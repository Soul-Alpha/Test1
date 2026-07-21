"""Prometheus Live Bot — autonomous XAUUSDm execution engine."""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import socket
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

# Ensure the Prometheus root is on sys.path so all engine/analysis imports
# resolve correctly whether the bot is launched as a module or a direct script.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    MT5_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# Local engine + analysis imports (all optional; bot degrades gracefully if missing)
try:
    from prometheus_core import Prometheus, PrometheusResult
    from analysis.confluence_scorer import ConfluenceScore, ConfluenceScorer
    from engines.multi_timeframe import TimeframeBias, MTFResult
    from engines.liquidity_smc import SMCResult
    from olympus.core.institutional_risk_performance import (
        build_institutional_risk_performance_runtime,
        write_institutional_risk_performance_runtime,
    )
    from live_bot.session_classifier import Session, SessionState, SessionClassifier
    from live_bot.regime_classifier import (
        Regime, RegimeState, RegimeClassifier, htf_alignment_score,
    )
    from live_bot.execution_quality import ExecutionQualityFilter, QualityResult
    from storage.database import init_db, list_analyses, list_trades, save_trade
    _ARCH_MODULES_OK = True
except Exception as _import_err:
    _ARCH_MODULES_OK = False
    Prometheus              = None  # type: ignore[assignment,misc]
    PrometheusResult        = None  # type: ignore[assignment,misc]
    ConfluenceScore         = None  # type: ignore[assignment,misc]
    ConfluenceScorer        = None  # type: ignore[assignment,misc]
    TimeframeBias           = None  # type: ignore[assignment,misc]
    SMCResult               = None  # type: ignore[assignment,misc]
    build_institutional_risk_performance_runtime = None  # type: ignore[assignment]
    write_institutional_risk_performance_runtime = None  # type: ignore[assignment]
    Session                 = None  # type: ignore[assignment,misc]
    SessionState            = None  # type: ignore[assignment,misc]
    SessionClassifier       = None  # type: ignore[assignment,misc]
    Regime                  = None  # type: ignore[assignment,misc]
    RegimeState             = None  # type: ignore[assignment,misc]
    RegimeClassifier        = None  # type: ignore[assignment,misc]
    htf_alignment_score     = None  # type: ignore[assignment]
    ExecutionQualityFilter  = None  # type: ignore[assignment,misc]
    QualityResult           = None  # type: ignore[assignment,misc]
    save_trade              = None  # type: ignore[assignment]
    init_db                 = None  # type: ignore[assignment]
    list_analyses           = None  # type: ignore[assignment]
    list_trades             = None  # type: ignore[assignment]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────────────────
_HERE         = pathlib.Path(__file__).parent
LEARNING_FILE = _HERE / "learning_state.json"
STATUS_FILE   = _HERE / "bot_status.json"
STOP_FLAG     = _HERE / "stop_flag"
RECOVERY_AUDIT_FILE = _HERE / "recovery_audit_log.jsonl"
RECOVERY_SESSIONS_FILE = _HERE / "recovery_sessions.jsonl"

# ── Constants missing from the original header (reconstructed) ─────────────────
MIN_BALANCE_USD     = 5.0    # hard floor — refuse entries if balance below this
CT_SCORE_BONUS      = 15.0   # extra score pts required for countertrend entries
CT_EXHAUSTION_EXTRA = 10.0   # additional CT premium in trend_exhaustion regime
M5_EXIT_MIN_USD     = 0.50   # minimum unrealised $ before 5M exit logic activates
M5_MIN_PROFIT_R     = 0.40   # minimum R multiple before weak/moderate 5M actions fire
M5_REVERSAL_CANDLES = 3      # consecutive opposing 5M bars to trigger exit
M5_STRONG_CANDLES   = 5      # consecutive bars required for "strong" severity rating
M5_STRONG_BODY_MULT   = 1.5  # avg 5M body/ATR ratio that triggers "strong" severity
M5_MODERATE_BODY_MULT = 0.8  # avg 5M body/ATR ratio that triggers "moderate" severity
ENABLE_5M_SEVERITY  = True   # True = weak/moderate/strong exits; False = original binary 5M full exit

# Time-aware profit capture (hybrid)
TIME_EXIT_ENABLE            = True
TIME_EXIT_SMART_MINUTES     = 15.0   # start evaluating smart timeout exits
TIME_EXIT_HARD_MINUTES      = 30.0   # hard fallback timeout
TIME_EXIT_PROFIT_USD_MIN    = 15.0   # minimum unrealised $ before timeout exits
TIME_EXIT_OPPOSING_BARS_REQ = 2      # smart trigger when >= N recent opposing 5M bars

# Small-account protection
# Accounts in the $50-$120 range are considered "small".
# They use limit-order-first entries (highest accuracy), single-leg TP1 only,
# full lot sized by the 2% rule (no artificial halving), and a tight SL gate.
SMALL_ACCOUNT_LOW        = 10.0    # USD floor — mirrors MIN_BALANCE_USD
SMALL_ACCOUNT_THRESHOLD  = 250.0   # USD ceiling for "small account" treatment
SMALL_ACCOUNT_SCALAR     = 1.0     # No lot halving -- 2% risk rule handles sizing correctly
SMALL_ACCOUNT_MAX_SL_ATR = 2.5     # Reject market entries whose SL > 2.5xATR (wide SL gate)
SMALL_ACCOUNT_MAX_OPEN   = 2       # max simultaneous positions on a small account
MEDIUM_ACCOUNT_THRESHOLD = 500.0   # USD -- above this is a normal account
MEDIUM_ACCOUNT_MAX_OPEN  = 3       # max simultaneous positions on a medium account
NORMAL_ACCOUNT_MAX_OPEN  = 5       # max simultaneous positions on a normal account

# Pyramid add-on constants
PYRAMID_MAX_ADD      = 1     # max extra positions beyond MAX_OPEN via pyramiding
PYRAMID_MIN_PROFIT_PTS = 50.0  # minimum unrealised points per position before adding
PYRAMID_RISK_MULT    = 0.5   # risk multiplier for pyramid legs (half normal size)

# Entry timing & direction gates
ENTRY_COOLDOWN_POLLS = 3     # polls between same-direction entries (3x60s=3min)
DIRECTION_LOSS_HALT  = 5     # consecutive SL losses before halting a direction (raised for $50k)
TRADE_DROUGHT_POLLS  = 30    # polls with no trade before secondary gates relax (30min)

# Risk circuit breakers
MAX_DAILY_LOSS_PCT       = 8.0   # halt entries if day loss >= 8% of session-start balance ($4k on $50k)
DAILY_PROFIT_PROTECT_PCT = 5.0   # scale lots 0.50x once daily P&L >= 5% ($2.5k on $50k)
DAILY_DEALS_REFRESH_SECONDS = 180  # throttle history_deals_get() to once per 3 min
DAILY_LOSS_HALT_SECONDS = 2 * 60 * 60  # fixed cooldown after daily-loss breaker trips (2h)

# Recovery authorization modes (additive safety layer above circuit breaker)
RECOVERY_MODE_HALTED = "halted"
RECOVERY_MODE_SHADOW = "shadow"
RECOVERY_MODE_PAPER = "paper"
RECOVERY_MODE_LIVE = "live"
RECOVERY_MODES = {
    RECOVERY_MODE_HALTED,
    RECOVERY_MODE_SHADOW,
    RECOVERY_MODE_PAPER,
    RECOVERY_MODE_LIVE,
}
RECOVERY_BALANCE_CHANGE_PCT = 0.02   # significant external balance change threshold (2%)
RECOVERY_BALANCE_CHANGE_MIN_USD = 100.0

# TP/SL & trailing constants
RR_MIN_LONG   = 2.0   # minimum reward:risk for long entries (1:2)
RR_MIN_SHORT  = 2.0   # minimum reward:risk for short entries (1:2)
TRAIL_ATR_MULT = 1.5  # trailing stop distance = TRAIL_ATR_MULT x ATR
BE_ATR_TRIGGER = 0.5  # default break-even trigger = 0.5 x ATR profit (overridden by regime)
BE_PROFIT_PTS  = 3.0  # price-point buffer above entry when moving SL to break-even

# Signal ranking dicts (used by qualification and gating logic)
TF_RANK: dict = {"1m":1,"5m":2,"15m":3,"30m":4,"1h":5,"4h":6,"1d":7,"1w":8}
GRADE_RANK: dict = {"A":4,"B":3,"C":2,"D":1,"F":0}

# ── Dashboard / manual override file paths ─────────────────────────────────────
MANUAL_TRADE_FILE = _HERE / "manual_trade.json"   # dashboard writes here to queue a manual trade
CONTROL_FILE      = _HERE / "bot_control.json"    # dashboard writes here to change entry_mode live

# ── Single-instance lock (TCP socket on loopback) ──────────────────────────────
BOT_LOCK_PORT   = 47_820
_LOCK_SOCKET: Optional[socket.socket] = None


def _acquire_pid_lock() -> None:
    """Bind a local TCP port to prevent duplicate bot instances."""
    global _LOCK_SOCKET
    _LOCK_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _LOCK_SOCKET.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        _LOCK_SOCKET.bind(("127.0.0.1", BOT_LOCK_PORT))
    except OSError:
        print(
            f"[lock] Another Prometheus instance is already running "
            f"(port {BOT_LOCK_PORT} in use). Exiting."
        )
        sys.exit(1)


def _release_pid_lock() -> None:
    """Release the TCP lock socket."""
    global _LOCK_SOCKET
    if _LOCK_SOCKET:
        try:
            _LOCK_SOCKET.close()
        except Exception:
            pass
        _LOCK_SOCKET = None





# LTF momentum scalp -- counter-trend entry when 5M+1M both strongly trend

LTF_SCALP_THRESHOLD = 0.60   # min |score| both 5M and 1M must reach to qualify

LTF_SCALP_ATR_SL    = 1.2    # SL = LTF_SCALP_ATR_SL x 5M_ATR (tight)

LTF_SCALP_RR        = 2.5    # TP reward:risk ratio for scalp entries







# Dual-TP split

TP1_RR     = 1.0    # first target at 1:1 R:R  (regime tp_scalar modulates)

TP2_RR     = 3.0    # second target at 1:3 R:R  (regime tp_scalar modulates)

TP1_MIN_USD = 5.0   # minimum $ profit on leg-1 — prevents hairline TPs on small lots



# Zone entry filter

ZONE_ATR_THRESHOLD = 1.0   # price must be within this many ATRs of an OB/S&R zone

LIMIT_ORDER_EXPIRY = 240   # pending limit orders expire after 240 polls (4h at 60s/poll -- matches 4H TF zone life)

MAX_LIMIT_DISTANCE_ATR = 3.0  # skip limit orders whose zone is >3xATR (tighter = higher fill probability)

MAGIC_LIMIT   = 777_001   # separate magic number for pending limit orders



# Entry mode

# "zone_only"  â€” only enter when price is at/near a fresh OB or S/R zone (strict)

# "market_any" â€” enter at market whenever score qualifies; use nearest S/R for SL

ENTRY_MODE = "zone_only"



# ── Architecture upgrade feature flags ────────────────────────────────────────

# Set False to revert an individual feature to its original (pre-upgrade) behaviour.

ENABLE_REGIME_CLASSIFIER  = True   # probabilistic HTF weighting + per-regime lot scaling

ENABLE_SESSION_FILTER     = True   # skip dead-zone hours, tighten spread tolerance

ENABLE_EXECUTION_QUALITY  = True   # reject entries when spread > X% of ATR

ENABLE_ADAPTIVE_BE        = True   # regime-specific break-even ATR multiplier

# Governance scaffold only: advisory data can be produced while enforcement remains disabled.
ENABLE_INSTITUTIONAL_POLICY_GATE = False



# How many consecutive polls showing a new direction before the bot is allowed

# to flip its primary trading direction.  2 polls @ 60s = 2 minutes minimum.

# Require 4 consecutive polls (4x60s=4 min) before accepting a direction flip.

DIRECTION_FLIP_MIN_CONFIRMS: int = 4

# Extra score required above threshold when entering countertrend trades.

# 15 pts is deliberately high — CT trades require a strong edge, not marginal signals.

CT_SCORE_BONUS: float = 15.0

# Additional score premium when CT during trend_exhaustion (exhaustion != reversal).

CT_EXHAUSTION_EXTRA: float = 10.0

# CT trade lot scalar — countertrend entries use 50% of normal position size.

CT_LOT_MULT: float = 0.5

# Periodic full MT5<->DB reconciliation interval (polls).  10 polls @ 60s = 10 min.

DB_SYNC_INTERVAL_POLLS: int = 10



# Learning state â€“ persisted to disk and reloaded on every restart.

# All fields are merged from the saved file at startup so knowledge compounds.

_LEARNING: dict = {

    "wins":             0,     # total closed positions in profit

    "losses":           0,     # total closed positions at a loss

    "total_seen":       0,     # all qualifying signals evaluated (cumulative)

    "score_adjust":     0.0,   # dynamic offset applied to min_score

    "grade_stats":      {},    # {"A": {"wins": n, "losses": n, "seen": n}, ...}

    "direction_stats":  {},    # {"SELL": {"wins": n, "losses": n}, ...}

    "open_pnl_history": [],    # rolling 20-poll unrealised PnL snapshots

    # â”€â”€ richer adaptive fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    "streak":           0,     # +N = N consecutive wins, -N = N consecutive losses

    "best_streak":      0,     # session best win streak

    "worst_streak":     0,     # session worst loss streak

    "total_pnl":        0.0,   # cumulative realised P&L ($)

    "last_20_results":  [],    # 1=win 0=loss for last 20 closed trades

    "ob_stats":         {},    # {"bearish": {"hits": n, "wins": n}, ...}

    "saved_at":         "",    # ISO timestamp of last disk write

    "ltf_stats":        {},    # {"both_confirmed": {"wins": 0, "losses": 0}, "one_counter": {...}, "trap": n, "unknown": {...}}

    # ── Extended learning dimensions (added in architecture upgrade) ──────────────
    "regime_stats":    {},   # {regime_name: {"wins": 0, "losses": 0, "pnl": 0.0}}
    "session_stats":   {},   # {session_name: {"wins": 0, "losses": 0, "pnl": 0.0}}
    "archetype_stats": {},   # {archetype: {"wins": 0, "losses": 0}} — reserved
    "spread_env_stats":{},   # {"tight":{"wins":0,"losses":0}, "medium":{...}, "wide":{...}}
    "vol_env_stats":   {},   # {"low":{"wins":0,"losses":0}, "medium":{...}, "high":{...}}

}





# â”€â”€ Persistent learning helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



def _save_learning(ob_stats: dict) -> None:

    """Merge ob_stats into _LEARNING and flush the whole dict to disk."""

    try:

        _LEARNING["ob_stats"]  = ob_stats

        _LEARNING["saved_at"] = datetime.utcnow().isoformat()

        LEARNING_FILE.write_text(

            json.dumps(_LEARNING, indent=2, default=str), encoding="utf-8"

        )

    except Exception as _exc:

        logger.debug("Could not save learning state: %s", _exc)





def _load_learning(ob_stats_ref: dict) -> None:

    """Load persisted learning state from disk and merge into _LEARNING.



    Unknown keys are ignored so old files stay compatible with new fields.

    ob_stats_ref is the bot instance's _ob_stats dict â€” updated in-place.

    """

    global _LEARNING

    if not LEARNING_FILE.exists():

        return

    try:

        saved = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))

        mergeable = {
            "wins", "losses", "total_seen", "score_adjust",
            "grade_stats", "direction_stats",
            "streak", "best_streak", "worst_streak",
            "total_pnl", "last_20_results", "ob_stats", "ltf_stats",
            "exit_reason_stats", "regime_stats", "session_stats",
            "open_trade_meta",
            # Entry quality positioning — added for zone/score/SL learning
            "zone_position_stats", "zone_type_stats",
            "sl_atr_stats", "score_bucket_stats",
        }

        for k in mergeable:

            if k in saved:

                _LEARNING[k] = saved[k]

        # Restore ob_stats into the instance dict too

        if "ob_stats" in saved:

            ob_stats_ref.update(saved["ob_stats"])

        logger.info(

            "[LML] Restored learning state: W=%d L=%d streak=%+d score_adjust=%+.1f",

            _LEARNING["wins"], _LEARNING["losses"],

            _LEARNING["streak"], _LEARNING["score_adjust"],

        )

    except Exception as _exc:

        logger.warning("[LML] Could not load learning state: %s", _exc)





def _bootstrap_from_db() -> None:

    """If the learning file is missing, reconstruct wins/losses from the

    trade DB so historical performance is never lost."""

    if LEARNING_FILE.exists():

        return   # already have a richer file â€” don't overwrite

    try:

        from storage.database import list_trades as _lt

        all_trades = _lt(source="live", limit=1000)

        for t in all_trades:

            status = (t.get("status") or "").lower()

            pnl    = t.get("pnl") or 0.0

            if status == "win":

                _LEARNING["wins"]      += 1

                _LEARNING["total_pnl"] += pnl

                _LEARNING["last_20_results"].append(1)

            elif status == "loss":

                _LEARNING["losses"]    += 1

                _LEARNING["total_pnl"] += pnl

                _LEARNING["last_20_results"].append(0)

        _LEARNING["last_20_results"] = _LEARNING["last_20_results"][-20:]

        if _LEARNING["wins"] + _LEARNING["losses"] > 0:

            logger.info(

                "[LML] Bootstrapped from DB: W=%d L=%d total_pnl=$%.2f",

                _LEARNING["wins"], _LEARNING["losses"], _LEARNING["total_pnl"],

            )

    except Exception as _exc:

        logger.debug("[LML] DB bootstrap error: %s", _exc)





# =============================================================================

class PrometheusLiveBot:

    """

    Autonomous trading bot.



    Every poll cycle:

      1. Pulls live candles from MT5

      2. Runs the full Prometheus analysis pipeline internally

      3. Executes a trade if the result qualifies

      4. Falls back to checking DB signals saved by the dashboard

      5. Processes manual_trade.json overrides from the dashboard UI

    """



    def __init__(

        self,

        asset:         str   = "XAUUSD",

        timeframe:     str   = "1H",

        min_grade:     str   = "B",

        min_score:     float = 65.0,

        risk_pct:      float = 1.0,

        poll_interval: int   = 60,

        n_candles:     int   = 500,

        use_db:        bool  = True,

        dry_run:       bool  = True,

        entry_mode:    str   = "zone_only",

    ) -> None:

        self.asset         = asset.upper()

        self.timeframe     = timeframe.upper()

        self.min_grade     = min_grade.upper()

        self.min_score     = min_score

        self.risk_fraction = risk_pct / 100.0

        self.poll_interval = poll_interval

        self.n_candles     = n_candles

        self.use_db        = use_db

        self.dry_run       = dry_run

        self.entry_mode    = entry_mode.lower()



        self._last_id             = 0          # highest DB record ID seen so far

        self._total_trades        = 0

        self._started_at          = datetime.utcnow().isoformat()

        self._session_start_balance: float | None = None   # captured first poll
        self._day_start_balance:   float | None = None   # keyed by calendar date
        self._day_start_date:      str   | None = None   # YYYY-MM-DD

        self._daily_loss_halted   = False       # latched True when daily loss limit hit
        self._daily_trade_pnl_cache: float = 0.0
        self._daily_loss_pct_cache:  float = 0.0
        self._daily_deals_last_ts:   float = 0.0
        self._daily_loss_halt_until_ts: float | None = None
        self._poll_error_count: int = 0
        self._poll_duration_history_ms: list[float] = []

        self._mt5_connected       = False      # set in run()
        self._symbol_selected_once = False     # avoid symbol_select() on every fetch

        # snapshot of open positions from previous poll (to detect closes)

        self._prev_open_tickets: set[int] = set()

        # tickets closed but not yet learned from (deferred for MT5 history-indexing delay)

        self._pending_close_tickets: set[int] = set()

        # cached last successful live analysis (persists across failed fetches)

        self._last_live_analysis: dict | None = None

        # OB learning: track which OB directions led to wins

        self._ob_stats: dict = {}   # {"bullish": {"hits": 0, "wins": 0}, "bearish": {"hits": 0, "wins": 0}}

        # Track pending limit orders placed at OB/zone levels {ticket: polls_remaining}

        self._pending_limits: dict[int, int] = {}

        # Track which positions are the "TP2 leg" of a dual-TP split

        self._tp2_tickets: set[int] = set()

        # High-water mark per ticket â€” best price seen (for tighter exit logic)

        self._hwm: dict[int, float] = {}   # {ticket: best_price}

        # Tickets that have already received a TP1-level partial close (avoid double-firing)

        self._tp1_partial_done: set[int] = set()

        # Tickets that already received the one-time smart-timeout partial close
        self._time_smart_partial_done: set[int] = set()

        self._last_ltf_state: str       = "unknown"   # LTF alignment at last qualifying signal
        self._current_ltf_biases: list  = []           # raw LTF biases last poll — used by pending limit monitor
        self._pending_limit_dirs: dict  = {}           # {ticket: is_long} — direction tracker for limit orders

        self._ltf_entry_state: dict     = {}           # {ticket: ltf_state} — set at open, read at close

        self._entry_grade: dict     = {}           # {ticket: grade}    — for win/loss attribution
        self._entry_regime: dict    = {}           # {ticket: regime}   — for regime_stats tracking
        self._entry_session: dict   = {}           # {ticket: session}  — for session_stats tracking

        self._last_tf_data:    dict = {}           # MTF candle dict from last poll (used by LTF scalp ATR)

        # Entry-quality positioning trackers -- set in _execute_from_result, consumed at close
        self._entry_zone_pos:  dict = {}  # {ticket: 'deep_fill'|'mid_fill'|'shallow_fill'|'no_zone'}
        self._entry_zone_type: dict = {}  # {ticket: 'ob'|'sr'|'no_zone'}
        self._entry_sl_atr:    dict = {}  # {ticket: float} -- sl_distance / atr at fill
        self._entry_score:     dict = {}  # {ticket: float} -- confluence score at fill
        self._startup_reconciled: bool = False     # True after first-poll DB reconciliation runs



        # ── Direction-flip guard & countertrend tracking ──────────────────────

        self._cached_open_directions: set[str] = set()  # "BUY"/"SELL" – updated each poll

        self._last_confirmed_direction: str = ""        # direction we most recently traded

        self._direction_flip_pending: str  = ""        # candidate new direction being confirmed

        self._direction_flip_polls: int    = 0         # consecutive polls showing the pending direction

        self._poll_count: int = 0            # monotonically increasing poll counter

        self._current_trade_is_ct: bool = False  # True if last qualified signal was countertrend

        # Session-direction loss halt: track consecutive SL hits per direction within
        # the current session. After DIRECTION_LOSS_HALT consecutive SL losses in one
        # direction, that direction is blocked for the rest of the session.
        # Reset when the session label changes (e.g., London -> NY Lunch).
        self._session_dir_losses: dict = {}   # {"BUY": n, "SELL": n}
        self._halt_session_label: str  = ""   # session when counters were last reset

        # Daily profit protection flag — set when daily P&L >= DAILY_PROFIT_PROTECT_PCT.
        # When active, lot sizes are scaled to 0.50x to lock in the day's gains.
        self._daily_profit_protect: bool = False



        # ── Architecture-upgrade classifiers (all optional / fail-soft) ───────

        _have_regime = ENABLE_REGIME_CLASSIFIER and _ARCH_MODULES_OK

        self._regime_clf   = RegimeClassifier()       if _have_regime else None

        self._session_clf  = SessionClassifier()      if (ENABLE_SESSION_FILTER and _ARCH_MODULES_OK) else None

        self._exec_quality = ExecutionQualityFilter() if (ENABLE_EXECUTION_QUALITY and _ARCH_MODULES_OK) else None

        self._current_regime  = None   # RegimeState – updated after each df fetch

        self._current_session = None   # SessionState – updated each poll

        self._asian_exhaustion_override_active: bool = False  # set each poll, consumed by _qualifies_result

        self._htf_lot_mult: float = 1.0  # set by _qualifies_result, consumed by _execute_from_result

        self._last_exec_quality_result: dict | None = None  # {"passed": bool, "reason": str, "spread_pips": float}

        # LTF reversal pending flag (set by micro-LTF gate, consumed by _poll_live).
        # Reversal auto-entry is DISABLED — flag retained for log visibility only.
        self._ltf_reversal_pending: dict | None = None

        # Entry cooldown: track the poll count when the last new primary entry was
        # placed, and its direction. Prevents rapid-fire accumulation into a moving
        # market when each poll sees a qualifying signal.
        # Rule: after any new entry, wait ENTRY_COOLDOWN_POLLS before next same-dir entry.
        self._last_entry_poll: int  = -999   # poll index of the most recent new entry
        self._last_entry_dir: str   = ""     # direction of that entry ("bullish"/"bearish")



        # â”€â”€ Prometheus analysis engine (initialized once, reused every poll) â”€â”€

        self.engine = None  # lazy-initialized in run() after imports settle



    # â”€â”€ Open positions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def _get_open_positions(self) -> list[dict]:

        """Fetch all MT5 positions for this bot's magic number.

        Returns a list of plain dicts safe for JSON serialisation."""

        if self.dry_run or not MT5_AVAILABLE:

            return []

        try:

            positions = mt5.positions_get(symbol=self.asset)

            if positions is None:

                return []

            result = []
            # Latency optimization: cache symbol metadata and ticks per symbol
            # once for this poll instead of querying MT5 repeatedly per position.
            _tick_by_symbol: dict[str, object] = {}
            _sym_by_symbol: dict[str, object] = {}

            for p in positions:

                if p.magic not in (777_000, 777_001, 777_002):   # all our trades

                    continue

                if p.symbol not in _tick_by_symbol:
                    _tick_by_symbol[p.symbol] = mt5.symbol_info_tick(p.symbol)
                tick = _tick_by_symbol[p.symbol]

                cur  = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask

                if cur:

                    unrealised = (cur - p.price_open) * p.volume * (

                        1 if p.type == mt5.POSITION_TYPE_BUY else -1

                    )

                    # approximate using tick value

                    if p.symbol not in _sym_by_symbol:
                        _sym_by_symbol[p.symbol] = mt5.symbol_info(p.symbol)
                    sym = _sym_by_symbol[p.symbol]

                    if sym and sym.trade_tick_size:

                        unrealised = unrealised / sym.trade_tick_size * sym.trade_tick_value

                else:

                    unrealised = p.profit

                _ticket = p.ticket

                _open_minutes = (datetime.utcnow() - datetime.utcfromtimestamp(p.time)).total_seconds() / 60.0

                _entry_regime = self._entry_regime.get(
                    _ticket,
                    getattr(getattr(self._current_regime, "regime", None), "value", "unknown"),
                )

                _tpol = self._time_exit_regime_policy(_entry_regime)

                _smart_min = float(_tpol.get("smart_minutes", TIME_EXIT_SMART_MINUTES))

                _hard_min = float(TIME_EXIT_HARD_MINUTES)

                _to_smart = max(0.0, _smart_min - _open_minutes)

                _to_hard = max(0.0, _hard_min - _open_minutes)

                _profit_gap = max(0.0, TIME_EXIT_PROFIT_USD_MIN - float(unrealised))

                result.append({

                    "ticket":     p.ticket,

                    "symbol":     p.symbol,

                    "direction":  "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",

                    "lots":       round(p.volume, 2),

                    "entry":      round(p.price_open, 5),

                    "sl":         round(p.sl, 5),

                    "tp":         round(p.tp, 5),

                    "current":    round(cur, 5) if cur else None,

                    "unrealised": round(unrealised, 2),

                    "open_since": datetime.utcfromtimestamp(p.time).isoformat(),

                    "open_minutes": round(_open_minutes, 2),

                    "entry_regime": _entry_regime,

                    "time_exit_smart_minutes": round(_smart_min, 2),

                    "time_exit_hard_minutes": round(_hard_min, 2),

                    "time_to_smart_min": round(_to_smart, 2),

                    "time_to_hard_min": round(_to_hard, 2),

                    "time_exit_profit_min_usd": round(TIME_EXIT_PROFIT_USD_MIN, 2),

                    "time_exit_profit_gap_usd": round(_profit_gap, 2),

                    "time_exit_smart_eligible": bool(_open_minutes >= _smart_min and unrealised >= TIME_EXIT_PROFIT_USD_MIN),

                    "time_exit_hard_due": bool(_open_minutes >= _hard_min and unrealised >= TIME_EXIT_PROFIT_USD_MIN),

                    "time_smart_partial_done": bool(_ticket in self._time_smart_partial_done),

                    "comment":    p.comment,

                })

            return result

        except Exception as exc:

            logger.warning("Could not fetch open positions: %s", exc)

            return []



    # â”€â”€ Learning from closed positions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€





    # -- Startup DB reconciliation -------------------------------------------------



    def _reconcile_db_on_startup(self) -> None:

        """On the first poll, backfill any DB trades still marked 'open' that

        MT5 has already closed.  Handles bot restarts where positions closed

        during downtime."""

        if self.dry_run or not MT5_AVAILABLE:

            return

        try:

            open_db = [t for t in list_trades(source="live", limit=1000)

                       if t.get("status") == "open" and t.get("trade_id")]

            if not open_db:

                return

            now   = datetime.utcnow()

            start = now - timedelta(days=30)

            all_deals = mt5.history_deals_get(

                datetime.utcfromtimestamp(start.timestamp()), now

            ) or []

            close_deals = {}

            for d in all_deals:

                if d.entry == mt5.DEAL_ENTRY_OUT:

                    close_deals.setdefault(d.position_id, []).append(d)



            updated = 0

            for trade in open_db:

                tid = str(trade["trade_id"])

                if not tid.startswith("live_"):

                    continue

                try:

                    pos_id = int(tid[5:])

                except ValueError:

                    continue

                closes = close_deals.get(pos_id)

                if not closes:

                    continue

                net_pnl    = sum(d.profit for d in closes)

                exit_price = closes[-1].price

                status     = "win" if net_pnl > 0 else "loss"

                try:

                    save_trade({

                        "trade_id":   tid,

                        "status":     status,

                        "pnl":        round(net_pnl, 2),

                        "exit_price": exit_price,

                        "exit_bar":   0,

                    })

                    updated += 1

                except Exception as _dbe:

                    logger.debug("startup reconcile save error for %s: %s", tid, _dbe)



            if updated:

                logger.info(

                    "Startup DB reconciliation: updated %d previously-open trades.", updated

                )

        except Exception as exc:

            logger.warning("Startup DB reconciliation failed: %s", exc)





    def _periodic_db_sync(self) -> None:

        """Full MT5 <-> DB reconciliation, run every DB_SYNC_INTERVAL_POLLS polls.



        Compares every DB trade marked 'open' against live MT5 positions and

        MT5 deal history.  Orphaned records (no matching MT5 position) are closed

        in the DB using deal history.  Handles crash recovery and terminal restarts.

        Called from _poll(); safe to call on every poll if needed.

        """

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            return

        try:

            open_db = [t for t in list_trades(source="live", limit=2000)

                       if t.get("status") == "open" and t.get("trade_id")]

            if not open_db:

                return



            # Current MT5 open ticket IDs (all our magic numbers)

            mt5_positions = mt5.positions_get(symbol=self.asset) or []

            mt5_tickets   = {p.ticket for p in mt5_positions

                             if p.magic in (777_000, 777_001, 777_002)}



            # Deal history for the last 60 days

            now   = datetime.utcnow()

            start = now - timedelta(days=60)

            all_deals = mt5.history_deals_get(

                datetime.utcfromtimestamp(start.timestamp()), now

            ) or []

            close_deals: dict = {}

            open_deals:  dict = {}

            for d in all_deals:

                if d.entry == mt5.DEAL_ENTRY_OUT:

                    close_deals.setdefault(d.position_id, []).append(d)

                elif d.entry == mt5.DEAL_ENTRY_IN:

                    open_deals[d.position_id] = d



            synced = 0

            for trade in open_db:

                tid = str(trade["trade_id"])

                if not tid.startswith("live_"):

                    continue

                try:

                    pos_id = int(tid[5:])

                except ValueError:

                    continue

                if pos_id in mt5_tickets:

                    continue   # still open in MT5, nothing to do

                closes = close_deals.get(pos_id)

                if not closes:

                    continue   # no history — old-session ghost, skip



                net_pnl    = round(sum(d.profit for d in closes), 2)

                exit_price = closes[-1].price

                status     = "win" if net_pnl > 0 else "loss"



                # Derive hold time from matched open deal

                _open_d  = open_deals.get(pos_id)

                hold_secs = int(closes[-1].time - _open_d.time) if _open_d else None



                # Exit reason from deal comment

                _cmt = (closes[-1].comment or "").lower()

                if "5m-exit" in _cmt or "5m_exit" in _cmt:

                    exit_reason = "5m_exit"

                elif "5m-partial" in _cmt or "partial" in _cmt:

                    exit_reason = "5m_partial"

                elif "tp1" in _cmt:

                    exit_reason = "tp1"

                elif "tp2" in _cmt:

                    exit_reason = "tp2"

                elif "manual" in _cmt:

                    exit_reason = "manual"

                elif "time-smart" in _cmt:

                    exit_reason = "time_smart"

                elif "time-hard" in _cmt:

                    exit_reason = "time_hard"

                else:

                    exit_reason = "sl"



                try:

                    save_trade({

                        "trade_id":     tid,

                        "status":       status,

                        "pnl":          net_pnl,

                        "exit_price":   exit_price,

                        "exit_bar":     0,

                        "exit_reason":  exit_reason,

                        "hold_seconds": hold_secs,

                    })

                    synced += 1

                except Exception as _dbe:

                    logger.debug("periodic sync save error %s: %s", tid, _dbe)



            if synced:

                logger.info("[DB-sync] Periodic reconciliation: closed %d orphaned record(s).", synced)

        except Exception as exc:

            logger.warning("[DB-sync] Periodic reconciliation error: %s", exc)

    def _learn_from_closes(self, now_open: list[dict]) -> None:
        """Detect closed positions via snapshot diff, queue for deferred learning,
        then process pending tickets with position-specific MT5 history queries.

        MT5 history_deals_get() by date range does not immediately index deals
        created by order_send() -- there is a ~60 s delay.  Queuing closed tickets
        and retrying on the next poll with history_deals_get(position=ticket)
        guarantees every close is eventually learned from.
        """
        if self.dry_run or not MT5_AVAILABLE:
            return

        now_tickets = {p["ticket"] for p in now_open}
        closed = self._prev_open_tickets - now_tickets

        # Queue newly detected closes; MT5 history needs ~60 s to index the deal.
        if closed:
            logger.debug(
                "_learn_from_closes: queuing %d closed ticket(s): %s",
                len(closed), closed,
            )
            self._pending_close_tickets |= closed

        if not self._pending_close_tickets:
            return

        try:
            processed: set[int] = set()

            for pos_ticket in list(self._pending_close_tickets):
                # Position-specific query: reliable, no date-range timing issues
                deals = mt5.history_deals_get(position=pos_ticket)
                if not deals:
                    continue  # not indexed yet -- retry next poll

                close_deal = next(
                    (d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT),
                    None,
                )
                if close_deal is None:
                    continue  # position still open or deal not indexed yet

                processed.add(pos_ticket)

                won = close_deal.profit > 0
                direction = "BUY" if close_deal.type == mt5.DEAL_TYPE_BUY else "SELL"

                if won:
                    _LEARNING["wins"] += 1
                    # Reset SL-loss counter for this direction on a win
                    self._session_dir_losses[direction] = 0
                else:
                    _LEARNING["losses"] += 1
                    # Increment session-direction SL loss counter
                    self._session_dir_losses[direction] = (
                        self._session_dir_losses.get(direction, 0) + 1
                    )
                    _cur_sess = getattr(
                        getattr(self._current_session, "session", None), "value", ""
                    ) or ""
                    _halt_n = self._session_dir_losses[direction]
                    if _halt_n >= DIRECTION_LOSS_HALT:
                        logger.warning(
                            "[halt] %s direction HALTED this session (%s): "
                            "%d consecutive SL losses",
                            direction, _cur_sess, _halt_n,
                        )

                # per-direction stats
                ds = _LEARNING["direction_stats"].setdefault(direction, {"wins": 0, "losses": 0})
                ds["wins" if won else "losses"] += 1

                # per-grade win/loss tracking
                _g = self._entry_grade.pop(pos_ticket, None)
                if _g:
                    _gs = _LEARNING["grade_stats"].setdefault(
                        _g, {"seen": 0, "acted": 0, "wins": 0, "losses": 0}
                    )
                    _gs.setdefault("wins", 0)
                    _gs.setdefault("losses", 0)
                    _gs["wins" if won else "losses"] += 1

                # -- Streak & rolling window tracking --
                _LEARNING["total_pnl"] = round(
                    _LEARNING.get("total_pnl", 0.0) + close_deal.profit, 2
                )
                last20 = _LEARNING.setdefault("last_20_results", [])
                last20.append(1 if won else 0)
                if len(last20) > 20:
                    last20.pop(0)

                streak = _LEARNING.get("streak", 0)
                if won:
                    streak = streak + 1 if streak >= 0 else 1
                else:
                    streak = streak - 1 if streak <= 0 else -1
                _LEARNING["streak"]       = streak
                _LEARNING["best_streak"]  = max(_LEARNING.get("best_streak", 0), streak)
                _LEARNING["worst_streak"] = min(_LEARNING.get("worst_streak", 0), streak)

                logger.info(
                    "Position %s closed | P&L=$%.2f | %s | W/L=%d/%d streak=%+d total_pnl=$%.2f",
                    pos_ticket, close_deal.profit, "WIN" if won else "LOSS",
                    _LEARNING["wins"], _LEARNING["losses"],
                    streak, _LEARNING["total_pnl"],
                )

                # Clean up partial-close tracker for this position
                self._tp1_partial_done.discard(pos_ticket)
                self._time_smart_partial_done.discard(pos_ticket)

                # LTF learning -- record which LTF state this entry had
                _ltf_key = self._ltf_entry_state.pop(pos_ticket, "unknown")
                _ls = _LEARNING["ltf_stats"].setdefault(_ltf_key, {"wins": 0, "losses": 0})
                _ls["wins" if won else "losses"] += 1

                # -- Regime / session performance tracking --
                _regime_key = self._entry_regime.pop(pos_ticket, "unknown")
                _rs = _LEARNING["regime_stats"].setdefault(_regime_key, {"wins": 0, "losses": 0, "pnl": 0.0})
                _rs["wins" if won else "losses"] += 1
                _rs["pnl"] = round(_rs.get("pnl", 0.0) + close_deal.profit, 2)

                _session_key = self._entry_session.pop(pos_ticket, "unknown")
                _ss = _LEARNING["session_stats"].setdefault(_session_key, {"wins": 0, "losses": 0, "pnl": 0.0})
                _ss["wins" if won else "losses"] += 1
                _ss["pnl"] = round(_ss.get("pnl", 0.0) + close_deal.profit, 2)


                # -- Entry-quality positioning stats (zone_pos, zone_type, sl_atr, score_bucket) --
                _zp_key = self._entry_zone_pos.pop(pos_ticket, 'unknown')
                _zp_st  = _LEARNING.setdefault('zone_position_stats', {})
                _zp_d   = _zp_st.setdefault(_zp_key, {'wins': 0, 'losses': 0})
                _zp_d['wins' if won else 'losses'] += 1

                _zt_key = self._entry_zone_type.pop(pos_ticket, 'unknown')
                _zt_st  = _LEARNING.setdefault('zone_type_stats', {})
                _zt_d   = _zt_st.setdefault(_zt_key, {'wins': 0, 'losses': 0})
                _zt_d['wins' if won else 'losses'] += 1

                _sa_val = self._entry_sl_atr.pop(pos_ticket, None)
                if _sa_val is not None:
                    _sa_bkt = 'tight' if _sa_val < 0.8 else ('wide' if _sa_val > 1.5 else 'normal')
                    _sa_st  = _LEARNING.setdefault('sl_atr_stats', {})
                    _sa_d   = _sa_st.setdefault(_sa_bkt, {'wins': 0, 'losses': 0, 'total_atr': 0.0, 'count': 0})
                    _sa_d['wins' if won else 'losses'] += 1
                    _sa_d['total_atr'] = round(_sa_d.get('total_atr', 0.0) + _sa_val, 3)
                    _sa_d['count']     += 1

                _sc_val = self._entry_score.pop(pos_ticket, None)
                if _sc_val is not None:
                    _sc_bkt = ('90+' if _sc_val >= 90 else
                               '85-89' if _sc_val >= 85 else
                               '75-84' if _sc_val >= 75 else '65-74')
                    _sc_st  = _LEARNING.setdefault('score_bucket_stats', {})
                    _sc_d   = _sc_st.setdefault(_sc_bkt, {'wins': 0, 'losses': 0})
                    _sc_d['wins' if won else 'losses'] += 1

                # -- Persist trade outcome to DB --
                try:
                    _open_deal = next(
                        (d for d in deals if d.entry == mt5.DEAL_ENTRY_IN),
                        None,
                    )
                    _hold_secs = int(close_deal.time - _open_deal.time) if _open_deal else None
                    _cmt = (close_deal.comment or "").lower()
                    if "5m-exit" in _cmt or "5m_exit" in _cmt:
                        _exit_reason = "5m_exit"
                    elif "partial" in _cmt:
                        _exit_reason = "5m_partial"
                    elif "tp1" in _cmt:
                        _exit_reason = "tp1"
                    elif "tp2" in _cmt:
                        _exit_reason = "tp2"
                    elif "trail" in _cmt:
                        _exit_reason = "trail"
                    elif "manual" in _cmt:
                        _exit_reason = "manual"
                    elif "time-smart" in _cmt:
                        _exit_reason = "time_smart"
                    elif "time-hard" in _cmt:
                        _exit_reason = "time_hard"
                    else:
                        _exit_reason = "sl"
                    save_trade({
                        "trade_id":     f"live_{pos_ticket}",
                        "exit_price":   close_deal.price,
                        "pnl":          close_deal.profit,
                        "status":       "win" if won else "loss",
                        "exit_bar":     0,
                        "exit_reason":  _exit_reason,
                        "hold_seconds": _hold_secs,
                    })
                    # Track exit-reason efficiency in learning state
                    _er = _LEARNING.setdefault("exit_reason_stats", {})
                    _er_e = _er.setdefault(_exit_reason, {"count": 0, "wins": 0, "pnl": 0.0})
                    _er_e["count"] += 1
                    _er_e["wins"]  += (1 if won else 0)
                    _er_e["pnl"]   = round(_er_e["pnl"] + close_deal.profit, 2)
                except Exception as _dbe:
                    logger.debug("DB save_trade (close) error: %s", _dbe)

            # -- Remove processed tickets from pending set --
            self._pending_close_tickets -= processed

            # -- Adaptive threshold: LML engine --
            # Only recalculate and save when we actually learned something new.
            if processed:
                total = _LEARNING["wins"] + _LEARNING["losses"]
                if total >= 3:
                    last20    = _LEARNING.get("last_20_results", [])
                    recent_wr = (sum(last20) / len(last20)) if last20 else None
                    alltime_wr = _LEARNING["wins"] / total
                    # Blend: 60% recent, 40% all-time
                    win_rate = (
                        0.6 * recent_wr + 0.4 * alltime_wr
                        if recent_wr is not None else alltime_wr
                    )

                    streak = _LEARNING.get("streak", 0)

                    # Base adjustment from blended win rate
                    if win_rate < 0.35:
                        adj = min(15.0, (0.35 - win_rate) * 60)
                    elif win_rate < 0.50:
                        adj = min(10.0, (0.50 - win_rate) * 40)
                    elif win_rate > 0.70:
                        adj = max(-8.0, (0.70 - win_rate) * 30)
                    elif win_rate > 0.55:
                        adj = max(-4.0, (0.55 - win_rate) * 20)
                    else:
                        adj = 0.0

                    if streak <= -3:
                        adj += min(5.0, abs(streak) * 1.0)
                    elif streak >= 3:
                        adj -= min(3.0, streak * 0.5)

                    _LEARNING["score_adjust"] = round(max(-10.0, min(8.0, adj)), 1)
                    logger.info(
                        "[LML] win_rate=%.0f%% (recent=%.0f%% all=%.0f%%) streak=%+d "
                        "-> score_adjust=%+.1f",
                        win_rate * 100,
                        (recent_wr or alltime_wr) * 100,
                        alltime_wr * 100,
                        streak,
                        _LEARNING["score_adjust"],
                    )

                # -- Flush learning state to disk --
                self._persist_open_trade_meta()
                _save_learning(self._ob_stats)

        except Exception as exc:
            logger.warning("Learning update error: %s", exc)

    def _persist_open_trade_meta(self) -> None:
        """Merge per-ticket attribution dicts into _LEARNING so they survive restarts."""
        all_tickets = (
            set(self._entry_grade)
            | set(self._ltf_entry_state)
            | set(self._entry_regime)
            | set(self._entry_session)
            | set(self._entry_zone_pos)
            | set(self._entry_zone_type)
            | set(self._entry_sl_atr)
            | set(self._entry_score)
        )
        _LEARNING['open_trade_meta'] = {
            str(tk): {
                'grade':     self._entry_grade.get(tk, 'F'),
                'ltf_state': self._ltf_entry_state.get(tk, 'unknown'),
                'regime':    self._entry_regime.get(tk, 'unknown'),
                'session':   self._entry_session.get(tk, 'unknown'),
                'zone_pos':  self._entry_zone_pos.get(tk, 'unknown'),
                'zone_type': self._entry_zone_type.get(tk, 'unknown'),
                'sl_atr':    self._entry_sl_atr.get(tk, None),
                'score':     self._entry_score.get(tk, None),
            }
            for tk in all_tickets
        }

    def _check_5m_exits(self, open_positions: list[dict]) -> list[str]:

        """Classify 5M reversal severity and act proportionally.



        Severity is determined by candle direction AND avg body size vs 5M ATR:

          weak     -> tighten trailing stop to lock 50% of accrued profit

          moderate -> partial close ~30% of the position

          strong   -> full exit (large bodies or 5+ consecutive opposing bars)



        A minimum profit gate (M5_MIN_PROFIT_R x SL distance) blocks weak/moderate

        actions when the position has not yet captured meaningful R.  Strong reversals

        bypass this gate to protect open equity.



        Set ENABLE_5M_SEVERITY=False to restore the original binary full-exit logic.

        """

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            return []

        profitable = [p for p in open_positions if p.get("unrealised", 0) >= M5_EXIT_MIN_USD]

        if not profitable:

            return []



        try:

            n_need = max(M5_REVERSAL_CANDLES, M5_STRONG_CANDLES if ENABLE_5M_SEVERITY else 3) + 3

            rates  = mt5.copy_rates_from_pos(self.asset, mt5.TIMEFRAME_M5, 0, n_need)

            if rates is None or len(rates) < M5_REVERSAL_CANDLES:

                return []

        except Exception as exc:

            logger.warning("5M bar fetch error: %s", exc)

            return []



        # 5M ATR (14-bar rolling) for body-size severity classification

        _atr_5m = 0.0

        if ENABLE_5M_SEVERITY and len(rates) >= 5:

            trs = [

                max(rates[i]["high"] - rates[i]["low"],

                    abs(rates[i]["high"] - rates[i - 1]["close"]),

                    abs(rates[i]["low"]  - rates[i - 1]["close"]))

                for i in range(1, len(rates))

            ]

            window  = trs[-min(14, len(trs)):]

            _atr_5m = sum(window) / len(window) if window else 0.0



        def _body_severity(bars: list) -> str:

            """Return 'strong' | 'moderate' | 'weak' by avg body size vs 5M ATR."""

            n = len(bars)

            if n == 0:

                return "weak"

            avg_body = sum(abs(r["close"] - r["open"]) for r in bars) / n

            ratio    = (avg_body / _atr_5m) if _atr_5m > 0 else 0.0

            if ratio >= M5_STRONG_BODY_MULT:

                return "strong"

            if ratio >= M5_MODERATE_BODY_MULT:

                return "moderate"

            return "weak"



        sym      = mt5.symbol_info(self.asset)

        fill     = (mt5.ORDER_FILLING_FOK

                    if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK)

                    else mt5.ORDER_FILLING_IOC)

        min_lot  = sym.volume_min  if sym else 0.01

        step_lot = sym.volume_step if sym else 0.01



        msgs: list[str] = []



        for pos in profitable:

            is_buy = pos["direction"] == "BUY"

            entry  = pos["entry"]

            sl     = pos["sl"]

            lots   = pos["lots"]



            # Check whether the base M5_REVERSAL_CANDLES bars all oppose the position

            base      = rates[-M5_REVERSAL_CANDLES:]

            base_dirs = ["bull" if r["close"] > r["open"] else "bear" for r in base]

            all_opp   = (

                (is_buy     and all(d == "bear" for d in base_dirs)) or

                (not is_buy and all(d == "bull" for d in base_dirs))

            )

            if not all_opp:

                continue   # no opposing reversal for this position direction



            # Classify severity

            if ENABLE_5M_SEVERITY:

                _sev = _body_severity(base)

                # Upgrade to 'strong' if enough bars are ALL opposing

                if _sev != "strong" and len(rates) >= M5_STRONG_CANDLES:

                    ext      = rates[-M5_STRONG_CANDLES:]

                    ext_dirs = ["bull" if r["close"] > r["open"] else "bear" for r in ext]

                    ext_opp  = (

                        (is_buy     and all(d == "bear" for d in ext_dirs)) or

                        (not is_buy and all(d == "bull" for d in ext_dirs))

                    )

                    if ext_opp:

                        _sev = "strong"

            else:

                _sev = "strong"   # original behaviour: all opposing = full exit



            tick = mt5.symbol_info_tick(self.asset)

            if not tick:

                continue

            close_price = tick.bid if is_buy else tick.ask

            profit_pts  = (close_price - entry) if is_buy else (entry - close_price)

            sl_dist     = abs(entry - sl) if sl != 0 else 0.0



            # Minimum profit gate: blocks noisy weak/moderate exits

            if _sev in ("weak", "moderate") and sl_dist > 0:

                min_pts = sl_dist * M5_MIN_PROFIT_R

                if profit_pts < min_pts:

                    logger.info(

                        "[5M] #%s %s reversal but profit %.2f pts < min %.2f (%.0f%% of 1R) -- held",

                        pos["ticket"], _sev, profit_pts, min_pts, M5_MIN_PROFIT_R * 100,

                    )

                    continue



            if _sev == "weak":

                # Tighten trail: lock 50% of accrued profit in the SL

                if sl_dist > 0 and sl != 0 and profit_pts > 0:

                    tighter = (

                        round(entry + profit_pts * 0.5, 5) if is_buy

                        else round(entry - profit_pts * 0.5, 5)

                    )

                    if (is_buy and tighter > sl) or (not is_buy and tighter < sl):

                        sl_req = {

                            "action":   mt5.TRADE_ACTION_SLTP,

                            "symbol":   self.asset,

                            "position": pos["ticket"],

                            "sl":       tighter,

                            "tp":       pos["tp"],

                            "magic":    777_000,

                        }

                        r_sl = mt5.order_send(sl_req)

                        if r_sl and r_sl.retcode == mt5.TRADE_RETCODE_DONE:

                            msg = (

                                f"[5M-weak] {'BUY' if is_buy else 'SELL'} #{pos['ticket']} "

                                f"SL tightened {sl:.4f}->{tighter:.4f} "

                                f"(weak reversal, profit {profit_pts:.2f} pts)"

                            )

                            logger.info(msg)

                            msgs.append(msg)



            elif _sev == "moderate":

                # Partial close ~30% of the position

                close_lots = max(min_lot, round(lots * 0.30 / step_lot) * step_lot)

                close_lots = round(min(close_lots, lots), 2)

                req = {

                    "action":       mt5.TRADE_ACTION_DEAL,

                    "symbol":       self.asset,

                    "volume":       close_lots,

                    "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,

                    "price":        close_price,

                    "position":     pos["ticket"],

                    "deviation":    20,

                    "magic":        777_000,

                    "comment":      "Prom-5M-partial",

                    "type_time":    mt5.ORDER_TIME_GTC,

                    "type_filling": fill,

                }

                r = mt5.order_send(req)

                if r and r.retcode == mt5.TRADE_RETCODE_DONE:

                    msg = (

                        f"[5M-moderate] {'BUY' if is_buy else 'SELL'} #{pos['ticket']} "

                        f"partial {close_lots} lots @ {close_price:.4f} "

                        f"| profit={pos['unrealised']:+.2f}"

                    )

                    logger.info(msg)

                    msgs.append(msg)

                else:

                    code = r.retcode if r else "None"

                    logger.warning("5M moderate partial failed #%s retcode=%s", pos["ticket"], code)



            else:   # strong -- full exit

                req = {

                    "action":       mt5.TRADE_ACTION_DEAL,

                    "symbol":       self.asset,

                    "volume":       lots,

                    "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,

                    "price":        close_price,

                    "position":     pos["ticket"],

                    "deviation":    20,

                    "magic":        777_000,

                    "comment":      "Prom-5M-exit",

                    "type_time":    mt5.ORDER_TIME_GTC,

                    "type_filling": fill,

                }

                r = mt5.order_send(req)

                if r and r.retcode == mt5.TRADE_RETCODE_DONE:

                    n_used = M5_STRONG_CANDLES if ENABLE_5M_SEVERITY else M5_REVERSAL_CANDLES

                    reversal_desc = f"{n_used}x bearish" if is_buy else f"{n_used}x bullish"

                    msg = (

                        f"[5M-exit] {'BUY' if is_buy else 'SELL'} #{pos['ticket']} "

                        f"closed @ {close_price:.4f} "

                        f"| profit=${pos['unrealised']:+.2f} "

                        f"| 5M: {reversal_desc} (strong)"

                    )

                    logger.info(msg)

                    msgs.append(msg)

                    self._hwm.pop(pos["ticket"], None)

                else:

                    code = r.retcode if r else "None"

                    logger.warning("5M exit failed #%s retcode=%s", pos["ticket"], code)



        return msgs


    def _time_exit_regime_policy(self, regime_name: str) -> dict:

        """Return regime-aware timeout policy.

        Trend-expansion trades get more room and require stronger opposing flow.
        Mean-reversion / exhaustion trades exit earlier with lighter confirmation.
        """

        r = (regime_name or "unknown").lower()

        if r == "trend_expansion":

            return {"smart_minutes": 20.0, "opp_bars_req": 3}

        if r in ("mean_reversion", "trend_exhaustion"):

            return {"smart_minutes": 15.0, "opp_bars_req": 1}

        if r == "volatility_expansion":

            return {"smart_minutes": 15.0, "opp_bars_req": 2}

        return {

            "smart_minutes": TIME_EXIT_SMART_MINUTES,

            "opp_bars_req": TIME_EXIT_OPPOSING_BARS_REQ,

        }


    def _time_exit_is_smart_risk_off(self, is_buy: bool, rates, opp_bars_req: int) -> tuple[bool, int]:

        """Return (should_close, opposing_count) from recent 5M directional pressure."""

        if rates is None or len(rates) < max(3, opp_bars_req):

            return False, 0

        lookback = rates[-max(3, opp_bars_req):]

        dirs = ["bull" if r["close"] > r["open"] else "bear" for r in lookback]

        opp = "bear" if is_buy else "bull"

        opp_count = sum(1 for d in dirs if d == opp)

        return opp_count >= opp_bars_req, opp_count


    def _manage_time_profit_exits(self, open_positions: list[dict]) -> list[str]:

        """Hybrid time-aware profit capture.

        Smart path: after TIME_EXIT_SMART_MINUTES, close profitable positions when
        recent 5M flow shows opposing pressure.
        Hard path: after TIME_EXIT_HARD_MINUTES, close profitable positions even
        if smart conditions did not trigger.
        """

        if (not TIME_EXIT_ENABLE or self.dry_run or not MT5_AVAILABLE
                or not self._mt5_connected):

            return []

        profitable = [
            p for p in open_positions
            if p.get("unrealised", 0.0) >= TIME_EXIT_PROFIT_USD_MIN
        ]

        if not profitable:

            return []

        try:

            rates = mt5.copy_rates_from_pos(self.asset, mt5.TIMEFRAME_M5, 0, 6)

        except Exception as exc:

            logger.warning("time-exit 5M fetch error: %s", exc)

            return []

        sym = mt5.symbol_info(self.asset)

        fill = (mt5.ORDER_FILLING_FOK

                if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK)

                else mt5.ORDER_FILLING_IOC)

        min_lot  = sym.volume_min  if sym else 0.01

        step_lot = sym.volume_step if sym else 0.01

        msgs: list[str] = []

        for pos in profitable:

            open_m = float(pos.get("open_minutes", 0.0) or 0.0)

            ticket = pos.get("ticket")

            regime_name = self._entry_regime.get(
                ticket,
                getattr(getattr(self._current_regime, "regime", None), "value", "unknown"),
            )

            policy = self._time_exit_regime_policy(regime_name)

            smart_minutes = float(policy.get("smart_minutes", TIME_EXIT_SMART_MINUTES))

            opp_req = int(policy.get("opp_bars_req", TIME_EXIT_OPPOSING_BARS_REQ))

            if open_m < smart_minutes:

                continue

            is_buy = pos["direction"] == "BUY"

            smart_risk_off, opp_count = self._time_exit_is_smart_risk_off(is_buy, rates, opp_req)

            hard_due = open_m >= TIME_EXIT_HARD_MINUTES

            if not smart_risk_off and not hard_due:

                continue

            # Smart timeout is one-time partial harvesting; hard timeout can still close remainder later.
            if smart_risk_off and not hard_due and ticket in self._time_smart_partial_done:

                continue

            tick = mt5.symbol_info_tick(self.asset)

            if not tick:

                continue

            close_price = tick.bid if is_buy else tick.ask

            lots = float(pos.get("lots", 0.0) or 0.0)

            if lots <= 0:

                continue

            if hard_due:

                close_lots = lots

                mode = "time-hard"

            else:

                close_lots = max(min_lot, round((lots * 0.50) / step_lot) * step_lot)

                close_lots = round(min(close_lots, lots), 2)

                mode = "time-smart-partial" if close_lots < lots else "time-smart"

            req = {

                "action":       mt5.TRADE_ACTION_DEAL,

                "symbol":       self.asset,

                "volume":       close_lots,

                "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,

                "price":        close_price,

                "position":     pos["ticket"],

                "deviation":    20,

                "magic":        777_000,

                "comment":      f"Prom-{mode}",

                "type_time":    mt5.ORDER_TIME_GTC,

                "type_filling": fill,

            }

            r = mt5.order_send(req)

            if r and r.retcode == mt5.TRADE_RETCODE_DONE:

                msg = (

                    f"[{mode}] {'BUY' if is_buy else 'SELL'} #{pos['ticket']} "

                    f"closed {close_lots:.2f} lot @ {close_price:.4f} | open={open_m:.1f}m "

                    f"| regime={regime_name} | opp5m={opp_count}/{opp_req} "

                    f"| profit=${pos['unrealised']:+.2f}"

                )

                logger.info(msg)

                msgs.append(msg)

                if mode == "time-smart-partial":

                    self._time_smart_partial_done.add(pos["ticket"])

                else:

                    self._time_smart_partial_done.discard(pos["ticket"])

                    self._hwm.pop(pos["ticket"], None)

            else:

                code = r.retcode if r else "None"

                logger.warning("%s exit failed #%s retcode=%s", mode, pos["ticket"], code)

        return msgs



    # â”€â”€ Trailing-stop management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def _manage_positions(self, open_positions: list[dict]) -> list[str]:

        """Ratchet SL towards price on every poll once a position is profitable.



        Strategy

        --------

        * Only activates when profit â‰¥ TRAIL_MIN_PROFIT Ã— ATR (avoids noise).

        * Trails SL at TRAIL_ATR_MULT Ã— ATR behind the current price.

        * SL is strictly ratcheted â€” it can only move in the profit direction,

          never widened.

        * Falls back to 0.15 % / 0.25 % of price when ATR is unavailable.

        """

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            return []

        if not open_positions:

            return []



        # ATR from the most recent live analysis (same symbol)

        atr: float | None = None

        if self._last_live_analysis:

            atr = self._last_live_analysis.get("atr")



        actions: list[str] = []

        for pos in open_positions:

            ticket  = pos["ticket"]

            entry   = pos["entry"]

            cur_sl  = pos["sl"]

            cur_tp  = pos["tp"]

            current = pos.get("current")

            is_buy  = pos["direction"] == "BUY"



            if current is None:

                continue



            if atr and atr > 0:

                trail_dist = atr * TRAIL_ATR_MULT

                # Adaptive BE: regime changes how aggressively we lock profit

                _be_mult = (

                    self._current_regime.be_atr_mult

                    if (ENABLE_ADAPTIVE_BE and self._current_regime is not None)

                    else BE_ATR_TRIGGER

                )

                be_trigger = atr * _be_mult

            else:

                trail_dist = current * 0.0025   # 0.25 % fallback

                be_trigger = current * 0.0015   # 0.15 % profit trigger



            if is_buy:

                profit_pts = current - entry

                if profit_pts < be_trigger:

                    continue                    # not profitable enough yet

                # Step 1: lock break-even+buffer if SL is still below entry

                if cur_sl < entry:

                    new_sl = round(entry + BE_PROFIT_PTS, 5)

                    req = {

                        "action":   mt5.TRADE_ACTION_SLTP,

                        "symbol":   self.asset,

                        "position": ticket,

                        "sl":       new_sl,

                        "tp":       cur_tp,

                        "magic":    777_000,

                    }

                    res = mt5.order_send(req)

                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:

                        msg = (f"[be-lock] BUY #{ticket} SL locked â†’ entry {entry:.4f}")

                        logger.info(msg)

                        actions.append(msg)

                    continue

                # Step 2: trail beyond break-even

                new_sl = round(current - trail_dist, 5)

                if new_sl <= cur_sl:

                    continue                    # already at or past the trail line

            else:

                profit_pts = entry - current

                if profit_pts < be_trigger:

                    continue

                # Step 1: break-even+buffer lock if SL is still above entry

                if cur_sl > entry:

                    new_sl = round(entry - BE_PROFIT_PTS, 5)

                    req = {

                        "action":   mt5.TRADE_ACTION_SLTP,

                        "symbol":   self.asset,

                        "position": ticket,

                        "sl":       new_sl,

                        "tp":       cur_tp,

                        "magic":    777_000,

                    }

                    res = mt5.order_send(req)

                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:

                        msg = (f"[be-lock] SELL #{ticket} SL locked â†’ entry {entry:.4f}")

                        logger.info(msg)

                        actions.append(msg)

                    continue

                # Step 2: trail beyond break-even

                new_sl = round(current + trail_dist, 5)

                if new_sl >= cur_sl:

                    continue



            req = {

                "action":   mt5.TRADE_ACTION_SLTP,

                "symbol":   self.asset,

                "position": ticket,

                "sl":       new_sl,

                "tp":       cur_tp,

                "magic":    777_000,

            }

            res = mt5.order_send(req)

            if res and res.retcode == mt5.TRADE_RETCODE_DONE:

                msg = (

                    f"[trail] {'BUY' if is_buy else 'SELL'} #{ticket} "

                    f"SL {cur_sl:.4f} â†’ {new_sl:.4f} "

                    f"(profit {profit_pts:+.2f} | dist {trail_dist:.2f})"

                )

                logger.info(msg)

                actions.append(msg)

            else:

                code = res.retcode if res else "None"

                logger.warning("Trail SL modify failed #%s retcode=%s", ticket, code)



        return actions



    # â”€â”€ MT5 helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def _connect_mt5(self) -> bool:

        if not MT5_AVAILABLE:

            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")

            return False

        if not mt5.initialize():

            logger.error("MT5 init failed: %s", mt5.last_error())

            return False

        acct = mt5.account_info()

        if not acct:

            logger.error("MT5 account info unavailable â€” check you are logged in")

            return False

        logger.info(

            "MT5 connected | account=%s | balance=$%.2f | server=%s",

            acct.login, acct.balance, acct.server,

        )

        if not acct.trade_allowed:

            logger.warning("MT5: trade_allowed=False on this account")



        # â”€â”€ Discover the correct symbol name on this broker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Brokers often suffix with 'm', '.', '#' etc (e.g. XAUUSDm, XAUUSD.)

        if self.asset.upper() in ("XAUUSD", "XAUUSDM"):

            candidates = mt5.symbols_get("*XAUUSD*") or []

            names = [s.name for s in candidates]

            logger.info("Available XAUUSD symbols on this broker: %s", names)

            if names and self.asset not in names:

                # Prefer exact match; fall back to first result

                for preferred in [self.asset, "XAUUSDm", "XAUUSD", "XAUUSD."]:

                    if preferred in names:

                        logger.info("Auto-correcting symbol %s â†’ %s", self.asset, preferred)

                        self.asset = preferred

                        break

                else:

                    self.asset = names[0]

                    logger.info("Using broker symbol: %s", self.asset)



        return True



    def _mt5_account(self) -> dict:

        if not MT5_AVAILABLE:

            return {}

        try:

            acct = mt5.account_info()

            if acct:

                return {

                    "balance":     round(float(acct.balance), 2),

                    "equity":      round(float(acct.equity),  2),

                    "margin_free": round(float(acct.margin_free), 2),

                    "server":      acct.server,

                    "login":       str(acct.login),

                }

        except Exception:

            pass

        return {}



    def _calc_lot(self, sl_distance: float) -> float:

        """Size a position so that the SL represents exactly risk_fraction of balance."""

        if not MT5_AVAILABLE or sl_distance <= 0:

            return 0.01

        acct = mt5.account_info()

        sym  = mt5.symbol_info(self.asset)

        if not acct or not sym:

            return 0.01

        tick_val = sym.trade_tick_value

        tick_sz  = sym.trade_tick_size

        if tick_sz == 0 or tick_val == 0:

            return 0.01

        # Use equity (not balance) so floating profits compound lot sizes

        risk_amount = acct.equity * self.risk_fraction

        lot = risk_amount / (sl_distance / tick_sz * tick_val)

        lot = round(lot / sym.volume_step) * sym.volume_step

        lot = max(sym.volume_min, min(sym.volume_max, lot))

        return round(lot, 2)



    # â”€â”€ Live data helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def _tf_constant(self) -> int:

        """Return the MT5 TIMEFRAME constant for self.timeframe."""

        if not MT5_AVAILABLE:

            return 240

        _map = {

            "M1":  mt5.TIMEFRAME_M1,

            "M5":  mt5.TIMEFRAME_M5,

            "M15": mt5.TIMEFRAME_M15,

            "M30": mt5.TIMEFRAME_M30,

            "H1":  mt5.TIMEFRAME_H1,   "1H": mt5.TIMEFRAME_H1,

            "H4":  mt5.TIMEFRAME_H4,   "4H": mt5.TIMEFRAME_H4,

            "D1":  mt5.TIMEFRAME_D1,   "1D": mt5.TIMEFRAME_D1,

            "W1":  mt5.TIMEFRAME_W1,

        }

        return _map.get(self.timeframe, mt5.TIMEFRAME_H4)



    def _fetch_candles(self, n: int = 500):

        """Pull n recent OHLCV bars from MT5.  Returns a pandas DataFrame or None."""

        if not MT5_AVAILABLE or not self._mt5_connected:

            return None

        try:

            import pandas as pd  # noqa: PLC0415 â€” lazy to avoid startup conflicts

            # Ensure the symbol is selected/visible in MT5

            if not self._symbol_selected_once:

                if not mt5.symbol_select(self.asset, True):

                    logger.warning("symbol_select(%s) failed: %s", self.asset, mt5.last_error())
                else:
                    self._symbol_selected_once = True

            tf    = self._tf_constant()

            rates = mt5.copy_rates_from_pos(self.asset, tf, 0, n)

            if rates is None or len(rates) == 0:

                logger.warning("No candle data from MT5 for %s %s â€” error: %s",

                               self.asset, self.timeframe, mt5.last_error())

                return None

            df = pd.DataFrame(rates)

            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

            df = df.rename(columns={"tick_volume": "volume"})

            df = df.set_index("time")

            df.index.name = "datetime"

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as exc:

            logger.warning("_fetch_candles error: %s", exc)

            return None



    def _fetch_mtf_candles(self, primary_df=None):

        """Fetch lower-timeframe bars for multi-timeframe alignment analysis.



        Returns a dict {timeframe_str: DataFrame} ready to pass as

        ``tf_data`` to ``engine.analyze_data()``.  The primary timeframe

        data (supplied as *primary_df*) is included under its own key so

        the MTF engine sees all four timeframes.

        Missing / insufficient data for any TF is silently skipped.

        """

        if not MT5_AVAILABLE or not self._mt5_connected:

            return {self.timeframe.lower(): primary_df} if primary_df is not None else None



        try:

            import pandas as pd  # noqa: PLC0415



            def _rates_to_df(rates):

                df = pd.DataFrame(rates)

                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

                df = df.rename(columns={"tick_volume": "volume"})

                df = df.set_index("time")

                df.index.name = "datetime"

                return df[["open", "high", "low", "close", "volume"]]



            # All timeframes always fetched, regardless of primary TF.

            # Bars chosen so each covers ~2-4 weeks of data minimum.

            # Primary TF is seeded from primary_df (already fetched) and skipped here.

            _ALL_TFS = [

                ("1m",  mt5.TIMEFRAME_M1,  300),   # ~5 h

                ("5m",  mt5.TIMEFRAME_M5,  300),   # ~25 h

                ("15m", mt5.TIMEFRAME_M15, 300),   # ~3 days

                ("30m", mt5.TIMEFRAME_M30, 300),   # ~6 days

                ("1h",  mt5.TIMEFRAME_H1,  300),   # ~12 days

                ("4h",  mt5.TIMEFRAME_H4,  300),   # ~50 days

                ("1d",  mt5.TIMEFRAME_D1,  200),   # ~200 days

            ]

            _primary_key = self.timeframe.lower()



            # Seed with primary TF bars (already fetched by the main poll loop)

            tf_data: dict = {}

            if primary_df is not None:

                tf_data[_primary_key] = primary_df



            for tf_str, mt5_tf, n_bars in _ALL_TFS:

                if tf_str == _primary_key:

                    continue   # already seeded above

                try:

                    rates = mt5.copy_rates_from_pos(self.asset, mt5_tf, 0, n_bars)

                    if rates is not None and len(rates) >= 20:

                        tf_data[tf_str] = _rates_to_df(rates)

                        logger.debug("MTF %s: %d bars", tf_str, len(rates))

                    else:

                        logger.warning("MTF %s: no data (%s)", tf_str, mt5.last_error())

                except Exception as exc:

                    logger.warning("MTF %s fetch error: %s", tf_str, exc)



            return tf_data or None



        except Exception as exc:

            logger.warning("_fetch_mtf_candles error: %s", exc)

            return {self.timeframe.lower(): primary_df} if primary_df is not None else None



    # â”€â”€ Signal evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def _qualifies(self, row: dict) -> bool:

        grade = (row.get("grade") or "F").upper()

        score = row.get("confluence_score") or 0.0

        effective_min = self.min_score + _LEARNING["score_adjust"]

        _LEARNING["total_seen"] += 1



        # Grade-level stat tracking

        gs = _LEARNING["grade_stats"].setdefault(grade, {"seen": 0, "acted": 0})

        gs["seen"] += 1



        qualifies = (

            GRADE_RANK.get(grade, 0) >= GRADE_RANK.get(self.min_grade, 3)

            and score >= effective_min

        )

        if qualifies:

            gs["acted"] += 1

        return qualifies



    def _qualifies_result(self, result) -> bool:

        """Check if a live PrometheusResult meets the trading thresholds."""

        if not result or not result.confluence:

            return False

        c     = result.confluence

        grade = (c.grade or "F").upper()

        score = c.total or 0.0

        direction = c.direction or "sideways"



        if direction == "sideways":

            return False



        effective_min = self.min_score + _LEARNING["score_adjust"]

        _LEARNING["total_seen"] += 1

        gs = _LEARNING["grade_stats"].setdefault(grade, {"seen": 0, "acted": 0})

        gs["seen"] += 1

        self._pending_grade        = grade   # remember for win/loss attribution at close
        self._pending_regime_name  = getattr(getattr(self._current_regime,  "regime",  None), "value", "unknown")
        self._pending_session_name = getattr(getattr(self._current_session, "session", None), "value", "unknown")



        qualifies = (

            GRADE_RANK.get(grade, 0) >= GRADE_RANK.get(self.min_grade, 3)

            and score >= effective_min

        )

        # -- SELL/bearish direction score premium (adaptive, sample-gated) -----
        # Only apply a SELL premium when there are >= 20 SELL outcomes with WR < 55%.
        # With < 20 samples the gate creates a self-starving loop (blocks SELL → no data
        # → WR stays low → gate stays on). Gate is lifted until data is statistically
        # meaningful. When active, premium is reduced to +3 (was +7, too aggressive).
        if qualifies and direction in ("bearish", "short"):
            _sell_ds = _LEARNING.get("direction_stats", {}).get("SELL", {})
            _sell_w  = _sell_ds.get("wins",   0)
            _sell_l  = _sell_ds.get("losses", 0)
            _sell_n  = _sell_w + _sell_l
            _sell_wr = _sell_w / _sell_n if _sell_n > 0 else 1.0
            if _sell_n >= 20 and _sell_wr < 0.55:
                _sell_floor = effective_min + 3.0
                if score < _sell_floor:
                    logger.info(
                        "[Learning] SELL premium (+3): need %.1f; got %.1f (SELL WR=%.0f%% n=%d) -- skipped",
                        _sell_floor, score, _sell_wr * 100, _sell_n,
                    )
                    qualifies = False
            else:
                logger.debug(
                    "[Learning] SELL premium inactive: SELL n=%d WR=%.0f%% (need n>=20 and WR<55%%)",
                    _sell_n, _sell_wr * 100,
                )

        # -- Regime kill-entries gate (NEWS_VOLATILITY / DEAD_LIQUIDITY) ----------
        # Hard-blocks entries when the market regime is unsafe regardless of score.
        # Also enforces regime-specific score floor premiums (COMPRESSION, etc.)
        if qualifies and ENABLE_REGIME_CLASSIFIER and self._current_regime is not None:
            if self._current_regime.kill_entries:
                logger.warning(
                    "[Regime] %s -- all new entries suppressed (kill_entries=True)",
                    self._current_regime.regime.value,
                )
                qualifies = False
            elif self._current_regime.score_floor_premium > 0:
                _floor = effective_min + self._current_regime.score_floor_premium
                if score < _floor:
                    logger.info(
                        "[Regime] %s -- score %.1f below regime-floor %.1f (premium +%.0f)",
                        self._current_regime.regime.value, score, _floor,
                        self._current_regime.score_floor_premium,
                    )
                    qualifies = False

        # -- Asian session score floor: reduced sizing, +5 pt premium ----------------
        # Asian session is open but low-volatility. A modest score floor ensures
        # only structurally sound setups (confluence does the selectivity work).
        if qualifies and self._current_session is not None:
            if self._current_session.session.value == "asian":
                _asian_floor = effective_min + 5.0
                if score < _asian_floor:
                    logger.info(
                        "[session/Asian] Score %.1f below Asian floor %.1f (+5 premium) -- skipping",
                        score, _asian_floor,
                    )
                    qualifies = False
                else:
                    self._htf_lot_mult = min(self._htf_lot_mult, 0.5)
                    logger.info(
                        "[session/Asian] Entry allowed -- lot scalar capped at %.2f",
                        self._htf_lot_mult,
                    )

        # -- HTF alignment gate (dynamic, relative to primary TF)

        # All TFs ABOVE the primary must agree with the signal direction.

        # TFs below the primary are excluded here and handled in the LTF gate.

        if qualifies and result.mtf and result.mtf.biases:

            _signal_bias = "bullish" if direction in ("bullish", "long") else "bearish"

            _primary_rank = TF_RANK.get(self.timeframe.lower(), 6)

            _htf_biases = [b for b in result.mtf.biases

                           if TF_RANK.get(b.timeframe.lower(), 0) > _primary_rank]

            _htf_aligned = [b for b in _htf_biases if b.bias == _signal_bias]

            if len(_htf_biases) >= 1:

                if ENABLE_REGIME_CLASSIFIER and _ARCH_MODULES_OK:

                    # Probabilistic: partial HTF alignment → reduced lot, not a hard block

                    _align_ratio, _htf_mult = htf_alignment_score(_htf_biases, _signal_bias)

                    if _htf_mult <= 0.0:

                        # Strongly opposed (≤35% aligned by weight) – block

                        _mismatched = [b.timeframe for b in _htf_biases if b.bias != _signal_bias]

                        logger.info(

                            "[MTF gate] Blocked %s: HTF fully opposed %s (align=%.0f%%)",

                            _signal_bias, _mismatched, _align_ratio * 100,

                        )

                        qualifies = False

                    elif _htf_mult < 1.0:

                        # Partial alignment – allow with proportionally reduced lot

                        logger.info(

                            "[MTF gate] Partial HTF alignment %.0f%% — lot×%.2f",

                            _align_ratio * 100, _htf_mult,

                        )

                        self._htf_lot_mult = _htf_mult

                    else:

                        self._htf_lot_mult = 1.0

                else:

                    # Legacy hard veto – any disagreeing HTF blocks the trade

                    if len(_htf_aligned) < len(_htf_biases):

                        _mismatched = [b.timeframe for b in _htf_biases if b.bias != _signal_bias]

                        logger.info(

                            "[MTF gate] Blocked %s entry: HTF misaligned %s (%d/%d aligned).",

                            _signal_bias, _mismatched, len(_htf_aligned), len(_htf_biases),

                        )

                        qualifies = False



        # -- 1H BOS invalidation gate ─────────────────────────────────────────
        # When the 1H (or 30M) timeframe has made a BOS *against* the 4H primary
        # signal direction in the current session, the 4H bias is contested: price
        # has broken the nearest 1H swing in the opposite direction, meaning
        # institutional order flow on the intermediate TF has shifted.
        # A professional ICT/SMC trader stops adding in the primary direction and
        # waits for the 1H reclaim to fail before re-entering.
        # Rule: if the closest sub-primary TF above 15M (i.e. 1H or 30M) has a
        #   BOS event opposing the signal within the last 6 bars → soft-block new
        #   entry unless the signal is Grade A (score >= 90) AND it's a pullback
        #   entry (price inside an OB or FVG — zone_only mode).
        if qualifies and result.mtf and result.mtf.biases:
            _inv_sig  = "bullish" if direction in ("bullish", "long") else "bearish"
            _opp_bos  = "bearish" if _inv_sig == "bullish" else "bullish"
            # Find 1H and 30M biases with their structure
            _itf_ranks = (5, 4)  # 1H=5, 30M=4
            _contested = False
            for _itf_b in result.mtf.biases:
                _itf_r = TF_RANK.get(_itf_b.timeframe.lower(), 0)
                if _itf_r not in _itf_ranks:
                    continue
                if _itf_b.structure is None:
                    continue
                # Check if recent BOS on this TF opposes signal
                _bos_list = getattr(_itf_b.structure, "bos_events", []) or []
                if _bos_list:
                    _last_bos = _bos_list[-1]
                    _bos_dir  = getattr(_last_bos, "direction", None)
                    _bos_bar  = getattr(_last_bos, "bar_index", 0)
                    _total_bars = len(result.ms.bos_events) if result.ms else 200
                    _bars_ago = _total_bars - _bos_bar
                    # Opposing BOS within last 6 bars = active invalidation
                    if isinstance(_bos_dir, str) and _bos_dir.lower() == _opp_bos and _bars_ago <= 6:
                        _contested = True
                        logger.info(
                            "[BOS invalidation] %s %s BOS %d bars ago contests %s signal"
                            " — 4H bias contested on intermediate TF",
                            _itf_b.timeframe, _bos_dir, _bars_ago, _inv_sig,
                        )
                        break
            if _contested:
                # Drought override: if no trade in TRADE_DROUGHT_POLLS in an active session,
                # downgrade BOS invalidation from a hard block to a lot reduction.
                # This ensures the bot always finds at least one entry per session window.
                _polls_no_trade = self._poll_count - self._last_entry_poll
                _drought_active = _polls_no_trade >= TRADE_DROUGHT_POLLS
                # Allow Grade A score>=90 in zone_only, OR any Grade A score>=85 in drought
                _zone_ok   = (self.entry_mode or "").lower() == "zone_only"
                _hard_pass = (score >= 90.0 and _zone_ok) or (score >= 85.0 and _drought_active)
                if _hard_pass:
                    self._htf_lot_mult = min(self._htf_lot_mult, 0.60)
                    logger.info(
                        "[BOS invalidation] Contested but allowed: score=%.1f drought=%s -> lot x0.60",
                        score, _drought_active,
                    )
                else:
                    logger.info(
                        "[BOS invalidation] Blocked %s: score=%.1f drought=%s"
                        " (need score>=90 zone_only, or score>=85 after %d poll drought)",
                        _inv_sig, score, _drought_active, TRADE_DROUGHT_POLLS,
                    )
                    qualifies = False



        # -- LTF momentum trap gate (dynamic, relative to primary TF)

        # The two TFs immediately BELOW the primary show short-term momentum.

        # "trap"           = all LTFs counter-trend  → block (momentum surge against trade)

        # "both_confirmed" = all LTFs aligned        → strong timing confirmation

        # "one_counter"    = mixed                   → allow (brief wick, not a surge)

        # "unknown"        = fewer than 2 LTFs available below primary

        if result.mtf and result.mtf.biases:

            _signal_bias = "bullish" if direction in ("bullish", "long") else "bearish"

            _primary_rank = TF_RANK.get(self.timeframe.lower(), 6)

            _ltf_biases = sorted(

                [b for b in result.mtf.biases

                 if TF_RANK.get(b.timeframe.lower(), 0) < _primary_rank],

                key=lambda b: TF_RANK.get(b.timeframe.lower(), 0),

                reverse=True,   # highest-rank first so we take the 2 closest to primary

            )[:2]               # only the two TFs immediately below primary

            _ltf_aligned = [b for b in _ltf_biases if b.bias == _signal_bias]

            _ltf_counter = [b for b in _ltf_biases if b.bias != _signal_bias]
            # Store raw biases for pending-limit LTF monitor (direction-agnostic)
            self._current_ltf_biases = [b.bias for b in _ltf_biases]

            if len(_ltf_biases) >= 2:

                if len(_ltf_counter) == len(_ltf_biases):

                    self._last_ltf_state = "trap"

                    if qualifies:

                        logger.info(

                            "[MTF gate] Blocked %s entry: LTF momentum trap -- "

                            "both %s counter-trend. Waiting for them to turn.",

                            _signal_bias, [b.timeframe for b in _ltf_biases],

                        )

                        qualifies = False

                elif len(_ltf_aligned) == len(_ltf_biases):

                    self._last_ltf_state = "both_confirmed"

                    if qualifies:

                        # both_confirmed: all LTFs agree with HTF direction.
                        # Stats: 42W/46L (48% WR) vs 90% for one_counter.
                        # However hard-blocking eliminates Grade A trend-aligned setups
                        # and starves data. Compromise: allow Grade A (score >= 85)
                        # with lot x0.7 to compensate for potentially extended entry.
                        # Grade B/C in both_confirmed remain blocked.
                        _bc_min_score = 85.0
                        if score >= _bc_min_score:
                            self._htf_lot_mult = min(self._htf_lot_mult, 0.70)
                            logger.info(
                                "[MTF gate] both_confirmed %s: Grade %s score %.1f >= %.0f -- allowed at lot x0.70",
                                _signal_bias, grade, score, _bc_min_score,
                            )
                        else:
                            logger.info(
                                "[MTF gate] Blocked %s entry: LTF both_confirmed -- "
                                "score %.1f below A-grade floor %.0f %s.",
                                _signal_bias, score, _bc_min_score,
                                [b.timeframe for b in _ltf_biases],
                            )
                            qualifies = False

                else:

                    self._last_ltf_state = "one_counter"

            else:

                self._last_ltf_state = "unknown"

        else:

            self._last_ltf_state = "unknown"



        # ── Micro-LTF momentum surge gate (15M / 5M / 1M) ─────────────────────
        # The primary LTF gate above checks only 1H + 30M (the two TFs
        # immediately below 4H).  When those agree with the signal direction
        # the gate passes, but 15M/5M/1M may already be surging the other way —
        # the exact scenario where SELL entries run into a rising market.
        # Rule: if ALL available micro-LTFs (rank < 4) oppose the signal:
        #   • Trend Exhaustion regime → hard block + set _ltf_reversal_pending so
        #     the poll loop can enter in the LTF direction (reversal momentum entry)
        #   • Any other regime        → allow but cut lot to ×0.50
        if qualifies and result.mtf and result.mtf.biases:
            _micro_sig = "bullish" if direction in ("bullish", "long") else "bearish"
            _micro_tfs = [
                b for b in result.mtf.biases
                if TF_RANK.get(b.timeframe.lower(), 0) < 4   # 15M, 5M, 1M
            ]
            _micro_counter = [b for b in _micro_tfs if b.bias != _micro_sig]
            if len(_micro_tfs) >= 2 and len(_micro_counter) == len(_micro_tfs):
                _reg_val = (
                    self._current_regime.regime.value
                    if ENABLE_REGIME_CLASSIFIER and self._current_regime
                    else ""
                )
                _avg_micro_score = (
                    sum(abs(b.score) for b in _micro_counter) / len(_micro_counter)
                    if _micro_counter else 0.0
                )
                _rev_dir = "bullish" if _micro_sig == "bearish" else "bearish"
                if "exhaustion" in _reg_val.lower():
                    logger.info(
                        "[MTF gate] Micro-LTF surge BLOCKED %s: %d/%d micro-TFs counter %s "
                        "in Trend Exhaustion (avg_score=%.2f) — reversal momentum active, "
                        "queuing %s reversal entry",
                        _micro_sig, len(_micro_counter), len(_micro_tfs),
                        [b.timeframe for b in _micro_counter],
                        _avg_micro_score, _rev_dir,
                    )
                    qualifies = False
                    # Signal to _poll_live: enter in LTF direction instead
                    if _avg_micro_score >= 0.60:
                        self._ltf_reversal_pending = {
                            "direction":  _rev_dir,
                            "avg_score":  _avg_micro_score,
                            "tfs":        [b.timeframe for b in _micro_counter],
                        }
                else:
                    self._htf_lot_mult = min(self._htf_lot_mult, 0.50)
                    logger.info(
                        "[MTF gate] Micro-LTF surge %s: %d/%d micro-TFs counter in %s — lot x0.50",
                        _micro_sig, len(_micro_counter), len(_micro_tfs),
                        _reg_val or "unknown",
                    )



        # -- Countertrend gate (uses regime allow_countertrend flag) --------------

        # Determines the dominant direction from 1D/1W HTF bias (or the direction

        # of currently-open positions as fallback).  If the signal opposes that:

        #   allow_countertrend=False  -> hard block

        #   allow_countertrend=True   -> require score >= threshold + CT_SCORE_BONUS

        if qualifies and ENABLE_REGIME_CLASSIFIER and self._current_regime is not None:

            _sig_ct = "bullish" if direction in ("bullish", "long") else "bearish"

            _dominant_dir: str | None = None

            if result.mtf and result.mtf.biases:

                _htf_day = [

                    b for b in result.mtf.biases

                    if b.timeframe.lower() in ("1d", "1w")

                    and b.bias not in ("unknown", "sideways")

                ]

                if _htf_day:

                    _dominant_dir = max(_htf_day, key=lambda b: abs(b.score)).bias

            if _dominant_dir is None and self._cached_open_directions:

                if len(self._cached_open_directions) == 1:

                    _mt5_d = next(iter(self._cached_open_directions))

                    _dominant_dir = "bullish" if _mt5_d == "BUY" else "bearish"

            if _dominant_dir and _dominant_dir != _sig_ct:

                self._current_trade_is_ct = True   # mark as countertrend

                if not self._current_regime.allow_countertrend:

                    logger.info(

                        "[Regime CT] Blocked countertrend %s -- regime=%s forbids it "

                        "(dominant=%s)",

                        _sig_ct, self._current_regime.regime.value, _dominant_dir,

                    )

                    qualifies = False

                else:

                    # Extra premium for trend_exhaustion: exhaustion != reversal.

                    # The regime allows CT but the bar is much higher here.

                    _ct_required = effective_min + CT_SCORE_BONUS

                    try:

                        from regime_classifier import Regime as _R

                        if self._current_regime.regime == _R.TREND_EXHAUSTION:

                            _ct_required += CT_EXHAUSTION_EXTRA

                            logger.info(

                                "[Regime CT] Exhaustion regime -- extra %.0f pt premium "

                                "(CT requires %.1f total, dominant=%s)",

                                CT_EXHAUSTION_EXTRA, _ct_required, _dominant_dir,

                            )

                    except Exception:

                        pass

                    if score < _ct_required:

                        logger.info(

                            "[Regime CT] Countertrend %s rejected -- score %.1f < %.1f "

                            "(dominant=%s)",

                            _sig_ct, score, _ct_required, _dominant_dir,

                        )

                        qualifies = False

                    else:

                        logger.info(

                            "[Regime CT] Countertrend %s ALLOWED -- score %.1f >= %.1f "

                            "(regime=%s, dominant=%s) -- CT_LOT_MULT=%.2f",

                            _sig_ct, score, _ct_required,

                            self._current_regime.regime.value, _dominant_dir, CT_LOT_MULT,

                        )

            else:

                self._current_trade_is_ct = False  # trend-continuation



        # -- Direction-flip guard -------------------------------------------------

        # Requires DIRECTION_FLIP_MIN_CONFIRMS consecutive polls showing the new

        # direction before the bot will act on a flip.  The counter is updated in

        # _poll (once per poll) to avoid double-counting from pyramid + main calls.

        if qualifies and self._last_confirmed_direction:

            _sig_flip = "bullish" if direction in ("bullish", "long") else "bearish"

            if _sig_flip != self._last_confirmed_direction:

                if self._direction_flip_polls < DIRECTION_FLIP_MIN_CONFIRMS:

                    logger.info(

                        "[Direction guard] Flip %s->%s needs %d confirms (%d/%d) -- blocked",

                        self._last_confirmed_direction, _sig_flip,

                        DIRECTION_FLIP_MIN_CONFIRMS,

                        self._direction_flip_polls, DIRECTION_FLIP_MIN_CONFIRMS,

                    )

                    qualifies = False

                else:

                    logger.info(

                        "[Direction guard] Flip %s->%s confirmed (%d polls) -- allowed",

                        self._last_confirmed_direction, _sig_flip,

                        self._direction_flip_polls,

                    )



        if qualifies:

            gs["acted"] += 1

        return qualifies



    # â”€â”€ Trade execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def _execute(self, row: dict) -> str:

        direction = (row.get("direction") or "").lower()

        is_long   = direction in ("bullish", "long")

        is_short  = direction in ("bearish", "short")



        if not is_long and not is_short:

            return f"direction='{direction}' is not actionable â€” skipped"





        # Record the direction we actually traded -- used by direction-flip guard

        _exec_dir = direction  # already lower-cased above

        if _exec_dir in ("bullish", "bearish"):

            self._last_confirmed_direction = _exec_dir

            self._direction_flip_pending   = ""

            self._direction_flip_polls     = 0

        # â”€â”€ Dry-run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        if self.dry_run or not MT5_AVAILABLE:

            side = "BUY" if is_long else "SELL"

            msg  = (f"[DRY RUN] Would {side} {self.asset} "

                    f"| grade={row.get('grade')} "

                    f"score={row.get('confluence_score', 0):.0f} "

                    f"dir={direction}")

            logger.info(msg)

            self._total_trades += 1

            return msg



        # â”€â”€ Live MT5 order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        tick = mt5.symbol_info_tick(self.asset)

        if not tick:

            return f"No tick data for {self.asset}"



        price = tick.ask if is_long else tick.bid

        sup   = row.get("nearest_support")

        res   = row.get("nearest_resistance")



        # SL/TP: use stored S/R levels; fall back to % of price

        buffer = price * 0.002   # 0.2% buffer beyond zone

        if is_long:

            sl = (float(sup) - buffer) if (sup and sup < price) else price * 0.997

            sl_dist = abs(price - sl)

            tp_min = price + sl_dist * RR_MIN_LONG          # enforce 1:2 minimum

            tp_sr  = float(res) if (res and res > tp_min) else None

            tp = tp_sr if tp_sr else tp_min

        else:

            sl = (float(res) + buffer) if (res and res > price) else price * 1.003

            sl_dist = abs(sl - price)

            tp_min = price - sl_dist * RR_MIN_SHORT          # enforce 1:2 minimum

            tp_sr  = float(sup) if (sup and sup < tp_min) else None

            tp = tp_sr if tp_sr else tp_min



        sl_dist = abs(price - sl)

        lot     = self._calc_lot(sl_dist)



        # Filling mode

        sym  = mt5.symbol_info(self.asset)

        fill = mt5.ORDER_FILLING_IOC

        if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK):

            fill = mt5.ORDER_FILLING_FOK



        req = {

            "action":       mt5.TRADE_ACTION_DEAL,

            "symbol":       self.asset,

            "volume":       lot,

            "type":         mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,

            "price":        price,

            "sl":           round(sl, 5),

            "tp":           round(tp, 5),

            "deviation":    20,

            "magic":        777_000,

            "comment":      f"Prom#{row['id']}",

            "type_time":    mt5.ORDER_TIME_GTC,

            "type_filling": fill,

        }



        result = mt5.order_send(req)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:

            code = result.retcode if result else "None"

            err  = result.comment if result else "no response"

            return f"Order FAILED retcode={code} ({err})"



        self._total_trades += 1

        msg = (f"Order executed ticket={result.order} "

               f"{'BUY' if is_long else 'SELL'} {lot} lot @ {price:.4f} "

               f"SL={sl:.4f} TP={tp:.4f}")

        logger.info(msg)

        return msg



    def _execute_from_result(self, result, label: str = "live") -> str:

        """Open a dual-TP split position from a live PrometheusResult.



        Flow

        ----

        1. Zone filter  â€” price must be within ZONE_ATR_THRESHOLD Ã— ATR of a

           fresh OB or S/R zone. If not in zone, place a Buy/SellLimit at the

           zone edge instead and return early.

        2. Market entry â€” if in zone, send TWO market orders (half lot each):

           â€¢ Leg 1 (TP1): closes at 1:2 R:R â€” locks in 50 % of profit quickly.

           â€¢ Leg 2 (TP2): closes at 1:5 R:R â€” trailing stop protects the rest.

        3. OB learning  â€” record which OB direction led to this entry.

        """

        confluence = result.confluence

        direction  = (confluence.direction if confluence else "sideways").lower()

        is_long  = direction == "bullish"

        is_short = direction == "bearish"



        if not is_long and not is_short:

            return f"direction='{direction}' is not actionable â€” skipped"





        # Record the direction we actually traded (only for trend-continuation).

        # Countertrend scalps are temporary deviations -- suppress direction state update

        # so they cannot trigger further CT follow-up trades.

        _exec_dir_r   = direction  # already lower-cased above

        _ct_lot_scale = CT_LOT_MULT if self._current_trade_is_ct else 1.0

        if _exec_dir_r in ("bullish", "bearish") and not self._current_trade_is_ct:

            self._last_confirmed_direction = _exec_dir_r

            self._direction_flip_pending   = ""

            self._direction_flip_polls     = 0

        # Reset CT flag immediately so it only applies to this single entry.

        self._current_trade_is_ct = False



        atr   = (result.ms.current_atr if result.ms and result.ms.current_atr else 0.0)

        grade = confluence.grade if confluence else "?"

        score = confluence.total if confluence else 0.0

        smc   = result.smc



        # Execution quality gate (spread vs ATR check)

        if ENABLE_EXECUTION_QUALITY and self._exec_quality and MT5_AVAILABLE and not self.dry_run:

            _sym_info = mt5.symbol_info(self.asset)

            _tol = getattr(self._current_session, "spread_tolerance", 1.0)

            _qr = self._exec_quality.check(_sym_info, atr if atr > 0 else None,

                                           spread_tolerance_mult=_tol)

            _spread_pips = None

            if _sym_info:

                _spread_pips = round(_sym_info.spread * _sym_info.point * 10, 2)

            self._last_exec_quality_result = {

                "passed":       _qr.passes,

                "reason":       _qr.reason,

                "spread_pips":  _spread_pips,

            }

            if not _qr.passes:

                msg = f"[exec_quality] Entry blocked: {_qr.reason}"

                logger.info(msg)

                return msg



        # â”€â”€ S/R levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        sup = (result.sr.nearest_support.level

               if result.sr and result.sr.nearest_support else None)

        res = (result.sr.nearest_resistance.level

               if result.sr and result.sr.nearest_resistance else None)



        # â”€â”€ Find best fresh Order Block for the signal direction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        ob_zone: tuple[float, float] | None = None   # (low, high) of best OB

        # Use last-bar close as an approximate price for OB proximity sorting

        # (actual tick price not yet fetched).

        _approx_price = result.current_price or 0.0

        _atr_approx   = (result.ms.current_atr if result.ms and result.ms.current_atr else 0.0)

        if smc and smc.order_blocks:

            ob_dir = "bullish" if is_long else "bearish"

            fresh_obs = [

                ob for ob in smc.order_blocks

                if ob.direction == ob_dir and not ob.mitigated

            ]

            if fresh_obs:

                # Sort by distance to current price (closest first) then strength.

                # This ensures we pick the nearest relevant OB, not the historical

                # one that originated a move 200 pts away.

                fresh_obs.sort(key=lambda b: (

                    abs((b.high if is_long else b.low) - _approx_price),

                    -b.strength,

                ))

                # Only select the OB if it is within MAX_LIMIT_DISTANCE_ATR of

                # price, otherwise fall through to the S/R check below.

                for _ob in fresh_obs:

                    _ob_ref = _ob.high if is_long else _ob.low

                    if (_atr_approx == 0 or

                            abs(_ob_ref - _approx_price) <= MAX_LIMIT_DISTANCE_ATR * _atr_approx):

                        ob_zone = (_ob.low, _ob.high)

                        break



        # ── LTF Order Block override for limit-order placement ──────────────────────
        # Sell limits and buy limits use LTF OBs for precise entry price.
        # HTF OBs identify the zone; LTF OBs pinpoint where price retraces to.
        if is_short:
            _ltf_ob = self._find_ltf_ob("bearish", _approx_price, _atr_approx)
            if _ltf_ob:
                logger.info(
                    "[ob] SellLimit: overriding HTF OB %s with LTF bearish OB %.4f-%.4f",
                    ob_zone, _ltf_ob[0], _ltf_ob[1],
                )
                ob_zone = _ltf_ob
        elif is_long:
            _ltf_ob = self._find_ltf_ob("bullish", _approx_price, _atr_approx)
            if _ltf_ob:
                logger.info(
                    "[ob] BuyLimit: overriding HTF OB %s with LTF bullish OB %.4f-%.4f",
                    ob_zone, _ltf_ob[0], _ltf_ob[1],
                )
                ob_zone = _ltf_ob

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            side = "BUY" if is_long else "SELL"

            msg = (f"[DRY RUN] [{label}] Would {side} {self.asset} "

                   f"| grade={grade} score={score:.0f} | ATR={atr:.4f} "

                   f"| OB={'yes' if ob_zone else 'none'}")

            logger.info(msg)

            self._total_trades += 1

            return msg



        tick = mt5.symbol_info_tick(self.asset)

        if not tick:

            return f"No tick data for {self.asset}"



        price = tick.ask if is_long else tick.bid



        # â”€â”€ Zone proximity filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        zone_threshold = atr * ZONE_ATR_THRESHOLD if atr else price * 0.002

        in_zone = False

        zone_entry_price: float | None = None



        if ob_zone:

            ob_low, ob_high = ob_zone

            if is_long:

                # In zone when price is inside or just above the bullish OB

                in_zone = price <= ob_high + zone_threshold

                zone_entry_price = ob_high   # limit buy at top of OB

            else:

                # In zone when price is at or just below the bearish OB

                in_zone = price <= ob_high + zone_threshold

                zone_entry_price = ob_high   # limit sell at TOP of OB (retrace into supply)



            # If the OB is too far and not in zone, still check S/R â€” price may be

            # sitting at a resistance/support even if the OB is out of range.

            if not in_zone:

                if sup and is_long and abs(price - sup) <= zone_threshold:

                    logger.info("[zone] OB not in range â€” using S/R support %.4f instead", sup)

                    in_zone = True

                    ob_zone = None           # treat as S/R entry (market order)

                    zone_entry_price = None

                elif res and is_short and abs(price - res) <= zone_threshold:

                    logger.info("[zone] OB not in range â€” using S/R resistance %.4f instead", res)

                    in_zone = True

                    ob_zone = None

                    zone_entry_price = None

        elif sup and is_long:

            in_zone = abs(price - sup) <= zone_threshold

            zone_entry_price = sup

        elif res and is_short:

            in_zone = abs(price - res) <= zone_threshold

            zone_entry_price = res

        else:

            # No OB and no clear S/R â€” allow entry (signals without zone context)

            in_zone = True



        if not in_zone and zone_entry_price is not None:

            if self.entry_mode == "market_any":

                # â”€â”€ market_any mode: skip the limit, enter at market now â”€â”€â”€â”€

                # SL is placed outside the nearest S/R zone with a 0.15Ã—ATR buffer

                logger.info(

                    "[market_any] Not in OB zone but entry_mode=market_any â€” entering at market."

                )

                in_zone = True   # fall through to the market order block below

            else:

                # â”€â”€ zone_only mode (default): place a pending limit order â”€â”€â”€

                return self._place_limit_order(

                    is_long=is_long, zone_price=zone_entry_price,

                    sup=sup, res=res, atr=atr, price=price, grade=grade, score=score,

                )



        # â”€â”€ SL calculation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # LTF trap gate — don't place a limit when both lower TFs already oppose direction.
        # A BuyLimit placed while 30M+1H are both bearish will fill into downside momentum.
        _lim_opp = "bearish" if is_long else "bullish"
        if (len(self._current_ltf_biases) >= 2
                and all(b == _lim_opp for b in self._current_ltf_biases)):
            _side = "BuyLimit" if is_long else "SellLimit"
            return (
                f"[limit] {_side} @ {zone_price:.4f} skipped — LTF trap "
                f"(both LTFs {_lim_opp}, opposing entry direction). "
                f"Waiting for LTF alignment before placing."
            )

        MIN_SL_ATR = 1.0

        fallback_sl_dist = max(atr * MIN_SL_ATR, price * 0.003)

        buf = atr * 0.15 if atr else price * 0.002



        # In market_any mode, the OB may be far away â€” only use it as SL anchor

        # if it's within MAX_LIMIT_DISTANCE_ATR of current price, otherwise fall

        # back to nearest S/R or pure ATR distance.

        ob_too_far = (

            ob_zone is not None and atr and

            abs((ob_zone[0] if is_long else ob_zone[1]) - price) > MAX_LIMIT_DISTANCE_ATR * atr

        )

        near_ob_zone = ob_zone if (ob_zone and not ob_too_far) else None



        if is_long:

            # SL just below the OB low (or S/R support) with a small buffer

            sl_anchor = near_ob_zone[0] if near_ob_zone else sup

            sl_cand = (float(sl_anchor) - buf) if sl_anchor else price - fallback_sl_dist

            sl = min(sl_cand, price - atr * MIN_SL_ATR) if atr else sl_cand

        else:

            sl_anchor = near_ob_zone[1] if near_ob_zone else res

            sl_cand = (float(sl_anchor) + buf) if sl_anchor else price + fallback_sl_dist

            sl = max(sl_cand, price + atr * MIN_SL_ATR) if atr else sl_cand



        sl_dist = abs(price - sl)

        if sl_dist <= 0:

            return f"[{label}] Invalid SL distance â€” skipped"



        # â”€â”€ Dual TP levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        _tp_scale = (self._current_regime.tp_scalar
                     if self._current_regime is not None else 1.0)

        if is_long:

            tp1 = round(price + sl_dist * TP1_RR * _tp_scale, 5)   # 1:1 x tp_scale

            tp2 = round(price + sl_dist * TP2_RR * _tp_scale, 5)   # 1:3 x tp_scale

        else:

            tp1 = round(price - sl_dist * TP1_RR * _tp_scale, 5)

            tp2 = round(price - sl_dist * TP2_RR * _tp_scale, 5)



        total_lot = self._calc_lot(sl_dist)

        # Grade-based risk multiplier: Grade A gets 1.5x, B 1.0x, C 0.8x.
        # Score-based boost: score >= 90 adds an additional 1.20x on top of grade mult.
        # This compounds capital toward the highest-conviction setups automatically.
        _grade_risk = {"A": 1.20, "B": 1.00, "C": 0.80}.get(grade or "B", 1.00)
        if score is not None and score >= 90.0:
            _grade_risk = round(_grade_risk * 1.20, 3)   # score 90+ = extra boost
        if abs(_grade_risk - 1.0) > 0.01:
            total_lot = max(0.01, round(total_lot * _grade_risk, 2))
            logger.info("[lot] Grade %s risk x%.2f -> %.2f lots", grade, _grade_risk, total_lot)

        # Daily profit protection: once daily target hit, scale back to 0.50x
        # to preserve the day's gains while still participating in strong setups.
        if self._daily_profit_protect:
            total_lot = max(0.01, round(total_lot * 0.50, 2))
            logger.info("[lot] Daily profit protect active -> lot x0.50 = %.2f", total_lot)

        # Apply CT lot reduction first (countertrend trades use smaller size)

        if abs(_ct_lot_scale - 1.0) > 0.01:

            total_lot = max(0.01, round(total_lot * _ct_lot_scale, 2))

            logger.info("[CT] Countertrend entry -- lot scaled by %.2f -> %.2f lots",

                        _ct_lot_scale, total_lot)

        # Apply HTF alignment + regime lot scalars

        _lot_scale = self._htf_lot_mult

        if ENABLE_REGIME_CLASSIFIER and self._current_regime is not None:

            _lot_scale *= self._current_regime.lot_scalar

        # lot_scalar=0.0 means the regime wants to suppress all entries (e.g. Mean
        # Reversion during a trending session). Return early — do not place order.
        if _lot_scale <= 0.0:
            msg = (
                f"[regime] Entry suppressed: regime={getattr(self._current_regime.regime, 'value', '?')} "
                f"lot_scalar={self._current_regime.lot_scalar:.2f}"
            )
            logger.info(msg)
            return msg

        if abs(_lot_scale - 1.0) > 0.01:

            total_lot = max(0.01, round(total_lot * _lot_scale, 2))

            logger.info("[lot] Scaled by HTF+regime mult=%.2f -> %.2f lots", _lot_scale, total_lot)

        self._htf_lot_mult = 1.0   # reset for next signal

        # ── Small-account adaptive execution ───────────────────────────────
        # SCALAR is 1.0 -- 2% risk rule sizes correctly; no artificial halving.
        # Instead protect via: (a) SL-width gate, (b) single-leg TP1 only.
        _acct_info = self._mt5_account()
        _cur_bal   = float(_acct_info.get("balance", 0) or 0)
        _is_small  = 0 < _cur_bal < SMALL_ACCOUNT_THRESHOLD

        # SL width gate: block wide-SL market entries that get stopped before
        # the trade can breathe. These are typically early entries ahead of a sweep.
        if _is_small and atr and atr > 0:
            _sl_check = abs(price - sl) if sl != 0 else 0.0
            if _sl_check > atr * SMALL_ACCOUNT_MAX_SL_ATR:
                msg = (
                    f"[small_acct] SL too wide ({_sl_check:.1f} pts > "
                    f"{atr * SMALL_ACCOUNT_MAX_SL_ATR:.1f} max = {SMALL_ACCOUNT_MAX_SL_ATR}xATR) "
                    f"-- entry skipped. Balance=${_cur_bal:.2f}"
                )
                logger.info(msg)
                return msg

        sym = mt5.symbol_info(self.asset)

        min_lot  = sym.volume_min  if sym else 0.01

        step_lot = sym.volume_step if sym else 0.01

        # Each leg = half; round down to step; at least min_lot
        # Small accounts: single-leg uses the full lot for proper 2% risk sizing
        half_lot        = max(min_lot, round(total_lot / 2 / step_lot) * step_lot)
        _single_leg_lot = max(min_lot, round(total_lot / step_lot) * step_lot)

        # ── TP1 equity-scaled minimum profit floor ──────────────────────────────
        # TP1_MIN_USD is calibrated for large accounts. On small accounts it pushes
        # TP1 far beyond a realistic 1:2 target, meaning TP1 almost never fills and
        # SL hits dominate. Scale the floor as a % of equity instead: 0.25% of
        # equity, capped at the full TP1_MIN_USD constant. On $600 → $1.50 minimum.
        # On $50k → $12.50 cap (original value). Guarantees TP1 stays reachable.
        if sym and sym.trade_tick_size > 0:
            _pnl_per_pt = half_lot * (sym.trade_tick_value / sym.trade_tick_size)
            if _pnl_per_pt > 0:
                _acct_eq = 0.0
                if MT5_AVAILABLE:
                    _ai = mt5.account_info()
                    _acct_eq = float(_ai.equity) if _ai else 0.0
                # 0.25% of equity, capped at TP1_MIN_USD constant
                _tp1_floor = min(TP1_MIN_USD, _acct_eq * 0.0025) if _acct_eq > 0 else TP1_MIN_USD
                _tp1_min_dist = _tp1_floor / _pnl_per_pt
                if is_long:
                    tp1 = max(tp1, round(price + _tp1_min_dist, 5))
                else:
                    tp1 = min(tp1, round(price - _tp1_min_dist, 5))
                logger.info(
                    "[TP1] $%.2f floor (equity-scaled) -> %.1f pts -> TP1=%.5f",
                    _tp1_floor, _tp1_min_dist, tp1,
                )


        fill = mt5.ORDER_FILLING_IOC

        if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK):

            fill = mt5.ORDER_FILLING_FOK



        order_type = mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL

        # -- Entry-quality zone positioning -----------------------------------------
        # Compute fill depth within the OB so _update_learning can attribute outcomes
        # to zone depth, zone type, and SL/ATR tightness.
        # deep_fill  = entered deep into zone (max room before SL) - ideal
        # shallow_fill = price barely tapped zone edge - higher noise risk
        _eq_zone_pos  = 'no_zone'
        _eq_zone_type = 'no_zone'
        if ob_zone:
            _ob_low, _ob_high = ob_zone
            _span = _ob_high - _ob_low
            if _span > 0:
                _fill_r = (price - _ob_low) / _span  # 0.0=OB bottom, 1.0=OB top
                # For BUY: ideal entry near OB bottom (fill_r low) = deep_fill
                # For SELL: ideal entry near OB top  (fill_r high) = deep_fill
                _depth = (1.0 - _fill_r) if is_long else _fill_r
                if _depth >= 0.67:
                    _eq_zone_pos = 'deep_fill'
                elif _depth >= 0.33:
                    _eq_zone_pos = 'mid_fill'
                else:
                    _eq_zone_pos = 'shallow_fill'
            else:
                _eq_zone_pos = 'mid_fill'
            _eq_zone_type = 'ob'
        elif in_zone:
            _eq_zone_pos  = 'sr_fill'
            _eq_zone_type = 'sr'

        _eq_sl_atr = round(abs(price - sl) / atr, 3) if atr and atr > 0 else None

        # Zone quality gate (data-driven, activates only when n >= sample threshold)
        _zp_stats = _LEARNING.get('zone_position_stats', {})
        _shallow_rec = _zp_stats.get('shallow_fill', {})
        _shallow_n   = _shallow_rec.get('wins', 0) + _shallow_rec.get('losses', 0)
        if _eq_zone_pos == 'shallow_fill' and _shallow_n >= 20:
            _shallow_wr = _shallow_rec.get('wins', 0) / _shallow_n
            if _shallow_wr < 0.45:
                _need_sc = score + 5.0
                if score < _need_sc:
                    msg = (f'[zone_quality] shallow fill blocked: '
                           f'WR={_shallow_wr:.0%} n={_shallow_n} '
                           f'score={score:.0f} < required {_need_sc:.0f}')
                    logger.info(msg)
                    return msg

        _nz_rec = _zp_stats.get('no_zone', {})
        _nz_n   = _nz_rec.get('wins', 0) + _nz_rec.get('losses', 0)
        if _eq_zone_pos == 'no_zone' and _nz_n >= 30:
            _nz_wr = _nz_rec.get('wins', 0) / _nz_n
            if _nz_wr < 0.45:
                logger.warning('[zone_quality] no_zone entries WR=%.0f%% (n=%d) -- '
                               'consider tightening zone filter', _nz_wr * 100, _nz_n)

        # SL tightness gate: tight SL < 0.8x ATR historically stopped by noise -> add buffer
        _sa_stats  = _LEARNING.get('sl_atr_stats', {})
        _tight_rec = _sa_stats.get('tight', {})
        _tight_n   = _tight_rec.get('wins', 0) + _tight_rec.get('losses', 0)
        if _eq_sl_atr is not None and _eq_sl_atr < 0.8 and _tight_n >= 20:
            _tight_wr = _tight_rec.get('wins', 0) / _tight_n
            if _tight_wr < 0.45:
                _buf = 0.15 * atr
                logger.info('[zone_quality] tight SL WR=%.0f%% -> adding %.4f buffer',
                            _tight_wr * 100, _buf)
                sl = round(sl - _buf, 5) if is_long else round(sl + _buf, 5)




        # â”€â”€ Leg 1: TP1 â€” close 50 % at 1:2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        req1 = {

            "action":       mt5.TRADE_ACTION_DEAL,

            "symbol":       self.asset,

            "volume":       _single_leg_lot if _is_small else half_lot,

            "type":         order_type,

            "price":        price,

            "sl":           round(sl, 5),

            "tp":           tp1,

            "deviation":    20,

            "magic":        777_000,

            "comment":      f"Prom-TP1-{label[:4]}",

            "type_time":    mt5.ORDER_TIME_GTC,

            "type_filling": fill,

        }

        r1 = mt5.order_send(req1)

        if r1 is None or r1.retcode != mt5.TRADE_RETCODE_DONE:

            code = r1.retcode if r1 else "None"

            return f"[{label}] Leg1 FAILED retcode={code}"



        # Tag this ticket with the LTF state so _update_learning can record it

        self._ltf_entry_state[r1.order] = self._last_ltf_state

        self._entry_grade[r1.order]     = getattr(self, "_pending_grade",       "F")
        self._entry_regime[r1.order]    = getattr(self, "_pending_regime_name",  "unknown")
        self._entry_session[r1.order]   = getattr(self, "_pending_session_name", "unknown")
        # Entry-quality positioning tags (zone position, zone type, SL/ATR, score)
        self._entry_zone_pos[r1.order]  = _eq_zone_pos
        self._entry_zone_type[r1.order] = _eq_zone_type
        self._entry_sl_atr[r1.order]    = _eq_sl_atr
        self._entry_score[r1.order]     = score

        # â”€â”€ Record open trade in DB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        try:

            save_trade({

                "trade_id":       f"live_{r1.order}",

                "source":         "live",

                "asset":          self.asset,

                "timeframe":      self.timeframe,

                "direction":      "BUY" if is_long else "SELL",

                "entry_price":    price,

                "sl_price":       round(sl, 5),

                "tp_price":       tp1 if _is_small else tp2,

                "size":           _single_leg_lot if _is_small else half_lot * 2,

                "status":         "open",

                "entry_bar":      0,

                "score_at_entry": score,

                "session":        getattr(self._current_session, "session", type("", (), {"value": None})()).value,

                "regime":         getattr(self._current_regime, "regime", type("", (), {"value": None})()).value,

                "spread_at_entry": (getattr(self._exec_quality, "_last_spread", None)

                                    if self._exec_quality else None),

            })

        except Exception as _dbe:

            logger.debug("DB save_trade (open) error: %s", _dbe)

        # Small accounts: single-leg mode -- skip Leg 2, keep MAX_OPEN slots free.
        if _is_small:
            logger.info(
                "[small_acct] Single-leg market entry: TP1=%.5f lot=%.2f bal=$%.2f",
                tp1, _single_leg_lot, _cur_bal,
            )
            self._total_trades += 1
            side_str = "BUY" if is_long else "SELL"
            return f"[{label}] Small-acct single-leg {side_str} @ {price:.5f}"



        # â”€â”€ Leg 2: TP2 â€” trail to 1:5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        req2 = {

            "action":       mt5.TRADE_ACTION_DEAL,

            "symbol":       self.asset,

            "volume":       half_lot,

            "type":         order_type,

            "price":        price,

            "sl":           round(sl, 5),

            "tp":           tp2,

            "deviation":    20,

            "magic":        777_000,

            "comment":      f"Prom-TP2-{label[:4]}",

            "type_time":    mt5.ORDER_TIME_GTC,

            "type_filling": fill,

        }

        r2 = mt5.order_send(req2)

        if r2 is not None and r2.retcode == mt5.TRADE_RETCODE_DONE:

            self._tp2_tickets.add(r2.order)

            # Tag leg-2 ticket for LTF/grade learning as well

            self._ltf_entry_state[r2.order] = self._last_ltf_state

            self._entry_grade[r2.order]     = getattr(self, "_pending_grade",       "F")
            self._entry_regime[r2.order]    = getattr(self, "_pending_regime_name",  "unknown")
            self._entry_session[r2.order]   = getattr(self, "_pending_session_name", "unknown")



        self._total_trades += 1



        # OB learning â€” record a hit for this OB direction

        ob_dir = "bullish" if is_long else "bearish"

        obs = self._ob_stats.setdefault(ob_dir, {"hits": 0, "wins": 0})

        obs["hits"] += 1



        msg = (

            f"[{label}] Dual-TP entry | {'BUY' if is_long else 'SELL'} "

            f"{half_lot}+{half_lot} lots @ {price:.4f} | "

            f"SL={sl:.4f} TP1={tp1:.4f}(1:{TP1_RR}) TP2={tp2:.4f}(1:{TP2_RR}) "

            f"| OB={'yes' if ob_zone else 'no'} grade={grade} score={score:.0f}"

        )

        logger.info(msg)

        return msg



    def _execute_ltf_scalp(self, direction: str, label: str = "ltf_scalp") -> str:

        """Open a tight trend-following scalp using 5M ATR for SL/TP.



        Called when both 5M and 1M are strongly aligned WITH the primary

        timeframe signal -- e.g. 4H bearish AND 5M+1M both bearish for a

        momentum continuation scalp.  When called with label='ltf_reversal'

        the entry is a counter-structure reversal driven by micro-LTF momentum

        surge (Trend Exhaustion regime blocking the HTF-direction signal).



        * SL  = LTF_SCALP_ATR_SL x 5M_ATR  (much tighter than 4H ATR)

        * TP  = SL_distance x LTF_SCALP_RR  (1:2.5 by default)

        * Lot = half of normal risk fraction (conservative sizing)

        * Magic number 777_001 distinguishes scalp tickets from primary ones.

        """

        is_long  = direction == "bullish"

        is_short = direction == "bearish"

        if not is_long and not is_short:

            return "LTF scalp: invalid direction -- skipped"



        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            side = "BUY" if is_long else "SELL"

            return f"[DRY RUN] LTF scalp would {side} {self.asset}"



        # -- Cap: max 2 LTF scalp positions open at a time --------------------

        _existing_scalps = [p for p in (mt5.positions_get(symbol=self.asset) or [])

                            if p.magic == 777_001]

        if len(_existing_scalps) >= 2:

            return f"LTF scalp: {len(_existing_scalps)} scalp position(s) already open — skipped"



        # -- Compute 5M ATR from stored candle data ---------------------------

        atr_5m = 0.0

        _m5_df = self._last_tf_data.get("5m")

        if _m5_df is not None and len(_m5_df) >= 14:

            try:

                import pandas as _pd  # noqa: PLC0415

                prev_close = _m5_df["close"].shift(1)

                tr = _pd.concat(

                    [

                        _m5_df["high"] - _m5_df["low"],

                        (_m5_df["high"] - prev_close).abs(),

                        (_m5_df["low"]  - prev_close).abs(),

                    ],

                    axis=1,

                ).max(axis=1)

                atr_5m = float(tr.rolling(14).mean().iloc[-1])

            except Exception as _exc:

                logger.debug("LTF scalp ATR calc error: %s", _exc)



        if atr_5m <= 0.0:

            return "LTF scalp: 5M ATR unavailable -- skipped"



        # Execution quality gate for LTF scalp

        if ENABLE_EXECUTION_QUALITY and self._exec_quality and not self.dry_run:

            _sym_info = mt5.symbol_info(self.asset) if MT5_AVAILABLE else None

            _tol = getattr(self._current_session, "spread_tolerance", 1.0)

            _qr = self._exec_quality.check(_sym_info, atr_5m, spread_tolerance_mult=_tol)

            if not _qr.passes:

                msg = f"[exec_quality] LTF scalp blocked: {_qr.reason}"

                logger.info(msg)

                return msg



        tick = mt5.symbol_info_tick(self.asset)

        if not tick:

            return f"LTF scalp: no tick data for {self.asset}"



        price   = tick.ask if is_long else tick.bid

        sl_dist = atr_5m * LTF_SCALP_ATR_SL

        sl      = round(price - sl_dist, 5) if is_long else round(price + sl_dist, 5)

        tp      = (round(price + sl_dist * LTF_SCALP_RR, 5) if is_long

                   else round(price - sl_dist * LTF_SCALP_RR, 5))



        # Half risk fraction -- scalp uses smaller position size

        orig_frac = self.risk_fraction

        self.risk_fraction *= 0.5

        lot = self._calc_lot(sl_dist)

        self.risk_fraction = orig_frac



        sym      = mt5.symbol_info(self.asset)

        min_lot  = sym.volume_min  if sym else 0.01

        step_lot = sym.volume_step if sym else 0.01

        lot      = max(min_lot, round(lot / step_lot) * step_lot)



        fill = mt5.ORDER_FILLING_IOC

        if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK):

            fill = mt5.ORDER_FILLING_FOK



        req = {

            "action":       mt5.TRADE_ACTION_DEAL,

            "symbol":       self.asset,

            "volume":       lot,

            "type":         mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,

            "price":        price,

            "sl":           sl,

            "tp":           tp,

            "deviation":    20,

            "magic":        777_001,           # distinct from primary-entry magic

            "comment":      "Prom-LTF-scalp",

            "type_time":    mt5.ORDER_TIME_GTC,

            "type_filling": fill,

        }

        r = mt5.order_send(req)

        if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:

            code = r.retcode if r else "None"

            return f"LTF scalp: order FAILED retcode={code}"



        self._ltf_entry_state[r.order] = label   # "ltf_scalp" or "ltf_reversal"

        self._total_trades += 1



        try:

            save_trade({

                "trade_id":       f"live_{r.order}",

                "source":         "live",

                "asset":          self.asset,

                "timeframe":      "5m",

                "direction":      "BUY" if is_long else "SELL",

                "entry_price":    price,

                "sl_price":       sl,

                "tp_price":       tp,

                "size":           lot,

                "status":         "open",

                "entry_bar":      0,

                "session":        getattr(self._current_session, "session", type("", (), {"value": None})()).value,

                "regime":         getattr(self._current_regime, "regime", type("", (), {"value": None})()).value,

            })

        except Exception as _dbe:

            logger.debug("DB save_trade (LTF scalp) error: %s", _dbe)



        msg = (

            f"[LTF-{label}] {'BUY' if is_long else 'SELL'} {lot} lots @ {price:.4f} "

            f"| SL={sl:.4f}(-{sl_dist:.2f}pts) TP={tp:.4f} "

            f"| 5M-ATR={atr_5m:.4f} RR=1:{LTF_SCALP_RR}"

        )

        logger.info(msg)

        return msg






    def _find_ltf_ob(
        self,
        direction: str,
        approx_price: float,
        atr_approx: float,
    ):
        """Return the nearest fresh LTF Order Block as (low, high) or None.

        Uses 5m bars first, falls back to 15m.  The LTF OBs give a more
        precise limit-order price than the HTF OBs from the main analysis.
        """
        if not self._last_tf_data or self.engine is None:
            return None
        try:
            ltf_df = (
                self._last_tf_data.get("5m")
                or self._last_tf_data.get("15m")
                or self._last_tf_data.get("30m")
            )
            if ltf_df is None or len(ltf_df) < 20:
                return None
            smc_eng = getattr(self.engine, "smc_engine", None)
            if smc_eng is None:
                return None
            import pandas as pd   # noqa: PLC0415
            df = ltf_df.copy()
            df.columns = [c.lower() for c in df.columns]
            atr_s = (df["high"] - df["low"]).rolling(14).mean()
            ltf_atr = float(atr_s.iloc[-1]) if not atr_s.empty else None
            obs = smc_eng.detect_order_blocks(df, ltf_atr)
            fresh = [ob for ob in obs if ob.direction == direction and not ob.mitigated]
            if not fresh:
                return None
            ref_fn = (lambda b: b.high) if direction == "bullish" else (lambda b: b.low)
            fresh.sort(key=lambda b: (
                abs(ref_fn(b) - approx_price),
                -b.strength,
            ))
            max_dist = MAX_LIMIT_DISTANCE_ATR * (atr_approx if atr_approx > 0 else 1.0)
            for ob in fresh:
                dist = abs(ref_fn(ob) - approx_price)
                if atr_approx == 0 or dist <= max_dist:
                    logger.info(
                        "[ltf_ob] LTF %s OB %.4f-%.4f selected (dist=%.4f, strength=%.2f)",
                        direction, ob.low, ob.high, dist, ob.strength,
                    )
                    return (ob.low, ob.high)
        except Exception as exc:
            logger.warning("[ltf_ob] Detection failed: %s", exc)
        return None

    def _place_limit_order(

        self, *, is_long: bool, zone_price: float,

        sup, res, atr: float, price: float, grade: str, score: float,

    ) -> str:

        """Place TWO pending Buy/SellLimit orders at the zone edge (dual-TP split).



        Leg 1 â€” half lot, TP = zone_price Â± sl_dist Ã— TP1_RR  (1:1.5 R:R)

        Leg 2 â€” half lot, TP = zone_price Â± sl_dist Ã— TP2_RR  (1:5   R:R)



        Both orders share the same limit price and SL. They expire after

        LIMIT_ORDER_EXPIRY polls if not filled.

        """

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            side = "BuyLimit" if is_long else "SellLimit"

            return (f"[DRY RUN] Would place {side} @ {zone_price:.4f} "

                    f"grade={grade} score={score:.0f}")



        # Skip if the zone is unreachably far from current price

        if atr and abs(zone_price - price) > MAX_LIMIT_DISTANCE_ATR * atr:

            dist_pts = abs(zone_price - price)

            return (f"[limit] Zone too far ({dist_pts:.1f} pts, {dist_pts/atr:.1f}Ã—ATR) "

                    f"â€” limit skipped (max {MAX_LIMIT_DISTANCE_ATR}Ã—ATR)")



        MIN_SL_ATR = 1.0

        buf  = atr * 0.15 if atr else price * 0.002

        fallback_sl_dist = max(atr * MIN_SL_ATR, price * 0.003)



        limit_price = round(zone_price, 5)

        if is_long:

            sl_cand = (float(sup) - buf) if (sup and sup < zone_price) else zone_price - fallback_sl_dist

            sl = round(min(sl_cand, zone_price - atr * MIN_SL_ATR) if atr else sl_cand, 5)

        else:

            sl_cand = (float(res) + buf) if (res and res > zone_price) else zone_price + fallback_sl_dist

            sl = round(max(sl_cand, zone_price + atr * MIN_SL_ATR) if atr else sl_cand, 5)



        sl_dist = abs(limit_price - sl)

        if sl_dist <= 0:

            return "[limit] Invalid SL for limit order â€” skipped"



        _tp_scale_lim = (self._current_regime.tp_scalar
                         if self._current_regime is not None else 1.0)

        if is_long:

            tp1 = round(limit_price + sl_dist * TP1_RR * _tp_scale_lim, 5)

            tp2 = round(limit_price + sl_dist * TP2_RR * _tp_scale_lim, 5)

        else:

            tp1 = round(limit_price - sl_dist * TP1_RR * _tp_scale_lim, 5)

            tp2 = round(limit_price - sl_dist * TP2_RR * _tp_scale_lim, 5)



        total_lot = self._calc_lot(sl_dist)

        sym      = mt5.symbol_info(self.asset)

        min_lot  = sym.volume_min  if sym else 0.01

        step_lot = sym.volume_step if sym else 0.01

        half_lot = max(min_lot, round(total_lot / 2 / step_lot) * step_lot)



        lim_type = mt5.ORDER_TYPE_BUY_LIMIT if is_long else mt5.ORDER_TYPE_SELL_LIMIT

        side_tag = "BL" if is_long else "SL"  # BuyLimit / SellLimit

        fill = mt5.ORDER_FILLING_RETURN        # pending orders require RETURN fill



        def _send_leg(tp: float, leg: int) -> tuple[int | None, str]:

            req = {

                "action":       mt5.TRADE_ACTION_PENDING,

                "symbol":       self.asset,

                "volume":       half_lot,

                "type":         lim_type,

                "price":        limit_price,

                "sl":           sl,

                "tp":           tp,

                "deviation":    20,

                "magic":        MAGIC_LIMIT,

                "comment":      f"Prom-{side_tag}-TP{leg}",

                "type_time":    mt5.ORDER_TIME_GTC,

                "type_filling": fill,

            }

            r = mt5.order_send(req)

            if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:

                code = r.retcode if r else "None"

                err  = r.comment if r else "no response"

                return None, f"[limit] Leg{leg} FAILED retcode={code} ({err})"

            return r.order, ""



        order1, err1 = _send_leg(tp1, 1)

        if order1 is None:

            return err1



        order2, err2 = _send_leg(tp2, 2)

        if order2 is None:

            # Cancel leg 1 since leg 2 failed â€” don't leave an orphan

            mt5.order_send({

                "action": mt5.TRADE_ACTION_REMOVE,

                "order":  order1,

                "comment": "Leg2 failed â€” orphan cleanup",

            })

            return f"{err2} (leg1 #{order1} also cancelled)"



        # Track both so they expire together

        self._pending_limits[order1] = LIMIT_ORDER_EXPIRY

        self._pending_limits[order2] = LIMIT_ORDER_EXPIRY
        # Track direction for each leg — used by the LTF opposition monitor
        self._pending_limit_dirs[order1] = is_long
        self._pending_limit_dirs[order2] = is_long

        # Tag both limit legs for LTF/grade learning (position_id == order on fill)

        self._ltf_entry_state[order1] = self._last_ltf_state

        self._ltf_entry_state[order2] = self._last_ltf_state

        self._entry_grade[order1]     = getattr(self, "_pending_grade",       "F")
        self._entry_grade[order2]     = getattr(self, "_pending_grade",       "F")
        self._entry_regime[order1]    = getattr(self, "_pending_regime_name",  "unknown")
        self._entry_regime[order2]    = getattr(self, "_pending_regime_name",  "unknown")
        self._entry_session[order1]   = getattr(self, "_pending_session_name", "unknown")
        self._entry_session[order2]   = getattr(self, "_pending_session_name", "unknown")



        lbl = "BuyLimit" if is_long else "SellLimit"

        msg = (

            f"[limit] {lbl} Ã—2 legs @ {limit_price:.4f}  "

            f"SL={sl:.4f}  TP1={tp1:.4f}  TP2={tp2:.4f}  "

            f"lot={half_lot}Ã—2  orders=#{order1},#{order2}  "

            f"expires={LIMIT_ORDER_EXPIRY} polls  grade={grade}  score={score:.0f}"

        )

        logger.info(msg)

        return msg



    def _manage_pending_limits(self) -> list[str]:

        """Expire stale pending limit orders that haven't been filled."""

        if self.dry_run or not MT5_AVAILABLE or not self._pending_limits:

            return []

        msgs: list[str] = []

        expired = []

        for ticket, polls_left in list(self._pending_limits.items()):

            polls_left -= 1

            if polls_left <= 0:

                req = {

                    "action":   mt5.TRADE_ACTION_REMOVE,

                    "order":    ticket,

                    "comment":  "Expired",

                }

                r = mt5.order_send(req)

                if r and r.retcode == mt5.TRADE_RETCODE_DONE:

                    msgs.append(f"[limit] Cancelled expired order #{ticket}")

                    logger.info("Cancelled expired limit order #%s", ticket)

                expired.append(ticket)

            else:

                # LTF opposition check — cancel early if both LTFs flipped against this limit's direction.
                # Example: BuyLimit placed while 30M+1H were aligned; they have since both turned bearish
                # — letting it fill now means filling into momentum opposing the trade.
                _lim_dir = self._pending_limit_dirs.get(ticket)
                if _lim_dir is not None and len(self._current_ltf_biases) >= 2:
                    _opp = "bearish" if _lim_dir else "bullish"
                    if all(b == _opp for b in self._current_ltf_biases):
                        r = mt5.order_send({
                            "action":  mt5.TRADE_ACTION_REMOVE,
                            "order":   ticket,
                            "comment": "LTF trap",
                        })
                        if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                            _lim_label = "BuyLimit" if _lim_dir else "SellLimit"
                            msgs.append(
                                f"[limit] Cancelled #{ticket} — LTF trap "
                                f"(both LTFs {self._current_ltf_biases} opposing {_lim_label})"
                            )
                            logger.info(
                                "[limit] Cancelled #%s — LTF trap: both LTFs %s opposing %s",
                                ticket, self._current_ltf_biases, _lim_label,
                            )
                        expired.append(ticket)
                        continue

                self._pending_limits[ticket] = polls_left

        for t in expired:

            self._pending_limits.pop(t, None)
            self._pending_limit_dirs.pop(t, None)  # clean direction tracker on expiry/cancel

        return msgs



    def _manage_partial_close(self, open_positions: list[dict]) -> list[str]:

        """Lock in 50 % profit on any position whose price has reached the TP1 level.



        TP1 is computed from the position's own entry / SL:

            tp1 = entry + |entry - sl| Ã— TP1_RR  (BUY)

            tp1 = entry - |entry - sl| Ã— TP1_RR  (SELL)



        If the position still has enough volume to split (â‰¥ 2 Ã— min_lot), exactly

        half the lots are closed at market and the remaining SL is locked to

        break-even.  If the position is already at minimum size (e.g. leg 1 of the

        dual-TP split), it is closed in full â€” MT5's own TP will usually beat

        this check, so this acts as a safety net.



        Each ticket is processed at most once per session via _tp1_partial_done.

        """

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            return []



        sym = mt5.symbol_info(self.asset)

        if not sym:

            return []

        min_lot  = sym.volume_min

        step_lot = sym.volume_step

        fill     = (mt5.ORDER_FILLING_FOK

                    if sym.filling_mode & mt5.ORDER_FILLING_FOK

                    else mt5.ORDER_FILLING_IOC)



        msgs: list[str] = []

        for pos in open_positions:

            ticket = pos["ticket"]

            if ticket in self._tp1_partial_done:

                continue



            entry  = pos["entry"]

            sl     = pos["sl"]

            cur    = pos.get("current")

            lots   = pos["lots"]

            is_buy = pos["direction"] == "BUY"



            if cur is None or sl == 0:

                continue



            sl_dist = abs(entry - sl)

            if sl_dist <= 0:

                continue



            # Compute where TP1 sits for this position

            tp1 = (round(entry + sl_dist * TP1_RR, 5) if is_buy

                   else round(entry - sl_dist * TP1_RR, 5))



            reached = (cur >= tp1) if is_buy else (cur <= tp1)

            if not reached:

                continue



            # Mark done immediately to prevent double-firing on the same poll

            self._tp1_partial_done.add(ticket)



            tick = mt5.symbol_info_tick(self.asset)

            if not tick:

                continue

            close_price = tick.bid if is_buy else tick.ask



            # How much to close: half if possible, else full

            half = round(lots / 2 / step_lot) * step_lot

            close_lots = max(min_lot, half) if half >= min_lot else lots



            req = {

                "action":       mt5.TRADE_ACTION_DEAL,

                "symbol":       self.asset,

                "volume":       round(close_lots, 2),

                "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,

                "price":        close_price,

                "position":     ticket,

                "deviation":    20,

                "magic":        777_000,

                "comment":      "Prom-TP1-partial",

                "type_time":    mt5.ORDER_TIME_GTC,

                "type_filling": fill,

            }

            r = mt5.order_send(req)

            if r and r.retcode == mt5.TRADE_RETCODE_DONE:

                profit_pts = abs(close_price - entry)

                remaining  = round(lots - close_lots, 2)

                msg = (

                    f"[partial] TP1 @ {tp1:.4f} â€” #{ticket} closed {close_lots} lots "

                    f"@ {close_price:.4f} (+{profit_pts:.2f} pts) | "

                    f"remaining {remaining} lots"

                )

                logger.info(msg)

                msgs.append(msg)



                # Move remaining SL to break-even (lock in 0 loss)

                if remaining >= min_lot:

                    _be_sl = round(entry + BE_PROFIT_PTS if is_buy else entry - BE_PROFIT_PTS, 5)

                    be_req = {

                        "action":   mt5.TRADE_ACTION_SLTP,

                        "symbol":   self.asset,

                        "position": ticket,

                        "sl":       _be_sl,

                        "tp":       pos["tp"],

                        "magic":    777_000,

                    }

                    be_res = mt5.order_send(be_req)

                    if be_res and be_res.retcode == mt5.TRADE_RETCODE_DONE:

                        logger.info("[partial] SL of #%s locked to BE+buf @ %.4f", ticket, _be_sl)



                # Record the OB hit as a win for learning

                ob_dir = "bullish" if is_buy else "bearish"

                self._ob_stats.setdefault(ob_dir, {"hits": 0, "wins": 0})["wins"] += 1

            else:

                # Remove from done set so we retry next poll

                self._tp1_partial_done.discard(ticket)

                code = r.retcode if r else "None"

                logger.warning("[partial] TP1 close failed #%s retcode=%s", ticket, code)



        return msgs



    def _check_manual_override(self) -> Optional[str]:

        """Check for a manual trade JSON written by the dashboard."""

        if not MANUAL_TRADE_FILE.exists():

            return None

        try:

            data = json.loads(MANUAL_TRADE_FILE.read_text(encoding="utf-8"))

        except Exception as exc:

            logger.warning("Cannot read manual_trade.json: %s", exc)

            MANUAL_TRADE_FILE.unlink(missing_ok=True)

            return None

        required = {"direction", "sl", "tp"}

        if not required.issubset(data.keys()):

            logger.warning("manual_trade.json missing fields: %s", required - data.keys())

            MANUAL_TRADE_FILE.unlink(missing_ok=True)

            return None

        MANUAL_TRADE_FILE.unlink(missing_ok=True)

        raw_dir     = str(data["direction"]).upper()

        is_long     = raw_dir in ("BUY", "LONG", "BULLISH")

        is_short    = raw_dir in ("SELL", "SHORT", "BEARISH")

        is_limit    = str(data.get("order_type", "market")).lower() == "limit"

        entry_price = data.get("entry")

        if entry_price is not None:

            entry_price = float(entry_price)

        if not is_long and not is_short:

            logger.warning("manual_trade.json -- unrecognised direction: %s", raw_dir)

            return None

        sl      = float(data["sl"])

        tp      = float(data["tp"])

        lots    = float(data.get("lots") or 0)

        comment = str(data.get("comment") or "Prom-manual")[:31]

        _otype  = ("BuyLimit" if is_long else "SellLimit") if is_limit else ("BUY" if is_long else "SELL")

        logger.info("Manual override: %s SL=%.4f TP=%.4f entry=%s lots=%s",

                    _otype, sl, tp, entry_price or "market", lots or "auto")

        if self.dry_run or not MT5_AVAILABLE or not self._mt5_connected:

            msg = (f"[DRY RUN] [manual] Would {_otype} {self.asset} "

                   f"entry={entry_price or 'market'} SL={sl:.4f} TP={tp:.4f}")

            logger.info(msg)

            self._total_trades += 1

            return msg

        tick = mt5.symbol_info_tick(self.asset)

        if not tick:

            return f"Manual override failed: no tick data for {self.asset}"

        sym  = mt5.symbol_info(self.asset)

        fill = mt5.ORDER_FILLING_IOC

        if sym and (sym.filling_mode & mt5.ORDER_FILLING_FOK):

            fill = mt5.ORDER_FILLING_FOK

        if is_limit and entry_price is not None:

            if lots <= 0:

                sl_dist = abs(entry_price - sl)

                lots    = self._calc_lot(sl_dist)

            lim_type = mt5.ORDER_TYPE_BUY_LIMIT if is_long else mt5.ORDER_TYPE_SELL_LIMIT

            req = {

                "action":       mt5.TRADE_ACTION_PENDING,

                "symbol":       self.asset,

                "volume":       lots,

                "type":         lim_type,

                "price":        round(entry_price, 5),

                "sl":           round(sl, 5),

                "tp":           round(tp, 5),

                "deviation":    20,

                "magic":        777_002,

                "comment":      comment,

                "type_time":    mt5.ORDER_TIME_GTC,

                "type_filling": fill,

            }

            r2 = mt5.order_send(req)

            if r2 is None or r2.retcode != mt5.TRADE_RETCODE_DONE:

                code = r2.retcode if r2 else "None"

                err  = r2.comment if r2 else "no response"

                return f"Manual limit FAILED retcode={code} ({err})"

            self._total_trades += 1

            msg = (f"[MANUAL-LIMIT] {_otype} ticket={r2.order} "

                   f"{lots} lots @ {entry_price:.4f} SL={sl:.4f} TP={tp:.4f}")

            logger.info(msg)

            return msg

        else:

            price = tick.ask if is_long else tick.bid

            if lots <= 0:

                sl_dist = abs(price - sl)

                lots    = self._calc_lot(sl_dist)

            req = {

                "action":       mt5.TRADE_ACTION_DEAL,

                "symbol":       self.asset,

                "volume":       lots,

                "type":         mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,

                "price":        price,

                "sl":           round(sl, 5),

                "tp":           round(tp, 5),

                "deviation":    20,

                "magic":        777_000,

                "comment":      comment,

                "type_time":    mt5.ORDER_TIME_GTC,

                "type_filling": fill,

            }

            r2 = mt5.order_send(req)

            if r2 is None or r2.retcode != mt5.TRADE_RETCODE_DONE:

                code = r2.retcode if r2 else "None"

                err  = r2.comment if r2 else "no response"

                return f"Manual order FAILED retcode={code} ({err})"

            self._total_trades += 1

            msg = (f"[MANUAL] {'BUY' if is_long else 'SELL'} ticket={r2.order} "

                   f"{lots} lots @ {price:.4f} SL={sl:.4f} TP={tp:.4f}")

            logger.info(msg)

            return msg





    def _poll(self) -> dict:

        update = {"last_action": "â€”", "last_signal_id": self._last_id}

        # Reset per-poll override flags so they never bleed between cycles
        self._asian_exhaustion_override_active = False



        # -- Account balance / circuit-breaker check --

        _acct    = self._mt5_account()
        _balance_raw = (_acct.get("balance") if isinstance(_acct, dict) else None)
        _equity_raw = (_acct.get("equity") if isinstance(_acct, dict) else None)
        _login_raw = (_acct.get("login") if isinstance(_acct, dict) else None)
        _has_verified_balance = (
            isinstance(_acct, dict)
            and (_balance_raw is not None)
            and (_equity_raw is not None)
            and (str(_login_raw or "").strip() not in {"", "0", "None"})
        )

        _balance = float(_acct.get("balance", 0) or 0) if _acct else 0.0



        # Capture starting balance once on first poll with real data

        if self._session_start_balance is None and _balance > 0:

            self._session_start_balance = _balance

            logger.info("[risk] Session-start balance: $%.2f", _balance)

        # ── Date-keyed daily baseline (persists across restarts) ──────────────
        # Only reset when the calendar date changes (midnight rollover), never on
        # bot restarts or balance events.  This prevents phantom 200%+ loss pcts
        # caused by capturing a tiny balance on a mid-session restart.
        _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._day_start_date != _today and _balance > 0:
            # Try to load a persisted value from STATUS_FILE first so a restart
            # mid-day keeps the same baseline.
            _persisted_date    = None
            _persisted_balance = None
            _persisted_halt_until_ts = None
            try:
                _sf = STATUS_FILE
                if _sf.exists():
                    _sd = json.loads(_sf.read_text(encoding="utf-8"))
                    _persisted_date    = _sd.get("day_start_date")
                    _persisted_balance = _sd.get("day_start_balance")
                    _persisted_halt_until_ts = _sd.get("daily_loss_halt_until_ts")
            except Exception:
                pass
            if _persisted_date == _today and _persisted_balance and _persisted_balance > 0:
                self._day_start_balance = float(_persisted_balance)
                self._day_start_date    = _today
                try:
                    if _persisted_halt_until_ts:
                        self._daily_loss_halt_until_ts = float(_persisted_halt_until_ts)
                except Exception:
                    self._daily_loss_halt_until_ts = None
                logger.info(
                    "[risk] Restored daily baseline from status file: $%.2f (%s)",
                    self._day_start_balance, _today,
                )
            else:
                self._day_start_balance = _balance
                self._day_start_date    = _today
                self._daily_loss_halt_until_ts = None
                logger.info(
                    "[risk] New daily baseline captured: $%.2f (%s)",
                    self._day_start_balance, _today,
                )



        _trading_halted = False

        _halt_reason    = ""



        if not self.dry_run:

            # Hard floor: refuse all new entries below minimum balance

            if _has_verified_balance and _balance > 0 and _balance < MIN_BALANCE_USD:

                _trading_halted = True

                _halt_reason = (f"CIRCUIT BREAKER: balance ${_balance:.2f} <= minimum"

                                f" ${MIN_BALANCE_USD:.2f} - all new entries suspended.")
            elif MT5_AVAILABLE and self._mt5_connected and not _has_verified_balance:
                _risk_msg = "RISK NOTICE: MT5 account balance unavailable; skipping hard-floor balance circuit breaker this poll."
                logger.warning(_risk_msg)
                update["risk_notice"] = _risk_msg



            # Daily loss cap — trading P&L only, never include balance operations
            if (not _trading_halted and self._day_start_balance
                    and MT5_AVAILABLE and self._mt5_connected):
                try:
                    import datetime as _dt_mod

                    _now = _dt_mod.datetime.now()
                    _refresh_needed = (time.time() - self._daily_deals_last_ts) >= DAILY_DEALS_REFRESH_SECONDS

                    if _refresh_needed:
                        _start = _now.replace(hour=0, minute=0, second=0, microsecond=0)
                        _deals = mt5.history_deals_get(_start, _now) or []

                        # Only count actual trade deals (type 0=BUY, type 1=SELL).
                        # Exclude balance operations (type 2=BALANCE), credit adjustments (3),
                        # corrections (4), bonuses (5), commissions (6) and charges (7).
                        # A withdrawal shows as type 2 with a large negative profit — without
                        # this filter the circuit breaker fires incorrectly on withdrawals.
                        _trade_deals = [d for d in _deals if d.type in (0, 1)]
                        _daily_pnl = float(sum(d.profit for d in _trade_deals))

                        _loss_pct  = 0.0
                        if _daily_pnl < 0 and self._day_start_balance:
                            _loss_pct = -_daily_pnl / self._day_start_balance * 100

                        self._daily_trade_pnl_cache = _daily_pnl
                        self._daily_loss_pct_cache = _loss_pct
                        self._daily_deals_last_ts = time.time()
                    else:
                        _daily_pnl = self._daily_trade_pnl_cache
                        _loss_pct = self._daily_loss_pct_cache

                    # Effective daily loss cap.
                    # Use whichever is larger: the % cap OR a fixed $50 floor so tiny
                    # accounts don't halt on a single normal-variance trade.
                    # Example: $600 account → max(8%=$48, $50) = $50 cap.
                    # Example: $50k account → max(8%=$4000, $50) = $4000 cap.
                    _pct_cap_usd = self._day_start_balance * MAX_DAILY_LOSS_PCT / 100
                    _effective_cap_usd = max(_pct_cap_usd, 50.0)
                    # Only show the % in the message when it's meaningful (<= 100%)
                    _effective_cap_pct = _effective_cap_usd / self._day_start_balance * 100

                    _now_ts = time.time()
                    # Backward-compatibility guard: if a legacy status restored the
                    # halt latch without a cooldown timestamp, clear it immediately.
                    if self._daily_loss_halted and not self._daily_loss_halt_until_ts:
                        self._daily_loss_halted = False
                    if self._daily_loss_halt_until_ts and _now_ts >= self._daily_loss_halt_until_ts:
                        self._daily_loss_halted = False
                        self._daily_loss_halt_until_ts = None
                        logger.info("[risk] Daily-loss circuit breaker cooldown expired; entries re-enabled.")

                    # Fixed-duration circuit breaker: once tripped, block entries for 2 hours.
                    if (not self._daily_loss_halted) and _daily_pnl < 0 and (-_daily_pnl >= _effective_cap_usd):
                        self._daily_loss_halted = True
                        self._daily_loss_halt_until_ts = _now_ts + DAILY_LOSS_HALT_SECONDS

                    if self._daily_loss_halted:
                        _remaining_sec = max(0, int((self._daily_loss_halt_until_ts or _now_ts) - _now_ts))
                        if _remaining_sec <= 0:
                            self._daily_loss_halted = False
                            self._daily_loss_halt_until_ts = None

                    if self._daily_loss_halted:
                        _trading_halted = True
                        if _effective_cap_pct <= 100:
                            _limit_str = f"{_effective_cap_pct:.1f}%"
                        else:
                            _limit_str = f"${_effective_cap_usd:.2f}"
                        _remaining_min = max(1, _remaining_sec // 60) if _remaining_sec > 0 else 0
                        _loss_usd = max(0.0, -_daily_pnl)
                        _halt_reason = (f"CIRCUIT BREAKER: daily trading loss ${_loss_usd:.2f}"
                                        f" ({_loss_pct:.1f}% >= {_limit_str} limit)"
                                        f" - no new entries for 2 hours"
                                        f" (remaining ~{_remaining_min}m).")

                    # Daily profit protection: once daily gain >= target %, scale lot to 0.5x
                    if _daily_pnl > 0 and self._day_start_balance:
                        _profit_pct = _daily_pnl / self._day_start_balance * 100
                        self._daily_profit_protect = _profit_pct >= DAILY_PROFIT_PROTECT_PCT
                        if self._daily_profit_protect:
                            logger.info(
                                "[risk] Daily profit target hit: +$%.2f (+%.1f%%) -- lot scale 0.50x",
                                _daily_pnl, _profit_pct,
                            )
                    else:
                        self._daily_profit_protect = False

                except Exception as _ce:
                    logger.debug("Daily loss check error: %s", _ce)



        # -- Manual override runs even when trading is halted --

        # A dashboard-initiated manual trade is an explicit user decision

        # and should be processed regardless of circuit-breaker state.

        _early_manual = self._check_manual_override()

        if _early_manual:

            update["last_action"]  = _early_manual

            update["total_trades"] = self._total_trades

            self._write_status(update)

            return update



        if _trading_halted:

            logger.warning(_halt_reason)

            update["last_action"]     = _halt_reason

            update["trading_halted"]  = True

            update["halt_reason"]     = _halt_reason

            _open = self._get_open_positions()

            self._manage_positions(_open)

            update["open_positions"]  = _open

            update["open_count"]      = len(_open)

            update["total_unrealised"]= round(sum(p["unrealised"] for p in _open), 2)

            return update





        # â”€â”€ Fetch open positions first â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # On the very first poll, reconcile any DB trades that closed during downtime

        if not self._startup_reconciled:

            self._reconcile_db_on_startup()

            self._startup_reconciled = True



        # Periodic full MT5<->DB reconciliation (every DB_SYNC_INTERVAL_POLLS polls)

        self._poll_count += 1

        if self._poll_count % DB_SYNC_INTERVAL_POLLS == 0:

            self._periodic_db_sync()

        # Session-end learning flush: save adaptive state at 17:00 UTC (end of
        # London/NY Overlap) so stats are persisted even if no trade closed that
        # session.  We track the last UTC hour we saved to avoid multiple saves
        # within the same minute window.
        _now_utc = datetime.utcnow()
        _save_hour = getattr(self, "_learning_last_save_hour", -1)
        if _now_utc.hour == 17 and _save_hour != _now_utc.date():
            _save_learning(self._ob_stats)
            self._learning_last_save_hour = _now_utc.date()
            self._persist_open_trade_meta()
            _save_learning(self._ob_stats)
            logger.info("[LML] Session-end flush: learning state saved at London/NY Overlap close (17:00 UTC)")

        open_positions = self._get_open_positions()

        self._learn_from_closes(open_positions)   # detect any just-closed trades

        self._prev_open_tickets      = {p["ticket"]    for p in open_positions}

        self._cached_open_directions = {p["direction"] for p in open_positions}



        # -- 5M reversal exit (highest priority -- runs before trail) ----------

        m5_exit_msgs = self._check_5m_exits(open_positions)

        if m5_exit_msgs:

            open_positions = self._get_open_positions()   # refresh after closes

            # Call _learn_from_closes immediately so DB is updated within this

            # poll -- prevents orphaned 'open' records if bot stops before next poll.

            self._learn_from_closes(open_positions)

            self._prev_open_tickets      = {p["ticket"]    for p in open_positions}

            self._cached_open_directions = {p["direction"] for p in open_positions}

            update["m5_exit_events"]   = m5_exit_msgs

            update["last_trail_action"] = m5_exit_msgs[-1]

            logger.info("5M exit events: %s", m5_exit_msgs)



        # â”€â”€ Trailing-stop management (runs every poll before new-signal logic) â”€

        trail_msgs = self._manage_positions(open_positions)

        partial_msgs = self._manage_partial_close(open_positions)

        if trail_msgs or partial_msgs:

            # Refresh so that updated SLs appear in the status file immediately

            open_positions = self._get_open_positions()

        # Hybrid time-aware profit capture runs after trail/partial updates so it
        # never acts on stale position snapshots from earlier in the same poll.
        time_exit_msgs = self._manage_time_profit_exits(open_positions)

        if time_exit_msgs:

            open_positions = self._get_open_positions()

            self._learn_from_closes(open_positions)

            self._prev_open_tickets      = {p["ticket"]    for p in open_positions}

            self._cached_open_directions = {p["direction"] for p in open_positions}

        if trail_msgs:

            update["last_trail_action"] = trail_msgs[-1]

            update["trail_events"]      = trail_msgs

        if partial_msgs:

            update["partial_close_events"] = partial_msgs

            logger.info("Partial close events: %s", partial_msgs)

        if time_exit_msgs:

            update["time_exit_events"] = time_exit_msgs

            update["last_trail_action"] = time_exit_msgs[-1]

            logger.info("Time exit events: %s", time_exit_msgs)



        # â”€â”€ Expire stale pending limit orders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        limit_msgs = self._manage_pending_limits()

        if limit_msgs:

            update["limit_events"] = limit_msgs



        # Track unrealised PnL history (last 20 snapshots)

        total_unrealised = sum(p["unrealised"] for p in open_positions)

        hist = _LEARNING["open_pnl_history"]

        hist.append(round(total_unrealised, 2))

        if len(hist) > 20:

            hist.pop(0)



        update["open_positions"]   = open_positions

        update["open_count"]       = len(open_positions)

        update["total_unrealised"] = round(total_unrealised, 2)

        update["learning"]         = {

            "wins":          _LEARNING["wins"],

            "losses":        _LEARNING["losses"],

            "total_seen":    _LEARNING["total_seen"],

            "score_adjust":  _LEARNING["score_adjust"],

            "grade_stats":   _LEARNING["grade_stats"],

            "direction_stats": _LEARNING["direction_stats"],

            "pnl_history":   hist[-20:],

            "ob_stats":      dict(self._ob_stats),

            "ltf_stats":     _LEARNING["ltf_stats"],

            "exit_reason_stats": _LEARNING.get("exit_reason_stats", {}),

            "time_exit_stats": {

                "time_smart": _LEARNING.get("exit_reason_stats", {}).get(
                    "time_smart", {"count": 0, "wins": 0, "pnl": 0.0}
                ),

                "time_hard": _LEARNING.get("exit_reason_stats", {}).get(
                    "time_hard", {"count": 0, "wins": 0, "pnl": 0.0}
                ),

            },

        }

        update["pending_limits"] = {

            str(t): polls for t, polls in self._pending_limits.items()

        }



        # â”€â”€ Live control overrides from dashboard (checked every poll) â”€â”€â”€â”€

        if CONTROL_FILE.exists():

            try:

                _ctrl = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))

                _mode = _ctrl.get("entry_mode", "").lower()

                if _mode in ("zone_only", "market_any") and _mode != self.entry_mode:

                    logger.info("[control] Entry mode changed: %s â†’ %s", self.entry_mode, _mode)

                    self.entry_mode = _mode

                if "live_mode" in _ctrl:

                    _want_live = bool(_ctrl["live_mode"])

                    if _want_live == self.dry_run:  # mismatch â€” flip

                        self.dry_run = not _want_live

                        logger.info(

                            "[control] Live mode changed by dashboard: dry_run=%s",

                            self.dry_run,

                        )

            except Exception:

                pass



        # â”€â”€ Manual override â€” always checked first (highest priority) â”€â”€â”€â”€â”€â”€

        # (Manual override already handled before circuit-breaker above)



        # â”€â”€ Live candle analysis â€” always runs every poll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Analysis runs regardless of open position count so the dashboard

        # always has fresh market intelligence to display.

        # Clear stale LTF reversal flag from previous poll -- gate re-sets it each cycle.
        self._ltf_reversal_pending = None

        live_result = None

        if self._mt5_connected and self.engine is not None:

            df = self._fetch_candles(self.n_candles)

            if df is not None and len(df) >= 50:

                # Update market regime (used for adaptive BE and lot sizing)

                if self._regime_clf:

                    self._current_regime = self._regime_clf.classify(df)

                    logger.debug("[regime] %s conf=%.2f lot_x%.2f",

                                 self._current_regime.regime.value,

                                 self._current_regime.confidence,

                                 self._current_regime.lot_scalar)

                    update["regime"] = {

                        "name":               self._current_regime.regime.value,

                        "confidence":         round(self._current_regime.confidence, 3),

                        "lot_scalar":         self._current_regime.lot_scalar,

                        "be_atr_mult":        self._current_regime.be_atr_mult,

                        "allow_countertrend": self._current_regime.allow_countertrend,

                        "tp_scalar":          self._current_regime.tp_scalar,

                    }

                logger.info("Running live analysis on %d bars (%s %s)",

                            len(df), self.asset, self.timeframe)

                try:

                    _tf_data = self._fetch_mtf_candles(primary_df=df)

                    self._last_tf_data = _tf_data or {}

                    logger.info("MTF data available: %s",

                                list(_tf_data.keys()) if _tf_data else "none")

                    live_result = self.engine.analyze_data(

                        df,

                        asset=self.asset,

                        timeframe=self.timeframe,

                        tf_data=_tf_data,

                        render_chart=False,

                        save_to_db=False,

                    )

                except Exception as exc:

                    logger.error("Live analysis pipeline error: %s", exc, exc_info=True)



        if live_result and live_result.confluence:

            c   = live_result.confluence

            ms  = live_result.ms

            sr  = live_result.sr

            vw  = live_result.vwap

            fib = live_result.fib

            smc = live_result.smc

            mtf = live_result.mtf



            _sup = sr.nearest_support.level   if (sr and sr.nearest_support)   else None

            _res = sr.nearest_resistance.level if (sr and sr.nearest_resistance) else None

            _atr = ms.current_atr if ms else None

            _price = live_result.current_price



            _la = {

                "grade":      c.grade,

                "score":      round(c.total, 1),

                "direction":  c.direction,

                "reasons":    (c.reasons or [])[:8],

                "price":      round(_price, 4) if _price else None,

                "atr":        round(_atr, 4)  if _atr  else None,

                "nearest_support":    round(_sup, 4) if _sup else None,

                "nearest_resistance": round(_res, 4) if _res else None,

                "structure":  ms.structure_type.name.lower() if ms else None,

                "strength":   round(ms.trend_strength * 100) if (ms and ms.trend_strength) else None,

                "bos_count":  len(ms.bos_events)   if (ms and ms.bos_events)   else 0,

                "choch_count":len(ms.choch_events)  if (ms and ms.choch_events) else 0,

                "vwap_signal": vw.signal         if vw else None,

                "vwap_value":  round(vw.vwap, 4) if vw else None,

                "fvg_count":   len(smc.fair_value_gaps) if (smc and smc.fair_value_gaps) else 0,

                "ob_count":    len(smc.order_blocks)    if (smc and smc.order_blocks)    else 0,

                "rr_min":      RR_MIN_LONG if (c.direction or "") == "bullish" else RR_MIN_SHORT,

                "updated_at":  datetime.utcnow().isoformat(),

            }

            # MTF per-timeframe breakdown

            if mtf:

                _la["mtf_score"]      = round(mtf.alignment_score, 3)

                _la["mtf_bias"]       = mtf.primary_bias

                _la["mtf_confluence"] = mtf.confluence_level

                _la["mtf_biases"]     = [

                    {

                        "tf":     b.timeframe,

                        "bias":   b.bias,

                        "score":  round(b.score, 2),

                        "weight": round(b.weight, 2),

                    }

                    for b in mtf.biases

                ]

            # Fibonacci key levels (top 4 closest to price)

            if fib and fib.levels and _price:

                _la["fib_levels"] = [

                    {"pct": l.label, "price": round(l.price, 4)}

                    for l in sorted(fib.levels, key=lambda x: abs(x.price - _price))[:4]

                ]

            # Chart pattern type at entry (feeds Architecture Engine Status card)

            _pat_result = live_result.pat

            if _pat_result and _pat_result.top_pattern:

                from ml.pattern_learner import classify_pattern_type

                _tp = _pat_result.top_pattern

                _pt_id = classify_pattern_type(_tp.name)

                _pt_labels = {

                    0: "Unknown", 1: "Cont. Bull", 2: "Cont. Bear",

                    3: "Rev. Bull", 4: "Rev. Bear", 5: "Breakout Neutral",

                }

                _la["top_pattern"] = {

                    "name":       _tp.name,

                    "confidence": round(_tp.confidence, 3),

                    "direction":  _tp.direction,

                    "type_id":    _pt_id,

                    "type_label": _pt_labels.get(_pt_id, "Unknown"),

                }

            else:

                _la["top_pattern"] = None



            self._last_live_analysis    = _la

            update["live_analysis"]       = _la

            effective_min = self.min_score + _LEARNING["score_adjust"]

            update["effective_min_score"] = round(effective_min, 1)

            logger.info("Live analysis: grade=%s score=%.0f direction=%s price=%s",

                        c.grade, c.total, c.direction, _price)



        # â”€â”€ Skip EXECUTION if max open positions reached â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # (analysis already ran above, so dashboard always has fresh data)

        # Session filter: update state and guard new entries

        if ENABLE_SESSION_FILTER and self._session_clf:

            self._current_session = self._session_clf.classify()

            update["session"] = self._current_session.session.value

            update["session_detail"] = {

                "name":             self._current_session.session.value,

                "spread_tolerance": self._current_session.spread_tolerance,

                "skip_new_entries": self._current_session.skip_new_entries,

            }

            if self._last_exec_quality_result:

                update["last_exec_quality"] = self._last_exec_quality_result

            update["last_htf_lot_mult"] = round(self._htf_lot_mult, 3)

            if self._current_session.skip_new_entries:

                # Special case: Asian session + Trend Exhaustion regime
                # Exhaustion during Asian hours can produce clean reversal setups
                # (wick rejection, stop-hunt + close above/below key level).
                # Allow entries but apply aggressive score floor (+10) and reduce
                # lot scalar to 0.35 so only high-quality setups pass.
                msg = (f"Session dead zone ({self._current_session.session.value})"
                       f" - skipping new entries")
                logger.info("[session] %s", msg)
                update.setdefault("last_action", msg)
                return update



        # -- Direction-flip counter update (read by _qualifies_result) --------

        # Runs once per poll so pyramid + main-signal calls share the same count.

        if live_result and live_result.confluence:

            _poll_dir = (live_result.confluence.direction or "").lower()

            if _poll_dir in ("bullish", "bearish"):

                if not self._last_confirmed_direction:

                    self._last_confirmed_direction = _poll_dir

                elif _poll_dir != self._last_confirmed_direction:

                    if _poll_dir == self._direction_flip_pending:

                        self._direction_flip_polls += 1

                    else:

                        self._direction_flip_pending = _poll_dir

                        self._direction_flip_polls = 1

                else:

                    self._direction_flip_pending = ""

                    self._direction_flip_polls = 0



        # Balance-tiered position limit
        # Small  (-): max 2 -- one SL cannot blow the account
        # Medium (-): max 3
        # Normal (>):    max 5
        if 0 < _balance < SMALL_ACCOUNT_THRESHOLD:
            MAX_OPEN = SMALL_ACCOUNT_MAX_OPEN
        elif 0 < _balance < MEDIUM_ACCOUNT_THRESHOLD:
            MAX_OPEN = MEDIUM_ACCOUNT_MAX_OPEN
        else:
            MAX_OPEN = NORMAL_ACCOUNT_MAX_OPEN

        n_open   = len(open_positions)

        if n_open >= MAX_OPEN:

            # â”€â”€ Pyramid add-on: ride a confirmed winning trend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

            # Allow ONE extra position beyond MAX_OPEN when all of these hold:

            #   a) all open positions are profitable â‰¥ PYRAMID_MIN_PROFIT_PTS

            #   b) all open positions are in the same direction

            #   c) live signal aligns with that direction

            #   d) we are not already at MAX_OPEN + PYRAMID_MAX_ADD

            pyramided = False

            if (

                live_result and live_result.confluence

                and n_open < MAX_OPEN + PYRAMID_MAX_ADD

                and open_positions

            ):

                all_profit = all(

                    pos.get("unrealised", 0) > 0

                    and abs((pos.get("current") or pos["entry"]) - pos["entry"])

                       >= PYRAMID_MIN_PROFIT_PTS

                    for pos in open_positions

                )

                directions = {pos["direction"] for pos in open_positions}

                signal_dir = (live_result.confluence.direction or "").upper()

                dir_map = {"BULLISH": "BUY", "BEARISH": "SELL"}

                signal_mt5 = dir_map.get(signal_dir, signal_dir)

                all_same_dir  = len(directions) == 1

                matches_signal = signal_mt5 in directions



                if all_profit and all_same_dir and matches_signal and self._qualifies_result(live_result):

                    try:

                        # Execute at half risk so the add-on is conservative

                        orig_frac = self.risk_fraction

                        self.risk_fraction *= PYRAMID_RISK_MULT

                        action_msg = self._execute_from_result(live_result, label="pyramid")

                        self.risk_fraction = orig_frac

                        pyramided = True

                    except Exception as _exc:

                        self.risk_fraction = orig_frac

                        action_msg = f"[pyramid] Execution error: {_exc}"

                        logger.error("Pyramid error: %s", _exc, exc_info=True)

                    update["last_action"]  = action_msg

                    update["total_trades"] = self._total_trades

                    logger.info("Pyramid add: %s", action_msg)

                    return update



            if not pyramided:

                msg = (f"Already {n_open} open position(s) on {self.asset} "

                       f"â€” skipping new signal until some close.")

                logger.info(msg)

                update["last_action"] = msg

                return update



        # â”€â”€ Execute if live signal qualifies â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        if live_result and live_result.confluence:

            c = live_result.confluence

            effective_min = self.min_score + _LEARNING["score_adjust"]

            if self._qualifies_result(live_result):

                # ── Session-direction loss halt gate ───────────────────────────
                # After DIRECTION_LOSS_HALT consecutive SL losses in the same
                # direction within the current session, block that direction.
                # Counters reset when the session label changes.
                _cur_sess_label = getattr(
                    getattr(self._current_session, "session", None), "value", ""
                ) or ""
                if _cur_sess_label != self._halt_session_label:
                    # New session — reset all counters
                    self._session_dir_losses  = {}
                    self._halt_session_label  = _cur_sess_label
                _sig_mt5_dir = "BUY" if (c.direction or "").lower() in ("bullish", "long") else "SELL"
                _opp_mt5_dir = "SELL" if _sig_mt5_dir == "BUY" else "BUY"
                _dir_losses  = self._session_dir_losses.get(_sig_mt5_dir, 0)
                # Drought override: raise halt threshold by 2 if no trade in TRADE_DROUGHT_POLLS
                # AND both directions have not simultaneously exceeded the base threshold.
                # This prevents the bot from halting both directions and going silent all day.
                _polls_no_trade_h = self._poll_count - self._last_entry_poll
                _drought_h = _polls_no_trade_h >= TRADE_DROUGHT_POLLS
                _opp_losses = self._session_dir_losses.get(_opp_mt5_dir, 0)
                _both_halted = (_dir_losses >= DIRECTION_LOSS_HALT
                                and _opp_losses >= DIRECTION_LOSS_HALT)
                _effective_halt = DIRECTION_LOSS_HALT + (2 if _drought_h else 0)
                if _dir_losses >= _effective_halt and not (_both_halted and _drought_h):
                    msg = (
                        f"[halt] {_sig_mt5_dir} direction halted in {_cur_sess_label}: "
                        f"{_dir_losses} SL losses (threshold={_effective_halt})"
                    )
                    logger.info(msg)
                    update["last_action"] = msg
                    # Do NOT execute — fall through to bot_status update
                else:
                    # ── Entry cooldown gate ────────────────────────────────────
                    # Prevent rapid-fire accumulation: after any new entry in a given
                    # direction, enforce a minimum gap of ENTRY_COOLDOWN_POLLS before
                    # the next entry in that same direction.
                    _sig_dir_cool = (c.direction or "").lower()
                    _polls_since  = self._poll_count - self._last_entry_poll
                    if (
                        _sig_dir_cool == self._last_entry_dir
                        and _polls_since < ENTRY_COOLDOWN_POLLS
                    ):
                        msg = (
                            f"[cooldown] Entry suppressed — last {_sig_dir_cool} entry was "
                            f"{_polls_since} poll(s) ago (need {ENTRY_COOLDOWN_POLLS})"
                        )
                        logger.info(msg)
                        update["last_action"] = msg
                    else:
                        try:
                            action_msg = self._execute_from_result(live_result, label="live")
                            # Record this entry for the cooldown gate
                            self._last_entry_poll = self._poll_count
                            self._last_entry_dir  = _sig_dir_cool
                        except Exception as _exc:
                            action_msg = f"[live] Execution error: {_exc}"
                            logger.error("_execute_from_result error: %s", _exc, exc_info=True)
                        update["last_action"]  = action_msg
                    update["total_trades"] = self._total_trades
                    return update

            else:

                _grade_ok = GRADE_RANK.get(c.grade or "F", 0) >= GRADE_RANK.get(self.min_grade, 3)
                _score_ok = c.total >= effective_min
                if _grade_ok and _score_ok:
                    # Grade and score cleared — a secondary gate blocked it (HTF/LTF/regime/session)
                    msg = (f"Live signal gated (grade/score OK — blocked by gate): "
                           f"grade={c.grade} score={c.total:.0f} "
                           f"ltf_state={self._last_ltf_state} "
                           f"(dir={c.direction}) — see logs for gate detail")
                else:
                    msg = (f"Live signal below threshold: "
                           f"grade={c.grade} score={c.total:.0f} vs min={effective_min:.0f} "
                           f"(dir={c.direction})")

                logger.info(msg)

                update["last_action"] = msg

                # LTF reversal entry removed: micro-LTF counter-signals during a
                # pullback/sweep are retracements in the macro trend, not true reversals.
                # Entering against the HTF direction at that point = buying the sweep top.
                # The hard block is the correct behaviour — sit idle until LTFs realign.
                self._ltf_reversal_pending = None   # always clear



        # -- LTF momentum scalp ----------------------------------------

        # When 5M+1M are both strongly aligned WITH the primary signal,

        # take a trend-following momentum scalp to capture the LTF surge

        # in the main trade direction (5M+1M momentum confirms the HTF bias).

        _la_now = self._last_live_analysis

        if _la_now:

            _ltf_map = {

                b["tf"].lower(): b

                for b in (_la_now.get("mtf_biases") or [])

                if b["tf"].lower() in ("5m", "1m", "m5", "m1")

            }

            _5m_b = _ltf_map.get("5m") or _ltf_map.get("m5")

            _1m_b = _ltf_map.get("1m") or _ltf_map.get("m1")

            if _5m_b and _1m_b:

                _5m_s = _5m_b["score"]

                _1m_s = _1m_b["score"]

                _primary_dir_now = _la_now.get("direction", "sideways")

                _scalp_dir = None

                if _5m_s >= LTF_SCALP_THRESHOLD and _1m_s >= LTF_SCALP_THRESHOLD:

                    _scalp_dir = "bullish"

                elif _5m_s <= -LTF_SCALP_THRESHOLD and _1m_s <= -LTF_SCALP_THRESHOLD:

                    _scalp_dir = "bearish"

                if _scalp_dir and _scalp_dir == _primary_dir_now:   # trend-aligned only

                    logger.info(

                        "LTF scalp signal: 5M=%+.2f 1M=%+.2f dir=%s (primary=%s) [aligned]",

                        _5m_s, _1m_s, _scalp_dir, _primary_dir_now,

                    )

                    try:

                        action_msg = self._execute_ltf_scalp(_scalp_dir)

                    except Exception as _exc:

                        action_msg = f"[LTF-scalp] Error: {_exc}"

                        logger.error("LTF scalp error: %s", _exc, exc_info=True)

                    update["last_action"]  = action_msg

                    update["total_trades"] = self._total_trades

                    return update





        # â”€â”€ DB signals (SECONDARY source â€” dashboard-saved analyses) â”€â”€â”€â”€â”€â”€â”€

        if not self.use_db:

            return update

        if not callable(list_analyses):

            logger.warning("DB fallback unavailable: storage.database.list_analyses is not loaded")

            return update



        try:

            rows = list_analyses(asset=self.asset, timeframe=self.timeframe, limit=10)

        except Exception as exc:

            logger.warning("DB poll error: %s", exc)

            return update



        fresh = [r for r in rows if r["id"] > self._last_id]

        if not fresh:

            return update



        # Take the highest-scoring fresh signal

        best = max(fresh, key=lambda r: r.get("confluence_score") or 0)

        self._last_id = max(r["id"] for r in fresh)



        effective_min = self.min_score + _LEARNING["score_adjust"]

        update.update({

            "last_signal_id":        self._last_id,

            "last_signal_grade":     best.get("grade", "?"),

            "last_signal_direction": best.get("direction", "?"),

            "last_signal_score":     best.get("confluence_score", 0),

            "effective_min_score":   round(effective_min, 1),

        })



        if not self._qualifies(best):

            msg = (f"Signal #{best['id']} below threshold "

                   f"(grade={best.get('grade')} score={best.get('confluence_score', 0):.0f} "

                   f"vs effective min {effective_min:.0f}) â€” skipped")

            logger.info(msg)

            update["last_action"] = msg

            return update



        action_msg = self._execute(best)

        update["last_action"]   = action_msg

        update["total_trades"]  = self._total_trades

        return update



    # â”€â”€ Status file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_runtime_telemetry(self, poll_duration_ms: float) -> dict:

        poll_duration_ms = max(0.0, float(poll_duration_ms))
        self._poll_duration_history_ms.append(round(poll_duration_ms, 2))
        if len(self._poll_duration_history_ms) > 200:
            self._poll_duration_history_ms = self._poll_duration_history_ms[-200:]

        ordered = sorted(self._poll_duration_history_ms)
        p50_idx = int(round(0.50 * (len(ordered) - 1))) if ordered else 0
        p95_idx = int(round(0.95 * (len(ordered) - 1))) if ordered else 0
        p50 = ordered[p50_idx] if ordered else 0.0
        p95 = ordered[p95_idx] if ordered else 0.0

        return {
            "poll_duration_ms": round(poll_duration_ms, 2),
            "poll_duration_p50_ms": round(float(p50), 2),
            "poll_duration_p95_ms": round(float(p95), 2),
            "recent_poll_durations_ms": self._poll_duration_history_ms[-20:],
            "poll_error_count": self._poll_error_count,
            "daily_trade_pnl": round(float(self._daily_trade_pnl_cache), 2),
            "daily_loss_pct": round(float(self._daily_loss_pct_cache), 4),
            "daily_loss_halted": bool(self._daily_loss_halted),
            "daily_loss_halt_until_ts": self._daily_loss_halt_until_ts,
            "daily_profit_protect": bool(self._daily_profit_protect),
        }



    def _write_status(self, extra: dict) -> None:

        # Always carry forward the most recent live analysis so the dashboard

        # doesn't go blank on a transient candle-fetch failure.

        merged: dict = {}

        if self._last_live_analysis and "live_analysis" not in extra:

            merged["live_analysis"] = self._last_live_analysis

        payload = {

            "version":        "2.0",

            "started_at":     self._started_at,

            "last_poll":      datetime.utcnow().isoformat(),

            "asset":          self.asset,

            "timeframe":      self.timeframe,

            "dry_run":        self.dry_run,

            "entry_mode":     self.entry_mode,

            "min_grade":      self.min_grade,

            "min_score":      self.min_score,

            "risk_pct":       self.risk_fraction * 100,

            "poll_interval":  self.poll_interval,

            "n_candles":      self.n_candles,

            "use_db":         self.use_db,

            "policy_gate_enabled": ENABLE_INSTITUTIONAL_POLICY_GATE,

            "mt5_available":  MT5_AVAILABLE,

            "mt5_connected":  self._mt5_connected,

            "total_trades":   self._total_trades,

            "open_positions": [],

            "open_count":     0,

            "total_unrealised": 0.0,

            "learning": _LEARNING,

            # Persist day-start baseline so restarts within the same calendar day
            # restore the same circuit-breaker reference point.
            "day_start_balance": self._day_start_balance,
            "day_start_date":    self._day_start_date,
            "daily_loss_halt_until_ts": self._daily_loss_halt_until_ts,

            **self._mt5_account(),

            **merged,

            **extra,

        }

        if build_institutional_risk_performance_runtime and write_institutional_risk_performance_runtime:

            try:

                institutional_profile = build_institutional_risk_performance_runtime(payload)

                payload["institutional_risk_performance"] = institutional_profile

                write_institutional_risk_performance_runtime(institutional_profile)

            except Exception as exc:

                logger.debug("Institutional runtime artifact write skipped: %s", exc)

        try:

            STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        except Exception as exc:

            logger.warning("Cannot write status file: %s", exc)



    # â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    def run(self) -> None:

        logger.info("=" * 60)

        logger.info(

            "Prometheus Live Bot v2 | %s %s | gradeâ‰¥%s | scoreâ‰¥%.0f | dry_run=%s",

            self.asset, self.timeframe, self.min_grade, self.min_score, self.dry_run,

        )

        if self.dry_run:

            logger.info("DRY RUN â€” signals logged but NO real orders placed.")

        else:

            logger.info("LIVE MODE â€” real orders will be sent to MT5.")

        logger.info("Entry mode: %s", self.entry_mode.upper())

        logger.info("=" * 60)



        init_db()



        # â”€â”€ Restore persistent learning state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        _bootstrap_from_db()          # rebuild from DB if no file yet

        _load_learning(self._ob_stats)  # merge saved file (overrides bootstrap)

        # Restore per-ticket attribution dicts that survive restarts
        _meta = _LEARNING.get("open_trade_meta", {})
        for _tk_str, _m in _meta.items():
            try:
                _tk = int(_tk_str)
                self._entry_grade[_tk]     = _m.get("grade",     "F")
                self._ltf_entry_state[_tk] = _m.get("ltf_state", "unknown")
                self._entry_regime[_tk]    = _m.get("regime",    "unknown")
                self._entry_session[_tk]   = _m.get("session",   "unknown")
                # Restore entry-quality positioning data
                self._entry_zone_pos[_tk]  = _m.get("zone_pos",  "unknown")
                self._entry_zone_type[_tk] = _m.get("zone_type", "unknown")
                self._entry_sl_atr[_tk]    = _m.get("sl_atr",    None)
                self._entry_score[_tk]     = _m.get("score",     None)
            except (ValueError, TypeError):
                pass
        if _meta:
            logger.info("[LML] Restored open_trade_meta for %d ticket(s)", len(_meta))

        # Always try MT5 (needed for live candle data + order execution)

        if MT5_AVAILABLE:

            self._mt5_connected = self._connect_mt5()

            if not self._mt5_connected:

                logger.warning(

                    "MT5 not available â€” live candle analysis disabled; will rely on DB signals."

                )

            if not self._mt5_connected and not self.dry_run:

                logger.warning("Switching to dry-run mode (MT5 unavailable).")

                self.dry_run = True

        else:

            logger.warning("MetaTrader5 package not installed â€” running in dry-run / DB-only mode.")



        # â”€â”€ Lazy-initialize the Prometheus engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        logger.info("Initializing Prometheus analysis engine â€¦")

        try:

            from prometheus_core import Prometheus  # noqa: PLC0415

            self.engine = Prometheus()

            logger.info("Prometheus engine ready.")

        except Exception as exc:

            logger.error("Could not initialize Prometheus engine: %s â€” live analysis disabled.", exc)

            self.engine = None



        # Remove stale stop flag (PID lock already acquired at __main__ entry)
        STOP_FLAG.unlink(missing_ok=True)

        self._write_status({"last_action": "started"})



        # â”€â”€ Backfill any already-open positions into the DB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Covers positions opened before this session (or before DB recording

        # was added). Upserts so restarting the bot is always safe.

        if not self.dry_run and MT5_AVAILABLE and self._mt5_connected:

            try:

                _existing = self._get_open_positions()

                for _p in _existing:

                    save_trade({

                        "trade_id":    f"live_{_p['ticket']}",

                        "source":      "live",

                        "asset":       self.asset,

                        "timeframe":   self.timeframe,

                        "direction":   _p["direction"],

                        "entry_price": _p["entry"],

                        "sl_price":    _p["sl"],

                        "tp_price":    _p["tp"],

                        "size":        _p["lots"],

                        "status":      "open",

                        "entry_bar":   0,

                    })

                if _existing:

                    logger.info("Backfilled %d open position(s) into trade DB.", len(_existing))

            except Exception as _bfe:

                logger.debug("Startup backfill error: %s", _bfe)



        try:

            while True:

                if STOP_FLAG.exists():

                    logger.info("Stop flag detected â€” shutting down.")

                    STOP_FLAG.unlink(missing_ok=True)

                    break



                try:

                    _poll_started = time.perf_counter()

                    update = self._poll()

                    _poll_elapsed_ms = (time.perf_counter() - _poll_started) * 1000.0

                    update["runtime_telemetry"] = self._build_runtime_telemetry(_poll_elapsed_ms)

                    self._write_status(update)

                except Exception as exc:

                    self._poll_error_count += 1

                    logger.error("Poll error (bot continues): %s", exc, exc_info=True)



                # Sleep in small increments so stop flag is checked promptly

                for _ in range(self.poll_interval):

                    if STOP_FLAG.exists():

                        break

                    time.sleep(1)



        except KeyboardInterrupt:

            logger.info("Interrupted by user (Ctrl+C)")

        except Exception as exc:

            logger.error("Fatal error in main loop: %s", exc, exc_info=True)

        finally:

            _release_pid_lock()

            self._write_status({"last_action": "stopped"})

            if self._mt5_connected and MT5_AVAILABLE:

                try:

                    mt5.shutdown()

                    logger.info("MT5 disconnected.")

                except Exception:

                    pass

            logger.info("Bot exited cleanly.")





# =============================================================================

def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(

        description="Prometheus Live Trading Bot",

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

    )

    p.add_argument("--asset",     default="XAUUSD", help="Symbol to trade")

    p.add_argument("--tf",        default="4H",     help="Timeframe label (M5/H1/H4/D1 â€¦)")

    p.add_argument("--min-grade", default="B",      help="Minimum grade (A/B/C/D)")

    p.add_argument("--min-score", type=float, default=65.0, help="Min confluence score (0-100)")

    p.add_argument("--risk",      type=float, default=1.0,  help="Risk %% of balance per trade")

    p.add_argument("--poll",      type=int,   default=60,   help="Seconds between polls")

    p.add_argument("--candles",   type=int,   default=500,  help="Candles to fetch per cycle")

    p.add_argument("--no-db",     action="store_true",      help="Skip DB signal check")

    p.add_argument(

        "--live",

        action="store_true",

        help="Enable real MT5 order placement (default: dry-run only)",

    )

    p.add_argument(

        "--entry-mode",

        default="zone_only",

        choices=["zone_only", "market_any"],

        help=(

            "zone_only: only enter when price is at/near a fresh OB or S/R zone (safer). "

            "market_any: enter at market whenever score qualifies without waiting for a zone."

        ),

    )

    return p.parse_args()





if __name__ == "__main__":

    # Acquire single-instance lock before any heavy init.
    # Uses a TCP socket bind — atomic on Windows; any duplicate process
    # will fail to bind port BOT_LOCK_PORT and exit immediately.
    _acquire_pid_lock()

    args = _parse_args()

    bot = PrometheusLiveBot(

        asset         = args.asset,

        timeframe     = args.tf,

        min_grade     = args.min_grade,

        min_score     = args.min_score,

        risk_pct      = args.risk,

        poll_interval = args.poll,

        n_candles     = args.candles,

        use_db        = not args.no_db,

        dry_run       = not args.live,

        entry_mode    = args.entry_mode,

    )

    bot.run()


