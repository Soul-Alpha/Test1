"""
Zeus — Prometheus LTF Scalp Backtester
========================================
Walk-forward backtester that replicates the exact live-trader logic on
Lower Time Frame (30M / 15M / 5M) historical data.

Faithful replicas of:
  - _qualifies_result()  : all gates (HTF, LTF trap, BOS, CT, session, regime)
  - _execute_from_result(): zone filter, OB detection, small-account single-leg
  - _place_limit_order() : pending limit placement and LTF-trap cancellation
  - _manage_positions()  : ATR trailing SL + BE lock
  - _check_5m_exits()   : 3-bar opposing candle early exit
  - _calc_lot()         : 2% risk rule, small-account detection

After the run:
  - Collects one feature vector per closed trade
  - Trains XGBoost to predict win probability
  - Outputs feature importance ranked by predictive power
  - Prints a detailed console report (by entry type / zone / LTF state /
    grade / session / pattern) with top win-rate combinations

Usage
-----
  from backtesting.scalp_backtester import ScalpBacktester, ScalpBacktestConfig
  from datetime import datetime

  cfg = ScalpBacktestConfig(
      asset          = "XAUUSDm",
      primary_tf     = "30m",
      date_from      = datetime(2025, 1, 1),
      date_to        = datetime(2026, 6, 1),
      initial_balance= 120.0,
      risk_pct       = 2.0,
      min_grade      = "B",
      min_score      = 65.0,
      entry_mode     = "zone_only",
      train_ml       = True,
  )
  bt  = ScalpBacktester(cfg)
  res = bt.run()
  bt.print_report(res)
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "live_bot"))

logger = logging.getLogger(__name__)

# ── Optional heavy imports — graceful degradation ─────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed — historical data must be loaded from CSV")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — ML training disabled")

try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — ML metrics disabled")

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


# =============================================================================
# Strategy signal — the universal contract for pluggable custom strategies
# =============================================================================

@dataclass
class StrategySignal:
    """
    Minimal signal returned by any strategy_fn callable.

    Zeus uses whatever fields are provided; None fields fall back to
    Zeus's own ATR-based SL/TP calculation so simple strategies only
    need to set direction + score.

    Required
    --------
    direction : "long" | "short" | "flat"
        "flat" / anything else → Zeus skips this bar.

    Optional overrides
    ------------------
    entry_price : float | None
        None → market order at current bar close.
        Not None → Zeus places a limit order at this price.
    sl : float | None
        None → Zeus calculates from ATR (1×ATR buffer from entry).
    tp1 : float | None
        None → Zeus calculates from sl_dist × tp1_rr.
    tp2 : float | None
        None → Zeus calculates from sl_dist × tp2_rr.
    score : float
        0–100.  Used as an ML feature and, when
        strategy_fn_apply_score_gate=True, as a min_score gate.
    grade : str
        "A"/"B"/"C"/"D"/"F".  Used as ML feature; gated when
        strategy_fn_apply_score_gate=True.
    meta : dict
        Extra key/value pairs forwarded into ML features.
        Keys must be numeric-valued for XGBoost training.
    """
    direction:   str   = "flat"            # "long" | "short" | "flat"
    score:       float = 0.0
    grade:       str   = "F"
    entry_price: Optional[float] = None    # None → market fill
    sl:          Optional[float] = None    # None → ATR-based
    tp1:         Optional[float] = None    # None → ATR-based
    tp2:         Optional[float] = None    # None → ATR-based
    meta:        Dict[str, Any]  = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """True only for a long or short signal — not flat/sideways."""
        return self.direction in ("long", "short")


# =============================================================================
# Config
# =============================================================================

@dataclass
class ScalpBacktestConfig:
    """All parameters for a scalp backtest run."""
    # Market
    asset:           str   = "XAUUSDm"
    primary_tf:      str   = "30m"        # "30m" | "15m" | "5m"
    context_tfs:     List[str] = field(
        default_factory=lambda: ["5m", "15m", "30m", "1h", "4h", "1d"]
    )
    date_from:       Optional[datetime] = None   # None = last 500 bars
    date_to:         Optional[datetime] = None   # None = now
    n_bars:          int   = 1000         # used when date_from/to not set

    # Account
    initial_balance: float = 120.0
    risk_pct:        float = 2.0          # % of equity per trade

    # Signal quality gates
    min_grade:       str   = "A"          # Phase 1: default A (OB entries are A-grade)
    min_score:       float = 80.0         # Phase 1: raised from 65 (all wins ≥87)
    entry_mode:      str   = "zone_only"  # "zone_only" | "market_any"
    enabled_sessions: List[str] = field(
        default_factory=lambda: [
            "asian",
            "london_open",
            "london",
            "ny_lunch",
            "london_ny_overlap",
            # ny_afternoon and dead_zone omitted — they are hard-blocked by
            # _skip_session regardless; listing them would be misleading.
        ]
    )
    strategy_name:   str   = "zeus_ltf"    # label for the active LTF strategy profile

    # ── Phase 1 optimisation filters ──────────────────────────────────────────
    # SR entries have near-zero edge (data: $0.08/trade avg P&L vs $12.81 for OB).
    # Require extra score premium to allow SR entries through.
    sr_min_score_premium: float = 15.0   # SR entry needs min_score + this extra pts
    # High-ATR-rank market regimes are noise-heavy; losses cluster at atr_rank ≥ 0.83.
    # Block one_counter entries when volatility rank is extreme.
    block_high_atr_rank: float = 0.85    # block one_counter if atr_rank ≥ this (0=off)
    # Relax both_confirmed score gate from 85 → 80 (100% WR at all score levels)
    both_confirmed_min_score: float = 80.0

    # Simulation costs
    slippage_pts:    float = 2.0          # XAUUSDm: ~2 point spread slip
    commission_pct:  float = 0.0003       # 0.03% per side

    # Small-account thresholds (mirrors live trader)
    small_acct_threshold: float = 120.0
    small_acct_max_open:  int   = 2
    medium_acct_threshold:float = 500.0
    medium_acct_max_open: int   = 3
    normal_acct_max_open: int   = 5

    # Signal evaluation cadence
    signal_stride:   int   = 3            # re-run analysis every N bars
    warmup_bars:     int   = 50           # skip first N bars

    # Risk management (Phase 2: tp1_rr 1.0→1.5, tp2_rr 3.0→5.0)
    tp1_rr:          float = 1.5          # Phase 2: lock 50% profit at better ratio
    tp2_rr:          float = 5.0          # Phase 2: let OB runners go further
    trail_atr_mult:  float = 1.2          # Phase 2: tighter trail on confirmed moves
    be_atr_trigger:  float = 0.35         # Phase 2: lock BE earlier
    be_profit_pts:   float = 3.0
    zone_atr_thresh: float = 1.0
    limit_order_expiry: int = 240         # bars before pending limit cancelled
    max_limit_dist_atr: float = 3.0
    small_acct_max_sl_atr: float = 2.5    # relaxed (was 1.5)

    # LTF exit
    m5_reversal_candles: int   = 3
    m5_min_profit_r:     float = 0.40

    # ML
    train_ml:        bool  = True
    ml_test_size:    float = 0.40
    unknown_regime_score_premium: float = 25.0  # fallback floor when regime classifier unavailable

    # Output
    report_path:     Optional[str] = None  # JSON output path
    verbose:         bool  = False

    # ── Strategy lab: pluggable strategy hook ─────────────────────────────
    # Set strategy_fn to a callable to replace Prometheus signal generation.
    # Signature: (df_slice, tf_data, bar_atr, regime, session_name) -> StrategySignal | None
    # None → Prometheus default path (_run_analysis + _qualifies) unchanged.
    strategy_fn: Optional[Any] = None
    # When True and strategy_fn is set, apply min_grade/min_score gate to
    # the signal's grade/score fields (same as Prometheus path).
    strategy_fn_apply_score_gate: bool = True

    # ── Phase 1: Entry discipline gates (off by default) ──────────────────
    # Minimum bars between any two entries (0 = disabled).
    entry_cooldown_bars:    int   = 0
    # Require N bars since direction change before allowing a flip entry.
    # e.g. 3 → must see 3 same-direction bars after last entry before flipping.
    direction_flip_min_bars: int  = 0
    # Halt entries in a session after N consecutive losses in that session.
    # Resets at each new session (0 = disabled).
    session_dir_loss_halt:  int   = 0

    # ── Phase 2: Exit parity with live bot (off by default) ───────────────
    # M5-style severity exit using current bar body ratio.
    m5_severity_enable:     bool  = False
    # Timeout exits: smart partial then hard full close.
    time_exit_enable:       bool  = False
    time_exit_smart_bars:   int   = 15   # bars open before 50 % partial
    time_exit_hard_bars:    int   = 30   # bars open before full close
    time_exit_profit_min:   float = 15.0 # min $ profit required for smart partial

    # ── Phase 3: Daily circuit breakers (off by default) ──────────────────
    # Halt all new entries when daily loss exceeds this % of day-start equity.
    max_daily_loss_pct:     float = 0.0
    # When daily gain exceeds this %, scale all new lots by daily_profit_lot_scalar.
    daily_profit_protect_pct: float = 0.0
    daily_profit_lot_scalar:  float = 0.5   # lot multiplier when protecting daily profits


# =============================================================================
# Trade record (per closed trade, carries ML features)
# =============================================================================

@dataclass
class ScalpTrade:
    trade_id:    str
    direction:   str       # "long" | "short"
    entry_type:  str       # "market" | "limit"
    entry_price: float
    sl_price:    float
    tp1_price:   float
    tp2_price:   float
    size:        float     # lots
    entry_bar:   int

    # Exit
    exit_bar:    Optional[int]   = None
    exit_price:  Optional[float] = None
    exit_reason: str             = "open"   # "tp1"|"tp2"|"sl"|"trail"|"5m_exit"|"eod"
    stop_type:   str             = "none"   # "initial_sl"|"be_sl"|"trail_sl"|"none"
    pnl:         float           = 0.0
    rr:          float           = 0.0
    status:      str             = "open"   # "open"|"won"|"lost"

    # ML features
    grade:       str   = "F"
    score:       float = 0.0
    ltf_state:   str   = "unknown"
    zone_type:   str   = "no_zone"      # "ob"|"sr"|"no_zone"
    zone_pos:    str   = "no_zone"      # "deep"|"mid"|"shallow"|"no_zone"
    sl_atr:      float = 0.0            # sl_distance / atr at entry
    ob_direction:str   = "none"
    pattern_type:int   = 0
    regime:      str   = "unknown"
    session:     str   = "unknown"
    atr_rank:    float = 0.0
    hour_utc:    int   = 0
    dow:         int   = 0              # day of week 0=Mon

    # Strategy lab extras
    bars_open:           int  = 0    # bars elapsed from entry to close
    session_loss_streak: int  = 0    # consecutive session losses at entry time
    exit_timeout:        bool = False # True if closed by timeout exit logic

    @property
    def is_win(self) -> bool:
        return self.status == "won"

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.sl_price) * self.size


@dataclass
class PendingLimit:
    order_id:    str
    direction:   str       # "long" | "short"
    limit_price: float
    sl_price:    float
    tp1_price:   float
    tp2_price:   float
    size:        float
    placed_bar:  int
    polls_left:  int
    meta:        Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Backtest result
# =============================================================================

@dataclass
class ScalpBacktestResult:
    # Trades
    trades:       List[ScalpTrade] = field(default_factory=list)
    equity_curve: List[float]      = field(default_factory=list)

    # Core metrics
    initial_balance:   float = 0.0
    final_equity:      float = 0.0
    total_return_pct:  float = 0.0
    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    expectancy:        float = 0.0
    max_drawdown_pct:  float = 0.0
    sharpe_ratio:      float = 0.0
    calmar_ratio:      float = 0.0
    avg_rr:            float = 0.0
    total_trades:      int   = 0
    winning_trades:    int   = 0
    losing_trades:     int   = 0

    # Segment breakdowns: {key: {"n": int, "wins": int, "pnl": float}}
    by_entry_type:  Dict[str, Dict] = field(default_factory=dict)
    by_zone_type:   Dict[str, Dict] = field(default_factory=dict)
    by_ltf_state:   Dict[str, Dict] = field(default_factory=dict)
    by_grade:       Dict[str, Dict] = field(default_factory=dict)
    by_session:     Dict[str, Dict] = field(default_factory=dict)
    by_pattern_type:Dict[str, Dict] = field(default_factory=dict)
    by_regime:      Dict[str, Dict] = field(default_factory=dict)
    by_hour:        Dict[str, Dict] = field(default_factory=dict)

    # Top win-rate combinations (list of dicts)
    winning_combos: List[Dict] = field(default_factory=list)

    # ML
    feature_importance: List[Dict] = field(default_factory=list)
    ml_accuracy:   Optional[float] = None
    ml_roc_auc:    Optional[float] = None
    ml_summary: Dict[str, Any] = field(default_factory=dict)

    # Validation / diagnostics
    invalid_trade_setup_count: int = 0
    regime_unavailable_count: int = 0
    skipped_by_regime_count: int = 0
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    # Strategy metadata
    strategy_name: str = "zeus_ltf"

    # What improves win rate (text bullets)
    insights: List[str] = field(default_factory=list)


# =============================================================================
# Main backtester
# =============================================================================

class ScalpBacktester:
    """Walk-forward LTF scalp backtester using the Prometheus analysis pipeline."""

    # Maps primary_tf → MT5 constant key (used for fetching)
    _TF_MAP = {
        "1m":  "M1",  "m1":  "M1",
        "5m":  "M5",  "m5":  "M5",
        "15m": "M15", "m15": "M15",
        "30m": "M30", "m30": "M30",
        "1h":  "H1",  "h1":  "H1",
        "4h":  "H4",  "h4":  "H4",
        "1d":  "D1",  "d1":  "D1",
    }

    _GRADE_RANK = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    _TF_RANK    = {"1m":1,"5m":2,"15m":3,"30m":4,"1h":5,"4h":6,"1d":7,"1w":8}

    def __init__(self, config: ScalpBacktestConfig) -> None:
        self.cfg = config
        self._engine = None          # lazy Prometheus engine
        self._regime_clf = None
        self._session_clf = None

        # Running state (reset each run())
        self._equity:         float = config.initial_balance
        self._open:           List[ScalpTrade]   = []
        self._closed:         List[ScalpTrade]   = []
        self._pending_limits: List[PendingLimit] = []
        self._equity_curve:   List[float]        = [config.initial_balance]
        self._current_ltf_biases: List[str]      = []
        self._last_live_result = None
        self._last_atr:        float = 0.0
        self._score_adjust:    float = 0.0  # LML running adjustment

        # ATR history for atr_rank calculation
        self._atr_history: List[float] = []

        # Diagnostics counters
        self._invalid_trade_setup_count: int = 0
        self._regime_unavailable_count: int = 0
        self._skipped_by_regime_count: int = 0
        self._regime_dist: Dict[str, int] = {}  # regime label → bar count

        # ── Strategy lab state ────────────────────────────────────────────
        # Phase 1 entry gate trackers
        self._last_entry_bar:    int   = -999   # bar index of most recent entry
        self._last_entry_dir:    str   = ""     # "long" | "short" of last entry
        self._flip_candidate:    str   = ""     # pending direction flip candidate
        self._flip_bar:          int   = -999   # bar index when flip was first seen
        self._sess_dir_losses:   Dict[str, int] = {}  # session → consecutive losses
        self._last_session_lbl:  str   = ""     # session label at last closed trade

        # Phase 3 daily circuit breaker trackers
        self._day_start_equity:  float = config.initial_balance
        self._day_start_date:    str   = ""
        self._daily_halted:      bool  = False  # True → no new entries today
        self._daily_protecting:  bool  = False  # True → scale lot by daily_profit_lot_scalar

        # Phase 2 timeout partial guard (set of trade_ids already smart-partialled)
        self._smart_partial_done: set  = set()

    # ─────────────────────────────────────────────────────────────────────────
    # Engine initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_engine(self) -> None:
        if self._engine is not None:
            return
        try:
            from prometheus_core import Prometheus
            self._engine = Prometheus()
            logger.info("Prometheus engine initialised")
        except Exception as exc:
            logger.error("Could not init Prometheus engine: %s", exc)
            raise

    def _init_classifiers(self) -> None:
        try:
            from regime_classifier import RegimeClassifier
            self._regime_clf = RegimeClassifier()
        except Exception as exc:
            logger.warning("Could not init RegimeClassifier: %s", exc)
            self._regime_clf = None
        try:
            from session_classifier import SessionClassifier
            self._session_clf = SessionClassifier()
        except Exception as exc:
            logger.warning("Could not init SessionClassifier: %s", exc)
            self._session_clf = None

    # ─────────────────────────────────────────────────────────────────────────
    # Data fetching
    # ─────────────────────────────────────────────────────────────────────────

    def _mt5_tf_const(self, tf_str: str):
        """Return the MT5 TIMEFRAME constant for a tf string."""
        key = self._TF_MAP.get(tf_str.lower(), "M30")
        return getattr(mt5, f"TIMEFRAME_{key}", mt5.TIMEFRAME_M30)

    def _rates_to_df(self, rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        df = df.set_index("time")
        df.index.name = "datetime"
        return df[["open", "high", "low", "close", "volume"]].copy()

    def fetch_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Pull historical OHLCV bars from MT5.

        Returns:
            primary_df — DataFrame for config.primary_tf
            context_dfs — {tf_str: DataFrame} for context timeframes
        """
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not available. "
                "Pass data manually to run(primary_df, context_dfs)."
            )

        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

        primary_df: Optional[pd.DataFrame] = None
        context_dfs: Dict[str, pd.DataFrame] = {}

        cfg = self.cfg
        asset = cfg.asset

        def _fetch(tf_str: str, n: int = 1500) -> Optional[pd.DataFrame]:
            tf_const = self._mt5_tf_const(tf_str)
            if cfg.date_from and cfg.date_to:
                rates = mt5.copy_rates_range(
                    asset, tf_const,
                    cfg.date_from, cfg.date_to,
                )
            else:
                rates = mt5.copy_rates_from_pos(asset, tf_const, 0, n)
            if rates is None or len(rates) < 20:
                logger.warning("No data for %s %s: %s", asset, tf_str, mt5.last_error())
                return None
            return self._rates_to_df(rates)

        logger.info("Fetching %s data for %s...", asset, cfg.primary_tf)
        primary_df = _fetch(cfg.primary_tf, cfg.n_bars)
        if primary_df is None:
            raise RuntimeError(f"Could not fetch primary data for {asset} {cfg.primary_tf}")

        logger.info("Primary: %d bars on %s", len(primary_df), cfg.primary_tf)

        for tf in cfg.context_tfs:
            if tf.lower() == cfg.primary_tf.lower():
                continue
            df = _fetch(tf)
            if df is not None:
                context_dfs[tf.lower()] = df
                logger.info("Context %s: %d bars", tf, len(df))

        return primary_df, context_dfs

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tf_data(
        self,
        bar_ts,
        context_dfs: Dict[str, pd.DataFrame],
        primary_slice: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Build tf_data dict sliced to bar_ts — no lookahead."""
        tf_data: Dict[str, pd.DataFrame] = {
            self.cfg.primary_tf.lower(): primary_slice,
        }
        for tf, df in context_dfs.items():
            sliced = df[df.index <= bar_ts]
            if len(sliced) >= 20:
                tf_data[tf] = sliced
        return tf_data

    def _run_analysis(
        self,
        primary_slice: pd.DataFrame,
        tf_data: Dict[str, pd.DataFrame],
    ):
        """Run Prometheus pipeline on current bar slice. Returns PrometheusResult."""
        try:
            result = self._engine.analyze_data(
                primary_slice,
                asset=self.cfg.asset,
                timeframe=self.cfg.primary_tf,
                tf_data=tf_data,
                render_chart=False,
                save_to_db=False,
            )
            self._last_live_result = result
            return result
        except Exception as exc:
            logger.debug("Analysis error at bar %s: %s", len(primary_slice), exc)
            return None

    def _get_session(self, bar_ts):
        """Classify session for bar timestamp (UTC-aware)."""
        if self._session_clf is None:
            return "unknown"
        try:
            if hasattr(bar_ts, "to_pydatetime"):
                dt = bar_ts.to_pydatetime()
            else:
                dt = bar_ts
            from datetime import timezone as tz
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz.utc)
            state = self._session_clf.classify(dt)
            return state.session.value if state else "unknown"
        except Exception:
            return "unknown"

    def _skip_session(self, session_name: str) -> bool:
        """Return True if this session hard-blocks new entries."""
        _SKIP = {"ny_afternoon", "dead_zone"}
        return session_name in _SKIP

    def _get_regime(self, df_slice: pd.DataFrame):
        """Classify regime from price data slice using Prometheus regime classifier."""
        if self._regime_clf is None:
            return None
        try:
            return self._regime_clf.classify(df_slice)
        except Exception as exc:
            logger.debug("Regime classification failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Lot sizing
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_lot(self, sl_dist: float, equity: float) -> float:
        """
        2 % risk rule matching live bot _calc_lot().
        XAUUSDm tick: 0.01 price move = tick_value per 0.01 lot.
        Approximate: $1 move × lot = $1 PnL per standard lot.
        For XAUUSDm (100 oz): 1 lot = 100 × price move / tick.
        Simplified: assume tick_val=0.01, tick_sz=0.01 → 1 pt ≈ $1/lot.
        """
        if sl_dist <= 0:
            return 0.01
        risk_amount = equity * (self.cfg.risk_pct / 100.0)
        # XAUUSDm: 1 lot ≈ 100 × point_value; approximation: 1 pt = $1/lot × 100
        # lot = risk / (sl_dist * 100) for standard Exness micro gold
        # This matches the broker's tick_value / tick_size ratio for XAUUSDm
        lot = risk_amount / (sl_dist * 100.0)
        # Clamp to min 0.01, max 10.0, step 0.01
        lot = max(0.01, min(10.0, round(lot / 0.01) * 0.01))
        return round(lot, 2)

    def _validate_trade_geometry(
        self,
        direction: str,
        entry_price: float,
        sl: float,
        tp1: float,
        tp2: float,
    ) -> Tuple[bool, str]:
        """Validate basic long/short geometry before trade creation."""
        if direction == "long":
            if not (sl < entry_price < tp1 and sl < entry_price < tp2):
                return False, "invalid_long_geometry"
            return True, "ok"
        if direction == "short":
            if not (tp1 < entry_price < sl and tp2 < entry_price < sl):
                return False, "invalid_short_geometry"
            return True, "ok"
        return False, "invalid_direction"

    def _max_open(self, equity: float) -> int:
        cfg = self.cfg
        if 0 < equity < cfg.small_acct_threshold:
            return cfg.small_acct_max_open
        if 0 < equity < cfg.medium_acct_threshold:
            return cfg.medium_acct_max_open
        return cfg.normal_acct_max_open

    def _is_small_account(self, equity: float) -> bool:
        return 0 < equity < self.cfg.small_acct_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Signal qualification (mirrors _qualifies_result)
    # ─────────────────────────────────────────────────────────────────────────

    def _qualifies(
        self,
        result,
        equity: float,
        regime,
        session_name: str,
        current_ltf_biases: List[str],
    ) -> Tuple[bool, dict]:
        """
        Gate the Prometheus result through all live-bot filters.
        Returns (qualifies, meta_dict).
        meta_dict carries features for ML and report segmentation.
        """
        meta: dict = {
            "grade": "F", "score": 0.0, "ltf_state": "unknown",
            "zone_type": "no_zone", "zone_pos": "no_zone",
            "sl_atr": 0.0, "ob_direction": "none",
            "pattern_type": 0, "regime": "unknown",
            "session": session_name,
            "entry_type": "market",
        }

        if not result or not result.confluence:
            return False, meta

        c = result.confluence
        grade     = (c.grade or "F").upper()
        score     = c.total or 0.0
        direction = (c.direction or "sideways").lower()

        meta["grade"] = grade
        meta["score"] = score

        if direction == "sideways":
            return False, meta

        is_long  = direction == "bullish"
        sig_bias = "bullish" if is_long else "bearish"

        # Grade / score threshold
        effective_min = self.cfg.min_score + self._score_adjust
        if self._GRADE_RANK.get(grade, 0) < self._GRADE_RANK.get(self.cfg.min_grade, 3):
            return False, meta
        if score < effective_min:
            return False, meta

        qualifies = True

        # Session allow-list / hard-blocks
        enabled_sessions = {s.lower() for s in (self.cfg.enabled_sessions or [])}
        if enabled_sessions and session_name not in enabled_sessions:
            return False, meta
        if self._skip_session(session_name):
            return False, meta

        # Asian +5pt floor
        if session_name == "asian":
            if score < effective_min + 5.0:
                return False, meta

        # ── London/NY Overlap SR gate (data-driven) ───────────────────────
        # Backtest data: london_ny_overlap + SR zone = 0% WR, -$60.80 across
        # both runs.  SR entries at London/NY overlap are swept before direction
        # establishes.  Require OB zone (not SR) OR score >= threshold + 12 pts.
        if session_name == "london_ny_overlap":
            # Will be filled in after zone detection, so use a score premium now
            # as a conservative pre-filter; OB entries bypass in _execute_signal.
            if score < effective_min + 8.0:
                return False, meta

        # Regime kill / score premium
        # Backtester should mirror Prometheus regime behavior; if regime is unavailable,
        # do not open new entries rather than running with ungated behavior.
        if regime is None:
            self._regime_unavailable_count += 1
            meta["regime"] = "unknown"
            fallback_floor = effective_min + self.cfg.unknown_regime_score_premium
            if score < fallback_floor:
                self._skipped_by_regime_count += 1
                return False, meta
        if getattr(regime, "kill_entries", False):
            self._skipped_by_regime_count += 1
            return False, meta
        floor_prem = getattr(regime, "score_floor_premium", 0.0)
        if floor_prem > 0 and score < effective_min + floor_prem:
            self._skipped_by_regime_count += 1
            return False, meta

        meta["regime"] = getattr(getattr(regime, "regime", None), "value", "unknown")

        # HTF alignment (all TFs above primary must lean with signal)
        if result.mtf and result.mtf.biases:
            _primary_rank = self._TF_RANK.get(self.cfg.primary_tf.lower(), 4)
            _htf_biases   = [b for b in result.mtf.biases
                             if self._TF_RANK.get(b.timeframe.lower(), 0) > _primary_rank]
            _htf_aligned  = [b for b in _htf_biases if b.bias == sig_bias]
            if len(_htf_biases) >= 1 and len(_htf_aligned) < len(_htf_biases):
                # Simple non-probabilistic gate for backtest clarity
                _align_ratio = len(_htf_aligned) / len(_htf_biases)
                if _align_ratio <= 0.35:
                    return False, meta

        # ── LTF trap + Phase 1 gates ──────────────────────────────────────────
        if result.mtf and result.mtf.biases:
            _primary_rank = self._TF_RANK.get(self.cfg.primary_tf.lower(), 4)
            _ltf_biases = sorted(
                [b for b in result.mtf.biases
                 if self._TF_RANK.get(b.timeframe.lower(), 0) < _primary_rank],
                key=lambda b: self._TF_RANK.get(b.timeframe.lower(), 0),
                reverse=True,
            )[:2]
            _ltf_aligned = [b for b in _ltf_biases if b.bias == sig_bias]
            _ltf_counter = [b for b in _ltf_biases if b.bias != sig_bias]
            current_ltf_biases[:] = [b.bias for b in _ltf_biases]

            if len(_ltf_biases) >= 2:
                if len(_ltf_counter) == len(_ltf_biases):
                    # Full trap: all LTFs opposing → block
                    meta["ltf_state"] = "trap"
                    return False, meta
                elif len(_ltf_aligned) == len(_ltf_biases):
                    meta["ltf_state"] = "both_confirmed"
                    # Phase 1: relaxed gate — both_confirmed_min_score (80, down from 85)
                    if not (grade == "A" and score >= self.cfg.both_confirmed_min_score):
                        return False, meta
                else:
                    meta["ltf_state"] = "one_counter"

                    # ── Phase 1 Gate A: High-ATR rank block ───────────────────
                    # Losses cluster at atr_rank >= block_high_atr_rank (default 0.85).
                    # Extreme volatility with mixed LTF = high noise → need score >=95.
                    _atr_rank_now = (
                        len([h for h in self._atr_history if h <= self._last_atr])
                        / max(1, len(self._atr_history))
                    ) if self._atr_history else 0.5
                    if (self.cfg.block_high_atr_rank > 0
                            and _atr_rank_now >= self.cfg.block_high_atr_rank
                            and score < 95.0):
                        meta["ltf_state"] = "one_counter_high_atr"
                        return False, meta

                    # ── Phase 1 Gate B: SR zone score premium ─────────────────
                    # SR entries earn $0.08/trade avg vs $12.81 for OB.
                    # Require sr_min_score_premium extra score UNLESS a fresh OB exists.
                    _sr_premium = self.cfg.sr_min_score_premium
                    if _sr_premium > 0 and score < effective_min + _sr_premium:
                        _has_ob = (
                            result.smc and result.smc.order_blocks
                            and any(
                                not ob.mitigated and ob.direction == sig_bias
                                for ob in result.smc.order_blocks
                            )
                        )
                        if not _has_ob:
                            return False, meta
            else:
                meta["ltf_state"] = "unknown"

        # Countertrend gate (regime-based)
        if regime is not None:
            allow_ct = getattr(regime, "allow_countertrend", False)
            if result.mtf and result.mtf.biases:
                _htf_day = [b for b in result.mtf.biases
                            if b.timeframe.lower() in ("1d", "1w")
                            and b.bias not in ("unknown", "sideways")]
                if _htf_day:
                    _dominant = max(_htf_day, key=lambda b: abs(b.score)).bias
                    if _dominant != sig_bias:
                        if not allow_ct:
                            return False, meta
                        # CT allowed but needs extra score
                        _ct_required = effective_min + 15.0
                        if score < _ct_required:
                            return False, meta

        # Pattern type for ML
        if result.pat and result.pat.patterns:
            _best_pat = result.pat.patterns[0]
            from ml.pattern_learner import classify_pattern_type
            meta["pattern_type"] = classify_pattern_type(getattr(_best_pat, "pattern", ""))

        return qualifies, meta

    # ─────────────────────────────────────────────────────────────────────────
    # Zone detection (mirrors _execute_from_result zone logic)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_zone(
        self,
        result,
        is_long: bool,
        price: float,
        atr: float,
    ) -> Tuple[Optional[float], str, str]:
        """
        Identify the best entry zone.
        Returns (zone_price, zone_type, zone_pos).
        zone_type: "ob" | "sr" | "no_zone"
        zone_pos:  "deep" | "mid" | "shallow" | "no_zone"
        """
        smc = result.smc
        sr  = result.sr
        zone_threshold = (atr * self.cfg.zone_atr_thresh) if atr else price * 0.002
        zone_price: Optional[float] = None
        zone_type   = "no_zone"

        # --- Order Block search ---
        if smc and smc.order_blocks:
            ob_dir    = "bullish" if is_long else "bearish"
            fresh_obs = [ob for ob in smc.order_blocks
                         if ob.direction == ob_dir and not ob.mitigated]
            if fresh_obs:
                fresh_obs.sort(key=lambda b: (
                    abs((b.high if is_long else b.low) - price),
                    -b.strength,
                ))
                for _ob in fresh_obs:
                    _ref = _ob.high if is_long else _ob.low
                    if atr == 0 or abs(_ref - price) <= self.cfg.max_limit_dist_atr * atr:
                        zone_price = _ref
                        zone_type  = "ob"
                        break

        # --- S/R fallback ---
        if zone_price is None and sr:
            if is_long and sr.nearest_support:
                _sp = sr.nearest_support.level
                if atr == 0 or abs(_sp - price) <= self.cfg.max_limit_dist_atr * atr:
                    zone_price = _sp
                    zone_type  = "sr"
            elif not is_long and sr.nearest_resistance:
                _rp = sr.nearest_resistance.level
                if atr == 0 or abs(_rp - price) <= self.cfg.max_limit_dist_atr * atr:
                    zone_price = _rp
                    zone_type  = "sr"

        # --- Zone position quality ---
        zone_pos = "no_zone"
        if zone_price is not None and atr > 0:
            dist = abs(price - zone_price)
            if dist <= zone_threshold * 0.33:
                zone_pos = "deep"
            elif dist <= zone_threshold:
                zone_pos = "mid"
            else:
                zone_pos = "shallow"

        return zone_price, zone_type, zone_pos

    # ─────────────────────────────────────────────────────────────────────────
    # SL/TP calculation (mirrors live bot logic)
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_sl_tp(
        self,
        is_long: bool,
        price: float,
        sup: Optional[float],
        res: Optional[float],
        atr: float,
        tp_scalar: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """Returns (sl, tp1, tp2, sl_dist)."""
        buf  = atr * 0.15 if atr else price * 0.002
        min_sl_atr = 1.0

        if is_long:
            sl_cand = (float(sup) - buf) if (sup and sup < price) else price - atr * min_sl_atr
            sl = round(min(sl_cand, price - atr * min_sl_atr) if atr else sl_cand, 5)
        else:
            sl_cand = (float(res) + buf) if (res and res > price) else price + atr * min_sl_atr
            sl = round(max(sl_cand, price + atr * min_sl_atr) if atr else sl_cand, 5)

        sl_dist = abs(price - sl)
        if sl_dist <= 0:
            sl_dist = atr * min_sl_atr if atr else price * 0.003

        if is_long:
            tp1 = round(price + sl_dist * self.cfg.tp1_rr * tp_scalar, 5)
            tp2 = round(price + sl_dist * self.cfg.tp2_rr * tp_scalar, 5)
        else:
            tp1 = round(price - sl_dist * self.cfg.tp1_rr * tp_scalar, 5)
            tp2 = round(price - sl_dist * self.cfg.tp2_rr * tp_scalar, 5)

        return sl, tp1, tp2, sl_dist

    # ─────────────────────────────────────────────────────────────────────────
    # Limit order SL/TP (matches _place_limit_order)
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_limit_sl_tp(
        self,
        is_long: bool,
        limit_price: float,
        sup: Optional[float],
        res: Optional[float],
        atr: float,
        tp_scalar: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        buf = atr * 0.15 if atr else limit_price * 0.002
        min_sl_atr = 1.0
        fallback_sl_dist = max(atr * min_sl_atr, limit_price * 0.003)

        if is_long:
            sl_cand = (float(sup) - buf) if (sup and sup < limit_price) else limit_price - fallback_sl_dist
            sl = round(min(sl_cand, limit_price - atr * min_sl_atr) if atr else sl_cand, 5)
        else:
            sl_cand = (float(res) + buf) if (res and res > limit_price) else limit_price + fallback_sl_dist
            sl = round(max(sl_cand, limit_price + atr * min_sl_atr) if atr else sl_cand, 5)

        sl_dist = abs(limit_price - sl)
        if sl_dist <= 0:
            return sl, limit_price, limit_price, 0.0

        if is_long:
            tp1 = round(limit_price + sl_dist * self.cfg.tp1_rr * tp_scalar, 5)
            tp2 = round(limit_price + sl_dist * self.cfg.tp2_rr * tp_scalar, 5)
        else:
            tp1 = round(limit_price - sl_dist * self.cfg.tp1_rr * tp_scalar, 5)
            tp2 = round(limit_price - sl_dist * self.cfg.tp2_rr * tp_scalar, 5)

        return sl, tp1, tp2, sl_dist

    # ─────────────────────────────────────────────────────────────────────────
    # Commission / slippage
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, is_long: bool) -> float:
        pts = self.cfg.slippage_pts
        return price + pts if is_long else price - pts

    def _commission_cost(self, trade: ScalpTrade) -> float:
        """Round-trip commission."""
        return trade.entry_price * trade.size * self.cfg.commission_pct * 2

    def _pnl(self, trade: ScalpTrade, exit_price: float) -> float:
        """Approximate P&L for XAUUSDm: 1 pt × 100 × lot = $1."""
        if trade.direction == "long":
            return (exit_price - trade.entry_price) * trade.size * 100.0
        return (trade.entry_price - exit_price) * trade.size * 100.0

    # ─────────────────────────────────────────────────────────────────────────
    # Open a market position
    # ─────────────────────────────────────────────────────────────────────────

    def _open_position(
        self,
        bar_i:      int,
        bar_ts,
        direction:  str,   # "long" | "short"
        price:      float,
        sl:         float,
        tp1:        float,
        tp2:        float,
        lot:        float,
        entry_type: str,
        meta:       dict,
        atr:        float,
    ) -> Optional[ScalpTrade]:
        """Create a SimPosition from meta + execution params."""
        equity = self._equity
        is_long = direction == "long"

        # Slippage
        fill_price = self._apply_slippage(price, is_long)

        ok_geom, geom_reason = self._validate_trade_geometry(direction, fill_price, sl, tp1, tp2)
        if not ok_geom:
            self._invalid_trade_setup_count += 1
            logger.debug(
                "[validate] Skipped %s entry due to %s (entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f)",
                direction, geom_reason, fill_price, sl, tp1, tp2,
            )
            return None

        # Commission
        comm = fill_price * lot * self.cfg.commission_pct

        # Small-account SL gate (market entries only)
        if entry_type == "market" and self._is_small_account(equity):
            sl_dist = abs(fill_price - sl)
            if atr > 0 and sl_dist > atr * self.cfg.small_acct_max_sl_atr:
                logger.debug("[small_acct] SL too wide — skip entry")
                return None

        # Deduct commission entry side
        self._equity -= comm

        dt = bar_ts if hasattr(bar_ts, "timetuple") else pd.Timestamp(bar_ts)
        dt_py = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
        hour = dt_py.hour if dt_py else 0
        dow  = dt_py.weekday() if dt_py else 0

        atr_rank = 0.0
        if self._atr_history and atr > 0:
            below = sum(1 for h in self._atr_history if h <= atr)
            atr_rank = below / len(self._atr_history)

        trade = ScalpTrade(
            trade_id    = str(uuid.uuid4())[:8],
            direction   = direction,
            entry_type  = entry_type,
            entry_price = fill_price,
            sl_price    = sl,
            tp1_price   = tp1,
            tp2_price   = tp2,
            size        = lot,
            entry_bar   = bar_i,
            grade       = meta.get("grade", "F"),
            score       = meta.get("score", 0.0),
            ltf_state   = meta.get("ltf_state", "unknown"),
            zone_type   = meta.get("zone_type", "no_zone"),
            zone_pos    = meta.get("zone_pos", "no_zone"),
            sl_atr      = (abs(fill_price - sl) / atr) if atr > 0 else 0.0,
            ob_direction= meta.get("ob_direction", "none"),
            pattern_type= meta.get("pattern_type", 0),
            regime      = meta.get("regime", "unknown"),
            session     = meta.get("session", "unknown"),
            atr_rank    = atr_rank,
            hour_utc    = hour,
            dow         = dow,
        )
        self._open.append(trade)
        logger.debug(
            "[entry] %s %s @ %.4f SL=%.4f TP1=%.4f lot=%.2f grade=%s score=%.0f",
            entry_type, direction, fill_price, sl, tp1, lot,
            meta.get("grade"), meta.get("score"),
        )
        return trade

    # ─────────────────────────────────────────────────────────────────────────
    # Close a position
    # ─────────────────────────────────────────────────────────────────────────

    def _close_position(
        self,
        trade:      ScalpTrade,
        bar_i:      int,
        exit_price: float,
        reason:     str,
    ) -> None:
        """Finalise trade P&L and move to closed list."""
        pnl = self._pnl(trade, exit_price) - self._commission_cost(trade)
        risk = abs(trade.entry_price - trade.sl_price)
        reward = abs(exit_price - trade.entry_price)
        rr    = reward / risk if risk > 0 else 0.0

        trade.exit_bar   = bar_i
        trade.exit_price = exit_price
        trade.exit_reason= reason
        trade.bars_open  = bar_i - trade.entry_bar   # strategy lab ML feature
        if reason == "sl":
            if trade.direction == "long":
                if exit_price > trade.entry_price:
                    trade.stop_type = "trail_sl"
                elif abs(exit_price - trade.entry_price) <= 1e-9:
                    trade.stop_type = "be_sl"
                else:
                    trade.stop_type = "initial_sl"
            else:
                if exit_price < trade.entry_price:
                    trade.stop_type = "trail_sl"
                elif abs(exit_price - trade.entry_price) <= 1e-9:
                    trade.stop_type = "be_sl"
                else:
                    trade.stop_type = "initial_sl"
        else:
            trade.stop_type = "none"
        trade.pnl        = round(pnl, 2)
        trade.rr         = round(rr, 2)
        trade.status     = "won" if pnl > 0 else "lost"

        self._equity += pnl
        self._open.remove(trade)
        self._closed.append(trade)

        # Running LML score_adjust (mirrors live bot)
        all_closed = len(self._closed)
        if all_closed >= 3:
            last20 = [1 if t.status == "won" else 0 for t in self._closed[-20:]]
            wr = sum(last20) / len(last20)
            if wr < 0.35:
                self._score_adjust = min(15.0, (0.35 - wr) * 60)
            elif wr < 0.50:
                self._score_adjust = min(10.0, (0.50 - wr) * 40)
            elif wr > 0.70:
                self._score_adjust = max(-8.0, (0.70 - wr) * 30)
            elif wr > 0.55:
                self._score_adjust = max(-4.0, (0.55 - wr) * 20)
            else:
                self._score_adjust = 0.0

        logger.debug(
            "[close] %s %s @ %.4f pnl=%.2f rr=%.2f status=%s reason=%s",
            trade.direction, trade.trade_id, exit_price,
            pnl, rr, trade.status, reason,
        )

        # ── Strategy lab bookkeeping ──────────────────────────────────────
        # Session consecutive-loss tracking (Phase 1 gate)
        if self.cfg.session_dir_loss_halt > 0:
            _sess = trade.session
            if trade.status == "lost":
                self._sess_dir_losses[_sess] = self._sess_dir_losses.get(_sess, 0) + 1
            else:
                # Win resets the streak for that session
                self._sess_dir_losses[_sess] = 0

        # Attach session loss streak at close time (ML feature)
        trade.session_loss_streak = self._sess_dir_losses.get(trade.session, 0)

        # Clean up smart partial guard so it doesn't linger
        self._smart_partial_done.discard(trade.trade_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Check SL/TP per bar
    # ─────────────────────────────────────────────────────────────────────────

    def _check_exits(
        self,
        bar_i:    int,
        bar_high: float,
        bar_low:  float,
        bar_close:float,
        bar_open: float = 0.0,   # needed for M5 severity body ratio
        bar_ts    = None,
        atr:      float = 0.0,
    ) -> None:
        """Check open positions for SL/TP hits, manage trailing SL,
        and apply Phase 2 timeout / M5-severity exits when enabled."""
        for trade in list(self._open):
            is_long = trade.direction == "long"
            entry   = trade.entry_price
            sl      = trade.sl_price
            cur_tp  = trade.tp1_price    # active target (tp1 until hit, then tp2)

            # ── TP1 → upgrade to TP2 (partial-close simulation) ──
            if trade.tp1_price != trade.tp2_price:
                if is_long and bar_high >= trade.tp1_price:
                    if trade.tp1_price != trade.tp2_price:
                        # "close" half at TP1 — adjust target to TP2
                        if self._is_small_account(self._equity):
                            # single-leg: close whole position at TP1
                            self._close_position(trade, bar_i, trade.tp1_price, "tp1")
                            continue
                        # dual-leg: first leg hits TP1, second continues to TP2
                        trade.tp1_price = trade.tp2_price   # sentinel: already hit
                        pnl_partial = self._pnl(trade, trade.tp1_price) * 0.5 - self._commission_cost(trade) * 0.5
                        self._equity   += pnl_partial
                        trade.size     *= 0.5  # half remaining
                        cur_tp          = trade.tp2_price
                elif not is_long and bar_low <= trade.tp1_price:
                    if trade.tp1_price != trade.tp2_price:
                        if self._is_small_account(self._equity):
                            self._close_position(trade, bar_i, trade.tp1_price, "tp1")
                            continue
                        trade.tp1_price = trade.tp2_price
                        pnl_partial = self._pnl(trade, trade.tp1_price) * 0.5 - self._commission_cost(trade) * 0.5
                        self._equity   += pnl_partial
                        trade.size     *= 0.5
                        cur_tp          = trade.tp2_price

            # ── SL hit ────────────────────────────────────────────────────
            if is_long and bar_low <= sl:
                self._close_position(trade, bar_i, sl, "sl")
                continue
            if not is_long and bar_high >= sl:
                self._close_position(trade, bar_i, sl, "sl")
                continue

            # ── TP2 hit ───────────────────────────────────────────────────
            if is_long and bar_high >= trade.tp2_price:
                self._close_position(trade, bar_i, trade.tp2_price, "tp2")
                continue
            if not is_long and bar_low <= trade.tp2_price:
                self._close_position(trade, bar_i, trade.tp2_price, "tp2")
                continue

            # ── Trailing SL ratchet ────────────────────────────────────────
            if atr > 0:
                trail_dist = atr * self.cfg.trail_atr_mult
                be_trigger = atr * self.cfg.be_atr_trigger

                if is_long:
                    profit_pts = bar_close - entry
                    if profit_pts >= be_trigger:
                        if sl < entry:
                            trade.sl_price = round(entry + self.cfg.be_profit_pts, 5)
                        else:
                            new_sl = round(bar_close - trail_dist, 5)
                            if new_sl > trade.sl_price:
                                trade.sl_price = new_sl
                else:
                    profit_pts = entry - bar_close
                    if profit_pts >= be_trigger:
                        if sl > entry:
                            trade.sl_price = round(entry - self.cfg.be_profit_pts, 5)
                        else:
                            new_sl = round(bar_close + trail_dist, 5)
                            if new_sl < trade.sl_price:
                                trade.sl_price = new_sl

            # ── Phase 2: Timeout exits ────────────────────────────────────
            if self.cfg.time_exit_enable:
                _bars_open = bar_i - trade.entry_bar

                # Smart partial (one-time, only if profit meets floor)
                _smart_threshold = self.cfg.time_exit_smart_bars
                if (
                    _bars_open >= _smart_threshold
                    and trade.trade_id not in self._smart_partial_done
                    and not self._is_small_account(self._equity)
                ):
                    _profit_usd = self._pnl(trade, bar_close)
                    if _profit_usd >= self.cfg.time_exit_profit_min:
                        _partial_pnl = (
                            _profit_usd * 0.5
                            - self._commission_cost(trade) * 0.5
                        )
                        self._equity += _partial_pnl
                        trade.size   = max(0.01, round(trade.size * 0.5 / 0.01) * 0.01)
                        trade.exit_timeout = True
                        self._smart_partial_done.add(trade.trade_id)
                        logger.debug(
                            "[timeout] smart-partial %s bars_open=%d pnl=%.2f",
                            trade.trade_id, _bars_open, _partial_pnl,
                        )

                # Hard full close
                if _bars_open >= self.cfg.time_exit_hard_bars:
                    trade.exit_timeout = True
                    self._close_position(trade, bar_i, bar_close, "timeout")
                    continue

            # ── Phase 2: M5-style severity exit ──────────────────────────
            if self.cfg.m5_severity_enable:
                _bar_range = bar_high - bar_low
                if _bar_range > 0 and bar_open > 0:
                    _body_ratio = abs(bar_close - bar_open) / _bar_range
                    _is_opposing = (
                        (is_long and bar_close < bar_open)
                        or (not is_long and bar_close > bar_open)
                    )
                    _profit_pts = (
                        (bar_close - entry) if is_long else (entry - bar_close)
                    )

                    if _is_opposing and _profit_pts > 0:
                        if _body_ratio >= 0.70:
                            # Strong opposing candle → full close
                            self._close_position(
                                trade, bar_i, bar_close, "5m_severity_strong"
                            )
                            continue
                        elif _body_ratio >= 0.40 and not self._is_small_account(self._equity):
                            # Moderate opposing candle → 30% partial
                            _mod_pnl = (
                                self._pnl(trade, bar_close) * 0.30
                                - self._commission_cost(trade) * 0.30
                            )
                            self._equity += _mod_pnl
                            trade.size = max(
                                0.01,
                                round(trade.size * 0.70 / 0.01) * 0.01,
                            )
                        elif _body_ratio < 0.40 and atr > 0:
                            # Weak opposing candle → tighten SL to 0.5 × ATR
                            if is_long:
                                _tight_sl = round(bar_close - atr * 0.5, 5)
                                if _tight_sl > trade.sl_price:
                                    trade.sl_price = _tight_sl
                            else:
                                _tight_sl = round(bar_close + atr * 0.5, 5)
                                if _tight_sl < trade.sl_price:
                                    trade.sl_price = _tight_sl

    # ─────────────────────────────────────────────────────────────────────────
    # 5M-style exit: 3+ opposing bars when in profit (LTF exit logic)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_ltf_exit(
        self,
        bar_i:         int,
        recent_closes: List[float],  # last N bar closes
        bar_close:     float,
    ) -> None:
        """
        If the last M5_REVERSAL_CANDLES bars all oppose an open position's
        direction AND profit >= 0.4 R → close early (mirrors _check_5m_exits).
        Works on whatever the primary TF is (30M/15M/5M).
        """
        n = self.cfg.m5_reversal_candles
        if len(recent_closes) < n + 1:
            return

        # Derive last n bar directions from closes
        bar_dirs = []
        for j in range(-n, 0):
            c_prev = recent_closes[j - 1]
            c_curr = recent_closes[j]
            bar_dirs.append("bull" if c_curr > c_prev else "bear")

        for trade in list(self._open):
            is_long    = trade.direction == "long"
            sl_dist    = abs(trade.entry_price - trade.sl_price)
            if sl_dist == 0:
                continue

            all_opp = (
                (is_long     and all(d == "bear" for d in bar_dirs)) or
                (not is_long and all(d == "bull" for d in bar_dirs))
            )
            if not all_opp:
                continue

            profit_pts = (
                (bar_close - trade.entry_price) if is_long
                else (trade.entry_price - bar_close)
            )
            if profit_pts < sl_dist * self.cfg.m5_min_profit_r:
                continue

            self._close_position(trade, bar_i, bar_close, "5m_exit")

    # ─────────────────────────────────────────────────────────────────────────
    # Pending limit management
    # ─────────────────────────────────────────────────────────────────────────

    def _check_limit_fills(
        self,
        bar_i:    int,
        bar_high: float,
        bar_low:  float,
        bar_ts,
        atr:      float,
    ) -> None:
        """Check if any pending limit should fill on this bar."""
        for lim in list(self._pending_limits):
            is_long = lim.direction == "long"

            # Fill condition: bar crosses the limit price
            fills = (is_long and bar_low <= lim.limit_price) or \
                    (not is_long and bar_high >= lim.limit_price)

            if fills:
                # Check slot availability
                if len(self._open) >= self._max_open(self._equity):
                    self._pending_limits.remove(lim)
                    continue

                # Open the filled limit as a position
                self._open_position(
                    bar_i      = bar_i,
                    bar_ts     = bar_ts,
                    direction  = lim.direction,
                    price      = lim.limit_price,
                    sl         = lim.sl_price,
                    tp1        = lim.tp1_price,
                    tp2        = lim.tp2_price,
                    lot        = lim.size,
                    entry_type = "limit",
                    meta       = lim.meta,
                    atr        = atr,
                )
                self._pending_limits.remove(lim)
                continue

            # LTF trap monitor: if both LTFs now oppose this limit → cancel
            if len(self._current_ltf_biases) >= 2:
                opp = "bearish" if is_long else "bullish"
                if all(b == opp for b in self._current_ltf_biases):
                    logger.debug("[limit] Cancelled #%s — LTF trap", lim.order_id)
                    self._pending_limits.remove(lim)
                    continue

            # Expiry countdown
            lim.polls_left -= 1
            if lim.polls_left <= 0:
                logger.debug("[limit] Expired #%s", lim.order_id)
                self._pending_limits.remove(lim)

    # ─────────────────────────────────────────────────────────────────────────
    # Place a pending limit
    # ─────────────────────────────────────────────────────────────────────────

    def _place_limit(
        self,
        bar_i:       int,
        direction:   str,
        zone_price:  float,
        sl:          float,
        tp1:         float,
        tp2:         float,
        lot:         float,
        meta:        dict,
    ) -> None:
        """Queue a pending limit order (mirrors _place_limit_order)."""
        # LTF trap gate at placement
        is_long = direction == "long"
        opp = "bearish" if is_long else "bullish"
        if (len(self._current_ltf_biases) >= 2
                and all(b == opp for b in self._current_ltf_biases)):
            logger.debug("[limit] Skipped — LTF trap at placement (both LTFs %s)", opp)
            return

        ok_geom, geom_reason = self._validate_trade_geometry(
            direction, zone_price, sl, tp1, tp2
        )
        if not ok_geom:
            self._invalid_trade_setup_count += 1
            logger.debug(
                "[validate] Skipped limit due to %s (entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f)",
                geom_reason, zone_price, sl, tp1, tp2,
            )
            return

        meta_copy = dict(meta)
        meta_copy["entry_type"] = "limit"

        lim = PendingLimit(
            order_id    = str(uuid.uuid4())[:8],
            direction   = direction,
            limit_price = zone_price,
            sl_price    = sl,
            tp1_price   = tp1,
            tp2_price   = tp2,
            size        = lot,
            placed_bar  = bar_i,
            polls_left  = self.cfg.limit_order_expiry,
            meta        = meta_copy,
        )
        self._pending_limits.append(lim)
        logger.debug(
            "[limit] Placed %s @ %.4f SL=%.4f TP1=%.4f lot=%.2f",
            direction, zone_price, sl, tp1, lot,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Execute signal (market or limit — mirrors _execute_from_result)
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_signal(
        self,
        result,
        bar_i:     int,
        bar_ts,
        bar_close: float,
        meta:      dict,
        atr:       float,
        regime,
    ) -> bool:
        """Decide market vs limit entry and create position/limit."""
        confluence = result.confluence
        direction  = (confluence.direction if confluence else "sideways").lower()
        is_long    = direction == "bullish"
        is_short   = direction == "bearish"
        if not is_long and not is_short:
            return False

        direction_str = "long" if is_long else "short"

        # S/R levels
        sup = (result.sr.nearest_support.level
               if result.sr and result.sr.nearest_support else None)
        res_lv = (result.sr.nearest_resistance.level
                  if result.sr and result.sr.nearest_resistance else None)

        # Zone
        zone_price, zone_type, zone_pos = self._find_zone(
            result, is_long, bar_close, atr
        )

        meta["zone_type"] = zone_type
        meta["zone_pos"]  = zone_pos
        if result.smc and result.smc.order_blocks:
            ob_dir = "bullish" if is_long else "bearish"
            fresh  = [ob for ob in result.smc.order_blocks
                      if ob.direction == ob_dir and not ob.mitigated]
            meta["ob_direction"] = ob_dir if fresh else "none"

        tp_scalar = getattr(regime, "tp_scalar", 1.0) if regime else 1.0

        # Zone proximity check — use bar_close as price proxy
        zone_threshold = atr * self.cfg.zone_atr_thresh if atr else bar_close * 0.002
        in_zone = False

        if zone_price is not None:
            if is_long:
                in_zone = bar_close <= zone_price + zone_threshold
            else:
                in_zone = bar_close >= zone_price - zone_threshold

        if zone_price is None:
            in_zone = True  # no zone context → allow market

        # ── zone_only mode: price NOT in zone → place limit ───────────────
        if self.cfg.entry_mode == "zone_only" and not in_zone and zone_price is not None:
            sl, tp1, tp2, sl_dist = self._calc_limit_sl_tp(
                is_long, zone_price, sup, res_lv, atr, tp_scalar
            )
            if sl_dist <= 0:
                return False

            lot = self._calc_lot(sl_dist, self._equity)
            if self._is_small_account(self._equity):
                # Small account: single-leg full lot
                self._place_limit(
                    bar_i, direction_str, zone_price, sl, tp1, tp1, lot, meta
                )
            else:
                # Normal: dual-leg half lots
                half = max(0.01, round(lot / 2 / 0.01) * 0.01)
                self._place_limit(
                    bar_i, direction_str, zone_price, sl, tp1, tp2, half, meta
                )
            return True

        # ── Market entry (in zone or market_any mode) ─────────────────────
        sl, tp1, tp2, sl_dist = self._calc_sl_tp(
            is_long, bar_close, sup, res_lv, atr, tp_scalar
        )
        if sl_dist <= 0:
            return False

        # Min RR check
        rr = abs(tp1 - bar_close) / sl_dist
        if rr < 2.0:
            return False

        lot = self._calc_lot(sl_dist, self._equity)

        if self._is_small_account(self._equity):
            # Small account: single-leg, TP1 only
            t = self._open_position(
                bar_i, bar_ts, direction_str,
                bar_close, sl, tp1, tp1, lot, "market", meta, atr
            )
        else:
            # Normal: two half-lot legs
            half = max(0.01, round(lot / 2 / 0.01) * 0.01)
            t1 = self._open_position(
                bar_i, bar_ts, direction_str,
                bar_close, sl, tp1, tp2, half, "market", meta, atr
            )

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Execute a StrategySignal (custom strategy path)
    # Reuses all existing helpers so custom strategies get full execution
    # logic (lot-sizing, slippage, BE/trail SL, dual-leg) for free.
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_from_strategy_signal(
        self,
        sig:       "StrategySignal",
        bar_i:     int,
        bar_ts,
        bar_close: float,
        meta:      dict,
        atr:       float,
        regime,
    ) -> bool:
        """
        Convert a StrategySignal into an open position or pending limit.

        Priority
        --------
        1. If sig.sl is explicitly provided → use it directly (no zone detection).
        2. If sig.entry_price is provided → limit order with calc_limit_sl_tp
           (or sig.sl if also provided).
        3. Otherwise → market entry with _calc_sl_tp (mirrors market_any path).

        Returns True if an entry was placed.
        """
        is_long      = sig.direction == "long"
        direction_str = sig.direction  # "long" or "short"
        tp_scalar = getattr(regime, "tp_scalar", 1.0) if regime else 1.0

        # ── Limit order path (sig.entry_price supplied) ───────────────────
        if sig.entry_price is not None:
            limit_price = sig.entry_price
            if sig.sl is not None:
                sl      = sig.sl
                sl_dist = abs(limit_price - sl)
                if sl_dist <= 0:
                    return False
                if sig.tp1 is not None:
                    tp1 = sig.tp1
                else:
                    tp1 = (limit_price + sl_dist * self.cfg.tp1_rr * tp_scalar
                           if is_long
                           else limit_price - sl_dist * self.cfg.tp1_rr * tp_scalar)
                if sig.tp2 is not None:
                    tp2 = sig.tp2
                else:
                    tp2 = (limit_price + sl_dist * self.cfg.tp2_rr * tp_scalar
                           if is_long
                           else limit_price - sl_dist * self.cfg.tp2_rr * tp_scalar)
            else:
                sl, tp1, tp2, sl_dist = self._calc_limit_sl_tp(
                    is_long, limit_price, None, None, atr, tp_scalar
                )
                if sig.tp1 is not None:
                    tp1 = sig.tp1
                if sig.tp2 is not None:
                    tp2 = sig.tp2
                if sl_dist <= 0:
                    return False

            lot = self._calc_lot(sl_dist, self._equity)
            # Apply daily profit lot scaling
            if self._daily_protecting and self.cfg.daily_profit_lot_scalar > 0:
                lot = max(0.01, round(lot * self.cfg.daily_profit_lot_scalar / 0.01) * 0.01)

            if self._is_small_account(self._equity):
                self._place_limit(bar_i, direction_str, limit_price, sl, tp1, tp1, lot, meta)
            else:
                half = max(0.01, round(lot / 2 / 0.01) * 0.01)
                self._place_limit(bar_i, direction_str, limit_price, sl, tp1, tp2, half, meta)
            return True

        # ── Market order path ─────────────────────────────────────────────
        if sig.sl is not None:
            sl      = sig.sl
            sl_dist = abs(bar_close - sl)
            if sl_dist <= 0:
                return False
            if sig.tp1 is not None:
                tp1 = sig.tp1
            else:
                tp1 = (bar_close + sl_dist * self.cfg.tp1_rr * tp_scalar
                       if is_long
                       else bar_close - sl_dist * self.cfg.tp1_rr * tp_scalar)
            if sig.tp2 is not None:
                tp2 = sig.tp2
            else:
                tp2 = (bar_close + sl_dist * self.cfg.tp2_rr * tp_scalar
                       if is_long
                       else bar_close - sl_dist * self.cfg.tp2_rr * tp_scalar)
        else:
            sl, tp1, tp2, sl_dist = self._calc_sl_tp(
                is_long, bar_close, None, None, atr, tp_scalar
            )
            if sig.tp1 is not None:
                tp1 = sig.tp1
            if sig.tp2 is not None:
                tp2 = sig.tp2
            if sl_dist <= 0:
                return False

        # Min RR sanity check
        if abs(tp1 - bar_close) / sl_dist < 1.0:
            return False

        lot = self._calc_lot(sl_dist, self._equity)
        # Apply daily profit lot scaling
        if self._daily_protecting and self.cfg.daily_profit_lot_scalar > 0:
            lot = max(0.01, round(lot * self.cfg.daily_profit_lot_scalar / 0.01) * 0.01)

        if self._is_small_account(self._equity):
            self._open_position(
                bar_i, bar_ts, direction_str,
                bar_close, sl, tp1, tp1, lot, "market", meta, atr
            )
        else:
            half = max(0.01, round(lot / 2 / 0.01) * 0.01)
            self._open_position(
                bar_i, bar_ts, direction_str,
                bar_close, sl, tp1, tp2, half, "market", meta, atr
            )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Main run() — the walk-forward loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        primary_df:   Optional[pd.DataFrame] = None,
        context_dfs:  Optional[Dict[str, pd.DataFrame]] = None,
        progress_cb = None,   # Optional[Callable[[int, int], None]] — (current, total)
    ) -> "ScalpBacktestResult":
        """
        Execute the walk-forward simulation.

        If primary_df / context_dfs are None, data is fetched from MT5.
        """
        # Reset state
        self._equity         = self.cfg.initial_balance
        self._open           = []
        self._closed         = []
        self._pending_limits = []
        self._equity_curve   = [self.cfg.initial_balance]
        self._current_ltf_biases = []
        self._last_live_result   = None
        self._last_atr           = 0.0
        self._score_adjust       = 0.0
        self._atr_history        = []
        self._invalid_trade_setup_count = 0
        self._regime_unavailable_count = 0
        self._skipped_by_regime_count = 0
        self._regime_dist = {}

        # Strategy lab state reset
        self._last_entry_bar     = -999
        self._last_entry_dir     = ""
        self._flip_candidate     = ""
        self._flip_bar           = -999
        self._sess_dir_losses    = {}
        self._last_session_lbl   = ""
        self._day_start_equity   = self.cfg.initial_balance
        self._day_start_date     = ""
        self._daily_halted       = False
        self._daily_protecting   = False
        self._smart_partial_done = set()

        # Init engine
        self._init_engine()
        self._init_classifiers()

        # Fetch data if not supplied
        if primary_df is None:
            logger.info("Fetching data from MT5...")
            primary_df, context_dfs = self.fetch_data()

        context_dfs = context_dfs or {}

        primary_df = primary_df.copy()
        primary_df.columns = [c.lower() for c in primary_df.columns]
        total_bars = len(primary_df)
        warmup     = self.cfg.warmup_bars
        stride     = self.cfg.signal_stride

        logger.info(
            "Starting backtest: %s %s | strategy=%s | enabled_sessions=%s | %d bars | balance=$%.2f",
            self.cfg.asset,
            self.cfg.primary_tf,
            self.cfg.strategy_name,
            ",".join(self.cfg.enabled_sessions),
            total_bars,
            self.cfg.initial_balance,
        )

        # Pre-compute ATR-14 for the whole series
        atr_series = (
            (primary_df["high"] - primary_df["low"])
            .rolling(14).mean()
            .fillna((primary_df["high"] - primary_df["low"]).mean())
        )
        close_list: List[float] = []  # running list for LTF exit check

        last_result    = None
        last_meta:     dict = {}
        last_qualified = False
        last_regime    = None

        for i in range(warmup, total_bars):
            bar_open  = float(primary_df["open"].iloc[i])
            bar_high  = float(primary_df["high"].iloc[i])
            bar_low   = float(primary_df["low"].iloc[i])
            bar_close = float(primary_df["close"].iloc[i])
            bar_ts    = primary_df.index[i]
            bar_atr   = float(atr_series.iloc[i])

            # ATR rank history
            self._atr_history.append(bar_atr)
            if len(self._atr_history) > 252:
                self._atr_history.pop(0)
            self._last_atr = bar_atr

            close_list.append(bar_close)
            if len(close_list) > 20:
                close_list.pop(0)

            # Progress log
            if self.cfg.verbose and (i - warmup) % max(1, (total_bars - warmup) // 20) == 0:
                pct = (i - warmup) / max(1, total_bars - warmup) * 100
                logger.info(
                    "[%.0f%%] bar %d/%d equity=$%.2f open=%d pending=%d closed=%d",
                    pct, i, total_bars, self._equity,
                    len(self._open), len(self._pending_limits), len(self._closed),
                )

            # ── 0. Daily circuit breakers ─────────────────────────────────
            _bar_date = str(bar_ts)[:10]
            if _bar_date != self._day_start_date:
                # New trading day
                self._day_start_date   = _bar_date
                self._day_start_equity = self._equity
                self._daily_halted     = False
                self._daily_protecting = False

            if self.cfg.max_daily_loss_pct > 0:
                _daily_loss_pct = (
                    (self._day_start_equity - self._equity)
                    / max(1e-8, self._day_start_equity) * 100.0
                )
                if _daily_loss_pct >= self.cfg.max_daily_loss_pct:
                    self._daily_halted = True

            if self.cfg.daily_profit_protect_pct > 0:
                _daily_gain_pct = (
                    (self._equity - self._day_start_equity)
                    / max(1e-8, self._day_start_equity) * 100.0
                )
                self._daily_protecting = _daily_gain_pct >= self.cfg.daily_profit_protect_pct

            # ── 1. LTF exit (reversal of open profit) ─────────────────────
            self._check_ltf_exit(i, close_list, bar_close)

            # ── 2. Check SL/TP on open positions ──────────────────────────
            self._check_exits(i, bar_high, bar_low, bar_close, bar_open, bar_ts, bar_atr)

            # ── 3. Check pending limit fills / expiry ─────────────────────
            self._check_limit_fills(i, bar_high, bar_low, bar_ts, bar_atr)

            # ── 4. New signal evaluation (every stride bars) ───────────────
            at_stride = (i - warmup) % stride == 0
            if at_stride and len(self._open) < self._max_open(self._equity):
                df_slice = primary_df.iloc[: i + 1]
                tf_data  = self._build_tf_data(bar_ts, context_dfs, df_slice)

                # Classify regime and session (needed by both paths)
                regime     = self._get_regime(df_slice)
                session_nm = self._get_session(bar_ts)
                last_regime = regime

                # Regime distribution diagnostic
                _r_label = (
                    getattr(getattr(regime, "regime", None), "value", "unavailable")
                    if regime is not None else "unavailable"
                )
                self._regime_dist[_r_label] = self._regime_dist.get(_r_label, 0) + 1

                # Update current_ltf_biases before qualification
                self._current_ltf_biases = []

                # ── Signal source: custom strategy_fn or Prometheus ────────
                _custom_sig: Optional["StrategySignal"] = None
                result      = None
                qualifies   = False
                meta: dict  = {}

                if self.cfg.strategy_fn is not None:
                    # Custom strategy path — bypasses _run_analysis + _qualifies
                    try:
                        _custom_sig = self.cfg.strategy_fn(
                            df_slice, tf_data, bar_atr, regime, session_nm
                        )
                    except Exception as _e:
                        logger.warning("[strategy_fn] raised %s — skipping bar", _e)
                        _custom_sig = None

                    if _custom_sig is not None and _custom_sig.is_actionable:
                        qualifies = True
                        # Optional score/grade gate (mirrors Prometheus min_score/min_grade)
                        if self.cfg.strategy_fn_apply_score_gate:
                            _eff_min = self.cfg.min_score + self._score_adjust
                            if (
                                self._GRADE_RANK.get(_custom_sig.grade, 0)
                                < self._GRADE_RANK.get(self.cfg.min_grade, 3)
                                or _custom_sig.score < _eff_min
                            ):
                                qualifies = False
                        if qualifies:
                            meta = {
                                "grade":       _custom_sig.grade,
                                "score":       _custom_sig.score,
                                "ltf_state":   "unknown",
                                "zone_type":   "no_zone",
                                "zone_pos":    "no_zone",
                                "sl_atr":      0.0,
                                "ob_direction":"none",
                                "pattern_type":0,
                                "regime":      _r_label,
                                "session":     session_nm,
                                "entry_type":  "limit" if _custom_sig.entry_price is not None else "market",
                                **{k: v for k, v in _custom_sig.meta.items()
                                   if isinstance(v, (int, float, bool))},
                            }
                else:
                    # Prometheus default path — completely unchanged
                    result = self._run_analysis(df_slice, tf_data)
                    if result is not None:
                        qualifies, meta = self._qualifies(
                            result, self._equity, regime, session_nm,
                            self._current_ltf_biases,
                        )

                # ── Phase 1 entry discipline gates (both paths) ───────────
                if qualifies:
                    if self._daily_halted:
                        qualifies = False
                        logger.debug("[gate] daily_halt active — skipping entry")

                if qualifies and self.cfg.entry_cooldown_bars > 0:
                    _bars_since_last = i - self._last_entry_bar
                    if _bars_since_last < self.cfg.entry_cooldown_bars:
                        qualifies = False
                        logger.debug(
                            "[gate] cooldown: %d/%d bars since last entry",
                            _bars_since_last, self.cfg.entry_cooldown_bars,
                        )

                if qualifies and self.cfg.direction_flip_min_bars > 0:
                    _new_dir = meta.get("direction") or (
                        _custom_sig.direction if _custom_sig else
                        ((meta.get("grade", "F") and
                          (result.confluence.direction if result and result.confluence else "sideways"))
                         or "sideways")
                    )
                    # Normalise to long/short
                    if _new_dir in ("bullish",):
                        _new_dir = "long"
                    elif _new_dir in ("bearish",):
                        _new_dir = "short"
                    if (
                        self._last_entry_dir
                        and _new_dir != self._last_entry_dir
                        and (i - self._last_entry_bar) < self.cfg.direction_flip_min_bars
                    ):
                        qualifies = False
                        logger.debug(
                            "[gate] direction_flip: too soon to flip %s→%s",
                            self._last_entry_dir, _new_dir,
                        )

                if qualifies and self.cfg.session_dir_loss_halt > 0:
                    _sess_losses = self._sess_dir_losses.get(session_nm, 0)
                    if _sess_losses >= self.cfg.session_dir_loss_halt:
                        qualifies = False
                        logger.debug(
                            "[gate] session_loss_halt: %d losses in %s",
                            _sess_losses, session_nm,
                        )

                # ── Execute ────────────────────────────────────────────────
                if qualifies:
                    _n_before = len(self._open) + len(self._closed)
                    if _custom_sig is not None:
                        _entered = self._execute_from_strategy_signal(
                            _custom_sig, i, bar_ts, bar_close, meta, bar_atr, regime
                        )
                    else:
                        _entered = self._execute_signal(
                            result, i, bar_ts, bar_close, meta, bar_atr, regime
                        )
                    _n_after = len(self._open) + len(self._closed)

                    if _entered and _n_after > _n_before:
                        # Record entry for gate tracking
                        _dir_entered = meta.get("direction") or (
                            _custom_sig.direction if _custom_sig else
                            (result.confluence.direction if result and result.confluence else "")
                        )
                        if _dir_entered in ("bullish",):
                            _dir_entered = "long"
                        elif _dir_entered in ("bearish",):
                            _dir_entered = "short"
                        self._last_entry_bar = i
                        if _dir_entered in ("long", "short"):
                            self._last_entry_dir = _dir_entered

                last_result    = result
                last_meta      = meta
                last_qualified = qualifies

            # ── 5. Equity snapshot ────────────────────────────────────────
            self._equity_curve.append(round(self._equity, 2))

            # ── 6. Progress callback ──────────────────────────────────────
            if progress_cb:
                _done = i - warmup
                _total = total_bars - warmup
                if _done % max(1, _total // 100) == 0:
                    progress_cb(_done, _total)

        # ── Close all remaining open positions at last close ──────────────
        if primary_df is not None and len(primary_df) > 0:
            _last_close = float(primary_df["close"].iloc[-1])
            for trade in list(self._open):
                self._close_position(trade, total_bars - 1, _last_close, "eod")

        logger.info(
            "Backtest complete: %d trades | equity=$%.2f",
            len(self._closed), self._equity,
        )

        # ── Compute metrics and build result ──────────────────────────────
        result_obj = self._compute_result()

        # ── ML training ───────────────────────────────────────────────────
        if self.cfg.train_ml and len(self._closed) >= 5:
            fi, acc, auc, ml_summary = self._train_ml(self._closed)
            result_obj.feature_importance = fi
            result_obj.ml_accuracy   = acc
            result_obj.ml_roc_auc    = auc
            result_obj.ml_summary    = ml_summary

        result_obj.invalid_trade_setup_count = self._invalid_trade_setup_count
        result_obj.regime_unavailable_count = self._regime_unavailable_count
        result_obj.skipped_by_regime_count = self._skipped_by_regime_count
        result_obj.regime_distribution = dict(self._regime_dist)

        return result_obj

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_result(self) -> ScalpBacktestResult:
        trades = self._closed
        eq     = np.array(self._equity_curve)

        wins   = [t for t in trades if t.status == "won"]
        losses = [t for t in trades if t.status == "lost"]

        gross_profit = sum(t.pnl for t in wins)   if wins   else 0.0
        gross_loss   = abs(sum(t.pnl for t in losses)) if losses else 1e-8
        total_pnl    = sum(t.pnl for t in trades)

        win_rate      = len(wins) / len(trades) if trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_rr        = float(np.mean([t.rr for t in wins])) if wins else 0.0

        avg_win  = gross_profit / len(wins)   if wins   else 0.0
        avg_loss = gross_loss   / len(losses) if losses else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        peaks  = np.maximum.accumulate(eq)
        dds    = (peaks - eq) / (peaks + 1e-8)
        max_dd = float(dds.max())

        returns = np.diff(eq) / (eq[:-1] + 1e-8)
        sharpe  = (float(np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
                   if len(returns) > 1 else 0.0)
        calmar  = (total_pnl / self.cfg.initial_balance) / (max_dd + 1e-8)
        tot_ret = (eq[-1] - eq[0]) / eq[0] if len(eq) > 0 else 0.0

        # Segment breakdowns
        def _segment(attr: str) -> Dict[str, Dict]:
            d: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
            for t in trades:
                key = str(getattr(t, attr, "unknown"))
                d[key]["n"]    += 1
                d[key]["wins"] += (1 if t.status == "won" else 0)
                d[key]["pnl"]  = round(d[key]["pnl"] + t.pnl, 2)
            for v in d.values():
                v["wr"] = round(v["wins"] / v["n"], 4) if v["n"] > 0 else 0.0
            return dict(d)

        _pattern_name = {0:"unknown",1:"cont_bull",2:"cont_bear",
                         3:"rev_bull",4:"rev_bear",5:"breakout"}

        by_pattern: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
        for t in trades:
            key = _pattern_name.get(t.pattern_type, "unknown")
            by_pattern[key]["n"]    += 1
            by_pattern[key]["wins"] += (1 if t.status == "won" else 0)
            by_pattern[key]["pnl"]  = round(by_pattern[key]["pnl"] + t.pnl, 2)
        for v in by_pattern.values():
            v["wr"] = round(v["wins"] / v["n"], 4) if v["n"] > 0 else 0.0

        # Hour buckets (UTC)
        by_hour: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
        for t in trades:
            key = str(t.hour_utc)
            by_hour[key]["n"]    += 1
            by_hour[key]["wins"] += (1 if t.status == "won" else 0)
            by_hour[key]["pnl"]  = round(by_hour[key]["pnl"] + t.pnl, 2)
        for v in by_hour.values():
            v["wr"] = round(v["wins"] / v["n"], 4) if v["n"] > 0 else 0.0

        # Top win-rate combos (entry_type + zone_type + ltf_state, n≥5)
        from itertools import product as _product
        combo_stats: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "wins": 0})
        for t in trades:
            key = f"{t.entry_type}|{t.zone_type}|{t.ltf_state}|{t.grade}"
            combo_stats[key]["n"]    += 1
            combo_stats[key]["wins"] += (1 if t.status == "won" else 0)

        winning_combos = []
        for combo, s in combo_stats.items():
            if s["n"] >= 5:
                wr = s["wins"] / s["n"]
                parts = combo.split("|")
                winning_combos.append({
                    "entry_type": parts[0],
                    "zone_type":  parts[1],
                    "ltf_state":  parts[2],
                    "grade":      parts[3],
                    "wr":         round(wr, 4),
                    "n":          s["n"],
                    "wins":       s["wins"],
                })
        winning_combos.sort(key=lambda x: (-x["wr"], -x["n"]))

        # Insights
        insights = self._derive_insights(
            trades, _segment("entry_type"), _segment("zone_type"),
            _segment("ltf_state"), _segment("grade"), _segment("session"),
        )

        return ScalpBacktestResult(
            trades          = trades,
            equity_curve    = list(eq),
            initial_balance = self.cfg.initial_balance,
            final_equity    = round(float(eq[-1]), 2),
            total_return_pct= round(tot_ret * 100, 2),
            win_rate        = round(win_rate, 4),
            profit_factor   = round(profit_factor, 2),
            expectancy      = round(expectancy, 2),
            max_drawdown_pct= round(max_dd, 4),
            sharpe_ratio    = round(sharpe, 3),
            calmar_ratio    = round(calmar, 3),
            avg_rr          = round(avg_rr, 2),
            total_trades    = len(trades),
            winning_trades  = len(wins),
            losing_trades   = len(losses),
            by_entry_type   = _segment("entry_type"),
            by_zone_type    = _segment("zone_type"),
            by_ltf_state    = _segment("ltf_state"),
            by_grade        = _segment("grade"),
            by_session      = _segment("session"),
            by_pattern_type = dict(by_pattern),
            by_regime       = _segment("regime"),
            by_hour         = dict(by_hour),
            winning_combos  = winning_combos[:10],
            strategy_name   = self.cfg.strategy_name,
            insights        = insights,
            invalid_trade_setup_count = self._invalid_trade_setup_count,
            regime_unavailable_count  = self._regime_unavailable_count,
            skipped_by_regime_count   = self._skipped_by_regime_count,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Insight generation
    # ─────────────────────────────────────────────────────────────────────────

    def _derive_insights(
        self,
        trades,
        by_entry, by_zone, by_ltf, by_grade, by_session,
    ) -> List[str]:
        insights = []
        MIN_N = 5

        def _wr(d: dict) -> Optional[float]:
            return d.get("wr") if d.get("n", 0) >= MIN_N else None

        # Entry type
        limit_wr  = _wr(by_entry.get("limit", {}))
        market_wr = _wr(by_entry.get("market", {}))
        if limit_wr and market_wr:
            diff = limit_wr - market_wr
            if abs(diff) >= 0.05:
                better = "Limit" if diff > 0 else "Market"
                insights.append(
                    f"{'Limit' if diff>0 else 'Market'} orders outperform by "
                    f"{abs(diff):.0%} WR ({limit_wr:.0%} vs {market_wr:.0%}) "
                    f"— {'prefer zone limit entries' if diff>0 else 'market entries more reliable here'}"
                )

        # Zone type
        ob_wr  = _wr(by_zone.get("ob", {}))
        sr_wr  = _wr(by_zone.get("sr", {}))
        nz_wr  = _wr(by_zone.get("no_zone", {}))
        if ob_wr:
            insights.append(
                f"Order Block entries: {ob_wr:.0%} WR "
                f"({by_zone['ob']['n']} trades)"
                + (f" vs S/R {sr_wr:.0%}" if sr_wr else "")
                + " — OB fills are highest accuracy entries"
            )
        if nz_wr and (ob_wr or sr_wr):
            best = max(v for v in [ob_wr, sr_wr] if v is not None)
            if best - nz_wr >= 0.10:
                insights.append(
                    f"Avoid no-zone market entries: {nz_wr:.0%} WR "
                    f"vs {best:.0%} in-zone — {best-nz_wr:.0%} WR penalty"
                )

        # LTF state
        oc_wr   = _wr(by_ltf.get("one_counter", {}))
        bc_wr   = _wr(by_ltf.get("both_confirmed", {}))
        unk_wr  = _wr(by_ltf.get("unknown", {}))
        if oc_wr:
            insights.append(
                f"one_counter LTF state: {oc_wr:.0%} WR "
                f"({by_ltf['one_counter']['n']} trades) "
                f"— mixed LTF momentum = ideal pullback timing"
            )
        if bc_wr and oc_wr and (oc_wr - bc_wr) >= 0.08:
            insights.append(
                f"both_confirmed ({bc_wr:.0%}) lags one_counter ({oc_wr:.0%}) "
                f"by {oc_wr-bc_wr:.0%} — entry is extended when all LTFs agree"
            )

        # Grade
        a_wr = _wr(by_grade.get("A", {}))
        b_wr = _wr(by_grade.get("B", {}))
        c_wr = _wr(by_grade.get("C", {}))
        if a_wr:
            insights.append(
                f"Grade A: {a_wr:.0%} WR ({by_grade['A']['n']} trades) "
                f"— never skip Grade A setups regardless of session"
            )
        if c_wr and b_wr and (b_wr - c_wr) >= 0.10:
            insights.append(
                f"Grade C ({c_wr:.0%}) is {b_wr-c_wr:.0%} below Grade B ({b_wr:.0%}) "
                f"— raise min_grade to B"
            )

        # Session
        best_sess = max(by_session.items(), key=lambda kv: kv[1].get("wr", 0)
                        if kv[1].get("n", 0) >= MIN_N else 0, default=None)
        worst_sess = min(
            [(k, v) for k, v in by_session.items() if v.get("n", 0) >= MIN_N],
            key=lambda kv: kv[1].get("wr", 1.0), default=None,
        )
        if best_sess and worst_sess and best_sess[0] != worst_sess[0]:
            insights.append(
                f"Best session: {best_sess[0]} ({best_sess[1]['wr']:.0%} WR) | "
                f"Worst: {worst_sess[0]} ({worst_sess[1]['wr']:.0%} WR) — "
                f"focus activity during peak sessions"
            )

        return insights

    # ─────────────────────────────────────────────────────────────────────────
    # ML training
    # ─────────────────────────────────────────────────────────────────────────

    def _build_features(self, trades: List[ScalpTrade]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert trade records to feature matrix X and label vector y."""
        _grade_enc = {"A": 3, "B": 2, "C": 1, "D": 0, "F": 0}
        _ltf_enc   = {"both_confirmed": 2, "one_counter": 1, "unknown": 0, "trap": -1}
        _zone_enc  = {"ob": 2, "sr": 1, "no_zone": 0}
        _entry_enc = {"limit": 1, "market": 0}

        feature_names = [
            "grade", "score", "ltf_state", "zone_type_enc",
            "entry_type", "sl_atr", "pattern_type",
            "atr_rank", "hour_utc", "dow",
            "is_london_ny",    # binary: london_ny_overlap session
            "score_gt85",      # binary: score >= 85
            "zone_ob",         # binary: zone_type == ob
            "entry_limit",     # binary: entry_type == limit
            # Strategy lab extras
            "bars_open",            # bars from entry to close
            "session_loss_streak",  # consecutive losses in session at entry
            "exit_timeout_flag",    # 1 if closed by timeout logic
        ]

        rows = []
        labels = []
        for t in trades:
            if t.status not in ("won", "lost"):
                continue
            row = [
                _grade_enc.get(t.grade, 0),
                t.score,
                _ltf_enc.get(t.ltf_state, 0),
                _zone_enc.get(t.zone_type, 0),
                _entry_enc.get(t.entry_type, 0),
                t.sl_atr,
                t.pattern_type,
                t.atr_rank,
                t.hour_utc,
                t.dow,
                1 if "london_ny" in t.session else 0,
                1 if t.score >= 85 else 0,
                1 if t.zone_type == "ob" else 0,
                1 if t.entry_type == "limit" else 0,
                t.bars_open,
                t.session_loss_streak,
                1 if t.exit_timeout else 0,
            ]
            rows.append(row)
            labels.append(1 if t.status == "won" else 0)

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)
        return X, y, feature_names

    def _train_ml(
        self,
        trades: List[ScalpTrade],
    ) -> Tuple[List[Dict], Optional[float], Optional[float], Dict[str, Any]]:
        """
        Train XGBoost on closed trade features.
        Returns (feature_importance_list, accuracy, roc_auc).

        Robustness layers:
        1. Drop zero-variance features (all same value → XGBoost can't split).
        2. Use point-biserial correlation as a fallback "importance" proxy when
           XGBoost produces all-zero importances (tiny/homogeneous datasets).
        3. Simplify model params for n < 30 to avoid memorising noise.
        4. Skip stratified split when a class has < 2 members.
        """
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("ML training skipped: xgboost or sklearn not available")
            return [], None, None, {"status": "skipped_missing_deps"}

        X, y, feat_names = self._build_features(trades)

        if len(X) < 5:
            logger.warning("ML training skipped: too few samples (%d)", len(X))
            return [], None, None, {"status": "skipped_too_few_samples", "samples": len(X)}

        try:
            # ── 1. Drop zero-variance features ────────────────────────────
            variances = X.var(axis=0)
            keep_mask = variances > 0
            if not keep_mask.any():
                logger.warning("ML: all features have zero variance — using correlation proxy")
                return (
                    self._correlation_importance(X, y, feat_names),
                    None,
                    None,
                    {"status": "all_zero_variance", "samples": len(X)},
                )

            X_red  = X[:, keep_mask]
            names_red = [feat_names[i] for i, k in enumerate(keep_mask) if k]
            dropped = [feat_names[i] for i, k in enumerate(keep_mask) if not k]
            if dropped:
                logger.info("ML: dropped %d zero-variance features: %s", len(dropped), dropped)

            # ── 2. Adaptive split: no stratify when class < 2 ─────────────
            n_train = int(len(X_red) * (1 - self.cfg.ml_test_size))
            class_counts = np.bincount(y)
            can_stratify = (len(class_counts) >= 2
                            and class_counts.min() >= 2
                            and n_train >= 4)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_red, y, test_size=self.cfg.ml_test_size,
                    random_state=42, stratify=(y if can_stratify else None),
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_red, y, test_size=self.cfg.ml_test_size, random_state=42,
                )

            # ── 3. Model complexity scaled to sample size ──────────────────
            n = len(X_red)
            _n_est    = 50  if n < 20 else 100 if n < 50 else 200
            _depth    = 2   if n < 20 else 3   if n < 50 else 4
            _min_child= max(1, n // 10)   # prevent memorising tiny leaves

            model = xgb.XGBClassifier(
                n_estimators     = _n_est,
                max_depth        = _depth,
                learning_rate    = 0.1,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                min_child_weight = _min_child,
                use_label_encoder= False,
                eval_metric      = "logloss",
                random_state     = 42,
                verbosity        = 0,
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            acc     = float(accuracy_score(y_test, y_pred))
            try:
                auc = float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else None
            except Exception:
                auc = None

            # ── 4. Feature importance — fall back to correlation if all-zero ──
            imp = model.feature_importances_
            if imp.sum() == 0:
                logger.warning("ML: XGBoost importances all zero — using correlation proxy")
                fi_full = self._correlation_importance(X, y, feat_names)
            else:
                # Reconstruct full-length importance list (zeros for dropped features)
                imp_full = np.zeros(len(feat_names))
                red_idx  = [i for i, k in enumerate(keep_mask) if k]
                for j, orig_i in enumerate(red_idx):
                    imp_full[orig_i] = imp[j]
                fi_full = sorted(
                    [{"feature": feat_names[j], "importance": round(float(imp_full[j]), 4)}
                     for j in range(len(feat_names))],
                    key=lambda x: -x["importance"],
                )

            baseline = float(max(np.mean(y_test), 1.0 - np.mean(y_test))) if len(y_test) > 0 else None
            brier = float(brier_score_loss(y_test, y_proba)) if len(set(y_test)) > 1 else None

            walkforward_summary: Dict[str, Any] = {}
            if len(X_red) >= 24:
                n_splits = min(5, max(2, len(X_red) // 12))
                tscv = TimeSeriesSplit(n_splits=n_splits)
                wf_rows: List[Dict[str, Any]] = []
                for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X_red), start=1):
                    Xtr, Xte = X_red[tr_idx], X_red[te_idx]
                    ytr, yte = y[tr_idx], y[te_idx]
                    if len(set(ytr)) < 2 or len(set(yte)) < 2:
                        continue
                    fold_model = xgb.XGBClassifier(
                        n_estimators=50,
                        max_depth=2,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=max(1, len(Xtr) // 10),
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=42,
                        verbosity=0,
                    )
                    fold_model.fit(Xtr, ytr, verbose=False)
                    yhat = fold_model.predict(Xte)
                    yprob = fold_model.predict_proba(Xte)[:, 1]
                    fold_acc = float(accuracy_score(yte, yhat))
                    try:
                        fold_auc = float(roc_auc_score(yte, yprob))
                    except Exception:
                        fold_auc = None
                    wf_rows.append(
                        {
                            "fold": fold_i,
                            "n_train": int(len(Xtr)),
                            "n_test": int(len(Xte)),
                            "accuracy": round(fold_acc, 4),
                            "auc": round(fold_auc, 4) if fold_auc is not None else None,
                        }
                    )
                if wf_rows:
                    wf_accs = [r["accuracy"] for r in wf_rows if r["accuracy"] is not None]
                    wf_aucs = [r["auc"] for r in wf_rows if r["auc"] is not None]
                    walkforward_summary = {
                        "folds": wf_rows,
                        "accuracy_mean": round(float(np.mean(wf_accs)), 4) if wf_accs else None,
                        "accuracy_std": round(float(np.std(wf_accs)), 4) if wf_accs else None,
                        "auc_mean": round(float(np.mean(wf_aucs)), 4) if wf_aucs else None,
                        "auc_std": round(float(np.std(wf_aucs)), 4) if wf_aucs else None,
                    }

            ml_summary = {
                "status": "ok",
                "samples": int(n),
                "active_features": int(keep_mask.sum()),
                "baseline_accuracy": round(baseline, 4) if baseline is not None else None,
                "brier": round(brier, 4) if brier is not None else None,
                "walkforward": walkforward_summary,
                "warning": "auc_near_random" if (auc is not None and auc <= 0.55) else None,
            }

            logger.info(
                "ML trained: %d samples (%d features active) | n_est=%d depth=%d | "
                "accuracy=%.2f | AUC=%s",
                n, keep_mask.sum(), _n_est, _depth, acc,
                f"{auc:.3f}" if auc is not None else "N/A",
            )

            # ── 5. Persist model and features ─────────────────────────────
            _out_dir = _ROOT / "outputs"
            _mod_dir = _ROOT / "models"
            _out_dir.mkdir(exist_ok=True)
            _mod_dir.mkdir(exist_ok=True)

            if PICKLE_AVAILABLE:
                try:
                    import pickle
                    with open(_mod_dir / "scalp_bt_model.pkl", "wb") as f:
                        pickle.dump(model, f)
                except Exception as e:
                    logger.warning("Could not save model: %s", e)

            try:
                feat_df = pd.DataFrame(X, columns=feat_names)
                feat_df["outcome"] = y
                feat_df.to_csv(_out_dir / "scalp_bt_features.csv", index=False)
            except Exception as e:
                logger.warning("Could not save features CSV: %s", e)

            return fi_full, acc, auc, ml_summary

        except Exception as exc:
            logger.error("ML training error: %s", exc, exc_info=True)
            return [], None, None, {"status": "error", "error": str(exc)}

    def _correlation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feat_names: List[str],
    ) -> List[Dict]:
        """
        Fallback importance proxy: absolute point-biserial correlation of each
        feature with the binary outcome.  Always produces non-zero values when
        there is any relationship in the data.
        """
        from scipy.stats import pointbiserialr
        fi = []
        for j, name in enumerate(feat_names):
            col = X[:, j]
            if col.std() == 0:
                fi.append({"feature": name, "importance": 0.0, "note": "zero_variance"})
                continue
            try:
                corr, _ = pointbiserialr(col, y)
                fi.append({"feature": name, "importance": round(abs(float(corr)), 4),
                           "note": "correlation_proxy"})
            except Exception:
                fi.append({"feature": name, "importance": 0.0, "note": "error"})
        fi.sort(key=lambda x: -x["importance"])
        logger.info("Correlation proxy importances: %s",
                    {d["feature"]: d["importance"] for d in fi[:5]})
        return fi

    # ─────────────────────────────────────────────────────────────────────────
    # Report printing
    # ─────────────────────────────────────────────────────────────────────────

    def print_report(self, res: ScalpBacktestResult) -> None:
        """Print detailed console report and optionally save JSON."""
        sep = "=" * 60

        def _bar(val: float, width: int = 20) -> str:
            filled = int(val * width)
            return "█" * filled + "░" * (width - filled)

        def _seg_table(d: Dict[str, Dict], title: str) -> str:
            if not d:
                return ""
            lines = [f"\n── {title} ──"]
            rows = sorted(d.items(), key=lambda kv: kv[1].get("wr", 0), reverse=True)
            for key, v in rows:
                n   = v.get("n", 0)
                wr  = v.get("wr", 0.0)
                pnl = v.get("pnl", 0.0)
                tag = "  ← BEST" if wr == max(vv.get("wr", 0) for vv in d.values()) else ""
                lines.append(
                    f"  {key:<22} {wr:>6.1%} WR  n={n:<4}  PnL=${pnl:>+8.2f}{tag}"
                )
            return "\n".join(lines)

        print("\n" + sep)
        print("PROMETHEUS SCALP BACKTEST REPORT")
        print(f"Asset: {self.cfg.asset} | TF: {self.cfg.primary_tf.upper()}")
        if self.cfg.date_from and self.cfg.date_to:
            print(f"Period: {self.cfg.date_from.date()} → {self.cfg.date_to.date()}")
        print(sep)
        print(f"Balance:      ${res.initial_balance:.2f} → ${res.final_equity:.2f}"
              f"  ({res.total_return_pct:+.1f}%)")
        print(f"Total trades: {res.total_trades}  ({res.winning_trades}W / {res.losing_trades}L)")
        print(f"Win rate:     {res.win_rate:.1%}")
        print(f"Profit factor:{res.profit_factor:.2f}  |  Expectancy: ${res.expectancy:.2f}/trade")
        print(f"Max drawdown: {res.max_drawdown_pct:.1%}  |  Sharpe: {res.sharpe_ratio:.3f}  |  Avg RR: {res.avg_rr:.2f}")
        print(
            f"Validation: invalid_setups={res.invalid_trade_setup_count} | "
            f"regime_unavailable={res.regime_unavailable_count} | "
            f"regime_skips={res.skipped_by_regime_count}"
        )
        if res.regime_distribution:
            _kill_names = {"compression", "news_volatility", "dead_liquidity"}
            print("\n── REGIME DISTRIBUTION (bars evaluated) ──")
            total_bars = sum(res.regime_distribution.values())
            for rname, cnt in sorted(
                res.regime_distribution.items(), key=lambda kv: -kv[1]
            ):
                pct = cnt / total_bars * 100
                flag = "  ← BLOCKS ENTRIES" if rname in _kill_names else ""
                print(f"  {rname:<24} {cnt:>5} bars  ({pct:.1f}%){flag}")
            blocked_bars = sum(
                v for k, v in res.regime_distribution.items() if k in _kill_names
            )
            if blocked_bars:
                print(
                    f"  \n  {blocked_bars}/{total_bars} evaluated bars ({blocked_bars/total_bars:.0%}) "
                    "had kill_entries=True — run on a period with more "
                    "trend_expansion / trend_exhaustion bars to see trades."
                )

        print(_seg_table(res.by_entry_type,  "BY ENTRY TYPE"))
        print(_seg_table(res.by_zone_type,   "BY ZONE TYPE"))
        print(_seg_table(res.by_ltf_state,   "BY LTF STATE"))
        print(_seg_table(res.by_grade,       "BY GRADE"))
        print(_seg_table(res.by_session,     "BY SESSION"))
        print(_seg_table(res.by_pattern_type,"BY PATTERN TYPE"))
        print(_seg_table(res.by_regime,      "BY REGIME"))

        # Top combos
        if res.winning_combos:
            print("\n── TOP WIN-RATE COMBINATIONS (n≥5) ──")
            for j, c in enumerate(res.winning_combos[:5], 1):
                print(
                    f"  {j}. {c['entry_type']} + {c['zone_type']} + "
                    f"{c['ltf_state']} + Grade {c['grade']} "
                    f"→ {c['wr']:.0%} WR  ({c['n']} trades)"
                )

        # ML
        if res.feature_importance:
            print("\n── ML FEATURE IMPORTANCE (XGBoost) ──")
            for fi in res.feature_importance[:8]:
                bar = _bar(fi["importance"])
                print(f"  {fi['feature']:<22} {fi['importance']:.3f}  {bar}")
            if res.ml_accuracy is not None:
                print(f"\n  Test accuracy: {res.ml_accuracy:.1%}"
                      + (f"  |  AUC: {res.ml_roc_auc:.3f}" if res.ml_roc_auc else ""))
            if res.ml_summary:
                bacc = res.ml_summary.get("baseline_accuracy")
                if bacc is not None:
                    print(f"  Baseline acc: {bacc:.1%}")
                brier = res.ml_summary.get("brier")
                if brier is not None:
                    print(f"  Brier: {brier:.4f}")
                wf = res.ml_summary.get("walkforward") or {}
                if wf:
                    print(
                        "  Walk-forward: "
                        f"acc={wf.get('accuracy_mean')}±{wf.get('accuracy_std')} "
                        f"auc={wf.get('auc_mean')}±{wf.get('auc_std')}"
                    )

        # Insights
        if res.insights:
            print("\n── WHAT IMPROVES WIN RATE ──")
            for ins in res.insights:
                print(f"  • {ins}")

        print(sep + "\n")

        # Save JSON report
        report_path = self.cfg.report_path
        if report_path:
            try:
                out = {
                    "asset":            self.cfg.asset,
                    "primary_tf":       self.cfg.primary_tf,
                    "date_from":        str(self.cfg.date_from),
                    "date_to":          str(self.cfg.date_to),
                    "initial_balance":  res.initial_balance,
                    "final_equity":     res.final_equity,
                    "total_return_pct": res.total_return_pct,
                    "win_rate":         res.win_rate,
                    "profit_factor":    res.profit_factor,
                    "expectancy":       res.expectancy,
                    "max_drawdown_pct": res.max_drawdown_pct,
                    "sharpe_ratio":     res.sharpe_ratio,
                    "calmar_ratio":     res.calmar_ratio,
                    "avg_rr":           res.avg_rr,
                    "total_trades":     res.total_trades,
                    "winning_trades":   res.winning_trades,
                    "losing_trades":    res.losing_trades,
                    "by_entry_type":    res.by_entry_type,
                    "by_zone_type":     res.by_zone_type,
                    "by_ltf_state":     res.by_ltf_state,
                    "by_grade":         res.by_grade,
                    "by_session":       res.by_session,
                    "by_pattern_type":  res.by_pattern_type,
                    "by_regime":        res.by_regime,
                    "by_hour":          res.by_hour,
                    "winning_combos":   res.winning_combos,
                    "feature_importance": res.feature_importance,
                    "ml_accuracy":      res.ml_accuracy,
                    "ml_roc_auc":       res.ml_roc_auc,
                    "ml_summary":       res.ml_summary,
                    "invalid_trade_setup_count": res.invalid_trade_setup_count,
                    "regime_unavailable_count": res.regime_unavailable_count,
                    "skipped_by_regime_count":  res.skipped_by_regime_count,
                    "regime_distribution":      res.regime_distribution,
                    "insights":         res.insights,
                    "trades": [
                        {
                            "trade_id":    t.trade_id,
                            "direction":   t.direction,
                            "entry_type":  t.entry_type,
                            "entry_price": t.entry_price,
                            "sl_price":    t.sl_price,
                            "tp1_price":   t.tp1_price,
                            "exit_price":  t.exit_price,
                            "pnl":         t.pnl,
                            "rr":          t.rr,
                            "status":      t.status,
                            "exit_reason": t.exit_reason,
                            "stop_type":   t.stop_type,
                            "grade":       t.grade,
                            "score":       t.score,
                            "ltf_state":   t.ltf_state,
                            "zone_type":   t.zone_type,
                            "zone_pos":    t.zone_pos,
                            "regime":      t.regime,
                            "session":     t.session,
                            "hour_utc":    t.hour_utc,
                        }
                        for t in res.trades
                    ],
                }
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(
                    json.dumps(out, indent=2, default=str), encoding="utf-8"
                )
                logger.info("Report saved to %s", report_path)
            except Exception as exc:
                logger.warning("Could not save report: %s", exc)
