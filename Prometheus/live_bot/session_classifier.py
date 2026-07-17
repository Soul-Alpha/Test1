"""
Session Classifier for Prometheus Trading Bot
==============================================
Classifies the current trading session based on UTC time and provides
session-specific behavioural parameters.

Sessions (all times UTC):
  ASIAN            – 00:00 – 07:59  (Tokyo / Sydney)
  LONDON_OPEN      – 08:00 – 09:59  (high momentum, trending)
  LONDON           – 10:00 – 11:59  (settled, continuation)
  NY_LUNCH         – 12:00 – 12:59  (choppy, reduced volume)
  LONDON_NY_OVERLAP– 13:00 – 16:59  (highest volume, best breakouts)
  NY_AFTERNOON     – 17:00 – 19:59  (NY afternoon, fading)
  DEAD_ZONE        – 20:00 – 23:59  (end of NY / rollover — avoid new entries)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class Session(str, Enum):
    ASIAN            = "asian"
    LONDON_OPEN      = "london_open"
    LONDON           = "london"
    NY_LUNCH         = "ny_lunch"
    LONDON_NY_OVERLAP= "london_ny_overlap"
    NY_AFTERNOON     = "ny_afternoon"
    DEAD_ZONE        = "dead_zone"
    UNKNOWN          = "unknown"


@dataclass
class SessionState:
    session: Session
    # 0.0 = skip new entries, 1.0 = full activity
    trade_allowed_scalar: float = 1.0
    # Spread tolerance multiplier (1.0 = base, 0.5 = only half the base spread allowed)
    spread_tolerance: float = 1.0
    # Preferred signal archetypes for this session
    preferred_archetypes: tuple[str, ...] = ()
    # Human-readable label
    description: str = ""
    # Should we skip NEW entries entirely (not exits/management)
    skip_new_entries: bool = False
    # Qualitative volatility expectation for the session
    volatility_expectation: str = "medium"   # "low" | "medium" | "high"
    # True during London 08-10 UTC — highest AMD manipulation probability
    amd_manipulation_window: bool = False


# ---------------------------------------------------------------------------
# Session profile table
# ---------------------------------------------------------------------------
_SESSION_PARAMS: dict[Session, dict] = {
    Session.ASIAN: {
        "trade_allowed_scalar": 0.5,          # reduced sizing — low volatility session
        "spread_tolerance": 0.75,             # spreads wider in Asian hours
        "preferred_archetypes": ("mean_reversion", "range_fade", "reversal"),
        # Unblocked 2026-06-03: live Grade A / Score 92 / Trend Exhaustion 100% conf
        # / Full HTF alignment / Exec Quality PASSED observed during Asian session.
        # Session is open but score floor premium (+5) and reduced lot scalar apply
        # to filter low-quality setups — confluence scoring handles selectivity.
        "skip_new_entries": False,
        "volatility_expectation": "low",
        "amd_manipulation_window": False,
        "description": "Asian session – open, reduced sizing (lot×0.5), +5 score floor",
    },
    Session.LONDON_OPEN: {
        "trade_allowed_scalar": 1.0,
        "spread_tolerance": 0.90,
        "preferred_archetypes": ("breakout", "continuation", "liquidity_sweep"),
        "skip_new_entries": False,
        "volatility_expectation": "high",
        "amd_manipulation_window": True,   # AMD sweep window: 08-10 UTC
        "description": "London open – strong momentum, breakout preferred",
    },
    Session.LONDON: {
        "trade_allowed_scalar": 0.9,
        "spread_tolerance": 1.0,
        "preferred_archetypes": ("continuation", "pullback"),
        "skip_new_entries": False,
        "volatility_expectation": "medium",
        "amd_manipulation_window": False,
        "description": "London mid-session – continuation setups",
    },
    Session.NY_LUNCH: {
        "trade_allowed_scalar": 0.5,
        "spread_tolerance": 0.70,          # spreads often wider at lunch
        "preferred_archetypes": (),
        "skip_new_entries": False,
        "volatility_expectation": "low",
        "amd_manipulation_window": False,
        "description": "NY lunch (12:00-12:59 UTC) – choppy, avoid new breakout entries",
    },
    Session.LONDON_NY_OVERLAP: {
        "trade_allowed_scalar": 1.0,          # highest directional volume — full sizing
        "spread_tolerance": 1.0,
        "preferred_archetypes": ("continuation", "breakout", "momentum_scalp"),
        "skip_new_entries": False,            # unblocked: 0W/8L too small to suppress best session
        "volatility_expectation": "high",
        "amd_manipulation_window": False,
        "description": "London/NY overlap – 13:00-16:59 UTC – highest directional volume, full activity",
    },
    Session.NY_AFTERNOON: {
        "trade_allowed_scalar": 0.0,
        "spread_tolerance": 0.85,
        "preferred_archetypes": ("mean_reversion", "continuation"),
        "skip_new_entries": True,           # learning: 0W/28L (-128 pnl) — hard block
        "volatility_expectation": "medium",
        "amd_manipulation_window": False,
        "description": "NY afternoon – 0W/28L historically, all new entries blocked",
    },
    Session.DEAD_ZONE: {
        "trade_allowed_scalar": 0.4,
        "spread_tolerance": 0.5,
        "preferred_archetypes": ("mean_reversion", "range_fade"),
        "skip_new_entries": False,          # unblocked 2026-06-04: Asian accumulation begins
        "volatility_expectation": "low",
        "amd_manipulation_window": False,
        "description": "Dead zone (20:00-23:59 UTC) – open, reduced sizing (lot×0.4), wide spread filter",
    },
    Session.UNKNOWN: {
        "trade_allowed_scalar": 0.8,
        "spread_tolerance": 0.9,
        "preferred_archetypes": (),
        "skip_new_entries": False,
        "volatility_expectation": "medium",
        "amd_manipulation_window": False,
        "description": "Session unknown",
    },
}


class SessionClassifier:
    """
    Classify the current trading session from a UTC datetime.

    Usage::
        clf = SessionClassifier()
        state = clf.classify()       # uses datetime.now(UTC)
        state = clf.classify(dt)     # pass explicit datetime
    """

    def classify(self, dt: datetime | None = None) -> SessionState:
        """Return the current SessionState."""
        try:
            if dt is None:
                dt = datetime.now(timezone.utc)
            elif dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return self._classify(dt)
        except Exception as exc:
            logger.warning("[session] Classification error: %s", exc)
            return self._make_state(Session.UNKNOWN)

    def _classify(self, dt: datetime) -> SessionState:
        h = dt.hour  # UTC hour 0-23

        # Weekend detection: Saturday (5) and Sunday (6) are always dead zone
        if dt.weekday() >= 5:
            return self._make_state(Session.DEAD_ZONE)

        if 0 <= h < 8:
            session = Session.ASIAN
        elif 8 <= h < 10:
            session = Session.LONDON_OPEN
        elif 10 <= h < 12:
            session = Session.LONDON
        elif h == 12:
            session = Session.NY_LUNCH
        elif 13 <= h < 17:
            session = Session.LONDON_NY_OVERLAP
        elif 17 <= h < 20:
            session = Session.NY_AFTERNOON
        else:  # 20-23
            session = Session.DEAD_ZONE

        return self._make_state(session)

    @staticmethod
    def _make_state(session: Session) -> SessionState:
        p = _SESSION_PARAMS[session]
        return SessionState(
            session                = session,
            trade_allowed_scalar   = p["trade_allowed_scalar"],
            spread_tolerance       = p["spread_tolerance"],
            preferred_archetypes   = p["preferred_archetypes"],
            skip_new_entries       = p["skip_new_entries"],
            volatility_expectation = p["volatility_expectation"],
            amd_manipulation_window= p["amd_manipulation_window"],
            description            = p["description"],
        )
