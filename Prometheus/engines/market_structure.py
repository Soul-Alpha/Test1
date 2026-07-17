"""
Market Structure Engine
=======================
Identifies swing highs/lows, classifies trend structure (bullish / bearish /
sideways), detects Break of Structure (BOS), and Change of Character (CHoCH).

Professional approach:
  - ZigZag-style pivot detection with configurable sensitivity
  - ATR-based filtering removes noise swings
  - Structure classification from the last N swing sequence
  - BOS  = price breaks through previous significant swing IN trend direction
  - CHoCH = price breaks AGAINST the prevailing structure (reversal signal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class StructureType(Enum):
    BULLISH   = auto()
    BEARISH   = auto()
    SIDEWAYS  = auto()
    UNDEFINED = auto()


class SwingType(Enum):
    HIGH = auto()
    LOW  = auto()


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Swing:
    index:      int
    price:      float
    swing_type: SwingType
    bar_time:   Optional[pd.Timestamp] = None
    strength:   float = 1.0   # 1 – 10, relative to surrounding price action


@dataclass
class StructureEvent:
    event_type: str              # "BOS" | "CHoCH"
    direction:  str              # "bullish" | "bearish"
    price:      float            # level that was broken
    index:      int
    bar_time:   Optional[pd.Timestamp] = None
    confirmed:  bool = False


@dataclass
class MarketStructureResult:
    structure_type: StructureType            = StructureType.UNDEFINED
    swing_highs:    List[Swing]              = field(default_factory=list)
    swing_lows:     List[Swing]              = field(default_factory=list)
    higher_highs:   List[Swing]              = field(default_factory=list)
    higher_lows:    List[Swing]              = field(default_factory=list)
    lower_highs:    List[Swing]              = field(default_factory=list)
    lower_lows:     List[Swing]              = field(default_factory=list)
    bos_events:     List[StructureEvent]     = field(default_factory=list)
    choch_events:   List[StructureEvent]     = field(default_factory=list)
    trend_strength: float                    = 0.0   # 0–1
    current_atr:    float                    = 0.0
    narrative:      str                      = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class MarketStructureEngine:
    """
    Full market structure analysis pipeline.

    Usage::

        engine = MarketStructureEngine(pivot_sensitivity=5)
        result = engine.analyze(df)   # df has columns: open, high, low, close, volume
    """

    def __init__(
        self,
        pivot_sensitivity:  int   = 5,
        min_swing_atr_mult: float = 0.5,
        atr_period:         int   = 14,
    ) -> None:
        self.sensitivity        = pivot_sensitivity
        self.min_swing_atr_mult = min_swing_atr_mult
        self.atr_period         = atr_period

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> MarketStructureResult:
        """
        Complete analysis: swings → structure → BOS → CHoCH → narrative.

        Args:
            df: OHLCV DataFrame.  Required columns: open, high, low, close, volume.
                Index should be a DatetimeIndex (not required but recommended).

        Returns:
            MarketStructureResult
        """
        df = self._validate_df(df)
        atr = self._calculate_atr(df)

        result = MarketStructureResult(current_atr=atr)

        swing_highs, swing_lows = self.detect_swings(df, atr)
        result.swing_highs = swing_highs
        result.swing_lows  = swing_lows

        (
            structure,
            result.higher_highs,
            result.higher_lows,
            result.lower_highs,
            result.lower_lows,
        ) = self.classify_structure(swing_highs, swing_lows)
        result.structure_type = structure

        aligned = (
            len(result.higher_highs) + len(result.higher_lows)
            if structure == StructureType.BULLISH
            else len(result.lower_highs) + len(result.lower_lows)
        )
        total = (
            len(result.higher_highs) + len(result.higher_lows)
            + len(result.lower_highs) + len(result.lower_lows)
        )
        result.trend_strength = aligned / total if total > 0 else 0.0

        # Mixed swing evidence with only moderate alignment should remain
        # sideways to avoid false directional classification in ranging regimes.
        if result.structure_type == StructureType.BULLISH:
            opposing = len(result.lower_highs) + len(result.lower_lows)
            if opposing >= 2 and result.trend_strength < 0.72:
                result.structure_type = StructureType.SIDEWAYS
        elif result.structure_type == StructureType.BEARISH:
            opposing = len(result.higher_highs) + len(result.higher_lows)
            if opposing >= 2 and result.trend_strength < 0.72:
                result.structure_type = StructureType.SIDEWAYS

        structure = result.structure_type
        result.bos_events   = self.detect_bos(df, swing_highs, swing_lows, structure)
        result.choch_events = self.detect_choch(df, swing_highs, swing_lows, structure)
        result.narrative    = self._build_narrative(result)

        logger.info(
            "Structure: %s | Strength: %.0f%% | BOS: %d | CHoCH: %d",
            result.structure_type.name,
            result.trend_strength * 100,
            len(result.bos_events),
            len(result.choch_events),
        )
        return result

    def detect_swings(
        self, df: pd.DataFrame, atr: Optional[float] = None
    ) -> Tuple[List[Swing], List[Swing]]:
        """
        Detect swing highs and lows via local-extrema on the high/low series.

        A bar is a swing high if its `high` is the maximum of a window of
        (sensitivity) bars to either side.  Symmetric for lows.
        """
        if len(df) < self.sensitivity * 2 + 3:
            logger.warning("Insufficient bars for swing detection (%d)", len(df))
            return [], []

        if atr is None:
            atr = self._calculate_atr(df)

        min_size     = atr * self.min_swing_atr_mult
        highs        = df["high"].values
        lows         = df["low"].values
        n            = self.sensitivity
        swing_highs: List[Swing] = []
        swing_lows:  List[Swing] = []

        for i in range(n, len(df) - n):
            window_h = highs[i - n : i + n + 1]
            if highs[i] == window_h.max() and int((window_h == highs[i]).sum()) == 1:
                strength = float(
                    min(10.0, (highs[i] - window_h.mean()) / (atr + 1e-8) * 2)
                )
                swing_highs.append(
                    Swing(
                        index=i,
                        price=float(highs[i]),
                        swing_type=SwingType.HIGH,
                        bar_time=_get_ts(df, i),
                        strength=max(1.0, strength),
                    )
                )

            window_l = lows[i - n : i + n + 1]
            if lows[i] == window_l.min() and int((window_l == lows[i]).sum()) == 1:
                strength = float(
                    min(10.0, (window_l.mean() - lows[i]) / (atr + 1e-8) * 2)
                )
                swing_lows.append(
                    Swing(
                        index=i,
                        price=float(lows[i]),
                        swing_type=SwingType.LOW,
                        bar_time=_get_ts(df, i),
                        strength=max(1.0, strength),
                    )
                )

        # Remove micro-swings smaller than ATR threshold
        swing_highs = _filter_small_swings(swing_highs, min_size)
        swing_lows  = _filter_small_swings(swing_lows,  min_size)

        logger.debug("%d swing highs, %d swing lows", len(swing_highs), len(swing_lows))
        return swing_highs, swing_lows

    def classify_structure(
        self,
        swing_highs: List[Swing],
        swing_lows:  List[Swing],
        n_swings:    int = 6,
    ) -> Tuple[StructureType, List[Swing], List[Swing], List[Swing], List[Swing]]:
        """
        Classify market structure from the last N swings.

        Returns:
            (structure_type, higher_highs, higher_lows, lower_highs, lower_lows)
        """
        empty: Tuple[StructureType, List, List, List, List] = (
            StructureType.UNDEFINED, [], [], [], []
        )
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return empty

        r_highs = sorted(swing_highs, key=lambda s: s.index)[-n_swings:]
        r_lows  = sorted(swing_lows,  key=lambda s: s.index)[-n_swings:]

        hh, lh, hl, ll = [], [], [], []

        for i in range(1, len(r_highs)):
            (hh if r_highs[i].price > r_highs[i - 1].price else lh).append(r_highs[i])

        for i in range(1, len(r_lows)):
            (hl if r_lows[i].price > r_lows[i - 1].price else ll).append(r_lows[i])

        bull_score = len(hh) + len(hl)
        bear_score = len(lh) + len(ll)
        total      = bull_score + bear_score

        if total == 0:
            return StructureType.SIDEWAYS, hh, hl, lh, ll

        # Treat near-balanced swing evidence as sideways to avoid overclassifying
        # noisy ranging markets as directional trends.
        if abs(bull_score - bear_score) <= 1:
            return StructureType.SIDEWAYS, hh, hl, lh, ll

        ratio = bull_score / total
        if ratio >= 0.60:
            structure = StructureType.BULLISH
        elif ratio <= 0.40:
            structure = StructureType.BEARISH
        else:
            structure = StructureType.SIDEWAYS

        return structure, hh, hl, lh, ll

    def detect_bos(
        self,
        df:          pd.DataFrame,
        swing_highs: List[Swing],
        swing_lows:  List[Swing],
        structure:   StructureType,
    ) -> List[StructureEvent]:
        """
        Break of Structure: price closes beyond a previous swing in the
        DIRECTION of the current trend (continuation signal).

        Bullish BOS → close above a previous swing high.
        Bearish BOS → close below a previous swing low.
        """
        events: List[StructureEvent] = []
        closes = df["close"].values

        sorted_highs = sorted(swing_highs, key=lambda s: s.index)
        sorted_lows  = sorted(swing_lows,  key=lambda s: s.index)

        if structure in (StructureType.BULLISH, StructureType.SIDEWAYS):
            for sh in sorted_highs[:-1]:
                segment = closes[sh.index :]
                mask    = segment > sh.price
                if mask.any():
                    idx = int(sh.index + int(np.argmax(mask)))
                    events.append(
                        StructureEvent(
                            event_type="BOS",
                            direction="bullish",
                            price=sh.price,
                            index=idx,
                            bar_time=_get_ts(df, idx),
                            confirmed=True,
                        )
                    )

        if structure in (StructureType.BEARISH, StructureType.SIDEWAYS):
            for sl in sorted_lows[:-1]:
                segment = closes[sl.index :]
                mask    = segment < sl.price
                if mask.any():
                    idx = int(sl.index + int(np.argmax(mask)))
                    events.append(
                        StructureEvent(
                            event_type="BOS",
                            direction="bearish",
                            price=sl.price,
                            index=idx,
                            bar_time=_get_ts(df, idx),
                            confirmed=True,
                        )
                    )

        # De-duplicate (keep earliest per level)
        seen: set[float] = set()
        unique: List[StructureEvent] = []
        for ev in sorted(events, key=lambda e: e.index):
            key = round(ev.price, 4)
            if key not in seen:
                seen.add(key)
                unique.append(ev)
        return unique

    def detect_choch(
        self,
        df:          pd.DataFrame,
        swing_highs: List[Swing],
        swing_lows:  List[Swing],
        structure:   StructureType,
    ) -> List[StructureEvent]:
        """
        Change of Character: price breaks AGAINST the prevailing structure.

        Bullish CHoCH → in uptrend, price breaks below a Higher Low.
        Bearish CHoCH → in downtrend, price breaks above a Lower High.
        """
        events: List[StructureEvent] = []
        closes = df["close"].values

        sorted_highs = sorted(swing_highs, key=lambda s: s.index)
        sorted_lows  = sorted(swing_lows,  key=lambda s: s.index)

        if structure == StructureType.BULLISH:
            # A Higher Low gets violated → bearish CHoCH
            for i in range(1, len(sorted_lows)):
                sl, prev = sorted_lows[i], sorted_lows[i - 1]
                if sl.price > prev.price:          # confirmed Higher Low
                    seg  = closes[sl.index :]
                    mask = seg < sl.price
                    if mask.any():
                        idx = int(sl.index + int(np.argmax(mask)))
                        events.append(
                            StructureEvent(
                                event_type="CHoCH",
                                direction="bearish",
                                price=sl.price,
                                index=idx,
                                bar_time=_get_ts(df, idx),
                                confirmed=True,
                            )
                        )

        elif structure == StructureType.BEARISH:
            # A Lower High gets violated → bullish CHoCH
            for i in range(1, len(sorted_highs)):
                sh, prev = sorted_highs[i], sorted_highs[i - 1]
                if sh.price < prev.price:          # confirmed Lower High
                    seg  = closes[sh.index :]
                    mask = seg > sh.price
                    if mask.any():
                        idx = int(sh.index + int(np.argmax(mask)))
                        events.append(
                            StructureEvent(
                                event_type="CHoCH",
                                direction="bullish",
                                price=sh.price,
                                index=idx,
                                bar_time=_get_ts(df, idx),
                                confirmed=True,
                            )
                        )

        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        trs: List[float] = [
            float(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            ))
            for i in range(1, len(df))
        ]
        period = min(self.atr_period, len(trs))
        return float(np.mean(trs[-period:])) if trs else 1.0

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - {c.lower() for c in df.columns}
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _build_narrative(self, r: MarketStructureResult) -> str:
        label = {
            StructureType.BULLISH:  "bullish",
            StructureType.BEARISH:  "bearish",
            StructureType.SIDEWAYS: "sideways / consolidating",
            StructureType.UNDEFINED:"undefined",
        }[r.structure_type]

        parts = [f"Market structure is {label}."]

        if r.structure_type == StructureType.BULLISH:
            parts.append(
                f"Price is printing higher highs and higher lows, with "
                f"{r.trend_strength:.0%} structural alignment to the upside."
            )
        elif r.structure_type == StructureType.BEARISH:
            parts.append(
                f"Price is printing lower highs and lower lows, with "
                f"{r.trend_strength:.0%} structural alignment to the downside."
            )
        else:
            parts.append(
                "Price is compressing between established swing highs and lows, "
                "suggesting accumulation or distribution."
            )

        if r.choch_events:
            latest = max(r.choch_events, key=lambda e: e.index)
            parts.append(
                f"A Change of Character (CHoCH) was registered at {latest.price:.4f}, "
                f"warning of a potential {latest.direction} reversal."
            )

        if r.bos_events:
            latest = max(r.bos_events, key=lambda e: e.index)
            parts.append(
                f"The most recent Break of Structure occurred at {latest.price:.4f} "
                f"({latest.direction})."
            )

        return "  ".join(parts)


# ─────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────

def _get_ts(df: pd.DataFrame, idx: int) -> Optional[pd.Timestamp]:
    try:
        return df.index[idx]
    except Exception:
        return None


def _filter_small_swings(swings: List[Swing], min_size: float) -> List[Swing]:
    """Remove swings that are within min_size of the previous swing."""
    if len(swings) < 2:
        return swings
    out = [swings[0]]
    for s in swings[1:]:
        if abs(s.price - out[-1].price) >= min_size:
            out.append(s)
    return out
