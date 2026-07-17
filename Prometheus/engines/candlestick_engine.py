"""
Candlestick Analysis Engine
============================
Context-aware candlestick pattern detection with psychological scoring.

Patterns detected:
  - Pin Bar (bullish / bearish)
  - Engulfing (bullish / bearish)
  - Doji
  - Morning Star / Evening Star
  - Inside Bar
  - Outside Bar
  - Hammer / Shooting Star
  - Marubozu

Professional principle: location matters.
  A bullish pin bar at strong support scores far higher than the same pattern
  mid-trend.  Each pattern's raw score is multiplied by a location multiplier
  derived from proximity to S/R, Fib levels, or structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class CandleSignal:
    pattern:    str          # pattern name
    direction:  str          # "bullish" | "bearish" | "neutral"
    bar_index:  int
    bar_time:   Optional[pd.Timestamp] = None
    raw_score:  float = 0.0  # 0–10 quality of the candlestick itself
    ctx_score:  float = 0.0  # 0–1 location/context multiplier
    final_score: float = 0.0 # raw_score * ctx_score

    def __post_init__(self) -> None:
        self.final_score = round(self.raw_score * (0.4 + 0.6 * self.ctx_score), 2)


@dataclass
class CandlestickResult:
    signals:      List[CandleSignal] = field(default_factory=list)
    top_signals:  List[CandleSignal] = field(default_factory=list)
    bullish_count: int = 0
    bearish_count: int = 0
    narrative:    str = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class CandlestickEngine:
    """
    Full candlestick pattern scanner with context scoring.

    Usage::

        engine = CandlestickEngine()
        result = engine.analyze(df, support_levels=[2300.0], resistance_levels=[2380.0])
    """

    def __init__(
        self,
        pin_bar_wick_ratio: float = 2.0,
        doji_body_pct:      float = 0.05,
        engulf_overlap:     float = 0.0,
        atr_period:         int   = 14,
    ) -> None:
        self.pin_wick_ratio  = pin_bar_wick_ratio
        self.doji_body_pct   = doji_body_pct
        self.engulf_overlap  = engulf_overlap
        self.atr_period      = atr_period

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        df:                 pd.DataFrame,
        support_levels:     Optional[List[float]] = None,
        resistance_levels:  Optional[List[float]] = None,
        fib_levels:         Optional[List[float]] = None,
        top_n:              int = 10,
    ) -> CandlestickResult:
        """
        Scan all bars for candlestick patterns and score each in context.

        Args:
            df:                 OHLCV DataFrame
            support_levels:     list of support prices
            resistance_levels:  list of resistance prices
            fib_levels:         list of Fibonacci level prices
            top_n:              how many top signals to return in result.top_signals
        """
        df      = self._validate(df)
        atr     = self._calc_atr(df)
        signals: List[CandleSignal] = []

        ctx_levels = list(support_levels or []) + list(resistance_levels or []) + list(fib_levels or [])

        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values

        for i in range(2, len(df)):
            bar_signals = self._scan_bar(i, opens, highs, lows, closes, atr, ctx_levels, df)
            signals.extend(bar_signals)

        # 3-candle patterns
        for i in range(2, len(df)):
            three = self._scan_three_candle(i, opens, highs, lows, closes, atr, ctx_levels, df)
            signals.extend(three)

        signals.sort(key=lambda s: s.final_score, reverse=True)

        result = CandlestickResult(
            signals      = signals,
            top_signals  = signals[:top_n],
            bullish_count = sum(1 for s in signals if s.direction == "bullish"),
            bearish_count = sum(1 for s in signals if s.direction == "bearish"),
        )
        result.narrative = self._build_narrative(result)
        logger.info(
            "Candlestick scan: %d signals (%d bullish, %d bearish)",
            len(signals), result.bullish_count, result.bearish_count,
        )
        return result

    # ── Pattern detectors ─────────────────────────────────────────────────────

    def detect_pinbar(self, o: float, h: float, l: float, c: float) -> Optional[str]:
        """
        Returns 'bullish' / 'bearish' / None.

        Bullish pin bar: long lower wick, small body near top.
        Bearish pin bar: long upper wick, small body near bottom.
        """
        body       = abs(c - o)
        bar_range  = h - l
        if bar_range == 0:
            return None
        lower_wick  = min(o, c) - l
        upper_wick  = h - max(o, c)

        if lower_wick >= self.pin_wick_ratio * max(body, 1e-8) and lower_wick > upper_wick:
            return "bullish"
        if upper_wick >= self.pin_wick_ratio * max(body, 1e-8) and upper_wick > lower_wick:
            return "bearish"
        return None

    def detect_engulfing(
        self, o: float, c: float, po: float, pc: float
    ) -> Optional[str]:
        """
        Bullish engulfing: current bullish body fully covers previous bearish body.
        Bearish engulfing: current bearish body fully covers previous bullish body.
        """
        # Bullish
        if c > o and pc < po:            # current bullish, prev bearish
            if c >= po and o <= pc:
                return "bullish"
        # Bearish
        if c < o and pc > po:            # current bearish, prev bullish
            if c <= po and o >= pc:
                return "bearish"
        return None

    def detect_doji(self, o: float, h: float, l: float, c: float) -> bool:
        bar_range = h - l
        if bar_range == 0:
            return True
        return abs(c - o) / bar_range <= self.doji_body_pct

    def detect_inside_bar(
        self, h: float, l: float, ph: float, pl: float
    ) -> bool:
        return h <= ph and l >= pl

    def detect_outside_bar(
        self, h: float, l: float, ph: float, pl: float
    ) -> bool:
        return h > ph and l < pl

    def detect_hammer(self, o: float, h: float, l: float, c: float) -> Optional[str]:
        """
        Hammer = small body near top, long lower wick (like bullish pin bar but
        at the bottom of a move).  Shooting star = small body near bottom, long
        upper wick.
        """
        return self.detect_pinbar(o, h, l, c)  # same geometry, context differs

    def detect_marubozu(self, o: float, h: float, l: float, c: float) -> Optional[str]:
        """Strong momentum candle: body ≥ 80 % of range, almost no wicks."""
        bar_range = h - l
        if bar_range == 0:
            return None
        body = abs(c - o)
        if body / bar_range >= 0.80:
            return "bullish" if c > o else "bearish"
        return None

    # ── Internal scanning helpers ─────────────────────────────────────────────

    def _scan_bar(
        self,
        i: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows:  np.ndarray,
        closes: np.ndarray,
        atr:   float,
        ctx_levels: List[float],
        df: pd.DataFrame,
    ) -> List[CandleSignal]:
        o, h, l, c   = opens[i], highs[i], lows[i], closes[i]
        po, ph, pl, pc = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
        ts = _get_ts(df, i)
        out: List[CandleSignal] = []

        # Pin bar
        pb = self.detect_pinbar(o, h, l, c)
        if pb:
            raw = self._pin_bar_score(o, h, l, c)
            ctx = self._context_score(c, ctx_levels, atr)
            out.append(CandleSignal("Pin Bar", pb, i, ts, raw, ctx))

        # Engulfing
        eng = self.detect_engulfing(o, c, po, pc)
        if eng:
            raw = self._engulf_score(o, h, l, c, po, ph, pl, pc)
            ctx = self._context_score(c, ctx_levels, atr)
            out.append(CandleSignal("Engulfing", eng, i, ts, raw, ctx))

        # Doji
        if self.detect_doji(o, h, l, c):
            ctx = self._context_score(c, ctx_levels, atr)
            out.append(CandleSignal("Doji", "neutral", i, ts, 4.0, ctx))

        # Inside bar
        if self.detect_inside_bar(h, l, ph, pl):
            direction = "bullish" if c > o else "bearish"
            ctx = self._context_score(c, ctx_levels, atr)
            out.append(CandleSignal("Inside Bar", direction, i, ts, 5.0, ctx))

        # Outside bar
        if self.detect_outside_bar(h, l, ph, pl):
            direction = "bullish" if c > o else "bearish"
            ctx = self._context_score(c, ctx_levels, atr)
            out.append(CandleSignal("Outside Bar", direction, i, ts, 6.0, ctx))

        # Marubozu
        maru = self.detect_marubozu(o, h, l, c)
        if maru:
            raw = 7.5
            ctx = self._context_score(c, ctx_levels, atr)
            out.append(CandleSignal("Marubozu", maru, i, ts, raw, ctx))

        return out

    def _scan_three_candle(
        self,
        i: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows:  np.ndarray,
        closes: np.ndarray,
        atr:   float,
        ctx_levels: List[float],
        df: pd.DataFrame,
    ) -> List[CandleSignal]:
        if i < 2:
            return []
        o1, h1, l1, c1 = opens[i-2], highs[i-2], lows[i-2], closes[i-2]
        o2, h2, l2, c2 = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
        o3, h3, l3, c3 = opens[i],   highs[i],   lows[i],   closes[i]
        ts = _get_ts(df, i)
        out: List[CandleSignal] = []

        # Morning Star
        if (c1 < o1                             # candle 1 bearish
                and self.detect_doji(o2, h2, l2, c2)   # candle 2 doji/small
                and c3 > o3                     # candle 3 bullish
                and c3 > (o1 + c1) / 2):        # closes above midpoint of candle 1
            ctx = self._context_score(c3, ctx_levels, atr)
            out.append(CandleSignal("Morning Star", "bullish", i, ts, 8.0, ctx))

        # Evening Star
        if (c1 > o1
                and self.detect_doji(o2, h2, l2, c2)
                and c3 < o3
                and c3 < (o1 + c1) / 2):
            ctx = self._context_score(c3, ctx_levels, atr)
            out.append(CandleSignal("Evening Star", "bearish", i, ts, 8.0, ctx))

        return out

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _pin_bar_score(self, o: float, h: float, l: float, c: float) -> float:
        """Score 1–10 based on wick-to-body ratio."""
        body      = max(abs(c - o), 1e-8)
        bar_range = h - l
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        dominant   = max(lower_wick, upper_wick)
        ratio      = dominant / body
        return float(min(10.0, ratio * 1.5))

    def _engulf_score(
        self,
        o: float, h: float, l: float, c: float,
        po: float, ph: float, pl: float, pc: float,
    ) -> float:
        """Score 1–10: how much does the current body exceed the previous body."""
        curr_body = abs(c - o)
        prev_body = max(abs(pc - po), 1e-8)
        ratio     = curr_body / prev_body
        return float(min(10.0, 3.0 + ratio * 2.0))

    def _context_score(self, price: float, levels: List[float], atr: float) -> float:
        """
        Returns 0–1 indicating how close price is to a key level.
        Full score (1.0) if within 0.5 ATR of a key level.
        """
        if not levels:
            return 0.5
        min_dist = min(abs(price - lvl) for lvl in levels)
        # Linear decay: full score at 0 ATR, zero at 2 ATR
        return float(max(0.0, 1.0 - min_dist / (2.0 * atr + 1e-8)))

    def _calc_atr(self, df: pd.DataFrame) -> float:
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        trs    = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            for i in range(1, len(df))
        ]
        period = min(self.atr_period, len(trs))
        return float(np.mean(trs[-period:])) if trs else 1.0

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _build_narrative(self, r: CandlestickResult) -> str:
        if not r.top_signals:
            return "No significant candlestick signals detected."
        top = r.top_signals[0]
        bias = "bullish" if r.bullish_count > r.bearish_count else "bearish"
        return (
            f"The strongest candlestick signal is a {top.direction} {top.pattern} "
            f"(score {top.final_score:.1f}/10) at bar index {top.bar_index}. "
            f"Overall candlestick sentiment leans {bias} "
            f"({r.bullish_count} bullish vs {r.bearish_count} bearish patterns)."
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_ts(df: pd.DataFrame, idx: int) -> Optional[pd.Timestamp]:
    try:
        return df.index[idx]
    except Exception:
        return None
