"""
Chart Pattern Engine
=====================
Geometric detection of classical chart patterns:
  - Double Top / Double Bottom
  - Head and Shoulders (H&S) / Inverse H&S
  - Ascending / Descending / Symmetrical Triangle
  - Rising / Falling Wedge
  - Bull / Bear Flag
  - Pennant
  - Channel (up / down / horizontal)

Approach:
  - Work entirely from swing highs / lows
  - Use slope calculations and trendline fitting
  - Validate geometric constraints before confirming
  - Score confidence 0–1 based on pattern quality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from engines.market_structure import Swing, SwingType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class ChartPattern:
    name:          str
    direction:     str              # "bullish" | "bearish" | "neutral"
    confidence:    float            # 0–1
    start_idx:     int
    end_idx:       int
    key_prices:    List[float]      = field(default_factory=list)
    target_price:  Optional[float]  = None
    invalidation:  Optional[float]  = None
    description:   str              = ""


@dataclass
class ChartPatternResult:
    patterns:       List[ChartPattern] = field(default_factory=list)
    top_pattern:    Optional[ChartPattern] = None
    bullish_count:  int = 0
    bearish_count:  int = 0
    narrative:      str = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class ChartPatternEngine:
    """
    Scans swing arrays for classical chart patterns.

    Usage::

        engine = ChartPatternEngine()
        result = engine.analyze(df, swing_highs, swing_lows)
    """

    def __init__(
        self,
        double_top_pct:    float = 0.015,
        triangle_tolerance: float = 0.02,
        lookback_swings:   int   = 10,
        min_pattern_bars:  int   = 10,
    ) -> None:
        self.double_top_pct    = double_top_pct
        self.tri_tol           = triangle_tolerance
        self.lookback_swings   = lookback_swings
        self.min_bars          = min_pattern_bars

    def analyze(
        self,
        df:          pd.DataFrame,
        swing_highs: List[Swing],
        swing_lows:  List[Swing],
    ) -> ChartPatternResult:
        """Detect all patterns and return ranked results."""
        patterns: List[ChartPattern] = []

        sh = sorted(swing_highs, key=lambda s: s.index)[-self.lookback_swings:]
        sl = sorted(swing_lows,  key=lambda s: s.index)[-self.lookback_swings:]

        patterns += self._detect_double_top_bottom(sh, sl)
        patterns += self._detect_head_and_shoulders(sh, sl)
        patterns += self._detect_triangles(sh, sl)
        patterns += self._detect_wedges(sh, sl)
        patterns += self._detect_flags_pennants(df, sh, sl)
        patterns += self._detect_channel(sh, sl)

        patterns.sort(key=lambda p: p.confidence, reverse=True)

        result = ChartPatternResult(
            patterns=patterns,
            top_pattern=patterns[0] if patterns else None,
            bullish_count=sum(1 for p in patterns if p.direction == "bullish"),
            bearish_count=sum(1 for p in patterns if p.direction == "bearish"),
        )
        result.narrative = self._build_narrative(result)
        logger.info("Detected %d chart patterns", len(patterns))
        return result

    # ── Pattern detectors ─────────────────────────────────────────────────────

    def _detect_double_top_bottom(
        self, highs: List[Swing], lows: List[Swing]
    ) -> List[ChartPattern]:
        """
        Double Top: two swing highs at similar price, with a valley between.
        Double Bottom: two swing lows at similar price, with a peak between.
        """
        patterns: List[ChartPattern] = []

        # Double Top
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                h1, h2 = highs[i], highs[j]
                price_diff = abs(h1.price - h2.price) / max(h1.price, h2.price)
                if price_diff <= self.double_top_pct and (h2.index - h1.index) >= self.min_bars:
                    # Valley must exist between them in lows
                    valley_lows = [l for l in lows if h1.index < l.index < h2.index]
                    if valley_lows:
                        valley = min(valley_lows, key=lambda l: l.price)
                        top    = max(h1.price, h2.price)
                        height = top - valley.price
                        target = valley.price - height   # measured move down
                        conf   = 1.0 - price_diff / self.double_top_pct
                        patterns.append(ChartPattern(
                            name="Double Top",
                            direction="bearish",
                            confidence=round(conf * 0.85, 3),
                            start_idx=h1.index,
                            end_idx=h2.index,
                            key_prices=[h1.price, valley.price, h2.price],
                            target_price=target,
                            invalidation=top * 1.005,
                            description=(
                                f"Double Top at {top:.4f}. Neckline at {valley.price:.4f}. "
                                f"Target on break: {target:.4f}."
                            ),
                        ))

        # Double Bottom
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                l1, l2 = lows[i], lows[j]
                price_diff = abs(l1.price - l2.price) / max(abs(l1.price), 1e-8)
                if price_diff <= self.double_top_pct and (l2.index - l1.index) >= self.min_bars:
                    peak_highs = [h for h in highs if l1.index < h.index < l2.index]
                    if peak_highs:
                        peak    = max(peak_highs, key=lambda h: h.price)
                        bottom  = min(l1.price, l2.price)
                        height  = peak.price - bottom
                        target  = peak.price + height
                        conf    = 1.0 - price_diff / self.double_top_pct
                        patterns.append(ChartPattern(
                            name="Double Bottom",
                            direction="bullish",
                            confidence=round(conf * 0.85, 3),
                            start_idx=l1.index,
                            end_idx=l2.index,
                            key_prices=[l1.price, peak.price, l2.price],
                            target_price=target,
                            invalidation=bottom * 0.995,
                            description=(
                                f"Double Bottom at {bottom:.4f}. Neckline at {peak.price:.4f}. "
                                f"Target on break: {target:.4f}."
                            ),
                        ))
        return patterns

    def _detect_head_and_shoulders(
        self, highs: List[Swing], lows: List[Swing]
    ) -> List[ChartPattern]:
        """Requires at least 3 swing highs."""
        patterns: List[ChartPattern] = []
        if len(highs) < 3:
            return patterns

        for i in range(len(highs) - 2):
            ls, head, rs = highs[i], highs[i + 1], highs[i + 2]
            # Head must be higher than both shoulders
            if not (head.price > ls.price and head.price > rs.price):
                continue
            # Shoulders roughly equal
            shoulder_diff = abs(ls.price - rs.price) / max(ls.price, 1e-8)
            if shoulder_diff > 0.03:       # within 3 %
                continue

            # Find neckline (two troughs between shoulders)
            t1_lows = [l for l in lows if ls.index < l.index < head.index]
            t2_lows = [l for l in lows if head.index < l.index < rs.index]
            if not t1_lows or not t2_lows:
                continue

            t1 = min(t1_lows, key=lambda l: l.price)
            t2 = min(t2_lows, key=lambda l: l.price)
            neckline  = (t1.price + t2.price) / 2.0
            height    = head.price - neckline
            target    = neckline - height
            conf      = max(0.5, 0.9 - shoulder_diff * 10)

            patterns.append(ChartPattern(
                name="Head and Shoulders",
                direction="bearish",
                confidence=round(conf, 3),
                start_idx=ls.index,
                end_idx=rs.index,
                key_prices=[ls.price, head.price, rs.price, neckline],
                target_price=target,
                invalidation=head.price * 1.005,
                description=(
                    f"H&S top with head at {head.price:.4f}, neckline at {neckline:.4f}. "
                    f"Target: {target:.4f}."
                ),
            ))

        # Inverse H&S (on lows)
        if len(lows) >= 3:
            for i in range(len(lows) - 2):
                ls, head, rs = lows[i], lows[i + 1], lows[i + 2]
                if not (head.price < ls.price and head.price < rs.price):
                    continue
                shoulder_diff = abs(ls.price - rs.price) / max(abs(ls.price), 1e-8)
                if shoulder_diff > 0.03:
                    continue
                t1_highs = [h for h in highs if ls.index < h.index < head.index]
                t2_highs = [h for h in highs if head.index < h.index < rs.index]
                if not t1_highs or not t2_highs:
                    continue
                t1 = max(t1_highs, key=lambda h: h.price)
                t2 = max(t2_highs, key=lambda h: h.price)
                neckline = (t1.price + t2.price) / 2.0
                height   = neckline - head.price
                target   = neckline + height
                conf     = max(0.5, 0.9 - shoulder_diff * 10)
                patterns.append(ChartPattern(
                    name="Inverse Head and Shoulders",
                    direction="bullish",
                    confidence=round(conf, 3),
                    start_idx=ls.index,
                    end_idx=rs.index,
                    key_prices=[ls.price, head.price, rs.price, neckline],
                    target_price=target,
                    invalidation=head.price * 0.995,
                    description=(
                        f"Inverse H&S with head at {head.price:.4f}, neckline at {neckline:.4f}. "
                        f"Target: {target:.4f}."
                    ),
                ))

        return patterns

    def _detect_triangles(
        self, highs: List[Swing], lows: List[Swing]
    ) -> List[ChartPattern]:
        """Detect ascending, descending, and symmetrical triangles."""
        patterns: List[ChartPattern] = []
        if len(highs) < 3 or len(lows) < 3:
            return patterns

        recent_h = highs[-5:]
        recent_l = lows[-5:]

        high_slope = _linear_slope([s.index for s in recent_h], [s.price for s in recent_h])
        low_slope  = _linear_slope([s.index for s in recent_l], [s.price for s in recent_l])

        start = min(recent_h[0].index, recent_l[0].index)
        end   = max(recent_h[-1].index, recent_l[-1].index)
        avg_price = (recent_h[-1].price + recent_l[-1].price) / 2.0

        if high_slope is None or low_slope is None:
            return patterns

        flat_tol = self.tri_tol / 5.0   # slope near-zero threshold

        if abs(high_slope) < flat_tol and low_slope > flat_tol:
            # Ascending triangle → bullish
            target = recent_h[-1].price + (recent_h[-1].price - recent_l[-1].price)
            patterns.append(ChartPattern(
                name="Ascending Triangle",
                direction="bullish",
                confidence=0.72,
                start_idx=start,
                end_idx=end,
                key_prices=[recent_h[-1].price, recent_l[-1].price],
                target_price=target,
                invalidation=recent_l[-1].price * 0.998,
                description=f"Ascending triangle with flat resistance at {recent_h[-1].price:.4f}.",
            ))

        elif abs(low_slope) < flat_tol and high_slope < -flat_tol:
            # Descending triangle → bearish
            target = recent_l[-1].price - (recent_h[-1].price - recent_l[-1].price)
            patterns.append(ChartPattern(
                name="Descending Triangle",
                direction="bearish",
                confidence=0.72,
                start_idx=start,
                end_idx=end,
                key_prices=[recent_h[-1].price, recent_l[-1].price],
                target_price=target,
                invalidation=recent_h[-1].price * 1.002,
                description=f"Descending triangle with flat support at {recent_l[-1].price:.4f}.",
            ))

        elif high_slope < -flat_tol and low_slope > flat_tol:
            # Symmetrical triangle → neutral, break in either direction
            patterns.append(ChartPattern(
                name="Symmetrical Triangle",
                direction="neutral",
                confidence=0.65,
                start_idx=start,
                end_idx=end,
                key_prices=[recent_h[-1].price, recent_l[-1].price],
                description="Symmetrical triangle compression. Await directional break.",
            ))

        return patterns

    def _detect_wedges(
        self, highs: List[Swing], lows: List[Swing]
    ) -> List[ChartPattern]:
        """Rising wedge (bearish) and falling wedge (bullish)."""
        patterns: List[ChartPattern] = []
        if len(highs) < 3 or len(lows) < 3:
            return patterns

        recent_h = highs[-5:]
        recent_l = lows[-5:]

        high_slope = _linear_slope([s.index for s in recent_h], [s.price for s in recent_h])
        low_slope  = _linear_slope([s.index for s in recent_l], [s.price for s in recent_l])

        if high_slope is None or low_slope is None:
            return patterns

        start = min(recent_h[0].index, recent_l[0].index)
        end   = max(recent_h[-1].index, recent_l[-1].index)

        both_up   = high_slope > 0 and low_slope > 0
        both_down = high_slope < 0 and low_slope < 0
        converging_up   = both_up   and high_slope < low_slope   # lows rising faster
        converging_down = both_down and high_slope > low_slope   # highs falling faster

        if converging_up:
            patterns.append(ChartPattern(
                name="Rising Wedge",
                direction="bearish",
                confidence=0.68,
                start_idx=start,
                end_idx=end,
                key_prices=[recent_h[-1].price, recent_l[-1].price],
                description="Rising wedge — bearish reversal pattern as ranges converge upward.",
            ))
        elif converging_down:
            patterns.append(ChartPattern(
                name="Falling Wedge",
                direction="bullish",
                confidence=0.68,
                start_idx=start,
                end_idx=end,
                key_prices=[recent_h[-1].price, recent_l[-1].price],
                description="Falling wedge — bullish reversal pattern as ranges converge downward.",
            ))

        return patterns

    def _detect_flags_pennants(
        self, df: pd.DataFrame, highs: List[Swing], lows: List[Swing]
    ) -> List[ChartPattern]:
        """
        Flag / Bull Flag: sharp impulse followed by tight counter-trend channel.
        Pennant: impulse followed by symmetrical triangle mini-consolidation.
        """
        patterns: List[ChartPattern] = []
        if len(df) < 20:
            return patterns

        closes = df["close"].values
        # Measure last 20-bar momentum
        impulse_bars = min(10, len(closes) - 1)
        impulse      = closes[-1] - closes[-(impulse_bars + 1)]
        atr_approx   = float(np.std(np.diff(closes[-20:])))
        if atr_approx == 0:
            return patterns

        impulse_strength = abs(impulse) / (atr_approx * impulse_bars)

        if impulse_strength > 1.5:      # strong impulse
            direction = "bullish" if impulse > 0 else "bearish"
            # Check if last 5 bars are tighter range (flag body)
            flag_range = max(closes[-5:]) - min(closes[-5:])
            flag_is_tight = flag_range < atr_approx * 3.0
            name = "Bull Flag" if direction == "bullish" else "Bear Flag"
            if flag_is_tight:
                patterns.append(ChartPattern(
                    name=name,
                    direction=direction,
                    confidence=0.70,
                    start_idx=len(closes) - impulse_bars - 1,
                    end_idx=len(closes) - 1,
                    key_prices=[closes[-(impulse_bars + 1)], closes[-1]],
                    description=f"{name} continuation setup following a strong impulse move.",
                ))

        return patterns

    def _detect_channel(
        self, highs: List[Swing], lows: List[Swing]
    ) -> List[ChartPattern]:
        """Identify parallel price channels."""
        patterns: List[ChartPattern] = []
        if len(highs) < 3 or len(lows) < 3:
            return patterns

        rh = highs[-min(6, len(highs)):]
        rl = lows[-min(6, len(lows)):]

        high_slope = _linear_slope([s.index for s in rh], [s.price for s in rh])
        low_slope  = _linear_slope([s.index for s in rl], [s.price for s in rl])

        if high_slope is None or low_slope is None:
            return patterns

        # Parallel within 20 %
        if abs(high_slope) < 1e-8:
            return patterns
        parallelism = abs((high_slope - low_slope) / high_slope)
        if parallelism > 0.20:
            return patterns

        start = min(rh[0].index, rl[0].index)
        end   = max(rh[-1].index, rl[-1].index)

        if high_slope > 0:
            name, direction = "Ascending Channel", "bullish"
        elif high_slope < 0:
            name, direction = "Descending Channel", "bearish"
        else:
            name, direction = "Horizontal Channel", "neutral"

        patterns.append(ChartPattern(
            name=name,
            direction=direction,
            confidence=0.60,
            start_idx=start,
            end_idx=end,
            key_prices=[rh[-1].price, rl[-1].price],
            description=(
                f"{name} with upper bound ~{rh[-1].price:.4f} "
                f"and lower bound ~{rl[-1].price:.4f}."
            ),
        ))
        return patterns

    # ─────────────────────────────────────────
    def _build_narrative(self, r: ChartPatternResult) -> str:
        if not r.patterns:
            return "No classical chart patterns detected."
        tp = r.top_pattern
        return (
            f"Primary pattern: {tp.name} ({tp.direction}, confidence {tp.confidence:.0%}). "
            f"{tp.description}  "
            f"Total: {r.bullish_count} bullish and {r.bearish_count} bearish patterns identified."
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _linear_slope(
    x: List[int], y: List[float]
) -> Optional[float]:
    """Least-squares slope of the series."""
    if len(x) < 2:
        return None
    xv, yv = np.array(x, dtype=float), np.array(y, dtype=float)
    if xv.std() == 0:
        return None
    slope = float(np.polyfit(xv, yv, 1)[0])
    return slope
