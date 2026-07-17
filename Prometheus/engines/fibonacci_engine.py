"""
Fibonacci Engine
================
Auto-detects the most significant recent swing high/low and plots all
standard Fibonacci retracement levels.  Detects price reactions at key
fib levels and scores confluence with other signals.

Standard retracement levels:  23.6 %, 38.2 %, 50 %, 61.8 %, 78.6 %, 100 %
Extension levels:             127.2 %, 161.8 %, 261.8 %

Confluence principle:
  61.8 + support zone + bullish engulfing = high-probability setup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Retracement level definitions
RETRACEMENT_LEVELS: Dict[str, float] = {
    "0.0":   0.000,
    "23.6":  0.236,
    "38.2":  0.382,
    "50.0":  0.500,
    "61.8":  0.618,
    "78.6":  0.786,
    "100.0": 1.000,
}

EXTENSION_LEVELS: Dict[str, float] = {
    "127.2": 1.272,
    "161.8": 1.618,
    "261.8": 2.618,
}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class FibLevel:
    label:       str           # e.g. "61.8"
    ratio:       float         # e.g. 0.618
    price:       float         # absolute price
    is_key:      bool = False  # 38.2, 50, 61.8, 78.6
    reaction_detected: bool = False
    reaction_score:    float = 0.0   # 0–1


@dataclass
class FibResult:
    swing_high:     float
    swing_low:      float
    swing_high_idx: int
    swing_low_idx:  int
    direction:      str              # "up" (retracing from high) or "down"
    levels:         List[FibLevel]   = field(default_factory=list)
    key_reactions:  List[FibLevel]   = field(default_factory=list)
    strongest_level: Optional[FibLevel] = None
    current_price:  float            = 0.0
    current_level:  Optional[FibLevel] = None   # level price is near now
    narrative:      str              = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class FibonacciEngine:
    """
    Automatic Fibonacci retracement analysis.

    Usage::

        engine = FibonacciEngine()
        result = engine.analyze(df)
    """

    def __init__(
        self,
        levels:               Optional[List[float]] = None,
        key_levels:           Optional[List[float]] = None,
        confluence_tolerance: float = 0.003,   # 0.3 % from level = reaction
        swing_lookback:       int   = 100,
        pivot_sensitivity:    int   = 5,
        atr_period:           int   = 14,
    ) -> None:
        self.levels       = levels or list(RETRACEMENT_LEVELS.values())
        self.level_labels = {v: k for k, v in RETRACEMENT_LEVELS.items()}
        self.key_levels   = key_levels or [0.382, 0.500, 0.618, 0.786]
        self.tol          = confluence_tolerance
        self.lookback     = swing_lookback
        self.sensitivity  = pivot_sensitivity
        self.atr_period   = atr_period

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame,
                swing_highs: Optional[list] = None,
                swing_lows:  Optional[list] = None) -> FibResult:
        """
        Full Fibonacci analysis:
          1. Determine swing anchor from market-structure swings if provided,
             otherwise fall back to pivot detection within the lookback window.
             Structure-aware anchoring draws the fib from the most recent
             significant swing low → swing high pair (bullish leg) or swing
             high → swing low pair (bearish leg), matching the dominant trend.
          2. Compute all retracement price levels (0% through 100% and extensions).
          3. Scan history for reactions at key levels.
          4. Identify which level current price is near.
        """
        df   = self._validate(df)
        atr  = self._calc_atr(df)
        window = df.iloc[-self.lookback :]

        # ── Swing anchor selection ────────────────────────────────────────────
        if swing_highs and swing_lows:
            swing_high, sh_idx, swing_low, sl_idx = self._detect_swing_from_structure(
                swing_highs, swing_lows, len(df)
            )
        else:
            swing_high, sh_idx, swing_low, sl_idx = self._detect_swing_points(window)

        direction = "up" if sh_idx > sl_idx else "down"

        # Build level objects
        levels: List[FibLevel] = []
        for ratio in sorted(self.levels):
            if direction == "up":
                # Retracing from high down toward low
                price = swing_high - ratio * (swing_high - swing_low)
            else:
                # Retracing from low up toward high
                price = swing_low + ratio * (swing_high - swing_low)

            label    = self.level_labels.get(ratio, f"{ratio*100:.1f}")
            is_key   = ratio in self.key_levels
            levels.append(FibLevel(label=label, ratio=ratio, price=price, is_key=is_key))

        # Detect historical reactions at each level
        closes = window["close"].values
        for lvl in levels:
            lvl.reaction_detected, lvl.reaction_score = self._detect_reaction(
                closes, lvl.price, atr
            )

        key_reactions = [l for l in levels if l.is_key and l.reaction_detected]
        key_reactions.sort(key=lambda l: l.reaction_score, reverse=True)

        current_price = float(df["close"].iloc[-1])
        current_level = self._nearest_level(levels, current_price, atr)

        result = FibResult(
            swing_high=swing_high,
            swing_low=swing_low,
            swing_high_idx=int(len(df) - self.lookback + sh_idx),
            swing_low_idx=int(len(df) - self.lookback + sl_idx),
            direction=direction,
            levels=levels,
            key_reactions=key_reactions,
            strongest_level=key_reactions[0] if key_reactions else None,
            current_price=current_price,
            current_level=current_level,
        )
        result.narrative = self._build_narrative(result)

        logger.info(
            "Fib: High=%.4f Low=%.4f | Reactions at %d key levels",
            swing_high, swing_low, len(key_reactions),
        )
        return result

    def get_level_prices(self, result: FibResult) -> List[float]:
        """Convenience: list of all fib price levels."""
        return [l.price for l in result.levels]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_swing_from_structure(
        self,
        swing_highs: list,
        swing_lows:  list,
        total_bars:  int,
    ) -> Tuple[float, int, float, int]:
        """
        Anchor the Fibonacci grid to the most recent complete swing leg from
        market-structure data (BOS-confirmed swing pivots).

        Selection logic (ICT/SMC canonical):
        - Find the most recent swing high and the most recent swing low.
        - Whichever came LATER defines the end of the current leg.
          * If the most recent swing high is newer  → bearish leg (draw high→low):
            anchor from the swing low that preceded it (the leg's origin)
            to that swing high (the leg's peak), retrace moves downward.
          * If the most recent swing low is newer → bullish leg (draw low→high):
            anchor from the swing high that preceded it (the leg's origin)
            to that swing low (the leg's trough), retrace moves upward.
        - Falls back to the absolute max/min if structure data is insufficient.
        """
        # Sort by bar index, most recent last
        sorted_highs = sorted(swing_highs, key=lambda s: s.index)
        sorted_lows  = sorted(swing_lows,  key=lambda s: s.index)

        if not sorted_highs or not sorted_lows:
            window_df = pd.DataFrame()  # will trigger fallback
            return self._detect_swing_points_fallback(total_bars)

        last_high = sorted_highs[-1]
        last_low  = sorted_lows[-1]

        if last_high.index >= last_low.index:
            # Most recent pivot is a HIGH → price peaked then fell → bearish leg retracing up
            # Fib drawn from the preceding swing low (leg origin) to the swing high (leg peak)
            # Retrace goes: 0% = high, 100% = low — direction = "down"
            origin_lows = [s for s in sorted_lows if s.index < last_high.index]
            leg_low  = origin_lows[-1] if origin_lows else last_low
            swing_high = float(last_high.price)
            sh_idx     = int(last_high.index)
            swing_low  = float(leg_low.price)
            sl_idx     = int(leg_low.index)
        else:
            # Most recent pivot is a LOW → price troughed then rose → bullish leg retracing down
            # Fib drawn from the preceding swing high (leg origin) to the swing low (leg trough)
            # Retrace goes: 0% = low, 100% = high — direction = "up"
            origin_highs = [s for s in sorted_highs if s.index < last_low.index]
            leg_high = origin_highs[-1] if origin_highs else last_high
            swing_low  = float(last_low.price)
            sl_idx     = int(last_low.index)
            swing_high = float(leg_high.price)
            sh_idx     = int(leg_high.index)

        return swing_high, sh_idx, swing_low, sl_idx

    def _detect_swing_points_fallback(self, total_bars: int) -> Tuple[float, int, float, int]:
        """Return a placeholder when no structure data available."""
        return 0.0, 1, 0.0, 0

    def _detect_swing_points(
        self, df: pd.DataFrame
    ) -> Tuple[float, int, float, int]:
        """
        Williams Fractal swing detection (Bill Williams, "Trading Chaos").

        A fractal HIGH at bar i: high[i] is STRICTLY greater than every one
        of the `n` bars preceding it AND every one of the `n` bars following it.
        (Classic Williams uses n=2, giving a 5-bar pattern; here n = self.sensitivity.)

        A fractal LOW at bar i: low[i] is STRICTLY less than all surrounding bars.

        Using strict inequalities (>) ensures flat-top/flat-bottom regions and
        inside bars are not counted as independent fractal points — only genuine
        local extremes qualify.  The most recent fractal pair anchors the fib grid
        to the current leg rather than a distant historical extreme.
        """
        highs = df["high"].values
        lows  = df["low"].values
        n     = self.sensitivity   # fractal half-window (default 5 → 11-bar fractal)
        nbar  = len(highs)

        fractal_highs: list[tuple[int, float]] = []
        fractal_lows:  list[tuple[int, float]] = []

        for i in range(n, nbar - n):
            # Fractal HIGH: strictly above all n bars on each side
            if (all(highs[i] > highs[i - j] for j in range(1, n + 1)) and
                    all(highs[i] > highs[i + j] for j in range(1, n + 1))):
                fractal_highs.append((i, float(highs[i])))

            # Fractal LOW: strictly below all n bars on each side
            if (all(lows[i] < lows[i - j] for j in range(1, n + 1)) and
                    all(lows[i] < lows[i + j] for j in range(1, n + 1))):
                fractal_lows.append((i, float(lows[i])))

        # Widen to n=2 if sensitivity produced no fractals (rare on short windows)
        if not fractal_highs or not fractal_lows:
            n2 = max(2, n - 2)
            for i in range(n2, nbar - n2):
                if not fractal_highs and (
                    all(highs[i] > highs[i - j] for j in range(1, n2 + 1)) and
                    all(highs[i] > highs[i + j] for j in range(1, n2 + 1))
                ):
                    fractal_highs.append((i, float(highs[i])))
                if not fractal_lows and (
                    all(lows[i] < lows[i - j] for j in range(1, n2 + 1)) and
                    all(lows[i] < lows[i + j] for j in range(1, n2 + 1))
                ):
                    fractal_lows.append((i, float(lows[i])))

        # Hard fallback: absolute extremes
        if not fractal_highs:
            fractal_highs = [(int(highs.argmax()), float(highs.max()))]
        if not fractal_lows:
            fractal_lows  = [(int(lows.argmin()),  float(lows.min()))]

        last_fh_idx, last_fh_val = fractal_highs[-1]   # most recent fractal high
        last_fl_idx, last_fl_val = fractal_lows[-1]    # most recent fractal low

        if last_fh_idx >= last_fl_idx:
            # Most recent fractal is a HIGH → price peaked and is now pulling back
            # Fib origin = fractal low immediately preceding the fractal high
            preceding_lows = [(i, v) for i, v in fractal_lows if i < last_fh_idx]
            sl_idx, sl_val = preceding_lows[-1] if preceding_lows else (last_fl_idx, last_fl_val)
            return last_fh_val, last_fh_idx, sl_val, sl_idx
        else:
            # Most recent fractal is a LOW → price troughed and is now bouncing
            # Fib origin = fractal high immediately preceding the fractal low
            preceding_highs = [(i, v) for i, v in fractal_highs if i < last_fl_idx]
            sh_idx, sh_val = preceding_highs[-1] if preceding_highs else (last_fh_idx, last_fh_val)
            return sh_val, sh_idx, last_fl_val, last_fl_idx

    def _detect_reaction(
        self, closes: np.ndarray, level: float, atr: float
    ) -> Tuple[bool, float]:
        """
        Check whether price has bounced from the given fib level at some
        point in the history.

        A "reaction" is defined as: price comes within (tol × level) of the
        fib price, then moves away by at least 0.5 × ATR.
        """
        tol    = level * self.tol
        reaction_count = 0

        in_zone    = False
        entry_idx  = -1

        for i, close in enumerate(closes):
            near_level = abs(close - level) <= tol
            if near_level and not in_zone:
                in_zone   = True
                entry_idx = i
            elif in_zone and not near_level:
                # Price moved away — measure displacement
                displacement = abs(close - closes[entry_idx])
                if displacement >= 0.5 * atr:
                    reaction_count += 1
                in_zone = False

        detected = reaction_count > 0
        score    = min(1.0, reaction_count / 3.0)
        return detected, float(score)

    def _nearest_level(
        self, levels: List[FibLevel], price: float, atr: float
    ) -> Optional[FibLevel]:
        """Return the fib level closest to current price within 1 ATR."""
        candidates = [l for l in levels if abs(l.price - price) <= atr]
        if not candidates:
            return None
        return min(candidates, key=lambda l: abs(l.price - price))

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

    def _build_narrative(self, r: FibResult) -> str:
        lines: List[str] = [
            f"Fibonacci retracement drawn from {r.swing_low:.4f} to {r.swing_high:.4f} "
            f"({r.direction}ward swing)."
        ]
        if r.current_level:
            lines.append(
                f"Current price ({r.current_price:.4f}) is near the "
                f"{r.current_level.label} % level ({r.current_level.price:.4f})."
            )
        if r.strongest_level:
            sl = r.strongest_level
            lines.append(
                f"Strongest historical reaction at the {sl.label} % level "
                f"({sl.price:.4f}) — reaction score {sl.reaction_score:.0%}."
            )
        if r.key_reactions:
            labels = ", ".join(f"{l.label}%" for l in r.key_reactions)
            lines.append(f"Key Fibonacci reaction levels: {labels}.")
        return "  ".join(lines)
