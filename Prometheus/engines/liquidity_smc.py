"""
Liquidity & Smart Money Concepts (SMC) Engine
===============================================
Implements institutional / Smart Money analysis concepts:

  - Equal Highs / Equal Lows (liquidity resting above/below)
  - Liquidity Pools (clusters of stop orders)
  - Stop Hunts / Liquidity Sweeps (price spikes beyond level then reversal)
  - Order Blocks (OB) — last opposing candle before a displacement move
  - Fair Value Gaps (FVG) — imbalances between candle wicks
  - Premium / Discount Zones relative to the current range
  - Inducement detection — minor liquidity grab before major move

Professional context:
  Smart money accumulates / distributes by sweeping retail stop-levels before
  the real move.  All concepts are derived from price action only — no
  proprietary data is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class LiquidityPool:
    direction:  str      # "buy_side" (above highs) | "sell_side" (below lows)
    price:      float    # central liquidity price
    strength:   float    # 0–1 probability this pool is targeted
    bar_idx:    int
    swept:      bool = False   # has the pool already been cleared?


@dataclass
class OrderBlock:
    direction:     str       # "bullish" OB | "bearish" OB
    high:          float
    low:           float
    open_:         float
    close:         float
    bar_idx:       int
    mitigated:     bool = False   # price has re-entered the block
    strength:      float = 0.0


@dataclass
class FairValueGap:
    direction:  str       # "bullish" (gap up) | "bearish" (gap down)
    high:       float     # top of the gap
    low:        float     # bottom of the gap
    mid:        float     # midpoint price
    start_idx:  int
    filled:     bool = False


@dataclass
class StopHunt:
    direction:   str       # "bullish_sweep" | "bearish_sweep"
    sweep_price: float     # the level that was swept
    reversal_idx: int
    bar_idx:     int


@dataclass
class SMCResult:
    liquidity_pools: List[LiquidityPool] = field(default_factory=list)
    order_blocks:    List[OrderBlock]    = field(default_factory=list)
    fair_value_gaps: List[FairValueGap]  = field(default_factory=list)
    stop_hunts:      List[StopHunt]      = field(default_factory=list)
    premium_zone:    Optional[Tuple[float, float]] = None   # (low, high)
    discount_zone:   Optional[Tuple[float, float]] = None
    equilibrium:     Optional[float] = None
    narrative:       str = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class LiquiditySMCEngine:
    """
    Full Smart Money Concepts analysis.

    Usage::

        engine = LiquiditySMCEngine()
        result = engine.analyze(df)
    """

    def __init__(
        self,
        equal_hl_tolerance_pct: float = 0.002,
        fvg_min_atr_mult:       float = 0.3,
        ob_lookback:            int   = 50,
        atr_period:             int   = 14,
    ) -> None:
        self.eq_tol       = equal_hl_tolerance_pct
        self.fvg_min_mult = fvg_min_atr_mult
        self.ob_lookback  = ob_lookback
        self.atr_period   = atr_period

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> SMCResult:
        """Full SMC analysis pipeline."""
        df  = self._validate(df)
        atr = self._calc_atr(df)

        result = SMCResult()
        result.liquidity_pools = self.detect_liquidity(df, atr)
        result.order_blocks    = self.detect_order_blocks(df, atr)
        result.fair_value_gaps = self.detect_fvg(df, atr)
        result.stop_hunts      = self.detect_stop_hunt(df, atr)
        result.premium_zone, result.discount_zone, result.equilibrium = \
            self._premium_discount_zones(df)

        result.narrative = self._build_narrative(result, df)
        logger.info(
            "SMC: %d pools | %d OBs | %d FVGs | %d stop-hunts",
            len(result.liquidity_pools),
            len(result.order_blocks),
            len(result.fair_value_gaps),
            len(result.stop_hunts),
        )
        return result

    def detect_liquidity(self, df: pd.DataFrame, atr: Optional[float] = None) -> List[LiquidityPool]:
        """
        Identify equal highs (buy-side liquidity) and equal lows (sell-side
        liquidity).  Two highs / lows within eq_tol % of each other form a pool.
        """
        if atr is None:
            atr = self._calc_atr(df)

        highs  = df["high"].values
        lows   = df["low"].values
        pools: List[LiquidityPool] = []

        # Equal highs → buy-side liquidity resting above
        for i in range(len(df)):
            for j in range(i + 3, min(i + 40, len(df))):
                diff = abs(highs[j] - highs[i]) / max(highs[i], 1e-8)
                if diff <= self.eq_tol:
                    price    = max(highs[i], highs[j])
                    strength = max(0.5, 1.0 - diff / self.eq_tol)
                    # Check if subsequently swept
                    swept = any(highs[k] > price * (1 + self.eq_tol) for k in range(j+1, len(df)))
                    pools.append(LiquidityPool(
                        direction="buy_side",
                        price=price,
                        strength=strength,
                        bar_idx=j,
                        swept=swept,
                    ))

        # Equal lows → sell-side liquidity resting below
        for i in range(len(df)):
            for j in range(i + 3, min(i + 40, len(df))):
                diff = abs(lows[j] - lows[i]) / max(abs(lows[i]), 1e-8)
                if diff <= self.eq_tol:
                    price    = min(lows[i], lows[j])
                    strength = max(0.5, 1.0 - diff / self.eq_tol)
                    swept = any(lows[k] < price * (1 - self.eq_tol) for k in range(j+1, len(df)))
                    pools.append(LiquidityPool(
                        direction="sell_side",
                        price=price,
                        strength=strength,
                        bar_idx=j,
                        swept=swept,
                    ))

        # Keep most significant pools (top 10 by strength, non-swept preferred)
        pools.sort(key=lambda p: (not p.swept, p.strength), reverse=True)
        return pools[:20]

    def detect_order_blocks(self, df: pd.DataFrame, atr: Optional[float] = None) -> List[OrderBlock]:
        """
        An Order Block is the last opposing candle BEFORE a strong displacement move.

        Bullish OB: last bearish candle before a strong bullish impulse.
        Bearish OB: last bullish candle before a strong bearish impulse.

        Displacement threshold: body of the displacement candle ≥ 1.5 × ATR.
        """
        if atr is None:
            atr = self._calc_atr(df)

        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        blocks: List[OrderBlock] = []

        displacement_threshold = 0.8 * atr   # 0.8× ATR — realistic for higher TFs like 4H/D1
        window = min(self.ob_lookback, len(df))

        for i in range(2, window):
            body = abs(closes[i] - opens[i])
            if body < displacement_threshold:
                continue

            if closes[i] > opens[i]:          # bullish displacement
                # Find last bearish candle before i
                for j in range(i - 1, max(0, i - 10), -1):
                    if closes[j] < opens[j]:  # bearish candle
                        # Check if mitigated (price re-enters the OB range)
                        ob_high = opens[j]
                        ob_low  = closes[j]
                        mitigated = any(
                            lows[k] <= ob_high and highs[k] >= ob_low
                            for k in range(i + 1, len(df))
                        )
                        strength  = min(1.0, body / (atr * 3.0))
                        blocks.append(OrderBlock(
                            direction="bullish",
                            high=ob_high,
                            low=ob_low,
                            open_=opens[j],
                            close=closes[j],
                            bar_idx=j,
                            mitigated=mitigated,
                            strength=strength,
                        ))
                        break

            elif closes[i] < opens[i]:        # bearish displacement
                for j in range(i - 1, max(0, i - 10), -1):
                    if closes[j] > opens[j]:  # bullish candle
                        ob_high = closes[j]
                        ob_low  = opens[j]
                        mitigated = any(
                            highs[k] >= ob_low and lows[k] <= ob_high
                            for k in range(i + 1, len(df))
                        )
                        strength = min(1.0, body / (atr * 3.0))
                        blocks.append(OrderBlock(
                            direction="bearish",
                            high=ob_high,
                            low=ob_low,
                            open_=opens[j],
                            close=closes[j],
                            bar_idx=j,
                            mitigated=mitigated,
                            strength=strength,
                        ))
                        break

        # Prefer fresh (un-mitigated) OBs, sorted by strength
        blocks.sort(key=lambda b: (not b.mitigated, b.strength), reverse=True)
        return blocks[:15]

    def detect_fvg(self, df: pd.DataFrame, atr: Optional[float] = None) -> List[FairValueGap]:
        """
        A Fair Value Gap (FVG) is a 3-candle imbalance:
          Bullish FVG: candle[i+2].low > candle[i].high   (gap left behind)
          Bearish FVG: candle[i+2].high < candle[i].low   (gap above)
        """
        if atr is None:
            atr = self._calc_atr(df)

        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        min_gap = atr * self.fvg_min_mult
        gaps: List[FairValueGap] = []

        for i in range(len(df) - 2):
            bullish_gap = lows[i + 2] - highs[i]
            bearish_gap = lows[i] - highs[i + 2]

            if bullish_gap >= min_gap:
                filled = any(
                    lows[k] <= highs[i] + bullish_gap / 2.0
                    for k in range(i + 3, len(df))
                )
                gaps.append(FairValueGap(
                    direction="bullish",
                    high=lows[i + 2],
                    low=highs[i],
                    mid=(highs[i] + lows[i + 2]) / 2.0,
                    start_idx=i,
                    filled=filled,
                ))

            if bearish_gap >= min_gap:
                filled = any(
                    highs[k] >= lows[i] - bearish_gap / 2.0
                    for k in range(i + 3, len(df))
                )
                gaps.append(FairValueGap(
                    direction="bearish",
                    high=lows[i],
                    low=highs[i + 2],
                    mid=(lows[i] + highs[i + 2]) / 2.0,
                    start_idx=i,
                    filled=filled,
                ))

        # Return unfilled FVGs (still tradeable), most recent first
        fresh = [g for g in gaps if not g.filled]
        fresh.sort(key=lambda g: g.start_idx, reverse=True)
        return fresh[:20]

    def detect_stop_hunt(self, df: pd.DataFrame, atr: Optional[float] = None) -> List[StopHunt]:
        """
        Stop Hunt / Liquidity Sweep:
          - Price briefly pierces a known swing level
          - Closes back on the other side within the same or next bar
          - Indicates engineered liquidity collection
        """
        if atr is None:
            atr = self._calc_atr(df)

        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        opens  = df["open"].values
        hunts: List[StopHunt] = []
        spike_threshold = 0.5 * atr

        for i in range(5, len(df)):
            # Recent swing high for the last 20 bars
            lookback_h = highs[max(0, i-20):i]
            lookback_l = lows[max(0, i-20):i]
            if len(lookback_h) == 0:
                continue
            recent_high = float(lookback_h.max())
            recent_low  = float(lookback_l.min())

            # Bullish sweep: wick above recent high, closes back below
            upper_wick = highs[i] - max(opens[i], closes[i])
            if (highs[i] > recent_high + spike_threshold
                    and closes[i] < recent_high          # closed back below
                    and upper_wick > atr * 0.3):
                hunts.append(StopHunt(
                    direction="bearish_sweep",
                    sweep_price=recent_high,
                    reversal_idx=i,
                    bar_idx=i,
                ))

            # Bearish sweep: wick below recent low, closes back above
            lower_wick = min(opens[i], closes[i]) - lows[i]
            if (lows[i] < recent_low - spike_threshold
                    and closes[i] > recent_low
                    and lower_wick > atr * 0.3):
                hunts.append(StopHunt(
                    direction="bullish_sweep",
                    sweep_price=recent_low,
                    reversal_idx=i,
                    bar_idx=i,
                ))

        return hunts[-10:]  # return 10 most recent

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _premium_discount_zones(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[float]]:
        """
        Compute premium (above 50 %) and discount (below 50 %) zones relative
        to the full high-low range of the lookback window.
        """
        highs = df["high"].values
        lows  = df["low"].values
        if len(highs) == 0:
            return None, None, None

        range_high = float(highs.max())
        range_low  = float(lows.min())
        mid        = (range_high + range_low) / 2.0

        premium  = (mid, range_high)
        discount = (range_low, mid)
        return premium, discount, mid

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

    def _build_narrative(self, r: SMCResult, df: pd.DataFrame) -> str:
        current = float(df["close"].iloc[-1])
        parts: List[str] = []

        # Premium / Discount
        if r.equilibrium:
            zone = "premium" if current > r.equilibrium else "discount"
            parts.append(f"Price is currently trading in a {zone} zone relative to the recent range.")

        # Order blocks
        fresh_obs = [ob for ob in r.order_blocks if not ob.mitigated]
        if fresh_obs:
            nearest = min(fresh_obs, key=lambda b: abs((b.high + b.low) / 2 - current))
            parts.append(
                f"Nearest unmitigated {nearest.direction} order block: "
                f"{nearest.low:.4f} – {nearest.high:.4f}."
            )

        # FVGs
        fresh_fvgs = [g for g in r.fair_value_gaps if not g.filled]
        if fresh_fvgs:
            nearest_fvg = min(fresh_fvgs, key=lambda g: abs(g.mid - current))
            parts.append(
                f"Open {nearest_fvg.direction} fair value gap at {nearest_fvg.low:.4f}–{nearest_fvg.high:.4f}."
            )

        # Stop hunts
        if r.stop_hunts:
            latest = r.stop_hunts[-1]
            parts.append(
                f"Recent {latest.direction} detected at {latest.sweep_price:.4f} — "
                "possible institutional accumulation / distribution."
            )

        # Liquidity
        buy_pools  = [p for p in r.liquidity_pools if not p.swept and p.direction == "buy_side"]
        sell_pools = [p for p in r.liquidity_pools if not p.swept and p.direction == "sell_side"]
        if buy_pools:
            nearest = min(buy_pools, key=lambda p: abs(p.price - current))
            parts.append(f"Buy-side liquidity resting above {nearest.price:.4f}.")
        if sell_pools:
            nearest = min(sell_pools, key=lambda p: abs(p.price - current))
            parts.append(f"Sell-side liquidity resting below {nearest.price:.4f}.")

        return "  ".join(parts) if parts else "No significant SMC structures detected."
