"""
Support & Resistance Engine
============================
Builds confidence-scored S/R zones by clustering repeated price rejections.

Professional approach:
  - Identify swing highs/lows across the lookback window
  - Cluster nearby levels using ATR-based tolerance
  - Score each zone by touch count, wick rejection strength, and volume
  - Classify as support (price above) or resistance (price below)
  - Output a heatmap-style ranked list of zones
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
class SRZone:
    zone_type:       str             # "support" | "resistance" | "both"
    level:           float           # central price of the zone
    upper:           float           # zone upper bound
    lower:           float           # zone lower bound
    touches:         int   = 0
    confidence:      float = 0.0     # 0–1  composite score
    volume_weight:   float = 0.0
    wick_score:      float = 0.0
    last_touched_idx: int  = -1
    is_fresh:        bool  = True    # not tested since formation

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def label(self) -> str:
        # Backward-compatible alias used by legacy tests/UI.
        return self.zone_type

    @property
    def touch_count(self) -> int:
        # Backward-compatible alias used by legacy tests/UI.
        return self.touches

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


@dataclass
class SRResult:
    support_zones:    List[SRZone] = field(default_factory=list)
    resistance_zones: List[SRZone] = field(default_factory=list)
    all_zones:        List[SRZone] = field(default_factory=list)
    current_price:    float        = 0.0
    nearest_support:  Optional[SRZone] = None
    nearest_resistance: Optional[SRZone] = None
    narrative:        str          = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class SupportResistanceEngine:
    """
    Detects and scores support / resistance zones.

    Usage::

        engine = SupportResistanceEngine(zone_tolerance_atr=0.3, min_touches=2)
        result = engine.analyze(df)
    """

    def __init__(
        self,
        zone_tolerance_atr: float = 0.3,
        min_touches:        int   = 2,
        lookback_period:    int   = 200,
        volume_weight:      bool  = True,
        pivot_sensitivity:  int   = 5,
        atr_period:         int   = 14,
    ) -> None:
        self.tolerance_atr    = zone_tolerance_atr
        self.min_touches      = min_touches
        self.lookback         = lookback_period
        self.volume_weight    = volume_weight
        self.sensitivity      = pivot_sensitivity
        self.atr_period       = atr_period

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> SRResult:
        """Full S/R analysis on OHLCV data."""
        df      = self._validate(df)
        window  = df.iloc[-self.lookback :]
        atr     = self._calc_atr(window)
        tol     = atr * self.tolerance_atr

        # Collect candidate levels from swing highs / lows + round numbers
        candidates = self._gather_candidates(window)

        # Cluster nearby levels
        zones = self._cluster_zones(candidates, tol)

        # Score each zone
        zones = [self._score_zone(z, window, atr) for z in zones]

        # Filter by minimum touches
        zones = [z for z in zones if z.touches >= self.min_touches]

        # Sort by confidence descending
        zones.sort(key=lambda z: z.confidence, reverse=True)

        current_price = float(df["close"].iloc[-1])
        support_zones    = [z for z in zones if z.level <= current_price]
        resistance_zones = [z for z in zones if z.level >  current_price]

        result = SRResult(
            all_zones        = zones,
            support_zones    = support_zones,
            resistance_zones = resistance_zones,
            current_price    = current_price,
        )

        # Nearest S/R
        if support_zones:
            result.nearest_support = max(support_zones, key=lambda z: z.level)
        if resistance_zones:
            result.nearest_resistance = min(resistance_zones, key=lambda z: z.level)

        result.narrative = self._build_narrative(result)
        logger.info(
            "Detected %d support zones, %d resistance zones",
            len(support_zones), len(resistance_zones),
        )
        return result

    # ── Internal methods ──────────────────────────────────────────────────────

    def _gather_candidates(self, df: pd.DataFrame) -> List[Tuple[float, int, float]]:
        """
        Return list of (price, bar_index, volume) tuples representing
        candidate S/R levels (pivot highs, pivot lows).
        """
        highs   = df["high"].values
        lows    = df["low"].values
        vols    = df["volume"].values
        n       = self.sensitivity
        cands: List[Tuple[float, int, float]] = []

        for i in range(n, len(df) - n):
            w_h = highs[i - n : i + n + 1]
            if highs[i] == w_h.max() and int((w_h == highs[i]).sum()) == 1:
                cands.append((float(highs[i]), i, float(vols[i])))

            w_l = lows[i - n : i + n + 1]
            if lows[i] == w_l.min() and int((w_l == lows[i]).sum()) == 1:
                cands.append((float(lows[i]), i, float(vols[i])))

        return cands

    def _cluster_zones(
        self, candidates: List[Tuple[float, int, float]], tol: float
    ) -> List[SRZone]:
        """Merge nearby candidates into zones using single-link clustering."""
        if not candidates:
            return []

        # Sort by price
        sorted_cands = sorted(candidates, key=lambda c: c[0])
        clusters: List[List[Tuple[float, int, float]]] = [[sorted_cands[0]]]

        for cand in sorted_cands[1:]:
            if cand[0] - clusters[-1][-1][0] <= tol:
                clusters[-1].append(cand)
            else:
                clusters.append([cand])

        zones: List[SRZone] = []
        for cluster in clusters:
            prices = [c[0] for c in cluster]
            level  = float(np.mean(prices))
            upper  = max(prices) + tol * 0.5
            lower  = min(prices) - tol * 0.5
            last_i = max(c[1] for c in cluster)
            zones.append(
                SRZone(
                    zone_type="both",
                    level=level,
                    upper=upper,
                    lower=lower,
                    last_touched_idx=last_i,
                )
            )

        return zones

    def _score_zone(self, zone: SRZone, df: pd.DataFrame, atr: float) -> SRZone:
        """
        Compute composite confidence score:
          - touch count (normalised)
          - wick rejection quality
          - volume at touches
          - freshness (not revisited recently)
        """
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        opens   = df["open"].values
        vols    = df["volume"].values
        mean_v  = float(np.mean(vols)) if vols.mean() > 0 else 1.0

        touch_count  = 0
        wick_scores: List[float] = []
        vol_scores:  List[float] = []

        for i in range(len(df)):
            bar_range = highs[i] - lows[i]
            if bar_range == 0:
                continue

            touched = zone.lower <= highs[i] and zone.upper >= lows[i]
            if not touched:
                continue

            touch_count += 1

            # Wick analysis: measure rejection relative to bar range
            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]

            if highs[i] > zone.upper:          # tested from below (resistance)
                wick_scores.append(upper_wick / bar_range)
            elif lows[i] < zone.lower:         # tested from above (support)
                wick_scores.append(lower_wick / bar_range)
            else:
                wick_scores.append(0.1)

            # Volume score
            vol_scores.append(min(2.0, vols[i] / mean_v))

        zone.touches      = touch_count
        zone.wick_score   = float(np.mean(wick_scores)) if wick_scores else 0.0
        zone.volume_weight = float(np.mean(vol_scores)) if vol_scores else 1.0

        # Freshness: was the zone tested in the last 10 bars?
        zone.is_fresh = zone.last_touched_idx < len(df) - 10

        # Composite confidence (0–1)
        touch_score   = min(1.0, touch_count / 6.0)
        vol_component = min(1.0, zone.volume_weight / 2.0) if self.volume_weight else 0.5
        fresh_bonus   = 0.1 if zone.is_fresh else 0.0

        zone.confidence = float(
            0.45 * touch_score
            + 0.30 * zone.wick_score
            + 0.20 * vol_component
            + 0.05 * fresh_bonus
        )

        # Set zone classification relative to current close
        current_close = float(closes[-1])
        if zone.level > current_close:
            zone.zone_type = "resistance"
        elif zone.level < current_close:
            zone.zone_type = "support"

        return zone

    def _calc_atr(self, df: pd.DataFrame) -> float:
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        trs    = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(1, len(df))
        ]
        period = min(self.atr_period, len(trs))
        return float(np.mean(trs[-period:])) if trs else 1.0

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        return df

    def _build_narrative(self, r: SRResult) -> str:
        parts: List[str] = []
        if r.nearest_support:
            ns = r.nearest_support
            parts.append(
                f"Nearest support zone is centred at {ns.level:.4f} "
                f"({ns.touches} touches, confidence {ns.confidence:.0%})."
            )
        if r.nearest_resistance:
            nr = r.nearest_resistance
            parts.append(
                f"Nearest resistance zone is centred at {nr.level:.4f} "
                f"({nr.touches} touches, confidence {nr.confidence:.0%})."
            )
        if not parts:
            return "No significant support or resistance zones identified."
        parts.append(
            f"Current price {r.current_price:.4f} is trading between "
            f"{len(r.support_zones)} support and {len(r.resistance_zones)} resistance zones."
        )
        return "  ".join(parts)
