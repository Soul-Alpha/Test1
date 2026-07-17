"""
Multi-Timeframe Analysis Engine
================================
Aggregates analysis results from multiple timeframes into a unified bias score.

Professional principle:
  - Higher timeframes establish the macro trend and key structural levels.
  - Lower timeframes provide precision entry context.
  - A trade should align with at least the two highest timeframes for highest probability.

Timeframe hierarchy (default): Daily → 4H → 1H → 15M
Weights reflect decreasing importance: 40 % / 30 % / 20 % / 10 %

When OHLCV data for multiple timeframes are supplied, the engine:
  1. Runs market structure analysis on each TF.
  2. Extracts a directional bias (-1 bearish, 0 neutral, +1 bullish).
  3. Computes a weighted alignment score.
  4. Generates a professional narrative of cross-TF confluence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engines.market_structure import MarketStructureEngine, MarketStructureResult, StructureType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class TimeframeBias:
    timeframe:  str
    bias:       str       # "bullish" | "bearish" | "sideways" | "unknown"
    score:      float     # -1.0 to +1.0
    weight:     float     # relative importance
    structure:  Optional[MarketStructureResult] = None
    note:       str = ""


@dataclass
class MTFResult:
    biases:           List[TimeframeBias] = field(default_factory=list)
    alignment_score:  float = 0.0      # -1 fully bearish, +1 fully bullish
    primary_bias:     str   = "sideways"
    confluence_level: str   = "low"    # "low" | "medium" | "high"
    narrative:        str   = ""


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class MultiTimeframeEngine:
    """
    Multi-timeframe analysis aggregator.

    Usage::

        engine = MultiTimeframeEngine()
        # Supply a dict {timeframe: DataFrame}
        result = engine.analyze({"1d": df_daily, "4h": df_4h, "1h": df_1h, "15m": df_15m})
    """

    DEFAULT_TIMEFRAMES = ["1d", "4h", "1h", "15m"]
    DEFAULT_WEIGHTS    = [0.40,  0.30,  0.20,  0.10]

    def __init__(
        self,
        timeframes: Optional[List[str]]  = None,
        weights:    Optional[List[float]] = None,
        pivot_sensitivity: int = 5,
    ) -> None:
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        raw_weights     = weights    or self.DEFAULT_WEIGHTS
        total           = sum(raw_weights)
        self.weights    = [w / total for w in raw_weights]
        self.ms_engine  = MarketStructureEngine(pivot_sensitivity=pivot_sensitivity)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, tf_data: Dict[str, pd.DataFrame]) -> MTFResult:
        """
        Args:
            tf_data: mapping of timeframe string → OHLCV DataFrame.
                     Missing timeframes are skipped gracefully.

        Returns:
            MTFResult with per-TF biases and aggregate alignment.
        """
        biases: List[TimeframeBias] = []

        # Normalize input keys to lowercase for case-insensitive matching
        tf_data_lower = {k.lower(): v for k, v in tf_data.items()}

        for tf, weight in zip(self.timeframes, self.weights):
            df = tf_data_lower.get(tf.lower())
            if df is None or len(df) < 20:
                logger.warning("No data for timeframe %s — skipping", tf)
                biases.append(TimeframeBias(
                    timeframe=tf, bias="unknown", score=0.0, weight=weight,
                    note="No data available."
                ))
                continue

            try:
                result = self.ms_engine.analyze(df)
                bias, score = self._structure_to_bias(result)
                biases.append(TimeframeBias(
                    timeframe=tf,
                    bias=bias,
                    score=score,
                    weight=weight,
                    structure=result,
                    note=result.narrative[:120],
                ))
            except Exception as exc:
                logger.error("Error analyzing TF %s: %s", tf, exc)
                biases.append(TimeframeBias(
                    timeframe=tf, bias="unknown", score=0.0, weight=weight,
                    note=f"Analysis error: {exc}"
                ))

        # Weighted alignment score
        valid     = [b for b in biases if b.bias != "unknown"]
        # Guard: if only 1 (or 0) timeframes have data we have no real
        # multi-timeframe information — return neutral so the confluence
        # scorer doesn't double-count the primary timeframe's own bias.
        if len(valid) < 2:
            primary = "sideways"
            alignment = 0.0
            conf = "low"
        else:
            alignment = (
                sum(b.score * b.weight for b in valid) / sum(b.weight for b in valid)
            )
            primary = self._score_to_bias(alignment)
            conf    = self._confluence_level(biases)

        result = MTFResult(
            biases=biases,
            alignment_score=round(alignment, 3),
            primary_bias=primary,
            confluence_level=conf,
        )
        result.narrative = self._build_narrative(result)
        logger.info(
            "MTF alignment: %.2f (%s) | Confluence: %s",
            alignment, primary, conf,
        )
        return result

    def resample_higher_tf(
        self, df: pd.DataFrame, source_tf: str, target_tf: str
    ) -> pd.DataFrame:
        """
        Resample a lower-timeframe DataFrame to a higher timeframe.

        Args:
            df:        OHLCV DataFrame with DatetimeIndex
            source_tf: e.g. "15m"
            target_tf: e.g. "1h"

        Returns:
            Resampled OHLCV DataFrame
        """
        period_map = {
            "1m": "1T", "3m": "3T", "5m": "5T", "15m": "15T",
            "30m": "30T", "1h": "1H", "2h": "2H", "4h": "4H",
            "6h": "6H", "12h": "12H", "1d": "1D", "1w": "1W",
        }
        rule = period_map.get(target_tf)
        if rule is None:
            raise ValueError(f"Unsupported target timeframe: {target_tf}")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        resampled = df.resample(rule).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()
        return resampled

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _structure_to_bias(
        self, ms: MarketStructureResult
    ) -> Tuple[str, float]:
        """Convert a market structure result to a named bias and scalar score."""
        if ms.structure_type == StructureType.BULLISH:
            return "bullish",  ms.trend_strength
        elif ms.structure_type == StructureType.BEARISH:
            return "bearish", -ms.trend_strength
        else:
            return "sideways", 0.0

    def _score_to_bias(self, score: float) -> str:
        if score > 0.25:
            return "bullish"
        elif score < -0.25:
            return "bearish"
        return "sideways"

    def _confluence_level(self, biases: List[TimeframeBias]) -> str:
        valid  = [b for b in biases if b.bias != "unknown"]
        if len(valid) < 2:
            return "low"
        directions   = [b.bias for b in valid if b.bias in ("bullish", "bearish")]
        if not directions:
            return "low"
        most_common  = max(set(directions), key=directions.count)
        aligned_frac = directions.count(most_common) / len(directions)
        if aligned_frac >= 0.85:
            return "high"
        elif aligned_frac >= 0.60:
            return "medium"
        return "low"

    def _build_narrative(self, r: MTFResult) -> str:
        lines: List[str] = [
            f"Multi-timeframe bias: {r.primary_bias.upper()} "
            f"(alignment score {r.alignment_score:+.2f} / 1.00, "
            f"{r.confluence_level} confluence)."
        ]
        for b in r.biases:
            icon = {"bullish": "↑", "bearish": "↓", "sideways": "→"}.get(b.bias, "?")
            lines.append(f"  {b.timeframe.upper():>4s}: {icon} {b.bias.capitalize()} ({b.note[:80]})")

        if r.confluence_level == "high":
            lines.append(
                "All timeframes are aligned → highest-probability directional environment."
            )
        elif r.confluence_level == "medium":
            lines.append(
                "Partial cross-timeframe alignment → exercise additional confirmation before entry."
            )
        else:
            lines.append(
                "Timeframes are conflicted.  Wait for higher-timeframe bias to clarify."
            )

        return "\n".join(lines)
