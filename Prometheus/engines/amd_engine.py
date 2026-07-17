"""
AMD Cycle Engine  (ICT — Accumulation · Manipulation · Distribution)
=====================================================================
Detects the three-phase intraday cycle used by institutional traders:

  ACCUMULATION  – Asian session (00:00–07:00 UTC)
      Price consolidates in a tight range.  Smart money builds positions.
      The high and low of this range become the key liquidity levels watched
      for the next phase.

  MANIPULATION  – London open (07:00–09:30 UTC)
      Price sweeps ABOVE the Asian high (hunting buy-stops) before reversing
      bearish, OR sweeps BELOW the Asian low (hunting sell-stops) before
      reversing bullish.  This is the "stop hunt" / inducement candle.

  DISTRIBUTION  – NY session / rest of day (09:30–17:00 UTC)
      Price delivers in the *opposite* direction of the manipulation sweep,
      targeting premium/discount zones.  FVGs created during or just after
      the manipulation are ideal low-risk entry points inside the distribution.

Key outputs
-----------
AMDResult.phase          – current phase ("accumulation"|"manipulation"|"distribution"|"unknown")
AMDResult.direction      – expected distribution direction ("bullish"|"bearish"|"neutral")
AMDResult.asian_high     – top of accumulation range
AMDResult.asian_low      – bottom of accumulation range
AMDResult.manipulation_swept – True when a sweep of Asian range confirmed
AMDResult.sweep_side     – "high" | "low" (which side was swept)
AMDResult.entry_fvgs     – FVGs formed AFTER sweep in distribution direction (ideal entries)
AMDResult.confidence     – 0.0–1.0  (used by confluence scorer)
AMDResult.note           – human-readable narrative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from engines.liquidity_smc import FairValueGap

logger = logging.getLogger(__name__)

# ── Session boundaries (UTC hours, inclusive lower bound exclusive upper) ──────
_ASIAN_START    =  0   # 00:00 UTC
_ASIAN_END      =  7   # 07:00 UTC
_MANIP_START    =  7   # London open
_MANIP_END      = 10   # 10:00 UTC — manipulation typically done
_DIST_START     =  9   # NY pre-market bleed-in (some days earlier)
_DIST_END       = 17   # 17:00 UTC

# Sweep threshold: price must close BEYOND Asian range by ≥ this fraction of
# the Asian range to be counted as a genuine manipulation sweep (not noise).
_SWEEP_CLOSE_BUFFER = 0.20   # 20% of Asian range


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AMDResult:
    phase:               str   = "unknown"   # accumulation|manipulation|distribution|unknown
    direction:           str   = "neutral"   # bullish|bearish|neutral
    asian_high:          Optional[float] = None
    asian_low:           Optional[float] = None
    asian_range:         Optional[float] = None   # high - low
    manipulation_swept:  bool  = False
    sweep_side:          str   = ""          # "high" | "low"
    sweep_price:         Optional[float] = None   # the price that was swept
    entry_fvgs:          List[FairValueGap] = field(default_factory=list)
    best_entry_fvg:      Optional[FairValueGap] = None
    confidence:          float = 0.0         # 0.0–1.0
    note:                str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class AMDEngine:
    """
    Detects the ICT AMD (Accumulation / Manipulation / Distribution) cycle
    from intraday OHLCV data.

    The engine requires a datetime-indexed DataFrame so it can split bars by
    UTC session.  If the index is not timezone-aware it is assumed UTC.

    Usage::

        engine = AMDEngine()
        result = engine.analyze(df, existing_fvgs=smc_result.fair_value_gaps)
    """

    def __init__(
        self,
        asian_start_hour:  int   = _ASIAN_START,
        asian_end_hour:    int   = _ASIAN_END,
        manip_end_hour:    int   = _MANIP_END,
        dist_end_hour:     int   = _DIST_END,
        sweep_buffer_pct:  float = _SWEEP_CLOSE_BUFFER,
        lookback_days:     int   = 1,   # how many past daily AMD cycles to use
    ) -> None:
        self.asian_start_hour = asian_start_hour
        self.asian_end_hour   = asian_end_hour
        self.manip_end_hour   = manip_end_hour
        self.dist_end_hour    = dist_end_hour
        self.sweep_buffer_pct = sweep_buffer_pct
        self.lookback_days    = lookback_days

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(
        self,
        df:            pd.DataFrame,
        existing_fvgs: Optional[List[FairValueGap]] = None,
    ) -> AMDResult:
        """
        Run AMD cycle detection.

        Args:
            df:            OHLCV DataFrame with DatetimeIndex (UTC assumed).
            existing_fvgs: FVGs already detected by LiquiditySMCEngine.
                           If supplied the engine filters for distribution-side
                           FVGs to surface as entry candidates.

        Returns:
            AMDResult
        """
        result = AMDResult()

        if df is None or len(df) < 10:
            result.note = "Insufficient data for AMD analysis."
            return result

        # ── Ensure UTC-aware datetime index ───────────────────────────────────
        df = self._ensure_utc_index(df)
        if df is None:
            result.note = "Non-datetime index — AMD requires time-series data."
            return result

        # ── Determine current UTC hour to label phase ─────────────────────────
        last_ts     = df.index[-1]
        current_hour = int(last_ts.hour)

        # ── Build Asian session range from the MOST RECENT completed session ──
        asian_hi, asian_lo, asian_bars = self._asian_range(df)
        if asian_hi is None:
            result.note = "No Asian session bars found in data window."
            result.phase = "unknown"
            return result

        asian_range = asian_hi - asian_lo
        result.asian_high  = round(asian_hi,  5)
        result.asian_low   = round(asian_lo,  5)
        result.asian_range = round(asian_range, 5)

        # ── Phase label ───────────────────────────────────────────────────────
        if self.asian_start_hour <= current_hour < self.asian_end_hour:
            result.phase = "accumulation"
        elif self.asian_end_hour <= current_hour < self.manip_end_hour:
            result.phase = "manipulation"
        elif self.manip_end_hour <= current_hour < self.dist_end_hour:
            result.phase = "distribution"
        else:
            result.phase = "unknown"

        # ── Detect manipulation sweep ─────────────────────────────────────────
        # Look at bars from London open to manip_end for a sweep of asian range
        manip_bars = df[
            (df.index.hour >= self.asian_end_hour) &
            (df.index.hour <  self.manip_end_hour)
        ]

        min_sweep   = asian_range * self.sweep_buffer_pct
        sweep_side  = None
        sweep_price = None

        if not manip_bars.empty:
            manip_hi = manip_bars["high"].max()
            manip_lo = manip_bars["low"].min()

            if manip_hi > asian_hi + min_sweep:
                sweep_side  = "high"
                sweep_price = manip_hi
                # Direction after a high sweep → bearish distribution (price collapses after trapping longs)
                result.direction           = "bearish"
                result.manipulation_swept  = True
            elif manip_lo < asian_lo - min_sweep:
                sweep_side  = "low"
                sweep_price = manip_lo
                # Direction after a low sweep → bullish distribution (price rallies after trapping shorts)
                result.direction           = "bullish"
                result.manipulation_swept  = True

            if sweep_side:
                result.sweep_side  = sweep_side
                result.sweep_price = round(sweep_price, 5)

        # ── Find distribution-side FVGs as entry candidates ───────────────────
        if result.manipulation_swept and existing_fvgs:
            # Determine the bar index at which the sweep / manipulation ended
            if not manip_bars.empty:
                sweep_end_idx = df.index.get_loc(manip_bars.index[-1])
            else:
                sweep_end_idx = 0

            dist_fvgs = []
            for fvg in existing_fvgs:
                if fvg.filled:
                    continue
                # FVG must be in the distribution direction
                if fvg.direction != result.direction:
                    continue
                # FVG must have formed at or after the manipulation window
                if fvg.start_idx < sweep_end_idx:
                    continue
                dist_fvgs.append(fvg)

            # Sort by recency (most recent first)
            dist_fvgs.sort(key=lambda g: g.start_idx, reverse=True)
            result.entry_fvgs = dist_fvgs

            if dist_fvgs:
                # Best entry = most recent unfilled FVG in distribution direction
                result.best_entry_fvg = dist_fvgs[0]

        # ── Confidence score ──────────────────────────────────────────────────
        result.confidence = self._confidence(result, asian_range)

        # ── Narrative ─────────────────────────────────────────────────────────
        result.note = self._narrative(result, current_hour)

        logger.info(
            "AMD | phase=%s dir=%s sweep=%s conf=%.0f%% FVG-entries=%d",
            result.phase, result.direction,
            f"{result.sweep_side} @ {result.sweep_price}" if result.manipulation_swept else "none",
            result.confidence * 100,
            len(result.entry_fvgs),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_utc_index(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Return df with DatetimeIndex in UTC, or None if not possible."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        if df.index.tzinfo is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.copy()
            df.index = df.index.tz_convert("UTC")
        return df

    def _asian_range(
        self, df: pd.DataFrame
    ) -> tuple[Optional[float], Optional[float], pd.DataFrame]:
        """
        Extract the high and low of the most recent completed Asian session.
        Falls back to last available Asian bars if today's Asian session has
        not yet started.
        """
        # Grab bars in Asian hours across the entire window, then take the
        # last full calendar day's worth (or last N bars if only partial).
        asian_mask = (
            (df.index.hour >= self.asian_start_hour) &
            (df.index.hour <  self.asian_end_hour)
        )
        asian_df = df[asian_mask]

        if asian_df.empty:
            return None, None, asian_df

        # Use only the most recent day's Asian session
        last_asian_date = asian_df.index[-1].date()
        today_asian     = asian_df[asian_df.index.date == last_asian_date]

        if today_asian.empty:
            today_asian = asian_df.tail(20)   # fallback: last 20 Asian bars

        hi = float(today_asian["high"].max())
        lo = float(today_asian["low"].min())
        return hi, lo, today_asian

    def _confidence(self, r: AMDResult, asian_range: float) -> float:
        """
        Compute a 0–1 confidence score for the AMD reading.

        Factors:
          - Asian range established (always true here)              +0.20
          - Manipulation sweep confirmed                            +0.35
          - Sweep magnitude ≥ 1.5× buffer threshold                +0.10
          - Distribution-side FVG found                            +0.25
          - Best FVG is recent (within last 20 bars)               +0.10
        """
        conf = 0.20   # baseline — we have an Asian range

        if not r.manipulation_swept:
            return round(conf, 2)

        conf += 0.35   # sweep confirmed

        # Bonus for sweep size
        if r.sweep_price is not None and r.asian_high and r.asian_low:
            swept_by = (abs(r.sweep_price - r.asian_high)
                        if r.sweep_side == "high"
                        else abs(r.asian_low - r.sweep_price))
            if swept_by >= asian_range * self.sweep_buffer_pct * 1.5:
                conf += 0.10

        if r.entry_fvgs:
            conf += 0.25

        if r.best_entry_fvg:
            conf += 0.10   # FVG depth bonus (already covered above as entry_fvgs bonus split)

        return round(min(1.0, conf), 2)

    def _narrative(self, r: AMDResult, current_hour: int) -> str:
        parts = []

        if r.asian_high is not None:
            parts.append(
                f"Asian range: {r.asian_low:.4f} – {r.asian_high:.4f} "
                f"({r.asian_range:.1f} pts)."
            )

        if not r.manipulation_swept:
            if r.phase == "accumulation":
                parts.append("Price still in accumulation (Asian session). Awaiting manipulation sweep.")
            elif r.phase == "manipulation":
                parts.append(
                    "London open active — watching for sweep of Asian "
                    f"high ({r.asian_high:.4f}) or low ({r.asian_low:.4f})."
                )
            else:
                parts.append("No manipulation sweep detected — AMD cycle unclear.")
        else:
            sweep_msg = (
                f"{'Bearish' if r.direction == 'bearish' else 'Bullish'} manipulation sweep of "
                f"Asian {'high' if r.sweep_side == 'high' else 'low'} "
                f"@ {r.sweep_price:.4f} confirmed."
            )
            parts.append(sweep_msg)
            parts.append(
                f"Expected distribution direction: {r.direction.upper()}."
            )

            if r.entry_fvgs:
                fvg = r.best_entry_fvg
                parts.append(
                    f"{len(r.entry_fvgs)} distribution FVG(s) identified. "
                    f"Best entry FVG: {fvg.low:.4f}–{fvg.high:.4f} "
                    f"(mid {fvg.mid:.4f}) — wait for price to retrace into this gap."
                )
            else:
                parts.append(
                    "No distribution-side FVG yet — wait for price to create an "
                    f"imbalance after the {'high' if r.sweep_side == 'high' else 'low'} sweep."
                )

        return "  ".join(parts)
