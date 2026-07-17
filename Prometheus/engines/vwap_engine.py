"""
VWAP Engine
===========
Computes Volume Weighted Average Price (VWAP) and its standard-deviation bands,
then scores the current bar's position relative to VWAP.

VWAP here is computed as a rolling/cumulative VWAP anchored to the start of
the supplied DataFrame (or, when pandas DatetimIndex is available, anchored to
the first bar of each trading day — "daily VWAP").

Outputs
-------
VWAPResult.vwap          – current VWAP value
VWAPResult.band1_upper   – VWAP + 1σ
VWAPResult.band1_lower   – VWAP − 1σ
VWAPResult.band2_upper   – VWAP + 2σ
VWAPResult.band2_lower   – VWAP − 2σ
VWAPResult.price         – last close
VWAPResult.signal        – "above" | "below" | "at"
VWAPResult.atr_distance  – |price − vwap| / ATR
VWAPResult.score         – 0–8  (used by ConfluenceScorer)
VWAPResult.note          – human-readable description
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_AT_THRESHOLD = 0.15   # within 0.15 × ATR → "at" VWAP (transition zone)


@dataclass
class VWAPResult:
    vwap:         float
    band1_upper:  float
    band1_lower:  float
    band2_upper:  float
    band2_lower:  float
    price:        float
    signal:       str    # "above" | "below" | "at"
    atr_distance: float  # |price − vwap| / ATR
    score:        float  # 0–12 (extended from 0–8 to account for band extremes)
    band_zone:    str    = "neutral"  # "extreme_high" | "upper_band" | "neutral" | "lower_band" | "extreme_low"
    note:         str    = ""


class VWAPEngine:
    """Computes VWAP, bands, and a directional score."""

    def __init__(self, atr_period: int = 14, band_mult: tuple = (1.0, 2.0)) -> None:
        self.atr_period = atr_period
        self.band_mult  = band_mult   # (1σ multiplier, 2σ multiplier)

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> Optional[VWAPResult]:
        """
        Compute VWAP analysis on *df*.

        Returns None if the DataFrame is too short or missing required columns.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning("VWAPEngine: missing columns %s", required - set(df.columns))
            return None
        if len(df) < max(self.atr_period + 1, 5):
            logger.warning("VWAPEngine: not enough bars (%d)", len(df))
            return None

        # ── VWAP (daily-anchored when DatetimIndex is available) ───────────────
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol     = df["volume"].replace(0, np.nan).ffill().fillna(1.0)

        if isinstance(df.index, pd.DatetimeIndex):
            # Anchor to the start of each calendar day
            df["_tp"]  = typical
            df["_vol"] = vol
            df["_date"] = df.index.normalize()
            df["_cum_tpv"] = df.groupby("_date").apply(
                lambda g: (g["_tp"] * g["_vol"]).cumsum()
            ).reset_index(level=0, drop=True)
            df["_cum_vol"] = df.groupby("_date").apply(
                lambda g: g["_vol"].cumsum()
            ).reset_index(level=0, drop=True)
        else:
            # Fallback: session-anchored VWAP (whole series)
            df["_cum_tpv"] = (typical * vol).cumsum()
            df["_cum_vol"] = vol.cumsum()

        vwap_series = df["_cum_tpv"] / df["_cum_vol"]

        # ── Standard-deviation bands ───────────────────────────────────────────
        # Rolling squared deviation from VWAP, volume-weighted
        sq_dev = ((typical - vwap_series) ** 2 * vol)

        if isinstance(df.index, pd.DatetimeIndex):
            cum_sq   = df.groupby(df.index.normalize()).apply(
                lambda g: sq_dev.loc[g.index].cumsum()
            ).reset_index(level=0, drop=True)
            cum_vol2 = df["_cum_vol"]
        else:
            cum_sq   = sq_dev.cumsum()
            cum_vol2 = df["_cum_vol"]

        std_series = np.sqrt(cum_sq / cum_vol2.replace(0, np.nan)).fillna(0)

        m1, m2 = self.band_mult
        vwap_val    = float(vwap_series.iloc[-1])
        std_val     = float(std_series.iloc[-1])
        b1u = vwap_val + m1 * std_val
        b1l = vwap_val - m1 * std_val
        b2u = vwap_val + m2 * std_val
        b2l = vwap_val - m2 * std_val

        # ── ATR ───────────────────────────────────────────────────────────────
        atr = float(
            (df["high"] - df["low"]).rolling(self.atr_period).mean().iloc[-1]
        )
        if atr == 0:
            atr = 1e-8

        price = float(df["close"].iloc[-1])
        dist  = abs(price - vwap_val)
        atr_dist = dist / atr

        # ── Band zone classification ───────────────────────────────────────────
        # Classify where price sits relative to the σ bands.
        # This is used by the confluence scorer to add a mean-reversion bonus
        # when price reaches the statistically extreme ±2σ band.
        if price >= b2u:
            band_zone = "extreme_high"   # price at or beyond +2σ — short extreme
        elif price >= b1u:
            band_zone = "upper_band"     # between +1σ and +2σ
        elif price <= b2l:
            band_zone = "extreme_low"    # price at or beyond −2σ — long extreme
        elif price <= b1l:
            band_zone = "lower_band"     # between −1σ and −2σ
        else:
            band_zone = "neutral"        # between ±1σ (fair-value zone)

        # ── Signal ────────────────────────────────────────────────────────────
        if atr_dist <= _AT_THRESHOLD:
            signal = "at"
        elif price > vwap_val:
            signal = "above"
        else:
            signal = "below"

        # ── Score (0–12) ─────────────────────────────────────────────────────
        # Base: 0–8 for directional VWAP position (same as before).
        # Band bonus: +4 when price is AT a ±2σ extreme (mean-reversion entry).
        # Being at an extreme is only a bonus when the trade is counter-directional
        # to that extreme — the confluence scorer handles direction gating.
        if signal == "at":
            base_score = 4.0
            note = f"Price at VWAP ({vwap_val:.4f}) — neutral zone."
        elif signal == "above":
            if band_zone == "extreme_high":
                # At +2sd: still bullish position but signals exhaustion
                base_score = 6.0
                note = (f"Price at VWAP +2sd ({b2u:.4f}) -- institutional exhaustion zone; "
                        "bearish mean-reversion setup.")
            elif band_zone == "upper_band":
                base_score = min(8.0, 4.0 + atr_dist * 1.5)
                note = (f"Price between VWAP +1sd/{b1u:.4f} and +2sd/{b2u:.4f} "
                        "-- bullish extension, watch for resistance.")
            else:
                base_score = min(8.0, 4.0 + atr_dist * 1.5)
                note = (f"Price {atr_dist:.1f}x ATR above VWAP ({vwap_val:.4f}) "
                        "-- bullish institutional bias.")
        else:  # below
            if band_zone == "extreme_low":
                base_score = 6.0
                note = (f"Price at VWAP -2sd ({b2l:.4f}) -- institutional exhaustion zone; "
                        "bullish mean-reversion setup.")
            elif band_zone == "lower_band":
                base_score = min(8.0, 4.0 + atr_dist * 1.5)
                note = (f"Price between VWAP -1sd/{b1l:.4f} and -2sd/{b2l:.4f} "
                        "-- bearish extension, watch for support.")
            else:
                base_score = min(8.0, 4.0 + atr_dist * 1.5)
                note = (f"Price {atr_dist:.1f}x ATR below VWAP ({vwap_val:.4f}) "
                        "-- bearish institutional bias.")

        # Band extreme bonus: +4 when price is at +/-2sd -- the confluence scorer
        # will only award this in the correct counter-directional context.
        band_bonus = 4.0 if band_zone in ("extreme_high", "extreme_low") else 0.0
        final_score = min(12.0, base_score + band_bonus)

        return VWAPResult(
            vwap=round(vwap_val, 4),
            band1_upper=round(b1u, 4),
            band1_lower=round(b1l, 4),
            band2_upper=round(b2u, 4),
            band2_lower=round(b2l, 4),
            price=round(price, 4),
            signal=signal,
            atr_distance=round(atr_dist, 3),
            score=round(final_score, 2),
            band_zone=band_zone,
            note=note,
        )
