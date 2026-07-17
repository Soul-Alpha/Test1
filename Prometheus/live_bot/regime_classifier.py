"""
Regime Classifier for Prometheus Trading Bot
=============================================
Classifies the current market regime using price action, ATR, Bollinger Bands,
candle body/wick ratios, and structure.

Regimes:
  TREND_EXPANSION      – strong directional momentum, expanding ATR
  TREND_EXHAUSTION     – momentum fading, large wicks, ATR plateauing
  MEAN_REVERSION       – oscillating between S/R, compressed ATR
  COMPRESSION          – Bollinger Band squeeze, low ATR, coiling
  VOLATILITY_EXPANSION – sudden ATR spike (news / spike)
  LIQUIDITY_SWEEP      – long wick beyond structure, fast reversal
  NEWS_VOLATILITY      – extreme impulsive spike from news event; suppress entries
  DEAD_LIQUIDITY       – ultra-low ATR, rollover or weekend; no valid setups
  UNKNOWN              – insufficient data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    TREND_EXPANSION      = "trend_expansion"
    TREND_EXHAUSTION     = "trend_exhaustion"
    MEAN_REVERSION       = "mean_reversion"
    COMPRESSION          = "compression"
    VOLATILITY_EXPANSION = "volatility_expansion"
    LIQUIDITY_SWEEP      = "liquidity_sweep"
    NEWS_VOLATILITY      = "news_volatility"
    DEAD_LIQUIDITY       = "dead_liquidity"
    UNKNOWN              = "unknown"


@dataclass
class RegimeState:
    regime: Regime = Regime.UNKNOWN
    confidence: float = 0.0          # 0-1
    # Risk scalars: multiply your base lot/risk by this
    lot_scalar: float = 1.0          # <1 = reduce size, >1 = increase
    # Break-even ATR multiplier override
    be_atr_mult: float = 0.25        # default mirrors BE_ATR_TRIGGER
    # Whether counter-trend scalps are allowed in this regime
    allow_countertrend: bool = False
    # Human readable description
    description: str = ""
    # per-regime suggested TP multiplier (relative to base RR)
    tp_scalar: float = 1.0
    # Hard block for new entries — regime is too dangerous to trade
    kill_entries: bool = False
    # Minimum additional score required above system threshold in this regime
    score_floor_premium: float = 0.0


# ---------------------------------------------------------------------------
# Regime parameter table – drives how bot adapts to each regime
# ---------------------------------------------------------------------------
_REGIME_PARAMS: dict[Regime, dict] = {
    Regime.TREND_EXPANSION: {
        "lot_scalar": 1.0,
        "be_atr_mult": 0.20,   # tight — lock profit early as it runs
        "allow_countertrend": False,
        "tp_scalar": 1.2,
        "kill_entries": False,
        "score_floor_premium": 0.0,
        "description": "Strong directional impulse – ride the wave",
    },
    Regime.TREND_EXHAUSTION: {
        "lot_scalar": 0.6,
        "be_atr_mult": 0.40,   # wider — give room for wicks before reversal
        "allow_countertrend": True,
        "tp_scalar": 0.8,
        "kill_entries": False,
        "score_floor_premium": 5.0,
        "description": "Momentum fading – tighten TP, allow counter scalps",
    },
    Regime.MEAN_REVERSION: {
        "lot_scalar": 0.30,
        "be_atr_mult": 0.35,
        "allow_countertrend": True,
        "tp_scalar": 0.7,
        "kill_entries": False,             # reopened: improved engines (fractals, VWAP bands, ATR rank)
        "score_floor_premium": 15.0,       # requires score >= 80 (base 65 + 15 premium)
        "description": "Range-bound -- reduced lot (0.30x), high-confidence only (score>=80)",
    },
    Regime.COMPRESSION: {
        "lot_scalar": 0.0,
        "be_atr_mult": 0.50,
        "allow_countertrend": False,
        "tp_scalar": 0.6,
        "kill_entries": True,              # learning: 0W/8L (-276 pnl) — hard block
        "score_floor_premium": 0.0,
        "description": "Squeeze – 0W/8L historically, all new entries suppressed",
    },
    Regime.VOLATILITY_EXPANSION: {
        "lot_scalar": 0.4,
        "be_atr_mult": 0.50,
        "allow_countertrend": False,
        "tp_scalar": 1.0,
        "kill_entries": False,
        "score_floor_premium": 5.0,
        "description": "Post-spike expansion – very small size, wait for settle",
    },
    Regime.LIQUIDITY_SWEEP: {
        "lot_scalar": 0.8,
        "be_atr_mult": 0.30,
        "allow_countertrend": True,
        "tp_scalar": 0.9,
        "kill_entries": False,
        "score_floor_premium": 0.0,
        "description": "Liquidity grab – potential reversal play",
    },
    Regime.NEWS_VOLATILITY: {
        # Extreme impulsive spike from news: spreads blow out, price
        # moves 3-8× normal ATR in seconds.  Do NOT enter new trades.
        # Let open positions ride with extra ATR buffer.
        "lot_scalar": 0.2,
        "be_atr_mult": 0.60,
        "allow_countertrend": False,
        "tp_scalar": 1.5,
        "kill_entries": True,
        "score_floor_premium": 20.0,
        "description": "News volatility – extreme spike; suppress all new entries",
    },
    Regime.DEAD_LIQUIDITY: {
        # Ultra-low ATR / rollover / weekend session.  Price moves are
        # meaningless; spreads relative to move are enormous.
        "lot_scalar": 0.3,
        "be_atr_mult": 0.40,
        "allow_countertrend": False,
        "tp_scalar": 0.5,
        "kill_entries": True,
        "score_floor_premium": 15.0,
        "description": "Dead liquidity – rollover/quiet period; no valid setups",
    },
    Regime.UNKNOWN: {
        "lot_scalar": 0.8,
        "be_atr_mult": 0.25,
        "allow_countertrend": False,
        "tp_scalar": 1.0,
        "kill_entries": False,
        "score_floor_premium": 0.0,
        "description": "Insufficient data",
    },
}


class RegimeClassifier:
    """
    Classifies market regime from a OHLCV DataFrame.

    Usage::
        clf = RegimeClassifier()
        state = clf.classify(df)
        logger.info("Regime: %s (conf=%.2f) lot×%.2f", state.regime, state.confidence, state.lot_scalar)
    """

    def __init__(
        self,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        lookback: int = 5,
    ) -> None:
        self.atr_period = atr_period
        self.bb_period  = bb_period
        self.bb_std     = bb_std
        self.lookback   = lookback   # candles to look back for regime assessment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, df: pd.DataFrame) -> RegimeState:
        """Classify the current regime from the last `lookback` candles."""
        try:
            return self._classify(df)
        except Exception as exc:
            logger.warning("[regime] Classification error: %s", exc)
            return self._make_state(Regime.UNKNOWN, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, df: pd.DataFrame) -> RegimeState:
        if df is None or len(df) < self.atr_period + self.bb_period:
            return self._make_state(Regime.UNKNOWN, 0.0)

        df = df.copy()
        self._add_indicators(df)

        recent = df.tail(self.lookback + 1).iloc[:-1]  # exclude current forming bar
        if len(recent) < 2:
            return self._make_state(Regime.UNKNOWN, 0.0)

        # -- Signal computations -----------------------------------------
        atr_now   = float(df["atr"].iloc[-2])
        atr_5ago  = float(df["atr"].iloc[-min(7, len(df) - 1)])
        atr_ratio = atr_now / atr_5ago if atr_5ago > 0 else 1.0

        # Long-baseline ATR (20 bars) for extreme-spike detection
        atr_20ago = float(df["atr"].iloc[-min(22, len(df) - 1)])
        atr_vs_baseline = atr_now / atr_20ago if atr_20ago > 0 else 1.0

        bb_width_now  = float(df["bb_width"].iloc[-2])
        bb_width_5ago = float(df["bb_width"].iloc[-min(7, len(df) - 1)])
        bb_ratio      = bb_width_now / bb_width_5ago if bb_width_5ago > 0 else 1.0

        # EWM-weighted body-to-range ratio (recent bars get higher weight)
        candle_ranges = (recent["high"] - recent["low"] + 1e-9)
        bodies_raw    = (recent["close"] - recent["open"]).abs() / candle_ranges
        weights       = np.exp(np.linspace(0, 1, len(bodies_raw)))  # exponential growth
        weights      /= weights.sum()
        avg_body_ratio = float((bodies_raw * weights).sum())

        # EWM-weighted upper + lower wick ratios
        up_wicks = (recent["high"] - recent[["close", "open"]].max(axis=1)) / candle_ranges
        dn_wicks = (recent[["close", "open"]].min(axis=1) - recent["low"]) / candle_ranges
        avg_wick = float(((up_wicks + dn_wicks) * weights).sum())

        # Directional consistency (fraction of candles bullish)
        n_bull = float((recent["close"] > recent["open"]).sum())
        dir_consistency = abs(n_bull / len(recent) - 0.5) * 2  # 0=split, 1=all same

        # Bollinger band position of close (>1 = outside band)
        close_now = float(df["close"].iloc[-2])
        bb_upper  = float(df["bb_upper"].iloc[-2])
        bb_lower  = float(df["bb_lower"].iloc[-2])
        bb_mid    = float(df["bb_mid"].iloc[-2])
        bb_pos    = (close_now - bb_lower) / max(bb_upper - bb_lower, 1e-9)

        # Last two completed bars
        last_bar = df.iloc[-2]
        prev_bar = df.iloc[-3] if len(df) > 3 else last_bar
        wick_range = last_bar["high"] - last_bar["low"]
        upper_wick = last_bar["high"] - max(last_bar["close"], last_bar["open"])
        lower_wick = min(last_bar["close"], last_bar["open"]) - last_bar["low"]
        max_wick   = max(upper_wick, lower_wick)

        # ── NEWS_VOLATILITY: classify FIRST (highest priority) ─────────────
        # Criteria: ATR spiked ≥ 2.5× 20-bar baseline AND ATR now > 1.5× 5-bar
        # This distinguishes a genuine news blowout from normal expansion
        is_news_spike = (atr_vs_baseline >= 2.5) and (atr_ratio >= 1.5)
        if is_news_spike:
            confidence = min(1.0, (atr_vs_baseline - 2.5) / 2.0 + 0.5)
            return self._make_state(Regime.NEWS_VOLATILITY, round(confidence, 3))

        # ── DEAD_LIQUIDITY: second highest priority ─────────────────────────
        # Criteria: ATR is in the bottom 10% of recent history (ultra-compressed)
        # AND Bollinger bands are extremely tight AND no directional conviction
        is_dead = (atr_vs_baseline <= 0.50) and (bb_ratio <= 0.70) and (dir_consistency < 0.25)
        if is_dead:
            confidence = min(1.0, (0.50 - atr_vs_baseline) / 0.30 * 0.8 + 0.2)
            return self._make_state(Regime.DEAD_LIQUIDITY, round(confidence, 3))

        # ── LIQUIDITY_SWEEP: long dominant wick that reversed direction ─────
        # Improved: require reversal confirmation (prev bar opposite direction)
        prev_bullish    = prev_bar["close"] > prev_bar["open"]
        last_bullish    = last_bar["close"] > last_bar["open"]
        reversal_exists = (prev_bullish != last_bullish)  # direction changed
        liq_sweep = (
            wick_range > 0
            and (max_wick / wick_range > 0.60)
            and (atr_ratio > 0.95)       # not in dead liquidity
            and reversal_exists           # reversal confirmation
        )

        # -- Scoring matrix -----------------------------------------------
        scores: dict[Regime, float] = {r: 0.0 for r in Regime
                                        if r not in (Regime.NEWS_VOLATILITY,
                                                     Regime.DEAD_LIQUIDITY,
                                                     Regime.UNKNOWN)}

        # TREND_EXPANSION: large bodies, consistent direction, ATR expanding
        scores[Regime.TREND_EXPANSION] = (
            (atr_ratio > 1.10) * 1.0
            + (avg_body_ratio > 0.60) * 1.0
            + (dir_consistency > 0.60) * 1.0
            + (bb_ratio > 1.05) * 0.5
        )

        # TREND_EXHAUSTION: decreasing body size, increasing wicks, ATR plateau
        scores[Regime.TREND_EXHAUSTION] = (
            (atr_ratio < 1.05) * 0.5
            + (avg_wick > 0.35) * 1.0
            + (avg_body_ratio < 0.40) * 1.0
            + (dir_consistency > 0.50) * 0.5   # still has direction but weakening
        )

        # MEAN_REVERSION: alternating candles, ATR stable, mid-band
        scores[Regime.MEAN_REVERSION] = (
            (dir_consistency < 0.30) * 1.5
            + (0.35 < bb_pos < 0.65) * 1.0     # hanging around mid-band
            + (0.85 < atr_ratio < 1.15) * 0.5
        )

        # COMPRESSION: BB squeeze, low ATR
        scores[Regime.COMPRESSION] = (
            (bb_ratio < 0.85) * 1.5
            + (atr_ratio < 0.90) * 1.0
            + (avg_body_ratio < 0.45) * 0.5
        )

        # VOLATILITY_EXPANSION: ATR spike but NOT extreme (already handled above)
        scores[Regime.VOLATILITY_EXPANSION] = (
            (1.40 <= atr_ratio < 2.0) * 2.0
            + (bb_pos > 1.0 or bb_pos < 0.0) * 1.0
        )

        # LIQUIDITY_SWEEP: long wick reversal bar with confirmation
        scores[Regime.LIQUIDITY_SWEEP] = 3.0 if liq_sweep else 0.0

        # -- Winner ---------------------------------------------------------
        best_regime = max(scores, key=lambda r: scores[r])
        best_score  = scores[best_regime]
        total_score = sum(scores.values()) or 1.0
        confidence  = min(best_score / total_score * 2.5, 1.0)  # normalise

        # Ensure minimum threshold – fall back to UNKNOWN if weak signal
        if best_score < 0.5:
            return self._make_state(Regime.UNKNOWN, confidence * 0.5)

        return self._make_state(best_regime, round(confidence, 3))

    def _add_indicators(self, df: pd.DataFrame) -> None:
        """Add ATR and Bollinger Bands columns in-place."""
        # ATR
        hi, lo, cl = df["high"], df["low"], df["close"]
        prev_cl    = cl.shift(1)
        tr = pd.concat([
            hi - lo,
            (hi - prev_cl).abs(),
            (lo - prev_cl).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # Bollinger Bands
        df["bb_mid"]   = df["close"].rolling(self.bb_period).mean()
        bb_std          = df["close"].rolling(self.bb_period).std()
        df["bb_upper"] = df["bb_mid"] + self.bb_std * bb_std
        df["bb_lower"] = df["bb_mid"] - self.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)

    @staticmethod
    def _make_state(regime: Regime, confidence: float) -> RegimeState:
        params = _REGIME_PARAMS[regime]
        return RegimeState(
            regime             = regime,
            confidence         = confidence,
            lot_scalar         = params["lot_scalar"],
            be_atr_mult        = params["be_atr_mult"],
            allow_countertrend = params["allow_countertrend"],
            tp_scalar          = params["tp_scalar"],
            kill_entries       = params["kill_entries"],
            score_floor_premium= params["score_floor_premium"],
            description        = params["description"],
        )


# ---------------------------------------------------------------------------
# HTF alignment → regime-aware confidence score
# ---------------------------------------------------------------------------

def htf_alignment_score(htf_biases: list, signal_bias: str) -> Tuple[float, float]:
    """
    Compute an alignment ratio and a weighted confidence multiplier from
    multi-timeframe bias list.

    Returns:
        (alignment_ratio, weight_multiplier)
        alignment_ratio:  0.0 = all opposed, 1.0 = fully aligned
        weight_multiplier: lot/score scale to apply (0.0 = block, 0.5-1.0 = allow)

    Key design decision:
        A SINGLE opposing HTF → reduce lot but never hard-block.
        Only block when 2+ HTFs ALL oppose the signal (consensus disagreement).
    """
    if not htf_biases:
        return 1.0, 1.0

    n_htf          = len(htf_biases)
    total_weight   = sum(getattr(b, "weight", 1.0) for b in htf_biases)
    aligned_weight = sum(
        getattr(b, "weight", 1.0) for b in htf_biases
        if getattr(b, "bias", "") == signal_bias
    )
    ratio = aligned_weight / max(total_weight, 1e-9)

    # Single-HTF case: treat as uncertainty, not opposition
    # One higher TF disagreeing is common in multi-TF analysis and should not
    # hard-block trades — only reduce position size.
    if n_htf == 1 and ratio == 0.0:
        mult = 0.50   # trade at half lot: cautious but not blocked

    # Multi-HTF alignment scoring
    elif ratio >= 0.85:
        mult = 1.0      # fully aligned — full lot
    elif ratio >= 0.65:
        mult = 0.85     # mostly aligned — slight reduction
    elif ratio >= 0.50:
        mult = 0.70     # marginal — reduce more, still tradeable
    elif ratio >= 0.35:
        mult = 0.50     # weak alignment — half lot maximum
    else:
        # Multiple HTFs ALL opposing — genuine consensus disagreement → block
        mult = 0.0

    return round(ratio, 3), round(mult, 3)
