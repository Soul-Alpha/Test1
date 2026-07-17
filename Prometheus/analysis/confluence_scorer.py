"""
Confluence Scorer
==================
Aggregates signals from ALL analysis engines into a single opportunity score
and identifies the highest-probability trade setups.

Scoring model (weighted sum, normalised to 0–100):

  Component                     Max points
  ─────────────────────────────────────────
  Market structure alignment         20
  Multi-timeframe confluence         20
  Support / resistance quality       15
  Candlestick signal score           15
  Fibonacci confluence               10
  Chart pattern quality              10
  SMC / liquidity alignment           5
  Trend strength                      5
  Moving averages (EMA 50/200)       10
  AMD cycle (ICT)                    12
  ─────────────────────────────────────────
  TOTAL                             122  (capped at 100)

A score ≥ 75 = high-probability setup.
50 – 74 = moderate probability.
< 50     = low probability / wait.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engines.market_structure import MarketStructureResult, StructureType
from engines.support_resistance import SRResult
from engines.candlestick_engine import CandlestickResult
from engines.chart_patterns import ChartPatternResult
from engines.fibonacci_engine import FibResult
from engines.liquidity_smc import SMCResult
from engines.multi_timeframe import MTFResult
from engines.vwap_engine import VWAPResult
from engines.amd_engine import AMDResult

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceScore:
    total:               float = 0.0    # 0–100
    grade:               str   = "F"    # A / B / C / D / F
    direction:           str   = "sideways"
    component_scores:    Dict[str, float] = field(default_factory=dict)
    reasons:             List[str]        = field(default_factory=list)
    invalidation_levels: List[float]      = field(default_factory=list)
    entry_zone:          Optional[tuple]  = None   # (low, high)


class ConfluenceScorer:
    """
    Scores a combined set of engine outputs.

    Usage::

        scorer  = ConfluenceScorer()
        score   = scorer.score(ms_result, sr_result, cs_result, pat_result,
                               fib_result, smc_result, mtf_result)
    """

    def score(
        self,
        ms:           Optional[MarketStructureResult] = None,
        sr:           Optional[SRResult]              = None,
        cs:           Optional[CandlestickResult]     = None,
        pat:          Optional[ChartPatternResult]    = None,
        fib:          Optional[FibResult]             = None,
        smc:          Optional[SMCResult]             = None,
        mtf:          Optional[MTFResult]             = None,
        vwap:         Optional[VWAPResult]            = None,
        amd:          Optional[AMDResult]             = None,
        ma_signal:    Optional[str]                  = None,   # "bullish" | "bearish" | "neutral"
        ma_strength:  float                          = 0.0,   # 0.0 – 1.0
        session_utc_hour: Optional[int]              = None,
        regime_name:  Optional[str]                  = None,   # e.g. "news_volatility", "compression"
        atr_rank:     Optional[float]                = None,   # 0–100: ATR percentile rank
    ) -> ConfluenceScore:

        comps: Dict[str, float] = {}
        reasons: List[str]      = []
        direction_votes: Dict[str, float] = {"bullish": 0.0, "bearish": 0.0, "sideways": 0.0}

        # ── Market structure (max 20) ──────────────────────────────────────────
        if ms is not None:
            s_type = ms.structure_type
            ts     = ms.trend_strength
            if s_type == StructureType.BULLISH:
                pts = 10 + ts * 10
                direction_votes["bullish"] += pts
                reasons.append(f"Bullish market structure ({ts:.0%} strength).")
            elif s_type == StructureType.BEARISH:
                pts = 10 + ts * 10
                direction_votes["bearish"] += pts
                reasons.append(f"Bearish market structure ({ts:.0%} strength).")
            else:
                pts = 5.0
                direction_votes["sideways"] += pts
                reasons.append("Sideways / undefined structure — reduced conviction.")
            if ms.choch_events:
                reasons.append("CHoCH detected — possible trend reversal in progress.")
            comps["market_structure"] = round(min(20.0, pts), 2)
        else:
            comps["market_structure"] = 0.0

        # ── Multi-timeframe (max 20) ───────────────────────────────────────────
        if mtf is not None:
            conf_map   = {"high": 1.0, "medium": 0.7, "low": 0.3}
            conf_mult  = conf_map.get(mtf.confluence_level, 0.3)
            raw_score  = abs(mtf.alignment_score)
            pts        = raw_score * conf_mult * 20
            direction  = mtf.primary_bias
            direction_votes[direction] = direction_votes.get(direction, 0.0) + pts
            reasons.append(
                f"MTF bias: {direction} (score {mtf.alignment_score:+.2f}, {mtf.confluence_level} confluence)."
            )
            comps["multi_timeframe"] = round(min(20.0, pts), 2)
        else:
            comps["multi_timeframe"] = 0.0

        # ── Support & resistance (max 15) ─────────────────────────────────────
        if sr is not None:
            max_conf = 0.0
            if sr.nearest_support:
                max_conf = max(max_conf, sr.nearest_support.confidence)
                reasons.append(
                    f"Support at {sr.nearest_support.level:.4f} "
                    f"(confidence {sr.nearest_support.confidence:.0%})."
                )
                direction_votes["bullish"] += sr.nearest_support.confidence * 7
            if sr.nearest_resistance:
                max_conf = max(max_conf, sr.nearest_resistance.confidence)
                reasons.append(
                    f"Resistance at {sr.nearest_resistance.level:.4f} "
                    f"(confidence {sr.nearest_resistance.confidence:.0%})."
                )
                direction_votes["bearish"] += sr.nearest_resistance.confidence * 7
            comps["support_resistance"] = round(min(15.0, max_conf * 15), 2)
        else:
            comps["support_resistance"] = 0.0

        # ── Candlestick signal (max 15) ────────────────────────────────────────
        if cs is not None and cs.top_signals:
            top = cs.top_signals[0]
            pts = top.final_score / 10.0 * 15.0
            direction_votes[top.direction] = direction_votes.get(top.direction, 0.0) + pts
            reasons.append(
                f"Candlestick: {top.direction} {top.pattern} at bar {top.bar_index} "
                f"(score {top.final_score:.1f})."
            )
            comps["candlestick"] = round(min(15.0, pts), 2)
        else:
            comps["candlestick"] = 0.0

        # ── Fibonacci (max 10) ─────────────────────────────────────────────────
        if fib is not None:
            pts = 0.0
            if fib.current_level and fib.current_level.is_key:
                pts += 5.0
                reasons.append(
                    f"Price at key Fibonacci level {fib.current_level.label}% "
                    f"({fib.current_level.price:.4f})."
                )
            if fib.strongest_level:
                pts += fib.strongest_level.reaction_score * 5.0
                reasons.append(
                    f"Historical reactions at {fib.strongest_level.label}% "
                    f"(score {fib.strongest_level.reaction_score:.0%})."
                )
            comps["fibonacci"] = round(min(10.0, pts), 2)
        else:
            comps["fibonacci"] = 0.0

        # ── Chart patterns (max 10) ────────────────────────────────────────────
        if pat is not None and pat.top_pattern:
            tp  = pat.top_pattern
            pts = tp.confidence * 10.0
            direction_votes[tp.direction] = direction_votes.get(tp.direction, 0.0) + pts
            reasons.append(
                f"Chart pattern: {tp.name} ({tp.direction}, confidence {tp.confidence:.0%})."
            )
            comps["chart_patterns"] = round(min(10.0, pts), 2)
        else:
            comps["chart_patterns"] = 0.0

        # ── SMC / liquidity (max 10) ─────────────────────────────────────────
        if smc is not None:
            pts = 0.0
            if smc.stop_hunts:
                pts += 2.5
                latest = smc.stop_hunts[-1]
                reasons.append(f"Stop hunt detected ({latest.direction}) — institutional move probable.")
                rev_dir = "bullish" if latest.direction == "bullish_sweep" else "bearish"
                direction_votes[rev_dir] = direction_votes.get(rev_dir, 0.0) + 2.5
            fresh_obs = [o for o in smc.order_blocks if not o.mitigated]
            if fresh_obs:
                pts += 2.5
                reasons.append(f"{len(fresh_obs)} unmitigated order block(s) present.")
                # OBs cast direction votes weighted by strength
                # Bullish OB below price = institutional demand = bullish bias
                # Bearish OB above price = institutional supply = bearish bias
                for ob in fresh_obs[:5]:   # top 5 by strength
                    vote_weight = 1.5 * ob.strength
                    direction_votes[ob.direction] = (
                        direction_votes.get(ob.direction, 0.0) + vote_weight
                    )
                strongest = max(fresh_obs, key=lambda o: o.strength)
                reasons.append(
                    f"Strongest fresh OB: {strongest.direction} "
                    f"({strongest.low:.4f}–{strongest.high:.4f}, strength {strongest.strength:.0%})."
                )
            # FVGs also carry directional information
            fresh_fvgs = [g for g in smc.fair_value_gaps if not g.filled]
            if fresh_fvgs:
                pts += 2.0
                for fvg in fresh_fvgs[:3]:   # top 3 most recent
                    direction_votes[fvg.direction] = (
                        direction_votes.get(fvg.direction, 0.0) + 0.5
                    )
                reasons.append(f"{len(fresh_fvgs)} unfilled FVG(s) — price imbalance zones active.")
            comps["smc_liquidity"] = round(min(10.0, pts), 2)
        else:
            comps["smc_liquidity"] = 0.0

        # ── Trend strength (max 5) ─────────────────────────────────────────────
        ts = ms.trend_strength if ms else 0.0
        comps["trend_strength"] = round(min(5.0, ts * 5.0), 2)

        # ── VWAP alignment (max 12 with band extreme bonus) ───────────────────────
        # Base 0–8 for directional VWAP position.
        # +4 bonus when price is at a ±2σ extreme AND direction confirms a
        # mean-reversion entry (short at extreme_high, long at extreme_low).
        # This is when institutions absorb retail trend-followers at extremes.
        if vwap is not None:
            vwap_pts = vwap.score   # 0–12 already computed by VWAPEngine
            if vwap.signal == "above":
                direction_votes["bullish"] = direction_votes.get("bullish", 0.0) + vwap_pts
            elif vwap.signal == "below":
                direction_votes["bearish"] = direction_votes.get("bearish", 0.0) + vwap_pts
            # ±2σ band extreme confirmation note
            if vwap.band_zone == "extreme_high":
                reasons.append(
                    f"VWAP +2sd extreme ({vwap.band2_upper:.4f}): institutional "
                    "exhaustion zone -- bearish mean-reversion confluence."
                )
            elif vwap.band_zone == "extreme_low":
                reasons.append(
                    f"VWAP -2sd extreme ({vwap.band2_lower:.4f}): institutional "
                    "exhaustion zone -- bullish mean-reversion confluence."
                )
            elif vwap.band_zone in ("upper_band", "lower_band"):
                reasons.append(vwap.note)
            else:
                reasons.append(vwap.note)
            comps["vwap"] = round(vwap_pts, 2)
        else:
            comps["vwap"] = 0.0

        # ── AMD cycle / ICT (max 12) ────────────────────────────────────────────
        # Manipulation sweep confirmed                    +5
        # Distribution FVG found (ideal entry candle)    +5
        # In active distribution phase right now         +2
        if amd is not None and amd.manipulation_swept:
            _amd_pts = 5.0
            direction_votes[amd.direction] = (
                direction_votes.get(amd.direction, 0.0) + _amd_pts
            )
            reasons.append(
                f"AMD: {amd.direction.upper()} distribution after {amd.sweep_side} sweep "
                f"@ {amd.sweep_price:.4f} (Asian range {amd.asian_low:.4f}–{amd.asian_high:.4f})."
            )
            if amd.entry_fvgs:
                _fvg = amd.best_entry_fvg
                _amd_pts += 5.0
                reasons.append(
                    f"AMD distribution FVG entry: {_fvg.low:.4f}–{_fvg.high:.4f} "
                    f"({len(amd.entry_fvgs)} FVG(s) in distribution direction)."
                )
            if amd.phase == "distribution":
                _amd_pts += 2.0
                reasons.append("AMD: currently in distribution phase — optimal entry window.")
            comps["amd_cycle"] = round(min(12.0, _amd_pts), 2)
        elif amd is not None and amd.asian_high is not None:
            # Asian range established but no sweep yet — minor bonus for knowing the range
            comps["amd_cycle"] = 0.5
            reasons.append(
                f"AMD: Asian range set ({amd.asian_low:.4f}–{amd.asian_high:.4f}). Awaiting manipulation sweep."
            )
        else:
            comps["amd_cycle"] = 0.0

        # ── Moving averages EMA 50 / EMA 200 (max 10) ───────────────────────────
        # Full alignment: price and EMA50 both on same side of EMA200 → max pts.
        # Partial: price beyond EMA200 but EMA50 not yet crossed → half pts.
        # A recent golden/death cross within last 20 bars adds 20% strength boost.
        if ma_signal and ma_signal != "neutral" and ma_strength > 0:
            _ma_pts = round(min(10.0, ma_strength * 10.0), 2)
            direction_votes[ma_signal] = direction_votes.get(ma_signal, 0.0) + _ma_pts
            _ma_align = "fully aligned" if ma_strength >= 0.9 else "partially aligned"
            reasons.append(
                f"EMA 50/200: {ma_signal} trend ({_ma_align}, strength {ma_strength:.0%})."
            )
            comps["moving_averages"] = _ma_pts
        else:
            comps["moving_averages"] = 0.0

        # ── Session timing bonus (±7) ─────────────────────────────────────────
        # London: 07-10 UTC  |  NY: 13-16 UTC  |  Overlap: 13-17 UTC
        # Off-session (Asian quiet or weekend): penalty.
        if session_utc_hour is not None:
            h = session_utc_hour % 24
            if 7 <= h < 10:          # London open — highest liquidity
                session_pts = 7.0
                session_note = f"London open session ({h:02d}:xx UTC) — peak liquidity."
            elif 13 <= h < 17:       # NY open / London-NY overlap
                session_pts = 6.0
                session_note = f"New York / overlap session ({h:02d}:xx UTC) — high liquidity."
            elif 10 <= h < 13:       # London mid-session
                session_pts = 3.0
                session_note = f"London mid-session ({h:02d}:xx UTC) — moderate liquidity."
            elif 17 <= h < 20:       # NY close / early Asian
                session_pts = 1.0
                session_note = f"NY close session ({h:02d}:xx UTC) — reducing liquidity."
            else:                    # Asian quiet hours / off-hours
                session_pts = -5.0
                session_note = (f"Off-session / Asian quiet ({h:02d}:xx UTC) — "
                                "low liquidity; setup probability reduced.")
            comps["session"] = session_pts
            reasons.append(session_note)
        else:
            comps["session"] = 0.0

        # ── ATR Percentile Rank / Volatility State (max +5 / min −3) ─────────
        # Low rank = compression → high-probability explosive setup brewing.
        # High rank = extreme expansion → move likely overextended; caution.
        # Mid rank = normal → neutral contribution.
        if atr_rank is not None:
            if atr_rank <= 20.0:
                atr_rank_pts = 5.0
                reasons.append(
                    f"ATR rank {atr_rank:.0f}% (compression): volatility coiling - "
                    "explosive directional move expected."
                )
            elif atr_rank >= 80.0:
                atr_rank_pts = -3.0
                reasons.append(
                    f"ATR rank {atr_rank:.0f}% (extreme expansion): market overextended - "
                    "continuation risk elevated."
                )
            else:
                atr_rank_pts = 0.0   # normal volatility — no adjustment
            comps["atr_rank"] = round(atr_rank_pts, 2)
        else:
            comps["atr_rank"] = 0.0

        raw_total = sum(comps.values())
        total     = min(100.0, max(0.0, raw_total))  # cap 0-100

        # ── Regime modulation (post-scoring, pre-grade) ───────────────────────
        # Adjusts the total or individual components based on the active market
        # regime to avoid over-confidence in ambiguous or dangerous conditions.
        if regime_name:
            _rn = regime_name.lower()
            if _rn == "news_volatility":
                # Extreme ATR spike — cap score; no high-conviction entries
                total = min(total, 70.0)
                reasons.append("[Regime] NEWS_VOLATILITY: score capped at 70 — avoid news entries.")
            elif _rn == "dead_liquidity":
                # Rollover / zero movement — no reliable signal
                total = min(total, 60.0)
                reasons.append("[Regime] DEAD_LIQUIDITY: score capped at 60 — rollover conditions.")
            elif _rn == "compression":
                # Low-ATR coiling — patterns unclear, reduce SMC and candlestick weight
                comps["smc_liquidity"] = round(comps.get("smc_liquidity", 0.0) * 0.70, 2)
                comps["candlestick"]   = round(comps.get("candlestick",   0.0) * 0.80, 2)
                total = min(100.0, max(0.0, sum(comps.values())))
                reasons.append("[Regime] COMPRESSION: SMC/candlestick reduced — coiling conditions.")
            elif _rn == "liquidity_sweep":
                # Active stop-hunt — boost institutional confluence signals
                comps["smc_liquidity"] = round(min(10.0, comps.get("smc_liquidity", 0.0) * 1.15), 2)
                comps["amd_cycle"]     = round(min(12.0, comps.get("amd_cycle",     0.0) * 1.15), 2)
                total = min(100.0, max(0.0, sum(comps.values())))
                reasons.append("[Regime] LIQUIDITY_SWEEP: SMC/AMD scores boosted +15%.")

        # Dominant direction from weighted votes
        primary_dir = max(direction_votes, key=direction_votes.get)  # type: ignore

        grade = self._grade(total)

        # Build invalidation levels from SR
        invalid_levels: List[float] = []
        if sr:
            if primary_dir == "bullish" and sr.nearest_support:
                invalid_levels.append(sr.nearest_support.lower)
            elif primary_dir == "bearish" and sr.nearest_resistance:
                invalid_levels.append(sr.nearest_resistance.upper)

        # Entry zone from nearest S/R
        entry: Optional[tuple] = None
        if sr and primary_dir == "bullish" and sr.nearest_support:
            z = sr.nearest_support
            entry = (z.lower, z.upper)
        elif sr and primary_dir == "bearish" and sr.nearest_resistance:
            z = sr.nearest_resistance
            entry = (z.lower, z.upper)

        result = ConfluenceScore(
            total=round(total, 1),
            grade=grade,
            direction=primary_dir,
            component_scores=comps,
            reasons=reasons,
            invalidation_levels=invalid_levels,
            entry_zone=entry,
        )
        logger.info(
            "Confluence score: %.1f / 100 (grade %s, %s)",
            total, grade, primary_dir,
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _grade(self, score: float) -> str:
        if score >= 85:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 65:
            return "C"
        elif score >= 50:
            return "D"
        return "F"
