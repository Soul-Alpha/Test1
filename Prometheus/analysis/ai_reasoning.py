"""
AI Reasoning Engine
====================
Generates institutional-grade market commentary that reads like it was written
by a professional discretionary analyst.

The engine:
  1. Aggregates all engine outputs into a structured context.
  2. Uses a modular template system with dynamic variable insertion.
  3. Constructs bullish AND bearish scenario narratives.
  4. States clear invalidation conditions.
  5. Produces a final structured analysis report (dict + formatted text).

This does NOT make financial recommendations.  It provides analytical scenarios
in the style of institutional research notes.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from analysis.confluence_scorer import ConfluenceScore
from engines.candlestick_engine import CandlestickResult
from engines.chart_patterns import ChartPatternResult
from engines.fibonacci_engine import FibResult
from engines.liquidity_smc import SMCResult
from engines.market_structure import MarketStructureResult, StructureType
from engines.multi_timeframe import MTFResult
from engines.support_resistance import SRResult
try:
    from vision.chart_analyzer import ChartVisionResult
except (ImportError, ModuleNotFoundError):
    ChartVisionResult = None  # type: ignore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────

@dataclass
class AnalysisReport:
    asset:            str
    timeframe:        str
    timestamp:        str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    # ── Structured sections ───────────────────────────────────────────────────
    trend_bias:       str = ""
    market_structure: str = ""
    sr_summary:       str = ""
    candlestick_summary: str = ""
    pattern_summary:  str = ""
    liquidity_summary: str = ""
    fib_summary:      str = ""
    mtf_summary:      str = ""
    vision_summary:   str = ""

    bullish_scenario: str = ""
    bearish_scenario: str = ""
    invalidation:     str = ""
    confluence_score: float = 0.0
    confidence_grade: str  = "F"
    risk_zones:       str  = ""

    # ── Final trading signal ─────────────────────────────────────────────────
    final_signal:     str = ""

    # ── Full narrative ─────────────────────────────────────────────────────────
    full_text:        str = ""

    # ── Raw data ──────────────────────────────────────────────────────────────
    component_scores: Dict[str, float] = field(default_factory=dict)
    key_levels:       List[float]      = field(default_factory=list)


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class AIReasoningEngine:
    """
    Produce a structured, professional analytical report.

    Usage::

        engine = AIReasoningEngine()
        report = engine.generate(
            asset="XAUUSD",
            timeframe="4H",
            ms=ms_result,
            sr=sr_result,
            ...
        )
        print(report.full_text)
    """

    def generate(
        self,
        asset:     str = "XAUUSD",
        timeframe: str = "4H",
        current_price: Optional[float] = None,
        ms:        Optional[MarketStructureResult]  = None,
        sr:        Optional[SRResult]               = None,
        cs:        Optional[CandlestickResult]      = None,
        pat:       Optional[ChartPatternResult]     = None,
        fib:       Optional[FibResult]              = None,
        smc:       Optional[SMCResult]              = None,
        mtf:       Optional[MTFResult]              = None,
        vision:    Optional[ChartVisionResult]      = None,
        confluence: Optional[ConfluenceScore]       = None,
    ) -> AnalysisReport:

        r = AnalysisReport(asset=asset.upper(), timeframe=timeframe.upper())

        if current_price is None and sr:
            current_price = sr.current_price

        # ── Trend Bias ─────────────────────────────────────────────────────────
        r.trend_bias = self._trend_bias(ms, mtf, asset, timeframe, current_price)

        # ── Market Structure ───────────────────────────────────────────────────
        r.market_structure = ms.narrative if ms else "Market structure data unavailable."

        # ── S/R ────────────────────────────────────────────────────────────────
        r.sr_summary = sr.narrative if sr else "Support/resistance data unavailable."

        # ── Candlesticks ───────────────────────────────────────────────────────
        r.candlestick_summary = cs.narrative if cs else "No candlestick data."

        # ── Patterns ───────────────────────────────────────────────────────────
        r.pattern_summary = pat.narrative if pat else "No chart pattern data."

        # ── Fibonacci ──────────────────────────────────────────────────────────
        r.fib_summary = fib.narrative if fib else "No Fibonacci data."

        # ── Liquidity / SMC ────────────────────────────────────────────────────
        r.liquidity_summary = smc.narrative if smc else "No SMC data."

        # ── MTF ────────────────────────────────────────────────────────────────
        r.mtf_summary = mtf.narrative if mtf else "No multi-timeframe data."

        # ── Vision ─────────────────────────────────────────────────────────────
        r.vision_summary = vision.narrative if vision else "No chart image supplied."

        # ── Scenarios ──────────────────────────────────────────────────────────
        direction = confluence.direction if confluence else self._infer_direction(ms, mtf)
        r.bullish_scenario = self._bullish_scenario(sr, fib, smc, cs, current_price)
        r.bearish_scenario = self._bearish_scenario(sr, fib, smc, cs, current_price)

        # ── Invalidation ───────────────────────────────────────────────────────
        r.invalidation = self._invalidation(ms, sr, fib, direction, current_price)

        # ── Confluence ─────────────────────────────────────────────────────────
        if confluence:
            r.confluence_score  = confluence.total
            r.confidence_grade  = confluence.grade
            r.component_scores  = confluence.component_scores

        # ── Risk zones ─────────────────────────────────────────────────────────
        r.risk_zones = self._risk_zones(sr, fib, smc, current_price)

        # ── Key levels list ────────────────────────────────────────────────────
        r.key_levels = self._extract_key_levels(sr, fib, smc, current_price)

        # ── Final signal ───────────────────────────────────────────────────────
        r.final_signal = self._final_signal(direction, sr, fib, smc, ms, current_price, asset)

        # ── Assemble full report ────────────────────────────────────────────────
        r.full_text = self._assemble(r)

        logger.info("AI report generated: %s %s | Score %.1f | Grade %s",
                    asset, timeframe, r.confluence_score, r.confidence_grade)
        return r

    # ── Section builders ──────────────────────────────────────────────────────

    def _trend_bias(
        self,
        ms:     Optional[MarketStructureResult],
        mtf:    Optional[MTFResult],
        asset:  str,
        tf:     str,
        price:  Optional[float],
    ) -> str:
        price_str = f" at {price:.4f}" if price else ""
        if ms is None and mtf is None:
            return f"{asset} {price_str}: insufficient data for trend assessment."

        # Priority: MTF alignment, then single-TF structure
        if mtf:
            direction = mtf.primary_bias
            conf      = mtf.confluence_level
            score     = mtf.alignment_score
            return (
                f"{asset} maintains a **{direction}** bias on the {tf} timeframe{price_str}.  "
                f"Cross-timeframe alignment is {conf} (score {score:+.2f}), "
                f"confirming the primary directional stance."
            )
        if ms:
            label_map = {
                StructureType.BULLISH:  "bullish",
                StructureType.BEARISH:  "bearish",
                StructureType.SIDEWAYS: "sideways / ranging",
                StructureType.UNDEFINED: "undefined",
            }
            direction = label_map[ms.structure_type]
            return (
                f"{asset} is exhibiting a **{direction}** market structure on the {tf} timeframe"
                f"{price_str}.  Trend strength is measured at "
                f"{ms.trend_strength:.0%}."
            )
        return f"{asset}{price_str}: no structural bias identifiable."

    def _bullish_scenario(
        self,
        sr:    Optional[SRResult],
        fib:   Optional[FibResult],
        smc:   Optional[SMCResult],
        cs:    Optional[CandlestickResult],
        price: Optional[float],
    ) -> str:
        # Key levels
        res_level = f"{sr.nearest_resistance.level:.4f}" if sr and sr.nearest_resistance else "the resistance zone"
        sup_level = f"{sr.nearest_support.level:.4f}"    if sr and sr.nearest_support    else "the support zone"
        res_upper = f"{sr.nearest_resistance.upper:.4f}" if sr and sr.nearest_resistance else "highs"

        # Fib extension target
        fib_target = ""
        if fib and fib.levels:
            top_fib = max(fib.levels, key=lambda l: l.price)
            fib_target = f" toward the {top_fib.label}% Fibonacci extension at {top_fib.price:.4f}"

        # Liquidity above
        liq_target = ""
        if smc:
            buy_pools = [p for p in smc.liquidity_pools if p.direction == "buy_side" and not p.swept]
            if buy_pools:
                nearest = min(buy_pools, key=lambda p: abs(p.price - (price or 0)))
                liq_target = f" and buy-side liquidity resting above {nearest.price:.4f}"

        # Candle signal
        candle_str = ""
        if cs and cs.top_signals:
            sig = next((s for s in cs.top_signals if s.direction == "bullish"), None)
            if sig:
                candle_str = f"\n    • Confirmation: {sig.pattern} candle close above {res_level}."

        return (
            "SCENARIO A — TRUE BREAKOUT (Bullish Continuation)\n"
            f"  Price action:\n"
            f"    • Closes strongly above resistance at {res_level}\n"
            f"    • Pulls back to retest {res_level} as new support\n"
            f"    • Holds above {sup_level} on the retest\n"
            f"    • Continues higher{fib_target}{liq_target}\n"
            f"{candle_str}\n"
            f"  Outcome: Bullish continuation confirmed — structure shifts higher.\n"
            f"  Target zone: above {res_upper}"
        )

    def _bearish_scenario(
        self,
        sr:    Optional[SRResult],
        fib:   Optional[FibResult],
        smc:   Optional[SMCResult],
        cs:    Optional[CandlestickResult],
        price: Optional[float],
    ) -> str:
        # Key levels
        res_level = f"{sr.nearest_resistance.level:.4f}" if sr and sr.nearest_resistance else "the resistance zone"
        sup_level = f"{sr.nearest_support.level:.4f}"    if sr and sr.nearest_support    else "the support zone"
        sup_lower = f"{sr.nearest_support.lower:.4f}"    if sr and sr.nearest_support    else "lows"

        # Fib target
        fib_target = ""
        if fib and fib.levels:
            bot_fib = min(fib.levels, key=lambda l: l.price)
            fib_target = f" toward the {bot_fib.label}% Fibonacci level at {bot_fib.price:.4f}"

        # Sell-side liquidity
        liq_target = ""
        if smc:
            sell_pools = [p for p in smc.liquidity_pools if p.direction == "sell_side" and not p.swept]
            if sell_pools:
                nearest = min(sell_pools, key=lambda p: abs(p.price - (price or 0)))
                liq_target = f" with sell-side liquidity resting below {nearest.price:.4f}"

        # Stop-hunt signal from SMC
        hunt_str = ""
        if smc and smc.stop_hunts:
            latest = smc.stop_hunts[-1]
            hunt_str = f"\n    • Recent stop-hunt detected at {latest.sweep_price:.4f} — breakout traders already trapped."

        # Candle signal
        candle_str = ""
        if cs and cs.top_signals:
            sig = next((s for s in cs.top_signals if s.direction == "bearish"), None)
            if sig:
                candle_str = f"\n    • Rejection signal: {sig.pattern} candle confirms sellers at {res_level}."

        return (
            "SCENARIO B — LIQUIDITY SWEEP (Bearish Reversal)\n"
            f"  Price action (very common on Gold):\n"
            f"    • Wicks above resistance at {res_level} — trapping breakout buyers\n"
            f"    • Fails to close above {res_level} on a 4H / daily candle\n"
            f"    • Reverses sharply downward{hunt_str}\n"
            f"    • Breaks below {sup_level}{liq_target}\n"
            f"    • Targets deeper levels{fib_target}\n"
            f"{candle_str}\n"
            f"  Outcome: Bearish reversal confirmed — structure shifts lower.\n"
            f"  Watch level: below {sup_lower}"
        )

    def _invalidation(
        self,
        ms:    Optional[MarketStructureResult],
        sr:    Optional[SRResult],
        fib:   Optional[FibResult],
        direction: str,
        price: Optional[float],
    ) -> str:
        conds: List[str] = []

        if direction == "bullish":
            if sr and sr.nearest_support:
                conds.append(
                    f"A daily close below the support zone at {sr.nearest_support.lower:.4f} "
                    "would invalidate the bullish bias."
                )
            if ms and ms.higher_lows:
                hl = sorted(ms.higher_lows, key=lambda s: s.index)[-1]
                conds.append(
                    f"Breach of the most recent higher low at {hl.price:.4f} "
                    "signals structure breakdown."
                )
        else:
            if sr and sr.nearest_resistance:
                conds.append(
                    f"A daily close above the resistance zone at {sr.nearest_resistance.upper:.4f} "
                    "would invalidate the bearish thesis."
                )
            if ms and ms.lower_highs:
                lh = sorted(ms.lower_highs, key=lambda s: s.index)[-1]
                conds.append(
                    f"Breach of the most recent lower high at {lh.price:.4f} "
                    "signals a potential trend reversal."
                )

        if not conds:
            conds.append("No specific invalidation levels identified — exercise caution.")

        return "**Invalidation Conditions:**\n" + "\n".join(f"  - {c}" for c in conds)

    def _risk_zones(
        self,
        sr:    Optional[SRResult],
        fib:   Optional[FibResult],
        smc:   Optional[SMCResult],
        price: Optional[float],
    ) -> str:
        zones: List[str] = []

        if sr:
            if sr.nearest_support:
                z = sr.nearest_support
                zones.append(f"Support entry zone: {z.lower:.4f} – {z.upper:.4f}")
            if sr.nearest_resistance:
                z = sr.nearest_resistance
                zones.append(f"Resistance zone (target / short): {z.lower:.4f} – {z.upper:.4f}")

        if fib and fib.current_level:
            lvl = fib.current_level
            zones.append(f"Fibonacci {lvl.label}% confluence zone near {lvl.price:.4f}")

        if smc:
            fresh_obs = [ob for ob in smc.order_blocks if not ob.mitigated]
            for ob in fresh_obs[:2]:
                zones.append(f"{ob.direction.capitalize()} order block: {ob.low:.4f} – {ob.high:.4f}")

        if not zones:
            return "Risk zones: insufficient data."

        return "**Risk / Entry Zones:**\n" + "\n".join(f"  - {z}" for z in zones)

    def _extract_key_levels(
        self,
        sr:    Optional[SRResult],
        fib:   Optional[FibResult],
        smc:   Optional[SMCResult],
        price: Optional[float],
    ) -> List[float]:
        levels: set[float] = set()
        if sr:
            for z in (sr.nearest_support, sr.nearest_resistance):
                if z:
                    levels.add(round(z.level, 4))
        if fib:
            for lvl in fib.levels:
                if lvl.is_key:
                    levels.add(round(lvl.price, 4))
        if smc:
            for ob in smc.order_blocks[:5]:
                levels.add(round((ob.high + ob.low) / 2, 4))
        return sorted(levels)

    def _final_signal(
        self,
        direction:     str,
        sr:            Optional[SRResult],
        fib:           Optional[FibResult],
        smc:           Optional[SMCResult],
        ms:            Optional[MarketStructureResult],
        current_price: Optional[float],
        asset:         str = "",
    ) -> str:
        """Compute a concrete entry / SL / TP trading signal from available data."""
        if current_price is None or current_price <= 0:
            return "Insufficient price data to generate a final signal."

        is_long = direction in ("bullish", "long")
        is_short = direction in ("bearish", "short")
        if not (is_long or is_short):
            return "Market is ranging — no directional signal. Wait for structure break."

        price = current_price

        # ── Entry zone ───────────────────────────────────────────────────────
        # For longs: nearest support mid; for shorts: nearest resistance mid
        if is_long:
            entry_zone = sr.nearest_support if sr else None
            opp_zone   = sr.nearest_resistance if sr else None
        else:
            entry_zone = sr.nearest_resistance if sr else None
            opp_zone   = sr.nearest_support if sr else None

        entry = entry_zone.level if entry_zone else price

        # ── Stop Loss ────────────────────────────────────────────────────────
        # Below entry zone lower (long) or above entry zone upper (short)
        # with a 0.2 % buffer so market noise doesn't immediate-trigger it
        buf = price * 0.002
        if is_long:
            sl_raw = (entry_zone.lower - buf) if entry_zone else (price - price * 0.005)
        else:
            sl_raw = (entry_zone.upper + buf) if entry_zone else (price + price * 0.005)
        stop_loss = round(sl_raw, 2)

        risk = abs(entry - stop_loss)
        if risk == 0:
            risk = price * 0.003   # fallback 0.3 %

        # ── TP1 — nearest opposing S/R zone ──────────────────────────────────
        tp1 = None
        if opp_zone:
            tp1 = round(opp_zone.level, 2)

        # ── TP2 — next S/R zone or Fib extension ─────────────────────────────
        tp2 = None
        if sr and is_long and len(sr.resistance_zones) >= 2:
            # second nearest resistance
            sorted_res = sorted(sr.resistance_zones, key=lambda z: z.level)
            above = [z for z in sorted_res if z.level > (tp1 or price)]
            if above:
                tp2 = round(above[0].level, 2)
        elif sr and is_short and len(sr.support_zones) >= 2:
            sorted_sup = sorted(sr.support_zones, key=lambda z: z.level, reverse=True)
            below = [z for z in sorted_sup if z.level < (tp1 or price)]
            if below:
                tp2 = round(below[0].level, 2)

        # Fib extension fallback for TP2
        if tp2 is None and fib and fib.levels:
            ext_labels = {"127.2", "161.8"}
            ext = [l for l in fib.levels if l.label in ext_labels]
            if ext:
                if is_long:
                    above_ext = [l for l in ext if l.price > (tp1 or price)]
                    if above_ext:
                        tp2 = round(min(above_ext, key=lambda l: l.price).price, 2)
                else:
                    below_ext = [l for l in ext if l.price < (tp1 or price)]
                    if below_ext:
                        tp2 = round(max(below_ext, key=lambda l: l.price).price, 2)

        # Mathematical fallback: TP1 = 1.5×R, TP2 = 3×R from entry
        if tp1 is None:
            tp1 = round(entry + (1.5 * risk if is_long else -1.5 * risk), 2)
        if tp2 is None:
            tp2 = round(entry + (3.0 * risk if is_long else -3.0 * risk), 2)

        # ── R:R ratios ────────────────────────────────────────────────────────
        rr1 = abs(tp1 - entry) / risk
        rr2 = abs(tp2 - entry) / risk

        action = "BUY" if is_long else "SELL"
        arrow  = "▲" if is_long else "▼"

        lines = [
            f"  {arrow} {action}  {asset.upper()}  — Institutional Signal",
            f"",
            f"  Entry Area   :  {entry:.2f}   (current: {price:.2f})",
            f"  Stop Loss    :  {stop_loss:.2f}   (risk: {risk:.2f} pts)",
            f"  Take Profit 1:  {tp1:.2f}   (R:R  1:{rr1:.1f})",
            f"  Take Profit 2:  {tp2:.2f}   (R:R  1:{rr2:.1f})",
            f"",
            f"  Position sizing: risk max 1-2 % of account per trade.",
            f"  Confirm entry on lower TF (15m / 1H) before executing.",
        ]
        # add SMC order block context if relevant
        if smc:
            fresh = [ob for ob in smc.order_blocks
                     if not ob.mitigated
                     and ob.direction == ("bullish" if is_long else "bearish")]
            if fresh:
                ob = fresh[0]
                lines.insert(3, f"  OB Zone      :  {ob.low:.2f} – {ob.high:.2f}   (order block confluence)")

        return "\n".join(lines)

    def _infer_direction(
        self,
        ms:  Optional[MarketStructureResult],
        mtf: Optional[MTFResult],
    ) -> str:
        if mtf:
            return mtf.primary_bias
        if ms:
            return {
                StructureType.BULLISH: "bullish",
                StructureType.BEARISH: "bearish",
            }.get(ms.structure_type, "sideways")
        return "sideways"

    def _assemble(self, r: AnalysisReport) -> str:
        divider = "=" * 60

        sections = [
            f"{divider}",
            f"  PROMETHEUS MARKET ANALYSIS",
            f"  {r.asset} | {r.timeframe} | {r.timestamp}",
            f"{divider}",
            "",
            "── TREND BIAS ──────────────────────────────────────────────",
            textwrap.fill(r.trend_bias, width=72),
            "",
            "── MARKET STRUCTURE ────────────────────────────────────────",
            textwrap.fill(r.market_structure, width=72),
            "",
            "── SUPPORT & RESISTANCE ────────────────────────────────────",
            textwrap.fill(r.sr_summary, width=72),
            "",
            "── FIBONACCI ANALYSIS ──────────────────────────────────────",
            textwrap.fill(r.fib_summary, width=72),
            "",
            "── CANDLESTICK SIGNALS ─────────────────────────────────────",
            textwrap.fill(r.candlestick_summary, width=72),
            "",
            "── CHART PATTERNS ──────────────────────────────────────────",
            textwrap.fill(r.pattern_summary, width=72),
            "",
            "── LIQUIDITY & SMART MONEY ─────────────────────────────────",
            textwrap.fill(r.liquidity_summary, width=72),
            "",
            "── MULTI-TIMEFRAME OVERVIEW ────────────────────────────────",
            r.mtf_summary,
            "",
            "── SCENARIO A — TRUE BREAKOUT (Bullish) ───────────────────",
            r.bullish_scenario,
            "",
            "── SCENARIO B — LIQUIDITY SWEEP (Bearish) ──────────────────",
            r.bearish_scenario,
            "",
            "── INVALIDATION ────────────────────────────────────────────",
            r.invalidation,
            "",
            "── RISK / ENTRY ZONES ──────────────────────────────────────",
            r.risk_zones,
            "",
            "── CONFLUENCE SCORE ────────────────────────────────────────",
            f"  Overall Score:  {r.confluence_score:.1f} / 100  (Grade: {r.confidence_grade})",
        ]

        if r.component_scores:
            for k, v in r.component_scores.items():
                sections.append(f"    {k.replace('_', ' ').capitalize():<28} {v:.1f}")

        if r.final_signal:
            sections += [
                "",
                "── 🎯 FINAL SIGNAL ─────────────────────────────────────────",
                r.final_signal,
            ]

        sections += [
            "",
            f"{divider}",
            "  ⚠  DISCLAIMER: This analysis is for informational purposes",
            "     only.  It does NOT constitute financial advice.  Always",
            "     apply your own risk management.",
            f"{divider}",
        ]

        return "\n".join(sections)
