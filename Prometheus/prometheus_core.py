"""
Prometheus Core Orchestrator
=============================
Single entry-point that wires all engines together and returns a complete
AnalysisReport for a given asset / timeframe / data combination.

Usage (programmatic)::

    from prometheus_core import Prometheus

    bot = Prometheus()
    result = bot.analyze_data(df, asset="XAUUSD", timeframe="4H")
    print(result.report.full_text)

Usage (image analysis)::

    result = bot.analyze_image("chart.png", asset="XAUUSD", timeframe="4H")
"""

from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG, BASE_DIR, CHARTS_DIR, OUTPUTS_DIR
from engines.market_structure import MarketStructureEngine, MarketStructureResult
from engines.support_resistance import SupportResistanceEngine, SRResult
from engines.candlestick_engine import CandlestickEngine, CandlestickResult
from engines.chart_patterns import ChartPatternEngine, ChartPatternResult
from engines.fibonacci_engine import FibonacciEngine, FibResult
from engines.liquidity_smc import LiquiditySMCEngine, SMCResult
from engines.multi_timeframe import MultiTimeframeEngine, MTFResult
from engines.vwap_engine import VWAPEngine, VWAPResult
from engines.amd_engine import AMDEngine, AMDResult
from analysis.confluence_scorer import ConfluenceScorer, ConfluenceScore
from analysis.ai_reasoning import AIReasoningEngine, AnalysisReport
from ml.pattern_learner import PatternLearner, SetupRecord, MLPrediction
from visualization.chart_renderer import ChartRenderer
from storage.database import init_db, save_analysis

logger = logging.getLogger(__name__)


@dataclass
class PrometheusResult:
    """Complete result object returned from any analysis run."""
    run_id:       str
    timestamp:    str
    asset:        str
    timeframe:    str

    # Engine outputs
    ms:           Optional[MarketStructureResult] = None
    sr:           Optional[SRResult]              = None
    cs:           Optional[CandlestickResult]     = None
    pat:          Optional[ChartPatternResult]    = None
    fib:          Optional[FibResult]             = None
    smc:          Optional[SMCResult]             = None
    mtf:          Optional[MTFResult]             = None
    vwap:         Optional[VWAPResult]            = None
    amd:          Optional[AMDResult]             = None

    # Higher-level outputs
    confluence:   Optional[ConfluenceScore]       = None
    report:       Optional[AnalysisReport]        = None
    ml_prediction: Optional[MLPrediction]         = None

    # Volatility rank (0–100): where current ATR sits in its 252-bar history
    atr_rank:     Optional[float]                 = None

    # Chart paths
    interactive_chart: Optional[str] = None
    static_chart:      Optional[str] = None

    # Convenience
    current_price: Optional[float] = None


class Prometheus:
    """
    Institutional-grade AI market analysis system.

    Initialize once, then call analyze_data() or analyze_image() repeatedly.
    """

    def __init__(self, config=None) -> None:
        cfg = config or CONFIG

        # Initialize engines
        self.ms_engine   = MarketStructureEngine(
            pivot_sensitivity=cfg.market_structure.pivot_sensitivity,
            min_swing_atr_mult=cfg.market_structure.min_swing_atr_mult,
        )
        self.sr_engine   = SupportResistanceEngine(
            zone_tolerance_atr=cfg.support_resistance.zone_tolerance_atr,
            min_touches=cfg.support_resistance.min_touches,
            lookback_period=cfg.support_resistance.lookback_period,
        )
        self.cs_engine   = CandlestickEngine(
            pin_bar_wick_ratio=cfg.candlestick.pin_bar_wick_ratio,
            doji_body_pct=cfg.candlestick.doji_body_pct,
        )
        self.pat_engine  = ChartPatternEngine(
            double_top_pct=cfg.chart_patterns.double_top_pct,
        )
        self.fib_engine  = FibonacciEngine(
            levels=cfg.fibonacci.levels,
            key_levels=cfg.fibonacci.key_levels,
        )
        self.smc_engine  = LiquiditySMCEngine(
            equal_hl_tolerance_pct=cfg.liquidity_smc.equal_hl_tolerance_pct,
            fvg_min_atr_mult=cfg.liquidity_smc.fvg_min_size_atr,
        )
        self.mtf_engine  = MultiTimeframeEngine(
            timeframes=cfg.multi_timeframe.timeframes,
            weights=cfg.multi_timeframe.tf_weights,
        )
        self.vwap_engine = VWAPEngine(atr_period=cfg.market_structure.atr_period)
        self.amd_engine  = AMDEngine()
        self.scorer      = ConfluenceScorer()
        self.reasoner    = AIReasoningEngine()
        self.learner     = PatternLearner(
            model_dir=str(BASE_DIR / "models"),
            model_type=cfg.ml.model_type,
            min_samples_train=cfg.ml.min_samples_train,
        )
        self.renderer    = ChartRenderer(output_dir=str(OUTPUTS_DIR))

        # Initialize DB
        try:
            init_db()
        except Exception as e:
            logger.warning("DB init warning: %s", e)

        # Vision analyzer (lazy import to avoid mandatory CV dependency)
        self._vision_analyzer = None

        logger.info("Prometheus v%s initialized for %s", cfg.version, cfg.default_asset)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_data(
        self,
        df:           pd.DataFrame,
        asset:        str = "XAUUSD",
        timeframe:    str = "4H",
        tf_data:      Optional[Dict[str, pd.DataFrame]] = None,
        render_chart: bool = True,
        save_to_db:   bool = True,
    ) -> PrometheusResult:
        """
        Full analysis pipeline on OHLCV data.

        Args:
            df:           Primary timeframe OHLCV DataFrame.
            asset:        Asset symbol (e.g. "XAUUSD").
            timeframe:    Primary timeframe (e.g. "4H").
            tf_data:      Optional dict {timeframe: df} for MTF analysis.
            render_chart: Whether to generate chart files.
            save_to_db:   Whether to persist result to database.

        Returns:
            PrometheusResult
        """
        run_id    = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        logger.info("=== Analysis run %s: %s %s | raw bars: %d ===",
                    run_id, asset, timeframe, len(df))

        # ── Cap to most-recent 1 500 bars ──────────────────────────────────────
        # Engines are O(n) to O(n²) in bar count; 1 500 bars gives ~3 months of
        # 30-min data or ~6 months of 1H — more than enough for any TF analysis.
        MAX_BARS = 1_500
        if len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:].copy()
            logger.info("DataFrame trimmed to last %d bars for speed", MAX_BARS)

        result = PrometheusResult(
            run_id=run_id, timestamp=timestamp,
            asset=asset, timeframe=timeframe,
            current_price=float(df["close"].iloc[-1]) if "close" in df.columns else None,
        )

        # ── Run all engines ────────────────────────────────────────────────────
        try:
            result.ms = self.ms_engine.analyze(df)
        except Exception as e:
            logger.error("Market structure error: %s", e)

        try:
            result.sr = self.sr_engine.analyze(df)
        except Exception as e:
            logger.error("S/R error: %s", e)

        # Supply S/R and fib levels to candlestick engine for context scoring
        support_lvls    = [z.level for z in (result.sr.support_zones    if result.sr else [])][:5]
        resistance_lvls = [z.level for z in (result.sr.resistance_zones if result.sr else [])][:5]

        try:
            result.fib = self.fib_engine.analyze(
                df,
                swing_highs=result.ms.swing_highs if result.ms else None,
                swing_lows =result.ms.swing_lows  if result.ms else None,
            )
        except Exception as e:
            logger.error("Fibonacci error: %s", e)

        fib_lvls = [l.price for l in (result.fib.levels if result.fib else [])]

        try:
            result.cs = self.cs_engine.analyze(
                df,
                support_levels=support_lvls,
                resistance_levels=resistance_lvls,
                fib_levels=fib_lvls,
            )
        except Exception as e:
            logger.error("Candlestick error: %s", e)

        try:
            sh = result.ms.swing_highs if result.ms else []
            sl = result.ms.swing_lows  if result.ms else []
            result.pat = self.pat_engine.analyze(df, sh, sl)
        except Exception as e:
            logger.error("Pattern error: %s", e)

        try:
            result.smc = self.smc_engine.analyze(df)
        except Exception as e:
            logger.error("SMC error: %s", e)

        try:
            tf_input = tf_data or {timeframe: df}
            result.mtf = self.mtf_engine.analyze(tf_input)
        except Exception as e:
            logger.error("MTF error: %s", e)

        try:
            result.vwap = self.vwap_engine.analyze(df)
        except Exception as e:
            logger.error("VWAP error: %s", e)

        try:
            result.amd = self.amd_engine.analyze(
                df,
                existing_fvgs=result.smc.fair_value_gaps if result.smc else None,
            )
        except Exception as e:
            logger.error("AMD error: %s", e)

        # Extract last-bar UTC hour for session scoring
        _session_hour: Optional[int] = None
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                _session_hour = int(df.index[-1].hour)
        except Exception:
            pass

        # ── EMA 50 / EMA 200 moving-average trend filter ──────────────────────────
        # bull_trend : price > EMA50 > EMA200   → full alignment (strength 1.0)
        # partial    : price > EMA200 but EMA50 not yet above EMA200 → strength 0.5
        # recent cross (golden/death) within last 20 bars → +0.2 strength bonus
        _ma_signal   = "neutral"
        _ma_strength = 0.0
        try:
            _close_s    = df["close"]
            _ema50_s    = _close_s.ewm(span=50,  adjust=False).mean()
            _ema200_s   = _close_s.ewm(span=200, adjust=False).mean()
            _price_now  = float(_close_s.iloc[-1])
            _ema50_now  = float(_ema50_s.iloc[-1])
            _ema200_now = float(_ema200_s.iloc[-1])

            if _price_now > _ema50_now and _ema50_now > _ema200_now:
                _ma_signal   = "bullish"
                _ma_strength = 1.0
            elif _price_now < _ema50_now and _ema50_now < _ema200_now:
                _ma_signal   = "bearish"
                _ma_strength = 1.0
            elif _price_now > _ema200_now:   # above slow MA but EMA50 not yet aligned
                _ma_signal   = "bullish"
                _ma_strength = 0.5
            elif _price_now < _ema200_now:   # below slow MA but EMA50 not yet aligned
                _ma_signal   = "bearish"
                _ma_strength = 0.5

            # Detect recent golden / death cross within last 20 bars → momentum boost
            _lookback      = min(20, max(1, len(_ema50_s) - 2))
            _ema50_prev    = float(_ema50_s.iloc[-1 - _lookback])
            _ema200_prev   = float(_ema200_s.iloc[-1 - _lookback])
            _golden_cross  = _ema50_prev <= _ema200_prev and _ema50_now > _ema200_now
            _death_cross   = _ema50_prev >= _ema200_prev and _ema50_now < _ema200_now
            if _golden_cross or _death_cross:
                _ma_strength = min(1.0, _ma_strength + 0.2)
            logger.debug(
                "EMA50=%.4f EMA200=%.4f price=%.4f → MA signal=%s strength=%.0f%%",
                _ema50_now, _ema200_now, _price_now, _ma_signal, _ma_strength * 100,
            )
        except Exception as _e:
            logger.debug("EMA computation skipped: %s", _e)
        # ── ATR Percentile Rank (Volatility Rank) ──────────────────────────────────────────
        # Where the current ATR sits within its own 252-bar history:
        #   0–20   = compression (tension building — explosive move coming)
        #  20–80   = normal volatility
        #  80–100  = extreme expansion (trend may be exhausting)
        _atr_rank: Optional[float] = None
        try:
            _atr_s   = (df["high"] - df["low"]).rolling(14).mean()
            _atr_cur = float(_atr_s.iloc[-1])
            _atr_win = _atr_s.dropna().iloc[-min(252, len(_atr_s.dropna())):]
            _atr_min = float(_atr_win.min())
            _atr_max = float(_atr_win.max())
            if _atr_max > _atr_min:
                _atr_rank = round((_atr_cur - _atr_min) / (_atr_max - _atr_min) * 100.0, 1)
            else:
                _atr_rank = 50.0
            result.atr_rank = _atr_rank
            logger.debug("ATR rank = %.1f%%  (ATR=%.4f, min=%.4f, max=%.4f)",
                         _atr_rank, _atr_cur, _atr_min, _atr_max)
        except Exception as _e:
            logger.debug("ATR rank computation skipped: %s", _e)
        # ── Score confluence ───────────────────────────────────────────────────
        try:
            result.confluence = self.scorer.score(
                ms=result.ms, sr=result.sr, cs=result.cs,
                pat=result.pat, fib=result.fib, smc=result.smc, mtf=result.mtf,
                vwap=result.vwap, amd=result.amd,
                ma_signal=_ma_signal, ma_strength=_ma_strength,
                session_utc_hour=_session_hour,
                atr_rank=_atr_rank,
            )
        except Exception as e:
            logger.error("Confluence scoring error: %s", e)

        # ── Generate AI report ─────────────────────────────────────────────────
        try:
            result.report = self.reasoner.generate(
                asset=asset, timeframe=timeframe,
                current_price=result.current_price,
                ms=result.ms, sr=result.sr, cs=result.cs,
                pat=result.pat, fib=result.fib, smc=result.smc,
                mtf=result.mtf, confluence=result.confluence,
            )
        except Exception as e:
            logger.error("AI reasoning error: %s", e)

        # ── ML prediction ──────────────────────────────────────────────────────
        try:
            setup = self._build_setup_record(result, run_id, asset, timeframe, df)
            result.ml_prediction = self.learner.predict(setup)
            self.learner.add_setup(setup)
        except Exception as e:
            logger.error("ML error: %s", e)

        # ── Render charts ──────────────────────────────────────────────────────
        if render_chart:
            ctx = self._build_render_ctx(result)
            try:
                result.interactive_chart = self.renderer.render_plotly(df, ctx, f"{asset}_{timeframe}_{run_id}")
            except Exception as e:
                logger.warning("Plotly render failed: %s", e)
            try:
                result.static_chart = self.renderer.render_static(df, ctx, f"{asset}_{timeframe}_{run_id}")
            except Exception as e:
                logger.warning("Static render failed: %s", e)

        # ── Save to DB ─────────────────────────────────────────────────────────
        if save_to_db:
            try:
                save_analysis(asset, timeframe, self._report_to_dict(result))
            except Exception as e:
                logger.warning("DB save failed: %s", e)

        logger.info(
            "Analysis complete: score=%.1f grade=%s direction=%s",
            result.confluence.total   if result.confluence else 0,
            result.confluence.grade   if result.confluence else "?",
            result.confluence.direction if result.confluence else "?",
        )
        return result

    def analyze_image(
        self,
        image_path:  str,
        asset:       str = "XAUUSD",
        timeframe:   str = "4H",
        df:          Optional[pd.DataFrame] = None,
    ) -> PrometheusResult:
        """
        Analyze a chart image.  If df is also supplied, combines vision
        analysis with quantitative engine outputs.

        Args:
            image_path: Path to PNG / JPG chart screenshot.
            asset:      Asset symbol.
            timeframe:  Timeframe string.
            df:         Optional OHLCV data for quantitative analysis.

        Returns:
            PrometheusResult (vision result embedded in report)
        """
        vision_result = self._get_vision_analyzer().analyze_path(image_path)
        logger.info("Vision analysis: %s | direction=%s",
                    image_path, vision_result.dominant_direction)

        if df is not None:
            result = self.analyze_data(df, asset=asset, timeframe=timeframe)
            if result.report:
                result.report.vision_summary = vision_result.narrative
        else:
            run_id = str(uuid.uuid4())[:8]
            result = PrometheusResult(
                run_id=run_id,
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                asset=asset,
                timeframe=timeframe,
            )

            # Derive bias from vision
            dir_map = {"bullish": "long", "bearish": "short", "ranging": "neutral"}
            vision_dir = dir_map.get(vision_result.dominant_direction.lower(), "neutral")

            # Build a minimal confluence result so the summary cards have data
            from analysis.confluence_scorer import ConfluenceScore
            vision_score = 45.0 if vision_dir == "neutral" else 55.0
            result.confluence = ConfluenceScore(
                total=vision_score,
                grade="C" if vision_score >= 50 else "D",
                direction=vision_dir,
                component_scores={},
                reasons=[
                    f"Vision-only analysis — direction: {vision_result.dominant_direction}",
                    f"Candles detected: {len(vision_result.candles)}",
                ],
            )

            # Build report
            result.report = AnalysisReport(
                asset=asset.upper(),
                timeframe=timeframe.upper(),
                trend_bias=vision_result.dominant_direction,
                vision_summary=vision_result.narrative,
                bullish_scenario=(
                    "SCENARIO A — TRUE BREAKOUT (Bullish Continuation)\n"
                    "  Price action:\n"
                    "    • Closes strongly above visible highs/resistance\n"
                    "    • Pulls back to retest the breakout level as new support\n"
                    "    • Holds and continues upward\n"
                    "  Outcome: Bullish continuation confirmed — structure shifts higher.\n"
                    "  ⚠️  Upload OHLCV data alongside the image for precise price levels."
                ),
                bearish_scenario=(
                    "SCENARIO B — LIQUIDITY SWEEP (Bearish Reversal)\n"
                    "  Price action (very common on Gold):\n"
                    "    • Wicks above resistance — trapping breakout buyers\n"
                    "    • Fails to close above highs on 4H / daily candle\n"
                    "    • Reverses sharply downward through support\n"
                    "    • Sweeps sell-side liquidity resting below lows\n"
                    "  Outcome: Bearish reversal confirmed — structure shifts lower.\n"
                    "  ⚠️  Upload OHLCV data alongside the image for precise price levels."
                ),
                invalidation=(
                    "Invalidation Conditions:\n"
                    "  - Scenario A invalidated if price wicks above highs but closes back below them.\n"
                    "  - Scenario B invalidated if price closes strongly above resistance with volume confirmation.\n"
                    "  Upload OHLCV data for precise invalidation levels."
                ),
            )
            result.report.full_text = (
                f"Image analysis of {asset} {timeframe}:\n{vision_result.narrative}\n\n"
                f"{result.report.bullish_scenario}\n\n"
                f"{result.report.bearish_scenario}"
            )
            result.report.final_signal = (
                "Upload OHLCV CSV data alongside the chart image to generate\n"
                "a precise entry / stop-loss / take-profit signal."
            )
            result.interactive_chart = vision_result.chart_preview_path

        return result

    def label_outcome(
        self,
        run_id:    str,
        outcome:   int,
        rr:        Optional[float] = None,
        exit_price: Optional[float] = None,
    ) -> None:
        """
        Record the outcome of a previous setup to improve ML model.

        Args:
            run_id:     The run_id returned from analyze_data().
            outcome:    1 = profitable, 0 = loss.
            rr:         Risk-reward achieved.
            exit_price: Exit price of the position.
        """
        self.learner.update_outcome(run_id, outcome, rr, exit_price)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_vision_analyzer(self):
        if self._vision_analyzer is None:
            from vision.chart_analyzer import ChartVisionAnalyzer
            self._vision_analyzer = ChartVisionAnalyzer(output_dir=str(OUTPUTS_DIR))
        return self._vision_analyzer

    def _build_setup_record(
        self, result: PrometheusResult, run_id: str, asset: str, tf: str,
        df: Optional[pd.DataFrame] = None,
    ) -> SetupRecord:
        from engines.market_structure import StructureType
        from ml.pattern_learner import SetupRecord as MLSetup, classify_pattern_type

        st_map = {
            StructureType.BULLISH:  1,
            StructureType.BEARISH:  2,
            StructureType.SIDEWAYS: 3,
        }
        structure_type = st_map.get(
            result.ms.structure_type if result.ms else None, 0
        )

        # ── Volume ratio: last bar vol ÷ 20-bar average ──────────────────────────
        volume_ratio = 1.0  # neutral default when volume unavailable
        if df is not None and "volume" in df.columns and len(df) >= 2:
            try:
                last_vol = float(df["volume"].iloc[-1])
                lookback = min(20, len(df))
                avg_vol = float(df["volume"].iloc[-lookback:].mean())
                if avg_vol > 0:
                    volume_ratio = round(last_vol / avg_vol, 3)
            except Exception:
                pass

        # ── Pattern type & prior-trend alignment ─────────────────────────────
        pat_name = (
            result.pat.top_pattern.name
            if result.pat and result.pat.top_pattern else ""
        )
        pattern_type_id = classify_pattern_type(pat_name)

        # prior_trend_aligned: 1 when the pattern is contextually valid in the prior trend.
        #   cont_bull (1): needs bullish structure       | cont_bear (2): needs bearish structure
        #   rev_bull  (3): needs prior bear (to reverse) | rev_bear  (4): needs prior bull
        #   neutral   (5): always aligned (direction-agnostic breakout)
        prior_trend_aligned = 0
        if pat_name:
            if pattern_type_id == 1 and structure_type == 1:   # bull continuation in bull trend
                prior_trend_aligned = 1
            elif pattern_type_id == 2 and structure_type == 2: # bear continuation in bear trend
                prior_trend_aligned = 1
            elif pattern_type_id == 3 and structure_type == 2: # bull reversal after bear trend
                prior_trend_aligned = 1
            elif pattern_type_id == 4 and structure_type == 1: # bear reversal after bull trend
                prior_trend_aligned = 1
            elif pattern_type_id == 5:                         # breakout neutral: always valid
                prior_trend_aligned = 1

        return MLSetup(
            setup_id=run_id,
            asset=asset,
            timeframe=tf,
            timestamp=result.timestamp,
            structure_type=structure_type,
            trend_strength=result.ms.trend_strength if result.ms else 0.0,
            mtf_score=result.mtf.alignment_score if result.mtf else 0.0,
            sr_confidence=(
                result.sr.nearest_support.confidence
                if result.sr and result.sr.nearest_support else 0.0
            ),
            candlestick_score=(
                result.cs.top_signals[0].final_score / 10
                if result.cs and result.cs.top_signals else 0.0
            ),
            pattern_confidence=(
                result.pat.top_pattern.confidence
                if result.pat and result.pat.top_pattern else 0.0
            ),
            fib_proximity=int(
                bool(result.fib and result.fib.current_level and result.fib.current_level.is_key)
            ),
            ob_present=int(
                bool(result.smc and any(not ob.mitigated for ob in result.smc.order_blocks))
            ),
            stop_hunt=int(bool(result.smc and result.smc.stop_hunts)),
            confluence_score=result.confluence.total if result.confluence else 0.0,
            volume_ratio=volume_ratio,
            pattern_type_id=pattern_type_id,
            prior_trend_aligned=prior_trend_aligned,
        )

    def _build_render_ctx(self, result: PrometheusResult) -> Dict[str, Any]:
        return {
            "asset":     result.asset,
            "timeframe": result.timeframe,
            "ms":        result.ms,
            "sr":        result.sr,
            "fib":       result.fib,
            "smc":       result.smc,
            "confluence": result.confluence,
        }

    def _report_to_dict(self, result: PrometheusResult) -> Dict[str, Any]:
        r = result.report
        nearest_sup = nearest_res = None
        if result.sr:
            ns = result.sr.nearest_support
            nr = result.sr.nearest_resistance
            nearest_sup = float(ns.level) if ns else None
            nearest_res = float(nr.level) if nr else None
        return {
            "run_id":            result.run_id,
            "current_price":     result.current_price,
            "trend_bias":        r.trend_bias[:32] if r else "",
            "structure_type":    (result.ms.structure_type.name if result.ms else "")[:32],
            "confluence_score":  result.confluence.total if result.confluence else 0,
            "confidence_grade":  result.confluence.grade if result.confluence else "F",
            "primary_direction": (result.confluence.direction if result.confluence else "sideways")[:16],
            "key_levels":        result.confluence.invalidation_levels if result.confluence else [],
            "nearest_support":   nearest_sup,
            "nearest_resistance": nearest_res,
            "full_text":         r.full_text if r else "",
        }
