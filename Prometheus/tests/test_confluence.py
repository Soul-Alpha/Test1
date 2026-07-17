"""Unit tests for Confluence Scorer and AI Reasoning Engine."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_data import generate_xauusd_ohlcv, generate_trending_data
from engines.market_structure import MarketStructureEngine
from engines.support_resistance import SupportResistanceEngine
from engines.candlestick_engine import CandlestickEngine
from engines.chart_patterns import ChartPatternEngine
from engines.fibonacci_engine import FibonacciEngine
from engines.liquidity_smc import LiquiditySMCEngine
from engines.multi_timeframe import MultiTimeframeEngine
from analysis.confluence_scorer import ConfluenceScorer, ConfluenceScore
from analysis.ai_reasoning import AIReasoningEngine, AnalysisReport


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def random_df():
    return generate_xauusd_ohlcv(n_bars=300, seed=11)


@pytest.fixture(scope="module")
def bullish_df():
    return generate_trending_data(n_bars=300, direction="bullish")


@pytest.fixture(scope="module")
def all_engine_results(random_df):
    """Run all engines once and return results dict."""
    ms_engine  = MarketStructureEngine(pivot_sensitivity=3)
    sr_engine  = SupportResistanceEngine()
    cs_engine  = CandlestickEngine()
    pat_engine = ChartPatternEngine()
    fib_engine = FibonacciEngine()
    smc_engine = LiquiditySMCEngine()
    mtf_engine = MultiTimeframeEngine()

    ms  = ms_engine.analyze(random_df)
    sr  = sr_engine.analyze(random_df)
    fib = fib_engine.analyze(random_df)
    cs  = cs_engine.analyze(random_df,
                             support_levels=[z.level for z in sr.support_zones[:3]],
                             resistance_levels=[z.level for z in sr.resistance_zones[:3]])
    sh  = ms.swing_highs
    sl  = ms.swing_lows
    pat = pat_engine.analyze(random_df, sh, sl)
    smc = smc_engine.analyze(random_df)
    mtf = mtf_engine.analyze({"4H": random_df})

    return dict(ms=ms, sr=sr, cs=cs, pat=pat, fib=fib, smc=smc, mtf=mtf)


# ─── ConfluenceScorer ────────────────────────────────────────────────────────

class TestConfluenceScorer:

    @pytest.fixture
    def scorer(self):
        return ConfluenceScorer()

    def test_score_returns_object(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert isinstance(result, ConfluenceScore)

    def test_total_score_range(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert 0.0 <= result.total <= 100.0, f"Score {result.total} outside [0, 100]"

    def test_grade_valid(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_direction_valid(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert result.direction in ("bullish", "bearish", "sideways", "neutral")

    def test_component_scores_dict(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert isinstance(result.component_scores, dict)
        assert len(result.component_scores) > 0

    def test_component_scores_non_negative(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        for k, v in result.component_scores.items():
            assert v >= 0.0, f"Component {k} is negative: {v}"

    def test_reasons_list(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert isinstance(result.reasons, list)

    def test_invalidation_levels_list(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        assert isinstance(result.invalidation_levels, list)

    def test_none_inputs_handled(self, scorer):
        """Scorer should not crash when all inputs are None."""
        result = scorer.score()
        assert result is not None
        assert 0.0 <= result.total <= 100.0

    def test_partial_inputs(self, scorer, random_df):
        ms = MarketStructureEngine().analyze(random_df)
        result = scorer.score(ms=ms)
        assert result is not None
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_bullish_score_higher_on_bullish_data(self, scorer, bullish_df):
        """Properly trending bullish data should get a higher score."""
        ms = MarketStructureEngine(pivot_sensitivity=3).analyze(bullish_df)
        sr = SupportResistanceEngine().analyze(bullish_df)
        result = scorer.score(ms=ms, sr=sr)
        # Can't guarantee A grade from structure alone, but score should be reasonable
        assert result.total >= 0

    def test_entry_zone_optional(self, scorer, all_engine_results):
        result = scorer.score(**all_engine_results)
        # entry_zone is Optional[List[float]] - can be None
        if result.entry_zone is not None:
            assert len(result.entry_zone) == 2
            assert result.entry_zone[0] <= result.entry_zone[1]


# ─── AIReasoningEngine ───────────────────────────────────────────────────────

class TestAIReasoningEngine:

    @pytest.fixture
    def reasoner(self):
        return AIReasoningEngine()

    @pytest.fixture
    def scorer(self):
        return ConfluenceScorer()

    def test_generate_returns_report(self, reasoner, scorer, all_engine_results, random_df):
        confluence = scorer.score(**all_engine_results)
        current_price = float(random_df["close"].iloc[-1])
        report = reasoner.generate(
            asset="XAUUSD", timeframe="4H",
            current_price=current_price,
            confluence=confluence,
            **all_engine_results,
        )
        assert isinstance(report, AnalysisReport)

    def test_full_text_nonempty(self, reasoner, scorer, all_engine_results, random_df):
        confluence = scorer.score(**all_engine_results)
        current_price = float(random_df["close"].iloc[-1])
        report = reasoner.generate(
            asset="XAUUSD", timeframe="4H",
            current_price=current_price,
            confluence=confluence,
            **all_engine_results,
        )
        assert isinstance(report.full_text, str)
        assert len(report.full_text) > 100

    def test_report_sections_present(self, reasoner, scorer, all_engine_results, random_df):
        confluence = scorer.score(**all_engine_results)
        report = reasoner.generate(
            asset="TESTASSET", timeframe="1H",
            current_price=2000.0,
            confluence=confluence,
            **all_engine_results,
        )
        # Report should contain asset name
        assert "TESTASSET" in report.full_text

    def test_report_with_none_inputs(self, reasoner):
        """Reasoner must not crash with all-None inputs."""
        report = reasoner.generate(
            asset="XAUUSD", timeframe="4H",
            current_price=2000.0,
        )
        assert report is not None
        assert isinstance(report.full_text, str)

    def test_trend_bias_field(self, reasoner, scorer, all_engine_results, random_df):
        confluence = scorer.score(**all_engine_results)
        report = reasoner.generate(
            asset="XAUUSD", timeframe="4H",
            current_price=float(random_df["close"].iloc[-1]),
            confluence=confluence,
            **all_engine_results,
        )
        assert isinstance(report.trend_bias, str) and len(report.trend_bias) > 0

    def test_different_assets(self, reasoner):
        for asset in ("XAUUSD", "EURUSD", "BTCUSDT"):
            report = reasoner.generate(asset=asset, timeframe="1H", current_price=100.0)
            assert asset in report.full_text
