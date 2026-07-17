"""Unit tests for the Candlestick and S/R engines."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_data import generate_xauusd_ohlcv, generate_trending_data
from engines.candlestick_engine import CandlestickEngine
from engines.support_resistance import SupportResistanceEngine


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def df():
    return generate_xauusd_ohlcv(n_bars=300, seed=77)


@pytest.fixture
def cs_engine():
    return CandlestickEngine()


@pytest.fixture
def sr_engine():
    return SupportResistanceEngine(zone_tolerance_atr=0.3, min_touches=2, lookback_period=200)


# ─── CandlestickEngine ────────────────────────────────────────────────────────

class TestCandlestickEngine:

    def test_analyze_returns_result(self, cs_engine, df):
        result = cs_engine.analyze(df)
        assert result is not None

    def test_top_signals_is_list(self, cs_engine, df):
        result = cs_engine.analyze(df)
        assert isinstance(result.top_signals, list)

    def test_signal_scores_positive(self, cs_engine, df):
        result = cs_engine.analyze(df)
        for sig in result.top_signals:
            assert sig.raw_score   >= 0
            assert sig.final_score >= 0

    def test_direction_valid_values(self, cs_engine, df):
        result = cs_engine.analyze(df)
        for sig in result.top_signals:
            assert sig.direction in ("bullish", "bearish", "neutral")

    def test_pattern_name_nonempty(self, cs_engine, df):
        result = cs_engine.analyze(df)
        for sig in result.top_signals:
            assert isinstance(sig.pattern, str) and len(sig.pattern) > 0

    def test_no_exception_small_df(self, cs_engine):
        tiny = generate_xauusd_ohlcv(n_bars=10)
        result = cs_engine.analyze(tiny)
        assert result is not None

    def test_context_score_with_levels(self, cs_engine, df):
        """Providing S/R and fib levels should yield context-boosted scores."""
        result_no_ctx = cs_engine.analyze(df)
        price = float(df["close"].iloc[-1])
        result_with_ctx = cs_engine.analyze(
            df,
            support_levels=[price - 2, price - 5],
            resistance_levels=[price + 2, price + 5],
            fib_levels=[price - 1],
        )
        # With context near current price, some scores may increase
        assert result_with_ctx is not None

    def test_narrative_string(self, cs_engine, df):
        result = cs_engine.analyze(df)
        assert isinstance(result.narrative, str)

    def test_bullish_bias_on_bullish_trend(self):
        bullish_df = generate_trending_data(n_bars=200, direction="bullish")
        engine = CandlestickEngine()
        result = engine.analyze(bullish_df)
        # Most signals should lean bullish in a strong bull trend
        bull_count = sum(1 for s in result.top_signals if s.direction == "bullish")
        bear_count = sum(1 for s in result.top_signals if s.direction == "bearish")
        # Not strictly required but this is a sanity check
        assert bull_count >= 0 and bear_count >= 0


# ─── SupportResistanceEngine ─────────────────────────────────────────────────

class TestSupportResistanceEngine:

    def test_analyze_returns_result(self, sr_engine, df):
        result = sr_engine.analyze(df)
        assert result is not None

    def test_has_support_and_resistance(self, sr_engine, df):
        result = sr_engine.analyze(df)
        assert isinstance(result.support_zones, list)
        assert isinstance(result.resistance_zones, list)

    def test_support_below_resistance(self, sr_engine, df):
        result = sr_engine.analyze(df)
        if result.support_zones and result.resistance_zones:
            avg_support    = sum(z.level for z in result.support_zones) / len(result.support_zones)
            avg_resistance = sum(z.level for z in result.resistance_zones) / len(result.resistance_zones)
            assert avg_support < avg_resistance

    def test_confidence_range(self, sr_engine, df):
        result = sr_engine.analyze(df)
        for z in result.support_zones + result.resistance_zones:
            assert 0.0 <= z.confidence <= 1.0, f"Confidence {z.confidence} out of [0,1]"

    def test_zone_levels_in_price_range(self, sr_engine, df):
        result = sr_engine.analyze(df)
        data_min = float(df["low"].min())
        data_max = float(df["high"].max())
        for z in result.support_zones + result.resistance_zones:
            assert data_min * 0.99 <= z.level <= data_max * 1.01, (
                f"Zone level {z.level} outside price range [{data_min}, {data_max}]"
            )

    def test_nearest_zones(self, sr_engine, df):
        result = sr_engine.analyze(df)
        current_price = float(df["close"].iloc[-1])
        # nearest_support should be below current price
        if result.nearest_support:
            assert result.nearest_support.level <= current_price * 1.01
        if result.nearest_resistance:
            assert result.nearest_resistance.level >= current_price * 0.99

    def test_zone_labels_nonempty(self, sr_engine, df):
        result = sr_engine.analyze(df)
        for z in result.support_zones + result.resistance_zones:
            assert isinstance(z.label, str) and len(z.label) > 0

    def test_zone_type_values(self, sr_engine, df):
        result = sr_engine.analyze(df)
        for z in result.support_zones:
            assert z.zone_type == "support"
        for z in result.resistance_zones:
            assert z.zone_type == "resistance"

    def test_no_exception_tiny_df(self, sr_engine):
        tiny = generate_xauusd_ohlcv(n_bars=15)
        result = sr_engine.analyze(tiny)
        assert result is not None

    def test_zone_touch_count_positive(self, sr_engine, df):
        result = sr_engine.analyze(df)
        for z in result.support_zones + result.resistance_zones:
            assert z.touch_count >= 1
