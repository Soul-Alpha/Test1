"""Unit tests for the Market Structure engine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_data import generate_trending_data, generate_ranging_data, generate_xauusd_ohlcv
from engines.market_structure import (
    MarketStructureEngine, StructureType, SwingType,
)


@pytest.fixture
def engine():
    return MarketStructureEngine(pivot_sensitivity=3)


@pytest.fixture
def bullish_df():
    return generate_trending_data(n_bars=200, direction="bullish")


@pytest.fixture
def bearish_df():
    return generate_trending_data(n_bars=200, direction="bearish")


@pytest.fixture
def ranging_df():
    return generate_ranging_data(n_bars=200)


@pytest.fixture
def random_df():
    return generate_xauusd_ohlcv(n_bars=300)


# ── Swing detection ──────────────────────────────────────────────────────────

class TestSwingDetection:
    def test_returns_lists(self, engine, random_df):
        sh, sl = engine.detect_swings(random_df)
        assert isinstance(sh, list)
        assert isinstance(sl, list)

    def test_swing_types(self, engine, random_df):
        sh, sl = engine.detect_swings(random_df)
        for s in sh:
            assert s.swing_type == SwingType.HIGH
        for s in sl:
            assert s.swing_type == SwingType.LOW

    def test_highs_above_lows(self, engine, random_df):
        sh, sl = engine.detect_swings(random_df)
        if sh and sl:
            assert max(s.price for s in sh) > min(s.price for s in sl)

    def test_prices_within_data_range(self, engine, random_df):
        sh, sl = engine.detect_swings(random_df)
        data_min = float(random_df["low"].min())
        data_max = float(random_df["high"].max())
        for s in sh:
            assert data_min <= s.price <= data_max, f"Swing HIGH {s.price} out of range"
        for s in sl:
            assert data_min <= s.price <= data_max, f"Swing LOW {s.price} out of range"

    def test_indices_within_bounds(self, engine, random_df):
        sh, sl = engine.detect_swings(random_df)
        n = len(random_df)
        for s in sh + sl:
            assert 0 <= s.index < n

    def test_minimum_data(self, engine):
        tiny_df = generate_xauusd_ohlcv(n_bars=5)
        sh, sl = engine.detect_swings(tiny_df)
        assert sh == [] and sl == []

    def test_strength_positive(self, engine, random_df):
        sh, sl = engine.detect_swings(random_df)
        for s in sh + sl:
            assert s.strength >= 1.0


# ── Structure classification ─────────────────────────────────────────────────

class TestStructureClassification:
    def test_bullish_trending(self, engine, bullish_df):
        result = engine.analyze(bullish_df)
        assert result.structure_type == StructureType.BULLISH, (
            f"Expected BULLISH, got {result.structure_type.name}"
        )

    def test_bearish_trending(self, engine, bearish_df):
        result = engine.analyze(bearish_df)
        assert result.structure_type == StructureType.BEARISH, (
            f"Expected BEARISH, got {result.structure_type.name}"
        )

    def test_ranging_sideways(self, engine, ranging_df):
        result = engine.analyze(ranging_df)
        assert result.structure_type in (StructureType.SIDEWAYS, StructureType.UNDEFINED)

    def test_trend_strength_range(self, engine, random_df):
        result = engine.analyze(random_df)
        assert 0.0 <= result.trend_strength <= 1.0


# ── BOS / CHoCH ──────────────────────────────────────────────────────────────

class TestStructureEvents:
    def test_bos_events_list(self, engine, random_df):
        result = engine.analyze(random_df)
        assert isinstance(result.bos_events, list)

    def test_choch_events_list(self, engine, random_df):
        result = engine.analyze(random_df)
        assert isinstance(result.choch_events, list)

    def test_bos_direction_values(self, engine, random_df):
        result = engine.analyze(random_df)
        for ev in result.bos_events:
            assert ev.direction in ("bullish", "bearish")
            assert ev.event_type == "BOS"

    def test_choch_direction_values(self, engine, random_df):
        result = engine.analyze(random_df)
        for ev in result.choch_events:
            assert ev.direction in ("bullish", "bearish")
            assert ev.event_type == "CHoCH"

    def test_bos_price_in_range(self, engine, random_df):
        result = engine.analyze(random_df)
        data_min = float(random_df["low"].min())
        data_max = float(random_df["high"].max())
        for ev in result.bos_events:
            assert data_min <= ev.price <= data_max


# ── Higher/Lower swing categories ────────────────────────────────────────────

class TestSwingCategories:
    def test_bullish_has_hh_hl(self, engine, bullish_df):
        result = engine.analyze(bullish_df)
        # In a clear bullish trend there should be some HH or HL
        total_bull = len(result.higher_highs) + len(result.higher_lows)
        total_bear = len(result.lower_highs) + len(result.lower_lows)
        assert total_bull > total_bear, "Bullish trend should have more HH/HL"

    def test_bearish_has_lh_ll(self, engine, bearish_df):
        result = engine.analyze(bearish_df)
        total_bull = len(result.higher_highs) + len(result.higher_lows)
        total_bear = len(result.lower_highs) + len(result.lower_lows)
        assert total_bear > total_bull, "Bearish trend should have more LH/LL"


# ── Narrative ────────────────────────────────────────────────────────────────

class TestNarrative:
    def test_narrative_is_string(self, engine, random_df):
        result = engine.analyze(random_df)
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 10

    def test_narrative_non_empty_on_bullish(self, engine, bullish_df):
        result = engine.analyze(bullish_df)
        assert "bullish" in result.narrative.lower() or "uptrend" in result.narrative.lower()


# ── Full pipeline smoke test ─────────────────────────────────────────────────

class TestFullPipeline:
    def test_analyze_returns_result(self, engine, random_df):
        result = engine.analyze(random_df)
        assert result is not None
        assert result.structure_type is not None

    def test_no_exceptions_on_minimal_data(self, engine):
        df = generate_xauusd_ohlcv(n_bars=25, seed=1)
        try:
            result = engine.analyze(df)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Raised {e} on minimal data")
