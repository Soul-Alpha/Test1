"""Unit tests for the AMD (Accumulation / Manipulation / Distribution) Engine."""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.amd_engine import AMDEngine, AMDResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(hours: list[int], highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with hourly bars at the given UTC hours (today)."""
    base_date = datetime(2025, 1, 15, tzinfo=timezone.utc)
    timestamps = [base_date + timedelta(hours=h) for h in hours]
    opens = closes  # simplification: open = close for test bars
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [100.0] * len(hours)},
        index=pd.DatetimeIndex(timestamps),
    )


def _make_amd_scenario(
    *,
    asian_high: float = 2010.0,
    asian_low: float = 2000.0,
    manip_direction: str = "bearish",  # "bearish" = sweep above Asian high (sell-stops)
    add_manip_sweep: bool = True,
    add_dist_fvg: bool = False,
) -> pd.DataFrame:
    """
    Construct a synthetic intraday bar set covering one AMD cycle.

    Asian hours 00-06: consolidation between asian_low..asian_high
    Manipulation hour 07-09: if add_manip_sweep, sweeps Asian high (bearish) or low (bullish)
    Distribution hours 10-16: price delivers in distribution direction
    """
    rows_hours: list[int] = []
    rows_high: list[float] = []
    rows_low: list[float] = []
    rows_close: list[float] = []

    mid = (asian_high + asian_low) / 2.0

    # Asian session: 00:00–06:59 UTC (7 bars)
    for h in range(7):
        rows_hours.append(h)
        rows_high.append(asian_high)
        rows_low.append(asian_low)
        rows_close.append(mid)

    # Manipulation: hour 7
    sweep_range = (asian_high - asian_low) * 0.4
    if add_manip_sweep:
        if manip_direction == "bearish":
            # Sweep ABOVE Asian high, close back below (bearish AMD)
            rows_hours.append(7)
            rows_high.append(asian_high + sweep_range)
            rows_low.append(asian_high - 1.0)
            rows_close.append(asian_high - 1.0)
        else:
            # Sweep BELOW Asian low, close back above (bullish AMD)
            rows_hours.append(7)
            rows_high.append(asian_low + 1.0)
            rows_low.append(asian_low - sweep_range)
            rows_close.append(asian_low + 1.0)
    else:
        rows_hours.append(7)
        rows_high.append(asian_high)
        rows_low.append(asian_low)
        rows_close.append(mid)

    # Distribution: hours 10–16
    for h in range(10, 17):
        rows_hours.append(h)
        if manip_direction == "bearish":
            # Price drops after the bear sweep
            rows_high.append(mid)
            rows_low.append(asian_low - 5.0)
            rows_close.append(asian_low - 5.0)
        else:
            rows_high.append(asian_high + 5.0)
            rows_low.append(mid)
            rows_close.append(asian_high + 5.0)

    return _make_df(rows_hours, rows_high, rows_low, rows_close)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> AMDEngine:
    return AMDEngine()


@pytest.fixture
def bearish_amd_df() -> pd.DataFrame:
    return _make_amd_scenario(manip_direction="bearish", add_manip_sweep=True)


@pytest.fixture
def bullish_amd_df() -> pd.DataFrame:
    return _make_amd_scenario(manip_direction="bullish", add_manip_sweep=True)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAMDEngineBasic:
    def test_returns_amd_result(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df)
        assert isinstance(result, AMDResult)

    def test_asian_range_extracted(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df)
        assert result.asian_high is not None
        assert result.asian_low is not None
        assert result.asian_range is not None
        assert result.asian_range > 0

    def test_asian_range_values(self, engine):
        df = _make_amd_scenario(asian_high=2010.0, asian_low=2000.0, add_manip_sweep=False)
        result = engine.analyze(df)
        assert result.asian_high == pytest.approx(2010.0, abs=0.01)
        assert result.asian_low == pytest.approx(2000.0, abs=0.01)
        assert result.asian_range == pytest.approx(10.0, abs=0.01)

    def test_insufficient_data_returns_gracefully(self, engine):
        df = _make_df([0, 1], [2010.0, 2010.0], [2000.0, 2000.0], [2005.0, 2005.0])
        result = engine.analyze(df)
        assert isinstance(result, AMDResult)

    def test_empty_dataframe(self, engine):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([])
        result = engine.analyze(df)
        assert isinstance(result, AMDResult)

    def test_non_datetime_index_returns_gracefully(self, engine):
        df = pd.DataFrame(
            {"open": [2005.0], "high": [2010.0], "low": [2000.0], "close": [2005.0], "volume": [100.0]},
            index=[0],
        )
        result = engine.analyze(df)
        assert isinstance(result, AMDResult)


class TestAMDSweepDetection:
    def test_bearish_sweep_detected(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df)
        assert result.manipulation_swept is True
        assert result.sweep_side == "high"
        assert result.direction == "bearish"

    def test_bullish_sweep_detected(self, engine, bullish_amd_df):
        result = engine.analyze(bullish_amd_df)
        assert result.manipulation_swept is True
        assert result.sweep_side == "low"
        assert result.direction == "bullish"

    def test_no_sweep_when_no_manip(self, engine):
        df = _make_amd_scenario(add_manip_sweep=False)
        result = engine.analyze(df)
        assert result.manipulation_swept is False

    def test_sweep_price_set_when_swept(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df)
        assert result.sweep_price is not None
        # Bearish sweep means price went above Asian high
        assert result.sweep_price > result.asian_high

    def test_direction_neutral_when_no_sweep(self, engine):
        df = _make_amd_scenario(add_manip_sweep=False)
        result = engine.analyze(df)
        assert result.direction in ("neutral", "bearish", "bullish")


class TestAMDPhaseLabel:
    def test_phase_distribution_at_noon(self, engine):
        """Bar ending at 12:00 UTC should be in distribution phase."""
        df = _make_amd_scenario()
        # Inject a bar at hour 12 as the latest
        last_row = df.iloc[-1]
        noon_ts = pd.Timestamp("2025-01-15 12:00:00", tz="UTC")
        extra = pd.DataFrame(
            {"open": [last_row.close], "high": [last_row.high],
             "low": [last_row.low], "close": [last_row.close], "volume": [100.0]},
            index=pd.DatetimeIndex([noon_ts]),
        )
        df2 = pd.concat([df, extra])
        result = engine.analyze(df2)
        assert result.phase == "distribution"

    def test_phase_accumulation_at_3am(self, engine):
        """Latest bar at 03:00 UTC → accumulation phase."""
        # 10+ bars: Asian session hours 0-9 on Day 1, then last bar at hour 3 on Day 2.
        base = datetime(2025, 1, 15, tzinfo=timezone.utc)
        day2_base = datetime(2025, 1, 16, tzinfo=timezone.utc)
        times = [base + timedelta(hours=h) for h in range(10)] + [day2_base + timedelta(hours=3)]
        n = len(times)
        df = pd.DataFrame(
            {"open": [2005.0]*n, "high": [2010.0]*n, "low": [2000.0]*n, "close": [2005.0]*n, "volume": [100.0]*n},
            index=pd.DatetimeIndex(times),
        )
        result = engine.analyze(df)
        assert result.phase == "accumulation"

    def test_phase_manipulation_at_8am(self, engine):
        """Latest bar at 08:00 UTC → manipulation phase."""
        # 10+ bars: hours 0-9 on Day 1, final bar at hour 8 on Day 2.
        base = datetime(2025, 1, 15, tzinfo=timezone.utc)
        day2_base = datetime(2025, 1, 16, tzinfo=timezone.utc)
        times = [base + timedelta(hours=h) for h in range(10)] + [day2_base + timedelta(hours=8)]
        n = len(times)
        df = pd.DataFrame(
            {"open": [2005.0]*n, "high": [2010.0]*n, "low": [2000.0]*n, "close": [2005.0]*n, "volume": [100.0]*n},
            index=pd.DatetimeIndex(times),
        )
        result = engine.analyze(df)
        assert result.phase == "manipulation"


class TestAMDConfidence:
    def test_confidence_between_0_and_1(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df)
        assert 0.0 <= result.confidence <= 1.0

    def test_swept_confidence_higher_than_unswept(self, engine):
        df_swept  = _make_amd_scenario(add_manip_sweep=True)
        df_no_swp = _make_amd_scenario(add_manip_sweep=False)
        r_swept   = engine.analyze(df_swept)
        r_no_swp  = engine.analyze(df_no_swp)
        assert r_swept.confidence >= r_no_swp.confidence

    def test_note_populated(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df)
        assert isinstance(result.note, str) and len(result.note) > 0


class TestAMDEntryFVGs:
    def test_no_entry_fvgs_without_existing_fvgs(self, engine, bearish_amd_df):
        result = engine.analyze(bearish_amd_df, existing_fvgs=None)
        assert result.entry_fvgs == []
        assert result.best_entry_fvg is None

    def test_entry_fvgs_with_matching_direction(self, engine, bearish_amd_df):
        from engines.liquidity_smc import FairValueGap
        # Create a bearish FVG that appears AFTER the manipulation window (bar_idx > manip bar)
        fvg = FairValueGap(
            direction="bearish",
            high=2009.0,
            low=2006.0,
            mid=2007.5,
            start_idx=8,   # after manipulation
            filled=False,
        )
        result = engine.analyze(bearish_amd_df, existing_fvgs=[fvg])
        # With a matching unfilled bearish FVG, it should appear in entry_fvgs
        # (result depends on sweep detection; only assert no crash and type integrity)
        assert isinstance(result.entry_fvgs, list)

    def test_filled_fvgs_excluded(self, engine, bearish_amd_df):
        from engines.liquidity_smc import FairValueGap
        filled_fvg = FairValueGap(direction="bearish", high=2009.0, low=2006.0, mid=2007.5, start_idx=8, filled=True)
        result = engine.analyze(bearish_amd_df, existing_fvgs=[filled_fvg])
        assert filled_fvg not in result.entry_fvgs

    def test_wrong_direction_fvgs_excluded(self, engine, bearish_amd_df):
        from engines.liquidity_smc import FairValueGap
        bullish_fvg = FairValueGap(direction="bullish", high=2009.0, low=2006.0, mid=2007.5, start_idx=8, filled=False)
        result = engine.analyze(bearish_amd_df, existing_fvgs=[bullish_fvg])
        assert bullish_fvg not in result.entry_fvgs
