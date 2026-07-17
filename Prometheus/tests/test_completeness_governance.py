"""Tests for analysis completeness governance (Phase 1.1).

Verifies that:
- Every engine run (success or failure) is recorded in EngineCompleteness.
- A fully-successful run reports is_complete=True and completeness_ratio=1.0.
- A run with simulated engine failures reports is_complete=False, captures
  failure messages, and does not silently appear complete.
- _report_to_dict always includes the completeness dict.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prometheus_core import EngineCompleteness, Prometheus, PrometheusResult
from data.sample_data import generate_trending_data


# ── Unit tests for EngineCompleteness ─────────────────────────────────────────


def test_engine_completeness_all_success():
    ec = EngineCompleteness()
    for name in ("ms", "sr", "fib", "cs", "pat", "smc", "mtf", "vwap", "amd"):
        ec.record_success(name)
    assert ec.is_complete is True
    assert ec.engines_succeeded == 9
    assert ec.engines_failed == 0
    assert ec.completeness_ratio == 1.0


def test_engine_completeness_partial_failure():
    ec = EngineCompleteness()
    ec.record_success("ms")
    ec.record_success("sr")
    ec.record_failure("fib", ValueError("test error"))
    ec.record_success("cs")

    assert ec.is_complete is False
    assert ec.engines_succeeded == 3
    assert ec.engines_failed == 1
    assert round(ec.completeness_ratio, 4) == round(3 / 4, 4)
    assert "fib" in ec.engine_errors
    assert "test error" in ec.engine_errors["fib"]


def test_engine_completeness_empty():
    ec = EngineCompleteness()
    assert ec.is_complete is False
    assert ec.completeness_ratio == 0.0
    assert ec.engines_run == 0


def test_completeness_to_dict():
    ec = EngineCompleteness()
    ec.record_success("ms")
    ec.record_failure("sr", RuntimeError("boom"))
    d = ec.to_dict()

    assert d["engines_run"] == 2
    assert d["engines_succeeded"] == 1
    assert d["engines_failed"] == 1
    assert d["is_complete"] is False
    assert 0.0 < d["completeness_ratio"] < 1.0
    assert "sr" in d["engine_errors"]


# ── Integration tests: completeness tracking in a full analysis run ────────────


@pytest.fixture
def small_df():
    return generate_trending_data(n_bars=200, direction="bullish")


def test_full_run_completeness_populated(small_df):
    """A successful analysis run must record all 9 engine statuses."""
    bot = Prometheus()
    result = bot.analyze_data(small_df, asset="XAUUSD", timeframe="4H",
                              render_chart=False, save_to_db=False)

    assert result.completeness.engines_run == 9
    assert result.completeness.engines_succeeded == 9
    assert result.completeness.is_complete is True


def test_degraded_run_is_not_silent(small_df):
    """When an engine raises, completeness.is_complete must be False."""
    bot = Prometheus()

    def _boom(_df, **kwargs):
        raise RuntimeError("simulated engine failure")

    with patch.object(bot.ms_engine, "analyze", side_effect=_boom):
        result = bot.analyze_data(small_df, asset="XAUUSD", timeframe="4H",
                                  render_chart=False, save_to_db=False)

    assert result.completeness.is_complete is False
    assert result.completeness.engines_failed >= 1
    assert "market_structure" in result.completeness.engine_errors


def test_report_to_dict_includes_completeness(small_df):
    """_report_to_dict must always embed the completeness governance block."""
    bot = Prometheus()
    result = bot.analyze_data(small_df, asset="XAUUSD", timeframe="4H",
                              render_chart=False, save_to_db=False)

    d = bot._report_to_dict(result)
    assert "completeness" in d
    assert "is_complete" in d["completeness"]
    assert "completeness_ratio" in d["completeness"]
    assert "engine_statuses" in d["completeness"]
    assert "engine_errors" in d["completeness"]
