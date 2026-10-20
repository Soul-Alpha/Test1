from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from olympus.contracts import BenchmarkCandidate, BenchmarkLimits, BenchmarkObservation
from olympus.core.execution_benchmark import (
    ExecutionBenchmarkLedger, build_benchmark_report, candidate_fingerprint,
)

START = "2026-10-01T00:00:00+00:00"
END = "2026-10-04T00:00:00+00:00"


def candidate(source: str = "hermes") -> BenchmarkCandidate:
    return BenchmarkCandidate(
        "candidate-1", source, "model-1", "features-1", "strategy-1", "dataset-1",
        "XAUUSDm", "M5", "USD", "test-broker-demo", '{"risk":0.01}',
        START, 100.0, BenchmarkLimits(2, 0.0, 1.1, 10.0, 200.0, 5.0),
    )


def observation(c: BenchmarkCandidate, signal: str = "signal-1",
                stage: str = "paper", **changes) -> BenchmarkObservation:
    row = BenchmarkObservation(
        f"{stage}-{signal}", c.candidate_id, candidate_fingerprint(c), c.source_system,
        stage, signal, "2026-10-02T00:00:00+00:00", "buy", "closed",
        intended_entry=100.0, intended_size=0.01, intended_stop=90.0, intended_target=120.0,
        filled_at="2026-10-02T00:00:01+00:00", actual_entry=101.0, actual_size=0.01,
        actual_stop=90.0, actual_target=120.0, closed_at="2026-10-02T01:00:00+00:00",
        actual_exit=110.0, gross_pnl=11.0, commission_cost=1.0, swap_cost=0.0,
        other_cost=0.0, spread_bps=2.0, exit_reason="target",
    )
    return replace(row, **changes)


def ledger_at(tmp_path: Path, c: BenchmarkCandidate) -> ExecutionBenchmarkLedger:
    ledger = ExecutionBenchmarkLedger(tmp_path / "benchmark.sqlite3")
    ledger.register(c)
    return ledger


def report(ledger: ExecutionBenchmarkLedger, source: str = "hermes"):
    return build_benchmark_report(ledger, source, "candidate-1", START, END)


def test_candidate_and_observations_are_immutable_and_idempotent(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    assert ledger.register(c) == candidate_fingerprint(c)
    with pytest.raises(ValueError, match="frozen"):
        ledger.register(replace(c, model_version="model-2"))
    row = observation(c)
    assert ledger.append(row)
    assert not ledger.append(row)
    with pytest.raises(ValueError, match="Conflicting"):
        ledger.append(replace(row, gross_pnl=99))
    with pytest.raises(ValueError, match="Conflicting"):
        ledger.append(replace(row, observation_id="new-id"))
    assert len(ledger.observations("hermes", c.candidate_id)) == 1
    with sqlite3.connect(ledger.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM observations")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE candidates SET payload='{}'")


def test_bot_identity_and_candidate_fingerprint_are_enforced(tmp_path: Path) -> None:
    hermes, prometheus = candidate(), candidate("prometheus")
    ledger = ledger_at(tmp_path, hermes)
    ledger.register(prometheus)
    ledger.append(observation(hermes))
    ledger.append(observation(prometheus))
    with pytest.raises(ValueError, match="fingerprint"):
        ledger.append(observation(hermes, "bad", candidate_fingerprint=candidate_fingerprint(prometheus)))
    assert len(ledger.observations("hermes", hermes.candidate_id)) == 1
    assert len(ledger.observations("prometheus", prometheus.candidate_id)) == 1


def test_net_metrics_include_costs_and_realized_drawdown(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    ledger.append(observation(c))
    ledger.append(observation(c, "signal-2", gross_pnl=-4.0,
                              closed_at="2026-10-02T02:00:00+00:00"))
    result = report(ledger)
    paper = result.stages[1]
    assert paper.net_pnl == pytest.approx(5.0)
    assert paper.expectancy == pytest.approx(2.5)
    assert paper.profit_factor == pytest.approx(2.0)
    assert paper.realized_drawdown_pct == pytest.approx(5 / 110 * 100)
    assert not paper.blockers
    assert result.no_auto_adopt and not result.approved_for_adoption
    assert "minimum_closed_trades" in result.stages[2].blockers
    json.dumps(result.as_dict(), allow_nan=False)


def test_unknown_costs_cannot_become_zero_or_pass_checks(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    ledger.append(observation(c, commission_cost=None))
    row = report(ledger).stages[1]
    assert row.closed_trades == 1 and row.complete_trades == 0
    assert row.net_pnl is None and row.expectancy is None
    assert "unknown_trade_fields_or_costs" in row.blockers


@pytest.mark.parametrize("changes", [
    {"decision_at": "2026-09-01T00:00:00+00:00"},
    {"filled_at": "2026-10-01T00:00:00+00:00"},
    {"closed_at": "2026-10-01T00:00:00+00:00"},
    {"decision_at": "2026-10-02T00:00:00"},
    {"gross_pnl": float("nan")},
    {"commission_cost": -1.0},
    {"actual_size": 0.0},
])
def test_invalid_observations_are_rejected(tmp_path: Path, changes: dict) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    with pytest.raises(ValueError):
        ledger.append(observation(c, **changes))


def test_historical_backtests_may_precede_freeze(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    assert ledger.append(observation(
        c, stage="backtest", decision_at="2025-01-01T00:00:00Z",
        filled_at="2025-01-01T00:00:01Z", closed_at="2025-01-01T01:00:00Z",
    ))


def test_matching_requires_identical_signal_intent_and_reports_unmatched(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    ledger.append(observation(c))
    ledger.append(observation(c, stage="live", actual_entry=102.0, gross_pnl=8.0))
    ledger.append(observation(c, "paper-only"))
    ledger.append(observation(c, "incompatible"))
    ledger.append(observation(c, "incompatible", "live", intended_size=0.02))
    comparison = report(ledger).comparisons[1]
    assert comparison.matched_signals == 1
    assert comparison.incompatible_signals == 1
    assert comparison.unmatched_left == 1 and comparison.unmatched_right == 0
    assert comparison.mean_entry_difference_bps == pytest.approx(1 / 101 * 10000)
    assert comparison.mean_net_pnl_difference == pytest.approx(-3)
    assert comparison.pnl_pairs == 1


def test_rejections_preserve_reason_without_fabricating_outcomes(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    row = observation(c, status="rejected", filled_at=None, actual_entry=None,
                      actual_size=None, closed_at=None, actual_exit=None, gross_pnl=None,
                      rejection_reason="spread limit")
    assert ledger.append(row)
    assert report(ledger).stages[1].closed_trades == 0
    with pytest.raises(ValueError, match="reason"):
        ledger.append(replace(row, observation_id="r2", signal_id="r2", rejection_reason=None))


def test_future_closed_outcomes_do_not_leak_into_window(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    ledger.append(observation(c, closed_at="2026-10-05T00:00:00Z"))
    stage = report(ledger).stages[1]
    assert stage.observations == 1 and stage.complete_trades == 0
    assert stage.net_pnl is None
    assert "outcomes_outside_evaluation_window" in stage.blockers


def test_all_wins_remain_json_safe_and_do_not_authorize_adoption(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    ledger.append(observation(c))
    ledger.append(observation(c, "signal-2"))
    result = report(ledger)
    assert result.stages[1].profit_factor is None
    assert result.stages[1].no_losing_trades
    assert result.approved_for_adoption is False
    json.dumps(result.as_dict(), allow_nan=False)


def test_sell_slippage_and_tail_execution_limits(tmp_path: Path) -> None:
    c = candidate()
    ledger = ledger_at(tmp_path, c)
    ledger.append(observation(c, side="sell", actual_entry=97.0,
                              filled_at="2026-10-02T00:00:10+00:00"))
    row = report(ledger).stages[1]
    assert row.mean_adverse_slippage_bps == pytest.approx(300)
    assert "adverse_slippage" in row.blockers
    assert "execution_latency" in row.blockers


def test_missing_read_does_not_create_database(tmp_path: Path) -> None:
    ledger = ExecutionBenchmarkLedger(tmp_path / "missing.sqlite3")
    assert ledger.candidates() == []
    assert not ledger.path.exists()


def test_concurrent_exact_replay_commits_once(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    c = candidate()
    ledger = ledger_at(tmp_path, c)
    row = observation(c)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ledger.append(row), range(8)))
    assert sum(results) == 1
    assert len(ledger.observations(c.source_system, c.candidate_id)) == 1


def test_cli_registration_import_and_report(tmp_path: Path, monkeypatch, capsys) -> None:
    from dataclasses import asdict
    from olympus.core.execution_benchmark_cli import main

    database = tmp_path / "benchmark.sqlite3"
    c = candidate()
    candidate_file = tmp_path / "candidate.json"
    candidate_file.write_text(json.dumps(asdict(c)), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["benchmark", "--database", str(database),
                                    "register", str(candidate_file)])
    main()
    assert json.loads(capsys.readouterr().out)["candidate_fingerprint"] == candidate_fingerprint(c)
    observations_file = tmp_path / "outcomes.jsonl"
    observations_file.write_text(json.dumps(asdict(observation(c))) + "\n", encoding="utf-8")
    for expected in ({"added": 1, "exact_replays": 0}, {"added": 0, "exact_replays": 1}):
        monkeypatch.setattr("sys.argv", ["benchmark", "--database", str(database),
                                        "observe", str(observations_file)])
        main()
        assert json.loads(capsys.readouterr().out) == expected
    monkeypatch.setattr("sys.argv", ["benchmark", "--database", str(database), "report",
                                    "--source", "hermes", "--candidate", c.candidate_id,
                                    "--start", START, "--end", END])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["no_auto_adopt"] is True
    assert result["stages"][1]["net_pnl"] == 10.0


def test_benchmark_section_renders_empty_and_populated_ledgers(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    script = (
        "from pathlib import Path\n"
        "from ui.execution_benchmark_section import render_execution_benchmark\n"
        f"render_execution_benchmark(Path({str(tmp_path)!r}))\n"
    )
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert "No benchmark candidates" in app.info[0].value
    assert not (tmp_path / "storage").exists()

    from config import CONFIG
    ledger = ExecutionBenchmarkLedger(
        tmp_path / "storage" / "olympus" / CONFIG.execution_benchmark.database_name
    )
    c = candidate()
    ledger.register(c)
    ledger.append(observation(c))
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert len(app.dataframe) == 3
