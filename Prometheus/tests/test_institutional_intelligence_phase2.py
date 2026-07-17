import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.core.decision_replay_counterfactual_intelligence import (
    build_decision_replay_counterfactual_intelligence,
)
from olympus.core.institutional_capital_intelligence import (
    build_capital_intelligence,
)
from olympus.core.institutional_knowledge_graph import (
    build_institutional_knowledge_graph,
    has_observed_decision_path,
)
from olympus.core.institutional_learning_scientist import (
    build_institutional_learning_scientist,
)
from olympus.core.institutional_decision_intelligence_platform import (
    build_institutional_decision_intelligence_platform,
    write_idip_artifacts,
)


def _closed_trade_samples() -> list[dict]:
    return [
        {
            "trade_id": "ph2-001",
            "signal_id": "sig-001",
            "direction": "long",
            "entry_price": 2400.0,
            "exit_price": 2406.0,
            "opened_at": "2026-07-13T08:00:00+00:00",
            "closed_at": "2026-07-13T08:30:00+00:00",
            "exit_reason": "tp",
            "mfe_pct": 0.42,
            "mae_pct": 0.12,
            "captured_return_pct": 78.0,
            "risk_utilization_pct": 27.0,
            "session": "London",
            "regime": "Trend Expansion",
            "pattern_name": "Liquidity Sweep Long",
            "signal_confidence": 0.81,
            "trend_state": "bullish",
            "volatility_state": "Moderate",
            "pnl": 1.2,
        },
        {
            "trade_id": "ph2-002",
            "signal_id": "sig-002",
            "direction": "short",
            "entry_price": 2410.0,
            "exit_price": 2412.0,
            "opened_at": "2026-07-13T09:00:00+00:00",
            "closed_at": "2026-07-13T09:20:00+00:00",
            "exit_reason": "micro_time_exit",
            "mfe_pct": 0.20,
            "mae_pct": 0.30,
            "captured_return_pct": 30.0,
            "risk_utilization_pct": 61.0,
            "session": "New York",
            "regime": "Volatility Expansion",
            "pattern_name": "CHOCH Reversal",
            "signal_confidence": 0.58,
            "trend_state": "sideways",
            "volatility_state": "High",
            "pnl": -0.4,
        },
    ]


def test_capital_intelligence_separates_ledgers_and_strategy_equity():
    payload = build_capital_intelligence(
        status={"start_balance": 1000.0},
        closed_trades=_closed_trade_samples(),
        account_events=[
            {"event_id": "evt-1", "event_type": "deposit", "amount": 500.0, "timestamp": "2026-07-13T10:00:00+00:00"},
            {"event_id": "evt-2", "event_type": "withdrawal", "amount": -200.0, "timestamp": "2026-07-13T10:05:00+00:00"},
        ],
    )

    summary = payload["summary"]
    assert summary["capital_injections"] == 500.0
    assert summary["capital_withdrawals"] == -200.0
    assert summary["strategy_equity"] != summary["raw_equity"]

    for row in payload["ledgers"]["capital_ledger"]:
        assert row["ml_eligible"] is False
        assert row["strategy_stat_eligible"] is False


def test_replay_engine_replays_all_completed_trades():
    replay = build_decision_replay_counterfactual_intelligence(closed_trades=_closed_trade_samples())
    assert replay["summary"]["completed_trade_replays"] == 2
    assert len(replay["replay_rows"]) == 2


def test_knowledge_graph_builds_decision_paths():
    graph = build_institutional_knowledge_graph(
        closed_trades=_closed_trade_samples(),
        attribution_rows=[{"trade_id": "ph2-001", "decision_score": 71.2}],
        version_seed="20260713120000",
    )
    assert graph["summary"]["path_count"] == 2
    path = [
        "structure:bullish",
        "pattern:Liquidity Sweep Long",
        "signal:sig-001",
        "entry:2400.0",
        "management:efficient",
        "exit:tp",
        "outcome:win",
        "lesson:71.2",
        "validated_knowledge:pending",
    ]
    assert has_observed_decision_path(graph, path) is True


def test_learning_scientist_outputs_required_sections(tmp_path: Path):
    storage = tmp_path / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "idip_history.jsonl").write_text("", encoding="utf-8")

    payload = build_institutional_learning_scientist(
        tmp_path,
        status={},
        closed_trades=_closed_trade_samples(),
        replay_rows=[{"trade_id": "ph2-001"}],
        zeus_reports=[{"evidence": {"confidence": 0.55}}],
        simulation_rows=[{"x": 1}],
        idip_payload={
            "summary": {"unknown_exit_count": 0},
            "engines": {
                "decision_attribution_intelligence": {"decision_attribution_rows": [{"decision_score": 72.5}]},
                "institutional_knowledge_intelligence": {"institutional_lessons": [{"knowledge_id": "IK-1"}]},
            },
        },
    )

    for key in [
        "institutional_learning",
        "hypotheses",
        "knowledge_growth",
        "learning_velocity",
        "research_queue",
        "concept_drift",
    ]:
        assert key in payload


def test_idip_writes_phase2_subsystem_artifacts(tmp_path: Path):
    storage = tmp_path / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "event_lineage.jsonl").write_text("", encoding="utf-8")
    (storage / "zeus_validation_reports.jsonl").write_text("", encoding="utf-8")

    payload = build_institutional_decision_intelligence_platform(
        tmp_path,
        status={"source_system": "hermes", "asset": "XAUUSDm", "timeframe": "M5", "start_balance": 1000.0},
        open_trades=[],
        closed_trades=_closed_trade_samples(),
    )
    artifact_paths = write_idip_artifacts(tmp_path, payload)

    assert "subsystem_artifacts" in artifact_paths
    subs = artifact_paths["subsystem_artifacts"]
    assert "institutional_learning" in subs
    assert "capital_intelligence" in subs
    assert "knowledge_graph" in subs
    assert "decision_replay" in subs

    for k in [
        "institutional_learning.json",
        "hypotheses.json",
        "knowledge_growth.json",
        "learning_velocity.json",
        "research_queue.json",
        "concept_drift.json",
        "capital_intelligence_runtime.json",
        "institutional_knowledge_graph.json",
        "decision_replay_counterfactual.json",
    ]:
        assert (storage / k).exists(), k
