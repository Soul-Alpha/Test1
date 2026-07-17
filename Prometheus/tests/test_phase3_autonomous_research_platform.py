import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.core.autonomous_research_orchestrator import build_autonomous_research_orchestrator
from olympus.core.explainability_engine import build_explainability_engine
from olympus.core.institutional_decision_intelligence_platform import (
    build_institutional_decision_intelligence_platform,
    write_idip_artifacts,
)
from olympus.core.institutional_research_director import build_institutional_research_director
from olympus.core.knowledge_coverage_intelligence import build_knowledge_coverage_intelligence
from olympus.core.knowledge_evolution_engine import build_knowledge_evolution_engine
from olympus.core.meta_learning_engine import build_meta_learning_engine


def _closed_trade_samples() -> list[dict]:
    return [
        {
            "trade_id": "ph3-001",
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
            "trade_id": "ph3-002",
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


def test_phase3_engines_generate_payloads(tmp_path: Path):
    storage = tmp_path / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "idip_history.jsonl").write_text("", encoding="utf-8")
    (storage / "research_queue.json").write_text(json.dumps({"rows": []}), encoding="utf-8")

    idip = {
        "summary": {"expectancy": 0.1, "decision_quality": 66.0, "risk_efficiency": 55.0, "sample_size": 2},
        "engines": {
            "pattern_lifecycle_intelligence": {"pattern_lifecycle_profiles": [{"pattern": "p1"}]},
            "institutional_knowledge_intelligence": {"institutional_lessons": [{"knowledge_id": "IK-1", "decision_quality": 70.0}]},
            "decision_replay_counterfactual_intelligence": {"summary": {"completed_trade_replays": 2}},
        },
    }

    meta = build_meta_learning_engine(tmp_path, idip_payload=idip)
    assert "metrics" in meta

    coverage = build_knowledge_coverage_intelligence(status={"closed_trades": _closed_trade_samples()}, idip_payload=idip)
    assert "coverage_rows" in coverage and len(coverage["coverage_rows"]) > 0

    aro = build_autonomous_research_orchestrator(
        tmp_path,
        hypotheses_rows=[{"hypothesis_id": "H1", "statement": "s1", "confidence": 0.5, "sample_size": 10}],
        recommendation_rows=[{"recommendation_id": "R1", "recommendation": "r1", "validation_domain": "recommendation", "evidence": {"confidence": 0.6, "sample_size": 20}}],
        knowledge_gap_rows=coverage["coverage_rows"][:2],
    )
    assert len(aro.get("zeus_submission_queue", [])) > 0

    rd = build_institutional_research_director(aro_payload=aro, hypotheses_rows=[])
    assert "merged_backlog" in rd

    evolution = build_knowledge_evolution_engine(tmp_path, idip_payload=idip)
    assert "knowledge_objects" in evolution

    explain = build_explainability_engine(
        recommendation_rows=[{"recommendation_id": "r1", "recommendation": "test", "evidence": {"confidence": 0.7, "sample_size": 30}}],
        context={"expectancy": 0.1, "decision_quality": 66.0, "risk_efficiency": 55.0},
    )
    assert explain["summary"]["explained_recommendations"] == 1


def test_phase3_idip_integration_and_artifacts(tmp_path: Path):
    storage = tmp_path / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "event_lineage.jsonl").write_text("", encoding="utf-8")
    (storage / "zeus_validation_reports.jsonl").write_text("", encoding="utf-8")
    (storage / "research_queue.json").write_text(json.dumps({"rows": []}), encoding="utf-8")

    payload = build_institutional_decision_intelligence_platform(
        tmp_path,
        status={"source_system": "hermes", "asset": "XAUUSDm", "timeframe": "M5", "start_balance": 1000.0, "closed_trades": _closed_trade_samples()},
        open_trades=[],
        closed_trades=_closed_trade_samples(),
    )

    engines = payload.get("engines", {})
    for key in [
        "meta_learning_engine",
        "autonomous_research_orchestrator",
        "knowledge_coverage_intelligence",
        "knowledge_evolution_engine",
        "explainability_engine",
        "institutional_research_director",
    ]:
        assert key in engines

    assert payload.get("meta", {}).get("zeus_validation_required") is True

    paths = write_idip_artifacts(tmp_path, payload)
    subs = paths.get("subsystem_artifacts", {})
    for key in [
        "meta_learning",
        "aro",
        "knowledge_coverage",
        "knowledge_evolution",
        "explainability",
        "research_director",
    ]:
        assert key in subs

    for file_name in [
        "meta_learning_runtime.json",
        "autonomous_research_orchestrator_runtime.json",
        "knowledge_coverage_runtime.json",
        "knowledge_evolution_runtime.json",
        "explainability_runtime.json",
        "institutional_research_director_runtime.json",
    ]:
        assert (storage / file_name).exists(), file_name
