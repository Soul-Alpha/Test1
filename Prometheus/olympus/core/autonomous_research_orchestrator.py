from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olympus.core.research_prioritization_engine import build_research_prioritization_engine

ARO_VERSION = "aro-v1.0"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_autonomous_research_orchestrator(
    root_dir: Path,
    *,
    hypotheses_rows: list[dict[str, Any]],
    recommendation_rows: list[dict[str, Any]],
    knowledge_gap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    storage = root_dir / "storage" / "olympus"

    queue = _load_json(storage / "research_queue.json", {})
    queued = queue.get("rows", []) if isinstance(queue, dict) else []

    candidates = []
    for h in hypotheses_rows:
        candidates.append(
            {
                "recommendation_id": f"aro-h-{str(h.get('hypothesis_id','unknown')).lower()}",
                "recommendation": h.get("statement", "Hypothesis-driven research"),
                "validation_domain": "recommendation",
                "expected_improvement": h.get("research_target", "Evidence pending"),
                "evidence": {
                    "confidence": h.get("confidence", 0.35),
                    "sample_size": max(1, int(h.get("sample_size", 1) or 1)),
                    "supporting_value": 0.15,
                },
                "knowledge_gap_severity": 0.6,
                "historical_importance": 0.55,
                "portfolio_impact": 0.55,
                "complexity": 0.45,
                "risk": 0.35,
            }
        )

    for r in recommendation_rows:
        candidates.append(
            {
                "recommendation_id": r.get("recommendation_id"),
                "recommendation": r.get("recommendation"),
                "validation_domain": r.get("validation_domain", "recommendation"),
                "expected_improvement": r.get("expected_improvement", "Evidence pending"),
                "evidence": r.get("evidence", {}),
                "knowledge_gap_severity": 0.7,
                "historical_importance": 0.65,
                "portfolio_impact": 0.6,
                "complexity": 0.5,
                "risk": 0.45,
            }
        )

    for g in knowledge_gap_rows:
        candidates.append(
            {
                "recommendation_id": f"aro-gap-{str(g.get('gap_id','unknown')).lower()}",
                "recommendation": g.get("research_recommendation", "Close knowledge gap"),
                "validation_domain": "pattern",
                "expected_improvement": "Increase coverage and reduce blind spots",
                "evidence": {"confidence": g.get("confidence", 0.35), "sample_size": g.get("missing_samples", 1), "supporting_value": g.get("severity", 0.2)},
                "knowledge_gap_severity": g.get("severity", 0.5),
                "historical_importance": 0.5,
                "portfolio_impact": 0.5,
                "complexity": 0.4,
                "risk": 0.3,
            }
        )

    dedupe = {}
    for c in candidates:
        rid = str(c.get("recommendation_id") or "")
        if rid:
            dedupe[rid] = c

    prioritized = build_research_prioritization_engine(candidates=list(dedupe.values()))
    roadmap = prioritized.get("prioritized_research_roadmap", [])

    for item in roadmap:
        item["research_lifecycle"] = "candidate"
        item["zeus_submission_ready"] = True
        item["estimated_validation_cost"] = round(max(0.01, item.get("required_samples", 1) / 500.0), 4)
        item["expected_value"] = round(item.get("priority_score", 0.0) * (1.0 + item.get("confidence", 0.0)), 6)

    payload = {
        "version": ARO_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "backlog_input_size": len(queued),
        "candidates": list(dedupe.values()),
        "prioritized_roadmap": roadmap,
        "research_lifecycle": {
            "candidate": len(roadmap),
            "submitted_to_zeus": len(roadmap),
            "validated": 0,
            "adopted": 0,
            "retired": 0,
        },
        "research_roi": {
            "estimated_value_sum": round(sum(float(x.get("expected_value", 0.0)) for x in roadmap), 6),
            "estimated_cost_sum": round(sum(float(x.get("estimated_validation_cost", 0.0)) for x in roadmap), 6),
        },
        "zeus_submission_queue": [
            {
                "recommendation_id": x.get("recommendation_id"),
                "recommendation": x.get("recommendation"),
                "validation_domain": x.get("validation_domain", "recommendation"),
                "priority": x.get("priority"),
                "priority_score": x.get("priority_score"),
                "confidence": x.get("confidence"),
                "required_samples": x.get("required_samples"),
                "lifecycle": "candidate",
                "operator_approved": False,
                "requires_zeus_validation": True,
                "governance_required": True,
                "timestamp": generated_at,
            }
            for x in roadmap[:200]
        ],
    }
    return payload


def write_aro_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "autonomous_research_orchestrator_runtime.json"
    history = storage / "autonomous_research_orchestrator_history.jsonl"

    runtime.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"aro_runtime": str(runtime), "aro_history": str(history)}
