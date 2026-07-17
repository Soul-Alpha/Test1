from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRIORITIZATION_VERSION = "research-prioritization-v1.0"


def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, str) and not v.strip():
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _score(row: dict[str, Any]) -> float:
    expected_return_improvement = _safe_float((row.get("evidence", {}) or {}).get("supporting_value"))
    drawdown_reduction = _safe_float(row.get("drawdown_reduction", 0.0))
    knowledge_gap = _safe_float(row.get("knowledge_gap_severity", 0.5))
    confidence = _safe_float((row.get("evidence", {}) or {}).get("confidence", 0.35))
    required_samples = max(1.0, _safe_float((row.get("evidence", {}) or {}).get("sample_size", 1)))
    historical_importance = _safe_float(row.get("historical_importance", 0.5))
    portfolio_impact = _safe_float(row.get("portfolio_impact", 0.5))
    complexity = _safe_float(row.get("complexity", 0.5))
    risk = _safe_float(row.get("risk", 0.5))

    return round(
        (expected_return_improvement * 0.22)
        + (drawdown_reduction * 0.15)
        + (knowledge_gap * 0.12)
        + (confidence * 0.14)
        + (historical_importance * 0.12)
        + (portfolio_impact * 0.13)
        - (complexity * 0.07)
        - (risk * 0.05)
        - ((required_samples / 1000.0) * 0.10),
        6,
    )


def build_research_prioritization_engine(*, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        score = _score(cand)
        priority = "High" if score >= 0.25 else "Medium" if score >= 0.08 else "Low"
        rows.append(
            {
                "recommendation_id": cand.get("recommendation_id"),
                "recommendation": cand.get("recommendation"),
                "validation_domain": cand.get("validation_domain", "recommendation"),
                "priority_score": score,
                "priority": priority,
                "confidence": _safe_float((cand.get("evidence", {}) or {}).get("confidence", 0.35)),
                "required_samples": int(max(1, _safe_float((cand.get("evidence", {}) or {}).get("sample_size", 1)))),
                "expected_improvement": cand.get("expected_improvement", "Evidence pending"),
            }
        )

    rows.sort(key=lambda x: x.get("priority_score", -9e9), reverse=True)

    return {
        "version": PRIORITIZATION_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "prioritized_research_roadmap": rows,
        "summary": {
            "backlog": len(rows),
            "high_priority": len([x for x in rows if x.get("priority") == "High"]),
            "medium_priority": len([x for x in rows if x.get("priority") == "Medium"]),
            "low_priority": len([x for x in rows if x.get("priority") == "Low"]),
        },
    }


def write_research_prioritization_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "research_prioritization_runtime.json"
    history = storage / "research_prioritization_history.jsonl"

    runtime.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"research_prioritization_runtime": str(runtime), "research_prioritization_history": str(history)}
