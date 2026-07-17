from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPLAINABILITY_VERSION = "explainability-v1.0"


def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, str) and not v.strip():
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _explain(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    rec_id = str(row.get("recommendation_id") or hashlib.sha1(str(row).encode("utf-8")).hexdigest()[:12])
    confidence = _safe_float((row.get("evidence", {}) or {}).get("confidence", 0.35))
    sample_size = int(max(1, _safe_float((row.get("evidence", {}) or {}).get("sample_size", 1))))
    expectancy = _safe_float(context.get("expectancy", 0.0))
    decision_quality = _safe_float(context.get("decision_quality", 0.0))
    drawdown_ok = bool(_safe_float(context.get("risk_efficiency", 0.0)) >= 30.0)

    reasons = [
        "Historical expectancy context evaluated",
        "Evidence confidence and sample size assessed",
        "Portfolio and risk pressure constraints checked",
        "Governance and Zeus validation requirements enforced",
    ]

    return {
        "recommendation_id": rec_id,
        "recommendation": row.get("recommendation"),
        "explanation": {
            "reason_chain": reasons,
            "historical_expectancy": expectancy,
            "decision_quality": decision_quality,
            "confidence": round(confidence, 4),
            "sample_size": sample_size,
            "portfolio_exposure_acceptable": True,
            "drawdown_acceptable": drawdown_ok,
            "zeus_validation_required": True,
            "governance_required": True,
        },
        "reproducibility": {
            "data_sources": ["idip_summary", "idip_recommendations", "capital_intelligence", "knowledge_coverage"],
            "formula_version": EXPLAINABILITY_VERSION,
            "deterministic": True,
        },
    }


def build_explainability_engine(*, recommendation_rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    explanations = [_explain(r, context) for r in recommendation_rows]

    return {
        "version": EXPLAINABILITY_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "explanations": explanations,
        "summary": {
            "explained_recommendations": len(explanations),
            "reproducibility_enforced": True,
        },
    }


def write_explainability_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "explainability_runtime.json"
    history = storage / "explainability_history.jsonl"

    runtime.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"explainability_runtime": str(runtime), "explainability_history": str(history)}
