from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

META_LEARNING_VERSION = "meta-learning-v1.0"
STATUS_AWAITING = "Awaiting Historical Data"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def _avg(values: list[float]) -> float | str:
    return round(mean(values), 4) if values else STATUS_AWAITING


def _drift(latest: list[float], baseline: list[float]) -> float:
    if not latest or not baseline:
        return 0.0
    lm = mean(latest)
    bm = mean(baseline)
    bsd = pstdev(baseline) if len(baseline) > 1 else abs(bm) + 1e-9
    return round(abs(lm - bm) / max(1e-9, bsd), 4)


def build_meta_learning_engine(root_dir: Path, *, idip_payload: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    storage = root_dir / "storage" / "olympus"

    history = _load_jsonl(storage / "idip_history.jsonl")[-240:]
    hypotheses = _load_json(storage / "hypotheses.json", {})
    research_queue = _load_json(storage / "research_queue.json", {})
    zeus_reports = _load_jsonl(storage / "zeus_validation_reports.jsonl")[-500:]

    summary = idip_payload.get("summary", {}) if isinstance(idip_payload, dict) else {}
    sample_size = int(summary.get("sample_size", 0) or 0)

    expectancies = [
        _safe_float((row.get("summary", {}) or {}).get("expectancy"))
        for row in history
        if isinstance(row, dict)
    ]
    expectancy_vals = [float(v) for v in expectancies if v is not None]

    decision_quality_series = [
        _safe_float((row.get("summary", {}) or {}).get("decision_quality"))
        for row in history
        if isinstance(row, dict)
    ]
    decision_vals = [float(v) for v in decision_quality_series if v is not None]

    latest_expect = expectancy_vals[-40:] if len(expectancy_vals) >= 40 else expectancy_vals
    base_expect = expectancy_vals[-160:-40] if len(expectancy_vals) >= 160 else expectancy_vals[:-40]

    latest_dec = decision_vals[-40:] if len(decision_vals) >= 40 else decision_vals
    base_dec = decision_vals[-160:-40] if len(decision_vals) >= 160 else decision_vals[:-40]

    concept_drift = _drift(latest_expect, base_expect)
    learning_drift = _drift(latest_dec, base_dec)

    passed = len([r for r in zeus_reports if str(r.get("status", "")).lower() == "passed"])
    validated_rate = round(passed / max(1, len(zeus_reports)), 4)

    hypothesis_rows = hypotheses.get("rows", []) if isinstance(hypotheses, dict) else []
    queue_rows = research_queue.get("rows", []) if isinstance(research_queue, dict) else []

    knowledge_growth_rate = _safe_float((_load_json(storage / "knowledge_growth.json", {})).get("growth_rate"))
    learning_velocity = _safe_float((_load_json(storage / "learning_velocity.json", {})).get("learning_velocity"))

    pattern_discovery_rate = round(
        len((idip_payload.get("engines", {}) or {}).get("pattern_lifecycle_intelligence", {}).get("pattern_lifecycle_profiles", []))
        / max(1, sample_size),
        4,
    )

    duplicate_learning_ratio = round(
        max(0, len(queue_rows) - len({str((r or {}).get("statement", "")).strip().lower() for r in queue_rows})) / max(1, len(queue_rows)),
        4,
    ) if queue_rows else 0.0

    plateau = bool(knowledge_growth_rate is not None and learning_velocity is not None and knowledge_growth_rate < 0.01 and learning_velocity < 0.2)
    stagnation = bool(len(queue_rows) > 0 and validated_rate < 0.2)
    overfitting_risk = bool(validated_rate < 0.25 and sample_size > 50)

    recommendations: list[dict[str, Any]] = []
    if plateau:
        recommendations.append({
            "improvement": "Increase hypothesis diversity across low-coverage contexts",
            "reason": "Learning plateau detected",
            "confidence": 0.62,
        })
    if stagnation:
        recommendations.append({
            "improvement": "Prioritize high-evidence low-cost research items",
            "reason": "Research stagnation detected",
            "confidence": 0.58,
        })
    if overfitting_risk:
        recommendations.append({
            "improvement": "Expand out-of-sample and regime-shift validation emphasis",
            "reason": "Potential overfitting signal",
            "confidence": 0.57,
        })
    if duplicate_learning_ratio > 0.3:
        recommendations.append({
            "improvement": "Merge duplicate hypotheses and enforce novelty scoring",
            "reason": "Duplicate learning risk",
            "confidence": 0.61,
        })

    payload = {
        "version": META_LEARNING_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "metrics": {
            "learning_velocity": learning_velocity if learning_velocity is not None else STATUS_AWAITING,
            "learning_efficiency": round((validated_rate * 0.6 + (float(learning_velocity or 0.0) * 0.4)), 4),
            "knowledge_growth_rate": knowledge_growth_rate if knowledge_growth_rate is not None else STATUS_AWAITING,
            "knowledge_saturation": round(min(1.0, max(0.0, sample_size / 500.0)), 4),
            "pattern_discovery_rate": pattern_discovery_rate,
            "research_productivity": round(len(queue_rows) / max(1, sample_size), 4),
            "hypothesis_success_rate": round(passed / max(1, len(hypothesis_rows)), 4) if hypothesis_rows else STATUS_AWAITING,
            "validation_success_rate": validated_rate,
            "knowledge_reuse": round(1.0 - duplicate_learning_ratio, 4),
            "knowledge_freshness": _load_json(storage / "knowledge_growth.json", {}).get("generated_at", STATUS_AWAITING),
            "concept_drift": concept_drift,
            "model_drift": _safe_float((_load_json(storage / "concept_drift.json", {})).get("model_drift_index")) or 0.0,
            "learning_drift": learning_drift,
        },
        "detections": {
            "learning_plateau": plateau,
            "overfitting_risk": overfitting_risk,
            "duplicate_learning": duplicate_learning_ratio > 0.3,
            "research_stagnation": stagnation,
        },
        "recommendations": recommendations,
    }
    return payload


def write_meta_learning_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "meta_learning_runtime.json"
    history = storage / "meta_learning_history.jsonl"

    runtime.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"meta_learning_runtime": str(runtime), "meta_learning_history": str(history)}
