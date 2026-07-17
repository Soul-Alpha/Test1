from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ILS_VERSION = "ils-v1.0"
STATUS_AWAITING = "Awaiting Historical Data"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


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


def _score_maturity(sample_size: int) -> str:
    if sample_size >= 300:
        return "Institutional"
    if sample_size >= 120:
        return "Validated"
    if sample_size >= 50:
        return "Developing"
    return "Emerging"


def _drift_index(latest: list[float], baseline: list[float]) -> float:
    if not latest or not baseline:
        return 0.0
    latest_mean = mean(latest)
    baseline_mean = mean(baseline)
    baseline_sd = pstdev(baseline) if len(baseline) > 1 else (abs(baseline_mean) + 1e-9)
    return round(abs(latest_mean - baseline_mean) / max(1e-9, baseline_sd), 4)


def build_institutional_learning_scientist(
    root_dir: Path,
    *,
    status: dict[str, Any],
    closed_trades: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    zeus_reports: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
    idip_payload: dict[str, Any],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    storage = root_dir / "storage" / "olympus"

    history_rows = _load_jsonl(storage / "idip_history.jsonl")
    prior_knowledge = _load_json(storage / "knowledge_growth.json", {})

    realized_returns = [
        _safe_float(row.get("realized_return_pct"))
        for row in closed_trades
        if _safe_float(row.get("realized_return_pct")) is not None
    ]
    realized = [float(v) for v in realized_returns if v is not None]

    decision_rows = (
        (idip_payload.get("engines", {}) or {})
        .get("decision_attribution_intelligence", {})
        .get("decision_attribution_rows", [])
    )
    decision_scores = [
        _safe_float(row.get("decision_score"))
        for row in decision_rows
        if _safe_float(row.get("decision_score")) is not None
    ]
    decision_vals = [float(v) for v in decision_scores if v is not None]

    trailing_returns = [
        _safe_float((row.get("summary", {}) or {}).get("expectancy"))
        for row in history_rows[-180:]
        if isinstance(row, dict)
    ]
    trailing = [float(v) for v in trailing_returns if v is not None]

    latest_window = trailing[-40:] if len(trailing) >= 40 else trailing
    baseline_window = trailing[-160:-40] if len(trailing) >= 160 else trailing[:-40]

    concept_drift_score = _drift_index(latest_window, baseline_window)
    behavior_drift_score = _drift_index(
        decision_vals[-40:] if len(decision_vals) >= 40 else decision_vals,
        decision_vals[-160:-40] if len(decision_vals) >= 160 else decision_vals[:-40],
    )

    model_conf = [
        _safe_float((r.get("evidence", {}) or {}).get("confidence"))
        for r in zeus_reports
        if isinstance(r, dict)
    ]
    model_conf_vals = [float(v) for v in model_conf if v is not None]
    model_drift_score = _drift_index(
        model_conf_vals[-40:] if len(model_conf_vals) >= 40 else model_conf_vals,
        model_conf_vals[-160:-40] if len(model_conf_vals) >= 160 else model_conf_vals[:-40],
    )

    hypothesis_rows: list[dict[str, Any]] = []
    if realized and _avg(realized) != STATUS_AWAITING and float(_avg(realized)) < 0:
        hypothesis_rows.append(
            {
                "hypothesis_id": "H-EXPECTANCY-RECOVERY-001",
                "priority": "High",
                "statement": "Negative expectancy indicates structural mismatch in exit and risk sequencing.",
                "research_target": "Validate stricter exit-style and session-regime gating through Zeus.",
                "confidence": round(min(1.0, 0.3 + len(realized) / 600.0), 4),
                "status": "Candidate",
                "generated_at": generated_at,
            }
        )
    if concept_drift_score >= 0.9:
        hypothesis_rows.append(
            {
                "hypothesis_id": "H-CONCEPT-DRIFT-001",
                "priority": "High",
                "statement": "Concept drift detected in expectancy distribution.",
                "research_target": "Re-validate dominant lifecycle and pattern pathways under current regime.",
                "confidence": round(min(1.0, 0.35 + concept_drift_score * 0.15), 4),
                "status": "Candidate",
                "generated_at": generated_at,
            }
        )
    if behavior_drift_score >= 0.9:
        hypothesis_rows.append(
            {
                "hypothesis_id": "H-BEHAVIOR-DRIFT-001",
                "priority": "Medium",
                "statement": "Decision score distribution drift suggests behavior inconsistency.",
                "research_target": "Replay decision-path variants and validate safer management templates.",
                "confidence": round(min(1.0, 0.3 + behavior_drift_score * 0.12), 4),
                "status": "Candidate",
                "generated_at": generated_at,
            }
        )

    growth_prev = int(prior_knowledge.get("total_knowledge_objects", 0) or 0)
    current_knowledge_objects = int(
        len(decision_rows)
        + len(replay_rows)
        + len(zeus_reports)
        + len(simulation_rows)
        + len((idip_payload.get("engines", {}) or {}).get("institutional_knowledge_intelligence", {}).get("institutional_lessons", []))
    )

    growth_delta = max(0, current_knowledge_objects - growth_prev)
    growth_rate = round(growth_delta / max(1.0, float(growth_prev if growth_prev > 0 else 100)), 4)

    learning_events = len(closed_trades) + len(replay_rows) + len(zeus_reports) + len(simulation_rows)
    learning_velocity = round(learning_events / max(1.0, len(closed_trades) if closed_trades else 50.0), 4)

    research_queue = []
    for hyp in hypothesis_rows:
        research_queue.append(
            {
                "research_id": f"RQ-{hyp['hypothesis_id']}",
                "source": "institutional_learning_scientist",
                "priority": hyp.get("priority", "Medium"),
                "statement": hyp.get("statement"),
                "requires_zeus_validation": True,
                "operator_approved": False,
                "generated_at": generated_at,
                "status": "Queued",
            }
        )

    institutional_learning = {
        "version": ILS_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "sample_size": len(closed_trades),
        "replay_samples": len(replay_rows),
        "zeus_validation_samples": len(zeus_reports),
        "simulation_samples": len(simulation_rows),
        "knowledge_maturity": _score_maturity(len(closed_trades)),
        "knowledge_confidence": _avg(model_conf_vals),
        "decision_quality": _avg(decision_vals),
        "expectancy": _avg(realized),
        "knowledge_gaps": [
            "Exit taxonomy residual unknowns" if int((idip_payload.get("summary", {}) or {}).get("unknown_exit_count", 0) or 0) > 0 else "None detected",
            "Concept drift investigation" if concept_drift_score >= 0.9 else "None detected",
            "Behaviour drift investigation" if behavior_drift_score >= 0.9 else "None detected",
        ],
    }

    return {
        "institutional_learning": institutional_learning,
        "hypotheses": {
            "version": ILS_VERSION,
            "generated_at": generated_at,
            "rows": hypothesis_rows,
        },
        "knowledge_growth": {
            "version": ILS_VERSION,
            "generated_at": generated_at,
            "total_knowledge_objects": current_knowledge_objects,
            "growth_delta": growth_delta,
            "growth_rate": growth_rate,
        },
        "learning_velocity": {
            "version": ILS_VERSION,
            "generated_at": generated_at,
            "learning_events": learning_events,
            "learning_velocity": learning_velocity,
            "learning_sources": {
                "completed_trades": len(closed_trades),
                "replays": len(replay_rows),
                "zeus_validations": len(zeus_reports),
                "simulations": len(simulation_rows),
            },
        },
        "research_queue": {
            "version": ILS_VERSION,
            "generated_at": generated_at,
            "rows": research_queue,
        },
        "concept_drift": {
            "version": ILS_VERSION,
            "generated_at": generated_at,
            "concept_drift_index": concept_drift_score,
            "behaviour_drift_index": behavior_drift_score,
            "model_drift_index": model_drift_score,
            "deteriorating_strategy": bool(_avg(realized) != STATUS_AWAITING and float(_avg(realized)) < 0),
        },
    }


def write_institutional_learning_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    mapping = {
        "institutional_learning": storage / "institutional_learning.json",
        "hypotheses": storage / "hypotheses.json",
        "knowledge_growth": storage / "knowledge_growth.json",
        "learning_velocity": storage / "learning_velocity.json",
        "research_queue": storage / "research_queue.json",
        "concept_drift": storage / "concept_drift.json",
    }

    for key, path in mapping.items():
        _write_json_atomic(path, payload.get(key, {}))

    return {k: str(v) for k, v in mapping.items()}
