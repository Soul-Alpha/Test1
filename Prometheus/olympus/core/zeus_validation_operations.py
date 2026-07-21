from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from olympus.core.validation_contracts import (
    ValidationDomain,
    ValidationLifecycle,
    ValidationStatus,
    evaluate_validation_gates,
    validation_leakage_safe,
)

ZVO_VERSION = "zvo-v1.0"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_str(ts: datetime | None = None) -> str:
    dt = ts or _utc_now()
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, str) and not value.strip():
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _norm_enumish(value: Any) -> str:
    if hasattr(value, "value"):
        try:
            return str(value.value).strip().lower()
        except Exception:
            pass
    text = str(value or "").strip().lower()
    if "." in text:
        return text.split(".")[-1]
    return text


def _safe_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(" UTC"):
        text = text[:-4] + "+00:00"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _candidate_key(report: dict[str, Any]) -> str:
    src = _norm_enumish(report.get("candidate_source_system") or report.get("source_system") or "unknown")
    domain = _norm_enumish(report.get("domain") or report.get("validation_domain") or "recommendation")
    cid = str(report.get("candidate_id") or report.get("recommendation_id") or report.get("report_id") or "unknown")
    return f"{src}|{domain}|{cid}"


def _domain_weight(domain: str) -> float:
    d = domain.lower()
    if d == "recommendation":
        return 1.15
    if d == "execution":
        return 1.1
    if d == "pattern":
        return 1.0
    if d == "feature":
        return 0.9
    if d == "capital":
        return 1.05
    return 0.95


def _priority_score(item: dict[str, Any], now: datetime) -> float:
    evidence = _safe_float(item.get("evidence_score"))
    confidence = _safe_float(item.get("confidence"))
    sample_size = max(1, _safe_int(item.get("sample_size")))
    domain = str(item.get("domain") or "recommendation")
    ts = _safe_dt(item.get("timestamp")) or now
    age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
    priority_boost = 0.2 if str(item.get("priority", "Normal")).lower() == "high" else 0.0

    raw = (
        (_domain_weight(domain) * 0.45)
        + (evidence * 0.8)
        + (confidence * 0.55)
        + (min(1.0, math.log1p(sample_size) / 7.0) * 0.5)
        + min(0.35, age_hours * 0.01)
        + priority_boost
    )
    return round(raw, 6)


def _record_transition(item: dict[str, Any], stage: str, status: str, reason: str, now: datetime) -> None:
    hist = item.setdefault("lifecycle_history", [])
    hist.append(
        {
            "timestamp": _utc_str(now),
            "stage": stage,
            "status": status,
            "reason": reason,
        }
    )
    item["last_transition_at"] = _utc_str(now)


def _pipeline_stage_order(item: dict[str, Any]) -> list[str]:
    pipeline = item.get("validation_pipeline") or {}
    base = [
        "Historical Validation",
        "Walk Forward",
        "Out-of-Sample",
        "Monte Carlo",
        "Robustness",
        "Statistical Confidence",
    ]
    domain_specific: list[str] = []
    if str(item.get("domain", "")).lower() == "feature":
        domain_specific = ["Feature Validation", "Leakage Detection"]
    elif str(item.get("domain", "")).lower() == "execution":
        domain_specific = ["Execution Validation", "Capital Validation"]
    elif str(item.get("domain", "")).lower() == "recommendation":
        domain_specific = ["Execution Validation"]

    out = [x for x in base + domain_specific if x in pipeline]
    return out or base


def _domain_from_item(item: dict[str, Any]) -> ValidationDomain:
    raw = _norm_enumish(item.get("domain") or item.get("validation_domain") or ValidationDomain.RECOMMENDATION.value)
    try:
        return ValidationDomain(raw)
    except Exception:
        return ValidationDomain.RECOMMENDATION


def _sync_pipeline_results(item: dict[str, Any]) -> None:
    pipeline = item.get("validation_pipeline") or {}
    item["historical_result"] = pipeline.get("Historical Validation", item.get("historical_result", "Pending"))
    item["walk_forward_result"] = pipeline.get("Walk Forward", item.get("walk_forward_result", "Pending"))
    item["out_of_sample_result"] = pipeline.get("Out-of-Sample", item.get("out_of_sample_result", "Pending"))
    item["monte_carlo_result"] = pipeline.get("Monte Carlo", item.get("monte_carlo_result", "Pending"))
    item["robustness_result"] = pipeline.get("Robustness", item.get("robustness_result", "Pending"))
    item["feature_validation_result"] = pipeline.get("Feature Validation", item.get("feature_validation_result", "N/A"))
    item["leakage_status"] = pipeline.get("Leakage Detection", item.get("leakage_status", "Pending"))
    item["statistical_confidence_result"] = pipeline.get("Statistical Confidence", item.get("statistical_confidence_result", "Pending"))
    item["execution_validation_result"] = pipeline.get("Execution Validation", item.get("execution_validation_result", "N/A"))
    item["capital_validation_result"] = pipeline.get("Capital Validation", item.get("capital_validation_result", "N/A"))


def _sync_gate_fields(item: dict[str, Any]) -> dict[str, Any]:
    _sync_pipeline_results(item)
    domain = _domain_from_item(item)
    evaluation = evaluate_validation_gates(
        domain=domain,
        sample_size=max(0, _safe_int(item.get("sample_size"))),
        confidence=max(0.0, min(1.0, _safe_float(item.get("confidence")))),
        evidence_score=max(0.0, min(1.0, _safe_float(item.get("evidence_score")))),
        statistical_confidence_result=str(item.get("statistical_confidence_result") or "Pending"),
        leakage_safe=validation_leakage_safe(domain=domain, payload=item, leakage_checks=item.get("leakage_checks")),
    )
    thresholds = evaluation["thresholds"]
    item["minimum_sample_size_required"] = int(thresholds["minimum_sample_size"])
    item["minimum_adoption_sample_size_required"] = int(thresholds["minimum_adoption_sample_size"])
    item["minimum_evidence_score_required"] = float(thresholds["minimum_evidence_score"])
    item["minimum_confidence_required"] = float(thresholds["minimum_confidence"])
    item["required_statistical_confidence_status"] = str(thresholds["required_statistical_confidence_status"])
    item["leakage_safe_required"] = bool(thresholds["leakage_safe_required"])
    item["validation_gate_passed"] = bool(evaluation["validation_gate_passed"])
    item["adoption_gate_passed"] = bool(evaluation["adoption_gate_passed"])
    item["gate_blockers"] = list(evaluation["gate_blockers"])
    item["gate_results"] = dict(evaluation["gates"])
    return evaluation


def _advance_pipeline_step(item: dict[str, Any], now: datetime) -> tuple[bool, str]:
    pipeline = item.setdefault("validation_pipeline", {})
    stages = _pipeline_stage_order(item)
    idx = _safe_int(item.get("scheduler_step"))
    sample = _safe_int(item.get("sample_size"))
    evidence = _safe_float(item.get("evidence_score"))
    conf = _safe_float(item.get("confidence"))

    if str(item.get("status", "pending")).lower() == ValidationStatus.PENDING.value:
        item["status"] = ValidationStatus.RUNNING.value
        item["queue_state"] = "Running"
        if stages:
            first = stages[0]
            if pipeline.get(first, "Pending") == "Pending":
                pipeline[first] = "Running"
        _sync_gate_fields(item)
        _record_transition(item, ValidationLifecycle.ZEUS_VALIDATION.value, "running", "Scheduler started validation run.", now)
        return True, "started"

    if str(item.get("status", "pending")).lower() in (ValidationStatus.FAILED.value, ValidationStatus.INCONCLUSIVE.value):
        return False, "terminal"

    gate_state = _sync_gate_fields(item)

    # Not enough data: pause and wait for evidence growth.
    if sample < int(gate_state["thresholds"]["minimum_sample_size"]):
        item["status"] = ValidationStatus.INCONCLUSIVE.value
        item["queue_state"] = "Paused"
        item["operator_approval_status"] = "Pending More Evidence"
        _sync_gate_fields(item)
        _record_transition(item, ValidationLifecycle.ZEUS_VALIDATION.value, "paused", "Sample size below institutional minimum.", now)
        return True, "paused"

    if idx < len(stages):
        stage = stages[idx]
        threshold = 0.35 + (0.08 * idx)
        quality = evidence + (conf * 0.5) + min(0.4, math.log1p(max(1, sample)) / 20.0)
        if quality >= threshold:
            pipeline[stage] = "Passed"
            item["scheduler_step"] = idx + 1
            _sync_gate_fields(item)
            _record_transition(item, ValidationLifecycle.ZEUS_VALIDATION.value, "progress", f"{stage} passed.", now)
            return True, "progress"
        pipeline[stage] = "Failed"
        item["status"] = ValidationStatus.FAILED.value
        item["queue_state"] = "Rejected"
        item["operator_approval_status"] = "Rejected by Zeus Validation"
        _sync_gate_fields(item)
        _record_transition(item, ValidationLifecycle.ZEUS_VALIDATION.value, "failed", f"{stage} failed quality threshold.", now)
        return True, "failed"

    item["status"] = ValidationStatus.PASSED.value
    item["queue_state"] = "Completed"
    item["lifecycle"] = ValidationLifecycle.VALIDATED.value
    gate_state = _sync_gate_fields(item)
    _record_transition(item, ValidationLifecycle.VALIDATED.value, "passed", "Validation pipeline completed.", now)

    # Zeus validates evidence; it must never impersonate an operator.  Passing
    # every adoption gate ends in an explicit waiting state.  A separately
    # authenticated operator workflow is the only authority allowed to create
    # OPERATOR_APPROVED or ACTIVE transitions.
    if item.get("operator_approval_required", True):
        item["approved_for_adoption"] = False
        if bool(gate_state.get("adoption_gate_passed")):
            item["operator_approval_status"] = "Awaiting Explicit Operator Approval"
            item["queue_state"] = "Awaiting Operator Approval"
            item["lifecycle"] = ValidationLifecycle.AWAITING_OPERATOR_APPROVAL.value
            _record_transition(
                item,
                ValidationLifecycle.AWAITING_OPERATOR_APPROVAL.value,
                "pending",
                "Automated validation passed; explicit operator approval is required.",
                now,
            )
        else:
            item["operator_approval_status"] = "Pending Zeus Gate Thresholds"
    return True, "completed"


def _build_status(items: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    queue_counts: dict[str, int] = {k: 0 for k in ("Queued", "Running", "Paused", "Completed", "Rejected", "Approved")}
    by_lifecycle: dict[str, int] = {}
    ages: list[float] = []
    approvals_24h = 0
    approved_candidates_24h: set[str] = set()
    validation_gates_passed = 0
    adoption_gates_passed = 0

    for item in items:
        status = _norm_enumish(item.get("status") or ValidationStatus.PENDING.value) or ValidationStatus.PENDING.value
        domain = _norm_enumish(item.get("domain") or "unknown") or "unknown"
        lifecycle = _norm_enumish(item.get("lifecycle") or ValidationLifecycle.ZEUS_VALIDATION.value) or ValidationLifecycle.ZEUS_VALIDATION.value
        by_status[status] = by_status.get(status, 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_lifecycle[lifecycle] = by_lifecycle.get(lifecycle, 0) + 1
        queue_state = str(item.get("queue_state", "Queued"))
        queue_counts[queue_state] = queue_counts.get(queue_state, 0) + 1
        if bool(item.get("validation_gate_passed", False)):
            validation_gates_passed += 1
        if bool(item.get("adoption_gate_passed", False)):
            adoption_gates_passed += 1

        ts = _safe_dt(item.get("timestamp"))
        if ts is not None:
            age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
            ages.append(age_h)

        candidate_key = _candidate_key(item)
        for ev in item.get("lifecycle_history", []):
            if _norm_enumish(ev.get("status")) == "approved":
                ev_dt = _safe_dt(ev.get("timestamp"))
                if ev_dt is not None and ev_dt >= (now - timedelta(hours=24)):
                    approved_candidates_24h.add(candidate_key)
                    break

        if candidate_key not in approved_candidates_24h and bool(item.get("approved_for_adoption")):
            approved_dt = _safe_dt(item.get("approved_at")) or _safe_dt(item.get("last_transition_at"))
            if approved_dt is not None and approved_dt >= (now - timedelta(hours=24)):
                approved_candidates_24h.add(candidate_key)

    approvals_24h = len(approved_candidates_24h)

    completed = queue_counts.get("Completed", 0) + queue_counts.get("Approved", 0)
    rejected = queue_counts.get("Rejected", 0)
    success_rate = round(completed / max(1, completed + rejected), 4)

    return {
        "validation_engine": "zeus",
        "generated_at": _utc_str(now),
        "version": "zeus-v2.1",
        "zvo_version": ZVO_VERSION,
        "reports": items,
        "summary": {
            "total": len(items),
            "by_status": by_status,
            "by_domain": by_domain,
            "operator_approval_required": sum(1 for x in items if bool(x.get("operator_approval_required", True))),
            "automatic_deployment": False,
            "pending_validations": queue_counts.get("Queued", 0),
            "running_validations": queue_counts.get("Running", 0),
            "validated_research": completed,
            "rejected_research": rejected,
            "validation_success_rate": success_rate,
            "average_queue_age_hours": round(sum(ages) / max(1, len(ages)), 3) if ages else 0.0,
            "institutional_confidence": round(sum(_safe_float(x.get("confidence")) for x in items) / max(1, len(items)), 4),
            "approval_velocity_24h": approvals_24h,
            "validation_gates_passed": validation_gates_passed,
            "adoption_gates_passed": adoption_gates_passed,
        },
        "queue": {
            "items": items,
            "counts": queue_counts,
        },
        "lifecycle": {
            "order": [stage.value for stage in ValidationLifecycle],
            "current_stage": ValidationLifecycle.ZEUS_VALIDATION.value if items else ValidationLifecycle.CANDIDATE.value,
            "counts": by_lifecycle,
        },
        "research_status": "Zeus Validation Operations active",
        "institutional_maturity": "Validated" if completed >= 10 else "Developing",
        "learning_velocity": round((completed + approvals_24h) / max(1, len(items)), 4),
    }


def run_zeus_validation_operations(
    *,
    root_dir: Path,
    incoming_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    now = _utc_now()
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime_f = storage / "zeus_validation_operations_runtime.json"
    history_f = storage / "zeus_validation_operations_history.jsonl"

    runtime = _load_json(runtime_f, {})
    existing_rows = runtime.get("queue", {}).get("items", []) if isinstance(runtime, dict) else []

    queue: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        key = _candidate_key(row)
        if key:
            queue[key] = dict(row)

    # Merge incoming candidates/reports by candidate identity.
    for raw in incoming_reports:
        if not isinstance(raw, dict):
            continue
        key = _candidate_key(raw)
        if not key:
            continue

        if key in queue:
            current = queue[key]
            # Keep lifecycle state, refresh evidence and telemetry fields.
            for fld in (
                "report_id",
                "timestamp",
                "sample_size",
                "confidence",
                "evidence",
                "metrics",
                "leakage_checks",
                "generalisation",
                "evidence_score",
                "priority",
                "validation_modes",
                "validation_pipeline",
                "trace_metadata",
                "research_origin",
                "improvement_estimate",
                "submission_time",
                "mission",
            ):
                if fld in raw:
                    current[fld] = raw.get(fld)
            _sync_gate_fields(current)
            queue[key] = current
        else:
            row = dict(raw)
            row.setdefault("status", ValidationStatus.PENDING.value)
            row.setdefault("queue_state", "Queued")
            row.setdefault("lifecycle", ValidationLifecycle.ZEUS_VALIDATION.value)
            row.setdefault("scheduler_step", 0)
            row.setdefault("operator_approval_required", True)
            row.setdefault("approved_for_adoption", False)
            row.setdefault("operator_approval_status", "Pending Operator Review")
            row.setdefault("lifecycle_history", [])
            _sync_gate_fields(row)
            _record_transition(row, ValidationLifecycle.CANDIDATE.value, "queued", "Candidate merged into Zeus queue.", now)
            queue[key] = row

    # Priority scoring and scheduler progression.
    rows = list(queue.values())
    for row in rows:
        row["priority_score"] = _priority_score(row, now)

    rows.sort(key=lambda x: (_safe_float(x.get("priority_score")), _safe_int(x.get("sample_size"))), reverse=True)

    max_transitions = _safe_int(os.getenv("ZEUS_SCHEDULER_MAX_TRANSITIONS", "30")) or 30
    transitions = 0
    transition_events: list[dict[str, Any]] = []
    for row in rows:
        if transitions >= max_transitions:
            break
        changed, mode = _advance_pipeline_step(row, now)
        if changed:
            transitions += 1
            transition_events.append(
                {
                    "timestamp": _utc_str(now),
                    "candidate_id": row.get("candidate_id"),
                    "domain": row.get("domain"),
                    "mode": mode,
                    "status": row.get("status"),
                    "lifecycle": row.get("lifecycle"),
                    "queue_state": row.get("queue_state"),
                }
            )

    for row in rows:
        _sync_gate_fields(row)

    status = _build_status(rows, now)
    runtime_payload = {
        "version": ZVO_VERSION,
        "generated_at": _utc_str(now),
        "scheduler": {
            "max_transitions": max_transitions,
            "transitions_executed": transitions,
        },
        "queue": {
            "items": rows,
            "counts": status.get("queue", {}).get("counts", {}),
        },
        "lifecycle": status.get("lifecycle", {}),
        "summary": status.get("summary", {}),
        "recent_transitions": transition_events,
    }
    _write_json_atomic(runtime_f, runtime_payload)

    for event in transition_events:
        _append_jsonl(history_f, event)

    return {
        "status": status,
        "reports": rows,
        "runtime_path": str(runtime_f),
        "history_path": str(history_f),
        "transitions": transition_events,
    }
