"""Zeus institutional validation scaffold.

This module makes Zeus' validation mission explicit without changing the
existing backtester.  It accepts Olympus validation candidates and emits
contracted validation reports for dashboards, operator review, or future
statistical validators.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional

from olympus.contracts import ExecutionType, SourceSystem, TraceMetadata
from olympus.core.validation_contracts import (
    RecommendationCandidate,
    ValidationDomain,
    evaluate_validation_gates,
    validation_leakage_safe,
    ValidationLifecycle,
    ValidationReport,
    ValidationStatus,
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _candidate_id(candidate: Dict[str, Any]) -> str:
    return str(
        candidate.get("candidate_id")
        or candidate.get("finding_id")
        or candidate.get("hypothesis_id")
        or candidate.get("recommendation_id")
        or candidate.get("pattern_id")
        or f"candidate-{uuid.uuid4().hex[:8]}"
    )


def _source_system(candidate: Dict[str, Any]) -> SourceSystem:
    raw = _enum_value(candidate.get("source_system") or candidate.get("source") or SourceSystem.PROMETHEUS.value)
    try:
        return SourceSystem(raw)
    except Exception:
        return SourceSystem.PROMETHEUS


def _domain(candidate: Dict[str, Any], default: ValidationDomain) -> ValidationDomain:
    raw = str(candidate.get("validation_domain") or candidate.get("domain") or candidate.get("finding_type") or default.value).lower()
    for domain in ValidationDomain:
        if domain.value in raw:
            return domain
    return default


def _trace(candidate: Dict[str, Any], source: SourceSystem) -> TraceMetadata:
    return TraceMetadata(
        source_system=source,
        model_version=str(candidate.get("model_version") or candidate.get("model_version_used") or candidate.get("version") or "0"),
        feature_version=str(candidate.get("feature_version") or candidate.get("feature_set_version") or "v1"),
        strategy_version=str(candidate.get("strategy_version") or candidate.get("validation_version") or "v1"),
        instrument=str(candidate.get("instrument") or candidate.get("asset") or "unknown"),
        timeframe=str(candidate.get("timeframe") or "unknown"),
        execution_type=ExecutionType.BACKTEST,
        timestamp=str(candidate.get("timestamp") or _utc_now()),
        dataset_generation=str(candidate.get("dataset_generation") or "unknown"),
    )


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        try:
            return str(value.value)
        except Exception:
            return str(value)
    text = str(value or "")
    if "." in text:
        return text.split(".")[-1].lower()
    return text.lower()


def _queue_state(status: ValidationStatus, approved_for_adoption: bool) -> str:
    if approved_for_adoption:
        return "Approved"
    if status == ValidationStatus.RUNNING:
        return "Running"
    if status == ValidationStatus.PASSED:
        return "Completed"
    if status == ValidationStatus.FAILED:
        return "Rejected"
    if status == ValidationStatus.INCONCLUSIVE:
        return "Paused"
    return "Queued"


def _validation_modes(domain: ValidationDomain) -> List[str]:
    base = ["Historical Backtest", "Walk Forward", "Out-of-Sample", "Monte Carlo", "Statistical Confidence"]
    if domain == ValidationDomain.FEATURE:
        return base + ["Feature Validation", "Leakage Detection", "Generalisation"]
    if domain == ValidationDomain.PATTERN:
        return base + ["Pattern Validation", "Robustness"]
    if domain == ValidationDomain.EXECUTION:
        return base + ["Execution Validation", "Capital Validation"]
    if domain == ValidationDomain.CAPITAL:
        return base + ["Capital Validation", "Robustness"]
    if domain == ValidationDomain.RECOMMENDATION:
        return base + ["Recommendation Validation", "Execution Validation"]
    return base + [f"{domain.value.title()} Validation"]


def _stage_status(domain: ValidationDomain, status: ValidationStatus, candidate: Dict[str, Any]) -> Dict[str, str]:
    pending = "Running" if status == ValidationStatus.RUNNING else "Pending"
    passed = "Passed" if status == ValidationStatus.PASSED else pending
    leakage = "Passed" if candidate.get("outcome_diagnostics_excluded") else ("Pending" if domain == ValidationDomain.FEATURE else "N/A")
    stages = {
        "Historical Validation": passed if domain in (ValidationDomain.PATTERN, ValidationDomain.RECOMMENDATION, ValidationDomain.STRATEGY, ValidationDomain.EXECUTION, ValidationDomain.CAPITAL) and status == ValidationStatus.PASSED else pending,
        "Walk Forward": pending,
        "Out-of-Sample": pending,
        "Monte Carlo": pending,
        "Robustness": pending,
        "Feature Validation": pending if domain == ValidationDomain.FEATURE else "N/A",
        "Leakage Detection": leakage,
        "Statistical Confidence": pending,
        "Execution Validation": pending if domain in (ValidationDomain.EXECUTION, ValidationDomain.RECOMMENDATION) else "N/A",
        "Capital Validation": pending if domain in (ValidationDomain.CAPITAL, ValidationDomain.EXECUTION) else "N/A",
    }
    if status == ValidationStatus.INCONCLUSIVE:
        for key, value in list(stages.items()):
            if value not in ("N/A", "Passed"):
                stages[key] = "Failed"
    return stages


def _origin(candidate: Dict[str, Any], source: SourceSystem, domain: ValidationDomain) -> Dict[str, Any]:
    return {
        "source": source.value,
        "version": str(candidate.get("validation_version") or candidate.get("strategy_version") or candidate.get("model_version") or candidate.get("version") or "v1"),
        "mission": (
            "Execution Intelligence" if source == SourceSystem.PROMETHEUS else "Pattern Intelligence" if source == SourceSystem.HERMES else "Institutional Validation"
        ),
        "submission_date": str(candidate.get("timestamp") or _utc_now()),
        "research_category": domain.value.replace("_", " ").title(),
    }


def _timeline(candidate: Dict[str, Any], status: ValidationStatus) -> List[Dict[str, Any]]:
    ts = str(candidate.get("timestamp") or _utc_now())
    started = _utc_now()
    timeline = [
        {"stage": "Generated", "status": "Completed", "timestamp": ts},
        {"stage": "Submitted", "status": "Completed", "timestamp": ts},
        {"stage": "Validation Started", "status": "Running" if status == ValidationStatus.RUNNING else "Pending", "timestamp": started},
        {"stage": "Historical Complete", "status": "Pending", "timestamp": ""},
        {"stage": "Walk Forward Complete", "status": "Pending", "timestamp": ""},
        {"stage": "Monte Carlo Complete", "status": "Pending", "timestamp": ""},
        {"stage": "Approved", "status": "Pending", "timestamp": ""},
        {"stage": "Adopted", "status": "Pending", "timestamp": ""},
    ]
    if status == ValidationStatus.PASSED:
        for idx in range(2, 6):
            timeline[idx]["status"] = "Completed"
            timeline[idx]["timestamp"] = started
    if status in (ValidationStatus.FAILED, ValidationStatus.INCONCLUSIVE):
        timeline[3]["status"] = "Failed"
        timeline[3]["timestamp"] = started
    return timeline


def _recommendation_for_domain(domain: ValidationDomain) -> str:
    if domain == ValidationDomain.PATTERN:
        return "Pattern research requires Zeus evidence and operator review before any downstream adoption."
    if domain == ValidationDomain.FEATURE:
        return "Feature candidate requires leakage-safe validation and generalisation review before model adoption."
    if domain == ValidationDomain.RECOMMENDATION:
        return "Recommendation requires simulation evidence and approval before Prometheus adoption."
    if domain == ValidationDomain.CAPITAL:
        return "Capital study requires survival, drawdown, and compounding validation before approval."
    if domain == ValidationDomain.EXECUTION:
        return "Execution policy requires fill-quality, slippage, and capital-impact validation."
    return "Validation requires Zeus evidence and operator review before adoption."


class ZeusValidationEngine:
    """Passive institutional validation engine facade.

    The current implementation is contract-first.  It does not mutate
    Prometheus, Hermes, dashboards, ML models, or execution policy.
    """

    validating_system = SourceSystem.ZEUS

    def build_report(
        self,
        *,
        candidate: Dict[str, Any],
        domain: ValidationDomain,
        status: ValidationStatus = ValidationStatus.PENDING,
        confidence: float = 0.0,
        sample_size: int = 0,
        evidence: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        leakage_checks: Optional[Dict[str, Any]] = None,
        generalisation: Optional[Dict[str, Any]] = None,
        recommendation: str = "Await Zeus statistical validation and operator review.",
    ) -> ValidationReport:
        source = _source_system(candidate)
        queue_state = _queue_state(status, False)
        pipeline = _stage_status(domain, status, candidate)
        origin = _origin(candidate, source, domain)
        evidence_score = round(max(0.0, min(1.0, ((float(confidence) if confidence else 0.0) + min(1.0, max(0, int(sample_size or 0)) / 200.0)) / 2.0)), 4)
        sample_text = f"Estimated edge impact derived from {max(0, int(sample_size or 0))} samples"
        statistical_result = pipeline.get("Statistical Confidence", "Pending")
        gate_evaluation = evaluate_validation_gates(
            domain=domain,
            sample_size=max(0, int(sample_size or 0)),
            confidence=round(max(0.0, min(1.0, float(confidence))), 4),
            evidence_score=evidence_score,
            statistical_confidence_result=statistical_result,
            leakage_safe=validation_leakage_safe(domain=domain, payload=candidate, leakage_checks=leakage_checks),
        )
        thresholds = gate_evaluation["thresholds"]
        return ValidationReport(
            report_id=f"zeus-validation-{uuid.uuid4().hex[:10]}",
            validating_system=self.validating_system,
            candidate_id=_candidate_id(candidate),
            candidate_source_system=source,
            domain=domain,
            status=status,
            timestamp=_utc_now(),
            lifecycle=ValidationLifecycle.ZEUS_VALIDATION,
            confidence=round(max(0.0, min(1.0, float(confidence))), 4),
            sample_size=max(0, int(sample_size or 0)),
            evidence=evidence or {},
            metrics=metrics or {},
            leakage_checks=leakage_checks or {},
            generalisation=generalisation or {},
            recommendation=recommendation or _recommendation_for_domain(domain),
            operator_approval_required=True,
            approved_for_adoption=False,
            trace_metadata=_trace(candidate, source),
            queue_state=queue_state,
            validation_modes=_validation_modes(domain),
            validation_pipeline=pipeline,
            research_origin=origin,
            evidence_score=evidence_score,
            improvement_estimate=str(candidate.get("expected_improvement") or candidate.get("expected_capital_impact") or sample_text),
            historical_result=pipeline.get("Historical Validation", "Pending"),
            walk_forward_result=pipeline.get("Walk Forward", "Pending"),
            out_of_sample_result=pipeline.get("Out-of-Sample", "Pending"),
            monte_carlo_result=pipeline.get("Monte Carlo", "Pending"),
            robustness_result=pipeline.get("Robustness", "Pending"),
            feature_validation_result=pipeline.get("Feature Validation", "N/A"),
            leakage_status=pipeline.get("Leakage Detection", "Pending"),
            statistical_confidence_result=statistical_result,
            execution_validation_result=pipeline.get("Execution Validation", "N/A"),
            capital_validation_result=pipeline.get("Capital Validation", "N/A"),
            operator_approval_status="Pending Operator Review",
            timeline=_timeline(candidate, status),
            validation_version="zeus-v2.0",
            mission="Institutional Validation Engine",
            submission_time=str(candidate.get("timestamp") or _utc_now()),
            priority=str(candidate.get("priority") or "Normal"),
            minimum_sample_size_required=int(thresholds["minimum_sample_size"]),
            minimum_adoption_sample_size_required=int(thresholds["minimum_adoption_sample_size"]),
            minimum_evidence_score_required=float(thresholds["minimum_evidence_score"]),
            minimum_confidence_required=float(thresholds["minimum_confidence"]),
            required_statistical_confidence_status=str(thresholds["required_statistical_confidence_status"]),
            leakage_safe_required=bool(thresholds["leakage_safe_required"]),
            validation_gate_passed=bool(gate_evaluation["validation_gate_passed"]),
            adoption_gate_passed=bool(gate_evaluation["adoption_gate_passed"]),
            gate_blockers=list(gate_evaluation["gate_blockers"]),
            gate_results=dict(gate_evaluation["gates"]),
        )

    def validate_recommendation(
        self,
        candidate: RecommendationCandidate | Dict[str, Any],
    ) -> ValidationReport:
        data = candidate.as_dict() if hasattr(candidate, "as_dict") else dict(candidate)
        evidence = dict(data.get("evidence") or {})
        sample_size = int(evidence.get("sample_size") or data.get("sample_size") or 0)
        confidence = float(evidence.get("confidence") or data.get("confidence") or 0.0)
        status = ValidationStatus.INCONCLUSIVE if sample_size == 0 else ValidationStatus.PENDING
        return self.build_report(
            candidate=data,
            domain=ValidationDomain.RECOMMENDATION,
            status=status,
            confidence=confidence / 100.0 if confidence > 1.0 else confidence,
            sample_size=sample_size,
            evidence=evidence,
            recommendation="Recommendation requires simulation evidence before adoption.",
        )

    def validate_feature_candidate(self, candidate: Dict[str, Any]) -> ValidationReport:
        leakage_checks = {
            "pre_entry_only_required": True,
            "outcome_diagnostics_excluded": bool(candidate.get("outcome_diagnostics_excluded", False)),
            "pending_static_leakage_scan": True,
        }
        status = (
            ValidationStatus.PENDING
            if leakage_checks["outcome_diagnostics_excluded"]
            else ValidationStatus.OPERATOR_REVIEW_REQUIRED
        )
        return self.build_report(
            candidate=candidate,
            domain=ValidationDomain.FEATURE,
            status=status,
            leakage_checks=leakage_checks,
            recommendation="Separate PreEntryFeatureSet from OutcomeDiagnostics before model adoption.",
        )

    def validate_pattern_candidate(self, candidate: Dict[str, Any]) -> ValidationReport:
        evidence = dict(candidate.get("evidence") or {})
        sample_size = int(candidate.get("sample_size") or candidate.get("occurrences") or evidence.get("sample_size") or 0)
        status = ValidationStatus.PENDING if sample_size > 0 else ValidationStatus.INCONCLUSIVE
        return self.build_report(
            candidate=candidate,
            domain=ValidationDomain.PATTERN,
            status=status,
            sample_size=sample_size,
            evidence=evidence,
            recommendation="Pattern research must pass historical, statistical, and edge validation.",
        )

    def validate_candidates(
        self,
        candidates: Iterable[Dict[str, Any]],
        default_domain: ValidationDomain = ValidationDomain.RECOMMENDATION,
    ) -> List[ValidationReport]:
        reports: List[ValidationReport] = []
        for candidate in candidates:
            domain = _domain(candidate, default_domain)
            if domain == ValidationDomain.FEATURE:
                reports.append(self.validate_feature_candidate(candidate))
            elif domain == ValidationDomain.PATTERN:
                reports.append(self.validate_pattern_candidate(candidate))
            elif domain == ValidationDomain.RECOMMENDATION:
                reports.append(self.validate_recommendation(candidate))
            else:
                reports.append(
                    self.build_report(
                        candidate=candidate,
                        domain=domain,
                        status=ValidationStatus.PENDING,
                        recommendation=f"{domain.value.title()} candidate requires dedicated Zeus validation.",
                    )
                )
        return reports


def build_validation_status(reports: Iterable[ValidationReport]) -> Dict[str, Any]:
    rows = [r.as_dict() if hasattr(r, "as_dict") else dict(r) for r in reports]
    by_status: Dict[str, int] = {}
    by_domain: Dict[str, int] = {}
    queue_counts: Dict[str, int] = {key: 0 for key in ("Queued", "Running", "Paused", "Completed", "Rejected", "Approved")}
    evidence_by_domain: Dict[str, int] = {
        "validated_strategies": 0,
        "validated_patterns": 0,
        "validated_features": 0,
        "validated_recommendations": 0,
        "capital_studies": 0,
        "execution_policies": 0,
        "institutional_reports": len(rows),
        "historical_validation_records": len(rows),
    }
    evidence_scores: List[float] = []
    durations: List[int] = []
    validation_gates_passed = 0
    adoption_gates_passed = 0
    for row in rows:
        status = _enum_value(row.get("status", "unknown"))
        domain = _enum_value(row.get("domain", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1
        queue_state = str(row.get("queue_state", "Queued"))
        queue_counts[queue_state] = queue_counts.get(queue_state, 0) + 1
        try:
            evidence_scores.append(float(row.get("evidence_score", 0.0) or 0.0))
        except Exception:
            pass
        if bool(row.get("validation_gate_passed", False)):
            validation_gates_passed += 1
        if bool(row.get("adoption_gate_passed", False)):
            adoption_gates_passed += 1
        timeline = row.get("timeline", []) or []
        if isinstance(timeline, list):
            durations.append(max(1, len([item for item in timeline if str((item or {}).get("status", "")).lower() == "completed"])))
        if domain == ValidationDomain.PATTERN.value:
            evidence_by_domain["validated_patterns"] += 1
        elif domain == ValidationDomain.FEATURE.value:
            evidence_by_domain["validated_features"] += 1
        elif domain == ValidationDomain.RECOMMENDATION.value:
            evidence_by_domain["validated_recommendations"] += 1
        elif domain == ValidationDomain.CAPITAL.value:
            evidence_by_domain["capital_studies"] += 1
        elif domain == ValidationDomain.EXECUTION.value:
            evidence_by_domain["execution_policies"] += 1
        else:
            evidence_by_domain["validated_strategies"] += 1
    completed = queue_counts.get("Completed", 0) + queue_counts.get("Approved", 0)
    rejected = queue_counts.get("Rejected", 0)
    success_rate = round(completed / max(1, completed + rejected), 4)
    return {
        "validation_engine": "zeus",
        "generated_at": _utc_now(),
        "reports": rows,
        "version": "zeus-v2.0",
        "summary": {
            "total": len(rows),
            "by_status": by_status,
            "by_domain": by_domain,
            "operator_approval_required": sum(1 for row in rows if row.get("operator_approval_required")),
            "automatic_deployment": False,
            "pending_validations": queue_counts.get("Queued", 0),
            "running_validations": queue_counts.get("Running", 0),
            "validated_research": completed,
            "rejected_research": rejected,
            "validation_success_rate": success_rate,
            "evidence_library_size": len(rows),
            "average_validation_duration": round(sum(durations) / max(1, len(durations)), 2),
            "research_throughput": round(len(rows) / max(1, len(durations)), 2),
            "average_evidence_score": round(sum(evidence_scores) / max(1, len(evidence_scores)), 4),
            "institutional_confidence": round(sum(float(row.get("confidence", 0.0) or 0.0) for row in rows) / max(1, len(rows)), 4),
            "validation_gates_passed": validation_gates_passed,
            "adoption_gates_passed": adoption_gates_passed,
        },
        "queue": {
            "items": rows,
            "counts": queue_counts,
        },
        "evidence_library": evidence_by_domain,
    }
