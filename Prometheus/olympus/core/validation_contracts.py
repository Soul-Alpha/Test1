"""Institutional validation contracts for Project Olympus.

These contracts are additive standards only.  They describe how Hera,
Prometheus, Hermes, Zeus, and Academy systems exchange candidate knowledge and
validation evidence without changing execution behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from olympus.contracts import ExecutionType, SourceSystem, TraceMetadata


class SystemMission(str, Enum):
    HERA = "institutional_governance"
    PROMETHEUS = "market_analysis_execution_intelligence"
    HERMES = "pattern_intelligence"
    ZEUS = "institutional_validation_engine"
    ACADEMY = "independent_evaluation"


class ValidationLifecycle(str, Enum):
    LEARNING = "learning"
    CANDIDATE = "candidate"
    ZEUS_VALIDATION = "zeus_validation"
    VALIDATED = "validated"
    AWAITING_OPERATOR_APPROVAL = "awaiting_operator_approval"
    OPERATOR_APPROVED = "operator_approved"
    ACTIVE = "active"
    MONITORING = "monitoring"
    RETIRED = "retired"


class ValidationDomain(str, Enum):
    STRATEGY = "strategy"
    EXECUTION = "execution"
    PATTERN = "pattern"
    FEATURE = "feature"
    CAPITAL = "capital"
    RECOMMENDATION = "recommendation"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    OPERATOR_REVIEW_REQUIRED = "operator_review_required"


@dataclass(frozen=True)
class ValidationGateThresholds:
    minimum_sample_size: int = 20
    minimum_adoption_sample_size: int = 120
    minimum_evidence_score: float = 0.55
    minimum_confidence: float = 0.55
    required_statistical_confidence_status: str = "Passed"
    leakage_safe_required: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validation_gate_thresholds(domain: ValidationDomain) -> ValidationGateThresholds:
    if domain == ValidationDomain.FEATURE:
        return ValidationGateThresholds(leakage_safe_required=True)
    return ValidationGateThresholds()


def evaluate_validation_gates(
    *,
    domain: ValidationDomain,
    sample_size: int,
    confidence: float,
    evidence_score: float,
    statistical_confidence_result: str,
    leakage_safe: bool,
) -> Dict[str, Any]:
    thresholds = validation_gate_thresholds(domain)
    observed_stat = str(statistical_confidence_result or "Pending")

    sample_ready = sample_size >= thresholds.minimum_sample_size
    adoption_sample_ready = sample_size >= thresholds.minimum_adoption_sample_size
    evidence_ready = evidence_score >= thresholds.minimum_evidence_score
    confidence_ready = confidence >= thresholds.minimum_confidence
    statistical_ready = observed_stat.lower() == thresholds.required_statistical_confidence_status.lower()
    leakage_ready = True if not thresholds.leakage_safe_required else bool(leakage_safe)

    blockers: List[str] = []
    if not sample_ready:
        blockers.append("minimum_sample_size")
    if not adoption_sample_ready:
        blockers.append("minimum_adoption_sample_size")
    if not evidence_ready:
        blockers.append("minimum_evidence_score")
    if not confidence_ready:
        blockers.append("minimum_confidence")
    if not statistical_ready:
        blockers.append("statistical_confidence")
    if not leakage_ready:
        blockers.append("leakage_safety")

    return {
        "thresholds": thresholds.as_dict(),
        "gates": {
            "minimum_sample_size": {
                "required": thresholds.minimum_sample_size,
                "observed": sample_size,
                "passed": sample_ready,
            },
            "minimum_adoption_sample_size": {
                "required": thresholds.minimum_adoption_sample_size,
                "observed": sample_size,
                "passed": adoption_sample_ready,
            },
            "minimum_evidence_score": {
                "required": thresholds.minimum_evidence_score,
                "observed": round(evidence_score, 4),
                "passed": evidence_ready,
            },
            "minimum_confidence": {
                "required": thresholds.minimum_confidence,
                "observed": round(confidence, 4),
                "passed": confidence_ready,
            },
            "statistical_confidence": {
                "required": thresholds.required_statistical_confidence_status,
                "observed": observed_stat,
                "passed": statistical_ready,
            },
            "leakage_safety": {
                "required": thresholds.leakage_safe_required,
                "observed": bool(leakage_safe),
                "passed": leakage_ready,
            },
        },
        "validation_gate_passed": sample_ready,
        "adoption_gate_passed": all(
            [
                sample_ready,
                adoption_sample_ready,
                evidence_ready,
                confidence_ready,
                statistical_ready,
                leakage_ready,
            ]
        ),
        "gate_blockers": blockers,
    }


def _resolve_outcome_diagnostics_excluded(
    payload: Dict[str, Any],
    leakage_checks: Optional[Dict[str, Any]] = None,
) -> bool | None:
    checks = leakage_checks or {}
    if "outcome_diagnostics_excluded" in checks:
        return bool(checks.get("outcome_diagnostics_excluded"))
    gate_results = payload.get("gate_results", {}) if isinstance(payload, dict) else {}
    leakage_gate = gate_results.get("leakage_safety", {}) if isinstance(gate_results, dict) else {}
    if "observed" in leakage_gate:
        return bool(leakage_gate.get("observed"))
    if "outcome_diagnostics_excluded" in payload:
        return bool(payload.get("outcome_diagnostics_excluded"))
    return None


def validation_leakage_safe(
    *,
    domain: ValidationDomain,
    payload: Dict[str, Any],
    leakage_checks: Optional[Dict[str, Any]] = None,
) -> bool:
    if domain != ValidationDomain.FEATURE:
        return True
    return bool(_resolve_outcome_diagnostics_excluded(payload, leakage_checks))


@dataclass(frozen=True)
class MissionBoundary:
    source_system: SourceSystem
    mission: SystemMission
    responsibilities: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalIntent:
    intent_id: str
    source_system: SourceSystem
    instrument: str
    timeframe: str
    timestamp: str
    direction: str
    confidence: float = 0.0
    score: float = 0.0
    grade: str = "F"
    rationale: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPolicy:
    policy_id: str
    source_system: SourceSystem
    execution_type: ExecutionType
    max_risk_pct: Optional[float] = None
    min_rr: Optional[float] = None
    allowed_entry_types: List[str] = field(default_factory=list)
    spread_rules: Dict[str, Any] = field(default_factory=dict)
    stop_rules: Dict[str, Any] = field(default_factory=dict)
    approval_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    signal_intent_id: str
    source_system: SourceSystem
    instrument: str
    timeframe: str
    timestamp: str
    direction: str
    entry_type: str
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp1_price: Optional[float] = None
    tp2_price: Optional[float] = None
    size: Optional[float] = None
    policy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    result_id: str
    execution_plan_id: str
    source_system: SourceSystem
    timestamp: str
    status: str
    fill_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    rr: Optional[float] = None
    slippage: Optional[float] = None
    fill_quality: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreEntryFeatureSet:
    feature_set_id: str
    source_system: SourceSystem
    instrument: str
    timeframe: str
    timestamp: str
    feature_version: str
    market_structure: Dict[str, Any] = field(default_factory=dict)
    liquidity: Dict[str, Any] = field(default_factory=dict)
    execution_location: Dict[str, Any] = field(default_factory=dict)
    volatility: Dict[str, Any] = field(default_factory=dict)
    session: Dict[str, Any] = field(default_factory=dict)
    statistical: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeDiagnostics:
    diagnostics_id: str
    source_system: SourceSystem
    instrument: str
    timeframe: str
    timestamp: str
    execution_result_id: Optional[str] = None
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    rr: Optional[float] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None
    hold_seconds: Optional[int] = None
    exit_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatternCandidate:
    candidate_id: str
    source_system: SourceSystem
    pattern_name: str
    timestamp: str
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    lifecycle: ValidationLifecycle = ValidationLifecycle.CANDIDATE
    operator_approved: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationCandidate:
    candidate_id: str
    source_system: SourceSystem
    timestamp: str
    recommendation_type: str
    recommendation: str
    expected_improvement: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    lifecycle: ValidationLifecycle = ValidationLifecycle.CANDIDATE
    operator_approved: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapitalStudy:
    study_id: str
    source_system: SourceSystem
    timestamp: str
    objective: str
    assumptions: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    lifecycle: ValidationLifecycle = ValidationLifecycle.CANDIDATE
    operator_approved: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    report_id: str
    validating_system: SourceSystem
    candidate_id: str
    candidate_source_system: SourceSystem
    domain: ValidationDomain
    status: ValidationStatus
    timestamp: str
    lifecycle: ValidationLifecycle = ValidationLifecycle.ZEUS_VALIDATION
    confidence: float = 0.0
    sample_size: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    leakage_checks: Dict[str, Any] = field(default_factory=dict)
    generalisation: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    operator_approval_required: bool = True
    approved_for_adoption: bool = False
    trace_metadata: Optional[TraceMetadata] = None
    queue_state: str = "Queued"
    validation_modes: List[str] = field(default_factory=list)
    validation_pipeline: Dict[str, str] = field(default_factory=dict)
    research_origin: Dict[str, Any] = field(default_factory=dict)
    evidence_score: float = 0.0
    improvement_estimate: str = "Pending validation"
    historical_result: str = "Pending"
    walk_forward_result: str = "Pending"
    out_of_sample_result: str = "Pending"
    monte_carlo_result: str = "Pending"
    robustness_result: str = "Pending"
    feature_validation_result: str = "Pending"
    leakage_status: str = "Pending"
    statistical_confidence_result: str = "Pending"
    execution_validation_result: str = "Pending"
    capital_validation_result: str = "Pending"
    operator_approval_status: str = "Pending Operator Review"
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    validation_version: str = "zeus-v2.0"
    mission: str = "Institutional Validation Engine"
    submission_time: str = ""
    priority: str = "Normal"
    minimum_sample_size_required: int = 20
    minimum_adoption_sample_size_required: int = 120
    minimum_evidence_score_required: float = 0.55
    minimum_confidence_required: float = 0.55
    required_statistical_confidence_status: str = "Passed"
    leakage_safe_required: bool = False
    validation_gate_passed: bool = False
    adoption_gate_passed: bool = False
    gate_blockers: List[str] = field(default_factory=list)
    gate_results: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.trace_metadata is not None:
            data["trace_metadata"] = self.trace_metadata.as_dict()
        return data


PatternValidationReport = ValidationReport


MISSION_BOUNDARIES: Dict[SourceSystem, MissionBoundary] = {
    SourceSystem.HERA: MissionBoundary(
        source_system=SourceSystem.HERA,
        mission=SystemMission.HERA,
        responsibilities=[
            "architecture_governance",
            "compatibility_governance",
            "olympus_standards",
            "intelligence_contracts",
            "dashboard_standards",
            "schema_governance",
            "version_governance",
            "institutional_policy",
        ],
        prohibited_actions=[
            "trade",
            "validate_strategies",
            "discover_patterns",
            "modify_execution",
        ],
    ),
    SourceSystem.PROMETHEUS: MissionBoundary(
        source_system=SourceSystem.PROMETHEUS,
        mission=SystemMission.PROMETHEUS,
        responsibilities=[
            "market_analysis",
            "execution_intelligence",
            "prometheus_ml_learning",
            "execution_learning",
            "capital_intelligence",
            "edge_intelligence",
            "recommendation_generation",
        ],
        prohibited_actions=[
            "train_using_hermes_datasets",
            "automatically_deploy_self_generated_improvements",
        ],
    ),
    SourceSystem.HERMES: MissionBoundary(
        source_system=SourceSystem.HERMES,
        mission=SystemMission.HERMES,
        responsibilities=[
            "pattern_discovery",
            "pattern_evolution",
            "context_intelligence",
            "pattern_research",
            "pattern_confidence",
            "pattern_edge_research",
        ],
        prohibited_actions=[
            "produce_execution_decisions",
            "inject_raw_observations_into_prometheus_ml",
            "directly_modify_prometheus",
        ],
    ),
    SourceSystem.ZEUS: MissionBoundary(
        source_system=SourceSystem.ZEUS,
        mission=SystemMission.ZEUS,
        responsibilities=[
            "historical_validation",
            "walk_forward_validation",
            "out_of_sample_validation",
            "monte_carlo_validation",
            "regime_validation",
            "execution_validation",
            "pattern_validation",
            "feature_validation",
            "capital_validation",
            "recommendation_validation",
        ],
        prohibited_actions=[
            "execute_live_trades",
            "automatically_deploy_validated_improvements",
            "override_operator_approval",
        ],
    ),
}


def validation_lifecycle_order() -> List[str]:
    return [stage.value for stage in ValidationLifecycle]


def mission_boundary_snapshot() -> Dict[str, Any]:
    return {system.value: boundary.as_dict() for system, boundary in MISSION_BOUNDARIES.items()}
