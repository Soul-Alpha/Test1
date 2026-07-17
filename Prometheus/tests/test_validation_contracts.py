import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.contracts import SourceSystem
from olympus.core.validation_contracts import (
    MISSION_BOUNDARIES,
    RecommendationCandidate,
    ValidationLifecycle,
    mission_boundary_snapshot,
    validation_lifecycle_order,
)
from backtesting.zeus_validation import ZeusValidationEngine, build_validation_status


def test_validation_lifecycle_order_preserves_operator_approval_gate():
    order = validation_lifecycle_order()

    assert order == [
        "learning",
        "candidate",
        "zeus_validation",
        "validated",
        "operator_approved",
        "active",
        "monitoring",
        "retired",
    ]
    assert order.index("operator_approved") < order.index("active")


def test_mission_boundaries_keep_systems_independent():
    snapshot = mission_boundary_snapshot()

    assert "train_using_hermes_datasets" in snapshot["prometheus"]["prohibited_actions"]
    assert "produce_execution_decisions" in snapshot["hermes"]["prohibited_actions"]
    assert "automatically_deploy_validated_improvements" in snapshot["zeus"]["prohibited_actions"]
    assert MISSION_BOUNDARIES[SourceSystem.HERA].mission.value == "institutional_governance"


def test_zeus_recommendation_validation_is_passive_and_requires_approval():
    candidate = RecommendationCandidate(
        candidate_id="rec-001",
        source_system=SourceSystem.PROMETHEUS,
        timestamp="2026-07-05 00:00:00 UTC",
        recommendation_type="execution_location",
        recommendation="Raise location quality threshold after validation.",
        evidence={"sample_size": 12, "confidence": 55.0},
        lifecycle=ValidationLifecycle.CANDIDATE,
    )

    report = ZeusValidationEngine().validate_recommendation(candidate)

    assert report.validating_system == SourceSystem.ZEUS
    assert report.candidate_source_system == SourceSystem.PROMETHEUS
    assert report.operator_approval_required is True
    assert report.approved_for_adoption is False
    assert report.lifecycle == ValidationLifecycle.ZEUS_VALIDATION
    assert report.minimum_sample_size_required == 20
    assert report.minimum_adoption_sample_size_required == 120
    assert report.validation_gate_passed is False
    assert report.adoption_gate_passed is False
    assert "minimum_sample_size" in report.gate_blockers
    assert "statistical_confidence" in report.gate_blockers
    assert report.gate_results["minimum_confidence"]["passed"] is True


def test_validation_status_never_implies_automatic_deployment():
    engine = ZeusValidationEngine()
    reports = engine.validate_candidates(
        [
            {"candidate_id": "feat-001", "validation_domain": "feature"},
            {"candidate_id": "pat-001", "validation_domain": "pattern", "occurrences": 8},
        ]
    )

    status = build_validation_status(reports)

    assert status["summary"]["total"] == 2
    assert status["summary"]["automatic_deployment"] is False
    assert status["summary"]["operator_approval_required"] == 2
    assert "validation_gates_passed" in status["summary"]
    assert "adoption_gates_passed" in status["summary"]


def test_feature_validation_requires_leakage_safe_gate():
    report = ZeusValidationEngine().validate_feature_candidate(
        {
            "candidate_id": "feat-001",
            "validation_domain": "feature",
            "outcome_diagnostics_excluded": False,
        }
    )

    assert report.leakage_safe_required is True
    assert report.validation_gate_passed is False
    assert report.adoption_gate_passed is False
    assert "leakage_safety" in report.gate_blockers
    assert report.gate_results["leakage_safety"]["passed"] is False
