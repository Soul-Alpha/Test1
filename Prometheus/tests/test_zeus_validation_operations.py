from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.zeus_validation import ZeusValidationEngine
from olympus.core.zeus_validation_operations import run_zeus_validation_operations


def _build_candidate(*, candidate_id: str, sample_size: int, confidence: float = 0.8) -> dict:
    engine = ZeusValidationEngine()
    report = engine.validate_pattern_candidate(
        {
            "candidate_id": candidate_id,
            "source_system": "hermes",
            "validation_domain": "pattern",
            "sample_size": sample_size,
            "evidence": {
                "sample_size": sample_size,
                "confidence": confidence,
            },
            "priority": "High",
        }
    ).as_dict()
    report["confidence"] = confidence
    report["evidence_score"] = 0.85
    return report


def test_zvo_dedupes_candidates_and_advances_lifecycle(tmp_path: Path) -> None:
    root = tmp_path

    incoming = [_build_candidate(candidate_id="pat-001", sample_size=280)]
    # Repeated runs emulate dashboard refresh cycles and scheduler cadence.
    for _ in range(10):
        result = run_zeus_validation_operations(root_dir=root, incoming_reports=incoming)

    reports = result["reports"]
    assert len(reports) == 1
    row = reports[0]

    assert row["candidate_id"] == "pat-001"
    assert row["status"] == "passed"
    assert row["approved_for_adoption"] is False
    assert row["queue_state"] == "Awaiting Operator Approval"
    assert row["lifecycle"] == "awaiting_operator_approval"
    assert row["operator_approval_status"] == "Awaiting Explicit Operator Approval"
    assert row["validation_gate_passed"] is True
    assert row["adoption_gate_passed"] is True
    assert row["statistical_confidence_result"] == "Passed"
    assert result["status"]["summary"]["adoption_gates_passed"] == 1


def test_zvo_never_synthesizes_operator_approval(tmp_path: Path) -> None:
    incoming = [_build_candidate(candidate_id="pat-no-auto-approval", sample_size=280)]

    for _ in range(20):
        result = run_zeus_validation_operations(root_dir=tmp_path, incoming_reports=incoming)

    row = result["reports"][0]
    assert row["adoption_gate_passed"] is True
    assert row["approved_for_adoption"] is False
    assert row.get("approved_at") in (None, "")
    assert row["lifecycle"] != "operator_approved"
    assert row["lifecycle"] != "active"


def test_zvo_pauses_low_sample_candidates(tmp_path: Path) -> None:
    root = tmp_path
    incoming = [_build_candidate(candidate_id="pat-low", sample_size=5)]

    result = run_zeus_validation_operations(root_dir=root, incoming_reports=incoming)
    row = result["reports"][0]

    # First run starts execution pipeline.
    assert row["status"] in ("running", "inconclusive")

    result = run_zeus_validation_operations(root_dir=root, incoming_reports=incoming)
    row = result["reports"][0]
    assert row["status"] == "inconclusive"
    assert row["queue_state"] == "Paused"
    assert row["validation_gate_passed"] is False
    assert row["adoption_gate_passed"] is False
    assert "minimum_sample_size" in row["gate_blockers"]


def test_zvo_backfills_gate_fields_for_legacy_runtime_rows(tmp_path: Path) -> None:
    root = tmp_path
    storage = root / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "zeus_validation_operations_runtime.json").write_text(
        json.dumps(
            {
                "queue": {
                    "items": [
                        {
                            "candidate_id": "legacy-001",
                            "candidate_source_system": "hermes",
                            "domain": "pattern",
                            "status": "pending",
                            "timestamp": "2026-07-01 00:00:00 UTC",
                            "sample_size": 10,
                            "confidence": 0.4,
                            "evidence_score": 0.2,
                            "validation_pipeline": {
                                "Historical Validation": "Pending",
                                "Walk Forward": "Pending",
                                "Out-of-Sample": "Pending",
                                "Monte Carlo": "Pending",
                                "Robustness": "Pending",
                                "Statistical Confidence": "Pending",
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_zeus_validation_operations(root_dir=root, incoming_reports=[])
    row = result["reports"][0]

    assert row["minimum_sample_size_required"] == 20
    assert row["minimum_adoption_sample_size_required"] == 120
    assert row["validation_gate_passed"] is False
    assert row["adoption_gate_passed"] is False
    assert "gate_results" in row
