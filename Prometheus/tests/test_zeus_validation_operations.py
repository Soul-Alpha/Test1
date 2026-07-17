from __future__ import annotations

import sys
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
    assert row["approved_for_adoption"] is True
    assert row["queue_state"] in ("Approved", "Completed")
    assert row["lifecycle"] in ("validated", "active", "completed")


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
