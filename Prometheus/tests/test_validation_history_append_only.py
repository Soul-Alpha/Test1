from __future__ import annotations

import json
from pathlib import Path

from olympus.core.prometheus_evolution_intelligence import _append_validation_report_events


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_validation_report_history_appends_new_states_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "zeus_validation_reports.jsonl"
    first = {"candidate_id": "c-1", "status": "running", "lifecycle": "zeus_validation", "last_transition_at": "t1"}
    second = {"candidate_id": "c-1", "status": "passed", "lifecycle": "validated", "last_transition_at": "t2"}

    _append_validation_report_events(path, [first])
    _append_validation_report_events(path, [first, second])

    assert _rows(path) == [first, second]


def test_validation_report_history_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "zeus_validation_reports.jsonl"
    historical = {"candidate_id": "historic", "status": "failed", "lifecycle": "validated", "timestamp": "old"}
    path.write_text(json.dumps(historical) + "\n", encoding="utf-8")

    current = {"candidate_id": "current", "status": "passed", "lifecycle": "validated", "timestamp": "new"}
    _append_validation_report_events(path, [current])

    assert _rows(path) == [historical, current]
