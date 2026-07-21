from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ui.dashboard_data_service import read_json_snapshot


def test_dashboard_entrypoints_define_three_command_centres() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        root / "ui" / "prometheus_command_center.py",
        root / "ui" / "hermes_command_center.py",
        root / "ui" / "olympus_command_center.py",
    }
    assert all(path.exists() for path in expected)

    startup = (root / "start_all.ps1").read_text(encoding="utf-8")
    assert startup.count("Start-Dashboard -Name") == 3
    assert "ui\\prometheus_command_center.py" in startup
    assert "ui\\hermes_command_center.py" in startup
    assert "ui\\olympus_command_center.py" in startup


def test_snapshot_reader_distinguishes_missing_invalid_and_current(tmp_path: Path) -> None:
    missing = read_json_snapshot(tmp_path / "missing.json")
    assert missing.status == "missing"
    assert missing.payload is None

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    invalid = read_json_snapshot(invalid_path)
    assert invalid.status == "invalid"
    assert invalid.error

    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    current = read_json_snapshot(current_path, stale_after_seconds=60)
    assert current.status == "current"
    assert current.payload == {"ok": True}


def test_snapshot_reader_marks_old_artifact_stale(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"generation": 1}), encoding="utf-8")
    old = datetime.now(UTC) - timedelta(minutes=10)
    timestamp = old.timestamp()
    path.touch()
    import os
    os.utime(path, (timestamp, timestamp))

    snapshot = read_json_snapshot(path, stale_after_seconds=60)
    assert snapshot.status == "stale"
    assert snapshot.age_seconds is not None and snapshot.age_seconds >= 60
