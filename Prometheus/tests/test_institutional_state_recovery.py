from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.core.institutional_state_recovery import recover_institutional_state


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_recovery_uses_idip_history_when_runtime_empty(tmp_path: Path) -> None:
    storage = tmp_path / "storage" / "olympus"
    live_bot = tmp_path / "live_bot"
    storage.mkdir(parents=True, exist_ok=True)
    live_bot.mkdir(parents=True, exist_ok=True)

    # Simulate truncated runtime file on restart.
    (storage / "idip_runtime.json").write_text("", encoding="utf-8")
    _append_jsonl(
        storage / "idip_history.jsonl",
        {
            "summary": {"sample_size": 77, "expectancy": 0.12},
            "engines": {
                "institutional_learning_scientist": {
                    "institutional_learning": {"sample_size": 77},
                    "knowledge_growth": {"total_knowledge_objects": 123},
                    "learning_velocity": {"learning_events": 88, "learning_velocity": 1.23},
                },
                "knowledge_evolution_engine": {
                    "summary": {
                        "objects_tracked": 123,
                        "action_counts": {"revalidate": 2, "refine": 3, "version_upgrade": 1},
                    }
                },
            },
        },
    )

    state = recover_institutional_state(tmp_path)
    assert state["idip_summary"].get("sample_size") == 77
    assert state["knowledge_growth"].get("total_knowledge_objects") == 123
    assert state["learning_velocity"].get("learning_events") == 88
    assert state["knowledge_evolution_payload"].get("summary", {}).get("objects_tracked") == 123


def test_recovery_rebuilds_zeus_summary_from_reports(tmp_path: Path) -> None:
    storage = tmp_path / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    (storage / "zeus_validation_status.json").write_text("", encoding="utf-8")
    _append_jsonl(storage / "zeus_validation_reports.jsonl", {"status": "passed", "queue_state": "Approved"})
    _append_jsonl(storage / "zeus_validation_reports.jsonl", {"status": "failed", "queue_state": "Rejected"})

    state = recover_institutional_state(tmp_path)
    summary = state["zeus_status"].get("summary", {})
    assert summary.get("total") == 2
    assert summary.get("validated_research") == 1
