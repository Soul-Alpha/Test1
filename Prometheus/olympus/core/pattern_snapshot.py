from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def append_pattern_snapshot(root_dir: Path, snapshot: Dict[str, Any]) -> None:
    """Append-only pattern snapshot store for Hermes market observations."""
    p = root_dir / "storage" / "olympus" / "pattern_snapshots.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(snapshot)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
