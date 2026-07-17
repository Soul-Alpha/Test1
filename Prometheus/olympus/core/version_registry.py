from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class VersionRecord:
    system: str
    model_version: str
    feature_version: str
    strategy_version: str
    dataset_generation: str
    build_date: str
    record_count: int
    active: bool
    metadata: Dict[str, Any]


class VersionRegistry:
    def __init__(self, root_dir: Path) -> None:
        self.path = root_dir / "storage" / "olympus" / "version_registry.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, rec: VersionRecord) -> None:
        line = json.dumps(asdict(rec), ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def register_simple(
        self,
        *,
        system: str,
        model_version: str,
        feature_version: str,
        strategy_version: str,
        dataset_generation: str,
        record_count: int,
        active: bool,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.register(
            VersionRecord(
                system=system,
                model_version=model_version,
                feature_version=feature_version,
                strategy_version=strategy_version,
                dataset_generation=dataset_generation,
                build_date=datetime.now(timezone.utc).isoformat(),
                record_count=int(record_count),
                active=bool(active),
                metadata=metadata or {},
            )
        )

    def list_records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
