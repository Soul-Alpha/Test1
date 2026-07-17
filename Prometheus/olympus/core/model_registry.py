from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelRegistryEntry:
    model_name: str
    system: str
    version: str
    training_record_count: int
    training_date: str
    feature_version: str
    dataset_generation: str
    description: str
    status: str
    metadata: Dict[str, Any]


class ModelRegistry:
    def __init__(self, root_dir: Path) -> None:
        self.path = root_dir / "storage" / "olympus" / "model_registry.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, entry: ModelRegistryEntry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=True) + "\n")

    def register_simple(
        self,
        *,
        model_name: str,
        system: str,
        version: str,
        training_record_count: int,
        feature_version: str,
        dataset_generation: str,
        description: str,
        status: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.register(
            ModelRegistryEntry(
                model_name=model_name,
                system=system,
                version=version,
                training_record_count=int(training_record_count),
                training_date=datetime.now(timezone.utc).isoformat(),
                feature_version=feature_version,
                dataset_generation=dataset_generation,
                description=description,
                status=status,
                metadata=metadata or {},
            )
        )

    def list_entries(self) -> List[Dict[str, Any]]:
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
