from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class ResearchArtifact:
    artifact_type: str
    source_system: str
    title: str
    summary: str
    timestamp: str
    payload: Dict[str, Any]


class ResearchRepository:
    """Stores research artifacts only; never influences live trading directly."""

    def __init__(self, root_dir: Path) -> None:
        self.path = root_dir / "storage" / "olympus_research" / "artifacts.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, artifact: ResearchArtifact) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(artifact), ensure_ascii=True) + "\n")

    def add_simple(
        self,
        *,
        artifact_type: str,
        source_system: str,
        title: str,
        summary: str,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        self.add(
            ResearchArtifact(
                artifact_type=artifact_type,
                source_system=source_system,
                title=title,
                summary=summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload=payload or {},
            )
        )
