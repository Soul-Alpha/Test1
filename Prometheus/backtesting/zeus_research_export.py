"""Manual Zeus research export helpers.

Exports research artifacts into Olympus Research Repository only.
No automatic deployment or strategy mutation is performed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from olympus.core.research_repository import ResearchRepository


def export_zeus_result(
    *,
    root_dir: str | Path,
    title: str,
    summary: str,
    payload: Dict[str, Any] | None = None,
    artifact_type: str = "strategy_comparison",
) -> None:
    repo = ResearchRepository(Path(root_dir))
    repo.add_simple(
        artifact_type=artifact_type,
        source_system="zeus",
        title=title,
        summary=summary,
        payload=payload or {},
    )
