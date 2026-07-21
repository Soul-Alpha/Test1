"""Read-only, freshness-aware access to dashboard runtime artifacts.

This module deliberately has no Streamlit dependency.  Command centres may
cache its immutable snapshots, while tests and non-UI health checks can use the
same semantics without triggering an intelligence build or a trading action.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SnapshotStatus = Literal["current", "stale", "missing", "invalid"]


@dataclass(frozen=True)
class ArtifactSnapshot:
    path: str
    status: SnapshotStatus
    payload: Any
    observed_at: str
    modified_at: str | None = None
    age_seconds: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "payload": self.payload,
            "observed_at": self.observed_at,
            "modified_at": self.modified_at,
            "age_seconds": self.age_seconds,
            "error": self.error,
        }


def read_json_snapshot(path: Path, *, stale_after_seconds: float = 180.0) -> ArtifactSnapshot:
    """Read one JSON artifact without constructing or mutating institutional state."""
    now = datetime.now(UTC)
    observed_at = now.isoformat()
    if not path.exists():
        return ArtifactSnapshot(str(path), "missing", None, observed_at, error="artifact_not_found")

    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    age = max(0.0, (now - modified).total_seconds())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ArtifactSnapshot(
            str(path), "invalid", None, observed_at,
            modified_at=modified.isoformat(), age_seconds=age,
            error=f"{type(exc).__name__}: {exc}",
        )

    status: SnapshotStatus = "stale" if age > stale_after_seconds else "current"
    return ArtifactSnapshot(
        str(path), status, payload, observed_at,
        modified_at=modified.isoformat(), age_seconds=age,
    )
