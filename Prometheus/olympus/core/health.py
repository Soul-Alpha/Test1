from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class HealthStatus:
    service: str
    healthy: bool
    checked_at: str
    details: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_health_status(service: str, healthy: bool, details: Dict[str, Any] | None = None) -> HealthStatus:
    return HealthStatus(
        service=service,
        healthy=healthy,
        checked_at=datetime.now(timezone.utc).isoformat(),
        details=details or {},
    )
