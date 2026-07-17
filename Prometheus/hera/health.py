from __future__ import annotations

from olympus.core.health import build_health_status


def get_hera_health() -> dict:
    return build_health_status(
        service="hera",
        healthy=True,
        details={"stage": "skeleton", "capabilities": ["registration", "scheduler", "gateway"]},
    ).as_dict()
