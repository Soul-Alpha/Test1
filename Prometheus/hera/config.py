from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeraConfig:
    system_id: str = "hera-core"
    system_name: str = "Hera"
    system_role: str = "orchestration_layer"
    development_stage: str = "skeleton"
    owner: str = "olympus"
