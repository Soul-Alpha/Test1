from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SystemIdentity:
    system_id: str
    system_name: str
    system_role: str
    development_stage: str
    owner: str
    model_version: str
    dataset_generation: str
    build_version: str
    feature_version: str
    strategy_version: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
