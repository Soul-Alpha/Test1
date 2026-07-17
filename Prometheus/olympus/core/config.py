from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OlympusCoreConfig:
    """Global Olympus infrastructure settings.

    Isolation defaults are strict (cross-system access denied) but configurable.
    """

    root_dir: Path
    allow_cross_system_models: bool = False
    allow_cross_system_datasets: bool = False
    allow_cross_system_configs: bool = False
    allow_cross_system_strategies: bool = False

    @property
    def storage_dir(self) -> Path:
        return self.root_dir / "storage" / "olympus"

    @property
    def research_dir(self) -> Path:
        return self.root_dir / "storage" / "olympus_research"

    @classmethod
    def from_env(cls, root_dir: Path) -> "OlympusCoreConfig":
        return cls(
            root_dir=root_dir,
            allow_cross_system_models=_env_bool("ALLOW_CROSS_SYSTEM_MODELS", False),
            allow_cross_system_datasets=_env_bool("ALLOW_CROSS_SYSTEM_DATASETS", False),
            allow_cross_system_configs=_env_bool("ALLOW_CROSS_SYSTEM_CONFIGS", False),
            allow_cross_system_strategies=_env_bool("ALLOW_CROSS_SYSTEM_STRATEGIES", False),
        )
