from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from olympus.core.logging_utils import get_olympus_logger

logger = get_olympus_logger("isolation")


@dataclass(frozen=True)
class IsolationPolicy:
    allow_cross_system_models: bool = False
    allow_cross_system_datasets: bool = False
    allow_cross_system_configs: bool = False
    allow_cross_system_strategies: bool = False


class IsolationGuard:
    """Runtime guard preventing accidental cross-system resource access."""

    def __init__(self, system: str, policy: IsolationPolicy) -> None:
        self.system = (system or "").lower().strip()
        self.policy = policy

    def _is_foreign(self, path: str | Path) -> bool:
        p = str(path).lower()
        if self.system == "":
            return False
        if self.system in p:
            return False
        # Known Olympus systems
        known = ["prometheus", "hermes", "zeus", "hera"]
        for s in known:
            if s != self.system and s in p:
                return True
        return False

    def guard_model_path(self, path: str | Path) -> bool:
        if self.policy.allow_cross_system_models:
            return True
        blocked = self._is_foreign(path)
        if blocked:
            logger.warning("Blocked cross-system model access | system=%s path=%s", self.system, path)
        return not blocked

    def guard_dataset_path(self, path: str | Path) -> bool:
        if self.policy.allow_cross_system_datasets:
            return True
        blocked = self._is_foreign(path)
        if blocked:
            logger.warning("Blocked cross-system dataset access | system=%s path=%s", self.system, path)
        return not blocked

    def guard_config_path(self, path: str | Path) -> bool:
        if self.policy.allow_cross_system_configs:
            return True
        blocked = self._is_foreign(path)
        if blocked:
            logger.warning("Blocked cross-system config access | system=%s path=%s", self.system, path)
        return not blocked

    def guard_strategy_path(self, path: str | Path) -> bool:
        if self.policy.allow_cross_system_strategies:
            return True
        blocked = self._is_foreign(path)
        if blocked:
            logger.warning("Blocked cross-system strategy access | system=%s path=%s", self.system, path)
        return not blocked
