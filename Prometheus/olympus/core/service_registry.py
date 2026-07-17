from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ServiceRegistration:
    name: str
    system: str
    version: str
    metadata: Dict[str, Any]


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: Dict[str, ServiceRegistration] = {}

    def register(self, registration: ServiceRegistration) -> None:
        self._services[registration.name] = registration

    def get(self, name: str) -> ServiceRegistration | None:
        return self._services.get(name)

    def list_services(self) -> list[ServiceRegistration]:
        return list(self._services.values())
