from __future__ import annotations

from olympus.core.service_registry import ServiceRegistration, ServiceRegistry


def create_hera_service_registry() -> ServiceRegistry:
    reg = ServiceRegistry()
    reg.register(
        ServiceRegistration(
            name="hera",
            system="hera",
            version="v0",
            metadata={"role": "orchestration_layer", "stage": "skeleton"},
        )
    )
    return reg
