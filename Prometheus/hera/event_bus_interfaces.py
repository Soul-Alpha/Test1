from __future__ import annotations

from typing import Protocol


class EventBusInterface(Protocol):
    def publish(self, topic: str, payload: dict) -> None:
        ...

    def subscribe(self, topic: str, handler_name: str) -> None:
        ...
