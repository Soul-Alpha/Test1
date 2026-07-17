from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeraLifecycle:
    running: bool = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
