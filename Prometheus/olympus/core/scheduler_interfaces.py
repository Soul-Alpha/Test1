from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ScheduledTask:
    name: str
    cron: str
    enabled: bool = True


class SchedulerInterface(Protocol):
    def register_task(self, task: ScheduledTask) -> None:
        ...

    def run_due_tasks(self) -> None:
        ...
