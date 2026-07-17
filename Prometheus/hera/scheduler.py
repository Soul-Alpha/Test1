from __future__ import annotations

from typing import List

from olympus.core.scheduler_interfaces import ScheduledTask


class HeraScheduler:
    def __init__(self) -> None:
        self._tasks: List[ScheduledTask] = []

    def register(self, task: ScheduledTask) -> None:
        self._tasks.append(task)

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks)
