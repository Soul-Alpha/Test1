from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class ApiRequest:
    route: str
    payload: Dict[str, Any]


@dataclass
class ApiResponse:
    ok: bool
    payload: Dict[str, Any]
    error: str = ""


class ApiGatewayInterface(Protocol):
    def handle(self, request: ApiRequest) -> ApiResponse:
        ...
