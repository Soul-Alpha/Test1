from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Protocol


@dataclass
class FeatureSnapshot:
    source_system: str
    feature_version: str
    instrument: str
    timeframe: str
    timestamp: str
    market_structure: Dict[str, Any]
    liquidity: Dict[str, Any]
    trend: Dict[str, Any]
    order_blocks: Dict[str, Any]
    fair_value_gaps: Dict[str, Any]
    volatility: Dict[str, Any]
    sessions: Dict[str, Any]
    statistical: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureStoreInterface(Protocol):
    def put_snapshot(self, snapshot: FeatureSnapshot) -> None:
        ...

    def get_latest(self, instrument: str, timeframe: str) -> FeatureSnapshot | None:
        ...


class FeatureProviderInterface(Protocol):
    def compute_features(self, *args: Any, **kwargs: Any) -> FeatureSnapshot:
        ...
