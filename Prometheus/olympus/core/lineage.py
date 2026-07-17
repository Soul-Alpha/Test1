from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict


class EventType(str, Enum):
    SIGNAL_GENERATED = "signal_generated"
    PREDICTION_CREATED = "prediction_created"
    TRADE_ENTERED = "trade_entered"
    TRADE_MANAGED = "trade_managed"
    TRADE_CLOSED = "trade_closed"
    SIMULATION_CREATED = "simulation_created"
    PATTERN_LEARNED = "pattern_learned"
    ML_UPDATED = "ml_updated"
    DASHBOARD_UPDATED = "dashboard_updated"
    VERSION_USED = "version_used"


@dataclass
class LineageEvent:
    event_type: str
    source_system: str
    instrument: str
    timeframe: str
    model_version: str
    feature_version: str
    strategy_version: str
    dataset_generation: str
    execution_type: str
    timestamp: str
    payload: Dict[str, Any]


def append_lineage_event(root_dir: Path, event: LineageEvent) -> None:
    p = root_dir / "storage" / "olympus" / "event_lineage.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=True) + "\n")


def build_event(
    *,
    event_type: EventType,
    source_system: str,
    instrument: str,
    timeframe: str,
    model_version: str,
    feature_version: str,
    strategy_version: str,
    dataset_generation: str,
    execution_type: str,
    payload: Dict[str, Any] | None = None,
) -> LineageEvent:
    return LineageEvent(
        event_type=event_type.value,
        source_system=source_system,
        instrument=instrument,
        timeframe=timeframe,
        model_version=model_version,
        feature_version=feature_version,
        strategy_version=strategy_version,
        dataset_generation=dataset_generation,
        execution_type=execution_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload=payload or {},
    )
