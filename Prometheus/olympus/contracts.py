"""Shared Olympus metadata contracts.

These contracts standardize traceability across systems while preserving
independent ML pipelines and datasets per system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict


class SourceSystem(str, Enum):
    PROMETHEUS = "prometheus"
    HERMES = "hermes"
    ZEUS = "zeus"
    HERA = "hera"


class ExecutionType(str, Enum):
    LIVE = "live"
    SIMULATED = "simulated"
    BACKTEST = "backtest"
    SHADOW = "shadow"


@dataclass(frozen=True)
class TraceMetadata:
    source_system: SourceSystem
    model_version: str
    feature_version: str
    strategy_version: str
    instrument: str
    timeframe: str
    execution_type: ExecutionType
    timestamp: str
    dataset_generation: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeContract:
    """Additive shared metadata envelope for Olympus learned objects.

    This contract is observational metadata only and never modifies trading logic.
    """

    knowledge_id: str
    source_system: SourceSystem
    dataset_generation: str
    model_version: str
    feature_version: str
    strategy_version: str
    timestamp: str
    knowledge_version: str
    evidence_version: str
    confidence_version: str
    pattern_version: str
    trace_metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceConfidenceContract:
    """Shared confidence payload to prevent unsupported certainty displays."""

    implementation_pct: float
    evidence_pct: float
    knowledge_confidence_pct: float
    reliability: str
    sample_size: int
    confidence_interval: Dict[str, Any]
    historical_stability: float | str
    concept_drift: float | str
    evidence_level: float
    current_grade: str
    pending_validation: bool
    estimated_samples_remaining: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
