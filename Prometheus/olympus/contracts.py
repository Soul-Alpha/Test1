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

# Execution benchmark v1 is observational and separate from legacy validation.
@dataclass(frozen=True)
class BenchmarkLimits:
    minimum_closed_trades: int
    minimum_expectancy: float
    minimum_profit_factor: float
    maximum_realized_drawdown_pct: float
    maximum_adverse_slippage_bps: float
    maximum_latency_seconds: float


@dataclass(frozen=True)
class BenchmarkCandidate:
    candidate_id: str
    source_system: str
    model_version: str
    feature_version: str
    strategy_version: str
    dataset_generation: str
    instrument: str
    timeframe: str
    account_currency: str
    venue: str
    parameters_json: str
    frozen_at: str
    initial_balance: float
    limits: BenchmarkLimits


@dataclass(frozen=True)
class BenchmarkObservation:
    observation_id: str
    candidate_id: str
    candidate_fingerprint: str
    source_system: str
    stage: str
    signal_id: str
    decision_at: str
    side: str
    status: str
    intended_entry: float | None = None
    intended_size: float | None = None
    intended_stop: float | None = None
    intended_target: float | None = None
    filled_at: str | None = None
    actual_entry: float | None = None
    actual_size: float | None = None
    actual_stop: float | None = None
    actual_target: float | None = None
    closed_at: str | None = None
    actual_exit: float | None = None
    gross_pnl: float | None = None
    commission_cost: float | None = None
    swap_cost: float | None = None
    other_cost: float | None = None
    spread_bps: float | None = None
    exit_reason: str | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BenchmarkStageReport:
    stage: str
    observations: int
    closed_trades: int
    complete_trades: int
    net_pnl: float | None
    expectancy: float | None
    profit_factor: float | None
    no_losing_trades: bool
    realized_drawdown_pct: float | None
    mean_adverse_slippage_bps: float | None
    mean_latency_seconds: float | None
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkComparison:
    left_stage: str
    right_stage: str
    matched_signals: int
    unmatched_left: int
    unmatched_right: int
    incompatible_signals: int
    entry_pairs: int
    size_pairs: int
    pnl_pairs: int
    mean_entry_difference_bps: float | None
    mean_size_difference: float | None
    mean_net_pnl_difference: float | None


@dataclass(frozen=True)
class ExecutionBenchmarkReport:
    candidate_id: str
    candidate_fingerprint: str
    source_system: str
    generated_at: str
    evaluation_start: str
    evaluation_end: str
    stages: tuple[BenchmarkStageReport, ...]
    comparisons: tuple[BenchmarkComparison, ...]
    evidence_score: float
    governance_flags: tuple[str, ...] = (
        "advisory_only", "explicit_operator_approval_required",
        "realized_drawdown_excludes_open_positions",
        "serial_trade_uncertainty_not_estimated",
        "legacy_zeus_gates_not_overridden",
    )
    no_auto_adopt: bool = True
    approved_for_adoption: bool = False
    benchmark_version: str = "execution-benchmark-v1"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
