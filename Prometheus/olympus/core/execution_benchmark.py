"""Immutable execution evidence and deterministic Zeus benchmark reports.

No MT5 imports, order submission, strategy adoption, or legacy gate mutation.
Observations are finalized per-signal outcomes, not individual partial fills.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from olympus.contracts import (
    BenchmarkCandidate, BenchmarkComparison, BenchmarkLimits,
    BenchmarkObservation, BenchmarkStageReport, ExecutionBenchmarkReport,
)

STAGES = ("backtest", "paper", "live")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Benchmark timestamps require a timezone")
    return result.astimezone(UTC)


def _number(value: float, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def candidate_from_dict(payload: dict[str, Any]) -> BenchmarkCandidate:
    data = dict(payload)
    data["limits"] = BenchmarkLimits(**data["limits"])
    return BenchmarkCandidate(**data)


def observation_from_dict(payload: dict[str, Any]) -> BenchmarkObservation:
    return BenchmarkObservation(**payload)


def candidate_fingerprint(candidate: BenchmarkCandidate) -> str:
    return hashlib.sha256(_json(asdict(candidate)).encode("utf-8")).hexdigest()


def _validate_candidate(candidate: BenchmarkCandidate) -> None:
    if candidate.source_system not in ("prometheus", "hermes"):
        raise ValueError("Benchmark source must be prometheus or hermes")
    for key, value in asdict(candidate).items():
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{key} must not be empty")
    _time(candidate.frozen_at)
    if not isinstance(json.loads(candidate.parameters_json), dict):
        raise ValueError("parameters_json must encode an object")
    _number(candidate.initial_balance, "initial_balance", positive=True)
    limits = candidate.limits
    if type(limits.minimum_closed_trades) is not int or limits.minimum_closed_trades < 2:
        raise ValueError("minimum_closed_trades must be an integer >= 2")
    for key, value in asdict(limits).items():
        _number(value, key)
    if limits.minimum_profit_factor < 1:
        raise ValueError("minimum_profit_factor must be >= 1")
    if not 0 < limits.maximum_realized_drawdown_pct <= 100:
        raise ValueError("maximum_realized_drawdown_pct must be in (0, 100]")
    if limits.maximum_adverse_slippage_bps < 0 or limits.maximum_latency_seconds < 0:
        raise ValueError("Execution limits must be nonnegative")


class ExecutionBenchmarkLedger:
    """Append-only SQLite ledger with atomic uniqueness and replay protection.

    Opening an existing ledger is read-only at the application level: schema
    setup happens only on explicit registration or observation writes.
    """
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30)
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    source TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(source, candidate_id)
                );
                CREATE TABLE IF NOT EXISTS observations (
                    source TEXT NOT NULL, observation_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL, stage TEXT NOT NULL,
                    signal_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(source, observation_id),
                    UNIQUE(source, candidate_id, stage, signal_id)
                );
                CREATE TRIGGER IF NOT EXISTS candidates_no_update
                BEFORE UPDATE ON candidates BEGIN SELECT RAISE(ABORT, 'immutable candidate'); END;
                CREATE TRIGGER IF NOT EXISTS candidates_no_delete
                BEFORE DELETE ON candidates BEGIN SELECT RAISE(ABORT, 'immutable candidate'); END;
                CREATE TRIGGER IF NOT EXISTS observations_no_update
                BEFORE UPDATE ON observations BEGIN SELECT RAISE(ABORT, 'immutable observation'); END;
                CREATE TRIGGER IF NOT EXISTS observations_no_delete
                BEFORE DELETE ON observations BEGIN SELECT RAISE(ABORT, 'immutable observation'); END;
            """)
            return connection
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)

    def register(self, candidate: BenchmarkCandidate) -> str:
        _validate_candidate(candidate)
        payload = _json(asdict(candidate))
        connection = self._connect(write=True)
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                old = connection.execute(
                    "SELECT payload FROM candidates WHERE source=? AND candidate_id=?",
                    (candidate.source_system, candidate.candidate_id),
                ).fetchone()
                if old and old[0] != payload:
                    raise ValueError("Candidate is frozen; register a new candidate_id")
                connection.execute(
                    "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?)",
                    (candidate.source_system, candidate.candidate_id, payload),
                )
        finally:
            connection.close()
        return candidate_fingerprint(candidate)

    def candidates(self) -> list[BenchmarkCandidate]:
        if not self.path.exists():
            return []
        connection = self._connect()
        try:
            return [candidate_from_dict(json.loads(row[0])) for row in connection.execute(
                "SELECT payload FROM candidates ORDER BY source, candidate_id"
            )]
        finally:
            connection.close()

    def candidate(self, source: str, candidate_id: str) -> BenchmarkCandidate:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload FROM candidates WHERE source=? AND candidate_id=?",
                (source, candidate_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("Candidate must be explicitly registered first")
        return candidate_from_dict(json.loads(row[0]))

    def append(self, observation: BenchmarkObservation) -> bool:
        candidate = self.candidate(observation.source_system, observation.candidate_id)
        if observation.candidate_fingerprint != candidate_fingerprint(candidate):
            raise ValueError("Observation does not match frozen candidate fingerprint")
        if observation.stage not in STAGES or observation.side not in ("buy", "sell"):
            raise ValueError("Unknown stage or side")
        if observation.status not in ("closed", "rejected"):
            raise ValueError("Only finalized closed or rejected outcomes are accepted")
        if not observation.observation_id.strip() or not observation.signal_id.strip():
            raise ValueError("Observation and signal identifiers are required")
        decision = _time(observation.decision_at)
        if observation.stage != "backtest" and decision < _time(candidate.frozen_at):
            raise ValueError("Prospective decisions must follow candidate freezing")
        for name in ("intended_entry", "intended_size", "intended_stop", "intended_target",
                     "actual_entry", "actual_size", "actual_stop", "actual_target", "actual_exit"):
            value = getattr(observation, name)
            if value is not None:
                _number(value, name, positive=True)
        for name in ("gross_pnl", "commission_cost", "swap_cost", "other_cost", "spread_bps"):
            value = getattr(observation, name)
            if value is not None:
                _number(value, name)
                if name in ("commission_cost", "other_cost", "spread_bps") and value < 0:
                    raise ValueError(f"{name} must be nonnegative")
        fill = _time(observation.filled_at) if observation.filled_at else None
        close = _time(observation.closed_at) if observation.closed_at else None
        if fill and fill < decision:
            raise ValueError("Fill precedes decision")
        if close and (close < decision or (fill and close < fill)):
            raise ValueError("Close precedes decision or fill")
        if observation.status == "rejected":
            if not observation.rejection_reason:
                raise ValueError("Rejected signals require a reason")
            if any(getattr(observation, name) is not None for name in (
                "filled_at", "actual_entry", "actual_size", "closed_at", "actual_exit", "gross_pnl"
            )):
                raise ValueError("Rejected signals cannot carry fills or trade P&L")
        payload = _json(asdict(observation))
        connection = self._connect(write=True)
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT payload FROM observations WHERE source=? AND "
                    "(observation_id=? OR (candidate_id=? AND stage=? AND signal_id=?))",
                    (observation.source_system, observation.observation_id, observation.candidate_id,
                     observation.stage, observation.signal_id),
                ).fetchall()
                if existing:
                    if len(existing) == 1 and existing[0][0] == payload:
                        return False
                    raise ValueError("Conflicting replay; historical evidence is immutable")
                connection.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)", (
                    observation.source_system, observation.observation_id, observation.candidate_id,
                    observation.stage, observation.signal_id, payload,
                ))
        finally:
            connection.close()
        return True

    def observations(self, source: str, candidate_id: str) -> list[BenchmarkObservation]:
        connection = self._connect()
        try:
            return [observation_from_dict(json.loads(row[0])) for row in connection.execute(
                "SELECT payload FROM observations WHERE source=? AND candidate_id=?",
                (source, candidate_id),
            )]
        finally:
            connection.close()


def _net(row: BenchmarkObservation) -> float | None:
    values = (row.gross_pnl, row.commission_cost, row.swap_cost, row.other_cost)
    if row.status != "closed" or any(value is None for value in values):
        return None
    return row.gross_pnl - row.commission_cost - row.swap_cost - row.other_cost


def _average(values: list[float]) -> float | None:
    return mean(values) if values else None


def stage_report(candidate: BenchmarkCandidate, stage: str,
                 rows: list[BenchmarkObservation], end: datetime) -> BenchmarkStageReport:
    closed = [row for row in rows if row.status == "closed"]
    required = ("intended_entry", "intended_size", "intended_stop", "intended_target",
                "filled_at", "actual_entry", "actual_size", "actual_stop", "actual_target",
                "closed_at", "actual_exit", "gross_pnl", "commission_cost", "swap_cost",
                "other_cost", "spread_bps", "exit_reason")
    complete = [row for row in closed if all(getattr(row, key) is not None for key in required)
                and bool(row.exit_reason) and _time(row.closed_at) < end]
    pnl = [_net(row) for row in sorted(complete, key=lambda item: (_time(item.closed_at), item.signal_id))]
    wins = sum(value for value in pnl if value > 0)
    losses = -sum(value for value in pnl if value < 0)
    profit_factor = wins / losses if losses else None
    equity = peak = candidate.initial_balance
    drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak * 100)
    slippage = [
        max(0.0, (row.actual_entry - row.intended_entry) / row.intended_entry * 10000
            * (1 if row.side == "buy" else -1))
        for row in complete
    ]
    latency = [(_time(row.filled_at) - _time(row.decision_at)).total_seconds() for row in complete]
    limits = candidate.limits
    blockers = []
    if any(row.closed_at and _time(row.closed_at) >= end for row in closed):
        blockers.append("outcomes_outside_evaluation_window")
    if len(closed) != len(complete):
        blockers.append("unknown_trade_fields_or_costs")
    if len(complete) < limits.minimum_closed_trades:
        blockers.append("minimum_closed_trades")
    expectancy = _average(pnl)
    if expectancy is None or expectancy <= limits.minimum_expectancy:
        blockers.append("net_expectancy")
    if not pnl or (losses > 0 and profit_factor < limits.minimum_profit_factor) or (losses == 0 and wins == 0):
        blockers.append("profit_factor")
    if pnl and drawdown > limits.maximum_realized_drawdown_pct:
        blockers.append("realized_drawdown")
    if slippage and max(slippage) > limits.maximum_adverse_slippage_bps:
        blockers.append("adverse_slippage")
    if latency and max(latency) > limits.maximum_latency_seconds:
        blockers.append("execution_latency")
    return BenchmarkStageReport(
        stage, len(rows), len(closed), len(complete), sum(pnl) if pnl else None,
        expectancy, profit_factor, bool(pnl) and losses == 0, drawdown if pnl else None,
        _average(slippage), _average(latency), tuple(blockers),
    )


def compare_stages(left_stage: str, right_stage: str,
                   rows: list[BenchmarkObservation]) -> BenchmarkComparison:
    left = {row.signal_id: row for row in rows if row.stage == left_stage}
    right = {row.signal_id: row for row in rows if row.stage == right_stage}
    common = left.keys() & right.keys()
    entry, size, pnl = [], [], []
    incompatible = 0
    intent_fields = ("side", "intended_entry", "intended_size", "intended_stop", "intended_target")
    for key in sorted(common):
        a, b = left[key], right[key]
        if _time(a.decision_at) != _time(b.decision_at) or any(
            getattr(a, field) != getattr(b, field) for field in intent_fields
        ) or any(getattr(a, field) is None for field in intent_fields):
            incompatible += 1
            continue
        if a.status == b.status == "closed":
            if a.actual_entry is not None and b.actual_entry is not None:
                entry.append((b.actual_entry - a.actual_entry) / a.actual_entry * 10000
                             * (1 if a.side == "buy" else -1))
            if a.actual_size is not None and b.actual_size is not None:
                size.append(b.actual_size - a.actual_size)
            if _net(a) is not None and _net(b) is not None:
                pnl.append(_net(b) - _net(a))
    return BenchmarkComparison(
        left_stage, right_stage, len(common) - incompatible,
        len(left.keys() - right.keys()), len(right.keys() - left.keys()),
        incompatible, len(entry), len(size), len(pnl),
        _average(entry), _average(size), _average(pnl),
    )


def build_benchmark_report(ledger: ExecutionBenchmarkLedger, source: str,
                           candidate_id: str, start: str, end: str) -> ExecutionBenchmarkReport:
    """Use a common [start, end) decision window; flag unfinished outcomes.

    All-stage comparison is descriptive. It never overrides Zeus adoption gates.
    """
    begin, finish = _time(start), _time(end)
    if begin >= finish:
        raise ValueError("Evaluation start must precede end")
    candidate = ledger.candidate(source, candidate_id)
    rows = [
        row for row in ledger.observations(source, candidate_id)
        if begin <= _time(row.decision_at) < finish
    ]
    stages = tuple(stage_report(candidate, stage, [row for row in rows if row.stage == stage], finish)
                   for stage in STAGES)
    comparisons = tuple(compare_stages(a, b, [row for row in rows if row.status == "rejected"
                               or (row.closed_at and _time(row.closed_at) < finish)]) for a, b in (
        ("backtest", "paper"), ("paper", "live"), ("backtest", "live")
    ))
    denominator = sum(stage.closed_trades for stage in stages)
    return ExecutionBenchmarkReport(
        candidate_id, candidate_fingerprint(candidate), source, datetime.now(UTC).isoformat(), start, end,
        stages, comparisons,
        sum(stage.complete_trades for stage in stages) / denominator if denominator else 0.0,
    )
