"""Durable, append-only historical evidence for Hermes analytics.

The runtime status JSON files are intentionally small snapshots.  They are not
historical stores and are replaced when a bot restarts.  This module provides a
separate append-only ledger for prediction and paper-trade evidence without
changing signal generation or execution behaviour.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
SAMPLE_THRESHOLDS = {
    "insufficient": 10,
    "emerging": 30,
    "developing": 75,
    "validated": 150,
    "elite": 300,
}
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_evidence_id(namespace: str, *parts: Any, length: int = 24) -> str:
    """Return a deterministic identifier for an evidence observation."""
    raw = "|".join([namespace, *(str(part).strip() for part in parts)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class HistoricalEvidenceLedger:
    """Append-only, idempotent evidence ledger scoped to one source system."""

    def __init__(self, root_dir: Path, source_system: str = "hermes") -> None:
        self.root_dir = Path(root_dir)
        self.source_system = source_system
        self.path = self.root_dir / "storage" / "olympus" / f"{source_system}_evidence.jsonl"
        self.last_error: str | None = None
        self._records = self._load_records()
        self._event_ids = {
            str(row.get("event_id"))
            for row in self._records
            if row.get("event_id")
        }

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def append(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
        *,
        occurred_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Append one event unless the same entity event already exists."""
        if not entity_id:
            return False
        event_id = stable_evidence_id(
            "historical_evidence",
            self.source_system,
            event_type,
            entity_type,
            entity_id,
        )
        if event_id in self._event_ids:
            return False

        row = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "source_system": self.source_system,
            "occurred_at": occurred_at or _utc_now(),
            "recorded_at": _utc_now(),
            "payload": payload,
            "metadata": metadata or {},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(row, default=_json_default, sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            self.last_error = str(exc)
            logger.error("Historical evidence append failed: %s", exc)
            return False

        self._records.append(row)
        self._event_ids.add(event_id)
        self.last_error = None
        return True

    def records(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type is None:
            return list(self._records)
        return [row for row in self._records if row.get("event_type") == event_type]

    def append_prediction(self, signal: dict[str, Any], metadata: dict[str, Any] | None = None) -> bool:
        signal_id = str(signal.get("signal_id", "") or "")
        return self.append(
            "prediction_created",
            "signal",
            signal_id,
            signal,
            occurred_at=str(signal.get("timestamp") or _utc_now()),
            metadata=metadata,
        )

    def append_trade_opened(self, trade: dict[str, Any], metadata: dict[str, Any] | None = None) -> bool:
        trade_id = str(trade.get("trade_id", "") or "")
        return self.append(
            "trade_opened",
            "trade",
            trade_id,
            trade,
            occurred_at=str(trade.get("opened_at") or _utc_now()),
            metadata=metadata,
        )

    def append_trade_closed(
        self,
        trade: dict[str, Any],
        *,
        closed_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        trade_id = str(trade.get("trade_id", "") or "")
        payload = dict(trade)
        timestamp_quality = str((metadata or {}).get("timestamp_quality") or "exact")
        payload["closed_at"] = closed_at or payload.get("closed_at") or _utc_now()
        payload["closed_at_quality"] = timestamp_quality
        return self.append(
            "trade_closed",
            "trade",
            trade_id,
            payload,
            occurred_at=str(payload["closed_at"]),
            metadata=metadata,
        )

    def ingest_runtime_status(self, status: dict[str, Any]) -> dict[str, int]:
        """Backfill the previous status snapshot before it is replaced on restart."""
        added = {"predictions": 0, "opened_trades": 0, "closed_trades": 0}
        signal_rows = [
            *(status.get("signals") or []),
            *(status.get("skipped_signals") or []),
        ]
        if status.get("last_signal"):
            signal_rows.append(status["last_signal"])
        for signal in signal_rows:
            if isinstance(signal, dict) and self.append_prediction(signal, {"backfill_source": "runtime_status"}):
                added["predictions"] += 1

        for trade in status.get("open_trades") or []:
            if isinstance(trade, dict) and self.append_trade_opened(trade, {"backfill_source": "runtime_status"}):
                added["opened_trades"] += 1

        for trade in status.get("closed_trades") or []:
            if isinstance(trade, dict) and self.append_trade_closed(
                trade,
                closed_at=str(trade.get("closed_at") or status.get("last_poll") or _utc_now()),
                metadata={
                    "backfill_source": "runtime_status",
                    "timestamp_quality": "snapshot_observed",
                },
            ):
                added["closed_trades"] += 1
        return added

    def trade_state(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Reconstruct current open and complete closed trade state."""
        opened: dict[str, dict[str, Any]] = {}
        closed: dict[str, dict[str, Any]] = {}
        for row in self._records:
            trade_id = str(row.get("entity_id", "") or "")
            payload = row.get("payload") or {}
            if not trade_id or not isinstance(payload, dict):
                continue
            if row.get("event_type") == "trade_opened":
                opened[trade_id] = dict(payload)
            elif row.get("event_type") == "trade_closed":
                closed[trade_id] = dict(payload)
                opened.pop(trade_id, None)
        return list(opened.values()), list(closed.values())

    def predictions_by_signal(self) -> dict[str, dict[str, Any]]:
        return {
            str(row.get("entity_id")): dict(row.get("payload") or {})
            for row in self._records
            if row.get("event_type") == "prediction_created" and row.get("entity_id")
        }


def _sample_stage(sample_count: int) -> str:
    if sample_count < SAMPLE_THRESHOLDS["insufficient"]:
        return "insufficient"
    if sample_count < SAMPLE_THRESHOLDS["emerging"]:
        return "emerging"
    if sample_count < SAMPLE_THRESHOLDS["developing"]:
        return "developing"
    if sample_count < SAMPLE_THRESHOLDS["validated"]:
        return "validated"
    if sample_count < SAMPLE_THRESHOLDS["elite"]:
        return "strongly_validated"
    return "elite"


def _family(
    *,
    sample_count: int,
    minimum_required: int,
    source: str,
    missing_fields: Iterable[str] = (),
) -> dict[str, Any]:
    missing = sorted(set(missing_fields))
    if sample_count <= 0:
        status = "missing_history"
    elif sample_count < minimum_required:
        status = "insufficient_samples"
    elif missing:
        status = "missing_fields"
    else:
        status = "ready"
    return {
        "status": status,
        "sample_count": sample_count,
        "minimum_required": minimum_required,
        "sample_stage": _sample_stage(sample_count),
        "source": source,
        "missing_fields": missing,
    }


def build_evidence_readiness(
    setup_records: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    *,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe why a metric family is ready, incomplete, or still awaiting data."""
    status = status or {}
    labeled = [row for row in setup_records if row.get("outcome") is not None]
    predictions = {
        str(row.get("entity_id")): row.get("payload") or {}
        for row in evidence_records
        if row.get("event_type") == "prediction_created" and row.get("entity_id")
    }
    closed = [
        row.get("payload") or {}
        for row in evidence_records
        if row.get("event_type") == "trade_closed"
    ]

    return_ready = [row for row in closed if row.get("exit_price") is not None and row.get("pnl") is not None]
    duration_ready = [
        row
        for row in closed
        if row.get("opened_at")
        and row.get("closed_at")
        and row.get("closed_at_quality") == "exact"
    ]
    execution_ready = [
        row
        for row in closed
        if row.get("mfe_pct") is not None
        and row.get("mae_pct") is not None
        and row.get("exit_reason")
    ]
    confidence_ready = [
        row
        for row in closed
        if str(row.get("signal_id", "") or "") in predictions
        and predictions[str(row.get("signal_id"))].get("confidence") is not None
    ]

    families = {
        "pattern_outcomes": _family(
            sample_count=len(labeled),
            minimum_required=SAMPLE_THRESHOLDS["emerging"],
            source="models/hermes/setups.json",
        ),
        "closed_trade_performance": _family(
            sample_count=len(return_ready),
            minimum_required=SAMPLE_THRESHOLDS["emerging"],
            source="storage/olympus/hermes_evidence.jsonl",
            missing_fields=("exit_price", "pnl") if closed and not return_ready else (),
        ),
        "confidence_calibration": _family(
            sample_count=len(confidence_ready),
            minimum_required=SAMPLE_THRESHOLDS["developing"],
            source="storage/olympus/hermes_evidence.jsonl",
            missing_fields=("prediction.confidence",) if closed and not confidence_ready else (),
        ),
        "trade_duration": _family(
            sample_count=len(duration_ready),
            minimum_required=SAMPLE_THRESHOLDS["emerging"],
            source="storage/olympus/hermes_evidence.jsonl",
            missing_fields=("opened_at", "closed_at") if closed and not duration_ready else (),
        ),
        "execution_quality": _family(
            sample_count=len(execution_ready),
            minimum_required=SAMPLE_THRESHOLDS["emerging"],
            source="storage/olympus/hermes_evidence.jsonl",
            missing_fields=("mfe_pct", "mae_pct", "exit_reason") if closed and not execution_ready else (),
        ),
        "runtime_health": {
            "status": "ready" if status.get("last_poll") else "missing_runtime_snapshot",
            "sample_count": 1 if status.get("last_poll") else 0,
            "minimum_required": 1,
            "sample_stage": "operational",
            "source": "live_bot/hermes_status.json",
            "missing_fields": [] if status.get("last_poll") else ["last_poll"],
        },
    }
    ready_count = sum(1 for row in families.values() if row.get("status") == "ready")
    latest = max(
        (str(row.get("recorded_at")) for row in evidence_records if row.get("recorded_at")),
        default=None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "last_evidence_at": latest,
        "sample_thresholds": SAMPLE_THRESHOLDS,
        "summary": {
            "ready_families": ready_count,
            "total_families": len(families),
            "readiness_pct": round((ready_count / max(1, len(families))) * 100.0, 2),
            "setup_records": len(setup_records),
            "labeled_setups": len(labeled),
            "prediction_records": len(predictions),
            "closed_trade_records": len(closed),
        },
        "families": families,
    }


def load_runtime_status(path: Path) -> dict[str, Any]:
    data = _read_json(path, {})
    return data if isinstance(data, dict) else {}
