"""Regression coverage for durable Hermes historical evidence."""
from __future__ import annotations

from pathlib import Path

from ml.pattern_learner import PatternLearner, SetupRecord
from olympus.core.historical_evidence import (
    HistoricalEvidenceLedger,
    build_evidence_readiness,
    stable_evidence_id,
)


def _trade(trade_id: str, signal_id: str, status: str = "open") -> dict:
    return {
        "trade_id": trade_id,
        "signal_id": signal_id,
        "direction": "long",
        "entry_price": 3300.0,
        "lots": 0.01,
        "sl_price": 3298.0,
        "tp_price": 3304.0,
        "opened_at": "2026-07-21T10:00:00+00:00",
        "status": status,
        "exit_price": 3304.0 if status != "open" else None,
        "exit_reason": "tp" if status != "open" else "",
        "pnl": 4.0 if status != "open" else 0.0,
        "mfe_pct": 0.15,
        "mae_pct": 0.03,
    }


def test_stable_evidence_id_is_deterministic_and_versioned() -> None:
    first = stable_evidence_id("bootstrap", "XAUUSDm", "M5", "2026-07-21T10:00:00Z", "v1")
    second = stable_evidence_id("bootstrap", "XAUUSDm", "M5", "2026-07-21T10:00:00Z", "v1")
    changed = stable_evidence_id("bootstrap", "XAUUSDm", "M5", "2026-07-21T10:00:00Z", "v2")

    assert first == second
    assert first != changed


def test_ledger_is_idempotent_and_reconstructs_trade_state(tmp_path: Path) -> None:
    ledger = HistoricalEvidenceLedger(tmp_path)
    signal = {
        "signal_id": "signal-1",
        "timestamp": "2026-07-21T10:00:00+00:00",
        "confidence": 0.74,
    }
    opened = _trade("trade-1", "signal-1")
    closed = _trade("trade-1", "signal-1", status="won")

    assert ledger.append_prediction(signal)
    assert not ledger.append_prediction(signal)
    assert ledger.append_trade_opened(opened)
    assert ledger.append_trade_closed(closed, closed_at="2026-07-21T10:15:00+00:00")

    reloaded = HistoricalEvidenceLedger(tmp_path)
    open_rows, closed_rows = reloaded.trade_state()

    assert open_rows == []
    assert len(closed_rows) == 1
    assert closed_rows[0]["trade_id"] == "trade-1"
    assert closed_rows[0]["closed_at"] == "2026-07-21T10:15:00+00:00"
    assert reloaded.predictions_by_signal()["signal-1"]["confidence"] == 0.74


def test_previous_runtime_status_is_backfilled_once(tmp_path: Path) -> None:
    status = {
        "last_poll": "2026-07-21T10:20:00+00:00",
        "signals": [{"signal_id": "signal-2", "timestamp": "2026-07-21T10:00:00+00:00", "confidence": 0.68}],
        "open_trades": [_trade("trade-open", "signal-2")],
        "closed_trades": [_trade("trade-closed", "signal-2", status="lost")],
    }
    ledger = HistoricalEvidenceLedger(tmp_path)

    first = ledger.ingest_runtime_status(status)
    second = ledger.ingest_runtime_status(status)

    assert first == {"predictions": 1, "opened_trades": 1, "closed_trades": 1}
    assert second == {"predictions": 0, "opened_trades": 0, "closed_trades": 0}
    open_rows, closed_rows = HistoricalEvidenceLedger(tmp_path).trade_state()
    assert {row["trade_id"] for row in open_rows} == {"trade-open"}
    assert {row["trade_id"] for row in closed_rows} == {"trade-closed"}


def test_readiness_explains_insufficient_samples(tmp_path: Path) -> None:
    ledger = HistoricalEvidenceLedger(tmp_path)
    ledger.append_prediction({"signal_id": "signal-3", "timestamp": "2026-07-21T10:00:00+00:00", "confidence": 0.71})
    ledger.append_trade_opened(_trade("trade-3", "signal-3"))
    ledger.append_trade_closed(
        _trade("trade-3", "signal-3", status="won"),
        closed_at="2026-07-21T10:05:00+00:00",
    )
    setups = [{"setup_id": "signal-3", "outcome": 1}]

    readiness = build_evidence_readiness(
        setups,
        ledger.records(),
        status={"last_poll": "2026-07-21T10:06:00+00:00"},
    )

    assert readiness["families"]["pattern_outcomes"]["status"] == "insufficient_samples"
    assert readiness["families"]["confidence_calibration"]["sample_count"] == 1
    assert readiness["families"]["runtime_health"]["status"] == "ready"


def test_pattern_learner_rejects_duplicate_stable_setup_ids(tmp_path: Path) -> None:
    learner = PatternLearner(model_dir=str(tmp_path), min_samples_train=50)
    record = SetupRecord(
        setup_id="stable-setup",
        asset="XAUUSDm",
        timeframe="M5",
        timestamp="2026-07-21T10:00:00+00:00",
    )

    assert learner.add_setup(record)
    assert not learner.add_setup(record)
    assert len(learner.records) == 1
    assert (tmp_path / "setups.json").exists()
    assert not (tmp_path / "setups.json.tmp").exists()
