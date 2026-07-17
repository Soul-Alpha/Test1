import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.core.trade_lifecycle_intelligence import (
    build_trade_lifecycle_intelligence,
    write_trade_lifecycle_intelligence_artifacts,
)


def _sample_closed_trades() -> list[dict]:
    return [
        {
            "trade_id": "tli-001",
            "signal_id": "sig-001",
            "direction": "long",
            "entry_price": 2400.0,
            "exit_price": 2404.0,
            "opened_at": "2026-07-13T08:00:00+00:00",
            "exit_reason": "tp",
            "predicted_distance_pts": 100.0,
            "realized_distance_pts": 120.0,
            "mfe_pct": 0.35,
            "mae_pct": 0.12,
            "captured_return_pct": 82.0,
            "risk_utilization_pct": 28.0,
            "session": "London",
            "regime": "Trend Expansion",
            "trend_state": "bullish",
            "volatility_state": "Moderate",
            "pattern_name": "Liquidity Sweep Long",
            "signal_confidence": 0.78,
        },
        {
            "trade_id": "tli-002",
            "signal_id": "sig-002",
            "direction": "short",
            "entry_price": 2410.0,
            "exit_price": 2412.5,
            "opened_at": "2026-07-13T09:00:00+00:00",
            "exit_reason": "sl",
            "predicted_distance_pts": 90.0,
            "realized_distance_pts": 70.0,
            "mfe_pct": 0.22,
            "mae_pct": 0.34,
            "captured_return_pct": 21.0,
            "risk_utilization_pct": 64.0,
            "session": "New York",
            "regime": "Volatility Expansion",
            "trend_state": "sideways",
            "volatility_state": "High",
            "pattern_name": "CHOCH Reversal",
            "signal_confidence": 0.61,
        },
    ]


def test_tli_payload_contract_contains_required_modules(tmp_path: Path):
    root = tmp_path
    storage = root / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    # Minimal lineage so duration and close-time inference can work.
    lineage = [
        {
            "event_type": "trade_closed",
            "timestamp": "2026-07-13T08:10:00+00:00",
            "payload": {"trade_id": "tli-001"},
        },
        {
            "event_type": "trade_closed",
            "timestamp": "2026-07-13T09:20:00+00:00",
            "payload": {"trade_id": "tli-002"},
        },
    ]
    (storage / "event_lineage.jsonl").write_text(
        "\n".join(json.dumps(r) for r in lineage) + "\n",
        encoding="utf-8",
    )

    payload = build_trade_lifecycle_intelligence(
        root,
        status={"bot": "Hermes", "asset": "XAUUSDm", "timeframe": "M5"},
        open_trades=[],
        closed_trades=_sample_closed_trades(),
    )

    assert payload["version"].startswith("tli-")
    assert payload["backward_compatible"] is True
    assert payload["preserves_execution_behavior"] is True

    modules = payload["modules"]
    for key in [
        "trade_duration_intelligence",
        "trade_state_machine",
        "trade_management_intelligence",
        "exit_intelligence",
        "reward_capture_intelligence",
        "trade_replay_intelligence",
        "pattern_lifecycle_intelligence",
        "adaptive_position_management",
        "trade_lifecycle_analytics",
        "continuous_learning",
    ]:
        assert key in modules


def test_tli_artifact_writer_persists_runtime_history_and_replay(tmp_path: Path):
    root = tmp_path
    payload = {
        "version": "tli-v1.0",
        "modules": {
            "trade_replay_intelligence": {
                "replay_cases": [
                    {"trade_id": "tli-001", "entry": {"price": 1.0}},
                    {"trade_id": "tli-002", "entry": {"price": 2.0}},
                ]
            }
        },
        "performance_profile": {"runtime_ms": 1.23},
    }

    paths = write_trade_lifecycle_intelligence_artifacts(root, payload)

    runtime = Path(paths["runtime"])
    history = Path(paths["history"])
    replay = Path(paths["replay"])

    assert runtime.exists()
    assert history.exists()
    assert replay.exists()

    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert runtime_payload["version"] == "tli-v1.0"

    history_lines = [x for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]
    replay_lines = [x for x in replay.read_text(encoding="utf-8").splitlines() if x.strip()]

    assert len(history_lines) == 1
    assert len(replay_lines) == 2

    # Re-writing the same payload should not duplicate replay trade IDs.
    write_trade_lifecycle_intelligence_artifacts(root, payload)
    replay_lines_after = [x for x in replay.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(replay_lines_after) == 2
