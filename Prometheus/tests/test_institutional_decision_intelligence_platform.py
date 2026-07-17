import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from olympus.core.institutional_decision_intelligence_platform import (
    build_institutional_decision_intelligence_platform,
    write_idip_artifacts,
)


def _closed_trade_samples() -> list[dict]:
    return [
        {
            "trade_id": "idip-001",
            "signal_id": "sig-001",
            "direction": "long",
            "entry_price": 2400.0,
            "exit_price": 2406.0,
            "opened_at": "2026-07-13T08:00:00+00:00",
            "closed_at": "2026-07-13T08:30:00+00:00",
            "exit_reason": "tp",
            "mfe_pct": 0.42,
            "mae_pct": 0.12,
            "captured_return_pct": 78.0,
            "risk_utilization_pct": 27.0,
            "session": "London",
            "regime": "Trend Expansion",
            "pattern_name": "Liquidity Sweep Long",
            "signal_confidence": 0.81,
            "trend_state": "bullish",
            "volatility_state": "Moderate",
        },
        {
            "trade_id": "idip-002",
            "signal_id": "sig-002",
            "direction": "short",
            "entry_price": 2410.0,
            "exit_price": 2412.0,
            "opened_at": "2026-07-13T09:00:00+00:00",
            "closed_at": "2026-07-13T09:20:00+00:00",
            "exit_reason": "micro_time_exit",
            "mfe_pct": 0.20,
            "mae_pct": 0.30,
            "captured_return_pct": 30.0,
            "risk_utilization_pct": 61.0,
            "session": "New York",
            "regime": "Volatility Expansion",
            "pattern_name": "CHOCH Reversal",
            "signal_confidence": 0.58,
            "trend_state": "sideways",
            "volatility_state": "High",
        },
    ]


def test_idip_build_contains_core_engines(tmp_path: Path):
    root = tmp_path
    storage = root / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "event_lineage.jsonl").write_text("", encoding="utf-8")

    payload = build_institutional_decision_intelligence_platform(
        root,
        status={"source_system": "hermes", "asset": "XAUUSDm", "timeframe": "M5"},
        open_trades=[],
        closed_trades=_closed_trade_samples(),
    )

    assert payload["meta"]["additive_only"] is True
    assert payload["meta"]["auto_execution_modification"] is False
    assert payload["summary"]["sample_size"] == 2
    assert payload["summary"]["maturity"] in ("Emerging", "Developing", "Validated", "Institutional")

    engines = payload["engines"]
    for key in [
        "trade_lifecycle_intelligence",
        "exit_intelligence",
        "duration_intelligence",
        "reward_capture_intelligence",
        "position_management_intelligence",
        "institutional_risk_intelligence",
        "portfolio_intelligence",
        "decision_attribution_intelligence",
        "counterfactual_intelligence",
        "pattern_lifecycle_intelligence",
        "institutional_knowledge_intelligence",
    ]:
        assert key in engines


def test_idip_artifacts_write_runtime_history_queue_and_knowledge(tmp_path: Path):
    root = tmp_path
    payload = {
        "meta": {"version": "idip-v1.0"},
        "summary": {"sample_size": 1},
        "performance": {"runtime_ms": 10.1},
        "zeus_research_recommendations": [
            {
                "recommendation_id": "idip-rec-001",
                "recommendation": "Test recommendation",
            }
        ],
        "engines": {
            "institutional_knowledge_intelligence": {
                "institutional_lessons": [
                    {
                        "knowledge_id": "IK-idip-001",
                        "trade_id": "idip-001",
                    }
                ]
            }
        },
    }

    paths = write_idip_artifacts(root, payload)
    runtime = Path(paths["runtime"])
    history = Path(paths["history"])
    queue = Path(paths["recommendation_queue"])
    knowledge = Path(paths["knowledge_base"])

    assert runtime.exists()
    assert history.exists()
    assert queue.exists()
    assert knowledge.exists()

    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert runtime_payload["meta"]["version"] == "idip-v1.0"

    assert len([x for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]) == 1
    assert len([x for x in queue.read_text(encoding="utf-8").splitlines() if x.strip()]) == 1
    assert len([x for x in knowledge.read_text(encoding="utf-8").splitlines() if x.strip()]) == 1

    write_idip_artifacts(root, payload)
    assert len([x for x in queue.read_text(encoding="utf-8").splitlines() if x.strip()]) == 1
    assert len([x for x in knowledge.read_text(encoding="utf-8").splitlines() if x.strip()]) == 1
