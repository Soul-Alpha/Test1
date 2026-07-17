from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COVERAGE_VERSION = "knowledge-coverage-v1.0"


def _uniq(values: list[Any]) -> int:
    return len({str(v) for v in values if v is not None and str(v).strip()})


def _coverage(count: int, target: int) -> float:
    if target <= 0:
        return 0.0
    return round(min(1.0, max(0.0, count / float(target))), 4)


def build_knowledge_coverage_intelligence(*, status: dict[str, Any], idip_payload: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()

    closed = status.get("closed_trades", []) if isinstance(status, dict) else []
    engines = idip_payload.get("engines", {}) if isinstance(idip_payload, dict) else {}

    sessions = _uniq([x.get("session") for x in closed if isinstance(x, dict)])
    regimes = _uniq([x.get("regime") for x in closed if isinstance(x, dict)])
    patterns = _uniq([x.get("pattern_name") for x in closed if isinstance(x, dict)])
    exits = _uniq([x.get("exit_reason") for x in closed if isinstance(x, dict)])

    liquidity_conditions = _uniq([x.get("volatility_state") for x in closed if isinstance(x, dict)])
    durations = len([x for x in closed if isinstance(x, dict) and x.get("hold_seconds") is not None])
    replay_coverage = int((engines.get("decision_replay_counterfactual_intelligence", {}) or {}).get("summary", {}).get("completed_trade_replays", 0) or 0)

    rows = [
        {"gap_id": "sessions", "dimension": "Sessions", "coverage_pct": _coverage(sessions, 8), "severity": round(1.0 - _coverage(sessions, 8), 4)},
        {"gap_id": "regimes", "dimension": "Market Regimes", "coverage_pct": _coverage(regimes, 8), "severity": round(1.0 - _coverage(regimes, 8), 4)},
        {"gap_id": "patterns", "dimension": "Patterns", "coverage_pct": _coverage(patterns, 25), "severity": round(1.0 - _coverage(patterns, 25), 4)},
        {"gap_id": "liquidity", "dimension": "Liquidity Conditions", "coverage_pct": _coverage(liquidity_conditions, 6), "severity": round(1.0 - _coverage(liquidity_conditions, 6), 4)},
        {"gap_id": "trade_duration", "dimension": "Trade Duration", "coverage_pct": _coverage(durations, max(1, len(closed))), "severity": round(1.0 - _coverage(durations, max(1, len(closed))), 4)},
        {"gap_id": "exit_styles", "dimension": "Exit Styles", "coverage_pct": _coverage(exits, 9), "severity": round(1.0 - _coverage(exits, 9), 4)},
        {"gap_id": "replay", "dimension": "Counterfactual Replay", "coverage_pct": _coverage(replay_coverage, max(1, len(closed))), "severity": round(1.0 - _coverage(replay_coverage, max(1, len(closed))), 4)},
    ]

    rows.sort(key=lambda x: x.get("severity", 0.0), reverse=True)
    for r in rows:
        r["missing_samples"] = max(1, int((1.0 - float(r.get("coverage_pct", 0.0))) * max(10, len(closed))))
        r["confidence"] = round(max(0.25, min(0.9, float(r.get("coverage_pct", 0.0)) * 0.8 + 0.2)), 4)
        r["research_recommendation"] = f"Increase {r.get('dimension')} coverage via targeted Zeus-validated research"

    return {
        "version": COVERAGE_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "coverage_rows": rows,
        "summary": {
            "average_coverage": round(sum(float(x.get("coverage_pct", 0.0)) for x in rows) / max(1, len(rows)), 4),
            "weak_areas": [x.get("dimension") for x in rows[:3]],
        },
    }


def write_knowledge_coverage_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "knowledge_coverage_runtime.json"
    history = storage / "knowledge_coverage_history.jsonl"

    runtime.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"knowledge_coverage_runtime": str(runtime), "knowledge_coverage_history": str(history)}
