from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPLAY_VERSION = "replay-v1.0"
STATUS_AWAITING = "Awaiting Historical Data"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _replay_trade(trade: dict[str, Any]) -> dict[str, Any]:
    trade_id = str(trade.get("trade_id") or "unknown")
    baseline = float(_safe_float(trade.get("realized_return_pct")) or 0.0)
    mfe = float(_safe_float(trade.get("mfe_pct")) or 0.0)
    mae = float(_safe_float(trade.get("mae_pct")) or 0.0)
    capture = float(_safe_float(trade.get("captured_return_pct")) or 0.0)

    tp_extended = baseline + max(0.0, mfe * 0.18)
    be_earlier = baseline + (0.08 if baseline < 0 else 0.0)
    be_later = baseline + max(0.0, mfe * 0.04)
    trail_later = baseline + max(0.0, mfe * 0.11)
    scale_in = baseline * 1.08
    scale_out = baseline + max(0.0, (capture - 45.0) * 0.004)
    alt_exit = baseline + max(0.0, (100.0 - capture) * 0.0022)
    alt_stop = baseline + (0.06 if mae > 0.25 else 0.0)

    scenario_rows = [
        {"scenario": "different_stop_placement", "counterfactual_return_pct": round(alt_stop, 4)},
        {"scenario": "different_tp", "counterfactual_return_pct": round(tp_extended, 4)},
        {"scenario": "earlier_break_even", "counterfactual_return_pct": round(be_earlier, 4)},
        {"scenario": "later_break_even", "counterfactual_return_pct": round(be_later, 4)},
        {"scenario": "scaling", "counterfactual_return_pct": round(scale_in, 4)},
        {"scenario": "position_sizing", "counterfactual_return_pct": round(scale_out, 4)},
        {"scenario": "trailing", "counterfactual_return_pct": round(trail_later, 4)},
        {"scenario": "alternative_exit", "counterfactual_return_pct": round(alt_exit, 4)},
    ]

    best = max(scenario_rows, key=lambda x: float(x.get("counterfactual_return_pct", -1e9)))
    uplift = float(best.get("counterfactual_return_pct", 0.0)) - baseline
    confidence = _bounded(0.35 + capture * 0.003 + abs(mfe - mae) * 0.15, 0.0, 1.0)

    return {
        "trade_id": trade_id,
        "baseline_return_pct": round(baseline, 4),
        "counterfactual_scenarios": scenario_rows,
        "best_scenario": best,
        "best_uplift_pct": round(uplift, 4),
        "replay_confidence": round(confidence, 4),
        "research_only": True,
    }


def build_decision_replay_counterfactual_intelligence(
    *,
    closed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    replay_rows = [_replay_trade(trade) for trade in closed_trades]

    positive = [row for row in replay_rows if float(row.get("best_uplift_pct", 0.0)) > 0]
    replay_accuracy = (
        round(len([r for r in replay_rows if float(r.get("replay_confidence", 0.0)) >= 0.5]) / len(replay_rows), 4)
        if replay_rows
        else STATUS_AWAITING
    )

    recommendations = []
    if positive:
        avg_uplift = sum(float(x.get("best_uplift_pct", 0.0)) for x in positive) / len(positive)
        text = "Counterfactual replay indicates uplift potential; validate staged exit and trailing regimes in Zeus."
        rec_id = f"replay-rec-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
        recommendations.append(
            {
                "recommendation_id": rec_id,
                "source_system": "hermes",
                "validation_domain": "recommendation",
                "timestamp": generated_at,
                "recommendation_type": "counterfactual_replay_optimization",
                "recommendation": text,
                "expected_improvement": "Improve reward capture via validated scenario adoption",
                "evidence": {
                    "sample_size": len(replay_rows),
                    "confidence": round(_bounded(0.32 + len(positive) / max(1.0, len(replay_rows)), 0.0, 1.0), 4),
                    "evidence_score": round(_bounded(0.32 + len(positive) / max(1.0, len(replay_rows)), 0.0, 1.0), 4),
                    "supporting_metric": "average_counterfactual_uplift",
                    "supporting_value": round(avg_uplift, 4),
                },
                "lifecycle": "candidate",
                "operator_approved": False,
                "requires_zeus_validation": True,
                "governance_required": True,
            }
        )

    return {
        "version": REPLAY_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "replay_rows": replay_rows,
        "summary": {
            "completed_trade_replays": len(replay_rows),
            "counterfactual_replays": len(replay_rows),
            "positive_uplift_replays": len(positive),
            "average_counterfactual_uplift": round(
                (sum(float(x.get("best_uplift_pct", 0.0)) for x in replay_rows) / len(replay_rows)), 4
            )
            if replay_rows
            else STATUS_AWAITING,
            "replay_accuracy": replay_accuracy,
        },
        "zeus_candidates": recommendations,
    }


def write_decision_replay_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime_path = storage / "decision_replay_counterfactual.json"
    history_path = storage / "decision_replay_counterfactual_history.jsonl"

    runtime_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {
        "decision_replay": str(runtime_path),
        "decision_replay_history": str(history_path),
    }
