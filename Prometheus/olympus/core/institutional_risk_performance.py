from __future__ import annotations

import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Mapping

_CONTRACT_VERSION = "v1.1"
_DEFAULT_STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "olympus"
_RUNTIME_STATUS_FILE = "institutional_risk_performance_runtime.json"
_RUNTIME_HISTORY_FILE = "institutional_risk_performance_runtime.jsonl"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)

    ordered = sorted(values)
    idx = int(round((percentile / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _derive_risk_state(payload: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    if _safe_bool(payload.get("trading_halted")) or _safe_bool(runtime.get("daily_loss_halted")):
        return "Halted"
    if _safe_bool(runtime.get("daily_profit_protect")):
        return "Defensive"
    if not _safe_bool(payload.get("mt5_connected", False)):
        return "Caution"
    if _safe_float(runtime.get("poll_duration_p95_ms"), 0.0) > 2_000.0:
        return "Caution"
    return "Normal"


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return sqrt(max(0.0, var))


def _sample_confidence_level(sample_size: int) -> str:
    if sample_size >= 200:
        return "Institutional"
    if sample_size >= 100:
        return "High"
    if sample_size >= 40:
        return "Medium"
    return "Low"


def _policy_recommendation(risk_state: str, sample_confidence: str) -> tuple[str, str]:
    if risk_state == "Halted":
        return "No New Entries", "Institutional circuit breaker active; preserve capital until reset criteria are met."
    if risk_state == "Defensive":
        return "Reduced Risk Mode", "Maintain reduced risk until runtime performance and confidence normalize."
    if risk_state == "Caution":
        return "Tighten Filters", "Raise quality thresholds and reduce discretionary exposure while telemetry stabilizes."
    if sample_confidence in {"Low", "Medium"}:
        return "Observe Only", "Telemetry sample confidence is still building; defer policy enforcement changes."
    return "Normal Operations", "Runtime governance and confidence are stable."


def build_institutional_risk_performance_runtime(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime_telemetry") if isinstance(payload.get("runtime_telemetry"), Mapping) else {}
    runtime = dict(runtime)

    recent_polls = runtime.get("recent_poll_durations_ms")
    recent_vals = [
        _safe_float(v) for v in (recent_polls if isinstance(recent_polls, list) else []) if _safe_float(v, -1.0) >= 0.0
    ]
    sample_size = len(recent_vals)
    duration_mean = (sum(recent_vals) / sample_size) if sample_size else 0.0
    duration_sd = _stddev(recent_vals)
    duration_sem = (duration_sd / sqrt(sample_size)) if sample_size > 1 else 0.0
    ci_margin = 1.96 * duration_sem
    ci_low = max(0.0, duration_mean - ci_margin)
    ci_high = max(ci_low, duration_mean + ci_margin)
    poll_error_count = int(runtime.get("poll_error_count") or 0)
    poll_error_rate_pct = (poll_error_count / sample_size * 100.0) if sample_size > 0 else 0.0
    sample_confidence = _sample_confidence_level(sample_size)

    risk_state = _derive_risk_state(payload, runtime)

    recommendation, rationale = _policy_recommendation(risk_state, sample_confidence)

    profile: dict[str, Any] = {
        "contract_version": _CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observational_only": True,
        "runtime_risk_governance": {
            "risk_state": risk_state,
            "daily_loss_halted": _safe_bool(runtime.get("daily_loss_halted")),
            "daily_profit_protect": _safe_bool(runtime.get("daily_profit_protect")),
            "current_risk_pct": round(_safe_float(payload.get("risk_pct"), 0.0), 4),
        },
        "runtime_performance": {
            "poll_duration_ms": round(_safe_float(runtime.get("poll_duration_ms"), 2), 2),
            "poll_duration_p50_ms": round(_safe_float(runtime.get("poll_duration_p50_ms"), _percentile(recent_vals, 50)), 2),
            "poll_duration_p95_ms": round(_safe_float(runtime.get("poll_duration_p95_ms"), _percentile(recent_vals, 95)), 2),
            "total_trades": int(payload.get("total_trades") or 0),
            "open_count": int(payload.get("open_count") or 0),
            "total_unrealised": round(_safe_float(payload.get("total_unrealised"), 0.0), 2),
            "daily_trade_pnl": round(_safe_float(runtime.get("daily_trade_pnl"), 0.0), 2),
            "daily_loss_pct": round(_safe_float(runtime.get("daily_loss_pct"), 0.0), 4),
            "poll_error_count": poll_error_count,
            "poll_error_rate_pct": round(poll_error_rate_pct, 4),
            "sample_size": sample_size,
        },
        "runtime_statistics_confidence": {
            "sample_confidence_level": sample_confidence,
            "sample_confidence_score": round(min(100.0, (sample_size / 200.0) * 100.0), 2),
            "poll_duration_ci95_ms": {
                "low": round(ci_low, 2),
                "high": round(ci_high, 2),
                "mean": round(duration_mean, 2),
                "margin": round(ci_margin, 2),
            },
            "poll_duration_ci95_label": f"{round(ci_low, 2)}-{round(ci_high, 2)} ms",
            "error_reliability": "Stable" if poll_error_rate_pct <= 1.0 else "Watch" if poll_error_rate_pct <= 3.0 else "Unstable",
            "error_rate_pct": round(poll_error_rate_pct, 4),
        },
        "policy_advisory": {
            "recommendation": recommendation,
            "rationale": rationale,
            "enforcement_active": _safe_bool(payload.get("policy_gate_enabled")),
        },
        "operational_health": {
            "mt5_connected": _safe_bool(payload.get("mt5_connected")),
            "mt5_available": _safe_bool(payload.get("mt5_available")),
            "trading_halted": _safe_bool(payload.get("trading_halted")),
        },
    }
    return profile


def write_institutional_risk_performance_runtime(
    profile: Mapping[str, Any],
    *,
    storage_dir: Path | None = None,
) -> dict[str, str]:
    target_dir = storage_dir or _DEFAULT_STORAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    status_path = target_dir / _RUNTIME_STATUS_FILE
    history_path = target_dir / _RUNTIME_HISTORY_FILE

    status_path.write_text(json.dumps(dict(profile), indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(profile), default=str) + "\n")

    return {
        "status_path": str(status_path),
        "history_path": str(history_path),
    }
