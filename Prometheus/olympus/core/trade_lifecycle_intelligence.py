from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

STATUS_AWAITING = "Awaiting Historical Data"
TLI_VERSION = "tli-v1.0"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _safe_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _avg(values: list[float]) -> float | str:
    return round(mean(values), 4) if values else STATUS_AWAITING


def _med(values: list[float]) -> float | str:
    return round(median(values), 4) if values else STATUS_AWAITING


def _std(values: list[float]) -> float | str:
    return round(pstdev(values), 4) if len(values) > 1 else (0.0 if values else STATUS_AWAITING)


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _trade_hold_seconds(trade: dict[str, Any], close_time: datetime | None) -> float | None:
    opened = _safe_dt(trade.get("opened_at"))
    if opened is None:
        return None
    if close_time is None:
        close_time = _safe_dt(trade.get("closed_at"))
    if close_time is None:
        return None
    return max(0.0, (close_time - opened).total_seconds())


def _risk_efficiency(trade: dict[str, Any]) -> float:
    capture = _safe_float(trade.get("captured_return_pct")) or 0.0
    risk_util = _safe_float(trade.get("risk_utilization_pct")) or 0.0
    return round(max(0.0, min(100.0, (capture * 0.65) + ((100.0 - risk_util) * 0.35))), 4)


def _lifecycle_path_for_trade(trade: dict[str, Any]) -> list[str]:
    path = ["Signal Detected", "Candidate", "Validated", "Entered"]
    capture = _safe_float(trade.get("captured_return_pct")) or 0.0
    mfe = _safe_float(trade.get("mfe_pct")) or 0.0
    mae = _safe_float(trade.get("mae_pct")) or 0.0
    exit_reason = str(trade.get("exit_reason", "unknown") or "unknown")

    if mfe >= 0.20:
        path.append("Protected")
    if capture >= 40.0 and mfe >= 0.40:
        path.append("Scaling")
    if exit_reason in ("tp", "take_profit") or capture >= 70.0:
        path.append("Trailing")

    path.append("Exit Candidate")
    path.append("Closed")
    path.append("Learning Complete")

    if mae > mfe and "Protected" in path:
        # If adverse pressure dominated, mark weak protection by removing the optimistic stage.
        path = [p for p in path if p != "Scaling"]
    return path


def _build_state_machine(closed_trades: list[dict[str, Any]], lineage: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()

    lineage_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in lineage:
        payload = ev.get("payload", {})
        trade_id = str(payload.get("trade_id", "") or "")
        if not trade_id:
            continue
        lineage_by_trade[trade_id].append(ev)

    for trade in closed_trades[-300:]:
        trade_id = str(trade.get("trade_id", "unknown") or "unknown")
        inferred_path = _lifecycle_path_for_trade(trade)
        for st in inferred_path:
            state_counts[st] += 1

        for i in range(len(inferred_path) - 1):
            frm = inferred_path[i]
            to = inferred_path[i + 1]
            transitions.append(
                {
                    "trade_id": trade_id,
                    "from_state": frm,
                    "to_state": to,
                    "transition_reason": f"{frm} criteria satisfied",
                    "confidence": 0.72,
                }
            )

        for ev in lineage_by_trade.get(trade_id, []):
            if str(ev.get("event_type", "")) == "trade_closed":
                transitions.append(
                    {
                        "trade_id": trade_id,
                        "from_state": "Exit Candidate",
                        "to_state": "Closed",
                        "transition_reason": f"lineage_event:{ev.get('event_type')}",
                        "confidence": 0.9,
                    }
                )

    total_paths = max(1, len(closed_trades))
    return {
        "states": [
            "Signal Detected",
            "Candidate",
            "Validated",
            "Entered",
            "Protected",
            "Scaling",
            "Trailing",
            "Exit Candidate",
            "Closed",
            "Learning Complete",
        ],
        "state_counts": dict(state_counts),
        "transition_count": len(transitions),
        "transition_history": transitions[-500:],
        "state_observability_score": round(min(100.0, (len(state_counts) / 10.0) * 100.0), 2),
        "transition_confidence": round(min(1.0, len(transitions) / max(1.0, total_paths * 8.0)), 4),
    }


def _build_duration_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [r["hold_seconds"] for r in rows if r.get("hold_seconds") is not None]
    time_to_target = [r["hold_seconds"] for r in rows if str(r.get("exit_reason", "")) in ("tp", "take_profit") and r.get("hold_seconds") is not None]
    time_to_stop = [r["hold_seconds"] for r in rows if str(r.get("exit_reason", "")) in ("sl", "stop_loss") and r.get("hold_seconds") is not None]

    favorable_durations = [
        r["hold_seconds"]
        for r in rows
        if (r.get("mfe_pct") or 0.0) >= 0.35 and r.get("hold_seconds") is not None
    ]
    adverse_durations = [
        r["hold_seconds"]
        for r in rows
        if (r.get("mae_pct") or 0.0) >= 0.25 and r.get("hold_seconds") is not None
    ]

    expectancy_curve = []
    if durations:
        for quantile in (0.25, 0.5, 0.75, 0.9):
            bucket = sorted(durations)
            idx = int(max(0, min(len(bucket) - 1, math.floor(quantile * (len(bucket) - 1)))))
            horizon = bucket[idx]
            returns = [r.get("realized_return_pct", 0.0) for r in rows if (r.get("hold_seconds") or 0.0) <= horizon]
            expectancy_curve.append(
                {
                    "duration_horizon_sec": round(horizon, 2),
                    "expectancy": round(mean(returns), 4) if returns else STATUS_AWAITING,
                    "samples": len(returns),
                }
            )

    return {
        "average_hold_time_sec": _avg(durations),
        "median_hold_time_sec": _med(durations),
        "max_favorable_duration_sec": round(max(favorable_durations), 2) if favorable_durations else STATUS_AWAITING,
        "max_adverse_duration_sec": round(max(adverse_durations), 2) if adverse_durations else STATUS_AWAITING,
        "time_to_break_even_sec": _med([r.get("time_to_break_even_sec") for r in rows if r.get("time_to_break_even_sec") is not None]),
        "time_to_target_sec": _avg(time_to_target),
        "time_to_stop_sec": _avg(time_to_stop),
        "optimal_exit_window_sec": {
            "start": _med([x.get("duration_horizon_sec") for x in expectancy_curve if isinstance(x.get("duration_horizon_sec"), (int, float))]),
            "end": round(max(durations), 2) if durations else STATUS_AWAITING,
        },
        "duration_expectancy": round(mean([r.get("realized_return_pct", 0.0) for r in rows]), 4) if rows else STATUS_AWAITING,
        "duration_distribution": {
            "count": len(durations),
            "stdev": _std(durations),
            "min": round(min(durations), 2) if durations else STATUS_AWAITING,
            "max": round(max(durations), 2) if durations else STATUS_AWAITING,
        },
        "duration_expectancy_curve": expectancy_curve,
    }


def _build_management_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recs: list[str] = []
    capture_vals = [r.get("captured_return_pct", 0.0) for r in rows]
    risk_vals = [r.get("risk_utilization_pct", 0.0) for r in rows]
    leak_vals = [r.get("return_leakage_pct", 0.0) for r in rows]

    avg_capture = mean(capture_vals) if capture_vals else 0.0
    avg_leak = mean(leak_vals) if leak_vals else 0.0
    avg_risk = mean(risk_vals) if risk_vals else 0.0

    if avg_capture < 45.0:
        recs.append("Increase trailing discipline in high-MFE states; reward capture is below target.")
    if avg_leak > 40.0:
        recs.append("Evaluate partial exits by session/regime to reduce return leakage.")
    if avg_risk > 55.0:
        recs.append("Tighten protective transition thresholds under adverse volatility regimes.")
    if not recs:
        recs.append("Current lifecycle management is stable; continue evidence accumulation.")

    return {
        "evaluation_dimensions": [
            "current_profit",
            "mfe",
            "mae",
            "atr",
            "liquidity",
            "market_structure",
            "trend_strength",
            "volatility",
            "session",
            "ml_confidence",
            "pattern_similarity",
            "historical_expectancy",
        ],
        "advisory_actions": {
            "should_move_stop": avg_risk > 50.0,
            "should_scale_out": avg_capture > 65.0,
            "should_extend_tp": avg_capture > 70.0 and avg_leak > 20.0,
            "should_trail": avg_capture >= 40.0,
            "should_close": avg_risk > 65.0,
            "should_continue": avg_capture >= 35.0,
        },
        "management_score": round(max(0.0, min(100.0, (avg_capture * 0.55) + ((100.0 - avg_risk) * 0.45))), 2),
        "recommendations": recs,
    }


def _build_exit_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_reason[str(row.get("exit_reason", "unknown") or "unknown")].append(row)

    style_rows: list[dict[str, Any]] = []
    for reason, items in by_reason.items():
        returns = [i.get("realized_return_pct", 0.0) for i in items]
        capture = [i.get("captured_return_pct", 0.0) for i in items]
        style_rows.append(
            {
                "exit_style": reason,
                "samples": len(items),
                "expectancy": round(mean(returns), 4) if returns else STATUS_AWAITING,
                "reward_efficiency": round(mean(capture), 4) if capture else STATUS_AWAITING,
                "drawdown_pressure": round(mean([i.get("mae_pct", 0.0) for i in items]), 4) if items else STATUS_AWAITING,
            }
        )

    style_rows.sort(key=lambda x: (x.get("samples", 0), _safe_float(x.get("expectancy")) or -999.0), reverse=True)
    top_style = style_rows[0]["exit_style"] if style_rows else STATUS_AWAITING

    return {
        "exit_style_performance": style_rows,
        "best_expectancy_exit_style": top_style,
        "institutional_objective": "Maximize long-term expectancy, not isolated win-rate",
    }


def _build_reward_capture(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capture = [r.get("captured_return_pct", 0.0) for r in rows]
    mfe = [r.get("mfe_pct", 0.0) for r in rows]
    leakage = [r.get("return_leakage_pct", 0.0) for r in rows]
    r_mult = [r.get("r_multiple", 0.0) for r in rows]

    return {
        "average_mfe_pct": _avg(mfe),
        "average_captured_profit_pct": _avg(capture),
        "reward_efficiency": _avg(capture),
        "capture_ratio": _avg(capture),
        "return_leakage": _avg(leakage),
        "exit_timing_quality": round(max(0.0, min(100.0, (mean(capture) if capture else 0.0) - (mean(leakage) if leakage else 0.0) * 0.25)), 2),
        "profit_left_on_table": _avg(leakage),
        "average_r_multiple": _avg(r_mult),
        "missed_opportunity": _avg(leakage),
    }


def _build_pattern_lifecycle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pattern = str(row.get("pattern_name", "unknown") or "unknown")
        grouped[pattern].append(row)

    profiles = []
    for pattern, items in grouped.items():
        sequences = [" -> ".join(_lifecycle_path_for_trade(item)) for item in items]
        seq_counts = Counter(sequences)
        returns = [item.get("realized_return_pct", 0.0) for item in items]
        captures = [item.get("captured_return_pct", 0.0) for item in items]
        profiles.append(
            {
                "pattern": pattern,
                "samples": len(items),
                "best_sequence": seq_counts.most_common(1)[0][0] if seq_counts else STATUS_AWAITING,
                "expectancy": round(mean(returns), 4) if returns else STATUS_AWAITING,
                "capture_ratio": round(mean(captures), 4) if captures else STATUS_AWAITING,
                "sequence_diversity": len(seq_counts),
            }
        )

    profiles.sort(key=lambda r: (r.get("samples", 0), _safe_float(r.get("expectancy")) or -999.0), reverse=True)
    return {
        "pattern_lifecycle_profiles": profiles,
        "top_expectancy_patterns": profiles[:10],
    }


def _build_replay_cases(rows: list[dict[str, Any]], lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in lineage:
        payload = ev.get("payload", {})
        trade_id = str(payload.get("trade_id", "") or "")
        if trade_id:
            events_by_trade[trade_id].append(ev)

    replays: list[dict[str, Any]] = []
    for row in rows[-200:]:
        trade_id = str(row.get("trade_id", "unknown") or "unknown")
        timeline = [
            {
                "timestamp": ev.get("timestamp"),
                "event_type": ev.get("event_type"),
                "payload": ev.get("payload", {}),
            }
            for ev in events_by_trade.get(trade_id, [])
        ]
        replays.append(
            {
                "trade_id": trade_id,
                "signal_id": row.get("signal_id"),
                "entry": {
                    "price": row.get("entry_price"),
                    "opened_at": row.get("opened_at"),
                    "direction": row.get("direction"),
                    "session": row.get("session"),
                    "regime": row.get("regime"),
                    "pattern": row.get("pattern_name"),
                    "confidence": row.get("signal_confidence"),
                },
                "management": {
                    "mfe_pct": row.get("mfe_pct"),
                    "mae_pct": row.get("mae_pct"),
                    "captured_return_pct": row.get("captured_return_pct"),
                    "risk_utilization_pct": row.get("risk_utilization_pct"),
                },
                "exit": {
                    "exit_price": row.get("exit_price"),
                    "exit_reason": row.get("exit_reason"),
                    "realized_return_pct": row.get("realized_return_pct"),
                    "hold_seconds": row.get("hold_seconds"),
                },
                "decision_timeline": timeline,
                "similarity_signature": {
                    "pattern": row.get("pattern_name", "unknown"),
                    "session": row.get("session", "unknown"),
                    "regime": row.get("regime", "unknown"),
                    "trend": row.get("trend_state", "unknown"),
                    "volatility": row.get("volatility_state", "unknown"),
                },
            }
        )
    return replays


def _build_adaptive_position_management(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "position_size_multiplier": STATUS_AWAITING,
            "partial_close_pct": STATUS_AWAITING,
            "trailing_distance_scalar": STATUS_AWAITING,
            "break_even_timing_sec": STATUS_AWAITING,
            "max_exposure_scalar": STATUS_AWAITING,
            "risk_allocation_score": STATUS_AWAITING,
            "capital_efficiency": STATUS_AWAITING,
            "advisory_only": True,
        }

    conf = [r.get("signal_confidence", 0.0) for r in rows if r.get("signal_confidence") is not None]
    expectancy = [r.get("realized_return_pct", 0.0) for r in rows]
    drawdown = [r.get("mae_pct", 0.0) for r in rows]

    conf_avg = mean(conf) if conf else 0.0
    exp_avg = mean(expectancy) if expectancy else 0.0
    dd_avg = mean(drawdown) if drawdown else 0.0

    return {
        "position_size_multiplier": round(max(0.5, min(1.5, 0.9 + conf_avg * 0.7 + exp_avg * 0.05 - dd_avg * 0.01)), 4),
        "partial_close_pct": round(max(0.1, min(0.7, 0.25 + dd_avg * 0.005)), 4),
        "trailing_distance_scalar": round(max(0.5, min(2.0, 1.2 - conf_avg * 0.4 + dd_avg * 0.01)), 4),
        "break_even_timing_sec": round(max(30.0, min(7200.0, (_avg([r.get("hold_seconds", 0.0) for r in rows if r.get("hold_seconds") is not None]) if isinstance(_avg([r.get("hold_seconds", 0.0) for r in rows if r.get("hold_seconds") is not None]), (int, float)) else 600.0) * 0.35)), 2),
        "max_exposure_scalar": round(max(0.4, min(1.2, 1.0 - dd_avg * 0.01)), 4),
        "risk_allocation_score": round(max(0.0, min(100.0, 60.0 + exp_avg * 8.0 - dd_avg * 1.5)), 2),
        "capital_efficiency": round(max(0.0, min(100.0, 55.0 + exp_avg * 10.0 + conf_avg * 20.0 - dd_avg)), 2),
        "advisory_only": True,
    }


def _lifecycle_analytics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [r.get("realized_return_pct", 0.0) for r in rows]
    capture = [r.get("captured_return_pct", 0.0) for r in rows]
    risk_eff = [_risk_efficiency(r) for r in rows]
    hold = [r.get("hold_seconds", 0.0) for r in rows if r.get("hold_seconds") is not None]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    payoff = (mean(wins) / abs(mean(losses))) if wins and losses and abs(mean(losses)) > 1e-9 else None
    pf = (sum(wins) / abs(sum(losses))) if wins and losses and abs(sum(losses)) > 1e-9 else None
    expectancy = mean(returns) if returns else None
    max_dd = max([r.get("mae_pct", 0.0) for r in rows], default=0.0)

    return {
        "trade_duration_score": round(max(0.0, min(100.0, 100.0 - (_safe_float(_std(hold)) or 0.0) * 10.0)), 2) if hold else STATUS_AWAITING,
        "trade_efficiency": _avg(capture),
        "reward_efficiency": _avg(capture),
        "entry_efficiency": _avg([100.0 - (r.get("mae_pct", 0.0) * 10.0) for r in rows]) if rows else STATUS_AWAITING,
        "exit_efficiency": _avg(capture),
        "lifecycle_efficiency": _avg([(r.get("captured_return_pct", 0.0) * 0.6) + (_risk_efficiency(r) * 0.4) for r in rows]) if rows else STATUS_AWAITING,
        "decision_quality": _avg([_risk_efficiency(r) for r in rows]) if rows else STATUS_AWAITING,
        "risk_efficiency": _avg(risk_eff),
        "capital_efficiency": _avg([max(0.0, min(100.0, (r.get("captured_return_pct", 0.0) * 0.5) + (100.0 - r.get("risk_utilization_pct", 0.0)) * 0.5)) for r in rows]) if rows else STATUS_AWAITING,
        "average_r": _avg([r.get("r_multiple", 0.0) for r in rows]),
        "payoff_ratio": round(payoff, 4) if payoff is not None else STATUS_AWAITING,
        "recovery_factor": round((sum(returns) / max(1e-9, max_dd)), 4) if rows and max_dd > 0 else STATUS_AWAITING,
        "maximum_drawdown": round(max_dd, 4),
        "average_holding_time": _avg(hold),
        "exit_distribution": dict(Counter([str(r.get("exit_reason", "unknown") or "unknown") for r in rows])),
        "duration_distribution": {
            "mean": _avg(hold),
            "median": _med(hold),
            "stdev": _std(hold),
        },
        "return_distribution": {
            "mean": round(expectancy, 4) if expectancy is not None else STATUS_AWAITING,
            "median": _med(returns),
            "stdev": _std(returns),
        },
        "expectancy": round(expectancy, 4) if expectancy is not None else STATUS_AWAITING,
        "profit_factor": round(pf, 4) if pf is not None else STATUS_AWAITING,
    }


def _continuous_learning(rows: list[dict[str, Any]], replay_cases: list[dict[str, Any]]) -> dict[str, Any]:
    improved = [r for r in rows if (r.get("realized_return_pct", 0.0) or 0.0) > 0 and (r.get("captured_return_pct", 0.0) or 0.0) >= 60.0]
    degraded = [r for r in rows if (r.get("realized_return_pct", 0.0) or 0.0) < 0 and (r.get("mae_pct", 0.0) or 0.0) >= 0.4]

    return {
        "learning_events": len(rows),
        "replay_cases_generated": len(replay_cases),
        "duration_intelligence_updates": len(rows),
        "exit_intelligence_updates": len(rows),
        "reward_capture_updates": len(rows),
        "pattern_lifecycle_updates": len(rows),
        "risk_intelligence_updates": len(rows),
        "capital_allocation_updates": len(rows),
        "trade_management_updates": len(rows),
        "improved_cases": len(improved),
        "degraded_cases": len(degraded),
        "append_only_learning": True,
    }


def _prepare_trade_rows(closed_trades: list[dict[str, Any]], lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    close_time_by_trade: dict[str, datetime] = {}
    for ev in lineage:
        if str(ev.get("event_type", "")) != "trade_closed":
            continue
        payload = ev.get("payload", {})
        trade_id = str(payload.get("trade_id", "") or "")
        ts = _safe_dt(ev.get("timestamp"))
        if trade_id and ts is not None:
            close_time_by_trade[trade_id] = ts

    rows: list[dict[str, Any]] = []
    for trade in closed_trades:
        row = dict(trade)
        trade_id = str(row.get("trade_id", "") or "")
        hold_seconds = _trade_hold_seconds(row, close_time_by_trade.get(trade_id))
        realized_return = _safe_float(row.get("realized_return_pct"))
        if realized_return is None:
            entry = _safe_float(row.get("entry_price")) or 0.0
            exitp = _safe_float(row.get("exit_price"))
            direction = str(row.get("direction", "long") or "long").lower()
            if entry > 0 and exitp is not None:
                raw = ((exitp - entry) / entry) * 100.0
                realized_return = raw if direction == "long" else -raw
            else:
                realized_return = 0.0

        captured = _safe_float(row.get("captured_return_pct")) or 0.0
        leakage = max(0.0, 100.0 - captured)
        risk_util = _safe_float(row.get("risk_utilization_pct")) or 0.0
        pred_dist = _safe_float(row.get("predicted_distance_pts")) or 0.0
        real_dist = _safe_float(row.get("realized_distance_pts")) or 0.0
        r_mult = real_dist / pred_dist if pred_dist > 1e-9 else 0.0

        row["hold_seconds"] = hold_seconds
        row["realized_return_pct"] = round(float(realized_return), 4)
        row["captured_return_pct"] = round(captured, 4)
        row["return_leakage_pct"] = round(leakage, 4)
        row["risk_utilization_pct"] = round(risk_util, 4)
        row["r_multiple"] = round(r_mult, 4)
        row["time_to_break_even_sec"] = hold_seconds * 0.4 if hold_seconds is not None and (captured >= 35.0 or row.get("exit_reason") in ("tp", "take_profit")) else None
        rows.append(row)
    return rows


def build_trade_lifecycle_intelligence(
    root_dir: Path,
    *,
    status: dict[str, Any],
    open_trades: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    start = datetime.now(timezone.utc)
    lineage = _jsonl(root_dir / "storage" / "olympus" / "event_lineage.jsonl")
    rows = _prepare_trade_rows(closed_trades, lineage)

    duration_intelligence = _build_duration_intelligence(rows)
    state_machine = _build_state_machine(closed_trades, lineage)
    management = _build_management_intelligence(rows)
    exit_intelligence = _build_exit_intelligence(rows)
    reward_capture = _build_reward_capture(rows)
    replay_cases = _build_replay_cases(rows, lineage)
    pattern_lifecycle = _build_pattern_lifecycle(rows)
    adaptive_position = _build_adaptive_position_management(rows)
    lifecycle_analytics = _lifecycle_analytics(rows)
    continuous_learning = _continuous_learning(rows, replay_cases)

    ended = datetime.now(timezone.utc)
    runtime_ms = round((ended - start).total_seconds() * 1000.0, 3)

    history_rows = _jsonl(root_dir / "storage" / "olympus" / "trade_lifecycle_intelligence_history.jsonl")
    baseline = [
        _safe_float(r.get("performance_profile", {}).get("runtime_ms"))
        for r in history_rows[-120:]
        if isinstance(r, dict)
    ]
    baseline_vals = [x for x in baseline if x is not None]
    baseline_ms = round(mean(baseline_vals), 3) if baseline_vals else None

    payload = {
        "version": TLI_VERSION,
        "generated_at": ended.isoformat(),
        "mode": "observational_advisory",
        "backward_compatible": True,
        "preserves_execution_behavior": True,
        "system": str(status.get("bot", "Hermes") or "Hermes"),
        "asset": str(status.get("asset", "unknown") or "unknown"),
        "timeframe": str(status.get("timeframe", "unknown") or "unknown"),
        "sample_sizes": {
            "open_trades": len(open_trades),
            "closed_trades": len(closed_trades),
            "replay_cases": len(replay_cases),
            "lineage_events": len(lineage),
        },
        "modules": {
            "trade_duration_intelligence": duration_intelligence,
            "trade_state_machine": state_machine,
            "trade_management_intelligence": management,
            "exit_intelligence": exit_intelligence,
            "reward_capture_intelligence": reward_capture,
            "trade_replay_intelligence": {
                "replay_cases": replay_cases[-120:],
                "similarity_ready": True,
            },
            "pattern_lifecycle_intelligence": pattern_lifecycle,
            "adaptive_position_management": adaptive_position,
            "trade_lifecycle_analytics": lifecycle_analytics,
            "continuous_learning": continuous_learning,
        },
        "summary": {
            "expectancy": lifecycle_analytics.get("expectancy", STATUS_AWAITING),
            "reward_efficiency": reward_capture.get("reward_efficiency", STATUS_AWAITING),
            "lifecycle_efficiency": lifecycle_analytics.get("lifecycle_efficiency", STATUS_AWAITING),
            "drawdown_pressure": lifecycle_analytics.get("maximum_drawdown", STATUS_AWAITING),
            "average_hold_time_sec": duration_intelligence.get("average_hold_time_sec", STATUS_AWAITING),
            "state_observability_score": state_machine.get("state_observability_score", STATUS_AWAITING),
        },
        "performance_profile": {
            "runtime_ms": runtime_ms,
            "baseline_runtime_ms": baseline_ms if baseline_ms is not None else STATUS_AWAITING,
            "delta_pct_vs_baseline": round(((runtime_ms - baseline_ms) / baseline_ms) * 100.0, 4) if baseline_ms and baseline_ms > 0 else STATUS_AWAITING,
            "rows_processed": len(rows),
            "computation_mode": "single_pass",
        },
        "research_hypotheses": [
            "Lifecycle-aware exits can increase reward capture while lowering leakage.",
            "Pattern-sequence quality predicts expectancy better than entry quality alone.",
            "Duration-normalized management may improve recovery factor in volatile regimes.",
        ],
        "recommendations": management.get("recommendations", []),
    }
    return payload


def write_trade_lifecycle_intelligence_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime_path = storage / "trade_lifecycle_intelligence_runtime.json"
    history_path = storage / "trade_lifecycle_intelligence_history.jsonl"
    replay_path = storage / "trade_lifecycle_replay_library.jsonl"

    runtime_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    _append_jsonl(history_path, payload)

    if not replay_path.exists():
        replay_path.write_text("", encoding="utf-8")

    existing_trade_ids = {
        str(row.get("trade_id", ""))
        for row in _jsonl(replay_path)
        if isinstance(row, dict)
    }
    replay_cases = payload.get("modules", {}).get("trade_replay_intelligence", {}).get("replay_cases", [])
    for case in replay_cases:
        trade_id = str((case or {}).get("trade_id", "") or "")
        if not trade_id or trade_id in existing_trade_ids:
            continue
        _append_jsonl(replay_path, case)
        existing_trade_ids.add(trade_id)

    return {
        "runtime": str(runtime_path),
        "history": str(history_path),
        "replay": str(replay_path),
    }
