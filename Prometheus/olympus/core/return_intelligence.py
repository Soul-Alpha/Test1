from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any


STATUS_AWAITING = "Awaiting Historical Data"
STATUS_PENDING = "Pending Initialization"


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if is_dataclass(row):
        return asdict(row)
    if hasattr(row, "_asdict"):
        try:
            return dict(row._asdict())
        except Exception:
            pass
    try:
        return dict(vars(row))
    except Exception:
        return {"value": row}


def _get(row: Any, key: str, default: Any = None) -> Any:
    data = _row_dict(row)
    return data.get(key, default)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _direction_multiplier(direction: Any) -> int:
    d = str(direction or "").lower()
    if d in ("long", "buy", "bullish"):
        return 1
    if d in ("short", "sell", "bearish"):
        return -1
    return 1


def _trade_return_pct(entry_price: float, exit_price: float, direction: Any) -> float:
    if entry_price <= 0:
        return 0.0
    raw = ((exit_price - entry_price) / entry_price) * 100.0
    return float(raw if _direction_multiplier(direction) > 0 else -raw)


def _mfe_mae_pct(trade: dict[str, Any]) -> tuple[float, float]:
    entry = _safe_float(trade.get("entry_price")) or 0.0
    if entry <= 0:
        return 0.0, 0.0

    direction = str(trade.get("direction", "long") or "long").lower()
    high = _safe_float(trade.get("mfe_price_high"))
    low = _safe_float(trade.get("mfe_price_low"))
    if high is None:
        high = _safe_float(trade.get("exit_price")) or entry
    if low is None:
        low = _safe_float(trade.get("exit_price")) or entry

    if direction in ("long", "buy", "bullish"):
        mfe_pct = max(0.0, ((high - entry) / entry) * 100.0)
        mae_pct = max(0.0, ((entry - low) / entry) * 100.0)
    else:
        mfe_pct = max(0.0, ((entry - low) / entry) * 100.0)
        mae_pct = max(0.0, ((high - entry) / entry) * 100.0)
    return round(mfe_pct, 4), round(mae_pct, 4)


def _skew(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    mu = mean(values)
    sd = pstdev(values)
    if sd <= 0:
        return 0.0
    m3 = sum((v - mu) ** 3 for v in values) / len(values)
    return round(m3 / (sd ** 3), 4)


def _trend(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    denom = sum((i - x_mean) ** 2 for i in range(n))
    if denom <= 0:
        return 0.0
    numer = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    return round(numer / denom, 6)


def _stability(values: list[float]) -> float | None:
    if not values:
        return None
    mu = abs(mean(values))
    sd = pstdev(values) if len(values) > 1 else 0.0
    scale = sd / max(1e-9, mu if mu > 1e-9 else 1.0)
    return round(max(0.0, 100.0 - min(100.0, scale * 100.0)), 2)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "average": STATUS_AWAITING,
            "median": STATUS_AWAITING,
            "variance": STATUS_AWAITING,
            "stability": STATUS_AWAITING,
            "trend": STATUS_AWAITING,
            "skew": STATUS_AWAITING,
        }
    variance = pstdev(values) ** 2 if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "average": round(mean(values), 4),
        "median": round(median(values), 4),
        "variance": round(variance, 4),
        "stability": _stability(values),
        "trend": _trend(values),
        "skew": _skew(values),
    }


def _group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown") or "unknown")].append(row)

    output: list[dict[str, Any]] = []
    for group_name, items in grouped.items():
        returns = [_safe_float(item.get("realized_return_pct")) or 0.0 for item in items]
        captures = [_safe_float(item.get("captured_return_pct")) or 0.0 for item in items]
        risk_utils = [_safe_float(item.get("risk_utilization_pct")) or 0.0 for item in items]
        qualities = [str(item.get("exit_quality", "Unknown") or "Unknown") for item in items]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        output.append(
            {
                key: group_name,
                "sample_size": len(items),
                "average_return": round(mean(returns), 4) if returns else STATUS_AWAITING,
                "average_win": round(mean(wins), 4) if wins else STATUS_AWAITING,
                "average_loss": round(mean(losses), 4) if losses else STATUS_AWAITING,
                "median_return": round(median(returns), 4) if returns else STATUS_AWAITING,
                "return_variance": round((pstdev(returns) ** 2) if len(returns) > 1 else 0.0, 4) if returns else STATUS_AWAITING,
                "return_stability": _stability(returns) if returns else STATUS_AWAITING,
                "historical_capture_ratio": round(mean(captures), 4) if captures else STATUS_AWAITING,
                "historical_risk_ratio": round(mean(risk_utils), 4) if risk_utils else STATUS_AWAITING,
                "exit_quality": max(set(qualities), key=qualities.count) if qualities else "Unknown",
            }
        )
    output.sort(key=lambda r: (r.get("sample_size", 0), _safe_float(r.get("average_return")) or -1e9), reverse=True)
    return output


def _trade_exit_quality(trade: dict[str, Any]) -> str:
    realized = _safe_float(trade.get("realized_return_pct")) or 0.0
    captured = _safe_float(trade.get("captured_return_pct")) or 0.0
    mfe = _safe_float(trade.get("mfe_pct")) or 0.0
    mae = _safe_float(trade.get("mae_pct")) or 0.0
    reason = str(trade.get("exit_reason", "") or "").lower()

    if mfe <= 0.0:
        return "Unknown"
    if reason in ("tp", "take_profit") and captured >= 85.0:
        return "Excellent Exit"
    if captured >= 70.0 and realized >= 0.0:
        return "Good Exit"
    if captured >= 45.0 and realized >= 0.0:
        return "Acceptable Exit"
    if reason in ("micro_time_exit", "time_exit") and realized > 0.0 and captured < 45.0:
        return "Premature Exit"
    if reason in ("micro_time_exit", "time_exit") and realized >= 0.0:
        return "Late Exit"
    if reason in ("sl", "stop_loss") and mfe > mae and mfe > 0.25:
        return "Reversal Exit"
    if reason in ("structure_exit", "structure"):
        return "Structure Exit"
    if reason in ("volatility_exit", "volatility"):
        return "Volatility Exit"
    if reason in ("liquidity_exit", "liquidity"):
        return "Liquidity Exit"
    if mfe > 0.5 and realized < mfe * 0.35:
        return "Missed Expansion"
    return "Unknown"


def _pattern_family(trade: dict[str, Any]) -> str:
    family = str(trade.get("pattern_family", "") or "").strip()
    if family:
        return family
    cluster = str(trade.get("pattern_cluster", "") or "").strip()
    if cluster:
        return cluster
    if trade.get("stop_hunt"):
        return "Liquidity Sweep"
    if trade.get("regime"):
        regime = str(trade.get("regime")).lower()
        if "trend" in regime:
            return "Trend Continuation"
        if "reversion" in regime:
            return "Mean Reversion"
        if "compression" in regime:
            return "Compression"
    return "Unknown"


def _pattern_cluster(trade: dict[str, Any]) -> str:
    cluster = str(trade.get("pattern_cluster", "") or "").strip()
    if cluster:
        return cluster
    direction = str(trade.get("direction", "unknown") or "unknown").lower()
    family = _pattern_family(trade)
    session = str(trade.get("session", "unknown") or "unknown")
    regime = str(trade.get("regime", "unknown") or "unknown")
    return f"{family} | {direction} | {session} | {regime}"


def _pattern_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        profiles[str(row.get("pattern_name", "unknown") or "unknown")].append(row)

    output: list[dict[str, Any]] = []
    for pattern_name, items in profiles.items():
        returns = [_safe_float(item.get("realized_return_pct")) or 0.0 for item in items]
        mfe_vals = [_safe_float(item.get("mfe_pct")) or 0.0 for item in items]
        mae_vals = [_safe_float(item.get("mae_pct")) or 0.0 for item in items]
        capture_vals = [_safe_float(item.get("captured_return_pct")) or 0.0 for item in items]
        risk_vals = [_safe_float(item.get("risk_utilization_pct")) or 0.0 for item in items]
        durations = [_safe_float(item.get("hold_seconds")) or 0.0 for item in items]
        confidences = [_safe_float(item.get("signal_confidence")) or 0.0 for item in items if _safe_float(item.get("signal_confidence")) is not None]
        exit_scores = []
        for item in items:
            quality = str(item.get("exit_quality", "Unknown") or "Unknown")
            if quality == "Excellent Exit":
                exit_scores.append(100.0)
            elif quality == "Good Exit":
                exit_scores.append(85.0)
            elif quality == "Acceptable Exit":
                exit_scores.append(70.0)
            elif quality == "Premature Exit":
                exit_scores.append(35.0)
            elif quality == "Late Exit":
                exit_scores.append(40.0)
            elif quality == "Missed Expansion":
                exit_scores.append(20.0)
            elif quality == "Reversal Exit":
                exit_scores.append(50.0)
            else:
                exit_scores.append(45.0)

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        avg_win = round(mean(wins), 4) if wins else STATUS_AWAITING
        avg_loss = round(mean(losses), 4) if losses else STATUS_AWAITING
        profit_factor = round(sum(wins) / abs(sum(losses)), 4) if wins and losses and abs(sum(losses)) > 1e-9 else STATUS_AWAITING
        payoff_ratio = round((mean(wins) / abs(mean(losses))), 4) if wins and losses and abs(mean(losses)) > 1e-9 else STATUS_AWAITING
        expectancy = round(mean(returns), 4) if returns else STATUS_AWAITING
        stability = _stability(returns) if returns else STATUS_AWAITING
        output.append(
            {
                "pattern_name": pattern_name,
                "sample_size": len(items),
                "average_return": expectancy,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "average_mfe": round(mean(mfe_vals), 4) if mfe_vals else STATUS_AWAITING,
                "average_mae": round(mean(mae_vals), 4) if mae_vals else STATUS_AWAITING,
                "average_duration": round(mean(durations), 4) if durations else STATUS_AWAITING,
                "average_expansion": round(mean(mfe_vals), 4) if mfe_vals else STATUS_AWAITING,
                "average_pullback": round(mean(mae_vals), 4) if mae_vals else STATUS_AWAITING,
                "historical_capture_ratio": round(mean(capture_vals), 4) if capture_vals else STATUS_AWAITING,
                "historical_risk_ratio": round(mean(risk_vals), 4) if risk_vals else STATUS_AWAITING,
                "historical_return_stability": stability,
                "historical_expectancy": expectancy,
                "historical_payoff_ratio": payoff_ratio,
                "historical_profit_factor": profit_factor,
                "historical_exit_quality": round(mean(exit_scores), 2) if exit_scores else STATUS_AWAITING,
                "historical_confidence": round(mean(confidences), 4) if confidences else STATUS_AWAITING,
                "historical_confidence_reliability": round(100.0 - min(100.0, (pstdev(confidences) if len(confidences) > 1 else 0.0) * 100.0), 2) if confidences else STATUS_AWAITING,
                "historical_optimal_tp": round(median(mfe_vals), 4) if mfe_vals else STATUS_AWAITING,
                "historical_optimal_break_even": round(median(mae_vals), 4) if mae_vals else STATUS_AWAITING,
                "historical_optimal_trailing_point": round(mean(mfe_vals) * 0.55, 4) if mfe_vals else STATUS_AWAITING,
                "knowledge_confidence": round(min(100.0, (len(items) / 30.0) * 100.0), 2),
                "evidence_level": round(min(100.0, (len(items) / 50.0) * 100.0), 2),
                "pattern_maturity": _pattern_maturity(len(items)),
                "pattern_family": _pattern_family(items[0]),
                "pattern_cluster": _pattern_cluster(items[0]),
            }
        )
    output.sort(key=lambda row: (row.get("sample_size", 0), _safe_float(row.get("average_return")) or -1e9), reverse=True)
    return output


def _pattern_maturity(samples: int) -> str:
    if samples < 10:
        return "Candidate"
    if samples < 30:
        return "Emerging"
    if samples < 75:
        return "Developing"
    if samples < 150:
        return "Validated"
    return "Elite"


def _period_series(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_trades = sorted(trades, key=lambda row: _safe_dt(row.get("opened_at")) or datetime.now(timezone.utc))
    running: list[float] = []
    for idx, row in enumerate(sorted_trades, start=1):
        running.append(_safe_float(row.get("realized_return_pct")) or 0.0)
        rows.append(
            {
                "index": idx,
                "timestamp": row.get("opened_at") or row.get("closed_at") or row.get("timestamp") or "",
                "return_pct": round(running[-1], 4),
                "rolling_average_return": round(mean(running[-10:]), 4),
                "cumulative_return": round(sum(running), 4),
                "capture_ratio": row.get("captured_return_pct", STATUS_AWAITING),
                "exit_quality": row.get("exit_quality", "Unknown"),
            }
        )
    return rows


def _research_report(summary: dict[str, Any], patterns: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return_by_session = groups.get("session", [])
    return_by_regime = groups.get("regime", [])
    return_by_pattern = groups.get("pattern_name", [])
    return_by_family = groups.get("pattern_family", [])

    improving_patterns = [p for p in return_by_pattern if (_safe_float(p.get("average_return")) or -1e9) > 0 and int(p.get("sample_size", 0) or 0) >= 5][:8]
    deteriorating_patterns = [p for p in return_by_pattern if (_safe_float(p.get("average_return")) or 0.0) < 0][:8]
    best_sessions = sorted(return_by_session, key=lambda r: (_safe_float(r.get("average_return")) or -1e9, int(r.get("sample_size", 0) or 0)), reverse=True)[:5]
    worst_sessions = sorted(return_by_session, key=lambda r: (_safe_float(r.get("average_return")) or 1e9, -int(r.get("sample_size", 0) or 0)))[:5]
    best_regimes = sorted(return_by_regime, key=lambda r: (_safe_float(r.get("average_return")) or -1e9, int(r.get("sample_size", 0) or 0)), reverse=True)[:5]
    worst_regimes = sorted(return_by_regime, key=lambda r: (_safe_float(r.get("average_return")) or 1e9, -int(r.get("sample_size", 0) or 0)))[:5]

    avg_capture = _safe_float(summary.get("average_captured_return_pct")) or 0.0
    avg_risk = _safe_float(summary.get("average_risk_utilization_pct")) or 0.0
    avg_return = _safe_float(summary.get("average_return_pct")) or 0.0
    hypotheses = []
    proposals = []
    if summary.get("sample_size", 0) >= 30 and avg_capture < 60.0:
        hypotheses.append("Hermes currently leaves a material share of favorable excursion unrealized.")
        proposals.append(
            {
                "proposal": "Investigate TP expansion",
                "reason": f"Average capture ratio is {avg_capture:.2f}% across statistically relevant samples.",
                "evidence": f"sample_size={summary.get('sample_size', 0)} | avg_capture={avg_capture:.2f}%",
                "scope": "Zeus controlled backtest only",
                "status": "Research-only",
            }
        )
    if summary.get("sample_size", 0) >= 30 and avg_risk > 45.0:
        hypotheses.append("Hermes is allowing too much adverse excursion before exit in some regimes.")
        proposals.append(
            {
                "proposal": "Investigate earlier break-even",
                "reason": f"Risk utilization averages {avg_risk:.2f}% and may be suppressing realized return.",
                "evidence": f"sample_size={summary.get('sample_size', 0)} | avg_risk_utilization={avg_risk:.2f}%",
                "scope": "Zeus controlled backtest only",
                "status": "Research-only",
            }
        )
    if best_sessions:
        top_session = best_sessions[0]
        hypotheses.append(f"{top_session.get('session')} currently concentrates the best realized return profile.")
        proposals.append(
            {
                "proposal": "Investigate session-specific exit logic",
                "reason": f"Best session: {top_session.get('session')} with average return {top_session.get('average_return')}",
                "evidence": f"best_session={top_session.get('session')} | sample_size={top_session.get('sample_size', 0)}",
                "scope": "Zeus controlled backtest only",
                "status": "Research-only",
            }
        )
    if best_regimes:
        top_regime = best_regimes[0]
        hypotheses.append(f"{top_regime.get('regime')} regimes appear to deliver the strongest return efficiency.")
        proposals.append(
            {
                "proposal": "Investigate volatility-adjusted exit logic",
                "reason": f"Best regime: {top_regime.get('regime')} with average return {top_regime.get('average_return')}",
                "evidence": f"best_regime={top_regime.get('regime')} | sample_size={top_regime.get('sample_size', 0)}",
                "scope": "Zeus controlled backtest only",
                "status": "Research-only",
            }
        )

    return {
        "current_observations": {
            "average_return_pct": summary.get("average_return_pct", STATUS_AWAITING),
            "average_capture_ratio_pct": summary.get("average_captured_return_pct", STATUS_AWAITING),
            "average_risk_utilization_pct": summary.get("average_risk_utilization_pct", STATUS_AWAITING),
            "best_sessions": best_sessions,
            "worst_sessions": worst_sessions,
            "best_regimes": best_regimes,
            "worst_regimes": worst_regimes,
            "best_patterns": improving_patterns[:5],
            "deteriorating_patterns": deteriorating_patterns[:5],
            "best_pattern_families": sorted(return_by_family, key=lambda r: (_safe_float(r.get("average_return")) or -1e9), reverse=True)[:5],
        },
        "emerging_trends": [
            f"Return trend slope: {summary.get('return_trend', STATUS_AWAITING)}",
            f"Return stability: {summary.get('return_stability', STATUS_AWAITING)}",
            f"Return skew: {summary.get('return_skew', STATUS_AWAITING)}",
        ],
        "improving_patterns": improving_patterns,
        "deteriorating_patterns": deteriorating_patterns,
        "return_efficiency_trends": {
            "capture_ratio": summary.get("average_captured_return_pct", STATUS_AWAITING),
            "risk_utilization": summary.get("average_risk_utilization_pct", STATUS_AWAITING),
            "execution_efficiency": summary.get("average_execution_efficiency_pct", STATUS_AWAITING),
        },
        "capture_efficiency_trends": {
            "average_capture_ratio_pct": summary.get("average_captured_return_pct", STATUS_AWAITING),
            "historical_capture_ratio": summary.get("historical_capture_ratio", STATUS_AWAITING),
        },
        "risk_efficiency_trends": {
            "average_risk_utilization_pct": summary.get("average_risk_utilization_pct", STATUS_AWAITING),
            "historical_risk_ratio": summary.get("historical_risk_ratio", STATUS_AWAITING),
        },
        "exit_efficiency_trends": {
            "average_exit_quality_score": summary.get("average_exit_quality_score", STATUS_AWAITING),
            "premature_exit_rate": summary.get("premature_exit_rate", STATUS_AWAITING),
            "missed_expansion_rate": summary.get("missed_expansion_rate", STATUS_AWAITING),
        },
        "new_hypotheses": hypotheses,
        "zeus_research_proposals": proposals,
        "academy_assessment": {
            "implementation": 100.0,
            "evidence_level": summary.get("sample_size", 0),
            "knowledge_confidence": summary.get("knowledge_confidence_score", STATUS_AWAITING),
            "mastery": summary.get("return_maturity_score", STATUS_AWAITING),
            "current_grade": summary.get("return_grade", STATUS_AWAITING),
            "reliability": summary.get("return_stability", STATUS_AWAITING),
            "sample_size": summary.get("sample_size", 0),
            "next_milestone": summary.get("next_milestone", STATUS_PENDING),
            "estimated_additional_samples": summary.get("estimated_additional_samples", 0),
        },
        "evidence_requirements": {
            "minimum_samples_for_high_grade": 30,
            "high_grade_requirement": "Stable capture efficiency and positive return expectancy over a statistically relevant sample.",
            "current_sample_size": summary.get("sample_size", 0),
        },
        "zeus_validation_recommendations": proposals,
    }


def build_return_intelligence(
    closed_trades: list[Any],
    *,
    status: dict[str, Any] | None = None,
    feature_version: str | None = None,
    model_version: str | None = None,
    report_interval: int = 10,
) -> dict[str, Any]:
    status = status or {}
    signal_rows = status.get("signals", []) or []
    skipped_rows = status.get("skipped_signals", []) or []
    signal_map: dict[str, dict[str, Any]] = {}
    for row in signal_rows + skipped_rows:
        sid = str(row.get("signal_id", "") or "")
        if sid:
            signal_map[sid] = row
    last_signal = status.get("last_signal") or {}
    if last_signal.get("signal_id"):
        signal_map[str(last_signal.get("signal_id"))] = last_signal

    trades: list[dict[str, Any]] = []
    for raw in closed_trades:
        trade = _row_dict(raw)
        entry = _safe_float(trade.get("entry_price")) or 0.0
        exit_price = _safe_float(trade.get("exit_price"))
        if exit_price is None:
            exit_price = entry
        direction = str(trade.get("direction", "long") or "long").lower()
        realized_return = trade.get("realized_return_pct")
        if _safe_float(realized_return) is None:
            realized_return = _trade_return_pct(entry, exit_price, direction)
        mfe_pct, mae_pct = _mfe_mae_pct(trade)
        potential_return_pct = max(0.0, mfe_pct)
        potential_loss_pct = max(0.0, mae_pct)
        captured_ratio = 0.0
        if potential_return_pct > 0:
            captured_ratio = max(0.0, min(100.0, (max(0.0, _safe_float(realized_return) or 0.0) / potential_return_pct) * 100.0))
        lost_opportunity = 0.0 if potential_return_pct <= 0 else max(0.0, 100.0 - captured_ratio)
        risk_utilization = 0.0 if (potential_return_pct + potential_loss_pct) <= 0 else max(0.0, min(100.0, (potential_loss_pct / (potential_return_pct + potential_loss_pct)) * 100.0))
        return_efficiency = 0.0 if (potential_return_pct + potential_loss_pct) <= 0 else max(-100.0, min(100.0, (float(realized_return) / (potential_return_pct + potential_loss_pct)) * 100.0))
        loss_efficiency = 0.0
        if (potential_loss_pct > 0) and (float(realized_return) < 0):
            loss_efficiency = max(0.0, min(100.0, (abs(float(realized_return)) / potential_loss_pct) * 100.0))
        opportunity_efficiency = captured_ratio
        execution_efficiency = max(0.0, min(100.0, (captured_ratio * 0.55) + ((100.0 - risk_utilization) * 0.45)))

        sid = str(trade.get("signal_id", "") or "")
        signal = signal_map.get(sid, {})
        ts = trade.get("opened_at") or trade.get("timestamp") or signal.get("timestamp")
        session = str(trade.get("session") or _derive_session(ts) or "unknown")
        regime = str(trade.get("regime") or _derive_regime(trade) or "unknown")
        trend = str(trade.get("trend_state") or _derive_trend(trade) or "unknown")
        volatility = str(trade.get("volatility_state") or _derive_volatility(trade) or "unknown")
        pattern_name = str(trade.get("pattern_name") or signal.get("pattern_name") or signal.get("skip_reason") or sid or "unknown")
        pattern_family = str(trade.get("pattern_family") or _pattern_family(trade) or "unknown")
        pattern_cluster = str(trade.get("pattern_cluster") or _pattern_cluster(trade) or "unknown")
        hold_seconds = _safe_float(trade.get("hold_seconds"))
        if hold_seconds is None:
            opened = _safe_dt(trade.get("opened_at"))
            closed = _safe_dt(trade.get("closed_at") or trade.get("closed_at_utc") or trade.get("updated_at"))
            if opened and closed:
                hold_seconds = max(0.0, (closed - opened).total_seconds())

        row = {
            **trade,
            "signal_id": sid,
            "realized_return_pct": round(float(realized_return), 4),
            "mfe_pct": round(mfe_pct, 4),
            "mae_pct": round(mae_pct, 4),
            "potential_return_pct": round(potential_return_pct, 4),
            "potential_loss_pct": round(potential_loss_pct, 4),
            "captured_return_pct": round(captured_ratio, 4),
            "lost_opportunity_pct": round(lost_opportunity, 4),
            "risk_utilization_pct": round(risk_utilization, 4),
            "return_efficiency_pct": round(return_efficiency, 4),
            "loss_efficiency_pct": round(loss_efficiency, 4),
            "opportunity_efficiency_pct": round(opportunity_efficiency, 4),
            "execution_efficiency_pct": round(execution_efficiency, 4),
            "historical_capture_ratio": round(captured_ratio, 4),
            "historical_risk_ratio": round(risk_utilization, 4),
            "exit_quality": _trade_exit_quality(
                {
                    **trade,
                    "realized_return_pct": float(realized_return),
                    "captured_return_pct": captured_ratio,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                }
            ),
            "session": session,
            "regime": regime,
            "trend_state": trend,
            "volatility_state": volatility,
            "pattern_name": pattern_name,
            "pattern_family": pattern_family,
            "pattern_cluster": pattern_cluster,
            "timeframe": str(trade.get("timeframe") or status.get("timeframe") or "unknown"),
            "model_version": str(trade.get("model_version") or model_version or trade.get("model_version_used") or signal.get("model_version") or status.get("ml", {}).get("model_version", "0")),
            "feature_version": str(trade.get("feature_version") or feature_version or status.get("feature_version", "v1")),
            "signal_confidence": _safe_float(trade.get("signal_confidence")) or _safe_float(signal.get("confidence")) or 0.0,
            "expected_distance_pts": _safe_float(trade.get("predicted_distance_pts")) or _safe_float(signal.get("expected_distance_pts")) or 0.0,
            "exit_reason": str(trade.get("exit_reason", "Unknown") or "Unknown"),
            "hold_seconds": round(hold_seconds, 2) if hold_seconds is not None else STATUS_AWAITING,
        }
        trades.append(row)

    returns = [_safe_float(row.get("realized_return_pct")) or 0.0 for row in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    captures = [_safe_float(row.get("captured_return_pct")) or 0.0 for row in trades]
    risk_utils = [_safe_float(row.get("risk_utilization_pct")) or 0.0 for row in trades]
    efficiencies = [_safe_float(row.get("execution_efficiency_pct")) or 0.0 for row in trades]
    durations = [_safe_float(row.get("hold_seconds")) or 0.0 for row in trades if _safe_float(row.get("hold_seconds")) is not None]
    mfe_vals = [_safe_float(row.get("mfe_pct")) or 0.0 for row in trades]
    mae_vals = [_safe_float(row.get("mae_pct")) or 0.0 for row in trades]
    exit_scores = []
    exit_counts: dict[str, int] = defaultdict(int)
    for row in trades:
        q = str(row.get("exit_quality", "Unknown") or "Unknown")
        exit_counts[q] += 1
        if q == "Excellent Exit":
            exit_scores.append(100.0)
        elif q == "Good Exit":
            exit_scores.append(85.0)
        elif q == "Acceptable Exit":
            exit_scores.append(70.0)
        elif q == "Premature Exit":
            exit_scores.append(35.0)
        elif q == "Late Exit":
            exit_scores.append(40.0)
        elif q == "Missed Expansion":
            exit_scores.append(20.0)
        elif q == "Reversal Exit":
            exit_scores.append(50.0)
        elif q == "Structure Exit":
            exit_scores.append(60.0)
        elif q == "Volatility Exit":
            exit_scores.append(65.0)
        elif q == "Liquidity Exit":
            exit_scores.append(60.0)
        else:
            exit_scores.append(45.0)

    grouped: dict[str, list[dict[str, Any]]] = {
        "pattern_name": _group_rows(trades, "pattern_name"),
        "pattern_family": _group_rows(trades, "pattern_family"),
        "pattern_cluster": _group_rows(trades, "pattern_cluster"),
        "session": _group_rows(trades, "session"),
        "regime": _group_rows(trades, "regime"),
        "volatility": _group_rows(trades, "volatility_state"),
        "trend": _group_rows(trades, "trend_state"),
        "timeframe": _group_rows(trades, "timeframe"),
        "model_version": _group_rows(trades, "model_version"),
        "feature_version": _group_rows(trades, "feature_version"),
    }

    summary = {
        "sample_size": len(trades),
        "average_return_pct": round(mean(returns), 4) if returns else STATUS_AWAITING,
        "average_winning_return_pct": round(mean(wins), 4) if wins else STATUS_AWAITING,
        "average_losing_return_pct": round(mean(losses), 4) if losses else STATUS_AWAITING,
        "return_distribution": _summary(returns),
        "return_variance": round((pstdev(returns) ** 2) if len(returns) > 1 else 0.0, 4) if returns else STATUS_AWAITING,
        "return_stability": _stability(returns) if returns else STATUS_AWAITING,
        "return_trend": _trend(returns) if returns else STATUS_AWAITING,
        "return_skew": _skew(returns) if returns else STATUS_AWAITING,
        "median_return_pct": round(median(returns), 4) if returns else STATUS_AWAITING,
        "rolling_average_return_pct": round(mean(returns[-10:]), 4) if returns else STATUS_AWAITING,
        "historical_return_evolution": _period_series(trades),
        "return_consistency": round((sum(1 for r in returns if r > 0) / len(returns)) * 100.0, 2) if returns else STATUS_AWAITING,
        "average_mfe_pct": round(mean(mfe_vals), 4) if mfe_vals else STATUS_AWAITING,
        "average_mae_pct": round(mean(mae_vals), 4) if mae_vals else STATUS_AWAITING,
        "average_captured_return_pct": round(mean(captures), 4) if captures else STATUS_AWAITING,
        "average_lost_opportunity_pct": round(mean([100.0 - c for c in captures]) if captures else 0.0, 4) if captures else STATUS_AWAITING,
        "average_risk_utilization_pct": round(mean(risk_utils), 4) if risk_utils else STATUS_AWAITING,
        "average_return_efficiency_pct": round(mean([_safe_float(row.get("return_efficiency_pct")) or 0.0 for row in trades]), 4) if trades else STATUS_AWAITING,
        "average_loss_efficiency_pct": round(mean([_safe_float(row.get("loss_efficiency_pct")) or 0.0 for row in trades]), 4) if trades else STATUS_AWAITING,
        "average_opportunity_efficiency_pct": round(mean([_safe_float(row.get("opportunity_efficiency_pct")) or 0.0 for row in trades]), 4) if trades else STATUS_AWAITING,
        "average_execution_efficiency_pct": round(mean(efficiencies), 4) if efficiencies else STATUS_AWAITING,
        "historical_capture_ratio": round(mean(captures), 4) if captures else STATUS_AWAITING,
        "historical_risk_ratio": round(mean(risk_utils), 4) if risk_utils else STATUS_AWAITING,
        "exit_intelligence": {
            "excellent_exit_count": exit_counts.get("Excellent Exit", 0),
            "good_exit_count": exit_counts.get("Good Exit", 0),
            "acceptable_exit_count": exit_counts.get("Acceptable Exit", 0),
            "premature_exit_count": exit_counts.get("Premature Exit", 0),
            "late_exit_count": exit_counts.get("Late Exit", 0),
            "missed_expansion_count": exit_counts.get("Missed Expansion", 0),
            "reversal_exit_count": exit_counts.get("Reversal Exit", 0),
            "structure_exit_count": exit_counts.get("Structure Exit", 0),
            "volatility_exit_count": exit_counts.get("Volatility Exit", 0),
            "liquidity_exit_count": exit_counts.get("Liquidity Exit", 0),
            "unknown_exit_count": exit_counts.get("Unknown", 0),
            "average_exit_quality_score": round(mean(exit_scores), 2) if exit_scores else STATUS_AWAITING,
            "premature_exit_rate": round((exit_counts.get("Premature Exit", 0) / len(trades)) * 100.0, 2) if trades else STATUS_AWAITING,
            "missed_expansion_rate": round((exit_counts.get("Missed Expansion", 0) / len(trades)) * 100.0, 2) if trades else STATUS_AWAITING,
        },
        "pattern_profiles": _pattern_profile(trades),
        "trade_metrics": trades,
    }
    summary["knowledge_confidence_score"] = round(min(100.0, (summary["sample_size"] / max(1.0, float(report_interval) * 3.0)) * 100.0), 2) if summary["sample_size"] else 0.0
    summary["return_maturity_score"] = round(min(100.0, (summary["sample_size"] / 150.0) * 100.0), 2) if summary["sample_size"] else 0.0
    summary["return_grade"] = _grade(summary["return_maturity_score"], summary["sample_size"], summary.get("average_return_pct"), summary.get("return_stability"))
    summary["next_milestone"] = _next_milestone(summary["sample_size"])
    summary["estimated_additional_samples"] = _estimated_samples(summary["sample_size"])
    summary["report_interval"] = int(report_interval)
    summary["sample_size_stage"] = _pattern_maturity(summary["sample_size"])

    research_report = _research_report(summary, summary["pattern_profiles"], grouped)
    edge_stability = {
        "prediction_edge": _edge_score_from_rate(_safe_float(status.get("learning_intelligence", {}).get("average_prediction_accuracy"))),
        "execution_edge": _edge_score_from_rate(_safe_float(summary.get("average_execution_efficiency_pct"))),
        "return_edge": _edge_score_from_rate(_safe_float(summary.get("average_return_efficiency_pct"))),
        "risk_edge": _edge_score_from_rate(100.0 - (_safe_float(summary.get("average_risk_utilization_pct")) or 0.0)),
        "pattern_edge": _edge_score_from_rate(_safe_float(status.get("pattern_intelligence", {}).get("pattern_stability"))),
        "knowledge_edge": _edge_score_from_rate(_safe_float(summary.get("knowledge_confidence_score"))),
        "confidence_edge": _edge_score_from_rate(_safe_float(status.get("confidence_intelligence", {}).get("confidence_stability"))),
        "adaptive_readiness": _edge_score_from_rate(_safe_float(status.get("execution_intelligence", {}).get("execution_readiness_score"))),
        "history": [
            {"date": row.get("timestamp"), "return_edge": row.get("capture_ratio", STATUS_AWAITING), "execution_edge": row.get("exit_quality", "Unknown")}
            for row in summary["historical_return_evolution"][-12:]
        ],
    }

    academy_subject = {
        "academy": "Return Intelligence",
        "implementation": 100.0,
        "evidence": min(100.0, (summary["sample_size"] / 150.0) * 100.0) if summary["sample_size"] else 0.0,
        "knowledge_confidence": summary.get("knowledge_confidence_score", 0.0),
        "mastery": round(min(100.0, max(0.0, (summary.get("return_maturity_score", 0.0) * 0.55) + (summary.get("return_stability", 0.0) if isinstance(summary.get("return_stability"), (int, float)) else 0.0) * 0.45)), 2) if summary["sample_size"] else 0.0,
        "weighted_competency": round(min(100.0, max(0.0, (summary.get("return_maturity_score", 0.0) * 0.25) + (summary.get("knowledge_confidence_score", 0.0) * 0.25) + ((summary.get("return_stability", 0.0) if isinstance(summary.get("return_stability"), (int, float)) else 0.0) * 0.25) + (min(100.0, (summary["sample_size"] / 150.0) * 100.0) * 0.25))), 2) if summary["sample_size"] else 0.0,
        "current_grade": summary["return_grade"],
        "reliability": summary.get("return_stability", STATUS_AWAITING),
        "evidence_level": min(100.0, (summary["sample_size"] / 150.0) * 100.0) if summary["sample_size"] else 0.0,
        "sample_size": summary["sample_size"],
        "next_milestone": summary["next_milestone"],
        "estimated_additional_samples": summary["estimated_additional_samples"],
    }

    summary["exit_intelligence"]["average_exit_quality_score"] = summary["exit_intelligence"].get("average_exit_quality_score", STATUS_AWAITING)
    summary["exit_intelligence"]["exit_quality_distribution"] = {k: v for k, v in sorted(exit_counts.items(), key=lambda kv: kv[1], reverse=True)}

    return {
        "status": "Foundation" if summary["sample_size"] < 10 else "Developing",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "trade_metrics": trades,
        "return_by_pattern": grouped["pattern_name"],
        "return_by_pattern_family": grouped["pattern_family"],
        "return_by_pattern_cluster": grouped["pattern_cluster"],
        "return_by_session": grouped["session"],
        "return_by_regime": grouped["regime"],
        "return_by_volatility": grouped["volatility"],
        "return_by_trend": grouped["trend"],
        "return_by_timeframe": grouped["timeframe"],
        "return_by_model_version": grouped["model_version"],
        "return_by_feature_version": grouped["feature_version"],
        "pattern_profiles": summary["pattern_profiles"],
        "exit_intelligence": summary["exit_intelligence"],
        "historical_return_evolution": summary["historical_return_evolution"],
        "research_report": research_report,
        "zeus_research_proposals": research_report["zeus_research_proposals"],
        "academy_subject": academy_subject,
        "edge_stability": edge_stability,
    }


def _next_milestone(samples: int) -> str:
    if samples < 10:
        return "Collect at least 10 completed trades"
    if samples < 30:
        return "Cross 30 completed trades for research-grade sampling"
    if samples < 75:
        return "Stabilize return trends across 75 samples"
    if samples < 150:
        return "Validate return stability across 150 samples"
    return "Maintain longitudinal stability and evolve hypotheses"


def _estimated_samples(samples: int) -> int:
    if samples < 10:
        return 10 - samples
    if samples < 30:
        return 30 - samples
    if samples < 75:
        return 75 - samples
    if samples < 150:
        return 150 - samples
    return 50


def _grade(maturity: float, samples: int, avg_return: Any, stability: Any) -> str:
    if samples < 10:
        return "F"
    if samples < 30:
        return "C-" if (isinstance(avg_return, (int, float)) and avg_return > 0) else "D"
    if samples < 75:
        return "B-" if (isinstance(avg_return, (int, float)) and avg_return > 0 and isinstance(stability, (int, float)) and stability >= 45.0) else "C"
    if samples < 150:
        return "B" if (isinstance(avg_return, (int, float)) and avg_return > 0 and isinstance(stability, (int, float)) and stability >= 55.0) else "B-"
    return "A" if (isinstance(avg_return, (int, float)) and avg_return > 0 and isinstance(stability, (int, float)) and stability >= 60.0) else "B"


def _edge_score_from_rate(value: float | None) -> float | str:
    if value is None:
        return STATUS_AWAITING
    if value <= 1.0:
        return round(max(0.0, min(100.0, value * 100.0)), 2)
    return round(max(0.0, min(100.0, value)), 2)


def _derive_session(trade: dict[str, Any] | Any) -> str:
    ts = _get(trade, "opened_at") or _get(trade, "timestamp") or _get(trade, "closed_at")
    dt = _safe_dt(ts)
    if dt is None:
        return "unknown"
    hour = dt.astimezone(timezone.utc).hour
    if 0 <= hour < 7:
        return "Asian"
    if 7 <= hour < 10:
        return "London Open"
    if 10 <= hour < 13:
        return "London"
    if 13 <= hour < 16:
        return "NY"
    if 16 <= hour < 19:
        return "NY Lunch"
    return "Rollover"


def _derive_regime(trade: dict[str, Any] | Any) -> str:
    regime = str(_get(trade, "regime", "") or "").strip()
    if regime:
        return regime
    trend = str(_get(trade, "trend_state", "") or "").lower()
    if "compression" in trend:
        return "Compression"
    if "reversion" in trend:
        return "Mean Reversion"
    if "bull" in trend or "bear" in trend:
        return "Trend Expansion"
    return "Dead Liquidity"


def _derive_trend(trade: dict[str, Any] | Any) -> str:
    trend = str(_get(trade, "trend_state", "") or "").strip()
    if trend:
        return trend
    direction = str(_get(trade, "direction", "unknown") or "unknown").lower()
    if direction in ("long", "buy", "bullish"):
        return "bullish"
    if direction in ("short", "sell", "bearish"):
        return "bearish"
    return "sideways"


def _derive_volatility(trade: dict[str, Any] | Any) -> str:
    vol = str(_get(trade, "volatility_state", "") or "").strip()
    if vol:
        return vol
    mfe = _safe_float(_get(trade, "mfe_pct")) or 0.0
    mae = _safe_float(_get(trade, "mae_pct")) or 0.0
    if mfe + mae >= 3.0:
        return "High"
    if mfe + mae >= 1.0:
        return "Moderate"
    return "Low"
