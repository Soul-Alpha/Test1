from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from olympus.core.decision_replay_counterfactual_intelligence import (
    build_decision_replay_counterfactual_intelligence,
    write_decision_replay_artifacts,
)
from olympus.core.autonomous_research_orchestrator import (
    build_autonomous_research_orchestrator,
    write_aro_artifacts,
)
from olympus.core.explainability_engine import (
    build_explainability_engine,
    write_explainability_artifacts,
)
from olympus.core.institutional_capital_intelligence import (
    build_capital_intelligence,
    write_capital_intelligence_artifacts,
)
from olympus.core.institutional_knowledge_graph import (
    build_institutional_knowledge_graph,
    write_institutional_knowledge_graph_artifacts,
)
from olympus.core.institutional_learning_scientist import (
    build_institutional_learning_scientist,
    write_institutional_learning_artifacts,
)
from olympus.core.institutional_research_director import (
    build_institutional_research_director,
    write_research_director_artifacts,
)
from olympus.core.knowledge_coverage_intelligence import (
    build_knowledge_coverage_intelligence,
    write_knowledge_coverage_artifacts,
)
from olympus.core.knowledge_evolution_engine import (
    build_knowledge_evolution_engine,
    write_knowledge_evolution_artifacts,
)
from olympus.core.meta_learning_engine import (
    build_meta_learning_engine,
    write_meta_learning_artifacts,
)
from olympus.core.research_prioritization_engine import (
    build_research_prioritization_engine,
    write_research_prioritization_artifacts,
)
from olympus.core.trade_lifecycle_intelligence import build_trade_lifecycle_intelligence

IDIP_VERSION = "idip-v1.0"
STATUS_AWAITING = "Awaiting Historical Data"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


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


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _avg(values: list[float]) -> float | str:
    return round(mean(values), 4) if values else STATUS_AWAITING


def _med(values: list[float]) -> float | str:
    return round(median(values), 4) if values else STATUS_AWAITING


def _stdev(values: list[float]) -> float | str:
    return round(pstdev(values), 4) if len(values) > 1 else (0.0 if values else STATUS_AWAITING)


def _maturity(sample_size: int) -> str:
    if sample_size >= 300:
        return "Institutional"
    if sample_size >= 120:
        return "Validated"
    if sample_size >= 50:
        return "Developing"
    return "Emerging"


def _exit_classifier(trade: dict[str, Any]) -> str:
    reason = str(trade.get("exit_reason", "") or "").strip().lower()
    mfe = _safe_float(trade.get("mfe_pct")) or 0.0
    mae = _safe_float(trade.get("mae_pct")) or 0.0
    captured = _safe_float(trade.get("captured_return_pct")) or 0.0
    realized = _safe_float(trade.get("realized_return_pct"))
    if realized is None:
        realized = 0.0

    if reason in ("tp", "take_profit"):
        if captured >= 80.0:
            return "Pattern Completion"
        return "Momentum Exit"
    if reason in ("sl", "stop_loss"):
        if mfe > mae and mfe > 0.2:
            return "Trailing Stop"
        return "Risk Exit"
    if reason in ("micro_time_exit", "time_exit"):
        return "Time Exit"
    if "liq" in reason or reason in ("liquidity_exit", "liquidity"):
        return "Liquidity Exit"
    if "vol" in reason or reason in ("volatility_exit", "volatility"):
        return "Volatility Exit"
    if "struct" in reason or reason in ("structure_exit", "structure"):
        return "Structure Exit"
    if reason in ("breakeven", "break_even", "be"):
        return "Break-even"
    if reason in ("manual", "manual_override"):
        return "Manual Override"
    if realized >= 0 and captured >= 55.0:
        return "Momentum Exit"
    return "Unknown"


def _trade_return_pct(trade: dict[str, Any]) -> float:
    val = _safe_float(trade.get("realized_return_pct"))
    if val is not None:
        return float(val)
    entry = _safe_float(trade.get("entry_price")) or 0.0
    exitp = _safe_float(trade.get("exit_price"))
    if entry <= 0 or exitp is None:
        return 0.0
    direction = str(trade.get("direction", "long") or "long").lower()
    raw = ((exitp - entry) / entry) * 100.0
    return float(raw if direction == "long" else -raw)


def _hold_seconds(trade: dict[str, Any]) -> float | None:
    hold = _safe_float(trade.get("hold_seconds"))
    if hold is not None:
        return hold
    opened = _safe_dt(trade.get("opened_at"))
    closed = _safe_dt(trade.get("closed_at"))
    if opened is None or closed is None:
        return None
    return max(0.0, (closed - opened).total_seconds())


def _prepare_closed_trade_rows(closed_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in closed_trades:
        row = dict(tr)
        hold = _hold_seconds(row)
        realized = _trade_return_pct(row)
        mfe = _safe_float(row.get("mfe_pct")) or 0.0
        mae = _safe_float(row.get("mae_pct")) or 0.0
        captured = _safe_float(row.get("captured_return_pct"))
        if captured is None:
            captured = 0.0 if mfe <= 0 else max(0.0, min(100.0, (max(0.0, realized) / max(1e-9, mfe)) * 100.0))
        risk_util = _safe_float(row.get("risk_utilization_pct"))
        if risk_util is None:
            denom = max(1e-9, mfe + mae)
            risk_util = max(0.0, min(100.0, (mae / denom) * 100.0))

        row["hold_seconds"] = hold
        row["realized_return_pct"] = round(realized, 4)
        row["mfe_pct"] = round(mfe, 4)
        row["mae_pct"] = round(mae, 4)
        row["captured_return_pct"] = round(captured, 4)
        row["risk_utilization_pct"] = round(risk_util, 4)
        row["return_leakage_pct"] = round(max(0.0, 100.0 - captured), 4)
        row["classified_exit_style"] = _exit_classifier(row)
        rows.append(row)
    return rows


def _duration_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [r["hold_seconds"] for r in rows if r.get("hold_seconds") is not None]

    def _group(key: str) -> list[dict[str, Any]]:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            d = r.get("hold_seconds")
            if d is None:
                continue
            k = str(r.get(key, "unknown") or "unknown")
            groups[k].append(float(d))
        out = []
        for k, vals in groups.items():
            out.append(
                {
                    key: k,
                    "samples": len(vals),
                    "average_hold_sec": round(mean(vals), 2),
                    "median_hold_sec": round(median(vals), 2),
                    "hold_stdev": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
                }
            )
        out.sort(key=lambda x: x.get("samples", 0), reverse=True)
        return out

    by_conf: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        d = r.get("hold_seconds")
        if d is None:
            continue
        conf = _safe_float(r.get("signal_confidence"))
        if conf is None:
            bucket = "unknown"
        elif conf < 0.4:
            bucket = "0.00-0.39"
        elif conf < 0.6:
            bucket = "0.40-0.59"
        elif conf < 0.8:
            bucket = "0.60-0.79"
        else:
            bucket = "0.80-1.00"
        by_conf[bucket].append(float(d))

    by_conf_rows = [
        {
            "confidence_bucket": k,
            "samples": len(v),
            "average_hold_sec": round(mean(v), 2),
            "median_hold_sec": round(median(v), 2),
        }
        for k, v in by_conf.items()
    ]
    by_conf_rows.sort(key=lambda x: x.get("confidence_bucket", ""))

    returns = [r.get("realized_return_pct", 0.0) for r in rows]
    return {
        "average_hold_time_sec": _avg(durations),
        "median_hold_time_sec": _med(durations),
        "optimal_holding_time_proxy_sec": _med([d for d, ret in zip(durations, returns) if ret > 0]) if durations else STATUS_AWAITING,
        "duration_by_pattern": _group("pattern_name"),
        "duration_by_session": _group("session"),
        "duration_by_market_regime": _group("regime"),
        "duration_by_volatility": _group("volatility_state"),
        "duration_by_confidence": by_conf_rows,
        "duration_by_liquidity": _group("classified_exit_style"),
        "duration_by_trend_strength": _group("trend_state"),
    }


def _exit_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("classified_exit_style", "Unknown"))].append(r)

    exit_rows = []
    best_style = STATUS_AWAITING
    best_expectancy = -1e9
    for style, items in grouped.items():
        rets = [x.get("realized_return_pct", 0.0) for x in items]
        expectancy = mean(rets) if rets else 0.0
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_style = style
        exit_rows.append(
            {
                "exit_style": style,
                "samples": len(items),
                "expectancy": round(expectancy, 4),
                "capture_ratio": round(mean([x.get("captured_return_pct", 0.0) for x in items]), 4) if items else STATUS_AWAITING,
                "risk_pressure": round(mean([x.get("mae_pct", 0.0) for x in items]), 4) if items else STATUS_AWAITING,
            }
        )

    exit_rows.sort(key=lambda x: (x.get("samples", 0), x.get("expectancy", -999.0)), reverse=True)
    unknown_count = len(grouped.get("Unknown", []))
    return {
        "classified_exit_distribution": exit_rows,
        "best_expectancy_exit_style": best_style,
        "unknown_exit_count": unknown_count,
        "unknown_exit_reduction_target": "Drive Unknown toward zero via evidence growth",
    }


def _reward_capture_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capture = [r.get("captured_return_pct", 0.0) for r in rows]
    leakage = [r.get("return_leakage_pct", 0.0) for r in rows]
    mfe = [r.get("mfe_pct", 0.0) for r in rows]
    mae = [r.get("mae_pct", 0.0) for r in rows]

    return {
        "maximum_favorable_excursion_pct": _avg(mfe),
        "maximum_adverse_excursion_pct": _avg(mae),
        "captured_reward_pct": _avg(capture),
        "missed_reward_pct": _avg(leakage),
        "capture_ratio": _avg(capture),
        "return_leakage": _avg(leakage),
        "exit_timing_quality": round(max(0.0, min(100.0, (mean(capture) if capture else 0.0) - 0.2 * (mean(leakage) if leakage else 0.0))), 2),
        "reward_efficiency": _avg(capture),
        "opportunity_cost": _avg(leakage),
    }


def _position_management_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "advisory_only": True,
            "should_move_stop": STATUS_AWAITING,
            "should_break_even_activate": STATUS_AWAITING,
            "should_partial_profits": STATUS_AWAITING,
            "should_tp_extend": STATUS_AWAITING,
            "should_reduce_exposure": STATUS_AWAITING,
            "should_trailing_begin": STATUS_AWAITING,
            "should_trade_unchanged": STATUS_AWAITING,
            "decision_quality": STATUS_AWAITING,
        }

    avg_capture = mean([r.get("captured_return_pct", 0.0) for r in rows])
    avg_risk = mean([r.get("risk_utilization_pct", 0.0) for r in rows])
    avg_mfe = mean([r.get("mfe_pct", 0.0) for r in rows])
    avg_mae = mean([r.get("mae_pct", 0.0) for r in rows])

    return {
        "advisory_only": True,
        "should_move_stop": avg_risk > 50.0,
        "should_break_even_activate": avg_mfe >= 0.25,
        "should_partial_profits": avg_capture > 60.0,
        "should_tp_extend": avg_capture > 70.0 and avg_mfe > 0.4,
        "should_reduce_exposure": avg_risk > 60.0 or avg_mae > 0.35,
        "should_trailing_begin": avg_mfe > 0.3,
        "should_trade_unchanged": avg_risk < 45.0 and avg_capture > 55.0,
        "decision_quality": round(max(0.0, min(100.0, (avg_capture * 0.6) + ((100.0 - avg_risk) * 0.4))), 2),
    }


def _institutional_risk_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [r.get("realized_return_pct", 0.0) for r in rows]
    drawdowns = [r.get("mae_pct", 0.0) for r in rows]
    cap = [max(0.0, min(100.0, (r.get("captured_return_pct", 0.0) * 0.5) + ((100.0 - r.get("risk_utilization_pct", 0.0)) * 0.5))) for r in rows]

    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    pf = STATUS_AWAITING
    if wins and losses and abs(sum(losses)) > 1e-9:
        pf = round(sum(wins) / abs(sum(losses)), 4)

    return {
        "advisory_only": True,
        "position_sizing_score": _avg(cap),
        "capital_allocation_score": _avg(cap),
        "risk_budgeting_score": round(max(0.0, min(100.0, 100.0 - (mean(drawdowns) * 180.0 if drawdowns else 0.0))), 2) if rows else STATUS_AWAITING,
        "drawdown_recovery_score": round(max(0.0, min(100.0, 45.0 + (mean(returns) * 20.0 if returns else 0.0))), 2) if rows else STATUS_AWAITING,
        "risk_utilization": _avg([r.get("risk_utilization_pct", 0.0) for r in rows]),
        "portfolio_exposure_proxy": _avg(drawdowns),
        "capital_preservation": round(max(0.0, min(100.0, 70.0 - (mean(drawdowns) * 100.0 if drawdowns else 0.0))), 2) if rows else STATUS_AWAITING,
        "recovery_discipline": round(max(0.0, min(100.0, (mean(cap) if cap else 0.0) * 0.7 + (mean(returns) if returns else 0.0) * 5.0)), 2) if rows else STATUS_AWAITING,
        "profit_factor": pf,
    }


def _portfolio_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        by_session[str(r.get("session", "unknown") or "unknown")].append(r)
        by_regime[str(r.get("regime", "unknown") or "unknown")].append(r)
        by_pattern[str(r.get("pattern_name", "unknown") or "unknown")].append(r)

    total = max(1, len(rows))

    def _concentration(groups: dict[str, list[dict[str, Any]]], key_name: str) -> list[dict[str, Any]]:
        out = []
        for k, vals in groups.items():
            rets = [x.get("realized_return_pct", 0.0) for x in vals]
            out.append(
                {
                    key_name: k,
                    "samples": len(vals),
                    "share_pct": round((len(vals) / total) * 100.0, 2),
                    "return_contribution": round(sum(rets), 4),
                    "expectancy": round(mean(rets), 4) if rets else STATUS_AWAITING,
                }
            )
        out.sort(key=lambda x: x.get("samples", 0), reverse=True)
        return out

    exposures = [len(v) / total for v in by_pattern.values()] if rows else []
    hhi = sum(x * x for x in exposures) if exposures else 0.0
    return {
        "session_exposure": _concentration(by_session, "session"),
        "regime_exposure": _concentration(by_regime, "regime"),
        "pattern_concentration": _concentration(by_pattern, "pattern"),
        "risk_concentration_hhi": round(hhi, 4),
        "capital_efficiency": _avg([max(0.0, min(100.0, (r.get("captured_return_pct", 0.0) * 0.4) + ((100.0 - r.get("risk_utilization_pct", 0.0)) * 0.6))) for r in rows]) if rows else STATUS_AWAITING,
        "portfolio_expectancy": _avg([r.get("realized_return_pct", 0.0) for r in rows]),
    }


def _decision_attribution_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attribution_rows: list[dict[str, Any]] = []
    improved, degraded = 0, 0
    for r in rows:
        capture = r.get("captured_return_pct", 0.0)
        risk = r.get("risk_utilization_pct", 0.0)
        ret = r.get("realized_return_pct", 0.0)

        improved_outcome = ret > 0 and capture >= 55.0
        increased_risk = risk > 60.0 or r.get("mae_pct", 0.0) > 0.35
        reduced_expectancy = ret < 0 or capture < 35.0

        score = max(0.0, min(100.0, (capture * 0.6) + ((100.0 - risk) * 0.4)))
        if improved_outcome:
            improved += 1
        if reduced_expectancy or increased_risk:
            degraded += 1

        attribution_rows.append(
            {
                "trade_id": r.get("trade_id"),
                "decision_score": round(score, 2),
                "improved_outcome": improved_outcome,
                "reduced_expectancy": reduced_expectancy,
                "increased_risk": increased_risk,
                "improved_reward_capture": capture >= 60.0,
                "caused_unnecessary_loss": ret < 0 and r.get("mfe_pct", 0.0) > r.get("mae_pct", 0.0),
                "evidence_score": round(min(1.0, 0.45 + min(0.55, abs(ret) * 0.08 + capture * 0.0025)), 4),
            }
        )

    return {
        "decision_attribution_rows": attribution_rows[-200:],
        "decisions_improved_outcomes": improved,
        "decisions_degraded_outcomes": degraded,
        "average_decision_score": _avg([x.get("decision_score", 0.0) for x in attribution_rows]),
    }


def _counterfactual_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for r in rows[-200:]:
        trade_id = str(r.get("trade_id", "unknown") or "unknown")
        ret = r.get("realized_return_pct", 0.0)
        capture = r.get("captured_return_pct", 0.0)
        mfe = r.get("mfe_pct", 0.0)

        scenarios.append(
            {
                "trade_id": trade_id,
                "baseline_return_pct": round(ret, 4),
                "what_if_tp_extended": round(ret + max(0.0, mfe * 0.15), 4),
                "what_if_trailing_later": round(ret + max(0.0, mfe * 0.10), 4),
                "what_if_break_even_earlier": round(ret + (0.05 if ret < 0 else 0.0), 4),
                "what_if_position_size_plus_10pct": round(ret * 1.1, 4),
                "what_if_liquidity_exit": round(ret + max(0.0, (100.0 - capture) * 0.002), 4),
                "assumption_confidence": round(min(1.0, 0.35 + min(0.5, capture * 0.004)), 4),
                "research_only": True,
            }
        )

    avg_uplift = _avg([
        s.get("what_if_tp_extended", 0.0) - s.get("baseline_return_pct", 0.0)
        for s in scenarios
    ]) if scenarios else STATUS_AWAITING
    return {
        "counterfactual_scenarios": scenarios[-120:],
        "average_counterfactual_uplift": avg_uplift,
        "research_only": True,
    }


def _pattern_lifecycle_intelligence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("pattern_name", "unknown") or "unknown")].append(r)

    lifecycle_profiles = []
    for pattern, items in grouped.items():
        seq_counter: Counter[str] = Counter()
        for r in items:
            seq = ["Signal", "Validation", "Entry", "Protection"]
            if r.get("captured_return_pct", 0.0) >= 40.0:
                seq.append("Scaling")
            if r.get("captured_return_pct", 0.0) >= 60.0:
                seq.append("Trailing")
            seq.append(str(r.get("classified_exit_style", "Unknown")))
            seq.append("Replay")
            seq.append("Expected Outcome")
            seq_counter.update([" -> ".join(seq)])

        rets = [x.get("realized_return_pct", 0.0) for x in items]
        lifecycle_profiles.append(
            {
                "pattern": pattern,
                "samples": len(items),
                "top_lifecycle_sequence": seq_counter.most_common(1)[0][0] if seq_counter else STATUS_AWAITING,
                "sequence_diversity": len(seq_counter),
                "expectancy": round(mean(rets), 4) if rets else STATUS_AWAITING,
                "capture_ratio": round(mean([x.get("captured_return_pct", 0.0) for x in items]), 4) if items else STATUS_AWAITING,
            }
        )

    lifecycle_profiles.sort(key=lambda x: (x.get("samples", 0), _safe_float(x.get("expectancy")) or -999.0), reverse=True)
    return {
        "pattern_lifecycle_profiles": lifecycle_profiles,
        "top_expectancy_lifecycles": lifecycle_profiles[:20],
    }


def _institutional_knowledge_intelligence(rows: list[dict[str, Any]], attribution: dict[str, Any]) -> dict[str, Any]:
    lessons = []
    for r in rows[-250:]:
        ret = r.get("realized_return_pct", 0.0)
        capture = r.get("captured_return_pct", 0.0)
        risk = r.get("risk_utilization_pct", 0.0)
        lesson = {
            "knowledge_id": f"IK-{str(r.get('trade_id', 'unknown'))}",
            "trade_id": r.get("trade_id"),
            "decision_quality": round(max(0.0, min(100.0, (capture * 0.6) + ((100.0 - risk) * 0.4))), 2),
            "lifecycle_quality": round(max(0.0, min(100.0, capture * 0.9)), 2),
            "risk_quality": round(max(0.0, min(100.0, 100.0 - risk)), 2),
            "reward_quality": round(max(0.0, min(100.0, capture)), 2),
            "exit_quality": str(r.get("classified_exit_style", "Unknown")),
            "duration_quality": "Efficient" if (_safe_float(r.get("hold_seconds")) or 0.0) > 0 and ret > 0 else "Developing",
            "pattern_evolution": str(r.get("pattern_name", "unknown")),
            "expectancy_evolution": round(ret, 4),
            "future_recommendation": "Preserve behavior" if ret > 0 and capture > 60.0 else "Investigate lifecycle optimization",
            "immutable": True,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        lessons.append(lesson)

    return {
        "institutional_lessons": lessons[-150:],
        "knowledge_count": len(lessons),
        "attribution_quality": attribution.get("average_decision_score", STATUS_AWAITING),
        "immutability_enforced": True,
    }


def _recommendations_for_zeus(
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    summary = payload.get("summary", {})
    capture = _safe_float(summary.get("reward_efficiency"))
    expectancy = _safe_float(summary.get("expectancy"))
    unknown_exits = int(payload.get("engines", {}).get("exit_intelligence", {}).get("unknown_exit_count", 0) or 0)

    def _make_id(seed: str) -> str:
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"idip-rec-{digest}"

    if capture is not None and capture < 55.0:
        text = "Reward capture below institutional target; validate adaptive trailing + partial exits by session/regime."
        recommendations.append(
            {
                "recommendation_id": _make_id(text),
                "source_system": "hermes",
                "validation_domain": "recommendation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recommendation_type": "reward_capture_optimization",
                "recommendation": text,
                "expected_improvement": "Increase realized return and reduce leakage",
                "evidence": {
                    "sample_size": len(rows),
                    "confidence": max(0.0, min(1.0, 0.35 + len(rows) / 600.0)),
                    "evidence_score": max(0.0, min(1.0, 0.35 + len(rows) / 600.0)),
                    "supporting_metric": "reward_efficiency",
                    "supporting_value": capture,
                },
                "lifecycle": "candidate",
                "operator_approved": False,
                "requires_zeus_validation": True,
                "governance_required": True,
            }
        )

    if expectancy is not None and expectancy < 0:
        text = "Negative expectancy observed; validate stricter lifecycle risk controls and exit-style selection."
        recommendations.append(
            {
                "recommendation_id": _make_id(text),
                "source_system": "hermes",
                "validation_domain": "execution",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recommendation_type": "expectancy_recovery",
                "recommendation": text,
                "expected_improvement": "Improve expectancy and recovery factor",
                "evidence": {
                    "sample_size": len(rows),
                    "confidence": max(0.0, min(1.0, 0.30 + len(rows) / 700.0)),
                    "evidence_score": max(0.0, min(1.0, 0.30 + len(rows) / 700.0)),
                    "supporting_metric": "expectancy",
                    "supporting_value": expectancy,
                },
                "lifecycle": "candidate",
                "operator_approved": False,
                "requires_zeus_validation": True,
                "governance_required": True,
            }
        )

    if unknown_exits > 0:
        text = "Unknown exit styles remain; validate expanded exit taxonomy and attribution rules."
        recommendations.append(
            {
                "recommendation_id": _make_id(text),
                "source_system": "hermes",
                "validation_domain": "pattern",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recommendation_type": "exit_classification_quality",
                "recommendation": text,
                "expected_improvement": "Reduce unknown exits and improve attribution confidence",
                "evidence": {
                    "sample_size": len(rows),
                    "confidence": max(0.0, min(1.0, 0.25 + len(rows) / 800.0)),
                    "evidence_score": max(0.0, min(1.0, 0.25 + len(rows) / 800.0)),
                    "supporting_metric": "unknown_exit_count",
                    "supporting_value": unknown_exits,
                },
                "lifecycle": "candidate",
                "operator_approved": False,
                "requires_zeus_validation": True,
                "governance_required": True,
            }
        )

    return recommendations


def build_institutional_decision_intelligence_platform(
    root_dir: Path,
    *,
    status: dict[str, Any],
    open_trades: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    feature_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    flags = feature_flags or {}

    def _enabled(name: str, default: bool = True) -> bool:
        raw = flags.get(name, default)
        if isinstance(raw, str):
            return raw.strip().lower() not in ("0", "false", "no", "off")
        return bool(raw)

    all_rows = _prepare_closed_trade_rows(closed_trades)
    max_rows_raw = str(os.getenv("IDIP_MAX_RUNTIME_ROWS", "600") or "600").strip()
    try:
        max_rows = max(100, int(max_rows_raw))
    except Exception:
        max_rows = 600
    rows = all_rows[-max_rows:]
    tli = build_trade_lifecycle_intelligence(
        root_dir,
        status=status,
        open_trades=open_trades,
        closed_trades=closed_trades,
    )

    exit_engine = _exit_intelligence(rows)
    duration_engine = _duration_intelligence(rows)
    reward_engine = _reward_capture_intelligence(rows)
    position_engine = _position_management_intelligence(rows)
    risk_engine = _institutional_risk_intelligence(rows)
    portfolio_engine = _portfolio_intelligence(rows)
    attribution_engine = _decision_attribution_intelligence(rows)
    counterfactual_engine = _counterfactual_intelligence(rows)
    pattern_engine = _pattern_lifecycle_intelligence(rows)
    knowledge_engine = _institutional_knowledge_intelligence(rows, attribution_engine)

    expectancy = _avg([r.get("realized_return_pct", 0.0) for r in rows])
    summary = {
        "expectancy": expectancy,
        "reward_efficiency": reward_engine.get("reward_efficiency", STATUS_AWAITING),
        "risk_efficiency": _avg([100.0 - r.get("risk_utilization_pct", 0.0) for r in rows]) if rows else STATUS_AWAITING,
        "decision_quality": attribution_engine.get("average_decision_score", STATUS_AWAITING),
        "unknown_exit_count": exit_engine.get("unknown_exit_count", 0),
        "sample_size": len(rows),
        "maturity": _maturity(len(rows)),
    }

    capital_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_capital_intelligence_engine", True):
        capital_engine = build_capital_intelligence(
            status=status,
            closed_trades=rows,
            account_events=[x for x in (status.get("account_events", []) or []) if isinstance(x, dict)],
        )

    replay_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
        "replay_rows": [],
        "zeus_candidates": [],
    }
    if _enabled("enable_decision_replay_counterfactual_intelligence", True):
        replay_engine = build_decision_replay_counterfactual_intelligence(closed_trades=rows)

    knowledge_graph_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
        "decision_paths": [],
    }
    if _enabled("enable_knowledge_graph_engine", True):
        knowledge_graph_engine = build_institutional_knowledge_graph(
            closed_trades=rows,
            attribution_rows=attribution_engine.get("decision_attribution_rows", []),
            version_seed=started.strftime("%Y%m%d%H%M%S"),
        )

    zeus_reports = _load_jsonl(root_dir / "storage" / "olympus" / "zeus_validation_reports.jsonl")
    learning_seed_payload = {
        "summary": {"unknown_exit_count": exit_engine.get("unknown_exit_count", 0)},
        "engines": {
            "decision_attribution_intelligence": attribution_engine,
            "institutional_knowledge_intelligence": knowledge_engine,
        },
    }
    learning_engine: dict[str, Any] = {
        "institutional_learning": {
            "version": "disabled",
            "status": "feature_flag_disabled",
        },
        "hypotheses": {"rows": []},
        "knowledge_growth": {},
        "learning_velocity": {},
        "research_queue": {"rows": []},
        "concept_drift": {},
    }
    if _enabled("enable_institutional_learning_scientist", True):
        learning_engine = build_institutional_learning_scientist(
            root_dir,
            status=status,
            closed_trades=rows,
            replay_rows=replay_engine.get("replay_rows", []),
            zeus_reports=zeus_reports,
            simulation_rows=[x for x in (capital_engine.get("equity_curves", {}) or {}).get("strategy_equity_curve", []) if isinstance(x, dict)],
            idip_payload=learning_seed_payload,
        )

    knowledge_coverage_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_knowledge_coverage_intelligence", True):
        knowledge_coverage_engine = build_knowledge_coverage_intelligence(status=status, idip_payload={"engines": {**learning_seed_payload.get("engines", {}), "decision_replay_counterfactual_intelligence": replay_engine}, "summary": learning_seed_payload.get("summary", {})})

    meta_learning_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_meta_learning_engine", True):
        meta_learning_engine = build_meta_learning_engine(root_dir, idip_payload={"summary": summary, "engines": {"pattern_lifecycle_intelligence": pattern_engine}})

    aro_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
        "zeus_submission_queue": [],
    }
    if _enabled("enable_autonomous_research_orchestrator", True):
        aro_engine = build_autonomous_research_orchestrator(
            root_dir,
            hypotheses_rows=learning_engine.get("hypotheses", {}).get("rows", []) if isinstance(learning_engine, dict) else [],
            recommendation_rows=[],
            knowledge_gap_rows=knowledge_coverage_engine.get("coverage_rows", []) if isinstance(knowledge_coverage_engine, dict) else [],
        )

    research_prioritization_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_research_prioritization_engine", True):
        research_prioritization_engine = build_research_prioritization_engine(
            candidates=aro_engine.get("candidates", []) if isinstance(aro_engine, dict) else []
        )

    research_director_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_institutional_research_director", True):
        research_director_engine = build_institutional_research_director(
            aro_payload=aro_engine,
            hypotheses_rows=learning_engine.get("hypotheses", {}).get("rows", []) if isinstance(learning_engine, dict) else [],
        )

    knowledge_evolution_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_knowledge_evolution_engine", True):
        knowledge_evolution_engine = build_knowledge_evolution_engine(root_dir, idip_payload={"engines": {"institutional_knowledge_intelligence": knowledge_engine}})

    explainability_engine: dict[str, Any] = {
        "version": "disabled",
        "status": "feature_flag_disabled",
    }
    if _enabled("enable_explainability_engine", True):
        explainability_engine = build_explainability_engine(
            recommendation_rows=[],
            context={
                "expectancy": expectancy,
                "decision_quality": summary.get("decision_quality", STATUS_AWAITING),
                "risk_efficiency": summary.get("risk_efficiency", STATUS_AWAITING),
            },
        )

    ended = datetime.now(timezone.utc)
    runtime_ms = round((ended - started).total_seconds() * 1000.0, 3)

    prior = _load_jsonl(root_dir / "storage" / "olympus" / "idip_history.jsonl")
    baselines = [
        _safe_float(r.get("performance", {}).get("runtime_ms"))
        for r in prior[-120:]
        if isinstance(r, dict)
    ]
    baseline_vals = [x for x in baselines if x is not None]
    baseline_runtime = round(mean(baseline_vals), 3) if baseline_vals else None

    payload = {
        "meta": {
            "version": IDIP_VERSION,
            "generated_at": ended.isoformat(),
            "source_system": str(status.get("source_system", "hermes") or "hermes"),
            "asset": str(status.get("asset", "unknown") or "unknown"),
            "timeframe": str(status.get("timeframe", "unknown") or "unknown"),
            "mode": "observational_and_research",
            "additive_only": True,
            "execution_behavior_unchanged": True,
            "auto_execution_modification": False,
            "zeus_validation_required": True,
            "governance_approval_required": True,
            "prometheus_consumption": "validated_knowledge_only",
        },
        "summary": summary,
        "engines": {
            "trade_lifecycle_intelligence": tli,
            "exit_intelligence": exit_engine,
            "duration_intelligence": duration_engine,
            "reward_capture_intelligence": reward_engine,
            "position_management_intelligence": position_engine,
            "institutional_risk_intelligence": risk_engine,
            "portfolio_intelligence": portfolio_engine,
            "decision_attribution_intelligence": attribution_engine,
            "counterfactual_intelligence": counterfactual_engine,
            "pattern_lifecycle_intelligence": pattern_engine,
            "institutional_knowledge_intelligence": knowledge_engine,
            "institutional_learning_scientist": learning_engine,
            "capital_intelligence_engine": capital_engine,
            "knowledge_graph_engine": knowledge_graph_engine,
            "decision_replay_counterfactual_intelligence": replay_engine,
            "knowledge_coverage_intelligence": knowledge_coverage_engine,
            "meta_learning_engine": meta_learning_engine,
            "autonomous_research_orchestrator": aro_engine,
            "research_prioritization_engine": research_prioritization_engine,
            "institutional_research_director": research_director_engine,
            "knowledge_evolution_engine": knowledge_evolution_engine,
            "explainability_engine": explainability_engine,
        },
        "feature_flags": {
            "enable_institutional_learning_scientist": _enabled("enable_institutional_learning_scientist", True),
            "enable_capital_intelligence_engine": _enabled("enable_capital_intelligence_engine", True),
            "enable_knowledge_graph_engine": _enabled("enable_knowledge_graph_engine", True),
            "enable_decision_replay_counterfactual_intelligence": _enabled("enable_decision_replay_counterfactual_intelligence", True),
            "enable_meta_learning_engine": _enabled("enable_meta_learning_engine", True),
            "enable_autonomous_research_orchestrator": _enabled("enable_autonomous_research_orchestrator", True),
            "enable_research_prioritization_engine": _enabled("enable_research_prioritization_engine", True),
            "enable_knowledge_evolution_engine": _enabled("enable_knowledge_evolution_engine", True),
            "enable_explainability_engine": _enabled("enable_explainability_engine", True),
            "enable_knowledge_coverage_intelligence": _enabled("enable_knowledge_coverage_intelligence", True),
            "enable_institutional_research_director": _enabled("enable_institutional_research_director", True),
        },
        "continuous_improvement_loop": {
            "stages": [
                "Market",
                "Trade",
                "Lifecycle Observation",
                "Decision Attribution",
                "Counterfactual Replay",
                "Hypothesis Generation",
                "Zeus Validation",
                "Evidence Score",
                "Institutional Knowledge Base",
                "Recommendation",
                "Governance Review",
                "Optional Adoption",
                "Performance Monitoring",
                "Continuous Improvement",
            ],
            "active": True,
            "automatic_adoption_allowed": False,
        },
        "self_improvement_loop": {
            "stages": [
                "Market",
                "Signal",
                "Trade",
                "Trade Lifecycle Observation",
                "Decision Attribution",
                "Replay",
                "Counterfactual Analysis",
                "Hypothesis",
                "Zeus Validation",
                "Evidence",
                "Institutional Knowledge",
                "Recommendation",
                "Governance Review",
                "Optional Adoption",
                "Performance Monitoring",
                "Continuous Learning",
            ],
            "active": True,
            "zeus_governance_required": True,
            "execution_bypass_allowed": False,
        },
        "performance": {
            "runtime_ms": runtime_ms,
            "baseline_runtime_ms": baseline_runtime if baseline_runtime is not None else STATUS_AWAITING,
            "delta_pct_vs_baseline": round(((runtime_ms - baseline_runtime) / baseline_runtime) * 100.0, 4) if baseline_runtime and baseline_runtime > 0 else STATUS_AWAITING,
            "rows_processed": len(rows),
            "total_closed_input_rows": len(all_rows),
            "runtime_window_rows": max_rows,
            "open_trades": len(open_trades),
            "closed_trades": len(closed_trades),
            "compute_mode": "single_pass_bounded",
        },
    }

    recommendations = _recommendations_for_zeus(payload=payload, rows=rows)
    recommendations.extend(replay_engine.get("zeus_candidates", []))
    recommendations.extend(aro_engine.get("zeus_submission_queue", []))
    for item in learning_engine.get("research_queue", {}).get("rows", []):
        rec_text = str(item.get("statement") or "Institutional learning research candidate")
        rec_id = f"ils-rec-{hashlib.sha1(rec_text.encode('utf-8')).hexdigest()[:12]}"
        recommendations.append(
            {
                "recommendation_id": rec_id,
                "source_system": "hermes",
                "validation_domain": "recommendation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recommendation_type": "institutional_learning_hypothesis",
                "recommendation": rec_text,
                "expected_improvement": "Reduce drift and improve institutional learning quality",
                "evidence": {
                    "sample_size": len(rows),
                    "confidence": 0.35,
                    "evidence_score": 0.35,
                    "supporting_metric": "learning_velocity",
                    "supporting_value": learning_engine.get("learning_velocity", {}).get("learning_velocity", STATUS_AWAITING),
                },
                "lifecycle": "candidate",
                "operator_approved": False,
                "requires_zeus_validation": True,
                "governance_required": True,
            }
        )

    dedupe: dict[str, dict[str, Any]] = {}
    for rec in recommendations:
        rid = str((rec or {}).get("recommendation_id", "") or "")
        if rid:
            dedupe[rid] = rec

    payload["zeus_research_recommendations"] = list(dedupe.values())

    if _enabled("enable_explainability_engine", True):
        payload["engines"]["explainability_engine"] = build_explainability_engine(
            recommendation_rows=payload["zeus_research_recommendations"],
            context={
                "expectancy": payload.get("summary", {}).get("expectancy", STATUS_AWAITING),
                "decision_quality": payload.get("summary", {}).get("decision_quality", STATUS_AWAITING),
                "risk_efficiency": payload.get("summary", {}).get("risk_efficiency", STATUS_AWAITING),
            },
        )
    return payload


def write_idip_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime_path = storage / "idip_runtime.json"
    history_path = storage / "idip_history.jsonl"
    recommendation_queue_path = storage / "idip_recommendation_queue.jsonl"
    knowledge_path = storage / "idip_knowledge_base.jsonl"

    _write_json_atomic(runtime_path, payload)
    _append_jsonl(history_path, payload)

    existing_recs = {
        str(r.get("recommendation_id", ""))
        for r in _load_jsonl(recommendation_queue_path)
        if isinstance(r, dict)
    }
    recommendations = payload.get("zeus_research_recommendations", [])
    for rec in recommendations:
        rec_id = str((rec or {}).get("recommendation_id", "") or "")
        if not rec_id or rec_id in existing_recs:
            continue
        _append_jsonl(recommendation_queue_path, rec)
        existing_recs.add(rec_id)

    existing_knowledge = {
        str(r.get("knowledge_id", ""))
        for r in _load_jsonl(knowledge_path)
        if isinstance(r, dict)
    }
    lessons = (
        payload.get("engines", {})
        .get("institutional_knowledge_intelligence", {})
        .get("institutional_lessons", [])
    )
    for lesson in lessons:
        kid = str((lesson or {}).get("knowledge_id", "") or "")
        if not kid or kid in existing_knowledge:
            continue
        _append_jsonl(knowledge_path, lesson)
        existing_knowledge.add(kid)

    if not recommendation_queue_path.exists():
        recommendation_queue_path.write_text("", encoding="utf-8")
    if not knowledge_path.exists():
        knowledge_path.write_text("", encoding="utf-8")

    subsystem_artifacts: dict[str, Any] = {}
    engines = payload.get("engines", {}) if isinstance(payload, dict) else {}

    learning_payload = engines.get("institutional_learning_scientist", {})
    if isinstance(learning_payload, dict) and learning_payload.get("institutional_learning"):
        try:
            subsystem_artifacts["institutional_learning"] = write_institutional_learning_artifacts(root_dir, learning_payload)
        except Exception:
            subsystem_artifacts["institutional_learning"] = {}

    capital_payload = engines.get("capital_intelligence_engine", {})
    if isinstance(capital_payload, dict) and capital_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["capital_intelligence"] = write_capital_intelligence_artifacts(root_dir, capital_payload)
        except Exception:
            subsystem_artifacts["capital_intelligence"] = {}

    graph_payload = engines.get("knowledge_graph_engine", {})
    if isinstance(graph_payload, dict) and graph_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["knowledge_graph"] = write_institutional_knowledge_graph_artifacts(root_dir, graph_payload)
        except Exception:
            subsystem_artifacts["knowledge_graph"] = {}

    replay_payload = engines.get("decision_replay_counterfactual_intelligence", {})
    if isinstance(replay_payload, dict) and replay_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["decision_replay"] = write_decision_replay_artifacts(root_dir, replay_payload)
        except Exception:
            subsystem_artifacts["decision_replay"] = {}

    coverage_payload = engines.get("knowledge_coverage_intelligence", {})
    if isinstance(coverage_payload, dict) and coverage_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["knowledge_coverage"] = write_knowledge_coverage_artifacts(root_dir, coverage_payload)
        except Exception:
            subsystem_artifacts["knowledge_coverage"] = {}

    meta_payload = engines.get("meta_learning_engine", {})
    if isinstance(meta_payload, dict) and meta_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["meta_learning"] = write_meta_learning_artifacts(root_dir, meta_payload)
        except Exception:
            subsystem_artifacts["meta_learning"] = {}

    aro_payload = engines.get("autonomous_research_orchestrator", {})
    if isinstance(aro_payload, dict) and aro_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["aro"] = write_aro_artifacts(root_dir, aro_payload)
        except Exception:
            subsystem_artifacts["aro"] = {}

    rp_payload = engines.get("research_prioritization_engine", {})
    if isinstance(rp_payload, dict) and rp_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["research_prioritization"] = write_research_prioritization_artifacts(root_dir, rp_payload)
        except Exception:
            subsystem_artifacts["research_prioritization"] = {}

    rd_payload = engines.get("institutional_research_director", {})
    if isinstance(rd_payload, dict) and rd_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["research_director"] = write_research_director_artifacts(root_dir, rd_payload)
        except Exception:
            subsystem_artifacts["research_director"] = {}

    ke_payload = engines.get("knowledge_evolution_engine", {})
    if isinstance(ke_payload, dict) and ke_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["knowledge_evolution"] = write_knowledge_evolution_artifacts(root_dir, ke_payload)
        except Exception:
            subsystem_artifacts["knowledge_evolution"] = {}

    ex_payload = engines.get("explainability_engine", {})
    if isinstance(ex_payload, dict) and ex_payload.get("version") not in (None, "disabled"):
        try:
            subsystem_artifacts["explainability"] = write_explainability_artifacts(root_dir, ex_payload)
        except Exception:
            subsystem_artifacts["explainability"] = {}

    return {
        "runtime": str(runtime_path),
        "history": str(history_path),
        "recommendation_queue": str(recommendation_queue_path),
        "knowledge_base": str(knowledge_path),
        "subsystem_artifacts": subsystem_artifacts,
    }
