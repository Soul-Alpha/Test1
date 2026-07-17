from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from olympus.contracts import SourceSystem

STATUS_AWAITING = "Awaiting Historical Data"
DATASET_VERSION = "idav-v1.0"

_VALIDATED_STATUSES = {"approved", "passed", "validated", "operator_approved", "completed"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).isoformat()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return default
        return json.loads(raw)
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return rows
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _safe_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"", "unknown", "none", "n/a", "awaiting historical data", "pending"}
    return False


def _trade_uid(row: dict[str, Any]) -> str:
    tid = str(row.get("trade_id") or "").strip()
    if tid:
        return tid
    sig = str(row.get("signal_id") or "").strip()
    opened = str(row.get("opened_at") or row.get("created_at") or "")
    direction = str(row.get("direction") or "")
    entry = str(row.get("entry_price") or row.get("entry") or "")
    raw = f"{sig}|{opened}|{direction}|{entry}".encode("utf-8", errors="ignore")
    return "anon-" + hashlib.sha1(raw).hexdigest()[:16]


def _day_of_week(ts: Any) -> str:
    dt = _safe_iso(ts)
    if dt is None:
        return "unknown"
    return dt.strftime("%A")


def _duration_seconds(entry_ts: Any, exit_ts: Any, fallback: Any = None) -> float | None:
    fb = _safe_float(fallback)
    if fb is not None:
        return float(fb)
    e1 = _safe_iso(entry_ts)
    e2 = _safe_iso(exit_ts)
    if e1 is None or e2 is None:
        return None
    return max(0.0, (e2 - e1).total_seconds())


def _calc_return_pct(row: dict[str, Any]) -> float | None:
    if row.get("return_pct") is not None:
        return _safe_float(row.get("return_pct"))
    entry = _safe_float(row.get("entry_price"))
    exit_p = _safe_float(row.get("exit_price"))
    if entry is None or exit_p is None or abs(entry) <= 1e-9:
        return None
    direction = str(row.get("direction") or "").lower()
    raw = (exit_p - entry) / entry * 100.0
    if direction in {"short", "sell", "bearish"}:
        raw *= -1.0
    return float(raw)


def _calc_r_multiple(row: dict[str, Any]) -> float | None:
    explicit = _safe_float(row.get("r_multiple"))
    if explicit is not None:
        return explicit
    rr = _safe_float(row.get("rr"))
    if rr is not None:
        return rr
    entry = _safe_float(row.get("entry_price"))
    sl = _safe_float(row.get("sl_price"))
    exit_p = _safe_float(row.get("exit_price"))
    if entry is None or sl is None or exit_p is None:
        return None
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return None
    direction = str(row.get("direction") or "").lower()
    reward = (exit_p - entry) if direction in {"long", "buy", "bullish"} else (entry - exit_p)
    return float(reward / risk)


def _prometheus_closed_trades(root_dir: Path) -> list[dict[str, Any]]:
    db_path = root_dir / "storage" / "prometheus.db"
    if not db_path.exists():
        return []
    query = (
        "SELECT trade_id, created_at, asset, timeframe, direction, entry_price, sl_price, tp_price, "
        "exit_price, size, pnl, rr, status, session, regime, spread_at_entry, score_at_entry, "
        "exit_reason, mae, mfe, hold_seconds "
        "FROM trades WHERE status IS NOT NULL AND lower(status) <> 'open'"
    )
    rows: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        for r in cur.fetchall():
            row = dict(r)
            row["source_system"] = SourceSystem.PROMETHEUS.value
            row["opened_at"] = row.get("created_at")
            row["exit_timestamp"] = row.get("created_at")
            row["lots"] = row.get("size")
            row["signal_confidence"] = row.get("score_at_entry")
            rows.append(row)
        conn.close()
    except Exception:
        return rows
    return rows


def _zeus_reports(root_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(root_dir / "storage" / "olympus" / "zeus_validation_reports.jsonl")


def _zeus_lookup(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in reports:
        keys = {
            str(row.get("candidate_id") or "").strip(),
            str(row.get("report_id") or "").strip(),
        }
        for key in keys:
            if key:
                lookup[key] = row
    return lookup


def _zeus_for_trade(trade: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        str(trade.get("trade_id") or "").strip(),
        str(trade.get("signal_id") or "").strip(),
        str(trade.get("pattern_name") or "").strip(),
    ]
    for key in keys:
        if key and key in lookup:
            row = lookup[key]
            status = str(row.get("status") or "pending").lower()
            approved = bool(row.get("approved_for_adoption", False))
            decision = "approved" if approved else ("passed" if status == "passed" else status)
            return {
                "validation_status": status,
                "zeus_decision": decision,
                "evidence_score": _safe_float(row.get("evidence_score")),
                "confidence_score": _safe_float(row.get("confidence")),
                "knowledge_object_id": str(row.get("candidate_id") or ""),
                "counterfactual_result": str((row.get("evidence") or {}).get("counterfactual_result") or "unknown"),
                "replay_result": str((row.get("evidence") or {}).get("replay_result") or "unknown"),
            }
    return {
        "validation_status": "pending",
        "zeus_decision": "pending",
        "evidence_score": None,
        "confidence_score": _safe_float(trade.get("signal_confidence")),
        "knowledge_object_id": "",
        "counterfactual_result": "unknown",
        "replay_result": "unknown",
    }


def _build_observation(trade: dict[str, Any], zeus: dict[str, Any], generated_at: str) -> dict[str, Any]:
    entry_ts = trade.get("opened_at") or trade.get("entry_timestamp") or trade.get("created_at")
    exit_ts = trade.get("closed_at") or trade.get("exit_timestamp") or trade.get("updated_at")
    stop_distance = None
    target_distance = None
    entry_price = _safe_float(trade.get("entry_price"))
    sl_price = _safe_float(trade.get("sl_price"))
    tp_price = _safe_float(trade.get("tp_price"))
    if entry_price is not None and sl_price is not None:
        stop_distance = abs(entry_price - sl_price)
    if entry_price is not None and tp_price is not None:
        target_distance = abs(tp_price - entry_price)

    observation_id = _trade_uid(trade)
    return_pct = _calc_return_pct(trade)
    r_multiple = _calc_r_multiple(trade)
    confidence = _safe_float(trade.get("signal_confidence"))
    evidence = zeus.get("evidence_score") if zeus.get("evidence_score") is not None else _safe_float(trade.get("evidence_score"))

    return {
        "observation_id": observation_id,
        "generated_at": generated_at,
        "source_system": str(trade.get("source_system") or SourceSystem.HERMES.value),
        "trade_id": str(trade.get("trade_id") or ""),
        "signal_id": str(trade.get("signal_id") or ""),
        "market_context": {
            "session": str(trade.get("session") or "unknown"),
            "day_of_week": _day_of_week(entry_ts),
            "volatility_regime": str(trade.get("volatility_state") or trade.get("regime") or "unknown"),
            "trend_range_classification": str(trade.get("trend_state") or "unknown"),
            "liquidity_conditions": str(trade.get("liquidity_conditions") or "unknown"),
            "higher_timeframe_bias": str(trade.get("htf_bias") or "unknown"),
            "time_to_major_news": str(trade.get("time_to_major_news") or "unknown"),
        },
        "structure_context": {
            "bos_sequence": str(trade.get("bos_sequence") or "unknown"),
            "choch_sequence": str(trade.get("choch_sequence") or "unknown"),
            "liquidity_sweep_type": str(trade.get("liquidity_sweep_type") or "unknown"),
            "order_block_quality": str(trade.get("order_block_quality") or "unknown"),
            "fvg_quality": str(trade.get("fvg_quality") or "unknown"),
            "displacement_strength": _safe_float(trade.get("displacement_strength")),
            "retracement_depth": _safe_float(trade.get("retracement_depth")),
        },
        "execution_context": {
            "entry_rationale": str(trade.get("entry_rationale") or "unknown"),
            "confirmation_chain": str(trade.get("confirmation_chain") or "unknown"),
            "confidence_score": confidence,
            "spread": _safe_float(trade.get("spread_at_entry") or trade.get("spread")),
            "slippage": _safe_float(trade.get("slippage")),
            "latency": _safe_float(trade.get("latency")),
            "position_size": _safe_float(trade.get("lots") or trade.get("size")),
            "risk_pct": _safe_float(trade.get("risk_pct")),
            "stop_distance": stop_distance,
            "target_distance": target_distance,
        },
        "trade_lifecycle_context": {
            "entry_timestamp": entry_ts,
            "exit_timestamp": exit_ts,
            "trade_duration_seconds": _duration_seconds(entry_ts, exit_ts, trade.get("hold_seconds")),
            "mfe": _safe_float(trade.get("mfe") or trade.get("mfe_pct")),
            "mae": _safe_float(trade.get("mae") or trade.get("mae_pct")),
            "partial_exits": str(trade.get("partial_exits") or "unknown"),
            "trailing_behaviour": str(trade.get("trailing_behaviour") or "unknown"),
            "exit_classification": str(trade.get("exit_reason") or "unknown"),
        },
        "outcome_context": {
            "r_multiple": r_multiple,
            "return_pct": return_pct,
            "expectancy_contribution": _safe_float(trade.get("pnl")),
            "drawdown_contribution": _safe_float(trade.get("drawdown_contribution")),
            "capital_utilization": _safe_float(trade.get("capital_utilization")),
            "strategy_equity_impact": _safe_float(trade.get("strategy_equity_impact") or trade.get("pnl")),
        },
        "learning_context": {
            "hypothesis_id": str(trade.get("hypothesis_id") or ""),
            "research_proposal_id": str(trade.get("research_proposal_id") or ""),
            "validation_status": str(zeus.get("validation_status") or "pending"),
            "knowledge_object_id": str(zeus.get("knowledge_object_id") or ""),
            "counterfactual_result": str(zeus.get("counterfactual_result") or "unknown"),
            "replay_result": str(zeus.get("replay_result") or "unknown"),
            "zeus_decision": str(zeus.get("zeus_decision") or "pending"),
            "evidence_score": evidence,
            "confidence_score": zeus.get("confidence_score") if zeus.get("confidence_score") is not None else confidence,
        },
    }


def _required_values(row: dict[str, Any]) -> list[Any]:
    return [
        row.get("market_context", {}).get("session"),
        row.get("market_context", {}).get("day_of_week"),
        row.get("market_context", {}).get("volatility_regime"),
        row.get("market_context", {}).get("trend_range_classification"),
        row.get("structure_context", {}).get("bos_sequence"),
        row.get("structure_context", {}).get("choch_sequence"),
        row.get("execution_context", {}).get("confidence_score"),
        row.get("execution_context", {}).get("position_size"),
        row.get("execution_context", {}).get("stop_distance"),
        row.get("trade_lifecycle_context", {}).get("entry_timestamp"),
        row.get("trade_lifecycle_context", {}).get("exit_timestamp"),
        row.get("trade_lifecycle_context", {}).get("exit_classification"),
        row.get("outcome_context", {}).get("r_multiple"),
        row.get("outcome_context", {}).get("return_pct"),
        row.get("learning_context", {}).get("validation_status"),
        row.get("learning_context", {}).get("zeus_decision"),
    ]


def _dataset_quality(rows: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    if not rows:
        metrics = {
            "completeness": 0.0,
            "coverage": 0.0,
            "label_quality": 0.0,
            "evidence_quality": 0.0,
            "unknown_classification_rate": 1.0,
            "duplicate_observation_rate": 0.0,
            "validation_coverage": 0.0,
            "historical_depth": 0.0,
            "statistical_significance": 0.0,
            "data_freshness": 0.0,
        }
        return {
            "generated_at": generated_at,
            "version": DATASET_VERSION,
            "dataset_quality_score": 0.0,
            "metrics": metrics,
        }

    required_total = 0
    required_ok = 0
    label_ok = 0
    evidence_scores: list[float] = []
    validation_ok = 0
    unknown_classifications = 0
    total_classifications = 0

    seen_ids: set[str] = set()
    dupes = 0

    sessions: set[str] = set()
    regimes: set[str] = set()
    days: set[str] = set()

    latest_exit: datetime | None = None

    for row in rows:
        rid = str(row.get("observation_id") or "")
        if rid in seen_ids:
            dupes += 1
        seen_ids.add(rid)

        req = _required_values(row)
        required_total += len(req)
        required_ok += sum(0 if _unknown(v) else 1 for v in req)

        exit_class = row.get("trade_lifecycle_context", {}).get("exit_classification")
        r_mult = row.get("outcome_context", {}).get("r_multiple")
        ret = row.get("outcome_context", {}).get("return_pct")
        if (not _unknown(exit_class)) and (r_mult is not None) and (ret is not None):
            label_ok += 1

        ev = _safe_float(row.get("learning_context", {}).get("evidence_score"))
        conf = _safe_float(row.get("learning_context", {}).get("confidence_score"))
        if ev is not None:
            evidence_scores.append(max(0.0, min(1.0, ev)))
        elif conf is not None:
            evidence_scores.append(max(0.0, min(1.0, conf)))

        v_status = str(row.get("learning_context", {}).get("validation_status") or "").lower()
        z_dec = str(row.get("learning_context", {}).get("zeus_decision") or "").lower()
        if v_status in _VALIDATED_STATUSES or z_dec in _VALIDATED_STATUSES:
            validation_ok += 1

        total_classifications += 1
        if _unknown(exit_class):
            unknown_classifications += 1

        session = str(row.get("market_context", {}).get("session") or "unknown")
        regime = str(row.get("market_context", {}).get("volatility_regime") or "unknown")
        day = str(row.get("market_context", {}).get("day_of_week") or "unknown")
        if not _unknown(session):
            sessions.add(session)
        if not _unknown(regime):
            regimes.add(regime)
        if not _unknown(day):
            days.add(day)

        exit_ts = _safe_iso(row.get("trade_lifecycle_context", {}).get("exit_timestamp"))
        if exit_ts is not None and (latest_exit is None or exit_ts > latest_exit):
            latest_exit = exit_ts

    completeness = required_ok / max(1, required_total)
    coverage = mean([
        min(1.0, len(sessions) / 6.0),
        min(1.0, len(regimes) / 6.0),
        min(1.0, len(days) / 7.0),
    ])
    label_quality = label_ok / max(1, len(rows))
    evidence_quality = mean(evidence_scores) if evidence_scores else 0.0
    unknown_rate = unknown_classifications / max(1, total_classifications)
    duplicate_rate = dupes / max(1, len(rows))
    validation_coverage = validation_ok / max(1, len(rows))
    historical_depth = min(1.0, len(rows) / 300.0)
    statistical_significance = min(1.0, validation_ok / 120.0)

    freshness = 0.0
    if latest_exit is not None:
        age_hours = max(0.0, (_utc_now() - latest_exit).total_seconds() / 3600.0)
        if age_hours <= 24:
            freshness = 1.0
        elif age_hours <= 72:
            freshness = 0.8
        elif age_hours <= 168:
            freshness = 0.6
        elif age_hours <= 336:
            freshness = 0.4
        else:
            freshness = 0.2

    quality_components = [
        completeness,
        coverage,
        label_quality,
        evidence_quality,
        1.0 - unknown_rate,
        1.0 - duplicate_rate,
        validation_coverage,
        historical_depth,
        statistical_significance,
        freshness,
    ]
    overall = max(0.0, min(1.0, mean(quality_components)))

    metrics = {
        "completeness": round(completeness, 4),
        "coverage": round(coverage, 4),
        "label_quality": round(label_quality, 4),
        "evidence_quality": round(evidence_quality, 4),
        "unknown_classification_rate": round(unknown_rate, 4),
        "duplicate_observation_rate": round(duplicate_rate, 4),
        "validation_coverage": round(validation_coverage, 4),
        "historical_depth": round(historical_depth, 4),
        "statistical_significance": round(statistical_significance, 4),
        "data_freshness": round(freshness, 4),
    }

    return {
        "generated_at": generated_at,
        "version": DATASET_VERSION,
        "dataset_quality_score": round(overall * 100.0, 2),
        "metrics": metrics,
    }


def _summarize_board(board_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    by_domain: dict[str, int] = defaultdict(int)
    confs: list[float] = []
    evidences: list[float] = []

    for row in rows:
        status = str(row.get("status") or "pending").lower()
        domain = str(row.get("domain") or "unknown").lower()
        by_status[status] += 1
        by_domain[domain] += 1
        c = _safe_float(row.get("confidence"))
        if c is not None:
            confs.append(c)
        e = _safe_float(row.get("evidence_score"))
        if e is not None:
            evidences.append(e)

    total = len(rows)
    passed = by_status.get("passed", 0) + by_status.get("approved", 0)
    approved = sum(1 for row in rows if bool(row.get("approved_for_adoption", False)))

    return {
        "board": board_name,
        "total_reports": total,
        "approved_reports": approved,
        "pass_rate": round(passed / max(1, total), 4),
        "average_confidence": round(mean(confs), 4) if confs else STATUS_AWAITING,
        "average_evidence_score": round(mean(evidences), 4) if evidences else STATUS_AWAITING,
        "by_status": dict(by_status),
        "by_domain": dict(by_domain),
        "ledger": [
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "source_system": str(row.get("source_system") or row.get("candidate_source_system") or ""),
                "domain": str(row.get("domain") or "unknown"),
                "status": str(row.get("status") or "pending"),
                "confidence": _safe_float(row.get("confidence")),
                "evidence_score": _safe_float(row.get("evidence_score")),
                "operator_approval_status": str(row.get("operator_approval_status") or "Pending Operator Review"),
                "timestamp": str(row.get("timestamp") or ""),
            }
            for row in rows[:200]
        ],
    }


def _split_zeus_boards(reports: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    pattern_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []

    for row in reports:
        src = str(row.get("source_system") or row.get("candidate_source_system") or "").lower()
        domain = str(row.get("domain") or row.get("validation_domain") or "").lower()
        if src == SourceSystem.HERMES.value or domain == "pattern":
            pattern_rows.append(row)
        else:
            strategy_rows.append(row)

    return {
        "version": DATASET_VERSION,
        "generated_at": generated_at,
        "zeus_validation_boards": {
            "pattern_validation_board": {
                "responsibilities": [
                    "pattern_validation",
                    "statistical_significance",
                    "counterfactual_validation",
                    "replay_validation",
                    "regime_robustness",
                    "pattern_stability",
                    "knowledge_approval",
                ],
                **_summarize_board("pattern_validation_board", pattern_rows),
            },
            "strategy_validation_board": {
                "responsibilities": [
                    "strategy_validation",
                    "position_sizing_validation",
                    "drawdown_validation",
                    "risk_analysis",
                    "expectancy_improvement",
                    "portfolio_impact",
                    "monte_carlo_validation",
                    "execution_approval",
                ],
                **_summarize_board("strategy_validation_board", strategy_rows),
            },
        },
    }


def _knowledge_promotion(rows: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    raw_count = len(rows)
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("learning_context", {}).get("validation_status") or "").lower()
        decision = str(row.get("learning_context", {}).get("zeus_decision") or "").lower()
        if status in _VALIDATED_STATUSES or decision in _VALIDATED_STATUSES:
            validated_rows.append(row)

    knowledge_objects = []
    for row in validated_rows:
        lc = row.get("learning_context", {}) or {}
        kid = str(lc.get("knowledge_object_id") or row.get("observation_id") or "")
        knowledge_objects.append(
            {
                "knowledge_object_id": kid,
                "observation_id": row.get("observation_id"),
                "zeus_decision": lc.get("zeus_decision"),
                "validation_status": lc.get("validation_status"),
                "evidence_score": lc.get("evidence_score"),
                "confidence_score": lc.get("confidence_score"),
                "promoted_at": generated_at,
            }
        )

    return {
        "version": DATASET_VERSION,
        "generated_at": generated_at,
        "raw_observations": {
            "count": raw_count,
            "source": "hermes_and_prometheus_trade_observations",
        },
        "validated_institutional_knowledge": {
            "count": len(validated_rows),
            "promotion_rate": round(len(validated_rows) / max(1, raw_count), 4),
            "knowledge_objects": knowledge_objects[:500],
        },
        "promotion_policy": {
            "raw_observations_promoted_automatically": False,
            "requires_zeus_approval": True,
            "execution_behavior_changed": False,
        },
    }


def build_institutional_dataset_architecture(
    root_dir: Path,
    *,
    status: dict[str, Any],
    open_trades: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    feature_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    generated_at = _utc_iso()
    reports = _zeus_reports(root_dir)
    lookup = _zeus_lookup(reports)

    hermes_closed: list[dict[str, Any]] = []
    for row in closed_trades:
        if isinstance(row, dict):
            item = dict(row)
            item["source_system"] = SourceSystem.HERMES.value
            hermes_closed.append(item)

    prometheus_closed = _prometheus_closed_trades(root_dir)

    rows_file = root_dir / "storage" / "olympus" / "institutional_dataset_rows.jsonl"
    existing_rows = _load_jsonl(rows_file)
    known_ids = {str(row.get("observation_id") or "") for row in existing_rows}

    new_rows: list[dict[str, Any]] = []
    for trade in hermes_closed + prometheus_closed:
        zeus = _zeus_for_trade(trade, lookup)
        obs = _build_observation(trade, zeus, generated_at)
        oid = str(obs.get("observation_id") or "")
        if oid and oid not in known_ids:
            known_ids.add(oid)
            new_rows.append(obs)

    all_rows = existing_rows + new_rows
    quality = _dataset_quality(all_rows, generated_at)
    boards = _split_zeus_boards(reports, generated_at)
    promotion = _knowledge_promotion(all_rows, generated_at)

    return {
        "meta": {
            "version": DATASET_VERSION,
            "generated_at": generated_at,
            "observational_only": True,
            "execution_behavior_unchanged": True,
            "governed_by_zeus": True,
            "feature_flags": feature_flags or {},
        },
        "responsibility_contract": {
            "hermes": {
                "scope": "market_intelligence",
                "owns": [
                    "pattern_intelligence",
                    "market_structure_intelligence",
                    "liquidity_intelligence",
                    "session_intelligence",
                    "trade_lifecycle_intelligence",
                    "return_intelligence",
                    "decision_intelligence",
                    "market_context_intelligence",
                ],
                "execution_mutation_allowed": False,
            },
            "prometheus": {
                "scope": "strategy_intelligence",
                "owns": [
                    "strategy_performance",
                    "position_sizing",
                    "risk_management",
                    "drawdown_control",
                    "expectancy",
                    "execution_quality",
                    "strategy_adaptation",
                    "portfolio_growth",
                ],
                "auto_adoption_allowed": False,
            },
            "zeus": {
                "scope": "institutional_validation",
                "independent_governance": True,
            },
        },
        "dataset": {
            "total_observations": len(all_rows),
            "new_observations": len(new_rows),
            "open_trades_observed": len(open_trades),
            "recent_observations": all_rows[-200:],
        },
        "dataset_quality": quality,
        "zeus_validation_boards": boards.get("zeus_validation_boards", {}),
        "institutional_knowledge_base": promotion,
        "artifacts": {
            "new_rows": new_rows,
        },
    }


def write_institutional_dataset_architecture_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    rows_path = storage / "institutional_dataset_rows.jsonl"
    dataset_runtime = storage / "institutional_dataset_architecture_runtime.json"
    dataset_history = storage / "institutional_dataset_architecture_history.jsonl"
    boards_runtime = storage / "zeus_validation_boards_runtime.json"
    boards_history = storage / "zeus_validation_boards_history.jsonl"
    knowledge_runtime = storage / "institutional_knowledge_base_runtime.json"
    knowledge_history = storage / "institutional_knowledge_base_history.jsonl"

    new_rows = list(payload.get("artifacts", {}).get("new_rows", []) or [])
    _append_jsonl(rows_path, new_rows)

    dataset_payload = {
        "meta": payload.get("meta", {}),
        "responsibility_contract": payload.get("responsibility_contract", {}),
        "dataset": payload.get("dataset", {}),
        "dataset_quality": payload.get("dataset_quality", {}),
    }
    boards_payload = {
        "meta": payload.get("meta", {}),
        "zeus_validation_boards": payload.get("zeus_validation_boards", {}),
    }
    knowledge_payload = {
        "meta": payload.get("meta", {}),
        "institutional_knowledge_base": payload.get("institutional_knowledge_base", {}),
    }

    dataset_runtime.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    boards_runtime.write_text(json.dumps(boards_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    knowledge_runtime.write_text(json.dumps(knowledge_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    _append_jsonl(dataset_history, [dataset_payload])
    _append_jsonl(boards_history, [boards_payload])
    _append_jsonl(knowledge_history, [knowledge_payload])

    return {
        "institutional_dataset_rows": str(rows_path),
        "institutional_dataset_runtime": str(dataset_runtime),
        "institutional_dataset_history": str(dataset_history),
        "zeus_validation_boards_runtime": str(boards_runtime),
        "zeus_validation_boards_history": str(boards_history),
        "institutional_knowledge_base_runtime": str(knowledge_runtime),
        "institutional_knowledge_base_history": str(knowledge_history),
    }
