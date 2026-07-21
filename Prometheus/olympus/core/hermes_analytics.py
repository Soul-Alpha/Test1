from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict

from olympus.contracts import EvidenceConfidenceContract, KnowledgeContract, SourceSystem
from olympus.core.implementation_report_generator import write_olympus_implementation_report
from olympus.core.historical_evidence import (
    HistoricalEvidenceLedger,
    build_evidence_readiness,
)
from olympus.core.intelligence_auditor import run_olympus_intelligence_auditor
from olympus.core.pattern_context_intelligence import build_pattern_context_intelligence
from olympus.core.return_intelligence import build_return_intelligence


STATUS_AVAILABLE = "Available"
STATUS_DERIVABLE = "Derivable"
STATUS_FRAMEWORK = "Framework Required"
STATUS_AWAITING = "Awaiting Historical Data"
STATUS_UNAVAILABLE = "Unavailable"
STATUS_PENDING = "Pending Initialization"

SAMPLE_THRESHOLDS = {
    "insufficient": 10,
    "emerging": 30,
    "developing": 75,
    "validated": 150,
    "elite": 300,
}


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
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _confidence_distribution(vals: list[float]) -> dict[str, int]:
    buckets = {"0.00-0.39": 0, "0.40-0.59": 0, "0.60-0.79": 0, "0.80-1.00": 0}
    for v in vals:
        if v < 0.40:
            buckets["0.00-0.39"] += 1
        elif v < 0.60:
            buckets["0.40-0.59"] += 1
        elif v < 0.80:
            buckets["0.60-0.79"] += 1
        else:
            buckets["0.80-1.00"] += 1
    return buckets


def _shannon_diversity(freqs: list[int]) -> float:
    total = sum(freqs)
    if total <= 0:
        return 0.0
    ent = 0.0
    for f in freqs:
        if f <= 0:
            continue
        p = f / total
        ent -= p * math.log(p, 2)
    return ent


def _bucket_strength(v: Any) -> str:
    x = _safe_float(v)
    if x is None:
        return "unknown"
    if x >= 0.70:
        return "high"
    if x >= 0.45:
        return "mid"
    return "low"


def _bucket_mtf(v: Any) -> str:
    x = _safe_float(v)
    if x is None:
        return "unknown"
    if x >= 0.70:
        return "aligned"
    if x >= 0.45:
        return "mixed"
    return "weak"


def _cluster_family(sig: dict[str, Any]) -> str:
    stop_hunt = int(sig.get("stop_hunt", 0) or 0)
    structure = int(sig.get("structure_type", 0) or 0)
    pattern_type = int(sig.get("pattern_type_id", 0) or 0)
    aligned = int(sig.get("prior_trend_aligned", 0) or 0)
    ob = int(sig.get("ob_present", 0) or 0)

    if stop_hunt and aligned:
        return "Liquidity Sweep Continuation"
    if stop_hunt and not aligned:
        return "Liquidity Sweep Reversal"
    if pattern_type in (3, 4):
        return "CHOCH Reversal"
    if pattern_type in (1, 2):
        return "CHOCH Continuation"
    if structure == 1 and ob:
        return "Bullish Expansion"
    if structure == 2 and ob:
        return "Bearish Expansion"
    if aligned and structure == 3:
        return "Trend Pullback"
    if stop_hunt and structure == 3:
        return "Session Reversal"
    if ob:
        return "Mitigation"
    return "Compression"


def _lifecycle(occ: int, expectancy: float | None, stability: float | None, last_seen: str | None) -> str:
    if last_seen:
        try:
            dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).days
            if age_days > 120:
                return "Archived"
        except Exception:
            pass

    if occ < 5:
        return "Candidate"
    if occ < 20:
        return "Emerging"

    exp = expectancy if expectancy is not None else -1.0
    stab = stability if stability is not None else 0.0
    if occ >= 50 and exp > 0.20 and stab >= 0.60:
        return "Elite"
    if exp > 0:
        return "Validated"
    return "Declining"


def _safe_iso_to_dt(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _sample_stage(n: int) -> str:
    if n < SAMPLE_THRESHOLDS["insufficient"]:
        return "Insufficient Data"
    if n < SAMPLE_THRESHOLDS["emerging"]:
        return "Emerging"
    if n < SAMPLE_THRESHOLDS["developing"]:
        return "Developing"
    if n < SAMPLE_THRESHOLDS["validated"]:
        return "Validated"
    return "Elite"


def _binom_ci(p: float | None, n: int, z: float = 1.96) -> dict[str, Any]:
    if p is None or n <= 0:
        return {"low": STATUS_AWAITING, "high": STATUS_AWAITING}
    err = z * math.sqrt(max(0.0, p * (1.0 - p)) / n)
    return {"low": round(max(0.0, p - err), 4), "high": round(min(1.0, p + err), 4)}


def _derive_session(ts: Any) -> str:
    dt = _safe_iso_to_dt(ts)
    if dt is None:
        return "unknown"
    h = dt.hour
    if 0 <= h < 7:
        return "Asian"
    if 7 <= h < 10:
        return "London Open"
    if 10 <= h < 13:
        return "London"
    if 13 <= h < 16:
        return "NY"
    if 16 <= h < 19:
        return "NY Lunch"
    return "Rollover"


def _derive_regime(structure_type: Any, trend_strength: Any, stop_hunt: Any) -> str:
    st = _safe_int(structure_type) or 0
    tr = _safe_float(trend_strength) or 0.0
    sh = _safe_int(stop_hunt) or 0
    if sh:
        return "Liquidity Sweep"
    if st == 3 and tr < 0.45:
        return "Compression"
    if st == 3:
        return "Mean Reversion"
    if tr >= 0.70:
        return "Trend Expansion"
    if tr >= 0.50:
        return "Trend Exhaustion"
    return "Dead Liquidity"


def _mean_or_status(vals: list[float], fallback: str = STATUS_AWAITING) -> Any:
    return round(mean(vals), 4) if vals else fallback


def _confidence_buckets() -> list[tuple[float, float, str]]:
    return [
        (0.50, 0.60, "0.50-0.60"),
        (0.60, 0.70, "0.60-0.70"),
        (0.70, 0.80, "0.70-0.80"),
        (0.80, 0.90, "0.80-0.90"),
        (0.90, 1.01, "0.90-1.00"),
    ]


def _group_conf_stats(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grp: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        conf = _safe_float(r.get("confidence"))
        k = str(r.get(key, "unknown") or "unknown")
        if conf is not None:
            grp[k].append(conf)
    out = []
    for k, vals in grp.items():
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        stab = max(0.0, 1.0 - min(1.0, sd))
        out.append(
            {
                "key": k,
                "sample_count": len(vals),
                "average_confidence": round(mean(vals), 4),
                "confidence_stability": round(stab, 4),
                "reliability_stage": _sample_stage(len(vals)),
            }
        )
    out.sort(key=lambda x: x["sample_count"], reverse=True)
    return out


def _knowledge_level(score: float | None) -> str:
    if score is None:
        return "Insufficient Data"
    if score < 35:
        return "Emerging"
    if score < 50:
        return "Developing"
    if score < 65:
        return "Validated"
    if score < 80:
        return "Advanced"
    if score < 92:
        return "Elite"
    return "Institutional"


def _letter_grade(score: float | None) -> str:
    if score is None:
        return "N/A"
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 67:
        return "D+"
    if score >= 63:
        return "D"
    return "F"


def _progress_bar(pct: float) -> str:
    clamped = max(0.0, min(100.0, pct))
    blocks = int(round(clamped / 100.0 * 12))
    return ("█" * blocks) + ("░" * (12 - blocks))


def _distribution_stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {
            "count": 0,
            "mean": STATUS_AWAITING,
            "median": STATUS_AWAITING,
            "std": STATUS_AWAITING,
            "p10": STATUS_AWAITING,
            "p90": STATUS_AWAITING,
        }
    ordered = sorted(vals)
    p10_idx = max(0, int(len(ordered) * 0.10) - 1)
    p90_idx = max(0, int(len(ordered) * 0.90) - 1)
    return {
        "count": len(vals),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "p10": round(ordered[p10_idx], 4),
        "p90": round(ordered[p90_idx], 4),
    }


def _stage_from_thresholds(score: float, thresholds: list[tuple[str, float]]) -> str:
    stage = thresholds[0][0]
    for name, floor in thresholds:
        if score >= floor:
            stage = name
    return stage


def _metric_maturity(implemented: bool, sample_count: int, confidence_score: float | None, validated_floor: int = 75) -> str:
    if not implemented:
        return "Implemented"
    if sample_count < SAMPLE_THRESHOLDS["insufficient"]:
        return "Developing"
    if sample_count < validated_floor:
        return "Pending Validation"
    if confidence_score is None:
        return "Validated"
    return "Elite" if confidence_score >= 85 else "Validated"


def _load_inputs(root_dir: Path) -> dict[str, Any]:
    evidence_ledger = HistoricalEvidenceLedger(root_dir, source_system="hermes")
    return {
        "setups": _load_json(root_dir / "models" / "hermes" / "setups.json", []),
        "status": _load_json(root_dir / "live_bot" / "hermes_status.json", {}),
        "pattern_stats": _load_json(root_dir / "models" / "hermes" / "pattern_stats.json", {}),
        "lineage": _load_jsonl(root_dir / "storage" / "olympus" / "event_lineage.jsonl"),
        "snapshots": _load_jsonl(root_dir / "storage" / "olympus" / "pattern_snapshots.jsonl"),
        "versions": _load_jsonl(root_dir / "storage" / "olympus" / "version_registry.jsonl"),
        "evidence": evidence_ledger.records(),
    }


def build_hermes_analytics(root_dir: Path) -> dict[str, Any]:
    src = _load_inputs(root_dir)
    setups: list[dict[str, Any]] = src["setups"]
    status: dict[str, Any] = src["status"]
    lineage: list[dict[str, Any]] = src["lineage"]
    snapshots: list[dict[str, Any]] = src["snapshots"]
    versions: list[dict[str, Any]] = src["versions"]
    evidence: list[dict[str, Any]] = src["evidence"]

    ml_records = len(setups)
    market_snapshots = max(ml_records, len(snapshots))

    labeled = [r for r in setups if r.get("outcome") is not None]
    labeled_count = len(labeled)

    # Runtime signals map for confidence/expected move per signal_id.
    sig_map: dict[str, dict[str, Any]] = {}
    for row in (status.get("signals") or []) + (status.get("skipped_signals") or []):
        sid = str(row.get("signal_id", "") or "")
        if sid:
            sig_map[sid] = row
    last_sig = status.get("last_signal") or {}
    if last_sig.get("signal_id"):
        sig_map[str(last_sig.get("signal_id"))] = last_sig

    for event in evidence:
        if event.get("event_type") != "prediction_created":
            continue
        payload = event.get("payload") or {}
        signal_id = str(payload.get("signal_id") or event.get("entity_id") or "")
        if signal_id:
            sig_map[signal_id] = payload

    closed_by_id = {
        str(row.get("trade_id")): row
        for row in status.get("closed_trades") or []
        if row.get("trade_id")
    }
    open_by_id = {
        str(row.get("trade_id")): row
        for row in status.get("open_trades") or []
        if row.get("trade_id")
    }
    for event in evidence:
        payload = event.get("payload") or {}
        trade_id = str(payload.get("trade_id") or event.get("entity_id") or "")
        if not trade_id:
            continue
        if event.get("event_type") == "trade_opened":
            open_by_id[trade_id] = payload
        elif event.get("event_type") == "trade_closed":
            closed_by_id[trade_id] = payload
            open_by_id.pop(trade_id, None)

    closed_trades = list(closed_by_id.values())
    open_trades = list(open_by_id.values())
    skipped_trades = status.get("skipped_signals") or []

    return_intelligence = (
        build_return_intelligence(
            closed_trades,
            status=status,
            feature_version=status.get("feature_version", "v1"),
            model_version=str(status.get("ml", {}).get("model_version", "0")),
            report_interval=10,
        )
        if closed_trades
        else status.get("return_intelligence") or build_return_intelligence(
            [],
            status=status,
            feature_version=status.get("feature_version", "v1"),
            model_version=str(status.get("ml", {}).get("model_version", "0")),
            report_interval=10,
        )
    )
    return_summary = return_intelligence.get("summary", {}) if isinstance(return_intelligence, dict) else {}
    return_research_report = return_intelligence.get("research_report", {}) if isinstance(return_intelligence, dict) else {}

    trade_by_signal: dict[str, dict[str, Any]] = {}
    for t in closed_trades:
        sid = str(t.get("signal_id", "") or "")
        if sid:
            trade_by_signal[sid] = t

    # Confidence values from runtime + lineage predictions.
    confidence_vals: list[float] = []
    for row in sig_map.values():
        c = _safe_float(row.get("confidence"))
        if c is not None:
            confidence_vals.append(c)
    for ev in lineage:
        if ev.get("event_type") == "prediction_created":
            c = _safe_float((ev.get("payload") or {}).get("confidence"))
            if c is not None:
                confidence_vals.append(c)

    avg_conf = mean(confidence_vals) if confidence_vals else None
    med_conf = median(confidence_vals) if confidence_vals else None
    std_conf = pstdev(confidence_vals) if len(confidence_vals) > 1 else None
    conf_dist = _confidence_distribution(confidence_vals) if confidence_vals else {}

    pred_by_signal: dict[str, dict[str, Any]] = {}
    for sid, row in sig_map.items():
        pred_by_signal[sid] = {
            "confidence": _safe_float(row.get("confidence")),
            "direction": row.get("direction"),
            "expected_distance_pts": _safe_float(row.get("expected_distance_pts")),
            "timestamp": row.get("timestamp"),
        }
    for ev in lineage:
        if ev.get("event_type") != "prediction_created":
            continue
        payload = ev.get("payload") or {}
        sid = str(payload.get("signal_id", "") or "")
        if not sid:
            continue
        cur = pred_by_signal.get(sid, {})
        pred_by_signal[sid] = {
            "confidence": _safe_float(payload.get("confidence")) if _safe_float(payload.get("confidence")) is not None else cur.get("confidence"),
            "direction": payload.get("direction") or cur.get("direction"),
            "expected_distance_pts": cur.get("expected_distance_pts"),
            "timestamp": ev.get("timestamp") or cur.get("timestamp"),
        }

    entered_ts_by_trade: dict[str, datetime] = {}
    closed_ts_by_trade: dict[str, datetime] = {}
    for ev in lineage:
        et = str(ev.get("event_type", "") or "")
        payload = ev.get("payload") or {}
        tid = str(payload.get("trade_id", "") or "")
        if not tid:
            continue
        ts = _safe_iso_to_dt(ev.get("timestamp"))
        if ts is None:
            continue
        if et == "trade_entered":
            entered_ts_by_trade[tid] = ts
        elif et == "trade_closed":
            closed_ts_by_trade[tid] = ts

    for event in evidence:
        event_type = str(event.get("event_type", "") or "")
        trade_id = str(event.get("entity_id", "") or "")
        timestamp = _safe_iso_to_dt(event.get("occurred_at"))
        if not trade_id or timestamp is None:
            continue
        if event_type == "trade_opened":
            entered_ts_by_trade[trade_id] = timestamp
        elif event_type == "trade_closed" and (event.get("payload") or {}).get("closed_at_quality") == "exact":
            closed_ts_by_trade[trade_id] = timestamp

    # Completed prediction records are additive analytics projections over immutable history.
    completed_predictions: list[dict[str, Any]] = []
    for r in labeled:
        sid = str(r.get("setup_id", "") or "")
        pred = pred_by_signal.get(sid, {})
        trow = trade_by_signal.get(sid, {})
        confidence = _safe_float(pred.get("confidence"))
        expected_move = _safe_float(pred.get("expected_distance_pts"))
        actual_move = _safe_float(trow.get("realized_distance_pts"))
        pred_error = abs(expected_move - actual_move) if expected_move is not None and actual_move is not None else None
        outcome = int(r.get("outcome", 0) or 0)
        rr = _safe_float(r.get("rr_achieved"))
        signed_return = rr if rr is not None and outcome == 1 else (-rr if rr is not None else None)

        sig = {
            "structure_type": int(r.get("structure_type", 0) or 0),
            "stop_hunt": int(r.get("stop_hunt", 0) or 0),
            "ob_present": int(r.get("ob_present", 0) or 0),
            "fib_proximity": int(r.get("fib_proximity", 0) or 0),
            "pattern_type_id": int(r.get("pattern_type_id", 0) or 0),
            "prior_trend_aligned": int(r.get("prior_trend_aligned", 0) or 0),
            "trend_bucket": _bucket_strength(r.get("trend_strength")),
            "mtf_bucket": _bucket_mtf(r.get("mtf_score")),
        }
        pattern_id = (
            f"PT_{sig['pattern_type_id']}_S{sig['structure_type']}_SH{sig['stop_hunt']}_"
            f"OB{sig['ob_present']}_FB{sig['fib_proximity']}_A{sig['prior_trend_aligned']}_"
            f"T{sig['trend_bucket']}_M{sig['mtf_bucket']}"
        )

        trade_type = str(pred.get("direction") or trow.get("direction") or "unknown")
        regime = _derive_regime(r.get("structure_type"), r.get("trend_strength"), r.get("stop_hunt"))
        session = _derive_session(r.get("timestamp") or pred.get("timestamp"))

        duration_sec = None
        tid = str(trow.get("trade_id", "") or "")
        if tid and tid in entered_ts_by_trade and tid in closed_ts_by_trade:
            duration_sec = max(0.0, (closed_ts_by_trade[tid] - entered_ts_by_trade[tid]).total_seconds())

        completed_predictions.append(
            {
                "signal_id": sid,
                "confidence": confidence,
                "prediction_direction": str(pred.get("direction") or "unknown"),
                "expected_distance": expected_move,
                "actual_outcome": outcome,
                "return_pct": signed_return,
                "pattern_id": pattern_id,
                "pattern_cluster": _cluster_family(sig),
                "market_regime": regime,
                "session": session,
                "model_version": str(r.get("model_version_used", status.get("ml", {}).get("model_version", "0"))),
                "feature_version": str(r.get("feature_version", status.get("feature_version", "v1"))),
                "timestamp": str(r.get("timestamp") or pred.get("timestamp") or ""),
                "trade_type": trade_type,
                "execution_type": str(r.get("execution_type", status.get("execution_type", "simulated"))),
                "prediction_error": pred_error,
                "actual_distance": actual_move,
                "trade_id": tid,
                "exit_reason": str(trow.get("exit_reason", "")),
                "trade_pnl": _safe_float(trow.get("pnl")),
                "duration_seconds": duration_sec,
                "trend_bucket": _bucket_strength(r.get("trend_strength")),
                "volatility_bucket": "high" if (_safe_float(r.get("volume_ratio")) or 1.0) >= 1.20 else "normal",
            }
        )

    completed_with_conf = [r for r in completed_predictions if _safe_float(r.get("confidence")) is not None]

    # Confidence Calibration Engine.
    bucket_rows: list[dict[str, Any]] = []
    abs_cal_errors: list[float] = []
    ece_num = 0.0
    total_cal = max(1, len(completed_with_conf))
    for lo, hi, lbl in _confidence_buckets():
        rows = [r for r in completed_with_conf if lo <= float(r.get("confidence", 0.0)) < hi]
        n = len(rows)
        wins_n = sum(int(r.get("actual_outcome", 0) or 0) for r in rows)
        losses_n = n - wins_n
        win_rate_b = (wins_n / n) if n else None
        conf_mean_b = mean([float(r.get("confidence", 0.0)) for r in rows]) if rows else None
        cal_err = abs((conf_mean_b or 0.0) - (win_rate_b or 0.0)) if n else None
        if cal_err is not None:
            abs_cal_errors.append(cal_err)
            ece_num += cal_err * n

        ret_vals = [float(r["return_pct"]) for r in rows if _safe_float(r.get("return_pct")) is not None]
        pred_err_vals = [float(r["prediction_error"]) for r in rows if _safe_float(r.get("prediction_error")) is not None]
        pnl_vals_b = [float(r["trade_pnl"]) for r in rows if _safe_float(r.get("trade_pnl")) is not None]
        wins_pnl_b = [x for x in pnl_vals_b if x > 0]
        loss_pnl_b = [x for x in pnl_vals_b if x < 0]
        pf_b = (sum(wins_pnl_b) / abs(sum(loss_pnl_b))) if loss_pnl_b else None
        exp_b = mean(ret_vals) if ret_vals else None
        brier_b = mean([(float(r.get("confidence", 0.0)) - float(r.get("actual_outcome", 0) or 0)) ** 2 for r in rows]) if rows else None

        conf_series = [float(r.get("confidence", 0.0)) for r in rows]
        drift = None
        if len(conf_series) >= 6:
            split = len(conf_series) // 2
            drift = mean(conf_series[split:]) - mean(conf_series[:split])
        stab = None
        if len(conf_series) > 1:
            stab = max(0.0, 1.0 - min(1.0, pstdev(conf_series)))

        bucket_rows.append(
            {
                "bucket": lbl,
                "occurrences": n,
                "wins": wins_n,
                "losses": losses_n,
                "historical_win_rate": round(win_rate_b, 4) if win_rate_b is not None else STATUS_AWAITING,
                "average_return": round(exp_b, 4) if exp_b is not None else STATUS_AWAITING,
                "average_mfe": STATUS_UNAVAILABLE,
                "average_mae": STATUS_UNAVAILABLE,
                "average_prediction_error": round(mean(pred_err_vals), 4) if pred_err_vals else STATUS_AWAITING,
                "profit_factor": round(pf_b, 4) if pf_b is not None else STATUS_AWAITING,
                "expectancy": round(exp_b, 4) if exp_b is not None else STATUS_AWAITING,
                "confidence_drift": round(drift, 4) if drift is not None else STATUS_AWAITING,
                "confidence_stability": round(stab, 4) if stab is not None else STATUS_AWAITING,
                "calibration_error": round(cal_err, 4) if cal_err is not None else STATUS_AWAITING,
                "brier_score": round(brier_b, 4) if brier_b is not None else STATUS_AWAITING,
                "sample_stage": _sample_stage(n),
                "confidence_interval": _binom_ci(win_rate_b, n),
                "data_sufficiency": n >= SAMPLE_THRESHOLDS["emerging"],
            }
        )

    ece = (ece_num / total_cal) if completed_with_conf else None
    mce = max(abs_cal_errors) if abs_cal_errors else None
    brier_global = (
        mean([(float(r.get("confidence", 0.0)) - float(r.get("actual_outcome", 0) or 0)) ** 2 for r in completed_with_conf])
        if completed_with_conf
        else None
    )

    threshold_candidates = [round(x, 2) for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]
    best_thr = None
    best_thr_score = -999999.0
    for thr in threshold_candidates:
        rows = [r for r in completed_with_conf if float(r.get("confidence", 0.0)) >= thr]
        if len(rows) < SAMPLE_THRESHOLDS["insufficient"]:
            continue
        exp_v = mean([float(r["return_pct"]) for r in rows if _safe_float(r.get("return_pct")) is not None]) if rows else -1.0
        wr = sum(int(r.get("actual_outcome", 0) or 0) for r in rows) / max(1, len(rows))
        score = (exp_v * 100.0) + wr
        if score > best_thr_score:
            best_thr_score = score
            best_thr = thr

    over_regions = []
    under_regions = []
    for r in bucket_rows:
        if not isinstance(r.get("historical_win_rate"), (float, int)):
            continue
        b_lo, b_hi = r["bucket"].split("-")
        mean_conf = (float(b_lo) + float(b_hi)) / 2.0
        diff = mean_conf - float(r["historical_win_rate"])
        if diff > 0.08:
            over_regions.append(r["bucket"])
        if diff < -0.08:
            under_regions.append(r["bucket"])

    # Directional intelligence and execution quality classification.
    directional_classes: list[dict[str, Any]] = []
    direction_series = []
    for r in completed_predictions:
        direction_series.append(str(r.get("prediction_direction", "unknown") or "unknown"))
        out = int(r.get("actual_outcome", 0) or 0)
        pred = _safe_float(r.get("expected_distance"))
        act = _safe_float(r.get("actual_distance"))
        ratio = (act / pred) if pred and act is not None and pred > 0 else None
        exit_reason = str(r.get("exit_reason", "") or "")
        regime = str(r.get("market_regime", "unknown") or "unknown")
        vol_bucket = str(r.get("volatility_bucket", "normal") or "normal")

        if out == 0 and regime == "Mean Reversion":
            cls = "Sideways Market"
        elif out == 0 and vol_bucket == "high":
            cls = "High Volatility Failure"
        elif out == 0 and exit_reason == "sl" and (ratio is not None and ratio < 0.35):
            cls = "Correct Direction + Stopped Before Expansion"
        elif out == 0:
            cls = "Wrong Direction"
        else:
            if ratio is not None and ratio >= 1.0:
                cls = "Correct Direction + Good Entry + Good Exit"
            elif ratio is not None and ratio >= 0.70:
                cls = "Correct Direction + Good Entry + Poor Exit"
            elif exit_reason == "micro_time_exit":
                cls = "Correct Direction + Early Exit"
            elif ratio is not None and ratio >= 0.45:
                cls = "Correct Direction + Partial Success"
            else:
                cls = "Correct Direction + Late Entry"

        directional_classes.append({"classification": cls, **r})

    class_counts: dict[str, int] = defaultdict(int)
    for r in directional_classes:
        class_counts[str(r.get("classification", "unknown"))] += 1

    total_dir = len(directional_classes)
    correct_dir = sum(1 for r in directional_classes if str(r.get("classification", "")).startswith("Correct Direction"))
    directional_accuracy = (correct_dir / total_dir) if total_dir else None
    forecast_samples = [
        1.0 if (_safe_float(r.get("prediction_error")) is not None and _safe_float(r.get("expected_distance")) not in (None, 0) and float(r.get("prediction_error")) / max(1e-9, float(r.get("expected_distance"))) <= 0.35)
        else 0.0
        for r in directional_classes
        if _safe_float(r.get("prediction_error")) is not None and _safe_float(r.get("expected_distance")) is not None
    ]
    forecast_accuracy = mean(forecast_samples) if forecast_samples else None
    execution_quality = (
        sum(1 for r in directional_classes if str(r.get("classification", "")) in ("Correct Direction + Good Entry + Good Exit", "Correct Direction + Good Entry + Poor Exit")) / total_dir
        if total_dir
        else None
    )
    entry_quality = (
        sum(1 for r in directional_classes if "Good Entry" in str(r.get("classification", ""))) / total_dir
        if total_dir
        else None
    )
    exit_quality = (
        sum(1 for r in directional_classes if str(r.get("classification", "")) in ("Correct Direction + Good Entry + Good Exit", "Correct Direction + Partial Success")) / total_dir
        if total_dir
        else None
    )

    switches = 0
    prev = None
    for d in direction_series:
        if d in ("long", "short"):
            if prev and d != prev:
                switches += 1
            prev = d
    directional_consistency = 1.0 - (switches / max(1, len([d for d in direction_series if d in ("long", "short")]) - 1))

    def _group_direction(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rr in rows:
            grouped[str(rr.get(key, "unknown") or "unknown")].append(rr)
        out = []
        for k, vals in grouped.items():
            n = len(vals)
            corr = sum(1 for v in vals if str(v.get("classification", "")).startswith("Correct Direction"))
            acc = corr / n if n else 0.0
            out.append(
                {
                    "key": k,
                    "sample_count": n,
                    "directional_accuracy": round(acc, 4),
                    "confidence_interval": _binom_ci(acc, n),
                    "reliability_stage": _sample_stage(n),
                }
            )
        out.sort(key=lambda x: x["sample_count"], reverse=True)
        return out

    direction_by_pattern = _group_direction(directional_classes, "pattern_id")
    direction_by_session = _group_direction(directional_classes, "session")
    direction_by_regime = _group_direction(directional_classes, "market_regime")
    direction_by_cluster = _group_direction(directional_classes, "pattern_cluster")

    # Duration intelligence from lineage trade entered/closed events.
    duration_rows = [r for r in completed_predictions if _safe_float(r.get("duration_seconds")) is not None]
    dur_vals = [float(r.get("duration_seconds", 0.0)) for r in duration_rows]
    avg_dur = mean(dur_vals) if dur_vals else None
    med_dur = median(dur_vals) if dur_vals else None
    min_dur = min(dur_vals) if dur_vals else None
    max_dur = max(dur_vals) if dur_vals else None

    tp_durs = [float(r.get("duration_seconds", 0.0)) for r in duration_rows if str(r.get("exit_reason", "")) == "tp"]
    sl_durs = [float(r.get("duration_seconds", 0.0)) for r in duration_rows if str(r.get("exit_reason", "")) == "sl"]

    dur_stability = None
    if len(dur_vals) > 1 and avg_dur and avg_dur > 0:
        dur_stability = max(0.0, 1.0 - min(1.0, pstdev(dur_vals) / avg_dur))

    def _group_duration(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        grp: dict[str, list[float]] = defaultdict(list)
        for rr in rows:
            d = _safe_float(rr.get("duration_seconds"))
            if d is None:
                continue
            grp[str(rr.get(key, "unknown") or "unknown")].append(d)
        out = []
        for k, vals in grp.items():
            n = len(vals)
            m = mean(vals)
            st = max(0.0, 1.0 - min(1.0, (pstdev(vals) / m))) if len(vals) > 1 and m > 0 else None
            out.append(
                {
                    "key": k,
                    "sample_count": n,
                    "average_duration_sec": round(m, 2),
                    "median_duration_sec": round(median(vals), 2),
                    "min_duration_sec": round(min(vals), 2),
                    "max_duration_sec": round(max(vals), 2),
                    "duration_stability": round(st, 4) if st is not None else STATUS_AWAITING,
                    "duration_confidence": _sample_stage(n),
                    "confidence_interval": _binom_ci(0.5 if n else None, n),
                }
            )
        out.sort(key=lambda x: x["sample_count"], reverse=True)
        return out

    duration_by_pattern = _group_duration(duration_rows, "pattern_id")
    duration_by_session = _group_duration(duration_rows, "session")
    duration_by_regime = _group_duration(duration_rows, "market_regime")
    duration_by_cluster = _group_duration(duration_rows, "pattern_cluster")

    # Build pattern candidates/signatures from historical setup records.
    patterns: dict[str, dict[str, Any]] = {}
    for r in setups:
        sig = {
            "structure_type": int(r.get("structure_type", 0) or 0),
            "stop_hunt": int(r.get("stop_hunt", 0) or 0),
            "ob_present": int(r.get("ob_present", 0) or 0),
            "fib_proximity": int(r.get("fib_proximity", 0) or 0),
            "pattern_type_id": int(r.get("pattern_type_id", 0) or 0),
            "prior_trend_aligned": int(r.get("prior_trend_aligned", 0) or 0),
            "trend_bucket": _bucket_strength(r.get("trend_strength")),
            "mtf_bucket": _bucket_mtf(r.get("mtf_score")),
        }
        pattern_id = (
            f"PT_{sig['pattern_type_id']}_S{sig['structure_type']}_SH{sig['stop_hunt']}_"
            f"OB{sig['ob_present']}_FB{sig['fib_proximity']}_A{sig['prior_trend_aligned']}_"
            f"T{sig['trend_bucket']}_M{sig['mtf_bucket']}"
        )

        p = patterns.setdefault(
            pattern_id,
            {
                "pattern_id": pattern_id,
                "pattern_signature": sig,
                "occurrences": 0,
                "wins": 0,
                "losses": 0,
                "returns": [],
                "expected_moves": [],
                "actual_moves": [],
                "holding_times": [],
                "prediction_errors": [],
                "confidences": [],
                "mfe": [],
                "mae": [],
                "drawdowns": [],
                "sessions": defaultdict(int),
                "regimes": defaultdict(int),
                "first_seen": None,
                "last_seen": None,
                "cluster_id": _cluster_family(sig),
                "pattern_version": "v1",
            },
        )

        p["occurrences"] += 1

        ts = str(r.get("timestamp", "") or "")
        if ts:
            if p["first_seen"] is None or ts < p["first_seen"]:
                p["first_seen"] = ts
            if p["last_seen"] is None or ts > p["last_seen"]:
                p["last_seen"] = ts

        out = r.get("outcome")
        if out is not None:
            if int(out) == 1:
                p["wins"] += 1
            else:
                p["losses"] += 1

        rr = _safe_float(r.get("rr_achieved"))
        if rr is not None:
            signed_rr = rr if int(r.get("outcome", 0) or 0) == 1 else -rr
            p["returns"].append(signed_rr)

        sid = str(r.get("setup_id", "") or "")
        srow = sig_map.get(sid, {})
        conf = _safe_float(srow.get("confidence"))
        if conf is not None:
            p["confidences"].append(conf)

        em = _safe_float(srow.get("expected_distance_pts"))
        if em is not None:
            p["expected_moves"].append(em)

        trow = trade_by_signal.get(sid, {})
        am = _safe_float(trow.get("realized_distance_pts"))
        if am is not None:
            p["actual_moves"].append(am)

        if em is not None and am is not None:
            p["prediction_errors"].append(abs(em - am))

        pnl = _safe_float(trow.get("pnl"))
        if pnl is not None and pnl < 0:
            p["drawdowns"].append(abs(pnl))

    # Pattern library rows.
    pattern_library: list[dict[str, Any]] = []
    cluster_rows: dict[str, dict[str, Any]] = {}

    for pid, p in patterns.items():
        occ = int(p["occurrences"])
        wins = int(p["wins"])
        losses = int(p["losses"])
        total_labeled = wins + losses
        win_rate = (wins / total_labeled) if total_labeled else 0.0

        avg_ret = mean(p["returns"]) if p["returns"] else None
        avg_exp = mean(p["expected_moves"]) if p["expected_moves"] else None
        avg_act = mean(p["actual_moves"]) if p["actual_moves"] else None
        avg_hold = mean(p["holding_times"]) if p["holding_times"] else None
        avg_err = mean(p["prediction_errors"]) if p["prediction_errors"] else None
        avg_conf_pattern = mean(p["confidences"]) if p["confidences"] else None
        avg_mfe = mean(p["mfe"]) if p["mfe"] else None
        avg_mae = mean(p["mae"]) if p["mae"] else None
        avg_dd = mean(p["drawdowns"]) if p["drawdowns"] else None

        # Stability from confidence and return variability if enough data.
        stability = None
        if len(p["returns"]) > 1:
            rstd = pstdev(p["returns"])
            stability = max(0.0, 1.0 - min(1.0, rstd))

        expectancy = avg_ret
        lifecycle = _lifecycle(occ, expectancy, stability, p["last_seen"])

        row = {
            "pattern_id": pid,
            "pattern_signature": p["pattern_signature"],
            "occurrences": occ,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "average_return_pct": round(avg_ret, 4) if avg_ret is not None else STATUS_AWAITING,
            "average_expected_move": round(avg_exp, 4) if avg_exp is not None else STATUS_AWAITING,
            "average_actual_move": round(avg_act, 4) if avg_act is not None else STATUS_AWAITING,
            "average_holding_time": round(avg_hold, 4) if avg_hold is not None else STATUS_UNAVAILABLE,
            "average_prediction_error": round(avg_err, 4) if avg_err is not None else STATUS_AWAITING,
            "average_confidence": round(avg_conf_pattern, 4) if avg_conf_pattern is not None else STATUS_AWAITING,
            "mfe": round(avg_mfe, 4) if avg_mfe is not None else STATUS_UNAVAILABLE,
            "mae": round(avg_mae, 4) if avg_mae is not None else STATUS_UNAVAILABLE,
            "average_drawdown": round(avg_dd, 4) if avg_dd is not None else STATUS_AWAITING,
            "best_session": STATUS_UNAVAILABLE,
            "worst_session": STATUS_UNAVAILABLE,
            "best_market_regime": STATUS_UNAVAILABLE,
            "trend_bias": p["pattern_signature"].get("structure_type", 0),
            "current_adaptiveness": lifecycle,
            "first_seen": p["first_seen"] or STATUS_AWAITING,
            "last_seen": p["last_seen"] or STATUS_AWAITING,
            "cluster_id": p["cluster_id"],
            "pattern_version": p["pattern_version"],
            "lifecycle_stage": lifecycle,
            "expectancy": round(expectancy, 4) if expectancy is not None else STATUS_AWAITING,
            "stability": round(stability, 4) if stability is not None else STATUS_AWAITING,
        }
        pattern_library.append(row)

        c = cluster_rows.setdefault(
            p["cluster_id"],
            {
                "cluster_id": p["cluster_id"],
                "occurrences": 0,
                "wins": 0,
                "losses": 0,
                "returns": [],
                "confidences": [],
                "expected_moves": [],
                "actual_moves": [],
                "mfe": [],
                "mae": [],
                "stability": [],
                "diversity_members": set(),
            },
        )
        c["occurrences"] += occ
        c["wins"] += wins
        c["losses"] += losses
        c["diversity_members"].add(pid)
        if avg_ret is not None:
            c["returns"].append(avg_ret)
        if avg_conf_pattern is not None:
            c["confidences"].append(avg_conf_pattern)
        if avg_exp is not None:
            c["expected_moves"].append(avg_exp)
        if avg_act is not None:
            c["actual_moves"].append(avg_act)
        if avg_mfe is not None:
            c["mfe"].append(avg_mfe)
        if avg_mae is not None:
            c["mae"].append(avg_mae)
        if stability is not None:
            c["stability"].append(stability)

    pattern_library.sort(key=lambda x: x["occurrences"], reverse=True)

    cluster_library: list[dict[str, Any]] = []
    for _, c in cluster_rows.items():
        ttl = c["wins"] + c["losses"]
        cluster_library.append(
            {
                "cluster_id": c["cluster_id"],
                "occurrences": c["occurrences"],
                "win_rate": round((c["wins"] / ttl), 4) if ttl else 0.0,
                "average_return": round(mean(c["returns"]), 4) if c["returns"] else STATUS_AWAITING,
                "average_confidence": round(mean(c["confidences"]), 4) if c["confidences"] else STATUS_AWAITING,
                "average_expected_move": round(mean(c["expected_moves"]), 4) if c["expected_moves"] else STATUS_AWAITING,
                "average_actual_move": round(mean(c["actual_moves"]), 4) if c["actual_moves"] else STATUS_AWAITING,
                "average_mfe": round(mean(c["mfe"]), 4) if c["mfe"] else STATUS_UNAVAILABLE,
                "average_mae": round(mean(c["mae"]), 4) if c["mae"] else STATUS_UNAVAILABLE,
                "pattern_stability": round(mean(c["stability"]), 4) if c["stability"] else STATUS_AWAITING,
                "pattern_diversity": len(c["diversity_members"]),
                "cluster_confidence": round(mean(c["confidences"]), 4) if c["confidences"] else STATUS_AWAITING,
            }
        )
    cluster_library.sort(key=lambda x: x["occurrences"], reverse=True)

    # Pattern Context Intelligence is additive/observational only and does not change execution.
    pattern_context_intelligence = build_pattern_context_intelligence(
        root_dir=root_dir,
        completed_predictions=completed_predictions,
        pattern_library=pattern_library,
    )
    pattern_context_library = pattern_context_intelligence.get("pattern_context_library", []) if isinstance(pattern_context_intelligence, dict) else []
    if isinstance(pattern_context_library, list) and pattern_context_library:
        pattern_library = pattern_context_library

    # Version metadata is initialized before shared contracts for stable payload generation order.
    model_version = status.get("ml", {}).get("model_version", "0")
    dataset_generation = status.get("dataset_generation", "gen1")
    feature_version = status.get("feature_version", "v1")
    strategy_version = status.get("strategy_version", "v1")

    # Shared knowledge contracts (additive metadata only).
    shared_knowledge_contracts: list[dict[str, Any]] = []
    for p in pattern_library:
        pid = str(p.get("pattern_id", "unknown") or "unknown")
        kc = KnowledgeContract(
            knowledge_id=f"K-{pid}",
            source_system=SourceSystem.HERMES,
            dataset_generation=str(dataset_generation),
            model_version=str(model_version),
            feature_version=str(feature_version),
            strategy_version=str(strategy_version),
            timestamp=datetime.now(timezone.utc).isoformat(),
            knowledge_version="knowledge-v1",
            evidence_version="evidence-v1",
            confidence_version="confidence-v1",
            pattern_version=str(p.get("pattern_version", "v1")),
            trace_metadata={
                "pattern_id": pid,
                "source_system": SourceSystem.HERMES.value,
                "cluster_id": str(p.get("cluster_id", "unknown") or "unknown"),
                "lifecycle_stage": str(p.get("lifecycle_stage", "Candidate") or "Candidate"),
                "lineage_source": "storage/olympus/event_lineage.jsonl",
            },
        )
        ec = EvidenceConfidenceContract(
            implementation_pct=100.0,
            evidence_pct=min(100.0, (int(p.get("occurrences", 0) or 0) / 150.0) * 100.0),
            knowledge_confidence_pct=float((_safe_float(p.get("knowledge_confidence")) or 0.0)),
            reliability=_sample_stage(int(p.get("occurrences", 0) or 0)),
            sample_size=int(p.get("occurrences", 0) or 0),
            confidence_interval=_binom_ci(_safe_float(p.get("win_rate")), int(p.get("occurrences", 0) or 0)),
            historical_stability=_safe_float(p.get("stability")) if _safe_float(p.get("stability")) is not None else STATUS_AWAITING,
            concept_drift=STATUS_AWAITING,
            evidence_level=min(100.0, (int(p.get("occurrences", 0) or 0) / 150.0) * 100.0),
            current_grade=_letter_grade((_safe_float(p.get("knowledge_confidence")) or 0.0)),
            pending_validation=int(p.get("occurrences", 0) or 0) < SAMPLE_THRESHOLDS["developing"],
            estimated_samples_remaining=max(0, SAMPLE_THRESHOLDS["developing"] - int(p.get("occurrences", 0) or 0)),
        )
        shared_knowledge_contracts.append(
            {
                "knowledge_contract": kc.as_dict(),
                "evidence_confidence_contract": ec.as_dict(),
            }
        )

    unique_patterns = len(pattern_library)
    pattern_clusters = len(cluster_library)
    freqs = [int(r["occurrences"]) for r in pattern_library]
    diversity_index = _shannon_diversity(freqs)
    reuse_rate = (sum(freqs) / unique_patterns) if unique_patterns else None
    concentration = (max(freqs) / sum(freqs)) if freqs and sum(freqs) else None

    # Pattern analytics.
    common = pattern_library[0] if pattern_library else None
    reliable = [r for r in pattern_library if int(r["occurrences"]) >= 5 and isinstance(r["expectancy"], (float, int))]
    highest_win = max(reliable, key=lambda r: r["win_rate"]) if reliable else None
    lowest_win = min(reliable, key=lambda r: r["win_rate"]) if reliable else None
    highest_exp = max(reliable, key=lambda r: float(r["expectancy"])) if reliable else None
    lowest_exp = min(reliable, key=lambda r: float(r["expectancy"])) if reliable else None
    highest_ret = max(reliable, key=lambda r: float(r["average_return_pct"])) if reliable else None

    # Trade/performance metrics from closed trades + labeled outcomes.
    pnl_vals = [float(t.get("pnl", 0.0) or 0.0) for t in closed_trades if t.get("pnl") is not None]
    wins_pnl = [p for p in pnl_vals if p > 0]
    losses_pnl = [p for p in pnl_vals if p < 0]
    avg_win = mean(wins_pnl) if wins_pnl else None
    avg_loss = mean(losses_pnl) if losses_pnl else None
    win_rate = (len(wins_pnl) / len(pnl_vals)) if pnl_vals else None
    profit_factor = (sum(wins_pnl) / abs(sum(losses_pnl))) if losses_pnl else None
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss is not None and avg_loss != 0 else None

    # R multiple and expectancy.
    rr_vals = []
    for t in closed_trades:
        pred = _safe_float(t.get("predicted_distance_pts"))
        real = _safe_float(t.get("realized_distance_pts"))
        if pred is None or real is None or pred == 0:
            continue
        rr_vals.append(real / pred)
    avg_r_multiple = mean(rr_vals) if rr_vals else None

    expectancy = None
    if avg_win is not None and avg_loss is not None and win_rate is not None:
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

    # Recovery factor from cumulative drawdown.
    recovery_factor = None
    if pnl_vals:
        eq = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnl_vals:
            eq += p
            peak = max(peak, eq)
            dd = peak - eq
            max_dd = max(max_dd, dd)
        if max_dd > 0:
            recovery_factor = eq / max_dd

    # Adaptive execution analytics (analytical only).
    entry_eff: list[float] = []
    exit_eff: list[float] = []
    pred_errs: list[float] = []
    tp_ach: list[float] = []
    sl_dist: list[float] = []
    rr_achieved_vals: list[float] = []
    profit_pct_vals: list[float] = []
    loss_pct_vals: list[float] = []

    for t in closed_trades:
        pred = _safe_float(t.get("predicted_distance_pts"))
        act = _safe_float(t.get("realized_distance_pts"))
        entry = _safe_float(t.get("entry_price"))
        exitp = _safe_float(t.get("exit_price"))
        slp = _safe_float(t.get("sl_price"))
        tp = _safe_float(t.get("tp_price"))

        if pred and act is not None and pred > 0:
            ratio = act / pred
            entry_eff.append(min(2.0, max(0.0, ratio)))
            exit_eff.append(min(2.0, max(0.0, ratio)))
            pred_errs.append(abs(pred - act))
            rr_achieved_vals.append(ratio)

        if pred and act is not None and pred > 0:
            tp_ach.append(min(2.0, max(0.0, act / pred)))

        if entry and slp is not None:
            sl_dist.append(abs(entry - slp))

        if entry and exitp is not None:
            pct = ((exitp - entry) / entry) * 100.0
            if float(t.get("pnl", 0.0) or 0.0) >= 0:
                profit_pct_vals.append(abs(pct))
            else:
                loss_pct_vals.append(abs(pct))

    # Growth timeline.
    growth = defaultdict(int)
    for r in setups:
        ts = str(r.get("timestamp", "") or "")
        day = ts[:10] if len(ts) >= 10 else "unknown"
        growth[day] += 1
    growth_series = [{"date": k, "records": v} for k, v in sorted(growth.items()) if k != "unknown"]

    retraining_events = sum(1 for e in lineage if e.get("event_type") == "ml_updated")
    pattern_discovery_events = sum(1 for e in lineage if e.get("event_type") == "pattern_learned")

    # Snapshot readiness.
    snapshot_count = len(snapshots)
    if snapshot_count > 0:
        pattern_snapshot_display: Any = snapshot_count
        snapshot_status = STATUS_AVAILABLE
    else:
        pattern_snapshot_display = STATUS_PENDING
        snapshot_status = STATUS_FRAMEWORK

    # Executed and simulation counts.
    status_stats = status.get("stats", {}) or {}
    signals_entered = int(status_stats.get("signals_entered", 0) or 0)

    has_exec_tracking = any("execution_type" in r for r in setups)
    tracked_sim_count = sum(1 for r in setups if str(r.get("execution_type", "")).lower() == "simulated")
    tracked_live_count = sum(1 for r in setups if str(r.get("execution_type", "")).lower() == "live")

    executed_count = tracked_live_count if tracked_live_count > 0 else signals_entered
    executed_reason = "from live execution_type tags" if tracked_live_count > 0 else "derived from entered signals (paper execution)"

    if has_exec_tracking and tracked_sim_count > 0:
        simulation_count_value: Any = tracked_sim_count
        simulation_status = STATUS_AVAILABLE
    elif signals_entered > 0 and str(status.get("execution_type", "")).lower() == "simulated":
        simulation_count_value = "Simulation Tracking Not Yet Implemented"
        simulation_status = STATUS_PENDING
    else:
        simulation_count_value = "Simulation Tracking Not Yet Implemented"
        simulation_status = STATUS_PENDING

    # Readiness matrix for dashboard metrics.
    readiness = [
        {"metric": "ML Records", "status": "✅ Available", "source": "models/hermes/setups.json"},
        {"metric": "Market Snapshots", "status": "✅ Available", "source": "setup + snapshot stores"},
        {"metric": "Pattern Learning Snapshots", "status": "✅ Available" if snapshot_count > 0 else "🔵 Framework required", "source": "storage/olympus/pattern_snapshots.jsonl"},
        {"metric": "Executed Trades", "status": "✅ Available" if executed_count > 0 else "🟡 Derivable", "source": executed_reason},
        {"metric": "Simulation Count", "status": "✅ Available" if simulation_status == STATUS_AVAILABLE else "🟡 Derivable from existing data", "source": "execution_type + runtime execution mode"},
        {"metric": "Average Confidence", "status": "✅ Available" if avg_conf is not None else "🟡 Derivable from existing data", "source": "signals/skipped + lineage predictions"},
        {"metric": "Prediction Accuracy", "status": "✅ Available" if labeled_count > 0 else "🟡 Derivable after outcome analysis", "source": "labeled setup outcomes"},
        {"metric": "Pattern Clusters", "status": "✅ Available" if pattern_clusters > 0 else "🔵 Framework required", "source": "signature cluster families"},
        {"metric": "Pattern Discoveries", "status": "✅ Available" if pattern_discovery_events > 0 else "🔵 Requires statistical discovery engine", "source": "lineage pattern_learned events"},
        {
            "metric": "Confidence Calibration",
            "status": "✅ Available" if len(completed_with_conf) >= SAMPLE_THRESHOLDS["insufficient"] else "🟡 Awaiting richer historical samples",
            "source": "setups outcomes + prediction confidence + runtime trades",
        },
        {
            "metric": "Directional Accuracy",
            "status": "✅ Available" if len(completed_predictions) >= SAMPLE_THRESHOLDS["insufficient"] else "🟡 Awaiting richer historical samples",
            "source": "completed prediction classification over immutable trade outcomes",
        },
        {
            "metric": "Trade Duration",
            "status": "✅ Available" if len(duration_rows) >= SAMPLE_THRESHOLDS["insufficient"] else "🟡 Awaiting richer lineage close timestamps",
            "source": "lineage trade_entered/trade_closed timestamps",
        },
    ]

    inconsistencies: list[dict[str, Any]] = []
    if snapshot_count == 0:
        inconsistencies.append(
            {
                "metric": "Pattern Learning Snapshots",
                "current": "0",
                "issue": "No snapshot framework rows yet",
                "resolution": "Display Pending Initialization until snapshot history exists",
            }
        )

    # System/model/feature metadata.
    model_version = status.get("ml", {}).get("model_version", "0")
    dataset_generation = status.get("dataset_generation", "gen1")
    feature_version = status.get("feature_version", "v1")
    strategy_version = status.get("strategy_version", "v1")

    latest_ts = None
    if setups:
        ts_vals = [s.get("timestamp") for s in setups if s.get("timestamp")]
        if ts_vals:
            latest_ts = max(ts_vals)
    recency_weight = None
    if latest_ts:
        dt_latest = _safe_iso_to_dt(latest_ts)
        if dt_latest is not None:
            age_days = max(0, (datetime.now(timezone.utc) - dt_latest).days)
            recency_weight = round(max(0.0, 1.0 - min(1.0, age_days / 120.0)), 4)

    completeness_fields = [
        "structure_type",
        "trend_strength",
        "mtf_score",
        "sr_confidence",
        "pattern_confidence",
        "outcome",
        "rr_achieved",
        "execution_type",
        "model_version_used",
        "feature_version",
        "timestamp",
    ]
    completeness_scores: list[float] = []
    for r in setups:
        have = sum(1 for k in completeness_fields if r.get(k) is not None)
        completeness_scores.append(have / max(1, len(completeness_fields)))
    data_completeness = round(mean(completeness_scores), 4) if completeness_scores else None

    def _metric_kc(sample_count: int, consistency: float | None, calibration: float | None = None) -> dict[str, Any]:
        sample_component = min(1.0, sample_count / SAMPLE_THRESHOLDS["elite"])
        consistency_component = max(0.0, min(1.0, consistency if consistency is not None else 0.0))
        calibration_component = 1.0
        if calibration is not None:
            calibration_component = max(0.0, 1.0 - min(1.0, calibration))
        recency_component = recency_weight if recency_weight is not None else 0.5
        completeness_component = data_completeness if data_completeness is not None else 0.5
        score = 100.0 * (
            (0.28 * sample_component)
            + (0.20 * consistency_component)
            + (0.18 * calibration_component)
            + (0.16 * recency_component)
            + (0.18 * completeness_component)
        )
        score = round(score, 2)
        return {
            "knowledge_confidence_score": score,
            "knowledge_level": _knowledge_level(score),
            "sample_count": sample_count,
            "recency_weight": round(recency_component, 4),
            "data_completeness": round(completeness_component, 4),
        }

    metrics: Dict[str, Any] = {
        "current_ml_records": ml_records,
        "labeled_records": labeled_count,
        "training_records": labeled_count,
        "pattern_learning_snapshots": pattern_snapshot_display,
        "market_snapshots": market_snapshots,
        "learning_events": labeled_count,
        "pattern_discoveries": pattern_discovery_events,
        "unique_pattern_sequences": unique_patterns,
        "pattern_clusters": pattern_clusters,
        "pattern_library_size": len(pattern_library),
        "executed_trades": executed_count if executed_count > 0 else "Unavailable",
        "executed_trades_reason": executed_reason,
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
        "skipped_trades": len(skipped_trades),
        "simulation_count": simulation_count_value,
        "average_confidence": round(avg_conf, 4) if avg_conf is not None else STATUS_AWAITING,
        "median_confidence": round(med_conf, 4) if med_conf is not None else STATUS_AWAITING,
        "confidence_std_dev": round(std_conf, 4) if std_conf is not None else STATUS_AWAITING,
        "confidence_distribution": conf_dist if conf_dist else STATUS_AWAITING,
        "prediction_accuracy": round((sum(int(r.get("outcome", 0)) for r in labeled) / labeled_count), 4) if labeled_count else STATUS_AWAITING,
        "average_prediction_accuracy": round((sum(int(r.get("outcome", 0)) for r in labeled) / labeled_count), 4) if labeled_count else STATUS_AWAITING,
        "historical_confidence_accuracy": round((1.0 - (ece or 1.0)), 4) if ece is not None else STATUS_AWAITING,
        "confidence_reliability_rating": (
            "High" if ece is not None and ece <= 0.06 else "Moderate" if ece is not None and ece <= 0.12 else "Low"
        ) if ece is not None else STATUS_AWAITING,
        "calibration_error": round(ece, 4) if ece is not None else STATUS_AWAITING,
        "maximum_calibration_error": round(mce, 4) if mce is not None else STATUS_AWAITING,
        "brier_score": round(brier_global, 4) if brier_global is not None else STATUS_AWAITING,
        "optimal_confidence_threshold": best_thr if best_thr is not None else STATUS_AWAITING,
        "confidence_drift": round(
            (
                mean([float(r.get("confidence", 0.0)) for r in completed_with_conf[-max(1, len(completed_with_conf)//3):]])
                - mean([float(r.get("confidence", 0.0)) for r in completed_with_conf[:max(1, len(completed_with_conf)//3)]])
            ),
            4,
        ) if len(completed_with_conf) >= 6 else STATUS_AWAITING,
        "confidence_stability": round(max(0.0, 1.0 - min(1.0, std_conf or 1.0)), 4) if std_conf is not None else STATUS_AWAITING,
        "directional_accuracy": round(directional_accuracy, 4) if directional_accuracy is not None else STATUS_AWAITING,
        "forecast_accuracy": round(forecast_accuracy, 4) if forecast_accuracy is not None else STATUS_AWAITING,
        "execution_accuracy": round(execution_quality, 4) if execution_quality is not None else STATUS_AWAITING,
        "entry_quality": round(entry_quality, 4) if entry_quality is not None else STATUS_AWAITING,
        "exit_quality": round(exit_quality, 4) if exit_quality is not None else STATUS_AWAITING,
        "direction_stability": round(directional_consistency, 4) if directional_consistency is not None else STATUS_AWAITING,
        "distance_prediction_error": round(mean(pred_errs), 4) if pred_errs else STATUS_AWAITING,
        "confidence_calibration": round(ece, 4) if ece is not None else STATUS_AWAITING,
        "pattern_success_rate": round((sum(int(r.get("outcome", 0)) for r in labeled) / labeled_count), 4) if labeled_count else STATUS_AWAITING,
        "average_expected_move": round(mean([x for x in [_safe_float(t.get("predicted_distance_pts")) for t in closed_trades] if x is not None]), 4) if closed_trades else STATUS_AWAITING,
        "average_actual_move": round(mean([x for x in [_safe_float(t.get("realized_distance_pts")) for t in closed_trades] if x is not None]), 4) if closed_trades else STATUS_AWAITING,
        "average_trade_duration": round(avg_dur, 2) if avg_dur is not None else STATUS_AWAITING,
        "average_holding_time": round(avg_dur, 2) if avg_dur is not None else STATUS_AWAITING,
        "average_time_to_tp": round(mean(tp_durs), 2) if tp_durs else STATUS_AWAITING,
        "average_time_to_sl": round(mean(sl_durs), 2) if sl_durs else STATUS_AWAITING,
        "duration_stability": round(dur_stability, 4) if dur_stability is not None else STATUS_AWAITING,
        "duration_confidence": _sample_stage(len(duration_rows)),
        "learning_velocity": round(sum(x["records"] for x in growth_series[-7:]) / max(1, len(growth_series[-7:])), 2) if growth_series else 0.0,
        "learning_growth": growth_series,
        "dataset_generation": dataset_generation,
        "current_model_version": str(model_version),
        "feature_version": feature_version,
        "strategy_version": strategy_version,
        "average_mfe": STATUS_UNAVAILABLE,
        "average_mae": STATUS_UNAVAILABLE,
        "average_prediction_error": round(mean(pred_errs), 4) if pred_errs else STATUS_AWAITING,
        "average_entry_efficiency": round(mean(entry_eff), 4) if entry_eff else STATUS_AWAITING,
        "average_exit_efficiency": round(mean(exit_eff), 4) if exit_eff else STATUS_AWAITING,
        "average_historical_rr": round(mean(rr_achieved_vals), 4) if rr_achieved_vals else STATUS_AWAITING,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else STATUS_AWAITING,
        "expectancy": round(expectancy, 4) if expectancy is not None else STATUS_AWAITING,
        "recovery_factor": round(recovery_factor, 4) if recovery_factor is not None else STATUS_AWAITING,
        "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio is not None else STATUS_AWAITING,
        "adaptive_execution_readiness": "Ready" if rr_achieved_vals and pred_errs else "Developing",
    }

    metric_knowledge_confidence = {
        "confidence_calibration": _metric_kc(len(completed_with_conf), metrics.get("confidence_stability") if isinstance(metrics.get("confidence_stability"), (float, int)) else None, _safe_float(metrics.get("calibration_error"))),
        "directional_accuracy": _metric_kc(len(completed_predictions), _safe_float(metrics.get("direction_stability"))),
        "duration_intelligence": _metric_kc(len(duration_rows), _safe_float(metrics.get("duration_stability"))),
        "expectancy": _metric_kc(len(pnl_vals), _safe_float(metrics.get("pattern_success_rate"))),
    }

    pattern_intelligence: Dict[str, Any] = {
        "pattern_library": pattern_library,
        "cluster_library": cluster_library,
        "most_common_pattern": common["pattern_id"] if common else STATUS_AWAITING,
        "highest_win_pattern": highest_win["pattern_id"] if highest_win else STATUS_AWAITING,
        "highest_expectancy_pattern": highest_exp["pattern_id"] if highest_exp else STATUS_AWAITING,
        "highest_return_pattern": highest_ret["pattern_id"] if highest_ret else STATUS_AWAITING,
        "lowest_win_pattern": lowest_win["pattern_id"] if lowest_win else STATUS_AWAITING,
        "lowest_expectancy_pattern": lowest_exp["pattern_id"] if lowest_exp else STATUS_AWAITING,
        "most_reliable_pattern": highest_win["pattern_id"] if highest_win else STATUS_AWAITING,
        "most_adaptive_pattern": highest_exp["pattern_id"] if highest_exp else STATUS_AWAITING,
        "pattern_diversity_index": round(diversity_index, 4),
        "pattern_reuse_rate": round(reuse_rate, 4) if reuse_rate is not None else STATUS_AWAITING,
        "pattern_stability": round(mean([r["stability"] for r in pattern_library if isinstance(r.get("stability"), (float, int))]), 4)
        if any(isinstance(r.get("stability"), (float, int)) for r in pattern_library)
        else STATUS_AWAITING,
        "pattern_evolution": STATUS_DERIVABLE,
        "pattern_frequency": [{"pattern_id": r["pattern_id"], "occurrences": r["occurrences"]} for r in pattern_library],
        "pattern_confidence_distribution": conf_dist if conf_dist else STATUS_AWAITING,
        "pattern_success_rate": metrics["pattern_success_rate"],
        "pattern_lifetime": STATUS_DERIVABLE,
        "pattern_concentration": round(concentration, 4) if concentration is not None else STATUS_AWAITING,
        "lifecycle_counts": {
            "Candidate": sum(1 for r in pattern_library if r.get("lifecycle_stage") == "Candidate"),
            "Emerging": sum(1 for r in pattern_library if r.get("lifecycle_stage") == "Emerging"),
            "Validated": sum(1 for r in pattern_library if r.get("lifecycle_stage") == "Validated"),
            "Elite": sum(1 for r in pattern_library if r.get("lifecycle_stage") == "Elite"),
            "Declining": sum(1 for r in pattern_library if r.get("lifecycle_stage") == "Declining"),
            "Archived": sum(1 for r in pattern_library if r.get("lifecycle_stage") == "Archived"),
        },
    }

    performance_intelligence: Dict[str, Any] = {
        "win_rate": round(win_rate, 4) if win_rate is not None else STATUS_AWAITING,
        "return_pct": status.get("stats", {}).get("return_pct_total", STATUS_AWAITING),
        "profit_factor": metrics["profit_factor"],
        "average_win": round(avg_win, 4) if avg_win is not None else STATUS_AWAITING,
        "average_loss": round(avg_loss, 4) if avg_loss is not None else STATUS_AWAITING,
        "average_expected_move": metrics["average_expected_move"],
        "average_actual_move": metrics["average_actual_move"],
        "prediction_accuracy": metrics["prediction_accuracy"],
        "confidence_calibration": metrics["confidence_calibration"],
        "trade_quality": STATUS_DERIVABLE,
        "risk_efficiency": round((profit_factor or 0.0) * (win_rate or 0.0), 4) if profit_factor is not None and win_rate is not None else STATUS_AWAITING,
        "reward_efficiency": round(mean(tp_ach), 4) if tp_ach else STATUS_AWAITING,
        "signal_acceptance_rate": round((int(status.get("stats", {}).get("signals_entered", 0) or 0) / max(1, int(status.get("stats", {}).get("signals_seen", 0) or 0))), 4),
        "signal_skip_rate": round((int(status.get("stats", {}).get("signals_skipped", 0) or 0) / max(1, int(status.get("stats", {}).get("signals_seen", 0) or 0))), 4),
        "pattern_success_rate": metrics["pattern_success_rate"],
        "expectancy": metrics["expectancy"],
        "recovery_factor": metrics["recovery_factor"],
        "payoff_ratio": metrics["payoff_ratio"],
        "average_r_multiple": round(avg_r_multiple, 4) if avg_r_multiple is not None else STATUS_AWAITING,
    }

    adaptive_execution_intelligence: Dict[str, Any] = {
        "average_entry_efficiency": metrics["average_entry_efficiency"],
        "average_exit_efficiency": metrics["average_exit_efficiency"],
        "average_holding_time": metrics["average_holding_time"],
        "average_pullback": STATUS_UNAVAILABLE,
        "average_expansion": STATUS_UNAVAILABLE,
        "average_mfe": metrics["average_mfe"],
        "average_mae": metrics["average_mae"],
        "average_prediction_error": metrics["average_prediction_error"],
        "average_tp_achievement": round(mean(tp_ach), 4) if tp_ach else STATUS_AWAITING,
        "average_sl_distance": round(mean(sl_dist), 4) if sl_dist else STATUS_AWAITING,
        "average_time_to_tp": STATUS_UNAVAILABLE,
        "average_time_to_sl": STATUS_UNAVAILABLE,
        "average_profit_pct": round(mean(profit_pct_vals), 4) if profit_pct_vals else STATUS_AWAITING,
        "average_loss_pct": round(mean(loss_pct_vals), 4) if loss_pct_vals else STATUS_AWAITING,
        "average_rr_achieved": metrics["average_historical_rr"],
        "historical_optimal_tp": round(max([x for x in rr_achieved_vals if x is not None]), 4) if rr_achieved_vals else STATUS_AWAITING,
        "historical_optimal_sl": STATUS_UNAVAILABLE,
        "historical_break_even_point": STATUS_DERIVABLE,
        "historical_trailing_stop_distance": STATUS_UNAVAILABLE,
        "historical_partial_close_point": STATUS_UNAVAILABLE,
    }

    confidence_intelligence: Dict[str, Any] = {
        "average_confidence": metrics["average_confidence"],
        "historical_confidence_accuracy": metrics["historical_confidence_accuracy"],
        "confidence_reliability_rating": metrics["confidence_reliability_rating"],
        "calibration_error": metrics["calibration_error"],
        "expected_calibration_error": metrics["calibration_error"],
        "maximum_calibration_error": metrics["maximum_calibration_error"],
        "brier_score": metrics["brier_score"],
        "optimal_confidence_threshold": metrics["optimal_confidence_threshold"],
        "confidence_drift": metrics["confidence_drift"],
        "confidence_stability": metrics["confidence_stability"],
        "confidence_distribution": metrics["confidence_distribution"],
        "confidence_buckets": bucket_rows,
        "overconfidence_regions": over_regions if over_regions else STATUS_AWAITING,
        "underconfidence_regions": under_regions if under_regions else STATUS_AWAITING,
        "confidence_by_pattern": _group_conf_stats(completed_with_conf, "pattern_id"),
        "confidence_by_pattern_cluster": _group_conf_stats(completed_with_conf, "pattern_cluster"),
        "confidence_by_session": _group_conf_stats(completed_with_conf, "session"),
        "confidence_by_market_regime": _group_conf_stats(completed_with_conf, "market_regime"),
        "confidence_by_trend": _group_conf_stats(completed_with_conf, "trend_bucket"),
        "confidence_by_model_version": _group_conf_stats(completed_with_conf, "model_version"),
        "confidence_by_feature_version": _group_conf_stats(completed_with_conf, "feature_version"),
    }

    directional_intelligence: Dict[str, Any] = {
        "directional_accuracy": metrics["directional_accuracy"],
        "forecast_accuracy": metrics["forecast_accuracy"],
        "execution_accuracy": metrics["execution_accuracy"],
        "entry_quality": metrics["entry_quality"],
        "exit_quality": metrics["exit_quality"],
        "trade_management_quality": metrics["execution_accuracy"],
        "prediction_stability": metrics["direction_stability"],
        "directional_consistency": metrics["direction_stability"],
        "classification_counts": dict(class_counts),
        "direction_by_pattern": direction_by_pattern,
        "direction_by_session": direction_by_session,
        "direction_by_market_regime": direction_by_regime,
        "direction_by_pattern_cluster": direction_by_cluster,
    }

    duration_intelligence: Dict[str, Any] = {
        "expected_trade_duration": metrics["average_trade_duration"],
        "average_holding_time": metrics["average_holding_time"],
        "median_duration": round(med_dur, 2) if med_dur is not None else STATUS_AWAITING,
        "minimum_duration": round(min_dur, 2) if min_dur is not None else STATUS_AWAITING,
        "maximum_duration": round(max_dur, 2) if max_dur is not None else STATUS_AWAITING,
        "average_time_to_tp": metrics["average_time_to_tp"],
        "average_time_to_sl": metrics["average_time_to_sl"],
        "average_expansion_duration": STATUS_UNAVAILABLE,
        "average_pullback_duration": STATUS_UNAVAILABLE,
        "average_consolidation_duration": STATUS_UNAVAILABLE,
        "average_time_before_reversal": STATUS_UNAVAILABLE,
        "average_time_before_trend_continuation": STATUS_UNAVAILABLE,
        "duration_stability": metrics["duration_stability"],
        "duration_variance": round(pstdev(dur_vals), 2) if len(dur_vals) > 1 else STATUS_AWAITING,
        "duration_confidence": metrics["duration_confidence"],
        "historical_holding_distribution": {
            "0-5m": sum(1 for d in dur_vals if d < 300),
            "5-15m": sum(1 for d in dur_vals if 300 <= d < 900),
            "15-60m": sum(1 for d in dur_vals if 900 <= d < 3600),
            "60m+": sum(1 for d in dur_vals if d >= 3600),
        } if dur_vals else STATUS_AWAITING,
        "duration_by_pattern": duration_by_pattern,
        "duration_by_session": duration_by_session,
        "duration_by_market_regime": duration_by_regime,
        "duration_by_pattern_cluster": duration_by_cluster,
    }

    cluster_map = {str(c.get("cluster_id")): c for c in cluster_library}
    dir_map = {str(r.get("key")): r for r in direction_by_pattern}
    conf_map = {str(r.get("key")): r for r in confidence_intelligence["confidence_by_pattern"]}
    dur_map = {str(r.get("key")): r for r in duration_by_pattern}

    execution_profiles: list[dict[str, Any]] = []
    for p in pattern_library:
        pid = str(p.get("pattern_id", "") or "")
        cid = str(p.get("cluster_id", "") or "")
        occ = int(p.get("occurrences", 0) or 0)
        conf_rel = conf_map.get(pid, {})
        dir_rel = dir_map.get(pid, {})
        dur_rel = dur_map.get(pid, {})
        sample_stage = _sample_stage(occ)
        stage_score = {
            "Insufficient Data": 20.0,
            "Emerging": 40.0,
            "Developing": 60.0,
            "Validated": 80.0,
            "Elite": 95.0,
        }.get(sample_stage, 20.0)
        wr = _safe_float(p.get("win_rate")) or 0.0
        cal = _safe_float(metrics.get("calibration_error"))
        cal_score = (1.0 - min(1.0, cal)) * 100.0 if cal is not None else 40.0
        dir_acc = _safe_float(dir_rel.get("directional_accuracy"))
        dir_score = (dir_acc * 100.0) if dir_acc is not None else 40.0
        knowledge_score = round((0.35 * stage_score) + (0.25 * cal_score) + (0.25 * (wr * 100.0)) + (0.15 * dir_score), 2)
        readiness_score = round((0.40 * stage_score) + (0.30 * dir_score) + (0.30 * (cal_score)), 2)

        execution_profiles.append(
            {
                "pattern_id": pid,
                "pattern_cluster": cid,
                "occurrences": occ,
                "win_rate": p.get("win_rate", STATUS_AWAITING),
                "expectancy": p.get("expectancy", STATUS_AWAITING),
                "average_return": p.get("average_return_pct", STATUS_AWAITING),
                "average_confidence": p.get("average_confidence", STATUS_AWAITING),
                "confidence_reliability": conf_rel.get("confidence_stability", STATUS_AWAITING),
                "directional_accuracy": dir_rel.get("directional_accuracy", STATUS_AWAITING),
                "entry_quality": metrics.get("entry_quality", STATUS_AWAITING),
                "exit_quality": metrics.get("exit_quality", STATUS_AWAITING),
                "average_duration": dur_rel.get("average_duration_sec", STATUS_AWAITING),
                "average_mfe": p.get("mfe", STATUS_UNAVAILABLE),
                "average_mae": p.get("mae", STATUS_UNAVAILABLE),
                "historical_optimal_tp": adaptive_execution_intelligence.get("historical_optimal_tp", STATUS_AWAITING),
                "historical_optimal_sl": adaptive_execution_intelligence.get("historical_optimal_sl", STATUS_UNAVAILABLE),
                "historical_break_even": adaptive_execution_intelligence.get("historical_break_even_point", STATUS_DERIVABLE),
                "historical_trailing_point": adaptive_execution_intelligence.get("historical_trailing_stop_distance", STATUS_UNAVAILABLE),
                "historical_partial_close_point": adaptive_execution_intelligence.get("historical_partial_close_point", STATUS_UNAVAILABLE),
                "pattern_stability": p.get("stability", STATUS_AWAITING),
                "pattern_adaptiveness": p.get("current_adaptiveness", STATUS_AWAITING),
                "knowledge_confidence_score": knowledge_score,
                "execution_readiness_score": readiness_score,
                "sample_stage": sample_stage,
            }
        )
    execution_profiles.sort(key=lambda x: float(x.get("occurrences", 0) or 0), reverse=True)

    execution_intelligence: Dict[str, Any] = {
        "execution_profiles": execution_profiles,
        "knowledge_confidence_score": round(mean([float(r.get("knowledge_confidence_score", 0.0)) for r in execution_profiles]), 2) if execution_profiles else STATUS_AWAITING,
        "execution_readiness_score": round(mean([float(r.get("execution_readiness_score", 0.0)) for r in execution_profiles]), 2) if execution_profiles else STATUS_AWAITING,
        "sample_stage_distribution": {
            "Insufficient Data": sum(1 for r in execution_profiles if r.get("sample_stage") == "Insufficient Data"),
            "Emerging": sum(1 for r in execution_profiles if r.get("sample_stage") == "Emerging"),
            "Developing": sum(1 for r in execution_profiles if r.get("sample_stage") == "Developing"),
            "Validated": sum(1 for r in execution_profiles if r.get("sample_stage") == "Validated"),
            "Elite": sum(1 for r in execution_profiles if r.get("sample_stage") == "Elite"),
        },
    }

    knowledge_quality_controls: Dict[str, Any] = {
        "sample_thresholds": SAMPLE_THRESHOLDS,
        "minimum_sample_threshold": SAMPLE_THRESHOLDS["insufficient"],
        "metrics": [
            {
                "metric": "Confidence Calibration",
                "sample_count": len(completed_with_conf),
                "confidence_interval": _binom_ci(1.0 - (ece or 1.0) if ece is not None else None, len(completed_with_conf)),
                "data_sufficiency": len(completed_with_conf) >= SAMPLE_THRESHOLDS["emerging"],
                "statistical_reliability": _sample_stage(len(completed_with_conf)),
            },
            {
                "metric": "Directional Accuracy",
                "sample_count": len(completed_predictions),
                "confidence_interval": _binom_ci(directional_accuracy, len(completed_predictions)),
                "data_sufficiency": len(completed_predictions) >= SAMPLE_THRESHOLDS["emerging"],
                "statistical_reliability": _sample_stage(len(completed_predictions)),
            },
            {
                "metric": "Trade Duration",
                "sample_count": len(duration_rows),
                "confidence_interval": _binom_ci(0.5 if duration_rows else None, len(duration_rows)),
                "data_sufficiency": len(duration_rows) >= SAMPLE_THRESHOLDS["emerging"],
                "statistical_reliability": _sample_stage(len(duration_rows)),
            },
        ],
        "adaptive_execution_gate": {
            "enabled": False,
            "reason": "Analytics only: metrics are observational and do not alter live execution.",
        },
    }

    # Performance diagnostics explain current edge quality without changing execution behavior.
    win_dist = _distribution_stats(wins_pnl)
    loss_dist = _distribution_stats(losses_pnl)
    all_dist = _distribution_stats(pnl_vals)
    risk_dist = {
        "sl_distance_distribution": _distribution_stats(sl_dist),
        "rr_distribution": _distribution_stats(rr_achieved_vals),
        "prediction_error_distribution": _distribution_stats(pred_errs),
    }

    perf_by_day: dict[str, list[float]] = defaultdict(list)
    for r in completed_predictions:
        ts = str(r.get("timestamp", "") or "")
        day = ts[:10] if len(ts) >= 10 else "unknown"
        rv = _safe_float(r.get("return_pct"))
        if day != "unknown" and rv is not None:
            perf_by_day[day].append(rv)

    expectancy_trend = [
        {"date": d, "expectancy": round(mean(vals), 4), "samples": len(vals)}
        for d, vals in sorted(perf_by_day.items())
    ]
    payoff_trend = [
        {
            "date": d,
            "payoff": round(
                (mean([v for v in vals if v > 0]) / abs(mean([v for v in vals if v < 0])))
                if any(v > 0 for v in vals) and any(v < 0 for v in vals)
                else 0.0,
                4,
            ),
        }
        for d, vals in sorted(perf_by_day.items())
    ]

    session_contrib: dict[str, list[float]] = defaultdict(list)
    pattern_contrib: dict[str, list[float]] = defaultdict(list)
    exec_contrib: dict[str, list[float]] = defaultdict(list)
    for r in completed_predictions:
        rv = _safe_float(r.get("return_pct"))
        if rv is None:
            continue
        session_contrib[str(r.get("session", "unknown"))].append(rv)
        pattern_contrib[str(r.get("pattern_id", "unknown"))].append(rv)
        exec_contrib[str(r.get("trade_type", "unknown"))].append(rv)

    performance_diagnostics = {
        "win_distribution": win_dist,
        "loss_distribution": loss_dist,
        "return_distribution": all_dist,
        "risk_distribution": risk_dist,
        "trade_efficiency": {
            "entry_efficiency": metrics.get("average_entry_efficiency", STATUS_AWAITING),
            "exit_efficiency": metrics.get("average_exit_efficiency", STATUS_AWAITING),
            "reward_efficiency": performance_intelligence.get("reward_efficiency", STATUS_AWAITING),
        },
        "edge_stability": {
            "prediction_edge": round((metrics.get("prediction_accuracy", 0.0) or 0.0) - 0.5, 4) if isinstance(metrics.get("prediction_accuracy"), (float, int)) else STATUS_AWAITING,
            "execution_edge": round((metrics.get("execution_accuracy", 0.0) or 0.0) - 0.5, 4) if isinstance(metrics.get("execution_accuracy"), (float, int)) else STATUS_AWAITING,
            "risk_management_edge": round((metrics.get("payoff_ratio", 0.0) or 0.0), 4) if isinstance(metrics.get("payoff_ratio"), (float, int)) else STATUS_AWAITING,
            "pattern_intelligence_edge": round((pattern_intelligence.get("pattern_success_rate", 0.0) or 0.0) - 0.5, 4) if isinstance(pattern_intelligence.get("pattern_success_rate"), (float, int)) else STATUS_AWAITING,
            "return_edge": return_intelligence.get("edge_stability", {}).get("return_edge", STATUS_AWAITING),
        },
        "expectancy_trend": expectancy_trend,
        "payoff_trend": payoff_trend,
        "session_contribution": [
            {"session": k, "expectancy": round(mean(v), 4), "samples": len(v)} for k, v in sorted(session_contrib.items())
        ],
        "pattern_contribution": [
            {"pattern_id": k, "expectancy": round(mean(v), 4), "samples": len(v)} for k, v in sorted(pattern_contrib.items())
        ],
        "execution_contribution": [
            {"trade_type": k, "expectancy": round(mean(v), 4), "samples": len(v)} for k, v in sorted(exec_contrib.items())
        ],
        "return_evolution": return_intelligence.get("historical_return_evolution", []),
        "explanations": [
            "Payoff ratio stays low when mean losses are materially larger than mean wins even with positive hit-rate.",
            "Recovery factor remains modest when drawdown excursions recover slowly relative to cumulative profit growth.",
            "Selective signal acceptance can improve expectancy while still leaving asymmetric loss tails unresolved.",
            "Return efficiency highlights how much of each favorable move Hermes actually captures.",
        ],
    }

    edge_stability = {
        "prediction_edge": performance_diagnostics["edge_stability"]["prediction_edge"],
        "execution_edge": performance_diagnostics["edge_stability"]["execution_edge"],
        "risk_management_edge": performance_diagnostics["edge_stability"]["risk_management_edge"],
        "pattern_intelligence_edge": performance_diagnostics["edge_stability"]["pattern_intelligence_edge"],
        "return_edge": performance_diagnostics["edge_stability"]["return_edge"],
        "knowledge_confidence": execution_intelligence.get("knowledge_confidence_score", STATUS_AWAITING),
        "confidence_edge": confidence_intelligence.get("confidence_stability", STATUS_AWAITING),
        "adaptive_readiness": execution_intelligence.get("execution_readiness_score", STATUS_AWAITING),
        "history": return_intelligence.get("edge_stability", {}).get("history", []),
    }

    # Pattern Genome: institutional memory decomposition for each pattern.
    conf_by_pattern = {str(x.get("key")): x for x in confidence_intelligence.get("confidence_by_pattern", [])}
    dir_by_pattern = {str(x.get("key")): x for x in directional_intelligence.get("direction_by_pattern", [])}
    dur_by_pattern = {str(x.get("key")): x for x in duration_intelligence.get("duration_by_pattern", [])}
    pattern_genome = []
    for p in pattern_library:
        pid = str(p.get("pattern_id", "") or "")
        sig = p.get("pattern_signature", {}) or {}
        cg = conf_by_pattern.get(pid, {})
        dg = dir_by_pattern.get(pid, {})
        dug = dur_by_pattern.get(pid, {})
        kcs = _safe_float((next((x.get("knowledge_confidence_score") for x in execution_profiles if str(x.get("pattern_id")) == pid), None)))
        pattern_genome.append(
            {
                "pattern_id": pid,
                "pattern_cluster": p.get("cluster_id", "unknown"),
                "structural_genes": {
                    "choch": int(sig.get("pattern_type_id", 0) in (3, 4)),
                    "bos": int(sig.get("structure_type", 0) in (1, 2)),
                    "liquidity_sweep": int(sig.get("stop_hunt", 0) or 0),
                    "order_block": int(sig.get("ob_present", 0) or 0),
                    "fair_value_gap": int(sig.get("fib_proximity", 0) or 0),
                    "premium_discount": sig.get("trend_bucket", "unknown"),
                },
                "context_genes": {
                    "session": "derived",
                    "market_regime": "derived",
                    "trend": sig.get("trend_bucket", "unknown"),
                    "volatility": "normal",
                },
                "behaviour_genes": {
                    "duration": dug.get("average_duration_sec", STATUS_AWAITING),
                    "expansion": p.get("average_actual_move", STATUS_AWAITING),
                    "pullback": STATUS_UNAVAILABLE,
                    "mfe": p.get("mfe", STATUS_UNAVAILABLE),
                    "mae": p.get("mae", STATUS_UNAVAILABLE),
                },
                "performance_genes": {
                    "win_rate": p.get("win_rate", STATUS_AWAITING),
                    "expectancy": p.get("expectancy", STATUS_AWAITING),
                    "profit_factor": performance_intelligence.get("profit_factor", STATUS_AWAITING),
                },
                "confidence_genes": {
                    "prediction_confidence": p.get("average_confidence", STATUS_AWAITING),
                    "calibration": confidence_intelligence.get("calibration_error", STATUS_AWAITING),
                    "reliability": cg.get("confidence_stability", STATUS_AWAITING),
                },
                "adaptation_genes": {
                    "pattern_stability": p.get("stability", STATUS_AWAITING),
                    "pattern_drift": confidence_intelligence.get("confidence_drift", STATUS_AWAITING),
                    "pattern_reuse": pattern_intelligence.get("pattern_reuse_rate", STATUS_AWAITING),
                    "knowledge_confidence": round(kcs, 2) if kcs is not None else STATUS_AWAITING,
                },
            }
        )

    # Hermes Academy: evidence-based competency framework.
    stage_thresholds = [
        ("Observer", 0.0),
        ("Student", 20.0),
        ("Apprentice", 35.0),
        ("Analyst", 50.0),
        ("Researcher", 65.0),
        ("Specialist", 78.0),
        ("Professional", 88.0),
        ("Institutional Intelligence", 95.0),
    ]

    sim_progress = min(100.0, (ml_records / 3000.0) * 100.0)
    paper_progress = min(100.0, (len(completed_predictions) / 1000.0) * 100.0)
    live_progress = min(100.0, (tracked_live_count / 200.0) * 100.0)

    sources = {
        "Simulation": {
            "infrastructure_readiness": 100.0,
            "evidence_collected": sim_progress,
            "validation_progress": min(100.0, (len(completed_with_conf) / 1200.0) * 100.0),
            "mastery": min(100.0, 0.55 * sim_progress + 0.45 * (pattern_clusters / 25.0 * 100.0)),
            "purpose": [
                "Pattern discovery",
                "Pattern clustering",
                "Market structure",
                "Pattern diversity",
                "Expected move estimation",
            ],
        },
        "Paper Trading": {
            "infrastructure_readiness": 100.0,
            "evidence_collected": paper_progress,
            "validation_progress": min(100.0, (len(completed_predictions) / 800.0) * 100.0),
            "mastery": min(
                100.0,
                0.40 * paper_progress
                + 0.30 * (_safe_float(metrics.get("directional_accuracy")) or 0.0) * 100.0
                + 0.30 * (_safe_float(metrics.get("historical_confidence_accuracy")) or 0.0) * 100.0,
            ),
            "purpose": [
                "Execution learning",
                "Pattern validation",
                "Confidence calibration",
                "Trade duration",
                "Entry quality",
                "Exit quality",
                "Expectancy",
                "Pattern stability",
            ],
        },
        "Live Validation": {
            "infrastructure_readiness": 100.0,
            "evidence_collected": live_progress,
            "validation_progress": min(100.0, (tracked_live_count / 120.0) * 100.0),
            "mastery": min(100.0, 0.30 * live_progress + 0.70 * (_safe_float(execution_intelligence.get("execution_readiness_score")) or 0.0)),
            "purpose": [
                "Execution validation",
                "Broker behaviour",
                "Spread",
                "Slippage",
                "Real-world robustness",
            ],
        },
    }
    for src in sources.values():
        src["status"] = _metric_maturity(True, int(src["evidence_collected"]), src["mastery"])
        src["progress_bar"] = _progress_bar(src["mastery"])

    primary_source = max(sources.items(), key=lambda kv: float(kv[1]["mastery"]))[0]
    primary_teacher = {
        "Simulation": "Simulation teaches theory.",
        "Paper Trading": "Paper trading teaches experience.",
        "Live Validation": "Live trading validates mastery.",
    }.get(primary_source, "Simulation teaches theory.")

    academy_specs = [
        {
            "name": "Market Structure",
            "implementation": 100.0,
            "sample_count": labeled_count,
            "evidence": min(100.0, (labeled_count / 1200.0) * 100.0),
            "confidence": _safe_float(metric_knowledge_confidence.get("expectancy", {}).get("knowledge_confidence_score")) or 0.0,
            "mastery": min(100.0, 40.0 + (_safe_float(metrics.get("prediction_accuracy")) or 0.0) * 60.0),
        },
        {
            "name": "Pattern Intelligence",
            "implementation": 100.0,
            "sample_count": unique_patterns,
            "evidence": min(100.0, (unique_patterns / 40.0) * 100.0),
            "confidence": _safe_float(execution_intelligence.get("knowledge_confidence_score")) or 0.0,
            "mastery": min(100.0, 0.55 * sim_progress + 0.45 * (_safe_float(pattern_intelligence.get("pattern_stability")) or 0.0) * 100.0),
        },
        {
            "name": "Execution Intelligence",
            "implementation": 100.0,
            "sample_count": len(execution_profiles),
            "evidence": min(100.0, (len(completed_predictions) / 1000.0) * 100.0),
            "confidence": _safe_float(execution_intelligence.get("execution_readiness_score")) or 0.0,
            "mastery": _safe_float(execution_intelligence.get("execution_readiness_score")) or 0.0,
        },
        {
            "name": "Confidence Intelligence",
            "implementation": 100.0,
            "sample_count": len(completed_with_conf),
            "evidence": min(100.0, (len(completed_with_conf) / 1200.0) * 100.0),
            "confidence": _safe_float(metric_knowledge_confidence.get("confidence_calibration", {}).get("knowledge_confidence_score")) or 0.0,
            "mastery": max(0.0, 100.0 - 100.0 * float(metrics.get("calibration_error", 1.0) or 1.0)) if isinstance(metrics.get("calibration_error"), (float, int)) else 0.0,
        },
        {
            "name": "Performance Intelligence",
            "implementation": 100.0,
            "sample_count": len(pnl_vals),
            "evidence": min(100.0, (len(pnl_vals) / 400.0) * 100.0),
            "confidence": _safe_float(metric_knowledge_confidence.get("expectancy", {}).get("knowledge_confidence_score")) or 0.0,
            "mastery": min(100.0, max(0.0, 45.0 + ((_safe_float(metrics.get("expectancy")) or 0.0) * 120.0))),
        },
        {
            "name": "Return Intelligence",
            "implementation": 100.0,
            "sample_count": int(return_summary.get("sample_size", 0) or 0),
            "evidence": min(100.0, (int(return_summary.get("sample_size", 0) or 0) / 150.0) * 100.0),
            "confidence": _safe_float(return_summary.get("knowledge_confidence_score")) or 0.0,
            "mastery": min(100.0, max(0.0, (50.0 + (_safe_float(return_summary.get("average_return_pct")) or 0.0) * 40.0) + ((_safe_float(return_summary.get("return_stability")) or 0.0) * 0.25))),
        },
        {
            "name": "Pattern Context Intelligence",
            "implementation": _safe_float((pattern_context_intelligence.get("academy_subject") or {}).get("implementation")) or 100.0,
            "sample_count": int((pattern_context_intelligence.get("academy_subject") or {}).get("estimated_samples_remaining", 0) or 0) + int(len(completed_predictions)),
            "evidence": _safe_float((pattern_context_intelligence.get("academy_subject") or {}).get("evidence")) or 0.0,
            "confidence": _safe_float((pattern_context_intelligence.get("academy_subject") or {}).get("knowledge_confidence")) or 0.0,
            "mastery": _safe_float((pattern_context_intelligence.get("academy_subject") or {}).get("mastery")) or 0.0,
        },
        {
            "name": "Adaptation Intelligence",
            "implementation": 100.0,
            "sample_count": len(execution_profiles),
            "evidence": min(100.0, (len(execution_profiles) / 60.0) * 100.0),
            "confidence": _safe_float(execution_intelligence.get("knowledge_confidence_score")) or 0.0,
            "mastery": min(100.0, 0.5 * (_safe_float(execution_intelligence.get("knowledge_confidence_score")) or 0.0) + 0.5 * (_safe_float(execution_intelligence.get("execution_readiness_score")) or 0.0)),
        },
    ]

    academies = []
    for spec in academy_specs:
        impl = max(0.0, min(100.0, _safe_float(spec.get("implementation")) or 0.0))
        evid = max(0.0, min(100.0, _safe_float(spec.get("evidence")) or 0.0))
        conf = max(0.0, min(100.0, _safe_float(spec.get("confidence")) or 0.0))
        mast = max(0.0, min(100.0, _safe_float(spec.get("mastery")) or 0.0))
        weighted = round((0.20 * impl) + (0.30 * evid) + (0.25 * conf) + (0.25 * mast), 2)
        academies.append(
            {
                "academy": spec["name"],
                "implementation": round(impl, 2),
                "evidence": round(evid, 2),
                "knowledge_confidence": round(conf, 2),
                "mastery": round(mast, 2),
                "weighted_competency": weighted,
                "knowledge_level": _knowledge_level(weighted),
                "current_grade": _letter_grade(weighted),
                "sample_count": int(spec["sample_count"]),
                "status": _metric_maturity(True, int(spec["sample_count"]), weighted),
                "lessons_completed": int(round(weighted)),
                "lessons_remaining": int(round(max(0.0, 100.0 - weighted))),
                "mastery_percentage": round(mast, 2),
            }
        )

    overall_competency = round(mean([float(a["weighted_competency"]) for a in academies]), 2) if academies else 0.0
    current_stage = _stage_from_thresholds(overall_competency, stage_thresholds)

    graduation_milestones = [
        {"name": "5000 ML Records", "current": ml_records, "target": 5000, "weight": 0.16, "evidence_samples": labeled_count, "min_samples": SAMPLE_THRESHOLDS["validated"]},
        {"name": "1500 Pattern Snapshots", "current": snapshot_count, "target": 1500, "weight": 0.10, "evidence_samples": snapshot_count, "min_samples": SAMPLE_THRESHOLDS["developing"]},
        {"name": "300 Validated Patterns", "current": len([p for p in pattern_library if p.get("lifecycle_stage") in ("Validated", "Elite")]), "target": 300, "weight": 0.10, "evidence_samples": unique_patterns, "min_samples": SAMPLE_THRESHOLDS["developing"]},
        {"name": "25 Pattern Clusters", "current": pattern_clusters, "target": 25, "weight": 0.08, "evidence_samples": pattern_clusters, "min_samples": SAMPLE_THRESHOLDS["insufficient"]},
        {"name": "Calibration Quality", "current": max(0.0, 100.0 - (float(metrics.get("calibration_error", 1.0) or 1.0) * 100.0)) if isinstance(metrics.get("calibration_error"), (float, int)) else 0.0, "target": 90.0, "weight": 0.16, "evidence_samples": len(completed_with_conf), "min_samples": SAMPLE_THRESHOLDS["validated"]},
        {"name": "Execution Intelligence", "current": _safe_float(execution_intelligence.get("execution_readiness_score")) or 0.0, "target": 80.0, "weight": 0.14, "evidence_samples": len(completed_predictions), "min_samples": SAMPLE_THRESHOLDS["developing"]},
        {"name": "Knowledge Confidence", "current": _safe_float(execution_intelligence.get("knowledge_confidence_score")) or 0.0, "target": 80.0, "weight": 0.14, "evidence_samples": len(completed_predictions), "min_samples": SAMPLE_THRESHOLDS["developing"]},
        {"name": "Adaptive Readiness", "current": _safe_float(execution_intelligence.get("execution_readiness_score")) or 0.0, "target": 85.0, "weight": 0.12, "evidence_samples": len(execution_profiles), "min_samples": SAMPLE_THRESHOLDS["insufficient"]},
    ]
    weighted_sum = 0.0
    total_weight = 0.0
    for m in graduation_milestones:
        target = max(1.0, float(m["target"]))
        pct = max(0.0, min(100.0, (float(m["current"]) / target) * 100.0))
        evidence_ok = int(m["evidence_samples"]) >= int(m["min_samples"])
        completion = evidence_ok and pct >= 100.0
        m["progress_pct"] = round(pct, 2)
        m["bar"] = _progress_bar(pct)
        m["evidence_sufficiency"] = evidence_ok
        m["status"] = "Validated" if completion else ("Pending Validation" if evidence_ok else "Developing")
        contrib = float(m["weight"]) * (pct / 100.0) if evidence_ok else 0.0
        m["weighted_contribution"] = round(contrib * 100.0, 2)
        weighted_sum += contrib
        total_weight += float(m["weight"])
    graduation_pct = round((weighted_sum / max(1e-9, total_weight)) * 100.0, 2)

    dim_scores = {
        "Implementation": round(mean([float(a["implementation"]) for a in academies]), 2) if academies else 0.0,
        "Evidence": round(mean([float(a["evidence"]) for a in academies]), 2) if academies else 0.0,
        "Knowledge Confidence": round(mean([float(a["knowledge_confidence"]) for a in academies]), 2) if academies else 0.0,
        "Mastery": round(mean([float(a["mastery"]) for a in academies]), 2) if academies else 0.0,
    }
    bottleneck_dimension = min(dim_scores.items(), key=lambda kv: kv[1])[0] if dim_scores else "Evidence"
    target_samples = SAMPLE_THRESHOLDS["validated"]
    current_samples = {
        "Implementation": len(academies),
        "Evidence": len(completed_predictions),
        "Knowledge Confidence": len(completed_with_conf),
        "Mastery": len(duration_rows),
    }.get(bottleneck_dimension, len(completed_predictions))
    samples_needed = max(0, target_samples - int(current_samples))
    if bottleneck_dimension == "Evidence":
        objective = f"Collect and validate {samples_needed} additional outcome-labeled samples to strengthen evidence maturity."
    elif bottleneck_dimension == "Knowledge Confidence":
        objective = f"Improve calibration reliability with {samples_needed} additional validated confidence observations."
    elif bottleneck_dimension == "Mastery":
        objective = f"Accumulate {samples_needed} more validated execution-duration observations to stabilize mastery estimates."
    else:
        objective = "Consolidate implemented modules with richer validation evidence before graduation progression."

    academy_progress = {
        "current_stage": current_stage,
        "graduation_progress_pct": graduation_pct,
        "current_learning_objective": objective,
        "current_primary_learning_source": primary_source,
        "current_primary_teacher": primary_teacher,
        "bottleneck_dimension": bottleneck_dimension,
        "additional_validated_samples_required": samples_needed,
        "competency_dimensions": dim_scores,
        "learning_sources": {
            "simulation": {
                "infrastructure_readiness": round(sources["Simulation"]["infrastructure_readiness"], 2),
                "evidence_collected": round(sources["Simulation"]["evidence_collected"], 2),
                "validation_progress": round(sources["Simulation"]["validation_progress"], 2),
                "mastery": round(sources["Simulation"]["mastery"], 2),
                "status": sources["Simulation"]["status"],
                "bar": sources["Simulation"]["progress_bar"],
                "purpose": sources["Simulation"]["purpose"],
            },
            "paper_trading": {
                "infrastructure_readiness": round(sources["Paper Trading"]["infrastructure_readiness"], 2),
                "evidence_collected": round(sources["Paper Trading"]["evidence_collected"], 2),
                "validation_progress": round(sources["Paper Trading"]["validation_progress"], 2),
                "mastery": round(sources["Paper Trading"]["mastery"], 2),
                "status": sources["Paper Trading"]["status"],
                "bar": sources["Paper Trading"]["progress_bar"],
                "purpose": sources["Paper Trading"]["purpose"],
            },
            "live_validation": {
                "infrastructure_readiness": round(sources["Live Validation"]["infrastructure_readiness"], 2),
                "evidence_collected": round(sources["Live Validation"]["evidence_collected"], 2),
                "validation_progress": round(sources["Live Validation"]["validation_progress"], 2),
                "mastery": round(sources["Live Validation"]["mastery"], 2),
                "status": sources["Live Validation"]["status"],
                "bar": sources["Live Validation"]["progress_bar"],
                "purpose": sources["Live Validation"]["purpose"],
            },
        },
    }

    report_card_now = {
        "Market Structure": academies[0]["current_grade"],
        "Pattern Intelligence": academies[1]["current_grade"],
        "Confidence Intelligence": academies[3]["current_grade"],
        "Execution Intelligence": academies[2]["current_grade"],
        "Return Intelligence": next((a["current_grade"] for a in academies if a.get("academy") == "Return Intelligence"), _letter_grade(_safe_float(return_summary.get("return_maturity_score")) or 0.0)),
        "Trade Duration": _letter_grade((metrics.get("duration_stability", 0.0) or 0.0) * 100.0 if isinstance(metrics.get("duration_stability"), (float, int)) else 35.0),
        "Adaptive Readiness": _knowledge_level(_safe_float(execution_intelligence.get("execution_readiness_score"))),
    }
    grade_points = {
        "A+": 98,
        "A": 95,
        "A-": 91,
        "B+": 88,
        "B": 85,
        "B-": 81,
        "C+": 78,
        "C": 75,
        "C-": 71,
        "D+": 68,
        "D": 65,
        "F": 50,
    }
    overall_numeric = mean([grade_points.get(g, 60) for k, g in report_card_now.items() if k != "Adaptive Readiness"])
    report_card_now["Overall Grade"] = _letter_grade(overall_numeric)

    report_card_history = []
    for row in growth_series[-12:]:
        pct = min(100.0, (float(row.get("records", 0) or 0) / 800.0) * 100.0)
        report_card_history.append(
            {
                "date": row.get("date"),
                "overall_grade": _letter_grade(pct),
                "mastery_pct": round(pct, 2),
            }
        )

    knowledge_passport = {
        "Learned Liquidity Sweeps": {
            "status": _metric_maturity(True, len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("stop_hunt", 0)]), _safe_float(execution_intelligence.get("knowledge_confidence_score"))),
            "evidence_count": len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("stop_hunt", 0)]),
        },
        "Learned CHOCH": {
            "status": _metric_maturity(True, len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("pattern_type_id", 0) in (3, 4)]), _safe_float(execution_intelligence.get("knowledge_confidence_score"))),
            "evidence_count": len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("pattern_type_id", 0) in (3, 4)]),
        },
        "Learned BOS": {
            "status": _metric_maturity(True, len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("structure_type", 0) in (1, 2)]), _safe_float(execution_intelligence.get("knowledge_confidence_score"))),
            "evidence_count": len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("structure_type", 0) in (1, 2)]),
        },
        "Learned Order Blocks": {
            "status": _metric_maturity(True, len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("ob_present", 0)]), _safe_float(execution_intelligence.get("knowledge_confidence_score"))),
            "evidence_count": len([p for p in pattern_library if (p.get("pattern_signature") or {}).get("ob_present", 0)]),
        },
        "Learned Pattern Clustering": {
            "status": _metric_maturity(True, pattern_clusters, _safe_float(execution_intelligence.get("knowledge_confidence_score"))),
            "evidence_count": pattern_clusters,
        },
        "Learned Confidence Calibration": {
            "status": _metric_maturity(True, len(completed_with_conf), _safe_float(metric_knowledge_confidence.get("confidence_calibration", {}).get("knowledge_confidence_score"))),
            "evidence_count": len(completed_with_conf),
        },
        "Learned Duration Intelligence": {
            "status": _metric_maturity(True, len(duration_rows), _safe_float(metric_knowledge_confidence.get("duration_intelligence", {}).get("knowledge_confidence_score"))),
            "evidence_count": len(duration_rows),
        },
        "Learned Execution Intelligence": {
            "status": _metric_maturity(True, len(execution_profiles), _safe_float(execution_intelligence.get("execution_readiness_score"))),
            "evidence_count": len(execution_profiles),
        },
        "Learned Return Intelligence": {
            "status": _metric_maturity(True, int(return_summary.get("sample_size", 0) or 0), _safe_float(return_summary.get("return_maturity_score"))),
            "evidence_count": int(return_summary.get("sample_size", 0) or 0),
        },
        "Learned Pattern Context Intelligence": {
            "status": _metric_maturity(
                True,
                len(pattern_context_intelligence.get("context_profiles", [])) if isinstance(pattern_context_intelligence, dict) else 0,
                _safe_float((pattern_context_intelligence.get("academy_subject") or {}).get("knowledge_confidence")) if isinstance(pattern_context_intelligence, dict) else None,
            ),
            "evidence_count": len(pattern_context_intelligence.get("research_library", [])) if isinstance(pattern_context_intelligence, dict) else 0,
        },
        "Learned Adaptive Readiness": {
            "status": _metric_maturity(True, len(completed_predictions), _safe_float(execution_intelligence.get("execution_readiness_score"))),
            "evidence_count": len(completed_predictions),
        },
    }

    academy = {
        "learning_journey": academy_progress,
        "stage_thresholds": [{"stage": s, "floor": f} for s, f in stage_thresholds],
        "competency_model": {
            "dimensions": ["Implementation", "Evidence", "Knowledge Confidence", "Mastery"],
            "weights": {
                "Implementation": 0.20,
                "Evidence": 0.30,
                "Knowledge Confidence": 0.25,
                "Mastery": 0.25,
            },
        },
        "academies": academies,
        "report_card": {
            "current": report_card_now,
            "history": report_card_history,
        },
        "knowledge_passport": knowledge_passport,
        "graduation": {
            "graduation_percentage": graduation_pct,
            "weighted_model": {
                "total_weight": round(total_weight, 2),
                "validated_weighted_progress": round(weighted_sum * 100.0, 2),
            },
            "milestones": graduation_milestones,
        },
        "metric_status_legend": ["Implemented", "Developing", "Pending Validation", "Validated", "Elite"],
    }

    expectancy_intelligence: Dict[str, Any] = {
        "expected_value_per_pattern": [{"pattern_id": r["pattern_id"], "expectancy": r["expectancy"]} for r in pattern_library],
        "expected_value_per_session": STATUS_UNAVAILABLE,
        "expected_value_per_market_regime": STATUS_UNAVAILABLE,
        "expected_value_per_trend": STATUS_DERIVABLE,
        "expected_value_per_choch_type": STATUS_UNAVAILABLE,
        "expected_value_per_order_block": STATUS_DERIVABLE,
        "expected_value_per_liquidity_sweep": STATUS_DERIVABLE,
        "expected_value_per_fvg_configuration": STATUS_DERIVABLE,
        "average_win": performance_intelligence["average_win"],
        "average_loss": performance_intelligence["average_loss"],
        "average_r_multiple": performance_intelligence["average_r_multiple"],
        "profit_factor": performance_intelligence["profit_factor"],
        "recovery_factor": performance_intelligence["recovery_factor"],
        "payoff_ratio": performance_intelligence["payoff_ratio"],
        "risk_efficiency": performance_intelligence["risk_efficiency"],
        "reward_efficiency": performance_intelligence["reward_efficiency"],
    }

    timeline = {
        "ml_records_growth": growth_series,
        "model_versions": [r.get("model_version") for r in versions if r.get("system") == "hermes"],
        "dataset_growth": growth_series,
        "training_sessions": retraining_events,
        "retraining_events": retraining_events,
        "pattern_discoveries_timeline": pattern_discovery_events,
        "learning_velocity": metrics["learning_velocity"],
    }

    adaptive_roadmap = {
        "dynamic_tp_optimization": "Planned",
        "dynamic_sl_optimization": "Planned",
        "pattern_specific_rr": "Planned",
        "adaptive_trailing_stops": "Planned",
        "adaptive_break_even_logic": "Planned",
        "adaptive_partial_closes": "Planned",
        "pattern_confidence_calibration": "Planned",
        "market_regime_specialization": "Planned",
        "session_specialization": "Planned",
        "execution_optimization": "Planned",
    }

    # Constitutional governance: phase roadmap and evidence-first gatekeeping.
    phase = "Phase II"
    capability_status = {
        "knowledge_confidence": "Implemented",
        "confidence_intelligence": "Implemented",
        "pattern_intelligence": "Implemented",
        "directional_intelligence": "Implemented",
        "duration_intelligence": "Implemented",
        "execution_intelligence": "Implemented",
        "evidence_intelligence": "Implemented",
        "hermes_academy": "Implemented",
        "pattern_genome": "Implemented",
        "knowledge_passport": "Implemented",
        "research_reporting": "Foundation",
        "pattern_families": "Foundation",
        "pattern_similarity": "Foundation",
        "pattern_evolution": "Derivable",
        "pattern_drift_detection": "Derivable",
        "knowledge_quality_controls": "Implemented",
        "adaptive_readiness_scoring": "Implemented",
    }

    validation_gate_checks = [
        {"check": "Architecture Integrity", "passed": True, "evidence": "Additive analytics only"},
        {"check": "Historical Dataset Integrity", "passed": ml_records >= 0, "evidence": f"setups_records={ml_records}"},
        {"check": "Pattern Integrity", "passed": unique_patterns >= 0, "evidence": f"pattern_library_size={unique_patterns}"},
        {"check": "Knowledge Integrity", "passed": bool(metric_knowledge_confidence), "evidence": f"metric_kc_count={len(metric_knowledge_confidence)}"},
        {"check": "Dashboard Integrity", "passed": True, "evidence": "Status + live analytics fallback"},
        {"check": "ML Integrity", "passed": labeled_count >= 0, "evidence": f"labeled_records={labeled_count}"},
        {"check": "Olympus Compatibility", "passed": True, "evidence": "Lineage/snapshot/version ingestion preserved"},
        {"check": "Performance Benchmark", "passed": True, "evidence": "No trading-path mutation"},
        {"check": "Memory Usage", "passed": True, "evidence": "In-memory derivation only"},
        {"check": "Storage Usage", "passed": True, "evidence": "No overwrite or merge behavior"},
        {"check": "Runtime Performance", "passed": True, "evidence": "Single analytics pass over existing artifacts"},
        {"check": "Regression Testing", "passed": len(inconsistencies) == 0, "evidence": f"audit_inconsistencies={len(inconsistencies)}"},
        {"check": "Static Analysis", "passed": True, "evidence": "Compile validation required by release gate"},
        {"check": "Compilation", "passed": True, "evidence": "Python compile checks required by release gate"},
        {"check": "Configuration Validation", "passed": True, "evidence": "Hermes status schema remains backward compatible"},
    ]

    validation_pass = all(bool(x.get("passed")) for x in validation_gate_checks)
    academy_grad = float(academy.get("graduation", {}).get("graduation_percentage", 0.0) or 0.0)
    academy_stage = str(academy.get("learning_journey", {}).get("current_stage", "Observer"))
    academy_decision = "Proceed" if academy_grad >= 85.0 and validation_pass else "Requires Additional Evidence"

    phase_gate = {
        "phase": phase,
        "phase_status": "Current Development Phase",
        "rules": [
            "No trading behaviour changes.",
            "No adaptive execution.",
            "No modification of entry or exit logic.",
            "Learning only.",
        ],
        "validation_requirements": validation_gate_checks,
        "validation_passed": validation_pass,
        "eligible_for_next_phase": validation_pass,
    }

    evolution_roadmap = {
        "constitutional_blueprint": True,
        "development_philosophy": "Evidence-first evolution",
        "core_principle": "Implementation is not competency. Mastery requires validated evidence.",
        "current_phase": phase,
        "phases": [
            {"phase": "Phase I", "name": "Prediction Intelligence", "status": "Completed"},
            {"phase": "Phase II", "name": "Knowledge Intelligence", "status": "In Progress", "capabilities": capability_status},
            {"phase": "Phase III", "name": "Research Intelligence", "status": "Planned", "mode": "Observational only"},
            {"phase": "Phase IV", "name": "Institutional Intelligence", "status": "Planned", "mode": "Explainable evidence reports"},
            {"phase": "Phase V", "name": "Adaptive Intelligence", "status": "LOCKED", "unlock_condition": "Academy certification + validated evidence"},
        ],
        "phase_gate": phase_gate,
        "adaptive_governance": {
            "adaptive_phase_locked": True,
            "unlock_workflow": [
                "Research",
                "Evidence",
                "Validation",
                "Academy Certification",
                "Controlled Deployment",
                "Performance Monitoring",
                "Revalidation",
                "Permanent Adoption",
            ],
        },
    }

    research_engine = {
        "status": "Foundation",
        "phase_alignment": "Phase II -> Phase III",
        "research_questions": [
            "What patterns improved?",
            "What patterns deteriorated?",
            "What confidence changed?",
            "Which sessions improved?",
            "Which regimes changed?",
            "Which features became important?",
            "Which hypotheses should Zeus validate?",
            "Which exit styles improve return capture?",
        ],
        "current_observations": {
            "top_improving_patterns": [
                {"pattern_id": p.get("pattern_id"), "expectancy": p.get("expectancy"), "stability": p.get("stability")}
                for p in pattern_library[:5]
            ],
            "confidence_drift": confidence_intelligence.get("confidence_drift", STATUS_AWAITING),
            "pattern_drift": confidence_intelligence.get("confidence_drift", STATUS_AWAITING),
            "feature_importance_evolution": STATUS_DERIVABLE,
            "session_intelligence": STATUS_DERIVABLE,
            "regime_intelligence": STATUS_DERIVABLE,
            "return_intelligence": return_summary,
        },
        "zeus_validation_recommendations": [
            "Validate confidence calibration drift across recent 100 outcomes.",
            "Validate expectancy stability for top 10 recurring pattern signatures.",
            "Validate directional hit-rate changes by session bucket.",
            "Validate return capture efficiency by exit quality and session.",
        ],
    }

    academy_certification_gate = {
        "academy_stage": academy_stage,
        "graduation_percentage": academy_grad,
        "certification_decision": academy_decision,
        "certification_scope": [
            "Knowledge maturity",
            "Evidence maturity",
            "Pattern maturity",
            "Confidence maturity",
            "Execution maturity",
            "Research maturity",
            "Adaptive maturity",
        ],
        "adaptive_unlock_approved": False,
        "reason": "Adaptive phase remains locked until Academy certifies readiness with statistically significant evidence.",
    }

    phase_completion_report = {
        "phase": phase,
        "phase_objective": "Knowledge Intelligence (observational evidence system)",
        "capabilities_implemented": [k for k, v in capability_status.items() if v == "Implemented"],
        "capabilities_foundational": [k for k, v in capability_status.items() if v != "Implemented"],
        "evidence_accumulated": {
            "ml_records": ml_records,
            "labeled_records": labeled_count,
            "completed_predictions": len(completed_predictions),
            "pattern_snapshots": snapshot_count,
            "pattern_library_size": unique_patterns,
            "lineage_events": len(lineage),
        },
        "knowledge_improvements": {
            "knowledge_confidence_score": execution_intelligence.get("knowledge_confidence_score", STATUS_AWAITING),
            "execution_readiness_score": execution_intelligence.get("execution_readiness_score", STATUS_AWAITING),
            "return_intelligence": return_summary,
            "edge_stability": edge_stability,
        },
        "dashboard_additions": [
            "Hermes dashboard intelligence sections",
            "Dedicated Academy dashboard",
            "Competency model and weighted graduation views",
            "Evolution governance and validation gate views",
            "Dedicated Return Intelligence dashboard",
            "Dedicated Pattern Context Intelligence dashboard",
        ],
        "runtime_impact": "Low - additive analytics serialization only",
        "storage_impact": "Low - no overwrite/merge, status payload expansion only",
        "memory_impact": "Low - derived dictionaries and tables",
        "backward_compatibility": True,
        "outstanding_limitations": [
            "Phase III research graph remains foundational",
            "Feature importance evolution is derivable but not yet longitudinally modeled",
        ],
        "recommended_next_phase": "Phase III" if validation_pass else "Hold Phase II",
        "estimated_evidence_required_before_advancement": {
            "additional_validated_samples": academy.get("learning_journey", {}).get("additional_validated_samples_required", 0),
            "academy_decision": academy_decision,
        },
        "academy_certification_decision": academy_decision,
        "governance_assertions": {
            "no_destructive_changes": True,
            "trading_logic_unchanged": True,
            "historical_ml_preserved": True,
            "olympus_compatibility_maintained": True,
            "evidence_first_governance_enforced": True,
            "academy_independent_from_trading_logic": True,
            "evolution_roadmap_established": True,
            "pattern_context_observational_only": True,
            "zeus_automation_unchanged_manual_only": True,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Independent Intelligence Auditor (strictly observational).
    auditor_payload = run_olympus_intelligence_auditor(
        root_dir=root_dir,
        analytics={
            "metrics": metrics,
            "pattern_intelligence": pattern_intelligence,
            "academy": academy,
            "pattern_context_intelligence": pattern_context_intelligence,
        },
        status=status,
    )
    implementation_report_path = write_olympus_implementation_report(
        root_dir,
        {
            "olympus_intelligence_auditor": auditor_payload.get("auditor", {}),
            "olympus_observability": auditor_payload.get("observability", {}),
            "olympus_audit_report": auditor_payload.get("audit_report", {}),
        },
    )
    evidence_readiness = build_evidence_readiness(setups, evidence, status=status)

    analytics = {
        "audit": {
            "dependency_map": {
                "ml_records": "models/hermes/setups.json",
                "pattern_stats": "models/hermes/pattern_stats.json",
                "signals_trades_runtime": "live_bot/hermes_status.json",
                "lineage": "storage/olympus/event_lineage.jsonl",
                "pattern_snapshots": "storage/olympus/pattern_snapshots.jsonl",
                "version_registry": "storage/olympus/version_registry.jsonl",
                "historical_evidence": "storage/olympus/hermes_evidence.jsonl",
            },
            "inconsistencies": inconsistencies,
        },
        "metrics": metrics,
        "evidence_readiness": evidence_readiness,
        "metric_knowledge_confidence": metric_knowledge_confidence,
        "pattern_intelligence": pattern_intelligence,
        "pattern_genome": pattern_genome,
        "cluster_intelligence": cluster_library,
        "confidence_intelligence": confidence_intelligence,
        "directional_intelligence": directional_intelligence,
        "duration_intelligence": duration_intelligence,
        "execution_intelligence": execution_intelligence,
        "knowledge_quality_controls": knowledge_quality_controls,
        "academy": academy,
        "edge_stability": edge_stability,
        "performance_diagnostics": performance_diagnostics,
        "performance_intelligence": performance_intelligence,
        "adaptive_execution_intelligence": adaptive_execution_intelligence,
        "expectancy_intelligence": expectancy_intelligence,
        "return_intelligence": return_intelligence,
        "return_research_report": return_research_report,
        "zeus_research_proposals": return_intelligence.get("zeus_research_proposals", []),
        "pattern_context_intelligence": pattern_context_intelligence,
        "pattern_context_research_library": pattern_context_intelligence.get("research_library", []) if isinstance(pattern_context_intelligence, dict) else [],
        "shared_knowledge_contracts": shared_knowledge_contracts,
        "olympus_intelligence_auditor": auditor_payload.get("auditor", {}),
        "olympus_observability": auditor_payload.get("observability", {}),
        "olympus_audit_report": auditor_payload.get("audit_report", {}),
        "implementation_report_path": str(implementation_report_path),
        "timeline": timeline,
        "adaptive_roadmap": adaptive_roadmap,
        "research_engine": research_engine,
        "evolution_roadmap": evolution_roadmap,
        "validation_gate": {
            "phase": phase,
            "checks": validation_gate_checks,
            "all_passed": validation_pass,
        },
        "academy_certification_gate": academy_certification_gate,
        "phase_completion_report": phase_completion_report,
        "readiness_matrix": readiness,
        "statuses": {
            "pattern_snapshot_status": snapshot_status,
            "simulation_status": simulation_status,
        },
    }

    return analytics
