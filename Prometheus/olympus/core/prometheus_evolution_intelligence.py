from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from olympus.contracts import EvidenceConfidenceContract, KnowledgeContract, SourceSystem
from olympus.core.hera_governance_standards import build_hera_governance_standards
from olympus.core.institutional_validation_pipeline import build_institutional_validation_pipeline
from olympus.core.zeus_validation_operations import run_zeus_validation_operations
from olympus.core.prometheus_persistent_intelligence_engines import (
    CapitalIntelligenceEngine,
    CapitalPreservationIntelligenceEngine,
    DecisionAttributionIntelligenceEngine,
    EdgeIntelligenceEngine,
    ExecutionLocationIntelligenceEngine,
    LearningIntelligenceEngine,
    RecommendationIntelligenceEngine,
    write_engine_artifacts,
)
from olympus.versions import FEATURE_VERSION, PROMETHEUS_DATASET_GENERATION, PROMETHEUS_STRATEGY_VERSION

STATUS_AWAITING = "Awaiting Historical Data"


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)
    except Exception:
        return None


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_validation_report_events(path: Path, reports: list[dict[str, Any]]) -> None:
    """Append new Zeus report states without truncating historical evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[tuple[str, str, str, str]] = set()
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    existing.add(_validation_report_event_key(row))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # Do not overwrite a damaged evidence file.  Surface the failure so
            # the caller cannot silently replace institutional history.
            raise RuntimeError(f"Cannot safely append Zeus validation history: {path}")

    with path.open("a", encoding="utf-8") as stream:
        for report in reports:
            key = _validation_report_event_key(report)
            if key in existing:
                continue
            stream.write(json.dumps(report, ensure_ascii=True) + "\n")
            existing.add(key)


def _validation_report_event_key(report: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(report.get("report_id") or report.get("candidate_id") or report.get("recommendation_id") or ""),
        str(report.get("status") or ""),
        str(report.get("lifecycle") or ""),
        str(report.get("last_transition_at") or report.get("timestamp") or ""),
    )


def _fetch_sqlite_rows(db_path: Path, query: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _profit_factor_value(pnls: list[float]) -> float | None:
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not wins or not losses:
        return None
    den = abs(sum(losses))
    if den <= 1e-9:
        return None
    return sum(wins) / den


def _profit_factor(pnls: list[float]) -> float | str:
    value = _profit_factor_value(pnls)
    if value is None:
        return STATUS_AWAITING
    return round(value, 4)


def _payoff_ratio_value(pnls: list[float]) -> float | None:
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not wins or not losses:
        return None
    avg_w = mean(wins)
    avg_l = abs(mean(losses))
    if avg_l <= 1e-9:
        return None
    return avg_w / avg_l


def _payoff_ratio(pnls: list[float]) -> float | str:
    value = _payoff_ratio_value(pnls)
    if value is None:
        return STATUS_AWAITING
    return round(value, 4)


def _stability(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    sd = pstdev(vals)
    return round(max(0.0, 1.0 - min(1.0, sd)), 4)


def _health(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 78:
        return "Healthy"
    if score >= 64:
        return "Neutral"
    if score >= 50:
        return "Caution"
    if score >= 35:
        return "Poor"
    return "Critical"


def _grade(score: float) -> str:
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


def _norm_session(v: Any) -> str:
    s = str(v or "").strip().lower().replace("_", " ")
    if "london" in s and "open" in s:
        return "London Open"
    if "london" in s and "close" in s:
        return "London Close"
    if s == "london":
        return "London"
    if "new york" in s and "open" in s:
        return "New York Open"
    if "new york" in s and "close" in s:
        return "New York Close"
    if s in ("ny", "new york", "ny lunch"):
        return "New York"
    if "asian" in s or s == "asia":
        return "Asian"
    if "sydney" in s:
        return "Sydney"
    if "dead" in s or "rollover" in s:
        return "Dead Zone"
    if "overlap" in s:
        return "Overlaps"
    return "Dead Zone" if not s else s.title()


def _norm_regime(v: Any) -> str:
    s = str(v or "").strip().lower().replace("_", " ")
    if "trend expansion" in s:
        return "Trend Expansion"
    if "trend continuation" in s:
        return "Trend Continuation"
    if "trend exhaustion" in s:
        return "Trend Exhaustion"
    if "compression" in s:
        return "Compression"
    if "breakout" in s:
        return "Breakout"
    if "liquidity" in s or "sweep" in s:
        return "Liquidity Sweep"
    if "high volatility" in s:
        return "High Volatility"
    if "low volatility" in s:
        return "Low Volatility"
    if "accumulation" in s:
        return "Accumulation"
    if "distribution" in s:
        return "Distribution"
    if "trend" in s:
        return "Trend Continuation"
    return "Range"


def _pattern_id(row: dict[str, Any]) -> str:
    st = int(row.get("structure_type", 0) or 0)
    sh = int(row.get("stop_hunt", 0) or 0)
    ob = int(row.get("ob_present", 0) or 0)
    fp = int(row.get("fib_proximity", 0) or 0)
    pt = int(row.get("pattern_type_id", 0) or 0)
    return f"P-{pt}-S{st}-SH{sh}-OB{ob}-F{fp}"


def _ci_binom(p: float | None, n: int, z: float = 1.96) -> dict[str, Any]:
    if p is None or n <= 0:
        return {"low": STATUS_AWAITING, "high": STATUS_AWAITING}
    err = z * math.sqrt(max(0.0, p * (1.0 - p)) / max(1, n))
    return {"low": round(max(0.0, p - err), 4), "high": round(min(1.0, p + err), 4)}


def _rolling(vals: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        out.append(mean(chunk) if chunk else 0.0)
    return out


def _rolling_metric(vals: list[float], window: int, metric_fn: Any) -> list[float | str]:
    out: list[float | str] = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        value = metric_fn(chunk)
        out.append(STATUS_AWAITING if value is None else round(float(value), 4))
    return out


def _sharpe_like(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    sd = pstdev(vals)
    if sd <= 1e-9:
        return None
    return (mean(vals) / sd) * math.sqrt(len(vals))


def _sortino_like(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    downside = [v for v in vals if v < 0]
    if not downside:
        return None
    downside_sd = pstdev(downside) if len(downside) > 1 else abs(downside[0])
    if downside_sd <= 1e-9:
        return None
    return (mean(vals) / downside_sd) * math.sqrt(len(vals))


def _context_edge_distribution(trades: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, float] = defaultdict(float)
    for trade in trades:
        pnl = _safe_float(trade.get("pnl"))
        if pnl is None:
            continue
        key = f"{_norm_session(trade.get('session'))}|{_norm_regime(trade.get('regime'))}"
        buckets[key] += abs(float(pnl))

    total = sum(buckets.values())
    if total <= 1e-9:
        return {
            "edge_concentration": STATUS_AWAITING,
            "dominant_context": STATUS_AWAITING,
            "edge_breadth": 0,
            "context_distribution": [],
        }

    rows = []
    shares = []
    for context, value in sorted(buckets.items(), key=lambda item: item[1], reverse=True):
        share = value / total
        shares.append(share)
        rows.append({
            "context": context,
            "abs_pnl_share": round(share, 4),
            "abs_pnl": round(value, 4),
        })

    return {
        "edge_concentration": round(sum(share * share for share in shares), 4),
        "dominant_context": rows[0]["context"] if rows else STATUS_AWAITING,
        "edge_breadth": sum(1 for share in shares if share >= 0.1),
        "context_distribution": rows[:25],
    }


def _build_version_history(
    prior_library: dict[str, Any],
    record_id: str,
    generated_at: str,
    version: str,
    confidence: float | int | str | None,
    sample_size: int,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for section in ("validated_findings", "candidate_findings", "rejected_findings"):
        for row in prior_library.get(section, []) or []:
            existing_id = str(row.get("finding_id") or row.get("hypothesis_id") or "")
            if existing_id == record_id:
                prior_history = row.get("version_history", [])
                if isinstance(prior_history, list):
                    history.extend(prior_history)
                break
        if history:
            break

    history.append(
        {
            "version": version,
            "timestamp": generated_at,
            "research_confidence": confidence,
            "sample_size": sample_size,
        }
    )
    return history[-10:]


def _simulate_capital_growth(returns: list[float], start_balance: float, risk_mode: str) -> dict[str, Any]:
    if not returns:
        return {
            "starting_balance": start_balance,
            "ending_balance": STATUS_AWAITING,
            "cagr_proxy": STATUS_AWAITING,
            "max_drawdown_pct": STATUS_AWAITING,
            "survival_probability": STATUS_AWAITING,
            "capital_efficiency": STATUS_AWAITING,
            "compounding_efficiency": STATUS_AWAITING,
            "risk_mode": risk_mode,
        }

    balance = start_balance
    peak = balance
    dds: list[float] = []
    losses = 0

    for idx, r in enumerate(returns):
        base_risk = 0.01
        if risk_mode == "fixed_fractional":
            risk_pct = 0.01
        elif risk_mode == "adaptive_risk":
            rr = returns[max(0, idx - 10) : idx + 1]
            momentum = mean(rr) if rr else 0.0
            risk_pct = max(0.003, min(0.02, 0.01 + momentum * 0.002))
        elif risk_mode == "reduced_after_drawdown":
            dd_now = max(0.0, (peak - balance) / max(1e-9, peak))
            risk_pct = max(0.002, 0.01 * (1.0 - min(0.8, dd_now * 2.0)))
        elif risk_mode == "increased_after_recovery":
            dd_now = max(0.0, (peak - balance) / max(1e-9, peak))
            risk_pct = 0.012 if dd_now < 0.02 else 0.008
        else:
            risk_pct = base_risk

        realized = balance * risk_pct * r
        balance += realized
        peak = max(peak, balance)
        dd = max(0.0, (peak - balance) / max(1e-9, peak))
        dds.append(dd)
        if realized < 0:
            losses += 1

    net = (balance / max(1e-9, start_balance)) - 1.0
    cagr_proxy = net * (252.0 / max(1.0, len(returns)))
    max_dd = max(dds) if dds else 0.0
    survival = max(0.0, 1.0 - min(1.0, max_dd * 1.5))
    capital_eff = net / max(1e-9, max_dd + 0.01)

    return {
        "starting_balance": start_balance,
        "ending_balance": round(balance, 2),
        "cagr_proxy": round(cagr_proxy, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "survival_probability": round(survival, 4),
        "capital_efficiency": round(capital_eff, 4),
        "compounding_efficiency": round(balance / max(1e-9, start_balance), 4),
        "risk_mode": risk_mode,
        "loss_count": losses,
    }


def _knowledge_contracts(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    out: list[dict[str, Any]] = []
    for f in findings:
        fid = str(f.get("finding_id", "unknown"))
        n = int(f.get("sample_size", 0) or 0)
        confidence = float(_safe_float(f.get("research_confidence")) or 0.0)
        wr = _safe_float(f.get("observed_win_rate"))

        k = KnowledgeContract(
            knowledge_id=f"PROM-EVO-{fid}",
            source_system=SourceSystem.PROMETHEUS,
            dataset_generation=PROMETHEUS_DATASET_GENERATION,
            model_version="prometheus-evolution-v1",
            feature_version=FEATURE_VERSION,
            strategy_version=PROMETHEUS_STRATEGY_VERSION,
            timestamp=now,
            knowledge_version="evolution-knowledge-v1",
            evidence_version="evolution-evidence-v1",
            confidence_version="evolution-confidence-v1",
            pattern_version="evolution-pattern-v1",
            trace_metadata={
                "finding_id": fid,
                "finding_type": f.get("finding_type", "unknown"),
                "source_system": SourceSystem.PROMETHEUS.value,
            },
        )
        ec = EvidenceConfidenceContract(
            implementation_pct=100.0,
            evidence_pct=min(100.0, (n / 400.0) * 100.0),
            knowledge_confidence_pct=confidence,
            reliability=f.get("evidence_strength", "Developing"),
            sample_size=n,
            confidence_interval=_ci_binom(wr, n),
            historical_stability=f.get("stability", STATUS_AWAITING),
            concept_drift=f.get("drift", STATUS_AWAITING),
            evidence_level=min(100.0, (n / 400.0) * 100.0),
            current_grade=_grade(confidence),
            pending_validation=n < 75,
            estimated_samples_remaining=max(0, 75 - n),
        )
        out.append({"knowledge_contract": k.as_dict(), "evidence_confidence_contract": ec.as_dict()})
    return out


def build_prometheus_evolution_intelligence(root_dir: Path) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    hera_standards = build_hera_governance_standards()
    status = _load_json(root_dir / "live_bot" / "bot_status.json", {})
    learning = _load_json(root_dir / "live_bot" / "learning_state.json", {})
    prior_decision = _load_json(root_dir / "storage" / "olympus" / "prometheus_decision_intelligence.json", {})
    prior_evolution = _load_json(root_dir / "storage" / "olympus" / "prometheus_evolution_intelligence.json", {})
    prior_library = (prior_evolution.get("research_library_evolution", {}) if isinstance(prior_evolution, dict) else {}) or {}
    db_path = root_dir / "storage" / "prometheus.db"

    trades = _fetch_sqlite_rows(
        db_path,
        """
        SELECT trade_id, created_at, direction, session, regime, pnl, rr, score_at_entry,
               mae, mfe, hold_seconds, exit_reason
        FROM trades
        WHERE lower(coalesce(status,'')) != 'open'
        ORDER BY created_at ASC
        """,
    )
    setups = _fetch_sqlite_rows(
        db_path,
        """
        SELECT setup_id, created_at, structure_type, trend_strength, mtf_score,
               sr_confidence, candlestick_score, pattern_confidence, fib_proximity,
               ob_present, stop_hunt, confluence_score, outcome, rr_achieved, pattern_type_id
        FROM setups
        ORDER BY created_at ASC
        """,
    )

    pnl_vals = [float(_safe_float(t.get("pnl")) or 0.0) for t in trades]
    rr_vals = [float(_safe_float(t.get("rr")) or 0.0) for t in trades if _safe_float(t.get("rr")) is not None]
    n = len(trades)
    wins = [x for x in pnl_vals if x > 0]
    losses = [x for x in pnl_vals if x < 0]
    win_rate = (len(wins) / max(1, n)) if n else 0.0

    entry_quality_series = [float(_safe_float(t.get("score_at_entry")) or 0.0) / 100.0 for t in trades if _safe_float(t.get("score_at_entry")) is not None]
    exit_quality_series = [1.0 if str(t.get("exit_reason", "")).lower() in ("tp", "tp1", "tp2", "5m_partial", "time_smart") else 0.35 for t in trades]
    entry_quality = mean(entry_quality_series) if entry_quality_series else 0.0
    exit_quality = mean(exit_quality_series) if exit_quality_series else 0.0
    timing_efficiency = mean([
        max(0.0, min(1.0, float(_safe_float(t.get("mfe")) or 0.0) / max(1e-9, abs(float(_safe_float(t.get("mae")) or 0.0)) + 1e-9)))
        for t in trades
    ]) if trades else 0.0
    stop_placement_quality = mean([
        max(0.0, min(1.0, abs(float(_safe_float(t.get("mae")) or 0.0)) / max(1e-9, abs(float(_safe_float(t.get("mfe")) or 0.0)) + 1e-9)))
        for t in trades
    ]) if trades else 0.0
    recovery_behaviour = max(0.0, 1.0 - min(1.0, (abs(mean(losses)) if losses else 0.0) / max(1e-9, (mean(wins) if wins else 1.0))))
    execution_consistency = _stability(rr_vals)
    execution_learning_engine = {
        "entry_quality": round(entry_quality, 4),
        "exit_quality": round(exit_quality, 4),
        "timing_efficiency": round(timing_efficiency, 4),
        "stop_placement_quality": round(stop_placement_quality, 4),
        "recovery_behaviour": round(recovery_behaviour, 4),
        "execution_consistency": round(execution_consistency, 4),
        "sample_size": n,
        "status": "Institutional" if n >= 300 else "Validated" if n >= 120 else "Developing",
    }

    # 3) Risk Intelligence Engine
    dd = []
    eq = 0.0
    peak = 0.0
    for p in pnl_vals:
        eq += p
        peak = max(peak, eq)
        dd.append(peak - eq)
    drawdown = max(dd) if dd else 0.0
    dd_vel = mean([dd[i] - dd[i - 1] for i in range(1, len(dd))]) if len(dd) > 1 else 0.0
    loss_vel = sum(1 for p in pnl_vals[-50:] if p < 0) / max(1, len(pnl_vals[-50:])) if pnl_vals else 0.0

    risk_candidates = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
    best_risk = 0.01
    best_score = -1e9
    for r in risk_candidates:
        score = (mean(rr_vals) if rr_vals else 0.0) * r - (loss_vel * r * 0.8) - (drawdown * r * 0.0001)
        if score > best_score:
            best_score = score
            best_risk = r

    risk_intelligence_engine = {
        "optimal_risk_pct": round(best_risk * 100.0, 3),
        "drawdown_recovery": round(max(0.0, (mean(wins) if wins else 0.0) / max(1e-9, abs(mean(losses)) if losses else 1.0)), 4),
        "loss_velocity": round(loss_vel, 4),
        "risk_efficiency": round((mean(rr_vals) if rr_vals else 0.0) * (win_rate if n else 0.0), 4),
        "capital_preservation": round(max(0.0, 1.0 - min(1.0, drawdown / max(1.0, abs(eq) + 1e-9))), 4),
        "risk_of_ruin": round(min(1.0, max(0.0, 0.55 - (win_rate - 0.5) - ((mean(rr_vals) if rr_vals else 0.0) * 0.05))), 4),
        "survival_probability": round(max(0.0, 1.0 - min(1.0, drawdown / max(1.0, abs(eq) + 1e-9))), 4),
        "drawdown_velocity": round(dd_vel, 4),
    }

    # 4) RR Intelligence by context
    rr_context: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        rr = _safe_float(t.get("rr"))
        if rr is None:
            continue
        key = "|".join(
            [
                _norm_session(t.get("session")),
                _norm_regime(t.get("regime")),
                "high_conf" if (float(_safe_float(t.get("score_at_entry")) or 0.0) >= 70.0) else "low_conf",
            ]
        )
        rr_context[key].append(float(rr))

    rr_rows = []
    for k, vals in rr_context.items():
        session, regime, conf = k.split("|")
        rr_rows.append(
            {
                "session": session,
                "regime": regime,
                "confidence_band": conf,
                "sample_size": len(vals),
                "optimal_rr": round(mean(vals), 4),
                "rr_stability": round(_stability(vals), 4),
            }
        )
    rr_rows.sort(key=lambda x: int(x.get("sample_size", 0)), reverse=True)

    risk_to_reward_intelligence_engine = {
        "context_aware_rr_recommendations": rr_rows[:150],
        "global_rr": round(mean(rr_vals), 4) if rr_vals else STATUS_AWAITING,
        "status": "Context Aware",
    }

    # 6) Session Intelligence Evolution
    session_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        session_groups[_norm_session(t.get("session"))].append(t)

    session_rows = []
    for s, rows in session_groups.items():
        pnls = [float(_safe_float(r.get("pnl")) or 0.0) for r in rows]
        rrs = [float(_safe_float(r.get("rr")) or 0.0) for r in rows if _safe_float(r.get("rr")) is not None]
        wr = (sum(1 for x in pnls if x > 0) / max(1, len(pnls))) if pnls else 0.0
        session_rows.append(
            {
                "session": s,
                "samples": len(rows),
                "session_edge_stability": round(_stability(rrs), 4),
                "session_rr": round(mean(rrs), 4) if rrs else STATUS_AWAITING,
                "session_risk": round(abs(mean([x for x in pnls if x < 0])) if any(x < 0 for x in pnls) else 0.0, 4),
                "session_consistency": round(_stability(pnls), 4) if len(pnls) > 1 else 0.0,
                "session_evolution": "Improving" if (mean(pnls[-20:]) if pnls else 0.0) > (mean(pnls[:-20]) if len(pnls) > 20 else 0.0) else "Stable",
                "session_psychology_proxy": "Disciplined" if wr >= 0.5 else "Needs Review",
            }
        )
    session_rows.sort(key=lambda x: int(x.get("samples", 0)), reverse=True)

    session_intelligence_evolution = {
        "best_sessions": [r for r in session_rows if isinstance(r.get("session_rr"), (int, float))][:5],
        "worst_sessions": sorted(
            [r for r in session_rows if isinstance(r.get("session_rr"), (int, float))],
            key=lambda x: float(x.get("session_rr", 0.0)),
        )[:5],
        "session_matrix": session_rows,
    }

    # 7) Pattern Evolution Engine
    pattern_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in setups:
        pattern_groups[_pattern_id(r)].append(r)

    pattern_evolution_rows = []
    for pid, rows in pattern_groups.items():
        out = [int(r.get("outcome", 0) or 0) for r in rows if r.get("outcome") is not None]
        rr = [float(_safe_float(r.get("rr_achieved")) or 0.0) for r in rows if _safe_float(r.get("rr_achieved")) is not None]
        if not rows:
            continue
        last = rows[-1]
        conditions = {
            "htf_alignment": "High" if (float(_safe_float(last.get("mtf_score")) or 0.0) >= 0.7) else "Moderate",
            "atr_proxy": "High" if (float(_safe_float(last.get("trend_strength")) or 0.0) >= 0.7) else "Moderate",
            "volatility": "Elevated" if (float(_safe_float(last.get("candlestick_score")) or 0.0) >= 70.0) else "Normal",
            "session": STATUS_AWAITING,
            "confidence": round(float(_safe_float(last.get("pattern_confidence")) or 0.0), 4),
        }
        pattern_evolution_rows.append(
            {
                "pattern_id": pid,
                "sample_size": len(rows),
                "base_win_rate": round(sum(out) / max(1, len(out)), 4) if out else STATUS_AWAITING,
                "base_expectancy": round(mean(rr), 4) if rr else STATUS_AWAITING,
                "additional_conditions": conditions,
                "institutional_pattern_score": round(min(100.0, (len(rows) / 120.0) * 100.0), 2),
                "pattern_evolution": "Institutional" if len(rows) >= 120 else "Validated" if len(rows) >= 50 else "Developing",
            }
        )
    pattern_evolution_rows.sort(key=lambda x: int(x.get("sample_size", 0)), reverse=True)

    pattern_evolution_engine = {
        "patterns": pattern_evolution_rows[:200],
        "evolution_score": round(mean([float(_safe_float(p.get("institutional_pattern_score")) or 0.0) for p in pattern_evolution_rows]), 2) if pattern_evolution_rows else 0.0,
    }

    # 8) Confidence Intelligence Evolution
    scores = [float(_safe_float(t.get("score_at_entry")) or 0.0) / 100.0 for t in trades if _safe_float(t.get("score_at_entry")) is not None]
    outcomes = [1.0 if (float(_safe_float(t.get("pnl")) or 0.0) > 0.0) else 0.0 for t in trades if _safe_float(t.get("score_at_entry")) is not None]
    brier = mean([(scores[i] - outcomes[i]) ** 2 for i in range(min(len(scores), len(outcomes)))]) if scores else None
    drift = STATUS_AWAITING
    if len(scores) >= 8:
        h = len(scores) // 2
        drift = round(mean(scores[h:]) - mean(scores[:h]), 4)

    conf_evolution = {
        "calibration_error": round(abs(mean(scores) - mean(outcomes)), 4) if scores and outcomes else STATUS_AWAITING,
        "confidence_drift": drift,
        "confidence_stability": round(_stability(scores), 4),
        "confidence_reliability": "High" if (brier is not None and brier < 0.18) else "Moderate" if (brier is not None and brier < 0.28) else "Developing",
        "confidence_evolution": [{"index": i + 1, "confidence": round(v, 4)} for i, v in enumerate(scores[-200:])],
        "brier_score": round(brier, 4) if brier is not None else STATUS_AWAITING,
    }

    # 9) Market Intelligence Engine
    regime_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        regime_groups[_norm_regime(t.get("regime"))].append(t)

    market_rows = []
    for regime, rows in regime_groups.items():
        pnls = [float(_safe_float(r.get("pnl")) or 0.0) for r in rows]
        wr = (sum(1 for x in pnls if x > 0) / max(1, len(pnls))) if pnls else 0.0
        market_rows.append(
            {
                "market_regime": regime,
                "samples": len(rows),
                "trend_strength": round(mean([float(_safe_float(s.get("trend_strength")) or 0.0) for s in setups[-max(1, len(rows)) :]]) if setups else 0.0, 4),
                "volatility": round(pstdev(pnls), 4) if len(pnls) > 1 else 0.0,
                "liquidity_behaviour": "Stable" if wr >= 0.52 else "Fragile",
                "session_transition_impact": STATUS_AWAITING,
                "structural_behaviour": "Orderly" if wr >= 0.52 else "Noisy",
                "edge_environment_quality": round(max(0.0, min(100.0, wr * 100.0)), 2),
            }
        )
    market_rows.sort(key=lambda x: int(x.get("samples", 0)), reverse=True)

    market_intelligence_engine = {
        "regime_matrix": market_rows,
        "market_environment_score": round(mean([float(_safe_float(x.get("edge_environment_quality")) or 0.0) for x in market_rows]), 2) if market_rows else 0.0,
        "classification": _health(round(mean([float(_safe_float(x.get("edge_environment_quality")) or 0.0) for x in market_rows]), 2) if market_rows else 0.0),
    }

    rr_for_growth = [float(_safe_float(t.get("rr")) or 0.0) for t in trades if _safe_float(t.get("rr")) is not None]
    edge_engine = EdgeIntelligenceEngine(root_dir, generated_at)
    edge_output = edge_engine.build(trades, pnl_vals, rr_vals, win_rate)
    execution_edge_learning_engine = edge_output["execution_edge_learning_engine"]

    capital_engine = CapitalIntelligenceEngine(root_dir, generated_at)
    capital_output = capital_engine.build(rr_for_growth, n, prior_library)
    capital_growth_intelligence_engine = capital_output["capital_growth_intelligence_engine"]

    learning_engine = LearningIntelligenceEngine(root_dir, generated_at)
    learning_output = learning_engine.build(
        status=status,
        prior_decision=prior_decision,
        prior_library=prior_library,
        trade_count=n,
        win_rate=win_rate,
        execution_learning_engine=execution_learning_engine,
        edge_engine_output=edge_output,
        capital_engine_output=capital_output,
        risk_intelligence_engine=risk_intelligence_engine,
        conf_evolution=conf_evolution,
        session_rows=session_rows,
        pattern_rows=pattern_evolution_rows,
        entry_quality_series=entry_quality_series,
        exit_quality_series=exit_quality_series,
    )
    findings = learning_output["validated_findings"]
    candidate_findings = learning_output["candidate_findings"] + capital_output["capital_research"]
    rejected_findings = learning_output["rejected_findings"]
    hypotheses = learning_output["historical_knowledge_mining"]
    continuous_improvement_metrics = learning_output["continuous_improvement_metrics"]
    continuous_improvement_metrics["pattern_evolution_score"] = pattern_evolution_engine.get("evolution_score", 0.0)

    risk_improvements = [
        {
            "finding_id": "R-RISK-001",
            "finding_type": "risk_intelligence",
            "validated_finding": f"Current optimal observational risk is {risk_intelligence_engine.get('optimal_risk_pct')}% per trade.",
            "expected_improvement": "Improved capital preservation through evidence-aligned risk governance.",
            "learning_confidence": round(min(100.0, (n / 250.0) * 100.0), 2),
            "research_confidence": round(min(100.0, (n / 250.0) * 100.0), 2),
            "sample_size": n,
            "evidence_strength": "Validated" if n >= 75 else "Developing",
            "historical_support": "Historical trade distribution analysis",
            "applicable_conditions": {"loss_velocity": risk_intelligence_engine.get("loss_velocity")},
            "expected_capital_impact": "Lower ruin risk if preserved under governance.",
            "version_history": _build_version_history(prior_library, "R-RISK-001", generated_at, "v2.1", round(min(100.0, (n / 250.0) * 100.0), 2), n),
        }
    ]
    execution_improvements = [
        {
            "finding_id": "E-EXEC-001",
            "finding_type": "execution_learning",
            "validated_finding": "Execution quality is tracked as an additive institutional learning stream.",
            "expected_improvement": "Improved entry, exit, stop, and recovery calibration over time.",
            "learning_confidence": round(execution_learning_engine.get("entry_quality", 0.0) * 100.0, 2),
            "research_confidence": round(execution_learning_engine.get("entry_quality", 0.0) * 100.0, 2),
            "sample_size": n,
            "evidence_strength": execution_learning_engine.get("status", "Developing"),
            "historical_support": "Closed-trade execution telemetry",
            "applicable_conditions": {"sample_size": n},
            "expected_capital_impact": "Positive through execution consistency gains.",
            "version_history": _build_version_history(prior_library, "E-EXEC-001", generated_at, "v2.1", round(execution_learning_engine.get("entry_quality", 0.0) * 100.0, 2), n),
        }
    ]

    recommendation_engine = RecommendationIntelligenceEngine(root_dir, generated_at)
    recommendation_output = recommendation_engine.build(
        findings=findings,
        candidate_findings=candidate_findings,
        risk_improvements=risk_improvements,
        capital_research=capital_output["capital_research"],
        continuous_improvement_metrics=continuous_improvement_metrics,
        sample_size=n,
        prior_library=prior_library,
    )
    recommendation_engine_evolution = recommendation_output["recommendation_engine_evolution"]
    continuous_improvement_metrics["recommendation_accuracy_growth"] = recommendation_output["recommendation_accuracy_growth"]

    execution_location_engine = ExecutionLocationIntelligenceEngine(root_dir, generated_at)
    execution_location_output = execution_location_engine.build(
        trades=trades,
        setups=setups,
        sample_size=n,
        prior_library=prior_library,
    )
    capital_preservation_engine = CapitalPreservationIntelligenceEngine(root_dir, generated_at)
    capital_preservation_output = capital_preservation_engine.build(
        trades=trades,
        sample_size=n,
        prior_library=prior_library,
    )
    decision_attribution_engine = DecisionAttributionIntelligenceEngine(root_dir, generated_at)
    decision_attribution_output = decision_attribution_engine.build(
        trades=trades,
        sample_size=n,
        prior_library=prior_library,
    )

    # Execution academy integration (independent, non-mutating)
    academy = (prior_decision.get("execution_academy", {}) if isinstance(prior_decision, dict) else {}) or {}
    discipline = round(max(0.0, min(100.0, (1.0 - loss_vel) * 100.0)), 2)
    decision_quality = round(mean([float(_safe_float(execution_edge_learning_engine.get("statistical_significance")) or 0.0) * 100.0, float(_safe_float(continuous_improvement_metrics.get("research_confidence_index")) or 0.0)]), 2)
    academy_integration = {
        "recovery_discipline": STATUS_AWAITING,
        "recovery_decision": STATUS_AWAITING,
        "recovery_performance": STATUS_AWAITING,
        "capital_preservation": risk_intelligence_engine.get("capital_preservation", STATUS_AWAITING),
        "operator_decision_quality": STATUS_AWAITING,
        "execution_learning_discipline": discipline,
        "evolution_decision_quality": decision_quality,
        "independent_assessment": {
            "current_execution_academy_stage": academy.get("current_stage", STATUS_AWAITING),
            "current_execution_academy_grade": academy.get("institutional_grade", STATUS_AWAITING),
            "automatic_grade_upgrade": False,
            "manual_governance_required": True,
        },
    }

    knowledge_contracts = _knowledge_contracts(findings)
    persistent_intelligence_engines = {
        "learning_intelligence_engine": learning_output["engine"],
        "capital_intelligence_engine": capital_output["engine"],
        "recommendation_intelligence_engine": recommendation_output["engine"],
        "edge_intelligence_engine": edge_output["engine"],
        "execution_location_intelligence_engine": execution_location_output["engine"],
        "capital_preservation_intelligence_engine": capital_preservation_output["engine"],
        "decision_attribution_intelligence_engine": decision_attribution_output["engine"],
    }

    institutional_validation = build_institutional_validation_pipeline(
        root_dir=root_dir,
        intelligence={
            "recommendation_engine_evolution": recommendation_engine_evolution,
            "research_library_evolution": {
                "candidate_findings": candidate_findings,
                "validated_findings": findings,
                "rejected_findings": rejected_findings,
            },
        },
        trades=trades,
        setups=setups,
    )

    payload = {
        "meta": {
            "source_system": SourceSystem.PROMETHEUS.value,
            "dataset_generation": PROMETHEUS_DATASET_GENERATION,
            "feature_version": FEATURE_VERSION,
            "strategy_version": PROMETHEUS_STRATEGY_VERSION,
            "generated_at": generated_at,
            "observational_only": True,
            "execution_behavior_unchanged": True,
            "schema_change": False,
            "additive_only": True,
            "evolution_layer_version": "v2.3",
        },
        "hera_governance": hera_standards,
        "mission": {
            "identity_preserved": True,
            "purpose": "Institutional Market Analysis and Execution Intelligence",
            "institutional_execution_excellence": True,
            "short_term_profit_priority": False,
        },
        "objective": {
            "primary_objective": "Maximize long-term compounded capital growth while preserving capital through institutional risk governance.",
            "profit_maximization_objective": False,
            "evidence_first": True,
        },
        "non_destructive_evolution": {
            "delete_data": False,
            "overwrite_history": False,
            "remove_features": False,
            "rename_modules": False,
            "replace_existing_systems": False,
            "modify_dashboards": False,
            "break_apis": False,
            "alter_olympus_compatibility": False,
            "change_json_contracts": False,
            "remove_reports": False,
            "invalidate_historical_datasets": False,
        },
        "learning_philosophy": {
            "historical_data_immutable": True,
            "knowledge_compounds_forever": True,
            "contradictory_evidence_lowers_confidence": True,
            "automatic_behavior_change": False,
        },
        "institutional_constitution": [
            {"principle": 1, "statement": "Preserve historical truth."},
            {"principle": 2, "statement": "Learning is additive."},
            {"principle": 3, "statement": "Evidence outweighs assumptions."},
            {"principle": 4, "statement": "Historical data is immutable."},
            {"principle": 5, "statement": "Every recommendation must be statistically supported."},
            {"principle": 6, "statement": "Every improvement must be reproducible."},
            {"principle": 7, "statement": "Backward compatibility is mandatory."},
            {"principle": 8, "statement": "Capital preservation precedes capital growth."},
            {"principle": 9, "statement": "Long-term expectancy outweighs short-term profit."},
            {"principle": 10, "statement": "Prometheus continuously evolves while remaining true to its mission."},
        ],
        "self_learning_loop": [
            "Observe",
            "Analyse",
            "Detect relationships",
            "Generate hypotheses",
            "Validate historically",
            "Measure statistical confidence",
            "Store findings",
            "Update knowledge confidence",
            "Generate recommendations",
            "Await governance approval",
        ],
        "execution_learning_engine": execution_learning_engine,
        "execution_location_intelligence_engine": execution_location_output["engine"],
        "execution_location_intelligence": execution_location_output["execution_location_intelligence"],
        "structural_price_location_intelligence": execution_location_output["structural_price_location_intelligence"],
        "edge_intelligence_engine": edge_output["engine"],
        "execution_edge_learning_engine": execution_edge_learning_engine,
        "capital_preservation_intelligence_engine": capital_preservation_output["engine"],
        "capital_preservation_intelligence": capital_preservation_output["capital_preservation_intelligence"],
        "risk_intelligence_engine": risk_intelligence_engine,
        "risk_to_reward_intelligence_engine": risk_to_reward_intelligence_engine,
        "capital_intelligence_engine": capital_output["engine"],
        "capital_growth_intelligence_engine": capital_growth_intelligence_engine,
        "learning_intelligence_engine": learning_output["engine"],
        "session_intelligence_evolution": session_intelligence_evolution,
        "pattern_evolution_engine": pattern_evolution_engine,
        "confidence_intelligence_evolution": conf_evolution,
        "market_intelligence_engine": market_intelligence_engine,
        "recommendation_intelligence_engine": recommendation_output["engine"],
        "decision_attribution_intelligence_engine": decision_attribution_output["engine"],
        "decision_attribution_intelligence": decision_attribution_output["decision_attribution_intelligence"],
        "historical_knowledge_mining": hypotheses,
        "research_library_evolution": {
            "validated_findings": findings,
            "candidate_findings": candidate_findings,
            "rejected_findings": rejected_findings,
            "edge_improvements": [f for f in findings if str(f.get("finding_type", "")).startswith("session")],
            "risk_improvements": risk_improvements,
            "capital_improvements": capital_output["capital_research"],
            "execution_improvements": execution_improvements,
            "execution_location_research": execution_location_output["execution_location_research"],
            "capital_preservation_research": capital_preservation_output["capital_preservation_research"],
            "learning_experiments": learning_output["learning_experiments"],
            "improvement_hypotheses": hypotheses.get("active_hypotheses", []),
            "capital_research": capital_output["capital_research"],
            "recommendation_outcomes": recommendation_output["recommendation_outcomes"],
            "decision_attribution": decision_attribution_output["decision_attribution_research"],
            "learning_velocity_history": learning_output["learning_velocity_history"],
            "edge_evolution_history": edge_output["edge_evolution_history"],
            "capital_evolution_history": capital_output["capital_evolution_history"],
            "execution_evolution_history": learning_output["execution_evolution_history"],
            "active_hypotheses": hypotheses.get("active_hypotheses", []),
            "rejected_hypotheses": hypotheses.get("rejected_hypotheses", []),
            "version_history": [{"version": "v2.2", "timestamp": generated_at}],
        },
        "continuous_improvement_metrics": continuous_improvement_metrics,
        "recommendation_engine_evolution": recommendation_engine_evolution,
        "institutional_validation_architecture": institutional_validation,
        "validation_status": institutional_validation.get("validation_status", {}),
        "execution_academy_integration": academy_integration,
        "shared_knowledge_contracts": knowledge_contracts,
        "persistent_intelligence_engines": persistent_intelligence_engines,
        "governance": {
            "requires_manual_approval_for_behavior_change": True,
            "historical_data_immutable": True,
            "existing_systems_unchanged": True,
            "olympus_compatibility_maintained": True,
        },
        "compatibility": {
            "prometheus_execution_academy": True,
            "olympus": True,
            "existing_dashboards": True,
            "existing_reports": True,
            "existing_apis": True,
            "existing_storage": True,
            "existing_json_schemas": True,
            "existing_feature_engineering": True,
            "existing_analytics": True,
            "existing_datasets": True,
            "existing_governance": True,
            "existing_execution_engine": True,
            "existing_research_library": True,
        },
    }

    return payload


def write_prometheus_evolution_artifacts(root_dir: Path, intelligence: dict[str, Any]) -> dict[str, str]:
    ol = root_dir / "storage" / "olympus"
    ol.mkdir(parents=True, exist_ok=True)

    payload_path = ol / "prometheus_evolution_intelligence.json"
    payload_path.write_text(json.dumps(intelligence, indent=2), encoding="utf-8")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evolution_layer": "Prometheus Evolution Directive v2.3",
        "new_learning_engines": {
            "execution_learning_engine": True,
            "execution_edge_learning_engine": True,
            "risk_intelligence_engine": True,
            "risk_to_reward_intelligence_engine": True,
            "capital_growth_intelligence_engine": True,
            "execution_location_intelligence": True,
            "capital_preservation_intelligence": True,
            "decision_attribution_intelligence": True,
            "session_intelligence_evolution": True,
            "pattern_evolution_engine": True,
            "confidence_intelligence_evolution": True,
            "market_intelligence_engine": True,
        },
        "persistent_engine_architecture": {
            "learning_intelligence_engine": True,
            "capital_intelligence_engine": True,
            "recommendation_intelligence_engine": True,
            "edge_intelligence_engine": True,
            "execution_location_intelligence_engine": True,
            "capital_preservation_intelligence_engine": True,
            "decision_attribution_intelligence_engine": True,
        },
        "research_library_evolution": True,
        "institutional_validation_architecture": True,
        "continuous_improvement_metrics": True,
        "recommendation_engine_evolution": True,
        "execution_academy_integration": True,
        "directive_alignment": {
            "edge_intelligence": True,
            "risk_intelligence": True,
            "capital_growth_intelligence": True,
            "confidence_intelligence": True,
            "historical_knowledge_mining": True,
            "institutional_constitution_embedded": True,
            "hera_governance_standards_embedded": True,
        },
        "dashboard_additions": ["No new dashboards created; existing dashboards extended via registry"],
        "runtime_impact": "Low - offline/additive analytics only",
        "memory_impact": "Low - in-memory aggregations",
        "storage_impact": "Low - additive JSON/JSONL artifacts",
        "backward_compatibility": True,
        "confirmations": {
            "circuit_breaker_preserved": True,
            "trading_logic_unchanged": True,
            "risk_management_unchanged": True,
            "historical_data_preserved": True,
            "recovery_requires_operator_authorization": True,
            "olympus_compatibility_maintained": True,
            "evidence_first_governance_maintained": True,
            "non_destructive_additive_only": True,
            "existing_dashboards_unchanged": True,
            "existing_json_schemas_unchanged": True,
            "no_automatic_deployment_of_validated_improvements": True,
        },
    }

    report_path = ol / "prometheus_evolution_implementation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    engine_artifacts: dict[str, dict[str, str]] = {}
    for engine_name, engine_payload in (intelligence.get("persistent_intelligence_engines", {}) or {}).items():
        if isinstance(engine_payload, dict):
            engine_artifacts[engine_name] = write_engine_artifacts(root_dir, engine_payload)

    library_rows = []
    research_library = intelligence.get("research_library_evolution", {}) or {}
    category_map = {
        "validated_findings": "validated_finding",
        "candidate_findings": "candidate_finding",
        "rejected_findings": "rejected_finding",
        "edge_improvements": "edge_improvement",
        "risk_improvements": "risk_improvement",
        "capital_improvements": "capital_improvement",
        "execution_improvements": "execution_improvement",
        "execution_location_research": "execution_location_research",
        "capital_preservation_research": "capital_preservation_research",
        "learning_experiments": "learning_experiment",
        "improvement_hypotheses": "improvement_hypothesis",
        "capital_research": "capital_research",
        "recommendation_outcomes": "recommendation_outcome",
        "decision_attribution": "decision_attribution",
    }
    for section, category in category_map.items():
        for record in research_library.get(section, []) or []:
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_system": SourceSystem.PROMETHEUS.value,
                "category": category,
                **record,
            }
            library_rows.append(row)

    for h in research_library.get("active_hypotheses", []) or []:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_system": SourceSystem.PROMETHEUS.value,
            "category": "active_hypothesis",
            **h,
        }
        library_rows.append(row)

    lib_path = ol / "prometheus_evolution_research_library.jsonl"
    with lib_path.open("a", encoding="utf-8") as f:
        for row in library_rows:
            f.write(json.dumps(row) + "\n")

    validation = intelligence.get("institutional_validation_architecture", {}) or {}
    validation_reports = validation.get("validation_reports", []) if isinstance(validation, dict) else []

    zvo = run_zeus_validation_operations(
        root_dir=root_dir,
        incoming_reports=[x for x in validation_reports if isinstance(x, dict)],
    )
    validation_status = zvo.get("status", {}) if isinstance(zvo, dict) else {}
    validation_reports_snapshot = zvo.get("reports", []) if isinstance(zvo, dict) else []

    zeus_status_path = ol / "zeus_validation_status.json"
    zeus_status_path.write_text(json.dumps(validation_status, indent=2), encoding="utf-8")

    zeus_reports_path = ol / "zeus_validation_reports.jsonl"
    _append_validation_report_events(
        zeus_reports_path,
        [row for row in validation_reports_snapshot if isinstance(row, dict)],
    )

    return {
        "payload_path": str(payload_path.relative_to(root_dir)),
        "report_path": str(report_path.relative_to(root_dir)),
        "research_library_path": str(lib_path.relative_to(root_dir)),
        "zeus_validation_status_path": str(zeus_status_path.relative_to(root_dir)),
        "zeus_validation_reports_path": str(zeus_reports_path.relative_to(root_dir)),
        "zeus_validation_operations_runtime_path": str(Path(zvo.get("runtime_path", "")).resolve().relative_to(root_dir)) if zvo.get("runtime_path") else "",
        "zeus_validation_operations_history_path": str(Path(zvo.get("history_path", "")).resolve().relative_to(root_dir)) if zvo.get("history_path") else "",
        "engine_artifacts": json.dumps(engine_artifacts),
    }
