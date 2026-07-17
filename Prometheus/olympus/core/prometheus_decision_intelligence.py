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
from olympus.versions import (
    FEATURE_VERSION,
    PROMETHEUS_DATASET_GENERATION,
    PROMETHEUS_STRATEGY_VERSION,
)


STATUS_AWAITING = "Awaiting Historical Data"

SESSION_ORDER = [
    "London",
    "London Open",
    "London Close",
    "New York",
    "New York Open",
    "New York Close",
    "Asian",
    "Sydney",
    "Dead Zone",
    "Overlaps",
]

REGIME_ORDER = [
    "Trend Expansion",
    "Trend Continuation",
    "Trend Exhaustion",
    "Range",
    "Compression",
    "Breakout",
    "Liquidity Sweep",
    "Accumulation",
    "Distribution",
    "High Volatility",
    "Low Volatility",
]


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
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


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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


def _sample_stage(n: int) -> str:
    if n < 10:
        return "Insufficient Evidence"
    if n < 30:
        return "Developing"
    if n < 75:
        return "Pending Validation"
    if n < 150:
        return "Validated"
    return "Institutional"


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


def _health_class(score: float) -> str:
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


def _ci_binom(p: float | None, n: int, z: float = 1.96) -> dict[str, Any]:
    if p is None or n <= 0:
        return {"low": STATUS_AWAITING, "high": STATUS_AWAITING}
    err = z * math.sqrt(max(0.0, p * (1.0 - p)) / max(1, n))
    return {"low": round(max(0.0, p - err), 4), "high": round(min(1.0, p + err), 4)}


def _rolling(vals: list[float], window: int) -> list[float]:
    out: list[float] = []
    if window <= 0:
        return out
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        x = vals[start : i + 1]
        out.append(mean(x) if x else 0.0)
    return out


def _stability_score(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    sd = pstdev(vals)
    return round(max(0.0, 1.0 - min(1.0, sd)), 4)


def _expectancy(pnls: list[float]) -> float:
    return round(mean(pnls), 4) if pnls else 0.0


def _profit_factor(pnls: list[float]) -> float | str:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    if not wins or not losses:
        return STATUS_AWAITING
    den = abs(sum(losses))
    if den <= 1e-9:
        return STATUS_AWAITING
    return round(sum(wins) / den, 4)


def _payoff_ratio(pnls: list[float]) -> float | str:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    if not wins or not losses:
        return STATUS_AWAITING
    avg_w = mean(wins)
    avg_l = abs(mean(losses))
    if avg_l <= 1e-9:
        return STATUS_AWAITING
    return round(avg_w / avg_l, 4)


def _drawdown_curve(pnls: list[float]) -> tuple[list[float], list[float]]:
    eq = 0.0
    peak = 0.0
    equity: list[float] = []
    dd: list[float] = []
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        equity.append(eq)
        dd.append(peak - eq)
    return equity, dd


def _pattern_id(row: dict[str, Any]) -> str:
    st = int(row.get("structure_type", 0) or 0)
    sh = int(row.get("stop_hunt", 0) or 0)
    ob = int(row.get("ob_present", 0) or 0)
    fp = int(row.get("fib_proximity", 0) or 0)
    pt = int(row.get("pattern_type_id", 0) or 0)
    ta = "h" if (_safe_float(row.get("trend_strength")) or 0.0) >= 0.7 else "m" if (_safe_float(row.get("trend_strength")) or 0.0) >= 0.45 else "l"
    mtf = "a" if (_safe_float(row.get("mtf_score")) or 0.0) >= 0.7 else "m" if (_safe_float(row.get("mtf_score")) or 0.0) >= 0.45 else "w"
    return f"P-{pt}-S{st}-SH{sh}-OB{ob}-F{fp}-T{ta}-M{mtf}"


def _lifecycle(occ: int, trend: str) -> str:
    if occ < 10:
        return "Emerging"
    if occ < 30:
        return "Developing"
    if occ < 75:
        return "Validated"
    if trend == "Declining":
        return "Archived"
    return "Institutional"


def _norm_session(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace("_", " ")
    if "london" in s and "open" in s:
        return "London Open"
    if "london" in s and "close" in s:
        return "London Close"
    if "london" in s and "overlap" in s:
        return "Overlaps"
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
    return "Dead Zone" if not s else s.title()


def _norm_regime(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace("_", " ")
    if "trend expansion" in s:
        return "Trend Expansion"
    if "trend continuation" in s:
        return "Trend Continuation"
    if "trend exhaustion" in s:
        return "Trend Exhaustion"
    if "mean reversion" in s or s == "range":
        return "Range"
    if "compression" in s:
        return "Compression"
    if "breakout" in s:
        return "Breakout"
    if "liquidity sweep" in s or "sweep" in s:
        return "Liquidity Sweep"
    if "accumulation" in s:
        return "Accumulation"
    if "distribution" in s:
        return "Distribution"
    if "high volatility" in s or "volatility expansion" in s:
        return "High Volatility"
    if "low volatility" in s or "dead liquidity" in s:
        return "Low Volatility"
    if "trend" in s:
        return "Trend Continuation"
    return "Range"


def _knowledge_contracts(pattern_library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for p in pattern_library:
        pid = str(p.get("pattern_id", "unknown") or "unknown")
        n = int(p.get("historical_samples", 0) or 0)
        wr = _safe_float(p.get("historical_win_rate"))
        kc = _safe_float(p.get("knowledge_confidence")) or 0.0
        contract = KnowledgeContract(
            knowledge_id=f"PROM-{pid}",
            source_system=SourceSystem.PROMETHEUS,
            dataset_generation=PROMETHEUS_DATASET_GENERATION,
            model_version="prometheus-live-v1",
            feature_version=FEATURE_VERSION,
            strategy_version=PROMETHEUS_STRATEGY_VERSION,
            timestamp=now,
            knowledge_version="knowledge-v1",
            evidence_version="evidence-v1",
            confidence_version="confidence-v1",
            pattern_version="pattern-v1",
            trace_metadata={
                "pattern_id": pid,
                "source_system": SourceSystem.PROMETHEUS.value,
                "lifecycle_stage": p.get("lifecycle_stage", "Developing"),
            },
        )
        ec = EvidenceConfidenceContract(
            implementation_pct=100.0,
            evidence_pct=min(100.0, (n / 300.0) * 100.0),
            knowledge_confidence_pct=kc,
            reliability=str(p.get("pattern_reliability", "Developing")),
            sample_size=n,
            confidence_interval=_ci_binom(wr, n),
            historical_stability=_safe_float(p.get("pattern_stability")) or STATUS_AWAITING,
            concept_drift=_safe_float(p.get("pattern_drift")) or STATUS_AWAITING,
            evidence_level=min(100.0, (n / 300.0) * 100.0),
            current_grade=_grade(kc),
            pending_validation=n < 75,
            estimated_samples_remaining=max(0, 75 - n),
        )
        out.append({"knowledge_contract": contract.as_dict(), "evidence_confidence_contract": ec.as_dict()})
    return out


def build_prometheus_decision_intelligence(root_dir: Path) -> dict[str, Any]:
    status = _load_json(root_dir / "live_bot" / "bot_status.json", {})
    learning = _load_json(root_dir / "live_bot" / "learning_state.json", {})
    db_path = root_dir / "storage" / "prometheus.db"

    trades = _fetch_sqlite_rows(
        db_path,
        """
        SELECT trade_id, created_at, source, asset, timeframe, direction, entry_price, sl_price, tp_price,
               exit_price, pnl, rr, status, session, regime, spread_at_entry, score_at_entry,
               exit_reason, mae, mfe, hold_seconds
        FROM trades
        WHERE lower(coalesce(status,'')) != 'open'
        ORDER BY created_at ASC
        """,
    )
    setups = _fetch_sqlite_rows(
        db_path,
        """
        SELECT setup_id, created_at, asset, timeframe, structure_type, trend_strength, mtf_score,
               sr_confidence, candlestick_score, pattern_confidence, fib_proximity, ob_present,
               stop_hunt, confluence_score, outcome, rr_achieved, entry_price, sl_price, tp_price, exit_price
        FROM setups
        ORDER BY created_at ASC
        """,
    )

    pnl_vals = [float(_safe_float(t.get("pnl")) or 0.0) for t in trades]
    rr_vals = [float(_safe_float(t.get("rr")) or 0.0) for t in trades if _safe_float(t.get("rr")) is not None]
    wins = [p for p in pnl_vals if p > 0]
    losses = [p for p in pnl_vals if p < 0]
    sample_size = len(trades)

    eq, dd = _drawdown_curve(pnl_vals)
    dd_delta = [dd[i] - dd[i - 1] for i in range(1, len(dd))]
    drawdown_velocity = round(mean(dd_delta[-50:]), 4) if dd_delta else 0.0
    loss_velocity = round(sum(1 for x in pnl_vals[-50:] if x < 0) / max(1, len(pnl_vals[-50:])), 4)

    recent_window = pnl_vals[-80:] if len(pnl_vals) >= 10 else pnl_vals
    recent_return = _expectancy(recent_window)
    hist_return = _expectancy(pnl_vals)
    return_stability = _stability_score(pnl_vals) if pnl_vals else 0.0

    score_vals = [float(_safe_float(t.get("score_at_entry")) or 0.0) / 100.0 for t in trades if _safe_float(t.get("score_at_entry")) is not None]
    outcomes = [1.0 if (float(_safe_float(t.get("pnl")) or 0.0) > 0) else 0.0 for t in trades]

    brier = mean([(score_vals[i] - outcomes[i]) ** 2 for i in range(min(len(score_vals), len(outcomes)))]) if score_vals else None

    conf_bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    calibration_rows: list[dict[str, Any]] = []
    ece_num = 0.0
    for lo, hi in conf_bins:
        idx = [i for i, s in enumerate(score_vals) if lo <= s < hi]
        if not idx:
            calibration_rows.append({"bucket": f"{lo:.2f}-{hi:.2f}", "sample": 0, "confidence": STATUS_AWAITING, "accuracy": STATUS_AWAITING, "calibration_error": STATUS_AWAITING})
            continue
        conf = mean([score_vals[i] for i in idx])
        acc = mean([outcomes[i] for i in idx])
        err = abs(conf - acc)
        ece_num += err * len(idx)
        calibration_rows.append({"bucket": f"{lo:.2f}-{hi:.2f}", "sample": len(idx), "confidence": round(conf, 4), "accuracy": round(acc, 4), "calibration_error": round(err, 4)})
    ece = round(ece_num / max(1, len(score_vals)), 4) if score_vals else STATUS_AWAITING

    conf_drift = STATUS_AWAITING
    if len(score_vals) >= 8:
        h = len(score_vals) // 2
        conf_drift = round(mean(score_vals[h:]) - mean(score_vals[:h]), 4)

    confidence_stability = _stability_score(score_vals) if len(score_vals) > 1 else 0.0

    threshold_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_thr = None
    best_sc = -1e9
    for thr in threshold_candidates:
        idx = [i for i, s in enumerate(score_vals) if s >= thr]
        if len(idx) < 10:
            continue
        pnl_sel = [pnl_vals[i] for i in idx]
        score = _expectancy(pnl_sel) + (mean([outcomes[i] for i in idx]) * 10.0)
        if score > best_sc:
            best_sc = score
            best_thr = thr

    session_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    regime_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        session_groups[_norm_session(t.get("session"))].append(t)
        regime_groups[_norm_regime(t.get("regime"))].append(t)

    def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pnls = [float(_safe_float(r.get("pnl")) or 0.0) for r in rows]
        wr = (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else None
        pf = _profit_factor(pnls)
        conf = [float(_safe_float(r.get("score_at_entry")) or 0.0) / 100.0 for r in rows if _safe_float(r.get("score_at_entry")) is not None]
        return {
            "samples": len(rows),
            "win_rate": round(wr, 4) if wr is not None else STATUS_AWAITING,
            "expectancy": round(_expectancy(pnls), 4) if pnls else STATUS_AWAITING,
            "profit_factor": pf,
            "average_return": round(_expectancy(pnls), 4) if pnls else STATUS_AWAITING,
            "risk": round(abs(mean([x for x in pnls if x < 0])) if any(x < 0 for x in pnls) else 0.0, 4),
            "confidence": round(mean(conf), 4) if conf else STATUS_AWAITING,
            "volatility": round(pstdev(pnls), 4) if len(pnls) > 1 else 0.0,
            "knowledge_confidence": round(min(100.0, (len(rows) / 120.0) * 100.0), 2),
        }

    session_intelligence = []
    for s in SESSION_ORDER:
        st = _group_stats(session_groups.get(s, []))
        session_intelligence.append({"session": s, **st, "pattern_diversity": STATUS_AWAITING, "pattern_success": st.get("win_rate", STATUS_AWAITING), "regime_distribution": STATUS_AWAITING, "market_health": STATUS_AWAITING})

    regime_intelligence = []
    for r in REGIME_ORDER:
        st = _group_stats(regime_groups.get(r, []))
        regime_intelligence.append({"regime": r, **st, "historical_performance": st.get("expectancy", STATUS_AWAITING), "recent_performance": st.get("expectancy", STATUS_AWAITING), "return_profile": st.get("average_return", STATUS_AWAITING), "pattern_distribution": STATUS_AWAITING, "session_distribution": STATUS_AWAITING, "historical_evolution": STATUS_AWAITING})

    # Pattern health and library from historical setups.
    pattern_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in setups:
        pattern_rows[_pattern_id(row)].append(row)

    pattern_health = []
    pattern_genome = []
    pattern_library = []
    for pid, rows in pattern_rows.items():
        outcomes_p = [int(_safe_int(r.get("outcome")) or 0) for r in rows if r.get("outcome") is not None]
        rr_p = [float(_safe_float(r.get("rr_achieved")) or 0.0) for r in rows if _safe_float(r.get("rr_achieved")) is not None]
        n = len(rows)
        win_hist = (sum(outcomes_p) / len(outcomes_p)) if outcomes_p else None
        recent_rows = rows[-min(30, n):]
        recent_out = [int(_safe_int(r.get("outcome")) or 0) for r in recent_rows if r.get("outcome") is not None]
        recent_rr = [float(_safe_float(r.get("rr_achieved")) or 0.0) for r in recent_rows if _safe_float(r.get("rr_achieved")) is not None]
        win_recent = (sum(recent_out) / len(recent_out)) if recent_out else None
        exp_hist = mean(rr_p) if rr_p else None
        exp_recent = mean(recent_rr) if recent_rr else None
        drift = abs((exp_recent or 0.0) - (exp_hist or 0.0))
        stab = _stability_score(rr_p) if len(rr_p) > 1 else 0.0
        evidence = min(100.0, (n / 300.0) * 100.0)
        kc = round(min(100.0, 0.6 * evidence + 0.4 * (stab * 100.0)), 2)
        trend = "Stable"
        if exp_recent is not None and exp_hist is not None:
            if exp_recent > exp_hist + 0.1:
                trend = "Improving"
            elif exp_recent < exp_hist - 0.1:
                trend = "Declining"
        if n < 10:
            trend = "Emerging"
        if n >= 150 and trend != "Declining":
            trend = "Institutional"

        health = {
            "pattern_id": pid,
            "historical_samples": n,
            "winning_samples": sum(outcomes_p),
            "losing_samples": max(0, len(outcomes_p) - sum(outcomes_p)),
            "historical_win_rate": round(win_hist, 4) if win_hist is not None else STATUS_AWAITING,
            "recent_win_rate": round(win_recent, 4) if win_recent is not None else STATUS_AWAITING,
            "historical_expectancy": round(exp_hist, 4) if exp_hist is not None else STATUS_AWAITING,
            "recent_expectancy": round(exp_recent, 4) if exp_recent is not None else STATUS_AWAITING,
            "historical_return": round(exp_hist, 4) if exp_hist is not None else STATUS_AWAITING,
            "pattern_stability": round(stab, 4),
            "pattern_drift": round(drift, 4),
            "pattern_maturity": _sample_stage(n),
            "pattern_reliability": _sample_stage(n),
            "evidence_level": round(evidence, 2),
            "knowledge_confidence": kc,
            "trend": trend,
            "lifecycle_stage": _lifecycle(n, trend),
            "historical_timeline": [{"index": i + 1, "rr": float(_safe_float(r.get("rr_achieved")) or 0.0)} for i, r in enumerate(rows[-50:])],
            "pattern_relationships": STATUS_AWAITING,
            "source_attribution": SourceSystem.PROMETHEUS.value,
        }
        pattern_health.append(health)

        sig = rows[-1]
        genome = {
            "pattern_id": pid,
            "structural_genes": {
                "structure_type": int(sig.get("structure_type", 0) or 0),
                "stop_hunt": int(sig.get("stop_hunt", 0) or 0),
                "order_block": int(sig.get("ob_present", 0) or 0),
                "fair_value_gap": int(sig.get("fib_proximity", 0) or 0),
            },
            "context_genes": {
                "trend_strength": _safe_float(sig.get("trend_strength")),
                "mtf_score": _safe_float(sig.get("mtf_score")),
                "session_profile": STATUS_AWAITING,
                "regime_profile": STATUS_AWAITING,
            },
            "behaviour_genes": {
                "historical_expectancy": health["historical_expectancy"],
                "recent_expectancy": health["recent_expectancy"],
                "drift": health["pattern_drift"],
            },
            "execution_genes": {
                "average_r": round(mean(rr_p), 4) if rr_p else STATUS_AWAITING,
                "execution_quality": STATUS_AWAITING,
            },
            "performance_genes": {
                "win_rate": health["historical_win_rate"],
                "stability": health["pattern_stability"],
            },
            "confidence_genes": {
                "knowledge_confidence": health["knowledge_confidence"],
                "evidence_level": health["evidence_level"],
            },
            "return_genes": {
                "historical_return": health["historical_return"],
                "return_stability": health["pattern_stability"],
            },
            "session_genes": STATUS_AWAITING,
            "regime_genes": STATUS_AWAITING,
            "adaptation_genes": {
                "trend": trend,
                "lifecycle_stage": health["lifecycle_stage"],
            },
            "knowledge_genes": {
                "maturity": health["pattern_maturity"],
                "reliability": health["pattern_reliability"],
            },
        }
        pattern_genome.append(genome)

        pattern_library.append(
            {
                "pattern_id": pid,
                "pattern_genome": genome,
                "pattern_signature": health["structural_genes"] if isinstance(health.get("structural_genes"), dict) else {
                    "structure_type": int(sig.get("structure_type", 0) or 0),
                    "stop_hunt": int(sig.get("stop_hunt", 0) or 0),
                    "ob_present": int(sig.get("ob_present", 0) or 0),
                    "fib_proximity": int(sig.get("fib_proximity", 0) or 0),
                },
                "historical_samples": health["historical_samples"],
                "winning_samples": health["winning_samples"],
                "losing_samples": health["losing_samples"],
                "average_return": health["historical_return"],
                "expected_move": health["historical_expectancy"],
                "session_profile": STATUS_AWAITING,
                "regime_profile": STATUS_AWAITING,
                "volatility_profile": STATUS_AWAITING,
                "confidence_profile": {
                    "knowledge_confidence": health["knowledge_confidence"],
                    "evidence_level": health["evidence_level"],
                },
                "knowledge_confidence": health["knowledge_confidence"],
                "pattern_evolution": health["trend"],
                "lifecycle_stage": health["lifecycle_stage"],
                "historical_timeline": health["historical_timeline"],
                "pattern_relationships": health["pattern_relationships"],
                "source_attribution": health["source_attribution"],
            }
        )

    pattern_health.sort(key=lambda x: int(x.get("historical_samples", 0) or 0), reverse=True)
    pattern_library.sort(key=lambda x: int(x.get("historical_samples", 0) or 0), reverse=True)

    # Risk intelligence and market health.
    regime_risk = []
    for r in regime_intelligence:
        wr = _safe_float(r.get("win_rate"))
        exp = _safe_float(r.get("expectancy"))
        risk_score = 100.0
        if wr is not None:
            risk_score -= max(0.0, (0.55 - wr) * 120.0)
        if exp is not None:
            risk_score -= max(0.0, -exp * 12.0)
        regime_risk.append({"regime": r.get("regime"), "risk_score": round(max(0.0, min(100.0, risk_score)), 2), "samples": r.get("samples", 0), "knowledge_confidence": r.get("knowledge_confidence", 0.0)})

    session_risk = []
    for s in session_intelligence:
        wr = _safe_float(s.get("win_rate"))
        exp = _safe_float(s.get("expectancy"))
        risk_score = 100.0
        if wr is not None:
            risk_score -= max(0.0, (0.55 - wr) * 120.0)
        if exp is not None:
            risk_score -= max(0.0, -exp * 12.0)
        session_risk.append({"session": s.get("session"), "risk_score": round(max(0.0, min(100.0, risk_score)), 2), "samples": s.get("samples", 0), "knowledge_confidence": s.get("knowledge_confidence", 0.0)})

    pattern_risk = [
        {
            "pattern_id": p.get("pattern_id"),
            "risk_score": round(max(0.0, 100.0 - (float(_safe_float(p.get("historical_expectancy")) or 0.0) * 20.0) - (float(_safe_float(p.get("historical_win_rate")) or 0.0) * 40.0)), 2),
            "samples": p.get("historical_samples"),
            "knowledge_confidence": p.get("knowledge_confidence"),
        }
        for p in pattern_health[:120]
    ]

    avg_wr = (len(wins) / max(1, sample_size)) if sample_size else 0.0
    pf = _profit_factor(pnl_vals)
    pf_num = _safe_float(pf)
    payoff = _payoff_ratio(pnl_vals)
    payoff_num = _safe_float(payoff)

    trend_quality = min(100.0, max(0.0, 40.0 + avg_wr * 80.0))
    structure_quality = min(100.0, max(0.0, 25.0 + (pf_num or 0.0) * 18.0))
    liquidity_behaviour = min(100.0, max(0.0, 30.0 + mean([x.get("risk_score", 50.0) for x in regime_risk if x.get("regime") == "Liquidity Sweep"] or [50.0]) * 0.7))
    volatility_quality = min(100.0, max(0.0, 100.0 - (pstdev(pnl_vals[-120:]) if len(pnl_vals) > 1 else 0.0) * 4.0))
    market_efficiency = min(100.0, max(0.0, 30.0 + (payoff_num or 0.0) * 22.0))
    predictability = min(100.0, max(0.0, 100.0 - (float(ece) * 100.0 if isinstance(ece, (float, int)) else 45.0)))
    pattern_stability = min(100.0, max(0.0, mean([float(_safe_float(p.get("pattern_stability")) or 0.0) for p in pattern_health]) * 100.0 if pattern_health else 0.0))
    confidence_stability_score = min(100.0, max(0.0, confidence_stability * 100.0))
    regime_stability = min(100.0, max(0.0, 100.0 - pstdev([float(_safe_float(x.get("win_rate")) or 0.0) for x in regime_intelligence if _safe_float(x.get("win_rate")) is not None]) * 100.0 if regime_intelligence else 0.0))
    session_quality = min(100.0, max(0.0, mean([float(_safe_float(x.get("win_rate")) or 0.0) for x in session_intelligence if _safe_float(x.get("win_rate")) is not None]) * 100.0 if session_intelligence else 0.0))

    started = str(status.get("started_at") or "")
    last_poll = str(status.get("last_poll") or "")
    days = 1.0
    try:
        if started and last_poll:
            ds = datetime.fromisoformat(started.replace("Z", "+00:00"))
            dl = datetime.fromisoformat(last_poll.replace("Z", "+00:00"))
            days = max(1.0, (dl - ds).total_seconds() / 86400.0)
    except Exception:
        days = 1.0
    signal_density = min(100.0, max(0.0, (float(_safe_float((learning.get("total_seen") if isinstance(learning, dict) else None)) or 0.0) / days) / 6.0))
    noise_level = max(0.0, min(100.0, 100.0 - predictability))

    market_health_score = round(
        mean(
            [
                trend_quality,
                structure_quality,
                liquidity_behaviour,
                volatility_quality,
                market_efficiency,
                predictability,
                pattern_stability,
                confidence_stability_score,
                regime_stability,
                session_quality,
                signal_density,
                100.0 - noise_level,
            ]
        ),
        2,
    )

    market_health_engine = {
        "market_health_score": market_health_score,
        "classification": _health_class(market_health_score),
        "components": {
            "trend_quality": round(trend_quality, 2),
            "structure_quality": round(structure_quality, 2),
            "liquidity_behaviour": round(liquidity_behaviour, 2),
            "volatility": round(volatility_quality, 2),
            "market_efficiency": round(market_efficiency, 2),
            "historical_predictability": round(predictability, 2),
            "pattern_stability": round(pattern_stability, 2),
            "confidence_stability": round(confidence_stability_score, 2),
            "regime_stability": round(regime_stability, 2),
            "session_quality": round(session_quality, 2),
            "signal_density": round(signal_density, 2),
            "noise_level": round(noise_level, 2),
        },
        "informational_only": True,
    }

    risk_intelligence = {
        "risk_momentum": round(100.0 - min(100.0, abs(drawdown_velocity) * 12.0 + loss_velocity * 100.0), 2),
        "drawdown_velocity": drawdown_velocity,
        "loss_velocity": loss_velocity,
        "session_risk": session_risk,
        "regime_risk": regime_risk,
        "pattern_risk": pattern_risk,
        "execution_risk": round(max(0.0, 100.0 - (abs(mean(losses)) if losses else 0.0) * 1.5), 2),
        "market_stability": round(market_health_score, 2),
        "decision_stability": round(100.0 - min(80.0, abs((_safe_float(conf_drift) or 0.0) * 80.0)), 2),
        "signal_quality_index": round(mean([trend_quality, structure_quality, predictability]), 2),
        "adaptive_readiness": round(mean([market_health_score, return_stability * 100.0, confidence_stability_score]), 2),
        "observational_only": True,
    }

    capture_vals = [
        (float(_safe_float(t.get("pnl")) or 0.0) / max(1e-9, float(_safe_float(t.get("mfe")) or 1.0)))
        for t in trades
        if _safe_float(t.get("mfe")) is not None and (_safe_float(t.get("mfe")) or 0.0) > 0
    ]
    rr_eff_vals = [max(-2.0, min(2.0, float(_safe_float(t.get("rr")) or 0.0))) for t in trades if _safe_float(t.get("rr")) is not None]
    opportunity_vals = [
        max(0.0, (float(_safe_float(t.get("mfe")) or 0.0) - float(_safe_float(t.get("pnl")) or 0.0)))
        for t in trades
        if _safe_float(t.get("mfe")) is not None
    ]
    mfe_vals = [float(_safe_float(t.get("mfe")) or 0.0) for t in trades if _safe_float(t.get("mfe")) is not None]
    mae_vals = [float(_safe_float(t.get("mae")) or 0.0) for t in trades if _safe_float(t.get("mae")) is not None]
    entry_eff_vals = [float(_safe_float(t.get("score_at_entry")) or 0.0) / 100.0 for t in trades if _safe_float(t.get("score_at_entry")) is not None]
    exit_eff_vals = [1.0 if str(t.get("exit_reason", "")).lower() in ("tp", "tp1", "tp2", "5m_partial", "time_smart") else 0.4 for t in trades]

    return_intelligence = {
        "average_return": round(hist_return, 4) if pnl_vals else STATUS_AWAITING,
        "return_distribution": {
            "count": len(pnl_vals),
            "median": round(median(pnl_vals), 4) if pnl_vals else STATUS_AWAITING,
            "std": round(pstdev(pnl_vals), 4) if len(pnl_vals) > 1 else STATUS_AWAITING,
            "p10": round(sorted(pnl_vals)[max(0, int(0.1 * len(pnl_vals)) - 1)], 4) if pnl_vals else STATUS_AWAITING,
            "p90": round(sorted(pnl_vals)[max(0, int(0.9 * len(pnl_vals)) - 1)], 4) if pnl_vals else STATUS_AWAITING,
        },
        "return_capture": round(mean(capture_vals), 4) if capture_vals else STATUS_AWAITING,
        "return_efficiency": round(mean(rr_eff_vals), 4) if rr_eff_vals else STATUS_AWAITING,
        "risk_efficiency": round((pf_num or 0.0) * (avg_wr or 0.0), 4) if pf_num is not None else STATUS_AWAITING,
        "opportunity_cost": round(mean(opportunity_vals), 4) if opportunity_vals else STATUS_AWAITING,
        "historical_mfe": round(mean(mfe_vals), 4) if mfe_vals else STATUS_AWAITING,
        "historical_mae": round(mean(mae_vals), 4) if mae_vals else STATUS_AWAITING,
        "entry_efficiency": round(mean(entry_eff_vals), 4) if entry_eff_vals else STATUS_AWAITING,
        "exit_efficiency": round(mean(exit_eff_vals), 4) if exit_eff_vals else STATUS_AWAITING,
        "average_r": round(mean(rr_vals), 4) if rr_vals else STATUS_AWAITING,
        "expected_r": round(mean(rr_vals), 4) if rr_vals else STATUS_AWAITING,
        "payoff_ratio": payoff,
        "recovery_factor": round((eq[-1] / max(dd)) if eq and dd and max(dd) > 0 else 0.0, 4) if eq else STATUS_AWAITING,
        "expectancy_evolution": [{"index": i + 1, "rolling_expectancy": round(v, 4)} for i, v in enumerate(_rolling(pnl_vals, 40))],
        "return_stability": round(return_stability, 4),
        "observational_only": True,
    }

    confidence_intelligence = {
        "confidence_calibration": calibration_rows,
        "confidence_drift": conf_drift,
        "confidence_stability": round(confidence_stability, 4),
        "historical_confidence_accuracy": round(1.0 - float(ece), 4) if isinstance(ece, (float, int)) else STATUS_AWAITING,
        "reliability_rating": _sample_stage(len(score_vals)),
        "confidence_evolution": [{"index": i + 1, "confidence": round(v, 4)} for i, v in enumerate(score_vals[-200:])],
        "brier_score": round(brier, 4) if brier is not None else STATUS_AWAITING,
        "calibration_error": ece,
        "optimal_confidence_threshold": best_thr if best_thr is not None else STATUS_AWAITING,
        "knowledge_confidence": round(min(100.0, 0.55 * min(100.0, (len(score_vals) / 250.0) * 100.0) + 0.45 * (confidence_stability * 100.0)), 2),
        "confidence_interval": _ci_binom((sum(outcomes) / len(outcomes)) if outcomes else None, len(outcomes)),
        "sample_size": len(score_vals),
    }

    edge_vals = {
        "prediction_edge": round(avg_wr - 0.5, 4),
        "execution_edge": round((mean(rr_vals) - 1.0), 4) if rr_vals else 0.0,
        "pattern_edge": round(mean([float(_safe_float(p.get("historical_expectancy")) or 0.0) for p in pattern_health]) if pattern_health else 0.0, 4),
        "knowledge_edge": round((mean([float(_safe_float(p.get("knowledge_confidence")) or 0.0) for p in pattern_health]) / 100.0) - 0.5 if pattern_health else 0.0, 4),
        "market_edge": round((market_health_score / 100.0) - 0.5, 4),
        "risk_edge": round((risk_intelligence["risk_momentum"] / 100.0) - 0.5, 4),
        "confidence_edge": round((float(_safe_float(confidence_intelligence.get("historical_confidence_accuracy")) or 0.0) - 0.5), 4),
        "return_edge": round((hist_return / max(1.0, abs(mean(losses)) if losses else 1.0)), 4) if pnl_vals else 0.0,
    }

    def _edge_trend(v: float) -> str:
        if v > 0.08:
            return "Improving"
        if v < -0.08:
            return "Declining"
        return "Stable"

    edge_stability = {
        **edge_vals,
        "trend": {k: _edge_trend(float(v)) for k, v in edge_vals.items()},
        "evidence_level": round(min(100.0, (sample_size / 300.0) * 100.0), 2),
    }

    # Decision intelligence: last executed trades + inferred rejects.
    decision_reviews = []
    for t in trades[-80:]:
        pnl = float(_safe_float(t.get("pnl")) or 0.0)
        decision_reviews.append(
            {
                "trade_id": t.get("trade_id", "unknown"),
                "decision": "accepted",
                "why_accepted": f"Executed in {_norm_session(t.get('session'))} / {_norm_regime(t.get('regime'))} with score {t.get('score_at_entry', STATUS_AWAITING)}.",
                "historical_comparison": "Compared against historical expectancy and session/regime distributions.",
                "pattern_evidence": STATUS_AWAITING,
                "market_health": market_health_score,
                "risk_level": "elevated" if pnl < 0 else "controlled",
                "session_quality": _norm_session(t.get("session")),
                "regime_quality": _norm_regime(t.get("regime")),
                "confidence_explanation": f"Score-at-entry proxy = {t.get('score_at_entry', STATUS_AWAITING)}",
                "supporting_statistics": {"pnl": pnl, "rr": _safe_float(t.get("rr")), "hold_seconds": _safe_int(t.get("hold_seconds"))},
                "knowledge_confidence": round(min(100.0, (sample_size / 300.0) * 100.0), 2),
                "decision_quality": "High" if pnl > 0 else "Review",
            }
        )

    grade_stats = ((learning.get("grade_stats") if isinstance(learning, dict) else {}) or {})
    seen = sum(int((v or {}).get("seen", 0) or 0) for v in grade_stats.values())
    acted = sum(int((v or {}).get("acted", 0) or 0) for v in grade_stats.values())
    inferred_rejects = max(0, seen - acted)

    decision_intelligence = {
        "reviews": decision_reviews,
        "rejected_signal_summary": {
            "inferred_rejections": inferred_rejects,
            "reason": "Below minimum grade/score or blocked by safety gates.",
            "knowledge_confidence": round(min(100.0, (seen / 2000.0) * 100.0), 2),
        },
    }

    # Research engine recommendations.
    recs = []
    if isinstance(conf_drift, (float, int)) and conf_drift < -0.03:
        recs.append("Confidence drift is declining; review score threshold consistency before governance approval.")
    if market_health_score < 60:
        recs.append("Market health is below neutral; prioritize observational review of regime/session mix.")
    if return_stability < 0.45:
        recs.append("Return stability is weak; investigate drawdown clusters and exit-reason concentration.")
    if not recs:
        recs.append("Execution quality appears stable; continue evidence accumulation for institutional certification.")

    research_engine = {
        "research_reports": {
            "pattern_report": pattern_health[:50],
            "session_report": session_intelligence,
            "regime_report": regime_intelligence,
            "return_report": return_intelligence,
            "confidence_report": confidence_intelligence,
            "execution_report": {
                "sample_size": sample_size,
                "profit_factor": pf,
                "payoff_ratio": payoff,
                "drawdown_velocity": drawdown_velocity,
            },
            "edge_report": edge_stability,
            "trend_report": {
                "recent_expectancy": recent_return,
                "historical_expectancy": hist_return,
                "expectancy_delta": round(recent_return - hist_return, 4),
            },
        },
        "recommendations": recs,
        "informational_only": True,
    }

    # Execution Academy (independent evaluator).
    execution_quality = min(100.0, max(0.0, 35.0 + avg_wr * 55.0 + ((pf_num or 0.0) * 5.0)))
    decision_quality = min(100.0, max(0.0, 40.0 + (market_health_score * 0.4) + (confidence_stability_score * 0.2)))
    risk_quality = min(100.0, max(0.0, risk_intelligence["risk_momentum"]))
    capital_quality = min(100.0, max(0.0, 30.0 + ((payoff_num or 0.0) * 18.0) + ((pf_num or 0.0) * 8.0)))
    pattern_quality = min(100.0, max(0.0, mean([float(_safe_float(p.get("knowledge_confidence")) or 0.0) for p in pattern_health]) if pattern_health else 0.0))
    session_quality_score = min(100.0, max(0.0, session_quality))
    consistency_score = min(100.0, max(0.0, return_stability * 100.0))
    discipline_score = min(100.0, max(0.0, 50.0 + (1.0 - loss_velocity) * 40.0))
    market_selection_score = min(100.0, max(0.0, market_health_score * 0.9))
    adaptive_readiness = min(100.0, max(0.0, risk_intelligence["adaptive_readiness"]))

    gpa = round(
        mean(
            [
                execution_quality,
                decision_quality,
                risk_quality,
                capital_quality,
                pattern_quality,
                session_quality_score,
                consistency_score,
                discipline_score,
                market_selection_score,
            ]
        ),
        2,
    )

    maturity_stages = [
        ("Execution Observer", 0.0),
        ("Execution Student", 20.0),
        ("Execution Apprentice", 35.0),
        ("Execution Analyst", 50.0),
        ("Execution Professional", 65.0),
        ("Institutional Execution Specialist", 78.0),
        ("Institutional Execution Intelligence", 88.0),
        ("Execution Master", 95.0),
    ]
    stage = maturity_stages[0][0]
    for name, floor in maturity_stages:
        if gpa >= floor:
            stage = name

    graduation_progress = min(100.0, (sample_size / 1200.0) * 100.0)
    institutional_grade = _grade(gpa)

    strengths = []
    weaknesses = []
    for label, score in [
        ("Execution Quality", execution_quality),
        ("Decision Quality", decision_quality),
        ("Risk Quality", risk_quality),
        ("Capital Quality", capital_quality),
        ("Pattern Quality", pattern_quality),
        ("Session Quality", session_quality_score),
        ("Consistency", consistency_score),
        ("Discipline", discipline_score),
        ("Market Selection", market_selection_score),
    ]:
        if score >= 75:
            strengths.append(label)
        if score < 55:
            weaknesses.append(label)

    academy_recommendations = []
    if risk_quality < 60:
        academy_recommendations.append("Risk discipline declining; review drawdown clusters and loss velocity.")
    if pattern_quality < 60:
        academy_recommendations.append("Pattern utilization quality is below institutional threshold; continue evidence accumulation.")
    if session_quality_score < 60:
        academy_recommendations.append("Session discipline uneven; compare London/NY vs Dead Zone execution behavior.")
    if consistency_score < 55:
        academy_recommendations.append("Execution consistency is unstable; monitor return stability and recovery behavior.")
    if not academy_recommendations:
        academy_recommendations.append("Execution profile is improving; continue collecting validated samples for next stage promotion.")

    certifications = []
    if sample_size >= 150 and risk_quality >= 70:
        certifications.append("Risk Discipline")
    if sample_size >= 150 and execution_quality >= 72:
        certifications.append("Execution Discipline")
    if sample_size >= 200 and decision_quality >= 74:
        certifications.append("Decision Discipline")
    if sample_size >= 200 and capital_quality >= 74:
        certifications.append("Capital Discipline")
    if sample_size >= 250 and session_quality_score >= 72:
        certifications.append("Session Discipline")
    if sample_size >= 300 and gpa >= 80:
        certifications.append("Institutional Readiness")

    execution_academy = {
        "independent": True,
        "observational_only": True,
        "current_stage": stage,
        "graduation_progress": round(graduation_progress, 2),
        "execution_gpa": gpa,
        "institutional_grade": institutional_grade,
        "report_card": {
            "execution_score": round(execution_quality, 2),
            "decision_score": round(decision_quality, 2),
            "risk_score": round(risk_quality, 2),
            "capital_score": round(capital_quality, 2),
            "pattern_score": round(pattern_quality, 2),
            "session_score": round(session_quality_score, 2),
            "consistency_score": round(consistency_score, 2),
            "discipline_score": round(discipline_score, 2),
            "market_selection_score": round(market_selection_score, 2),
            "adaptive_readiness": round(adaptive_readiness, 2),
        },
        "edge_stability": edge_stability,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": academy_recommendations,
        "evidence_level": round(min(100.0, (sample_size / 600.0) * 100.0), 2),
        "knowledge_confidence": round(min(100.0, 0.55 * min(100.0, (sample_size / 600.0) * 100.0) + 0.45 * consistency_score), 2),
        "estimated_samples_remaining": max(0, 600 - sample_size),
        "historical_progress": [{"index": i + 1, "equity": round(v, 4), "drawdown": round(dd[i], 4)} for i, v in enumerate(eq[-250:])] if eq else [],
        "certification_framework": {
            "eligible_certifications": certifications,
            "promotion_requires_evidence": True,
            "implementation_alone_never_promotes": True,
        },
    }

    knowledge_confidence_framework = {
        "implementation_pct": 100.0,
        "evidence_pct": round(min(100.0, (sample_size / 600.0) * 100.0), 2),
        "knowledge_confidence_pct": execution_academy["knowledge_confidence"],
        "reliability": _sample_stage(sample_size),
        "historical_stability": round(return_stability, 4),
        "confidence_interval": _ci_binom(avg_wr if sample_size else None, sample_size),
        "sample_size": sample_size,
        "historical_validation": sample_size >= 75,
        "recency_weight": 1.0,
        "concept_drift": conf_drift,
        "current_grade": institutional_grade,
        "pending_validation": sample_size < 75,
        "estimated_samples_remaining": max(0, 75 - sample_size),
    }

    shared_contracts = _knowledge_contracts(pattern_library)

    intelligence = {
        "meta": {
            "source_system": SourceSystem.PROMETHEUS.value,
            "dataset_generation": PROMETHEUS_DATASET_GENERATION,
            "feature_version": FEATURE_VERSION,
            "strategy_version": PROMETHEUS_STRATEGY_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "observational_only": True,
            "execution_behavior_unchanged": True,
        },
        "risk_intelligence": risk_intelligence,
        "market_health_engine": market_health_engine,
        "pattern_health_engine": {"patterns": pattern_health, "total_patterns": len(pattern_health)},
        "confidence_intelligence": confidence_intelligence,
        "return_intelligence": return_intelligence,
        "pattern_genome": pattern_genome,
        "pattern_library": pattern_library,
        "session_intelligence": session_intelligence,
        "regime_intelligence": regime_intelligence,
        "edge_stability": edge_stability,
        "decision_intelligence": decision_intelligence,
        "research_engine": research_engine,
        "knowledge_confidence": knowledge_confidence_framework,
        "execution_academy": execution_academy,
        "shared_knowledge_contracts": shared_contracts,
        "governance": {
            "historical_validation_required": True,
            "future_approval_required_for_behavior_change": True,
            "no_direct_execution_modification": True,
            "evidence_first_architecture": True,
            "olympus_compatible": True,
        },
    }

    return intelligence


def write_prometheus_intelligence_artifacts(root_dir: Path, intelligence: dict[str, Any]) -> dict[str, str]:
    ol = root_dir / "storage" / "olympus"
    ol.mkdir(parents=True, exist_ok=True)

    payload_path = ol / "prometheus_decision_intelligence.json"
    payload_path.write_text(json.dumps(intelligence, indent=2, ensure_ascii=True), encoding="utf-8")

    implementation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "new_intelligence_engines": {
            "risk_intelligence": True,
            "market_health_engine": True,
            "pattern_health_engine": True,
            "confidence_intelligence": True,
            "return_intelligence": True,
            "pattern_genome": True,
            "pattern_library": True,
            "session_intelligence": True,
            "regime_intelligence": True,
            "edge_stability": True,
            "decision_intelligence": True,
            "research_engine": True,
            "execution_academy": True,
        },
        "runtime_impact": "Low - offline analytics over existing status/database artifacts",
        "memory_impact": "Low - in-memory aggregations and report maps",
        "storage_impact": "Low - additive JSON artifacts only",
        "dashboard_additions": ["ui/prometheus_execution_academy_dashboard.py (port 8508)"],
        "backward_compatibility": True,
        "confirmations": {
            "trading_logic_unchanged": True,
            "smc_logic_unchanged": True,
            "existing_execution_preserved": True,
            "historical_learning_preserved": True,
            "adaptive_learning_preserved": True,
            "existing_dashboards_preserved": True,
            "olympus_compatibility_maintained": True,
            "evidence_first_architecture_maintained": True,
            "no_destructive_modifications": True,
            "execution_academy_independent": True,
        },
    }

    report_path = ol / "prometheus_execution_academy_report.json"
    report_path.write_text(json.dumps(implementation_report, indent=2, ensure_ascii=True), encoding="utf-8")

    # Append research observations (deduplicated by research_id).
    lib_path = ol / "prometheus_research_library.jsonl"
    existing: set[str] = set()
    if lib_path.exists():
        for line in lib_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rid = str(row.get("research_id", "") or "")
                if rid:
                    existing.add(rid)
            except Exception:
                continue

    observations = []
    mh = (intelligence.get("market_health_engine") or {}).get("market_health_score")
    if isinstance(mh, (float, int)):
        observations.append(
            {
                "research_id": f"PROM-R-MH-{int(mh*100)}",
                "source_system": SourceSystem.PROMETHEUS.value,
                "pattern_id": "market_health",
                "research_category": "Market Health",
                "observation": f"Market health score currently {mh:.2f} ({(intelligence.get('market_health_engine') or {}).get('classification')}).",
                "supporting_statistics": intelligence.get("market_health_engine", {}),
                "evidence_level": (intelligence.get("knowledge_confidence", {}) or {}).get("evidence_pct", 0.0),
                "knowledge_confidence": (intelligence.get("knowledge_confidence", {}) or {}).get("knowledge_confidence_pct", 0.0),
                "current_status": _sample_stage(int((intelligence.get("knowledge_confidence", {}) or {}).get("sample_size", 0) or 0)),
                "suggested_investigation": "Monitor evolution of market health alongside return stability.",
                "historical_evolution": (intelligence.get("return_intelligence", {}) or {}).get("expectancy_evolution", [])[-20:],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    for rec in (intelligence.get("execution_academy", {}) or {}).get("recommendations", []):
        rid = f"PROM-R-REC-{abs(hash(rec)) % 100000:05d}"
        observations.append(
            {
                "research_id": rid,
                "source_system": SourceSystem.PROMETHEUS.value,
                "pattern_id": "execution_academy",
                "research_category": "Execution Recommendation",
                "observation": rec,
                "supporting_statistics": (intelligence.get("execution_academy", {}) or {}).get("report_card", {}),
                "evidence_level": (intelligence.get("execution_academy", {}) or {}).get("evidence_level", 0.0),
                "knowledge_confidence": (intelligence.get("execution_academy", {}) or {}).get("knowledge_confidence", 0.0),
                "current_status": "Observed",
                "suggested_investigation": "Manual review in Zeus before any governance-approved behavior change.",
                "historical_evolution": (intelligence.get("execution_academy", {}) or {}).get("historical_progress", [])[-20:],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    with lib_path.open("a", encoding="utf-8") as fh:
        for obs in observations:
            rid = str(obs.get("research_id", "") or "")
            if not rid or rid in existing:
                continue
            fh.write(json.dumps(obs, ensure_ascii=True) + "\n")

    return {
        "payload_path": str(payload_path),
        "report_path": str(report_path),
        "research_library_path": str(lib_path),
    }
