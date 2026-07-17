from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


STATUS_AWAITING = "Awaiting Historical Data"

SESSION_ORDER = [
    "Sydney",
    "Tokyo",
    "Asia",
    "Frankfurt",
    "London Open",
    "London",
    "New York Open",
    "London-New York Overlap",
    "New York",
    "Late New York",
    "After-hours",
]

REGIME_ORDER = [
    "Strong Trend",
    "Trending",
    "Weak Trend",
    "Range",
    "Compression",
    "Expansion",
    "Accumulation",
    "Distribution",
    "Breakout",
    "Reversal",
    "High Volatility",
    "Normal Volatility",
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


def _safe_dt(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _session_label(ts: Any) -> str:
    dt = _safe_dt(ts)
    if dt is None:
        return "After-hours"
    h = dt.astimezone(timezone.utc).hour
    if 21 <= h <= 23:
        return "Sydney"
    if 0 <= h <= 2:
        return "Tokyo"
    if 3 <= h <= 5:
        return "Asia"
    if h == 6:
        return "Frankfurt"
    if 7 <= h <= 8:
        return "London Open"
    if 9 <= h <= 11:
        return "London"
    if 12 <= h <= 13:
        return "New York Open"
    if 14 <= h <= 16:
        return "London-New York Overlap"
    if 17 <= h <= 18:
        return "New York"
    if 19 <= h <= 20:
        return "Late New York"
    return "After-hours"


def _time_labels(ts: Any) -> dict[str, Any]:
    dt = _safe_dt(ts)
    if dt is None:
        return {
            "hour": "unknown",
            "day_of_week": "unknown",
            "trading_week": "unknown",
            "month": "unknown",
            "quarter": "unknown",
            "time_window": "unknown",
        }
    dt = dt.astimezone(timezone.utc)
    return {
        "hour": dt.hour,
        "day_of_week": dt.strftime("%A"),
        "trading_week": f"W{dt.isocalendar().week}",
        "month": dt.strftime("%Y-%m"),
        "quarter": f"Q{((dt.month - 1) // 3) + 1}",
        "time_window": f"{dt.hour:02d}:00-{(dt.hour + 1) % 24:02d}:00",
    }


def _context_regime(row: dict[str, Any]) -> str:
    base = str(row.get("market_regime", "") or "").lower()
    trend = str(row.get("trend_bucket", "") or "").lower()
    vol = str(row.get("volatility_bucket", "") or "").lower()

    if "sweep" in base:
        return "Reversal"
    if "expansion" in base or vol == "high":
        if trend == "high":
            return "Strong Trend"
        return "Expansion"
    if "exhaustion" in base:
        return "Weak Trend"
    if "mean reversion" in base:
        return "Range"
    if "compression" in base:
        return "Compression"
    if "dead" in base:
        return "Accumulation"
    if trend == "high":
        return "Trending"
    if trend == "mid":
        return "Weak Trend"
    if vol == "low":
        return "Low Volatility"
    return "Normal Volatility"


def _metric_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "occurrences": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": STATUS_AWAITING,
            "average_return": STATUS_AWAITING,
            "average_mfe": STATUS_AWAITING,
            "average_mae": STATUS_AWAITING,
            "average_duration": STATUS_AWAITING,
            "average_confidence": STATUS_AWAITING,
            "average_prediction_accuracy": STATUS_AWAITING,
            "profit_factor": STATUS_AWAITING,
            "expectancy": STATUS_AWAITING,
            "return_stability": STATUS_AWAITING,
            "pattern_stability": STATUS_AWAITING,
            "knowledge_confidence": 0.0,
            "evidence_level": 0.0,
            "estimated_samples_remaining": 30,
        }

    returns = [float(_safe_float(r.get("return_pct")) or 0.0) for r in rows]
    mfe = [float(_safe_float(r.get("actual_distance")) or 0.0) for r in rows]
    mae = [float(abs(_safe_float(r.get("trade_pnl")) or 0.0)) for r in rows if (_safe_float(r.get("trade_pnl")) or 0.0) < 0]
    durs = [float(_safe_float(r.get("duration_seconds")) or 0.0) for r in rows if _safe_float(r.get("duration_seconds")) is not None]
    confs = [float(_safe_float(r.get("confidence")) or 0.0) for r in rows if _safe_float(r.get("confidence")) is not None]
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    win_n = len(wins)
    loss_n = len(losses)
    n = len(rows)

    pf = STATUS_AWAITING
    if wins and losses and abs(sum(losses)) > 1e-9:
        pf = round(sum(wins) / abs(sum(losses)), 4)

    acc = [1.0 if int(r.get("actual_outcome", 0) or 0) == 1 else 0.0 for r in rows]
    rstd = pstdev(returns) if len(returns) > 1 else 0.0
    stability = max(0.0, 1.0 - min(1.0, rstd))
    evidence = min(100.0, (n / 150.0) * 100.0)
    kc = round(min(100.0, (0.65 * evidence) + (0.35 * (stability * 100.0))), 2)

    return {
        "occurrences": n,
        "wins": win_n,
        "losses": loss_n,
        "win_rate": round(win_n / n, 4),
        "average_return": round(mean(returns), 4),
        "average_mfe": round(mean(mfe), 4) if mfe else STATUS_AWAITING,
        "average_mae": round(mean(mae), 4) if mae else STATUS_AWAITING,
        "average_duration": round(mean(durs), 2) if durs else STATUS_AWAITING,
        "average_confidence": round(mean(confs), 4) if confs else STATUS_AWAITING,
        "average_prediction_accuracy": round(mean(acc), 4),
        "profit_factor": pf,
        "expectancy": round(mean(returns), 4),
        "return_stability": round(stability, 4),
        "pattern_stability": round(stability, 4),
        "knowledge_confidence": kc,
        "evidence_level": round(evidence, 2),
        "estimated_samples_remaining": max(0, 30 - n),
    }


def _rank_table(grouped: dict[str, list[dict[str, Any]]], key_name: str) -> list[dict[str, Any]]:
    rows = []
    for key, items in grouped.items():
        metrics = _metric_bundle(items)
        rows.append({key_name: key, **metrics})
    rows.sort(key=lambda r: ((r.get("expectancy") if isinstance(r.get("expectancy"), (float, int)) else -1e9), r.get("occurrences", 0)), reverse=True)
    return rows


def _pick(rows: list[dict[str, Any]], key: str) -> str:
    if not rows:
        return STATUS_AWAITING
    valid = [r for r in rows if isinstance(r.get("expectancy"), (float, int))]
    if not valid:
        return STATUS_AWAITING
    if key == "best":
        return str(max(valid, key=lambda r: float(r.get("expectancy", -1e9))).get(next(k for k in r_keys(valid[0]) if k not in CORE_METRIC_KEYS), STATUS_AWAITING))
    return str(min(valid, key=lambda r: float(r.get("expectancy", 1e9))).get(next(k for k in r_keys(valid[0]) if k not in CORE_METRIC_KEYS), STATUS_AWAITING))


def r_keys(row: dict[str, Any]) -> list[str]:
    return list(row.keys())


CORE_METRIC_KEYS = {
    "occurrences",
    "wins",
    "losses",
    "win_rate",
    "average_return",
    "average_mfe",
    "average_mae",
    "average_duration",
    "average_confidence",
    "average_prediction_accuracy",
    "profit_factor",
    "expectancy",
    "return_stability",
    "pattern_stability",
    "knowledge_confidence",
    "evidence_level",
    "estimated_samples_remaining",
}


def _entity_key(row: dict[str, Any], fallback: str) -> str:
    return str(row.get(fallback, row.get("pattern_id", "unknown")) or "unknown")


def _context_profile(pattern_rows: list[dict[str, Any]], session_rank: list[dict[str, Any]], regime_rank: list[dict[str, Any]], vol_rank: list[dict[str, Any]], hour_rank: list[dict[str, Any]], day_rank: list[dict[str, Any]]) -> dict[str, Any]:
    base_stats = _metric_bundle(pattern_rows)

    def _best(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return STATUS_AWAITING
        col = next((k for k in rows[0].keys() if k not in CORE_METRIC_KEYS), None)
        if not col:
            return STATUS_AWAITING
        ranked = [r for r in rows if isinstance(r.get("expectancy"), (float, int))]
        if not ranked:
            return STATUS_AWAITING
        return str(max(ranked, key=lambda r: float(r.get("expectancy", -1e9))).get(col, STATUS_AWAITING))

    def _worst(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return STATUS_AWAITING
        col = next((k for k in rows[0].keys() if k not in CORE_METRIC_KEYS), None)
        if not col:
            return STATUS_AWAITING
        ranked = [r for r in rows if isinstance(r.get("expectancy"), (float, int))]
        if not ranked:
            return STATUS_AWAITING
        return str(min(ranked, key=lambda r: float(r.get("expectancy", 1e9))).get(col, STATUS_AWAITING))

    return {
        "preferred_session": _best(session_rank),
        "session_performance_ranking": session_rank,
        "preferred_market_regime": _best(regime_rank),
        "preferred_volatility": _best(vol_rank),
        "preferred_trend_strength": _best(regime_rank),
        "preferred_htf_bias": _best(regime_rank),
        "preferred_ltf_bias": _best(session_rank),
        "preferred_liquidity_context": _best(regime_rank),
        "preferred_choch_state": _best(regime_rank),
        "preferred_bos_state": _best(regime_rank),
        "preferred_order_block_context": _best(regime_rank),
        "preferred_fvg_context": _best(regime_rank),
        "preferred_time_window": _best(hour_rank),
        "worst_session": _worst(session_rank),
        "worst_market_regime": _worst(regime_rank),
        "best_hour": _best(hour_rank),
        "worst_hour": _worst(hour_rank),
        "best_day": _best(day_rank),
        "worst_day": _worst(day_rank),
        "historical_return": base_stats.get("average_return", STATUS_AWAITING),
        "historical_mfe": base_stats.get("average_mfe", STATUS_AWAITING),
        "historical_mae": base_stats.get("average_mae", STATUS_AWAITING),
        "historical_holding_time": base_stats.get("average_duration", STATUS_AWAITING),
        "historical_confidence": base_stats.get("average_confidence", STATUS_AWAITING),
        "historical_prediction_accuracy": base_stats.get("average_prediction_accuracy", STATUS_AWAITING),
        "historical_exit_quality": STATUS_AWAITING,
        "knowledge_confidence": base_stats.get("knowledge_confidence", 0.0),
        "evidence_level": base_stats.get("evidence_level", 0.0),
        "pattern_maturity": _maturity(base_stats.get("occurrences", 0)),
    }


def _maturity(samples: int) -> str:
    if samples < 10:
        return "Candidate"
    if samples < 30:
        return "Emerging"
    if samples < 75:
        return "Developing"
    if samples < 150:
        return "Validated"
    return "Elite"


def _research_observations(pattern_id: str, context_profile: dict[str, Any], session_rank: list[dict[str, Any]], regime_rank: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    obs: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    def _mk(category: str, text: str, evidence: float, kc: float, stats: dict[str, Any], suggestion: str) -> dict[str, Any]:
        rid = f"CTX-{pattern_id}-{category}-{abs(hash(text)) % 100000:05d}"
        return {
            "research_id": rid,
            "timestamp": now,
            "pattern_id": pattern_id,
            "evidence_level": round(evidence, 2),
            "knowledge_confidence": round(kc, 2),
            "supporting_statistics": stats,
            "historical_sample_size": sample_size,
            "current_status": "Observed",
            "research_category": category,
            "observation": text,
            "suggested_investigation": suggestion,
        }

    if session_rank:
        best = session_rank[0]
        weak = session_rank[-1]
        skey = next((k for k in best.keys() if k not in CORE_METRIC_KEYS), "session")
        obs.append(
            _mk(
                "Session Context",
                f"Pattern {pattern_id} demonstrates stronger expectancy during {best.get(skey)}.",
                best.get("evidence_level", 0.0),
                best.get("knowledge_confidence", 0.0),
                {"best_session": best, "weak_session": weak},
                "Manually compare session-specific entry and exit efficiency in Zeus.",
            )
        )
    if regime_rank:
        best_r = regime_rank[0]
        weak_r = regime_rank[-1]
        rkey = next((k for k in best_r.keys() if k not in CORE_METRIC_KEYS), "regime")
        obs.append(
            _mk(
                "Regime Context",
                f"Pattern {pattern_id} underperforms during {weak_r.get(rkey)} and improves in {best_r.get(rkey)}.",
                best_r.get("evidence_level", 0.0),
                best_r.get("knowledge_confidence", 0.0),
                {"best_regime": best_r, "weak_regime": weak_r},
                "Manually validate regime-conditioned expectancy with Zeus before any policy changes.",
            )
        )

    if isinstance(context_profile.get("historical_return"), (float, int)):
        hr = float(context_profile.get("historical_return", 0.0))
        if hr > 0:
            obs.append(
                _mk(
                    "Return Context",
                    f"Pattern {pattern_id} maintains positive contextual return expectancy ({hr:.4f}).",
                    float(context_profile.get("evidence_level", 0.0) or 0.0),
                    float(context_profile.get("knowledge_confidence", 0.0) or 0.0),
                    {"historical_return": hr, "context_profile": context_profile},
                    "Review whether contextual strength persists across recent quarters.",
                )
            )

    return obs


def _archive_observations(root_dir: Path, observations: list[dict[str, Any]]) -> None:
    if not observations:
        return
    lib = root_dir / "storage" / "olympus" / "hermes_research_library.jsonl"
    lib.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    if lib.exists():
        for line in lib.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rid = str(row.get("research_id", "") or "")
            if rid:
                existing_ids.add(rid)

    to_write = [o for o in observations if str(o.get("research_id", "") or "") not in existing_ids]
    if not to_write:
        return

    with lib.open("a", encoding="utf-8") as fh:
        for row in to_write:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_research_library(root_dir: Path, limit: int = 500) -> list[dict[str, Any]]:
    lib = root_dir / "storage" / "olympus" / "hermes_research_library.jsonl"
    if not lib.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in lib.read_text(encoding="utf-8").splitlines()[-max(1, limit):]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def build_pattern_context_intelligence(
    root_dir: Path,
    completed_predictions: list[dict[str, Any]],
    pattern_library: list[dict[str, Any]],
) -> dict[str, Any]:
    enriched_rows: list[dict[str, Any]] = []
    for row in completed_predictions:
        ts = row.get("timestamp")
        session = _session_label(ts)
        time_info = _time_labels(ts)
        regime = _context_regime(row)
        enriched_rows.append(
            {
                **row,
                "context_session": session,
                "context_regime": regime,
                "context_volatility": "High Volatility" if str(row.get("volatility_bucket", "normal")).lower() == "high" else "Normal Volatility" if str(row.get("volatility_bucket", "normal")).lower() == "normal" else "Low Volatility",
                "context_trend": "Strong Trend" if str(row.get("trend_bucket", "")).lower() == "high" else "Weak Trend" if str(row.get("trend_bucket", "")).lower() == "mid" else "Range",
                **time_info,
            }
        )

    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_rows:
        by_pattern[str(row.get("pattern_id", "unknown") or "unknown")].append(row)

    context_profiles: list[dict[str, Any]] = []
    session_genomes: dict[str, list[dict[str, Any]]] = {}
    regime_profiles: dict[str, list[dict[str, Any]]] = {}
    volatility_profiles: dict[str, list[dict[str, Any]]] = {}
    timing_profiles: dict[str, dict[str, Any]] = {}
    research_obs: list[dict[str, Any]] = []

    expanded_library: list[dict[str, Any]] = []
    lib_map = {str(p.get("pattern_id", "") or ""): p for p in pattern_library}

    for pid, rows in by_pattern.items():
        session_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        regime_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        vol_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        hour_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        day_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        week_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        month_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        quarter_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for r in rows:
            session_groups[str(r.get("context_session", "After-hours"))].append(r)
            regime_groups[str(r.get("context_regime", "Normal Volatility"))].append(r)
            vol_groups[str(r.get("context_volatility", "Normal Volatility"))].append(r)
            hour_groups[str(r.get("hour", "unknown"))].append(r)
            day_groups[str(r.get("day_of_week", "unknown"))].append(r)
            week_groups[str(r.get("trading_week", "unknown"))].append(r)
            month_groups[str(r.get("month", "unknown"))].append(r)
            quarter_groups[str(r.get("quarter", "unknown"))].append(r)

        session_rank = _rank_table(session_groups, "session")
        regime_rank = _rank_table(regime_groups, "regime")
        vol_rank = _rank_table(vol_groups, "volatility")
        hour_rank = _rank_table(hour_groups, "hour")
        day_rank = _rank_table(day_groups, "day")
        week_rank = _rank_table(week_groups, "trading_week")
        month_rank = _rank_table(month_groups, "month")
        quarter_rank = _rank_table(quarter_groups, "quarter")

        profile = {
            "pattern_id": pid,
            **_context_profile(rows, session_rank, regime_rank, vol_rank, hour_rank, day_rank),
            "session_reliability": session_rank[0].get("knowledge_confidence", 0.0) if session_rank else 0.0,
            "regime_reliability": regime_rank[0].get("knowledge_confidence", 0.0) if regime_rank else 0.0,
            "time_reliability": hour_rank[0].get("knowledge_confidence", 0.0) if hour_rank else 0.0,
        }
        context_profiles.append(profile)

        session_genomes[pid] = session_rank
        regime_profiles[pid] = regime_rank
        volatility_profiles[pid] = vol_rank
        timing_profiles[pid] = {
            "hour_profile": hour_rank,
            "day_profile": day_rank,
            "week_profile": week_rank,
            "month_profile": month_rank,
            "quarter_profile": quarter_rank,
            "best_hour": profile.get("best_hour", STATUS_AWAITING),
            "worst_hour": profile.get("worst_hour", STATUS_AWAITING),
            "best_day": profile.get("best_day", STATUS_AWAITING),
            "worst_day": profile.get("worst_day", STATUS_AWAITING),
            "seasonality": month_rank,
            "historical_consistency": profile.get("knowledge_confidence", 0.0),
        }

        obs = _research_observations(pid, profile, session_rank, regime_rank, len(rows))
        research_obs.extend(obs)

        base = dict(lib_map.get(pid, {"pattern_id": pid}))
        base.update(
            {
                "context_profile": profile,
                "session_genome": session_rank,
                "market_regime_profile": regime_rank,
                "volatility_profile": vol_rank,
                "timing_profile": timing_profiles[pid],
                "execution_profile": {
                    "average_duration": _metric_bundle(rows).get("average_duration", STATUS_AWAITING),
                    "average_confidence": _metric_bundle(rows).get("average_confidence", STATUS_AWAITING),
                    "average_prediction_accuracy": _metric_bundle(rows).get("average_prediction_accuracy", STATUS_AWAITING),
                },
                "return_profile": {
                    "average_return": _metric_bundle(rows).get("average_return", STATUS_AWAITING),
                    "average_mfe": _metric_bundle(rows).get("average_mfe", STATUS_AWAITING),
                    "average_mae": _metric_bundle(rows).get("average_mae", STATUS_AWAITING),
                    "profit_factor": _metric_bundle(rows).get("profit_factor", STATUS_AWAITING),
                    "expectancy": _metric_bundle(rows).get("expectancy", STATUS_AWAITING),
                },
                "confidence_profile": {
                    "average_confidence": _metric_bundle(rows).get("average_confidence", STATUS_AWAITING),
                    "knowledge_confidence": _metric_bundle(rows).get("knowledge_confidence", 0.0),
                },
                "evidence_profile": {
                    "evidence_level": _metric_bundle(rows).get("evidence_level", 0.0),
                    "sample_size": len(rows),
                    "estimated_samples_remaining": _metric_bundle(rows).get("estimated_samples_remaining", 30),
                },
                "knowledge_confidence": _metric_bundle(rows).get("knowledge_confidence", 0.0),
                "pattern_maturity": _maturity(len(rows)),
                "historical_evolution": {
                    "week": week_rank,
                    "month": month_rank,
                    "quarter": quarter_rank,
                },
            }
        )
        expanded_library.append(base)

    _archive_observations(root_dir, research_obs)
    research_library = load_research_library(root_dir, limit=1000)

    all_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_regimes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_hours: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_rows:
        all_sessions[str(row.get("context_session", "After-hours"))].append(row)
        all_regimes[str(row.get("context_regime", "Normal Volatility"))].append(row)
        all_hours[str(row.get("hour", "unknown"))].append(row)
        all_days[str(row.get("day_of_week", "unknown"))].append(row)

    session_ranking = _rank_table(all_sessions, "session")
    regime_ranking = _rank_table(all_regimes, "regime")
    hour_ranking = _rank_table(all_hours, "hour")
    day_ranking = _rank_table(all_days, "day")

    pattern_context_academy = {
        "academy": "Pattern Context Intelligence",
        "implementation": 100.0,
        "evidence": round(min(100.0, (len(enriched_rows) / 300.0) * 100.0), 2),
        "knowledge_confidence": round(min(100.0, (len(context_profiles) / 40.0) * 100.0), 2),
        "mastery": round(mean([float(c.get("knowledge_confidence", 0.0) or 0.0) for c in context_profiles]), 2) if context_profiles else 0.0,
        "pattern_context_coverage": round(min(100.0, (len(context_profiles) / max(1, len(pattern_library))) * 100.0), 2) if pattern_library else 0.0,
        "session_coverage": round(min(100.0, (len([s for s in session_ranking if int(s.get("occurrences", 0) or 0) > 0]) / max(1, len(SESSION_ORDER))) * 100.0), 2),
        "market_regime_coverage": round(min(100.0, (len([r for r in regime_ranking if int(r.get("occurrences", 0) or 0) > 0]) / max(1, len(REGIME_ORDER))) * 100.0), 2),
        "time_coverage": round(min(100.0, (len([h for h in hour_ranking if int(h.get("occurrences", 0) or 0) > 0]) / 24.0) * 100.0), 2),
        "reliability": round(mean([float(c.get("session_reliability", 0.0) or 0.0) for c in context_profiles]), 2) if context_profiles else 0.0,
        "current_grade": _grade(len(enriched_rows), context_profiles),
        "next_milestone": _next_milestone(len(enriched_rows)),
        "estimated_samples_remaining": max(0, 150 - len(enriched_rows)),
    }

    observations = {
        "session_comparison": [
            f"Patterns collectively show stronger expectancy in {session_ranking[0].get('session')} than {session_ranking[-1].get('session')}" if session_ranking else STATUS_AWAITING,
        ],
        "market_context_discovery": [
            f"Patterns collectively show best regime expectancy in {regime_ranking[0].get('regime')}" if regime_ranking else STATUS_AWAITING,
        ],
        "research_observations": research_obs[-100:],
    }

    return {
        "pattern_context_library": expanded_library,
        "context_profiles": context_profiles,
        "session_genome": session_genomes,
        "market_regime_profiles": regime_profiles,
        "volatility_profiles": volatility_profiles,
        "timing_profiles": timing_profiles,
        "session_ranking": session_ranking,
        "regime_ranking": regime_ranking,
        "hour_ranking": hour_ranking,
        "day_ranking": day_ranking,
        "best_session": session_ranking[0].get("session", STATUS_AWAITING) if session_ranking else STATUS_AWAITING,
        "worst_session": session_ranking[-1].get("session", STATUS_AWAITING) if session_ranking else STATUS_AWAITING,
        "best_market_regime": regime_ranking[0].get("regime", STATUS_AWAITING) if regime_ranking else STATUS_AWAITING,
        "worst_market_regime": regime_ranking[-1].get("regime", STATUS_AWAITING) if regime_ranking else STATUS_AWAITING,
        "best_hour": hour_ranking[0].get("hour", STATUS_AWAITING) if hour_ranking else STATUS_AWAITING,
        "worst_hour": hour_ranking[-1].get("hour", STATUS_AWAITING) if hour_ranking else STATUS_AWAITING,
        "best_day": day_ranking[0].get("day", STATUS_AWAITING) if day_ranking else STATUS_AWAITING,
        "worst_day": day_ranking[-1].get("day", STATUS_AWAITING) if day_ranking else STATUS_AWAITING,
        "session_heatmap": session_ranking,
        "market_regime_heatmap": regime_ranking,
        "time_heatmap": hour_ranking,
        "context_reliability": pattern_context_academy.get("reliability", 0.0),
        "knowledge_confidence": pattern_context_academy.get("knowledge_confidence", 0.0),
        "evidence_level": pattern_context_academy.get("evidence", 0.0),
        "pattern_maturity": _maturity(len(enriched_rows)),
        "historical_evolution": {
            "hour": hour_ranking,
            "day": day_ranking,
        },
        "observations": observations,
        "research_library": research_library,
        "academy_subject": pattern_context_academy,
        "manual_research_workflow": {
            "mode": "Manual",
            "rules": [
                "Hermes observes and archives context evidence.",
                "No automatic strategy mutation.",
                "No automatic Zeus backtesting.",
                "Operator manually exports selected observations to Zeus.",
            ],
            "zeus_integration": "Manual export only",
        },
    }


def _grade(sample_size: int, profiles: list[dict[str, Any]]) -> str:
    if sample_size < 10:
        return "F"
    if sample_size < 30:
        return "D"
    rel = mean([float(p.get("knowledge_confidence", 0.0) or 0.0) for p in profiles]) if profiles else 0.0
    if sample_size < 75:
        return "C" if rel >= 35 else "D+"
    if sample_size < 150:
        return "B-" if rel >= 45 else "C+"
    return "B" if rel >= 55 else "B-"


def _next_milestone(sample_size: int) -> str:
    if sample_size < 30:
        return "Reach 30 contextual outcomes for early reliability scoring"
    if sample_size < 75:
        return "Reach 75 outcomes for developing session/regime confidence"
    if sample_size < 150:
        return "Reach 150 outcomes for validated context maturity"
    return "Continue longitudinal context evolution monitoring"
