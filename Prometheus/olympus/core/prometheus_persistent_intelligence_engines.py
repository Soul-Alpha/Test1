from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from olympus.contracts import SourceSystem

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


def _load_jsonl(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return rows[-limit:]


def _stability(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    sd = pstdev(vals)
    return round(max(0.0, 1.0 - min(1.0, sd)), 4)


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


def _profit_factor_value(pnls: list[float]) -> float | None:
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not wins or not losses:
        return None
    den = abs(sum(losses))
    if den <= 1e-9:
        return None
    return sum(wins) / den


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


def _rolling_metric(vals: list[float], window: int, metric_fn: Any) -> list[float | str]:
    out: list[float | str] = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        value = metric_fn(chunk)
        out.append(STATUS_AWAITING if value is None else round(float(value), 4))
    return out


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
    for section in (
        "validated_findings",
        "candidate_findings",
        "rejected_findings",
        "improvement_hypotheses",
        "capital_research",
        "recommendation_outcomes",
    ):
        for row in prior_library.get(section, []) or []:
            existing_id = str(row.get("finding_id") or row.get("hypothesis_id") or row.get("recommendation_id") or "")
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
            "growth_stability": STATUS_AWAITING,
            "growth_consistency": STATUS_AWAITING,
            "recovery_efficiency": STATUS_AWAITING,
            "risk_mode": risk_mode,
        }

    balance = start_balance
    peak = balance
    dds: list[float] = []
    equity_curve = [balance]

    for idx, ret in enumerate(returns):
        recent = returns[max(0, idx - 10) : idx + 1]
        recent_mean = mean(recent) if recent else 0.0
        current_dd = max(0.0, (peak - balance) / max(1e-9, peak))

        if risk_mode == "fixed_fractional":
            risk_pct = 0.01
        elif risk_mode == "adaptive_risk":
            risk_pct = max(0.003, min(0.02, 0.01 + recent_mean * 0.002))
        elif risk_mode == "reduced_after_drawdown":
            risk_pct = max(0.002, 0.01 * (1.0 - min(0.8, current_dd * 2.0)))
        elif risk_mode == "increased_after_recovery":
            risk_pct = 0.012 if current_dd < 0.02 else 0.008
        elif risk_mode == "dynamic_rr":
            risk_pct = 0.011 if recent_mean > 1.2 else 0.0085
        elif risk_mode == "partial_exits":
            risk_pct = 0.0095
            ret = ret * 0.88 if ret > 0 else ret * 0.72
        elif risk_mode == "scaling":
            risk_pct = 0.013 if recent_mean > 0.8 else 0.007
        elif risk_mode == "trailing_exits":
            risk_pct = 0.009
            ret = ret * 0.92 if ret > 0 else ret * 0.82
        elif risk_mode == "reduced_exposure":
            risk_pct = 0.006
        elif risk_mode == "aggressive_recovery":
            risk_pct = 0.016 if recent_mean < 0 else 0.01
        elif risk_mode == "conservative_recovery":
            risk_pct = 0.0065 if recent_mean < 0 else 0.009
        elif risk_mode == "dynamic_position_sizing":
            risk_pct = max(0.003, min(0.018, 0.009 + (recent_mean * 0.0015) - current_dd * 0.01))
        else:
            risk_pct = 0.01

        realized = balance * risk_pct * ret
        balance += realized
        peak = max(peak, balance)
        dd = max(0.0, (peak - balance) / max(1e-9, peak))
        dds.append(dd)
        equity_curve.append(balance)

    net = (balance / max(1e-9, start_balance)) - 1.0
    cagr_proxy = net * (252.0 / max(1.0, len(returns)))
    max_dd = max(dds) if dds else 0.0
    survival = max(0.0, 1.0 - min(1.0, max_dd * 1.5))
    capital_eff = net / max(1e-9, max_dd + 0.01)
    growth_returns = [
        (equity_curve[i] / max(1e-9, equity_curve[i - 1])) - 1.0
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] > 0
    ]
    growth_stability = _stability(growth_returns)
    growth_consistency = sum(1 for x in growth_returns if x > 0) / max(1, len(growth_returns))
    recovery_efficiency = max(0.0, 1.0 - min(1.0, max_dd / max(0.01, abs(net) + 0.01)))
    geometric_return = (balance / max(1e-9, start_balance)) ** (1.0 / max(1.0, len(returns))) - 1.0

    return {
        "starting_balance": start_balance,
        "ending_balance": round(balance, 2),
        "cagr_proxy": round(cagr_proxy, 4),
        "expected_cagr": round(cagr_proxy, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "survival_probability": round(survival, 4),
        "capital_efficiency": round(capital_eff, 4),
        "compounding_efficiency": round(balance / max(1e-9, start_balance), 4),
        "growth_stability": round(growth_stability, 4),
        "growth_consistency": round(growth_consistency, 4),
        "recovery_efficiency": round(recovery_efficiency, 4),
        "capital_survival": round(survival, 4),
        "risk_adjusted_growth": round(net / max(0.01, max_dd + 0.01), 4),
        "geometric_return": round(geometric_return, 4),
        "risk_mode": risk_mode,
    }


class PersistentIntelligenceEngine:
    def __init__(self, root_dir: Path, engine_key: str, generated_at: str, version: str = "v2.2") -> None:
        self.root_dir = root_dir
        self.engine_key = engine_key
        self.generated_at = generated_at
        self.version = version
        self.storage_dir = self.root_dir / "storage" / "olympus"

    def state_path(self) -> Path:
        return self.storage_dir / f"{self.engine_key}.json"

    def history_path(self) -> Path:
        return self.storage_dir / f"{self.engine_key}_history.jsonl"

    def research_path(self) -> Path:
        return self.storage_dir / f"{self.engine_key}_research.jsonl"

    def load_state(self) -> dict[str, Any]:
        return _load_json(self.state_path(), {}) or {}

    def load_history(self, limit: int = 200) -> list[dict[str, Any]]:
        return _load_jsonl(self.history_path(), limit=limit)

    def contract(self) -> dict[str, Any]:
        return {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "source_system": SourceSystem.PROMETHEUS.value,
            "generated_at": self.generated_at,
            "observational_only": True,
            "governed_adoption_only": True,
            "historical_data_immutable": True,
            "backward_compatible": True,
        }

    def dataset_contract(self) -> dict[str, str]:
        return {
            "state_path": str(self.state_path().relative_to(self.root_dir)),
            "history_path": str(self.history_path().relative_to(self.root_dir)),
            "research_path": str(self.research_path().relative_to(self.root_dir)),
        }


class EdgeIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "edge_intelligence_engine", generated_at)

    def build(self, trades: list[dict[str, Any]], pnl_vals: list[float], rr_vals: list[float], win_rate: float) -> dict[str, Any]:
        rolling_exp = _rolling(pnl_vals, 40)
        rolling_pf = _rolling_metric(pnl_vals, 40, _profit_factor_value)
        rolling_sharpe = _rolling_metric(pnl_vals, 40, _sharpe_like)
        rolling_sortino = _rolling_metric(pnl_vals, 40, _sortino_like)
        half = max(1, len(rolling_exp) // 2)
        early = mean(rolling_exp[:half]) if rolling_exp else 0.0
        late = mean(rolling_exp[half:]) if rolling_exp else 0.0
        edge_decay = round(early - late, 4)
        edge_durability = round(max(0.0, 1.0 - abs(edge_decay) / max(1.0, abs(early) + abs(late) + 1e-9)), 4)
        edge_stability = _stability(rolling_exp)
        ci = _ci_binom(win_rate, len(trades))
        significance_proxy = round(min(1.0, math.sqrt(max(1, len(trades)) / 500.0) * (0.5 + abs(win_rate - 0.5))), 4)
        ci_width = 1.0
        if isinstance(ci.get("low"), (int, float)) and isinstance(ci.get("high"), (int, float)):
            ci_width = max(0.0, min(1.0, float(ci["high"]) - float(ci["low"])))
        edge_distribution = _context_edge_distribution(trades)
        sharpe_value = _sharpe_like(pnl_vals)
        sortino_value = _sortino_like(pnl_vals)
        profit_factor_value = _profit_factor_value(pnl_vals)
        payoff_ratio_value = _payoff_ratio_value(pnl_vals)
        sample_strength = min(1.0, len(trades) / 400.0)
        edge_confidence = round(mean([significance_proxy, sample_strength, max(0.0, 1.0 - ci_width)]), 4)
        edge_robustness = round(
            mean(
                [
                    edge_stability,
                    significance_proxy,
                    sample_strength,
                    min(1.0, max(0.0, (profit_factor_value or 0.0) / 2.0)),
                ]
            ),
            4,
        )
        rolling_metrics = []
        rolling_start = max(0, len(rolling_exp) - 120)
        for idx in range(rolling_start, len(rolling_exp)):
            rolling_metrics.append(
                {
                    "trade_index": idx + 1,
                    "expectancy": round(rolling_exp[idx], 4),
                    "profit_factor": rolling_pf[idx],
                    "sharpe": rolling_sharpe[idx],
                    "sortino": rolling_sortino[idx],
                }
            )

        execution_edge_learning_engine = {
            "expectancy": round(mean(pnl_vals), 4) if pnl_vals else STATUS_AWAITING,
            "edge_durability": edge_durability,
            "edge_decay": edge_decay,
            "edge_stability": edge_stability,
            "edge_concentration": edge_distribution.get("edge_concentration", STATUS_AWAITING),
            "edge_breadth": edge_distribution.get("edge_breadth", 0),
            "dominant_edge_context": edge_distribution.get("dominant_context", STATUS_AWAITING),
            "edge_confidence": edge_confidence,
            "edge_robustness": edge_robustness,
            "confidence_interval": ci,
            "statistical_significance": significance_proxy,
            "repeatability": round(_stability(rr_vals), 4),
            "rolling_metrics": rolling_metrics,
            "context_distribution": edge_distribution.get("context_distribution", []),
            "rolling_sharpe": round(sharpe_value, 4) if sharpe_value is not None else STATUS_AWAITING,
            "rolling_sortino": round(sortino_value, 4) if sortino_value is not None else STATUS_AWAITING,
            "profit_factor": round(profit_factor_value, 4) if profit_factor_value is not None else STATUS_AWAITING,
            "payoff_ratio": round(payoff_ratio_value, 4) if payoff_ratio_value is not None else STATUS_AWAITING,
        }

        prior_history = self.load_history(limit=240)
        history_entry = {
            "timestamp": self.generated_at,
            "expectancy": execution_edge_learning_engine.get("expectancy"),
            "edge_decay": execution_edge_learning_engine.get("edge_decay"),
            "edge_stability": execution_edge_learning_engine.get("edge_stability"),
            "edge_robustness": execution_edge_learning_engine.get("edge_robustness"),
            "profit_factor": execution_edge_learning_engine.get("profit_factor"),
        }
        edge_history = (prior_history + [history_entry])[-180:]

        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "current_summary": execution_edge_learning_engine,
                "edge_evolution_history": edge_history,
                "edge_evolution_series": rolling_metrics,
            },
            "history_entry": history_entry,
            "versioned_outputs": [
                {
                    "timestamp": self.generated_at,
                    "category": "edge_evolution_snapshot",
                    "version": self.version,
                    "summary": history_entry,
                }
            ],
        }
        return {
            "engine": engine,
            "execution_edge_learning_engine": execution_edge_learning_engine,
            "edge_evolution_history": edge_history,
        }


class CapitalIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "capital_intelligence_engine", generated_at)

    def build(self, rr_for_growth: list[float], sample_size: int, prior_library: dict[str, Any]) -> dict[str, Any]:
        growth_start_balances = [50, 100, 250, 500, 1000, 5000, 10000]
        risk_modes = [
            "fixed_fractional",
            "adaptive_risk",
            "dynamic_rr",
            "partial_exits",
            "scaling",
            "trailing_exits",
            "reduced_exposure",
            "aggressive_recovery",
            "conservative_recovery",
            "dynamic_position_sizing",
            "reduced_after_drawdown",
            "increased_after_recovery",
        ]
        growth_runs = []
        for start in growth_start_balances:
            for mode in risk_modes:
                growth_runs.append(_simulate_capital_growth(rr_for_growth, float(start), mode))

        best_growth = sorted(
            [g for g in growth_runs if isinstance(g.get("capital_efficiency"), (int, float))],
            key=lambda x: float(x.get("capital_efficiency", -1e9)),
            reverse=True,
        )
        best_path = best_growth[0] if best_growth else {}
        capital_growth_intelligence_engine = {
            "simulations": growth_runs,
            "best_capital_path": best_path,
            "objective": "Long-term compounded capital growth with preservation",
            "learning_status": "Institutional" if len(growth_runs) >= 20 else "Developing",
        }

        capital_research = []
        if best_path:
            confidence = round(min(100.0, (sample_size / 300.0) * 100.0), 2)
            capital_research.append(
                {
                    "finding_id": "C-CAPITAL-GROWTH-001",
                    "finding_type": "capital_growth",
                    "validated_finding": f"{best_path.get('risk_mode', 'unknown')} currently leads the capital efficiency simulation set.",
                    "expected_improvement": "Potential capital efficiency improvement through validated risk-mode governance.",
                    "learning_confidence": confidence,
                    "research_confidence": confidence,
                    "sample_size": sample_size,
                    "evidence_strength": "Candidate",
                    "historical_support": "Simulation-backed observational candidate",
                    "applicable_conditions": {"risk_mode": best_path.get("risk_mode"), "starting_balance": best_path.get("starting_balance")},
                    "risk_assessment": "Requires replay and governance approval before live adaptation.",
                    "expected_capital_impact": "Positive if validated across regime and drawdown states.",
                    "version_history": _build_version_history(
                        prior_library,
                        "C-CAPITAL-GROWTH-001",
                        self.generated_at,
                        self.version,
                        confidence,
                        sample_size,
                    ),
                }
            )

        prior_history = self.load_history(limit=240)
        history_entry = {
            "timestamp": self.generated_at,
            "capital_efficiency": best_path.get("capital_efficiency", STATUS_AWAITING),
            "compounding_efficiency": best_path.get("compounding_efficiency", STATUS_AWAITING),
            "survival_probability": best_path.get("survival_probability", STATUS_AWAITING),
            "risk_mode": best_path.get("risk_mode", STATUS_AWAITING),
        }
        capital_history = (prior_history + [history_entry])[-180:]

        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "current_summary": best_path,
                "capital_evolution_history": capital_history,
                "capital_research": capital_research,
            },
            "history_entry": history_entry,
            "versioned_outputs": [
                {
                    "timestamp": self.generated_at,
                    "category": "capital_optimization_snapshot",
                    "version": self.version,
                    "summary": history_entry,
                }
            ],
        }
        return {
            "engine": engine,
            "capital_growth_intelligence_engine": capital_growth_intelligence_engine,
            "capital_research": capital_research,
            "capital_evolution_history": capital_history,
        }


class LearningIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "learning_intelligence_engine", generated_at)

    def build(
        self,
        *,
        status: dict[str, Any],
        prior_decision: dict[str, Any],
        prior_library: dict[str, Any],
        trade_count: int,
        win_rate: float,
        execution_learning_engine: dict[str, Any],
        edge_engine_output: dict[str, Any],
        capital_engine_output: dict[str, Any],
        risk_intelligence_engine: dict[str, Any],
        conf_evolution: dict[str, Any],
        session_rows: list[dict[str, Any]],
        pattern_rows: list[dict[str, Any]],
        entry_quality_series: list[float],
        exit_quality_series: list[float],
    ) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        best_session = None
        if session_rows:
            candidates = [r for r in session_rows if isinstance(r.get("session_rr"), (int, float))]
            if candidates:
                best_session = max(candidates, key=lambda x: float(x.get("session_rr", -1e9)))
                confidence = round(min(100.0, (int(best_session.get("samples", 0)) / 120.0) * 100.0), 2)
                findings.append(
                    {
                        "finding_id": "F-SESSION-EDGE-001",
                        "finding_type": "session_edge",
                        "validated_finding": f"{best_session.get('session')} currently has highest session RR.",
                        "expected_improvement": "Improved expectancy by session selection discipline",
                        "learning_confidence": confidence,
                        "research_confidence": confidence,
                        "sample_size": int(best_session.get("samples", 0) or 0),
                        "evidence_strength": "Validated" if int(best_session.get("samples", 0) or 0) >= 75 else "Developing",
                        "historical_support": "Validated" if int(best_session.get("samples", 0) or 0) >= 75 else "Developing",
                        "applicable_conditions": {"session": best_session.get("session")},
                        "risk_assessment": "Observational only",
                        "expected_capital_impact": "Positive if regime and confidence filters align",
                        "observed_win_rate": win_rate,
                        "stability": best_session.get("session_edge_stability", STATUS_AWAITING),
                        "drift": STATUS_AWAITING,
                        "version_history": _build_version_history(
                            prior_library,
                            "F-SESSION-EDGE-001",
                            self.generated_at,
                            self.version,
                            confidence,
                            int(best_session.get("samples", 0) or 0),
                        ),
                    }
                )

        top_pattern = pattern_rows[0] if pattern_rows else None
        if top_pattern is not None:
            confidence = round(min(100.0, (int(top_pattern.get("sample_size", 0)) / 150.0) * 100.0), 2)
            findings.append(
                {
                    "finding_id": "F-PATTERN-EVO-001",
                    "finding_type": "pattern_evolution",
                    "validated_finding": f"{top_pattern.get('pattern_id')} has strongest evidence footprint.",
                    "expected_improvement": "Improved repeatability via conditional pattern selection",
                    "learning_confidence": confidence,
                    "research_confidence": confidence,
                    "sample_size": int(top_pattern.get("sample_size", 0) or 0),
                    "evidence_strength": "Validated" if int(top_pattern.get("sample_size", 0) or 0) >= 75 else "Developing",
                    "historical_support": "Validated" if int(top_pattern.get("sample_size", 0) or 0) >= 75 else "Developing",
                    "applicable_conditions": top_pattern.get("additional_conditions", {}),
                    "risk_assessment": "Requires governance approval before execution changes",
                    "expected_capital_impact": "Moderate positive under matching market context",
                    "observed_win_rate": top_pattern.get("base_win_rate", STATUS_AWAITING),
                    "stability": top_pattern.get("institutional_pattern_score", STATUS_AWAITING),
                    "drift": STATUS_AWAITING,
                    "version_history": _build_version_history(
                        prior_library,
                        "F-PATTERN-EVO-001",
                        self.generated_at,
                        self.version,
                        confidence,
                        int(top_pattern.get("sample_size", 0) or 0),
                    ),
                }
            )

        active_hypotheses = []
        active_hypotheses.append(
            {
                "hypothesis_id": "H-CAGR-ADAPTIVE-001",
                "statement": "Adaptive risk mode improves long-term compounding efficiency versus fixed fractional in current dataset.",
                "status": "Active",
                "research_confidence": round(min(100.0, (trade_count / 400.0) * 100.0), 2),
                "sample_size": trade_count,
                "version_history": _build_version_history(
                    prior_library,
                    "H-CAGR-ADAPTIVE-001",
                    self.generated_at,
                    self.version,
                    round(min(100.0, (trade_count / 400.0) * 100.0), 2),
                    trade_count,
                ),
            }
        )
        if float(_safe_float(conf_evolution.get("calibration_error")) or 0.0) > 0.12:
            active_hypotheses.append(
                {
                    "hypothesis_id": "H-CONFIDENCE-CAL-001",
                    "statement": "Confidence calibration thresholds should be tightened to reduce forecast drift.",
                    "status": "Active",
                    "research_confidence": round(min(100.0, (trade_count / 350.0) * 100.0), 2),
                    "sample_size": trade_count,
                    "version_history": _build_version_history(
                        prior_library,
                        "H-CONFIDENCE-CAL-001",
                        self.generated_at,
                        self.version,
                        round(min(100.0, (trade_count / 350.0) * 100.0), 2),
                        trade_count,
                    ),
                }
            )
        dominant_context = (edge_engine_output.get("execution_edge_learning_engine", {}) or {}).get("dominant_edge_context")
        if dominant_context and dominant_context != STATUS_AWAITING:
            active_hypotheses.append(
                {
                    "hypothesis_id": "H-EDGE-CONTEXT-001",
                    "statement": f"Edge concentration in {dominant_context} should be stress-tested against adjacent sessions and regimes.",
                    "status": "Active",
                    "research_confidence": round(min(100.0, (trade_count / 320.0) * 100.0), 2),
                    "sample_size": trade_count,
                    "version_history": _build_version_history(
                        prior_library,
                        "H-EDGE-CONTEXT-001",
                        self.generated_at,
                        self.version,
                        round(min(100.0, (trade_count / 320.0) * 100.0), 2),
                        trade_count,
                    ),
                }
            )

        prev_sample = int((prior_decision.get("knowledge_confidence", {}) or {}).get("sample_size", 0) or 0)
        knowledge_growth = max(0, trade_count - prev_sample)
        knowledge_growth_rate = round(max(0, trade_count - prev_sample) / max(1.0, prev_sample if prev_sample > 0 else 100.0), 4)
        learning_velocity = 0.0
        try:
            started = str(status.get("started_at") or "")
            last_poll = str(status.get("last_poll") or "")
            if started and last_poll:
                ds = datetime.fromisoformat(started.replace("Z", "+00:00"))
                dl = datetime.fromisoformat(last_poll.replace("Z", "+00:00"))
                days = max(1e-9, (dl - ds).total_seconds() / 86400.0)
                learning_velocity = round(trade_count / days, 4)
        except Exception:
            learning_velocity = 0.0

        prior_history = self.load_history(limit=240)
        prev_learning = prior_history[-1] if prior_history else {}
        edge_improvement_rate = float(_safe_float((edge_engine_output.get("execution_edge_learning_engine", {}) or {}).get("expectancy")) or 0.0) - float(_safe_float(prev_learning.get("expectancy")) or 0.0)
        cap_eff = (capital_engine_output.get("capital_growth_intelligence_engine", {}) or {}).get("best_capital_path", {}).get("capital_efficiency", STATUS_AWAITING)
        capital_improvement_rate = 0.0
        if isinstance(cap_eff, (int, float)):
            capital_improvement_rate = round(float(cap_eff) - float(_safe_float(prev_learning.get("capital_efficiency")) or 0.0), 4)
        risk_of_ruin = float(_safe_float(risk_intelligence_engine.get("risk_of_ruin")) or 0.0)
        prev_ror = float(_safe_float(prev_learning.get("risk_of_ruin")) or risk_of_ruin)
        risk_improvement_rate = round(prev_ror - risk_of_ruin, 4)
        model_improvement_rate = round(float(_safe_float((conf_evolution.get("confidence_stability"))) or 0.0) - float(_safe_float(prev_learning.get("confidence_stability")) or 0.0), 4)
        confidence_calibration_growth = round(float(_safe_float(prev_learning.get("calibration_error")) or 0.0) - float(_safe_float(conf_evolution.get("calibration_error")) or 0.0), 4)
        research_growth_rate = round((len(findings) + len(active_hypotheses)) / max(1.0, len(prior_history) if prior_history else 1.0), 4)
        entry_half = max(1, len(entry_quality_series) // 2)
        exit_half = max(1, len(exit_quality_series) // 2)
        early_entry = mean(entry_quality_series[:entry_half]) if entry_quality_series else 0.0
        late_entry = mean(entry_quality_series[entry_half:]) if len(entry_quality_series) > 1 else early_entry
        early_exit = mean(exit_quality_series[:exit_half]) if exit_quality_series else 0.0
        late_exit = mean(exit_quality_series[exit_half:]) if len(exit_quality_series) > 1 else early_exit
        execution_improvement_rate = round(mean([late_entry - early_entry, late_exit - early_exit]), 4)
        learning_efficiency = round(knowledge_growth / max(1.0, len(active_hypotheses)), 4)
        institutional_learning_index = round(
            mean(
                [
                    min(100.0, learning_velocity),
                    min(100.0, knowledge_growth_rate * 100.0),
                    min(100.0, max(0.0, execution_improvement_rate) * 100.0),
                    min(100.0, max(0.0, capital_improvement_rate) * 100.0),
                    min(100.0, max(0.0, risk_improvement_rate) * 100.0),
                ]
            ),
            2,
        )
        adaptive_intelligence = round(
            mean(
                [
                    float(_safe_float((edge_engine_output.get("execution_edge_learning_engine", {}) or {}).get("edge_confidence")) or 0.0) * 100.0,
                    min(100.0, learning_velocity),
                    min(100.0, knowledge_growth_rate * 100.0),
                    float(_safe_float((edge_engine_output.get("execution_edge_learning_engine", {}) or {}).get("edge_robustness")) or 0.0) * 100.0,
                ]
            ),
            2,
        )

        continuous_improvement_metrics = {
            "learning_velocity": learning_velocity,
            "knowledge_growth": knowledge_growth,
            "knowledge_growth_rate": knowledge_growth_rate,
            "execution_improvement_rate": execution_improvement_rate,
            "edge_improvement_rate": round(edge_improvement_rate, 4),
            "capital_improvement_rate": round(capital_improvement_rate, 4),
            "risk_improvement_rate": round(risk_improvement_rate, 4),
            "model_improvement_rate": model_improvement_rate,
            "recommendation_accuracy_growth": STATUS_AWAITING,
            "confidence_calibration_growth": confidence_calibration_growth,
            "research_growth_rate": research_growth_rate,
            "learning_efficiency": learning_efficiency,
            "institutional_learning_index": institutional_learning_index,
            "capital_growth_efficiency": cap_eff,
            "risk_learning_score": round(max(0.0, min(100.0, (1.0 - risk_of_ruin) * 100.0)), 2),
            "pattern_evolution_score": 0.0,
            "session_mastery_score": round(mean([float(_safe_float(x.get("session_edge_stability")) or 0.0) for x in session_rows]) * 100.0, 2) if session_rows else 0.0,
            "execution_improvement_score": round(mean([execution_learning_engine.get("execution_consistency", 0.0), execution_learning_engine.get("timing_efficiency", 0.0)]) * 100.0, 2),
            "research_confidence": round(mean([float(_safe_float(f.get("research_confidence")) or 0.0) for f in findings]), 2) if findings else 0.0,
            "research_confidence_index": round(mean([float(_safe_float(f.get("research_confidence")) or 0.0) for f in findings]), 2) if findings else 0.0,
            "adaptive_intelligence": adaptive_intelligence,
            "institutional_knowledge_score": round(min(100.0, (trade_count / 1200.0) * 100.0), 2),
        }

        candidate_findings: list[dict[str, Any]] = []
        rejected_findings: list[dict[str, Any]] = []
        learning_experiments = []
        for hypothesis in active_hypotheses:
            learning_experiments.append(
                {
                    "experiment_id": hypothesis.get("hypothesis_id"),
                    "objective": hypothesis.get("statement"),
                    "status": "Queued for historical validation",
                    "research_confidence": hypothesis.get("research_confidence"),
                    "sample_size": hypothesis.get("sample_size"),
                    "version": self.version,
                    "timestamp": self.generated_at,
                }
            )

        improved_areas = []
        deteriorated_areas = []
        if execution_improvement_rate > 0:
            improved_areas.append("execution_quality")
        elif execution_improvement_rate < 0:
            deteriorated_areas.append("execution_quality")
        if capital_improvement_rate > 0:
            improved_areas.append("capital_growth")
        elif capital_improvement_rate < 0:
            deteriorated_areas.append("capital_growth")
        if risk_improvement_rate > 0:
            improved_areas.append("risk_preservation")
        elif risk_improvement_rate < 0:
            deteriorated_areas.append("risk_preservation")
        if edge_improvement_rate > 0:
            improved_areas.append("edge_quality")
        elif edge_improvement_rate < 0:
            deteriorated_areas.append("edge_quality")

        history_entry = {
            "timestamp": self.generated_at,
            "learning_velocity": learning_velocity,
            "knowledge_growth": knowledge_growth,
            "expectancy": (edge_engine_output.get("execution_edge_learning_engine", {}) or {}).get("expectancy"),
            "capital_efficiency": cap_eff,
            "risk_of_ruin": risk_of_ruin,
            "confidence_stability": conf_evolution.get("confidence_stability"),
            "calibration_error": conf_evolution.get("calibration_error"),
        }
        learning_velocity_history = (prior_history + [history_entry])[-180:]
        execution_evolution_history = [
            {
                "timestamp": self.generated_at,
                "entry_quality": execution_learning_engine.get("entry_quality", 0.0),
                "exit_quality": execution_learning_engine.get("exit_quality", 0.0),
                "timing_efficiency": execution_learning_engine.get("timing_efficiency", 0.0),
            }
        ]

        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "current_summary": continuous_improvement_metrics,
                "learning_velocity_history": learning_velocity_history,
                "self_improvement_assessment": {
                    "improved_areas": improved_areas,
                    "deteriorated_areas": deteriorated_areas,
                    "learning_faster": learning_velocity > float(_safe_float(prev_learning.get("learning_velocity")) or 0.0),
                    "institutional_learning_index": institutional_learning_index,
                },
                "learning_experiments": learning_experiments,
            },
            "history_entry": history_entry,
            "versioned_outputs": [
                {
                    "timestamp": self.generated_at,
                    "category": "learning_velocity_snapshot",
                    "version": self.version,
                    "summary": history_entry,
                }
            ],
        }
        return {
            "engine": engine,
            "historical_knowledge_mining": {
                "validated_findings": findings,
                "active_hypotheses": active_hypotheses,
                "rejected_hypotheses": rejected_findings,
            },
            "continuous_improvement_metrics": continuous_improvement_metrics,
            "validated_findings": findings,
            "candidate_findings": candidate_findings,
            "rejected_findings": rejected_findings,
            "learning_experiments": learning_experiments,
            "learning_velocity_history": learning_velocity_history,
            "execution_evolution_history": execution_evolution_history,
        }


class RecommendationIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "recommendation_intelligence_engine", generated_at)

    def build(
        self,
        *,
        findings: list[dict[str, Any]],
        candidate_findings: list[dict[str, Any]],
        risk_improvements: list[dict[str, Any]],
        capital_research: list[dict[str, Any]],
        continuous_improvement_metrics: dict[str, Any],
        sample_size: int,
        prior_library: dict[str, Any],
    ) -> dict[str, Any]:
        prior_state = self.load_state()
        prior_recommendations = (prior_state.get("state", {}) or {}).get("recommendations", []) or []
        prior_map = {str(row.get("recommendation_id") or ""): row for row in prior_recommendations}

        recommendations = []
        source_rows = findings + candidate_findings + risk_improvements + capital_research
        for idx, row in enumerate(source_rows, start=1):
            confidence = float(_safe_float(row.get("research_confidence")) or 0.0)
            recommendation_id = str(row.get("finding_id") or row.get("hypothesis_id") or f"REC-{idx:03d}")
            priority = "High" if confidence >= 80 else "Medium" if confidence >= 55 else "Low"
            previous = prior_map.get(recommendation_id, {})
            prior_conf = float(_safe_float(previous.get("confidence")) or 0.0)
            recommendations.append(
                {
                    "recommendation_id": recommendation_id,
                    "type": row.get("finding_type", row.get("type", "validated_finding")),
                    "finding": row.get("validated_finding") or row.get("statement") or row.get("expected_improvement"),
                    "expected_improvement": row.get("expected_improvement"),
                    "confidence": confidence,
                    "sample_size": int(row.get("sample_size", sample_size) or sample_size),
                    "historical_evidence": row.get("evidence_strength", row.get("historical_support", "Developing")),
                    "applicable_conditions": row.get("applicable_conditions", {}),
                    "expected_capital_impact": row.get("expected_capital_impact", "Neutral"),
                    "expected_risk_impact": row.get("risk_assessment", "Governance review required."),
                    "recommendation_priority": priority,
                    "research_confidence": confidence,
                    "version": self.version,
                    "implementation_status": previous.get("implementation_status", "Governance Pending"),
                    "governance_required": True,
                    "confidence_change": round(confidence - prior_conf, 2),
                }
            )

        if not recommendations:
            recommendations.append(
                {
                    "recommendation_id": "REC-EMPTY-001",
                    "type": "validated_finding",
                    "finding": "Insufficient evidence for high-confidence recommendation.",
                    "expected_improvement": "Continue data collection and historical validation.",
                    "confidence": 0.0,
                    "sample_size": sample_size,
                    "historical_evidence": "Developing",
                    "applicable_conditions": {},
                    "expected_capital_impact": "Neutral",
                    "expected_risk_impact": "No governed change proposed.",
                    "recommendation_priority": "Low",
                    "research_confidence": 0.0,
                    "version": self.version,
                    "implementation_status": "Governance Pending",
                    "governance_required": True,
                    "confidence_change": 0.0,
                }
            )

        avg_confidence = mean([float(_safe_float(r.get("confidence")) or 0.0) for r in recommendations]) if recommendations else 0.0
        outcome_tracking = []
        for rec in recommendations:
            outcome_tracking.append(
                {
                    "recommendation_id": rec.get("recommendation_id"),
                    "implemented": False,
                    "implementation_status": rec.get("implementation_status"),
                    "outcome_status": "Awaiting governed adoption",
                    "confidence_delta": rec.get("confidence_change", 0.0),
                    "version_history": _build_version_history(
                        prior_library,
                        str(rec.get("recommendation_id")),
                        self.generated_at,
                        self.version,
                        rec.get("confidence"),
                        int(rec.get("sample_size", sample_size) or sample_size),
                    ),
                }
            )

        prior_history = self.load_history(limit=240)
        prev_accuracy = float(_safe_float((prior_history[-1] if prior_history else {}).get("calibration_score")) or 0.0)
        calibration_score = round(min(100.0, avg_confidence), 2)
        recommendation_accuracy_growth = round(calibration_score - prev_accuracy, 4)
        history_entry = {
            "timestamp": self.generated_at,
            "recommendation_count": len(recommendations),
            "average_confidence": round(avg_confidence, 2),
            "calibration_score": calibration_score,
            "recommendation_accuracy_growth": recommendation_accuracy_growth,
        }
        recommendation_history = (prior_history + [history_entry])[-180:]

        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "recommendations": recommendations,
                "recommendation_history": recommendation_history,
                "outcome_tracking": outcome_tracking,
                "calibration_summary": {
                    "average_confidence": round(avg_confidence, 2),
                    "recommendation_accuracy_growth": recommendation_accuracy_growth,
                    "governed_adoption_only": True,
                },
            },
            "history_entry": history_entry,
            "versioned_outputs": [
                {
                    "timestamp": self.generated_at,
                    "category": "recommendation_snapshot",
                    "version": self.version,
                    "summary": history_entry,
                }
            ],
        }
        return {
            "engine": engine,
            "recommendation_engine_evolution": {
                "recommendations": recommendations,
                "descriptive_to_evidence_based": True,
                "feedback_loop": {
                    "outcome_tracking": outcome_tracking,
                    "recommendation_accuracy_growth": recommendation_accuracy_growth,
                    "calibration_score": calibration_score,
                },
            },
            "recommendation_outcomes": outcome_tracking,
            "recommendation_accuracy_growth": recommendation_accuracy_growth,
        }


class ExecutionLocationIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "execution_location_intelligence_engine", generated_at)

    def build(
        self,
        *,
        trades: list[dict[str, Any]],
        setups: list[dict[str, Any]],
        sample_size: int,
        prior_library: dict[str, Any],
    ) -> dict[str, Any]:
        scores = [float(_safe_float(t.get("score_at_entry")) or 0.0) / 100.0 for t in trades if _safe_float(t.get("score_at_entry")) is not None]
        location_quality = mean(scores) if scores else 0.0
        sweep_exposure = sum(1 for t in trades if "sweep" in _norm_regime(t.get("regime")).lower()) / max(1, len(trades))
        late_entries = [t for t in trades if float(_safe_float(t.get("score_at_entry")) or 0.0) < 55.0 and float(_safe_float(t.get("pnl")) or 0.0) < 0.0]
        premature_entries = [t for t in trades if float(_safe_float(t.get("hold_seconds")) or 0.0) < 300.0 and float(_safe_float(t.get("pnl")) or 0.0) < 0.0]
        chasing_expansion = [t for t in trades if "trend expansion" in _norm_regime(t.get("regime")).lower() and float(_safe_float(t.get("pnl")) or 0.0) < 0.0]
        liquidity_proximity_proxy = mean([1.0 if "liquidity" in _norm_regime(t.get("regime")).lower() else 0.45 for t in trades]) if trades else 0.0
        stop_location_quality = mean([
            max(0.0, min(1.0, abs(float(_safe_float(t.get("mae")) or 0.0)) / max(1e-9, abs(float(_safe_float(t.get("mfe")) or 0.0)) + 1e-9)))
            for t in trades
        ]) if trades else 0.0

        setup_window = setups[-max(1, min(250, len(setups))):] if setups else []
        structural_price_location_intelligence = {
            "retracement_depth_proxy": round(mean([float(_safe_float(s.get("fib_proximity")) or 0.0) for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "extension_depth_proxy": round(mean([float(_safe_float(s.get("trend_strength")) or 0.0) for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "premium_discount_proxy": round(mean([float(_safe_float(s.get("mtf_score")) or 0.0) for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "swing_location_proxy": round(mean([1.0 if int(s.get("ob_present", 0) or 0) else 0.35 for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "liquidity_location_proxy": round(mean([1.0 if int(s.get("stop_hunt", 0) or 0) else 0.3 for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "order_block_proximity_proxy": round(mean([float(int(s.get("ob_present", 0) or 0)) for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "fair_value_gap_proxy": round(mean([float(_safe_float(s.get("candlestick_score")) or 0.0) / 100.0 for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "volatility_phase_proxy": round(mean([float(_safe_float(s.get("trend_strength")) or 0.0) for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
            "trend_maturity_proxy": round(mean([float(_safe_float(s.get("mtf_score")) or 0.0) for s in setup_window]), 4) if setup_window else STATUS_AWAITING,
        }
        execution_location_research = [
            {
                "finding_id": "X-LOC-001",
                "finding_type": "execution_location",
                "validated_finding": "Execution location quality is now tracked as a permanent research stream.",
                "expected_improvement": "Reduce structurally poor entries and improve location-dependent execution quality.",
                "research_confidence": round(min(100.0, (sample_size / 300.0) * 100.0), 2),
                "sample_size": sample_size,
                "historical_evidence": "Closed trade execution telemetry and setup context",
                "applicable_conditions": {"dominant_regimes": sorted({_norm_regime(t.get('regime')) for t in trades})[:5]},
                "expected_execution_improvement": "Higher location quality and lower sweep exposure.",
                "version_history": _build_version_history(prior_library, "X-LOC-001", self.generated_at, self.version, round(min(100.0, (sample_size / 300.0) * 100.0), 2), sample_size),
            }
        ]
        location_errors = [
            {"error": "entering_into_liquidity", "count": len(chasing_expansion), "impact_proxy": round(sum(abs(float(_safe_float(t.get('pnl')) or 0.0)) for t in chasing_expansion), 4)},
            {"error": "late_execution", "count": len(late_entries), "impact_proxy": round(sum(abs(float(_safe_float(t.get('pnl')) or 0.0)) for t in late_entries), 4)},
            {"error": "premature_execution", "count": len(premature_entries), "impact_proxy": round(sum(abs(float(_safe_float(t.get('pnl')) or 0.0)) for t in premature_entries), 4)},
        ]
        summary = {
            "execution_location_quality": round(location_quality, 4),
            "liquidity_proximity": round(liquidity_proximity_proxy, 4),
            "structural_position": structural_price_location_intelligence.get("swing_location_proxy", STATUS_AWAITING),
            "breakout_quality_proxy": round(mean([float(_safe_float(t.get("score_at_entry")) or 0.0) / 100.0 for t in trades if float(_safe_float(t.get("score_at_entry")) or 0.0) >= 60.0]), 4) if trades else STATUS_AWAITING,
            "inducement_exposure": structural_price_location_intelligence.get("liquidity_location_proxy", STATUS_AWAITING),
            "execution_timing": round(location_quality, 4),
            "sweep_exposure": round(sweep_exposure, 4),
            "stop_location_quality": round(stop_location_quality, 4),
            "execution_efficiency": round(mean([max(0.0, min(1.0, float(_safe_float(t.get("mfe")) or 0.0) / max(1e-9, abs(float(_safe_float(t.get("mae")) or 0.0)) + 1e-9))) for t in trades]), 4) if trades else STATUS_AWAITING,
            "recurring_execution_location_errors": location_errors,
        }
        prior_history = self.load_history(limit=240)
        history_entry = {"timestamp": self.generated_at, **{k: v for k, v in summary.items() if k != "recurring_execution_location_errors"}}
        history = (prior_history + [history_entry])[-180:]
        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "current_summary": summary,
                "structural_price_location_intelligence": structural_price_location_intelligence,
                "execution_location_research": execution_location_research,
                "execution_location_history": history,
            },
            "history_entry": history_entry,
            "versioned_outputs": [{"timestamp": self.generated_at, "category": "execution_location_snapshot", "version": self.version, "summary": summary}],
        }
        return {
            "engine": engine,
            "execution_location_intelligence": summary,
            "structural_price_location_intelligence": structural_price_location_intelligence,
            "execution_location_research": execution_location_research,
        }


class CapitalPreservationIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "capital_preservation_intelligence_engine", generated_at)

    def build(self, *, trades: list[dict[str, Any]], sample_size: int, prior_library: dict[str, Any]) -> dict[str, Any]:
        leakage_rows = []
        for trade in trades:
            pnl = float(_safe_float(trade.get("pnl")) or 0.0)
            if pnl >= 0:
                continue
            score = float(_safe_float(trade.get("score_at_entry")) or 0.0)
            hold_seconds = float(_safe_float(trade.get("hold_seconds")) or 0.0)
            regime = _norm_regime(trade.get("regime"))
            if "Liquidity Sweep" == regime:
                cause = "liquidity_sweep_loss"
            elif score < 50.0:
                cause = "late_entry"
            elif hold_seconds < 300.0:
                cause = "poor_execution_timing"
            elif abs(float(_safe_float(trade.get("mae")) or 0.0)) > abs(float(_safe_float(trade.get("mfe")) or 0.0)):
                cause = "stop_location_quality"
            else:
                cause = "general_capital_leakage"
            leakage_rows.append(
                {
                    "cause": cause,
                    "capital_impact": round(abs(pnl), 4),
                    "recovery_cost": round(abs(pnl) * 1.25, 4),
                    "expected_improvement": "Reduce recurrence through governed execution refinement.",
                    "session": _norm_session(trade.get("session")),
                    "regime": regime,
                }
            )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in leakage_rows:
            grouped[str(row.get("cause"))].append(row)
        cause_rows = []
        for cause, rows in grouped.items():
            cause_rows.append(
                {
                    "cause": cause,
                    "historical_frequency": len(rows),
                    "expected_capital_impact": round(sum(float(_safe_float(r.get("capital_impact")) or 0.0) for r in rows), 4),
                    "recovery_cost": round(sum(float(_safe_float(r.get("recovery_cost")) or 0.0) for r in rows), 4),
                    "potential_improvement": "Avoidable capital erosion can reduce through evidence-backed gating.",
                }
            )
        cause_rows.sort(key=lambda row: float(row.get("expected_capital_impact", 0.0)), reverse=True)
        capital_preservation_research = [
            {
                "finding_id": "CP-001",
                "finding_type": "capital_preservation",
                "validated_finding": "Capital leakage is tracked as a persistent institutional intelligence domain.",
                "expected_improvement": "Reduce avoidable capital erosion and recovery cost.",
                "research_confidence": round(min(100.0, (sample_size / 280.0) * 100.0), 2),
                "sample_size": sample_size,
                "historical_evidence": "Closed loss trades with causal attribution proxies",
                "applicable_conditions": {"top_causes": [row.get("cause") for row in cause_rows[:3]]},
                "expected_capital_impact": "Lower leakage and faster recovery.",
                "version_history": _build_version_history(prior_library, "CP-001", self.generated_at, self.version, round(min(100.0, (sample_size / 280.0) * 100.0), 2), sample_size),
            }
        ]
        summary = {
            "capital_leakage_events": len(leakage_rows),
            "total_capital_leakage": round(sum(float(_safe_float(r.get("capital_impact")) or 0.0) for r in leakage_rows), 4),
            "capital_preservation_score": round(max(0.0, 1.0 - min(1.0, sum(float(_safe_float(r.get("capital_impact")) or 0.0) for r in leakage_rows) / max(1.0, sample_size * 10.0))), 4),
            "top_leakage_causes": cause_rows[:10],
        }
        prior_history = self.load_history(limit=240)
        history_entry = {"timestamp": self.generated_at, "capital_leakage_events": summary["capital_leakage_events"], "total_capital_leakage": summary["total_capital_leakage"], "capital_preservation_score": summary["capital_preservation_score"]}
        history = (prior_history + [history_entry])[-180:]
        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "current_summary": summary,
                "capital_preservation_research": capital_preservation_research,
                "capital_preservation_history": history,
            },
            "history_entry": history_entry,
            "versioned_outputs": [{"timestamp": self.generated_at, "category": "capital_preservation_snapshot", "version": self.version, "summary": summary}],
        }
        return {
            "engine": engine,
            "capital_preservation_intelligence": summary,
            "capital_preservation_research": capital_preservation_research,
        }


class DecisionAttributionIntelligenceEngine(PersistentIntelligenceEngine):
    def __init__(self, root_dir: Path, generated_at: str) -> None:
        super().__init__(root_dir, "decision_attribution_intelligence_engine", generated_at)

    def build(self, *, trades: list[dict[str, Any]], sample_size: int, prior_library: dict[str, Any]) -> dict[str, Any]:
        success_rows = [t for t in trades if float(_safe_float(t.get("pnl")) or 0.0) > 0.0]
        failure_rows = [t for t in trades if float(_safe_float(t.get("pnl")) or 0.0) < 0.0]
        def factor_score(rows: list[dict[str, Any]], factor: str) -> float:
            if not rows:
                return 0.0
            if factor == "market_selection":
                return mean([1.0 if _norm_regime(r.get("regime")) not in ("Compression", "Range") else 0.45 for r in rows])
            if factor == "session_selection":
                return mean([1.0 if _norm_session(r.get("session")) not in ("Dead Zone",) else 0.25 for r in rows])
            if factor == "market_structure":
                return mean([float(_safe_float(r.get("score_at_entry")) or 0.0) / 100.0 for r in rows])
            if factor == "structural_price_location":
                return mean([max(0.0, min(1.0, float(_safe_float(r.get("mfe")) or 0.0) / max(1e-9, abs(float(_safe_float(r.get("mae")) or 0.0)) + 1e-9))) for r in rows])
            if factor == "liquidity":
                return mean([1.0 if "liquidity" in _norm_regime(r.get("regime")).lower() else 0.4 for r in rows])
            if factor == "confidence":
                return mean([float(_safe_float(r.get("score_at_entry")) or 0.0) / 100.0 for r in rows])
            if factor == "risk":
                return mean([1.0 - min(1.0, abs(float(_safe_float(r.get("mae")) or 0.0)) / max(1.0, abs(float(_safe_float(r.get("mfe")) or 0.0)) + 1.0)) for r in rows])
            if factor == "entry_timing":
                return mean([1.0 if float(_safe_float(r.get("hold_seconds")) or 0.0) >= 300.0 else 0.4 for r in rows])
            if factor == "stop_placement":
                return mean([max(0.0, min(1.0, abs(float(_safe_float(r.get("mae")) or 0.0)) / max(1e-9, abs(float(_safe_float(r.get("mfe")) or 0.0)) + 1e-9))) for r in rows])
            if factor == "exit_timing":
                return mean([1.0 if str(r.get("exit_reason", "")).lower() in ("tp", "tp1", "tp2", "time_smart") else 0.35 for r in rows])
            if factor == "trade_management":
                return mean([1.0 if float(_safe_float(r.get("pnl")) or 0.0) > 0 else 0.0 for r in rows])
            return 0.0

        factors = [
            "market_selection",
            "session_selection",
            "market_structure",
            "structural_price_location",
            "liquidity",
            "confidence",
            "risk",
            "entry_timing",
            "stop_placement",
            "exit_timing",
            "trade_management",
        ]
        contributions = []
        for factor in factors:
            success_score = round(factor_score(success_rows, factor), 4)
            failure_score = round(factor_score(failure_rows, factor), 4)
            contributions.append(
                {
                    "factor": factor,
                    "success_contribution": success_score,
                    "failure_contribution": failure_score,
                    "net_institutional_edge": round(success_score - failure_score, 4),
                }
            )
        contributions.sort(key=lambda row: abs(float(_safe_float(row.get("net_institutional_edge")) or 0.0)), reverse=True)
        decision_attribution = {
            "top_success_factors": [row for row in contributions if float(_safe_float(row.get("net_institutional_edge")) or 0.0) > 0][:5],
            "top_failure_factors": [row for row in contributions if float(_safe_float(row.get("net_institutional_edge")) or 0.0) < 0][:5],
            "capital_leakage_behaviours": [row for row in contributions if row.get("factor") in ("entry_timing", "risk", "stop_placement")][:3],
            "decision_attribution_matrix": contributions,
        }
        attribution_research = [
            {
                "finding_id": "DA-001",
                "finding_type": "decision_attribution",
                "validated_finding": "Decision attribution is persisted as institutional execution research.",
                "expected_improvement": "Identify the decisions contributing most to success, failure, and capital leakage.",
                "research_confidence": round(min(100.0, (sample_size / 280.0) * 100.0), 2),
                "sample_size": sample_size,
                "historical_evidence": "Closed trade outcome attribution proxies",
                "applicable_conditions": {"factors": factors},
                "expected_execution_improvement": "Higher decision quality through causal ranking.",
                "version_history": _build_version_history(prior_library, "DA-001", self.generated_at, self.version, round(min(100.0, (sample_size / 280.0) * 100.0), 2), sample_size),
            }
        ]
        prior_history = self.load_history(limit=240)
        history_entry = {"timestamp": self.generated_at, "top_factor": (contributions[0].get("factor") if contributions else STATUS_AWAITING), "top_factor_net_edge": (contributions[0].get("net_institutional_edge") if contributions else STATUS_AWAITING)}
        history = (prior_history + [history_entry])[-180:]
        engine = {
            "engine_key": self.engine_key,
            "engine_version": self.version,
            "json_contract": self.contract(),
            "historical_dataset": self.dataset_contract(),
            "state": {
                "current_summary": decision_attribution,
                "decision_attribution_history": history,
                "decision_attribution_research": attribution_research,
            },
            "history_entry": history_entry,
            "versioned_outputs": [{"timestamp": self.generated_at, "category": "decision_attribution_snapshot", "version": self.version, "summary": decision_attribution}],
        }
        return {
            "engine": engine,
            "decision_attribution_intelligence": decision_attribution,
            "decision_attribution_research": attribution_research,
        }


def write_engine_artifacts(root_dir: Path, engine_payload: dict[str, Any]) -> dict[str, str]:
    state = engine_payload.get("state", {}) or {}
    history_entry = engine_payload.get("history_entry")
    versioned_outputs = engine_payload.get("versioned_outputs", []) or []
    historical_dataset = engine_payload.get("historical_dataset", {}) or {}

    state_path = root_dir / str(historical_dataset.get("state_path", ""))
    history_path = root_dir / str(historical_dataset.get("history_path", ""))
    research_path = root_dir / str(historical_dataset.get("research_path", ""))

    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(engine_payload, indent=2), encoding="utf-8")

    if history_entry and history_path:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_entry) + "\n")

    if versioned_outputs and research_path:
        research_path.parent.mkdir(parents=True, exist_ok=True)
        with research_path.open("a", encoding="utf-8") as handle:
            for row in versioned_outputs:
                handle.write(json.dumps(row) + "\n")

    return {
        "state_path": str(state_path.relative_to(root_dir)) if state_path else "",
        "history_path": str(history_path.relative_to(root_dir)) if history_path else "",
        "research_path": str(research_path.relative_to(root_dir)) if research_path else "",
    }
