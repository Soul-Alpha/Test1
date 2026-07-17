from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olympus.contracts import SourceSystem
from olympus.core.hermes_analytics import build_hermes_analytics
from olympus.core.validation_contracts import (
    OutcomeDiagnostics,
    PatternCandidate,
    PreEntryFeatureSet,
    RecommendationCandidate,
    ValidationDomain,
    ValidationLifecycle,
    validation_lifecycle_order,
)
from backtesting.zeus_validation import ZeusValidationEngine, build_validation_status


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _maturity_from_samples(sample_size: int) -> str:
    if sample_size >= 300:
        return "Institutional"
    if sample_size >= 120:
        return "Validated"
    if sample_size >= 50:
        return "Developing"
    return "Emerging"


def _prometheus_recommendation_candidates(intelligence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (
        (intelligence.get("recommendation_engine_evolution", {}) or {}).get("recommendations", [])
        if isinstance(intelligence, dict)
        else []
    )
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        rec = RecommendationCandidate(
            candidate_id=f"prom-rec-{idx:04d}",
            source_system=SourceSystem.PROMETHEUS,
            timestamp=_utc_now(),
            recommendation_type=str(row.get("recommendation_type") or "execution_improvement"),
            recommendation=str(row.get("finding") or row.get("recommendation") or "Awaiting Historical Data"),
            expected_improvement=str(row.get("expected_capital_impact") or "Evidence pending"),
            evidence={
                "confidence": row.get("confidence", 0.0),
                "sample_size": row.get("sample_size", 0),
                "priority": row.get("recommendation_priority", "Low"),
                "source": "prometheus_evolution",
            },
            lifecycle=ValidationLifecycle.CANDIDATE,
            operator_approved=False,
        )
        record = rec.as_dict()
        record.update(
            {
                "source_system": SourceSystem.PROMETHEUS.value,
                "version": str(row.get("version") or row.get("validation_version") or "2.2"),
                "mission": "Execution Intelligence",
                "submission_time": record["timestamp"],
                "priority": str(row.get("recommendation_priority") or "Normal"),
            }
        )
        out.append(record)
    return out


def _prometheus_feature_candidates(trades: list[dict[str, Any]], setups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not setups:
        return []

    # Feature contracts are intentionally separated from outcome diagnostics to avoid leakage.
    closed = [row for row in trades if row.get("trade_id") is not None]
    trade_idx = {str(row.get("trade_id")): row for row in closed}

    out: list[dict[str, Any]] = []
    for idx, setup in enumerate(setups[-120:], start=1):
        setup_id = str(setup.get("setup_id") or f"setup-{idx:04d}")
        pre = PreEntryFeatureSet(
            feature_set_id=f"pre-entry-{setup_id}",
            source_system=SourceSystem.PROMETHEUS,
            instrument="XAUUSDm",
            timeframe="M5",
            timestamp=_utc_now(),
            feature_version=str(setup.get("feature_version") or "v1"),
            market_structure={
                "structure_type": setup.get("structure_type"),
                "trend_strength": setup.get("trend_strength"),
                "mtf_score": setup.get("mtf_score"),
            },
            liquidity={
                "stop_hunt": setup.get("stop_hunt"),
                "ob_present": setup.get("ob_present"),
            },
            execution_location={
                "fib_proximity": setup.get("fib_proximity"),
                "sr_confidence": setup.get("sr_confidence"),
            },
            volatility={"candlestick_score": setup.get("candlestick_score")},
            session={"session": setup.get("session")},
            statistical={"pattern_confidence": setup.get("pattern_confidence")},
            metadata={"setup_id": setup_id},
        )

        trade_row = trade_idx.get(setup_id, {})
        outc = OutcomeDiagnostics(
            diagnostics_id=f"outcome-{setup_id}",
            source_system=SourceSystem.PROMETHEUS,
            instrument="XAUUSDm",
            timeframe="M5",
            timestamp=_utc_now(),
            execution_result_id=setup_id,
            outcome="win" if (_safe_float(trade_row.get("pnl")) or 0.0) > 0 else "loss",
            pnl=_safe_float(trade_row.get("pnl")),
            rr=_safe_float(trade_row.get("rr")),
            mae=_safe_float(trade_row.get("mae")),
            mfe=_safe_float(trade_row.get("mfe")),
            hold_seconds=int(trade_row.get("hold_seconds", 0) or 0),
            exit_reason=str(trade_row.get("exit_reason") or "unknown"),
            diagnostics={"raw_trade_link": setup_id},
        )

        out.append(
            {
                "candidate_id": f"prom-feature-{idx:04d}",
                "source_system": SourceSystem.PROMETHEUS.value,
                "validation_domain": ValidationDomain.FEATURE.value,
                "timestamp": _utc_now(),
                "version": "2.2",
                "mission": "Execution Intelligence",
                "submission_time": _utc_now(),
                "priority": "Normal",
                "outcome_diagnostics_excluded": True,
                "pre_entry_feature_set": pre.as_dict(),
                "outcome_diagnostics": outc.as_dict(),
                "evidence": {"sample_size": 1, "confidence": 0.0},
            }
        )
    return out


def _hermes_pattern_candidates(root_dir: Path) -> list[dict[str, Any]]:
    try:
        analytics = build_hermes_analytics(root_dir)
    except Exception:
        analytics = {}

    pattern_rows = (
        (analytics.get("pattern_context_intelligence", {}) or {}).get("pattern_context_library", [])
        if isinstance(analytics, dict)
        else []
    )
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(pattern_rows[:120], start=1):
        pattern = PatternCandidate(
            candidate_id=f"hermes-pattern-{idx:04d}",
            source_system=SourceSystem.HERMES,
            pattern_name=str(row.get("pattern_id") or f"pattern-{idx:04d}"),
            timestamp=_utc_now(),
            instrument="XAUUSDm",
            timeframe="M5",
            context={
                "session": row.get("best_session"),
                "regime": row.get("best_market_regime"),
                "structural_behaviour": row.get("structural_behaviour"),
            },
            evidence={
                "sample_size": row.get("occurrences", 0),
                "win_rate": row.get("win_rate"),
                "expectancy": row.get("expectancy"),
                "knowledge_confidence": row.get("knowledge_confidence"),
            },
            lifecycle=ValidationLifecycle.CANDIDATE,
            operator_approved=False,
        )
        record = pattern.as_dict()
        record.update(
            {
                "source_system": SourceSystem.HERMES.value,
                "version": str(row.get("version") or row.get("pattern_version") or "2.0"),
                "mission": "Pattern Intelligence",
                "submission_time": record["timestamp"],
                "priority": str(row.get("priority") or "Normal"),
            }
        )
        out.append(record)
    return out


def _lifecycle_summary(*, candidate_count: int, report_count: int, approved_count: int, active_count: int) -> dict[str, Any]:
    return {
        "order": validation_lifecycle_order(),
        "counts": {
            ValidationLifecycle.LEARNING.value: candidate_count,
            ValidationLifecycle.CANDIDATE.value: candidate_count,
            ValidationLifecycle.ZEUS_VALIDATION.value: report_count,
            ValidationLifecycle.VALIDATED.value: 0,
            ValidationLifecycle.OPERATOR_APPROVED.value: approved_count,
            ValidationLifecycle.ACTIVE.value: active_count,
            ValidationLifecycle.MONITORING.value: active_count,
            ValidationLifecycle.RETIRED.value: 0,
        },
        "current_stage": ValidationLifecycle.ZEUS_VALIDATION.value if report_count else ValidationLifecycle.CANDIDATE.value,
    }


def build_institutional_validation_pipeline(
    *,
    root_dir: Path,
    intelligence: dict[str, Any],
    trades: list[dict[str, Any]],
    setups: list[dict[str, Any]],
) -> dict[str, Any]:
    zeus = ZeusValidationEngine()

    recommendation_candidates = _prometheus_recommendation_candidates(intelligence)
    feature_candidates = _prometheus_feature_candidates(trades, setups)
    hermes_pattern_candidates = _hermes_pattern_candidates(root_dir)

    recommendation_reports = [zeus.validate_recommendation(row) for row in recommendation_candidates]
    feature_reports = [zeus.validate_feature_candidate(row) for row in feature_candidates]
    pattern_reports = [zeus.validate_pattern_candidate(row) for row in hermes_pattern_candidates]
    reports = recommendation_reports + feature_reports + pattern_reports

    status = build_validation_status(reports)
    total_candidates = len(recommendation_candidates) + len(feature_candidates) + len(hermes_pattern_candidates)
    lifecycle = _lifecycle_summary(
        candidate_count=total_candidates,
        report_count=len(reports),
        approved_count=0,
        active_count=0,
    )

    status["lifecycle"] = lifecycle
    status["version"] = "zeus-v2.0"
    status["research_status"] = "Hermes and Prometheus candidates in Zeus validation queue"
    status["learning_velocity"] = round(total_candidates / max(1, len(setups[-120:])), 4)
    status["institutional_maturity"] = _maturity_from_samples(len(setups))

    return {
        "validation_status": status,
        "validation_reports": [report.as_dict() for report in reports],
        "pipelines": {
            "hermes_research_pipeline": [
                "Hermes Research",
                "Research Candidate",
                "Zeus Validation",
                "Validation Report",
                "Operator Review",
                "Approved Knowledge",
                "Prometheus Adoption (Optional)",
            ],
            "prometheus_evolution_pipeline": [
                "Candidate Improvements",
                "Research Candidates",
                "Zeus Validation",
                "Evidence",
                "Operator Approval",
                "Prometheus Version Update",
            ],
        },
        "contracts": {
            "shared_contracts": [
                "SignalIntent",
                "ExecutionPlan",
                "ExecutionPolicy",
                "ExecutionResult",
                "PreEntryFeatureSet",
                "OutcomeDiagnostics",
                "PatternCandidate",
                "PatternValidationReport",
                "RecommendationCandidate",
                "ValidationReport",
                "CapitalStudy",
            ],
            "feature_leakage_policy": {
                "separate_pre_entry_and_outcome_diagnostics": True,
                "prometheus_training_uses_pre_entry_only": True,
                "outcome_diagnostics_for_post_trade_validation_only": True,
            },
        },
        "candidate_counts": {
            "prometheus_recommendations": len(recommendation_candidates),
            "prometheus_features": len(feature_candidates),
            "hermes_patterns": len(hermes_pattern_candidates),
            "total": total_candidates,
        },
    }