"""Knowledge Growth Dashboard (KGD).

Run:
    streamlit run ui/knowledge_growth_dashboard.py --server.port=8511
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from olympus.core.institutional_state_recovery import recover_institutional_state

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_STORAGE = _ROOT / "storage" / "olympus"

STATUS_AWAITING = "Awaiting Historical Data"


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
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)
    except Exception:
        return None


def _count_unknown(value: Any) -> tuple[int, int]:
    unknown = 0
    total = 0

    def walk(node: Any) -> None:
        nonlocal unknown, total
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
            return
        if isinstance(node, list):
            for v in node:
                walk(v)
            return
        total += 1
        s = str(node).strip().lower()
        if s in {
            "awaiting historical data",
            "pending",
            "unknown",
            "none",
            "",
        }:
            unknown += 1

    walk(value)
    return unknown, total


st.set_page_config(page_title="Knowledge Growth Dashboard", page_icon="KGD", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: radial-gradient(circle at 20% 20%, #18301f 0%, #141b2a 45%, #0e1320 100%); border:1px solid #375f48; border-radius:16px; padding:20px 26px; margin-bottom:18px;}
    .hero h1 {margin:0; color:#e6f6ea; font-size:2.0rem;}
    .hero p {margin:8px 0 0; color:#b8d9bf;}
    </style>
    <div class="hero">
      <h1>KGD Knowledge Growth Dashboard</h1>
      <p>Central institutional intelligence growth monitor for learning, decision quality, lifecycle evidence, capital integrity, and governance maturity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

recovered = recover_institutional_state(_ROOT)

status = recovered.get("status", {}) if isinstance(recovered, dict) else {}
idip = recovered.get("idip", {}) if isinstance(recovered, dict) else {}
idip_summary = recovered.get("idip_summary", {}) if isinstance(recovered, dict) else {}
feature_flags = (idip.get("feature_flags", {}) if isinstance(idip, dict) else {}) or {}
idip_engines = (idip.get("engines", {}) if isinstance(idip, dict) else {}) or {}

institutional_learning = recovered.get("institutional_learning", {})
hypotheses = recovered.get("hypotheses", {})
knowledge_growth = recovered.get("knowledge_growth", {})
learning_velocity = recovered.get("learning_velocity", {})
research_queue = recovered.get("research_queue", {})
concept_drift = recovered.get("concept_drift", {})
capital_intelligence = recovered.get("capital_intelligence", {})
knowledge_graph = recovered.get("knowledge_graph", {})
replay_payload = recovered.get("replay_payload", {})
meta_learning_payload = recovered.get("meta_learning_payload", {})
aro_payload = recovered.get("aro_payload", {})
coverage_payload = recovered.get("coverage_payload", {})
knowledge_evolution_payload = recovered.get("knowledge_evolution_payload", {})
explainability_payload = recovered.get("explainability_payload", {})
research_director_payload = recovered.get("research_director_payload", {})
idip_history = recovered.get("idip_history", [])
zeus_reports = recovered.get("zeus_reports", [])
zeus_status = recovered.get("zeus_status", {})
zvo_runtime = recovered.get("zvo_runtime", {})
institutional_dataset_architecture = recovered.get("institutional_dataset_architecture", {})
zeus_validation_boards = recovered.get("zeus_validation_boards", {})
institutional_knowledge_base = recovered.get("institutional_knowledge_base", {})
institutional_dataset_rows = recovered.get("institutional_dataset_rows", [])
recovery_audit = recovered.get("recovery_audit", {}) if isinstance(recovered, dict) else {}

recovered_sources = [
    k for k, v in recovery_audit.items()
    if str(k).endswith("_source") and str(v) in {"history", "idip_runtime"}
]
if recovered_sources:
    st.caption("Institutional recovery active: historical artifacts restored before rendering metrics.")

knowledge_total = int(knowledge_growth.get("total_knowledge_objects", 0) or 0)
knowledge_rate = knowledge_growth.get("growth_rate", STATUS_AWAITING)
knowledge_conf = institutional_learning.get("knowledge_confidence", STATUS_AWAITING)
knowledge_maturity = institutional_learning.get("knowledge_maturity", STATUS_AWAITING)
kg_size = (knowledge_graph.get("summary", {}) or {}).get("node_count", 0)
z_summary = (zeus_status.get("summary", {}) if isinstance(zeus_status, dict) else {}) or {}
z_queue = (zeus_status.get("queue", {}).get("counts", {}) if isinstance(zeus_status, dict) else {}) or {}
validated_research = z_summary.get("validated_research", len([r for r in zeus_reports if str(r.get("status", "")).lower() == "passed"]))
approved_count = int(z_queue.get("Approved", 0) or 0)
total_reports = int(z_summary.get("total", len(zeus_reports)) or 0)
meta_metrics = (meta_learning_payload.get("metrics", {}) or {})

# ── Executive KPI Strip ────────────────────────────────────────────────────
_kx1, _kx2, _kx3, _kx4, _kx5, _kx6 = st.columns(6)
_kx1.metric("Knowledge Objects",   knowledge_total, help="Total institutional knowledge objects tracked")
_kx2.metric("Knowledge Confidence",knowledge_conf,  help="0.0–1.0; 0.65+ = Validated; 0.80+ = Elite")
_kx3.metric("Learning Velocity",   learning_velocity.get("learning_velocity", STATUS_AWAITING), help="Knowledge growth events per cycle")
_kx4.metric("Validated Research",  validated_research, help="Research candidates approved by Zeus")
_kx5.metric("Research Backlog",    (aro_payload.get("research_lifecycle", {}) or {}).get("candidate", 0), help="Pending research candidates awaiting validation")
_kx6.metric("Dataset Quality",     (((institutional_dataset_architecture.get("dataset_quality", {}) if isinstance(institutional_dataset_architecture, dict) else {}) or {}).get("dataset_quality_score", STATUS_AWAITING)), help="Composite dataset quality score 0–100")

# ── Tab Navigation ─────────────────────────────────────────────────────────
_kg_t1, _kg_t2, _kg_t3, _kg_t4, _kg_t5 = st.tabs([
    "🏛 Governance & Quality", "📚 Knowledge & Learning", "⚡ Research & Validation", "💰 Capital & Lifecycle", "📊 History & Trends"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GOVERNANCE & QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with _kg_t1:
    st.caption("Dataset integrity, Zeus board governance, and knowledge promotion status. These are the primary trust indicators for institutional knowledge quality.")

    # ── Dataset Quality Framework ──────────────────────────────────────────
    st.markdown("### Dataset Quality Framework")
    st.caption("Dataset Quality Score = composite across completeness, coverage, label quality, evidence quality, validation coverage, historical depth, freshness, and statistical significance.")
    dataset_quality = (institutional_dataset_architecture.get("dataset_quality", {}) if isinstance(institutional_dataset_architecture, dict) else {}) or {}
    quality_metrics = (dataset_quality.get("metrics", {}) if isinstance(dataset_quality, dict) else {}) or {}

    _dq_score = dataset_quality.get("dataset_quality_score", STATUS_AWAITING)
    st.metric("**Dataset Quality Score**", _dq_score)

    dq1, dq2, dq3, dq4, dq5 = st.columns(5)
    dq1.metric("Completeness",    quality_metrics.get("completeness", STATUS_AWAITING),    help="Required fields populated across all observations")
    dq2.metric("Coverage",        quality_metrics.get("coverage", STATUS_AWAITING),        help="Session / regime / day-of-week distribution breadth")
    dq3.metric("Label Quality",   quality_metrics.get("label_quality", STATUS_AWAITING),   help="Observations with full exit classification + R-multiple")
    dq4.metric("Evidence Quality",quality_metrics.get("evidence_quality", STATUS_AWAITING),help="Average evidence / confidence score across observations")
    dq5.metric("Validation Coverage", quality_metrics.get("validation_coverage", STATUS_AWAITING), help="Proportion of observations linked to Zeus validation")

    dq6, dq7, dq8, dq9, dq10 = st.columns(5)
    dq6.metric("Unknown Classification Rate", quality_metrics.get("unknown_classification_rate", STATUS_AWAITING), help="Lower = better: unknown exits reduce label quality")
    dq7.metric("Duplicate Rate",  quality_metrics.get("duplicate_observation_rate", STATUS_AWAITING), help="Duplicate trade observation rate (0 = clean)")
    dq8.metric("Historical Depth",quality_metrics.get("historical_depth", STATUS_AWAITING),help="Depth relative to 300-observation institutional baseline")
    dq9.metric("Stat Significance",quality_metrics.get("statistical_significance", STATUS_AWAITING), help="Validation sample significance (1.0 = fully significant)")
    dq10.metric("Data Freshness", quality_metrics.get("data_freshness", STATUS_AWAITING),  help="1.0 = updated <24h; 0.2 = >14 days stale")

    st.markdown("---")

    # ── Zeus Board Separation ──────────────────────────────────────────────
    st.markdown("### Zeus Validation Boards")
    st.caption("Pattern and Strategy boards are independent governance workflows with separate ledgers. Do not aggregate them.")
    boards = (zeus_validation_boards.get("zeus_validation_boards", {}) if isinstance(zeus_validation_boards, dict) else {}) or {}
    pattern_board = boards.get("pattern_validation_board", {}) if isinstance(boards, dict) else {}
    strategy_board = boards.get("strategy_validation_board", {}) if isinstance(boards, dict) else {}

    st.markdown("#### 🔬 Pattern Validation Board — Hermes research")
    pb1, pb2, pb3, pb4, pb5 = st.columns(5)
    pb1.metric("Total Reports",   pattern_board.get("total_reports", 0))
    pb2.metric("Pass Rate",       pattern_board.get("pass_rate", STATUS_AWAITING))
    pb3.metric("Avg Confidence",  pattern_board.get("average_confidence", STATUS_AWAITING))
    pb4.metric("Evidence Quality",pattern_board.get("average_evidence_score", STATUS_AWAITING))
    pb5.metric("Approved",        pattern_board.get("approved_reports", 0))

    _pb_ledger = pattern_board.get("ledger", []) if isinstance(pattern_board, dict) else []
    if _pb_ledger:
        st.dataframe(pd.DataFrame(_pb_ledger).head(50), use_container_width=True, hide_index=True)

    st.markdown("#### ⚙️ Strategy Validation Board — Prometheus research")
    sb1, sb2, sb3, sb4, sb5 = st.columns(5)
    sb1.metric("Total Reports",   strategy_board.get("total_reports", 0))
    sb2.metric("Pass Rate",       strategy_board.get("pass_rate", STATUS_AWAITING))
    sb3.metric("Avg Confidence",  strategy_board.get("average_confidence", STATUS_AWAITING))
    sb4.metric("Evidence Quality",strategy_board.get("average_evidence_score", STATUS_AWAITING))
    sb5.metric("Approved",        strategy_board.get("approved_reports", 0))

    _sb_ledger = strategy_board.get("ledger", []) if isinstance(strategy_board, dict) else []
    if _sb_ledger:
        st.dataframe(pd.DataFrame(_sb_ledger).head(50), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Knowledge Promotion Governance ────────────────────────────────────
    st.markdown("### Knowledge Promotion Governance")
    st.caption("Only Zeus-approved knowledge is promoted to the Institutional Knowledge Base. Raw observations are never automatically promoted.")
    ikb = (institutional_knowledge_base.get("institutional_knowledge_base", {}) if isinstance(institutional_knowledge_base, dict) else {}) or {}
    raw_obs = (ikb.get("raw_observations", {}) if isinstance(ikb, dict) else {}) or {}
    validated = (ikb.get("validated_institutional_knowledge", {}) if isinstance(ikb, dict) else {}) or {}
    policy = (ikb.get("promotion_policy", {}) if isinstance(ikb, dict) else {}) or {}

    kp1, kp2, kp3, kp4, kp5 = st.columns(5)
    kp1.metric("Raw Observations", raw_obs.get("count", len(institutional_dataset_rows)))
    kp2.metric("Validated Knowledge", validated.get("count", 0))
    kp3.metric("Promotion Rate", validated.get("promotion_rate", STATUS_AWAITING))
    kp4.metric("Auto Promotion", "No" if not policy.get("raw_observations_promoted_automatically", False) else "Yes")
    kp5.metric("Zeus Gate Required", "Yes" if policy.get("requires_zeus_approval", True) else "No")

    st.markdown("---")

    # ── Zeus Validation Operations Summary ────────────────────────────────
    st.markdown("### Zeus Validation Operations")
    zs1, zs2, zs3, zs4, zs5, zs6 = st.columns(6)
    zs1.metric("Queue Total",    z_summary.get("total", len(zeus_reports)))
    zs2.metric("Queued",         z_queue.get("Queued", 0))
    zs3.metric("Running",        z_queue.get("Running", 0))
    zs4.metric("Paused",         z_queue.get("Paused", 0))
    zs5.metric("Completed",      z_queue.get("Completed", 0))
    zs6.metric("Approved",       z_queue.get("Approved", 0))

    zs7, zs8, zs9, zs10 = st.columns(4)
    zs7.metric("Validation Success Rate",   z_summary.get("validation_success_rate", STATUS_AWAITING))
    zs8.metric("Avg Queue Age (hrs)",        z_summary.get("average_queue_age_hours", STATUS_AWAITING))
    zs9.metric("Approval Velocity (24h)",    z_summary.get("approval_velocity_24h", STATUS_AWAITING))
    zs10.metric("Scheduler Transitions",     (zvo_runtime.get("scheduler", {}) or {}).get("transitions_executed", STATUS_AWAITING))

    recent_transitions = zvo_runtime.get("recent_transitions", []) if isinstance(zvo_runtime, dict) else []
    if recent_transitions:
        st.dataframe(pd.DataFrame(recent_transitions).head(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE & LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with _kg_t2:
    st.caption("Institutional knowledge growth, learning velocity, pattern discoveries, and knowledge coverage across dimensions.")

    st.markdown("### Knowledge Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Knowledge Objects", knowledge_total)
    k2.metric("Knowledge Growth Rate",   knowledge_rate)
    k3.metric("Knowledge Density",       (knowledge_graph.get("summary", {}) or {}).get("edge_count", 0))
    k4.metric("Knowledge Confidence",    knowledge_conf)
    k5.metric("Knowledge Maturity",      knowledge_maturity)

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric("Knowledge Stability",     max(0.0, 1.0 - float(_safe_float(concept_drift.get("concept_drift_index")) or 0.0)))
    k7.metric("Knowledge Freshness",     knowledge_growth.get("generated_at", STATUS_AWAITING))
    k8.metric("Knowledge Coverage",      institutional_learning.get("sample_size", 0))
    k9.metric("Institutional Memory",    len((idip.get("engines", {}) or {}).get("institutional_knowledge_intelligence", {}).get("institutional_lessons", [])))
    k10.metric("Knowledge Graph Size",   kg_size)

    st.markdown("### Learning Metrics")
    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Learning Velocity",   learning_velocity.get("learning_velocity", STATUS_AWAITING))
    l2.metric("Learning Events",     learning_velocity.get("learning_events", 0))
    l3.metric("New Concepts Learned",len((hypotheses.get("rows", []) or [])))
    l4.metric("Pattern Discoveries", len((idip.get("engines", {}) or {}).get("pattern_lifecycle_intelligence", {}).get("pattern_lifecycle_profiles", [])))
    l5.metric("Lifecycle Discoveries",len((idip.get("engines", {}) or {}).get("trade_lifecycle_intelligence", {}).get("modules", {}).get("replay_engine", {}).get("replay_cases", [])))

    l6, l7, l8, l9, l10 = st.columns(5)
    l6.metric("Decision Discoveries",len((idip.get("engines", {}) or {}).get("decision_attribution_intelligence", {}).get("decision_attribution_rows", [])))
    l7.metric("Regime Discoveries",  len((idip.get("engines", {}) or {}).get("portfolio_intelligence", {}).get("regime_exposure", [])))
    l8.metric("Research Generated",  len((research_queue.get("rows", []) or [])))
    l9.metric("Research Validated",  validated_research)
    l10.metric("Zeus Approval Rate", round(approved_count / max(1, total_reports), 4) if total_reports else STATUS_AWAITING)

    st.markdown("### Knowledge Coverage and Evolution")
    cov_rows = coverage_payload.get("coverage_rows", []) if isinstance(coverage_payload, dict) else []
    if cov_rows:
        cov_df = pd.DataFrame(cov_rows)
        st.dataframe(cov_df.head(200), use_container_width=True, hide_index=True)
        if "coverage_pct" in cov_df.columns and "dimension" in cov_df.columns:
            st.plotly_chart(px.bar(cov_df, x="dimension", y="coverage_pct", title="Knowledge Coverage % by Dimension"), use_container_width=True)
    else:
        st.caption("Knowledge coverage data awaiting historical data.")

    ev_summary = (knowledge_evolution_payload.get("summary", {}) if isinstance(knowledge_evolution_payload, dict) else {}) or {}
    ke1, ke2, ke3, ke4 = st.columns(4)
    ke1.metric("Knowledge Objects Tracked", ev_summary.get("objects_tracked", 0))
    ke2.metric("Revalidation Candidates",   (ev_summary.get("action_counts", {}) or {}).get("revalidate", 0))
    ke3.metric("Refinement Candidates",     (ev_summary.get("action_counts", {}) or {}).get("refine", 0))
    ke4.metric("Version Upgrade Candidates",(ev_summary.get("action_counts", {}) or {}).get("version_upgrade", 0))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESEARCH & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with _kg_t3:
    st.caption("Autonomous research pipeline: backlog, priorities, ROI, completion, and retirement. Research queue shows pending hypotheses.")

    st.markdown("### Olympus Command Center — Autonomous Research")
    ar1, ar2, ar3, ar4, ar5 = st.columns(5)
    ar1.metric("Research Backlog",    (aro_payload.get("research_lifecycle", {}) or {}).get("candidate", 0))
    ar2.metric("Research ROI (Est)",  (aro_payload.get("research_roi", {}) or {}).get("estimated_value_sum", STATUS_AWAITING))
    ar3.metric("Research Priorities", len((aro_payload.get("prioritized_roadmap", []) or [])))
    ar4.metric("Research Completion", len([r for r in zeus_reports if str(r.get("status", "")).lower() == "passed"]))
    ar5.metric("Research Retirement", len((research_director_payload.get("archived_research", []) or [])))

    ar6, ar7, ar8, ar9, ar10 = st.columns(5)
    ar6.metric("Learning Health",  meta_metrics.get("learning_efficiency", STATUS_AWAITING))
    ar7.metric("Knowledge Drift",  meta_metrics.get("concept_drift", STATUS_AWAITING))
    ar8.metric("Model Drift",      meta_metrics.get("model_drift", STATUS_AWAITING))
    ar9.metric("Learning Drift",   meta_metrics.get("learning_drift", STATUS_AWAITING))
    _unknown_count_local, _total_count_local = _count_unknown({
        "idip": idip, "institutional_learning": institutional_learning,
        "knowledge_growth": knowledge_growth, "learning_velocity": learning_velocity,
        "concept_drift": concept_drift, "capital_intelligence": capital_intelligence,
    })
    ar10.metric("Subsystem Health", round(1.0 - (_unknown_count_local / max(1, _total_count_local)), 4))

    if (research_queue.get("rows", []) or []):
        st.markdown("### Research Queue")
        st.dataframe(pd.DataFrame(research_queue.get("rows", [])).head(200), use_container_width=True, hide_index=True)

    st.markdown("### Explainability and Governance")
    ex_summary = (explainability_payload.get("summary", {}) if isinstance(explainability_payload, dict) else {}) or {}
    gx1, gx2, gx3, gx4 = st.columns(4)
    gx1.metric("Explained Recommendations", ex_summary.get("explained_recommendations", 0))
    gx2.metric("Reproducibility Enforced",  ex_summary.get("reproducibility_enforced", STATUS_AWAITING))
    gx3.metric("Zeus Submission Queue",     len((aro_payload.get("zeus_submission_queue", []) if isinstance(aro_payload, dict) else [])))
    gx4.metric("Governance Mandatory",      bool((idip.get("meta", {}) or {}).get("zeus_validation_required", True)))

    if (explainability_payload.get("explanations", []) if isinstance(explainability_payload, dict) else []):
        st.dataframe(pd.DataFrame(explainability_payload.get("explanations", [])).head(100), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CAPITAL & LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════
with _kg_t4:
    st.caption("Capital integrity, trade lifecycle coverage, decision quality, and institutional risk metrics.")

    st.markdown("### Capital Intelligence Metrics")
    st.caption("Strategy equity excludes deposits/withdrawals; trading growth = P&L from execution only.")
    cap_summary = (capital_intelligence.get("summary", {}) or {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy Equity",      cap_summary.get("strategy_equity", STATUS_AWAITING))
    c2.metric("Raw Equity",           cap_summary.get("raw_equity", STATUS_AWAITING))
    c3.metric("Organic Growth",       cap_summary.get("organic_growth", STATUS_AWAITING))
    c4.metric("Capital Injections",   cap_summary.get("capital_injections", STATUS_AWAITING))
    c5.metric("Capital Withdrawals",  cap_summary.get("capital_withdrawals", STATUS_AWAITING))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Trading Growth",       cap_summary.get("trading_growth", STATUS_AWAITING))
    c7.metric("Capital Efficiency",   cap_summary.get("capital_efficiency", STATUS_AWAITING))
    c8.metric("Drawdown Attribution", (cap_summary.get("drawdown_attribution", {}) or {}).get("trading_only", STATUS_AWAITING))
    c9.metric("Recovery Attribution", (cap_summary.get("recovery_attribution", {}) or {}).get("trading_only", STATUS_AWAITING))

    st.markdown("### Trade Lifecycle Metrics")
    st.caption("Unknown exit % should decrease over time as exit classification matures. Reward capture = % of theoretical move captured.")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Lifecycle Coverage",        idip_summary.get("sample_size", 0))
    t2.metric("Exit Classification Coverage", max(0, (idip_summary.get("sample_size", 0) or 0) - (idip_summary.get("unknown_exit_count", 0) or 0)))
    t3.metric("Unknown Exit %",            round((idip_summary.get("unknown_exit_count", 0) or 0) / max(1, (idip_summary.get("sample_size", 0) or 1)), 4))
    t4.metric("Trade Duration Coverage",   len((idip.get("engines", {}) or {}).get("duration_intelligence", {}).get("duration_by_pattern", [])))
    t5.metric("MFE Coverage",              len([x for x in status.get("closed_trades", []) if x.get("mfe_pct") is not None]))

    t6, t7, t8, t9 = st.columns(4)
    t6.metric("MAE Coverage",             len([x for x in status.get("closed_trades", []) if x.get("mae_pct") is not None]))
    t7.metric("Reward Capture",           idip_summary.get("reward_efficiency", STATUS_AWAITING))
    t8.metric("Opportunity Capture",      (idip.get("engines", {}) or {}).get("reward_capture_intelligence", {}).get("capture_ratio", STATUS_AWAITING))
    t9.metric("Lifecycle Efficiency",     (idip.get("engines", {}) or {}).get("trade_lifecycle_intelligence", {}).get("summary", {}).get("lifecycle_efficiency", STATUS_AWAITING))

    st.markdown("### Decision Intelligence Metrics")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Decision Coverage",         len((idip.get("engines", {}) or {}).get("decision_attribution_intelligence", {}).get("decision_attribution_rows", [])))
    d2.metric("Decision Attribution Sample",(idip_summary.get("sample_size") or 0))
    d3.metric("Counterfactual Replays",    (replay_payload.get("summary", {}) or {}).get("counterfactual_replays", 0))
    d4.metric("Replay Accuracy",           (replay_payload.get("summary", {}) or {}).get("replay_accuracy", STATUS_AWAITING))
    d5.metric("Decision Quality",          idip_summary.get("decision_quality", STATUS_AWAITING))

    d6, d7 = st.columns(2)
    d6.metric("Decision Confidence", knowledge_conf)
    d7.metric("Decision Drift / Behaviour Drift", f"{concept_drift.get('concept_drift_index', 0)} / {concept_drift.get('behaviour_drift_index', 0)}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY & TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with _kg_t5:
    st.caption("7-day institutional progress trends, data integrity scoring, feature flags, and version status.")

    st.markdown("### Data Quality Metrics")
    unknown_count, total_count = _count_unknown(
        {
            "idip": idip,
            "institutional_learning": institutional_learning,
            "knowledge_growth": knowledge_growth,
            "learning_velocity": learning_velocity,
            "concept_drift": concept_drift,
            "capital_intelligence": capital_intelligence,
            "knowledge_graph": knowledge_graph,
            "replay": replay_payload,
        }
    )

    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("Awaiting Metrics",         unknown_count)
    q2.metric("Unknown Fields",           unknown_count)
    q3.metric("Pending Intelligence",     len([x for x in (research_queue.get("rows", []) or []) if str(x.get("status", "")).lower() == "queued"]))
    q4.metric("Missing Evidence",         len([r for r in zeus_reports if not r.get("evidence")]))
    q5.metric("Missing Labels",           len([x for x in status.get("closed_trades", []) if not x.get("exit_reason")]))
    q6.metric("Classification Accuracy",  round(1.0 - ((idip_summary.get("unknown_exit_count", 0) or 0) / max(1, (idip_summary.get("sample_size", 0) or 1))), 4))

    st.metric("Data Integrity Score", round((1.0 - (unknown_count / max(1, total_count))) * 100.0, 2))

    st.markdown("### Institutional Progress Trends")
    progress_rows = []
    for row in idip_history[-200:]:
        if not isinstance(row, dict):
            continue
        summary = row.get("summary", {}) or {}
        progress_rows.append(
            {
                "timestamp": (row.get("meta", {}) or {}).get("generated_at"),
                "knowledge_growth": knowledge_growth.get("growth_rate", STATUS_AWAITING),
                "mastery": summary.get("maturity", STATUS_AWAITING),
                "intelligence_maturity": summary.get("maturity", STATUS_AWAITING),
                "research_completion": len(zeus_reports),
                "pattern_coverage": len((row.get("engines", {}) or {}).get("pattern_lifecycle_intelligence", {}).get("pattern_lifecycle_profiles", [])),
                "decision_coverage": summary.get("sample_size", 0),
                "lifecycle_coverage": summary.get("sample_size", 0),
                "portfolio_intelligence": (row.get("engines", {}) or {}).get("portfolio_intelligence", {}).get("portfolio_expectancy", STATUS_AWAITING),
                "return_intelligence": summary.get("expectancy", STATUS_AWAITING),
                "risk_intelligence": (row.get("engines", {}) or {}).get("institutional_risk_intelligence", {}).get("risk_budgeting_score", STATUS_AWAITING),
            }
        )

    progress_df = pd.DataFrame(progress_rows)
    if not progress_df.empty:
        numeric_cols = ["decision_coverage", "lifecycle_coverage", "pattern_coverage", "return_intelligence", "risk_intelligence", "portfolio_intelligence"]
        for col in numeric_cols:
            progress_df[col + "_n"] = progress_df[col].apply(_safe_float)

        _trend_pairs = [
            ("decision_coverage_n", "Decision Coverage Trend"),
            ("lifecycle_coverage_n", "Lifecycle Coverage Trend"),
            ("pattern_coverage_n", "Pattern Coverage Trend"),
        ]
        for col, title in _trend_pairs:
            chart_df = progress_df[progress_df[col].notnull() & progress_df["timestamp"].notnull()][["timestamp", col]]
            if not chart_df.empty:
                st.plotly_chart(px.line(chart_df, x="timestamp", y=col, title=title), use_container_width=True)

        st.dataframe(progress_df.tail(200), use_container_width=True, hide_index=True)
    else:
        st.caption("Progress history is awaiting sufficient data.")

    st.markdown("### Feature Flag and Version Status")
    ff1, ff2, ff3, ff4 = st.columns(4)
    ff1.metric("Feature Flags Enabled", len([k for k, v in feature_flags.items() if bool(v)]))
    ff2.metric("Feature Flags Total",   len(feature_flags))
    ff3.metric("IDIP Version",          (idip.get("meta", {}) or {}).get("version", STATUS_AWAITING))
    ff4.metric("KGD Version",           "kgd-v2.0")

    if feature_flags:
        ff_df = pd.DataFrame([{"flag": k, "enabled": bool(v)} for k, v in feature_flags.items()])
        st.dataframe(ff_df, use_container_width=True, hide_index=True)
st.markdown("### Zeus Validation Operations")
st.caption("KGD is observational and governance-first. No execution-path mutation is performed by this dashboard.")
