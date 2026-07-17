"""Prometheus Evolution Dashboard.

Run:
    streamlit run ui/prometheus_evolution_dashboard.py --server.port=8510
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from olympus.core.prometheus_evolution_intelligence import (  # noqa: E402
    build_prometheus_evolution_intelligence,
    write_prometheus_evolution_artifacts,
)


def _arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        s = safe[col]
        if s.dtype != object:
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        as_num = pd.to_numeric(non_null, errors="coerce")
        if as_num.notna().all():
            safe[col] = pd.to_numeric(s, errors="coerce")
        else:
            safe[col] = s.astype(str)
    return safe


def _metric(columns: list[Any], idx: int, name: str, value: Any) -> None:
    columns[idx].metric(name, value if value is not None else "Awaiting Historical Data")


st.set_page_config(page_title="Prometheus Evolution", page_icon="EVO", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#132033 0%,#1d2513 100%); border:1px solid #3e5f39; border-radius:14px; padding:22px 28px; margin-bottom:20px;}
    .hero h1 {margin:0; color:#e9f3ff; font-size:2.1rem;}
    .hero p {margin:6px 0 0; color:#caddc8;}
    </style>
    <div class="hero">
      <h1>Prometheus Evolution Layer v1</h1>
      <p>Additive-only institutional self-learning intelligence. Existing execution behavior remains unchanged.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

intel = build_prometheus_evolution_intelligence(_ROOT)
paths = write_prometheus_evolution_artifacts(_ROOT, intel)

meta = intel.get("meta", {})
improve = intel.get("continuous_improvement_metrics", {})
exec_lrn = intel.get("execution_learning_engine", {})
edge_lrn = intel.get("execution_edge_learning_engine", {})
risk = intel.get("risk_intelligence_engine", {})
rr = intel.get("risk_to_reward_intelligence_engine", {})
growth = intel.get("capital_growth_intelligence_engine", {})
session = intel.get("session_intelligence_evolution", {})
pattern = intel.get("pattern_evolution_engine", {})
conf = intel.get("confidence_intelligence_evolution", {})
market = intel.get("market_intelligence_engine", {})
research = intel.get("research_library_evolution", {})
recs = intel.get("recommendation_engine_evolution", {})
academy = intel.get("execution_academy_integration", {})

st.markdown("### Evolution Summary")
c = st.columns(6)
_metric(c, 0, "Source", meta.get("source_system"))
_metric(c, 1, "Dataset", meta.get("dataset_generation"))
_metric(c, 2, "Feature Version", meta.get("feature_version"))
_metric(c, 3, "Strategy Version", meta.get("strategy_version"))
_metric(c, 4, "Observational", "Yes" if meta.get("observational_only") else "No")
_metric(c, 5, "Execution Unchanged", "Yes" if meta.get("execution_behavior_unchanged") else "No")

st.markdown("### Continuous Improvement Metrics")
m = st.columns(5)
_metric(m, 0, "Learning Velocity", improve.get("learning_velocity"))
_metric(m, 1, "Knowledge Growth Rate", improve.get("knowledge_growth_rate"))
_metric(m, 2, "Edge Improvement Rate", improve.get("edge_improvement_rate"))
_metric(m, 3, "Capital Growth Efficiency", improve.get("capital_growth_efficiency"))
_metric(m, 4, "Institutional Knowledge Score", improve.get("institutional_knowledge_score"))

st.markdown("### Execution / Edge / Risk")
e1, e2, e3 = st.columns(3)
with e1:
    st.json(exec_lrn)
with e2:
    st.json(edge_lrn)
with e3:
    st.json(risk)

st.markdown("### Risk-to-Reward Intelligence")
rr_rows = pd.DataFrame(rr.get("context_aware_rr_recommendations", []))
if not rr_rows.empty:
    st.dataframe(_arrow_safe_df(rr_rows.head(200)), use_container_width=True, hide_index=True)
else:
    st.caption("Awaiting context-aware RR evidence.")

st.markdown("### Capital Growth Intelligence")
growth_runs = pd.DataFrame(growth.get("simulations", []))
if not growth_runs.empty:
    st.dataframe(_arrow_safe_df(growth_runs), use_container_width=True, hide_index=True)
else:
    st.caption("Awaiting growth simulation evidence.")

best = growth.get("best_capital_path", {})
if best:
    st.markdown("Best Capital Path")
    st.json(best)

st.markdown("### Session / Pattern / Confidence / Market Evolution")
t1, t2 = st.columns(2)
with t1:
    session_df = pd.DataFrame(session.get("session_matrix", []))
    if not session_df.empty:
        st.dataframe(_arrow_safe_df(session_df.head(200)), use_container_width=True, hide_index=True)
    pattern_df = pd.DataFrame(pattern.get("patterns", []))
    if not pattern_df.empty:
        st.dataframe(_arrow_safe_df(pattern_df.head(200)), use_container_width=True, hide_index=True)
with t2:
    st.json(conf)
    market_df = pd.DataFrame(market.get("regime_matrix", []))
    if not market_df.empty:
        st.dataframe(_arrow_safe_df(market_df), use_container_width=True, hide_index=True)

st.markdown("### Research Library Evolution")
res_left, res_right = st.columns(2)
with res_left:
    vf = pd.DataFrame(research.get("validated_findings", []))
    if not vf.empty:
        st.dataframe(_arrow_safe_df(vf), use_container_width=True, hide_index=True)
with res_right:
    ah = pd.DataFrame(research.get("active_hypotheses", []))
    if not ah.empty:
        st.dataframe(_arrow_safe_df(ah), use_container_width=True, hide_index=True)

st.markdown("### Evidence-Based Recommendations")
rec_df = pd.DataFrame(recs.get("recommendations", []))
if not rec_df.empty:
    st.dataframe(_arrow_safe_df(rec_df), use_container_width=True, hide_index=True)

st.markdown("### Execution Academy Integration (Independent)")
st.json(academy)

st.markdown("### Governance and Compatibility")
st.json(
    {
        "governance": intel.get("governance", {}),
        "compatibility": intel.get("compatibility", {}),
        "artifact_paths": paths,
        "shared_knowledge_contracts": len(intel.get("shared_knowledge_contracts", [])),
    }
)

st.caption("Evolution layer is additive-only and observational. No automatic execution behavior changes are performed.")
