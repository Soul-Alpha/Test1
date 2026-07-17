"""Hermes Academy Dashboard.

Run:
    streamlit run ui/hermes_academy_dashboard.py --server.port=8504
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from olympus.core.hermes_analytics import build_hermes_analytics

_STATUS_F = _ROOT / "live_bot" / "hermes_status.json"

st.set_page_config(page_title="Hermes Academy", page_icon="🎓", layout="wide")
st.title("Hermes Academy")
st.caption("Institutional learning intelligence layer for Hermes (analytics-only).")

status = {}
if _STATUS_F.exists():
    status = json.loads(_STATUS_F.read_text(encoding="utf-8"))

analytics = build_hermes_analytics(_ROOT)
academy = status.get("academy") or analytics.get("academy", {})
edge = status.get("edge_stability") or analytics.get("edge_stability", {})
perf_diag = status.get("performance_diagnostics") or analytics.get("performance_diagnostics", {})
pattern_genome = status.get("pattern_genome") or analytics.get("pattern_genome", [])
metric_kc = status.get("metric_knowledge_confidence") or analytics.get("metric_knowledge_confidence", {})
evolution = status.get("evolution_roadmap") or analytics.get("evolution_roadmap", {})
validation_gate = status.get("validation_gate") or analytics.get("validation_gate", {})
academy_gate = status.get("academy_certification_gate") or analytics.get("academy_certification_gate", {})
phase_report = status.get("phase_completion_report") or analytics.get("phase_completion_report", {})
research_engine = status.get("research_engine") or analytics.get("research_engine", {})

learning_journey = academy.get("learning_journey", {})
learning_sources = learning_journey.get("learning_sources", {})

st.markdown("### Learning Journey")
j1, j2, j3, j4 = st.columns(4)
j1.metric("Current Stage", learning_journey.get("current_stage", "Observer"))
j2.metric("Graduation Progress %", learning_journey.get("graduation_progress_pct", 0.0))
j3.metric("Primary Source", learning_journey.get("current_primary_learning_source", "Simulation"))
j4.metric("Primary Teacher", learning_journey.get("current_primary_teacher", "Simulation teaches theory."))
st.info(f"Current Objective: {learning_journey.get('current_learning_objective', 'Expand historical evidence quality.')}")
if learning_journey.get("bottleneck_dimension"):
    st.caption(
        f"Bottleneck dimension: {learning_journey.get('bottleneck_dimension')} | Additional validated samples required: {learning_journey.get('additional_validated_samples_required', 0)}"
    )

competency_dims = learning_journey.get("competency_dimensions", {})
if competency_dims:
    st.markdown("#### Competency Dimensions")
    cdf = pd.DataFrame([{"dimension": k, "score": v} for k, v in competency_dims.items()])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

st.markdown("### Learning Sources")
for key, label in [("simulation", "Simulation"), ("paper_trading", "Paper Trading"), ("live_validation", "Live Validation")]:
    src = learning_sources.get(key, {})
    st.write(
        f"{label}: {src.get('bar', '')}  Mastery {src.get('mastery', 0.0)}% | "
        f"Evidence {src.get('evidence_collected', 0.0)}% | Validation {src.get('validation_progress', 0.0)}% | "
        f"Infrastructure {src.get('infrastructure_readiness', 0.0)}% | Status {src.get('status', 'Developing')}"
    )
    purpose = src.get("purpose", [])
    if purpose:
        st.caption("Purpose: " + ", ".join(purpose))

academies = academy.get("academies", [])
st.markdown("### Academy Schools")
if academies:
    st.dataframe(pd.DataFrame(academies), use_container_width=True, hide_index=True)
else:
    st.caption("Academy schools are awaiting initialization.")

stage_thresholds = academy.get("stage_thresholds", [])
if stage_thresholds:
    st.markdown("#### Stage Thresholds")
    st.dataframe(pd.DataFrame(stage_thresholds), use_container_width=True, hide_index=True)

competency_model = academy.get("competency_model", {})
if competency_model:
    st.markdown("#### Competency Model Weights")
    weights = competency_model.get("weights", {})
    if weights:
        wdf = pd.DataFrame([{"dimension": k, "weight": v} for k, v in weights.items()])
        st.dataframe(wdf, use_container_width=True, hide_index=True)

report = academy.get("report_card", {})
st.markdown("### Hermes Report Card")
if report.get("current"):
    st.json(report.get("current"))

hist = report.get("history", [])
if hist:
    hdf = pd.DataFrame(hist)
    st.plotly_chart(px.line(hdf, x="date", y="mastery_pct", title="Mastery Progress History"), use_container_width=True)

passport = academy.get("knowledge_passport", {})
st.markdown("### Knowledge Passport")
if passport:
    rows = []
    for k, v in passport.items():
        if isinstance(v, dict):
            rows.append({"achievement": k, **v})
        else:
            rows.append({"achievement": k, "status": "Validated" if bool(v) else "Developing", "evidence_count": None})
    pdf = pd.DataFrame(rows)
    st.dataframe(pdf, use_container_width=True, hide_index=True)

graduation = academy.get("graduation", {})
st.markdown("### Graduation System")
g1, g2, g3 = st.columns(3)
g1.metric("Graduation %", graduation.get("graduation_percentage", 0.0))
g2.metric("Milestones", len(graduation.get("milestones", [])))
g3.metric("Weighted Progress", graduation.get("weighted_model", {}).get("validated_weighted_progress", 0.0))
if graduation.get("milestones"):
    st.dataframe(pd.DataFrame(graduation.get("milestones", [])), use_container_width=True, hide_index=True)

st.markdown("### Evolution Governance")
v1, v2, v3 = st.columns(3)
v1.metric("Current Phase", evolution.get("current_phase", "Phase II"))
v2.metric("Academy Decision", academy_gate.get("certification_decision", "Requires Additional Evidence"))
v3.metric("Adaptive Unlock", "Approved" if academy_gate.get("adaptive_unlock_approved", False) else "LOCKED")

if validation_gate.get("checks"):
    st.markdown("#### Validation Gate")
    st.dataframe(pd.DataFrame(validation_gate.get("checks", [])), use_container_width=True, hide_index=True)

if research_engine:
    st.markdown("#### Research Foundation")
    st.caption(f"Status: {research_engine.get('status', 'Foundation')}")
    for q in research_engine.get("research_questions", [])[:8]:
        st.caption(f"- {q}")

if phase_report:
    st.markdown("#### Phase Completion Summary")
    st.json(
        {
            "phase": phase_report.get("phase"),
            "recommended_next_phase": phase_report.get("recommended_next_phase"),
            "academy_certification_decision": phase_report.get("academy_certification_decision"),
            "governance_assertions": phase_report.get("governance_assertions", {}),
        }
    )

st.markdown("### Edge Stability")
e1, e2, e3, e4, e5, e6, e7, e8 = st.columns(8)
e1.metric("Prediction Edge", edge.get("prediction_edge", "Awaiting Historical Data"))
e2.metric("Execution Edge", edge.get("execution_edge", "Awaiting Historical Data"))
e3.metric("Risk Mgmt Edge", edge.get("risk_management_edge", "Awaiting Historical Data"))
e4.metric("Pattern Edge", edge.get("pattern_intelligence_edge", "Awaiting Historical Data"))
e5.metric("Return Edge", edge.get("return_edge", "Awaiting Historical Data"))
e6.metric("Knowledge Confidence", edge.get("knowledge_confidence", "Awaiting Historical Data"))
e7.metric("Confidence Edge", edge.get("confidence_edge", "Awaiting Historical Data"))
e8.metric("Adaptive Readiness", edge.get("adaptive_readiness", "Awaiting Historical Data"))

if edge.get("history"):
    st.markdown("#### Edge History")
    st.dataframe(pd.DataFrame(edge.get("history", [])).tail(20), use_container_width=True, hide_index=True)

st.markdown("### Performance Diagnostics")
for key in ["win_distribution", "loss_distribution", "return_distribution", "trade_efficiency"]:
    val = perf_diag.get(key)
    if val is not None:
        st.write(f"{key.replace('_', ' ').title()}")
        st.json(val)

for key, title in [
    ("expectancy_trend", "Expectancy Trend"),
    ("payoff_trend", "Payoff Trend"),
    ("session_contribution", "Session Contribution"),
    ("execution_contribution", "Execution Contribution"),
]:
    rows = perf_diag.get(key, [])
    if rows:
        st.markdown(f"#### {title}")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

if perf_diag.get("explanations"):
    st.markdown("#### Why Current Metrics Look This Way")
    for x in perf_diag.get("explanations", []):
        st.write(f"- {x}")

st.markdown("### Pattern Genome")
if pattern_genome:
    st.dataframe(pd.DataFrame(pattern_genome).head(200), use_container_width=True, hide_index=True)
else:
    st.caption("Pattern genome is awaiting sufficient historical data.")

st.markdown("### Metric Knowledge Confidence")
if metric_kc:
    mdf = pd.DataFrame([{"metric": k, **v} for k, v in metric_kc.items()])
    st.dataframe(mdf, use_container_width=True, hide_index=True)

st.caption("Hermes Academy is evidence-driven and does not alter live trading behavior.")
