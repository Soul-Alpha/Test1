"""Prometheus Execution Academy Dashboard.

Run:
    streamlit run ui/prometheus_execution_academy_dashboard.py --server.port=8508
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from olympus.core.prometheus_decision_intelligence import (
    build_prometheus_decision_intelligence,
    write_prometheus_intelligence_artifacts,
)
from ui.dashboard_registry_support import render_registry_metrics, render_registry_tables


def _load_json_silent(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            import json
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)
    except Exception:
        return None


st.set_page_config(page_title="Prometheus Execution Academy", page_icon="PEA", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#0f1f2d 0%,#231611 100%); border:1px solid #2c546d; border-radius:14px; padding:22px 28px; margin-bottom:20px;}
    .hero h1 {margin:0; color:#e9f4ff; font-size:2.1rem;}
    .hero p {margin:6px 0 0; color:#c9dae9;}
    </style>
    <div class="hero">
      <h1>PEA Prometheus Execution Academy</h1>
      <p>Independent institutional execution evaluation, decision intelligence, and evidence-first governance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

intel = build_prometheus_decision_intelligence(_ROOT)
paths = write_prometheus_intelligence_artifacts(_ROOT, intel)
evo = _load_json_silent(_ROOT / "storage" / "olympus" / "prometheus_evolution_intelligence.json")
zeus_validation = _load_json_silent(_ROOT / "storage" / "olympus" / "zeus_validation_status.json")
institutional_runtime = _load_json_silent(_ROOT / "storage" / "olympus" / "institutional_risk_performance_runtime.json")

academy = intel.get("execution_academy", {})
risk = intel.get("risk_intelligence", {})
mh = intel.get("market_health_engine", {})
conf = intel.get("confidence_intelligence", {})
ret = intel.get("return_intelligence", {})
edge = intel.get("edge_stability", {})
evo_meta = evo.get("meta", {}) if isinstance(evo, dict) else {}
evo_exec_loc = evo.get("execution_location_intelligence", {}) if isinstance(evo, dict) else {}
evo_cap_pres = evo.get("capital_preservation_intelligence", {}) if isinstance(evo, dict) else {}
evo_capital = evo.get("capital_growth_intelligence_engine", {}) if isinstance(evo, dict) else {}
evo_capital_best = (evo_capital.get("best_capital_path", {}) if isinstance(evo_capital, dict) else {}) or {}
evo_learning = evo.get("continuous_improvement_metrics", {}) if isinstance(evo, dict) else {}
evo_edge = evo.get("execution_edge_learning_engine", {}) if isinstance(evo, dict) else {}
evo_recommend = evo.get("recommendation_engine_evolution", {}) if isinstance(evo, dict) else {}
evo_decision = evo.get("decision_attribution_intelligence", {}) if isinstance(evo, dict) else {}

st.markdown("### Academy Stage and Grade")
a1, a2, a3, a4, a5, a6 = st.columns(6)
a1.metric("Current Stage", academy.get("current_stage", "Awaiting Historical Data"))
a2.metric("Graduation Progress", academy.get("graduation_progress", "Awaiting Historical Data"))
a3.metric("Execution GPA", academy.get("execution_gpa", "Awaiting Historical Data"))
a4.metric("Institutional Grade", academy.get("institutional_grade", "Awaiting Historical Data"))
a5.metric("Evidence Level", academy.get("evidence_level", "Awaiting Historical Data"))
a6.metric("Knowledge Confidence", academy.get("knowledge_confidence", "Awaiting Historical Data"), delta=f"Version {evo_meta.get('evolution_layer_version', 'v2.2')}", delta_color="off")

st.markdown("### Market Health and Risk Momentum")
b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Market Health Score", mh.get("market_health_score", "Awaiting Historical Data"))
b2.metric("Market Class", mh.get("classification", "Awaiting Historical Data"))
b3.metric("Risk Momentum", risk.get("risk_momentum", "Awaiting Historical Data"))
b4.metric("Decision Stability", risk.get("decision_stability", "Awaiting Historical Data"))
b5.metric("Signal Quality Index", risk.get("signal_quality_index", "Awaiting Historical Data"))

st.markdown("### Execution Report Card")
report_card = pd.DataFrame([academy.get("report_card", {})])
if not report_card.empty:
    st.dataframe(report_card, use_container_width=True, hide_index=True)

st.markdown("### Execution Intelligence")
_registry_sources = {"prometheus_evolution": evo, "prometheus_decision": intel}
_registry_sources["zeus_validation"] = zeus_validation
_registry_sources["institutional_risk_performance_runtime"] = institutional_runtime
render_registry_metrics(dashboard="prometheus_execution_academy", section="execution_intelligence", sources=_registry_sources, columns_count=5)

st.markdown("### Capital Intelligence")
render_registry_metrics(dashboard="prometheus_execution_academy", section="capital_intelligence", sources=_registry_sources, columns_count=5)

st.markdown("### Learning Intelligence")
render_registry_metrics(dashboard="prometheus_execution_academy", section="learning_intelligence", sources=_registry_sources, columns_count=5)

st.markdown("### Recommendation Intelligence")
rec_rows = evo_recommend.get("recommendations", []) if isinstance(evo_recommend, dict) else []
outcome_rows = (evo_recommend.get("feedback_loop", {}) if isinstance(evo_recommend, dict) else {}).get("outcome_tracking", [])
render_registry_metrics(dashboard="prometheus_execution_academy", section="recommendation_intelligence", sources=_registry_sources, columns_count=4)

st.markdown("### Edge Intelligence")
render_registry_metrics(dashboard="prometheus_execution_academy", section="edge_intelligence", sources=_registry_sources, columns_count=5)

st.markdown("### Decision Attribution")
render_registry_tables(dashboard="prometheus_execution_academy", section="decision_attribution", sources=_registry_sources, max_rows=20)

st.markdown("### Confidence Intelligence")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Calibration Error", conf.get("calibration_error", "Awaiting Historical Data"))
c2.metric("Brier Score", conf.get("brier_score", "Awaiting Historical Data"))
c3.metric("Confidence Drift", conf.get("confidence_drift", "Awaiting Historical Data"))
c4.metric("Confidence Stability", conf.get("confidence_stability", "Awaiting Historical Data"))
c5.metric("Optimal Threshold", conf.get("optimal_confidence_threshold", "Awaiting Historical Data"))

cal = pd.DataFrame(conf.get("confidence_calibration", []))
if not cal.empty:
    st.dataframe(cal, use_container_width=True, hide_index=True)

st.markdown("### Return Intelligence")
d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("Average Return", ret.get("average_return", "Awaiting Historical Data"))
d2.metric("Return Efficiency", ret.get("return_efficiency", "Awaiting Historical Data"))
d3.metric("Risk Efficiency", ret.get("risk_efficiency", "Awaiting Historical Data"))
d4.metric("Average R", ret.get("average_r", "Awaiting Historical Data"))
d5.metric("Payoff Ratio", ret.get("payoff_ratio", "Awaiting Historical Data"))
d6.metric("Recovery Factor", ret.get("recovery_factor", "Awaiting Historical Data"))

exp_evo = pd.DataFrame(ret.get("expectancy_evolution", []))
if not exp_evo.empty and "rolling_expectancy" in exp_evo.columns:
    st.plotly_chart(px.line(exp_evo, x="index", y="rolling_expectancy", title="Expectancy Evolution"), use_container_width=True)

st.markdown("### Pattern Health and Pattern Library")
pattern_health = pd.DataFrame((intel.get("pattern_health_engine", {}) or {}).get("patterns", []))
pattern_lib = pd.DataFrame(intel.get("pattern_library", []))
left, right = st.columns(2)
with left:
    if not pattern_health.empty:
        st.dataframe(pattern_health.head(200), use_container_width=True, hide_index=True)
with right:
    if not pattern_lib.empty:
        cols = [
            c
            for c in [
                "pattern_id",
                "historical_samples",
                "winning_samples",
                "losing_samples",
                "average_return",
                "knowledge_confidence",
                "pattern_evolution",
                "lifecycle_stage",
                "source_attribution",
            ]
            if c in pattern_lib.columns
        ]
        st.dataframe(pattern_lib[cols] if cols else pattern_lib, use_container_width=True, hide_index=True)

st.markdown("### Session and Regime Intelligence")
si = pd.DataFrame(intel.get("session_intelligence", []))
ri = pd.DataFrame(intel.get("regime_intelligence", []))
col_s, col_r = st.columns(2)
with col_s:
    if not si.empty:
        st.dataframe(si, use_container_width=True, hide_index=True)
with col_r:
    if not ri.empty:
        st.dataframe(ri, use_container_width=True, hide_index=True)

st.markdown("### Edge Stability")
edge_rows = [{"edge": k, "value": v} for k, v in edge.items() if k not in ("trend",)]
if edge_rows:
    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
if isinstance(edge.get("trend"), dict):
    st.dataframe(pd.DataFrame([{"edge": k, "trend": v} for k, v in edge.get("trend", {}).items()]), use_container_width=True, hide_index=True)

st.markdown("### Decision Intelligence")
reviews = pd.DataFrame((intel.get("decision_intelligence", {}) or {}).get("reviews", []))
if not reviews.empty:
    show = [
        c
        for c in [
            "trade_id",
            "decision",
            "why_accepted",
            "market_health",
            "risk_level",
            "session_quality",
            "regime_quality",
            "decision_quality",
            "knowledge_confidence",
        ]
        if c in reviews.columns
    ]
    st.dataframe(reviews[show].tail(200) if show else reviews.tail(200), use_container_width=True, hide_index=True)

st.markdown("### Academy Recommendations")
for rec in academy.get("recommendations", []):
    st.write(f"- {rec}")
for rec in rec_rows[:8]:
    st.write(
        f"- [{rec.get('recommendation_priority', 'Low')}] {rec.get('finding', 'Awaiting Historical Data')} "
        f"| confidence {rec.get('confidence', 'Awaiting Historical Data')} "
        f"| capital {rec.get('expected_capital_impact', 'Neutral')}"
    )

st.markdown("### Olympus Compatibility and Governance")
st.json(
    {
        "meta": intel.get("meta", {}),
        "governance": intel.get("governance", {}),
        "artifact_paths": paths,
        "shared_knowledge_contracts": len(intel.get("shared_knowledge_contracts", [])),
    }
)

st.caption("Execution Academy is independent and observational. It evaluates and recommends only; it never modifies Prometheus trading behavior.")
