"""Hermes Return Intelligence Dashboard.

Run:
    streamlit run ui/hermes_return_dashboard.py --server.port=8505
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

st.set_page_config(page_title="Hermes Return Intelligence", page_icon="↗", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#1d1111 0%,#1a2415 100%); border:1px solid #5b4630; border-radius:14px; padding:22px 28px; margin-bottom:20px;}
    .hero h1 {margin:0; color:#f5ead7; font-size:2.15rem;}
    .hero p {margin:6px 0 0; color:#d6c8b0;}
    .panel {background: linear-gradient(135deg,#151515 0%,#1a1f24 100%); border:1px solid #39424a; border-radius:12px; padding:16px 18px; margin:10px 0 18px 0;}
    .panel pre {margin:0; color:#edf2f4; font-size:0.93rem; line-height:1.45; white-space:pre-wrap;}
    </style>
    <div class="hero">
      <h1>↗ Hermes Return Intelligence</h1>
      <p>Observational return-efficiency research for completed simulated, paper, and live trades.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not _STATUS_F.exists():
    st.warning("Hermes status file not found yet. Start the bot first.")
    st.stop()

status = json.loads(_STATUS_F.read_text(encoding="utf-8"))
analytics = {}
try:
    analytics = build_hermes_analytics(_ROOT)
except Exception:
    analytics = {}

return_intel = status.get("return_intelligence") or analytics.get("return_intelligence", {})
summary = return_intel.get("summary", {}) if isinstance(return_intel, dict) else {}
report = return_intel.get("research_report", {}) if isinstance(return_intel, dict) else {}
academy = return_intel.get("academy_subject", {}) if isinstance(return_intel, dict) else {}
edge = return_intel.get("edge_stability", {}) if isinstance(return_intel, dict) else {}
proposals = return_intel.get("zeus_research_proposals", []) if isinstance(return_intel, dict) else []
trades = pd.DataFrame(return_intel.get("trade_metrics", []) if isinstance(return_intel, dict) else [])

st.markdown("### Return Summary")
a1, a2, a3, a4, a5, a6 = st.columns(6)
a1.metric("Sample Size", summary.get("sample_size", 0))
a2.metric("Average Return %", summary.get("average_return_pct", "Awaiting Historical Data"))
a3.metric("Average Win %", summary.get("average_winning_return_pct", "Awaiting Historical Data"))
a4.metric("Average Loss %", summary.get("average_losing_return_pct", "Awaiting Historical Data"))
a5.metric("Return Stability", summary.get("return_stability", "Awaiting Historical Data"))
a6.metric("Return Trend", summary.get("return_trend", "Awaiting Historical Data"))

b1, b2, b3, b4 = st.columns(4)
b1.metric("Return Efficiency", summary.get("average_return_efficiency_pct", "Awaiting Historical Data"))
b2.metric("Loss Efficiency", summary.get("average_loss_efficiency_pct", "Awaiting Historical Data"))
b3.metric("Opportunity Efficiency", summary.get("average_opportunity_efficiency_pct", "Awaiting Historical Data"))
b4.metric("Execution Efficiency", summary.get("average_execution_efficiency_pct", "Awaiting Historical Data"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capture Ratio", summary.get("average_captured_return_pct", "Awaiting Historical Data"))
c2.metric("Risk Utilization", summary.get("average_risk_utilization_pct", "Awaiting Historical Data"))
c3.metric("Median Return", summary.get("median_return_pct", "Awaiting Historical Data"))
c4.metric("Return Skew", summary.get("return_skew", "Awaiting Historical Data"))

st.markdown("### Exit Intelligence")
if return_intel.get("exit_intelligence"):
    st.json(return_intel.get("exit_intelligence"))

st.markdown("### Historical Return Evolution")
evolution = return_intel.get("historical_return_evolution", []) if isinstance(return_intel, dict) else []
if evolution:
    evo_df = pd.DataFrame(evolution)
    if "cumulative_return" in evo_df.columns:
        st.plotly_chart(px.line(evo_df, x="index", y="cumulative_return", title="Cumulative Return Evolution"), use_container_width=True)
    if "rolling_average_return" in evo_df.columns:
        st.plotly_chart(px.line(evo_df, x="index", y="rolling_average_return", title="Rolling Average Return"), use_container_width=True)
else:
    st.caption("Return evolution is awaiting historical samples.")

left, right = st.columns(2)
with left:
    st.markdown("### Best Performing Patterns")
    if return_intel.get("return_by_pattern"):
        st.dataframe(pd.DataFrame(return_intel.get("return_by_pattern", [])).head(20), use_container_width=True, hide_index=True)
    else:
        st.caption("No pattern-level return profile yet.")

    st.markdown("### Best Sessions")
    if return_intel.get("return_by_session"):
        st.dataframe(pd.DataFrame(return_intel.get("return_by_session", [])).head(20), use_container_width=True, hide_index=True)

with right:
    st.markdown("### Best Market Regimes")
    if return_intel.get("return_by_regime"):
        st.dataframe(pd.DataFrame(return_intel.get("return_by_regime", [])).head(20), use_container_width=True, hide_index=True)

    st.markdown("### Worst Performing Patterns")
    if trades is not None and not trades.empty:
        worst = trades.sort_values("realized_return_pct", ascending=True).head(20)
        st.dataframe(worst[[c for c in ["pattern_name", "pattern_family", "session", "regime", "realized_return_pct", "exit_quality"] if c in worst.columns]], use_container_width=True, hide_index=True)

if not trades.empty:
    st.markdown("### Return Distribution")
    if "realized_return_pct" in trades.columns:
        st.plotly_chart(px.histogram(trades, x="realized_return_pct", nbins=25, title="Realized Return Distribution"), use_container_width=True)

st.markdown("### Knowledge Confidence")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Evidence Level", academy.get("evidence_level", "Awaiting Historical Data"))
k2.metric("Knowledge Confidence", academy.get("knowledge_confidence", "Awaiting Historical Data"))
k3.metric("Mastery", academy.get("mastery", "Awaiting Historical Data"))
k4.metric("Current Grade", academy.get("current_grade", "Awaiting Historical Data"))
k5.metric("Reliability", academy.get("reliability", "Awaiting Historical Data"))

if summary.get("next_milestone"):
    st.info(f"Next Milestone: {summary.get('next_milestone')} | Estimated Additional Samples: {summary.get('estimated_additional_samples', 0)}")

st.markdown("### Return Edge Stability")
if edge:
    st.dataframe(pd.DataFrame([{"edge": k, "value": v} for k, v in edge.items() if k != "history"]), use_container_width=True, hide_index=True)
    history = edge.get("history", []) if isinstance(edge, dict) else []
    if history:
        st.dataframe(pd.DataFrame(history).tail(20), use_container_width=True, hide_index=True)

st.markdown("### Current Research Focus")
if report:
    st.json({
        "current_observations": report.get("current_observations", {}),
        "emerging_trends": report.get("emerging_trends", []),
        "new_hypotheses": report.get("new_hypotheses", []),
    })

if report.get("zeus_research_proposals"):
    st.markdown("### Zeus Research Proposals")
    st.dataframe(pd.DataFrame(report.get("zeus_research_proposals", [])), use_container_width=True, hide_index=True)

st.caption("Return Intelligence is observational only and does not influence trading behavior.")
