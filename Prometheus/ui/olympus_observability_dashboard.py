"""Olympus Observability Dashboard.

Run:
    streamlit run ui/olympus_observability_dashboard.py --server.port=8507
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from olympus.core.hermes_analytics import build_hermes_analytics

_STATUS_F = _ROOT / "live_bot" / "hermes_status.json"


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_hermes_status() -> dict:
    if not _STATUS_F.exists():
        return {}
    try:
        return json.loads(_STATUS_F.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=120)
def _build_hermes_analytics_cached():
    try:
        return build_hermes_analytics(_ROOT)
    except Exception:
        return {}


st.set_page_config(page_title="Olympus Observability", page_icon="OBS", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#161f16 0%,#17112a 100%); border:1px solid #3a5c42; border-radius:14px; padding:22px 28px; margin-bottom:20px;}
    .hero h1 {margin:0; color:#e7f7ea; font-size:2.1rem;}
    .hero p {margin:6px 0 0; color:#c7dccb;}
    </style>
    <div class="hero">
      <h1>OBS Olympus Intelligence Observability</h1>
      <p>Independent trustworthiness telemetry for data integrity, analytics integrity, evidence maturity, and dashboard consistency.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status = _load_hermes_status()
analytics = _build_hermes_analytics_cached()
if not analytics:
    analytics = {}

obs = status.get("olympus_observability") or analytics.get("olympus_observability", {})
audit = status.get("olympus_audit_report") or analytics.get("olympus_audit_report", {})
auditor = status.get("olympus_intelligence_auditor") or analytics.get("olympus_intelligence_auditor", {})

if not obs:
    st.warning("Observability payload not available yet. Ensure Hermes analytics is running.")
    st.stop()

st.markdown("### Overall System Health")
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric("Overall Health", obs.get("overall_system_health", "Awaiting Historical Data"))
a2.metric("Overall Grade", obs.get("overall_grade", "Awaiting Historical Data"))
a3.metric("Critical Findings", (obs.get("severity_counts", {}) or {}).get("critical", 0))
a4.metric("High Findings", (obs.get("severity_counts", {}) or {}).get("high", 0))
a5.metric("Pending Validation", obs.get("pending_validation", 0))

st.markdown("### Integrity and Coverage")
b1, b2, b3, b4, b5, b6 = st.columns(6)
b1.metric("Data Integrity", obs.get("data_integrity_score", "Awaiting Historical Data"))
b2.metric("Analytics Integrity", obs.get("analytics_integrity_score", "Awaiting Historical Data"))
b3.metric("Evidence Coverage", obs.get("evidence_coverage", "Awaiting Historical Data"))
b4.metric("Knowledge Coverage", obs.get("knowledge_coverage", "Awaiting Historical Data"))
b5.metric("Pattern Coverage", obs.get("pattern_coverage", "Awaiting Historical Data"))
b6.metric("Research Coverage", obs.get("research_coverage", "Awaiting Historical Data"))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Dashboard Health", obs.get("dashboard_health", "Awaiting Historical Data"))
c2.metric("Version Consistency", obs.get("version_consistency", "Awaiting Historical Data"))
c3.metric("Traceability Health", obs.get("traceability_health", "Awaiting Historical Data"))
c4.metric("Storage Health", obs.get("storage_health", "Awaiting Historical Data"))
c5.metric("Memory Health", obs.get("memory_health", "Awaiting Historical Data"))
c6.metric("Runtime Health", obs.get("runtime_health", "Awaiting Historical Data"))

st.markdown("### Synchronization and Storage")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Data Synchronization", obs.get("data_synchronization", "Awaiting Historical Data"))
d2.metric("Storage Growth MB", obs.get("storage_growth_mb", "Awaiting Historical Data"))
d3.metric("Historical Integrity", obs.get("historical_integrity", "Awaiting Historical Data"))
d4.metric("Traceability Status", obs.get("traceability_status", "Awaiting Historical Data"))

st.markdown("### Audit Findings")
findings = pd.DataFrame(obs.get("findings", []))
if not findings.empty:
    st.dataframe(findings, use_container_width=True, hide_index=True)
    sev_count = (obs.get("severity_counts", {}) or {})
    sev_df = pd.DataFrame(
        [
            {"severity": "critical", "count": sev_count.get("critical", 0)},
            {"severity": "high", "count": sev_count.get("high", 0)},
            {"severity": "medium", "count": sev_count.get("medium", 0)},
            {"severity": "low", "count": sev_count.get("low", 0)},
        ]
    )
    st.plotly_chart(px.bar(sev_df, x="severity", y="count", title="Findings by Severity", color="severity"), use_container_width=True)
else:
    st.caption("No active audit findings.")

st.markdown("### Historical Audit Timeline")
timeline = pd.DataFrame(obs.get("historical_audit_timeline", []))
if not timeline.empty:
    st.dataframe(timeline.tail(200), use_container_width=True, hide_index=True)
    if "overall_system_health" in timeline.columns:
        timeline["health_num"] = timeline["overall_system_health"].apply(_safe_float)
        t = timeline[timeline["health_num"].notnull()]
        if not t.empty and "timestamp" in t.columns:
            st.plotly_chart(px.line(t, x="timestamp", y="health_num", title="Historical System Health"), use_container_width=True)

st.markdown("### Auditor Governance")
if auditor:
    st.json(auditor)

st.markdown("### Latest Audit Report")
if audit:
    st.json(
        {
            "executive_summary": audit.get("executive_summary", {}),
            "outstanding_issues": audit.get("outstanding_issues", []),
            "recommended_actions": audit.get("recommended_actions", []),
            "runtime_impact": audit.get("runtime_impact"),
            "memory_impact": audit.get("memory_impact"),
            "storage_impact": audit.get("storage_impact"),
            "backward_compatibility": audit.get("backward_compatibility"),
        }
    )

st.caption("Olympus Intelligence Auditor is strictly observational. No automatic correction or trading behavior modification occurs.")
