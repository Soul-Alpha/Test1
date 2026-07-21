"""Prometheus Academy Observability Dashboard.

Run:
    streamlit run ui/prometheus_academy_observability_dashboard.py --server.port=8509
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DECISION_INTEL = _ROOT / "storage" / "olympus" / "prometheus_decision_intelligence.json"
_ACADEMY_REPORT = _ROOT / "storage" / "olympus" / "prometheus_execution_academy_report.json"
_RESEARCH_LIB = _ROOT / "storage" / "olympus" / "prometheus_research_library.jsonl"

@st.cache_data(ttl=60)
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _file_age_minutes(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - updated).total_seconds() / 60.0
    except Exception:
        return None


@st.cache_data(ttl=60)
def _read_research_tail(path: Path, max_lines: int = 100) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max_lines:]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return []
    return rows


st.set_page_config(page_title="Prometheus Academy Observability", page_icon="PAO", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#142018 0%,#17263a 100%); border:1px solid #2e6446; border-radius:14px; padding:20px 24px; margin-bottom:18px;}
    .hero h1 {margin:0; color:#ebfff4; font-size:2rem;}
    .hero p {margin:6px 0 0; color:#c7dece;}
    </style>
    <div class="hero">
      <h1>Prometheus Academy Observability</h1>
      <p>Lightweight operational panel for artifact freshness, key academy telemetry, and evidence throughput.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

intel = _read_json(_DECISION_INTEL)
report = _read_json(_ACADEMY_REPORT)
research_rows = _read_research_tail(_RESEARCH_LIB)

if not intel:
    st.warning("Academy intelligence artifact not found yet. Run the academy builder/dashboard first.")
    st.stop()

meta = intel.get("meta", {})
academy = intel.get("execution_academy", {})
risk = intel.get("risk_intelligence", {})
market = intel.get("market_health_engine", {})
confidence = intel.get("confidence_intelligence", {})
returns = intel.get("return_intelligence", {})
knowledge = intel.get("knowledge_confidence", {})

st.markdown("### Operational Health")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Decision Artifact", "Present" if _DECISION_INTEL.exists() else "Missing")
c2.metric("Academy Report", "Present" if _ACADEMY_REPORT.exists() else "Missing")
c3.metric("Research Library", "Present" if _RESEARCH_LIB.exists() else "Missing")

age_decision = _file_age_minutes(_DECISION_INTEL)
age_report = _file_age_minutes(_ACADEMY_REPORT)
c4.metric("Decision Age (min)", f"{age_decision:.1f}" if age_decision is not None else "n/a")
c5.metric("Report Age (min)", f"{age_report:.1f}" if age_report is not None else "n/a")

st.markdown("### Academy and Market Snapshot")
a1, a2, a3, a4, a5, a6 = st.columns(6)
a1.metric("Academy Stage", academy.get("current_stage", "Awaiting Historical Data"))
a2.metric("Execution GPA", academy.get("execution_gpa", "Awaiting Historical Data"))
a3.metric("Institutional Grade", academy.get("institutional_grade", "Awaiting Historical Data"))
a4.metric("Market Health", market.get("market_health_score", "Awaiting Historical Data"))
a5.metric("Risk Momentum", risk.get("risk_momentum", "Awaiting Historical Data"))
a6.metric("Adaptive Readiness", risk.get("adaptive_readiness", "Awaiting Historical Data"))

st.markdown("### Intelligence Integrity")
i1, i2, i3, i4, i5, i6 = st.columns(6)
i1.metric("Knowledge Confidence", knowledge.get("knowledge_confidence", "Awaiting Historical Data"))
i2.metric("Evidence Level", knowledge.get("evidence_level", "Awaiting Historical Data"))
i3.metric("Sample Size", knowledge.get("sample_size", "Awaiting Historical Data"))
i4.metric("Confidence Stability", confidence.get("confidence_stability", "Awaiting Historical Data"))
i5.metric("Calibration Error", confidence.get("calibration_error", "Awaiting Historical Data"))
i6.metric("Return Stability", returns.get("return_stability", "Awaiting Historical Data"))

st.markdown("### Artifact Meta")
left, right = st.columns(2)
with left:
    st.json(
        {
            "source_system": meta.get("source_system"),
            "generated_at": meta.get("generated_at"),
            "observational_only": meta.get("observational_only"),
            "execution_behavior_unchanged": meta.get("execution_behavior_unchanged"),
        }
    )
with right:
    st.json(
        {
            "runtime_impact": report.get("runtime_impact", "n/a"),
            "memory_impact": report.get("memory_impact", "n/a"),
            "storage_impact": report.get("storage_impact", "n/a"),
            "backward_compatibility": report.get("backward_compatibility", "n/a"),
        }
    )

st.markdown("### Recent Research Throughput")
if research_rows:
    df = pd.DataFrame(research_rows)
    cols = [
        c
        for c in [
            "timestamp",
            "research_id",
            "research_category",
            "pattern_id",
            "knowledge_confidence",
            "current_status",
            "observation",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols].tail(50) if cols else df.tail(50), use_container_width=True, hide_index=True)

    if "research_category" in df.columns:
        counts = (
            df["research_category"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("research_category")
            .reset_index(name="count")
        )
        st.bar_chart(counts.set_index("research_category"))
else:
    st.caption("No research rows available yet.")

issues: list[str] = []
if _safe_float(market.get("market_health_score")) is not None and float(market.get("market_health_score")) < 45:
    issues.append("Market health is below 45 and in fragile territory.")
if _safe_float(academy.get("execution_gpa")) is not None and float(academy.get("execution_gpa")) < 50:
    issues.append("Execution GPA is below 50; maintain observation-first governance.")
if _safe_float(risk.get("risk_momentum")) is not None and float(risk.get("risk_momentum")) < 40:
    issues.append("Risk momentum is weak; review drawdown and loss velocity clusters.")
if _safe_float(knowledge.get("sample_size")) is not None and float(knowledge.get("sample_size")) < 300:
    issues.append("Sample size is low for institutional confidence stability.")

st.markdown("### Observability Alerts")
if issues:
    for item in issues:
        st.warning(item)
else:
    st.success("No immediate observability alerts from current academy telemetry.")

st.caption("This panel is standalone, lightweight, and observational only. It does not alter Prometheus execution behavior.")
