"""Hermes Pattern Context Intelligence Dashboard.

Run:
    streamlit run ui/hermes_pattern_context_dashboard.py --server.port=8506
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

@st.cache_data(ttl=60)
def _load_json_cached(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from olympus.core.hermes_analytics import build_hermes_analytics
from ui.dashboard_registry_support import render_registry_metrics, render_registry_tables

_STATUS_F = _ROOT / "live_bot" / "hermes_status.json"


@st.cache_data(ttl=120)
def _build_hermes_analytics_cached():
    try:
        return build_hermes_analytics(_ROOT)
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _load_hermes_status_cached():
    if not _STATUS_F.exists():
        return {}
    try:
        return json.loads(_STATUS_F.read_text(encoding="utf-8"))
    except Exception:
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


def _arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe that avoids mixed object types for Arrow transport."""
    safe = df.copy()
    for col in safe.columns:
        series = safe[col]
        if series.dtype != object:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        as_num = pd.to_numeric(non_null, errors="coerce")
        if as_num.notna().all():
            safe[col] = pd.to_numeric(series, errors="coerce")
        else:
            safe[col] = series.astype(str)

    return safe


st.set_page_config(page_title="Hermes Pattern Context Intelligence", page_icon="GEN", layout="wide")
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#10212d 0%,#1f1616 100%); border:1px solid #365168; border-radius:14px; padding:22px 28px; margin-bottom:20px;}
    .hero h1 {margin:0; color:#e8f4ff; font-size:2.1rem;}
    .hero p {margin:6px 0 0; color:#c5d8e8;}
    </style>
    <div class="hero">
      <h1>GEN Hermes Pattern Context Intelligence</h1>
      <p>Session, regime, and timing evidence mapped per pattern. Observational research only.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status = _load_hermes_status_cached()
if not status:
    st.warning("Hermes status file not found yet. Start Hermes first.")
    st.stop()

analytics = _build_hermes_analytics_cached()

ctx = analytics.get("pattern_context_intelligence", {}) if isinstance(analytics, dict) else {}
academy = ctx.get("academy_subject", {}) if isinstance(ctx, dict) else {}
metrics = analytics.get("metrics", {}) if isinstance(analytics, dict) else {}
identity = status.get("system_identity", {})
hermes_version = identity.get("build_version") or identity.get("model_version") or "v2.0"

st.markdown("### Context Intelligence Summary")
st.caption(f"Version {hermes_version}")
_registry_sources = {"hermes_status": status, "hermes_analytics": analytics}
render_registry_metrics(dashboard="hermes_pattern_context", section="context_summary", sources=_registry_sources, columns_count=6)

a1, a2, a3, a4, a5, a6 = st.columns(6)
a1.metric("Evidence", academy.get("evidence", "Awaiting Historical Data"))
a2.metric("Knowledge Confidence", academy.get("knowledge_confidence", "Awaiting Historical Data"))
a3.metric("Mastery", academy.get("mastery", "Awaiting Historical Data"))
a4.metric("Grade", academy.get("current_grade", "Awaiting Historical Data"))
a5.metric("Samples Remaining", academy.get("estimated_samples_remaining", "Awaiting Historical Data"))
a6.metric("Maturity", academy.get("status", academy.get("current_grade", "Developing")))

render_registry_metrics(dashboard="hermes_pattern_context", section="learning_summary", sources=_registry_sources, columns_count=6)

if academy.get("next_milestone"):
    st.info(f"Next Milestone: {academy.get('next_milestone')}")

left, right = st.columns(2)

with left:
    st.markdown("### Session Ranking")
    session_rank = pd.DataFrame(ctx.get("session_ranking", []) if isinstance(ctx, dict) else [])
    if not session_rank.empty:
        st.dataframe(_arrow_safe_df(session_rank), use_container_width=True, hide_index=True)
        if "session" in session_rank.columns and "expectancy" in session_rank.columns:
            chart_df = session_rank.copy()
            chart_df["expectancy_num"] = chart_df["expectancy"].apply(_safe_float)
            chart_df = chart_df[chart_df["expectancy_num"].notnull()]
            if not chart_df.empty:
                st.plotly_chart(
                    px.bar(chart_df, x="session", y="expectancy_num", title="Session Expectancy", color="expectancy_num"),
                    use_container_width=True,
                )
    else:
        st.caption("Awaiting session context samples.")

    st.markdown("### Time Heatmap (Hour)")
    hour_rank = pd.DataFrame(ctx.get("hour_ranking", []) if isinstance(ctx, dict) else [])
    if not hour_rank.empty:
        hour_rank["expectancy_num"] = hour_rank.get("expectancy", pd.Series(dtype=float)).apply(_safe_float)
        heat_df = hour_rank[hour_rank["expectancy_num"].notnull()].copy()
        if not heat_df.empty and "hour" in heat_df.columns:
            heat_df["hour"] = heat_df["hour"].astype(str)
            st.plotly_chart(
                px.bar(heat_df, x="hour", y="expectancy_num", title="Hourly Context Expectancy", color="expectancy_num"),
                use_container_width=True,
            )
        st.dataframe(_arrow_safe_df(hour_rank), use_container_width=True, hide_index=True)

with right:
    st.markdown("### Market Regime Ranking")
    regime_rank = pd.DataFrame(ctx.get("regime_ranking", []) if isinstance(ctx, dict) else [])
    if not regime_rank.empty:
        st.dataframe(_arrow_safe_df(regime_rank), use_container_width=True, hide_index=True)
        if "regime" in regime_rank.columns and "expectancy" in regime_rank.columns:
            chart_df = regime_rank.copy()
            chart_df["expectancy_num"] = chart_df["expectancy"].apply(_safe_float)
            chart_df = chart_df[chart_df["expectancy_num"].notnull()]
            if not chart_df.empty:
                st.plotly_chart(
                    px.bar(chart_df, x="regime", y="expectancy_num", title="Regime Expectancy", color="expectancy_num"),
                    use_container_width=True,
                )
    else:
        st.caption("Awaiting market-regime context samples.")

    st.markdown("### Day Ranking")
    day_rank = pd.DataFrame(ctx.get("day_ranking", []) if isinstance(ctx, dict) else [])
    if not day_rank.empty:
        st.dataframe(_arrow_safe_df(day_rank), use_container_width=True, hide_index=True)

st.markdown("### Pattern Context Profiles")
profiles = pd.DataFrame(ctx.get("context_profiles", []) if isinstance(ctx, dict) else [])
if not profiles.empty:
    render_registry_tables(dashboard="hermes_pattern_context", section="pattern_context_profiles", sources=_registry_sources, max_rows=200, arrow_safe=_arrow_safe_df, title_prefix=None)
else:
    st.caption("No pattern context profiles available yet.")

st.markdown("### Expanded Pattern Context Library")
lib = pd.DataFrame(ctx.get("pattern_context_library", []) if isinstance(ctx, dict) else [])
if not lib.empty:
    render_registry_tables(dashboard="hermes_pattern_context", section="pattern_context_library", sources=_registry_sources, max_rows=200, arrow_safe=_arrow_safe_df, title_prefix=None)

st.markdown("### Research Library (Manual Zeus Export Only)")
research = pd.DataFrame(ctx.get("research_library", []) if isinstance(ctx, dict) else [])
if not research.empty:
    st.dataframe(_arrow_safe_df(research.tail(200)), use_container_width=True, hide_index=True)
else:
    st.caption("No archived pattern-context observations yet.")

workflow = ctx.get("manual_research_workflow", {}) if isinstance(ctx, dict) else {}
if workflow:
    st.markdown("### Governance")
    st.json(workflow)

st.caption("Pattern Context Intelligence is observational and does not change entries, exits, sizing, or adaptive execution logic.")
