from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from olympus.core.dashboard_registry import resolve_dashboard_section


def render_registry_metrics(
    *,
    dashboard: str,
    section: str,
    sources: dict[str, Any],
    columns_count: int,
) -> None:
    entries = [row for row in resolve_dashboard_section(dashboard, section, sources) if row.get("component") == "metric"]
    if not entries:
        return
    cols = st.columns(columns_count)
    for idx, entry in enumerate(entries):
        cols[idx % columns_count].metric(entry.get("label", "Metric"), entry.get("resolved_value", entry.get("default_value", "Awaiting Historical Data")))


def render_registry_tables(
    *,
    dashboard: str,
    section: str,
    sources: dict[str, Any],
    max_rows: int = 100,
    title_prefix: str | None = "#### ",
    arrow_safe: Any | None = None,
) -> None:
    entries = [row for row in resolve_dashboard_section(dashboard, section, sources) if row.get("component") == "table"]
    for entry in entries:
        value = entry.get("resolved_value")
        if not isinstance(value, list) or not value:
            continue
        if title_prefix is not None:
            st.markdown(f"{title_prefix}{entry.get('label', 'Table')}")
        df = pd.DataFrame(value)
        cols = [c for c in entry.get("table_columns", []) if c in df.columns]
        if cols:
            df = df[cols]
        if arrow_safe is not None:
            df = arrow_safe(df)
        st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)


def render_registry_texts(*, dashboard: str, section: str, sources: dict[str, Any]) -> None:
    entries = [row for row in resolve_dashboard_section(dashboard, section, sources) if row.get("component") == "text"]
    for entry in entries:
        value = entry.get("resolved_value")
        if value and value != entry.get("default_value"):
            st.caption(str(value))