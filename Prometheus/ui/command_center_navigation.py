"""Shared lazy navigation for the three institutional dashboard applications.

The existing dashboard scripts remain compatibility pages.  ``st.navigation``
executes only the selected page, so expensive dashboard builders are not run
merely because another page belongs to the same command centre.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

UI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UI_ROOT.parent


def _page(path: Path, *, title: str, icon: str, default: bool = False) -> st.Page:
    return st.Page(path, title=title, icon=icon, default=default)


def prometheus_pages() -> dict[str, list[st.Page]]:
    return {
        "Trading": [
            _page(UI_ROOT / "dashboard.py", title="Trading Command Center", icon="📈", default=True),
        ],
        "Intelligence": [
            _page(UI_ROOT / "prometheus_evolution_dashboard.py", title="Evolution", icon="🧬"),
            _page(UI_ROOT / "prometheus_execution_academy_dashboard.py", title="Execution Academy", icon="🎓"),
            _page(UI_ROOT / "prometheus_academy_observability_dashboard.py", title="Academy Health", icon="🩺"),
        ],
    }


def hermes_pages() -> dict[str, list[st.Page]]:
    return {
        "Execution": [
            _page(UI_ROOT / "hermes_dashboard.py", title="Execution Overview", icon="🪽", default=True),
        ],
        "Learning": [
            _page(UI_ROOT / "hermes_pattern_context_dashboard.py", title="Pattern Context", icon="🧩"),
            _page(UI_ROOT / "hermes_return_dashboard.py", title="Return Intelligence", icon="↗"),
            _page(UI_ROOT / "hermes_academy_dashboard.py", title="Academy", icon="🎓"),
        ],
    }


def olympus_pages() -> dict[str, list[st.Page]]:
    return {
        "Governance": [
            _page(UI_ROOT / "knowledge_growth_dashboard.py", title="Knowledge Growth", icon="🏛️", default=True),
            _page(PROJECT_ROOT / "backtesting" / "zeus_dashboard.py", title="Zeus Validation", icon="⚖️"),
        ],
        "Operations": [
            _page(UI_ROOT / "olympus_observability_dashboard.py", title="Olympus Observability", icon="🔭"),
        ],
    }


def run_command_center(pages: dict[str, list[st.Page]]) -> None:
    """Render top navigation and execute only the selected compatibility page."""
    selected = st.navigation(pages, position="top")
    selected.run()
