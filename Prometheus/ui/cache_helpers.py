"""Centralized caching helpers for Streamlit dashboards.

This module provides reusable cached functions for expensive operations
to ensure consistent performance optimization across all dashboards.

Usage:
    from ui.cache_helpers import cache_json_load, cache_institutional_state, cache_hermes_analytics
    status = cache_json_load(path, ttl=60)
    state = cache_institutional_state(root)
    analytics = cache_hermes_analytics(root)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


@st.cache_data(ttl=60)
def cache_json_load(path: Path, default: Any = None) -> Any:
    """Load JSON file with 60-second cache TTL.
    
    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or fails to parse
        
    Returns:
        Parsed JSON data or default value
    """
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@st.cache_data(ttl=60)
def cache_jsonl_load(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file (one JSON per line) with 60-second cache TTL.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception:
                continue
    except Exception:
        return []
    return rows


@st.cache_data(ttl=120)
def cache_institutional_state(root: Path) -> dict[str, Any]:
    """Cache institutional state recovery with 120-second TTL.
    
    Heavy operation that reconstructs entire institutional intelligence state.
    
    Args:
        root: Prometheus root directory
        
    Returns:
        Recovered institutional state dictionary
    """
    try:
        from olympus.core.institutional_state_recovery import recover_institutional_state
        return recover_institutional_state(root)
    except Exception:
        return {}


@st.cache_data(ttl=120)
def cache_prometheus_decision_intelligence(root: Path) -> dict[str, Any]:
    """Cache Prometheus decision intelligence builder with 120-second TTL.
    
    Args:
        root: Prometheus root directory
        
    Returns:
        Decision intelligence payload
    """
    try:
        from olympus.core.prometheus_decision_intelligence import build_prometheus_decision_intelligence
        return build_prometheus_decision_intelligence(root)
    except Exception:
        return {}


@st.cache_data(ttl=120)
def cache_prometheus_evolution_intelligence(root: Path) -> dict[str, Any]:
    """Cache Prometheus evolution intelligence builder with 120-second TTL.
    
    Args:
        root: Prometheus root directory
        
    Returns:
        Evolution intelligence payload
    """
    try:
        from olympus.core.prometheus_evolution_intelligence import build_prometheus_evolution_intelligence
        return build_prometheus_evolution_intelligence(root)
    except Exception:
        return {}


@st.cache_data(ttl=120)
def cache_hermes_analytics(root: Path) -> dict[str, Any]:
    """Cache Hermes analytics builder with 120-second TTL.
    
    Args:
        root: Prometheus root directory
        
    Returns:
        Hermes analytics payload
    """
    try:
        from olympus.core.hermes_analytics import build_hermes_analytics
        return build_hermes_analytics(root)
    except Exception:
        return {}
