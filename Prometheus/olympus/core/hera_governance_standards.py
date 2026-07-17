from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_hera_governance_standards() -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    shared_domains = [
        {
            "domain": "structural_price_location_intelligence",
            "mission_scope": [
                "retracement_depth",
                "extension_depth",
                "premium_discount",
                "swing_distance",
                "liquidity_proximity",
                "range_position",
                "impulse_maturity",
                "volatility_phase",
                "expansion",
                "compression",
            ],
        },
        {
            "domain": "liquidity_intelligence",
            "mission_scope": [
                "equal_highs",
                "equal_lows",
                "resting_liquidity",
                "liquidity_sweeps",
                "inducement",
                "external_liquidity",
                "internal_liquidity",
                "post_sweep_behaviour",
            ],
        },
        {
            "domain": "market_structure_intelligence",
            "mission_scope": ["bos", "choch", "trend_transitions", "structure_hierarchy", "internal_structure", "external_structure"],
        },
        {"domain": "session_intelligence", "mission_scope": ["session_dependent_behaviour"]},
        {"domain": "market_regime_intelligence", "mission_scope": ["trend", "range", "compression", "expansion", "volatility"]},
        {"domain": "volatility_intelligence", "mission_scope": ["atr", "expansion", "compression", "volatility_cycles"]},
        {
            "domain": "higher_timeframe_context_intelligence",
            "mission_scope": ["htf_alignment", "htf_liquidity", "htf_structure", "htf_momentum"],
        },
        {
            "domain": "learning_velocity_intelligence",
            "mission_scope": ["knowledge_growth", "confidence_growth", "learning_efficiency", "research_growth", "institutional_learning_index"],
        },
    ]

    return {
        "authority": "Hera",
        "authority_role": "Institutional Architecture and Governance Authority",
        "generated_at": generated_at,
        "olympus_standards_version": "hera-v1",
        "system_independence": {
            "prometheus_independent_from_hermes": True,
            "hermes_independent_from_prometheus": True,
            "shared_implementations_forbidden": True,
            "shared_standards_only": True,
        },
        "governance_principles": {
            "hera_does_not_trade": True,
            "hera_does_not_perform_pattern_recognition": True,
            "evidence_first": True,
            "backward_compatibility_required": True,
            "non_destructive_additive_only": True,
        },
        "shared_intelligence_domains": shared_domains,
        "system_expectations": {
            "prometheus": "Implements standards independently for execution intelligence.",
            "hermes": "Implements standards independently for pattern intelligence.",
            "olympus_compatibility_required": True,
        },
    }