from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_bot.hermes import _update_idip_status
from ui.knowledge_growth_dashboard_support import (
    summarize_research_prioritization,
    top_research_prioritization_rows,
)


def test_update_idip_status_exposes_research_prioritization_payload() -> None:
    status: dict[str, object] = {}
    idip = {
        "summary": {"sample_size": 12},
        "engines": {
            "research_prioritization_engine": {
                "summary": {"backlog": 2, "high_priority": 1, "medium_priority": 1, "low_priority": 0},
                "prioritized_research_roadmap": [
                    {"recommendation_id": "R-2", "priority": "Medium", "priority_score": 0.12, "confidence": 0.51},
                ],
            }
        },
        "zeus_research_recommendations": [],
        "self_improvement_loop": {"active": True},
    }

    _update_idip_status(status, idip=idip, idip_artifacts={"runtime": "runtime.json"})

    assert status["idip_summary"] == {"sample_size": 12}
    assert status["idip_research_prioritization"] == idip["engines"]["research_prioritization_engine"]
    assert status["idip_artifacts"] == {"runtime": "runtime.json"}


def test_research_prioritization_dashboard_helpers_handle_empty_payload() -> None:
    assert summarize_research_prioritization({}) == {
        "backlog": 0,
        "high_priority": 0,
        "medium_priority": 0,
        "low_priority": 0,
    }
    assert top_research_prioritization_rows({}, limit=5) == []


def test_research_prioritization_dashboard_helpers_rank_rows() -> None:
    payload = {
        "prioritized_research_roadmap": [
            {
                "recommendation_id": "R-low",
                "recommendation": "Lower impact",
                "validation_domain": "recommendation",
                "priority": "Low",
                "priority_score": 0.03,
                "confidence": 0.25,
                "required_samples": 10,
            },
            {
                "recommendation_id": "R-high",
                "recommendation": "Higher impact",
                "validation_domain": "pattern",
                "priority": "High",
                "priority_score": 0.41,
                "confidence": 0.77,
                "required_samples": 40,
            },
        ]
    }

    summary = summarize_research_prioritization(payload)
    rows = top_research_prioritization_rows(payload, limit=1)

    assert summary == {
        "backlog": 2,
        "high_priority": 1,
        "medium_priority": 0,
        "low_priority": 1,
    }
    assert rows == [
        {
            "recommendation_id": "R-high",
            "recommendation": "Higher impact",
            "validation_domain": "pattern",
            "priority": "High",
            "priority_score": 0.41,
            "confidence": 0.77,
            "required_samples": 40,
        }
    ]
