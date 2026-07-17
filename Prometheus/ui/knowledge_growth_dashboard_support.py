from __future__ import annotations

from typing import Any


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("-inf")
        if isinstance(value, str) and not value.strip():
            return float("-inf")
        return float(value)
    except Exception:
        return float("-inf")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(value)
    except Exception:
        return default


def summarize_research_prioritization(payload: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(payload, dict):
        payload = {}
    rows = payload.get("prioritized_research_roadmap", [])
    if not isinstance(rows, list):
        rows = []
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    high = _safe_int(summary.get("high_priority"), 0)
    medium = _safe_int(summary.get("medium_priority"), 0)
    low = _safe_int(summary.get("low_priority"), 0)
    backlog = _safe_int(summary.get("backlog"), len(rows))

    if rows and not any((high, medium, low)):
        high = len([row for row in rows if str((row or {}).get("priority", "")).strip().lower() == "high"])
        medium = len([row for row in rows if str((row or {}).get("priority", "")).strip().lower() == "medium"])
        low = len([row for row in rows if str((row or {}).get("priority", "")).strip().lower() == "low"])
        backlog = len(rows)

    return {
        "backlog": backlog,
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
    }


def top_research_prioritization_rows(
    payload: dict[str, Any] | None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("prioritized_research_roadmap", [])
    if not isinstance(rows, list):
        return []

    trimmed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        trimmed.append(
            {
                "recommendation_id": row.get("recommendation_id"),
                "recommendation": row.get("recommendation"),
                "validation_domain": row.get("validation_domain", "recommendation"),
                "priority": row.get("priority"),
                "priority_score": row.get("priority_score"),
                "confidence": row.get("confidence"),
                "required_samples": row.get("required_samples"),
            }
        )

    trimmed.sort(key=lambda row: _safe_float(row.get("priority_score")), reverse=True)
    return trimmed[: max(0, int(limit))]
