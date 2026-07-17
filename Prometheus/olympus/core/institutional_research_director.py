from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESEARCH_DIRECTOR_VERSION = "research-director-v1.0"


def _key(row: dict[str, Any]) -> str:
    text = str(row.get("recommendation") or row.get("statement") or "").strip().lower()
    dom = str(row.get("validation_domain") or "recommendation").strip().lower()
    return f"{dom}|{text}"


def build_institutional_research_director(*, aro_payload: dict[str, Any], hypotheses_rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()

    roadmap = aro_payload.get("prioritized_roadmap", []) if isinstance(aro_payload, dict) else []

    dedupe: dict[str, dict[str, Any]] = {}
    overlaps = []
    for row in roadmap:
        k = _key(row)
        if k in dedupe:
            overlaps.append({"primary": dedupe[k].get("recommendation_id"), "duplicate": row.get("recommendation_id"), "key": k})
            continue
        dedupe[k] = row

    merged = list(dedupe.values())

    for row in merged:
        row["research_stage"] = "active_backlog"
        row["recommended_zeus_workload_slot"] = "high" if row.get("priority") == "High" else "normal"
        row["estimated_roi"] = round(float(row.get("priority_score", 0.0)) * (1.0 + float(row.get("confidence", 0.0))), 6)

    archived = [
        {
            "item": h.get("hypothesis_id"),
            "reason": "superseded_or_duplicate",
            "timestamp": generated_at,
        }
        for h in hypotheses_rows
        if str(h.get("status", "")).lower() in ("retired", "obsolete")
    ]

    return {
        "version": RESEARCH_DIRECTOR_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "execution_modification_allowed": False,
        "merged_backlog": merged,
        "overlaps_detected": overlaps,
        "archived_research": archived,
        "long_term_strategy": {
            "objective": "Maximize institutional knowledge growth and validated edge while preserving execution integrity",
            "zeus_workload_allocation": {
                "high_priority_slots": len([x for x in merged if x.get("priority") == "High"]),
                "normal_priority_slots": len([x for x in merged if x.get("priority") != "High"]),
            },
            "future_projects": [
                "Regime-specific replay optimization",
                "Portfolio-aware hypothesis clustering",
                "Meta-learning adaptive validation cadence",
            ],
        },
        "research_roi": {
            "estimated_roi_total": round(sum(float(x.get("estimated_roi", 0.0)) for x in merged), 6),
            "active_projects": len(merged),
            "retired_projects": len(archived),
        },
    }


def write_research_director_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "institutional_research_director_runtime.json"
    history = storage / "institutional_research_director_history.jsonl"

    runtime.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"research_director_runtime": str(runtime), "research_director_history": str(history)}
