from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWLEDGE_EVOLUTION_VERSION = "knowledge-evolution-v1.0"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def build_knowledge_evolution_engine(root_dir: Path, *, idip_payload: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    storage = root_dir / "storage" / "olympus"

    graph = _load_json(storage / "institutional_knowledge_graph.json", {})
    lessons = (idip_payload.get("engines", {}) or {}).get("institutional_knowledge_intelligence", {}).get("institutional_lessons", [])
    history = _load_jsonl(storage / "institutional_knowledge_graph_history.jsonl")[-240:]

    objects = []
    for lesson in lessons[-300:]:
        kid = str(lesson.get("knowledge_id") or "unknown")
        confidence = float(lesson.get("decision_quality", 0.0) or 0.0) / 100.0
        freshness = 1.0
        stability = min(1.0, max(0.0, confidence + 0.15))
        maturity = "Validated" if confidence >= 0.6 else "Developing"
        utilization = min(1.0, len(history) / 120.0)
        age = 0
        obsolescence = max(0.0, 1.0 - (confidence * 0.7 + utilization * 0.3))

        action = "revalidate" if obsolescence > 0.45 else "refine" if confidence < 0.55 else "expand"
        objects.append(
            {
                "knowledge_id": kid,
                "age": age,
                "freshness": round(freshness, 4),
                "confidence": round(confidence, 4),
                "stability": round(stability, 4),
                "maturity": maturity,
                "utilization": round(utilization, 4),
                "obsolescence": round(obsolescence, 4),
                "recommended_action": action,
                "recommended_version_upgrade": "v+1" if action in ("expand", "refine") else "none",
            }
        )

    action_counts = {
        "revalidate": len([x for x in objects if x.get("recommended_action") == "revalidate"]),
        "retire": len([x for x in objects if x.get("recommended_action") == "retire"]),
        "expand": len([x for x in objects if x.get("recommended_action") == "expand"]),
        "refine": len([x for x in objects if x.get("recommended_action") == "refine"]),
        "version_upgrade": len([x for x in objects if x.get("recommended_version_upgrade") == "v+1"]),
    }

    return {
        "version": KNOWLEDGE_EVOLUTION_VERSION,
        "generated_at": generated_at,
        "additive_only": True,
        "immutable_history_preserved": True,
        "knowledge_objects": objects,
        "summary": {
            "knowledge_graph_nodes": (graph.get("summary", {}) or {}).get("node_count", 0),
            "knowledge_graph_edges": (graph.get("summary", {}) or {}).get("edge_count", 0),
            "objects_tracked": len(objects),
            "action_counts": action_counts,
        },
    }


def write_knowledge_evolution_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime = storage / "knowledge_evolution_runtime.json"
    history = storage / "knowledge_evolution_history.jsonl"

    _write_json_atomic(runtime, payload)
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {"knowledge_evolution_runtime": str(runtime), "knowledge_evolution_history": str(history)}
