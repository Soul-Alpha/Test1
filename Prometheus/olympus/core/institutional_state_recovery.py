from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return default
        return json.loads(raw)
    except Exception:
        return default


def _load_jsonl(path: Path, *, tail: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        if tail is not None and tail > 0:
            line_iter: Any = deque(maxlen=tail)
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        line_iter.append(line)
        else:
            with path.open("r", encoding="utf-8") as fh:
                line_iter = [line for line in fh if line.strip()]

        for line in line_iter:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return rows
    return rows


def _has_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value not in (None, "", 0)


def _latest_history_row(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path, tail=200)
    for row in reversed(rows):
        if isinstance(row, dict) and row:
            return row
    return {}


def _recover_runtime(runtime_path: Path, history_path: Path | None = None) -> tuple[dict[str, Any], str]:
    runtime = _load_json(runtime_path, {})
    if isinstance(runtime, dict) and _has_payload(runtime):
        return runtime, "runtime"
    if history_path is not None:
        hist = _latest_history_row(history_path)
        if hist:
            return hist, "history"
    return {}, "default"


def _summarize_zeus_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"summary": {}, "queue": {"counts": {}}}
    total = len(reports)
    passed = 0
    queue_counts: dict[str, int] = {}
    for row in reports:
        status = str((row or {}).get("status", "") or "").lower()
        if status == "passed":
            passed += 1
        state = str((row or {}).get("queue_state", "Unknown") or "Unknown")
        queue_counts[state] = queue_counts.get(state, 0) + 1

    return {
        "summary": {
            "total": total,
            "validated_research": passed,
            "validation_success_rate": round(passed / max(1, total), 4),
        },
        "queue": {"counts": queue_counts},
    }


def _summarize_zvo_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    transitions = [r for r in rows if isinstance(r, dict)]
    return {
        "scheduler": {
            "transitions_executed": len(transitions),
            "recovered_from_history": True,
        },
        "recent_transitions": transitions[-50:],
    }


def recover_institutional_state(root_dir: Path, *, write_snapshot: bool = False) -> dict[str, Any]:
    storage = root_dir / "storage" / "olympus"
    status_path = root_dir / "live_bot" / "hermes_status.json"

    idip_runtime, idip_source = _recover_runtime(storage / "idip_runtime.json", storage / "idip_history.jsonl")
    idip_history = _load_jsonl(storage / "idip_history.jsonl", tail=1000)
    idip_last = idip_runtime if idip_source == "runtime" else (idip_history[-1] if idip_history else {})
    idip_engines = (idip_last.get("engines", {}) if isinstance(idip_last, dict) else {}) or {}

    status = _load_json(status_path, {})
    if not isinstance(status, dict):
        status = {}

    institutional_learning = _load_json(storage / "institutional_learning.json", {})
    if not _has_payload(institutional_learning):
        institutional_learning = (idip_engines.get("institutional_learning_scientist", {}) or {}).get("institutional_learning", {})

    hypotheses = _load_json(storage / "hypotheses.json", {})
    if not _has_payload(hypotheses):
        hypotheses = (idip_engines.get("institutional_learning_scientist", {}) or {}).get("hypotheses", {})

    knowledge_growth = _load_json(storage / "knowledge_growth.json", {})
    if not _has_payload(knowledge_growth):
        knowledge_growth = (idip_engines.get("institutional_learning_scientist", {}) or {}).get("knowledge_growth", {})

    learning_velocity = _load_json(storage / "learning_velocity.json", {})
    if not _has_payload(learning_velocity):
        learning_velocity = (idip_engines.get("institutional_learning_scientist", {}) or {}).get("learning_velocity", {})

    research_queue = _load_json(storage / "research_queue.json", {})
    if not _has_payload(research_queue):
        research_queue = (idip_engines.get("institutional_learning_scientist", {}) or {}).get("research_queue", {})

    concept_drift = _load_json(storage / "concept_drift.json", {})
    if not _has_payload(concept_drift):
        concept_drift = (idip_engines.get("institutional_learning_scientist", {}) or {}).get("concept_drift", {})

    capital_intelligence = _load_json(storage / "capital_intelligence_runtime.json", {})
    if not _has_payload(capital_intelligence):
        capital_intelligence = idip_engines.get("capital_intelligence_engine", {}) or {}

    knowledge_graph, knowledge_graph_source = _recover_runtime(
        storage / "institutional_knowledge_graph.json",
        storage / "institutional_knowledge_graph_history.jsonl",
    )
    if not _has_payload(knowledge_graph):
        knowledge_graph = idip_engines.get("knowledge_graph_engine", {}) or {}
        knowledge_graph_source = "idip_runtime" if _has_payload(knowledge_graph) else knowledge_graph_source

    replay_payload, replay_source = _recover_runtime(
        storage / "decision_replay_counterfactual.json",
        storage / "decision_replay_counterfactual_history.jsonl",
    )
    if not _has_payload(replay_payload):
        replay_payload = idip_engines.get("decision_replay_counterfactual_intelligence", {}) or {}
        replay_source = "idip_runtime" if _has_payload(replay_payload) else replay_source

    meta_learning_payload, meta_source = _recover_runtime(storage / "meta_learning_runtime.json", storage / "meta_learning_history.jsonl")
    aro_payload, aro_source = _recover_runtime(storage / "autonomous_research_orchestrator_runtime.json", storage / "autonomous_research_orchestrator_history.jsonl")
    coverage_payload, coverage_source = _recover_runtime(storage / "knowledge_coverage_runtime.json", storage / "knowledge_coverage_history.jsonl")
    explainability_payload, explainability_source = _recover_runtime(storage / "explainability_runtime.json", storage / "explainability_history.jsonl")
    research_director_payload, research_director_source = _recover_runtime(
        storage / "institutional_research_director_runtime.json",
        storage / "institutional_research_director_history.jsonl",
    )

    knowledge_evolution_payload, knowledge_evolution_source = _recover_runtime(
        storage / "knowledge_evolution_runtime.json",
        storage / "knowledge_evolution_history.jsonl",
    )
    if not _has_payload(knowledge_evolution_payload):
        knowledge_evolution_payload = idip_engines.get("knowledge_evolution_engine", {}) or {}
        knowledge_evolution_source = "idip_runtime" if _has_payload(knowledge_evolution_payload) else knowledge_evolution_source

    zeus_reports = _load_jsonl(storage / "zeus_validation_reports.jsonl", tail=2000)
    zeus_status = _load_json(storage / "zeus_validation_status.json", {})
    zeus_source = "runtime"
    if not _has_payload(zeus_status):
        zeus_status = _summarize_zeus_reports(zeus_reports)
        zeus_source = "history"

    zvo_runtime = _load_json(storage / "zeus_validation_operations_runtime.json", {})
    zvo_source = "runtime"
    if not _has_payload(zvo_runtime):
        zvo_runtime = _summarize_zvo_history(_load_jsonl(storage / "zeus_validation_operations_history.jsonl", tail=2000))
        zvo_source = "history" if _has_payload(zvo_runtime) else "default"

    dataset_architecture, dataset_architecture_source = _recover_runtime(
        storage / "institutional_dataset_architecture_runtime.json",
        storage / "institutional_dataset_architecture_history.jsonl",
    )
    zeus_boards, zeus_boards_source = _recover_runtime(
        storage / "zeus_validation_boards_runtime.json",
        storage / "zeus_validation_boards_history.jsonl",
    )
    institutional_knowledge_base, institutional_knowledge_base_source = _recover_runtime(
        storage / "institutional_knowledge_base_runtime.json",
        storage / "institutional_knowledge_base_history.jsonl",
    )
    institutional_dataset_rows = _load_jsonl(storage / "institutional_dataset_rows.jsonl", tail=5000)

    recovery_audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "idip_source": idip_source,
        "knowledge_graph_source": knowledge_graph_source,
        "knowledge_evolution_source": knowledge_evolution_source,
        "replay_source": replay_source,
        "meta_learning_source": meta_source,
        "aro_source": aro_source,
        "coverage_source": coverage_source,
        "explainability_source": explainability_source,
        "research_director_source": research_director_source,
        "zeus_source": zeus_source,
        "zvo_source": zvo_source,
        "dataset_architecture_source": dataset_architecture_source,
        "zeus_boards_source": zeus_boards_source,
        "institutional_knowledge_base_source": institutional_knowledge_base_source,
    }

    state = {
        "status": status,
        "idip": idip_last if isinstance(idip_last, dict) else {},
        "idip_summary": (idip_last.get("summary", {}) if isinstance(idip_last, dict) else {}) or {},
        "institutional_learning": institutional_learning if isinstance(institutional_learning, dict) else {},
        "hypotheses": hypotheses if isinstance(hypotheses, dict) else {},
        "knowledge_growth": knowledge_growth if isinstance(knowledge_growth, dict) else {},
        "learning_velocity": learning_velocity if isinstance(learning_velocity, dict) else {},
        "research_queue": research_queue if isinstance(research_queue, dict) else {},
        "concept_drift": concept_drift if isinstance(concept_drift, dict) else {},
        "capital_intelligence": capital_intelligence if isinstance(capital_intelligence, dict) else {},
        "knowledge_graph": knowledge_graph if isinstance(knowledge_graph, dict) else {},
        "replay_payload": replay_payload if isinstance(replay_payload, dict) else {},
        "meta_learning_payload": meta_learning_payload if isinstance(meta_learning_payload, dict) else {},
        "aro_payload": aro_payload if isinstance(aro_payload, dict) else {},
        "coverage_payload": coverage_payload if isinstance(coverage_payload, dict) else {},
        "knowledge_evolution_payload": knowledge_evolution_payload if isinstance(knowledge_evolution_payload, dict) else {},
        "explainability_payload": explainability_payload if isinstance(explainability_payload, dict) else {},
        "research_director_payload": research_director_payload if isinstance(research_director_payload, dict) else {},
        "idip_history": idip_history,
        "zeus_reports": zeus_reports,
        "zeus_status": zeus_status if isinstance(zeus_status, dict) else {},
        "zvo_runtime": zvo_runtime if isinstance(zvo_runtime, dict) else {},
        "institutional_dataset_architecture": dataset_architecture if isinstance(dataset_architecture, dict) else {},
        "zeus_validation_boards": zeus_boards if isinstance(zeus_boards, dict) else {},
        "institutional_knowledge_base": institutional_knowledge_base if isinstance(institutional_knowledge_base, dict) else {},
        "institutional_dataset_rows": institutional_dataset_rows,
        "recovery_audit": recovery_audit,
    }

    if write_snapshot:
        snapshot_path = storage / "institutional_state_recovery_runtime.json"
        # Keep on-disk recovery snapshots compact to avoid disk-pressure loops.
        compact_state = {
            **state,
            "idip_history": state.get("idip_history", [])[-25:],
            "zeus_reports": state.get("zeus_reports", [])[-100:],
        }
        try:
            snapshot_path.write_text(json.dumps(compact_state, indent=2, ensure_ascii=True), encoding="utf-8")
        except OSError as exc:
            recovery_audit["snapshot_write_error"] = str(exc)

    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover Olympus institutional state from persisted artifacts.")
    parser.add_argument("--root", default=".", help="Prometheus root directory")
    parser.add_argument("--write-snapshot", action="store_true")
    args = parser.parse_args()

    state = recover_institutional_state(Path(args.root).resolve(), write_snapshot=args.write_snapshot)
    print(
        json.dumps(
            {
                "ok": True,
                "idip_history_rows": len(state.get("idip_history", [])),
                "zeus_reports": len(state.get("zeus_reports", [])),
                "recovery_audit": state.get("recovery_audit", {}),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
