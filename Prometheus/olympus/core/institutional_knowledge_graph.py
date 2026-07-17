from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWLEDGE_GRAPH_VERSION = "kg-v1.0"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _decision_path(trade: dict[str, Any], attributed: dict[str, Any] | None) -> list[str]:
    return [
        f"structure:{trade.get('trend_state', 'unknown')}",
        f"pattern:{trade.get('pattern_name', 'unknown')}",
        f"signal:{trade.get('signal_id', trade.get('trade_id', 'unknown'))}",
        f"entry:{trade.get('entry_price', 'unknown')}",
        f"management:{'efficient' if float(trade.get('captured_return_pct', 0.0) or 0.0) >= 55.0 else 'developing'}",
        f"exit:{trade.get('classified_exit_style', trade.get('exit_reason', 'unknown'))}",
        f"outcome:{'win' if float(trade.get('realized_return_pct', 0.0) or 0.0) >= 0 else 'loss'}",
        f"lesson:{(attributed or {}).get('decision_score', 'unknown')}",
        "validated_knowledge:pending",
    ]


def build_institutional_knowledge_graph(
    *,
    closed_trades: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    version_seed: str,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    attribution_idx = {str(row.get("trade_id")): row for row in attribution_rows if isinstance(row, dict)}

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    decision_paths: list[dict[str, Any]] = []

    def _node(node_id: str, node_type: str, label: str) -> None:
        if node_id in nodes:
            return
        nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "created_at": generated_at,
        }

    for trade in closed_trades:
        trade_id = str(trade.get("trade_id", "unknown") or "unknown")
        attr = attribution_idx.get(trade_id, {})
        path = _decision_path(trade, attr)
        digest = hashlib.sha1("|".join(path).encode("utf-8")).hexdigest()[:16]
        path_id = f"DP-{digest}"

        last = None
        typed = [
            ("market_structure", path[0]),
            ("pattern", path[1]),
            ("signal", path[2]),
            ("entry", path[3]),
            ("trade_management", path[4]),
            ("exit", path[5]),
            ("outcome", path[6]),
            ("lessons", path[7]),
            ("validated_knowledge", path[8]),
        ]
        for node_type, label in typed:
            node_id = f"{node_type}:{label}"
            _node(node_id, node_type, label)
            if last is not None:
                edge_id = f"E-{hashlib.sha1((last + '->' + node_id).encode('utf-8')).hexdigest()[:18]}"
                edges.append(
                    {
                        "edge_id": edge_id,
                        "from": last,
                        "to": node_id,
                        "relation": "leads_to",
                        "trade_id": trade_id,
                        "path_id": path_id,
                        "version_seed": version_seed,
                        "timestamp": generated_at,
                    }
                )
            last = node_id

        decision_paths.append(
            {
                "path_id": path_id,
                "trade_id": trade_id,
                "path": path,
                "observed_before": False,
                "outcome": "win" if float(trade.get("realized_return_pct", 0.0) or 0.0) >= 0 else "loss",
                "generated_at": generated_at,
            }
        )

    seen: set[str] = set()
    for row in decision_paths:
        pid = str(row.get("path_id") or "")
        if pid in seen:
            row["observed_before"] = True
        seen.add(pid)

    return {
        "version": KNOWLEDGE_GRAPH_VERSION,
        "generated_at": generated_at,
        "graph_version": f"{KNOWLEDGE_GRAPH_VERSION}:{version_seed}",
        "additive_only": True,
        "execution_modification_allowed": False,
        "nodes": list(nodes.values()),
        "edges": edges,
        "decision_paths": decision_paths,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "path_count": len(decision_paths),
            "unique_paths": len({x.get('path_id') for x in decision_paths}),
        },
    }


def has_observed_decision_path(graph_payload: dict[str, Any], path_components: list[str]) -> bool:
    digest = hashlib.sha1("|".join(path_components).encode("utf-8")).hexdigest()[:16]
    target = f"DP-{digest}"
    for row in graph_payload.get("decision_paths", []):
        if str(row.get("path_id")) == target:
            return True
    return False


def write_institutional_knowledge_graph_artifacts(root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    storage = root_dir / "storage" / "olympus"
    storage.mkdir(parents=True, exist_ok=True)

    runtime_path = storage / "institutional_knowledge_graph.json"
    history_path = storage / "institutional_knowledge_graph_history.jsonl"

    runtime_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {
        "knowledge_graph": str(runtime_path),
        "knowledge_graph_history": str(history_path),
    }
