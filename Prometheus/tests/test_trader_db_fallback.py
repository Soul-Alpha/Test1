"""Regression coverage for the Prometheus database-analysis fallback."""
from __future__ import annotations

import ast
from pathlib import Path


TRADER_PATH = Path(__file__).resolve().parents[1] / "live_bot" / "trader.py"


def _trader_tree() -> ast.Module:
    return ast.parse(TRADER_PATH.read_text(encoding="utf-8-sig"))


def test_list_analyses_is_imported_from_storage_database() -> None:
    imported_names: set[str] = set()

    for node in ast.walk(_trader_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "storage.database":
            imported_names.update(alias.name for alias in node.names)

    assert "list_analyses" in imported_names


def test_optional_import_failure_defines_list_analyses_fallback() -> None:
    assigned_names = {
        target.id
        for node in ast.walk(_trader_tree())
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    }

    assert "list_analyses" in assigned_names
