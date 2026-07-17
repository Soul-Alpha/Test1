from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from olympus.core.hermes_analytics import build_hermes_analytics


def compute_learning_statistics_from_root(root_dir: Path) -> Dict[str, Any]:
    """Compute additive Hermes learning analytics from existing historical/runtime data.

    This function is non-destructive and read-only with respect to historical records.
    """
    return build_hermes_analytics(root_dir)
