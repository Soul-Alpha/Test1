"""Prometheus package init — exposes CONFIG for convenience."""
from config import CONFIG, setup_logging

__all__ = ["CONFIG", "setup_logging"]
