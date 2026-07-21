"""Prometheus Trading Command Center — consolidated lazy dashboard entrypoint."""
from __future__ import annotations

from ui.command_center_navigation import prometheus_pages, run_command_center

run_command_center(prometheus_pages())
