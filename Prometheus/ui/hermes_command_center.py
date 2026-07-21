"""Hermes Execution and Learning Center — consolidated lazy entrypoint."""
from __future__ import annotations

from ui.command_center_navigation import hermes_pages, run_command_center

run_command_center(hermes_pages())
