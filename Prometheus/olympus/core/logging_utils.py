from __future__ import annotations

import logging


def get_olympus_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"olympus.{name}")
    return logger
