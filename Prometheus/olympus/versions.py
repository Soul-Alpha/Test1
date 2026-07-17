"""Olympus version registry constants.

Additive-only metadata constants used for event traceability.
"""
from __future__ import annotations

# Shared feature schema/version consumed by Olympus components.
FEATURE_VERSION = "olympus-feature-v1"

# System-scoped strategy/model versions.
HERMES_STRATEGY_VERSION = "hermes-strategy-v1"
HERMES_DATASET_GENERATION = "hermes-gen1"
HERMES_MODEL_FAMILY = "hermes-pattern-learner"

PROMETHEUS_STRATEGY_VERSION = "prometheus-strategy-v1"
PROMETHEUS_DATASET_GENERATION = "prometheus-gen1"
PROMETHEUS_MODEL_FAMILY = "prometheus-core"
