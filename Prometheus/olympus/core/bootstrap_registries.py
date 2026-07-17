from __future__ import annotations

from pathlib import Path

from olympus.core.model_registry import ModelRegistry
from olympus.core.version_registry import VersionRegistry
from olympus.versions import (
    FEATURE_VERSION,
    HERMES_DATASET_GENERATION,
    HERMES_MODEL_FAMILY,
    HERMES_STRATEGY_VERSION,
    PROMETHEUS_DATASET_GENERATION,
    PROMETHEUS_MODEL_FAMILY,
    PROMETHEUS_STRATEGY_VERSION,
)


def register_existing_models(root_dir: Path) -> None:
    """Registers existing model artifacts without moving or modifying them."""
    mr = ModelRegistry(root_dir)
    vr = VersionRegistry(root_dir)

    hermes_model = root_dir / "models" / "hermes" / "ml_model.pkl"
    prometheus_model = root_dir / "models" / "advanced_ml_models.pkl"

    if hermes_model.exists():
        mr.register_simple(
            model_name=HERMES_MODEL_FAMILY,
            system="hermes",
            version="existing",
            training_record_count=0,
            feature_version=FEATURE_VERSION,
            dataset_generation=HERMES_DATASET_GENERATION,
            description="Existing Hermes model artifact discovered",
            status="registered",
            metadata={"path": str(hermes_model)},
        )
        vr.register_simple(
            system="hermes",
            model_version="existing",
            feature_version=FEATURE_VERSION,
            strategy_version=HERMES_STRATEGY_VERSION,
            dataset_generation=HERMES_DATASET_GENERATION,
            record_count=0,
            active=False,
            metadata={"path": str(hermes_model)},
        )

    if prometheus_model.exists():
        mr.register_simple(
            model_name=PROMETHEUS_MODEL_FAMILY,
            system="prometheus",
            version="existing",
            training_record_count=0,
            feature_version=FEATURE_VERSION,
            dataset_generation=PROMETHEUS_DATASET_GENERATION,
            description="Existing Prometheus model artifact discovered",
            status="registered",
            metadata={"path": str(prometheus_model)},
        )
        vr.register_simple(
            system="prometheus",
            model_version="existing",
            feature_version=FEATURE_VERSION,
            strategy_version=PROMETHEUS_STRATEGY_VERSION,
            dataset_generation=PROMETHEUS_DATASET_GENERATION,
            record_count=0,
            active=False,
            metadata={"path": str(prometheus_model)},
        )
