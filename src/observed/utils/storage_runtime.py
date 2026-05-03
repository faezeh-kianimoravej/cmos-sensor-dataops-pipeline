"""Shared runtime helpers for storage environment and model version resolution."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from src.observed.utils.model_reference import resolve_model_reference
from src.observed.utils.storage_paths import StoragePathBuilder


def resolve_storage_environment(builder: StoragePathBuilder | None = None) -> str:
    """Resolve the active storage environment with a stable default."""

    active_builder = builder or StoragePathBuilder.from_default_config()
    return (
        str(os.environ.get("OBSERVED_STORAGE_ENV", "")).strip()
        or active_builder.default_environment
    )


def stage_model_version(stage: str) -> str:
    """Resolve the runtime model version for a stage, falling back to a UTC timestamp."""

    reference = resolve_model_reference(stage)
    if reference.version:
        return reference.version
    return f"model-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def stage_dataset_version() -> str:
    """Resolve a deterministic dataset version suffix for storage publishing."""

    return f"data-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


__all__ = [
    "resolve_storage_environment",
    "stage_dataset_version",
    "stage_model_version",
]
