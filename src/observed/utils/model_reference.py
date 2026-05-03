"""Small helpers for resolving inference model references from env/config."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelReference:
    stage: str
    version: str | None
    source: str


def resolve_model_reference(stage: str) -> ModelReference:
    """Resolve model version/reference for a stage using env precedence.

    Precedence:
    1. ``OBSERVED_<STAGE>_MODEL_VERSION``
    2. ``OBSERVED_MODEL_VERSION``
    3. no configured version
    """

    normalized_stage = str(stage).strip().lower()
    stage_specific_var = f"OBSERVED_{normalized_stage.upper()}_MODEL_VERSION"

    stage_specific = str(os.environ.get(stage_specific_var, "")).strip()
    if stage_specific:
        return ModelReference(
            stage=normalized_stage,
            version=stage_specific,
            source=stage_specific_var,
        )

    shared = str(os.environ.get("OBSERVED_MODEL_VERSION", "")).strip()
    if shared:
        return ModelReference(
            stage=normalized_stage,
            version=shared,
            source="OBSERVED_MODEL_VERSION",
        )

    return ModelReference(
        stage=normalized_stage,
        version=None,
        source="unconfigured",
    )


__all__ = ["ModelReference", "resolve_model_reference"]
