"""Model loading and caching helpers for stage-1/stage-2 API inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from src.observed.utils.artifact_consumer import (
    materialize_model_bundle_access_plan,
    resolve_model_bundle_access,
)

LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Load and cache stage model bundles with local-first, storage-fallback behavior."""

    _STAGE_DIRS = {
        "stage1": "stage1_mixture_detector",
        "stage2": "stage2_gas_probability_model",
    }
    _STAGE_MODEL_NAMES = {
        "stage1": "stage1-mixture-detector",
        "stage2": "stage2-gas-probability-model",
    }

    def __init__(self, models_dir: Path = Path("artifacts/models")):
        self.models_dir = Path(models_dir)
        self._cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    def _bundle_paths(self, stage: str) -> tuple[Path, Path]:
        stage_dir_name = self._STAGE_DIRS[stage]
        stage_dir = self.models_dir / stage_dir_name
        return stage_dir / "model.joblib", stage_dir / "metadata.json"

    def _ensure_local_bundle_from_object_storage(
        self, stage: str
    ) -> tuple[Path, Path] | None:
        model_path, metadata_path = self._bundle_paths(stage)
        access_plan = resolve_model_bundle_access(
            stage=stage,
            model_name=self._STAGE_MODEL_NAMES[stage],
            local_model_path=model_path,
            local_metadata_path=metadata_path,
        )
        LOGGER.info(
            "Model loader access resolution for stage '%s': %s",
            stage,
            access_plan.reason,
        )
        if access_plan.source == "local":
            return model_path, metadata_path
        if access_plan.source != "object_storage":
            LOGGER.warning(
                "Model loader could not resolve stage '%s': %s",
                stage,
                access_plan.reason,
            )
            return None
        try:
            return materialize_model_bundle_access_plan(access_plan)
        except Exception:
            LOGGER.exception(
                "Model loader failed to materialize stage '%s' from object storage.",
                stage,
            )
            return None

    def load_stage_bundle(self, stage: str) -> tuple[Any, dict[str, Any]] | None:
        if stage in self._cache:
            return self._cache[stage]

        if stage not in self._STAGE_DIRS:
            return None

        model_path, metadata_path = self._bundle_paths(stage)
        if not model_path.exists() or not metadata_path.exists():
            downloaded = self._ensure_local_bundle_from_object_storage(stage)
            if downloaded is None:
                return None
            model_path, metadata_path = downloaded

        try:
            model = joblib.load(model_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._cache[stage] = (model, metadata)
            LOGGER.info(
                "Model bundle loaded for stage '%s' from '%s'.",
                stage,
                model_path.parent,
            )
            return model, metadata
        except Exception:
            LOGGER.exception(
                "Failed to load model bundle for stage '%s' from '%s' and '%s'.",
                stage,
                model_path,
                metadata_path,
            )
            return None

    def load_stage_model(self, stage: str) -> Any | None:
        bundle = self.load_stage_bundle(stage)
        if bundle is None:
            return None
        return bundle[0]

    def load_stage_metadata(self, stage: str) -> dict[str, Any]:
        bundle = self.load_stage_bundle(stage)
        if bundle is None:
            return {}
        return bundle[1]

    def list_available_models(self) -> list[str]:
        available: list[str] = []
        for stage in ["stage1", "stage2"]:
            if self.load_stage_bundle(stage) is not None:
                available.append(stage)
        return available

    def model_artifact_paths(self) -> dict[str, dict[str, str]]:
        info: dict[str, dict[str, str]] = {}
        for stage in ["stage1", "stage2"]:
            model_path, metadata_path = self._bundle_paths(stage)
            info[stage] = {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "manifest_path": str(model_path.parent / "bundle-manifest.json"),
            }
        return info
