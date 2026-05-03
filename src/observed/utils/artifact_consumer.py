"""Small shared helpers for consuming published project artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.observed.utils.model_manifest import parse_model_bundle_manifest
from src.observed.utils.model_reference import resolve_model_reference
from src.observed.utils.object_storage import (
    download_from_storage_target,
    read_text_from_storage_target,
)
from src.observed.utils.storage_access import (
    resolve_local_vs_object_storage,
    resolve_manifest_vs_legacy_fallback,
)
from src.observed.utils.storage_paths import StoragePathBuilder
from src.observed.utils.storage_runtime import resolve_storage_environment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundleAccessPlan:
    """Resolved local-vs-storage decision for an inference model bundle."""

    source: str
    reason: str
    stage: str
    model_name: str
    model_version: str | None
    storage_environment: str | None
    local_model_path: Path
    local_metadata_path: Path
    model_target: dict[str, str] | None = None
    metadata_target: dict[str, str] | None = None
    manifest_target: dict[str, str] | None = None
    manifest_strategy: str | None = None


def resolve_model_bundle_access(
    *,
    stage: str,
    model_name: str,
    local_model_path: str | Path,
    local_metadata_path: str | Path,
    fallback_env_var: str = "OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK",
    legacy_fallback_env_var: str = "OBSERVED_ENABLE_LEGACY_MODEL_PATH_FALLBACK",
) -> ModelBundleAccessPlan:
    """Resolve the best available source for a stage model bundle."""

    model_path = Path(local_model_path)
    metadata_path = Path(local_metadata_path)

    decision = resolve_local_vs_object_storage(
        local_paths=[model_path, metadata_path],
        fallback_env_var=fallback_env_var,
    )
    if decision.source == "local":
        return ModelBundleAccessPlan(
            source="local",
            reason=decision.reason,
            stage=stage,
            model_name=model_name,
            model_version=None,
            storage_environment=None,
            local_model_path=model_path,
            local_metadata_path=metadata_path,
        )
    if decision.source != "object_storage":
        return ModelBundleAccessPlan(
            source="unavailable",
            reason=decision.reason,
            stage=stage,
            model_name=model_name,
            model_version=None,
            storage_environment=None,
            local_model_path=model_path,
            local_metadata_path=metadata_path,
        )

    reference = resolve_model_reference(stage)
    if not reference.version:
        return ModelBundleAccessPlan(
            source="unavailable",
            reason=f"object storage fallback is enabled for {stage} but no model version was resolved",
            stage=stage,
            model_name=model_name,
            model_version=None,
            storage_environment=None,
            local_model_path=model_path,
            local_metadata_path=metadata_path,
        )

    storage_env = resolve_storage_environment()
    default_targets = _default_model_bundle_targets(
        stage=stage,
        model_name=model_name,
        model_version=reference.version,
        storage_env=storage_env,
    )
    resolved_targets = _manifest_first_bundle_targets(
        stage=stage,
        model_version=reference.version,
        default_targets=default_targets,
        legacy_fallback_env_var=legacy_fallback_env_var,
    )
    if resolved_targets is None:
        plan = ModelBundleAccessPlan(
            source="unavailable",
            reason="model bundle manifest could not be used and legacy path fallback is disabled",
            stage=stage,
            model_name=model_name,
            model_version=reference.version,
            storage_environment=storage_env,
            local_model_path=model_path,
            local_metadata_path=metadata_path,
            manifest_target=default_targets["manifest"],
            manifest_strategy="unavailable",
        )
        LOGGER.warning(
            "Model bundle access is unavailable for stage '%s': %s", stage, plan.reason
        )
        return plan

    plan = ModelBundleAccessPlan(
        source="object_storage",
        reason=resolved_targets["reason"],
        stage=stage,
        model_name=model_name,
        model_version=reference.version,
        storage_environment=storage_env,
        local_model_path=model_path,
        local_metadata_path=metadata_path,
        model_target=resolved_targets["model_target"],
        metadata_target=resolved_targets["metadata_target"],
        manifest_target=default_targets["manifest"],
        manifest_strategy=resolved_targets["strategy"],
    )
    LOGGER.info(
        "Model bundle access resolved for stage '%s': source=%s, strategy=%s, version=%s",
        stage,
        plan.source,
        plan.manifest_strategy,
        plan.model_version,
    )
    return plan


def materialize_model_bundle_access_plan(
    plan: ModelBundleAccessPlan,
) -> tuple[Path, Path] | None:
    """Ensure the resolved bundle exists locally for the loader to consume."""

    if plan.source == "local":
        LOGGER.info("Using local model bundle for stage '%s'.", plan.stage)
        return plan.local_model_path, plan.local_metadata_path
    if (
        plan.source != "object_storage"
        or plan.model_target is None
        or plan.metadata_target is None
    ):
        LOGGER.warning(
            "Model bundle materialization skipped for stage '%s': %s",
            plan.stage,
            plan.reason,
        )
        return None

    plan.local_model_path.parent.mkdir(parents=True, exist_ok=True)
    model_download = download_from_storage_target(
        plan.model_target,
        local_path=plan.local_model_path,
    )
    metadata_download = download_from_storage_target(
        plan.metadata_target,
        local_path=plan.local_metadata_path,
    )
    if (
        model_download.status == "downloaded"
        and metadata_download.status == "downloaded"
    ):
        LOGGER.info(
            "Model bundle materialized from object storage for stage '%s' using %s strategy.",
            plan.stage,
            plan.manifest_strategy,
        )
        return plan.local_model_path, plan.local_metadata_path
    LOGGER.warning(
        "Model bundle materialization for stage '%s' did not complete: model=%s metadata=%s",
        plan.stage,
        model_download.status,
        metadata_download.status,
    )
    return None


def _default_model_bundle_targets(
    *,
    stage: str,
    model_name: str,
    model_version: str,
    storage_env: str,
) -> dict[str, dict[str, str]]:
    builder = StoragePathBuilder.from_default_config()
    model_target = builder.model_path(
        env=storage_env,
        model_family=stage,
        model_name=model_name,
        model_version=model_version,
        filename="model.joblib",
    )
    metadata_target = builder.model_path(
        env=storage_env,
        model_family=stage,
        model_name=model_name,
        model_version=model_version,
        filename="metadata.json",
    )
    manifest_target = builder.model_path(
        env=storage_env,
        model_family=stage,
        model_name=model_name,
        model_version=model_version,
        filename="bundle-manifest.json",
    )
    return {
        "model": _storage_target_dict(
            model_target, storage_env=storage_env, model_version=model_version
        ),
        "metadata": _storage_target_dict(
            metadata_target, storage_env=storage_env, model_version=model_version
        ),
        "manifest": _storage_target_dict(
            manifest_target, storage_env=storage_env, model_version=model_version
        ),
    }


def _manifest_first_bundle_targets(
    *,
    stage: str,
    model_version: str,
    default_targets: dict[str, dict[str, str]],
    legacy_fallback_env_var: str,
) -> dict[str, Any] | None:
    manifest_text, manifest_read = read_text_from_storage_target(
        default_targets["manifest"]
    )
    manifest_state = "missing"
    manifest = None
    if manifest_read.status == "read" and manifest_text:
        try:
            manifest = parse_model_bundle_manifest(manifest_text)
            manifest_state = "valid"
        except Exception:
            manifest_state = "invalid"

    if manifest is not None:
        if (
            manifest.stage.strip().lower() != stage
            or manifest.model_version.strip() != model_version
        ):
            manifest = None
            manifest_state = "invalid"

    strategy = resolve_manifest_vs_legacy_fallback(
        manifest_state=manifest_state,
        legacy_fallback_env_var=legacy_fallback_env_var,
    )
    if strategy.source == "manifest" and manifest is not None:
        return {
            "strategy": "manifest",
            "reason": strategy.reason,
            "model_target": dict(manifest.model_artifact),
            "metadata_target": dict(manifest.metadata_artifact),
        }
    if strategy.source == "legacy":
        return {
            "strategy": "legacy",
            "reason": strategy.reason,
            "model_target": dict(default_targets["model"]),
            "metadata_target": dict(default_targets["metadata"]),
        }
    return None


def _storage_target_dict(
    location: Any, *, storage_env: str, model_version: str
) -> dict[str, str]:
    return {
        "environment": storage_env,
        "model_version": model_version,
        "bucket": location.bucket,
        "prefix": location.prefix,
        "key": location.key,
        "uri": location.uri,
    }


__all__ = [
    "ModelBundleAccessPlan",
    "materialize_model_bundle_access_plan",
    "resolve_model_bundle_access",
]
