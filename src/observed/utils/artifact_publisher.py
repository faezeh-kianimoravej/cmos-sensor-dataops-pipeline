"""Small shared helpers for publishing project artifacts to object storage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.observed.utils.model_manifest import (
    build_model_bundle_manifest,
    write_model_bundle_manifest,
)
from src.observed.utils.object_storage import upload_file_to_object_storage
from src.observed.utils.storage_paths import StoragePathBuilder
from src.observed.utils.storage_runtime import (
    resolve_storage_environment,
    stage_model_version,
)

LOGGER = logging.getLogger(__name__)


def build_storage_target(
    *,
    local_path: str | Path,
    domain: str,
    env: str | None = None,
    builder: StoragePathBuilder | None = None,
    **fields: str,
) -> dict[str, str]:
    """Build a standardized ``storage_target`` payload for a local artifact."""

    active_builder = builder or StoragePathBuilder.from_default_config()
    resolved_env = env or resolve_storage_environment(active_builder)
    artifact_path = Path(local_path)
    filename = artifact_path.name

    if domain == "reports":
        location = active_builder.report_path(
            env=resolved_env,
            pipeline=str(fields["pipeline"]),
            report_family=str(fields["report_family"]),
            run_id=str(fields["run_id"]),
            filename=filename,
        )
        metadata: dict[str, str] = {
            "environment": resolved_env,
            "pipeline": str(fields["pipeline"]),
            "report_family": str(fields["report_family"]),
            "run_id": str(fields["run_id"]),
        }
    elif domain == "batch_predictions":
        location = active_builder.batch_prediction_path(
            env=resolved_env,
            prediction_job=str(fields["prediction_job"]),
            job_date=str(fields["job_date"]),
            job_id=str(fields["job_id"]),
            filename=filename,
        )
        metadata = {
            "environment": resolved_env,
            "prediction_job": str(fields["prediction_job"]),
            "job_date": str(fields["job_date"]),
            "job_id": str(fields["job_id"]),
        }
    elif domain == "models":
        location = active_builder.model_path(
            env=resolved_env,
            model_family=str(fields["model_family"]),
            model_name=str(fields["model_name"]),
            model_version=str(fields["model_version"]),
            filename=filename,
        )
        metadata = {
            "environment": resolved_env,
            "model_family": str(fields["model_family"]),
            "model_name": str(fields["model_name"]),
            "model_version": str(fields["model_version"]),
        }
    elif domain == "datasets":
        location = active_builder.dataset_path(
            env=resolved_env,
            pipeline=str(fields["pipeline"]),
            dataset_name=str(fields["dataset_name"]),
            dataset_version=str(fields["dataset_version"]),
            filename=filename,
        )
        metadata = {
            "environment": resolved_env,
            "pipeline": str(fields["pipeline"]),
            "dataset_name": str(fields["dataset_name"]),
            "dataset_version": str(fields["dataset_version"]),
        }
    else:
        raise ValueError(f"Unsupported artifact publish domain: {domain}")

    return {
        **metadata,
        "bucket": location.bucket,
        "prefix": location.prefix,
        "key": location.key,
        "uri": location.uri,
    }


def publish_local_artifact(
    *,
    local_path: str | Path,
    domain: str,
    env: str | None = None,
    builder: StoragePathBuilder | None = None,
    **fields: str,
) -> dict[str, Any]:
    """Publish one local artifact and return normalized storage metadata."""

    source = Path(local_path)
    storage_target = build_storage_target(
        local_path=source,
        domain=domain,
        env=env,
        builder=builder,
        **fields,
    )
    upload_result = upload_file_to_object_storage(
        local_path=source,
        bucket=storage_target["bucket"],
        key=storage_target["key"],
        uri=storage_target["uri"],
    )
    published = {
        "local_path": str(source),
        "domain": domain,
        "storage_target": storage_target,
        "storage_upload": upload_result.to_dict(),
    }
    LOGGER.info(
        "Artifact publish result for '%s' [%s]: %s -> %s",
        source,
        domain,
        published["storage_upload"]["status"],
        storage_target["uri"],
    )
    return published


def publish_model_bundle(
    *,
    stage: str,
    model_name: str,
    model_dir: str | Path,
    model_version: str | None = None,
    storage_env: str | None = None,
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the stage model bundle and its manifest using the shared flow."""

    bundle_dir = Path(model_dir)
    active_builder = StoragePathBuilder.from_default_config()
    resolved_env = storage_env or resolve_storage_environment(active_builder)
    resolved_version = model_version or stage_model_version(stage)

    model_artifact = bundle_dir / "model.joblib"
    metadata_artifact = bundle_dir / "metadata.json"
    manifest_artifact = bundle_dir / "bundle-manifest.json"

    model_publish = publish_local_artifact(
        local_path=model_artifact,
        domain="models",
        env=resolved_env,
        builder=active_builder,
        model_family=stage,
        model_name=model_name,
        model_version=resolved_version,
    )
    metadata_publish = publish_local_artifact(
        local_path=metadata_artifact,
        domain="models",
        env=resolved_env,
        builder=active_builder,
        model_family=stage,
        model_name=model_name,
        model_version=resolved_version,
    )

    manifest = build_model_bundle_manifest(
        stage=stage,
        model_name=model_name,
        model_version=resolved_version,
        storage_environment=resolved_env,
        model_artifact=dict(model_publish["storage_target"]),
        metadata_artifact=dict(metadata_publish["storage_target"]),
        source_info=source_info,
    )
    write_model_bundle_manifest(manifest, manifest_artifact)

    manifest_publish = publish_local_artifact(
        local_path=manifest_artifact,
        domain="models",
        env=resolved_env,
        builder=active_builder,
        model_family=stage,
        model_name=model_name,
        model_version=resolved_version,
    )

    published_bundle = {
        "storage_environment": resolved_env,
        "model_version": resolved_version,
        "artifacts": {
            "model": model_publish,
            "metadata": metadata_publish,
            "manifest": manifest_publish,
        },
    }
    LOGGER.info(
        "Model bundle publish completed for stage '%s' version '%s' in environment '%s'.",
        stage,
        resolved_version,
        resolved_env,
    )
    return published_bundle


__all__ = [
    "build_storage_target",
    "publish_local_artifact",
    "publish_model_bundle",
    "resolve_storage_environment",
    "stage_model_version",
]
