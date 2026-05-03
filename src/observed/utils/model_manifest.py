"""Helpers for serializing and consuming published model bundle manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelBundleManifest:
    stage: str
    model_name: str
    model_version: str
    storage_environment: str
    published_at: str
    model_artifact: dict[str, str]
    metadata_artifact: dict[str, str]
    source: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "storage_environment": self.storage_environment,
            "published_at": self.published_at,
            "artifacts": {
                "model": dict(self.model_artifact),
                "metadata": dict(self.metadata_artifact),
            },
        }
        if self.source:
            payload["source"] = dict(self.source)
        return payload


def build_model_bundle_manifest(
    *,
    stage: str,
    model_name: str,
    model_version: str,
    storage_environment: str,
    model_artifact: dict[str, str],
    metadata_artifact: dict[str, str],
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = ModelBundleManifest(
        stage=stage,
        model_name=model_name,
        model_version=model_version,
        storage_environment=storage_environment,
        published_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_artifact=dict(model_artifact),
        metadata_artifact=dict(metadata_artifact),
        source=dict(source_info) if source_info else None,
    )
    return manifest.to_dict()


def parse_model_bundle_manifest(
    payload: str | bytes | dict[str, Any],
) -> ModelBundleManifest:
    if isinstance(payload, bytes):
        raw_payload: Any = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        raw_payload = json.loads(payload)
    else:
        raw_payload = payload

    if not isinstance(raw_payload, dict):
        raise ValueError("Model bundle manifest payload must be a JSON object.")

    artifacts = raw_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Model bundle manifest must contain an 'artifacts' object.")

    model_artifact = _validated_artifact_target(
        artifacts.get("model"), artifact_name="model"
    )
    metadata_artifact = _validated_artifact_target(
        artifacts.get("metadata"), artifact_name="metadata"
    )

    return ModelBundleManifest(
        stage=_required_non_empty_string(raw_payload.get("stage"), field_name="stage"),
        model_name=_required_non_empty_string(
            raw_payload.get("model_name"), field_name="model_name"
        ),
        model_version=_required_non_empty_string(
            raw_payload.get("model_version"), field_name="model_version"
        ),
        storage_environment=_required_non_empty_string(
            raw_payload.get("storage_environment"),
            field_name="storage_environment",
        ),
        published_at=_required_non_empty_string(
            raw_payload.get("published_at"), field_name="published_at"
        ),
        model_artifact=model_artifact,
        metadata_artifact=metadata_artifact,
        source=(
            dict(raw_payload["source"])
            if isinstance(raw_payload.get("source"), dict)
            else None
        ),
    )


def write_model_bundle_manifest(
    manifest: dict[str, Any], destination: str | Path
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _required_non_empty_string(value: Any, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(
            f"Model bundle manifest field '{field_name}' must be a non-empty string."
        )
    return normalized


def _validated_artifact_target(value: Any, *, artifact_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Model bundle manifest artifact '{artifact_name}' must be an object."
        )

    validated: dict[str, str] = {}
    for key in ("bucket", "key", "uri"):
        validated[key] = _required_non_empty_string(
            value.get(key), field_name=f"artifacts.{artifact_name}.{key}"
        )
    return validated


__all__ = [
    "ModelBundleManifest",
    "build_model_bundle_manifest",
    "parse_model_bundle_manifest",
    "write_model_bundle_manifest",
]
