"""Object storage path helpers for MinIO/S3-compatible naming conventions.

This module is intentionally thin:

- loads the storage layout from ``configs/storage_layout.yaml``
- validates environment and required template fields
- builds deterministic bucket/prefix/key values

It does not perform any upload/download operations and has no storage client
dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

from src.observed.utils.config import get_repo_root


@dataclass(frozen=True)
class ObjectStoragePath:
    """Deterministic object storage location."""

    bucket: str
    prefix: str
    filename: str | None = None

    @property
    def key(self) -> str:
        if self.filename:
            return f"{self.prefix}{self.filename}"
        return self.prefix

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class StoragePathError(ValueError):
    """Raised when object storage path inputs are invalid."""


class StoragePathBuilder:
    """Build object storage paths from ``configs/storage_layout.yaml``."""

    _DOMAIN_TO_PREFIX_KEY = {
        "datasets": "datasets",
        "reports": "reports",
        "models": "models",
        "batch_predictions": "batch_predictions",
    }

    def __init__(self, layout_path: str | Path | None = None) -> None:
        self.layout_path = (
            Path(layout_path)
            if layout_path is not None
            else get_repo_root() / "configs" / "storage_layout.yaml"
        )
        self._layout = self._load_layout(self.layout_path)
        storage_cfg = self._layout["storage"]
        self._buckets = dict(storage_cfg["buckets"])
        self._prefixes = dict(storage_cfg["prefixes"])
        self._naming_rules = dict(storage_cfg["naming_rules"])
        self._local_mapping = dict(storage_cfg.get("local_to_object_mapping", {}))

    @property
    def valid_environments(self) -> tuple[str, ...]:
        values = self._naming_rules.get("env_values", [])
        return tuple(str(v) for v in values)

    @property
    def default_environment(self) -> str:
        return str(self._layout["storage"].get("default_environment", "dev"))

    def dataset_path(
        self,
        *,
        env: str | None = None,
        pipeline: str,
        dataset_name: str,
        dataset_version: str,
        filename: str | None = None,
    ) -> ObjectStoragePath:
        return self._build_for_domain(
            "datasets",
            env=env,
            filename=filename,
            pipeline=pipeline,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
        )

    def report_path(
        self,
        *,
        env: str | None = None,
        pipeline: str,
        report_family: str,
        run_id: str,
        filename: str | None = None,
    ) -> ObjectStoragePath:
        return self._build_for_domain(
            "reports",
            env=env,
            filename=filename,
            pipeline=pipeline,
            report_family=report_family,
            run_id=run_id,
        )

    def model_path(
        self,
        *,
        env: str | None = None,
        model_family: str,
        model_name: str,
        model_version: str,
        filename: str | None = None,
    ) -> ObjectStoragePath:
        return self._build_for_domain(
            "models",
            env=env,
            filename=filename,
            model_family=model_family,
            model_name=model_name,
            model_version=model_version,
        )

    def batch_prediction_path(
        self,
        *,
        env: str | None = None,
        prediction_job: str,
        job_date: str,
        job_id: str,
        filename: str | None = None,
    ) -> ObjectStoragePath:
        return self._build_for_domain(
            "batch_predictions",
            env=env,
            filename=filename,
            prediction_job=prediction_job,
            job_date=job_date,
            job_id=job_id,
        )

    def path_for_local_output(
        self,
        local_path: str | Path,
        *,
        env: str | None = None,
        filename: str | None = None,
        **fields: str,
    ) -> ObjectStoragePath:
        normalized = self._normalize_local_path(str(local_path))
        mapping = self._local_mapping.get(normalized)
        if mapping is None:
            raise StoragePathError(
                f"No object storage mapping is defined for local path: {normalized}"
            )

        bucket_alias = str(mapping["bucket"])
        prefix_template = str(mapping["prefix"])
        bucket = self.bucket_name(bucket_alias)
        effective_env = self._validate_environment(env)
        prefix = self._format_prefix(prefix_template, {"env": effective_env, **fields})
        return ObjectStoragePath(
            bucket=bucket, prefix=prefix, filename=self._normalize_filename(filename)
        )

    def bucket_name(self, alias: str) -> str:
        override_key = f"OBSERVED_STORAGE_BUCKET_{str(alias).strip().upper()}"
        override_value = str(os.environ.get(override_key, "")).strip()
        if override_value:
            return override_value
        try:
            return str(self._buckets[alias])
        except KeyError as exc:
            known = ", ".join(sorted(self._buckets))
            raise StoragePathError(
                f"Unknown storage bucket alias '{alias}'. Known aliases: {known}"
            ) from exc

    @classmethod
    def from_default_config(cls) -> "StoragePathBuilder":
        return cls()

    @staticmethod
    def _load_layout(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Storage layout config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict) or "storage" not in loaded:
            raise StoragePathError(f"Invalid storage layout config: {path}")
        return loaded

    def _build_for_domain(
        self, domain: str, *, env: str | None, filename: str | None, **fields: str
    ) -> ObjectStoragePath:
        prefix_key = self._DOMAIN_TO_PREFIX_KEY[domain]
        bucket = self.bucket_name(domain)
        effective_env = self._validate_environment(env)
        prefix_template = str(self._prefixes[prefix_key])
        prefix = self._format_prefix(prefix_template, {"env": effective_env, **fields})
        return ObjectStoragePath(
            bucket=bucket,
            prefix=prefix,
            filename=self._normalize_filename(filename),
        )

    def _validate_environment(self, env: str | None) -> str:
        effective_env = str(env or self.default_environment).strip()
        if effective_env not in self.valid_environments:
            allowed = ", ".join(self.valid_environments)
            raise StoragePathError(
                f"Invalid storage environment '{effective_env}'. Allowed values: {allowed}"
            )
        return effective_env

    def _format_prefix(self, template: str, values: dict[str, str]) -> str:
        required_fields = self._template_fields(template)
        missing = [
            name for name in required_fields if not str(values.get(name, "")).strip()
        ]
        if missing:
            raise StoragePathError(
                f"Missing required storage path fields for template '{template}': {', '.join(sorted(missing))}"
            )

        normalized_values = {
            key: self._normalize_segment(value) for key, value in values.items()
        }
        prefix = template.format(**normalized_values)
        return self._normalize_prefix(prefix)

    @staticmethod
    def _template_fields(template: str) -> set[str]:
        fields: set[str] = set()
        for _, field_name, _, _ in Formatter().parse(template):
            if field_name:
                fields.add(field_name)
        return fields

    @staticmethod
    def _normalize_segment(value: Any) -> str:
        text = str(value).strip().strip("/")
        if not text:
            raise StoragePathError("Storage path segments must not be empty")
        return text

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        cleaned = prefix.strip().strip("/")
        if not cleaned:
            raise StoragePathError("Storage path prefix must not be empty")
        return f"{cleaned}/"

    @staticmethod
    def _normalize_filename(filename: str | None) -> str | None:
        if filename is None:
            return None
        cleaned = str(filename).strip().strip("/")
        if not cleaned:
            raise StoragePathError("Filename must not be empty")
        return cleaned

    @staticmethod
    def _normalize_local_path(path: str) -> str:
        return path.replace("\\", "/").strip()


__all__ = ["ObjectStoragePath", "StoragePathBuilder", "StoragePathError"]
