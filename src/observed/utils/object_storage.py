"""Thin S3-compatible object storage upload/download helpers.

This module is intentionally small and side-effect free except for the explicit
network operations. It is designed for MinIO/S3-compatible access driven by env
vars.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ObjectStorageUploadError(RuntimeError):
    """Raised when an object storage upload is requested but cannot complete."""


class ObjectStorageDownloadError(RuntimeError):
    """Raised when an object storage download/read cannot complete."""


@dataclass(frozen=True)
class ObjectStorageUploadResult:
    status: str
    bucket: str
    key: str
    uri: str
    endpoint_url: str | None
    local_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectStorageDownloadResult:
    status: str
    bucket: str
    key: str
    uri: str
    endpoint_url: str | None
    local_path: str | None = None
    bytes_read: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolved_endpoint_url() -> str | None:
    candidates = [
        os.environ.get("OBSERVED_STORAGE_S3_ENDPOINT_URL"),
        os.environ.get("MINIO_PUBLIC_ENDPOINT"),
        os.environ.get("DVC_S3_ENDPOINT_URL"),
        os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def _resolved_storage_backend(endpoint_url: str | None) -> str:
    selected = str(os.environ.get("OBSERVED_STORAGE_BACKEND", "auto")).strip().lower()
    if selected in {"minio", "s3"}:
        return selected
    return "minio" if endpoint_url else "s3"


def _resolved_region_name() -> str | None:
    for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    return None


def _minio_credentials_present() -> bool:
    required = [
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
    ]
    return all(str(value or "").strip() for value in required)


def _aws_runtime_hint_present() -> bool:
    # Signals that runtime may use native AWS auth (env creds or role-based providers).
    hints = [
        os.environ.get("AWS_REGION"),
        os.environ.get("AWS_DEFAULT_REGION"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
        os.environ.get("AWS_PROFILE"),
        os.environ.get("AWS_ROLE_ARN"),
        os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"),
        os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"),
        os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI"),
    ]
    return any(str(value or "").strip() for value in hints)


def object_storage_access_configured() -> bool:
    endpoint_url = _resolved_endpoint_url()
    backend = _resolved_storage_backend(endpoint_url)

    if backend == "minio":
        return endpoint_url is not None and _minio_credentials_present()

    return _aws_runtime_hint_present()


def _storage_client():
    endpoint_url = _resolved_endpoint_url()
    backend = _resolved_storage_backend(endpoint_url)

    if backend == "minio" and (
        endpoint_url is None or not _minio_credentials_present()
    ):
        return None, endpoint_url
    if backend == "s3" and not _aws_runtime_hint_present():
        return None, endpoint_url

    addressing_style = str(
        os.environ.get(
            "AWS_S3_ADDRESSING_STYLE", "path" if backend == "minio" else "virtual"
        )
    ).strip() or ("path" if backend == "minio" else "virtual")
    region_name = _resolved_region_name() or (
        "us-east-1" if backend == "minio" else None
    )
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ObjectStorageUploadError(
            "boto3/botocore are required for object storage access. Install dependencies from requirements.txt."
        ) from exc

    client_kwargs: dict[str, Any] = {
        "service_name": "s3",
        "region_name": region_name,
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
        ),
    }
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    access_key = str(os.environ.get("AWS_ACCESS_KEY_ID", "")).strip()
    secret_key = str(os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip()
    session_token = str(os.environ.get("AWS_SESSION_TOKEN", "")).strip()
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            client_kwargs["aws_session_token"] = session_token

    client = boto3.client(**client_kwargs)
    return client, endpoint_url


def upload_file_to_object_storage(
    *,
    local_path: str | Path,
    bucket: str,
    key: str,
    uri: str,
) -> ObjectStorageUploadResult:
    """Upload a local file to S3-compatible storage when env config is present.

    Behavior:
    - If endpoint/credentials are not configured, returns ``status='skipped'``.
    - If config is present but upload fails, raises ``ObjectStorageUploadError``.
    """

    source = Path(local_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Local file for object storage upload was not found: {source}"
        )

    client, endpoint_url = _storage_client()
    if client is None:
        result = ObjectStorageUploadResult(
            status="skipped",
            bucket=bucket,
            key=key,
            uri=uri,
            endpoint_url=endpoint_url,
            local_path=str(source),
        )
        LOGGER.info(
            "Object storage upload skipped for '%s': endpoint or credentials are not fully configured.",
            source,
        )
        return result

    try:
        client.upload_file(str(source), bucket, key)
    except Exception as exc:
        raise ObjectStorageUploadError(
            f"Failed to upload '{source}' to object storage at s3://{bucket}/{key} via {endpoint_url}: {exc}"
        ) from exc

    result = ObjectStorageUploadResult(
        status="uploaded",
        bucket=bucket,
        key=key,
        uri=uri,
        endpoint_url=endpoint_url,
        local_path=str(source),
    )
    LOGGER.info("Object storage upload completed: '%s' -> %s", source, uri)
    return result


def download_file_from_object_storage(
    *,
    bucket: str,
    key: str,
    uri: str,
    local_path: str | Path,
) -> ObjectStorageDownloadResult:
    """Download an object to a local file when object storage config is present."""

    destination = Path(local_path)
    client, endpoint_url = _storage_client()
    if client is None:
        result = ObjectStorageDownloadResult(
            status="skipped",
            bucket=bucket,
            key=key,
            uri=uri,
            endpoint_url=endpoint_url,
            local_path=str(destination),
        )
        LOGGER.info(
            "Object storage download skipped for '%s': endpoint or credentials are not fully configured.",
            destination,
        )
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(destination))
    except Exception as exc:
        raise ObjectStorageDownloadError(
            f"Failed to download s3://{bucket}/{key} to '{destination}' via {endpoint_url}: {exc}"
        ) from exc

    size = destination.stat().st_size if destination.exists() else None
    result = ObjectStorageDownloadResult(
        status="downloaded",
        bucket=bucket,
        key=key,
        uri=uri,
        endpoint_url=endpoint_url,
        local_path=str(destination),
        bytes_read=size,
    )
    LOGGER.info("Object storage download completed: %s -> '%s'", uri, destination)
    return result


def read_bytes_from_object_storage(
    *,
    bucket: str,
    key: str,
    uri: str,
) -> tuple[bytes | None, ObjectStorageDownloadResult]:
    """Read a small object into memory as bytes when config is present."""

    client, endpoint_url = _storage_client()
    if client is None:
        result = ObjectStorageDownloadResult(
            status="skipped",
            bucket=bucket,
            key=key,
            uri=uri,
            endpoint_url=endpoint_url,
            local_path=None,
            bytes_read=None,
        )
        LOGGER.info(
            "Object storage read skipped for %s: endpoint or credentials are not fully configured.",
            uri,
        )
        return None, result

    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
    except Exception as exc:
        raise ObjectStorageDownloadError(
            f"Failed to read s3://{bucket}/{key} via {endpoint_url}: {exc}"
        ) from exc

    result = ObjectStorageDownloadResult(
        status="read",
        bucket=bucket,
        key=key,
        uri=uri,
        endpoint_url=endpoint_url,
        local_path=None,
        bytes_read=len(body),
    )
    LOGGER.info("Object storage read completed: %s (%s bytes)", uri, len(body))
    return body, result


def read_text_from_object_storage(
    *,
    bucket: str,
    key: str,
    uri: str,
    encoding: str = "utf-8",
) -> tuple[str | None, ObjectStorageDownloadResult]:
    """Read a small object into memory as decoded text when config is present."""

    payload, result = read_bytes_from_object_storage(bucket=bucket, key=key, uri=uri)
    if payload is None:
        return None, result
    return payload.decode(encoding), result


def download_from_storage_target(
    storage_target: dict[str, Any],
    *,
    local_path: str | Path,
) -> ObjectStorageDownloadResult:
    """Download an object using existing ``bucket/key/uri`` metadata."""

    return download_file_from_object_storage(
        bucket=str(storage_target["bucket"]),
        key=str(storage_target["key"]),
        uri=str(storage_target["uri"]),
        local_path=local_path,
    )


def read_text_from_storage_target(
    storage_target: dict[str, Any],
    *,
    encoding: str = "utf-8",
) -> tuple[str | None, ObjectStorageDownloadResult]:
    """Read text from an object using existing ``bucket/key/uri`` metadata."""

    return read_text_from_object_storage(
        bucket=str(storage_target["bucket"]),
        key=str(storage_target["key"]),
        uri=str(storage_target["uri"]),
        encoding=encoding,
    )


__all__ = [
    "ObjectStorageDownloadError",
    "ObjectStorageDownloadResult",
    "ObjectStorageUploadError",
    "ObjectStorageUploadResult",
    "download_file_from_object_storage",
    "download_from_storage_target",
    "object_storage_access_configured",
    "read_bytes_from_object_storage",
    "read_text_from_object_storage",
    "read_text_from_storage_target",
    "upload_file_to_object_storage",
]
