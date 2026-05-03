from __future__ import annotations

from pathlib import Path

import pytest

from src.observed.utils.object_storage import (
    download_file_from_object_storage,
    download_from_storage_target,
    read_text_from_object_storage,
    read_text_from_storage_target,
)

pytestmark = pytest.mark.unit


def test_download_file_from_object_storage_skips_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    local_path = tmp_path / "report.json"
    result = download_file_from_object_storage(
        bucket="reports",
        key="dev/evaluation/stage1/evaluation-report/run/report.json",
        uri="s3://reports/dev/evaluation/stage1/evaluation-report/run/report.json",
        local_path=local_path,
    )

    assert result.status == "skipped"
    assert result.local_path == str(local_path)


def test_download_from_storage_target_reuses_bucket_key_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    storage_target = {
        "bucket": "batch-predictions",
        "key": "dev/batch-inference/2026/04/07/job/predictions.parquet",
        "uri": "s3://batch-predictions/dev/batch-inference/2026/04/07/job/predictions.parquet",
    }
    result = download_from_storage_target(
        storage_target, local_path=tmp_path / "predictions.parquet"
    )

    assert result.status == "skipped"
    assert result.bucket == "batch-predictions"
    assert result.key.endswith("predictions.parquet")


def test_read_text_from_object_storage_skips_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    text, result = read_text_from_object_storage(
        bucket="reports",
        key="dev/evaluation/stage2/evaluation-report/run/report.json",
        uri="s3://reports/dev/evaluation/stage2/evaluation-report/run/report.json",
    )

    assert text is None
    assert result.status == "skipped"


def test_read_text_from_storage_target_reuses_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    storage_target = {
        "bucket": "reports",
        "key": "dev/evaluation/stage1/evaluation-report/run/report.json",
        "uri": "s3://reports/dev/evaluation/stage1/evaluation-report/run/report.json",
    }
    text, result = read_text_from_storage_target(storage_target)

    assert text is None
    assert result.status == "skipped"
