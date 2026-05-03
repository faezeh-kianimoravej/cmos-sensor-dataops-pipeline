from __future__ import annotations

from pathlib import Path

from src.observed.utils.object_storage import upload_file_to_object_storage


def test_upload_file_to_object_storage_returns_skipped_when_storage_is_unconfigured(
    monkeypatch,
) -> None:
    sample_path = Path("artifacts/test_tmp/object-storage-skip.txt")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("sample", encoding="utf-8")

    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = upload_file_to_object_storage(
        local_path=sample_path,
        bucket="reports",
        key="dev/stage1/evaluation/manual-1/report.json",
        uri="s3://reports/dev/stage1/evaluation/manual-1/report.json",
    )

    assert result.status == "skipped"
    assert result.bucket == "reports"
