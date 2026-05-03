from __future__ import annotations

from pathlib import Path

from src.observed.utils.storage_access import (
    resolve_local_vs_object_storage,
    resolve_manifest_vs_legacy_fallback,
)


def test_policy_prefers_local_when_files_exist(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "model.joblib"
    second = tmp_path / "metadata.json"
    first.write_text("x", encoding="utf-8")
    second.write_text("y", encoding="utf-8")

    monkeypatch.delenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", raising=False)

    decision = resolve_local_vs_object_storage(
        local_paths=[first, second],
        fallback_env_var="OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK",
    )

    assert decision.source == "local"
    assert decision.local_available is True


def test_policy_reports_unavailable_when_local_missing_and_fallback_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", raising=False)

    decision = resolve_local_vs_object_storage(
        local_paths=[tmp_path / "missing.bin"],
        fallback_env_var="OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK",
    )

    assert decision.source == "unavailable"
    assert "disabled" in decision.reason


def test_policy_reports_unavailable_when_fallback_enabled_but_storage_not_configured(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", "true")
    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    decision = resolve_local_vs_object_storage(
        local_paths=[tmp_path / "missing.bin"],
        fallback_env_var="OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK",
    )

    assert decision.source == "unavailable"
    assert decision.storage_configured is False


def test_policy_selects_object_storage_when_local_missing_and_fallback_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", "true")
    monkeypatch.setenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    decision = resolve_local_vs_object_storage(
        local_paths=[tmp_path / "missing.bin"],
        fallback_env_var="OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK",
    )

    assert decision.source == "object_storage"
    assert decision.storage_configured is True


def test_policy_selects_native_s3_when_endpoint_not_set_but_aws_runtime_is_present(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OBSERVED_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", "true")
    monkeypatch.delenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.delenv("DVC_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    decision = resolve_local_vs_object_storage(
        local_paths=[tmp_path / "missing.bin"],
        fallback_env_var="OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK",
    )

    assert decision.source == "object_storage"
    assert decision.storage_configured is True


def test_manifest_policy_prefers_manifest_when_valid(monkeypatch) -> None:
    monkeypatch.delenv("OBSERVED_ENABLE_LEGACY_MODEL_PATH_FALLBACK", raising=False)

    decision = resolve_manifest_vs_legacy_fallback(manifest_state="valid")

    assert decision.source == "manifest"
    assert decision.manifest_state == "valid"


def test_manifest_policy_allows_legacy_by_default_when_manifest_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OBSERVED_ENABLE_LEGACY_MODEL_PATH_FALLBACK", raising=False)

    decision = resolve_manifest_vs_legacy_fallback(manifest_state="missing")

    assert decision.source == "legacy"
    assert decision.legacy_fallback_enabled is True


def test_manifest_policy_disables_legacy_when_env_is_false(monkeypatch) -> None:
    monkeypatch.setenv("OBSERVED_ENABLE_LEGACY_MODEL_PATH_FALLBACK", "false")

    decision = resolve_manifest_vs_legacy_fallback(manifest_state="invalid")

    assert decision.source == "unavailable"
    assert "disabled" in decision.reason
