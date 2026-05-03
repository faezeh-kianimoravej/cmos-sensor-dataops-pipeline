from __future__ import annotations

import pytest

from src.observed.utils.storage_paths import StoragePathBuilder, StoragePathError

pytestmark = pytest.mark.unit


def test_dataset_path_uses_configured_bucket_and_prefix() -> None:
    builder = StoragePathBuilder.from_default_config()

    location = builder.dataset_path(
        env="dev",
        pipeline="feature-engineering",
        dataset_name="features-timeseries",
        dataset_version="dvc-a1b2c3d4",
        filename="dataset.parquet",
    )

    assert location.bucket == "datasets"
    assert (
        location.prefix
        == "processed/feature-engineering/features-timeseries/dvc-a1b2c3d4/"
    )
    assert (
        location.key
        == "processed/feature-engineering/features-timeseries/dvc-a1b2c3d4/dataset.parquet"
    )
    assert (
        location.uri
        == "s3://datasets/processed/feature-engineering/features-timeseries/dvc-a1b2c3d4/dataset.parquet"
    )


def test_report_path_uses_default_environment_when_not_provided() -> None:
    builder = StoragePathBuilder.from_default_config()

    location = builder.report_path(
        pipeline="stage1",
        report_family="cv-metrics",
        run_id="manual-20260407T102200Z",
    )

    assert location.bucket == "reports"
    assert location.prefix == "evaluation/stage1/cv-metrics/manual-20260407T102200Z/"
    assert location.key == location.prefix


def test_model_path_rejects_invalid_environment() -> None:
    builder = StoragePathBuilder.from_default_config()

    with pytest.raises(StoragePathError, match="Invalid storage environment"):
        builder.model_path(
            env="qa",
            model_family="stage1",
            model_name="stage1-mixture-detector",
            model_version="git-abc1234",
        )


def test_batch_prediction_path_requires_template_fields() -> None:
    builder = StoragePathBuilder.from_default_config()

    with pytest.raises(StoragePathError, match="Missing required storage path fields"):
        builder.batch_prediction_path(
            env="prod",
            prediction_job="batch-inference",
            job_date="2026/04/07",
            job_id="",
        )


def test_local_output_mapping_builds_expected_model_prefix() -> None:
    builder = StoragePathBuilder.from_default_config()

    location = builder.path_for_local_output(
        "artifacts/models/stage2_gas_probability_model",
        env="staging",
        model_version="git-3fa92c1",
        filename="metadata.json",
    )

    assert location.bucket == "models"
    assert location.prefix == "stage2/stage2-gas-probability-model/git-3fa92c1/"
    assert location.key.endswith("/metadata.json")


def test_local_output_mapping_rejects_unknown_path() -> None:
    builder = StoragePathBuilder.from_default_config()

    with pytest.raises(StoragePathError, match="No object storage mapping"):
        builder.path_for_local_output(
            "artifacts/models/unknown_model",
            env="dev",
            model_version="git-1234567",
        )


def test_model_bucket_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("OBSERVED_STORAGE_BUCKET_MODELS", "observed-api-models-123")
    builder = StoragePathBuilder.from_default_config()

    location = builder.model_path(
        env="prod",
        model_family="stage1",
        model_name="stage1-mixture-detector",
        model_version="git-abc1234",
        filename="model.joblib",
    )

    assert location.bucket == "observed-api-models-123"
