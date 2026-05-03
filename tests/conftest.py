from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STORAGE_ENV_VARS = [
    "OBSERVED_STORAGE_BACKEND",
    "OBSERVED_STORAGE_S3_ENDPOINT_URL",
    "MINIO_PUBLIC_ENDPOINT",
    "MINIO_ENDPOINT",
    "DVC_S3_ENDPOINT_URL",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
]


def pytest_configure(config: pytest.Config) -> None:
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        Path(str(basetemp)).mkdir(parents=True, exist_ok=True)

    cache_dir = config.getini("cache_dir")
    if cache_dir:
        Path(str(cache_dir)).mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _default_test_storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests isolated from any real AWS/MinIO credentials present in CI."""

    monkeypatch.setenv("OBSERVED_STORAGE_BACKEND", "minio")
    for name in _STORAGE_ENV_VARS:
        if name != "OBSERVED_STORAGE_BACKEND":
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def metadata_columns() -> list[str]:
    return [
        "run_id",
        "experiment",
        "experiment_folder",
        "run_folder",
        "repeat_index",
        "window_id",
        "start_idx",
        "end_idx",
        "time_start",
        "time_end",
    ]


@pytest.fixture
def synthetic_windowed_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    runs = [
        ("run_toluene_1", "Toluene"),
        ("run_toluene_2", "Toluene"),
        ("run_butanone_1", "2-butanone"),
        ("run_mixing_1", "Mixing"),
    ]
    for i, (run_id, exp) in enumerate(runs):
        for w in range(2):
            signal = np.linspace(0.0 + i, 1.0 + i, 20)
            rows.append(
                {
                    "run_id": run_id,
                    "experiment": exp,
                    "experiment_folder": exp.lower(),
                    "run_folder": f"folder_{run_id}",
                    "repeat_index": i,
                    "window_id": f"{run_id}_w{w}",
                    "start_idx": w * 10,
                    "end_idx": w * 10 + 19,
                    "time_start": float(w),
                    "time_end": float(w + 1),
                    "signal_window": signal.tolist(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_features_df(synthetic_windowed_df: pd.DataFrame) -> pd.DataFrame:
    from src.features.extractor import build_feature_table

    return build_feature_table(synthetic_windowed_df)


@pytest.fixture
def temp_config_path(tmp_path: Path) -> Path:
    cfg = tmp_path / "default.yaml"
    cfg.write_text(
        """
hierarchical_pipeline:
  paths:
    processed_timeseries: data/processed/processed_timeseries.parquet
    runs: data/processed/runs.parquet
    ingestion_summary: artifacts/reports/ingestion_summary.json
    windowed_timeseries: data/processed/windowed_timeseries.parquet
    train_windows: data/processed/train_windows.parquet
    test_windows: data/processed/test_windows.parquet
    features_timeseries: data/processed/features_timeseries.parquet
    feature_engineering_summary: artifacts/reports/feature_engineering_summary.json
    stage1_model_dir: artifacts/models/stage1_mixture_detector
    stage1_comparison: artifacts/reports/stage1_model_comparison.json
    stage1_metrics: artifacts/reports/stage1_cv_metrics.json
    stage1_predictions: artifacts/predictions/stage1_training_predictions.parquet
    features_single_gas: data/processed/features_single_gas.parquet
    stage2_subset_summary: artifacts/reports/stage2_subset_summary.json
    stage2_model_dir: artifacts/models/stage2_gas_probability_model
    stage2_comparison: artifacts/reports/stage2_model_comparison.json
    stage2_tuning: artifacts/reports/stage2_tuning_results.json
    stage2_metrics: artifacts/reports/stage2_cv_metrics.json
    stage2_predictions: artifacts/predictions/stage2_training_predictions.parquet
    evaluate_stage1_report: artifacts/reports/evaluate_stage1_report.json
    evaluate_stage2_report: artifacts/reports/evaluate_stage2_report.json
    batch_predictions: artifacts/predictions/batch_predictions.parquet
    batch_prediction_summary: artifacts/reports/batch_prediction_summary.json
  feature_engineering:
    window_size: 20
    step_size: 10
    test_size: 0.25
  seeds:
    global: 42
  mlflow:
    stage1_experiment: stage1_mixture_detection
    stage2_experiment: stage2_single_gas_probability
mlflow:
  tracking_uri: sqlite:///artifacts/mlflow/mlflow_tracking.db
  registry_uri: sqlite:///artifacts/mlflow/mlflow_tracking.db
  artifacts_dir: artifacts/mlruns
""".strip(),
        encoding="utf-8",
    )
    return cfg
