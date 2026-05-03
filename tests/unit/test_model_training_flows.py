from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.prepare_stage2_single_gas_subset_flow import run_prepare_stage2_subset
from pipelines.train_stage1_mixture_detection_flow import (
    run_train_stage1_mixture_detection,
)
from pipelines.train_stage2_single_gas_probability_flow import (
    run_train_stage2_single_gas_probability,
)

pytestmark = pytest.mark.unit


def test_train_stage1_wrapper_calls_trainer(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def _fake(**kwargs):
        called.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(
        "pipelines.train_stage1_mixture_detection_flow.train_stage1_mixture_detector",
        _fake,
    )
    out = run_train_stage1_mixture_detection(
        tmp_path / "features.parquet",
        tmp_path / "model",
        tmp_path / "comparison.json",
        tmp_path / "metrics.json",
        tmp_path / "predictions.parquet",
        "stage1_exp",
        tmp_path / "cfg.yaml",
    )
    assert out["status"] == "success"
    assert called["mlflow_experiment_name"] == "stage1_exp"


def test_train_stage2_wrapper_calls_trainer(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def _fake(**kwargs):
        called.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(
        "pipelines.train_stage2_single_gas_probability_flow.train_stage2_probability_model",
        _fake,
    )
    out = run_train_stage2_single_gas_probability(
        tmp_path / "single.parquet",
        tmp_path / "model",
        tmp_path / "comparison.json",
        tmp_path / "tuning.json",
        tmp_path / "metrics.json",
        tmp_path / "predictions.parquet",
        "stage2_exp",
        tmp_path / "cfg.yaml",
    )
    assert out["status"] == "success"
    assert called["mlflow_experiment_name"] == "stage2_exp"


def test_prepare_stage2_subset_creates_single_gas_artifact(
    tmp_path: Path, synthetic_features_df: pd.DataFrame
) -> None:
    features_path = tmp_path / "features.parquet"
    synthetic_features_df.to_parquet(features_path, index=False)

    subset_path = tmp_path / "data" / "processed" / "features_single_gas.parquet"
    summary_path = tmp_path / "artifacts" / "reports" / "stage2_subset_summary.json"

    result = run_prepare_stage2_subset(features_path, subset_path, summary_path)
    assert result["status"] == "success"
    assert subset_path.exists()
    subset_df = pd.read_parquet(subset_path)
    assert all(
        "mixing" not in str(v).lower() for v in subset_df["experiment"]
    )  # deterministic filtering
