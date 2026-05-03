from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.inference import run_two_stage_batch_prediction

pytestmark = pytest.mark.unit


class DummyStage1Model:
    def predict(self, X):
        arr = X.to_numpy()
        return (arr[:, 0] > 0.5).astype(int)

    def predict_proba(self, X):
        p1 = self.predict(X).astype(float)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


class DummyStage2Model:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (X.to_numpy()[:, 0] > 0.2).astype(int)

    def predict_proba(self, X):
        p1 = np.clip(X.to_numpy()[:, 0], 0.0, 1.0)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


def _write_bundle(
    model_dir: Path, model, feature_cols: list[str], class_labels=None
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    metadata = {
        "model_name": "dummy",
        "feature_columns": feature_cols,
        "class_labels": class_labels or {"0": "2-butanone", "1": "Toluene"},
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_two_stage_batch_prediction_outputs_probabilities(
    tmp_path: Path, metadata_columns
) -> None:
    stage1_dir = tmp_path / "artifacts" / "models" / "stage1"
    stage2_dir = tmp_path / "artifacts" / "models" / "stage2"
    _write_bundle(stage1_dir, DummyStage1Model(), ["f1", "f2"])
    _write_bundle(stage2_dir, DummyStage2Model(), ["f1", "f2"])

    df = pd.DataFrame(
        [
            {
                **{c: 0 for c in metadata_columns},
                "f1": 0.1,
                "f2": 0.2,
            },
            {
                **{c: 1 for c in metadata_columns},
                "f1": 0.9,
                "f2": 0.2,
            },
        ]
    )
    for c in metadata_columns:
        if c in [
            "run_id",
            "experiment",
            "experiment_folder",
            "run_folder",
            "window_id",
        ]:
            df[c] = ["a", "b"]
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    out_path = tmp_path / "predictions.parquet"
    summary_path = tmp_path / "summary.json"

    summary = run_two_stage_batch_prediction(
        features_path, stage1_dir, stage2_dir, out_path, summary_path, 20, 10
    )
    result_df = pd.read_parquet(out_path)

    assert summary["status"] == "success"
    assert "stage1_label" in result_df.columns
    assert "stage2_prob_toluene" in result_df.columns
    assert result_df["stage2_prob_toluene"].dropna().between(0, 1).all()


def test_two_stage_batch_prediction_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_two_stage_batch_prediction(
            tmp_path / "missing.parquet",
            tmp_path / "stage1",
            tmp_path / "stage2",
            tmp_path / "out.parquet",
            tmp_path / "summary.json",
        )
