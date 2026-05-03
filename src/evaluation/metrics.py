from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.observed.utils.config import load_config
from src.observed.utils.mlflow_utils import configure_mlflow

LOGGER = logging.getLogger(__name__)


def evaluate_stage1_from_predictions(predictions_path: Path) -> dict[str, Any]:
    df = pd.read_parquet(predictions_path)
    if df.empty:
        raise ValueError("stage1 predictions are empty")
    if "target_mixture" not in df.columns or "stage1_prediction" not in df.columns:
        raise ValueError(
            "stage1 predictions parquet must contain target_mixture and stage1_prediction"
        )

    y_true = df["target_mixture"].astype(int)
    y_pred = df["stage1_prediction"].astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "stage": "stage6",
        "target_stage": "stage1",
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def evaluate_stage2_from_predictions(predictions_path: Path) -> dict[str, Any]:
    df = pd.read_parquet(predictions_path)
    if df.empty:
        raise ValueError("stage2 predictions are empty")
    if "target_gas_type" not in df.columns or "stage2_prediction" not in df.columns:
        raise ValueError(
            "stage2 predictions parquet must contain target_gas_type and stage2_prediction"
        )

    y_true = df["target_gas_type"].astype(int)
    y_pred = df["stage2_prediction"].astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "stage": "stage6",
        "target_stage": "stage2",
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def log_evaluation_to_mlflow(
    *,
    experiment_name: str,
    run_name: str,
    dataset_path: Path,
    report: dict[str, Any],
    output_path: Path,
    artifacts: list[Path],
    selected_best_model: str,
    run_count: int | None = None,
    feature_shape: tuple[int, int] | None = None,
) -> None:
    config = load_config("configs/default.yaml")
    global_mlflow_cfg = config.get("mlflow", {})
    hp_mlflow_cfg = config.get("hierarchical_pipeline", {}).get("mlflow", {})
    mlflow_cfg = {**global_mlflow_cfg, **hp_mlflow_cfg}
    try:
        configure_mlflow(mlflow_cfg=mlflow_cfg, experiment_name=experiment_name)
    except Exception as exc:
        LOGGER.warning(
            "Skipping MLflow evaluation logging because MLflow is unavailable: %s",
            exc,
        )
        return

    inferred_rows: int | None = None
    inferred_runs: int | None = None
    if dataset_path.exists():
        try:
            preview_df = pd.read_parquet(dataset_path)
            inferred_rows = int(len(preview_df))
            if "run_id" in preview_df.columns:
                inferred_runs = int(preview_df["run_id"].astype(str).nunique())
            elif "run" in preview_df.columns:
                inferred_runs = int(preview_df["run"].astype(str).nunique())
        except Exception:
            pass

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("run_name", run_name)
            mlflow.log_param("dataset_path", str(dataset_path))
            mlflow.log_param("output_path", str(output_path))
            mlflow.log_param("selected_best_model", selected_best_model)
            if run_count is not None:
                mlflow.log_param("n_runs", int(run_count))
                mlflow.log_metric("n_runs", int(run_count))
            elif inferred_runs is not None:
                mlflow.log_param("n_runs", inferred_runs)
                mlflow.log_metric("n_runs", inferred_runs)
            if inferred_rows is not None:
                mlflow.log_metric("dataset_rows", inferred_rows)
            if feature_shape is not None:
                mlflow.log_metric("feature_rows", int(feature_shape[0]))
                mlflow.log_metric("feature_columns", int(feature_shape[1]))

            for key, value in report.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    mlflow.log_metric(f"report_{key}", float(value))
                elif isinstance(value, str):
                    mlflow.log_param(f"report_{key}", value)

            mlflow.log_dict(report, "evaluation_report.json")
            mlflow.log_artifact(str(output_path))
            for artifact_path in artifacts:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))
    except Exception as exc:
        LOGGER.warning(
            "MLflow evaluation logging failed for run '%s' in experiment '%s': %s",
            run_name,
            experiment_name,
            exc,
        )


def write_report(report: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
