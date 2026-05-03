from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.observed.utils.artifact_publisher import (
    publish_local_artifact,
    publish_model_bundle,
)
from src.observed.utils.config import load_config
from src.observed.utils.mlflow_utils import configure_mlflow
from src.observed.utils.storage_runtime import (
    resolve_storage_environment,
    stage_dataset_version,
)

try:
    import mlflow as _mlflow
except Exception:  # pragma: no cover - environment-specific dependency resolution
    _mlflow = None

LOGGER = logging.getLogger(__name__)


METADATA_COLUMNS = [
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


def _build_stage1_target(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df["target_mixture"] = df["experiment"].apply(
        lambda x: 1 if "mixing" in str(x).lower() else 0
    )
    return df


def _stage1_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    y = df["target_mixture"].astype(int)
    groups = df["run_id"].astype(str)
    drop_cols = METADATA_COLUMNS + ["target_mixture"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_columns = list(X.columns)
    return X, y, groups, feature_columns


def _build_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(kernel="rbf", random_state=random_state, probability=True),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, random_state=random_state, n_jobs=-1
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def _to_serializable(cv_result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cv_result.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _log_dataset_context(
    features_path: Path, features_df: pd.DataFrame, groups: pd.Series
) -> None:
    if _mlflow is None:
        return
    _mlflow.log_param("dataset_path", str(features_path))
    _mlflow.log_metric("feature_rows", int(len(features_df)))
    _mlflow.log_metric("feature_columns", int(features_df.shape[1]))
    _mlflow.log_metric("n_groups", int(groups.nunique()))
    _mlflow.log_metric("n_runs", int(groups.nunique()))


def _log_cv_summary(prefix: str, scores: dict[str, Any]) -> None:
    if _mlflow is None:
        return
    _mlflow.log_metrics(
        {
            f"{prefix}_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            f"{prefix}_accuracy_std": float(np.std(scores["test_accuracy"])),
            f"{prefix}_precision_mean": float(np.mean(scores["test_precision"])),
            f"{prefix}_recall_mean": float(np.mean(scores["test_recall"])),
            f"{prefix}_f1_mean": float(np.mean(scores["test_f1"])),
        }
    )


def _publish_model_bundle(
    *,
    stage: str,
    model_name: str,
    model_dir: Path,
) -> dict[str, Any]:
    return publish_model_bundle(
        stage=stage,
        model_name=model_name,
        model_dir=model_dir,
        source_info={
            "publisher": "train_stage1_mixture_detector",
        },
    )


def train_stage1_mixture_detector(
    features_path: Path,
    model_dir: Path,
    comparison_path: Path,
    metrics_path: Path,
    predictions_path: Path | None = None,
    mlflow_experiment_name: str | None = None,
    config_path: Path = Path("configs/default.yaml"),
) -> dict[str, Any]:
    config = load_config(config_path)
    hp_cfg = config.get("hierarchical_pipeline", {})
    stage1_cfg = hp_cfg.get("stage1", {})
    seed_cfg = hp_cfg.get("seeds", {})
    cv_cfg = hp_cfg.get("cv", {})
    paths_cfg = hp_cfg.get("paths", {})
    global_mlflow_cfg = config.get("mlflow", {})
    hp_mlflow_cfg = hp_cfg.get("mlflow", {})
    mlflow_cfg = {**global_mlflow_cfg, **hp_mlflow_cfg}

    random_state = int(seed_cfg.get("global", 42))
    cv_n_splits = int(cv_cfg.get("n_splits", 3))
    cv_shuffle = bool(cv_cfg.get("shuffle", True))
    selection_metric = str(stage1_cfg.get("selection_metric", "mean_f1"))

    if predictions_path is None:
        predictions_path = Path(
            paths_cfg.get(
                "stage1_predictions",
                "artifacts/predictions/stage1_training_predictions.parquet",
            )
        )
    if mlflow_experiment_name is None:
        mlflow_experiment_name = str(
            mlflow_cfg.get("stage1_experiment", "stage1_mixture_detection")
        )

    features_df = pd.read_parquet(features_path)
    if features_df.empty:
        raise ValueError("features_timeseries is empty")

    features_df = _build_stage1_target(features_df)
    X, y, groups, feature_columns = _stage1_xy(features_df)

    cv = StratifiedGroupKFold(
        n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=random_state
    )
    models = _build_models(random_state=random_state)

    scores: list[dict[str, Any]] = []
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    mlflow_enabled = _mlflow is not None
    if mlflow_enabled:
        try:
            configure_mlflow(
                mlflow_cfg=mlflow_cfg, experiment_name=mlflow_experiment_name
            )
        except Exception as exc:
            LOGGER.warning(
                "Disabling MLflow logging for stage1 due to configuration/import error: %s",
                exc,
            )
            mlflow_enabled = False
    else:
        LOGGER.warning(
            "MLflow import failed in current runtime; stage1 training will continue without MLflow logging."
        )

    if mlflow_enabled:
        with _mlflow.start_run(run_name="stage1_baseline_comparison"):
            _mlflow.log_param("stage", "stage1")
            _mlflow.log_param("run_name", "stage1_baseline_comparison")
            _mlflow.log_param("cv_strategy", "StratifiedGroupKFold")
            _mlflow.log_param("cv_splits", cv_n_splits)
            _mlflow.log_param("scoring", "accuracy_precision_recall_f1")
            _log_dataset_context(features_path, features_df, groups)
            try:
                _mlflow.log_text(
                    json.dumps({"feature_columns": feature_columns}, indent=2),
                    "feature_columns.json",
                )
            except Exception as exc:
                LOGGER.warning(
                    "Skipping MLflow feature_columns.json text artifact for stage1: %s",
                    exc,
                )
                _mlflow.log_param("feature_columns_artifact_logging", "skipped")
            for model_name, model in models.items():
                result = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    groups=groups,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=-1,
                )
                scores.append(
                    {
                        "model": model_name,
                        "mean_accuracy": float(np.mean(result["test_accuracy"])),
                        "std_accuracy": float(np.std(result["test_accuracy"])),
                        "mean_f1": float(np.mean(result["test_f1"])),
                        "raw": _to_serializable(result),
                    }
                )
                with _mlflow.start_run(
                    run_name=f"stage1_baseline_{model_name}", nested=True
                ):
                    _mlflow.log_param("stage", "stage1")
                    _mlflow.log_param("run_name", f"stage1_baseline_{model_name}")
                    _mlflow.log_param("model_name", model_name)
                    _mlflow.log_param("dataset_path", str(features_path))
                    _mlflow.log_param("feature_count", len(feature_columns))
                    _mlflow.log_param("group_count", int(groups.nunique()))
                    _log_cv_summary("cv", result)
                    _mlflow.log_dict(
                        _json_safe(result), f"baseline_{model_name}_cv_results.json"
                    )

            comparison_df = (
                pd.DataFrame(scores)
                .sort_values([selection_metric, "mean_accuracy"], ascending=False)
                .reset_index(drop=True)
            )
            _mlflow.log_dict(
                _json_safe(comparison_df.to_dict(orient="records")),
                "baseline_model_comparison.json",
            )
    else:
        for model_name, model in models.items():
            result = cross_validate(
                model,
                X,
                y,
                cv=cv,
                groups=groups,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )
            scores.append(
                {
                    "model": model_name,
                    "mean_accuracy": float(np.mean(result["test_accuracy"])),
                    "std_accuracy": float(np.std(result["test_accuracy"])),
                    "mean_f1": float(np.mean(result["test_f1"])),
                    "raw": _to_serializable(result),
                }
            )
        comparison_df = (
            pd.DataFrame(scores)
            .sort_values([selection_metric, "mean_accuracy"], ascending=False)
            .reset_index(drop=True)
        )

    best_model_name = str(comparison_df.iloc[0]["model"])
    final_model = models[best_model_name]
    final_model.fit(X, y)

    selected_row = comparison_df.loc[comparison_df["model"] == best_model_name].iloc[0]

    model_dir.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, model_dir / "model.joblib")

    model_meta = {
        "stage": "stage1",
        "task": "single_vs_mixture",
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "classes": [0, 1],
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(model_meta, indent=2), encoding="utf-8"
    )
    published_bundle = _publish_model_bundle(
        stage="stage1",
        model_name="stage1-mixture-detector",
        model_dir=model_dir,
    )

    comparison_df.to_json(comparison_path, orient="records", indent=2)

    metrics = {
        "stage": "stage1",
        "best_model": best_model_name,
        "mean_accuracy": float(selected_row["mean_accuracy"]),
        "mean_f1": float(selected_row["mean_f1"]),
        "n_samples": int(len(X)),
        "n_runs": int(groups.nunique()),
        "selection_metric": selection_metric,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_pred = final_model.predict(X)
    if hasattr(final_model, "predict_proba"):
        probs = final_model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            pred_prob = probs[:, 1]
        else:
            pred_prob = probs.ravel()
    else:
        pred_prob = train_pred.astype(float)

    pred_df = pd.DataFrame(
        {
            "run_id": groups.astype(str).values,
            "target_mixture": y.values.astype(int),
            "stage1_prediction": train_pred.astype(int),
            "stage1_prob_mixture": pred_prob.astype(float),
        }
    )
    pred_df.to_parquet(predictions_path, index=False)

    if mlflow_enabled:
        with _mlflow.start_run(run_name="stage1_final_selected_model"):
            _mlflow.log_param("stage", "stage1")
            _mlflow.log_param("run_name", "stage1_final_selected_model")
            _mlflow.log_param("dataset_path", str(features_path))
            _mlflow.log_param("feature_count", len(feature_columns))
            _mlflow.log_param("group_count", int(groups.nunique()))
            _mlflow.log_param("selected_best_model", best_model_name)
            _mlflow.log_param("saved_model_path", str(model_dir / "model.joblib"))
            _mlflow.log_param("metadata_path", str(model_dir / "metadata.json"))
            _mlflow.log_metric("feature_rows", int(len(features_df)))
            _mlflow.log_metric("n_runs", int(groups.nunique()))
            _mlflow.log_metrics(
                {
                    "selected_cv_mean_accuracy": metrics["mean_accuracy"],
                    "selected_cv_mean_f1": metrics["mean_f1"],
                }
            )
            _mlflow.log_dict(_json_safe(metrics), "selected_model_metrics.json")
            try:
                model_info = _mlflow.sklearn.log_model(
                    sk_model=final_model,
                    artifact_path="model",
                )
                try:
                    _mlflow.register_model(
                        model_uri=model_info.model_uri,
                        name="stage1-mixture-detector",
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Skipping explicit MLflow model registration for stage1: %s",
                        exc,
                    )
            except Exception as exc:
                LOGGER.warning("Skipping MLflow model logging for stage1: %s", exc)
            _mlflow.log_artifact(str(comparison_path))
            _mlflow.log_artifact(str(metrics_path))
            _mlflow.log_artifact(str(model_dir / "metadata.json"))
            _mlflow.log_artifact(str(model_dir / "model.joblib"))
            _mlflow.log_artifact(str(predictions_path))

    return {
        "status": "success",
        "best_model": best_model_name,
        "model_dir": str(model_dir),
        "model_bundle_storage": published_bundle,
        "comparison_path": str(comparison_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }


def create_true_single_gas_subset(
    features_path: Path,
    subset_output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    features_df = pd.read_parquet(features_path)
    features_df = _build_stage1_target(features_df)
    single_df = features_df[features_df["target_mixture"] == 0].copy()

    subset_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    single_df.to_parquet(subset_output_path, index=False)

    storage_env = resolve_storage_environment()
    dataset_version = stage_dataset_version()
    published_subset = publish_local_artifact(
        local_path=subset_output_path,
        domain="datasets",
        env=storage_env,
        pipeline="stage2-subset",
        dataset_name="features-single-gas",
        dataset_version=dataset_version,
    )

    summary = {
        "status": "success",
        "subset_rows": int(len(single_df)),
        "source_rows": int(len(features_df)),
        "subset_path": str(subset_output_path),
        "dataset_storage": {
            "environment": storage_env,
            "dataset_version": dataset_version,
            "features_single_gas": published_subset,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
