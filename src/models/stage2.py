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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.observed.utils.artifact_publisher import publish_model_bundle
from src.observed.utils.config import load_config
from src.observed.utils.mlflow_utils import configure_mlflow

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


def _map_gas_type(exp_name: str) -> int | None:
    v = str(exp_name).lower()
    if "toluene" in v:
        return 1
    if "2-butanone" in v or "butanone" in v:
        return 0
    return None


def _prepare_xy(
    single_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = single_df.copy()
    df["target_gas_type"] = df["experiment"].apply(_map_gas_type)
    df = df.dropna(subset=["target_gas_type"]).copy()
    df["target_gas_type"] = df["target_gas_type"].astype(int)

    y = df["target_gas_type"]
    groups = df["run_id"].astype(str)

    drop_cols = METADATA_COLUMNS + ["target_mixture", "target_gas_type"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y, groups, list(X.columns)


def _baseline_models(random_state: int) -> dict[str, Any]:
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
    single_features_path: Path, single_df: pd.DataFrame, groups: pd.Series
) -> None:
    if _mlflow is None:
        return
    _mlflow.log_param("dataset_path", str(single_features_path))
    _mlflow.log_metric("feature_rows", int(len(single_df)))
    _mlflow.log_metric("feature_columns", int(single_df.shape[1]))
    _mlflow.log_metric("n_groups", int(groups.nunique()))
    _mlflow.log_metric("n_runs", int(groups.nunique()))


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
            "publisher": "train_stage2_probability_model",
        },
    )


def train_stage2_probability_model(
    single_features_path: Path,
    model_dir: Path,
    comparison_path: Path,
    tuning_path: Path,
    metrics_path: Path,
    predictions_path: Path | None = None,
    mlflow_experiment_name: str | None = None,
    config_path: Path = Path("configs/default.yaml"),
) -> dict[str, Any]:
    config = load_config(config_path)
    hp_cfg = config.get("hierarchical_pipeline", {})
    seed_cfg = hp_cfg.get("seeds", {})
    cv_cfg = hp_cfg.get("cv", {})
    paths_cfg = hp_cfg.get("paths", {})
    global_mlflow_cfg = config.get("mlflow", {})
    hp_mlflow_cfg = hp_cfg.get("mlflow", {})
    mlflow_cfg = {**global_mlflow_cfg, **hp_mlflow_cfg}
    stage2_cfg = hp_cfg.get("stage2", {})
    tuning_cfg = stage2_cfg.get("tuning", {})

    random_state = int(seed_cfg.get("global", 42))
    cv_n_splits = int(cv_cfg.get("n_splits", 3))
    cv_shuffle = bool(cv_cfg.get("shuffle", True))

    if predictions_path is None:
        predictions_path = Path(
            paths_cfg.get(
                "stage2_predictions",
                "artifacts/predictions/stage2_training_predictions.parquet",
            )
        )
    if mlflow_experiment_name is None:
        mlflow_experiment_name = str(
            mlflow_cfg.get("stage2_experiment", "stage2_single_gas_probability")
        )

    lr_tuning_cfg = tuning_cfg.get("logistic_regression", {})
    gb_tuning_cfg = tuning_cfg.get("gradient_boosting", {})

    lr_c_grid = lr_tuning_cfg.get("C", [0.01, 0.1, 1, 10, 50])
    lr_class_weight_grid = lr_tuning_cfg.get("class_weight", [None, "balanced"])
    gb_n_estimators_grid = gb_tuning_cfg.get("n_estimators", [50, 100, 200])
    gb_learning_rate_grid = gb_tuning_cfg.get("learning_rate", [0.01, 0.05, 0.1])
    gb_max_depth_grid = gb_tuning_cfg.get("max_depth", [1, 2, 3])

    single_df = pd.read_parquet(single_features_path)
    if single_df.empty:
        raise ValueError("features_single_gas is empty")

    X, y, groups, feature_columns = _prepare_xy(single_df)
    cv = StratifiedGroupKFold(
        n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=random_state
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    baseline_rows: list[dict[str, Any]] = []

    mlflow_enabled = _mlflow is not None
    if mlflow_enabled:
        try:
            configure_mlflow(
                mlflow_cfg=mlflow_cfg, experiment_name=mlflow_experiment_name
            )
        except Exception as exc:
            LOGGER.warning(
                "Disabling MLflow logging for stage2 due to configuration/import error: %s",
                exc,
            )
            mlflow_enabled = False
    else:
        LOGGER.warning(
            "MLflow import failed in current runtime; stage2 training will continue without MLflow logging."
        )

    if mlflow_enabled:
        with _mlflow.start_run(run_name="stage2_baseline_comparison"):
            _mlflow.log_param("stage", "stage2")
            _mlflow.log_param("run_name", "stage2_baseline_comparison")
            _mlflow.log_param("cv_strategy", "StratifiedGroupKFold")
            _mlflow.log_param("cv_splits", cv_n_splits)
            _mlflow.log_param("scoring", "accuracy_precision_recall_f1")
            _log_dataset_context(single_features_path, single_df, groups)
            try:
                _mlflow.log_text(
                    json.dumps({"feature_columns": feature_columns}, indent=2),
                    "feature_columns.json",
                )
            except Exception as exc:
                LOGGER.warning(
                    "Skipping MLflow feature_columns.json text artifact for stage2: %s",
                    exc,
                )
                _mlflow.log_param("feature_columns_artifact_logging", "skipped")
            for model_name, model in _baseline_models(
                random_state=random_state
            ).items():
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
                baseline_rows.append(
                    {
                        "model": model_name,
                        "mean_accuracy": float(np.mean(result["test_accuracy"])),
                        "std_accuracy": float(np.std(result["test_accuracy"])),
                        "mean_f1": float(np.mean(result["test_f1"])),
                    }
                )
                with _mlflow.start_run(
                    run_name=f"stage2_baseline_{model_name}", nested=True
                ):
                    _mlflow.log_param("stage", "stage2")
                    _mlflow.log_param("run_name", f"stage2_baseline_{model_name}")
                    _mlflow.log_param("model_name", model_name)
                    _mlflow.log_param("dataset_path", str(single_features_path))
                    _mlflow.log_param("feature_count", len(feature_columns))
                    _mlflow.log_param("group_count", int(groups.nunique()))
                    _mlflow.log_metrics(
                        {
                            "cv_accuracy_mean": float(np.mean(result["test_accuracy"])),
                            "cv_accuracy_std": float(np.std(result["test_accuracy"])),
                            "cv_precision_mean": float(
                                np.mean(result["test_precision"])
                            ),
                            "cv_recall_mean": float(np.mean(result["test_recall"])),
                            "cv_f1_mean": float(np.mean(result["test_f1"])),
                        }
                    )
                    _mlflow.log_dict(
                        _json_safe(result), f"baseline_{model_name}_cv_results.json"
                    )

            baseline_df = (
                pd.DataFrame(baseline_rows)
                .sort_values("mean_f1", ascending=False)
                .reset_index(drop=True)
            )
            _mlflow.log_dict(
                _json_safe(baseline_df.to_dict(orient="records")),
                "baseline_model_comparison.json",
            )
    else:
        for model_name, model in _baseline_models(random_state=random_state).items():
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
            baseline_rows.append(
                {
                    "model": model_name,
                    "mean_accuracy": float(np.mean(result["test_accuracy"])),
                    "std_accuracy": float(np.std(result["test_accuracy"])),
                    "mean_f1": float(np.mean(result["test_f1"])),
                }
            )
        baseline_df = (
            pd.DataFrame(baseline_rows)
            .sort_values("mean_f1", ascending=False)
            .reset_index(drop=True)
        )

    # Tune LR and GB candidates with group-aware cross-validation.
    lr_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    lr_grid = GridSearchCV(
        estimator=lr_pipeline,
        param_grid={
            "model__C": lr_c_grid,
            "model__class_weight": lr_class_weight_grid,
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    lr_grid.fit(X, y, groups=groups)

    gb_grid = GridSearchCV(
        estimator=Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
        param_grid={
            "model__n_estimators": gb_n_estimators_grid,
            "model__learning_rate": gb_learning_rate_grid,
            "model__max_depth": gb_max_depth_grid,
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    gb_grid.fit(X, y, groups=groups)

    # Auto-select the best tuned model by CV F1 (no hard-coded final estimator).
    tuned_candidates = {
        "logistic_regression_tuned": {
            "score": float(lr_grid.best_score_),
            "estimator": lr_grid.best_estimator_,
            "best_params": lr_grid.best_params_,
        },
        "gradient_boosting_tuned": {
            "score": float(gb_grid.best_score_),
            "estimator": gb_grid.best_estimator_,
            "best_params": gb_grid.best_params_,
        },
    }
    selected_model_name = max(
        tuned_candidates,
        key=lambda name: tuned_candidates[name]["score"],
    )
    final_model = tuned_candidates[selected_model_name]["estimator"]
    final_model.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, model_dir / "model.joblib")

    metadata = {
        "stage": "stage2",
        "task": "toluene_vs_2butanone_probability",
        "model_name": selected_model_name,
        "feature_columns": feature_columns,
        "classes": [0, 1],
        "class_labels": {"0": "2-butanone", "1": "Toluene"},
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    published_bundle = _publish_model_bundle(
        stage="stage2",
        model_name="stage2-gas-probability-model",
        model_dir=model_dir,
    )

    baseline_df.to_json(comparison_path, orient="records", indent=2)

    tuning = {
        "logistic_regression": {
            "best_params": lr_grid.best_params_,
            "best_cv_f1": float(lr_grid.best_score_),
        },
        "gradient_boosting": {
            "best_params": gb_grid.best_params_,
            "best_cv_f1": float(gb_grid.best_score_),
        },
        "final_selected_model": selected_model_name,
        "final_selected_score_f1": float(
            tuned_candidates[selected_model_name]["score"]
        ),
    }
    tuning_path.write_text(json.dumps(tuning, indent=2), encoding="utf-8")

    metrics = {
        "stage": "stage2",
        "final_model": selected_model_name,
        "baseline_best_model": str(baseline_df.iloc[0]["model"]),
        "baseline_best_f1": float(baseline_df.iloc[0]["mean_f1"]),
        "tuned_lr_f1": float(lr_grid.best_score_),
        "tuned_gb_f1": float(gb_grid.best_score_),
        "final_selected_f1": float(tuned_candidates[selected_model_name]["score"]),
        "n_samples": int(len(X)),
        "n_runs": int(groups.nunique()),
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
            "target_gas_type": y.values.astype(int),
            "stage2_prediction": train_pred.astype(int),
            "stage2_prob_toluene": pred_prob.astype(float),
        }
    )
    pred_df.to_parquet(predictions_path, index=False)

    if mlflow_enabled:
        with _mlflow.start_run(run_name="stage2_final_tuned_model"):
            _mlflow.log_param("stage", "stage2")
            _mlflow.log_param("run_name", "stage2_final_tuned_model")
            _mlflow.log_param("dataset_path", str(single_features_path))
            _mlflow.log_param("feature_count", len(feature_columns))
            _mlflow.log_param("group_count", int(groups.nunique()))
            _mlflow.log_param("selected_best_model", selected_model_name)
            _mlflow.log_param("saved_model_path", str(model_dir / "model.joblib"))
            _mlflow.log_param("metadata_path", str(model_dir / "metadata.json"))
            _mlflow.log_param(
                "tuned_hyperparameters",
                json.dumps(tuned_candidates[selected_model_name]["best_params"]),
            )
            _mlflow.log_metrics(
                {
                    "baseline_best_f1": metrics["baseline_best_f1"],
                    "tuned_lr_f1": metrics["tuned_lr_f1"],
                    "tuned_gb_f1": metrics["tuned_gb_f1"],
                    "selected_model_f1": metrics["final_selected_f1"],
                    "feature_rows": int(len(single_df)),
                    "n_runs": int(groups.nunique()),
                }
            )
            _mlflow.log_dict(_json_safe(tuning), "tuned_hyperparameters.json")
            _mlflow.log_dict(_json_safe(metrics), "selected_model_metrics.json")
            try:
                model_info = _mlflow.sklearn.log_model(
                    sk_model=final_model,
                    artifact_path="model",
                )
                try:
                    _mlflow.register_model(
                        model_uri=model_info.model_uri,
                        name="stage2-gas-probability-model",
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Skipping explicit MLflow model registration for stage2: %s",
                        exc,
                    )
            except Exception as exc:
                LOGGER.warning("Skipping MLflow model logging for stage2: %s", exc)
            _mlflow.log_artifact(str(comparison_path))
            _mlflow.log_artifact(str(tuning_path))
            _mlflow.log_artifact(str(metrics_path))
            _mlflow.log_artifact(str(model_dir / "metadata.json"))
            _mlflow.log_artifact(str(model_dir / "model.joblib"))
            _mlflow.log_artifact(str(predictions_path))

    return {
        "status": "success",
        "model_dir": str(model_dir),
        "model_bundle_storage": published_bundle,
        "comparison_path": str(comparison_path),
        "tuning_path": str(tuning_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
