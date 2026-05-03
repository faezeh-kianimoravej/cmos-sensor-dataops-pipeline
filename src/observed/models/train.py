"""Baseline model training helpers for the Observed project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

logger = logging.getLogger(__name__)

PREFERRED_FEATURE_COLUMNS = [
    "peak_value",
    "rise_time",
    "recovery_time",
    "duration",
    "baseline_noise",
]

ID_COLUMNS = {"event_id", "run_id"}


def load_feature_table(features_path: str | Path) -> pd.DataFrame:
    """Load the feature table from parquet."""
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_parquet(path)


def infer_task_type(target: pd.Series) -> str:
    """Infer modeling task from target dtype."""
    if pd.api.types.is_numeric_dtype(target):
        return "regression"
    return "classification"


def select_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    preferred_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Pick model feature columns with a preferred-column-first strategy."""
    preferred = list(preferred_columns or PREFERRED_FEATURE_COLUMNS)

    available_preferred = [
        col for col in preferred if col in df.columns and col != target_column
    ]
    missing_preferred = [col for col in preferred if col not in df.columns]

    if available_preferred:
        return available_preferred, missing_preferred

    fallback_numeric = [
        col
        for col in df.columns
        if col not in ID_COLUMNS
        and col != target_column
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return fallback_numeric, missing_preferred


def prepare_feature_matrix(
    df: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Create a numeric feature matrix and handle missing values safely."""
    if not feature_columns:
        raise ValueError("No usable feature columns were selected")

    x_df = df[list(feature_columns)].copy()
    for column in x_df.columns:
        x_df[column] = pd.to_numeric(x_df[column], errors="coerce")

    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x_df = x_df.fillna(0.0)
    return x_df


def fit_baseline_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    random_state: int = 42,
) -> RandomForestRegressor | RandomForestClassifier:
    """Train a RandomForest baseline model."""
    if task_type == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    model.fit(x_train, y_train)
    return model


def fit_unsupervised_model(
    x_train: pd.DataFrame,
    random_state: int = 42,
    contamination: float = 0.1,
) -> IsolationForest:
    """Train a deterministic Isolation Forest anomaly detector."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(x_train)
    return model


def fit_deep_learning_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    random_state: int = 42,
) -> MLPRegressor | MLPClassifier:
    """Train a lightweight deterministic MLP model for deep-learning compliance."""
    if task_type == "regression":
        model = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=random_state,
            early_stopping=False,
        )
    elif task_type == "classification":
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=random_state,
            early_stopping=False,
        )
    else:
        raise ValueError(f"Unsupported task_type for deep learning model: {task_type}")

    model.fit(x_train, y_train)
    return model


def train_baseline_model(
    features_path: str,
    target_column: str = "auc",
    random_state: int = 42,
) -> tuple[RandomForestRegressor | RandomForestClassifier, np.ndarray, Dict[str, Any]]:
    """Train a baseline model from the feature table and predict on same rows.

    This helper keeps things simple for quick experimentation. For proper
    train/test evaluation, use pipelines/train_flow.py.
    """
    df = load_feature_table(features_path)
    if df.empty:
        raise ValueError(f"Feature table is empty: {features_path}")

    resolved_target = target_column
    if resolved_target not in df.columns:
        fallback_targets = [
            col
            for col in df.columns
            if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not fallback_targets:
            raise ValueError(
                f"Target column '{target_column}' not found and no numeric fallback is available"
            )
        resolved_target = fallback_targets[0]
        logger.info(
            f"[train] WARNING: target '{target_column}' missing; using '{resolved_target}' instead"
        )

    feature_columns, missing_preferred = select_feature_columns(df, resolved_target)
    if not feature_columns:
        raise ValueError("Could not determine usable feature columns for training")

    x_df = prepare_feature_matrix(df, feature_columns)
    y = df[resolved_target]
    task_type = infer_task_type(y)

    model = fit_baseline_model(x_df, y, task_type=task_type, random_state=random_state)
    predictions = model.predict(x_df)

    summary: Dict[str, Any] = {
        "target_column": resolved_target,
        "task_type": task_type,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "missing_preferred_features": missing_preferred,
    }
    return model, np.asarray(predictions), summary


__all__ = [
    "ID_COLUMNS",
    "PREFERRED_FEATURE_COLUMNS",
    "fit_baseline_model",
    "fit_deep_learning_model",
    "fit_unsupervised_model",
    "infer_task_type",
    "load_feature_table",
    "prepare_feature_matrix",
    "select_feature_columns",
    "train_baseline_model",
]
