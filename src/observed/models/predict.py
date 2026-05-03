"""Prediction helpers for the Observed baseline model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_model(model_path: str) -> Any:
    """Load a pickled model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with path.open("rb") as handle:
        return pickle.load(handle)


def _prepare_numeric_features(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Build a numeric feature matrix compatible with the trained model."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns found in feature table")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    numeric_df = numeric_df.fillna(0.0)

    if hasattr(model, "feature_names_in_"):
        expected = [str(name) for name in model.feature_names_in_]
        return numeric_df.reindex(columns=expected, fill_value=0.0)

    if hasattr(model, "n_features_in_"):
        expected_n = int(model.n_features_in_)
        if numeric_df.shape[1] > expected_n:
            return numeric_df.iloc[:, :expected_n]
        if numeric_df.shape[1] < expected_n:
            for idx in range(expected_n - numeric_df.shape[1]):
                numeric_df[f"_pad_{idx}"] = 0.0
            return numeric_df

    return numeric_df


def _id_column_or_default(
    df: pd.DataFrame, column: str, default_prefix: str
) -> pd.Series:
    """Return existing ID column or generate a stable fallback."""
    if column in df.columns:
        return df[column].astype(str)
    return pd.Series([f"{default_prefix}-{i:05d}" for i in range(len(df))], dtype=str)


def predict_from_features(model_path: str, features_path: str) -> pd.DataFrame:
    """Load model + features and return predictions with event/run identifiers."""
    model = _load_model(model_path)

    features_file = Path(features_path)
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")

    features_df = pd.read_parquet(features_file)
    if features_df.empty:
        raise ValueError(f"Feature table is empty: {features_file}")

    x_df = _prepare_numeric_features(model, features_df)
    predictions = model.predict(x_df)

    output = pd.DataFrame(
        {
            "event_id": _id_column_or_default(features_df, "event_id", "event"),
            "run_id": _id_column_or_default(features_df, "run_id", "run"),
            "prediction": np.asarray(predictions),
        }
    )
    return output


__all__ = ["predict_from_features"]
