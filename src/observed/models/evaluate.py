"""Evaluation helpers for baseline model outputs."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    task_type: str,
) -> Dict[str, Any]:
    """Evaluate predictions with task-specific metrics."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0:
        raise ValueError("Cannot evaluate empty y_true")

    if task_type == "regression":
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

        if y_true_arr.size < 2:
            r2 = None
        else:
            try:
                r2 = float(r2_score(y_true_arr, y_pred_arr))
            except Exception:
                r2 = None

        return {
            "task_type": "regression",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    if task_type == "classification":
        return {
            "task_type": "classification",
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "f1": float(
                f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
            ),
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


__all__ = ["evaluate_predictions"]
