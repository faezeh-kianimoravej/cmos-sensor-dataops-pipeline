from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import build_windowed_timeseries
from src.features.extractor import build_feature_table
from src.observed.utils.artifact_publisher import publish_local_artifact
from src.observed.utils.storage_runtime import resolve_storage_environment

_METADATA_COLUMNS = [
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


def _default_batch_job_id() -> str:
    return f"manual-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _default_job_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y/%m/%d")


def _load_model_bundle(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"missing model file: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format: {suffix}; expected .parquet or .csv")


def _parse_list_like(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return value
    return value


def _normalize_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["signal_window", "signal_normalized", "signal", "time"]:
        if col in out.columns:
            out[col] = out[col].map(_parse_list_like)
    return out


def _load_or_compute_features(
    features_path: Path,
    stage1_feature_columns: list[str],
    stage2_feature_columns: list[str],
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, str]:
    input_df = _normalize_signal_columns(_read_table(features_path))
    if input_df.empty:
        raise ValueError("features input is empty")

    required = set(stage1_feature_columns) | set(stage2_feature_columns)
    if required.issubset(set(input_df.columns)):
        return input_df.copy(), "loaded"

    if "signal_window" in input_df.columns:
        features_df = build_feature_table(input_df)
        if not required.issubset(set(features_df.columns)):
            missing = sorted(required - set(features_df.columns))
            raise ValueError(
                f"computed features missing required columns: {missing[:10]}"
            )
        return features_df, "computed_from_windowed"

    if {"signal_normalized", "time", "run_id"}.issubset(set(input_df.columns)):
        windowed_df = build_windowed_timeseries(
            processed_df=input_df,
            window_size=window_size,
            step_size=step_size,
        )
        if windowed_df.empty:
            raise ValueError("windowed dataset is empty after processing")
        features_df = build_feature_table(windowed_df)
        if not required.issubset(set(features_df.columns)):
            missing = sorted(required - set(features_df.columns))
            raise ValueError(
                f"computed features missing required columns: {missing[:10]}"
            )
        return features_df, "computed_from_processed_timeseries"

    missing = sorted(required - set(input_df.columns))
    raise ValueError(
        "input does not contain model-ready features and lacks recognized raw/windowed signal columns; "
        f"missing required feature columns include: {missing[:10]}"
    )


def _predict_with_confidence(
    model: Any, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(model.predict(X))
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
        if proba.ndim == 2:
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.nan_to_num(proba.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        confidence = np.full(shape=(len(y_pred),), fill_value=np.nan, dtype=float)
    return y_pred, confidence


def _predict_stage2_probabilities(
    model: Any, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pred = np.asarray(model.predict(X))

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)

        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = np.arange(proba.shape[1])

        idx_toluene = None
        idx_butanone = None
        for idx, cls in enumerate(classes):
            cls_i = (
                int(cls)
                if isinstance(cls, (int, np.integer, float, np.floating))
                else None
            )
            if cls_i == 1:
                idx_toluene = idx
            if cls_i == 0:
                idx_butanone = idx

        if idx_toluene is None:
            idx_toluene = 1 if proba.shape[1] > 1 else 0
        if idx_butanone is None:
            idx_butanone = 0

        prob_toluene = proba[:, idx_toluene]
        prob_butanone = proba[:, idx_butanone]
        return y_pred, prob_toluene, prob_butanone

    pred_int = y_pred.astype(int)
    prob_toluene = np.where(pred_int == 1, 1.0, 0.0).astype(float)
    prob_butanone = np.where(pred_int == 0, 1.0, 0.0).astype(float)
    return y_pred, prob_toluene, prob_butanone


def _save_table(df: pd.DataFrame, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return "parquet"
    if suffix == ".csv":
        csv_df = df.copy()
        if "stage2_probabilities" in csv_df.columns:
            csv_df["stage2_probabilities"] = csv_df["stage2_probabilities"].map(
                lambda x: json.dumps(x) if isinstance(x, dict) else ""
            )
        csv_df.to_csv(output_path, index=False)
        return "csv"

    raise ValueError(f"unsupported output format: {suffix}; expected .parquet or .csv")


def run_two_stage_batch_prediction(
    features_path: Path,
    stage1_model_dir: Path,
    stage2_model_dir: Path,
    output_path: Path,
    summary_path: Path,
    window_size: int = 100,
    step_size: int = 50,
) -> dict[str, Any]:
    stage1_model, stage1_meta = _load_model_bundle(stage1_model_dir)
    stage2_model, stage2_meta = _load_model_bundle(stage2_model_dir)

    stage1_features = [str(c) for c in stage1_meta.get("feature_columns", [])]
    stage2_features = [str(c) for c in stage2_meta.get("feature_columns", [])]

    if not stage1_features:
        raise ValueError("stage1 metadata missing feature_columns")
    if not stage2_features:
        raise ValueError("stage2 metadata missing feature_columns")

    features_df, features_source = _load_or_compute_features(
        features_path=features_path,
        stage1_feature_columns=stage1_features,
        stage2_feature_columns=stage2_features,
        window_size=window_size,
        step_size=step_size,
    )

    X1 = features_df[stage1_features]
    X2 = features_df[stage2_features]

    stage1_pred, stage1_conf = _predict_with_confidence(stage1_model, X1)
    stage2_pred, stage2_prob_toluene, stage2_prob_butanone = (
        _predict_stage2_probabilities(stage2_model, X2)
    )

    class_labels = stage2_meta.get("class_labels", {"0": "2-butanone", "1": "Toluene"})

    stage1_is_single = stage1_pred.astype(int) == 0
    stage1_labels = np.where(stage1_is_single, "single_gas", "mixture")

    stage2_labels_full = np.array(
        [
            class_labels.get(str(int(v)), "Toluene" if int(v) == 1 else "2-butanone")
            for v in stage2_pred
        ],
        dtype=object,
    )

    result_df = features_df[
        [c for c in _METADATA_COLUMNS if c in features_df.columns]
    ].copy()
    result_df["stage1_prediction"] = stage1_pred.astype(int)
    result_df["stage1_label"] = stage1_labels
    result_df["stage1_confidence"] = stage1_conf.astype(float)

    result_df["stage2_prediction"] = np.where(
        stage1_is_single, stage2_pred.astype(int), np.nan
    )
    result_df["stage2_label"] = np.where(stage1_is_single, stage2_labels_full, None)
    result_df["stage2_prob_toluene"] = np.where(
        stage1_is_single, stage2_prob_toluene.astype(float), np.nan
    )
    result_df["stage2_prob_2_butanone"] = np.where(
        stage1_is_single, stage2_prob_butanone.astype(float), np.nan
    )

    result_df["stage2_probabilities"] = [
        (
            {
                "Toluene": float(tol),
                "2-butanone": float(but),
            }
            if bool(is_single)
            else None
        )
        for is_single, tol, but in zip(
            stage1_is_single, stage2_prob_toluene, stage2_prob_butanone
        )
    ]

    result_df["route"] = np.where(stage1_is_single, "stage1_stage2", "stage1_only")

    output_format = _save_table(result_df, output_path)

    storage_env = resolve_storage_environment()
    prediction_job = "batch-inference"
    job_date = _default_job_date()
    job_id = _default_batch_job_id()
    published_predictions = publish_local_artifact(
        local_path=output_path,
        domain="batch_predictions",
        env=storage_env,
        prediction_job=prediction_job,
        job_date=job_date,
        job_id=job_id,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "success",
        "rows": int(len(result_df)),
        "features_input_path": str(features_path),
        "features_source": features_source,
        "stage1_model_dir": str(stage1_model_dir),
        "stage2_model_dir": str(stage2_model_dir),
        "predictions_path": str(output_path),
        "predictions_format": output_format,
        "single_gas_predicted_rows": int(stage1_is_single.sum()),
        "mixture_predicted_rows": int((~stage1_is_single).sum()),
        "prediction_job": prediction_job,
        "job_date": job_date,
        "job_id": job_id,
        "storage_environment": storage_env,
        "predictions_storage_target": published_predictions["storage_target"],
        "predictions_storage_upload": published_predictions["storage_upload"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    published_summary = publish_local_artifact(
        local_path=summary_path,
        domain="reports",
        env=storage_env,
        pipeline="batch-inference",
        report_family="prediction-summary",
        run_id=job_id,
    )

    summary["summary_storage_target"] = published_summary["storage_target"]
    summary["summary_storage_upload"] = published_summary["storage_upload"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
