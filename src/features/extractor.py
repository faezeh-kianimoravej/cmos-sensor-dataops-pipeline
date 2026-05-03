from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def _clean_signal(signal_like: np.ndarray | list[float]) -> np.ndarray:
    """Return finite numeric signal values; empty array when unusable."""
    signal = np.asarray(signal_like, dtype=float)
    if signal.ndim != 1:
        signal = signal.ravel()
    signal = signal[np.isfinite(signal)]
    return signal


def extract_shape_features(signal: np.ndarray) -> dict[str, float | int]:
    x = np.arange(len(signal), dtype=float)
    slope, _ = np.polyfit(x, signal, 1)

    trend_threshold = 1e-3
    if slope > trend_threshold:
        trend_type = 1
    elif slope < -trend_threshold:
        trend_type = -1
    else:
        trend_type = 0

    peak_position = int(np.argmax(signal))
    peak_value = float(np.max(signal))

    rise_strength = float(peak_value - signal[0])
    fall_strength = float(peak_value - signal[-1])

    return {
        "trend_slope": float(slope),
        "trend_type": int(trend_type),
        "peak_value": peak_value,
        "peak_position": peak_position,
        "rise_strength": rise_strength,
        "fall_strength": fall_strength,
    }


def extract_stat_features(signal: np.ndarray) -> dict[str, float]:
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    q25 = float(np.percentile(signal, 25))
    q75 = float(np.percentile(signal, 75))

    return {
        "mean": mean_val,
        "std": std_val,
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "median": float(np.median(signal)),
        "range": float(np.max(signal) - np.min(signal)),
        "iqr": float(q75 - q25),
        "cv": float(std_val / mean_val) if mean_val != 0 else 0.0,
    }


def extract_temporal_features(signal: np.ndarray) -> dict[str, float]:
    first_diff = np.diff(signal)
    if len(first_diff) == 0:
        return {
            "first_diff_mean": 0.0,
            "first_diff_std": 0.0,
            "first_diff_max": 0.0,
            "first_diff_min": 0.0,
            "abs_diff_mean": 0.0,
            "sign_changes": 0.0,
            "diff_sign_changes": 0.0,
        }

    diff_sign = np.sign(first_diff)
    sign_changes = np.sum(diff_sign[:-1] != diff_sign[1:]) if len(diff_sign) > 1 else 0

    return {
        "first_diff_mean": float(np.mean(first_diff)),
        "first_diff_std": float(np.std(first_diff)),
        "first_diff_max": float(np.max(first_diff)),
        "first_diff_min": float(np.min(first_diff)),
        "abs_diff_mean": float(np.mean(np.abs(first_diff))),
        "sign_changes": float(sign_changes),
        "diff_sign_changes": float(sign_changes),
    }


def extract_variability_features(signal: np.ndarray) -> dict[str, float | int]:
    diff = np.diff(signal)
    net_change = float(signal[-1] - signal[0])
    energy = float(np.sum(signal**2))

    flatness_threshold = 1e-3
    flatness = (
        float(np.mean(np.abs(diff) < flatness_threshold)) if len(diff) > 0 else 0.0
    )

    diff_sign = np.sign(diff)
    zcr_diff = (
        float(np.sum(diff_sign[:-1] != diff_sign[1:]) / len(diff_sign[1:]))
        if len(diff_sign) > 1
        else 0.0
    )

    peak_position = int(np.argmax(signal))

    return {
        "net_change": net_change,
        "energy": energy,
        "skewness": float(np.nan_to_num(skew(signal), nan=0.0)),
        "kurtosis": float(np.nan_to_num(kurtosis(signal), nan=0.0)),
        "flatness": flatness,
        "zero_crossing_rate_diff": zcr_diff,
        "signal_start": float(signal[0]),
        "signal_end": float(signal[-1]),
        "peak_to_start_distance": int(peak_position),
        "peak_to_end_distance": int(len(signal) - 1 - peak_position),
        "relative_peak_position": (
            float(peak_position / (len(signal) - 1)) if len(signal) > 1 else 0.0
        ),
    }


def build_feature_table(windowed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in windowed_df.itertuples(index=False):
        signal = _clean_signal(row.signal_window)
        if len(signal) < 2:
            # Drop unusable rows rather than failing the pipeline.
            continue

        shape = extract_shape_features(signal)
        stat = extract_stat_features(signal)
        temporal = extract_temporal_features(signal)
        variability = extract_variability_features(signal)

        merged = {
            "run_id": row.run_id,
            "experiment": row.experiment,
            "experiment_folder": row.experiment_folder,
            "run_folder": row.run_folder,
            "repeat_index": row.repeat_index,
            "window_id": row.window_id,
            "start_idx": row.start_idx,
            "end_idx": row.end_idx,
            "time_start": row.time_start,
            "time_end": row.time_end,
        }
        merged.update(shape)
        merged.update(stat)
        merged.update(temporal)
        merged.update(variability)
        rows.append(merged)

    metadata_cols = [
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
    if not rows:
        return pd.DataFrame(columns=metadata_cols)

    features_df = pd.DataFrame(rows)
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]

    # Keep pipeline resilient: coerce invalid numeric values and fill missing feature values.
    features_df[feature_cols] = features_df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    if not features_df.empty:
        medians = features_df[feature_cols].median(numeric_only=True)
        features_df[feature_cols] = features_df[feature_cols].fillna(medians)
        features_df[feature_cols] = features_df[feature_cols].fillna(0.0)

    features_df = features_df[metadata_cols + feature_cols]
    return features_df
