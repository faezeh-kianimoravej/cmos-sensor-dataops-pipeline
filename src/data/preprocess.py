from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.parser import load_vector_file


def _load_raw_numeric_column(file_path: Path) -> pd.Series | None:
    """Load first column as numeric while preserving NaN positions for pairwise cleaning."""
    df = None
    for sep in [",", "\t", r"\s+"]:
        try:
            if sep == r"\s+":
                temp_df = pd.read_csv(file_path, sep=sep, engine="python", header=None)
            else:
                temp_df = pd.read_csv(file_path, delimiter=sep, header=None)
            if temp_df.shape[0] >= 10:
                df = temp_df
                break
        except Exception:
            continue

    if df is None:
        return None

    return pd.to_numeric(df.iloc[:, 0], errors="coerce").reset_index(drop=True)


def _smooth_signal(signal: np.ndarray, window_size: int = 7) -> np.ndarray:
    if len(signal) < window_size:
        return signal.astype(float)
    series = pd.Series(signal.astype(float))
    return (
        series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    )


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    min_v = float(np.min(signal))
    max_v = float(np.max(signal))
    denom = max_v - min_v
    if denom <= 0:
        return np.zeros_like(signal, dtype=float)
    return (signal - min_v) / denom


def build_processed_timeseries(runs_df: pd.DataFrame) -> pd.DataFrame:
    processed_rows: list[dict[str, Any]] = []

    for row in runs_df.itertuples(index=False):
        # Keep original validation behavior to skip fully invalid files.
        adc_values = load_vector_file(Path(row.adc_file))
        time_values = load_vector_file(Path(row.time_file))
        if adc_values is None or time_values is None:
            continue

        # Pairwise clean to avoid misalignment when only one side has missing values.
        adc_raw = _load_raw_numeric_column(Path(row.adc_file))
        time_raw = _load_raw_numeric_column(Path(row.time_file))
        if adc_raw is None or time_raw is None:
            continue

        min_len_raw = min(len(adc_raw), len(time_raw))
        if min_len_raw < 10:
            continue

        paired_df = pd.DataFrame(
            {
                "time": time_raw.iloc[:min_len_raw].reset_index(drop=True),
                "signal": adc_raw.iloc[:min_len_raw].reset_index(drop=True),
            }
        )
        paired_df = paired_df.dropna(subset=["time", "signal"]).reset_index(drop=True)
        if len(paired_df) < 10 or paired_df["signal"].nunique() < 3:
            continue

        signal = paired_df["signal"].to_numpy(dtype=float)
        time_axis = paired_df["time"].to_numpy(dtype=float)

        # Rebase time and enforce monotonicity.
        time_axis = time_axis - float(time_axis[0])
        if np.any(np.diff(time_axis) < 0) or not np.all(np.isfinite(time_axis)):
            continue

        smoothed = _smooth_signal(signal)
        normalized = _normalize_signal(smoothed)

        processed_rows.append(
            {
                "run_id": row.run_id,
                "experiment": row.experiment,
                "experiment_folder": row.experiment_folder,
                "run_folder": row.run_folder,
                "repeat_index": row.repeat_index,
                "time": time_axis.tolist(),
                "signal": signal.tolist(),
                "signal_smoothed": smoothed.tolist(),
                "signal_normalized": normalized.tolist(),
                "n_samples": int(len(signal)),
            }
        )

    return pd.DataFrame(processed_rows)


def build_windowed_timeseries(
    processed_df: pd.DataFrame,
    window_size: int = 100,
    step_size: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in processed_df.itertuples(index=False):
        signal = np.asarray(row.signal_normalized, dtype=float)
        time_axis = np.asarray(row.time, dtype=float)

        valid_mask = np.isfinite(signal) & np.isfinite(time_axis)
        signal = signal[valid_mask]
        time_axis = time_axis[valid_mask]

        if len(signal) < window_size:
            continue

        for start_idx in range(0, len(signal) - window_size + 1, step_size):
            end_idx = start_idx + window_size - 1
            signal_window = signal[start_idx : end_idx + 1]
            time_window = time_axis[start_idx : end_idx + 1]
            if len(signal_window) != window_size or len(time_window) != window_size:
                continue
            if (
                not np.isfinite(signal_window).all()
                or not np.isfinite(time_window).all()
            ):
                continue

            rows.append(
                {
                    "run_id": row.run_id,
                    "experiment": row.experiment,
                    "experiment_folder": row.experiment_folder,
                    "run_folder": row.run_folder,
                    "repeat_index": row.repeat_index,
                    "window_id": f"{row.run_id}_w{start_idx}",
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "time_start": float(time_window[0]),
                    "time_end": float(time_window[-1]),
                    "signal_window": signal_window.tolist(),
                }
            )

    return pd.DataFrame(rows)


def split_windowed_by_run(
    windowed_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if windowed_df.empty:
        return windowed_df.copy(), windowed_df.copy()

    groups = windowed_df["run_id"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(windowed_df, groups=groups))
    train_df = windowed_df.iloc[train_idx].reset_index(drop=True)
    test_df = windowed_df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df
