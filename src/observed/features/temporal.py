"""
Temporal feature extraction for detected events in the Observed pipeline.

This module extracts temporal features (peak value, duration, rise time, recovery time, AUC, baseline noise) for each detected event.
It reuses Tolouene scientific logic when available, and provides a robust fallback implementation.
Output is aligned with the canonical FeatureRecord and Parquet export contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from src.observed.utils.io import write_parquet

# Canonical schema and I/O helpers
from src.observed.utils.schema import FeatureRecord

# Try to import Tolouene's scientific metrics
try:
    from Tolouene.modules.module3_sensor_characterization import (
        compute_response_metrics,
    )

    _TOLOUENE_METRICS_AVAILABLE = True
except Exception as _import_exc:
    _TOLOUENE_METRICS_AVAILABLE = False
    _import_exc_msg = str(_import_exc)


def extract_temporal_features(
    signal: np.ndarray,
    timestamps: np.ndarray,
    events: Sequence[Any],
    run_id: str,
    baseline: float = 0.0,
    baseline_noise: float = np.nan,
) -> List[FeatureRecord]:
    """
    Extract temporal features for each detected event.
    Computes peak value, duration, rise time, recovery time, AUC, and baseline noise.
    Uses Tolouene's compute_response_metrics if available, otherwise falls back to a simple threshold-based method.
    Handles edge cases gracefully (returns NaN for invalid/short events).

    Args:
        signal: 1D numpy array of sensor signal (e.g., ΔADC values)
        timestamps: 1D numpy array of time axis (seconds)
        events: Sequence of event objects (must have event_id, run_id, start_idx, end_idx)
        run_id: Run identifier
        baseline: Baseline value to subtract from signal (default 0.0)
        baseline_noise: Baseline noise estimate (optional, default NaN)
    Returns:
        List of FeatureRecord objects, one per event.
    """
    # Convert inputs to arrays before validation
    signal = np.asarray(signal, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    # Input validation
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D numpy array")
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D numpy array")
    if len(signal) != len(timestamps):
        raise ValueError("signal and timestamps must have the same length")

    # Normalize baseline_noise to a float or NaN
    try:
        baseline_noise_val = float(baseline_noise)
    except Exception:
        baseline_noise_val = float("nan")

    features = []
    for event in events:
        # Robust event_id/run_id fallback
        event_id = getattr(event, "event_id", None)
        if event_id is None:
            event_id = getattr(event, "id", None)
        if event_id is None:
            event_id = str(getattr(event, "index", "unknown"))
        event_run_id = getattr(event, "run_id", None) or run_id
        # Defensive: ensure event boundaries are valid
        start_idx = int(getattr(event, "start_idx", 0))
        end_idx = int(getattr(event, "end_idx", -1))
        if end_idx < start_idx or start_idx < 0 or end_idx >= len(signal):
            features.append(
                FeatureRecord(
                    event_id=str(event_id),
                    run_id=str(event_run_id),
                    peak_value=np.nan,
                    auc=np.nan,
                    rise_time=np.nan,
                    recovery_time=np.nan,
                    duration=np.nan,
                    baseline_noise=baseline_noise_val,
                    mean_value=np.nan,
                    std_value=np.nan,
                    min_value=np.nan,
                    skewness=np.nan,
                    kurtosis=np.nan,
                    max_slope=np.nan,
                    mean_slope=np.nan,
                    slope_std=np.nan,
                    signal_energy=np.nan,
                    peak_duration_ratio=np.nan,
                    auc_duration_ratio=np.nan,
                )
            )
            continue
        event_signal = signal[start_idx : end_idx + 1] - baseline
        event_times = timestamps[start_idx : end_idx + 1]
        # Compute AUC (NumPy compatibility across versions)
        if len(event_signal) > 1:
            if hasattr(np, "trapezoid"):
                auc = float(np.trapezoid(event_signal, event_times))
            else:
                auc = float(np.trapz(event_signal, event_times))
        else:
            auc = np.nan
        # Compute peak value
        peak_value = float(np.max(event_signal)) if len(event_signal) > 0 else np.nan
        mean_value = float(np.mean(event_signal)) if len(event_signal) > 0 else np.nan
        std_value = (
            float(np.std(event_signal, ddof=0)) if len(event_signal) > 0 else np.nan
        )
        min_value = float(np.min(event_signal)) if len(event_signal) > 0 else np.nan
        # Compute duration
        duration = (
            float(event_times[-1] - event_times[0]) if len(event_times) > 1 else 0.0
        )

        # Shape features
        skewness = np.nan
        kurtosis = np.nan
        if len(event_signal) > 2 and np.isfinite(std_value) and std_value > 1e-12:
            centered = event_signal - mean_value
            m2 = float(np.mean(centered**2))
            if m2 > 1e-12:
                m3 = float(np.mean(centered**3))
                m4 = float(np.mean(centered**4))
                skewness = m3 / (m2**1.5)
                kurtosis = (m4 / (m2**2)) - 3.0

        # Dynamic features from first derivative
        max_slope = np.nan
        mean_slope = np.nan
        slope_std = np.nan
        if len(event_signal) > 1:
            dt = np.diff(event_times)
            ds = np.diff(event_signal)
            valid = dt > 0
            if np.any(valid):
                slopes = ds[valid] / dt[valid]
                max_slope = float(np.max(slopes))
                mean_slope = float(np.mean(slopes))
                slope_std = float(np.std(slopes, ddof=0))

        # Energy feature
        signal_energy = (
            float(np.sum(np.square(event_signal))) if len(event_signal) > 0 else np.nan
        )

        # Ratio features
        peak_duration_ratio = np.nan
        auc_duration_ratio = np.nan
        if duration > 0:
            peak_duration_ratio = (
                float(peak_value / duration) if np.isfinite(peak_value) else np.nan
            )
            auc_duration_ratio = float(auc / duration) if np.isfinite(auc) else np.nan
        # Try Tolouene metrics first
        rise_time = np.nan
        recovery_time = np.nan
        if _TOLOUENE_METRICS_AVAILABLE:
            try:
                metrics = compute_response_metrics(signal, timestamps, event, baseline)
                if metrics is not None:
                    rise_time = (
                        float(metrics.rise_time)
                        if metrics.rise_time is not None
                        else np.nan
                    )
                    recovery_time = (
                        float(metrics.recovery_time)
                        if metrics.recovery_time is not None
                        else np.nan
                    )
            except Exception:
                pass
        # Fallback: simple threshold-based rise/recovery time
        if (
            not _TOLOUENE_METRICS_AVAILABLE
            or np.isnan(rise_time)
            or np.isnan(recovery_time)
        ):
            try:
                if len(event_signal) > 2 and peak_value > 0:
                    thresh_10 = 0.10 * peak_value
                    thresh_90 = 0.90 * peak_value
                    above_10 = np.where(event_signal >= thresh_10)[0]
                    above_90 = np.where(event_signal >= thresh_90)[0]
                    if above_10.size > 0 and above_90.size > 0:
                        t10 = event_times[above_10[0]]
                        t90 = event_times[above_90[0]]
                        rise_time = float(t90 - t10) if t90 > t10 else 0.0
                    # Recovery time: time from 90% to 10% of peak (after peak)
                    peak_idx = np.argmax(event_signal)
                    after_peak = event_signal[peak_idx:]
                    after_times = event_times[peak_idx:]
                    below_90 = np.where(after_peak <= thresh_90)[0]
                    below_10 = np.where(after_peak <= thresh_10)[0]
                    if below_90.size > 0 and below_10.size > 0:
                        t90r = after_times[below_90[0]]
                        t10r = after_times[below_10[0]]
                        recovery_time = float(t10r - t90r) if t10r > t90r else 0.0
            except Exception:
                pass
        features.append(
            FeatureRecord(
                event_id=str(event_id),
                run_id=str(event_run_id),
                peak_value=peak_value,
                auc=auc,
                rise_time=rise_time,
                recovery_time=recovery_time,
                duration=duration,
                baseline_noise=baseline_noise_val,
                mean_value=mean_value,
                std_value=std_value,
                min_value=min_value,
                skewness=skewness,
                kurtosis=kurtosis,
                max_slope=max_slope,
                mean_slope=mean_slope,
                slope_std=slope_std,
                signal_energy=signal_energy,
                peak_duration_ratio=peak_duration_ratio,
                auc_duration_ratio=auc_duration_ratio,
            )
        )
    return features


def save_features_parquet(
    features: List[FeatureRecord],
    output_path: (
        str | Path
    ) = "data/processed/dat_runs/features_event_table_dat.parquet",
) -> Path:
    """
    Save feature records to a Parquet file using the shared pipeline helper.
    Args:
        features: List of FeatureRecord objects.
        output_path: Destination path for the Parquet file.
    Returns:
        Path to the written Parquet file.
    """
    import pandas as pd

    df = pd.DataFrame([f.to_dict() for f in features])
    return write_parquet(df, output_path)


def example_extract_and_save(
    signal: np.ndarray,
    timestamps: np.ndarray,
    events: Sequence[Any],
    run_id: str,
    baseline: float = 0.0,
    baseline_noise: float = np.nan,
    output_path: (
        str | Path
    ) = "data/processed/dat_runs/features_event_table_dat.parquet",
) -> Path:
    """
    Example integration: extract features and save to Parquet in one step.
    """
    features = extract_temporal_features(
        signal, timestamps, events, run_id, baseline, baseline_noise
    )
    return save_features_parquet(features, output_path)


__all__ = [
    "extract_temporal_features",
    "save_features_parquet",
    "example_extract_and_save",
]
