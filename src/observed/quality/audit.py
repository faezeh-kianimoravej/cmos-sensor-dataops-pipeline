"""Data quality audit utilities for ingestion and preprocessing outputs.

Why:
These checks provide fast, explainable run-level quality signals before model
training and evaluation, reducing time spent debugging low-quality inputs.

Assumptions:
- Parsed runs expose frame-level data and timestamps.
- Quality checks should not alter core parsing or model behavior.

Edge cases:
- Sparse or malformed inputs return conservative default audit values instead
    of raising non-essential exceptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TimestampAudit:
    expected_interval_s: float
    n_gaps: int
    estimated_missing_timestamps: int
    missing_ratio: float
    interpolation_strategy: str
    used_interpolated_time_axis: bool


@dataclass(slots=True)
class DeadPixelAudit:
    layer_id: int
    dead_pixel_count: int
    dead_pixel_ratio: float
    variance_threshold: float
    dynamic_range_threshold: float


@dataclass(slots=True)
class DriftAudit:
    baseline_shift: float
    drift_per_hour: float
    drift_slope_per_sec: float
    baseline_noise: float
    drift_flag: bool


@dataclass(slots=True)
class RunQualityAudit:
    run_id: str
    timestamp: TimestampAudit
    dead_pixels: DeadPixelAudit
    drift: DriftAudit
    outlier_flag: bool
    outlier_reasons: list[str]

    def to_flat_dict(self) -> Dict[str, Any]:
        row = {
            "run_id": self.run_id,
            "outlier_flag": self.outlier_flag,
            "outlier_reasons": (
                "; ".join(self.outlier_reasons) if self.outlier_reasons else ""
            ),
        }
        row.update({f"timestamp_{k}": v for k, v in asdict(self.timestamp).items()})
        row.update({f"dead_pixels_{k}": v for k, v in asdict(self.dead_pixels).items()})
        row.update({f"drift_{k}": v for k, v in asdict(self.drift).items()})
        return row


def audit_timestamps_and_interpolate(
    time_seconds: np.ndarray,
) -> tuple[np.ndarray, TimestampAudit]:
    """Detect timestamp gaps and return a regularized axis by interpolation strategy.

    Why:
    Sensor logs may contain dropped or duplicated timestamps. A regularized axis
    preserves sequence length while making time-dependent features consistent.

    Assumptions:
    - Input is a monotonic elapsed-time vector under normal operation.

    Edge cases:
    - Non-positive intervals or single-sample signals bypass interpolation.
    - Empty or non-1D arrays raise ``ValueError``.
    """
    axis = np.asarray(time_seconds, dtype=np.float64)
    if axis.ndim != 1 or len(axis) == 0:
        raise ValueError("time_seconds must be a non-empty 1D array")

    if len(axis) == 1:
        audit = TimestampAudit(
            expected_interval_s=0.0,
            n_gaps=0,
            estimated_missing_timestamps=0,
            missing_ratio=0.0,
            interpolation_strategy="none-single-sample",
            used_interpolated_time_axis=False,
        )
        return axis, audit

    diffs = np.diff(axis)
    positive = diffs[diffs > 0]
    expected = float(np.median(positive)) if len(positive) else 0.0

    if expected <= 0.0:
        audit = TimestampAudit(
            expected_interval_s=0.0,
            n_gaps=0,
            estimated_missing_timestamps=0,
            missing_ratio=0.0,
            interpolation_strategy="none-nonpositive-interval",
            used_interpolated_time_axis=False,
        )
        return axis, audit

    gap_idx = np.where(diffs > 1.5 * expected)[0]
    n_gaps = int(len(gap_idx))

    estimated_missing = 0
    for idx in gap_idx:
        estimate = int(round(diffs[idx] / expected)) - 1
        if estimate > 0:
            estimated_missing += estimate

    denom = (
        float(len(axis) + estimated_missing)
        if len(axis) + estimated_missing > 0
        else 1.0
    )
    missing_ratio = float(estimated_missing / denom)

    regularized = axis[0] + np.arange(len(axis), dtype=np.float64) * expected
    use_interpolated = bool(n_gaps > 0)
    output_axis = regularized if use_interpolated else axis

    audit = TimestampAudit(
        expected_interval_s=expected,
        n_gaps=n_gaps,
        estimated_missing_timestamps=int(estimated_missing),
        missing_ratio=missing_ratio,
        interpolation_strategy=(
            "regularized-time-axis-by-index" if use_interpolated else "none"
        ),
        used_interpolated_time_axis=use_interpolated,
    )
    return output_axis, audit


def detect_dead_pixels(
    parsed_run: Any,
    *,
    layer_id: int = 1,
    variance_threshold: float = 1.0,
    dynamic_range_threshold: float = 3.0,
) -> DeadPixelAudit:
    """Detect dead pixels across 32x32 frames for a given layer.

    Why:
    Persistently inactive pixels distort spatial features and can bias models.

    Assumptions:
    - Layer grids are shape-consistent across frames.

    Edge cases:
    - Missing frames or missing layer data return zero dead-pixel ratio.
    - Thresholds are conservative defaults and may need sensor-specific tuning.
    """
    frames = getattr(parsed_run, "frames", [])
    if not frames:
        return DeadPixelAudit(
            layer_id=layer_id,
            dead_pixel_count=0,
            dead_pixel_ratio=0.0,
            variance_threshold=variance_threshold,
            dynamic_range_threshold=dynamic_range_threshold,
        )

    layer_arrays: list[np.ndarray] = []
    for frame in frames:
        grid = frame.layer_grids.get(layer_id)
        if grid is not None:
            layer_arrays.append(np.asarray(grid, dtype=np.float32))

    if not layer_arrays:
        return DeadPixelAudit(
            layer_id=layer_id,
            dead_pixel_count=0,
            dead_pixel_ratio=0.0,
            variance_threshold=variance_threshold,
            dynamic_range_threshold=dynamic_range_threshold,
        )

    stack = np.stack(layer_arrays, axis=0)
    pixel_var = np.var(stack, axis=0)
    pixel_range = np.ptp(stack, axis=0)
    dead_mask = (pixel_var <= variance_threshold) & (
        pixel_range <= dynamic_range_threshold
    )

    dead_count = int(np.sum(dead_mask))
    total = int(dead_mask.size)
    ratio = float(dead_count / total) if total > 0 else 0.0

    return DeadPixelAudit(
        layer_id=layer_id,
        dead_pixel_count=dead_count,
        dead_pixel_ratio=ratio,
        variance_threshold=variance_threshold,
        dynamic_range_threshold=dynamic_range_threshold,
    )


def detect_drift(signal: np.ndarray, time_seconds: np.ndarray) -> DriftAudit:
    """Estimate baseline drift from early-vs-late signal behavior.

    Why:
    Baseline drift can indicate sensor instability and should be surfaced before
    downstream interpretation.

    Assumptions:
    - Drift is approximated by median shift and linear trend over elapsed time.

    Edge cases:
    - Too few samples or mismatched array lengths return a neutral audit.
    """
    y = np.asarray(signal, dtype=np.float64)
    t = np.asarray(time_seconds, dtype=np.float64)

    if len(y) < 5 or len(t) != len(y):
        return DriftAudit(
            baseline_shift=0.0,
            drift_per_hour=0.0,
            drift_slope_per_sec=0.0,
            baseline_noise=0.0,
            drift_flag=False,
        )

    n = len(y)
    k = max(1, int(n * 0.1))
    early = y[:k]
    late = y[-k:]

    baseline_shift = float(np.median(late) - np.median(early))
    baseline_noise = float(np.std(early))

    duration_s = float(max(t[-1] - t[0], 1e-9))
    drift_per_hour = float(baseline_shift / duration_s * 3600.0)

    slope = 0.0
    if len(t) >= 2 and np.ptp(t) > 0:
        slope = float(np.polyfit(t, y, 1)[0])

    threshold = max(0.1, 3.0 * baseline_noise)
    drift_flag = bool(abs(baseline_shift) > threshold)

    return DriftAudit(
        baseline_shift=baseline_shift,
        drift_per_hour=drift_per_hour,
        drift_slope_per_sec=slope,
        baseline_noise=baseline_noise,
        drift_flag=drift_flag,
    )


def flag_outlier_run(
    timestamp_audit: TimestampAudit, dead_pixels: DeadPixelAudit, drift: DriftAudit
) -> tuple[bool, list[str]]:
    """Flag outlier runs using explicit, non-destructive criteria."""
    reasons: list[str] = []
    if timestamp_audit.missing_ratio > 0.02:
        reasons.append("missing_timestamp_ratio_gt_2pct")
    if dead_pixels.dead_pixel_ratio > 0.20:
        reasons.append("dead_pixel_ratio_gt_20pct")
    if drift.drift_flag:
        reasons.append("baseline_drift_flag")
    return bool(reasons), reasons


def audit_run_quality(
    *,
    run_id: str,
    parsed_run: Any,
    signal: np.ndarray,
    time_seconds: np.ndarray,
) -> tuple[np.ndarray, RunQualityAudit]:
    """Run all quality checks and return (time_axis_for_pipeline, audit)."""
    regularized_time, ts_audit = audit_timestamps_and_interpolate(time_seconds)
    dead_pixels = detect_dead_pixels(parsed_run, layer_id=1)
    drift = detect_drift(signal, regularized_time)
    outlier_flag, reasons = flag_outlier_run(ts_audit, dead_pixels, drift)

    audit = RunQualityAudit(
        run_id=run_id,
        timestamp=ts_audit,
        dead_pixels=dead_pixels,
        drift=drift,
        outlier_flag=outlier_flag,
        outlier_reasons=reasons,
    )
    return regularized_time, audit


def append_quality_audit_parquet(
    audit: RunQualityAudit, output_path: str | Path
) -> Path:
    """Append/overwrite run-level quality audit row in parquet by run_id."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([audit.to_flat_dict()])
    if out.exists():
        existing = pd.read_parquet(out)
        merged = pd.concat([existing, new_row], ignore_index=True)
        if "run_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["run_id"], keep="last")
        merged.to_parquet(out, index=False)
    else:
        new_row.to_parquet(out, index=False)

    return out


def build_leakage_safe_groups(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Build group keys for leakage-safe splitting by run/day/sensor when possible."""
    if df.empty:
        return None, "none"

    if "run_id" not in df.columns:
        return None, "none"

    run_part = df["run_id"].fillna("unknown_run").astype(str)

    if "sensor_id" in df.columns:
        sensor_part = df["sensor_id"].fillna("unknown_sensor").astype(str)
    else:
        sensor_part = pd.Series(["unknown_sensor"] * len(df), index=df.index, dtype=str)

    day_part: pd.Series
    if "start_time" in df.columns:
        parsed = pd.to_datetime(df["start_time"], errors="coerce")
        day_part = parsed.dt.date.astype(str).replace("NaT", "unknown_day")
    elif "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        day_part = parsed.dt.date.astype(str).replace("NaT", "unknown_day")
    else:
        day_part = pd.Series(["unknown_day"] * len(df), index=df.index, dtype=str)

    groups = run_part + "|" + day_part + "|" + sensor_part
    return groups, "run_day_sensor"


__all__ = [
    "RunQualityAudit",
    "audit_run_quality",
    "append_quality_audit_parquet",
    "build_leakage_safe_groups",
]
