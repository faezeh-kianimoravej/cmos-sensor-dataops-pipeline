"""Schema standardization adapter for client-required tables.

This module adds a thin conversion layer on top of existing pipeline outputs
without modifying core parsing, detection, or modeling logic.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from src.observed.ingestion.parser import ParsedRun
from src.observed.utils.io import write_parquet

from .schema_models import (
    StandardEventRecord,
    StandardFeatureRecord,
    StandardLabelRecord,
    StandardRawFrameRecord,
    StandardRunRecord,
)


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        return value
    return value


def build_runs_table(parsed: ParsedRun) -> pd.DataFrame:
    row = StandardRunRecord(
        run_id=parsed.run_id,
        sensor_id=None,
        start_time=parsed.start_time,
        temp_c=None,
        rh_percent=None,
        sampling_rate_hz=parsed.sampling_rate_hz,
        duration_s=parsed.duration_s,
        flow_rate=None,
        pressure=None,
        experiment_type=None,
        operator=None,
        notes=None,
    )
    return pd.DataFrame([row.to_dict()])


def build_raw_frames_table(parsed: ParsedRun) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in parsed.frames:
        for layer_id, grid in frame.layer_grids.items():
            rows.append(
                StandardRawFrameRecord(
                    run_id=parsed.run_id,
                    timestamp=frame.timestamp,
                    layer_id=int(layer_id),
                    frame=grid.astype("int32").tolist(),
                ).to_dict()
            )

    if not rows:
        return pd.DataFrame(columns=["run_id", "timestamp", "layer_id", "frame"])

    return pd.DataFrame(rows)


def build_events_table(events: Iterable[Any], run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for event in events:
        event_index = int(getattr(event, "event_index", len(rows)))
        event_id = str(getattr(event, "event_id", f"{run_id}-evt-{event_index:03d}"))
        rows.append(
            StandardEventRecord(
                event_id=event_id,
                run_id=run_id,
                t_start=_none_if_nan(
                    getattr(event, "t_start", getattr(event, "start_time", None))
                ),
                t_end=_none_if_nan(
                    getattr(event, "t_end", getattr(event, "end_time", None))
                ),
                event_type=_none_if_nan(getattr(event, "event_type", None)),
                gas_label=None,
                target_ppm=None,
                confidence_score=_none_if_nan(
                    getattr(
                        event, "confidence_score", getattr(event, "confidence", None)
                    )
                ),
            ).to_dict()
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "run_id",
                "t_start",
                "t_end",
                "event_type",
                "gas_label",
                "target_ppm",
                "confidence_score",
            ]
        )

    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    value = _none_if_nan(value)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _event_frame_stack(
    parsed: ParsedRun, t_start: float, t_end: float, layer_id: int = 1
) -> np.ndarray | None:
    if not parsed.frames:
        return None
    run_start = parsed.start_time
    arrays: list[np.ndarray] = []
    for frame in parsed.frames:
        ts = (frame.timestamp - run_start).total_seconds()
        if ts < t_start or ts > t_end:
            continue
        grid = frame.layer_grids.get(layer_id)
        if grid is not None:
            arrays.append(np.asarray(grid, dtype=np.float64))
    if not arrays:
        return None
    return np.stack(arrays, axis=0)


def _compute_spatial_features(
    parsed: ParsedRun,
    t_start: float | None,
    t_end: float | None,
    baseline_map: np.ndarray | None,
) -> dict[str, float | None]:
    defaults = {
        "spatial_map_variance": None,
        "spatial_map_skewness": None,
        "spatial_hotspot_fraction": None,
        "spatial_pixel_corr_mean": None,
        "spatial_pca_component_1": None,
        "spatial_pca_component_2": None,
    }
    if t_start is None or t_end is None:
        return defaults

    stack = _event_frame_stack(parsed, t_start=t_start, t_end=t_end, layer_id=1)
    if stack is None:
        return defaults

    mean_map = np.mean(stack, axis=0)
    if baseline_map is not None:
        resp_map = mean_map - baseline_map
    else:
        resp_map = mean_map

    flat = resp_map.reshape(-1)
    map_var = float(np.var(flat))
    centered = flat - float(np.mean(flat))
    std = float(np.std(flat))
    skew = None
    if std > 1e-12:
        skew = float(np.mean((centered / std) ** 3))

    abs_flat = np.abs(flat)
    p90 = float(np.percentile(abs_flat, 90))
    hotspot_frac = float(np.mean(abs_flat >= p90)) if len(abs_flat) else None

    # Time x pixels matrix for compact spatial dynamics metrics.
    x = stack.reshape(stack.shape[0], -1)
    x = x - np.mean(x, axis=0, keepdims=True)

    pca1 = None
    pca2 = None
    if x.shape[0] >= 2 and x.shape[1] >= 2:
        try:
            _, svals, _ = np.linalg.svd(x, full_matrices=False)
            denom = float(np.sum(svals**2))
            if denom > 0:
                var_exp = (svals**2) / denom
                pca1 = float(var_exp[0])
                if len(var_exp) > 1:
                    pca2 = float(var_exp[1])
        except Exception:
            pass

    corr_mean = None
    if x.shape[0] >= 3 and x.shape[1] >= 4:
        pixel_std = np.std(x, axis=0)
        sel = np.argsort(pixel_std)[-min(32, x.shape[1]) :]
        xs = x[:, sel]
        corr = np.corrcoef(xs, rowvar=False)
        if corr.ndim == 2 and corr.shape[0] > 1:
            tri = np.triu_indices(corr.shape[0], k=1)
            vals = np.abs(corr[tri])
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                corr_mean = float(np.mean(finite))

    return {
        "spatial_map_variance": map_var,
        "spatial_map_skewness": skew,
        "spatial_hotspot_fraction": hotspot_frac,
        "spatial_pixel_corr_mean": corr_mean,
        "spatial_pca_component_1": pca1,
        "spatial_pca_component_2": pca2,
    }


def build_features_table(
    features_df: pd.DataFrame,
    *,
    parsed: ParsedRun,
    events_df: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if features_df.empty:
        return pd.DataFrame(
            columns=[
                field
                for field in asdict(
                    StandardFeatureRecord(
                        event_id="",
                        temporal_peak_value=None,
                        temporal_auc=None,
                        temporal_rise_time=None,
                        temporal_recovery_time=None,
                        temporal_duration=None,
                        temporal_mean_value=None,
                        temporal_std_value=None,
                        temporal_min_value=None,
                        temporal_skewness=None,
                        temporal_kurtosis=None,
                        temporal_max_slope=None,
                        temporal_mean_slope=None,
                        temporal_slope_std=None,
                        temporal_signal_energy=None,
                        temporal_peak_duration_ratio=None,
                        temporal_auc_duration_ratio=None,
                        spatial_map_variance=None,
                        spatial_map_skewness=None,
                        spatial_hotspot_fraction=None,
                        spatial_pixel_corr_mean=None,
                        spatial_pca_component_1=None,
                        spatial_pca_component_2=None,
                        spatial_coated_percentage=None,
                        spatial_top_half_coated=None,
                        spatial_bottom_half_coated=None,
                        stability_baseline_noise=None,
                        stability_dead_pixel_ratio=None,
                        stability_sensor_health_index=None,
                        stability_calibration_status=None,
                        stability_drift_slope=None,
                        stability_drift_per_hour=None,
                        stability_long_term_stability=None,
                    )
                ).keys()
            ]
        )

    events_meta = (
        events_df.set_index("event_id").to_dict(orient="index")
        if not events_df.empty
        else {}
    )

    baseline_map: np.ndarray | None = None
    if parsed.frames:
        n_base = max(1, int(len(parsed.frames) * 0.1))
        maps = []
        for frame in parsed.frames[:n_base]:
            grid = frame.layer_grids.get(1)
            if grid is not None:
                maps.append(np.asarray(grid, dtype=np.float64))
        if maps:
            baseline_map = np.mean(np.stack(maps, axis=0), axis=0)

    quality_row: dict[str, Any] = {}
    if (
        quality_df is not None
        and not quality_df.empty
        and "run_id" in quality_df.columns
    ):
        sub = quality_df[quality_df["run_id"].astype(str) == str(parsed.run_id)]
        if not sub.empty:
            quality_row = sub.iloc[-1].to_dict()

    dead_pixel_ratio = _safe_float(quality_row.get("dead_pixels_dead_pixel_ratio"))
    drift_per_hour = _safe_float(quality_row.get("drift_drift_per_hour"))
    drift_slope = _safe_float(quality_row.get("drift_drift_slope_per_sec"))

    rows: list[dict[str, Any]] = []
    for _, row in features_df.iterrows():
        event_id = str(row.get("event_id", ""))
        event_meta = events_meta.get(event_id, {})
        t_start = _safe_float(event_meta.get("t_start"))
        t_end = _safe_float(event_meta.get("t_end"))
        spatial = _compute_spatial_features(
            parsed, t_start=t_start, t_end=t_end, baseline_map=baseline_map
        )

        baseline_noise = _safe_float(row.get("baseline_noise"))
        dead_norm = (
            min((dead_pixel_ratio or 0.0) / 0.2, 1.0)
            if dead_pixel_ratio is not None
            else 1.0
        )
        drift_norm = (
            min(abs(drift_per_hour or 0.0) / 1.0, 1.0)
            if drift_per_hour is not None
            else 1.0
        )
        noise_norm = (
            min((baseline_noise or 0.0) / 0.2, 1.0)
            if baseline_noise is not None
            else 1.0
        )
        sensor_health = float(
            100.0 * (1.0 - (0.4 * dead_norm + 0.3 * drift_norm + 0.3 * noise_norm))
        )
        sensor_health = max(0.0, min(100.0, sensor_health))

        rows.append(
            StandardFeatureRecord(
                event_id=event_id,
                temporal_peak_value=_none_if_nan(row.get("peak_value")),
                temporal_auc=_none_if_nan(row.get("auc")),
                temporal_rise_time=_none_if_nan(row.get("rise_time")),
                temporal_recovery_time=_none_if_nan(row.get("recovery_time")),
                temporal_duration=_none_if_nan(row.get("duration")),
                temporal_mean_value=_none_if_nan(row.get("mean_value")),
                temporal_std_value=_none_if_nan(row.get("std_value")),
                temporal_min_value=_none_if_nan(row.get("min_value")),
                temporal_skewness=_none_if_nan(row.get("skewness")),
                temporal_kurtosis=_none_if_nan(row.get("kurtosis")),
                temporal_max_slope=_none_if_nan(row.get("max_slope")),
                temporal_mean_slope=_none_if_nan(row.get("mean_slope")),
                temporal_slope_std=_none_if_nan(row.get("slope_std")),
                temporal_signal_energy=_none_if_nan(row.get("signal_energy")),
                temporal_peak_duration_ratio=_none_if_nan(
                    row.get("peak_duration_ratio")
                ),
                temporal_auc_duration_ratio=_none_if_nan(row.get("auc_duration_ratio")),
                spatial_map_variance=_none_if_nan(spatial.get("spatial_map_variance")),
                spatial_map_skewness=_none_if_nan(spatial.get("spatial_map_skewness")),
                spatial_hotspot_fraction=_none_if_nan(
                    spatial.get("spatial_hotspot_fraction")
                ),
                spatial_pixel_corr_mean=_none_if_nan(
                    spatial.get("spatial_pixel_corr_mean")
                ),
                spatial_pca_component_1=_none_if_nan(
                    spatial.get("spatial_pca_component_1")
                ),
                spatial_pca_component_2=_none_if_nan(
                    spatial.get("spatial_pca_component_2")
                ),
                spatial_coated_percentage=None,
                spatial_top_half_coated=None,
                spatial_bottom_half_coated=None,
                stability_baseline_noise=_none_if_nan(baseline_noise),
                stability_dead_pixel_ratio=_none_if_nan(dead_pixel_ratio),
                stability_sensor_health_index=_none_if_nan(sensor_health),
                stability_calibration_status="unknown",
                stability_drift_slope=_none_if_nan(drift_slope),
                stability_drift_per_hour=_none_if_nan(drift_per_hour),
                stability_long_term_stability=_none_if_nan(
                    quality_row.get("drift_baseline_noise")
                ),
            ).to_dict()
        )

    return pd.DataFrame(rows)


def build_labels_table(event_ids: Iterable[str]) -> pd.DataFrame:
    rows = [
        StandardLabelRecord(
            event_id=str(event_id),
            disease_or_gas_class=None,
            severity_label=None,
            concentration_label=None,
            label_provenance=None,
        ).to_dict()
        for event_id in event_ids
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "disease_or_gas_class",
                "severity_label",
                "concentration_label",
                "label_provenance",
            ]
        )

    return pd.DataFrame(rows)


def export_standardized_tables(
    *,
    parsed: ParsedRun,
    events: Iterable[Any],
    features_path: str | Path,
    output_dir: str | Path = "data/processed",
    quality_audit_path: str | Path | None = None,
) -> Dict[str, str]:
    """Build and export standardized runs/raw_frames/events/features/labels tables."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = build_runs_table(parsed)
    raw_frames_df = build_raw_frames_table(parsed)
    events_list = list(events)
    events_df = build_events_table(events_list, parsed.run_id)

    features_df = pd.read_parquet(Path(features_path).resolve())
    quality_df: pd.DataFrame | None = None
    if quality_audit_path is not None:
        quality_path = Path(quality_audit_path).resolve()
        if quality_path.exists():
            quality_df = pd.read_parquet(quality_path)

    standardized_features_df = build_features_table(
        features_df,
        parsed=parsed,
        events_df=events_df,
        quality_df=quality_df,
    )

    labels_df = build_labels_table(
        events_df.get("event_id", pd.Series([], dtype=str)).tolist()
    )

    paths = {
        "runs": str(write_parquet(runs_df, out_dir / "runs.parquet")),
        "raw_frames": str(write_parquet(raw_frames_df, out_dir / "raw_frames.parquet")),
        "events": str(write_parquet(events_df, out_dir / "events.parquet")),
        "features": str(
            write_parquet(standardized_features_df, out_dir / "features.parquet")
        ),
        "labels": str(write_parquet(labels_df, out_dir / "labels.parquet")),
    }

    return paths


__all__ = [
    "build_runs_table",
    "build_raw_frames_table",
    "build_events_table",
    "build_features_table",
    "build_labels_table",
    "export_standardized_tables",
]
