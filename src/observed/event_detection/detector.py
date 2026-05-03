"""
Observed event detection wrapper — integrates Tolouene EventDetector.

This module provides a wrapper around the Tolouene EventDetector to detect exposure events from filtered ΔADC signals.
It converts Tolouene's event outputs into a structured EventRecord format for the Observed pipeline and provides a function to save results as a Parquet file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Import Tolouene EventDetector and DetectedEvent
# This allows us to use the proven event detection logic from Tolouene.
try:
    from Tolouene.modules.module2_signal_processing import (
        DetectedEvent as _ToloueneDetectedEvent,
    )
    from Tolouene.modules.module2_signal_processing import (
        EventDetector as _ToloueneEventDetector,
    )

    _TOLOUENE_AVAILABLE = True
except Exception as _import_exc:
    _TOLOUENE_AVAILABLE = False
    _import_exc_msg = str(_import_exc)


# Structured event record for Observed pipeline
@dataclass
class EventRecord:
    event_id: str  # Stable event identifier
    run_id: str  # Unique identifier for the run
    event_index: int  # Index of the event in the run
    start_time: float  # Event start time (seconds from run start)
    end_time: float  # Event end time (seconds from run start)
    duration: float  # Duration of the event in seconds
    peak_value: float  # Maximum signal value in event
    mean_value: float  # Mean signal value in event
    integrated_value: float  # Area under the curve for event
    event_type: str  # baseline | exposure | recovery | unknown
    confidence: float  # Backward-compatible alias field
    confidence_score: float  # Schema-aligned confidence in [0, 1]

    @property
    def t_start(self) -> float:
        """Schema alias for event start time."""
        return self.start_time

    @property
    def t_end(self) -> float:
        """Schema alias for event end time."""
        return self.end_time


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _compute_confidence(
    ev: _ToloueneDetectedEvent, peak_ref: float, duration_ref: float
) -> float:
    """Simple explainable confidence based on strength, duration, and contrast."""
    peak_term = _clip01(float(ev.peak_value) / max(peak_ref, 1e-9))
    duration_term = _clip01(float(ev.duration) / max(duration_ref, 1e-9))

    contrast = float(abs(ev.peak_value - ev.mean_value))
    contrast_term = _clip01(contrast / max(abs(ev.peak_value), 1e-9))

    return _clip01(0.5 * peak_term + 0.3 * duration_term + 0.2 * contrast_term)


def _classify_gap_state(
    signal: np.ndarray,
    timestamps: np.ndarray,
    start_time: float,
    end_time: float,
    baseline_noise: float,
) -> str:
    """Classify non-exposure intervals as baseline or recovery.

    Why:
    Labeling gaps explicitly improves downstream interpretability and supports
    timeline-aware analytics beyond pure exposure detection.

    Assumptions:
    - Residual activity can be approximated by interval standard deviation.

    Edge cases:
    - Invalid time ranges or very short segments return ``unknown``.
    """
    if end_time <= start_time:
        return "unknown"

    t = np.asarray(timestamps, dtype=np.float64)
    s = np.asarray(signal, dtype=np.float64)

    start_idx = int(np.searchsorted(t, start_time, side="left"))
    end_idx = int(np.searchsorted(t, end_time, side="right") - 1)
    start_idx = max(0, min(start_idx, len(s) - 1))
    end_idx = max(start_idx, min(end_idx, len(s) - 1))

    seg = s[start_idx : end_idx + 1]
    if len(seg) < 3:
        return "unknown"

    activity = float(np.std(seg))
    threshold = max(2.0 * baseline_noise, 1e-6)
    if activity > threshold:
        return "recovery"
    return "baseline"


def _build_state_augmented_events(
    raw_events: list[_ToloueneDetectedEvent],
    signal: np.ndarray,
    timestamps: np.ndarray,
    run_id: str,
) -> list[EventRecord]:
    """Convert detections into schema-aligned exposure and gap-state events.

    Why:
    Producing a complete event timeline (exposure plus non-exposure states)
    keeps downstream feature extraction and quality analysis consistent.

    Edge cases:
    - Empty timestamps return an empty list.
    - Empty detections still produce baseline/recovery intervals when possible.
    """
    if len(timestamps) == 0:
        return []

    t = np.asarray(timestamps, dtype=np.float64)
    s = np.asarray(signal, dtype=np.float64)

    # Baseline activity estimate from lower-response portion of the run.
    baseline_mask = s <= np.percentile(s, 25)
    baseline_noise = (
        float(np.std(s[baseline_mask])) if np.any(baseline_mask) else float(np.std(s))
    )

    peak_ref = float(
        max((ev.peak_value for ev in raw_events), default=float(np.max(np.abs(s))))
    )
    duration_ref = float(
        max((ev.duration for ev in raw_events), default=max(t[-1] - t[0], 1.0))
    )

    records: list[EventRecord] = []
    idx = 0
    cursor = float(t[0])

    # Ensure detector outputs are ordered for stable state-interval construction.
    sorted_events = sorted(raw_events, key=lambda ev: float(ev.start_time))

    for ev in sorted_events:
        ev_start = float(ev.start_time)
        ev_end = float(ev.end_time)

        if ev_start > cursor:
            gap_type = _classify_gap_state(s, t, cursor, ev_start, baseline_noise)
            gap_conf = (
                0.8
                if gap_type == "baseline"
                else 0.7 if gap_type == "recovery" else 0.5
            )
            records.append(
                EventRecord(
                    event_id=f"{run_id}-evt-{idx:03d}",
                    run_id=run_id,
                    event_index=idx,
                    start_time=cursor,
                    end_time=ev_start,
                    duration=float(max(0.0, ev_start - cursor)),
                    peak_value=float(np.nanmax(s)) if len(s) else 0.0,
                    mean_value=float(np.nanmean(s)) if len(s) else 0.0,
                    integrated_value=0.0,
                    event_type=gap_type,
                    confidence=gap_conf,
                    confidence_score=gap_conf,
                )
            )
            idx += 1

        exposure_conf = _compute_confidence(
            ev, peak_ref=peak_ref, duration_ref=duration_ref
        )
        records.append(
            EventRecord(
                event_id=f"{run_id}-evt-{idx:03d}",
                run_id=run_id,
                event_index=idx,
                start_time=ev_start,
                end_time=ev_end,
                duration=float(ev.duration),
                peak_value=float(ev.peak_value),
                mean_value=float(ev.mean_value),
                integrated_value=float(ev.integrated_value),
                event_type="exposure",
                confidence=exposure_conf,
                confidence_score=exposure_conf,
            )
        )
        idx += 1
        cursor = max(cursor, ev_end)

    if cursor < float(t[-1]):
        tail_type = _classify_gap_state(s, t, cursor, float(t[-1]), baseline_noise)
        tail_conf = (
            0.8 if tail_type == "baseline" else 0.7 if tail_type == "recovery" else 0.5
        )
        records.append(
            EventRecord(
                event_id=f"{run_id}-evt-{idx:03d}",
                run_id=run_id,
                event_index=idx,
                start_time=cursor,
                end_time=float(t[-1]),
                duration=float(max(0.0, float(t[-1]) - cursor)),
                peak_value=float(np.nanmax(s)) if len(s) else 0.0,
                mean_value=float(np.nanmean(s)) if len(s) else 0.0,
                integrated_value=0.0,
                event_type=tail_type,
                confidence=tail_conf,
                confidence_score=tail_conf,
            )
        )

    return records


def detect_events(
    signal: np.ndarray,
    timestamps: np.ndarray,
    run_id: str,
    detector: _ToloueneEventDetector | None = None,
) -> list[EventRecord]:
    """Detect exposure events and emit schema-aligned timeline records.

    Why:
    This wrapper reuses proven Tolouene detection while normalizing output to
    the project EventRecord contract used by ingestion and feature pipelines.

    Assumptions:
    - Input arrays are aligned 1D vectors of equal length.

    Edge cases:
    - Empty signals return an empty list.
    - Missing Tolouene dependency raises ``ImportError``.
    """
    if not _TOLOUENE_AVAILABLE:
        raise ImportError(f"Tolouene EventDetector not available: {_import_exc_msg}")
    signal = np.asarray(signal, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if signal.ndim != 1 or timestamps.ndim != 1 or len(signal) != len(timestamps):
        raise ValueError("signal and timestamps must be 1D arrays with equal length")
    if len(signal) == 0:
        return []

    if detector is None:
        detector = _ToloueneEventDetector()
    events = detector.detect(signal, timestamps)
    return _build_state_augmented_events(events, signal, timestamps, run_id)


def save_events_parquet(events: list[EventRecord], output_path: str | Path):
    """Write detected events to parquet with schema-compatible aliases.

    Why:
    Parquet outputs are the interchange format for downstream training and
    evaluation jobs.

    Edge cases:
    - Empty event lists write an empty parquet table with inferred columns.
    """
    rows = []
    for e in events:
        row = dict(e.__dict__)
        # Schema-aligned aliases for standardized event table compatibility.
        row["t_start"] = e.t_start
        row["t_end"] = e.t_end
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)


__all__ = ["EventRecord", "detect_events", "save_events_parquet"]
