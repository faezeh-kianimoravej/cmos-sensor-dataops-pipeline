"""Core structured schema records used across the pipeline.

These records define the internal contract for run metadata, detected events,
engineered features, and labels. The shape and field names are intentionally
stable so records can be serialized directly to Parquet tables later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True)
class RunRecord:
    """Canonical run-level metadata.

    Fields:
        run_id: Unique run identifier.
        sensor_id: Identifier for the sensor/source device.
        start_time: Timestamp of first frame in the run.
        end_time: Timestamp of last frame in the run.
        duration_s: Run duration in seconds.
        source_file: Source raw data filename/path.
        sampling_rate_hz: Effective sampling rate in Hz.
        data_version: Dataset/version tag for lineage.
    """

    run_id: str
    sensor_id: str
    start_time: datetime
    end_time: datetime
    duration_s: float
    source_file: str
    sampling_rate_hz: float
    data_version: str

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be on or after start_time")
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serialization-friendly dictionary."""
        return asdict(self)


@dataclass(slots=True)
class EventRecord:
    """Canonical event-level metadata.

    Fields:
        event_id: Unique event identifier.
        run_id: Parent run identifier.
        start_idx: Start frame index (inclusive).
        end_idx: End frame index (inclusive).
        start_time: Event start timestamp.
        end_time: Event end timestamp.
        peak_value: Peak signal value observed in event window.
        event_type: Event category label (for example exposure/recovery).
        confidence_score: Confidence in [0.0, 1.0].
    """

    event_id: str
    run_id: str
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    peak_value: float
    event_type: str
    confidence_score: float

    def __post_init__(self) -> None:
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if self.end_idx < self.start_idx:
            raise ValueError("end_idx must be greater than or equal to start_idx")
        if self.end_time < self.start_time:
            raise ValueError("end_time must be on or after start_time")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serialization-friendly dictionary."""
        return asdict(self)


@dataclass(slots=True)
class FeatureRecord:
    """Canonical feature row associated with one event.

    Fields:
        event_id: Parent event identifier.
        run_id: Parent run identifier.
        peak_value: Event peak signal magnitude.
        auc: Area under the event response curve.
        rise_time: Rise time in seconds.
        recovery_time: Recovery time in seconds.
        duration: Event duration in seconds.
        baseline_noise: Baseline noise estimate for the event/run.
        mean_value: Mean event signal magnitude.
        std_value: Standard deviation of event signal.
        min_value: Minimum event signal magnitude.
        skewness: Event signal skewness.
        kurtosis: Event signal kurtosis (excess).
        max_slope: Maximum first derivative value.
        mean_slope: Mean first derivative value.
        slope_std: Standard deviation of first derivative values.
        signal_energy: Sum of squared event signal values.
        peak_duration_ratio: Peak-to-duration ratio.
        auc_duration_ratio: AUC-to-duration ratio.
    """

    event_id: str
    run_id: str
    peak_value: float
    auc: float
    rise_time: float
    recovery_time: float
    duration: float
    baseline_noise: float
    mean_value: float = float("nan")
    std_value: float = float("nan")
    min_value: float = float("nan")
    skewness: float = float("nan")
    kurtosis: float = float("nan")
    max_slope: float = float("nan")
    mean_slope: float = float("nan")
    slope_std: float = float("nan")
    signal_energy: float = float("nan")
    peak_duration_ratio: float = float("nan")
    auc_duration_ratio: float = float("nan")

    def __post_init__(self) -> None:
        if self.rise_time < 0:
            raise ValueError("rise_time must be non-negative")
        if self.recovery_time < 0:
            raise ValueError("recovery_time must be non-negative")
        if self.duration < 0:
            raise ValueError("duration must be non-negative")
        if self.baseline_noise < 0:
            raise ValueError("baseline_noise must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serialization-friendly dictionary."""
        return asdict(self)


@dataclass(slots=True)
class LabelRecord:
    """Canonical label row associated with one event.

    Fields:
        event_id: Parent event identifier.
        run_id: Parent run identifier.
        gas_label: Gas class label (for example toluene).
        concentration_proxy: Numeric concentration proxy value, if available.
        severity_label: Severity class label.
        label_provenance: Label source/provenance marker.
        is_synthetic: Flag indicating synthetic/generated label.
    """

    event_id: str
    run_id: str
    gas_label: str
    concentration_proxy: Optional[float]
    severity_label: str
    label_provenance: str
    is_synthetic: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serialization-friendly dictionary."""
        return asdict(self)


__all__ = [
    "RunRecord",
    "EventRecord",
    "FeatureRecord",
    "LabelRecord",
]
