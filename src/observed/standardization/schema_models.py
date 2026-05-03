"""Standardized client-facing schema models.

Why:
These dataclasses provide a stable contract for exported tables so ingestion,
feature generation, evaluation, and API layers can exchange data reliably.

Assumptions:
- Optional fields may be unavailable for some datasets and remain ``None``.

Edge cases:
- Serialization keeps field names unchanged even when values are null, which
    simplifies downstream joins and schema validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True)
class StandardRunRecord:
    run_id: str
    sensor_id: Optional[str]
    start_time: Optional[datetime]
    temp_c: Optional[float]
    rh_percent: Optional[float]
    sampling_rate_hz: Optional[float]
    duration_s: Optional[float]
    flow_rate: Optional[float]
    pressure: Optional[float]
    experiment_type: Optional[str]
    operator: Optional[str]
    notes: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StandardRawFrameRecord:
    run_id: str
    timestamp: Optional[datetime]
    layer_id: Optional[int]
    frame: Optional[list[list[int]]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StandardEventRecord:
    event_id: str
    run_id: str
    t_start: Optional[float]
    t_end: Optional[float]
    event_type: Optional[str]
    gas_label: Optional[str]
    target_ppm: Optional[float]
    confidence_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StandardFeatureRecord:
    event_id: str
    temporal_peak_value: Optional[float]
    temporal_auc: Optional[float]
    temporal_rise_time: Optional[float]
    temporal_recovery_time: Optional[float]
    temporal_duration: Optional[float]
    temporal_mean_value: Optional[float]
    temporal_std_value: Optional[float]
    temporal_min_value: Optional[float]
    temporal_skewness: Optional[float]
    temporal_kurtosis: Optional[float]
    temporal_max_slope: Optional[float]
    temporal_mean_slope: Optional[float]
    temporal_slope_std: Optional[float]
    temporal_signal_energy: Optional[float]
    temporal_peak_duration_ratio: Optional[float]
    temporal_auc_duration_ratio: Optional[float]
    spatial_map_variance: Optional[float]
    spatial_map_skewness: Optional[float]
    spatial_hotspot_fraction: Optional[float]
    spatial_pixel_corr_mean: Optional[float]
    spatial_pca_component_1: Optional[float]
    spatial_pca_component_2: Optional[float]
    spatial_coated_percentage: Optional[float]
    spatial_top_half_coated: Optional[float]
    spatial_bottom_half_coated: Optional[float]
    stability_baseline_noise: Optional[float]
    stability_dead_pixel_ratio: Optional[float]
    stability_sensor_health_index: Optional[float]
    stability_calibration_status: Optional[str]
    stability_drift_slope: Optional[float]
    stability_drift_per_hour: Optional[float]
    stability_long_term_stability: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StandardLabelRecord:
    event_id: str
    disease_or_gas_class: Optional[str]
    severity_label: Optional[str]
    concentration_label: Optional[str]
    label_provenance: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "StandardRunRecord",
    "StandardRawFrameRecord",
    "StandardEventRecord",
    "StandardFeatureRecord",
    "StandardLabelRecord",
]
