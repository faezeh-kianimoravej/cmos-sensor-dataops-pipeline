"""Signal filtering wrapper for Observed preprocessing.

This module wraps Tolouene's Module 2 filtering primitives so the Observed
pipeline can apply baseline correction and smoothing consistently without
depending on Tolouene types downstream.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.observed.preprocessing.preprocess import ObservedProcessedDataset
from src.observed.utils.config import get_repo_root

# ---------------------------------------------------------------------------
# Ensure Tolouene module imports resolve from any working directory.
# ---------------------------------------------------------------------------
_repo_root = get_repo_root()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from Tolouene.modules.module2_signal_processing import (
        FilterPipeline as _ToloueneFilterPipeline,  # type: ignore[import]
    )
    from Tolouene.modules.module2_signal_processing import (
        apply_butterworth_lowpass as _apply_butterworth_lowpass,
    )
    from Tolouene.modules.module2_signal_processing import (
        apply_moving_average as _apply_moving_average,
    )
    from Tolouene.modules.module2_signal_processing import (
        apply_savitzky_golay as _apply_savitzky_golay,
    )

    _TOLOUENE_FILTER_AVAILABLE = True
except Exception as _import_exc:  # pragma: no cover
    _TOLOUENE_FILTER_AVAILABLE = False
    _import_exc_msg = str(_import_exc)


@dataclass(slots=True)
class FilterConfig:
    """Config for signal filtering.

    Attributes:
        baseline_window: Rolling window for baseline drift estimation.
        smoothing_method: One of "ma", "savgol", "butter".
        smoothing_window: Window size for MA/Savitzky-Golay.
        savgol_order: Polynomial order for Savitzky-Golay.
        butter_cutoff: Normalized cutoff frequency for Butterworth filter.
        butter_order: Filter order for Butterworth filter.
    """

    baseline_window: int = 300
    smoothing_method: str = "savgol"
    smoothing_window: int = 15
    savgol_order: int = 3
    butter_cutoff: float = 0.1
    butter_order: int = 2

    def validate(self) -> None:
        """Validate the filter configuration."""
        allowed = {"ma", "savgol", "butter"}
        if self.smoothing_method not in allowed:
            raise ValueError(
                f"Unknown smoothing_method={self.smoothing_method!r}. "
                f"Expected one of {sorted(allowed)}"
            )

        if self.baseline_window < 1:
            raise ValueError("baseline_window must be >= 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if self.savgol_order < 1:
            raise ValueError("savgol_order must be >= 1")
        if self.butter_order < 1:
            raise ValueError("butter_order must be >= 1")
        if not (0.0 < self.butter_cutoff < 1.0):
            raise ValueError("butter_cutoff must be in (0, 1)")


@dataclass(slots=True)
class FilteredSignal:
    """Filtered representation ready for event detection.

    Attributes:
        run_id: Source run identifier.
        timestamps: Timestamp series aligned with all signal arrays.
        raw_delta_adc: Original processed signal before filtering.
        baseline_trend: Estimated baseline drift curve.
        baseline_corrected: Signal after baseline correction.
        filtered_delta_adc: Final smoothed signal for event detection.
        smoothing_method: Method actually used.
        config: Snapshot of filter config values.
        start_time: Run start time.
        end_time: Run end time.
        duration_s: Run duration in seconds.
        sampling_rate_hz: Sampling frequency estimate.
    """

    run_id: str
    timestamps: List[datetime]
    raw_delta_adc: np.ndarray
    baseline_trend: np.ndarray
    baseline_corrected: np.ndarray
    filtered_delta_adc: np.ndarray
    smoothing_method: str
    config: Dict[str, object]
    start_time: datetime
    end_time: datetime
    duration_s: float
    sampling_rate_hz: float

    @property
    def event_signal(self) -> np.ndarray:
        """Alias used by event-detection code paths."""
        return self.filtered_delta_adc

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a plain dictionary for plotting or API responses."""
        return {
            "run_id": self.run_id,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "raw_delta_adc": self.raw_delta_adc.tolist(),
            "baseline_trend": self.baseline_trend.tolist(),
            "baseline_corrected": self.baseline_corrected.tolist(),
            "filtered_delta_adc": self.filtered_delta_adc.tolist(),
            "smoothing_method": self.smoothing_method,
            "config": dict(self.config),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_s": self.duration_s,
            "sampling_rate_hz": self.sampling_rate_hz,
        }


def _resolve_savgol_window(length: int, requested_window: int, order: int) -> int:
    """Return a valid odd Savitzky-Golay window for signal length/order."""
    if length <= 2:
        return 1

    window = max(1, requested_window)
    max_window = length if length % 2 == 1 else length - 1

    if window > max_window:
        window = max_window
    if window % 2 == 0:
        window = max(1, window - 1)

    min_window = order + 2
    if min_window % 2 == 0:
        min_window += 1

    if window < min_window:
        window = min_window

    if window > max_window:
        window = max_window
    if window % 2 == 0:
        window = max(1, window - 1)

    return max(1, window)


def _smooth_signal(signal_data: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Apply Tolouene smoothing helpers with robust small-signal handling."""
    if cfg.smoothing_method == "ma":
        return np.asarray(
            _apply_moving_average(signal_data, window=cfg.smoothing_window),
            dtype=np.float32,
        )

    if cfg.smoothing_method == "savgol":
        window = _resolve_savgol_window(
            len(signal_data), cfg.smoothing_window, cfg.savgol_order
        )
        if window <= 1:
            return np.asarray(signal_data, dtype=np.float32)
        order = min(cfg.savgol_order, max(1, window - 2))
        return np.asarray(
            _apply_savitzky_golay(signal_data, window=window, order=order),
            dtype=np.float32,
        )

    # Butterworth can fail on very short sequences due filtfilt padding limits.
    try:
        return np.asarray(
            _apply_butterworth_lowpass(
                signal_data,
                cutoff=cfg.butter_cutoff,
                order=cfg.butter_order,
            ),
            dtype=np.float32,
        )
    except Exception:
        fallback_window = min(max(3, cfg.smoothing_window), max(3, len(signal_data)))
        return np.asarray(
            _apply_moving_average(signal_data, window=fallback_window),
            dtype=np.float32,
        )


def filter_signal(
    processed: ObservedProcessedDataset,
    config: Optional[FilterConfig] = None,
) -> FilteredSignal:
    """Apply baseline correction + smoothing to a processed data run.

    Args:
        processed: Output of preprocessing step containing delta_adc signal.
        config: Optional filtering config. Uses defaults when omitted.

    Returns:
        FilteredSignal suitable for event detection.

    Raises:
        RuntimeError: If Tolouene filtering components are unavailable.
        ValueError: For invalid config or malformed processed input.
    """
    if not _TOLOUENE_FILTER_AVAILABLE:
        raise RuntimeError(
            "Tolouene filtering components could not be imported. "
            f"Original error: {_import_exc_msg}"
        )

    cfg = config or FilterConfig()
    cfg.validate()

    raw_signal = np.asarray(processed.delta_adc, dtype=np.float32)
    if raw_signal.ndim != 1:
        raise ValueError("processed.delta_adc must be a 1D signal")
    if len(raw_signal) == 0:
        raise ValueError("processed.delta_adc is empty")
    if len(raw_signal) != len(processed.timestamps):
        raise ValueError("processed timestamps and delta_adc must have matching length")

    pipeline = _ToloueneFilterPipeline(
        baseline_window=cfg.baseline_window,
        smoothing_method=cfg.smoothing_method,
        smoothing_window=cfg.smoothing_window,
        savgol_order=cfg.savgol_order,
        butter_cutoff=cfg.butter_cutoff,
        butter_order=cfg.butter_order,
    )

    # Reuse Tolouene baseline-correction stage directly.
    baseline_corrected = np.asarray(
        pipeline._correct_baseline(raw_signal),  # pylint: disable=protected-access
        dtype=np.float32,
    )
    baseline_trend = np.asarray(pipeline.baseline, dtype=np.float32)

    filtered = _smooth_signal(baseline_corrected, cfg)

    return FilteredSignal(
        run_id=processed.run_id,
        timestamps=processed.timestamps,
        raw_delta_adc=raw_signal,
        baseline_trend=baseline_trend,
        baseline_corrected=baseline_corrected,
        filtered_delta_adc=filtered,
        smoothing_method=cfg.smoothing_method,
        config=asdict(cfg),
        start_time=processed.start_time,
        end_time=processed.end_time,
        duration_s=processed.duration_s,
        sampling_rate_hz=processed.sampling_rate_hz,
    )


__all__ = [
    "FilterConfig",
    "FilteredSignal",
    "filter_signal",
]
