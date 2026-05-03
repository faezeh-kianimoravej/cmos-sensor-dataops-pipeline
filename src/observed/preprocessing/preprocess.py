"""Preprocessing wrapper — converts parsed frames to processed time series.

This module wraps Tolouene's core signal processing algorithms (baseline
estimation, coated/reference pixel subtraction for ΔADC) into a clean API
that downstream analysis modules consume without depending on Tolouene.

The key transformation is from raw ADC counts to normalized ΔADC signal,
with layer sums and baseline metadata captured for quality inspection.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.observed.ingestion.parser import ParsedRun
from src.observed.utils.config import get_repo_root

# ---------------------------------------------------------------------------
# Ensure Tolouene is importable
# ---------------------------------------------------------------------------
_repo_root = get_repo_root()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from Tolouene.modules.module1_data_preprocessing import (
        process_dataset as _tolouene_process_dataset,  # type: ignore[import]
    )

    _TOLOUENE_AVAILABLE = True
except Exception as _import_exc:  # pragma: no cover
    _TOLOUENE_AVAILABLE = False
    _import_exc_msg = str(_import_exc)


# ---------------------------------------------------------------------------
# Processed dataset type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObservedProcessedDataset:
    """Processed time series representation of a sensor run.

    This is the main intermediate artifact produced by preprocessing and
    consumed by downstream analysis (event detection, feature extraction).

    Fields:
        run_id: Identifier of the source run.
        n_frames: Number of frames in the time series.
        timestamps: List of frame timestamps (UTC, datetime objects).
        layer_sums: Dict[layer_id → np.ndarray(n_frames, dtype=int64)]
            Total ADC count per layer per frame (raw pixel sums).
        delta_adc: np.ndarray(n_frames, dtype=float32)
            Normalized signal using coated/reference pixel subtraction.
            Automatically compensates for environmental drift.
        baseline: float
            Mean reference pixel value (environmental baseline).
        start_time: Timestamp of first frame.
        end_time: Timestamp of last frame.
        duration_s: Total elapsed time in seconds.
        sampling_rate_hz: Estimated frames-per-second.
    """

    run_id: str
    n_frames: int
    timestamps: List[datetime]
    layer_sums: Dict[int, np.ndarray]  # layer_id -> (n_frames,) int64
    delta_adc: np.ndarray  # (n_frames,) float32
    baseline: float
    start_time: datetime
    end_time: datetime
    duration_s: float
    sampling_rate_hz: float

    @property
    def n_samples(self) -> int:
        """Alias for n_frames (convenient for signal processing code)."""
        return self.n_frames

    @property
    def time_axis(self) -> np.ndarray:
        """Relative time in seconds from start_time."""
        return np.array(
            [(ts - self.start_time).total_seconds() for ts in self.timestamps],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary (datetimes as ISO strings, arrays as nested lists)."""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        data["timestamps"] = [ts.isoformat() for ts in self.timestamps]
        data["delta_adc"] = self.delta_adc.tolist()
        data["layer_sums"] = {k: v.tolist() for k, v in self.layer_sums.items()}
        return data


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess(
    parsed: ParsedRun, sensor_config: Optional[str] = None
) -> ObservedProcessedDataset:
    """Convert a ParsedRun into a processed time series.

    This is the main entry point for preprocessing. It:
      1. Validates that raw frames are available in the ParsedRun
      2. Calls Tolouene's process_dataset() for signal processing
      3. Wraps the result in an Observed-native type
      4. Performs basic validation

    Args:
        parsed: A ParsedRun returned by parse_dat_file().
        sensor_config: (Unused in current implementation; kept for API
            compatibility with future sensor-specific configs.)

    Returns:
        ObservedProcessedDataset ready for downstream analysis modules.

    Raises:
        RuntimeError: If Tolouene is not available or preprocessing fails.
        ValueError: If parsed run is missing required data.
    """
    if not _TOLOUENE_AVAILABLE:
        raise RuntimeError(
            "Tolouene package could not be imported — ensure the repository "
            f"root is a Python package or on sys.path. Original error: {_import_exc_msg}"
        )

    if not parsed._raw_frames:
        raise ValueError(
            f"ParsedRun {parsed.run_id!r} has no raw frames stored. "
            "This usually means the parser is out of sync with the preprocessing module."
        )

    # --- Call Tolouene's core preprocessing ----
    try:
        processed = _tolouene_process_dataset(parsed._raw_frames)
    except Exception as exc:
        raise RuntimeError(
            f"Tolouene preprocessing failed for run {parsed.run_id}: {exc}"
        ) from exc

    # --- Wrap in Observed type ----
    # Collect layer sums into a consistent format (dict of arrays)
    layer_sums: Dict[int, np.ndarray] = {
        1: np.asarray(processed.layer1_sums, dtype=np.int64),
        2: np.asarray(processed.layer2_sums, dtype=np.int64),
        3: np.asarray(processed.layer3_sums, dtype=np.int64),
    }

    # Convert ΔADC to float32 for downstream use
    delta_adc = np.asarray(processed.delta_adc, dtype=np.float32)

    return ObservedProcessedDataset(
        run_id=parsed.run_id,
        n_frames=parsed.n_frames,
        timestamps=processed.timestamps,
        layer_sums=layer_sums,
        delta_adc=delta_adc,
        baseline=float(processed.baseline),
        start_time=parsed.start_time,
        end_time=parsed.end_time,
        duration_s=parsed.duration_s,
        sampling_rate_hz=parsed.sampling_rate_hz,
    )


__all__ = [
    "ObservedProcessedDataset",
    "preprocess",
]
