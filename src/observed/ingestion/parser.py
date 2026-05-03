"""Observed ingestion parser — wraps the Tolouene .dat file loader.

This module is the first contact point between raw sensor logs and the
Observed pipeline.  It delegates all low-level parsing (timestamp extraction,
packet ID reading, 32×32 serpentine grid reconstruction) to the proven
Tolouene implementation and returns a clean, Observed-native data structure.

No Tolouene source code is modified.  The wrapper adds:
  - run-level metadata (run_id, source_file, duration, sampling rate)
  - a normalized frame type (ObservedFrame) that downstream modules consume
  - consistent error messages at the boundary between raw data and the pipeline
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repository root is importable so Tolouene can be resolved regardless
# of the caller's working directory.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from Tolouene.modules.module1_data_preprocessing import (
        FrameRecord as _ToloueneFrameRecord,  # type: ignore[import]
    )
    from Tolouene.modules.module1_data_preprocessing import (
        load_frames as _tolouene_load_frames,
    )

    _TOLOUENE_AVAILABLE = True
except Exception as _import_exc:  # pragma: no cover
    _TOLOUENE_AVAILABLE = False
    _import_exc_msg = str(_import_exc)


# ---------------------------------------------------------------------------
# Observed-native frame / run types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObservedFrame:
    """Normalized representation of a single sensor frame.

    Fields:
        frame_index: Zero-based position in the run sequence.
        timestamp: UTC datetime of the frame as parsed from the .dat file.
        packet_id: Hardware packet identifier from the sensor header.
        layer_grids: Mapping of layer_id → 32×32 ADC array (uint16).
    """

    frame_index: int
    timestamp: datetime
    packet_id: int
    layer_grids: Dict[int, np.ndarray]  # layer_id -> (32, 32) uint16


@dataclass
class ParsedRun:
    """Run-level container produced by parse_dat_file().

    Fields:
        run_id: Unique identifier derived from the source filename stem.
        source_file: Absolute path to the parsed .dat file.
        n_frames: Total number of frames successfully parsed.
        frames: Ordered list of ObservedFrame objects.
        start_time: Timestamp of the first frame.
        end_time: Timestamp of the last frame.
        duration_s: Wall-clock duration in seconds.
        sampling_rate_hz: Median frame rate inferred from timestamps.
        layer_ids: Sorted list of layer IDs present in the data.
        _raw_frames: (Internal) Original Tolouene FrameRecord objects for preprocessing.
    """

    run_id: str
    source_file: str
    n_frames: int
    frames: List[ObservedFrame]
    start_time: datetime
    end_time: datetime
    duration_s: float
    sampling_rate_hz: float
    layer_ids: List[int] = field(default_factory=list)
    _raw_frames: List[_ToloueneFrameRecord] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_run_id(path: Path) -> str:
    """Derive a pipeline-safe run identifier from a file stem."""
    return path.stem.lower().replace(" ", "-")


def _infer_sampling_rate(frames: List[ObservedFrame]) -> float:
    """Estimate sampling rate in Hz from median inter-frame interval."""
    if len(frames) < 2:
        return 0.0

    intervals = [
        (frames[i].timestamp - frames[i - 1].timestamp).total_seconds()
        for i in range(1, len(frames))
    ]
    median_interval = float(np.median(intervals))
    if median_interval <= 0.0:
        return 0.0

    return 1.0 / median_interval


def _convert_frame(raw: _ToloueneFrameRecord, index: int) -> ObservedFrame:
    """Convert a Tolouene FrameRecord into an ObservedFrame.

    Casts grid arrays from int32 (Tolouene internal) to uint16 which matches
    the physical ADC range (0–4095) and is consistent with io.write_frames_hdf5.
    """
    layer_grids: Dict[int, np.ndarray] = {
        layer_id: np.asarray(grid, dtype=np.uint16)
        for layer_id, grid in raw.grids.items()
    }
    return ObservedFrame(
        frame_index=index,
        timestamp=raw.timestamp,
        packet_id=raw.packet_id,
        layer_grids=layer_grids,
    )


def _read_numeric_series(path: str | Path) -> np.ndarray:
    """Read one numeric time series from a text file.

    Supports:
    - one value per line
    - multiple whitespace/comma separated values per line
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Numeric series file not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {file_path}")

    values: list[float] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            line = line.replace(",", " ")
            for token in line.split():
                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric token {token!r} in {file_path}"
                    ) from exc

    if not values:
        raise ValueError(f"No numeric values found in file: {file_path}")

    series = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(series)):
        raise ValueError(f"Non-finite numeric values found in: {file_path}")
    return series


def load_text_signal_pair(
    adc_path: str | Path,
    time_path: str | Path,
    *,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ADC signal + elapsed-seconds time series from text files.

    Returns:
        (adc_values, time_seconds), each as float64 1D arrays with equal length.
    """
    adc_values = _read_numeric_series(adc_path)
    time_seconds = _read_numeric_series(time_path)

    if len(adc_values) != len(time_seconds):
        raise ValueError(
            "ADC/time series length mismatch: "
            f"adc={len(adc_values)}, time={len(time_seconds)}"
        )

    if len(adc_values) < 2:
        raise ValueError("ADC/time series must contain at least 2 samples")

    diffs = np.diff(time_seconds)
    if np.any(diffs < 0):
        raise ValueError("time series must be non-decreasing")

    if max_frames is not None and max_frames < len(adc_values):
        adc_values = adc_values[:max_frames]
        time_seconds = time_seconds[:max_frames]

    return adc_values, time_seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dat_file(
    path: str | Path,
    *,
    run_id: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> ParsedRun:
    """Parse a CMOS sensor .dat file and return a normalized ParsedRun.

    Delegates actual parsing to the Tolouene loader (Tolouene.modules
    .module1_data_preprocessing.load_frames) and wraps the result in
    Observed-native data structures.

    Args:
        path: Path to the raw .dat sensor log file.
        run_id: Override for the run identifier.  When omitted the file stem
            is used (e.g. ``tolouene_aug_2024`` for ``tolouene_aug_2024.dat``).
        max_frames: If given, only the first *max_frames* frames are parsed.
            Useful for quick inspection and fast unit tests on large files.

    Returns:
        ParsedRun with run metadata and an ordered list of ObservedFrame objects.

    Raises:
        ImportError: When the Tolouene package cannot be imported.
        FileNotFoundError: When the .dat file does not exist.
        ValueError: When the file is present but contains no parseable frames,
            or when Tolouene raises a format error.
    """
    if not _TOLOUENE_AVAILABLE:
        raise ImportError(
            "Tolouene package could not be imported — ensure the repository "
            f"root is a Python package or on sys.path. Original error: {_import_exc_msg}"
        )

    dat_path = Path(path).resolve()

    if not dat_path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {dat_path}")

    if not dat_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {dat_path}")

    # --- Call Tolouene parser -------------------------------------------
    try:
        raw_frames: List[_ToloueneFrameRecord] = _tolouene_load_frames(dat_path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Tolouene parser failed on {dat_path}: {exc}") from exc

    if not raw_frames:
        raise ValueError(
            f"No frames parsed from {dat_path} — file may be empty or malformed."
        )

    # Apply optional frame cap after parsing (Tolouene loads lazily anyway)
    if max_frames is not None and max_frames < len(raw_frames):
        raw_frames = raw_frames[:max_frames]

    # --- Convert to Observed types ------------------------------------
    observed_frames = [_convert_frame(raw, i) for i, raw in enumerate(raw_frames)]

    # --- Infer run-level metadata ------------------------------------
    derived_run_id = run_id if run_id is not None else _derive_run_id(dat_path)
    start_time = observed_frames[0].timestamp
    end_time = observed_frames[-1].timestamp
    duration_s = (end_time - start_time).total_seconds()
    sampling_rate_hz = _infer_sampling_rate(observed_frames)
    layer_ids = sorted(observed_frames[0].layer_grids.keys())

    return ParsedRun(
        run_id=derived_run_id,
        source_file=str(dat_path),
        n_frames=len(observed_frames),
        frames=observed_frames,
        start_time=start_time,
        end_time=end_time,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        layer_ids=layer_ids,
        _raw_frames=raw_frames,
    )


__all__ = [
    "ObservedFrame",
    "ParsedRun",
    "parse_dat_file",
    "load_text_signal_pair",
]
