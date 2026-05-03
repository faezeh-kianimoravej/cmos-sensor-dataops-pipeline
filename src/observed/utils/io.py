"""Pipeline artifact I/O helpers.

Provides stable, inspectable persistence for the two artifact classes used in
this project:

* Tabular data (runs, events, features, labels) → Parquet via pandas/pyarrow
* Raw frame arrays (32 × 32 × time, per layer) → HDF5 via h5py

All write helpers create parent directories automatically.  Read helpers raise
clear errors for missing or malformed files so callers never get silent
failures.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Create parent directory of *path* if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def write_parquet(
    df: pd.DataFrame, path: str | Path, *, compression: str = "snappy"
) -> Path:
    """Persist *df* to a Parquet file at *path*.

    Creates all intermediate directories automatically.  Returns resolved path.

    Args:
        df: DataFrame to persist.
        path: Destination file path (absolute or relative to cwd).
        compression: Parquet compression codec.  'snappy' is fast and portable.

    Returns:
        Resolved Path to the written file.
    """
    dest = Path(path).resolve()
    _ensure_dir(dest)

    df.to_parquet(dest, engine="pyarrow", compression=compression, index=True)
    return dest


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When the file cannot be read as Parquet.
    """
    src = Path(path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Parquet file not found: {src}")

    if not src.is_file():
        raise FileNotFoundError(f"Path is not a file: {src}")

    try:
        return pd.read_parquet(src, engine="pyarrow")
    except Exception as exc:
        raise ValueError(f"Failed to read Parquet file {src}: {exc}") from exc


# ---------------------------------------------------------------------------
# HDF5 raw-frame I/O
# ---------------------------------------------------------------------------

# HDF5 group / dataset naming conventions
_GROUP_TMPL = "run/{run_id}"
_DS_FRAMES = "frames_layer{layer_id}"  # shape (n_frames, 32, 32), dtype uint16
_DS_SUMS = "sums_layer{layer_id}"  # shape (n_frames,), dtype int64
_DS_TIMESTAMPS = "timestamps"  # shape (n_frames,), dtype int64 (ns since epoch)
_DS_PACKET_IDS = "packet_ids"  # shape (n_frames,), dtype int64


def _encode_timestamps(timestamps: Sequence[datetime]) -> np.ndarray:
    """Convert datetime sequence to int64 nanoseconds-since-epoch array."""
    return np.array(
        [int(ts.timestamp() * 1_000_000_000) for ts in timestamps],
        dtype=np.int64,
    )


def _decode_timestamps(arr: np.ndarray) -> List[datetime]:
    """Convert int64 nanoseconds-since-epoch array back to datetime list."""
    return [datetime.utcfromtimestamp(int(ns) / 1_000_000_000) for ns in arr]


def write_frames_hdf5(
    path: str | Path,
    run_id: str,
    timestamps: Sequence[datetime],
    frames_by_layer: Dict[int, np.ndarray],
    packet_ids: Sequence[int],
    *,
    overwrite: bool = False,
) -> Path:
    """Save raw sensor frame arrays to an HDF5 file.

    Data is organised under a group keyed by run_id:

        run/<run_id>/timestamps          int64[n_frames]          (ns since epoch)
        run/<run_id>/packet_ids          int64[n_frames]
        run/<run_id>/frames_layer<L>     uint16[n_frames, 32, 32]
        run/<run_id>/sums_layer<L>       int64[n_frames]

    Args:
        path: HDF5 file path (.h5 or .hdf5).
        run_id: Unique run identifier — becomes the HDF5 group key.
        timestamps: Per-frame datetime objects, length n_frames.
        frames_by_layer: Mapping {layer_id: ndarray of shape (n_frames, H, W)}.
        packet_ids: Per-frame packet IDs, length n_frames.
        overwrite: If True and the run_id group already exists it is replaced.

    Returns:
        Resolved Path to the written file.

    Raises:
        ValueError: When run_id already exists in the file and overwrite=False.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 frame storage: pip install h5py"
        ) from exc

    dest = Path(path).resolve()
    _ensure_dir(dest)

    n_frames = len(timestamps)
    group_key = _GROUP_TMPL.format(run_id=run_id)

    mode = "a"  # append / create
    with h5py.File(dest, mode) as hf:
        if group_key in hf:
            if not overwrite:
                raise ValueError(
                    f"Run '{run_id}' already exists in {dest}. "
                    "Pass overwrite=True to replace it."
                )
            del hf[group_key]

        grp = hf.require_group(group_key)

        grp.create_dataset(
            _DS_TIMESTAMPS,
            data=_encode_timestamps(timestamps),
            dtype=np.int64,
        )
        grp.create_dataset(
            _DS_PACKET_IDS,
            data=np.array(packet_ids, dtype=np.int64),
            dtype=np.int64,
        )

        for layer_id, array in frames_by_layer.items():
            arr = np.asarray(array, dtype=np.uint16)
            if arr.ndim != 3 or arr.shape[0] != n_frames:
                raise ValueError(
                    f"frames_by_layer[{layer_id}] must have shape "
                    f"(n_frames={n_frames}, H, W), got {arr.shape}"
                )
            grp.create_dataset(
                _DS_FRAMES.format(layer_id=layer_id),
                data=arr,
                dtype=np.uint16,
                compression="gzip",
                compression_opts=4,
            )
            layer_sums = arr.reshape(n_frames, -1).sum(axis=1).astype(np.int64)
            grp.create_dataset(
                _DS_SUMS.format(layer_id=layer_id),
                data=layer_sums,
                dtype=np.int64,
            )

        # Persist run_id as an attribute for human inspection
        grp.attrs["run_id"] = run_id
        grp.attrs["n_frames"] = n_frames

    return dest


def read_frames_hdf5(
    path: str | Path,
    run_id: str,
) -> Dict[str, Any]:
    """Load raw sensor frame arrays previously saved with write_frames_hdf5.

    Returns a dictionary with the following keys:
        run_id: str
        timestamps: List[datetime]
        packet_ids: np.ndarray (int64)
        frames_by_layer: Dict[int, np.ndarray] (uint16, shape n_frames×H×W)
        sums_by_layer: Dict[int, np.ndarray] (int64, shape n_frames)
        n_frames: int

    Args:
        path: HDF5 file path.
        run_id: Run identifier to retrieve.

    Raises:
        FileNotFoundError: When the file does not exist.
        KeyError: When run_id is not present in the file.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 frame storage: pip install h5py"
        ) from exc

    src = Path(path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"HDF5 file not found: {src}")

    group_key = _GROUP_TMPL.format(run_id=run_id)

    with h5py.File(src, "r") as hf:
        if group_key not in hf:
            available = sorted(hf.get("run", {}).keys())
            raise KeyError(
                f"Run '{run_id}' not found in {src}. Available run IDs: {available}"
            )

        grp = hf[group_key]

        timestamps = _decode_timestamps(grp[_DS_TIMESTAMPS][()])
        packet_ids = grp[_DS_PACKET_IDS][()].copy()

        frames_by_layer: Dict[int, np.ndarray] = {}
        sums_by_layer: Dict[int, np.ndarray] = {}

        for key in grp.keys():
            if key.startswith("frames_layer"):
                layer_id = int(key[len("frames_layer") :])
                frames_by_layer[layer_id] = grp[key][()].copy()
            elif key.startswith("sums_layer"):
                layer_id = int(key[len("sums_layer") :])
                sums_by_layer[layer_id] = grp[key][()].copy()

        n_frames = int(grp.attrs.get("n_frames", len(timestamps)))

    return {
        "run_id": run_id,
        "timestamps": timestamps,
        "packet_ids": packet_ids,
        "frames_by_layer": frames_by_layer,
        "sums_by_layer": sums_by_layer,
        "n_frames": n_frames,
    }


def list_runs_hdf5(path: str | Path) -> List[str]:
    """Return all run IDs stored in an HDF5 file.

    Args:
        path: HDF5 file path.

    Returns:
        Sorted list of run ID strings.

    Raises:
        FileNotFoundError: When the file does not exist.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 frame storage: pip install h5py"
        ) from exc

    src = Path(path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"HDF5 file not found: {src}")

    with h5py.File(src, "r") as hf:
        run_group = hf.get("run", {})
        return sorted(run_group.keys())


__all__ = [
    "write_parquet",
    "read_parquet",
    "write_frames_hdf5",
    "read_frames_hdf5",
    "list_runs_hdf5",
]
