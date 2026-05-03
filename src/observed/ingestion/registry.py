"""Run registry — tracks metadata about all ingested sensor datasets.

This module maintains a persistent registry of processed .dat files,
indexed by run_id. Each entry captures:
  - Unique run identifier and source file path
  - Start & end timestamps, frame count, duration, sampling rate
  - Processing timestamp (when the file was registered)
  - Layer configuration

The registry is the entry point for joining downstream artifacts
(events, features, labels) back to their original raw data.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.observed.ingestion.parser import ParsedRun
from src.observed.utils.io import read_parquet, write_parquet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry entry: one row per ingested run
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RegistryEntry:
    """Metadata for a single ingested run, to be stored in runs.parquet.

    Fields:
        run_id: Unique identifier for the run (derived from file stem).
        source_file: Absolute path to the source .dat file.
        start_time: Timestamp of the first frame (UTC).
        end_time: Timestamp of the last frame (UTC).
        n_frames: Total number of frames parsed.
        duration_s: Elapsed time in seconds between first and last frame.
        sampling_rate_hz: Median frame rate during the run.
        layer_ids: Comma-separated list of sensor layer IDs (e.g. '1,2,3').
        processing_timestamp: When this entry was registered (UTC, now).
    """

    run_id: str
    source_file: str
    start_time: datetime
    end_time: datetime
    n_frames: int
    duration_s: float
    sampling_rate_hz: float
    layer_ids: str  # Stored as comma-separated string
    processing_timestamp: datetime

    @classmethod
    def from_parsed_run(cls, parsed: ParsedRun) -> RegistryEntry:
        """Create a registry entry from a freshly parsed run."""
        return cls(
            run_id=parsed.run_id,
            source_file=parsed.source_file,
            start_time=parsed.start_time,
            end_time=parsed.end_time,
            n_frames=parsed.n_frames,
            duration_s=parsed.duration_s,
            sampling_rate_hz=parsed.sampling_rate_hz,
            layer_ids=",".join(str(lid) for lid in parsed.layer_ids),
            processing_timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary (dates serialized as ISO strings)."""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        data["processing_timestamp"] = self.processing_timestamp.isoformat()
        return data


# ---------------------------------------------------------------------------
# Registry manager
# ---------------------------------------------------------------------------


class RunRegistry:
    """In-memory registry of ingested runs, backed by parquet storage.

    Usage:
        # Register a new run
        registry = RunRegistry()
        parsed = parse_dat_file("data.dat")
        registry.register(parsed)
        registry.save("runs.parquet")

        # Later, load existing registry
        registry = RunRegistry.load("runs.parquet")
        logger.info(registry.entries)  # List of RegistryEntry objects
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._entries: Dict[str, RegistryEntry] = {}

    @property
    def entries(self) -> List[RegistryEntry]:
        """Return all registry entries in insertion order."""
        return list(self._entries.values())

    def register(self, parsed: ParsedRun) -> RegistryEntry:
        """Register a freshly parsed run in the registry.

        Args:
            parsed: A ParsedRun object to register.

        Returns:
            The RegistryEntry created and stored.

        Raises:
            ValueError: If run_id already exists in the registry (duplicates not allowed).
        """
        if parsed.run_id in self._entries:
            raise ValueError(
                f"Run {parsed.run_id!r} is already registered. "
                f"Duplicate run IDs are not allowed."
            )

        entry = RegistryEntry.from_parsed_run(parsed)
        self._entries[parsed.run_id] = entry
        return entry

    def get(self, run_id: str) -> Optional[RegistryEntry]:
        """Retrieve a registry entry by run_id, if it exists."""
        return self._entries.get(run_id)

    def has_run(self, run_id: str) -> bool:
        """Check if a run_id is already registered."""
        return run_id in self._entries

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the registry to a pandas DataFrame.

        Datetime columns remain as Python datetime objects (will be
        serialized by write_parquet via pyarrow).
        """
        if not self._entries:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                {
                    "run_id": pd.Series(dtype=str),
                    "source_file": pd.Series(dtype=str),
                    "start_time": pd.Series(dtype=object),
                    "end_time": pd.Series(dtype=object),
                    "n_frames": pd.Series(dtype="int64"),
                    "duration_s": pd.Series(dtype="float64"),
                    "sampling_rate_hz": pd.Series(dtype="float64"),
                    "layer_ids": pd.Series(dtype=str),
                    "processing_timestamp": pd.Series(dtype=object),
                }
            )

        return pd.DataFrame([asdict(e) for e in self._entries.values()])

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> RunRegistry:
        """Reconstruct a registry from a DataFrame.

        Assumes columns: run_id, source_file, start_time, end_time, n_frames,
        duration_s, sampling_rate_hz, layer_ids, processing_timestamp.
        """
        reg = RunRegistry()

        for _, row in df.iterrows():
            # Parse datetime strings back to datetime objects if needed
            start_time = row["start_time"]
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)

            end_time = row["end_time"]
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)

            proc_ts = row["processing_timestamp"]
            if isinstance(proc_ts, str):
                proc_ts = datetime.fromisoformat(proc_ts)

            entry = RegistryEntry(
                run_id=str(row["run_id"]),
                source_file=str(row["source_file"]),
                start_time=start_time,
                end_time=end_time,
                n_frames=int(row["n_frames"]),
                duration_s=float(row["duration_s"]),
                sampling_rate_hz=float(row["sampling_rate_hz"]),
                layer_ids=str(row["layer_ids"]),
                processing_timestamp=proc_ts,
            )
            reg._entries[entry.run_id] = entry

        return reg

    def save(self, path: str | Path) -> Path:
        """Save the registry to a parquet file.

        Args:
            path: Destination parquet file.

        Returns:
            The absolute Path to the saved file.
        """
        df = self.to_dataframe()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(df, path, compression="snappy")
        return path

    @staticmethod
    def load(path: str | Path) -> RunRegistry:
        """Load a registry from a parquet file.

        Args:
            path: Path to runs.parquet.

        Returns:
            A populated RunRegistry.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        df = read_parquet(path)
        return RunRegistry.from_dataframe(df)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def register_run(
    parsed: ParsedRun,
    registry_path: str | Path = "artifacts/runs.parquet",
) -> RegistryEntry:
    """One-shot convenience: register a parsed run and save the registry.

    If the registry file already exists, it is loaded, updated, and saved.
    If it doesn't exist, a new registry is created.

    Args:
        parsed: A ParsedRun object to register.
        registry_path: Path to runs.parquet (auto-created if needed).

    Returns:
        The RegistryEntry that was registered.

    Raises:
        ValueError: If the run_id is already in the registry.
    """
    registry_path = Path(registry_path)

    if registry_path.exists():
        registry = RunRegistry.load(registry_path)
    else:
        registry = RunRegistry()

    entry = registry.register(parsed)
    registry.save(registry_path)
    return entry


__all__ = [
    "RegistryEntry",
    "RunRegistry",
    "register_run",
]
