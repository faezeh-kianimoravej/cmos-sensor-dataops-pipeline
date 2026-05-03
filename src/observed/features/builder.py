"""
Feature table builder for the Observed pipeline.

Assembles a structured feature table from detected events and exports features.parquet.
The implementation keeps feature extraction and persistence in one place so
upstream event detection and downstream training use a consistent table format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.observed.features.temporal import (
    extract_temporal_features,
    save_features_parquet,
)


def build_feature_table(
    signal: np.ndarray,
    timestamps: np.ndarray,
    events: Sequence[Any],
    run_id: str,
    baseline: float = 0.0,
    baseline_noise: float = np.nan,
    output_path: (
        str | Path
    ) = "data/processed/dat_runs/features_event_table_dat.parquet",
) -> Path:
    """
        Build and persist event-level features in a schema-stable parquet table.

        Why:
        Centralizing table construction prevents drift between training-time and
        serving-time feature definitions.

        Assumptions:
        - `events` include start/end indices and IDs compatible with temporal
            feature extraction.
        - `signal` and `timestamps` are aligned 1D arrays.

        Edge cases:
        - Empty event lists produce an empty but valid feature table.
        - `baseline_noise` may be NaN when unavailable; downstream consumers should
            treat it as optional metadata.

    Args:
        signal: 1D numpy array of sensor signal (ΔADC values)
        timestamps: 1D numpy array of time axis (seconds)
        events: Sequence of event objects (must have event_id, run_id, start_idx, end_idx)
        run_id: Run identifier
        baseline: Baseline value to subtract from signal (default 0.0)
        baseline_noise: Baseline noise estimate (optional, default NaN)
        output_path: Destination path for the Parquet file
    Returns:
        Path to the written Parquet file.
    """
    # Compute temporal features for all events
    features = extract_temporal_features(
        signal=signal,
        timestamps=timestamps,
        events=events,
        run_id=run_id,
        baseline=baseline,
        baseline_noise=baseline_noise,
    )
    # Additional feature families can be appended here without changing table I/O.
    # Save to Parquet
    return save_features_parquet(features, output_path)


__all__ = ["build_feature_table"]
