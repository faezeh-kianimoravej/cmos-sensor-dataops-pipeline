"""Event detection integration script for Observed pipeline.

Loads a filtered ΔADC signal, detects events, and saves them as events.parquet.
"""

# ruff: noqa: E402

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Ensure project root is in sys.path for src imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.observed.event_detection.detector import detect_events  # noqa: E402
from src.observed.event_detection.detector import save_events_parquet
from src.observed.ingestion.parser import parse_dat_file  # noqa: E402

# Example usage: process a run and save events


def main():
    # Path to a real run's .dat file (update as needed)
    dat_path = Path(
        "data/raw/tolouene/tolouene_aug_2024.dat"
    )  # Updated path to real .dat file
    run = parse_dat_file(dat_path)
    run_id = run.run_id
    # For demonstration, use the first layer's ΔADC signal
    # (Replace with actual filtering as needed)
    layer_id = run.layer_ids[0]
    signal = np.array([frame.layer_grids[layer_id].mean() for frame in run.frames])
    timestamps = np.array(
        [(frame.timestamp - run.start_time).total_seconds() for frame in run.frames]
    )
    # Detect events
    events = detect_events(signal, timestamps, run_id)
    # Save to Parquet
    output_path = Path("data/processed/events.parquet")
    save_events_parquet(events, output_path)
    logger.info(f"Saved {len(events)} events to {output_path}")


if __name__ == "__main__":
    main()
