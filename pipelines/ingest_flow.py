from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.parser import discover_text_runs
from src.data.preprocess import build_processed_timeseries
from src.observed.utils.artifact_publisher import publish_local_artifact
from src.observed.utils.config import load_config
from src.observed.utils.storage_runtime import (
    resolve_storage_environment,
    stage_dataset_version,
)


def run_ingestion(
    input_path: Path,
    config_path: Path,
    processed_timeseries_path: Path,
    runs_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    raw_root = input_path if input_path.is_dir() else input_path.parent

    runs_df = discover_text_runs(raw_root)
    processed_df = build_processed_timeseries(runs_df)

    runs_path.parent.mkdir(parents=True, exist_ok=True)
    processed_timeseries_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    runs_df.to_parquet(runs_path, index=False)
    processed_df.to_parquet(processed_timeseries_path, index=False)

    storage_env = resolve_storage_environment()
    dataset_version = stage_dataset_version()
    published_runs = publish_local_artifact(
        local_path=runs_path,
        domain="datasets",
        env=storage_env,
        pipeline="ingest",
        dataset_name="runs",
        dataset_version=dataset_version,
    )
    published_processed = publish_local_artifact(
        local_path=processed_timeseries_path,
        domain="datasets",
        env=storage_env,
        pipeline="ingest",
        dataset_name="processed-timeseries",
        dataset_version=dataset_version,
    )

    summary = {
        "status": "success",
        "raw_root": str(raw_root),
        "runs_discovered": int(len(runs_df)),
        "runs_processed": int(len(processed_df)),
        "runs_path": str(runs_path),
        "processed_timeseries_path": str(processed_timeseries_path),
        "dataset_storage": {
            "environment": storage_env,
            "dataset_version": dataset_version,
            "runs": published_runs,
            "processed_timeseries": published_processed,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    paths_cfg = hp_cfg.get("paths", {})

    parser = argparse.ArgumentParser(
        description="Stage 1 ingestion and structured processing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(paths_cfg.get("raw_input", "data/raw")),
        help="Raw data root folder",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--processed-timeseries",
        type=Path,
        default=Path(
            paths_cfg.get(
                "processed_timeseries", "data/processed/processed_timeseries.parquet"
            )
        ),
    )
    parser.add_argument(
        "--runs-out",
        type=Path,
        default=Path(paths_cfg.get("runs", "data/processed/runs.parquet")),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "ingestion_summary", "artifacts/reports/ingestion_summary.json"
            )
        ),
    )
    args = parser.parse_args()

    result = run_ingestion(
        input_path=args.input,
        config_path=args.config,
        processed_timeseries_path=args.processed_timeseries,
        runs_path=args.runs_out,
        summary_path=args.summary_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
