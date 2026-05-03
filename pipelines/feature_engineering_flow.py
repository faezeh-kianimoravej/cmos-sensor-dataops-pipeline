from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocess import build_windowed_timeseries, split_windowed_by_run
from src.features.extractor import build_feature_table
from src.observed.utils.artifact_publisher import publish_local_artifact
from src.observed.utils.config import load_config
from src.observed.utils.storage_runtime import (
    resolve_storage_environment,
    stage_dataset_version,
)


def run_feature_engineering(
    processed_timeseries_path: Path,
    windowed_path: Path,
    train_windows_path: Path,
    test_windows_path: Path,
    features_output_path: Path,
    summary_path: Path,
    window_size: int = 100,
    step_size: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    processed_df = pd.read_parquet(processed_timeseries_path)
    if processed_df.empty:
        raise ValueError("processed_timeseries input is empty")

    windowed_df = build_windowed_timeseries(
        processed_df=processed_df,
        window_size=window_size,
        step_size=step_size,
    )
    train_df, test_df = split_windowed_by_run(
        windowed_df=windowed_df,
        test_size=test_size,
        random_state=random_state,
    )
    features_df = build_feature_table(windowed_df)

    windowed_path.parent.mkdir(parents=True, exist_ok=True)
    train_windows_path.parent.mkdir(parents=True, exist_ok=True)
    test_windows_path.parent.mkdir(parents=True, exist_ok=True)
    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    windowed_df.to_parquet(windowed_path, index=False)
    train_df.to_parquet(train_windows_path, index=False)
    test_df.to_parquet(test_windows_path, index=False)
    features_df.to_parquet(features_output_path, index=False)

    storage_env = resolve_storage_environment()
    dataset_version = stage_dataset_version()
    published_windowed = publish_local_artifact(
        local_path=windowed_path,
        domain="datasets",
        env=storage_env,
        pipeline="feature-engineering",
        dataset_name="windowed-timeseries",
        dataset_version=dataset_version,
    )
    published_train = publish_local_artifact(
        local_path=train_windows_path,
        domain="datasets",
        env=storage_env,
        pipeline="feature-engineering",
        dataset_name="train-windows",
        dataset_version=dataset_version,
    )
    published_test = publish_local_artifact(
        local_path=test_windows_path,
        domain="datasets",
        env=storage_env,
        pipeline="feature-engineering",
        dataset_name="test-windows",
        dataset_version=dataset_version,
    )
    published_features = publish_local_artifact(
        local_path=features_output_path,
        domain="datasets",
        env=storage_env,
        pipeline="feature-engineering",
        dataset_name="features-timeseries",
        dataset_version=dataset_version,
    )

    summary = {
        "status": "success",
        "windowed_rows": int(len(windowed_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_rows": int(len(features_df)),
        "feature_columns": int(features_df.shape[1]),
        "windowed_path": str(windowed_path),
        "features_output_path": str(features_output_path),
        "dataset_storage": {
            "environment": storage_env,
            "dataset_version": dataset_version,
            "windowed": published_windowed,
            "train_windows": published_train,
            "test_windows": published_test,
            "features_timeseries": published_features,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    fe_cfg = hp_cfg.get("feature_engineering", {})
    paths_cfg = hp_cfg.get("paths", {})
    seeds_cfg = hp_cfg.get("seeds", {})

    parser = argparse.ArgumentParser(
        description="Stage 2 preprocessing/windowing/feature extraction"
    )
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
        "--windowed-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "windowed_timeseries", "data/processed/windowed_timeseries.parquet"
            )
        ),
    )
    parser.add_argument(
        "--train-windows-out",
        type=Path,
        default=Path(
            paths_cfg.get("train_windows", "data/processed/train_windows.parquet")
        ),
    )
    parser.add_argument(
        "--test-windows-out",
        type=Path,
        default=Path(
            paths_cfg.get("test_windows", "data/processed/test_windows.parquet")
        ),
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "features_timeseries", "data/processed/features_timeseries.parquet"
            )
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "feature_engineering_summary",
                "artifacts/reports/feature_engineering_summary.json",
            )
        ),
    )
    parser.add_argument(
        "--window-size", type=int, default=int(fe_cfg.get("window_size", 100))
    )
    parser.add_argument(
        "--step-size", type=int, default=int(fe_cfg.get("step_size", 50))
    )
    parser.add_argument(
        "--test-size", type=float, default=float(fe_cfg.get("test_size", 0.2))
    )
    parser.add_argument(
        "--random-state", type=int, default=int(seeds_cfg.get("global", 42))
    )
    args = parser.parse_args()

    result = run_feature_engineering(
        processed_timeseries_path=args.processed_timeseries,
        windowed_path=args.windowed_out,
        train_windows_path=args.train_windows_out,
        test_windows_path=args.test_windows_out,
        features_output_path=args.features_out,
        summary_path=args.summary_out,
        window_size=args.window_size,
        step_size=args.step_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
