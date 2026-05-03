from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.stage1 import train_stage1_mixture_detector


def run_train_stage1_mixture_detection(
    features_path: Path,
    model_dir: Path,
    comparison_path: Path,
    metrics_path: Path,
    predictions_path: Path | None = None,
    mlflow_experiment_name: str | None = None,
    config_path: Path = Path("configs/default.yaml"),
) -> dict[str, Any]:
    return train_stage1_mixture_detector(
        features_path=features_path,
        model_dir=model_dir,
        comparison_path=comparison_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        mlflow_experiment_name=mlflow_experiment_name,
        config_path=config_path,
    )


def main() -> None:
    from src.observed.utils.config import load_config

    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    paths_cfg = hp_cfg.get("paths", {})
    mlflow_cfg = hp_cfg.get("mlflow", {})

    parser = argparse.ArgumentParser(
        description="Stage 3 training: single-vs-mixture detector"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            paths_cfg.get(
                "features_timeseries", "data/processed/features_timeseries.parquet"
            )
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage1_model_dir", "artifacts/models/stage1_mixture_detector"
            )
        ),
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage1_comparison", "artifacts/reports/stage1_model_comparison.json"
            )
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path(
            paths_cfg.get("stage1_metrics", "artifacts/reports/stage1_cv_metrics.json")
        ),
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage1_predictions",
                "artifacts/predictions/stage1_training_predictions.parquet",
            )
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=str(mlflow_cfg.get("stage1_experiment", "stage1_mixture_detection")),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    result = run_train_stage1_mixture_detection(
        features_path=args.features,
        model_dir=args.model_dir,
        comparison_path=args.comparison_out,
        metrics_path=args.metrics_out,
        predictions_path=args.predictions_out,
        mlflow_experiment_name=args.mlflow_experiment,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
