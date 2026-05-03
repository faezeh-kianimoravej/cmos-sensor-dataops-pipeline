from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.stage2 import train_stage2_probability_model


def run_train_stage2_single_gas_probability(
    single_features_path: Path,
    model_dir: Path,
    comparison_path: Path,
    tuning_path: Path,
    metrics_path: Path,
    predictions_path: Path | None = None,
    mlflow_experiment_name: str | None = None,
    config_path: Path = Path("configs/default.yaml"),
) -> dict[str, Any]:
    return train_stage2_probability_model(
        single_features_path=single_features_path,
        model_dir=model_dir,
        comparison_path=comparison_path,
        tuning_path=tuning_path,
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
        description="Stage 5 training: stage2 gas-type probability model"
    )
    parser.add_argument(
        "--single-features",
        type=Path,
        default=Path(
            paths_cfg.get(
                "features_single_gas", "data/processed/features_single_gas.parquet"
            )
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_model_dir", "artifacts/models/stage2_gas_probability_model"
            )
        ),
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_comparison", "artifacts/reports/stage2_model_comparison.json"
            )
        ),
    )
    parser.add_argument(
        "--tuning-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_tuning", "artifacts/reports/stage2_tuning_results.json"
            )
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path(
            paths_cfg.get("stage2_metrics", "artifacts/reports/stage2_cv_metrics.json")
        ),
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_predictions",
                "artifacts/predictions/stage2_training_predictions.parquet",
            )
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=str(
            mlflow_cfg.get("stage2_experiment", "stage2_single_gas_probability")
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    result = run_train_stage2_single_gas_probability(
        single_features_path=args.single_features,
        model_dir=args.model_dir,
        comparison_path=args.comparison_out,
        tuning_path=args.tuning_out,
        metrics_path=args.metrics_out,
        predictions_path=args.predictions_out,
        mlflow_experiment_name=args.mlflow_experiment,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
