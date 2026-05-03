from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.inference import run_two_stage_batch_prediction
from src.observed.utils.config import load_config


def run_batch_prediction(
    features_path: Path,
    stage1_model_dir: Path,
    stage2_model_dir: Path,
    output_path: Path,
    summary_path: Path,
    window_size: int = 100,
    step_size: int = 50,
) -> dict[str, Any]:
    return run_two_stage_batch_prediction(
        features_path=features_path,
        stage1_model_dir=stage1_model_dir,
        stage2_model_dir=stage2_model_dir,
        output_path=output_path,
        summary_path=summary_path,
        window_size=window_size,
        step_size=step_size,
    )


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    paths_cfg = hp_cfg.get("paths", {})
    fe_cfg = hp_cfg.get("feature_engineering", {})

    parser = argparse.ArgumentParser(
        description="Stage 7 batch prediction and probability artifacting"
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
        "--stage1-model-dir",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage1_model_dir", "artifacts/models/stage1_mixture_detector"
            )
        ),
    )
    parser.add_argument(
        "--stage2-model-dir",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_model_dir", "artifacts/models/stage2_gas_probability_model"
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            paths_cfg.get(
                "batch_predictions", "artifacts/predictions/batch_predictions.parquet"
            )
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "batch_prediction_summary",
                "artifacts/reports/batch_prediction_summary.json",
            )
        ),
    )
    parser.add_argument(
        "--window-size", type=int, default=int(fe_cfg.get("window_size", 100))
    )
    parser.add_argument(
        "--step-size", type=int, default=int(fe_cfg.get("step_size", 50))
    )
    args = parser.parse_args()

    summary = run_batch_prediction(
        features_path=args.features,
        stage1_model_dir=args.stage1_model_dir,
        stage2_model_dir=args.stage2_model_dir,
        output_path=args.output,
        summary_path=args.summary_out,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
