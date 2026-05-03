from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipelines.evaluate_stage1_flow import run_evaluate_stage1
from pipelines.evaluate_stage2_flow import run_evaluate_stage2
from src.observed.utils.config import load_config


def _evaluation_run_id() -> str:
    return f"manual-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def run_evaluation(
    stage1_predictions_path: Path,
    stage2_predictions_path: Path,
    stage1_output_path: Path,
    stage2_output_path: Path,
) -> dict[str, Any]:
    evaluation_run_id = _evaluation_run_id()
    storage_env = os.environ.get("OBSERVED_STORAGE_ENV")
    stage1_report = run_evaluate_stage1(
        stage1_predictions_path=stage1_predictions_path,
        output_report_path=stage1_output_path,
        evaluation_run_id=evaluation_run_id,
        storage_env=storage_env,
    )
    stage2_report = run_evaluate_stage2(
        stage2_predictions_path=stage2_predictions_path,
        output_report_path=stage2_output_path,
        evaluation_run_id=evaluation_run_id,
        storage_env=storage_env,
    )
    return {
        "stage1": stage1_report,
        "stage2": stage2_report,
        "evaluation_run_id": evaluation_run_id,
        "storage_environment": storage_env or "dev",
        "stage1_output_path": str(stage1_output_path),
        "stage2_output_path": str(stage2_output_path),
    }


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    paths_cfg = hp_cfg.get("paths", {})

    parser = argparse.ArgumentParser(
        description="Combined evaluation for stage1 and stage2 outputs"
    )
    parser.add_argument(
        "--stage1-predictions",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage1_predictions",
                "artifacts/predictions/stage1_training_predictions.parquet",
            )
        ),
    )
    parser.add_argument(
        "--stage2-predictions",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_predictions",
                "artifacts/predictions/stage2_training_predictions.parquet",
            )
        ),
    )
    parser.add_argument(
        "--stage1-output",
        type=Path,
        default=Path(
            paths_cfg.get(
                "evaluate_stage1_report",
                "artifacts/reports/evaluate_stage1_report.json",
            )
        ),
    )
    parser.add_argument(
        "--stage2-output",
        type=Path,
        default=Path(
            paths_cfg.get(
                "evaluate_stage2_report",
                "artifacts/reports/evaluate_stage2_report.json",
            )
        ),
    )
    args = parser.parse_args()

    summary = run_evaluation(
        stage1_predictions_path=args.stage1_predictions,
        stage2_predictions_path=args.stage2_predictions,
        stage1_output_path=args.stage1_output,
        stage2_output_path=args.stage2_output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
