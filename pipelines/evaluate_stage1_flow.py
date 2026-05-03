from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    evaluate_stage1_from_predictions,
    log_evaluation_to_mlflow,
    write_report,
)
from src.observed.utils.artifact_publisher import publish_local_artifact
from src.observed.utils.config import load_config


def _default_evaluation_run_id() -> str:
    return f"manual-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def run_evaluate_stage1(
    stage1_predictions_path: Path,
    output_report_path: Path,
    mlflow_experiment_name: str = "stage1_mixture_detection",
    evaluation_run_id: str | None = None,
    storage_env: str | None = None,
) -> dict[str, Any]:
    evaluation_run_id = evaluation_run_id or _default_evaluation_run_id()
    report = evaluate_stage1_from_predictions(stage1_predictions_path)
    written = write_report(report, output_report_path)
    published_report = publish_local_artifact(
        local_path=output_report_path,
        domain="reports",
        env=storage_env,
        pipeline="stage1",
        report_family="evaluation-report",
        run_id=evaluation_run_id,
    )
    log_evaluation_to_mlflow(
        experiment_name=mlflow_experiment_name,
        run_name="stage1_evaluation",
        dataset_path=stage1_predictions_path,
        report=written,
        output_path=output_report_path,
        artifacts=[output_report_path],
        selected_best_model="stage1_final_selected_model",
    )
    return {
        **written,
        "output_report_path": str(output_report_path),
        "output_storage_target": published_report["storage_target"],
        "output_storage_upload": published_report["storage_upload"],
    }


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    paths_cfg = hp_cfg.get("paths", {})
    mlflow_cfg = hp_cfg.get("mlflow", {})

    parser = argparse.ArgumentParser(
        description="Stage 6 evaluation for stage1 outputs"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage1_predictions",
                "artifacts/predictions/stage1_training_predictions.parquet",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            paths_cfg.get(
                "evaluate_stage1_report",
                "artifacts/reports/evaluate_stage1_report.json",
            )
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=str(mlflow_cfg.get("stage1_experiment", "stage1_mixture_detection")),
    )
    args = parser.parse_args()

    report = run_evaluate_stage1(
        stage1_predictions_path=args.predictions,
        output_report_path=args.output,
        mlflow_experiment_name=args.mlflow_experiment,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
