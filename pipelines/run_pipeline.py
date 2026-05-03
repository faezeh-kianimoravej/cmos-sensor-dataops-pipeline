from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipelines.batch_predict_flow import run_batch_prediction
from pipelines.evaluate_stage1_flow import run_evaluate_stage1
from pipelines.evaluate_stage2_flow import run_evaluate_stage2
from pipelines.feature_engineering_flow import run_feature_engineering
from pipelines.ingest_flow import run_ingestion
from pipelines.prepare_stage2_single_gas_subset_flow import run_prepare_stage2_subset
from pipelines.train_stage1_mixture_detection_flow import (
    run_train_stage1_mixture_detection,
)
from pipelines.train_stage2_single_gas_probability_flow import (
    run_train_stage2_single_gas_probability,
)
from src.observed.utils.config import load_config

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
# No-op CI touchpoint: changing this file intentionally triggers pipeline-related jobs.


def _log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def _load_pipeline_context(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hierarchical_cfg = config.get("hierarchical_pipeline", {})
    return {
        "config": config,
        "paths": hierarchical_cfg.get("paths", {}),
        "feature_engineering": hierarchical_cfg.get("feature_engineering", {}),
        "seeds": hierarchical_cfg.get("seeds", {}),
        "mlflow": hierarchical_cfg.get("mlflow", {}),
    }


def _path_from(config_value: Any, fallback: str) -> Path:
    return Path(str(config_value)) if config_value is not None else Path(fallback)


def run_stage1_pipeline(
    input_path: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    context = _load_pipeline_context(config_path)
    paths_cfg = context["paths"]
    feature_cfg = context["feature_engineering"]
    seeds_cfg = context["seeds"]
    mlflow_cfg = context["mlflow"]

    processed_timeseries_path = _path_from(
        paths_cfg.get("processed_timeseries"),
        "data/processed/processed_timeseries.parquet",
    )
    runs_path = _path_from(paths_cfg.get("runs"), "data/processed/runs.parquet")
    ingest_summary_path = _path_from(
        paths_cfg.get("ingestion_summary"), "artifacts/reports/ingestion_summary.json"
    )

    windowed_path = _path_from(
        paths_cfg.get("windowed_timeseries"),
        "data/processed/windowed_timeseries.parquet",
    )
    train_windows_path = _path_from(
        paths_cfg.get("train_windows"), "data/processed/train_windows.parquet"
    )
    test_windows_path = _path_from(
        paths_cfg.get("test_windows"), "data/processed/test_windows.parquet"
    )
    features_path = _path_from(
        paths_cfg.get("features_timeseries"),
        "data/processed/features_timeseries.parquet",
    )
    feature_summary_path = _path_from(
        paths_cfg.get("feature_engineering_summary"),
        "artifacts/reports/feature_engineering_summary.json",
    )

    stage1_model_dir = _path_from(
        paths_cfg.get("stage1_model_dir"), "artifacts/models/stage1_mixture_detector"
    )
    stage1_comparison = _path_from(
        paths_cfg.get("stage1_comparison"),
        "artifacts/reports/stage1_model_comparison.json",
    )
    stage1_cv_metrics = _path_from(
        paths_cfg.get("stage1_metrics"), "artifacts/reports/stage1_cv_metrics.json"
    )
    stage1_predictions_path = _path_from(
        paths_cfg.get("stage1_predictions"),
        "artifacts/predictions/stage1_training_predictions.parquet",
    )

    _log(
        "Starting stage 1 workflow: ingest -> feature engineering -> mixture detection"
    )

    _log("[1/3] Ingesting raw data")
    ingest_result = run_ingestion(
        input_path=input_path,
        config_path=config_path,
        processed_timeseries_path=processed_timeseries_path,
        runs_path=runs_path,
        summary_path=ingest_summary_path,
    )
    _log(f"[1/3] Ingestion complete: {ingest_summary_path}")

    _log("[2/3] Building features")
    feature_result = run_feature_engineering(
        processed_timeseries_path=processed_timeseries_path,
        windowed_path=windowed_path,
        train_windows_path=train_windows_path,
        test_windows_path=test_windows_path,
        features_output_path=features_path,
        summary_path=feature_summary_path,
        window_size=int(feature_cfg.get("window_size", 100)),
        step_size=int(feature_cfg.get("step_size", 50)),
        test_size=float(feature_cfg.get("test_size", 0.2)),
        random_state=int(seeds_cfg.get("global", 42)),
    )
    _log(f"[2/3] Feature engineering complete: {features_path}")

    _log("[3/3] Training stage 1 mixture detector")
    stage1_result = run_train_stage1_mixture_detection(
        features_path=features_path,
        model_dir=stage1_model_dir,
        comparison_path=stage1_comparison,
        metrics_path=stage1_cv_metrics,
        config_path=config_path,
        mlflow_experiment_name=str(
            mlflow_cfg.get("stage1_experiment", "stage1_mixture_detection")
        ),
        predictions_path=stage1_predictions_path,
    )
    _log(f"[3/3] Stage 1 training complete: {stage1_model_dir}")

    return {
        "stage": "stage1",
        "input_path": str(input_path),
        "ingest": ingest_result,
        "feature_engineering": feature_result,
        "train_stage1": stage1_result,
    }


def run_stage2_pipeline(
    config_path: Path = DEFAULT_CONFIG_PATH,
    features_path: Path | None = None,
) -> dict[str, Any]:
    context = _load_pipeline_context(config_path)
    paths_cfg = context["paths"]
    mlflow_cfg = context["mlflow"]

    features_path = features_path or _path_from(
        paths_cfg.get("features_timeseries"),
        "data/processed/features_timeseries.parquet",
    )
    stage2_subset_path = _path_from(
        paths_cfg.get("features_single_gas"),
        "data/processed/features_single_gas.parquet",
    )
    stage2_subset_summary = _path_from(
        paths_cfg.get("stage2_subset_summary"),
        "artifacts/reports/stage2_subset_summary.json",
    )

    stage2_model_dir = _path_from(
        paths_cfg.get("stage2_model_dir"),
        "artifacts/models/stage2_gas_probability_model",
    )
    stage2_comparison = _path_from(
        paths_cfg.get("stage2_comparison"),
        "artifacts/reports/stage2_model_comparison.json",
    )
    stage2_tuning = _path_from(
        paths_cfg.get("stage2_tuning"), "artifacts/reports/stage2_tuning_results.json"
    )
    stage2_cv_metrics = _path_from(
        paths_cfg.get("stage2_metrics"), "artifacts/reports/stage2_cv_metrics.json"
    )
    stage2_predictions_path = _path_from(
        paths_cfg.get("stage2_predictions"),
        "artifacts/predictions/stage2_training_predictions.parquet",
    )

    _log("Starting stage 2 workflow: single-gas subset -> probability model")

    _log("[1/2] Preparing single-gas subset")
    subset_result = run_prepare_stage2_subset(
        features_path=features_path,
        subset_output_path=stage2_subset_path,
        summary_path=stage2_subset_summary,
    )
    _log(f"[1/2] Stage 2 subset complete: {stage2_subset_path}")

    _log("[2/2] Training stage 2 probability model")
    stage2_result = run_train_stage2_single_gas_probability(
        single_features_path=stage2_subset_path,
        model_dir=stage2_model_dir,
        comparison_path=stage2_comparison,
        tuning_path=stage2_tuning,
        metrics_path=stage2_cv_metrics,
        config_path=config_path,
        mlflow_experiment_name=str(
            mlflow_cfg.get("stage2_experiment", "stage2_single_gas_probability")
        ),
        predictions_path=stage2_predictions_path,
    )
    _log(f"[2/2] Stage 2 training complete: {stage2_model_dir}")

    return {
        "stage": "stage2",
        "features_path": str(features_path),
        "subset": subset_result,
        "train_stage2": stage2_result,
    }


def run_evaluation_pipeline(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    context = _load_pipeline_context(config_path)
    paths_cfg = context["paths"]

    stage1_predictions_path = _path_from(
        paths_cfg.get("stage1_predictions"),
        "artifacts/predictions/stage1_training_predictions.parquet",
    )
    stage2_predictions_path = _path_from(
        paths_cfg.get("stage2_predictions"),
        "artifacts/predictions/stage2_training_predictions.parquet",
    )
    stage1_output_path = _path_from(
        paths_cfg.get("evaluate_stage1_report"),
        "artifacts/reports/evaluate_stage1_report.json",
    )
    stage2_output_path = _path_from(
        paths_cfg.get("evaluate_stage2_report"),
        "artifacts/reports/evaluate_stage2_report.json",
    )

    _log("Starting evaluation workflow")

    _log("[1/2] Evaluating stage 1 predictions")
    stage1_report = run_evaluate_stage1(
        stage1_predictions_path=stage1_predictions_path,
        output_report_path=stage1_output_path,
    )
    _log(f"[1/2] Stage 1 evaluation complete: {stage1_output_path}")

    _log("[2/2] Evaluating stage 2 predictions")
    stage2_report = run_evaluate_stage2(
        stage2_predictions_path=stage2_predictions_path,
        output_report_path=stage2_output_path,
    )
    _log(f"[2/2] Stage 2 evaluation complete: {stage2_output_path}")

    return {
        "stage": "evaluation",
        "stage1": stage1_report,
        "stage2": stage2_report,
        "stage1_output_path": str(stage1_output_path),
        "stage2_output_path": str(stage2_output_path),
    }


def run_batch_prediction_pipeline(
    config_path: Path = DEFAULT_CONFIG_PATH,
    features_path: Path | None = None,
    stage1_model_dir: Path | None = None,
    stage2_model_dir: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
    window_size: int | None = None,
    step_size: int | None = None,
) -> dict[str, Any]:
    context = _load_pipeline_context(config_path)
    paths_cfg = context["paths"]
    feature_cfg = context["feature_engineering"]

    features_path = features_path or _path_from(
        paths_cfg.get("features_timeseries"),
        "data/processed/features_timeseries.parquet",
    )
    stage1_model_dir = stage1_model_dir or _path_from(
        paths_cfg.get("stage1_model_dir"), "artifacts/models/stage1_mixture_detector"
    )
    stage2_model_dir = stage2_model_dir or _path_from(
        paths_cfg.get("stage2_model_dir"),
        "artifacts/models/stage2_gas_probability_model",
    )
    batch_predictions_path = output_path or _path_from(
        paths_cfg.get("batch_predictions"),
        "artifacts/predictions/batch_predictions.parquet",
    )
    batch_summary_path = summary_path or _path_from(
        paths_cfg.get("batch_prediction_summary"),
        "artifacts/reports/batch_prediction_summary.json",
    )
    effective_window_size = (
        window_size
        if window_size is not None
        else int(feature_cfg.get("window_size", 100))
    )
    effective_step_size = (
        step_size if step_size is not None else int(feature_cfg.get("step_size", 50))
    )

    _log("Starting batch prediction workflow")
    _log("[1/1] Running hierarchical batch prediction")
    batch_result = run_batch_prediction(
        features_path=features_path,
        stage1_model_dir=stage1_model_dir,
        stage2_model_dir=stage2_model_dir,
        output_path=batch_predictions_path,
        summary_path=batch_summary_path,
        window_size=effective_window_size,
        step_size=effective_step_size,
    )
    _log(f"[1/1] Batch prediction complete: {batch_predictions_path}")

    return {
        "stage": "batch_prediction",
        "features_path": str(features_path),
        "stage1_model_dir": str(stage1_model_dir),
        "stage2_model_dir": str(stage2_model_dir),
        "output_path": str(batch_predictions_path),
        "summary_path": str(batch_summary_path),
        "window_size": effective_window_size,
        "step_size": effective_step_size,
        "batch_prediction": batch_result,
    }


def run_full_pipeline(
    input_path: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    _log("Starting full end-to-end workflow")
    stage1_result = run_stage1_pipeline(input_path=input_path, config_path=config_path)
    stage2_result = run_stage2_pipeline(config_path=config_path)
    evaluation_result = run_evaluation_pipeline(config_path=config_path)
    batch_result = run_batch_prediction_pipeline(config_path=config_path)
    _log("Full workflow complete")

    return {
        "stage": "full",
        "stage1": stage1_result,
        "stage2": stage2_result,
        "evaluation": evaluation_result,
        "batch_prediction": batch_result,
    }


def run_ml_pipeline(
    input_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    return run_full_pipeline(input_path=input_path, config_path=config_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the observed gas sensor ML workflow from the command line"
    )
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file that defines pipeline defaults",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    full_parser = subparsers.add_parser(
        "full",
        parents=[common_parser],
        help="Run ingest, feature engineering, training, evaluation, and batch prediction",
    )
    full_parser.add_argument(
        "--input", type=Path, required=True, help="Path to the raw data root folder"
    )

    stage1_parser = subparsers.add_parser(
        "stage1",
        parents=[common_parser],
        help="Run ingest, feature engineering, and stage 1 training only",
    )
    stage1_parser.add_argument(
        "--input", type=Path, required=True, help="Path to the raw data root folder"
    )

    stage2_parser = subparsers.add_parser(
        "stage2",
        parents=[common_parser],
        help="Run stage 2 subset preparation and training only",
    )
    stage2_parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional path to the stage 1 feature table; defaults to the config value",
    )

    evaluation_parser = subparsers.add_parser(
        "evaluation",
        parents=[common_parser],
        help="Run stage 1 and stage 2 evaluation only",
    )
    evaluation_parser.add_argument(
        "--stage1-predictions",
        type=Path,
        default=None,
        help="Optional path to the stage 1 prediction artifact",
    )
    evaluation_parser.add_argument(
        "--stage2-predictions",
        type=Path,
        default=None,
        help="Optional path to the stage 2 prediction artifact",
    )
    evaluation_parser.add_argument(
        "--stage1-output",
        type=Path,
        default=None,
        help="Optional path to the stage 1 evaluation report",
    )
    evaluation_parser.add_argument(
        "--stage2-output",
        type=Path,
        default=None,
        help="Optional path to the stage 2 evaluation report",
    )

    batch_parser = subparsers.add_parser(
        "batch", parents=[common_parser], help="Run hierarchical batch prediction only"
    )
    batch_parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional path to the feature table used for batch prediction",
    )
    batch_parser.add_argument(
        "--stage1-model-dir",
        type=Path,
        default=None,
        help="Optional path to the stage 1 model directory",
    )
    batch_parser.add_argument(
        "--stage2-model-dir",
        type=Path,
        default=None,
        help="Optional path to the stage 2 model directory",
    )
    batch_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to the batch prediction parquet output",
    )
    batch_parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path to the batch prediction summary report",
    )
    batch_parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Optional sliding window size for batch prediction",
    )
    batch_parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Optional sliding step size for batch prediction",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "full":
        summary = run_full_pipeline(
            input_path=Path(args.input), config_path=config_path
        )
    elif args.command == "stage1":
        summary = run_stage1_pipeline(
            input_path=Path(args.input), config_path=config_path
        )
    elif args.command == "stage2":
        summary = run_stage2_pipeline(
            config_path=config_path, features_path=args.features
        )
    elif args.command == "evaluation":
        context = _load_pipeline_context(config_path)
        paths_cfg = context["paths"]
        if (
            args.stage1_predictions is not None
            or args.stage2_predictions is not None
            or args.stage1_output is not None
            or args.stage2_output is not None
        ):
            summary = {
                "stage": "evaluation",
                "stage1": run_evaluate_stage1(
                    stage1_predictions_path=args.stage1_predictions
                    or _path_from(
                        paths_cfg.get("stage1_predictions"),
                        "artifacts/predictions/stage1_training_predictions.parquet",
                    ),
                    output_report_path=args.stage1_output
                    or _path_from(
                        paths_cfg.get("evaluate_stage1_report"),
                        "artifacts/reports/evaluate_stage1_report.json",
                    ),
                ),
                "stage2": run_evaluate_stage2(
                    stage2_predictions_path=args.stage2_predictions
                    or _path_from(
                        paths_cfg.get("stage2_predictions"),
                        "artifacts/predictions/stage2_training_predictions.parquet",
                    ),
                    output_report_path=args.stage2_output
                    or _path_from(
                        paths_cfg.get("evaluate_stage2_report"),
                        "artifacts/reports/evaluate_stage2_report.json",
                    ),
                ),
            }
        else:
            summary = run_evaluation_pipeline(config_path=config_path)
    elif args.command == "batch":
        summary = run_batch_prediction_pipeline(
            config_path=config_path,
            features_path=args.features,
            stage1_model_dir=args.stage1_model_dir,
            stage2_model_dir=args.stage2_model_dir,
            output_path=args.output,
            summary_path=args.summary_out,
            window_size=args.window_size,
            step_size=args.step_size,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
