from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
RUNTIME_TMP = REPO_ROOT / ".runtime_tmp"


def _configure_runtime_tempdir() -> Path:
    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    resolved = RUNTIME_TMP.resolve()
    resolved_str = str(resolved)
    os.environ["TMPDIR"] = resolved_str
    os.environ["TMP"] = resolved_str
    os.environ["TEMP"] = resolved_str
    tempfile.tempdir = resolved_str
    return resolved


def _pipeline_api() -> dict[str, Any]:
    from pipelines.run_pipeline import (
        run_batch_prediction_pipeline,
        run_evaluation_pipeline,
        run_full_pipeline,
        run_stage1_pipeline,
        run_stage2_pipeline,
    )

    return {
        "run_batch_prediction_pipeline": run_batch_prediction_pipeline,
        "run_evaluation_pipeline": run_evaluation_pipeline,
        "run_full_pipeline": run_full_pipeline,
        "run_stage1_pipeline": run_stage1_pipeline,
        "run_stage2_pipeline": run_stage2_pipeline,
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2), flush=True)


def _clean_generated_artifacts() -> dict[str, Any]:
    targets = [
        Path("artifacts/models"),
        Path("artifacts/reports"),
        Path("artifacts/predictions"),
        Path("data/processed/processed_timeseries.parquet"),
        Path("data/processed/runs.parquet"),
        Path("data/processed/windowed_timeseries.parquet"),
        Path("data/processed/train_windows.parquet"),
        Path("data/processed/test_windows.parquet"),
        Path("data/processed/features_timeseries.parquet"),
        Path("data/processed/features_single_gas.parquet"),
    ]

    removed: list[str] = []
    missing: list[str] = []

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
        elif target.is_file():
            target.unlink()
            removed.append(str(target))
        else:
            missing.append(str(target))

    return {
        "status": "success",
        "removed": removed,
        "missing": missing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple developer entrypoint for end-to-end pipeline orchestration.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_pipeline.py run --input data/raw/CMOS_addtional_datasets\n"
            "  python scripts/run_pipeline.py train --input data/raw/CMOS_addtional_datasets\n"
            "  python scripts/run_pipeline.py evaluate\n"
            "  python scripts/run_pipeline.py predict\n"
            "  python scripts/run_pipeline.py clean"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline config YAML (default: configs/default.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run full pipeline in order: ingest -> feature_engineering -> train_stage1 -> prepare_stage2_subset -> train_stage2 -> evaluate -> batch_prediction",
    )
    run_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw data root for ingest stage",
    )
    run_parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip optional batch_prediction stage",
    )

    train_parser = subparsers.add_parser(
        "train",
        parents=[common],
        help="Run training path only: ingest -> feature_engineering -> train_stage1 -> prepare_stage2_subset -> train_stage2",
    )
    train_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw data root for ingest stage",
    )

    subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Run evaluation stage only (stage1 + stage2 reports)",
    )

    subparsers.add_parser(
        "predict",
        parents=[common],
        help="Run batch_prediction stage only",
    )

    subparsers.add_parser(
        "clean",
        help="Remove generated model/report/prediction artifacts and processed parquet outputs",
    )

    return parser


def main() -> None:
    _configure_runtime_tempdir()
    parser = build_parser()
    args = parser.parse_args()

    api = None if args.command == "clean" else _pipeline_api()

    if args.command == "run":
        if args.skip_batch:
            stage1 = api["run_stage1_pipeline"](
                input_path=args.input, config_path=args.config
            )
            stage2 = api["run_stage2_pipeline"](config_path=args.config)
            evaluation = api["run_evaluation_pipeline"](config_path=args.config)
            _print_json(
                {
                    "stage": "full_without_batch",
                    "stage1": stage1,
                    "stage2": stage2,
                    "evaluation": evaluation,
                    "batch_prediction": "skipped",
                }
            )
        else:
            _print_json(
                api["run_full_pipeline"](input_path=args.input, config_path=args.config)
            )
        return

    if args.command == "train":
        _print_json(
            {
                "stage": "train",
                "stage1": api["run_stage1_pipeline"](
                    input_path=args.input, config_path=args.config
                ),
                "stage2": api["run_stage2_pipeline"](config_path=args.config),
            }
        )
        return

    if args.command == "evaluate":
        _print_json(api["run_evaluation_pipeline"](config_path=args.config))
        return

    if args.command == "predict":
        _print_json(api["run_batch_prediction_pipeline"](config_path=args.config))
        return

    if args.command == "clean":
        _print_json(_clean_generated_artifacts())
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
