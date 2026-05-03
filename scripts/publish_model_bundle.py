from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.observed.utils.artifact_publisher import publish_model_bundle

_STAGE_CONFIG: dict[str, dict[str, str]] = {
    "stage1": {
        "model_name": "stage1-mixture-detector",
        "model_dir": "artifacts/models/stage1_mixture_detector",
    },
    "stage2": {
        "model_name": "stage2-gas-probability-model",
        "model_dir": "artifacts/models/stage2_gas_probability_model",
    },
}


def _default_model_version() -> str:
    return f"git-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _validate_bundle_inputs(model_dir: Path) -> None:
    required = [model_dir / "model.joblib", model_dir / "metadata.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model bundle is incomplete. Missing required files: " + ", ".join(missing)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish one stage model bundle (model.joblib + metadata.json + bundle-manifest.json) to S3-compatible storage."
    )
    parser.add_argument("--stage", required=True, choices=sorted(_STAGE_CONFIG.keys()))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local model directory containing model.joblib and metadata.json",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Version suffix for S3 key path (defaults to UTC timestamp if omitted)",
    )
    parser.add_argument(
        "--storage-env",
        type=str,
        default="prod",
        help="Storage environment segment used by storage path builder (default: prod)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write publish report JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _STAGE_CONFIG[args.stage]
    model_dir = args.model_dir or Path(cfg["model_dir"])
    model_version = (args.model_version or "").strip() or _default_model_version()
    report_path = args.report or Path(f"reports/{args.stage}_bundle_publish.json")

    _validate_bundle_inputs(model_dir)

    result: dict[str, Any] = publish_model_bundle(
        stage=args.stage,
        model_name=cfg["model_name"],
        model_dir=model_dir,
        model_version=model_version,
        storage_env=args.storage_env,
        source_info={"publisher": "ci-model-bundle-publish"},
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
