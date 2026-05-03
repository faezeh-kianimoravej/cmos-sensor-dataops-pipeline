from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.stage1 import create_true_single_gas_subset
from src.observed.utils.config import load_config


def run_prepare_stage2_subset(
    features_path: Path,
    subset_output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    return create_true_single_gas_subset(
        features_path=features_path,
        subset_output_path=subset_output_path,
        summary_path=summary_path,
    )


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_cfg = config.get("hierarchical_pipeline", {})
    paths_cfg = hp_cfg.get("paths", {})

    parser = argparse.ArgumentParser(
        description="Stage 4 subset creation: true single-gas rows"
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
        "--subset-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "features_single_gas", "data/processed/features_single_gas.parquet"
            )
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            paths_cfg.get(
                "stage2_subset_summary", "artifacts/reports/stage2_subset_summary.json"
            )
        ),
    )
    args = parser.parse_args()

    result = run_prepare_stage2_subset(
        features_path=args.features,
        subset_output_path=args.subset_out,
        summary_path=args.summary_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
