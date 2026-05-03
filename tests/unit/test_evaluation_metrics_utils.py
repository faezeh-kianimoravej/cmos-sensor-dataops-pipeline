from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.metrics import (
    evaluate_stage1_from_predictions,
    evaluate_stage2_from_predictions,
    write_report,
)

pytestmark = pytest.mark.unit


def test_evaluate_stage1_from_predictions(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"target_mixture": [0, 1, 0, 1], "stage1_prediction": [0, 1, 1, 1]}
    )
    p = tmp_path / "s1.parquet"
    df.to_parquet(p, index=False)

    report = evaluate_stage1_from_predictions(p)
    assert report["target_stage"] == "stage1"
    assert report["rows"] == 4


def test_evaluate_stage2_from_predictions_and_write_report(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"target_gas_type": [0, 1, 0, 1], "stage2_prediction": [0, 1, 0, 0]}
    )
    p = tmp_path / "s2.parquet"
    df.to_parquet(p, index=False)

    report = evaluate_stage2_from_predictions(p)
    out = tmp_path / "report.json"
    write_report(report, out)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["target_stage"] == "stage2"
    assert "accuracy" in saved
