from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_engineering_flow import run_feature_engineering
from pipelines.ingest_flow import run_ingestion

pytestmark = pytest.mark.unit


def test_run_ingestion_writes_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    fake_runs = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "experiment": "Toluene",
                "experiment_folder": "exp",
                "run_folder": "run",
                "repeat_index": 1,
                "adc_file": "a",
                "time_file": "t",
            }
        ]
    )
    fake_processed = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "experiment": "Toluene",
                "experiment_folder": "exp",
                "run_folder": "run",
                "repeat_index": 1,
                "time": list(range(20)),
                "signal": list(range(20)),
                "signal_smoothed": list(range(20)),
                "signal_normalized": [float(i) / 20.0 for i in range(20)],
                "n_samples": 20,
            }
        ]
    )

    monkeypatch.setattr("pipelines.ingest_flow.discover_text_runs", lambda _: fake_runs)
    monkeypatch.setattr(
        "pipelines.ingest_flow.build_processed_timeseries", lambda _: fake_processed
    )

    out_proc = tmp_path / "data" / "processed" / "processed_timeseries.parquet"
    out_runs = tmp_path / "data" / "processed" / "runs.parquet"
    out_summary = tmp_path / "artifacts" / "reports" / "ingestion_summary.json"

    result = run_ingestion(
        tmp_path / "raw", tmp_path / "cfg.yaml", out_proc, out_runs, out_summary
    )
    assert out_proc.exists()
    assert out_runs.exists()
    assert out_summary.exists()
    assert result["status"] == "success"
    assert str(out_proc).endswith("processed_timeseries.parquet")


def test_run_feature_engineering_outputs_dvc_compatible_paths(tmp_path: Path) -> None:
    processed = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "experiment": "Toluene",
                "experiment_folder": "exp",
                "run_folder": "run",
                "repeat_index": 1,
                "time": [float(i) for i in range(40)],
                "signal_normalized": [float(i) / 40.0 for i in range(40)],
            },
            {
                "run_id": "r2",
                "experiment": "2-butanone",
                "experiment_folder": "exp",
                "run_folder": "run2",
                "repeat_index": 2,
                "time": [float(i) for i in range(40)],
                "signal_normalized": [float(i + 1) / 41.0 for i in range(40)],
            },
        ]
    )
    processed_path = tmp_path / "data" / "processed" / "processed_timeseries.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(processed_path, index=False)

    windowed = tmp_path / "data" / "processed" / "windowed_timeseries.parquet"
    train_w = tmp_path / "data" / "processed" / "train_windows.parquet"
    test_w = tmp_path / "data" / "processed" / "test_windows.parquet"
    features = tmp_path / "data" / "processed" / "features_timeseries.parquet"
    summary = tmp_path / "artifacts" / "reports" / "feature_engineering_summary.json"

    result = run_feature_engineering(
        processed_path, windowed, train_w, test_w, features, summary, 20, 10, 0.2, 42
    )
    assert result["status"] == "success"
    assert windowed.exists() and features.exists() and summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["windowed_path"].endswith("windowed_timeseries.parquet")
    assert payload["features_output_path"].endswith("features_timeseries.parquet")
