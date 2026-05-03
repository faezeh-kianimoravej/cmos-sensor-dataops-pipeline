from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    build_processed_timeseries,
    build_windowed_timeseries,
    split_windowed_by_run,
)

pytestmark = pytest.mark.unit


def _write_series(path: Path, values: list[float]) -> None:
    path.write_text("\n".join(str(v) for v in values), encoding="utf-8")


def test_build_processed_timeseries_and_windowing(tmp_path: Path) -> None:
    run_dir_1 = tmp_path / "exp" / "1"
    run_dir_2 = tmp_path / "exp" / "2"
    run_dir_1.mkdir(parents=True)
    run_dir_2.mkdir(parents=True)
    adc_1 = run_dir_1 / "adc.txt"
    t_1 = run_dir_1 / "time.txt"
    adc_2 = run_dir_2 / "adc.txt"
    t_2 = run_dir_2 / "time.txt"
    _write_series(adc_1, [float(i) for i in range(50)])
    _write_series(t_1, [float(i) for i in range(50)])
    _write_series(adc_2, [float(i) + 0.5 for i in range(50)])
    _write_series(t_2, [float(i) for i in range(50)])

    runs_df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "experiment": "Toluene",
                "experiment_folder": "exp",
                "run_folder": str(run_dir_1),
                "repeat_index": 1,
                "adc_file": str(adc_1),
                "time_file": str(t_1),
            },
            {
                "run_id": "run_2",
                "experiment": "Toluene",
                "experiment_folder": "exp",
                "run_folder": str(run_dir_2),
                "repeat_index": 2,
                "adc_file": str(adc_2),
                "time_file": str(t_2),
            },
        ]
    )

    processed = build_processed_timeseries(runs_df)
    assert len(processed) == 2
    assert processed.iloc[0]["n_samples"] == 50

    windowed = build_windowed_timeseries(processed, window_size=20, step_size=10)
    assert not windowed.empty
    train_df, test_df = split_windowed_by_run(windowed, test_size=0.5, random_state=42)
    assert len(train_df) + len(test_df) == len(windowed)
