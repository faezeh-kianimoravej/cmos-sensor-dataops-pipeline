from __future__ import annotations

from pathlib import Path

import pytest

from src.data.parser import discover_text_runs, load_vector_file

pytestmark = pytest.mark.unit


def test_load_vector_file_reads_numeric_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "adc.txt"
    file_path.write_text("\n".join(str(i) for i in range(20)), encoding="utf-8")

    series = load_vector_file(file_path)
    assert series is not None
    assert len(series) == 20


def test_discover_text_runs_case_insensitive(tmp_path: Path) -> None:
    run_dir = tmp_path / "Toluene_Test" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "ADC_values.CSV").write_text(
        "\n".join(str(i) for i in range(20)), encoding="utf-8"
    )
    (run_dir / "TIME_VALUES.txt").write_text(
        "\n".join(str(i) for i in range(20)), encoding="utf-8"
    )

    runs = discover_text_runs(tmp_path)
    assert len(runs) == 1
    assert "adc_file" in runs.columns
    assert "time_file" in runs.columns
