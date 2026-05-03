from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_vector_file(file_path: Path) -> pd.Series | None:
    df = None
    for sep in [",", "\t", r"\s+"]:
        try:
            if sep == r"\s+":
                temp_df = pd.read_csv(file_path, sep=sep, engine="python", header=None)
            else:
                temp_df = pd.read_csv(file_path, delimiter=sep, header=None)
            if temp_df.shape[0] >= 10:
                df = temp_df
                break
        except Exception:
            continue

    if df is None:
        return None

    values = (
        pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().reset_index(drop=True)
    )
    if len(values) < 10 or values.nunique() < 3:
        return None
    return values


def _normalize_experiment_name(folder_name: str) -> str:
    return " ".join(folder_name.replace("_", " ").split()).strip()


def _infer_repeat_index(run_folder: Path) -> int | None:
    try:
        return int(run_folder.name)
    except ValueError:
        return None


def discover_text_runs(raw_root: Path) -> pd.DataFrame:
    """
    Discover measurement runs with robust, case-insensitive file matching.
    Aligns with notebook's data understanding approach.
    """
    records: list[dict[str, Any]] = []
    supported_exts = {".txt", ".csv", ".tsv"}

    # Group files by parent folder to match runs
    run_folders: dict[Path, dict[str, list]] = {}

    for file in raw_root.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in supported_exts:
            continue
        if "explanation" in file.name.lower():
            continue

        run_folder = file.parent
        if run_folder not in run_folders:
            run_folders[run_folder] = {"adc": [], "time": []}

        # Case-insensitive classification, matching notebook logic
        file_name_lower = file.name.lower()
        if "adc" in file_name_lower:
            run_folders[run_folder]["adc"].append(file)
        elif "time" in file_name_lower:
            run_folders[run_folder]["time"].append(file)

    # Process runs that have both ADC and time files
    for run_folder, files in run_folders.items():
        if not files["adc"] or not files["time"]:
            continue

        adc_file = sorted(files["adc"])[0]
        time_file = sorted(files["time"])[0]

        experiment_folder = run_folder.parent
        experiment_name = _normalize_experiment_name(experiment_folder.name)
        repeat_index = _infer_repeat_index(run_folder)

        if pd.notna(repeat_index):
            run_id = f"{experiment_name}__run_{repeat_index}"
        else:
            run_id = f"{experiment_name}__{run_folder.name}"
        records.append(
            {
                "run_id": run_id,
                "experiment": experiment_name,
                "experiment_folder": experiment_folder.name,
                "run_folder": str(run_folder),
                "repeat_index": repeat_index,
                "adc_file": str(adc_file),
                "time_file": str(time_file),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "run_id",
                "experiment",
                "experiment_folder",
                "run_folder",
                "repeat_index",
                "adc_file",
                "time_file",
            ]
        )

    runs_df = (
        pd.DataFrame(records)
        .sort_values(["experiment", "run_id"])
        .reset_index(drop=True)
    )
    return runs_df
