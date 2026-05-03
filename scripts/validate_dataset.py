from __future__ import annotations

import ast
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure repo root is importable when running as `python scripts/validate_dataset.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observed.utils.config import load_config


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _flatten_numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []

    # Handle pyarrow-style scalar wrappers.
    if hasattr(value, "as_py"):
        try:
            value = value.as_py()
        except Exception:
            pass

    # Handle numpy/pandas containers.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass

    # Handle stringified sequences, e.g. "[0.1, 0.2]".
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
        value = parsed

    if not isinstance(value, (list, tuple, set)):
        return []

    out: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple, set)):
            out.extend(_flatten_numeric_sequence(item))
            continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _write_reports(
    results: list[CheckResult], output_json: Path, output_md: Path
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "pass" if all(r.passed for r in results) else "fail",
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "details": r.details,
            }
            for r in results
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Data Validation Report",
        "",
        f"Overall status: **{payload['status'].upper()}**",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"| {result.name} | {status} | {result.details} |")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_console_summary(results: list[CheckResult]) -> None:
    status = "PASS" if all(r.passed for r in results) else "FAIL"
    print(f"[validate-dataset] Overall status: {status}")
    for result in results:
        result_status = "PASS" if result.passed else "FAIL"
        print(f"[validate-dataset] {result.name}: {result_status} - {result.details}")


def _check_required_files(required_files: list[Path]) -> CheckResult:
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        return CheckResult(
            name="required-files-exist",
            passed=False,
            details=f"Missing files: {', '.join(missing)}",
        )
    return CheckResult(
        name="required-files-exist",
        passed=True,
        details="All required files are present.",
    )


def _check_columns_exist(
    df: pd.DataFrame, required_columns: list[str], label: str
) -> CheckResult:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return CheckResult(
            name=f"expected-columns-{label}",
            passed=False,
            details=f"Missing columns: {', '.join(missing)}",
        )
    return CheckResult(
        name=f"expected-columns-{label}",
        passed=True,
        details=f"All expected columns exist in {label}.",
    )


def _check_columns_any_schema(
    df: pd.DataFrame, schema_options: list[list[str]], label: str
) -> CheckResult:
    for columns in schema_options:
        missing = [c for c in columns if c not in df.columns]
        if not missing:
            return CheckResult(
                name=f"expected-columns-{label}",
                passed=True,
                details=f"Columns match a supported schema for {label}.",
            )

    option_missing = []
    for idx, columns in enumerate(schema_options, start=1):
        missing = [c for c in columns if c not in df.columns]
        option_missing.append(f"option{idx}: {', '.join(missing)}")

    return CheckResult(
        name=f"expected-columns-{label}",
        passed=False,
        details="No supported schema matched. " + " | ".join(option_missing),
    )


def _check_missing_values(
    df: pd.DataFrame, columns: list[str], label: str
) -> CheckResult:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return CheckResult(
            name=f"missing-values-{label}",
            passed=False,
            details="No overlapping columns to validate missing values.",
        )

    missing_counts = df[cols].isna().sum()
    offenders = {c: int(v) for c, v in missing_counts.items() if int(v) > 0}

    if offenders:
        details = ", ".join(f"{k}={v}" for k, v in offenders.items())
        return CheckResult(
            name=f"missing-values-{label}",
            passed=False,
            details=f"Unexpected missing values found: {details}",
        )
    return CheckResult(
        name=f"missing-values-{label}",
        passed=True,
        details=f"No unexpected missing values in {label}.",
    )


def _check_sensor_ranges(
    processed_df: pd.DataFrame, windowed_df: pd.DataFrame
) -> CheckResult:
    issues: list[str] = []

    normalized_col = None
    if "signal_normalized" in processed_df.columns:
        normalized_col = "signal_normalized"
    elif "signal_norm" in processed_df.columns:
        normalized_col = "signal_norm"

    if normalized_col is None:
        issues.append("processed_timeseries missing normalized signal column")
    else:
        normalized_values: list[float] = []
        for item in processed_df[normalized_col].tolist():
            seq = _flatten_numeric_sequence(item)
            if seq:
                normalized_values.extend(seq)
                continue
            try:
                normalized_values.append(float(item))
            except (TypeError, ValueError):
                continue

        if not normalized_values:
            issues.append(
                f"processed_timeseries {normalized_col} has no numeric values"
            )
        else:
            min_val = min(normalized_values)
            max_val = max(normalized_values)
            if min_val < -1e-9 or max_val > 1 + 1e-9:
                issues.append(
                    f"processed_timeseries {normalized_col} out of [0, 1]: min={min_val:.6f}, max={max_val:.6f}"
                )

    if "signal_window" not in windowed_df.columns:
        issues.append("windowed_timeseries missing signal_window column")
    else:
        for idx, values in enumerate(windowed_df["signal_window"].tolist()):
            seq = _flatten_numeric_sequence(values)
            if not seq:
                issues.append(
                    f"windowed_timeseries row {idx} has empty/non-numeric signal_window"
                )
                continue
            # Allow a small tolerance because window slices can include minor
            # numerical spillover around normalization boundaries.
            if min(seq) < -0.05 or max(seq) > 1.05:
                issues.append(
                    f"windowed_timeseries row {idx} signal_window out of tolerance [-0.05, 1.05]"
                )
                break

    if issues:
        return CheckResult(
            name="sensor-value-ranges",
            passed=False,
            details="; ".join(issues),
        )
    return CheckResult(
        name="sensor-value-ranges",
        passed=True,
        details="Normalized sensor values are within [0, 1].",
    )


def _check_train_test_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> CheckResult:
    issues: list[str] = []

    if "run_id" not in train_df.columns or "run_id" not in test_df.columns:
        issues.append("run_id column missing from train/test windows")
    else:
        train_runs = set(train_df["run_id"].astype(str).tolist())
        test_runs = set(test_df["run_id"].astype(str).tolist())
        overlap = sorted(train_runs.intersection(test_runs))
        if overlap:
            sample = ", ".join(overlap[:10])
            issues.append(f"run_id overlap between train and test: {sample}")

    if "window_id" in train_df.columns and "window_id" in test_df.columns:
        train_windows = set(train_df["window_id"].astype(str).tolist())
        test_windows = set(test_df["window_id"].astype(str).tolist())
        overlap_windows = train_windows.intersection(test_windows)
        if overlap_windows:
            issues.append(f"window_id overlap count: {len(overlap_windows)}")

    if issues:
        return CheckResult(
            name="train-test-leakage",
            passed=False,
            details="; ".join(issues),
        )
    return CheckResult(
        name="train-test-leakage",
        passed=True,
        details="No train/test leakage detected.",
    )


def main() -> None:
    config = load_config("configs/default.yaml")
    hp_paths = config.get("hierarchical_pipeline", {}).get("paths", {})

    processed_path = Path(
        hp_paths.get(
            "processed_timeseries", "data/processed/processed_timeseries.parquet"
        )
    )
    windowed_path = Path(
        hp_paths.get(
            "windowed_timeseries", "data/processed/windowed_timeseries.parquet"
        )
    )
    train_path = Path(
        hp_paths.get("train_windows", "data/processed/train_windows.parquet")
    )
    test_path = Path(
        hp_paths.get("test_windows", "data/processed/test_windows.parquet")
    )
    features_path = Path(
        hp_paths.get(
            "features_timeseries", "data/processed/features_timeseries.parquet"
        )
    )

    report_dir = Path("reports")
    json_report_path = report_dir / "data_validation.json"
    md_report_path = report_dir / "data_validation.md"

    required_files = [
        processed_path,
        windowed_path,
        train_path,
        test_path,
        features_path,
    ]
    results: list[CheckResult] = []
    results.append(_check_required_files(required_files))

    if not results[-1].passed:
        _write_reports(results, json_report_path, md_report_path)
        _print_console_summary(results)
        raise SystemExit(1)

    processed_df = pd.read_parquet(processed_path)
    windowed_df = pd.read_parquet(windowed_path)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    features_df = pd.read_parquet(features_path)

    expected_processed_cols = [
        "run_id",
        "time",
        "signal",
    ]
    expected_processed_missing_cols = expected_processed_cols + [
        "signal_smoothed",
        "signal_normalized",
        "signal_smooth",
        "signal_norm",
        "n_samples",
    ]
    processed_schema_options = [
        expected_processed_cols + ["signal_smoothed", "signal_normalized", "n_samples"],
        expected_processed_cols + ["signal_smooth", "signal_norm"],
    ]
    expected_windowed_cols = [
        "run_id",
        "window_id",
        "start_idx",
        "end_idx",
        "signal_window",
    ]
    expected_feature_cols_base = [
        "run_id",
        "window_id",
        "trend_slope",
        "trend_type",
        "peak_value",
        "peak_position",
        "rise_strength",
        "fall_strength",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "iqr",
        "first_diff_mean",
        "first_diff_std",
        "first_diff_max",
        "first_diff_min",
        "abs_diff_mean",
        "net_change",
        "energy",
        "skewness",
        "kurtosis",
        "flatness",
        "zero_crossing_rate_diff",
        "signal_start",
        "signal_end",
        "peak_to_start_distance",
        "peak_to_end_distance",
        "relative_peak_position",
    ]
    feature_schema_options = [
        expected_feature_cols_base + ["q25", "q75", "sign_changes"],
        expected_feature_cols_base + ["range", "cv", "diff_sign_changes"],
    ]
    expected_feature_cols = expected_feature_cols_base + [
        "q25",
        "q75",
        "sign_changes",
        "range",
        "cv",
        "diff_sign_changes",
    ]

    results.append(
        _check_columns_any_schema(
            processed_df, processed_schema_options, "processed_timeseries"
        )
    )
    results.append(
        _check_columns_exist(windowed_df, expected_windowed_cols, "windowed_timeseries")
    )
    results.append(
        _check_columns_any_schema(
            features_df, feature_schema_options, "features_timeseries"
        )
    )

    results.append(
        _check_missing_values(
            processed_df, expected_processed_missing_cols, "processed_timeseries"
        )
    )
    results.append(
        _check_missing_values(
            windowed_df, expected_windowed_cols, "windowed_timeseries"
        )
    )
    results.append(
        _check_missing_values(features_df, expected_feature_cols, "features_timeseries")
    )

    results.append(_check_sensor_ranges(processed_df, windowed_df))
    results.append(_check_train_test_leakage(train_df, test_df))

    _write_reports(results, json_report_path, md_report_path)
    _print_console_summary(results)

    failed = [r for r in results if not r.passed]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        report_dir = Path("reports")
        json_report_path = report_dir / "data_validation.json"
        md_report_path = report_dir / "data_validation.md"
        failure = CheckResult(
            name="validation-runtime-error",
            passed=False,
            details=f"{type(exc).__name__}: {exc}",
        )
        _write_reports([failure], json_report_path, md_report_path)
        _print_console_summary([failure])
        traceback.print_exc()
        raise SystemExit(1)
