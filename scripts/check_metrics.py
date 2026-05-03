from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MetricCheck:
    name: str
    value: float | None
    threshold: float
    passed: bool
    details: str


@dataclass
class BaselineCheck:
    name: str
    current: float
    baseline: float
    delta: float
    passed: bool


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return loaded


def _find_metric_value(payload: Any, aliases: list[str]) -> float | None:
    normalized_aliases = {a.lower() for a in aliases}

    def walk(node: Any) -> float | None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in normalized_aliases:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
            for value in node.values():
                result = walk(value)
                if result is not None:
                    return result
        elif isinstance(node, list):
            for item in node:
                result = walk(item)
                if result is not None:
                    return result
        return None

    return walk(payload)


def _write_reports(
    metric_checks: list[MetricCheck],
    baseline_checks: list[BaselineCheck],
    output_json: Path,
    output_md: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    failed_thresholds = [c for c in metric_checks if not c.passed]
    failed_baselines = [c for c in baseline_checks if not c.passed]
    overall_pass = not failed_thresholds and not failed_baselines

    payload = {
        "status": "pass" if overall_pass else "fail",
        "threshold_checks": [
            {
                "name": c.name,
                "value": c.value,
                "threshold": c.threshold,
                "passed": c.passed,
                "details": c.details,
            }
            for c in metric_checks
        ],
        "baseline_checks": [
            {
                "name": c.name,
                "current": c.current,
                "baseline": c.baseline,
                "delta": c.delta,
                "passed": c.passed,
            }
            for c in baseline_checks
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Model Validation Report",
        "",
        f"Overall status: **{payload['status'].upper()}**",
        "",
        "## Threshold Checks",
        "",
        "| Metric | Value | Threshold | Status | Details |",
        "|---|---:|---:|---|---|",
    ]

    for c in metric_checks:
        value_text = "N/A" if c.value is None else f"{c.value:.6f}"
        lines.append(
            f"| {c.name} | {value_text} | {c.threshold:.6f} | {'PASS' if c.passed else 'FAIL'} | {c.details} |"
        )

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            "| Metric | Current | Baseline | Delta | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )

    if baseline_checks:
        for c in baseline_checks:
            lines.append(
                f"| {c.name} | {c.current:.6f} | {c.baseline:.6f} | {c.delta:+.6f} | {'PASS' if c.passed else 'FAIL'} |"
            )
    else:
        lines.append(
            "| - | - | - | - | SKIPPED (baseline file not found or incomplete) |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate model metrics against thresholds and optional baseline"
    )
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("reports/baseline_metrics.json")
    )
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/model_validation.json")
    )
    parser.add_argument(
        "--report-md", type=Path, default=Path("reports/model_validation.md")
    )

    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--min-precision", type=float, default=0.70)
    parser.add_argument("--min-recall", type=float, default=0.70)
    args = parser.parse_args()

    metrics_payload = _load_json(args.metrics)

    metric_specs = [
        ("f1", ["f1", "f1_score", "macro_f1", "weighted_f1"], args.min_f1),
        (
            "precision",
            ["precision", "precision_score", "macro_precision", "weighted_precision"],
            args.min_precision,
        ),
        (
            "recall",
            ["recall", "recall_score", "macro_recall", "weighted_recall"],
            args.min_recall,
        ),
    ]

    threshold_checks: list[MetricCheck] = []
    current_values: dict[str, float] = {}

    for metric_name, aliases, threshold in metric_specs:
        value = _find_metric_value(metrics_payload, aliases)
        if value is None:
            threshold_checks.append(
                MetricCheck(
                    name=metric_name,
                    value=None,
                    threshold=threshold,
                    passed=False,
                    details=f"Metric not found. Looked for aliases: {aliases}",
                )
            )
            continue

        current_values[metric_name] = value
        passed = value >= threshold
        threshold_checks.append(
            MetricCheck(
                name=metric_name,
                value=value,
                threshold=threshold,
                passed=passed,
                details=(
                    f"Value {value:.6f} meets threshold {threshold:.6f}"
                    if passed
                    else f"Value {value:.6f} below threshold {threshold:.6f}"
                ),
            )
        )

    baseline_checks: list[BaselineCheck] = []
    if args.baseline.exists():
        baseline_payload = _load_json(args.baseline)
        for metric_name, aliases, _ in metric_specs:
            if metric_name not in current_values:
                continue
            baseline_value = _find_metric_value(baseline_payload, aliases)
            if baseline_value is None:
                continue
            current = current_values[metric_name]
            delta = current - baseline_value
            baseline_checks.append(
                BaselineCheck(
                    name=metric_name,
                    current=current,
                    baseline=baseline_value,
                    delta=delta,
                    passed=(delta >= 0),
                )
            )

    _write_reports(threshold_checks, baseline_checks, args.report_json, args.report_md)

    failed_thresholds = [c for c in threshold_checks if not c.passed]
    failed_baselines = [c for c in baseline_checks if not c.passed]
    if failed_thresholds or failed_baselines:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
