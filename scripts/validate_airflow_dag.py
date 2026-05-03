from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Airflow DAG parsing")
    parser.add_argument("--dag-dir", default="dags")
    parser.add_argument("--report", default="reports/dag_parse_validation.json")
    args = parser.parse_args()

    dag_dir = Path(args.dag_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "status": "fail",
        "dag_dir": str(dag_dir),
        "checks": [],
    }

    if not dag_dir.exists() or not dag_dir.is_dir():
        result["checks"] = [
            {
                "name": "dag-directory-exists",
                "passed": False,
                "details": f"DAG directory not found: {dag_dir}",
            }
        ]
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit(1)

    dag_files = sorted(
        p for p in dag_dir.glob("*.py") if p.is_file() and not p.name.startswith("__")
    )
    if not dag_files:
        result["checks"] = [
            {
                "name": "dag-files-present",
                "passed": False,
                "details": f"No DAG Python files found in {dag_dir}",
            }
        ]
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit(1)

    syntax_errors: list[str] = []
    for dag_file in dag_files:
        try:
            source = dag_file.read_text(encoding="utf-8")
            compile(source, str(dag_file), "exec")
        except Exception as exc:
            syntax_errors.append(f"{dag_file}: {exc}")

    syntax_ok = not syntax_errors
    syntax_details = (
        f"Python syntax parse passed for {len(dag_files)} DAG file(s)."
        if syntax_ok
        else f"Syntax parse failed for DAG files: {syntax_errors}"
    )

    checks: list[dict[str, object]] = [
        {
            "name": "python-syntax-parse-all-dags",
            "passed": syntax_ok,
            "details": syntax_details,
        }
    ]

    airflow_parse_ok = True
    airflow_parse_details = "DagBag import and parse passed with no import errors."
    try:
        os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
        from airflow.models import DagBag  # type: ignore

        dagbag = DagBag(dag_folder=str(dag_dir), include_examples=False)
        if dagbag.import_errors:
            airflow_parse_ok = False
            airflow_parse_details = f"Import errors: {dagbag.import_errors}"
        elif not dagbag.dags:
            airflow_parse_ok = False
            airflow_parse_details = "No DAGs were loaded from dag directory"
        else:
            airflow_parse_details = (
                f"Loaded {len(dagbag.dags)} DAG(s) with no import errors."
            )
    except Exception as exc:
        airflow_parse_ok = False
        airflow_parse_details = f"Airflow parse failed: {exc}"

    checks.append(
        {
            "name": "airflow-dag-parse",
            "passed": airflow_parse_ok,
            "details": airflow_parse_details,
        }
    )

    overall_ok = all(bool(c["passed"]) for c in checks)
    result["status"] = "pass" if overall_ok else "fail"
    result["checks"] = checks
    result["dag_files"] = [str(p) for p in dag_files]

    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
