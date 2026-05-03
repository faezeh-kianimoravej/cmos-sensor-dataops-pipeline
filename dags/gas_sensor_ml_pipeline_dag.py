from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from src.observed.utils.config import load_config

_config = load_config("configs/default.yaml")
_hp_cfg = _config.get("hierarchical_pipeline", {})
_paths_cfg = _hp_cfg.get("paths", {})
_airflow_cfg = _config.get("airflow", {})

PROJECT_ROOT = os.environ.get(
    "OBSERVED_PROJECT_ROOT",
    str(_airflow_cfg.get("project_root", "/opt/airflow/workspace")),
)
PYTHON_BIN = os.environ.get(
    "OBSERVED_PYTHON_BIN", str(_airflow_cfg.get("python_bin", "python"))
)


def _to_date(value: str, fallback: datetime) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return fallback


def _run(cmd: str) -> str:
    return (
        "set -euo pipefail\n"
        f"echo '[airflow] Running: {cmd}'\n"
        f"cd {PROJECT_ROOT}\n"
        f"{PYTHON_BIN} -m {cmd}\n"
    )


DEFAULT_ARGS = {
    "owner": str(_airflow_cfg.get("owner", "dataops")),
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": int(_airflow_cfg.get("retries", 2)),
    "retry_delay": timedelta(minutes=int(_airflow_cfg.get("retry_delay_minutes", 5))),
}


def _log_run_metadata() -> None:
    # Read metadata from externally triggered runs (e.g., GitLab CI) while remaining
    # compatible with scheduled runs where dag_run.conf may be empty.
    from airflow.operators.python import get_current_context

    context = get_current_context()
    dag_run = context.get("dag_run")

    raw_conf = dag_run.conf if dag_run and dag_run.conf else {}
    gitlab_conf = raw_conf.get("gitlab", {}) if isinstance(raw_conf, dict) else {}

    commit_sha = gitlab_conf.get("commit_sha") or "N/A"
    branch = gitlab_conf.get("branch") or "N/A"
    pipeline_id = gitlab_conf.get("pipeline_id") or "N/A"
    trigger_source = gitlab_conf.get("trigger_source") or (
        dag_run.run_type if dag_run else "N/A"
    )

    print("[airflow] Run metadata")
    print(f"[airflow]   commit_sha={commit_sha}")
    print(f"[airflow]   branch={branch}")
    print(f"[airflow]   pipeline_id={pipeline_id}")
    print(f"[airflow]   trigger_source={trigger_source}")
    if dag_run:
        print(f"[airflow]   run_id={dag_run.run_id}")
    print(f"[airflow]   execution_date={context.get('logical_date')}")


def _bash_task(
    task_id: str,
    command: str,
    *,
    retries: int | None = None,
    trigger_rule: str | None = None,
) -> BashOperator:
    kwargs = {
        "task_id": task_id,
        "bash_command": _run(command),
    }
    if retries is not None:
        kwargs["retries"] = retries
    if trigger_rule is not None:
        kwargs["trigger_rule"] = trigger_rule
    return BashOperator(**kwargs)


with DAG(
    dag_id="gas_sensor_ml_pipeline",
    # Keep description explicit; this no-op text change is used for CI trigger checks.
    description="Gas sensor ML pipeline (orchestration).",
    start_date=_to_date(
        str(_airflow_cfg.get("start_date", "2026-03-01")), datetime(2026, 3, 1)
    ),
    schedule=str(_airflow_cfg.get("schedule", "@daily")),
    catchup=bool(_airflow_cfg.get("catchup", False)),
    max_active_runs=int(_airflow_cfg.get("max_active_runs", 1)),
    default_args=DEFAULT_ARGS,
    tags=["observed", "mlops", "pipeline", "gas-sensor"],
) as dag:
    log_trigger_metadata = PythonOperator(
        task_id="log_trigger_metadata",
        python_callable=_log_run_metadata,
    )

    stage1_ingest = _bash_task(
        "stage1_ingest",
        "pipelines.ingest_flow "
        f"--input {str(_paths_cfg.get('raw_input', 'data/raw'))} "
        "--config configs/default.yaml "
        f"--processed-timeseries {str(_paths_cfg.get('processed_timeseries', 'data/processed/processed_timeseries.parquet'))} "
        f"--runs-out {str(_paths_cfg.get('runs', 'data/processed/runs.parquet'))} "
        f"--summary-out {str(_paths_cfg.get('ingestion_summary', 'artifacts/reports/ingestion_summary.json'))}",
    )

    stage2_feature_engineering = _bash_task(
        "stage2_feature_engineering",
        "pipelines.feature_engineering_flow "
        f"--processed-timeseries {str(_paths_cfg.get('processed_timeseries', 'data/processed/processed_timeseries.parquet'))} "
        f"--windowed-out {str(_paths_cfg.get('windowed_timeseries', 'data/processed/windowed_timeseries.parquet'))} "
        f"--train-windows-out {str(_paths_cfg.get('train_windows', 'data/processed/train_windows.parquet'))} "
        f"--test-windows-out {str(_paths_cfg.get('test_windows', 'data/processed/test_windows.parquet'))} "
        f"--features-out {str(_paths_cfg.get('features_timeseries', 'data/processed/features_timeseries.parquet'))} "
        f"--summary-out {str(_paths_cfg.get('feature_engineering_summary', 'artifacts/reports/feature_engineering_summary.json'))}",
    )

    stage3_train_stage1 = _bash_task(
        "stage3_train_stage1",
        "pipelines.train_stage1_mixture_detection_flow "
        f"--features {str(_paths_cfg.get('features_timeseries', 'data/processed/features_timeseries.parquet'))} "
        f"--model-dir {str(_paths_cfg.get('stage1_model_dir', 'artifacts/models/stage1_mixture_detector'))} "
        f"--comparison-out {str(_paths_cfg.get('stage1_comparison', 'artifacts/reports/stage1_model_comparison.json'))} "
        f"--metrics-out {str(_paths_cfg.get('stage1_metrics', 'artifacts/reports/stage1_cv_metrics.json'))} "
        f"--predictions-out {str(_paths_cfg.get('stage1_predictions', 'artifacts/predictions/stage1_training_predictions.parquet'))}",
    )

    stage4_prepare_stage2_subset = _bash_task(
        "stage4_prepare_stage2_subset",
        "pipelines.prepare_stage2_single_gas_subset_flow "
        f"--features {str(_paths_cfg.get('features_timeseries', 'data/processed/features_timeseries.parquet'))} "
        f"--subset-out {str(_paths_cfg.get('features_single_gas', 'data/processed/features_single_gas.parquet'))} "
        f"--summary-out {str(_paths_cfg.get('stage2_subset_summary', 'artifacts/reports/stage2_subset_summary.json'))}",
    )

    stage5_train_stage2 = _bash_task(
        "stage5_train_stage2",
        "pipelines.train_stage2_single_gas_probability_flow "
        f"--single-features {str(_paths_cfg.get('features_single_gas', 'data/processed/features_single_gas.parquet'))} "
        f"--model-dir {str(_paths_cfg.get('stage2_model_dir', 'artifacts/models/stage2_gas_probability_model'))} "
        f"--comparison-out {str(_paths_cfg.get('stage2_comparison', 'artifacts/reports/stage2_model_comparison.json'))} "
        f"--tuning-out {str(_paths_cfg.get('stage2_tuning', 'artifacts/reports/stage2_tuning_results.json'))} "
        f"--metrics-out {str(_paths_cfg.get('stage2_metrics', 'artifacts/reports/stage2_cv_metrics.json'))} "
        f"--predictions-out {str(_paths_cfg.get('stage2_predictions', 'artifacts/predictions/stage2_training_predictions.parquet'))}",
    )

    stage6_evaluate = _bash_task(
        "stage6_evaluate",
        "pipelines.evaluate_flow "
        f"--stage1-predictions {str(_paths_cfg.get('stage1_predictions', 'artifacts/predictions/stage1_training_predictions.parquet'))} "
        f"--stage2-predictions {str(_paths_cfg.get('stage2_predictions', 'artifacts/predictions/stage2_training_predictions.parquet'))} "
        f"--stage1-output {str(_paths_cfg.get('evaluate_stage1_report', 'artifacts/reports/evaluate_stage1_report.json'))} "
        f"--stage2-output {str(_paths_cfg.get('evaluate_stage2_report', 'artifacts/reports/evaluate_stage2_report.json'))}",
    )

    stage7_batch_prediction = _bash_task(
        "stage7_batch_prediction",
        "pipelines.batch_predict_flow "
        f"--features {str(_paths_cfg.get('features_timeseries', 'data/processed/features_timeseries.parquet'))} "
        f"--stage1-model-dir {str(_paths_cfg.get('stage1_model_dir', 'artifacts/models/stage1_mixture_detector'))} "
        f"--stage2-model-dir {str(_paths_cfg.get('stage2_model_dir', 'artifacts/models/stage2_gas_probability_model'))} "
        f"--output {str(_paths_cfg.get('batch_predictions', 'artifacts/predictions/batch_predictions.parquet'))} "
        f"--summary-out {str(_paths_cfg.get('batch_prediction_summary', 'artifacts/reports/batch_prediction_summary.json'))}",
        retries=0,
        trigger_rule="all_done",
    )

    (
        log_trigger_metadata
        >> stage1_ingest
        >> stage2_feature_engineering
        >> stage3_train_stage1
        >> stage4_prepare_stage2_subset
        >> stage5_train_stage2
        >> stage6_evaluate
        >> stage7_batch_prediction
    )
