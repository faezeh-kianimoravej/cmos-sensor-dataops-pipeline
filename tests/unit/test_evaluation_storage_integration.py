from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.evaluate_flow import run_evaluation
from pipelines.evaluate_stage1_flow import run_evaluate_stage1
from pipelines.evaluate_stage2_flow import run_evaluate_stage2

pytestmark = pytest.mark.unit


def test_run_evaluate_stage1_returns_storage_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.evaluate_stage1_flow.log_evaluation_to_mlflow", lambda **_: None
    )
    monkeypatch.setattr(
        "pipelines.evaluate_stage1_flow.publish_local_artifact",
        lambda **kwargs: {
            "storage_target": {
                "environment": kwargs["env"],
                "pipeline": kwargs["pipeline"],
                "report_family": kwargs["report_family"],
                "run_id": kwargs["run_id"],
                "bucket": "reports",
                "prefix": f"{kwargs['env']}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/",
                "key": f"{kwargs['env']}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/evaluate_stage1_report.json",
                "uri": f"s3://reports/{kwargs['env']}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/evaluate_stage1_report.json",
            },
            "storage_upload": {
                "status": "uploaded",
                "local_path": str(kwargs["local_path"]),
            },
        },
    )

    predictions = pd.DataFrame({"target_mixture": [0, 1], "stage1_prediction": [0, 1]})
    predictions_path = tmp_path / "stage1_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    output_path = tmp_path / "artifacts" / "reports" / "evaluate_stage1_report.json"

    result = run_evaluate_stage1(
        stage1_predictions_path=predictions_path,
        output_report_path=output_path,
        evaluation_run_id="manual-20260407T120000Z",
        storage_env="staging",
    )

    assert output_path.exists()
    assert result["output_storage_target"]["bucket"] == "reports"
    assert result["output_storage_target"]["environment"] == "staging"
    assert result["output_storage_target"]["uri"].endswith(
        "/evaluate_stage1_report.json"
    )
    assert result["output_storage_upload"]["status"] == "uploaded"


def test_run_evaluate_stage2_returns_storage_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.evaluate_stage2_flow.log_evaluation_to_mlflow", lambda **_: None
    )
    monkeypatch.setattr(
        "pipelines.evaluate_stage2_flow.publish_local_artifact",
        lambda **kwargs: {
            "storage_target": {
                "environment": kwargs["env"],
                "pipeline": kwargs["pipeline"],
                "report_family": kwargs["report_family"],
                "run_id": kwargs["run_id"],
                "bucket": "reports",
                "prefix": f"{kwargs['env']}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/",
                "key": f"{kwargs['env']}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/evaluate_stage2_report.json",
                "uri": f"s3://reports/{kwargs['env']}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/evaluate_stage2_report.json",
            },
            "storage_upload": {
                "status": "uploaded",
                "local_path": str(kwargs["local_path"]),
            },
        },
    )

    predictions = pd.DataFrame({"target_gas_type": [0, 1], "stage2_prediction": [0, 1]})
    predictions_path = tmp_path / "stage2_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    output_path = tmp_path / "artifacts" / "reports" / "evaluate_stage2_report.json"

    result = run_evaluate_stage2(
        stage2_predictions_path=predictions_path,
        output_report_path=output_path,
        evaluation_run_id="manual-20260407T120000Z",
        storage_env="prod",
    )

    assert output_path.exists()
    assert result["output_storage_target"]["bucket"] == "reports"
    assert result["output_storage_target"]["environment"] == "prod"
    assert "stage2/evaluation-report" in result["output_storage_target"]["prefix"]
    assert result["output_storage_upload"]["status"] == "uploaded"


def test_run_evaluation_reuses_single_run_id_for_both_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.evaluate_stage1_flow.log_evaluation_to_mlflow", lambda **_: None
    )
    monkeypatch.setattr(
        "pipelines.evaluate_stage2_flow.log_evaluation_to_mlflow", lambda **_: None
    )
    monkeypatch.setattr(
        "pipelines.evaluate_stage1_flow.publish_local_artifact",
        lambda **kwargs: {
            "storage_target": {
                "environment": kwargs["env"] or "dev",
                "pipeline": kwargs["pipeline"],
                "report_family": kwargs["report_family"],
                "run_id": kwargs["run_id"],
                "bucket": "reports",
                "prefix": f"{kwargs['env'] or 'dev'}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/",
                "key": f"{kwargs['env'] or 'dev'}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/{Path(kwargs['local_path']).name}",
                "uri": f"s3://reports/{kwargs['env'] or 'dev'}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/{Path(kwargs['local_path']).name}",
            },
            "storage_upload": {
                "status": "uploaded",
                "local_path": str(kwargs["local_path"]),
            },
        },
    )
    monkeypatch.setattr(
        "pipelines.evaluate_stage2_flow.publish_local_artifact",
        lambda **kwargs: {
            "storage_target": {
                "environment": kwargs["env"] or "dev",
                "pipeline": kwargs["pipeline"],
                "report_family": kwargs["report_family"],
                "run_id": kwargs["run_id"],
                "bucket": "reports",
                "prefix": f"{kwargs['env'] or 'dev'}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/",
                "key": f"{kwargs['env'] or 'dev'}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/{Path(kwargs['local_path']).name}",
                "uri": f"s3://reports/{kwargs['env'] or 'dev'}/{kwargs['pipeline']}/{kwargs['report_family']}/{kwargs['run_id']}/{Path(kwargs['local_path']).name}",
            },
            "storage_upload": {
                "status": "uploaded",
                "local_path": str(kwargs["local_path"]),
            },
        },
    )
    monkeypatch.setattr(
        "pipelines.evaluate_flow._evaluation_run_id", lambda: "manual-20260407T130000Z"
    )
    monkeypatch.setenv("OBSERVED_STORAGE_ENV", "dev")

    stage1_predictions = pd.DataFrame(
        {"target_mixture": [0, 1], "stage1_prediction": [0, 1]}
    )
    stage2_predictions = pd.DataFrame(
        {"target_gas_type": [0, 1], "stage2_prediction": [0, 1]}
    )

    stage1_predictions_path = tmp_path / "stage1_predictions.parquet"
    stage2_predictions_path = tmp_path / "stage2_predictions.parquet"
    stage1_predictions.to_parquet(stage1_predictions_path, index=False)
    stage2_predictions.to_parquet(stage2_predictions_path, index=False)

    stage1_output = tmp_path / "artifacts" / "reports" / "evaluate_stage1_report.json"
    stage2_output = tmp_path / "artifacts" / "reports" / "evaluate_stage2_report.json"

    result = run_evaluation(
        stage1_predictions_path=stage1_predictions_path,
        stage2_predictions_path=stage2_predictions_path,
        stage1_output_path=stage1_output,
        stage2_output_path=stage2_output,
    )

    assert result["evaluation_run_id"] == "manual-20260407T130000Z"
    assert (
        result["stage1"]["output_storage_target"]["run_id"] == "manual-20260407T130000Z"
    )
    assert (
        result["stage2"]["output_storage_target"]["run_id"] == "manual-20260407T130000Z"
    )
    assert result["stage1"]["output_storage_upload"]["status"] == "uploaded"
    assert result["stage2"]["output_storage_upload"]["status"] == "uploaded"
