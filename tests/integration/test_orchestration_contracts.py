from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


def test_dvc_stage_graph_validation() -> None:
    dvc = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))
    stages = dvc["stages"]
    expected = [
        "stage1_ingest",
        "stage2_feature_engineering",
        "stage3_train_stage1",
        "stage4_prepare_stage2_subset",
        "stage5_train_stage2",
        "stage6_evaluate",
        "stage7_batch_prediction",
    ]
    assert list(stages.keys()) == expected
    for stage_name, cfg in stages.items():
        assert "deps" in cfg and cfg["deps"]
        assert "outs" in cfg and cfg["outs"]


def test_airflow_dag_has_expected_task_structure() -> None:
    dag_path = Path("dags/gas_sensor_ml_pipeline_dag.py")
    source = dag_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    text = source
    for task_id in [
        "stage1_ingest",
        "stage2_feature_engineering",
        "stage3_train_stage1",
        "stage4_prepare_stage2_subset",
        "stage5_train_stage2",
        "stage6_evaluate",
        "stage7_batch_prediction",
    ]:
        assert task_id in text

    assert "gas_sensor_ml_pipeline" in text
    assert any(isinstance(node, ast.With) for node in module.body)


def test_makefile_command_smoke_contract() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    for target in [
        "pipeline-run",
        "pipeline-train",
        "pipeline-eval",
        "pipeline-predict",
        "clean-artifacts",
    ]:
        assert f"{target}:" in makefile
    assert "scripts/run_pipeline.py" in makefile
