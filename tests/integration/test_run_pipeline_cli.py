from __future__ import annotations

import sys

import pytest

from scripts import run_pipeline

pytestmark = pytest.mark.integration


def test_build_parser_supports_expected_commands() -> None:
    parser = run_pipeline.build_parser()
    args = parser.parse_args(["evaluate"])
    assert args.command == "evaluate"


def test_main_dispatches_run(monkeypatch) -> None:
    calls = {}

    def _fake_api():
        return {
            "run_full_pipeline": lambda **kwargs: {"ok": True, "kwargs": kwargs},
            "run_stage1_pipeline": lambda **kwargs: {"ok": True},
            "run_stage2_pipeline": lambda **kwargs: {"ok": True},
            "run_evaluation_pipeline": lambda **kwargs: {"ok": True},
            "run_batch_prediction_pipeline": lambda **kwargs: {"ok": True},
        }

    def _capture(payload):
        calls["payload"] = payload

    monkeypatch.setattr(run_pipeline, "_pipeline_api", _fake_api)
    monkeypatch.setattr(run_pipeline, "_print_json", _capture)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "run", "--input", "data/raw"])

    run_pipeline.main()
    assert calls["payload"]["ok"] is True


def test_main_dispatches_clean(monkeypatch) -> None:
    monkeypatch.setattr(
        run_pipeline, "_clean_generated_artifacts", lambda: {"status": "success"}
    )
    captured = {}
    monkeypatch.setattr(
        run_pipeline,
        "_print_json",
        lambda payload: captured.setdefault("payload", payload),
    )
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "clean"])

    run_pipeline.main()
    assert captured["payload"]["status"] == "success"
