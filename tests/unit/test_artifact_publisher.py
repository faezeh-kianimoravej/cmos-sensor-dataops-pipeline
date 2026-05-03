from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.observed.utils import artifact_publisher as publisher_module


def test_publish_local_artifact_returns_standardized_storage_metadata(
    monkeypatch,
) -> None:
    base_dir = Path("artifacts/test_tmp") / f"publisher-{uuid.uuid4().hex}"
    report_path = base_dir / "evaluate_stage1_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("OBSERVED_STORAGE_ENV", "staging")
    monkeypatch.setattr(
        publisher_module,
        "upload_file_to_object_storage",
        lambda **kwargs: type(
            "UploadResult",
            (),
            {"to_dict": lambda self: {"status": "uploaded", **kwargs}},
        )(),
    )

    published = publisher_module.publish_local_artifact(
        local_path=report_path,
        domain="reports",
        pipeline="stage1",
        report_family="evaluation-report",
        run_id="manual-20260407T120000Z",
    )

    assert published["domain"] == "reports"
    assert published["storage_target"]["bucket"] == "reports"
    assert published["storage_target"]["environment"] == "staging"
    assert published["storage_upload"]["status"] == "uploaded"


def test_publish_model_bundle_returns_unified_artifact_structure(monkeypatch) -> None:
    model_dir = (
        Path("artifacts/test_tmp")
        / f"bundle-{uuid.uuid4().hex}"
        / "stage1_mixture_detector"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"model")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_name": "dummy"}), encoding="utf-8"
    )

    uploads: list[dict[str, str]] = []

    monkeypatch.setenv("OBSERVED_STORAGE_ENV", "prod")
    monkeypatch.setenv("OBSERVED_STAGE1_MODEL_VERSION", "git-abc1234")

    def _fake_upload(**kwargs):
        uploads.append(kwargs)
        return type(
            "UploadResult",
            (),
            {"to_dict": lambda self: {"status": "uploaded", **kwargs}},
        )()

    monkeypatch.setattr(publisher_module, "upload_file_to_object_storage", _fake_upload)

    published = publisher_module.publish_model_bundle(
        stage="stage1",
        model_name="stage1-mixture-detector",
        model_dir=model_dir,
        source_info={"publisher": "unit-test"},
    )

    assert published["storage_environment"] == "prod"
    assert published["model_version"] == "git-abc1234"
    assert set(published["artifacts"]) == {"model", "metadata", "manifest"}
    assert len(uploads) == 3
    assert published["artifacts"]["model"]["storage_upload"]["status"] == "uploaded"
