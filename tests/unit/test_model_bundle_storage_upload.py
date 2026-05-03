from __future__ import annotations

import json
from pathlib import Path

from src.models import stage1 as stage1_module
from src.models import stage2 as stage2_module
from src.observed.utils import artifact_publisher as publisher_module


def test_stage1_publish_model_bundle_uses_storage_convention(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "artifacts" / "models" / "stage1_mixture_detector"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"model")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_name": "dummy"}), encoding="utf-8"
    )

    uploads: list[dict[str, str]] = []

    class _UploadResult:
        def __init__(self, payload: dict[str, str]) -> None:
            self.payload = payload

        def to_dict(self) -> dict[str, str]:
            return self.payload

    def _fake_upload(**kwargs):
        uploads.append(kwargs)
        return _UploadResult({"status": "uploaded", **kwargs})

    monkeypatch.setenv("OBSERVED_STORAGE_ENV", "staging")
    monkeypatch.setenv("OBSERVED_STAGE1_MODEL_VERSION", "git-abc1234")
    monkeypatch.setattr(publisher_module, "upload_file_to_object_storage", _fake_upload)

    result = stage1_module._publish_model_bundle(
        stage="stage1",
        model_name="stage1-mixture-detector",
        model_dir=model_dir,
    )

    assert result["storage_environment"] == "staging"
    assert result["model_version"] == "git-abc1234"
    assert len(uploads) == 3
    assert uploads[0]["bucket"] == "models"
    assert "stage1-mixture-detector/git-abc1234/model.joblib" in uploads[0]["key"]
    assert uploads[2]["key"].endswith("/bundle-manifest.json")
    manifest_payload = json.loads(
        Path(result["artifacts"]["manifest"]["local_path"]).read_text(encoding="utf-8")
    )
    assert manifest_payload["stage"] == "stage1"
    assert manifest_payload["model_version"] == "git-abc1234"
    assert result["artifacts"]["model"]["storage_upload"]["status"] == "uploaded"
    assert result["artifacts"]["manifest"]["storage_upload"]["status"] == "uploaded"


def test_stage2_publish_model_bundle_uses_storage_convention(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "artifacts" / "models" / "stage2_gas_probability_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.joblib").write_bytes(b"model")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_name": "dummy"}), encoding="utf-8"
    )

    uploads: list[dict[str, str]] = []

    class _UploadResult:
        def __init__(self, payload: dict[str, str]) -> None:
            self.payload = payload

        def to_dict(self) -> dict[str, str]:
            return self.payload

    def _fake_upload(**kwargs):
        uploads.append(kwargs)
        return _UploadResult({"status": "uploaded", **kwargs})

    monkeypatch.setenv("OBSERVED_STORAGE_ENV", "prod")
    monkeypatch.setenv("OBSERVED_STAGE2_MODEL_VERSION", "git-def5678")
    monkeypatch.setattr(publisher_module, "upload_file_to_object_storage", _fake_upload)

    result = stage2_module._publish_model_bundle(
        stage="stage2",
        model_name="stage2-gas-probability-model",
        model_dir=model_dir,
    )

    assert result["storage_environment"] == "prod"
    assert result["model_version"] == "git-def5678"
    assert len(uploads) == 3
    assert uploads[0]["bucket"] == "models"
    assert "stage2-gas-probability-model/git-def5678/model.joblib" in uploads[0]["key"]
    assert uploads[2]["key"].endswith("/bundle-manifest.json")
    manifest_payload = json.loads(
        Path(result["artifacts"]["manifest"]["local_path"]).read_text(encoding="utf-8")
    )
    assert manifest_payload["stage"] == "stage2"
    assert manifest_payload["model_version"] == "git-def5678"
    assert result["artifacts"]["metadata"]["storage_upload"]["status"] == "uploaded"
    assert result["artifacts"]["manifest"]["storage_upload"]["status"] == "uploaded"
