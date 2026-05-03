from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.observed.utils import artifact_consumer as consumer_module


def test_resolve_model_bundle_access_prefers_manifest_targets(monkeypatch) -> None:
    base_dir = Path("artifacts/test_tmp") / f"consumer-{uuid.uuid4().hex}"
    local_model = base_dir / "stage1" / "model.joblib"
    local_metadata = base_dir / "stage1" / "metadata.json"

    monkeypatch.setenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", "true")
    monkeypatch.setenv("OBSERVED_STAGE1_MODEL_VERSION", "git-abc1234")
    monkeypatch.setenv("OBSERVED_STORAGE_ENV", "dev")
    monkeypatch.setenv("OBSERVED_STORAGE_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setattr(
        consumer_module,
        "read_text_from_storage_target",
        lambda target: (
            json.dumps(
                {
                    "stage": "stage1",
                    "model_name": "stage1-mixture-detector",
                    "model_version": "git-abc1234",
                    "storage_environment": "dev",
                    "published_at": "2026-04-07T10:00:00Z",
                    "artifacts": {
                        "model": {
                            "bucket": "models",
                            "key": "from-manifest/model.joblib",
                            "uri": "s3://models/from-manifest/model.joblib",
                        },
                        "metadata": {
                            "bucket": "models",
                            "key": "from-manifest/metadata.json",
                            "uri": "s3://models/from-manifest/metadata.json",
                        },
                    },
                }
            ),
            type("ReadResult", (), {"status": "read"})(),
        ),
    )

    plan = consumer_module.resolve_model_bundle_access(
        stage="stage1",
        model_name="stage1-mixture-detector",
        local_model_path=local_model,
        local_metadata_path=local_metadata,
    )

    assert plan.source == "object_storage"
    assert plan.manifest_strategy == "manifest"
    assert plan.model_target is not None
    assert plan.model_target["key"] == "from-manifest/model.joblib"


def test_materialize_model_bundle_access_plan_downloads_to_local_paths(
    monkeypatch,
) -> None:
    base_dir = Path("artifacts/test_tmp") / f"consumer-{uuid.uuid4().hex}"
    local_model = base_dir / "stage1" / "model.joblib"
    local_metadata = base_dir / "stage1" / "metadata.json"

    downloaded: list[dict[str, str]] = []

    def _fake_download(storage_target, *, local_path):
        downloaded.append({"key": storage_target["key"], "local_path": str(local_path)})
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
        return type("DownloadResult", (), {"status": "downloaded"})()

    monkeypatch.setattr(consumer_module, "download_from_storage_target", _fake_download)

    plan = consumer_module.ModelBundleAccessPlan(
        source="object_storage",
        reason="manifest bundle is available",
        stage="stage1",
        model_name="stage1-mixture-detector",
        model_version="git-abc1234",
        storage_environment="dev",
        local_model_path=local_model,
        local_metadata_path=local_metadata,
        model_target={
            "bucket": "models",
            "key": "bundle/model.joblib",
            "uri": "s3://models/bundle/model.joblib",
        },
        metadata_target={
            "bucket": "models",
            "key": "bundle/metadata.json",
            "uri": "s3://models/bundle/metadata.json",
        },
        manifest_target={
            "bucket": "models",
            "key": "bundle/bundle-manifest.json",
            "uri": "s3://models/bundle/bundle-manifest.json",
        },
        manifest_strategy="manifest",
    )

    materialized = consumer_module.materialize_model_bundle_access_plan(plan)

    assert materialized == (local_model, local_metadata)
    assert len(downloaded) == 2
    assert local_model.exists()
    assert local_metadata.exists()
