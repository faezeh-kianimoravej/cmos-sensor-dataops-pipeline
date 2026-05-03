from __future__ import annotations

import json
import uuid
from pathlib import Path

import joblib

from api.models_loader import ModelLoader
from src.observed.utils.artifact_consumer import ModelBundleAccessPlan


class _DummyModel:
    def predict(self, X):
        return [0]


def test_model_loader_uses_local_bundle_when_present(tmp_path: Path) -> None:
    models_dir = tmp_path / "artifacts" / "models"
    stage_dir = models_dir / "stage1_mixture_detector"
    stage_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(_DummyModel(), stage_dir / "model.joblib")
    (stage_dir / "metadata.json").write_text(
        json.dumps({"feature_columns": ["f1"], "model_name": "dummy"}), encoding="utf-8"
    )

    loader = ModelLoader(models_dir=models_dir)
    bundle = loader.load_stage_bundle("stage1")

    assert bundle is not None
    _, metadata = bundle
    assert metadata["model_name"] == "dummy"


def test_model_loader_can_materialize_bundle_from_resolved_access_plan(
    monkeypatch,
) -> None:
    models_dir = Path("artifacts/test_tmp") / f"loader-{uuid.uuid4().hex}" / "models"
    loader = ModelLoader(models_dir=models_dir)

    monkeypatch.setattr(
        "api.models_loader.resolve_model_bundle_access",
        lambda **kwargs: ModelBundleAccessPlan(
            source="object_storage",
            reason="manifest bundle is available",
            stage="stage1",
            model_name="stage1-mixture-detector",
            model_version="git-abc1234",
            storage_environment="dev",
            local_model_path=Path(kwargs["local_model_path"]),
            local_metadata_path=Path(kwargs["local_metadata_path"]),
            model_target={
                "bucket": "models",
                "key": "x/model.joblib",
                "uri": "s3://models/x/model.joblib",
            },
            metadata_target={
                "bucket": "models",
                "key": "x/metadata.json",
                "uri": "s3://models/x/metadata.json",
            },
            manifest_target={
                "bucket": "models",
                "key": "x/bundle-manifest.json",
                "uri": "s3://models/x/bundle-manifest.json",
            },
            manifest_strategy="manifest",
        ),
    )

    def _fake_materialize(plan: ModelBundleAccessPlan):
        plan.local_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(_DummyModel(), plan.local_model_path)
        plan.local_metadata_path.write_text(
            json.dumps({"feature_columns": ["f1"], "model_name": "downloaded"}),
            encoding="utf-8",
        )
        return plan.local_model_path, plan.local_metadata_path

    monkeypatch.setattr(
        "api.models_loader.materialize_model_bundle_access_plan", _fake_materialize
    )

    bundle = loader.load_stage_bundle("stage1")

    assert bundle is not None
    _, metadata = bundle
    assert metadata["model_name"] == "downloaded"
    assert (models_dir / "stage1_mixture_detector" / "model.joblib").exists()


def test_model_loader_returns_none_when_access_plan_is_unavailable(monkeypatch) -> None:
    models_dir = Path("artifacts/test_tmp") / f"loader-{uuid.uuid4().hex}" / "models"
    loader = ModelLoader(models_dir=models_dir)

    monkeypatch.setattr(
        "api.models_loader.resolve_model_bundle_access",
        lambda **kwargs: ModelBundleAccessPlan(
            source="unavailable",
            reason="legacy fallback is disabled",
            stage="stage1",
            model_name="stage1-mixture-detector",
            model_version="git-abc1234",
            storage_environment="dev",
            local_model_path=Path(kwargs["local_model_path"]),
            local_metadata_path=Path(kwargs["local_metadata_path"]),
        ),
    )

    bundle = loader.load_stage_bundle("stage1")

    assert bundle is None


def test_model_loader_returns_none_when_object_storage_fallback_not_configured(
    tmp_path: Path, monkeypatch
) -> None:
    loader = ModelLoader(models_dir=tmp_path / "artifacts" / "models")

    monkeypatch.delenv("OBSERVED_ENABLE_OBJECT_STORAGE_MODEL_FALLBACK", raising=False)
    monkeypatch.delenv("OBSERVED_STAGE1_MODEL_VERSION", raising=False)

    assert loader.load_stage_bundle("stage1") is None


def test_model_loader_returns_none_when_materialization_raises(monkeypatch) -> None:
    models_dir = Path("artifacts/test_tmp") / f"loader-{uuid.uuid4().hex}" / "models"
    loader = ModelLoader(models_dir=models_dir)

    monkeypatch.setattr(
        "api.models_loader.resolve_model_bundle_access",
        lambda **kwargs: ModelBundleAccessPlan(
            source="object_storage",
            reason="manifest bundle is available",
            stage="stage1",
            model_name="stage1-mixture-detector",
            model_version="git-abc1234",
            storage_environment="dev",
            local_model_path=Path(kwargs["local_model_path"]),
            local_metadata_path=Path(kwargs["local_metadata_path"]),
            model_target={
                "bucket": "models",
                "key": "x/model.joblib",
                "uri": "s3://models/x/model.joblib",
            },
            metadata_target={
                "bucket": "models",
                "key": "x/metadata.json",
                "uri": "s3://models/x/metadata.json",
            },
            manifest_target={
                "bucket": "models",
                "key": "x/bundle-manifest.json",
                "uri": "s3://models/x/bundle-manifest.json",
            },
            manifest_strategy="manifest",
        ),
    )

    def _raise_materialize(_plan: ModelBundleAccessPlan):
        raise RuntimeError("download failed")

    monkeypatch.setattr(
        "api.models_loader.materialize_model_bundle_access_plan", _raise_materialize
    )

    assert loader.load_stage_bundle("stage1") is None
