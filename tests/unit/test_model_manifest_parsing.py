from __future__ import annotations

import json

import pytest

from src.observed.utils.model_manifest import parse_model_bundle_manifest


def test_parse_model_bundle_manifest_returns_structured_manifest() -> None:
    manifest = parse_model_bundle_manifest(
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
                        "key": "dev/stage1/stage1-mixture-detector/git-abc1234/model.joblib",
                        "uri": "s3://models/dev/stage1/stage1-mixture-detector/git-abc1234/model.joblib",
                    },
                    "metadata": {
                        "bucket": "models",
                        "key": "dev/stage1/stage1-mixture-detector/git-abc1234/metadata.json",
                        "uri": "s3://models/dev/stage1/stage1-mixture-detector/git-abc1234/metadata.json",
                    },
                },
            }
        )
    )

    assert manifest.stage == "stage1"
    assert manifest.model_version == "git-abc1234"
    assert manifest.model_artifact["bucket"] == "models"


def test_parse_model_bundle_manifest_rejects_missing_artifact_fields() -> None:
    with pytest.raises(ValueError, match="artifacts.model.uri"):
        parse_model_bundle_manifest(
            {
                "stage": "stage1",
                "model_name": "stage1-mixture-detector",
                "model_version": "git-abc1234",
                "storage_environment": "dev",
                "published_at": "2026-04-07T10:00:00Z",
                "artifacts": {
                    "model": {
                        "bucket": "models",
                        "key": "dev/stage1/stage1-mixture-detector/git-abc1234/model.joblib",
                    },
                    "metadata": {
                        "bucket": "models",
                        "key": "dev/stage1/stage1-mixture-detector/git-abc1234/metadata.json",
                        "uri": "s3://models/dev/stage1/stage1-mixture-detector/git-abc1234/metadata.json",
                    },
                },
            }
        )
