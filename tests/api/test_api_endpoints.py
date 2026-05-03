from __future__ import annotations

import numpy as np
import pytest

from api import app as app_module

pytestmark = pytest.mark.api


class _Stage1MixtureModel:
    def predict(self, X):
        return np.array([1])


class _Stage1SingleModel:
    def predict(self, X):
        return np.array([0])


class _Stage2Model:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.25, 0.75]])


class _Stage1SingleModelWithProba:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.82, 0.18]])


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_health_endpoint():
    from fastapi.testclient import TestClient

    def _fail_on_load(stage: str):
        raise AssertionError("health endpoint should not load model bundles")

    original_load_stage_bundle = app_module._stage_loader.load_stage_bundle
    app_module._stage_loader.load_stage_bundle = _fail_on_load
    try:
        client = TestClient(app_module.app)
        response = client.get("/health")
    finally:
        app_module._stage_loader.load_stage_bundle = original_load_stage_bundle

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_model_info_validation_error():
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    response = client.get("/model-info", params={"model_family": "unknown"})
    assert response.status_code == 400


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_predict_mixture_route(monkeypatch):
    from fastapi.testclient import TestClient

    def _bundle(stage: str):
        if stage == "stage1":
            return _Stage1MixtureModel(), {
                "feature_columns": ["f1"],
                "model_name": "gradient_boosting",
            }
        return _Stage2Model(), {
            "feature_columns": ["f1"],
            "model_name": "gradient_boosting_tuned",
            "class_labels": {"0": "2-butanone", "1": "Toluene"},
        }

    monkeypatch.setattr(app_module._stage_loader, "load_stage_bundle", _bundle)
    monkeypatch.setattr(
        app_module._stage_loader, "list_available_models", lambda: ["stage1", "stage2"]
    )

    client = TestClient(app_module.app)
    response = client.post("/predict", json={"run_id": "r1", "features": {"f1": 0.9}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "stage1_only"
    assert payload["stage1_result"]["predicted_class"] == "mixture"
    assert payload["stage1_result"]["class_probabilities"] == {
        "single_gas": 0.0,
        "mixture": 1.0,
    }
    assert "stage1_prediction" not in payload
    assert "stage1_label" not in payload
    assert "mixture_classification" not in payload
    assert payload["stage2_result"] is None
    assert payload["model_versions"] == {
        "stage1": "gradient_boosting",
        "stage2": "gradient_boosting_tuned",
    }


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_predict_single_gas_route(monkeypatch):
    from fastapi.testclient import TestClient

    def _bundle(stage: str):
        if stage == "stage1":
            return _Stage1SingleModelWithProba(), {
                "feature_columns": ["f1"],
                "model_name": "gradient_boosting",
            }
        return _Stage2Model(), {
            "feature_columns": ["f1"],
            "model_name": "gradient_boosting_tuned",
            "class_labels": {"0": "2-butanone", "1": "Toluene"},
        }

    monkeypatch.setattr(app_module._stage_loader, "load_stage_bundle", _bundle)
    monkeypatch.setattr(
        app_module._stage_loader, "list_available_models", lambda: ["stage1", "stage2"]
    )

    client = TestClient(app_module.app)
    response = client.post("/predict", json={"run_id": "r2", "features": {"f1": 0.1}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "stage1_stage2"
    assert payload["stage1_result"]["predicted_class"] == "single_gas"
    assert payload["stage1_result"]["class_probabilities"] == {
        "single_gas": 0.82,
        "mixture": 0.18,
    }
    assert "stage1_prediction" not in payload
    assert "stage1_label" not in payload
    assert "mixture_classification" not in payload
    assert payload["stage2_result"]["predicted_class"] in {"Toluene", "2-butanone"}
    assert 0.0 <= payload["stage2_result"]["class_probabilities"]["Toluene"] <= 1.0
    assert payload["model_versions"] == {
        "stage1": "gradient_boosting",
        "stage2": "gradient_boosting_tuned",
    }


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_predict_bad_request_when_no_numeric_features(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        app_module._stage_loader,
        "load_stage_bundle",
        lambda stage: (
            _Stage1SingleModel(),
            {"feature_columns": ["f1"], "model_name": "gradient_boosting"},
        ),
    )
    client = TestClient(app_module.app)
    response = client.post("/predict", json={"run_id": "r3"})
    assert response.status_code == 400


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_predict_response_schema_validation(monkeypatch):
    from fastapi.testclient import TestClient

    def _bundle(stage: str):
        if stage == "stage1":
            return _Stage1SingleModelWithProba(), {
                "feature_columns": ["f1"],
                "model_name": "gradient_boosting",
            }
        return _Stage2Model(), {
            "feature_columns": ["f1"],
            "model_name": "gradient_boosting_tuned",
            "class_labels": {"0": "2-butanone", "1": "Toluene"},
        }

    monkeypatch.setattr(app_module._stage_loader, "load_stage_bundle", _bundle)
    monkeypatch.setattr(
        app_module._stage_loader, "list_available_models", lambda: ["stage1", "stage2"]
    )

    client = TestClient(app_module.app)
    response = client.post("/predict", json={"run_id": "r4", "features": {"f1": 0.1}})
    assert response.status_code == 200

    from api.schemas import PredictResponse

    validated = PredictResponse.model_validate(response.json())
    assert validated.stage1_result.predicted_class == "single_gas"
    assert validated.stage1_result.class_probabilities["single_gas"] == 0.82
    assert validated.model_versions["stage1"] == "gradient_boosting"


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_predict_raw_single_gas_route(monkeypatch):
    from fastapi.testclient import TestClient

    def _bundle(stage: str):
        if stage == "stage1":
            return _Stage1SingleModelWithProba(), {
                "feature_columns": ["trend_slope", "mean"],
                "model_name": "gradient_boosting",
            }
        return _Stage2Model(), {
            "feature_columns": ["trend_slope", "mean"],
            "model_name": "gradient_boosting_tuned",
            "class_labels": {"0": "2-butanone", "1": "Toluene"},
        }

    monkeypatch.setattr(app_module._stage_loader, "load_stage_bundle", _bundle)
    monkeypatch.setattr(
        app_module._stage_loader, "list_available_models", lambda: ["stage1", "stage2"]
    )

    client = TestClient(app_module.app)
    samples = [[float(i), float(i * 0.1)] for i in range(12)]
    response = client.post(
        "/predict-raw",
        json={"run_id": "raw-r1", "event_id": "e1", "samples": samples},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "stage1_stage2"
    assert payload["run_id"] == "raw-r1"
    assert payload["event_id"] == "e1"
    assert payload["stage1_result"]["predicted_class"] == "single_gas"
    assert payload["stage2_result"] is not None


@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_predict_raw_rejects_non_increasing_time(monkeypatch):
    from fastapi.testclient import TestClient

    def _bundle(stage: str):
        if stage == "stage1":
            return _Stage1SingleModelWithProba(), {
                "feature_columns": ["trend_slope"],
                "model_name": "gradient_boosting",
            }
        return _Stage2Model(), {
            "feature_columns": ["trend_slope"],
            "model_name": "gradient_boosting_tuned",
            "class_labels": {"0": "2-butanone", "1": "Toluene"},
        }

    monkeypatch.setattr(app_module._stage_loader, "load_stage_bundle", _bundle)
    client = TestClient(app_module.app)
    samples = [[float(i), float(i)] for i in range(10)]
    samples[5][0] = samples[4][0]

    response = client.post("/predict-raw", json={"samples": samples})
    assert response.status_code == 400


@pytest.mark.e2e_synthetic
@pytest.mark.skipif(
    not app_module.FASTAPI_AVAILABLE or app_module.app is None,
    reason="FastAPI not available",
)
def test_api_e2e_synthetic_smoke() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    assert client.get("/health").status_code == 200
    assert (
        client.get("/model-info", params={"model_family": "__invalid__"}).status_code
        == 400
    )

    stage1_bundle = app_module._stage_loader.load_stage_bundle("stage1")
    stage2_bundle = app_module._stage_loader.load_stage_bundle("stage2")
    if stage1_bundle is None or stage2_bundle is None:
        pytest.skip("stage model artifacts not present")

    _, stage1_meta = stage1_bundle
    _, stage2_meta = stage2_bundle
    feature_payload = {
        **{str(column): 0.0 for column in stage1_meta.get("feature_columns", [])},
        **{str(column): 0.0 for column in stage2_meta.get("feature_columns", [])},
    }
    if not feature_payload:
        pytest.skip("feature columns not available in stage metadata")

    prediction_response = client.post(
        "/predict",
        json={"run_id": "e2e-synthetic", "features": feature_payload},
    )
    assert prediction_response.status_code == 200
