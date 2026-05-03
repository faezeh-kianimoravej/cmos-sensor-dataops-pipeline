"""API schema definitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for hierarchical stage-1/stage-2 inference."""

    run_id: Optional[str] = Field(None, description="Associated run ID")
    event_id: Optional[str] = Field(None, description="Associated event ID")
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature dictionary used for stage-1 and stage-2 inference",
    )

    class Config:
        extra = "allow"


class PredictRawRequest(BaseModel):
    """Request schema for raw time-series hierarchical inference."""

    run_id: Optional[str] = Field(None, description="Associated run ID")
    event_id: Optional[str] = Field(None, description="Associated event ID")
    samples: List[Tuple[float, float]] = Field(
        ...,
        min_length=10,
        description=(
            "Raw samples as [second, sensor_value] pairs. "
            "First number = time in seconds, second number = sensor reading."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "raw-r1",
                "event_id": "e1",
                "samples": [
                    [10.0, 20.0],
                    [15.0, 25.0],
                    [20.0, 22.0],
                    [25.0, 30.0],
                    [30.0, 28.0],
                    [35.0, 31.0],
                    [40.0, 33.0],
                    [45.0, 34.0],
                    [50.0, 36.0],
                    [55.0, 35.0],
                ],
            }
        }


class Stage1ProbabilityResult(BaseModel):
    predicted_class: str = Field(..., description="Predicted stage-1 class label")
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Class probabilities for single_gas and mixture",
    )


class Stage2ProbabilityResult(BaseModel):
    predicted_class: str = Field(..., description="Predicted gas class label")
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Class probabilities for Toluene and 2-butanone",
    )


class PredictResponse(BaseModel):
    """Hierarchical response schema for two-stage inference."""

    run_id: Optional[str] = Field(None, description="Associated run ID")
    event_id: Optional[str] = Field(None, description="Associated event ID")
    route: str = Field(..., description="Inference route: stage1_only or stage1_stage2")
    stage1_result: Stage1ProbabilityResult = Field(
        ..., description="Stage-1 prediction result with class probabilities"
    )
    stage2_result: Optional[Stage2ProbabilityResult] = Field(
        None,
        description="Stage-2 probability result when stage-1 predicts single gas",
    )
    model_versions: Dict[str, str] = Field(
        ...,
        description="Resolved stage model identifiers",
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    service_name: str = Field(..., description="Service identifier")
    version: str = Field(..., description="API version")
    model_family: Optional[str] = Field(None, description="Reserved for compatibility")
    models_available: List[str] = Field(
        ..., description="Available stage model bundles"
    )


class ModelMetrics(BaseModel):
    """Metrics summary from training."""

    task_type: Optional[str] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Response schema for model info."""

    model_family: str = Field(..., description="Model scope (stage1 or stage2)")
    model_type: str = Field(..., description="Specific model type")
    training_data_version: Optional[str] = Field(
        None, description="Training data version/path"
    )
    git_commit: Optional[str] = Field(
        None, description="Git commit hash at model training"
    )
    metrics_summary: Optional[ModelMetrics] = Field(
        None, description="Key training metrics"
    )
    registry_stage: str = Field("local", description="Model registry stage")
    available_models: List[str] = Field(
        ..., description="Other available model families"
    )
