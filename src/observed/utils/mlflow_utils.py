from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from src.observed.utils.config import get_repo_root


def _normalize_sqlite_uri(uri: str) -> str:
    if not uri.startswith("sqlite:///"):
        return uri

    raw_path = uri[len("sqlite:///") :]
    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = (get_repo_root() / db_path).resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _normalize_file_uri(uri: str) -> str:
    if not uri:
        return uri
    if uri.startswith("file:///"):
        return uri
    if uri.startswith("file:"):
        rest = uri[len("file:") :]
        # Normalize Windows drive file URI form: file:F:/... -> file:///F:/...
        if len(rest) >= 3 and rest[1] == ":" and rest[2] == "/":
            return f"file:///{rest}"
    return uri


def _repair_sqlite_artifact_uris(tracking_uri: str) -> None:
    if not tracking_uri.startswith("sqlite:///"):
        return

    db_path = Path(tracking_uri[len("sqlite:///") :])
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        exp_rows = cur.execute(
            "SELECT experiment_id, artifact_location FROM experiments"
        ).fetchall()
        for exp_id, artifact_location in exp_rows:
            fixed = _normalize_file_uri(str(artifact_location or ""))
            if fixed != artifact_location:
                cur.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                    (fixed, exp_id),
                )

        run_rows = cur.execute("SELECT run_uuid, artifact_uri FROM runs").fetchall()
        for run_id, artifact_uri in run_rows:
            fixed = _normalize_file_uri(str(artifact_uri or ""))
            if fixed != artifact_uri:
                cur.execute(
                    "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                    (fixed, run_id),
                )

        conn.commit()
    finally:
        conn.close()


def configure_mlflow(mlflow_cfg: dict[str, Any], experiment_name: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    env_tracking_uri = str(os.environ.get("MLFLOW_TRACKING_URI", "")).strip()
    tracking_uri = env_tracking_uri or str(mlflow_cfg.get("tracking_uri", "")).strip()
    if tracking_uri:
        tracking_uri = _normalize_sqlite_uri(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    env_registry_uri = str(os.environ.get("MLFLOW_REGISTRY_URI", "")).strip()
    registry_uri = env_registry_uri or str(mlflow_cfg.get("registry_uri", "")).strip()
    if registry_uri:
        registry_uri = _normalize_sqlite_uri(registry_uri)
        mlflow.set_registry_uri(registry_uri)

    if tracking_uri:
        _repair_sqlite_artifact_uris(tracking_uri)

    # When running the pipeline on the host, MLflow artifact uploads need a
    # host-reachable S3 endpoint rather than the Docker-internal MinIO name.
    s3_endpoint = (
        str(os.environ.get("OBSERVED_STORAGE_S3_ENDPOINT_URL", "")).strip()
        or str(os.environ.get("MINIO_PUBLIC_ENDPOINT", "")).strip()
        or str(os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")).strip()
    )
    if s3_endpoint:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

    artifacts_dir = str(mlflow_cfg.get("artifacts_dir", "")).strip()
    artifact_location = None
    if artifacts_dir and not s3_endpoint:
        artifact_path = Path(artifacts_dir)
        if not artifact_path.is_absolute():
            artifact_path = (get_repo_root() / artifact_path).resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_path.as_uri()

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None and artifact_location:
        client.create_experiment(experiment_name, artifact_location=artifact_location)

    mlflow.set_experiment(experiment_name)
