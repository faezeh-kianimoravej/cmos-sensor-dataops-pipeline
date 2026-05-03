from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_BUCKETS = [
    "mlflow-artifacts",
    "models",
    "datasets",
    "reports",
    "batch-predictions",
]


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout if capture else ""


def _compose_cmd(compose_file: Path) -> list[str]:
    compose_bin = os.environ.get("DOCKER_COMPOSE_BIN", "docker-compose")
    return [*shlex.split(compose_bin), "-f", str(compose_file)]


def _wait_for_mlflow(tracking_uri: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{tracking_uri.rstrip('/')}/health"

    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except URLError:
            time.sleep(2)

    raise TimeoutError(
        f"MLflow did not become reachable at {health_url} within {timeout_seconds} seconds"
    )


def _verify_buckets(compose_file: Path) -> None:
    bucket_output = _run(
        _compose_cmd(compose_file)
        + [
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            "minio-init",
            "-c",
            "mc alias set observed ${MINIO_ENDPOINT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} >/dev/null && mc ls observed",
        ]
    )
    for bucket in EXPECTED_BUCKETS:
        if bucket not in bucket_output:
            raise RuntimeError(
                f"Expected bucket '{bucket}' was not found in MinIO bucket listing:\n{bucket_output}"
            )


def _verify_mlflow_artifact(compose_file: Path, tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "infra-smoke-mlflow"
    mlflow.set_experiment(experiment_name)

    artifact_filename = "smoke-artifact.json"
    payload = {
        "kind": "infra-smoke",
        "timestamp": int(time.time()),
        "token": uuid.uuid4().hex,
    }

    with tempfile.TemporaryDirectory(prefix="observed-mlflow-smoke-") as tmp_dir:
        artifact_path = Path(tmp_dir) / artifact_filename
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with mlflow.start_run(run_name=f"infra-smoke-{payload['token']}") as run:
            mlflow.log_text("ok", "status.txt")
            mlflow.log_artifact(str(artifact_path))
            run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    run_info = client.get_run(run_id).info
    artifact_uri = run_info.artifact_uri
    parsed = urlparse(artifact_uri)
    if parsed.scheme != "s3" or parsed.netloc != "mlflow-artifacts":
        raise RuntimeError(
            f"MLflow artifact URI is not using MinIO-backed S3 storage: {artifact_uri}"
        )

    artifact_key = parsed.path.strip("/")
    object_path = f"observed/mlflow-artifacts/{artifact_key}/{artifact_filename}"
    _run(
        _compose_cmd(compose_file)
        + [
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            "minio-init",
            "-c",
            f"mc alias set observed ${{MINIO_ENDPOINT}} ${{MINIO_ROOT_USER}} ${{MINIO_ROOT_PASSWORD}} >/dev/null && mc stat {object_path}",
        ]
    )


def _verify_dvc_remote() -> None:
    required_env = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "DVC_S3_ENDPOINT_URL",
    ]
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables for DVC smoke test: {', '.join(missing)}"
        )

    dvc_env = {
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
    }

    smoke_prefix = f"s3://datasets/dvc/smoke-tests/{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory(prefix="observed-dvc-smoke-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_path = tmp_path / "sample.txt"
        sample_content = f"infra-smoke-{uuid.uuid4().hex}"

        _run(["dvc", "init", "--no-scm"], cwd=tmp_path, env=dvc_env)
        _run(
            ["dvc", "remote", "add", "-d", "minio", smoke_prefix],
            cwd=tmp_path,
            env=dvc_env,
        )
        _run(
            [
                "dvc",
                "remote",
                "modify",
                "minio",
                "endpointurl",
                os.environ["DVC_S3_ENDPOINT_URL"],
            ],
            cwd=tmp_path,
            env=dvc_env,
        )
        _run(
            ["dvc", "remote", "modify", "minio", "use_ssl", "false"],
            cwd=tmp_path,
            env=dvc_env,
        )
        _run(
            [
                "dvc",
                "remote",
                "modify",
                "minio",
                "access_key_id",
                os.environ["AWS_ACCESS_KEY_ID"],
            ],
            cwd=tmp_path,
            env=dvc_env,
        )
        _run(
            [
                "dvc",
                "remote",
                "modify",
                "minio",
                "secret_access_key",
                os.environ["AWS_SECRET_ACCESS_KEY"],
            ],
            cwd=tmp_path,
            env=dvc_env,
        )
        _run(
            [
                "dvc",
                "remote",
                "modify",
                "minio",
                "region",
                os.environ["AWS_DEFAULT_REGION"],
            ],
            cwd=tmp_path,
            env=dvc_env,
        )
        _run(
            ["dvc", "remote", "modify", "minio", "listobjects", "true"],
            cwd=tmp_path,
            env=dvc_env,
        )

        sample_path.write_text(sample_content, encoding="utf-8")
        _run(["dvc", "add", "sample.txt"], cwd=tmp_path, env=dvc_env)
        _run(["dvc", "push"], cwd=tmp_path, env=dvc_env)

        cache_dir = tmp_path / ".dvc" / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if sample_path.exists():
            sample_path.unlink()

        _run(["dvc", "pull", "sample.txt.dvc"], cwd=tmp_path, env=dvc_env)
        restored = sample_path.read_text(encoding="utf-8")
        if restored != sample_content:
            raise RuntimeError("DVC pull restored unexpected file content")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test MinIO + MLflow + DVC integration"
    )
    parser.add_argument(
        "--compose-file", type=Path, default=REPO_ROOT / "docker" / "docker-compose.yml"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    args = parser.parse_args()

    _load_dotenv(REPO_ROOT / ".env")

    print("[smoke] Starting MinIO/MLflow services", flush=True)
    _run(
        _compose_cmd(args.compose_file) + ["up", "-d", "minio", "minio-init", "mlflow"],
        capture=True,
    )

    print("[smoke] Verifying MinIO buckets", flush=True)
    _verify_buckets(args.compose_file)

    print("[smoke] Waiting for MLflow", flush=True)
    _wait_for_mlflow(args.tracking_uri)

    print("[smoke] Verifying MLflow artifact storage on MinIO", flush=True)
    _verify_mlflow_artifact(args.compose_file, args.tracking_uri)

    print("[smoke] Verifying DVC push/pull against MinIO", flush=True)
    _verify_dvc_remote()

    print("[smoke] All storage integration checks passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[smoke] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
