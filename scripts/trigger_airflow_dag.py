from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else default


def _build_auth_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
    return f"Basic {token}"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_airflow_api_base(url: str) -> str:
    base = url.strip().rstrip("/")
    if not base:
        return ""
    if "/api/" in base:
        return base
    return f"{base}/api/v1"


def _post_json(
    url: str, payload: dict, auth_header: str, timeout_sec: int
) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if auth_header:
        request.add_header("Authorization", auth_header)

    with urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
        return int(response.status), body


def _trigger_via_rest(
    dag_id: str,
    metadata: dict,
    airflow_api_base: str,
    username: str,
    password: str,
    timeout_sec: int,
) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pipeline_id = metadata.get("pipeline_id") or "na"
    short_sha = (metadata.get("commit_sha") or "na")[:8]
    run_id = f"gitlab_{pipeline_id}_{short_sha}_{ts}"

    payload = {
        "dag_run_id": run_id,
        "conf": {
            "gitlab": metadata,
        },
        "note": "Triggered by GitLab CI event-driven orchestration",
    }

    trigger_url = f"{airflow_api_base.rstrip('/')}/dags/{dag_id}/dagRuns"
    auth_header = _build_auth_header(username, password) if username or password else ""

    status_code, response_body = _post_json(
        url=trigger_url,
        payload=payload,
        auth_header=auth_header,
        timeout_sec=timeout_sec,
    )

    if status_code not in {200, 201}:
        raise RuntimeError(
            f"REST trigger returned unexpected status code: {status_code}"
        )

    return {
        "mode": "rest",
        "dag_id": dag_id,
        "dag_run_id": run_id,
        "status_code": status_code,
        "response": response_body,
        "trigger_url": trigger_url,
    }


def _trigger_via_cli(dag_id: str, metadata: dict) -> dict:
    # CLI fallback is intentionally lightweight; use REST by default for CI portability.
    import subprocess

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pipeline_id = metadata.get("pipeline_id") or "na"
    short_sha = (metadata.get("commit_sha") or "na")[:8]
    run_id = f"gitlab_{pipeline_id}_{short_sha}_{ts}"

    conf_json = json.dumps({"gitlab": metadata})
    cmd = [
        "airflow",
        "dags",
        "trigger",
        dag_id,
        "--run-id",
        run_id,
        "--conf",
        conf_json,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(
            f"CLI trigger failed with code {completed.returncode}. "
            f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )

    return {
        "mode": "cli",
        "dag_id": dag_id,
        "dag_run_id": run_id,
        "stdout": completed.stdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger Airflow DAG from GitLab CI")
    parser.add_argument(
        "--dag-id", default=_env("AIRFLOW_DAG_ID", "gas_sensor_ml_pipeline")
    )
    parser.add_argument("--report", default="reports/airflow_trigger.json")
    parser.add_argument(
        "--mode", choices=["rest", "cli"], default=_env("AIRFLOW_TRIGGER_MODE", "rest")
    )
    parser.add_argument(
        "--required",
        action="store_true",
        default=_is_truthy(_env("AIRFLOW_TRIGGER_REQUIRED", "0")),
        help="Fail if trigger cannot run due to missing endpoint configuration",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "commit_sha": _env("CI_COMMIT_SHA"),
        "commit_short_sha": _env("CI_COMMIT_SHORT_SHA"),
        "branch": _env("CI_COMMIT_REF_NAME"),
        "pipeline_id": _env("CI_PIPELINE_ID"),
        "pipeline_url": _env("CI_PIPELINE_URL"),
        "project_path": _env("CI_PROJECT_PATH"),
        "trigger_source": _env("CI_PIPELINE_SOURCE"),
        "job_url": _env("CI_JOB_URL"),
    }

    result: dict[str, object] = {
        "status": "fail",
        "dag_id": args.dag_id,
        "required": bool(args.required),
        "metadata": metadata,
        "trigger": {},
    }

    try:
        if args.mode == "cli":
            trigger_result = _trigger_via_cli(args.dag_id, metadata)
        else:
            airflow_api_base = _normalize_airflow_api_base(_env("AIRFLOW_API_URL", ""))
            if not airflow_api_base:
                result["status"] = "fail" if args.required else "skipped"
                result["reason"] = "AIRFLOW_API_URL is not configured"
                report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                if args.required:
                    raise SystemExit(1)
                return

            airflow_username = _env("AIRFLOW_USERNAME", "admin")
            airflow_password = _env("AIRFLOW_PASSWORD", "admin")
            timeout_sec = int(_env("AIRFLOW_REQUEST_TIMEOUT_SEC", "30") or "30")
            trigger_result = _trigger_via_rest(
                dag_id=args.dag_id,
                metadata=metadata,
                airflow_api_base=airflow_api_base,
                username=airflow_username,
                password=airflow_password,
                timeout_sec=timeout_sec,
            )

        result["status"] = "pass"
        result["trigger"] = trigger_result
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (HTTPError, URLError, RuntimeError, ValueError, OSError) as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
