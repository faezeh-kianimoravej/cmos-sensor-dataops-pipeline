from __future__ import annotations

import argparse
import base64
import json
import os
import time
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


def _get(url: str, auth_header: str, timeout_sec: int) -> tuple[int, str]:
    request = Request(url=url, method="GET")
    if auth_header:
        request.add_header("Authorization", auth_header)

    with urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
        return int(response.status), body


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Airflow API health")
    parser.add_argument("--report", default="reports/airflow_health.json")
    parser.add_argument(
        "--retries", type=int, default=int(_env("AIRFLOW_HEALTH_RETRIES", "10") or "10")
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=float,
        default=float(_env("AIRFLOW_HEALTH_RETRY_DELAY_SEC", "3") or "3"),
    )
    parser.add_argument(
        "--required",
        action="store_true",
        default=_is_truthy(_env("AIRFLOW_HEALTH_REQUIRED", "0")),
        help="Fail if Airflow endpoint is not configured or never becomes healthy",
    )
    args = parser.parse_args()

    airflow_api_base = _normalize_airflow_api_base(_env("AIRFLOW_API_URL", ""))
    health_url = f"{airflow_api_base.rstrip('/')}/health"
    username = _env("AIRFLOW_USERNAME", "admin")
    password = _env("AIRFLOW_PASSWORD", "admin")
    timeout_sec = int(_env("AIRFLOW_REQUEST_TIMEOUT_SEC", "20") or "20")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "status": "fail",
        "airflow_api_url": airflow_api_base,
        "health_url": health_url,
        "retries": int(args.retries),
        "retry_delay_sec": float(args.retry_delay_sec),
        "required": bool(args.required),
    }

    if not airflow_api_base:
        payload.update(
            {
                "status": "fail" if args.required else "skipped",
                "reason": "AIRFLOW_API_URL is not configured",
                "checks": {
                    "webserver": False,
                    "metadatabase": False,
                    "scheduler": False,
                },
                "attempts": [],
            }
        )
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.required:
            raise SystemExit(1)
        return

    auth_header = _build_auth_header(username, password) if username or password else ""
    last_error: str | None = None
    attempts: list[dict[str, object]] = []

    for attempt in range(1, max(1, int(args.retries)) + 1):
        try:
            status_code, response_body = _get(health_url, auth_header, timeout_sec)
            parsed = json.loads(response_body) if response_body else {}

            metadatabase_ok = (
                str(parsed.get("metadatabase", {}).get("status", "")).lower()
                == "healthy"
            )
            scheduler_ok = (
                str(parsed.get("scheduler", {}).get("status", "")).lower() == "healthy"
            )
            webserver_ok = status_code == 200
            is_healthy = webserver_ok and metadatabase_ok and scheduler_ok

            attempts.append(
                {
                    "attempt": attempt,
                    "status_code": status_code,
                    "checks": {
                        "webserver": webserver_ok,
                        "metadatabase": metadatabase_ok,
                        "scheduler": scheduler_ok,
                    },
                }
            )

            payload.update(
                {
                    "attempts": attempts,
                    "status": "pass" if is_healthy else "fail",
                    "status_code": status_code,
                    "response": parsed,
                    "checks": {
                        "webserver": webserver_ok,
                        "metadatabase": metadatabase_ok,
                        "scheduler": scheduler_ok,
                    },
                }
            )

            if is_healthy:
                report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return

            last_error = "Health endpoint reachable but not healthy yet"

        except (HTTPError, URLError, ValueError, OSError) as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt, "error": last_error})

        if attempt < int(args.retries):
            time.sleep(float(args.retry_delay_sec))

    payload["status"] = "fail"
    payload["attempts"] = attempts
    if last_error:
        payload["error"] = last_error
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
