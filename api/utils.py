"""Utility helpers for API metadata and artifact discovery.

Why:
The API needs lightweight, failure-tolerant helpers for reading local metadata
without coupling endpoint logic to filesystem and subprocess details.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit_hash() -> Optional[str]:
    """Return current git commit hash when available.

    Why:
    Exposing commit SHA in API metadata improves traceability across builds.

    Edge cases:
    - Returns ``None`` when git is unavailable, command times out, or the
      directory is not a repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics JSON and return a dictionary.

    Why:
    Endpoints should degrade gracefully when metrics artifacts are missing or
    malformed.

    Edge cases:
    - Missing or invalid JSON returns an empty dict.
    """
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def find_latest_model_artifacts(models_dir: Path) -> Dict[str, Optional[Path]]:
    """Resolve known model artifact paths by family.

    Assumptions:
    - One canonical filename per family is used in local artifacts.

    Edge cases:
    - Missing artifacts remain mapped to ``None``.
    """
    artifacts = {
        "classical_ml": None,
        "deep_learning": None,
        "unsupervised": None,
    }

    model_files = {
        "classical_ml_model.pkl": "classical_ml",
        "deep_learning_model.pkl": "deep_learning",
        "unsupervised_model.pkl": "unsupervised",
    }

    for filename, family in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            artifacts[family] = model_path

    return artifacts


def find_latest_metrics(reports_dir: Path) -> Dict[str, Optional[Path]]:
    """Resolve metrics artifact paths by model family.

    Why:
    Model-info responses should be assembled from whichever family artifacts are
    currently present.
    """
    metrics = {
        "classical_ml": None,
        "deep_learning": None,
        "unsupervised": None,
    }

    metric_files = {
        "training_metrics_classical_ml.json": "classical_ml",
        "training_metrics_deep_learning.json": "deep_learning",
        "training_metrics_unsupervised.json": "unsupervised",
    }

    for filename, family in metric_files.items():
        metrics_path = reports_dir / filename
        if metrics_path.exists():
            metrics[family] = metrics_path

    return metrics


def get_default_model_family(models_dir: Path) -> Optional[str]:
    """Pick a default model family from available local artifacts.

    Why:
    API endpoints should have deterministic behavior when callers do not pass a
    model family.

    Assumptions:
    - Preference order is classical_ml, then deep_learning, then unsupervised.
    """
    artifacts = find_latest_model_artifacts(models_dir)
    if artifacts["classical_ml"]:
        return "classical_ml"
    if artifacts["deep_learning"]:
        return "deep_learning"
    if artifacts["unsupervised"]:
        return "unsupervised"
    return None
