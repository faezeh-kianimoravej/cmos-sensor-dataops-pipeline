"""Reusable YAML configuration loader for pipeline settings.

This module provides a stable way to load project configuration from YAML files,
including support for config inheritance, repository-root path resolution, and
optional global random seed injection.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

ConfigDict = Dict[str, Any]


def get_repo_root() -> Path:
    """Locate the repository root by searching for a configs directory."""
    starts = [Path.cwd(), Path(__file__).resolve().parent]

    for start in starts:
        for parent in [start, *start.parents]:
            if (parent / "configs").is_dir() and (parent / "README.md").exists():
                return parent

    # Fallback for standard layout: <repo>/src/observed/utils/config.py
    return Path(__file__).resolve().parents[3]


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config file path with repository-root semantics for relative inputs."""
    raw = Path(path).expanduser()

    if raw.is_absolute():
        resolved = raw
    else:
        resolved = get_repo_root() / raw

    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved} (requested: {path})"
        )

    if not resolved.is_file():
        raise FileNotFoundError(f"Config path is not a file: {resolved}")

    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must be YAML (.yaml/.yml), got: {resolved}")

    return resolved


def _load_yaml_file(path: Path) -> ConfigDict:
    """Load a YAML file safely and ensure the root object is a mapping."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {path}. {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read config file: {path}. {exc}") from exc

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping in file: {path}")

    return loaded


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge dictionaries where override values take precedence."""
    merged: ConfigDict = dict(base)

    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value

    return merged


def _resolve_extends_path(base_ref: str, current_file: Path) -> Path:
    """Resolve an extends reference relative to config file then repository root."""
    raw = Path(base_ref).expanduser()

    if raw.is_absolute():
        candidate = raw
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Extended config not found: {candidate} (from {current_file})"
        )

    local_candidate = current_file.parent / raw
    if local_candidate.exists():
        return local_candidate

    repo_candidate = get_repo_root() / raw
    if repo_candidate.exists():
        return repo_candidate

    configs_candidate = get_repo_root() / "configs" / raw
    if configs_candidate.exists():
        return configs_candidate

    raise FileNotFoundError(
        f"Extended config not found: {base_ref} (from {current_file})"
    )


def _load_with_extends(path: Path, stack: list[Path]) -> ConfigDict:
    """Load a YAML config recursively, resolving optional extends references."""
    resolved = path.resolve()

    if resolved in stack:
        chain = " -> ".join(str(p) for p in [*stack, resolved])
        raise ValueError(f"Circular config inheritance detected: {chain}")

    stack.append(resolved)
    try:
        cfg = _load_yaml_file(resolved)

        extends_ref = cfg.pop("extends", None)
        if not extends_ref:
            return cfg

        if not isinstance(extends_ref, str):
            raise ValueError(
                f"Invalid extends value in {resolved}: expected string, got {type(extends_ref).__name__}"
            )

        base_path = _resolve_extends_path(extends_ref, resolved)
        base_cfg = _load_with_extends(base_path, stack)
        return _deep_merge(base_cfg, cfg)
    finally:
        stack.pop()


def _resolve_paths_section(config: ConfigDict) -> ConfigDict:
    """Resolve relative entries under top-level paths section from repository root."""
    paths = config.get("paths")
    if not isinstance(paths, dict):
        return config

    repo_root = get_repo_root()
    resolved_paths: ConfigDict = {}

    for key, value in paths.items():
        if isinstance(value, str):
            path_obj = Path(value).expanduser()
            if not path_obj.is_absolute():
                path_obj = repo_root / path_obj
            resolved_paths[key] = str(path_obj)
        else:
            resolved_paths[key] = value

    output = dict(config)
    output["paths"] = resolved_paths
    return output


def _extract_seed(config: ConfigDict) -> int | None:
    """Extract integer seed from config if present."""
    seed_section = config.get("seed")

    if isinstance(seed_section, dict):
        raw_value = seed_section.get("value")
    else:
        raw_value = seed_section

    if raw_value is None:
        return None

    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid seed value in config: {raw_value}") from exc


def apply_global_seed(seed: int) -> None:
    """Apply deterministic global seeding for Python and NumPy (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except Exception:
        return

    np.random.seed(seed)


def _to_namespace(value: Any) -> Any:
    """Convert nested dictionaries to a SimpleNamespace recursively."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(
    path: str | Path,
    *,
    inject_seed: bool = False,
    as_namespace: bool = False,
) -> ConfigDict | SimpleNamespace:
    """Load YAML config with inheritance and normalized project-relative paths.

    Args:
        path: Config file path. Relative paths are resolved from repository root.
        inject_seed: If True, applies seed.value (if present) globally.
        as_namespace: If True, returns a dot-access object instead of dict.

    Returns:
        Loaded configuration as dict or SimpleNamespace.
    """
    resolved = resolve_config_path(path)
    config = _load_with_extends(resolved, stack=[])
    config = _resolve_paths_section(config)

    if inject_seed:
        seed = _extract_seed(config)
        if seed is not None:
            apply_global_seed(seed)

    if as_namespace:
        return _to_namespace(config)

    return config


__all__ = [
    "ConfigDict",
    "apply_global_seed",
    "get_repo_root",
    "load_config",
    "resolve_config_path",
]
