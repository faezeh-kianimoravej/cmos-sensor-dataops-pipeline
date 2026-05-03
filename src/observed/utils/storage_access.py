"""Small policy helpers for resolving local-vs-object-storage access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.observed.utils.object_storage import object_storage_access_configured


@dataclass(frozen=True)
class StorageSourceDecision:
    source: str
    reason: str
    local_available: bool
    fallback_enabled: bool
    storage_configured: bool


@dataclass(frozen=True)
class ManifestFallbackDecision:
    source: str
    reason: str
    manifest_state: str
    legacy_fallback_enabled: bool


def env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def resolve_local_vs_object_storage(
    *,
    local_paths: list[Path],
    fallback_env_var: str,
) -> StorageSourceDecision:
    """Resolve whether a consumer should use local files or object storage.

    Policy:
    - If all required local files exist, always use local.
    - If local files are missing and fallback is disabled, return a clear local-missing decision.
    - If local files are missing and fallback is enabled, use object storage only when config is complete.
    - If local files are missing, fallback is enabled, but config is incomplete, return a clear disabled decision.
    """

    local_available = all(path.exists() for path in local_paths)
    fallback_enabled = env_flag_enabled(fallback_env_var)
    storage_configured = object_storage_access_configured()

    if local_available:
        return StorageSourceDecision(
            source="local",
            reason="local files are available",
            local_available=True,
            fallback_enabled=fallback_enabled,
            storage_configured=storage_configured,
        )

    if not fallback_enabled:
        return StorageSourceDecision(
            source="unavailable",
            reason=f"local files are missing and {fallback_env_var} is disabled",
            local_available=False,
            fallback_enabled=False,
            storage_configured=storage_configured,
        )

    if not storage_configured:
        return StorageSourceDecision(
            source="unavailable",
            reason="local files are missing and object storage access is not fully configured",
            local_available=False,
            fallback_enabled=True,
            storage_configured=False,
        )

    return StorageSourceDecision(
        source="object_storage",
        reason="local files are missing and object storage fallback is enabled",
        local_available=False,
        fallback_enabled=True,
        storage_configured=True,
    )


def resolve_manifest_vs_legacy_fallback(
    *,
    manifest_state: str,
    legacy_fallback_env_var: str = "OBSERVED_ENABLE_LEGACY_MODEL_PATH_FALLBACK",
) -> ManifestFallbackDecision:
    """Resolve whether object-storage loading should use manifest or legacy paths.

    Policy:
    - If the manifest is valid, always use it.
    - If the manifest is missing or invalid, only fall back to legacy path guessing
      when the legacy fallback flag is enabled.
    - Legacy fallback is enabled by default for backward compatibility.
    """

    normalized_state = str(manifest_state).strip().lower()
    legacy_enabled = env_flag_enabled(legacy_fallback_env_var, default=True)

    if normalized_state == "valid":
        return ManifestFallbackDecision(
            source="manifest",
            reason="bundle manifest is available and valid",
            manifest_state="valid",
            legacy_fallback_enabled=legacy_enabled,
        )

    if normalized_state in {"missing", "invalid"} and legacy_enabled:
        return ManifestFallbackDecision(
            source="legacy",
            reason=f"bundle manifest is {normalized_state} and legacy path fallback is enabled",
            manifest_state=normalized_state,
            legacy_fallback_enabled=True,
        )

    return ManifestFallbackDecision(
        source="unavailable",
        reason=f"bundle manifest is {normalized_state or 'unavailable'} and legacy path fallback is disabled",
        manifest_state=normalized_state or "unavailable",
        legacy_fallback_enabled=legacy_enabled,
    )


__all__ = [
    "ManifestFallbackDecision",
    "StorageSourceDecision",
    "env_flag_enabled",
    "resolve_manifest_vs_legacy_fallback",
    "resolve_local_vs_object_storage",
]
