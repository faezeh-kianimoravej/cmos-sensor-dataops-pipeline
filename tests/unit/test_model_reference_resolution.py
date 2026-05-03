from __future__ import annotations

from src.observed.utils.model_reference import resolve_model_reference


def test_resolve_model_reference_prefers_stage_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("OBSERVED_STAGE1_MODEL_VERSION", "git-stage1")
    monkeypatch.setenv("OBSERVED_MODEL_VERSION", "git-shared")

    reference = resolve_model_reference("stage1")

    assert reference.version == "git-stage1"
    assert reference.source == "OBSERVED_STAGE1_MODEL_VERSION"


def test_resolve_model_reference_uses_shared_env_when_stage_specific_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OBSERVED_STAGE2_MODEL_VERSION", raising=False)
    monkeypatch.setenv("OBSERVED_MODEL_VERSION", "git-shared")

    reference = resolve_model_reference("stage2")

    assert reference.version == "git-shared"
    assert reference.source == "OBSERVED_MODEL_VERSION"


def test_resolve_model_reference_reports_unconfigured_when_no_env_is_set(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OBSERVED_STAGE1_MODEL_VERSION", raising=False)
    monkeypatch.delenv("OBSERVED_MODEL_VERSION", raising=False)

    reference = resolve_model_reference("stage1")

    assert reference.version is None
    assert reference.source == "unconfigured"
