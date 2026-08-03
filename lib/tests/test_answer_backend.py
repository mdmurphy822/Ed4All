"""Tests for the WS3 answer backend resolution + loopback enforcement (D1)."""
from __future__ import annotations

import pytest

from lib.retrieval.answer_backend import (
    AnswerProviderNotLocal,
    ENV_ANSWER_MODEL,
    ENV_ANSWER_PROVIDER,
    ENV_ANSWER_TIMEOUT,
    ENV_EVAL_COMPOSER_MODEL,
    ENV_EVAL_COMPOSER_PROVIDER,
    ResolvedAnswerBackend,
    build_answer_client,
    resolve_answer_backend,
    resolve_eval_composer_backend,
)


# ---------------------------------------------------------------------------
# Env-var hygiene — clear answer + local-server vars per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        ENV_ANSWER_PROVIDER, ENV_ANSWER_MODEL, ENV_ANSWER_TIMEOUT,
        ENV_EVAL_COMPOSER_PROVIDER, ENV_EVAL_COMPOSER_MODEL,
        "LOCAL_SYNTHESIS_BASE_URL", "LOCAL_SYNTHESIS_MODEL",
        "LOCAL_SYNTHESIS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_default_resolves_to_canonical_local_loopback_model():
    resolved = resolve_answer_backend()
    assert isinstance(resolved, ResolvedAnswerBackend)
    assert resolved.provider_name == "local"
    # Default model chain resolves through the canonical local registry row;
    # an explicit LOCAL_SYNTHESIS_MODEL still overrides this identity.
    assert resolved.model_id == "nemotron-3-nano-30b-a3b"
    assert "localhost" in resolved.base_url or "127.0.0.1" in resolved.base_url
    assert resolved.timeout == 120.0


def test_env_provider_default_is_local(monkeypatch):
    monkeypatch.setenv(ENV_ANSWER_PROVIDER, "local")
    assert resolve_answer_backend().provider_name == "local"


def test_model_resolution_precedence_kwarg_over_env(monkeypatch):
    monkeypatch.setenv(ENV_ANSWER_MODEL, "env-model")
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "registry-env-model")
    # kwarg wins over both env vars
    assert resolve_answer_backend(model_id="kwarg-model").model_id == "kwarg-model"
    # ED4ALL_ANSWER_MODEL wins over LOCAL_SYNTHESIS_MODEL
    assert resolve_answer_backend().model_id == "env-model"


def test_model_resolution_falls_back_to_local_synthesis_model(monkeypatch):
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    assert resolve_answer_backend().model_id == "qwen2.5:7b-instruct-q4_K_M"


def test_timeout_from_env(monkeypatch):
    monkeypatch.setenv(ENV_ANSWER_TIMEOUT, "45.5")
    assert resolve_answer_backend().timeout == 45.5


def test_timeout_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(ENV_ANSWER_TIMEOUT, "not-a-number")
    assert resolve_answer_backend().timeout == 120.0


@pytest.mark.parametrize(
    "provider", ["together", "groq", "fireworks", "deepseek", "nvidia"]
)
def test_cloud_providers_refused_as_not_local(provider):
    # ``nvidia`` is included now that the unified registry projects it into
    # _OPENAI_COMPATIBLE_PROVIDERS — the loopback gate must keep refusing it.
    with pytest.raises(AnswerProviderNotLocal):
        resolve_answer_backend(provider_name=provider)


def test_non_loopback_local_base_url_refused(monkeypatch):
    # A reverse-proxied cloud URL placed in LOCAL_SYNTHESIS_BASE_URL must
    # still fail the loopback check (check is on the resolved URL).
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "https://api.example.com/v1")
    with pytest.raises(AnswerProviderNotLocal):
        resolve_answer_backend(provider_name="local")


def test_ipv6_loopback_accepted(monkeypatch):
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://[::1]:11434/v1")
    resolved = resolve_answer_backend(provider_name="local")
    assert "[::1]" in resolved.base_url


def test_unknown_provider_raises_valueerror():
    with pytest.raises(ValueError):
        resolve_answer_backend(provider_name="does-not-exist")


def test_build_answer_client_uses_json_mode_and_provider_label(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import Trainforge.generators._openai_compatible_client as occ

    monkeypatch.setattr(occ, "OpenAICompatibleClient", _FakeClient)
    resolved = resolve_answer_backend()
    build_answer_client(resolved, capture="cap-handle")

    assert captured["json_mode"] is True
    assert captured["provider_label"] == "answer:local"
    assert captured["capture"] == "cap-handle"
    assert captured["model"] == resolved.model_id
    assert captured["base_url"] == resolved.base_url
    assert captured["timeout"] == resolved.timeout


def test_build_answer_client_default_json_mode_is_true(monkeypatch):
    # Default preserves the grounded pipeline's structured-envelope behavior
    # byte-for-byte: no json_mode kwarg → json_mode=True on the wire client.
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import Trainforge.generators._openai_compatible_client as occ

    monkeypatch.setattr(occ, "OpenAICompatibleClient", _FakeClient)
    build_answer_client()  # resolved=None → env/default resolution
    assert captured["json_mode"] is True


# ---------------------------------------------------------------------------
# E7a — diagnostic-composer backend resolution
# ---------------------------------------------------------------------------


def test_eval_composer_absent_by_default_returns_none():
    # Both envs unset + no kwargs → arm OFF (production resolution untouched).
    assert resolve_eval_composer_backend() is None


def test_eval_composer_env_provider_activates(monkeypatch):
    monkeypatch.setenv(ENV_EVAL_COMPOSER_PROVIDER, "local")
    resolved = resolve_eval_composer_backend()
    assert isinstance(resolved, ResolvedAnswerBackend)
    assert resolved.provider_name == "local"
    assert "localhost" in resolved.base_url or "127.0.0.1" in resolved.base_url


def test_eval_composer_env_model_only_activates(monkeypatch):
    # A model override alone activates the arm on the default (local) provider.
    monkeypatch.setenv(ENV_EVAL_COMPOSER_MODEL, "super-120b")
    resolved = resolve_eval_composer_backend()
    assert resolved is not None
    assert resolved.model_id == "super-120b"
    assert resolved.provider_name == "local"


def test_eval_composer_kwarg_provider_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_EVAL_COMPOSER_PROVIDER, "local")
    monkeypatch.setenv(ENV_EVAL_COMPOSER_MODEL, "env-model")
    resolved = resolve_eval_composer_backend(model_id="kwarg-model")
    assert resolved.model_id == "kwarg-model"


def test_eval_composer_enforces_loopback(monkeypatch):
    # The diagnostic seat is loopback-gated exactly like the production path.
    monkeypatch.setenv(ENV_EVAL_COMPOSER_PROVIDER, "together")
    with pytest.raises(AnswerProviderNotLocal):
        resolve_eval_composer_backend()


def test_eval_composer_non_loopback_local_url_refused(monkeypatch):
    monkeypatch.setenv(ENV_EVAL_COMPOSER_PROVIDER, "local")
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "https://api.example.com/v1")
    with pytest.raises(AnswerProviderNotLocal):
        resolve_eval_composer_backend()


def test_build_answer_client_json_mode_false_threads_through(monkeypatch):
    # The eval BASE arm needs raw free text, so it passes json_mode=False.
    # Assert the flag threads into the OpenAICompatibleClient construction.
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import Trainforge.generators._openai_compatible_client as occ

    monkeypatch.setattr(occ, "OpenAICompatibleClient", _FakeClient)
    resolved = resolve_answer_backend()
    build_answer_client(resolved, capture="cap-handle", json_mode=False)
    assert captured["json_mode"] is False
    # Everything else is unchanged from the default-mode build.
    assert captured["provider_label"] == "answer:local"
