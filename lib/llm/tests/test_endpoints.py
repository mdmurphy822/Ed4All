"""Unit tests for the unified LLM/API endpoint registry loader.

Schema-validates the shipped ``config/endpoints.yaml``, exercises the
frozen resolution precedence (``explicit kwarg > *_env var > registry
default``), the ``api_key_required`` fail-loud, the closed provenance
set the Touch enum derives from, the generic
``build_openai_compatible_client`` attach point, and the R11 sync
invariant against the legacy ``_OPENAI_COMPATIBLE_PROVIDERS`` registry.
"""

from __future__ import annotations

import pytest

from lib.llm import endpoints as ep


# ---------------------------------------------------------------------------
# Registry load + schema
# ---------------------------------------------------------------------------


def test_shipped_registry_loads_and_validates():
    reg = ep.load_endpoint_registry()
    # All six canonical endpoints plus the migrated stubs are present.
    for name in ("anthropic", "local", "together", "nvidia", "claude_session"):
        assert name in reg, f"{name} missing from shipped registry"
    for name in ("together-vision", "groq", "fireworks", "deepseek"):
        assert name in reg, f"{name} (migrated stub) missing"


def test_endpoint_names_nonempty_and_includes_nvidia():
    names = ep.endpoint_names()
    assert "nvidia" in names
    assert "claude_session" in names
    assert len(names) >= 5


# ---------------------------------------------------------------------------
# NVIDIA path (verified working — must keep resolving identically)
# ---------------------------------------------------------------------------


def test_resolve_nvidia_defaults(monkeypatch):
    # Ensure no env override leaks from the operator shell.
    for v in ("NVIDIA_BASE_URL", "NVIDIA_LARGE_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-printed")
    r = ep.resolve_endpoint("nvidia")
    assert r.base_url == "https://integrate.api.nvidia.com/v1"
    assert r.model == "nvidia/nemotron-3-nano-30b-a3b"
    assert r.api_key == "test-key-not-printed"
    assert r.api_key_required is True
    assert r.provenance_provider == "nvidia"
    assert r.kind == "openai_compatible"


def test_resolve_nvidia_env_overrides_win(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nim.local/v1")
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "nvidia/custom-model")
    r = ep.resolve_endpoint("nvidia")
    assert r.base_url == "https://nim.local/v1"
    assert r.model == "nvidia/custom-model"


def test_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "env-model")
    r = ep.resolve_endpoint(
        "nvidia",
        model_override="explicit-model",
        api_key_override="explicit-key",
        base_url_override="https://explicit/v1",
    )
    assert r.model == "explicit-model"
    assert r.api_key == "explicit-key"
    assert r.base_url == "https://explicit/v1"


# ---------------------------------------------------------------------------
# api_key_required fail-loud
# ---------------------------------------------------------------------------


def test_api_key_required_raises_when_unset(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ep.EndpointKeyRequired):
        ep.resolve_endpoint("nvidia")


def test_local_no_key_required_uses_default(monkeypatch):
    for v in ("LOCAL_SYNTHESIS_API_KEY", "LOCAL_SYNTHESIS_MODEL",
              "LOCAL_SYNTHESIS_BASE_URL", "LOCAL_VISION_CAPABLE"):
        monkeypatch.delenv(v, raising=False)
    r = ep.resolve_endpoint("local")
    assert r.api_key_required is False
    assert r.api_key == "local"  # api_key_default floor
    assert r.base_url == "http://localhost:11434/v1"
    # 2-tier design: the local row default_model is the 7B that fits an 8GB
    # GPU fully resident (LOCAL_SYNTHESIS_MODEL still overrides per-run).
    assert r.model == "qwen2.5:7b-instruct-q4_K_M"
    assert r.provenance_provider == "local"


def test_unknown_endpoint_lists_names():
    with pytest.raises(ep.UnknownEndpoint) as exc:
        ep.resolve_endpoint("does-not-exist")
    assert "does-not-exist" in str(exc.value)
    assert "nvidia" in str(exc.value)


# ---------------------------------------------------------------------------
# Provenance set (the Touch enum derives from THIS)
# ---------------------------------------------------------------------------


def test_provenance_provider_names_is_frozen_set():
    got = set(ep.provenance_provider_names())
    expected = {
        "anthropic",
        "local",
        "together",
        "nvidia",
        "claude_session",
        "deterministic",
    }
    assert got == expected, (
        "provenance set drifted from the historical hand-maintained Touch "
        f"enum; got {sorted(got)}"
    )


def test_provenance_names_sorted():
    names = ep.provenance_provider_names()
    assert list(names) == sorted(names)


def test_groq_fireworks_deepseek_map_to_local_provenance():
    reg = ep.load_endpoint_registry()
    for name in ("groq", "fireworks", "deepseek"):
        assert reg[name]["provenance_provider"] == "local"
    assert ep.resolve_endpoint(
        "together-vision", api_key_override="k"
    ).provenance_provider == "together"


# ---------------------------------------------------------------------------
# Generic openai-compatible attach point
# ---------------------------------------------------------------------------


def test_build_client_nvidia(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    for v in ("NVIDIA_BASE_URL", "NVIDIA_LARGE_MODEL"):
        monkeypatch.delenv(v, raising=False)
    client = ep.build_openai_compatible_client("nvidia")
    assert client.base_url == "https://integrate.api.nvidia.com/v1"
    assert client.model == "nvidia/nemotron-3-nano-30b-a3b"


def test_build_client_injected_client_skips_key_check(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    import httpx

    sentinel = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    client = ep.build_openai_compatible_client("nvidia", client=sentinel)
    assert client.client is sentinel


def test_build_client_rejects_non_openai_kind(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ep.EndpointKindError):
        ep.build_openai_compatible_client("anthropic")
    with pytest.raises(ep.EndpointKindError):
        ep.build_openai_compatible_client("claude_session")


def test_provider_label_defaults_to_name(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    client = ep.build_openai_compatible_client("together")
    # provider_label is the audit string surfaced in capture rationales.
    assert client._provider_label == "together"


# ---------------------------------------------------------------------------
# B1 regression net: identity modules re-source FROM the registry
# (asserted here so the test lands with the loader; B1 makes it pass).
# ---------------------------------------------------------------------------


def test_nvidia_identity_module_matches_registry(monkeypatch):
    for v in ("NVIDIA_BASE_URL", "NVIDIA_LARGE_MODEL"):
        monkeypatch.delenv(v, raising=False)
    from Trainforge.generators import _nvidia_provider as nv

    r = ep.resolve_endpoint("nvidia", api_key_override="k")
    assert nv.DEFAULT_SYNTHESIS_MODEL == r.model
    assert nv.DEFAULT_BASE_URL == r.base_url
    assert nv.ENV_API_KEY == "NVIDIA_API_KEY"
    assert nv.ENV_MODEL == "NVIDIA_LARGE_MODEL"
    assert nv.ENV_BASE_URL == "NVIDIA_BASE_URL"


# ---------------------------------------------------------------------------
# Option B (landed): _OPENAI_COMPATIBLE_PROVIDERS is now a PROJECTION of the
# openai_compatible rows in config/endpoints.yaml, so the two registries are
# structurally guaranteed to agree (one IS derived from the other). The
# answer path consumes the unified registry transitively.
# ---------------------------------------------------------------------------


def test_legacy_openai_registry_stays_in_sync():
    from MCP.orchestrator.llm_backend import _OPENAI_COMPATIBLE_PROVIDERS

    reg = ep.load_endpoint_registry()
    for name, legacy in _OPENAI_COMPATIBLE_PROVIDERS.items():
        assert name in reg, (
            f"_OPENAI_COMPATIBLE_PROVIDERS has {name!r} with no matching "
            f"config/endpoints.yaml row"
        )
        row = reg[name]
        assert row["kind"] == "openai_compatible"
        assert row.get("base_url") == legacy.get("base_url_default"), name
        assert row.get("default_model") == legacy.get("model_default"), name
        assert row.get("api_key_env") == legacy.get("api_key_env"), name
        assert bool(row["api_key_required"]) == bool(
            legacy.get("api_key_required", False)
        ), name


def test_legacy_registry_is_exactly_the_projection():
    """_OPENAI_COMPATIBLE_PROVIDERS must BE the YAML projection.

    Locks in the single-source-of-truth contract: nobody may
    re-introduce a hand-maintained literal dict in llm_backend.py that
    drifts from config/endpoints.yaml. The module-level dict is built
    once at import from openai_compatible_legacy_registry(); this asserts
    it equals a fresh projection.
    """
    from MCP.orchestrator.llm_backend import _OPENAI_COMPATIBLE_PROVIDERS

    assert _OPENAI_COMPATIBLE_PROVIDERS == ep.openai_compatible_legacy_registry()


def test_projection_includes_nvidia_with_verified_identity():
    """The verified NVIDIA endpoint now projects into the legacy registry.

    Previously nvidia lived only as a _base.py branch + identity module;
    folding the YAML in surfaces it as an OpenAI-compatible provider with
    the verified base_url / model / key-env.
    """
    proj = ep.openai_compatible_legacy_registry()
    assert "nvidia" in proj
    nv = proj["nvidia"]
    assert nv["base_url_default"] == "https://integrate.api.nvidia.com/v1"
    assert nv["api_key_env"] == "NVIDIA_API_KEY"
    assert nv["api_key_required"] is True


def test_projection_is_a_fresh_mutable_dict():
    """The projection returns a fresh dict each call (monkeypatch-safe).

    The W-D12 dynamic-extension tests mutate _OPENAI_COMPATIBLE_PROVIDERS
    via monkeypatch.setitem; the projection must not share row objects
    with the cached registry, so a mutation can't poison the cache.
    """
    a = ep.openai_compatible_legacy_registry()
    b = ep.openai_compatible_legacy_registry()
    assert a == b
    assert a is not b
    a["local"]["model_default"] = "MUTATED"
    # A fresh projection is unaffected by the mutation above.
    assert ep.openai_compatible_legacy_registry()["local"]["model_default"] != "MUTATED"
