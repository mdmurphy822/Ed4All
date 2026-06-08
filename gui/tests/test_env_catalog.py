"""Tests for ``gui.env_catalog`` — provider registry fidelity, defaults, routing.

No fastapi needed. The provider registry mirror is asserted against the REAL
``_OPENAI_COMPATIBLE_PROVIDERS`` registry when importable (skipped tolerantly if
the heavy MCP backend isn't importable in this environment).
"""

from __future__ import annotations

import pytest

from gui import env_catalog


def test_provider_names_cover_spec_set():
    names = set(env_catalog.provider_names())
    expected = {
        "anthropic",
        "local",
        "together",
        "together-vision",
        "groq",
        "fireworks",
        "deepseek",
        "mock",
    }
    assert expected == names


def test_every_provider_traces_to_real_registry():
    """Each OpenAI-compatible provider name must exist in the real registry."""
    try:
        from MCP.orchestrator.llm_backend import _OPENAI_COMPATIBLE_PROVIDERS
    except Exception:  # noqa: BLE001 — backend not importable here
        pytest.skip("MCP.orchestrator.llm_backend not importable in this env")

    registry_names = set(_OPENAI_COMPATIBLE_PROVIDERS.keys())
    # Every catalog provider except the native anthropic/mock backends must be
    # present in the OpenAI-compatible registry.
    native = {"anthropic", "mock"}
    for entry in env_catalog.PROVIDERS:
        name = entry["name"]
        if name in native:
            continue
        assert name in registry_names, f"{name} missing from real registry"


def test_provider_entries_have_required_keys():
    required = {
        "name",
        "label",
        "api_key_env",
        "base_url_default",
        "model_env",
        "model_default",
        "api_key_required",
        "vision_capable",
        "unverified",
    }
    for entry in env_catalog.PROVIDERS:
        assert required <= set(entry.keys()), f"{entry['name']} missing keys"


def test_default_settings_shape():
    doc = env_catalog.default_settings()
    assert doc["version"] == 1
    assert set(doc.keys()) >= {"env", "model_routing", "retrieval", "flags"}
    routing = doc["model_routing"]
    assert routing["global"] == {"mode": "local", "provider": "anthropic", "model": None}
    # Each routing task block is a dict.
    for task, block in routing.items():
        assert isinstance(block, dict), task
    # Fresh dict each call.
    assert env_catalog.default_settings() is not doc


def test_routing_to_env_emits_expected_keys():
    routing = {
        "global": {"mode": "api", "provider": "anthropic", "model": "claude-x"},
        "dart": {"provider": "local", "vision_provider": "together-vision"},
        "courseforge_outline": {"provider": "local", "model": "qwen-x"},
        "trainforge_synthesis": {"provider": "together", "model": "llama-x"},
    }
    env = env_catalog.routing_to_env(routing)
    assert env["LLM_MODE"] == "api"
    assert env["LLM_PROVIDER"] == "anthropic"
    assert env["LLM_MODEL"] == "claude-x"
    assert env["DART_PROVIDER"] == "local"
    assert env["DART_VISION_PROVIDER"] == "together-vision"
    assert env["COURSEFORGE_OUTLINE_PROVIDER"] == "local"
    assert env["COURSEFORGE_OUTLINE_MODEL"] == "qwen-x"
    assert env["TRAINFORGE_SYNTHESIS_PROVIDER"] == "together"
    # trainforge_synthesis model maps to ANTHROPIC_SYNTHESIS_MODEL per the map.
    assert env["ANTHROPIC_SYNTHESIS_MODEL"] == "llama-x"


def test_routing_to_env_vision_model_provider_conditional():
    # together-vision => TOGETHER_VISION_MODEL emitted.
    env = env_catalog.routing_to_env(
        {"vision": {"provider": "together-vision", "model": "vlm-90b"}}
    )
    assert env.get("TOGETHER_VISION_MODEL") == "vlm-90b"
    assert env.get("DART_VISION_PROVIDER") == "together-vision"

    # local vision provider => no invented DART_VISION_MODEL / TOGETHER var.
    env2 = env_catalog.routing_to_env(
        {"vision": {"provider": "local", "model": "llava:13b"}}
    )
    assert "TOGETHER_VISION_MODEL" not in env2
    assert env2.get("DART_VISION_PROVIDER") == "local"


def test_routing_to_env_drops_nulls():
    env = env_catalog.routing_to_env(
        {"courseforge_outline": {"provider": None, "model": ""}}
    )
    assert env == {}


def test_vision_provider_names_subset():
    vision = set(env_catalog.vision_provider_names())
    assert vision <= set(env_catalog.provider_names())
    # The spec's vision-capable trio is present.
    assert {"anthropic", "together-vision", "local"} <= vision


def test_base_models_nonempty_and_stringly():
    assert env_catalog.BASE_MODELS, "BASE_MODELS must not be empty"
    assert all(isinstance(m, str) for m in env_catalog.BASE_MODELS)


def test_by_category_groups_catalog():
    grouped = env_catalog.by_category()
    assert "credentials" in grouped
    cred_keys = {e["key"] for e in grouped["credentials"]}
    assert {"ANTHROPIC_API_KEY", "TOGETHER_API_KEY", "LOCAL_SYNTHESIS_API_KEY"} <= cred_keys
    # Every catalog entry surfaces in exactly one category bucket.
    total = sum(len(v) for v in grouped.values())
    assert total == len(env_catalog.CATALOG)


def test_catalog_keys_cover_credentials_and_global():
    keys = set(env_catalog.catalog_keys())
    assert {"ANTHROPIC_API_KEY", "LLM_MODE", "LLM_PROVIDER", "LLM_MODEL"} <= keys
