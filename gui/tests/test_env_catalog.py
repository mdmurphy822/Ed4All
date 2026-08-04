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
        "vision": {"provider": "local", "model": "qwen2.5vl:7b"},
        "courseforge_outline": {"provider": "local", "model": "qwen-x"},
        "trainforge_synthesis": {"provider": "together", "model": "llama-x"},
    }
    env = env_catalog.routing_to_env(routing)
    assert env["LLM_MODE"] == "api"
    assert env["LLM_PROVIDER"] == "anthropic"
    assert env["LLM_MODEL"] == "claude-x"
    assert env["SEMANTIK_VLM_PROVIDER"] == "local"
    assert env["SEMANTIK_VLM_MODEL"] == "qwen2.5vl:7b"
    assert env["COURSEFORGE_OUTLINE_PROVIDER"] == "local"
    assert env["COURSEFORGE_OUTLINE_MODEL"] == "qwen-x"
    assert env["TRAINFORGE_SYNTHESIS_PROVIDER"] == "together"
    # trainforge_synthesis model maps to ANTHROPIC_SYNTHESIS_MODEL per the map.
    assert env["ANTHROPIC_SYNTHESIS_MODEL"] == "llama-x"


def test_routing_to_env_vision_maps_semantik_vlm_seat():
    # Vision routing maps to the real SemantiK VLM seat knobs, universally:
    # the seat is provider-agnostic, so the model is always SEMANTIK_VLM_MODEL.
    env = env_catalog.routing_to_env(
        {"vision": {"provider": "together-vision", "model": "vlm-90b"}}
    )
    assert env.get("SEMANTIK_VLM_PROVIDER") == "together-vision"
    assert env.get("SEMANTIK_VLM_MODEL") == "vlm-90b"

    # local vision provider still carries the model on SEMANTIK_VLM_MODEL.
    env2 = env_catalog.routing_to_env(
        {"vision": {"provider": "local", "model": "llava:13b"}}
    )
    assert env2.get("SEMANTIK_VLM_PROVIDER") == "local"
    assert env2.get("SEMANTIK_VLM_MODEL") == "llava:13b"
    # No invented TOGETHER vision-model var.
    assert "TOGETHER_VISION_MODEL" not in env2


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


# ---------------------------------------------------------------------------
# Catalog completeness for the embedding / chunking / retrieval knobs.
#
# gui/CLAUDE.md: "New env knob the pipeline reads -> add it to
# gui/env_catalog.py, not ad-hoc os.environ reads in a router." These ten
# landed without catalog entries; the defaults below are asserted against the
# REAL resolvers (never a doc) further down.
# ---------------------------------------------------------------------------
_NEW_KNOBS = {
    "ED4ALL_EMBEDDING_DTYPE": ("embedding", "enum", "fp32"),
    "ED4ALL_EMBEDDING_CLIENT_CACHE": ("embedding", "bool", True),
    "ED4ALL_EMBEDDING_CLIENT_CACHE_MAX": ("embedding", "number", 2),
    "ED4ALL_CHUNK_DEDUP": ("chunking", "bool", False),
    "ED4ALL_CHUNK_DEDUP_MIN_TOKENS": ("chunking", "number", 8),
    "ED4ALL_HTML_PARSE_WORKERS": ("chunking", "number", 10),
    "ED4ALL_HTML_PARSE_START_METHOD": ("chunking", "enum", "spawn"),
    "ED4ALL_HTML_ASSET_REJECT": ("chunking", "bool", True),
    "ED4ALL_RETRIEVAL_BLAS_THREADS": ("embedding", "number", 8),
    "ED4ALL_RETRIEVAL_TOPK_LEGACY": ("embedding", "bool", False),
}


@pytest.mark.parametrize("key", sorted(_NEW_KNOBS))
def test_new_knob_has_catalog_entry(key):
    category, type_, default = _NEW_KNOBS[key]
    entry = env_catalog.catalog_entry(key)
    assert entry is not None, f"{key} is missing from the GUI env catalog"
    assert entry["category"] == category
    assert entry["type"] == type_
    assert entry["default"] == default
    # Same required shape every other catalog row carries.
    assert {"key", "label", "category", "type", "default", "help", "applies_to"} <= set(entry)
    assert entry["label"] and entry["help"]
    # An enum row must actually declare its closed list.
    if type_ == "enum":
        assert isinstance(entry.get("enum"), list) and entry["enum"]
        assert entry["default"] in entry["enum"]


def test_catalog_defaults_match_embedding_resolvers():
    """The embedding-family defaults trace to lib/embedding/providers.py."""
    try:
        from lib.embedding import providers
    except Exception:  # noqa: BLE001 — embedding extras absent in this env
        pytest.skip("lib.embedding.providers not importable in this env")

    dtype = env_catalog.catalog_entry("ED4ALL_EMBEDDING_DTYPE")
    assert dtype["default"] == providers.DEFAULT_DTYPE
    assert dtype["enum"] == list(providers.VALID_DTYPES)

    # Client cache: default ON, LRU bound 2 — asserted through the resolvers
    # with an EMPTY env so a stray process env cannot make this pass.
    assert providers.resolve_client_cache_enabled({}) is True
    assert (
        env_catalog.catalog_entry("ED4ALL_EMBEDDING_CLIENT_CACHE")["default"] is True
    )
    assert providers.resolve_client_cache_max({}) == 2
    assert env_catalog.catalog_entry("ED4ALL_EMBEDDING_CLIENT_CACHE_MAX")["default"] == 2


def test_catalog_defaults_match_chunk_dedup_resolvers():
    try:
        from Trainforge.chunker import cross_course_dedup
    except Exception:  # noqa: BLE001
        pytest.skip("Trainforge.chunker.cross_course_dedup not importable in this env")

    assert cross_course_dedup.resolve_chunk_dedup_enabled({}) is False
    assert env_catalog.catalog_entry("ED4ALL_CHUNK_DEDUP")["default"] is False
    assert cross_course_dedup.resolve_chunk_dedup_min_tokens({}) == 8
    assert env_catalog.catalog_entry("ED4ALL_CHUNK_DEDUP_MIN_TOKENS")["default"] == 8


def test_catalog_defaults_match_retrieval_resolvers():
    try:
        from LibV2.tools.libv2 import vector_index
    except Exception:  # noqa: BLE001 — numpy / LibV2 deps absent
        pytest.skip("LibV2.tools.libv2.retrieval.vector_index not importable in this env")

    assert vector_index.resolve_retrieval_blas_threads({}) == 8
    assert env_catalog.catalog_entry("ED4ALL_RETRIEVAL_BLAS_THREADS")["default"] == 8
    assert vector_index.resolve_topk_legacy({}) is False
    assert env_catalog.catalog_entry("ED4ALL_RETRIEVAL_TOPK_LEGACY")["default"] is False


def test_catalog_defaults_match_html_parse_resolvers(monkeypatch):
    try:
        from MCP.tools import pipeline_tools
    except Exception:  # noqa: BLE001 — heavy MCP backend not importable here
        pytest.skip("MCP.tools.pipeline_tools not importable in this env")

    # These resolvers read os.environ directly; clear the knobs so the
    # assertion is about the CODE default, not the operator's shell.
    for name in (
        "ED4ALL_HTML_PARSE_WORKERS",
        "ED4ALL_HTML_PARSE_START_METHOD",
        "ED4ALL_HTML_ASSET_REJECT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert pipeline_tools._DEFAULT_HTML_PARSE_WORKERS == 10
    assert env_catalog.catalog_entry("ED4ALL_HTML_PARSE_WORKERS")["default"] == 10
    assert pipeline_tools._resolve_html_parse_start_method() == "spawn"
    start = env_catalog.catalog_entry("ED4ALL_HTML_PARSE_START_METHOD")
    assert start["default"] == pipeline_tools._DEFAULT_HTML_PARSE_START_METHOD
    assert start["enum"] == list(pipeline_tools._HTML_PARSE_START_METHODS)
    assert pipeline_tools._resolve_html_asset_reject() is True
    assert env_catalog.catalog_entry("ED4ALL_HTML_ASSET_REJECT")["default"] is True


def test_parse_start_method_never_offers_fork():
    """``fork`` deadlocks the chunking phases — the GUI must not offer it."""
    entry = env_catalog.catalog_entry("ED4ALL_HTML_PARSE_START_METHOD")
    assert "fork" not in entry["enum"]
    assert entry["enum"] == ["spawn", "forkserver", "serial"]
    assert env_catalog.value_is_valid("ED4ALL_HTML_PARSE_START_METHOD", "fork") is False
    for good in entry["enum"]:
        assert env_catalog.value_is_valid("ED4ALL_HTML_PARSE_START_METHOD", good) is True


# ---------------------------------------------------------------------------
# ED4ALL_EMBEDDING_DEVICE — widened from enum[cpu,cuda] to a CLOSED pattern so
# the ``cuda:N`` token the resolver accepts is representable.
# ---------------------------------------------------------------------------
def test_embedding_device_entry_is_pattern_validated():
    entry = env_catalog.catalog_entry("ED4ALL_EMBEDDING_DEVICE")
    assert entry["default"] == "cuda"
    assert entry["type"] == "string"
    assert entry["pattern"] == env_catalog.DEVICE_TOKEN_PATTERN
    # The old two-value enum is gone (it made cuda:N unrepresentable), but the
    # entry is NOT a free-text field.
    assert "enum" not in entry


@pytest.mark.parametrize("token", ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:7"])
def test_embedding_device_accepts_resolver_tokens(token):
    assert env_catalog.value_is_valid("ED4ALL_EMBEDDING_DEVICE", token) is True


@pytest.mark.parametrize(
    "token",
    [
        "auto",  # never accepted — auto-detection is silent degradation
        "gpu",
        "cuda:",
        "cuda:x",
        "cuda:1:2",
        "mps",
        "CUDA",  # canonical lower-case form only in the GUI
    ],
)
def test_embedding_device_rejects_invalid_tokens(token):
    assert env_catalog.value_is_valid("ED4ALL_EMBEDDING_DEVICE", token) is False


def test_embedding_device_pattern_agrees_with_the_real_resolver():
    """Every token the pattern accepts must normalize; ``auto`` must not.

    Guards the widening against inventing a token the embedding stack would
    reject at load time (silent degradation in the other direction).
    """
    try:
        from lib.embedding.providers import normalize_device_token
    except Exception:  # noqa: BLE001
        pytest.skip("lib.embedding.providers not importable in this env")

    for token in ("cpu", "cuda", "cuda:0", "cuda:3"):
        assert normalize_device_token(token) == token
    with pytest.raises(ValueError):
        normalize_device_token("auto")


def test_value_is_valid_treats_unset_and_unknown_as_no_opinion():
    # Unset = "let the resolver default win", never a fabricated selection.
    assert env_catalog.value_is_valid("ED4ALL_EMBEDDING_DEVICE", None) is True
    assert env_catalog.value_is_valid("ED4ALL_EMBEDDING_DEVICE", "  ") is True
    # An unknown key gets no invented constraint.
    assert env_catalog.value_is_valid("NOT_A_CATALOG_KEY", "whatever") is True
    # A row with neither enum nor pattern is unconstrained.
    assert env_catalog.value_is_valid("LLM_MODEL", "some-model-id") is True


def test_new_knob_categories_group_cleanly():
    grouped = env_catalog.by_category()
    assert "chunking" in grouped
    chunk_keys = {e["key"] for e in grouped["chunking"]}
    assert {
        "ED4ALL_CHUNK_DEDUP",
        "ED4ALL_CHUNK_DEDUP_MIN_TOKENS",
        "ED4ALL_HTML_PARSE_WORKERS",
        "ED4ALL_HTML_PARSE_START_METHOD",
        "ED4ALL_HTML_ASSET_REJECT",
    } == chunk_keys
    # The knobs stay out of the Studio-facing subset (operator surface only).
    embedding_keys = {e["key"] for e in grouped["embedding"]}
    assert {
        "ED4ALL_EMBEDDING_DTYPE",
        "ED4ALL_EMBEDDING_CLIENT_CACHE",
        "ED4ALL_EMBEDDING_CLIENT_CACHE_MAX",
        "ED4ALL_RETRIEVAL_BLAS_THREADS",
        "ED4ALL_RETRIEVAL_TOPK_LEGACY",
    } <= embedding_keys
    # No duplicate keys anywhere in the catalog.
    keys = env_catalog.catalog_keys()
    assert len(keys) == len(set(keys))
