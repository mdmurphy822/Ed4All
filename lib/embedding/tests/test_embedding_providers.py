"""Tests for the embedding provider registry (lib/embedding/providers.py).

All tests run with ZERO real weights and ZERO network — the ``fake``
provider exercises the build/read path deterministically; the ``st`` and
``openai-embeddings`` paths are tested via monkeypatch / stubbed httpx
transport so CI never downloads a model or hits a server.

A single ``@pytest.mark.real_models`` smoke for all-MiniLM-L6-v2 (the only
cached model per the WS2 environment facts) skips when the model is not
in the HF cache, so CI without a cache stays green.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from lib.embedding.providers import (
    EMBEDDING_MODEL_REGISTRY,
    EmbeddingBackendUnavailable,
    EmbeddingClient,
    ResolvedEmbeddingProvider,
    allow_fake_enabled,
    build_embedding_client,
    resolve_embedding_provider,
)


# ---------------------------------------------------------------------------
# Registry resolution + env precedence
# ---------------------------------------------------------------------------
def test_default_provider_is_st(monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("ED4ALL_EMBEDDING_MODEL", raising=False)
    resolved = resolve_embedding_provider()
    assert resolved.provider_name == "st"
    assert resolved.kind == "st"
    assert resolved.model_id == "BAAI/bge-large-en-v1.5"
    assert resolved.device == "cpu"  # determinism default (D4)
    assert resolved.batch_size == 16


def test_env_selects_provider(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    resolved = resolve_embedding_provider()
    assert resolved.provider_name == "fake"
    assert resolved.kind == "fake"
    assert resolved.model_id == "fake-deterministic-v1"


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    resolved = resolve_embedding_provider(provider_name="st")
    assert resolved.provider_name == "st"


def test_model_env_precedence(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    resolved = resolve_embedding_provider()
    assert resolved.model_id == "BAAI/bge-large-en-v1.5"
    # explicit model_id arg beats the env var
    resolved2 = resolve_embedding_provider(model_id="sentence-transformers/all-MiniLM-L6-v2")
    assert resolved2.model_id == "sentence-transformers/all-MiniLM-L6-v2"


def test_batch_size_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_BATCH_SIZE", "64")
    assert resolve_embedding_provider().batch_size == 64


def test_batch_size_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_BATCH_SIZE", "not-an-int")
    assert resolve_embedding_provider().batch_size == 16
    monkeypatch.setenv("ED4ALL_EMBEDDING_BATCH_SIZE", "-3")
    assert resolve_embedding_provider().batch_size == 16


def test_device_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cuda")
    assert resolve_embedding_provider().device == "cuda"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_PROVIDER", raising=False)
    with pytest.raises(ValueError) as exc:
        resolve_embedding_provider(provider_name="does-not-exist")
    # Lists the registered providers inline.
    assert "fake" in str(exc.value)
    assert "st" in str(exc.value)
    assert "local-openai" in str(exc.value)


def test_local_openai_resolution(monkeypatch):
    for var in (
        "ED4ALL_EMBEDDING_BASE_URL",
        "ED4ALL_EMBEDDING_API_KEY",
        "ED4ALL_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    resolved = resolve_embedding_provider(provider_name="local-openai")
    assert resolved.kind == "openai-embeddings"
    assert resolved.base_url == "http://localhost:11434/v1"
    assert resolved.api_key == "local"
    assert resolved.model_id == "nomic-embed-text"


def test_local_openai_base_url_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_BASE_URL", "http://localhost:8000/v1")
    resolved = resolve_embedding_provider(provider_name="local-openai")
    assert resolved.base_url == "http://localhost:8000/v1"


def test_model_metadata_threaded(monkeypatch):
    # gte-large carries trust_remote_code=True (R2 — operator-visible).
    resolved = resolve_embedding_provider(
        provider_name="st", model_id="Alibaba-NLP/gte-large-en-v1.5"
    )
    assert resolved.trust_remote_code is True
    # nomic carries the search_query / search_document prefixes (D6).
    resolved2 = resolve_embedding_provider(
        provider_name="local-openai", model_id="nomic-ai/nomic-embed-text-v1.5"
    )
    assert resolved2.query_prefix == "search_query: "
    assert resolved2.document_prefix == "search_document: "
    # bge carries a query-side instruction prefix only; passages get none.
    resolved3 = resolve_embedding_provider(
        provider_name="st", model_id="BAAI/bge-large-en-v1.5"
    )
    assert resolved3.query_prefix.startswith("Represent this sentence")
    assert resolved3.document_prefix == ""
    assert resolved3.trust_remote_code is False


def test_model_registry_all_license_clean():
    # D5: all candidates Apache-2.0 / MIT.
    for model_id, meta in EMBEDDING_MODEL_REGISTRY.items():
        assert meta["license"] in {"Apache-2.0", "MIT"}, model_id
        assert isinstance(meta["dim"], int) and meta["dim"] > 0


# ---------------------------------------------------------------------------
# Fake provider determinism + no-network
# ---------------------------------------------------------------------------
def test_fake_determinism_same_text_same_vector():
    c1 = build_embedding_client(provider_name="fake")
    c2 = build_embedding_client(provider_name="fake")
    v1 = c1.encode_batch(["the quick brown fox"])
    v2 = c2.encode_batch(["the quick brown fox"])
    assert np.array_equal(v1, v2)  # byte-identical across client instances


def test_fake_vectors_are_normalized():
    client = build_embedding_client(provider_name="fake")
    vecs = client.encode_batch(["alpha", "beta", "gamma"])
    assert vecs.dtype == np.float32
    assert vecs.shape == (3, 32)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_fake_distinct_texts_distinct_vectors():
    client = build_embedding_client(provider_name="fake")
    vecs = client.encode_batch(["alpha", "beta"])
    assert not np.array_equal(vecs[0], vecs[1])


def test_fake_dim_property():
    client = build_embedding_client(provider_name="fake")
    assert client.dim == 32


def test_fake_empty_input():
    client = build_embedding_client(provider_name="fake")
    out = client.encode_batch([])
    assert out.shape == (0, 32)
    assert out.dtype == np.float32


def test_fake_fingerprint():
    client = build_embedding_client(provider_name="fake")
    client.encode_batch(["x"])
    fp = client.model_fingerprint()
    assert fp["kind"] == "fake"
    assert fp["provider_name"] == "fake"
    assert fp["dim"] == 32
    assert fp["model_id"] == "fake-deterministic-v1"


def test_fake_makes_no_network_calls(monkeypatch):
    """Guard: building + encoding with the fake provider touches no socket."""
    import socket

    def _boom(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("fake provider attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    client = build_embedding_client(provider_name="fake")
    vecs = client.encode_batch(["no network here", "still none"])
    assert vecs.shape == (2, 32)


# ---------------------------------------------------------------------------
# allow_fake gate
# ---------------------------------------------------------------------------
def test_allow_fake_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_ALLOW_FAKE", raising=False)
    assert allow_fake_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "On"])
def test_allow_fake_truthy(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", val)
    assert allow_fake_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "", "off"])
def test_allow_fake_falsy(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", val)
    assert allow_fake_enabled() is False


# ---------------------------------------------------------------------------
# st provider — offline fail-closed (no real weights needed)
# ---------------------------------------------------------------------------
def test_st_missing_deps_raises_unavailable(monkeypatch):
    """When sentence-transformers is unimportable, raise typed Unavailable."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("simulated missing extras")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    client = build_embedding_client(provider_name="st", offline=True)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["x"])


def test_st_offline_load_failure_raises_unavailable(monkeypatch):
    """A failed SentenceTransformer load (e.g. cache miss offline) → typed error.

    We stub the SentenceTransformer constructor to raise so the test runs
    with zero weights and asserts the fail-closed contract.
    """
    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")

    class _BoomST:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights not found in local cache (simulated)")

    fake_mod.SentenceTransformer = _BoomST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    resolved = resolve_embedding_provider(provider_name="st")
    client = EmbeddingClient(resolved, offline=True)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["x"])


def test_st_offline_sets_hf_hub_offline_and_restores(monkeypatch):
    """offline=True sets HF_HUB_OFFLINE=1 during load and restores after."""
    import sys
    import types

    seen = {}
    fake_mod = types.ModuleType("sentence_transformers")

    class _ProbeST:
        def __init__(self, *args, **kwargs):
            seen["hf_offline"] = os.environ.get("HF_HUB_OFFLINE")
            seen["local_files_only"] = kwargs.get("local_files_only")
            raise RuntimeError("stop here — we only wanted the env snapshot")

    fake_mod.SentenceTransformer = _ProbeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    resolved = resolve_embedding_provider(provider_name="st")
    client = EmbeddingClient(resolved, offline=True)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["x"])
    assert seen["hf_offline"] == "1"
    assert seen["local_files_only"] is True
    # Restored (was unset before).
    assert "HF_HUB_OFFLINE" not in os.environ


# ---------------------------------------------------------------------------
# openai-embeddings — stubbed httpx transport (success / retry / malformed)
# ---------------------------------------------------------------------------
class _StubResp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _StubHttpClient:
    """Records POSTs and replays a queued sequence of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None):  # noqa: A002
        self.calls.append({"url": url, "json": json, "headers": headers})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_openai_embeddings_success():
    body = {"data": [{"embedding": [3.0, 4.0]}, {"embedding": [0.0, 5.0]}]}
    stub = _StubHttpClient([_StubResp(200, body)])
    resolved = resolve_embedding_provider(provider_name="local-openai")
    client = EmbeddingClient(resolved, http_client=stub)
    vecs = client.encode_batch(["a", "b"])
    assert vecs.shape == (2, 2)
    # Rows are L2-normalized: [3,4] → [0.6, 0.8]; [0,5] → [0,1].
    assert np.allclose(vecs[0], [0.6, 0.8], atol=1e-5)
    assert np.allclose(vecs[1], [0.0, 1.0], atol=1e-5)
    # Hits {base_url}/embeddings.
    assert stub.calls[0]["url"].endswith("/embeddings")
    assert stub.calls[0]["headers"]["Authorization"] == "Bearer local"


def test_openai_embeddings_503_then_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real backoff wait
    body = {"data": [{"embedding": [1.0, 0.0]}]}
    stub = _StubHttpClient([_StubResp(503, {}), _StubResp(200, body)])
    resolved = resolve_embedding_provider(provider_name="local-openai")
    client = EmbeddingClient(resolved, http_client=stub)
    vecs = client.encode_batch(["a"])
    assert vecs.shape == (1, 2)
    assert len(stub.calls) == 2  # retried once


def test_openai_embeddings_persistent_503_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    stub = _StubHttpClient([_StubResp(503, {}), _StubResp(503, {}), _StubResp(503, {})])
    resolved = resolve_embedding_provider(provider_name="local-openai")
    client = EmbeddingClient(resolved, http_client=stub)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["a"])


def test_openai_embeddings_malformed_body_raises():
    stub = _StubHttpClient([_StubResp(200, {"unexpected": "shape"})])
    resolved = resolve_embedding_provider(provider_name="local-openai")
    client = EmbeddingClient(resolved, http_client=stub)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["a"])


def test_openai_embeddings_transport_error_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    import httpx

    err = httpx.ConnectError("connection refused")
    stub = _StubHttpClient([err, err, err])
    resolved = resolve_embedding_provider(provider_name="local-openai")
    client = EmbeddingClient(resolved, http_client=stub)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["a"])


def test_openai_embeddings_count_mismatch_raises():
    body = {"data": [{"embedding": [1.0, 0.0]}]}  # only 1 for 2 inputs
    stub = _StubHttpClient([_StubResp(200, body)])
    resolved = resolve_embedding_provider(provider_name="local-openai")
    client = EmbeddingClient(resolved, http_client=stub)
    with pytest.raises(EmbeddingBackendUnavailable):
        client.encode_batch(["a", "b"])


# ---------------------------------------------------------------------------
# Resolved provider is immutable (frozen contract)
# ---------------------------------------------------------------------------
def test_resolved_provider_is_frozen():
    resolved = resolve_embedding_provider(provider_name="fake")
    assert isinstance(resolved, ResolvedEmbeddingProvider)
    with pytest.raises(Exception):
        resolved.model_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Opt-in real-model smoke (cached weights only).
#
# The shared ``real_models`` pytest marker is registered only in
# ``MCP/tests/conftest.py`` + ``Trainforge/tests/conftest.py`` (not for
# ``lib/embedding/tests/``), and ``--strict-markers`` is on, so this suite
# gates the real-model smoke with an env-var opt-in + a cache-presence skip
# instead of the marker. CI (no cache, env unset) always skips → stays green.
# E2's ``LibV2/tools/libv2/tests/`` carries the marker-based ``real_models``
# smoke per the WS2 plan; this is the registry-level equivalent.
# ---------------------------------------------------------------------------
def test_real_minilm_smoke():
    """Smoke the only cached model (all-MiniLM-L6-v2). Skips if uncached/opt-out."""
    if not _env_truthy_test("EMBEDDING_REAL_MODEL_SMOKE"):
        pytest.skip("set EMBEDDING_REAL_MODEL_SMOKE=1 to run the real-model smoke")
    pytest.importorskip("sentence_transformers")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    client = build_embedding_client(
        provider_name="st", model_id=model_id, offline=True
    )
    try:
        vecs = client.encode_batch(["hello world", "a second passage"])
    except EmbeddingBackendUnavailable:
        pytest.skip(f"{model_id} not in HF cache; skipping real-model smoke")
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert client.dim == 384


def _env_truthy_test(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"true", "1", "yes", "on"}
