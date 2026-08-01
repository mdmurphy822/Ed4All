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

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from lib.embedding.providers import (
    EMBEDDING_MODEL_REGISTRY,
    EmbeddingBackendUnavailable,
    EmbeddingClient,
    ResolvedEmbeddingProvider,
    allow_fake_enabled,
    build_embedding_client,
    normalize_device_token,
    normalize_dtype_token,
    resolve_embedding_device,
    resolve_embedding_provider,
)

#: Every env var that participates in ``st`` resolution. The default-assertion
#: tests clear all of them so they stay hermetic for an operator who has
#: sourced a run-env template — the old default test cleared only provider +
#: model and would already fail on a box with the device pinned.
_ST_RESOLUTION_ENVS = (
    "ED4ALL_EMBEDDING_PROVIDER",
    "ED4ALL_EMBEDDING_MODEL",
    "ED4ALL_EMBEDDING_DEVICE",
    "ED4ALL_EMBEDDING_DTYPE",
    "ED4ALL_EMBEDDING_BATCH_SIZE",
)


@pytest.fixture(autouse=True)
def _isolate_client_cache():
    """Drop the process-level resident client between tests.

    ``build_embedding_client`` memoizes by resolved provider, so without this
    a client built under one test's monkeypatched env could be handed to the
    next test.
    """
    from lib.embedding.providers import _reset_embedding_client_cache_for_tests

    _reset_embedding_client_cache_for_tests()
    yield
    _reset_embedding_client_cache_for_tests()


def _clear_st_envs(monkeypatch):
    for var in _ST_RESOLUTION_ENVS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Registry resolution + env precedence
# ---------------------------------------------------------------------------
def test_default_provider_is_st(monkeypatch):
    _clear_st_envs(monkeypatch)
    resolved = resolve_embedding_provider()
    assert resolved.provider_name == "st"
    assert resolved.kind == "st"
    assert resolved.model_id == "BAAI/bge-large-en-v1.5"
    # CUDA is the product default for index builds AND query encoding; CPU is
    # an explicit operator selection, never an auto-detected fallback.
    assert resolved.device == "cuda"
    assert resolved.dtype == "fp32"
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
    # CPU stays a fully-supported explicit selection.
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    assert resolve_embedding_provider().device == "cpu"


# ---------------------------------------------------------------------------
# Device-token discipline (W10 item 2) — the token used to be handed verbatim
# to SentenceTransformer, so a typo produced an opaque wrapped load error.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        (" cuda ", "cuda"),
        ("CUDA", "cuda"),
        ("cuda:0", "cuda:0"),
        ("Cuda:1", "cuda:1"),
        ("cuda:00", "cuda:0"),
    ],
)
def test_device_token_normalized(raw, expected):
    assert normalize_device_token(raw) == expected


@pytest.mark.parametrize("raw", ["auto", "gpu", "", "cuda:", "cuda:x", "mps", None])
def test_device_token_invalid_raises_naming_the_optout(raw):
    with pytest.raises(ValueError) as exc:
        normalize_device_token(raw)
    message = str(exc.value)
    assert "ED4ALL_EMBEDDING_DEVICE=cpu" in message
    assert "'cuda:N'" in message


def test_auto_is_not_a_recognized_device(monkeypatch):
    """``auto`` must be a config error — auto-detection is silent degradation."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "auto")
    with pytest.raises(ValueError):
        resolve_embedding_provider()


def test_resolve_embedding_device_chain(monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_DEVICE", raising=False)
    assert resolve_embedding_device() == "cuda"  # registry default
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    assert resolve_embedding_device() == "cpu"  # env
    assert resolve_embedding_device("cuda:1") == "cuda:1"  # explicit arg wins


# ---------------------------------------------------------------------------
# Encoder precision seam (W9) — default fp32, CPU + half is a hard failure.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [("fp32", "fp32"), ("BF16", "bf16"), (" fp16 ", "fp16")],
)
def test_dtype_token_normalized(raw, expected):
    assert normalize_dtype_token(raw) == expected


@pytest.mark.parametrize("raw", ["bfloat16", "float32", "half", "", None])
def test_dtype_token_invalid_raises(raw):
    with pytest.raises(ValueError) as exc:
        normalize_dtype_token(raw)
    assert "ED4ALL_EMBEDDING_DTYPE" in str(exc.value)


def test_dtype_default_is_fp32(monkeypatch):
    _clear_st_envs(monkeypatch)
    assert resolve_embedding_provider(provider_name="st").dtype == "fp32"


def test_dtype_env_resolves(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "bf16")
    assert resolve_embedding_provider(provider_name="st").dtype == "bf16"


def test_half_precision_on_cpu_raises_not_ignored(monkeypatch):
    """Non-fp32 + cpu must FAIL — silently encoding fp32 would record a
    precision in the index provenance that the run never used."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "bf16")
    with pytest.raises(EmbeddingBackendUnavailable) as exc:
        resolve_embedding_provider(provider_name="st")
    assert "ED4ALL_EMBEDDING_DTYPE=fp32" in str(exc.value)


def test_dtype_is_st_only(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "bf16")
    # The remote/fake kinds run no local encoder, so the knob never applies.
    assert resolve_embedding_provider(provider_name="fake").dtype == "fp32"
    assert resolve_embedding_provider(provider_name="local-openai").dtype == "fp32"


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


# ---------------------------------------------------------------------------
# W8 (emit half) — the fingerprint carries the three replay parameters.
#
# Before this, ``model_fingerprint`` emitted neither device nor batch_size and
# the client exposed neither attribute, so every consumer fell through to the
# literals ``"cpu"`` / ``1`` and every index on disk recorded a device and a
# batch size the build never used.
# ---------------------------------------------------------------------------
def test_fingerprint_carries_device_batch_and_dtype(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cuda:1")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "bf16")
    monkeypatch.setenv("ED4ALL_EMBEDDING_BATCH_SIZE", "64")
    resolved = resolve_embedding_provider(provider_name="st")
    fp = EmbeddingClient(resolved).model_fingerprint()
    assert fp["device"] == "cuda:1"
    assert fp["dtype"] == "bf16"
    assert fp["batch_size"] == 64
    # Sourced from ``resolved``, never independently recomputed.
    assert fp["device"] == resolved.device
    assert fp["dtype"] == resolved.dtype
    assert fp["batch_size"] == resolved.batch_size


def test_fingerprint_provenance_never_fabricates_a_torch_device():
    """Kinds with no local encoder report the documented server sentinel
    rather than a torch device this process never used."""
    remote = EmbeddingClient(resolve_embedding_provider(provider_name="local-openai"))
    fp = remote.model_fingerprint()
    assert fp["device"] == "server"
    assert fp["dtype"] == "server"

    # ``fake`` genuinely computes in-process in float32 on the CPU.
    local = build_embedding_client(provider_name="fake")
    fp2 = local.model_fingerprint()
    assert fp2["device"] == "cpu"
    assert fp2["dtype"] == "fp32"
    assert fp2["batch_size"] == 16


def test_fingerprint_batch_size_is_never_the_literal_one(monkeypatch):
    """Regression for the fabricated ``or 1`` fall-through: the registry
    default is 16, so an unset batch size must report 16, not 1."""
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    fp = EmbeddingClient(
        resolve_embedding_provider(provider_name="st")
    ).model_fingerprint()
    assert fp["batch_size"] == 16


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


def test_st_cuda_unavailable_raises_and_never_downgrades(monkeypatch):
    """CUDA selected but unavailable → typed unavailable naming the CPU opt-out.

    Fail-without-fix: this is the regression test that was missing, so a
    future "for safety" ``torch.cuda.is_available()`` downgrade branch would
    have passed the whole suite. It asserts BOTH that the raise happens and
    that no client silently ends up on CPU.
    """
    import sys
    import types

    seen = {}
    fake_mod = types.ModuleType("sentence_transformers")

    class _NoCudaST:
        def __init__(self, *args, **kwargs):
            seen["device"] = kwargs.get("device")
            raise RuntimeError(
                "Torch not compiled with CUDA enabled (simulated)"
            )

    fake_mod.SentenceTransformer = _NoCudaST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cuda")

    client = EmbeddingClient(
        resolve_embedding_provider(provider_name="st"), offline=True
    )
    with pytest.raises(EmbeddingBackendUnavailable) as exc:
        client.encode_batch(["x"])
    message = str(exc.value)
    assert "ED4ALL_EMBEDDING_DEVICE=cpu" in message
    assert "device=cuda" in message
    # The load was attempted on the REQUESTED device, and no retry on cpu.
    assert seen["device"] == "cuda"
    assert client._st_model is None


def test_st_dtype_threaded_into_model_kwargs(monkeypatch):
    """A non-fp32 dtype reaches SentenceTransformer as ``model_kwargs`` and
    leaves the default fp32 load byte-identical (no ``model_kwargs`` at all)."""
    import sys
    import types

    calls = []
    fake_mod = types.ModuleType("sentence_transformers")

    class _CaptureST:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("stop after the kwargs snapshot")

    fake_mod.SentenceTransformer = _CaptureST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cuda")

    # Default (fp32): no model_kwargs on the wire.
    monkeypatch.delenv("ED4ALL_EMBEDDING_DTYPE", raising=False)
    with pytest.raises(EmbeddingBackendUnavailable):
        EmbeddingClient(
            resolve_embedding_provider(provider_name="st")
        ).encode_batch(["x"])
    assert "model_kwargs" not in calls[-1]

    # bf16: torch dtype threaded through. The seam also flips TF32 matmul,
    # which is PROCESS-GLOBAL — save and restore it so this test cannot move
    # another test's fp32 numerics.
    torch = pytest.importorskip("torch")
    prior_tf32 = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "bf16")
        with pytest.raises(EmbeddingBackendUnavailable):
            EmbeddingClient(
                resolve_embedding_provider(provider_name="st")
            ).encode_batch(["x"])
        assert calls[-1]["model_kwargs"] == {"torch_dtype": torch.bfloat16}
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32[0]
        torch.backends.cudnn.allow_tf32 = prior_tf32[1]


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
