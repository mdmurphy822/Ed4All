"""Process-level resident embedding-client cache (W3).

``build_embedding_client`` used to construct a brand-new client on every
call, and the retrieval path calls it once per query — so every query paid a
fresh model load (measured CPU-pinned: ~0.48 s warm, 3.67 s cold, multiplied
by the number of indexed courses on the library-wide path). With CUDA as the
product default that is worse than slow: an unmemoized client allocates and
frees ~1.25 GiB of the same unified pool a co-resident inference seat sizes
its KV cache against, per query.

Zero weights, zero network, zero GPU — every test uses the ``fake`` provider
or a resolved provider that is never loaded. Run CPU-pinned:

    ED4ALL_EMBEDDING_DEVICE=cpu ED4ALL_NLI_DEVICE=cpu \
      pytest lib/embedding/tests/test_embedding_client_cache.py
"""
from __future__ import annotations

import pytest

from lib.embedding.providers import (
    DEFAULT_CLIENT_CACHE_MAX,
    EmbeddingClient,
    _reset_embedding_client_cache_for_tests,
    build_embedding_client,
    embedding_client_cache_stats,
    resolve_client_cache_enabled,
    resolve_client_cache_max,
    resolve_embedding_provider,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    _reset_embedding_client_cache_for_tests()
    yield
    _reset_embedding_client_cache_for_tests()


# ---------------------------------------------------------------------------
# Flag resolution
# ---------------------------------------------------------------------------
def test_cache_enabled_by_default():
    assert resolve_client_cache_enabled({}) is True


@pytest.mark.parametrize("token", ["0", "false", "no", "off", "OFF", " Off "])
def test_cache_explicit_falsey_opts_out(token):
    assert resolve_client_cache_enabled({"ED4ALL_EMBEDDING_CLIENT_CACHE": token}) is False


@pytest.mark.parametrize("token", ["1", "true", "on", "garbage", ""])
def test_cache_anything_else_keeps_the_default(token):
    # Parse-with-fallback: garbage resolves to the documented default (on).
    assert resolve_client_cache_enabled({"ED4ALL_EMBEDDING_CLIENT_CACHE": token}) is True


def test_cache_max_parse_with_fallback():
    assert resolve_client_cache_max({}) == DEFAULT_CLIENT_CACHE_MAX
    assert resolve_client_cache_max({"ED4ALL_EMBEDDING_CLIENT_CACHE_MAX": "5"}) == 5
    for bad in ("0", "-3", "not-an-int", ""):
        assert (
            resolve_client_cache_max({"ED4ALL_EMBEDDING_CLIENT_CACHE_MAX": bad})
            == DEFAULT_CLIENT_CACHE_MAX
        )


# ---------------------------------------------------------------------------
# Identity: same resolution → same object
# ---------------------------------------------------------------------------
def test_identical_resolution_returns_the_same_object(monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_CLIENT_CACHE", raising=False)
    first = build_embedding_client(provider_name="fake")
    second = build_embedding_client(provider_name="fake")
    assert first is second


def test_offline_flag_separates_entries():
    online = build_embedding_client(provider_name="fake", offline=False)
    offline = build_embedding_client(provider_name="fake", offline=True)
    assert online is not offline


def test_device_dtype_model_and_batch_all_separate_entries(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.delenv("ED4ALL_EMBEDDING_DTYPE", raising=False)
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.setenv("ED4ALL_EMBEDDING_CLIENT_CACHE_MAX", "16")
    baseline = build_embedding_client(provider_name="st")

    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cuda")
    assert build_embedding_client(provider_name="st") is not baseline

    monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "bf16")
    bf16 = build_embedding_client(provider_name="st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DTYPE", "fp32")
    assert build_embedding_client(provider_name="st") is not bf16

    monkeypatch.setenv("ED4ALL_EMBEDDING_BATCH_SIZE", "64")
    assert build_embedding_client(provider_name="st") is not baseline

    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)
    # Back to the original resolution → the original object, not a rebuild.
    assert build_embedding_client(provider_name="st") is baseline


def test_env_change_is_never_served_stale(monkeypatch):
    """Resolution runs on EVERY call, so a changed env lands on a different
    entry rather than being answered from a stale one."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    first = build_embedding_client(provider_name="st")
    assert first.resolved.model_id == "BAAI/bge-large-en-v1.5"

    monkeypatch.setenv("ED4ALL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    second = build_embedding_client(provider_name="st")
    assert second is not first
    assert second.resolved.model_id == "BAAI/bge-base-en-v1.5"


# ---------------------------------------------------------------------------
# Opt-out + non-cacheable inputs
# ---------------------------------------------------------------------------
def test_flag_off_restores_construct_per_call(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_CLIENT_CACHE", "0")
    first = build_embedding_client(provider_name="fake")
    second = build_embedding_client(provider_name="fake")
    assert first is not second
    assert embedding_client_cache_stats()["entries"] == 0


def test_injected_http_client_is_never_cached():
    """An injected transport is caller-owned state; sharing it across callers
    would leak one caller's stub into another's client."""
    stub = object()
    first = build_embedding_client(provider_name="local-openai", http_client=stub)
    second = build_embedding_client(provider_name="local-openai", http_client=stub)
    assert first is not second
    assert embedding_client_cache_stats()["entries"] == 0


# ---------------------------------------------------------------------------
# LRU bound
# ---------------------------------------------------------------------------
def test_lru_bound_evicts_least_recently_used(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_CLIENT_CACHE_MAX", "2")
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    a = build_embedding_client(provider_name="st", model_id="model-a")
    b = build_embedding_client(provider_name="st", model_id="model-b")
    assert embedding_client_cache_stats()["entries"] == 2

    # Touch A so B becomes least-recently-used, then insert C.
    assert build_embedding_client(provider_name="st", model_id="model-a") is a
    c = build_embedding_client(provider_name="st", model_id="model-c")
    assert embedding_client_cache_stats()["entries"] == 2
    assert build_embedding_client(provider_name="st", model_id="model-a") is a
    assert build_embedding_client(provider_name="st", model_id="model-c") is c
    # B was evicted → a fresh object.
    assert build_embedding_client(provider_name="st", model_id="model-b") is not b


# ---------------------------------------------------------------------------
# Status reporting — resident bytes are "unknown", never a fabricated 0.
# ---------------------------------------------------------------------------
def test_stats_report_unmeasured_rather_than_zero(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    build_embedding_client(provider_name="st")
    stats = embedding_client_cache_stats()
    assert stats["enabled"] is True
    assert stats["entries"] == 1
    assert stats["measured_entries"] == 0
    assert stats["unmeasured_entries"] == 1
    assert stats["resident_bytes"] is None  # not measured, NOT "0 bytes"
    assert stats["clients"][0]["model_id"] == "BAAI/bge-large-en-v1.5"
    assert stats["clients"][0]["device"] == "cpu"


def test_resident_bytes_sums_parameters_and_buffers(monkeypatch):
    """With a model in hand the estimate is a real byte count."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")

    class _Tensor:
        def __init__(self, count, size):
            self._count = count
            self._size = size

        def numel(self):
            return self._count

        def element_size(self):
            return self._size

    class _Model:
        def parameters(self):
            return [_Tensor(1000, 4), _Tensor(500, 4)]

        def buffers(self):
            return [_Tensor(10, 8)]

    client = EmbeddingClient(resolve_embedding_provider(provider_name="st"))
    assert client.resident_bytes() is None  # nothing loaded yet
    client._st_model = _Model()
    assert client.resident_bytes() == 1000 * 4 + 500 * 4 + 10 * 8


def test_resident_bytes_is_none_for_non_st_kinds():
    assert build_embedding_client(provider_name="fake").resident_bytes() is None


def test_the_bound_is_entries_and_there_is_no_byte_ceiling(monkeypatch):
    """The cap is ENTRIES; nothing here enforces a byte ceiling.

    Documented rather than fixed: the encoder load is deferred to the first
    encode, so an entry's size is unknown at insert time, and eviction only
    drops this dict's reference — ``build_embedding_client`` hands out SHARED
    clients, so an evicted client stays alive as long as any caller holds it.
    A byte cap that cannot free what it evicts would enforce nothing. The
    honest surface is measurement, so the stats report ``max_entries`` (never
    a ``max_bytes``) alongside measured ``resident_bytes``.
    """
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("ED4ALL_EMBEDDING_CLIENT_CACHE_MAX", "1")

    evicted = build_embedding_client(provider_name="st", model_id="model-a")
    build_embedding_client(provider_name="st", model_id="model-b")

    stats = embedding_client_cache_stats()
    assert "max_bytes" not in stats  # no byte ceiling is claimed anywhere
    assert stats["max_entries"] == 1
    assert stats["entries"] == 1

    # The evicted client is still a live, usable object — eviction bounded the
    # dict, it did not release anything the caller still references.
    assert evicted.resolved.model_id == "model-a"
    assert build_embedding_client(provider_name="st", model_id="model-a") is not evicted


# ---------------------------------------------------------------------------
# Exact-tokenizer seam (W11 resolver half)
# ---------------------------------------------------------------------------
def test_token_counter_uses_the_real_tokenizer(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")

    class _Tokenizer:
        def encode(self, text, add_special_tokens=True):
            # Two special tokens around one id per character.
            return [0] + [1] * len(text) + [2]

    class _Model:
        tokenizer = _Tokenizer()

    client = EmbeddingClient(resolve_embedding_provider(provider_name="st"))
    assert client.token_counter() is None  # model not loaded
    client._st_model = _Model()
    counter = client.token_counter()
    assert counter is not None
    assert counter("abcd") == 6


def test_token_counter_none_when_no_usable_tokenizer(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")

    class _Broken:
        class tokenizer:  # noqa: N801 — mimics an ST model attribute
            @staticmethod
            def encode(text, add_special_tokens=True):
                raise RuntimeError("tokenizer API mismatch")

    client = EmbeddingClient(resolve_embedding_provider(provider_name="st"))
    client._st_model = _Broken()
    # Falls back to the DOCUMENTED estimate seam (caller passes nothing), it
    # does not raise and does not invent a count.
    assert client.token_counter() is None
    assert build_embedding_client(provider_name="fake").token_counter() is None
