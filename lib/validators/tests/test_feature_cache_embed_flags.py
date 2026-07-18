"""Builder-C wiring: ``BlockFeatureCache.embed`` honours the incremental-embed
flags (``ED4ALL_EMBED_BATCH_TUNE`` / ``ED4ALL_EMBED_PERSIST_CACHE``).

Stub embedder only — asserts WHICH method + kwargs the cache dispatches under
each flag combination. Default (both unset) MUST be the byte-identical legacy
``encode_batch(misses)`` call with no extra kwargs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

np = pytest.importorskip("numpy")

from lib.validators.feature_cache import BlockFeatureCache  # noqa: E402


class _SpyEmbedder:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, List[str], Dict[str, Any]]] = []

    def encode_batch(self, texts, **kwargs):
        self.calls.append(("encode_batch", list(texts), dict(kwargs)))
        return np.asarray([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    def encode_batch_cached(self, texts, **kwargs):
        self.calls.append(("encode_batch_cached", list(texts), dict(kwargs)))
        return [np.asarray([float(len(t)), 1.0], dtype=np.float32) for t in texts]


def _cache(emb: _SpyEmbedder) -> BlockFeatureCache:
    return BlockFeatureCache({}, {}, embedder=emb)


def test_default_uses_legacy_encode_batch_no_kwargs(monkeypatch) -> None:
    monkeypatch.delenv("ED4ALL_EMBED_BATCH_TUNE", raising=False)
    monkeypatch.delenv("ED4ALL_EMBED_PERSIST_CACHE", raising=False)
    emb = _SpyEmbedder()
    _cache(emb).embed(["alpha", "beta"])
    assert len(emb.calls) == 1
    method, texts, kwargs = emb.calls[0]
    assert method == "encode_batch"
    assert texts == ["alpha", "beta"]
    assert kwargs == {}  # byte-identical legacy signature


def test_batch_tune_adds_kwargs_on_encode_batch(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_EMBED_BATCH_TUNE", "1")
    monkeypatch.delenv("ED4ALL_EMBED_PERSIST_CACHE", raising=False)
    emb = _SpyEmbedder()
    _cache(emb).embed(["alpha", "beta"])
    method, _texts, kwargs = emb.calls[0]
    assert method == "encode_batch"
    assert kwargs == {"batch_size": 256, "length_sort": True}


def test_persist_cache_routes_to_encode_batch_cached(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_EMBED_PERSIST_CACHE", "1")
    monkeypatch.delenv("ED4ALL_EMBED_BATCH_TUNE", raising=False)
    emb = _SpyEmbedder()
    _cache(emb).embed(["alpha", "beta"])
    method, _texts, kwargs = emb.calls[0]
    assert method == "encode_batch_cached"
    assert kwargs == {}


def test_both_flags_route_cached_with_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_EMBED_PERSIST_CACHE", "on")
    monkeypatch.setenv("ED4ALL_EMBED_BATCH_TUNE", "true")
    emb = _SpyEmbedder()
    _cache(emb).embed(["alpha", "beta"])
    method, _texts, kwargs = emb.calls[0]
    assert method == "encode_batch_cached"
    assert kwargs == {"batch_size": 256, "length_sort": True}


def test_persist_falls_back_when_embedder_lacks_cached_method(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_EMBED_PERSIST_CACHE", "1")
    monkeypatch.delenv("ED4ALL_EMBED_BATCH_TUNE", raising=False)

    class _OldEmbedder:
        def __init__(self) -> None:
            self.calls: List[List[str]] = []

        def encode_batch(self, texts, **kwargs):
            self.calls.append(list(texts))
            return np.asarray([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    emb = _OldEmbedder()
    BlockFeatureCache({}, {}, embedder=emb).embed(["x"])
    # No encode_batch_cached → gracefully falls back to encode_batch.
    assert emb.calls == [["x"]]
