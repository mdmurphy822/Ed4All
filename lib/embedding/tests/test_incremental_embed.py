"""Builder-C incremental-embedding tests (stub model only — NO GPU / real load).

Covers the SentenceEmbedder surface added for the entailment-gate embed cost
work:

* :func:`resolve_embed_batch_tune` / ``_persist_cache`` / ``_fp16`` — flag
  parse-with-fallback, all default OFF.
* ``encode_batch(batch_size=, length_sort=)`` — batch_size forwarded,
  longest-first packing restores the caller's input order, default call
  byte-identical (no extra kwargs on the wire).
* ``encode_batch_cached`` — disk-persisted, model-folded content addressing:
  input-order vectors, a second call is a pure cache hit (no re-encode), a
  different model_name never serves another model's vector, JSONL persists.
* ``_maybe_half`` — fp16 cast fires only on CUDA + flag on.

All tests inject a fake SentenceTransformer via ``embedder._model`` so no
sentence-transformers install / GPU is touched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

np = pytest.importorskip("numpy")

from lib.embedding.sentence_embedder import (  # noqa: E402
    EmbeddingCache,
    SentenceEmbedder,
    resolve_embed_batch_tune,
    resolve_embed_fp16,
    resolve_embed_persist_cache,
)


class _FakeModel:
    """Records every ``encode`` call; deterministic length-based vectors."""

    def __init__(self, device_type: Optional[str] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        if device_type is not None:
            self.device = type("_Dev", (), {"type": device_type})()
        self.halved = False

    def encode(self, texts, normalize_embeddings=True, batch_size=None):
        single = isinstance(texts, str)
        seq = [texts] if single else list(texts)
        self.calls.append(
            {
                "texts": seq,
                "normalize_embeddings": normalize_embeddings,
                "batch_size": batch_size,
            }
        )
        # First component = text length so input-order restoration is verifiable.
        vecs = np.asarray([[float(len(t)), 1.0] for t in seq], dtype=np.float32)
        return vecs[0] if single else vecs

    def half(self):
        self.halved = True
        return self


def _embedder(model: _FakeModel, **kw) -> SentenceEmbedder:
    emb = SentenceEmbedder(**kw)
    emb._model = model  # bypass the heavy _ensure_model load
    return emb


# --------------------------------------------------------------------------- #
# Flag resolvers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "resolver, env_key",
    [
        (resolve_embed_batch_tune, "ED4ALL_EMBED_BATCH_TUNE"),
        (resolve_embed_persist_cache, "ED4ALL_EMBED_PERSIST_CACHE"),
        (resolve_embed_fp16, "ED4ALL_EMBED_FP16"),
    ],
)
def test_flag_resolvers_parse_with_fallback(resolver, env_key) -> None:
    assert resolver({}) is False
    for truthy in ("1", "true", "TRUE", "yes", "on", " On "):
        assert resolver({env_key: truthy}) is True
    for off in ("0", "false", "no", "off", "garbage", ""):
        assert resolver({env_key: off}) is False


# --------------------------------------------------------------------------- #
# encode_batch — batch_size + length_sort
# --------------------------------------------------------------------------- #


def test_encode_batch_default_is_legacy_signature() -> None:
    model = _FakeModel()
    emb = _embedder(model)
    texts = ["b", "aaaa", "cc"]
    out = emb.encode_batch(texts)
    # No length sort, no batch_size on the wire (byte-identical legacy call).
    assert model.calls[-1]["texts"] == texts
    assert model.calls[-1]["batch_size"] is None
    assert [float(v[0]) for v in out] == [1.0, 4.0, 2.0]


def test_encode_batch_length_sort_restores_input_order() -> None:
    model = _FakeModel()
    emb = _embedder(model)
    texts = ["b", "aaaa", "cc"]
    out = emb.encode_batch(texts, batch_size=256, length_sort=True)
    # Model saw longest-first; batch_size forwarded.
    assert model.calls[-1]["texts"] == ["aaaa", "cc", "b"]
    assert model.calls[-1]["batch_size"] == 256
    # Returned vectors are back in the caller's input order.
    assert [float(v[0]) for v in out] == [1.0, 4.0, 2.0]


def test_encode_batch_length_sort_single_text_noop() -> None:
    model = _FakeModel()
    emb = _embedder(model)
    out = emb.encode_batch(["only"], length_sort=True)
    assert model.calls[-1]["texts"] == ["only"]
    assert [float(v[0]) for v in out] == [4.0]


# --------------------------------------------------------------------------- #
# encode_batch_cached — disk persistence, input order, model folding
# --------------------------------------------------------------------------- #


def test_encode_batch_cached_input_order_and_hit(tmp_path: Path) -> None:
    model = _FakeModel()
    cache = EmbeddingCache(cache_path=tmp_path / "batch.jsonl")
    emb = _embedder(model, batch_cache=cache)

    texts = ["aaa", "bb", "c"]
    v1 = emb.encode_batch_cached(texts)
    assert [float(v[0]) for v in v1] == [3.0, 2.0, 1.0]
    assert len(model.calls) == 1  # one encode for the three misses

    # Second call: every text is a cache hit → no new encode.
    v2 = emb.encode_batch_cached(texts)
    assert len(model.calls) == 1
    assert [float(v[0]) for v in v2] == [3.0, 2.0, 1.0]

    # JSONL persisted one row per distinct text.
    rows = (tmp_path / "batch.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3


def test_encode_batch_cached_dedups_within_call(tmp_path: Path) -> None:
    model = _FakeModel()
    cache = EmbeddingCache(cache_path=tmp_path / "batch.jsonl")
    emb = _embedder(model, batch_cache=cache)

    out = emb.encode_batch_cached(["dup", "dup", "other"])
    # Only the two DISTINCT texts hit the model in the single miss batch.
    assert model.calls[-1]["texts"] == ["dup", "other"]
    assert [float(v[0]) for v in out] == [3.0, 3.0, 5.0]


def test_encode_batch_cached_model_folded_key(tmp_path: Path) -> None:
    shared = tmp_path / "batch.jsonl"
    m1 = _FakeModel()
    e1 = _embedder(m1, model_name="model-A", batch_cache=EmbeddingCache(cache_path=shared))
    e1.encode_batch_cached(["shared text"])
    assert len(m1.calls) == 1

    # A different model over the SAME cache file must NOT reuse model-A's vector.
    m2 = _FakeModel()
    e2 = _embedder(m2, model_name="model-B", batch_cache=EmbeddingCache(cache_path=shared))
    e2.encode_batch_cached(["shared text"])
    assert len(m2.calls) == 1  # cache miss → re-encoded under the folded key


def test_encode_batch_cached_forwards_tune_kwargs(tmp_path: Path) -> None:
    model = _FakeModel()
    cache = EmbeddingCache(cache_path=tmp_path / "batch.jsonl")
    emb = _embedder(model, batch_cache=cache)
    out = emb.encode_batch_cached(
        ["b", "aaaa", "cc"], batch_size=256, length_sort=True
    )
    assert model.calls[-1]["batch_size"] == 256
    assert model.calls[-1]["texts"] == ["aaaa", "cc", "b"]  # longest-first
    assert [float(v[0]) for v in out] == [1.0, 4.0, 2.0]  # restored order


def test_encode_batch_cached_empty_returns_empty(tmp_path: Path) -> None:
    model = _FakeModel()
    emb = _embedder(model, batch_cache=EmbeddingCache(cache_path=tmp_path / "b.jsonl"))
    assert emb.encode_batch_cached([]) == []
    assert model.calls == []


def test_batch_cache_key_folds_model_name() -> None:
    emb = SentenceEmbedder(model_name="foo/bar-v2")
    assert emb._batch_cache_key("x") == "foo/bar-v2\x00x"


def test_ensure_batch_cache_slugifies_model_name(tmp_path: Path, monkeypatch) -> None:
    import lib.embedding.sentence_embedder as m

    monkeypatch.setattr(m, "_DEFAULT_BATCH_CACHE_DIR", tmp_path)
    emb = SentenceEmbedder(model_name="foo/bar v2")
    cache = emb._ensure_batch_cache()
    assert cache.cache_path == tmp_path / "embedding_cache.batch.foo_bar_v2.jsonl"


# --------------------------------------------------------------------------- #
# fp16 — CUDA-only, flag-gated
# --------------------------------------------------------------------------- #


def test_maybe_half_off_by_default() -> None:
    model = _FakeModel(device_type="cuda")
    SentenceEmbedder._maybe_half(model)  # flag unset
    assert model.halved is False


def test_maybe_half_casts_on_cuda_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_EMBED_FP16", "1")
    model = _FakeModel(device_type="cuda")
    SentenceEmbedder._maybe_half(model)
    assert model.halved is True


def test_maybe_half_skips_cpu_when_flag_on(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_EMBED_FP16", "1")
    model = _FakeModel(device_type="cpu")
    SentenceEmbedder._maybe_half(model)
    assert model.halved is False
