"""Tests for `lib.embedding.sentence_embedder` (Phase 4 Subtask 9).

Six tests covering:
- :func:`test_encode_returns_unit_vectors` — model wrapping + normalize=True
- :func:`test_cosine_similarity_perfect_match_is_1` — math helper
- :func:`test_cache_hit_returns_cached_vector` — EmbeddingCache.get/put
- :func:`test_cache_persists_across_runs` — JSONL round-trip
- :func:`test_try_load_embedder_returns_none_when_extras_missing` — fallback policy
- :func:`test_strict_mode_raises_when_extras_missing` — Subtask 8 strict mode

Tests that need ``sentence-transformers`` skip when the extras are
missing (``importlib.util.find_spec("sentence_transformers") is None``).
The cache + math + fallback-policy tests run regardless.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


# ---------------------------------------------------------------------------
# Math helper — runs without extras.
# ---------------------------------------------------------------------------


def test_cosine_similarity_perfect_match_is_1() -> None:
    """Identical vectors must return cosine similarity 1.0."""
    from lib.embedding._math import cosine_similarity

    v = [0.5, 0.5, 0.5, 0.5]
    sim = cosine_similarity(v, v)
    assert sim == pytest.approx(1.0, abs=1e-6)

    # Also verify orthogonal vectors return 0.0 and zero-norm short-circuits.
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# EmbeddingCache — runs without extras (numpy is the only soft-dep, the
# cache falls back to plain Python lists when numpy is absent).
# ---------------------------------------------------------------------------


def test_cache_hit_returns_cached_vector(tmp_path: Path) -> None:
    """A subsequent ``get`` for the same text returns the inserted vector."""
    from lib.embedding.sentence_embedder import EmbeddingCache

    cache_path = tmp_path / "embedding_cache.jsonl"
    cache = EmbeddingCache(cache_path=cache_path)

    text = "the quick brown fox jumps over the lazy dog"
    vector = [0.1, 0.2, 0.3, 0.4]
    assert cache.get(text) is None
    cache.put(text, vector)

    cached = cache.get(text)
    assert cached is not None
    # Compare element-wise so the test handles numpy or list equally.
    assert list(cached) == pytest.approx(vector)
    assert text in cache
    assert len(cache) == 1


def test_cache_persists_across_runs(tmp_path: Path) -> None:
    """A second EmbeddingCache instance over the same JSONL re-reads entries."""
    from lib.embedding.sentence_embedder import EmbeddingCache

    cache_path = tmp_path / "embedding_cache.jsonl"

    cache1 = EmbeddingCache(cache_path=cache_path)
    cache1.put("alpha", [1.0, 0.0, 0.0])
    cache1.put("beta", [0.0, 1.0, 0.0])
    assert len(cache1) == 2

    # Verify the JSONL file shape on disk.
    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {r["hash"] for r in rows} == {
        EmbeddingCache._hash_text("alpha"),
        EmbeddingCache._hash_text("beta"),
    }

    # Second instance over the same file must re-read both entries.
    cache2 = EmbeddingCache(cache_path=cache_path)
    assert len(cache2) == 2
    assert cache2.get("alpha") is not None
    assert cache2.get("beta") is not None
    assert list(cache2.get("alpha")) == pytest.approx([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Fallback policy + strict mode — runs without extras.
# ---------------------------------------------------------------------------


def _block_sentence_transformers_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force-fail any future ``import sentence_transformers``.

    Used by the two fallback-policy tests so they're deterministic
    regardless of whether the extras happen to be installed in the
    test environment.
    """
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    class _Blocker:
        def find_spec(self, name, path, target=None):
            if name == "sentence_transformers":
                raise ImportError("blocked for test")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + list(sys.meta_path))


def test_try_load_embedder_returns_none_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default mode: missing extras → ``None`` (graceful degrade)."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    _block_sentence_transformers_import(monkeypatch)

    # Reload the module so the import-probe re-runs against the patched
    # meta_path. Without the reload, a previously-cached import in the
    # test session would short-circuit the probe.
    import lib.embedding.sentence_embedder as m

    importlib.reload(m)
    embedder = m.try_load_embedder()
    assert embedder is None


def test_strict_mode_raises_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode: missing extras → :class:`EmbeddingDepsMissing`."""
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")
    _block_sentence_transformers_import(monkeypatch)

    import lib.embedding.sentence_embedder as m

    importlib.reload(m)
    with pytest.raises(m.EmbeddingDepsMissing) as excinfo:
        m.try_load_embedder()

    # Operator-actionable error message must mention the install hint.
    assert "TRAINFORGE_REQUIRE_EMBEDDINGS" in str(excinfo.value)
    assert "embedding" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Real-model encode test — skipped without extras.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed; install via pip install -e .[embedding]",
)
def test_encode_returns_unit_vectors() -> None:
    """SentenceEmbedder.encode(normalize=True) returns ~unit-length vectors."""
    import math

    from lib.embedding.sentence_embedder import SentenceEmbedder

    embedder = SentenceEmbedder()
    vector = embedder.encode("hello world", normalize=True)

    # Compute L2 norm whether vector is np.ndarray or list[float].
    try:
        import numpy as _np

        norm = float(_np.linalg.norm(vector))
    except ImportError:
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))

    assert norm == pytest.approx(1.0, abs=1e-3)
