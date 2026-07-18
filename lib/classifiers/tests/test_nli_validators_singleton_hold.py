"""Singleton-hold regression tests (post_rewrite_validation overhaul, feature 4).

The NLI-reload investigation found TWO compounding reload mechanisms in the
slow ``post_rewrite_validation`` phase:

* the NLI ``DeBERTa`` singleton is correct (loads once per process via
  ``NliClassifier.get_or_load``) — these tests PIN that contract so a future
  refactor can't silently break it; and
* ``sentence_embedder.try_load_embedder`` historically constructed a BRAND-NEW
  ``SentenceEmbedder`` on every call, so the MiniLM reloaded several times per
  process across the embedding-tier validators. The overhaul adds a
  process-level singleton cache keyed on ``model_name`` — pinned here.

Both are GPU-free (the NLI check injects a sentinel instance; the embedder
wrapper defers the heavy ``SentenceTransformer`` load to ``encode()``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.classifiers.nli_classifier import NliClassifier  # noqa: E402


# --------------------------------------------------------------------- #
# NLI singleton — loads once per process (do NOT change: investigation).
# --------------------------------------------------------------------- #


def test_nli_get_or_load_returns_same_instance() -> None:
    """A resolved singleton is returned by-identity on every subsequent call."""
    NliClassifier._reset_for_tests()
    try:
        sentinel = object.__new__(NliClassifier)  # bypass __init__ (no torch)
        NliClassifier._INSTANCE = sentinel  # type: ignore[assignment]
        assert NliClassifier.get_or_load() is sentinel
        assert NliClassifier.get_or_load() is sentinel  # no reload
    finally:
        NliClassifier._reset_for_tests()


def test_nli_negative_result_is_cached() -> None:
    """A load failure is cached so get_or_load stays O(1) None (no re-probe)."""
    NliClassifier._reset_for_tests()
    try:
        NliClassifier._LOAD_FAILED = True
        assert NliClassifier.get_or_load() is None
        assert NliClassifier.get_or_load() is None
    finally:
        NliClassifier._reset_for_tests()


# --------------------------------------------------------------------- #
# Embedder singleton — the reload fix (feature 4).
# --------------------------------------------------------------------- #


def test_try_load_embedder_is_process_singleton() -> None:
    import importlib.util

    from lib.embedding import sentence_embedder as se

    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")

    se._reset_embedder_cache_for_tests()
    try:
        a = se.try_load_embedder("all-MiniLM-L6-v2")
        b = se.try_load_embedder("all-MiniLM-L6-v2")
        if a is None:
            pytest.skip("embedding extras unavailable in this env")
        # Same model_name → the SAME wrapper instance (loaded once per process).
        assert a is b
        # A different model_name gets its own cached instance.
        c = se.try_load_embedder("all-mpnet-base-v2")
        assert c is not a
        assert c is se.try_load_embedder("all-mpnet-base-v2")
        # The reset seam clears the cache → a fresh instance after.
        se._reset_embedder_cache_for_tests()
        d = se.try_load_embedder("all-MiniLM-L6-v2")
        assert d is not a
    finally:
        se._reset_embedder_cache_for_tests()
