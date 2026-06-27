"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``dart_semantic.qwen_specialists.*``. The package root
is ``SemantiK/`` (two levels above this file's ``dart_semantic/`` parent),
so we prepend it when not already importable. Mirrors how the lib/semantik
adapter tests reach the vendored tree, but kept local to the vendored
package so the SemantiK tree stays self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# .../SemantiK/dart_semantic/qwen_specialists/tests/conftest.py
#   parents[0]=tests parents[1]=qwen_specialists parents[2]=dart_semantic
#   parents[3]=SemantiK  -> the importable package root for ``dart_semantic``.
_SEMANTIK_ROOT = Path(__file__).resolve().parents[3]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))


@pytest.fixture(autouse=True)
def _isolate_semantik_cache(tmp_path, monkeypatch):
    """Redirect the SemantiK disk cache to a per-test tmp dir.

    The Phase-4b block-review op-cache (``block_review_cache.py``, gated by the
    default-ON ``SEMANTIK_BLOCK_REVIEW_CACHE``) persists each window's parsed
    op-list under ``paths.resolve_cache(...)``. Without isolation, a populated
    real cache (``<repo>/SemantiK/data/block_review_ops/``) suppresses the
    ``generate_batch`` dispatch that the windowed-reviewer tests assert on AND
    pollutes the repo tree. Pointing ``SEMANTIK_CACHE_DIR`` at a fresh per-test
    tmp dir makes every test start cache-cold and write nothing real. Tests
    that need a specific cache dir override the env themselves (later setenv
    wins). Honored by ``paths.resolve_cache`` for the extract / glm-ocr /
    prerender caches too — all benefit from the hermeticity.
    """
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "semantik_cache"))
