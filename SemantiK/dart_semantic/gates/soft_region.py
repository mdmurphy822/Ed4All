"""Stage 8 per-region soft reranker — historical import path.

The actual implementation lives in
:mod:`dart_semantic.soft_reranker.region`. This module re-exports the
public surface so callers built against the original
``dart_semantic.gates.soft_region`` import path keep working.

See ``architecture.md:120-126`` for the Stage 8 box and the
``soft_reranker`` package docstring for axis details.
"""
from __future__ import annotations

from dart_semantic.soft_reranker.region import (
    DEFAULT_WEIGHTS,
    rerank_per_region,
)
from dart_semantic.soft_reranker.types import RankedCandidate


__all__ = ["DEFAULT_WEIGHTS", "RankedCandidate", "rerank_per_region"]
