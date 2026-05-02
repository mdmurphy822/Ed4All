"""Embedding infrastructure for Phase 4 statistical-tier validators.

The package wraps `sentence-transformers` behind a lazy-load wrapper so
the rest of the codebase can import names from `lib.embedding` whether
or not the heavy ML extras are installed. When the extras are missing,
``try_load_embedder()`` returns ``None`` and downstream validators fall
back to a warning-severity GateIssue (``EMBEDDING_DEPS_MISSING``)
instead of failing closed. Strict-mode opt-in via
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` flips the policy to critical.

Public surface:
- :class:`SentenceEmbedder` — wraps a SentenceTransformer model.
- :func:`try_load_embedder` — returns a ``SentenceEmbedder`` or ``None``.
- :func:`cosine_similarity` — port of the helper in
  ``Trainforge/eval/key_term_precision.py``, np.ndarray-aware.

Phase 4 plan reference: ``plans/phase4_statistical_tier_detailed.md``
Subtasks 5-9. Subtask 6 will add :class:`EmbeddingCache` to this
public surface.
"""
from __future__ import annotations

from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    EmbeddingDepsMissing,
    SentenceEmbedder,
    try_load_embedder,
)

__all__ = [
    "EmbeddingDepsMissing",
    "SentenceEmbedder",
    "cosine_similarity",
    "try_load_embedder",
]
