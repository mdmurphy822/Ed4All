"""Embedding infrastructure for Phase 4 statistical-tier validators.

The package wraps `sentence-transformers` behind a lazy-load wrapper so
the rest of the codebase can import names from `lib.embedding` whether
or not the heavy ML extras are installed. When the extras are missing,
``try_load_embedder()`` returns ``None`` and downstream validators fall
back to a warning-severity GateIssue (``EMBEDDING_DEPS_MISSING``)
instead of failing closed. Strict-mode opt-in via
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` flips the policy to critical.

Distinct from that extras contract: a REQUESTED DEVICE that is unavailable is
a hard failure (:class:`EmbeddingModelUnavailable` on the validator-tier
wrapper, :class:`EmbeddingBackendUnavailable` on the index/query client).
``ED4ALL_EMBEDDING_DEVICE`` defaults to ``cuda`` on both; CPU is an explicit
operator selection and there is no automatic downgrade between them.

Public surface:
- :class:`SentenceEmbedder` — wraps a SentenceTransformer model.
- :class:`EmbeddingCache` — content-addressed LRU on disk (Subtask 6).
- :func:`try_load_embedder` — returns a ``SentenceEmbedder`` or ``None``.
- :func:`cosine_similarity` — port of the helper in
  ``Trainforge/eval/metrics/key_term_precision.py``, np.ndarray-aware.

The embedding-provider registry (``lib.embedding.providers``) adds the
retrieval-index embedding surface: a registry-driven
:class:`EmbeddingClient` (st / openai-embeddings / fake kinds),
:func:`resolve_embedding_provider`, :func:`build_embedding_client`, and
the fail-closed :class:`EmbeddingBackendUnavailable`. Downstream
retrieval consumers import from ``lib.embedding.providers`` directly to
avoid ``__init__`` import-coupling; the re-exports here are for
convenience only.
"""
from __future__ import annotations

from lib.embedding._math import cosine_similarity
from lib.embedding.providers import (
    EmbeddingBackendUnavailable,
    EmbeddingClient,
    ResolvedEmbeddingProvider,
    build_embedding_client,
    resolve_embedding_provider,
)
from lib.embedding.sentence_embedder import (
    EmbeddingCache,
    EmbeddingDepsMissing,
    EmbeddingModelUnavailable,
    SentenceEmbedder,
    is_strict_mode,
    try_load_embedder,
)

__all__ = [
    "EmbeddingCache",
    "EmbeddingDepsMissing",
    "EmbeddingModelUnavailable",
    "SentenceEmbedder",
    "cosine_similarity",
    "is_strict_mode",
    "try_load_embedder",
    # Provider registry (lib.embedding.providers)
    "EmbeddingBackendUnavailable",
    "EmbeddingClient",
    "ResolvedEmbeddingProvider",
    "build_embedding_client",
    "resolve_embedding_provider",
]
