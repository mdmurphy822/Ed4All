"""Document-level soft reranker (Stage 11) — compatibility shim.

Thin re-export over :mod:`dart_semantic.soft_reranker.document`, which
owns the real Stage 11 implementation (``rerank_documents`` /
``score_document``). Kept so callers importing the gates-side name keep
working; new code should import from ``soft_reranker`` directly.
"""
from __future__ import annotations

from typing import Any

from ..soft_reranker import rerank_documents


def rerank_document(candidates: list[Any]) -> list[Any]:
    """Reorder DocCandidates by composite score.

    Delegates to :func:`dart_semantic.soft_reranker.rerank_documents`.
    """
    return rerank_documents(candidates)
