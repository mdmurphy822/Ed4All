"""Soft-reranker package — Stage 8 (per-region) + Stage 11 (per-doc).

Both tiers are rule-based v1 — deterministic axes, weighted composite,
top-1 winner. The ``soft_*_reranker_path`` config slots remain ``None``
in v1; they are reserved for future learned upgrades.

Stage 8 — per-region (``region.py``)
    Cross-encoder-style scorer over the K candidates that survived the
    per-region hard gate (Stage 7). Picks the top-1 to feed into the
    deterministic assembler at Stage 9. Mirrors the rule-based pattern
    used by Stage 4 (``cross_reranker.py``).

    Axes (architecture.md:120-126):

        1. diff_from_src       — text-preservation ratio (prose-ish kinds)
        2. heading_fit         — Stage-6 emitted heading level vs. hint
        3. table_semantics     — WCAG-richness of <table> markup
        4. aria_restraint      — penalize redundant ARIA per First Rule
        5. neighbor_consistency — outermost block element vs. prior region

Stage 11 — per-document (``document.py``)
    Whole-document scorer over the assembler output. v1 typically scores
    a single fast-lane :class:`AssembledDoc` and emits per-axis scores
    for Stage 12 to consume; the API takes a sequence so the v1.5
    fast-vs-offline comparison plumbing is ready.

    Axes (architecture.md:185-188):

        1. heading_tree_balance — skipped levels, lopsided depth
        2. landmark_coverage    — <main> required + aux landmarks bonus
        3. ref_link_integrity   — fraction of internal anchors that resolve
        4. outline_cleanliness  — empty / duplicate / near-empty headings
"""
from __future__ import annotations

from .document import rerank_documents, score_document
from .region import DEFAULT_WEIGHTS, rerank_per_region
from .types import DEFAULT_DOC_WEIGHTS, RankedCandidate, ScoredDoc

__all__ = [
    # Stage 8 — per-region
    "DEFAULT_WEIGHTS",
    "RankedCandidate",
    "rerank_per_region",
    # Stage 11 — per-document
    "DEFAULT_DOC_WEIGHTS",
    "ScoredDoc",
    "score_document",
    "rerank_documents",
]
