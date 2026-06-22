"""Stage 11 — document-level soft reranker (rule-based v1).

DART's per-document analogue of Stage 8. Scores one (or more) assembled
documents on the four DP-10.1 axes from ``architecture.md:179-192``:

    composite = 0.30 · heading_tree_balance
              + 0.20 · landmark_coverage
              + 0.30 · ref_link_integrity
              + 0.20 · outline_cleanliness

Tie-break: smaller diff-from-source by character count wins (we use
``len(html)`` as the proxy because the source-text length is invariant
across lanes; the shorter rendering is the one that diverged less).

v1 scope (per architecture.md): ONLY fast-lane vs. offline-lane
multi-assembly. Per-region top-2 fan-out and gap-fill K^M cross-product
are deferred to v2 — combinatorial trap. v1 today typically scores a
single fast-lane doc; the API takes ``Sequence[AssembledDoc]`` so the
moment the offline lane materializes the comparison plumbing is ready.

Per-axis scorers are pure functions of the :class:`AssembledDoc`. Each
returns a float in ``[0, 1]``; 1.0 = best. They never raise; missing
fields degrade gracefully to a neutral score so a half-formed doc still
gets a comparable ranking.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from dart_semantic.assembler.types import AssembledDoc

from .types import DEFAULT_DOC_WEIGHTS, ScoredDoc


# ---------------------------------------------------------------------------
# Per-axis scorers — each returns float in [0, 1]; 1.0 is best.
# ---------------------------------------------------------------------------


def _heading_levels(doc: AssembledDoc) -> list[int]:
    """Extract just the level integers from ``doc.heading_tree``.

    The contracted shape is ``list[tuple[int, str]]`` (per
    ``assembler/types.py``). Defensively handle a legacy shape where
    callers pass plain ints.
    """
    tree = doc.heading_tree or []
    if not tree:
        return []
    first = tree[0]
    if isinstance(first, tuple):
        return [int(t[0]) for t in tree]
    # Legacy / defensive: list of bare ints.
    if isinstance(first, int):
        return [int(t) for t in tree]
    return []


def _heading_texts(doc: AssembledDoc) -> list[str]:
    """Extract just the text strings from ``doc.heading_tree``.

    Returns ``[]`` if the tree is empty or in the legacy bare-int shape
    (no texts available — caller decides what to do).
    """
    tree = doc.heading_tree or []
    if not tree:
        return []
    first = tree[0]
    if isinstance(first, tuple):
        return [str(t[1]) for t in tree]
    return []


def _score_heading_tree_balance(doc: AssembledDoc) -> float:
    """Penalize broken heading hierarchies.

    Deductions from a baseline of 1.0:
      * empty tree but doc has content → 0.5 (neutral; a one-paragraph
        doc with no headings is legitimately heading-less).
      * each skipped level (h1 → h3, h2 → h5, …) → −0.20.
      * lopsided depth: when 5+ headings all live at a single level
        (almost certainly an over-flat outline) → −0.20.
    """
    levels = _heading_levels(doc)
    if not levels:
        return 0.5  # neutral — could legitimately be heading-less.
    score = 1.0
    last = 0
    skips = 0
    for lv in levels:
        if last > 0 and lv > last + 1:
            skips += 1
        last = lv
    score -= 0.2 * skips
    if len(levels) >= 5 and len(set(levels)) == 1:
        score -= 0.2
    return max(0.0, min(1.0, score))


def _score_landmark_coverage(doc: AssembledDoc) -> float:
    """Score by ARIA-landmark coverage.

    Required: at least one ``<main>``. Without ``<main>`` the score is
    0.0 — there is no acceptable document outline without it.

    Auxiliary landmarks that bump the score (each +0.08, capped at 1.0
    with a base of 0.6 for ``<main>``-only): ``address``, ``header``,
    ``footer``, ``nav``, ``aside``.
    """
    landmarks = doc.landmarks or {}
    if landmarks.get("main", 0) < 1:
        return 0.0
    bonus = sum(
        1
        for k in ("address", "header", "footer", "nav", "aside")
        if landmarks.get(k, 0) > 0
    )
    return min(1.0, 0.6 + 0.08 * bonus)


_HREF_HASH_RE = re.compile(r'<a\s+[^>]*href="#([^"]+)"', re.IGNORECASE)
_ID_ATTR_RE = re.compile(r'\bid="([^"]+)"', re.IGNORECASE)


def _score_ref_link_integrity(doc: AssembledDoc) -> float:
    """Ratio of internal anchor links that resolve to an in-doc id.

    1.0 if the doc has zero internal anchors (vacuously satisfied —
    nothing to break). Otherwise: ``resolved / total``.
    """
    html = doc.html or ""
    refs = set(_HREF_HASH_RE.findall(html))
    if not refs:
        return 1.0
    ids = set(_ID_ATTR_RE.findall(html))
    resolved = sum(1 for r in refs if r in ids)
    return resolved / len(refs)


def _score_outline_cleanliness(doc: AssembledDoc) -> float:
    """Penalize empty / near-empty / duplicate heading texts.

    Deductions (each weighted by share of headings affected):
      * empty heading text → −0.40 × share.
      * heading text ≤ 2 chars → −0.10 × share.
      * duplicates (case- and whitespace-folded) → −0.05 × share.

    No headings → 1.0 (vacuous). Legacy bare-int tree shape → 0.7
    (can't measure; mark uncertain).
    """
    tree = doc.heading_tree or []
    if not tree:
        return 1.0
    texts = _heading_texts(doc)
    if not texts:
        return 0.7
    score = 1.0
    n = len(texts)
    empty = sum(1 for t in texts if not (t or "").strip())
    too_short = sum(1 for t in texts if 0 < len((t or "").strip()) <= 2)
    duplicates = n - len({(t or "").strip().lower() for t in texts})
    score -= 0.4 * (empty / n)
    score -= 0.1 * (too_short / n)
    score -= 0.05 * (duplicates / n)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Composite + driver
# ---------------------------------------------------------------------------


def _composite(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted sum of per-axis scores.

    Iterates over ``weights`` so an axis missing from ``scores`` raises
    a clear ``KeyError`` (programmer error) instead of silently scoring
    zero.
    """
    return sum(weights[k] * scores[k] for k in weights)


def score_document(
    doc: AssembledDoc,
    *,
    weights: Mapping[str, float] | None = None,
    lane: str = "fast",
) -> ScoredDoc:
    """Score a single :class:`AssembledDoc` on the four DP-10.1 axes.

    Always returns a :class:`ScoredDoc` — the per-axis scorers never
    raise, so neither does this entry point.

    Parameters
    ----------
    doc
        The assembled document to score.
    weights
        Optional override for the per-axis weights. Defaults to
        :data:`DEFAULT_DOC_WEIGHTS`. Must contain every axis name in
        the default mapping (other axes are ignored).
    lane
        Lane tag to attach to the returned :class:`ScoredDoc`. Must be
        ``"fast"`` (default) or ``"offline"``; v1 emits ``"fast"`` only.
    """
    w = weights or DEFAULT_DOC_WEIGHTS
    scores: dict[str, float] = {
        "heading_tree_balance": _score_heading_tree_balance(doc),
        "landmark_coverage":    _score_landmark_coverage(doc),
        "ref_link_integrity":   _score_ref_link_integrity(doc),
        "outline_cleanliness":  _score_outline_cleanliness(doc),
    }
    return ScoredDoc(
        doc=doc,
        score=_composite(scores, w),
        breakdown=dict(scores),
        lane=lane,
    )


def rerank_documents(
    docs: Sequence[AssembledDoc],
    *,
    weights: Mapping[str, float] | None = None,
    lanes: Sequence[str] | None = None,
) -> list[ScoredDoc]:
    """Rank a sequence of assembled documents (best-first).

    v1 typically gets a single fast-lane doc; the function still accepts
    a sequence so the v1.5 fast-vs-offline comparison plumbing is in
    place. Tie-break is smaller ``len(doc.html)`` (the
    diff-from-source proxy from ``architecture.md:189``).

    Parameters
    ----------
    docs
        Documents to score. Empty input → empty output.
    weights
        Optional override for the per-axis weights.
    lanes
        Optional per-doc lane tags, parallel to ``docs``. When ``None``
        every doc is tagged ``"fast"``.
    """
    if lanes is not None and len(lanes) != len(docs):
        raise ValueError(
            f"lanes length ({len(lanes)}) must match docs length ({len(docs)})"
        )
    scored = [
        score_document(
            d,
            weights=weights,
            lane=(lanes[i] if lanes is not None else "fast"),
        )
        for i, d in enumerate(docs)
    ]
    return sorted(scored, key=lambda s: (-s.score, len(s.doc.html)))


__all__ = [
    "DEFAULT_DOC_WEIGHTS",
    "ScoredDoc",
    "score_document",
    "rerank_documents",
]
