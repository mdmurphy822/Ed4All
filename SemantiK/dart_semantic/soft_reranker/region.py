"""Stage 8 per-region soft reranker — implementation.

Rule-based v1. Five private scoring functions, each returning a float in
``[0, 1]`` (1.0 = best). A composite is the weighted dot product against
``DEFAULT_WEIGHTS``; tie-break prefers the shorter candidate so the
reranker never inflates document size on ties.

Public entry point: :func:`rerank_per_region`. Returns either top-1
:class:`RankedCandidate` per region (default) or the full sorted ranking
when ``return_full_ranking=True``.

Axes
----
1. ``diff_from_src``        — text-preservation ratio for prose-like
   kinds. Math + table return 0.5 (neutral); the structured nature of
   their candidate output makes a flat-text comparison unreliable. The
   hard gate already enforces a ``tau`` floor for prose so this axis
   discriminates among survivors, not against them.

2. ``heading_fit``          — for heading regions: parse the first
   ``<h1..6>`` and compare its level to ``payload['level_hint']``.
   Off-by-zero = 1.0, off-by-one = 0.7, off-by-two = 0.3, else 0.0.
   For non-heading regions: 1.0 (axis is N/A — neutral).

3. ``table_semantics``      — for table regions: WCAG-richness of the
   markup. Sum of small bonuses for ``<caption>``, ``<thead>``,
   ``<th scope=...>``, ``<tbody>``, ``<colgroup>/<col>``. Non-table
   regions: 1.0 (neutral).

4. ``aria_restraint``       — penalize redundant ``role="..."`` on
   native elements (First Rule of ARIA per ``docs/ontology.md``).

5. ``neighbor_consistency`` — sanity check on the outermost block
   element vs. the previous region's kind. Catches cases like a
   paragraph region whose candidate emits ``<li>`` at top level.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Mapping

# Direct module import (not via ``from dart_semantic.gates``) so that
# importing this module does not trigger the gates package __init__ —
# which itself re-exports symbols from this module via soft_region.py
# and would form a circular import.
from dart_semantic.gates.text_preserve import text_preservation_ratio  # noqa: E501
from dart_semantic.qwen_specialists.types import Candidate
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock

from .types import DEFAULT_WEIGHTS, RankedCandidate


# ---------------------------------------------------------------------------
# Per-axis scorers — each returns float in [0, 1]; 1.0 is best.
# ---------------------------------------------------------------------------


_HEADING_TAG_RE = re.compile(r"<h([1-6])\b", re.IGNORECASE)
_ROLE_ATTR_RE = re.compile(r'\brole\s*=\s*"([^"]+)"', re.IGNORECASE)
_ARIA_ATTR_RE = re.compile(r"\baria-\w+\s*=", re.IGNORECASE)

# Roles that duplicate the native semantics of an HTML element with the
# same tag name. First Rule of ARIA: prefer native elements over role=.
_REDUNDANT_ROLES = frozenset({
    "main", "navigation", "banner", "contentinfo",
    "list", "listitem", "table", "figure", "heading",
    "button", "link", "form", "article", "complementary",
})


def _score_diff_from_src(
    candidate: Candidate,
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
) -> float:
    """Text-preservation ratio for prose-like kinds; 0.5 (neutral)
    for math + table where the comparison would be unfair."""
    if region.kind in ("math", "table"):
        return 0.5
    chunks: list[str] = []
    for fb_idx in region.feature_block_indices:
        if 0 <= fb_idx < len(feature_blocks):
            fb = feature_blocks[fb_idx]
            text = getattr(getattr(fb, "raw", None), "text", "") or ""
            text = text.strip()
            if text:
                chunks.append(text)
    if not chunks:
        # No source text to compare against — neutral.
        return 0.5
    source = " ".join(chunks)
    return text_preservation_ratio(source, candidate.text)


def _score_heading_fit(candidate: Candidate, region: Region) -> float:
    """Parse the first ``<h1..6>`` tag and grade vs. ``level_hint``."""
    if region.kind != "heading":
        return 1.0
    expected_raw = (region.payload or {}).get("level_hint")
    try:
        expected = int(expected_raw) if expected_raw is not None else 2
    except (TypeError, ValueError):
        expected = 2
    expected = max(1, min(6, expected))
    m = _HEADING_TAG_RE.search(candidate.text)
    if not m:
        return 0.0
    actual = int(m.group(1))
    delta = abs(actual - expected)
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.7
    if delta == 2:
        return 0.3
    return 0.0


def _score_table_semantics(candidate: Candidate, region: Region) -> float:
    """WCAG-richness of <table> markup. Non-table → 1.0."""
    if region.kind != "table":
        return 1.0
    text_lc = candidate.text.lower()
    score = 0.0
    if "<caption" in text_lc:
        score += 0.30
    if "<thead" in text_lc:
        score += 0.20
    if "scope=" in text_lc:
        score += 0.30
    if "<tbody" in text_lc:
        score += 0.10
    if "<colgroup" in text_lc or "<col " in text_lc:
        score += 0.10
    return min(1.0, score)


def _score_aria_restraint(candidate: Candidate, region: Region) -> float:
    """Penalize redundant ``role="..."`` on native elements.

    Counts matches of ``role="x"`` where ``x`` is in
    ``_REDUNDANT_ROLES``. Score is ``1 / (1 + redundant_count)`` so
    one redundant role drops the score to 0.5, two to 0.33, etc.

    aria-* attributes are *not* directly penalized here — many are
    legitimately required by WCAG 2.2 AA — but their presence is
    surfaced in the breakdown for debugging via the regex.
    """
    text = candidate.text
    redundant = 0
    for m in _ROLE_ATTR_RE.finditer(text):
        role_val = m.group(1).strip().lower()
        if role_val in _REDUNDANT_ROLES:
            redundant += 1
    return 1.0 / (1 + redundant)


_LEADING_TAG_RE = re.compile(r"^\s*<\s*([a-zA-Z][a-zA-Z0-9]*)", re.MULTILINE)


def _score_neighbor_consistency(
    candidate: Candidate,
    region: Region,
    regions: Sequence[Region],
    idx: int,
) -> float:
    """Heuristic check on outermost tag vs. prior region's kind.

    First region: 1.0 (no prior context).

    Penalizes obvious mismatches:
      * candidate starts with ``<li>`` but region is not a list/list_item
      * candidate starts with ``<dt>``/``<dd>`` but region is not a
        definition_list

    Everything else: 1.0. This is intentionally narrow — the heuristic
    is brittle and Stage 9 owns the document-level reading-order check.
    """
    if idx == 0:
        return 1.0
    text_stripped = candidate.text.strip().lower()
    if text_stripped.startswith("<li"):
        if region.kind not in ("list", "list_item"):
            return 0.5
    if text_stripped.startswith("<dt") or text_stripped.startswith("<dd"):
        if region.kind != "definition_list":
            return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# Composite + driver
# ---------------------------------------------------------------------------


def _composite(scores: dict[str, float], weights: Mapping[str, float]) -> float:
    """Weighted sum of per-axis scores."""
    return sum(weights[k] * scores[k] for k in weights)


def rerank_per_region(
    survivors: dict[int, list[Candidate]],
    regions: Sequence[Region],
    feature_blocks: Sequence[FeatureBlock],
    *,
    weights: Mapping[str, float] | None = None,
    return_full_ranking: bool = False,
) -> dict[int, Any]:
    """Score every Stage-7 survivor; return top-1 per region.

    Parameters
    ----------
    survivors
        Stage-7 output — region index → list of surviving candidates.
        ``survivors[i] == []`` is the all-K-fail signal Stage 9 reads.
    regions
        Stage-5 typed regions; indexed by the keys in ``survivors``.
    feature_blocks
        Stage-2 FeatureBlock stream — used for text-preservation
        source text on the ``diff_from_src`` axis.
    weights
        Optional override of per-axis weights. Defaults to
        :data:`DEFAULT_WEIGHTS`.
    return_full_ranking
        When ``True``, the value at each region index is the full sorted
        list of :class:`RankedCandidate` (best-first). When ``False``
        (default), it's the single top-1 :class:`RankedCandidate` (or
        ``None`` for all-K-fail).

    Returns
    -------
    dict
        Region index → :class:`RankedCandidate` (top-1 mode) or
        ``list[RankedCandidate]`` (full-ranking mode).

    Notes
    -----
    * All-K-fail (``survivors[i] == []``) → ``None`` (top-1) or
      ``[]`` (full-ranking).
    * Singleton (one survivor) → that candidate with ``score=1.0`` and
      ``breakdown={"singleton": True}``. No scoring math runs.
    * Identical ``candidate.text`` strings are deduplicated; only the
      first instance is scored. (Stage 6's mock runtime can repeat text;
      so can a tied real-runtime sample.)
    * Tie-break: shorter ``candidate.text`` wins.
    """
    weights = weights or DEFAULT_WEIGHTS
    out: dict[int, Any] = {}

    for region_idx, cand_list in survivors.items():
        if region_idx < 0 or region_idx >= len(regions):
            # Defensive: a region index outside the regions list cannot
            # be scored. Skip rather than crash.
            continue
        region = regions[region_idx]

        # All-K-fail: no candidates to rank.
        if not cand_list:
            out[region_idx] = [] if return_full_ranking else None
            continue

        # Singleton: skip scoring, return as-is with marker breakdown.
        if len(cand_list) == 1:
            rc = RankedCandidate(
                candidate=cand_list[0],
                score=1.0,
                breakdown={"singleton": True},
            )
            out[region_idx] = [rc] if return_full_ranking else rc
            continue

        # Score each unique candidate text.
        seen: dict[str, RankedCandidate] = {}
        for c in cand_list:
            if c.text in seen:
                continue
            scores: dict[str, float] = {
                "diff_from_src":        _score_diff_from_src(c, region, feature_blocks),
                "heading_fit":          _score_heading_fit(c, region),
                "table_semantics":      _score_table_semantics(c, region),
                "aria_restraint":       _score_aria_restraint(c, region),
                "neighbor_consistency": _score_neighbor_consistency(
                    c, region, regions, region_idx,
                ),
            }
            composite = _composite(scores, weights)
            seen[c.text] = RankedCandidate(
                candidate=c,
                score=composite,
                breakdown=dict(scores),
            )

        ranked = sorted(
            seen.values(),
            key=lambda rc: (-rc.score, len(rc.candidate.text)),
        )
        out[region_idx] = ranked if return_full_ranking else ranked[0]

    return out


__all__ = ["DEFAULT_WEIGHTS", "RankedCandidate", "rerank_per_region"]
