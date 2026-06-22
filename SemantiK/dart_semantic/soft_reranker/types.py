"""Soft-reranker data contracts.

Two reranker tiers share this module:

* **Stage 8** (per-region) — :class:`RankedCandidate` + :data:`DEFAULT_WEIGHTS`.
* **Stage 11** (per-document) — :class:`ScoredDoc` + :data:`DEFAULT_DOC_WEIGHTS`.

``RankedCandidate`` wraps one Stage-6 :class:`Candidate` with its
composite score and per-axis breakdown so callers (and the smoke
harness) can introspect *why* a candidate won.

``DEFAULT_WEIGHTS`` is the Stage-8 v1 tuning. Sum = 1.0. The values are
intentional, not arbitrary:

    diff_from_src        0.30   text fidelity is the primary signal for
                                prose / heading / list / code / quote.
    heading_fit          0.20   matters only on heading regions but,
                                when it matters, it's load-bearing for
                                downstream Stage 9 hierarchy normalization.
    table_semantics      0.20   WCAG-richness lives in the markup —
                                <caption>, <thead>, <th scope=...>.
    aria_restraint       0.20   First Rule of ARIA: native semantics win.
                                Penalize redundant role= on native elements.
    neighbor_consistency 0.10   sanity check on outermost tag against the
                                prior region's kind. Lowest weight because
                                the heuristic is brittle.

Non-applicable axes return 1.0 (neutral) so they don't drag the
composite down for kinds where the axis can't be measured.

``ScoredDoc`` is the Stage-11 analogue: one :class:`AssembledDoc` with
its composite + per-axis breakdown and a lane tag (``"fast"`` /
``"offline"``). v1 typically scores a single fast-lane doc; the type
is ranking-ready for the v1.5 fast-vs-offline comparison.

``DEFAULT_DOC_WEIGHTS`` is the architecture.md:185-188 tuning for
Stage 11 (DP-10.1 default). Sum = 1.0.

    heading_tree_balance 0.30   skipped levels + lopsided depth break
                                screen-reader navigation.
    landmark_coverage    0.20   <main> required; aux landmarks bonus.
    ref_link_integrity   0.30   broken intra-doc anchors are WCAG 2.4.4
                                failures.
    outline_cleanliness  0.20   empty / duplicate / near-empty headings
                                degrade the document outline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from dart_semantic.qwen_specialists.types import Candidate

if TYPE_CHECKING:  # avoid runtime cycle; assembler.types is leaf-ish but be safe
    from dart_semantic.assembler.types import AssembledDoc


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "diff_from_src":        0.30,
    "heading_fit":          0.20,
    "table_semantics":      0.20,
    "aria_restraint":       0.20,
    "neighbor_consistency": 0.10,
}


DEFAULT_DOC_WEIGHTS: Mapping[str, float] = {
    "heading_tree_balance": 0.30,
    "landmark_coverage":    0.20,
    "ref_link_integrity":   0.30,
    "outline_cleanliness":  0.20,
}


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate with its composite score and per-axis breakdown.

    Attributes
    ----------
    candidate
        The :class:`Candidate` that was scored.
    score
        Composite in ``[0, 1]``; higher is better. ``1.0`` when this
        is the only survivor (singleton — no real ranking happened).
    breakdown
        Per-axis raw scores keyed by axis name. Carries
        ``{"singleton": True}`` instead when the candidate was a
        singleton (no ranking performed).
    """
    candidate: Candidate
    score: float
    breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredDoc:
    """One :class:`AssembledDoc` scored by the Stage-11 doc reranker.

    Parallel of :class:`RankedCandidate` but one tier up — for whole
    assembled documents instead of per-region candidates. ``lane`` lets
    Stage 12 distinguish fast-lane output from offline-lane output when
    the v1.5 multi-assembly comparison lands.

    Attributes
    ----------
    doc
        The :class:`AssembledDoc` that was scored. Held by reference;
        the dataclass is ``frozen`` but the doc itself is mutable.
    score
        Composite in ``[0, 1]``; higher is better.
    breakdown
        Per-axis raw scores keyed by axis name. All four
        :data:`DEFAULT_DOC_WEIGHTS` axes are always present.
    lane
        ``"fast"`` (default) or ``"offline"``. v1 emits ``"fast"``
        only; the field exists to keep Stage 12's contract stable
        when the offline lane lands.
    """
    doc: "AssembledDoc"
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    lane: str = "fast"


__all__ = [
    "DEFAULT_WEIGHTS",
    "DEFAULT_DOC_WEIGHTS",
    "RankedCandidate",
    "ScoredDoc",
]
