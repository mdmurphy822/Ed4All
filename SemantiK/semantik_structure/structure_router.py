"""Genre/confidence-gated STRUCTURE ROUTER (deterministic, CPU-only).

The concrete Pillar-A generalization lever that came out of the task-#25
structure-authority measurement. An oracle A/B (a local, untracked
structure-authority harness under the gitignored ``inputs/`` corpus root;
ground truth = a born-digital PDF's ToC bookmarks) established that
section-structure authority is **DOMAIN-CONDITIONAL**:

* ON the tuning domain (the textbook-scan corpus) the council BERTs WIN section
  recall + chapter-title recovery; the VLM markdown loses.
* On an INACCESSIBLE image-only scan of a NEW GENRE the textbook-tuned BERTs
  COLLAPSE (recall ~0.25 — a couple of real sections buried under ~20
  false-positive apparatus / pedagogical-label headings) while the VLM markdown
  HOLDS (recall ~0.875, section order 1.0, chapter title exact).

So the right design is NOT "retire the BERTs" and NOT "VLM as a mere repair
signal" — it is: **default BERT-authoritative structure; detect off-domain;
switch to VLM-authoritative structure off-domain.** This module is that switch.

It is a **deterministic scorer over signals that already exist** on the built
region list (heading kinds, ``payload['confidence']``, per-FeatureBlock VLM
heading hints, and the co-homed domain-agnostic furniture/apparatus/pedagogical
predicates in ``qwen_specialists.deterministic_structure``). It adds **NO new
LLM call site**, so — per the root no-silent-LLM-call contract — it carries NO
``DecisionCapture`` obligation (flagged loudly here: a pure function).

Contract:

* **Gate ``SEMANTIK_STRUCTURE_ROUTER`` (default OFF).** Off → the detector never
  runs, the switch never fires, the region list is byte-identical.
* **Natural NO-OP without VLM extract+fusion.** The VLM heading structure is
  read from the per-FeatureBlock ``vlm_hint`` (surfaced only when the
  ``SEMANTIK_VLM_EXTRACT`` / ``_FUSION`` / ``_STRUCT_HINTS`` lane ran). With no
  VLM heading hint on any region, ``vlm_present`` is False, the detector returns
  BERT-authoritative, and the switch is a no-op — a dual-gate identical in
  spirit to ``SEMANTIK_VLM_STRUCT_HINTS`` requiring ``SEMANTIK_VLM_EXTRACT``.
* **Order/level only — the wire contract is untouched.** The switch is a
  partition-immutable re-type (kind + heading ``level_hint``): it PROMOTES a
  region the VLM marks a whole-block heading that the council typed as prose,
  and DEMOTES a council heading that is unambiguous apparatus/pedagogical
  furniture the VLM does NOT corroborate. FeatureBlock partitions,
  ``feature_block_indices`` and every FB-derived ``data-semantik-block-id`` /
  ``sourceId`` stay stable. It rides the reviewer's token-conservation invariant
  and **fails closed** (whole-revert to the input list) on any violation,
  mirroring ``clean_structure``.

Thresholds are CALIBRATION constants (marked ``# TODO(calibration)``), NOT
corpus targets — no course slug, publisher name, or corpus-specific gate lives
here (SemantiK wide-net doctrine). Oracle-validation on the real scan corpora
(an off-domain scan vs the on-domain textbook-scan corpus) is the FOLLOW-UP.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

from .structure_graph import Region
from .types import FeatureBlock

# Co-homed domain-agnostic furniture / apparatus / pedagogical predicates + the
# VLM-hint accessor + the promotion mechanic. These are the SoT (block_resegment
# reuses them the same way); the router NEVER re-derives a furniture vocabulary.
from .qwen_specialists.deterministic_structure import (
    _BARE_CHAPTER_SHAPE_RE,
    _detect_repeated_chapter_furniture,
    _is_heading,
    _is_pedagogical_label,
    _is_running_header_region,
    _matches_apparatus_opener,
    _promote_to_heading,
    _region_seed_vlm_hint,
    _region_text,
)
from .qwen_specialists.reviewer import (
    TokenConservationError,
    assert_token_conservation,
)

__all__ = [
    "SEMANTIK_STRUCTURE_ROUTER_ENV",
    "RouterDecision",
    "resolve_structure_router_mode",
    "route_decision",
    "apply_structure_router",
]

# ---------------------------------------------------------------------------
# Flag resolver (SEMANTIK_STRUCTURE_ROUTER) — parse-with-fallback, default OFF.
# ---------------------------------------------------------------------------

SEMANTIK_STRUCTURE_ROUTER_ENV = "SEMANTIK_STRUCTURE_ROUTER"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_structure_router_mode() -> bool:
    """Whether the genre/confidence structure router may run (default OFF).

    Parse-with-fallback truthy set (``1`` / ``true`` / ``yes`` / ``on``,
    case-insensitive); unset / blank / falsey / garbage → off (byte-identical
    pass-through — the detector never runs and the region list is unchanged).
    Mirrors ``structure_graph.resolve_bert_authoritative``. Read at CALL time so
    tests / Docker can flip it without a re-import. Even when ON the router is a
    natural NO-OP without VLM heading hints (see the module docstring dual-gate).
    """
    return (os.environ.get(SEMANTIK_STRUCTURE_ROUTER_ENV) or "").strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Calibration constants — tunable, NOT corpus targets. All # TODO(calibration).
# ---------------------------------------------------------------------------

# Minimum VLM whole-block headings for the VLM structure to be a credible
# authority at all (below this the VLM saw ~no structure → keep BERT).
_MIN_VLM_HEADINGS = 3  # TODO(calibration)
# Apparatus/pedagogical-furniture share of the council heading set at/above which
# the BERT heading set is judged furniture-polluted (the off-domain FP
# signature: the oracle measured ~20 furniture headings burying 2 real ones).
_APPARATUS_RATIO_FLOOR = 0.50  # TODO(calibration)
# Mean council heading confidence below which the heading heads are judged
# low-confidence for this document.
_HEADING_CONF_FLOOR = 0.55  # TODO(calibration)
# Surviving (non-furniture) council section headings per page below which the
# BERT heading recall is judged collapsed for this document.
_RECALL_COLLAPSE_PER_PAGE = 0.15  # TODO(calibration)
# Share of the VLM heading set that the council did NOT also type as a heading,
# at/above which the two structures MATERIALLY diverge (the VLM found sections
# the BERTs missed — the recall-collapse fingerprint).
_DIVERGENCE_FLOOR = 0.50  # TODO(calibration)
# Fallback heading level when a promoted VLM heading carries no usable level.
_DEFAULT_PROMOTE_LEVEL = 2


# ---------------------------------------------------------------------------
# Decision record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterDecision:
    """Deterministic off-domain / structure-authority decision + its signals.

    ``bert_authoritative`` is the load-bearing verdict: True → keep the council
    (BERT) structure (on-domain / no credible VLM); False → the VLM heading
    structure is promoted to authoritative for this document. Every signal is
    reported so the number is auditable (mirrors the oracle's posture).
    """

    vlm_present: bool
    bert_authoritative: bool
    n_pages: int
    n_bert_headings: int
    n_furniture_headings: int
    apparatus_ratio: float
    mean_heading_confidence: float
    section_headings_per_page: float
    n_vlm_headings: int
    n_vlm_only_headings: int
    divergence: float
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Signal helpers (region-native; all reused predicates are domain-agnostic).
# ---------------------------------------------------------------------------


def _n_pages(feature_blocks: list[FeatureBlock]) -> int:
    """Distinct positive physical pages across the FeatureBlocks (>= 1)."""
    pages: set[int] = set()
    for fb in feature_blocks or ():
        raw = getattr(fb, "raw", None)
        try:
            p = int(getattr(raw, "page", 0))
        except (TypeError, ValueError):
            continue
        if p > 0:
            pages.add(p)
    return max(1, len(pages))


def _vlm_heading_hint(region: Region, feature_blocks: list[FeatureBlock]) -> dict | None:
    """The region seed-FB VLM hint IFF it is a WHOLE-BLOCK heading, else None.

    ``whole_block`` is the load-bearing coverage — a ``prefix`` hint (a title
    fused with body prose) must never re-type a prose region into a heading (the
    exact conflation ``mint_vlm_hint`` guards). Reuses the co-homed
    ``_region_seed_vlm_hint`` accessor.
    """
    hint = _region_seed_vlm_hint(region, feature_blocks)
    if (
        isinstance(hint, dict)
        and hint.get("kind") == "heading"
        and hint.get("coverage") == "whole_block"
    ):
        return hint
    return None


def _is_furniture_heading(
    region: Region, index: int, repeated_furniture: set[int]
) -> bool:
    """Whether a council HEADING region is apparatus / pedagogical furniture.

    Union of the co-homed domain-agnostic predicates — NOT a new vocabulary:
    a repeated running-header ``Chapter N`` (``_detect_repeated_chapter_furniture``),
    a pedagogical label (Example / Try It / Solution / Step —
    ``_is_pedagogical_label``, which itself never flags a real numbered
    heading), a whole-match end-of-chapter apparatus opener
    (``_matches_apparatus_opener``), a ``Chapter N … <page>`` running header
    (``_is_running_header_region``), or a bare ``Chapter N`` shape
    (``_BARE_CHAPTER_SHAPE_RE``). Non-heading regions are never furniture here.
    """
    if not _is_heading(region):
        return False
    if index in repeated_furniture:
        return True
    if _is_pedagogical_label(region):
        return True
    if _is_running_header_region(region):
        return True
    text = _region_text(region)
    if not text:
        return False
    if _matches_apparatus_opener(text):
        return True
    if _BARE_CHAPTER_SHAPE_RE.match(text):
        return True
    return False


def _heading_confidence(region: Region) -> float | None:
    """A council heading region's confidence, from ``payload['confidence']``.

    Falls back to the top-1 of ``provenance['role_top_k']`` (the always-on role
    distribution) when the direct confidence key is absent. Returns None when
    neither is present (excluded from the mean rather than counted as 0.0).
    """
    payload = region.payload or {}
    conf = payload.get("confidence")
    try:
        if conf is not None:
            return float(conf)
    except (TypeError, ValueError):
        pass
    top_k = (region.provenance or {}).get("role_top_k")
    if isinstance(top_k, (list, tuple)) and top_k:
        first = top_k[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                return float(first[1])
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Detector.
# ---------------------------------------------------------------------------


def route_decision(
    regions: list[Region], feature_blocks: list[FeatureBlock]
) -> RouterDecision:
    """Score whether the council structure is trustworthy for THIS document.

    Deterministic, pure (no I/O, no LLM). Combines three off-domain signals over
    the built region list + the per-FB VLM heading hints:

      * **BERT recall-collapse proxy** — few surviving (non-furniture) council
        section headings per page, OR a low mean council heading confidence.
      * **High apparatus/furniture ratio** among the council heading set (the
        off-domain false-positive signature).
      * **VLM-vs-BERT heading divergence** — the VLM proposes materially more /
        different whole-block headings than the council surfaced.

    The verdict is VLM-authoritative (``bert_authoritative=False``) IFF the VLM
    is credibly present (``>= _MIN_VLM_HEADINGS`` whole-block headings) AND at
    least one BERT-untrust symptom fires (apparatus ratio / low confidence /
    recall collapse) AND the two structures materially diverge. Any missing
    condition keeps the council authoritative — the switch is a no-op.
    """
    n_pages = _n_pages(feature_blocks)

    repeated_furniture = _detect_repeated_chapter_furniture(regions)

    bert_heading_idxs = [i for i, r in enumerate(regions) if _is_heading(r)]
    n_bert_headings = len(bert_heading_idxs)

    furniture_idxs = {
        i
        for i in bert_heading_idxs
        if _is_furniture_heading(regions[i], i, repeated_furniture)
    }
    n_furniture = len(furniture_idxs)
    apparatus_ratio = (n_furniture / n_bert_headings) if n_bert_headings else 0.0

    section_heading_idxs = [i for i in bert_heading_idxs if i not in furniture_idxs]
    section_per_page = len(section_heading_idxs) / n_pages

    confs = [
        c
        for c in (_heading_confidence(regions[i]) for i in bert_heading_idxs)
        if c is not None
    ]
    mean_conf = (sum(confs) / len(confs)) if confs else 1.0

    # VLM-proposed whole-block headings (region seed-FB hints).
    vlm_heading_idxs = [
        i
        for i, r in enumerate(regions)
        if _vlm_heading_hint(r, feature_blocks) is not None
    ]
    n_vlm = len(vlm_heading_idxs)
    # VLM headings the council did NOT also type as a heading = the recall gap.
    vlm_only = [i for i in vlm_heading_idxs if not _is_heading(regions[i])]
    n_vlm_only = len(vlm_only)
    divergence = (n_vlm_only / n_vlm) if n_vlm else 0.0

    vlm_present = n_vlm > 0

    reasons: list[str] = []
    symptom = False
    if apparatus_ratio >= _APPARATUS_RATIO_FLOOR:
        reasons.append("apparatus_ratio")
        symptom = True
    if mean_conf < _HEADING_CONF_FLOOR:
        reasons.append("low_heading_confidence")
        symptom = True
    if section_per_page < _RECALL_COLLAPSE_PER_PAGE:
        reasons.append("recall_collapse")
        symptom = True
    diverged = divergence >= _DIVERGENCE_FLOOR
    if diverged:
        reasons.append("vlm_divergence")

    off_domain = (
        vlm_present
        and n_vlm >= _MIN_VLM_HEADINGS
        and symptom
        and diverged
    )

    return RouterDecision(
        vlm_present=vlm_present,
        bert_authoritative=not off_domain,
        n_pages=n_pages,
        n_bert_headings=n_bert_headings,
        n_furniture_headings=n_furniture,
        apparatus_ratio=round(apparatus_ratio, 6),
        mean_heading_confidence=round(mean_conf, 6),
        section_headings_per_page=round(section_per_page, 6),
        n_vlm_headings=n_vlm,
        n_vlm_only_headings=n_vlm_only,
        divergence=round(divergence, 6),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# The switch (partition-immutable re-type; fail-closed on token-conservation).
# ---------------------------------------------------------------------------


def _demote_to_paragraph(region: Region) -> Region:
    """Return a copy of a furniture heading demoted heading→paragraph.

    Kind + a ``structure_router`` breadcrumb only; text + the FB partition are
    preserved verbatim (frozen-``replace``), so this is partition-immutable and
    token-conserving (the FB stays in a non-``metadata_drop`` region).
    """
    payload = dict(region.payload or {})
    prov = dict(payload.get("structure_router") or {})
    prov["from_kind"] = str(region.kind)
    prov["demoted"] = "apparatus_furniture"
    payload["structure_router"] = prov
    return dataclasses.replace(region, kind="paragraph", payload=payload)


def _promote_vlm_heading(region: Region, hint: dict) -> Region:
    """Promote a region the VLM marks a whole-block heading (level from the hint).

    Reuses the co-homed ``_promote_to_heading`` mechanic (frozen-``replace`` of
    kind + ``level_hint``, text preserved) with the VLM markdown ``#``-level as
    the level hint (the assembler owns the final ``<hN>``). Stamps a
    ``structure_router`` breadcrumb so the promotion is auditable.
    """
    level = hint.get("level")
    try:
        level_hint = int(level) if level else _DEFAULT_PROMOTE_LEVEL
    except (TypeError, ValueError):
        level_hint = _DEFAULT_PROMOTE_LEVEL
    if level_hint < 1:
        level_hint = _DEFAULT_PROMOTE_LEVEL
    promoted = _promote_to_heading(region, level_hint=level_hint)
    payload = dict(promoted.payload or {})
    prov = dict(payload.get("structure_router") or {})
    prov["from_kind"] = str(region.kind)
    prov["promoted"] = "vlm_heading"
    payload["structure_router"] = prov
    return dataclasses.replace(promoted, payload=payload)


def apply_structure_router(
    regions: list[Region], feature_blocks: list[FeatureBlock]
) -> tuple[list[Region], dict]:
    """Route the structure authority for one document; return (regions, diag).

    Runs :func:`route_decision`; when it verdicts VLM-authoritative (off-domain
    AND a credible VLM structure), promotes each region the VLM marks a
    whole-block heading that the council typed as prose AND demotes each council
    heading that is unambiguous apparatus/pedagogical furniture the VLM does not
    corroborate — a partition-immutable re-type (kind + heading level only).
    Fails closed (whole-revert to the input list) on any token-conservation
    violation, mirroring ``clean_structure``.

    On-domain / no credible VLM (``bert_authoritative``) → the input list is
    returned UNCHANGED (byte-identical), so the caller's gate-off path plus this
    natural no-op are the two-layer safety the campaign wants.
    """
    decision = route_decision(regions, feature_blocks)
    diag: dict = {
        "decision": dataclasses.asdict(decision),
        "promoted": 0,
        "demoted": 0,
        "reverted_for_invariant": False,
    }

    if decision.bert_authoritative:
        return regions, diag

    # Recompute the repeated-running-header furniture set over the SAME region
    # list the detector scored, so the demote decision matches the signal.
    repeated_furniture = _detect_repeated_chapter_furniture(regions)

    corrected: list[Region] = []
    promoted = demoted = 0
    for i, region in enumerate(regions):
        hint = _vlm_heading_hint(region, feature_blocks)
        if (
            hint is not None
            and not _is_heading(region)
            and region.source_region_id is None
        ):
            corrected.append(_promote_vlm_heading(region, hint))
            promoted += 1
            continue
        if (
            _is_heading(region)
            and hint is None
            and _is_furniture_heading(region, i, repeated_furniture)
        ):
            corrected.append(_demote_to_paragraph(region))
            demoted += 1
            continue
        corrected.append(region)

    try:
        assert_token_conservation(regions, corrected, feature_blocks)
    except TokenConservationError:
        diag["reverted_for_invariant"] = True
        return list(regions), diag

    diag["promoted"] = promoted
    diag["demoted"] = demoted
    return corrected, diag
