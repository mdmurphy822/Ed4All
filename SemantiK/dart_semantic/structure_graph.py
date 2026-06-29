"""Stage 5 — deterministic structure-graph candidate generator.

Stage 5 sits between the BERT council (Stages 3-4) and the Qwen
region specialists (Stage 6). It is a **pure function** over the
:class:`CouncilState`, the FeatureBlock stream, the typed
:class:`RegionCandidate`s, and the Stage-4 :class:`RoutingDecision`s.
It emits a flat list of typed :class:`Region` objects which Stage 6
expands into HTML and Stage 9 then assembles into a document.

See ``architecture.md`` lines 84-108 for the pipeline diagram and
lines 743-754 for the legacy mapping (Stage 5 = part of legacy
"ontology_map" stage). This is the deterministic mapping from typed
BERT signals to structural regions; the Qwen specialists in Stage 6
expand each region into HTML, and Stage 9 assembles the document.

Critical boundary
-----------------

Stage 5 emits a **flat list of typed Region objects**. It does NOT:

* normalize heading hierarchy (Stage 9's job, ``architecture.md``
  lines 130-168 — "promote-the-first, demote-forward, never-skip"),
* merge lists across regions (Stage 9's "4-clause continuation"),
* emit any HTML (Stage 6 + Stage 9 own that),
* do reading-order placement (Stage 9 "reading-order DOM placement"),
* emit ARIA landmarks like ``<main>`` / ``<nav>`` (Stage 9 owns those).

Stage 5 ONLY groups consecutive prose FeatureBlocks into typed
regions and passes through table/math regions from Stage 4.

Six deterministic passes
------------------------

The driver maintains a ``claimed: set[int]`` of FeatureBlock indices
already assigned to a region. Each pass writes to the same output
list and adds claimed indices.

1. **Claim region members.** For every Stage-4 RoutingDecision whose
   route is ``"table"`` or ``"math"``, mark the corresponding
   RegionCandidate's ``member_block_indices`` as claimed (so the
   subsequent prose passes skip them).
2. **Heading.** Per FeatureBlock: if Structure's ``is_heading`` head
   says ``"heading"`` with confidence ≥ threshold, emit a single-FB
   heading region with a ``level_hint`` derived from the relative
   font ratio. (Stage 9 owns final h-level normalization.)
3. **Table & Math passthrough.** Emit one ``Region(kind="table")``
   per confirmed table decision (carrying the candidate's cell_grid /
   header_row_indices / bordered fields), one ``Region(kind="math")``
   per confirmed math decision (carrying src_text + glyph features).
   Demoted regions (flagged ``detector_demoted`` or
   ``math_detector_stand_in_demoted``) are skipped here — their
   FBs flow through to the prose passes below.
4. **List.** Maximal runs of consecutive unclaimed FBs whose
   ``structural_role`` top-1 is ``"list_item"`` (≥ 0.4) collapse
   into one ``Region(kind="list")`` carrying per-item depth and
   ordered/unordered hint.
5. **Paragraph.** Maximal runs of consecutive unclaimed FBs whose
   ``structural_role`` top-1 is ``"paragraph"`` (≥ 0.3) collapse,
   gated by MergeOrSplit's ``same_logical_block`` head between each
   adjacent pair (≥ ``same_logical_block_threshold``, default 0.5;
   missing signal merges defensively for v1).
6. **Code / blockquote / metadata-drop / figure / form fallback.**
   Code-block and blockquote runs from Structure's ``structural_role``
   head; single-FB metadata drops when Semantic's ``doc_role`` says
   ``"footer"`` or ``"metadata"`` with high confidence; single-FB
   figures when Stage-4 flagged ``image_block_demoted``; runs of
   ``in_widget=True`` FBs as ``Region(kind="form")``. Any FB still
   unclaimed after pass 6 falls back to a 1-FB paragraph.

Coverage invariant
------------------

After all six passes, every FeatureBlock index must appear in
exactly one Region's ``feature_block_indices``. This is asserted at
the bottom of :func:`build_structure_graph` — a regression here
silently drops content from the document.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

# RELATIVE imports (``.`` = the ``dart_semantic`` package) so the typed
# candidate classes bind to the SAME module object ``features`` /
# ``orchestrator`` build candidates from and ``cross_reranker`` routes
# against. The cascade is importable under both bare ``dart_semantic`` and
# package-prefixed ``SemantiK.dart_semantic``; an ABSOLUTE ``from
# dart_semantic.region_detection import ...`` here would bind the bare
# module even when the cascade was loaded as ``SemantiK.dart_semantic``,
# splitting class identity so Pass-3 / Pass-3a ``isinstance(cand,
# TableCandidate)`` / ``ImageCandidate`` returned False and no table /
# figure region ever formed. See the matching note in ``council/types.py``.
from .region_detection import (
    ImageCandidate,
    MathCandidate,
    RegionCandidate,
    TableCandidate,
)
from .types import FeatureBlock


# ---------------------------------------------------------------------------
# Reading-order fix gate.
# ---------------------------------------------------------------------------

_READING_ORDER_FALSEY = frozenset({"0", "false", "no", "off"})


def resolve_reading_order_fix() -> bool:
    """Return True when the Stage-5 region reading-order re-sort runs.

    Reads ``SEMANTIK_READING_ORDER_FIX``. **Default ON** (unset / blank /
    truthy / garbage -> on): the final stable sort of
    ``build_structure_graph``'s region list into document reading order runs,
    so the assembled HTML + ``region_provenance`` are in reading order. Only
    an EXPLICIT falsey value (``0``/``false``/``no``/``off``, case-insensitive)
    reverts to the legacy segregated pass-order list (the revert lever).
    Default-on parse-with-fallback semantics mirror
    ``deterministic_structure.resolve_structure_clean_mode`` /
    ``toc_frontmatter_detector._flag_enabled``.
    """
    raw = (os.environ.get("SEMANTIK_READING_ORDER_FIX") or "").strip().lower()
    return raw not in _READING_ORDER_FALSEY


# ---------------------------------------------------------------------------
# BERT-authoritative precedence flip (Phase 5).
#
# Default OFF. When SEMANTIK_BERT_AUTHORITATIVE is truthy, the council
# Structure head's ``structural_role`` argmax IS the region kind (authoritative)
# rather than a non-authoritative recommendation that downstream band-aids
# override. The flip consumes the Phase-2 expanded taxonomy (9-role
# ``structural_role`` + the AXIS-2 ``pedagogical_role`` head) and the per-label
# confidence distribution. Byte-identical to the legacy passes when off (the
# post-pass short-circuits before touching a single region).
# ---------------------------------------------------------------------------

_BERT_AUTHORITATIVE_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_bert_authoritative() -> bool:
    """Return True when the council Structure head's role is authoritative.

    Reads ``SEMANTIK_BERT_AUTHORITATIVE``. **Default OFF** (unset / blank /
    falsey / garbage -> off; byte-identical to the legacy recommendation-only
    contract). A truthy value (``1``/``true``/``yes``/``on``, case-insensitive)
    flips the precedence so the ``structural_role`` argmax becomes the region
    kind and the Stage-5d LLM reviewer demotes to a confidence-gated span
    adjuster. Parse-with-fallback, mirroring ``resolve_structure_review_mode``.
    """
    raw = (os.environ.get("SEMANTIK_BERT_AUTHORITATIVE") or "").strip().lower()
    return raw in _BERT_AUTHORITATIVE_TRUTHY


# The role->kind table over the 9-role ``structural_role`` taxonomy (must match
# ``council/structure.ROLE_NAMES``). table / math / figure are NEVER produced
# here — those kinds are Stage-4 passthrough regions with structured payloads
# (cell_grid / glyph features) and are skipped by the post-pass entirely.
_ROLE_TO_REGION_KIND: dict[str, str] = {
    "paragraph": "paragraph",
    "heading": "heading",
    "list_item": "list",
    "form_label": "form",
    "blockquote": "blockquote",
    "code_block": "code_block",
    "definition_term": "definition_list",
    "definition_def": "definition_list",
    "caption": "paragraph",
}

# Region kinds the authoritative re-type MAY rewrite. Prose-built content kinds
# only — passthrough (table/math/figure) regions are skipped via
# ``source_region_id``; metadata_drop / form stay as emitted.
_AUTHORITATIVE_RETYPE_KINDS: frozenset[str] = frozenset(
    {"paragraph", "list", "code_block", "blockquote", "definition_list"}
)

# AXIS-2 pedagogical_role label -> gold accessible semantic_class component.
# Only the unambiguous pedagogical containers map; "none" / "solution" / "step"
# stay unmapped (a body fragment, not its own container) so no semantic_class is
# stamped. Must reference real ``gold_shell_markup._CONTAINER_SPECS`` keys.
_PEDAGOGICAL_ROLE_TO_SEMANTIC_CLASS: dict[str, str] = {
    "example_open": "worked_example",
    "how_to": "worked_example",
    "exercise_open": "exercise",
    "try_it": "exercise",
    "practice": "exercise",
    "be_prepared": "exercise",
    "objectives": "objectives",
}


def _apply_bert_authoritative_kinds(
    regions: list[Region], state: Any
) -> list[Region]:
    """Re-derive each prose region's kind from the council Structure head.

    GATED by :func:`resolve_bert_authoritative` (the caller short-circuits when
    off, so this is byte-identical pass-through when the flag is unset). For
    every non-passthrough content region (``source_region_id is None`` and
    ``kind in _AUTHORITATIVE_RETYPE_KINDS``):

    * the region's representative FeatureBlock (its ``min`` claimed index) is
      looked up in the council ``structural_role`` signal; its argmax label is
      mapped through :data:`_ROLE_TO_REGION_KIND` and becomes the region kind
      (heading membership is NEVER changed here — that stays with the
      deterministic ``is_heading`` pass and the heading-level normalizer, so an
      argmax of ``heading`` leaves the kind untouched);
    * ``payload['role_confidence']`` is stamped from the head's top-1
      confidence (the per-label distribution the gate consumes);
    * when the AXIS-2 ``pedagogical_role`` head is present and its argmax maps
      to a gold component, ``payload['semantic_class']`` is stamped so the
      pedagogical kind routes through ``gold_shell_markup._wrap_semantic_class``
      (NOT the retired ``<p class="…">`` band-aid).

    The FeatureBlock partition (``feature_block_indices``) is immutable, so the
    coverage invariant + reading-order sort are unaffected.
    """
    out: list[Region] = []
    for region in regions:
        if region.source_region_id is not None or (
            str(region.kind) not in _AUTHORITATIVE_RETYPE_KINDS
        ):
            out.append(region)
            continue
        fb_indices = region.feature_block_indices or ()
        if not fb_indices:
            out.append(region)
            continue
        rep_idx = min(fb_indices)
        role_label, role_conf = _top1(
            _get_signal(state, "structure", "structural_role", rep_idx)
        )
        if role_label is None:
            out.append(region)
            continue
        payload = dict(region.payload or {})
        payload["role_confidence"] = role_conf
        new_kind = _ROLE_TO_REGION_KIND.get(role_label, str(region.kind))
        # Heading membership is owned by the deterministic is_heading pass; the
        # authoritative role flip never promotes a content region TO heading.
        if new_kind == "heading":
            new_kind = str(region.kind)
        ped_label, _ = _top1(
            _get_signal(state, "structure", "pedagogical_role", rep_idx)
        )
        if ped_label is not None and ped_label != "none":
            semantic_class = _PEDAGOGICAL_ROLE_TO_SEMANTIC_CLASS.get(ped_label)
            if semantic_class:
                payload["semantic_class"] = semantic_class
        out.append(replace(region, kind=new_kind, payload=payload))
    return out


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

REGION_KINDS = (
    "paragraph",
    "heading",
    "list",
    "definition_list",
    "table",
    "math",
    "code_block",
    "blockquote",
    "figure",
    "form",
    "metadata_drop",
)
RegionKind = Literal[
    "paragraph",
    "heading",
    "list",
    "definition_list",
    "table",
    "math",
    "code_block",
    "blockquote",
    "figure",
    "form",
    "metadata_drop",
]


@dataclass(frozen=True)
class Region:
    """One Stage-5 typed region — the unit Stage 6 expands.

    ``feature_block_indices`` is a tuple of FeatureBlock indices the
    region claims. The Stage-5 coverage invariant guarantees that
    every FB appears in exactly one Region's ``feature_block_indices``.

    ``payload`` carries kind-specific data (``cell_grid`` for tables,
    ``items`` for lists, ``text`` for prose, etc.). Stage 6
    specialists read keys they understand.

    ``aria_hints`` is a tuple of best-effort ARIA hint strings; Stage
    9 owns the actual ARIA wiring on emission.

    ``provenance`` carries arbitrary debug information (which pass
    created the region, which signals fired). Useful for the smoke
    harness but not load-bearing.

    ``source_region_id`` indexes back into the Stage-4 ``regions``
    list for table/math passthrough; ``None`` for prose-built
    regions that have no Stage-4 RegionCandidate ancestor.
    """

    kind: RegionKind
    feature_block_indices: tuple[int, ...]
    payload: dict[str, Any] = field(default_factory=dict)
    aria_hints: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    source_region_id: int | None = None


# ---------------------------------------------------------------------------
# Tunables — kept private; pass-through to ``build_structure_graph`` kwargs.
# ---------------------------------------------------------------------------

_LIST_ITEM_CONFIDENCE_FLOOR = 0.4
_PARAGRAPH_CONFIDENCE_FLOOR = 0.3
_CODE_CONFIDENCE_FLOOR = 0.4
_BLOCKQUOTE_CONFIDENCE_FLOOR = 0.4
# AXIS-1: definition-list collapse floor. A span whose structural_role top-1
# is definition_term / definition_def at/above this floor seeds/extends a
# `definition_list` Region (the previously-dead REGION_KINDS member).
_DEFINITION_CONFIDENCE_FLOOR = 0.4

# Semantic doc_role labels that, when high-confidence, drop the
# corresponding FB into a metadata_drop region (Stage 9 will surface
# them as artifacts / aside / address per docs/ontology.md §7).
_METADATA_DOC_ROLES = ("footer", "metadata")
# Semantic doc_role labels that, when attached to a paragraph,
# annotate the paragraph payload without minting a new region kind.
# Stage 6's prose specialist reads ``payload["doc_role"]`` and
# Stage 9 uses it for landmark wiring.
_PARAGRAPH_DOC_ROLE_OVERRIDES = (
    "abstract",
    "author",
    "copyright",
    "legal",
    "citation",
    "epigraph",
    "footer",
    "metadata",
)

_ORDERED_LIST_RE = re.compile(r"^\s*(\d+\.|\d+\)|[a-z]\)|[ivx]+\.)\s")
_BULLET_LIST_RE = re.compile(r"^\s*[•‣◦▪·*•●◦‣]\s")


def _has_list_marker(text: str) -> bool:
    """True if ``text`` opens with an ordered (1./a)/iv.) or bullet marker."""
    t = text or ""
    return bool(_ORDERED_LIST_RE.match(t) or _BULLET_LIST_RE.match(t))


# Numbered-section heading prefixes (Stage-5 ``_level_hint`` companion).
# Order matters: more-specific patterns must precede the single-letter
# pattern so ``"I. INTRODUCTION"`` resolves as a Roman numeral, not as a
# subsection-style ``"A."`` (the Roman pattern wins because it's tried
# first in :func:`_numbered_section_depth`).
_CHAPTER_RE = re.compile(r"^Chapter\s+\d+\b", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^Appendix\s+[A-Z]\b")
_SECTION_RE = re.compile(r"^Section\s+\d+\b", re.IGNORECASE)
_ROMAN_RE = re.compile(r"^[IVX]{1,4}\.\s")
_SINGLE_LETTER_RE = re.compile(r"^[A-Z]\.\s")
_DECIMAL_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


# ---------------------------------------------------------------------------
# Private helpers — TypedSignal / state lookup
# ---------------------------------------------------------------------------


def _top1(signal: Any) -> tuple[str | None, float]:
    """Return ``(top_label, top_confidence)`` for a TypedSignal-like
    object. Returns ``(None, 0.0)`` if the signal is missing or has
    no labels. Mirrors :func:`cross_reranker._top1`.
    """
    if signal is None:
        return (None, 0.0)
    labels = getattr(signal, "top_k_labels", None) or []
    confs = getattr(signal, "top_k_confidences", None) or []
    if not labels or not confs:
        return (None, 0.0)
    return (labels[0], float(confs[0]))


def _get_signal(
    state: Any,
    bert_name: str,
    head_name: str,
    region_id: int,
) -> Any:
    """Pull one TypedSignal out of CouncilState by (bert, head, region_id).

    Returns ``None`` if any layer is missing. Mirrors
    :func:`cross_reranker._get_signal` but uses ``getattr`` so the
    helper survives non-strict tests where ``state.outputs`` is a
    plain dict.
    """
    outputs = getattr(state, "outputs", None) or {}
    bert_out = outputs.get(bert_name)
    if bert_out is None:
        return None
    for sig in getattr(bert_out, "signals", []) or []:
        if (
            getattr(sig, "head_name", None) == head_name
            and getattr(sig, "region_id", None) == region_id
        ):
            return sig
    return None


def _merge_says_same(state: Any, prev_idx: int, threshold: float) -> bool:
    """True if MergeOrSplit decides 'merge' for the pair (prev_idx, prev_idx+1).

    MergeOrSplit's ``same_logical_block`` region_id k corresponds to the
    adjacent pair (k, k+1). A missing signal merges defensively (v1
    contract). Shared by the heading pass (to overrule is_heading
    false-positives mid-paragraph) and the paragraph pass.
    """
    if prev_idx < 0:
        return False
    sig = _get_signal(state, "merge_or_split", "same_logical_block", prev_idx)
    if sig is None:
        return True
    label, conf = _top1(sig)
    if label == "same":
        return conf >= threshold
    # Top-1 says "not_same" → flip the inequality.
    return (1.0 - conf) >= threshold


def _merge_present_same(state: Any, prev_idx: int, threshold: float) -> bool:
    """Strict variant: True only when MergeOrSplit has a PRESENT signal that
    says 'same' at/above threshold for pair (prev_idx, prev_idx+1).

    Unlike :func:`_merge_says_same`, a MISSING signal returns False — used
    where merging must be evidence-driven (overruling is_heading, bridging
    mislabeled prose), so we never speculatively merge unknown blocks. The
    defensive (missing→merge) variant is for splitting within an
    already-established paragraph run only.
    """
    if prev_idx < 0:
        return False
    sig = _get_signal(state, "merge_or_split", "same_logical_block", prev_idx)
    if sig is None:
        return False
    label, conf = _top1(sig)
    return label == "same" and conf >= threshold


def _merge_present_not_same(state: Any, prev_idx: int, threshold: float) -> bool:
    """Strict variant: True only when MergeOrSplit has a PRESENT signal that
    CONFIDENTLY says 'not_same' (a real split) for pair (prev_idx, prev_idx+1).

    Used where a deterministic text heuristic must NOT overrule an explicit
    merge-head split (the merge head is the authority on boundaries). A
    missing or low-confidence signal returns False (the heuristic may apply).
    """
    if prev_idx < 0:
        return False
    sig = _get_signal(state, "merge_or_split", "same_logical_block", prev_idx)
    if sig is None:
        return False
    label, conf = _top1(sig)
    return label == "not_same" and conf >= threshold


def _is_heading_shaped(state: Any, idx: int, threshold: float) -> bool:
    """True if Structure's is_heading head (calibrated if present) confidently
    marks ``idx`` as a heading at/above ``threshold``. Callers pair this with
    :func:`_plausible_heading` on the block text before treating it as a real
    heading (e.g. to avoid demoting one into a paragraph run)."""
    sig = _get_signal(state, "structure", "is_heading_calibrated", idx)
    if sig is None:
        sig = _get_signal(state, "structure", "is_heading", idx)
    label, conf = _top1(sig)
    return label == "heading" and conf >= threshold


def _is_paragraph_role(state: Any, idx: int) -> bool:
    """True if Structure's ``structural_role`` top-1 is paragraph (≥ floor)."""
    label, conf = _top1(_get_signal(state, "structure", "structural_role", idx))
    return label == "paragraph" and conf >= _PARAGRAPH_CONFIDENCE_FLOOR


def _is_prose_role(state: Any, idx: int) -> bool:
    """True if ``structural_role`` top-1 is a BODY-text role (paragraph or
    list_item). The Structure head scatters paragraph/list_item across the
    lines of one body paragraph, so callers that just need "this is prose, not
    a structural break" must accept either (see the plain-heading FP guard)."""
    label, conf = _top1(_get_signal(state, "structure", "structural_role", idx))
    return label in ("paragraph", "list_item") and conf >= _PARAGRAPH_CONFIDENCE_FLOOR


_RULE_CHARS = frozenset("_-–—=·*‾⎯─━") | {" "}


def _is_horizontal_rule(text: str) -> bool:
    """True if ``text`` is a decorative horizontal rule (only rule glyphs)."""
    t = (text or "").strip()
    return len(t) >= 5 and all(c in _RULE_CHARS for c in t)


def _is_noise_block(text: str) -> bool:
    """True if ``text`` carries no readable content — a decorative rule, or a
    block with no alphanumeric characters at all (a stray '.', box-drawing
    glyph, or OCR speck). Such blocks should be dropped, not emitted as
    paragraphs/headings."""
    t = (text or "").strip()
    if not t:
        return True
    if _is_horizontal_rule(t):
        return True
    return not any(c.isalnum() for c in t)


def _running_header_norm(text: str) -> str:
    """Normalize a margin block for cross-page matching: lowercase, page
    numbers → ``#``, collapse whitespace. So 'Shades of Accessibility 3'
    and '... 7' both map to 'shades of accessibility #'."""
    return re.sub(r"\d+", "#", " ".join((text or "").split()).lower()).strip()


def _detect_running_headers(
    feature_blocks: list[FeatureBlock],
    *,
    band_frac: float = 0.12,
    min_page_frac: float = 0.4,
    max_words: int = 12,
) -> set[int]:
    """Indices of FeatureBlocks that are running headers/footers.

    Deterministic, model-free: a short text block sitting in the top or
    bottom margin band whose page-number-normalized text recurs in the
    SAME band on a large fraction of pages. These are page furniture
    (the doc title + page number repeated every page) — never content —
    yet Structure's is_heading head fires on them and they shatter
    page-spanning paragraphs. Dropping them deterministically is far more
    reliable than hoping the boilerplate head catches every one.
    """
    page_h: dict[int, float] = {}
    pages: set[int] = set()
    for fb in feature_blocks:
        pg = fb.raw.page
        pages.add(pg)
        # Authoritative page height (types.RawBlock.page_height) — NOT the
        # tallest content bbox, which on a short/sparse/footer-absent page is
        # smaller than the real page and pulls the bottom margin band UP into
        # genuine body text (dropping a real last/first line as a "footer").
        ph = float(getattr(fb.raw, "page_height", 0.0) or 0.0)
        page_h[pg] = ph if ph > 0 else max(page_h.get(pg, 0.0), float(fb.raw.bbox[3]))
    n_pages = len(pages)
    if n_pages < 3:
        return set()  # too few pages to establish recurrence

    # key=(zone, normalized_text) -> {pages}, {indices}
    by_key_pages: dict[tuple[str, str], set[int]] = {}
    by_key_idx: dict[tuple[str, str], list[int]] = {}
    for i, fb in enumerate(feature_blocks):
        text = (fb.raw.text or "").strip()
        if not text or len(text.split()) > max_words:
            continue
        H = page_h.get(fb.raw.page) or 1.0
        y0, y1 = float(fb.raw.bbox[1]), float(fb.raw.bbox[3])
        if y0 < band_frac * H:
            zone = "top"
        elif y1 > (1.0 - band_frac) * H:
            zone = "bot"
        else:
            continue
        norm = _running_header_norm(text)
        if not norm:
            continue
        # Require a page-number token (digits → '#' placeholder) in the
        # normalized text. This is the strongest furniture signal and the
        # WCAG-safe gate: a vanished real heading is worse than an
        # over-promotion, so a digit-LESS recurring band block (a section /
        # part banner "Personal Information", "Part II", "Table of Contents",
        # or reader-facing navigation) is KEPT rather than dropped. Both
        # verified running headers ("Shades of Accessibility N", "No. 24-4068
        # … Page N") carry a page number, so this does not regress them.
        if "#" not in norm:
            continue
        # Require ≥2 alphabetic words so a real running header ("shades of
        # accessibility #") matches but short numeric furniture-LOOKING
        # content does NOT — bibliography year fragments ("2002.", "Rep.,
        # 2002.") that ragged-end in the bottom band would otherwise
        # normalize to "#." / "rep., #." and be falsely dropped as footers.
        if sum(1 for w in norm.split() if w.isalpha() and len(w) >= 2) < 2:
            continue
        key = (zone, norm)
        by_key_pages.setdefault(key, set()).add(fb.raw.page)
        by_key_idx.setdefault(key, []).append(i)

    threshold = max(2, round(min_page_frac * n_pages))
    out: set[int] = set()
    for key, pgs in by_key_pages.items():
        if len(pgs) >= threshold:
            out.update(by_key_idx[key])
    return out


# ---------------------------------------------------------------------------
# Private helpers — per-FB derivations
# ---------------------------------------------------------------------------


def _numbered_section_depth(text: str) -> int | None:
    """Detect a numbered-section prefix in ``text`` and return the
    implied h-level. Returns ``None`` if no prefix matches.

    Patterns (first match wins):

    ===========================================  ========
    Pattern                                       h-level
    ===========================================  ========
    ``Chapter N`` (case-insensitive)              1
    ``Appendix X`` (single uppercase letter)      2
    ``Section N`` (case-insensitive)              2
    ``[IVX]{1,4}.`` (Roman numeral)               2
    ``[A-Z].`` (single uppercase letter)          3
    ``N(.N)*[.]?`` (dotted decimal)               depth+1, capped at 6
    ===========================================  ========

    Decimal numbering counts dot-separated parts: ``"1"`` → depth 1
    → h2, ``"1.1"`` → depth 2 → h3, ``"3.1.2"`` → depth 3 → h4.

    Rationale (companion to :func:`_level_hint`): academic papers
    that use small-caps body-font for headings (e.g. ``"I. INTRODUCTION"``)
    starve the font-ratio bucketing of signal. The numbered-section
    prefix is the only structural clue available in those cases.
    """
    if not text:
        return None
    s = text.lstrip()
    if not s:
        return None
    if _CHAPTER_RE.match(s):
        return 1
    if _APPENDIX_RE.match(s):
        return 2
    if _SECTION_RE.match(s):
        return 2
    if _ROMAN_RE.match(s):
        return 2
    if _SINGLE_LETTER_RE.match(s):
        return 3
    m = _DECIMAL_RE.match(s)
    if m:
        depth = m.group(1).count(".") + 1
        return min(depth + 1, 6)
    return None


def _level_hint(fb: FeatureBlock) -> int:
    """Best-effort heading-level hint from font ratio + numbered-section
    prefix.

    Stage 9 owns final h-level normalization; this is just a hint so
    Stage 6 can size text consistently within a region. Larger
    relative font → smaller hint integer.

    Font-ratio mapping (chosen so a "title-ish" 2× font lands at h1
    and body text at h6):
        ratio ≥ 1.8  → 1
        ratio ≥ 1.5  → 2
        ratio ≥ 1.3  → 3
        ratio ≥ 1.15 → 4
        ratio ≥ 1.05 → 5
        else         → 6

    Combination rule: if the heading text carries a numbered-section
    prefix (see :func:`_numbered_section_depth`), return the
    ``min(numbered_level, font_level)``. The smaller integer wins
    because both signals are independently meaningful — a heading
    marked ``"I. INTRODUCTION"`` is structurally h2 even if its font
    is body-sized; conversely a heading with a 2× font but no
    numbered prefix stays h1.
    """
    ratio = float(getattr(fb, "relative_font_ratio", 1.0) or 1.0)
    if ratio >= 1.8:
        font_level = 1
    elif ratio >= 1.5:
        font_level = 2
    elif ratio >= 1.3:
        font_level = 3
    elif ratio >= 1.15:
        font_level = 4
    elif ratio >= 1.05:
        font_level = 5
    else:
        font_level = 6

    text = getattr(getattr(fb, "raw", None), "text", "") or ""
    numbered = _numbered_section_depth(text)
    if numbered is not None:
        return min(numbered, font_level)
    return font_level


# Equation-crumb glyphs the Structure ``is_heading`` head over-fires on
# (math-heavy papers): operators, relations, and lone Greek letters.
_MATH_GLYPHS = frozenset(
    "∑∫√∞∂∇±×÷≤≥≠≈≡→←↔∈∉⊂⊆⊃⊇∪∩∧∨¬∀∃∅⊥∥⟨⟩∗⋅∝∼≅αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
)
_HEADING_MAX_WORDS = 12


def _plausible_heading(text: str) -> bool:
    """Reject blocks the Structure ``is_heading`` head over-fires on.

    Stage 1 of ``Plans/11_heading_overdetection.md``. A heading region is
    emitted ONLY for text that is heading-shaped; the head is a
    NON-AUTHORITATIVE recommendation (see ``council/structure.py``) and on
    math-heavy papers fires on empty blocks, stray glyphs, equation
    crumbs, full body sentences, and space-stripped extraction artefacts
    (e.g. ``"frameworkforderivingvariousEPIs"``). Rejected blocks are NOT
    claimed, so they fall through to the prose passes (ultimately the
    1-FB paragraph fallback) — no FeatureBlock is lost.

    Conservative by construction so real headings survive: ``:`` endings
    (``"Abstract:"`` / ``"Keywords:"``) are kept, the word cap is generous,
    and a real numbered-section prefix exempts the no-space rule.
    """
    t = (text or "").strip()
    if not t:
        return False  # empty / whitespace
    if len(t) <= 2:
        return False  # stray char / number
    if t.endswith("-"):
        return False  # line-break hyphenation ("…of ac-") → paragraph
        # continuation, never a heading. Catches the is_heading FP on the
        # first wrapped line of a body paragraph.
    if t[0].isalpha() and t[0].islower():
        # Starts lowercase: a wrapped body line ("modation that…",
        # "broadened the…", "versally available…") continues in lowercase,
        # whereas a genuine lowercase-start heading ("via Normal Transport
        # and Rotation") has a Title-Cased remainder. Reject only when the
        # SECOND word is also lowercase.
        _parts = t.split()
        _second = _parts[1] if len(_parts) > 1 else ""
        if not _second or _second[0].islower():
            return False
    words = t.split()
    if len(words) > _HEADING_MAX_WORDS:
        return False  # full body sentence
    if t.endswith(","):
        return False  # a comma-terminated line is a list/caption item that
        # continues onto the next line (e.g. a case-caption party name
        # "R. OPERATIONS, LTD," whose "R." trips the lettered-heading pattern),
        # never a standalone heading. Unconditional: no real heading ends in a
        # comma, and this keeps the party list as one paragraph instead of
        # splitting it and orphaning the continuation as a fragment.
    # Sentence fragment / body crumb: terminal punctuation on a single
    # token ("derivenewones.", "distributed.") or on several words is body
    # text, not a heading. ``:`` is intentionally excluded — label headings
    # legitimately end in a colon ("Abstract:" / "Keywords:").
    if t[-1] in ".,;" and (len(words) == 1 or len(words) > 4):
        return False
    # Equation crumb: a CID extraction artefact.
    if "(cid:" in t:
        return False
    # Tiny-token dominance: a run of mostly ≤2-char tokens is an equation
    # crumb ("r i r i", "i = 1", "1 1", "h ( X ) ="), not a heading. Uses
    # a strict majority so genuine short headings like "3. An Information
    # Inequality" (exactly half tiny) survive. Math glyphs lower the bar.
    if len(words) >= 2:
        tiny = sum(1 for w in words if len(w) <= 2)
        has_glyph = any(ch in _MATH_GLYPHS for ch in t)
        if tiny > len(words) / 2 or (has_glyph and tiny >= len(words) / 2):
            return False
    # Concatenation artefact: a single ≥26-char token means inter-word
    # spaces were lost in extraction ("frameworkforderiving…",
    # "InformationTheoreticProofsofEntropyPowerInequalities"). Real
    # headings don't carry tokens that long.
    if any(len(w) > 25 for w in words):
        return False
    # Space-stripped single run with no spaces at all.
    if " " not in t and len(t) > 25 and _numbered_section_depth(t) is None:
        return False
    return True


def _is_ordered_list(text: str) -> bool:
    """Detect ordered-list marker prefixes (``1.``, ``a)``, ``iv.``, ...).

    Used as the ``payload["ordered"]`` hint on list regions. Stage 6's
    prose specialist may override per-region; Stage 9 owns final
    list-marker normalization.
    """
    if not text:
        return False
    return bool(_ORDERED_LIST_RE.match(text))


def _list_depth(state: Any, fb_index: int) -> int:
    """Read Structure's ``list_nesting`` top-1 → integer depth.

    Mapping mirrors
    :data:`dart_semantic.council.structure.LIST_NESTING_LABELS`:
    ``depth_0`` → 0, ``depth_1`` → 1, ``depth_2`` → 2,
    ``depth_3plus`` → 3. Default 0 if the signal is missing or
    unparseable.
    """
    sig = _get_signal(state, "structure", "list_nesting", fb_index)
    label, _ = _top1(sig)
    if not label:
        return 0
    if label.startswith("depth_"):
        suffix = label[len("depth_") :]
        if suffix == "3plus":
            return 3
        try:
            return int(suffix)
        except ValueError:
            return 0
    return 0


def _doc_role_override(
    state: Any,
    fb_index: int,
    threshold: float,
) -> str | None:
    """Return Semantic's ``doc_role`` top-1 label IF its confidence
    crosses ``threshold`` AND the label is in
    :data:`_PARAGRAPH_DOC_ROLE_OVERRIDES`. Otherwise ``None``.

    The Phase-3c trained Semantic head emits the labels: ``title``,
    ``author``, ``body``, ``citation``, ``footer``, ``legal``,
    ``metadata`` (``abstract`` and ``epigraph`` were dropped at <0.2%
    prevalence). The override list is the union of those labels with
    legacy ontology-driven prose-role labels (``abstract``,
    ``copyright``, ``epigraph``) so the helper still returns a useful
    annotation when synthetic test data — or a future Semantic
    retrain — emits them.
    """
    sig = _get_signal(state, "semantic", "doc_role", fb_index)
    label, conf = _top1(sig)
    if label is None or conf < threshold:
        return None
    if label in _PARAGRAPH_DOC_ROLE_OVERRIDES:
        return label
    return None


def _find_caption_neighbor(
    tc: TableCandidate,
    feature_blocks: list[FeatureBlock],
    state: Any,
    *,
    doc_role_threshold: float = 0.5,
) -> int | None:
    """Best-effort caption pick for a table.

    Scans the FBs immediately before the FIRST member block index and
    immediately after the LAST member block index. Picks one if its
    Semantic ``doc_role`` is ``"caption"`` (high-confidence) OR its
    text begins with ``"Table N"`` / ``"Figure N"``.

    Returns the FeatureBlock index, or ``None`` if no neighbor
    qualifies. Caption detection is intentionally approximate; Stage
    6's table specialist will validate.
    """
    members = sorted(int(i) for i in (tc.member_block_indices or []))
    if not members:
        return None
    member_set = set(members)
    first, last = members[0], members[-1]

    # WCAG 1.3.1 table-caption resolution. A "Table N" label is frequently
    # separated from the table body by a blank / spacer / page-furniture FB
    # (header band, rule line), so the immediate ``first-1`` / ``last+1``
    # neighbour misses it and the table ships with no accessible name. Scan a
    # small window before ``first`` and after ``last`` (closest-out) for an
    # explicit "Table N" / "Figure N" label — the canonical, NON-fabricated
    # accessible name. The wider window only ever LINKS an existing label FB;
    # it never invents caption text (anti-fabrication).
    window = 3
    scan: list[int] = []
    for d in range(1, window + 1):
        scan.append(first - d)
        scan.append(last + d)
    label_re = re.compile(r"^(Table|Figure)\s+\d", flags=re.IGNORECASE)
    for idx in scan:
        if idx < 0 or idx >= len(feature_blocks) or idx in member_set:
            continue
        text = (getattr(getattr(feature_blocks[idx], "raw", None), "text", "") or "").strip()
        if label_re.match(text):
            return idx

    # Secondary signal — a high-confidence Semantic ``doc_role == "caption"``
    # on the IMMEDIATE neighbour (kept narrow: a model "caption" verdict three
    # FBs away is too weak to trust without an explicit label).
    for idx in (first - 1, last + 1):
        if idx < 0 or idx >= len(feature_blocks) or idx in member_set:
            continue
        sig = _get_signal(state, "semantic", "doc_role", idx)
        label, conf = _top1(sig)
        if label == "caption" and conf >= doc_role_threshold:
            return idx
    return None


def _find_figure_caption_below(
    image_fb_idx: int,
    feature_blocks: list[FeatureBlock],
    *,
    max_lookahead: int = 4,
) -> int | None:
    """Nearest ``"Figure N..."`` text FB below the image on the SAME page.

    Part F — scans up to ``max_lookahead`` FBs after the image FB on the
    same physical page; returns the first whose text opens with
    ``"Figure N"`` (the conventional below-image caption). Returns
    ``None`` if none qualifies (the figure ships with the type-level alt).
    The caption FB is NOT claimed here — it stays a normal paragraph so
    its prose is not lost; the figure region only RECORDS its index for
    the assembler to render a ``<figcaption>``.
    """
    if image_fb_idx < 0 or image_fb_idx >= len(feature_blocks):
        return None
    img_raw = getattr(feature_blocks[image_fb_idx], "raw", None)
    img_page = int(getattr(img_raw, "page", -1)) if img_raw is not None else -1
    seen = 0
    j = image_fb_idx + 1
    while j < len(feature_blocks) and seen < max_lookahead:
        raw = getattr(feature_blocks[j], "raw", None)
        page = int(getattr(raw, "page", -2)) if raw is not None else -2
        if page != img_page:
            break
        text = (getattr(raw, "text", "") or "").strip()
        if re.match(r"^Figure\s+\d", text, flags=re.IGNORECASE):
            return j
        seen += 1
        j += 1
    return None


def _iter_runs(
    predicate: Callable[[int], bool],
    feature_blocks: list[FeatureBlock],
    claimed: set[int],
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end_exclusive)`` half-open intervals for
    maximal runs of consecutive unclaimed FBs satisfying ``predicate``.

    ``predicate`` is called with the FB index — it must NOT consult
    ``claimed`` itself; this helper handles that.
    """
    n = len(feature_blocks)
    i = 0
    while i < n:
        if i in claimed or not predicate(i):
            i += 1
            continue
        j = i + 1
        while j < n and j not in claimed and predicate(j):
            j += 1
        yield (i, j)
        i = j


def _resolve_cell_roles(
    state: Any,
    regions: list[RegionCandidate],
) -> dict[int, list[list[str] | list[str]] | None]:
    """Bucket BERT-TableSpecialist ``cell_role_scope`` signals back
    onto each :class:`TableCandidate` as a 2D label grid.

    The orchestrator (``council/orchestrator.py``) aggregates cells
    across every ``TableCandidate`` into one flat list and invokes
    ``run_bert("table_specialist", all_cells)``. The returned signals
    carry ``region_id`` = index in that flat list. To map back we walk
    candidates in the **same order** the orchestrator did — i.e.
    iterate ``regions`` and pick out ``TableCandidate`` instances —
    and use :func:`~dart_semantic.council.table_cell_builder.build_cells`
    to know each candidate's cell footprint.

    Returned dict keys are indices into ``regions`` (the same indices
    Stage 5 hands to Stage 4 / uses internally). The value is a
    row-major 2D list of label strings parallel to
    ``TableCandidate.cell_grid`` (with ragged rows padded — see
    :func:`build_cells`); or ``None`` for a region that:

    * isn't a ``TableCandidate`` (math regions never get cell roles),
    * is a too-small TableCandidate that ``build_cells`` skipped, or
    * has no signals because the BERT-TableSpecialist didn't run.
    """
    out: dict[int, list[list[str] | list[str]] | None] = {}
    outputs = getattr(state, "outputs", None) or {}
    bert_out = outputs.get("table_specialist")
    if bert_out is None:
        # BERT didn't run — every TableCandidate gets None so Stage 6
        # falls back to "data" for every cell.
        for region_idx, cand in enumerate(regions):
            if isinstance(cand, TableCandidate):
                out[region_idx] = None
        return out

    # Index signals by region_id (= flat cell index) for O(1) lookup.
    sig_by_id: dict[int, Any] = {}
    for sig in getattr(bert_out, "signals", []) or []:
        if getattr(sig, "head_name", None) != "cell_role_scope":
            continue
        sig_by_id[int(getattr(sig, "region_id", -1))] = sig

    # Lazy import — avoid pulling council.table_cell_builder at module
    # import time (the council depends on heavy transformers stuff).
    from dart_semantic.council.table_cell_builder import build_cells

    cursor = 0  # walks the orchestrator's flat-list index space
    for region_idx, cand in enumerate(regions):
        if not isinstance(cand, TableCandidate):
            continue
        cells = build_cells(cand, table_idx=region_idx)
        if not cells:
            # Size filter rejected this candidate; orchestrator would
            # also have skipped it. No cells consumed from cursor.
            out[region_idx] = None
            continue
        # Recover (n_rows, n_cols) the same way build_cells did, so
        # padding here matches the flat-list ordering.
        cell_grid: list[list[Any]] = list(cand.cell_grid or [])
        n_rows = len(cell_grid)
        n_cols = max((len(r) for r in cell_grid), default=0)
        labels: list[list[str]] = []
        for i in range(n_rows):
            row: list[str] = []
            for j in range(n_cols):
                flat_idx = cursor + i * n_cols + j
                sig = sig_by_id.get(flat_idx)
                top, _ = _top1(sig)
                # Default unknown / missing top-1 to "data" — same
                # default Stage 6's prompt builder applies anyway.
                row.append(top or "data")
            labels.append(row)
        out[region_idx] = labels
        cursor += n_rows * n_cols
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_structure_graph(
    state: Any,
    feature_blocks: list[FeatureBlock],
    regions: list[RegionCandidate],
    decisions: list[Any],
    *,
    same_logical_block_threshold: float = 0.5,
    is_heading_threshold: float = 0.8,
    doc_role_threshold: float = 0.5,
    pdf_path: Any = None,
) -> list[Region]:
    """Run the six deterministic passes described in the module
    docstring. Returns a flat list of :class:`Region` covering every
    FeatureBlock exactly once.

    Parameters
    ----------
    state
        :class:`CouncilState` from the Stage-3 council.
    feature_blocks
        The Stage-2 FeatureBlock stream that the council operated on.
    regions
        The Stage-2 RegionCandidate list (table + math). Indices must
        align with ``decisions`` 1:1.
    decisions
        Stage-4 RoutingDecisions, one per ``regions`` entry. ``decisions[i]``
        applies to ``regions[i]``.
    same_logical_block_threshold
        MergeOrSplit ``same_logical_block`` confidence floor for
        merging adjacent paragraph FBs. Default 0.5. Missing signal
        is treated as "merge" (defensive in v1).
    is_heading_threshold
        Structure ``is_heading`` confidence floor (on the CALIBRATED
        probability scale — see ``council/structure.py`` temperature
        plumbing) for emitting a heading region. Default 0.8. History:
        0.5 → 0.7 on 2026-05-06 (over-fire diagnostic); → 0.8 on
        2026-06-08 (Plans/11 Stage 2) after fitting the post-hoc
        temperature T=1.6553 (`scripts/calibrate_structure_heads.py`,
        ECE 0.0237→0.0042). On val.jsonl (n=12348) the calibrated 0.8
        operating point is P=0.926 / R=0.902 / F1=0.914 — the max-F1
        point, +5.4pt precision vs the old raw-0.7 (P=0.872 / R=0.935).
        Complements the ``_plausible_heading`` text-shape guard (Stage 1).
    doc_role_threshold
        Semantic ``doc_role`` confidence floor for paragraph
        ``doc_role`` overrides AND for metadata_drop emission.
        Default 0.5.
    """
    if len(decisions) != len(regions):
        raise ValueError(
            f"decisions length {len(decisions)} != regions length "
            f"{len(regions)} (Stage-4 contract violation)"
        )

    regions_out: list[Region] = []
    claimed: set[int] = set()

    # ------------------------------------------------------------------
    # Pass 1 — Claim table/math member FBs so prose passes skip them.
    # Multiple Stage-2 candidates can overlap on the same FB (typical
    # for math: pdfplumber + glyph-density detectors emit per-line
    # equations that share inline blocks). Track per-FB ownership so
    # the Pass-3 passthrough emits ONE region per FB.
    # ------------------------------------------------------------------
    table_math_decisions: list[tuple[int, Any, RegionCandidate]] = []
    for region_idx, decision in enumerate(decisions):
        route = getattr(decision, "route", None)
        if route not in ("table", "math"):
            continue
        cand = regions[region_idx]
        for fb_idx in cand.member_block_indices or []:
            claimed.add(int(fb_idx))
        table_math_decisions.append((region_idx, decision, cand))

    # FBs already emitted via a table/math passthrough (Pass 3). Used
    # to dedup overlapping math candidates — second one sees its
    # members all-already-emitted and is skipped.
    emitted_passthrough: set[int] = set()

    # ------------------------------------------------------------------
    # Pass 1b — Running headers/footers. Deterministic page-furniture
    # removal (model-free): a short margin block whose page-normalized
    # text recurs across pages is the doc's running header/footer, not
    # content. Claim BEFORE the heading/paragraph passes so it can't be
    # mis-emitted as a heading and so it stops shattering page-spanning
    # paragraphs. Recorded as metadata_drop (no FB is lost).
    # ------------------------------------------------------------------
    running_header_idx = _detect_running_headers(feature_blocks)
    for i in sorted(running_header_idx):
        if i in claimed:
            continue
        # Part F — never drop a synthetic image FB as a running header
        # (empty text would recur across pages); it is the figure pass's.
        if getattr(feature_blocks[i], "is_image", False):
            continue
        fb = feature_blocks[i]
        regions_out.append(
            Region(
                kind="metadata_drop",
                feature_block_indices=(i,),
                payload={
                    "text": (getattr(getattr(fb, "raw", None), "text", "") or "").strip(),
                    "reason": "running_header",
                },
                provenance={"pass": "running_header"},
            )
        )
        claimed.add(i)

    # Content-free blocks — decorative rules ("_____"), stray punctuation
    # ("."), box-drawing glyphs, OCR specks (no alphanumerics at all) — carry
    # nothing; drop them so they are not emitted as paragraphs/headings and
    # so they stop shattering paragraph runs.
    for i in range(len(feature_blocks)):
        if i in claimed:
            continue
        # Part F — a SYNTHETIC image FeatureBlock carries empty text (it is
        # a figure placeholder, not OCR noise); it is claimed by the
        # figure-formation Pass-3a. Skip it here so it is not dropped as
        # noise before the figure region can claim it.
        if getattr(feature_blocks[i], "is_image", False):
            continue
        if _is_noise_block(_fb_text(feature_blocks, i)):
            regions_out.append(
                Region(
                    kind="metadata_drop",
                    feature_block_indices=(i,),
                    payload={"text": _fb_text(feature_blocks, i), "reason": "noise"},
                    provenance={"pass": "noise"},
                )
            )
            claimed.add(i)

    # ------------------------------------------------------------------
    # Pre-pass — pluck per-cell role labels from CouncilState, if the
    # table_specialist BERT ran. The orchestrator aggregates cells
    # across every TableCandidate into one flat list before invoking
    # ``run_bert("table_specialist", all_cells)``; signals come back
    # with ``region_id`` equal to that flat-list index. To bucket the
    # signals back to per-table 2D grids we walk the candidates in the
    # same order the orchestrator did and call ``build_cells`` to
    # recover each candidate's cell footprint.
    # ------------------------------------------------------------------
    table_candidate_to_cell_roles: dict[int, list[list[str]] | None] = _resolve_cell_roles(
        state, regions
    )

    # ------------------------------------------------------------------
    # Pass 2 — Heading. Single-FB regions where Structure says heading.
    #
    # Cascade-preservation contract: the gate-side check reads the
    # ``is_heading_calibrated`` signal (post-hoc temperature applied at
    # inference time — see ``dart_semantic/council/structure.py``), so
    # the threshold here lives on a calibrated probability scale. The
    # raw ``is_heading`` signal is still on the council state and is
    # what the orchestrator's cascade builder feeds Semantic — that
    # data path stays untouched. Pre-calibration checkpoints emit a
    # calibrated signal numerically identical to the raw one (T=1.0),
    # so this code path is no-op-safe on legacy heads.
    # If the calibrated signal is missing (very-old checkpoints that
    # predate the second TypedSignal — none in production today), we
    # fall back to the raw ``is_heading`` signal so behavior is
    # bit-for-bit identical to pre-calibration Stage 5.
    # ------------------------------------------------------------------
    n_fb = len(feature_blocks)
    for i in range(n_fb):
        if i in claimed:
            continue
        sig = _get_signal(state, "structure", "is_heading_calibrated", i)
        if sig is None:
            sig = _get_signal(state, "structure", "is_heading", i)
        label, conf = _top1(sig)
        if label != "heading" or conf < is_heading_threshold:
            continue
        fb = feature_blocks[i]
        text = (getattr(getattr(fb, "raw", None), "text", "") or "").strip()
        # Stage-1 over-detection guard (Plans/11): skip blocks that are not
        # heading-shaped. Leaving ``i`` unclaimed re-routes the FB to the
        # prose passes (1-FB paragraph fallback) — no FB is lost.
        if not _plausible_heading(text):
            continue
        # MergeOrSplit override (2026-06-16): a block MergeOrSplit ties to
        # the PREVIOUS prose block as the same logical block cannot be a
        # heading — headings are structural breaks, not paragraph
        # continuations. This corrects is_heading false-positives on
        # paragraph line-fragments (the whitepaper "every body line → <h6>"
        # failure: is_heading fires on a mid-paragraph line, shattering the
        # paragraph run). Gated on the previous block being paragraph-role
        # and unclaimed, so genuine section headings — preceded by a
        # not_same boundary (or a heading/list/table) — are unaffected.
        if (
            i > 0
            and (i - 1) not in claimed
            and _is_paragraph_role(state, i - 1)
            and _merge_present_same(state, i - 1, same_logical_block_threshold)
        ):
            continue
        # Plain-heading FP guard (2026-06-16): the is_heading head fires on
        # mid-paragraph BODY lines that look title-ish ("…procedural due
        # process claim. Following"). A GENUINE heading is set off
        # stylistically (bold / enlarged) OR numbered ("A.", "II."), AND
        # breaks the text flow — it follows a sentence-ENDING paragraph and
        # precedes an uppercase one. So decline to claim a candidate that is
        # (a) PLAIN: not bold, font ratio < 1.15, not numbered; (b) has prose
        # flowing THROUGH it: the previous unclaimed prose-role block (para OR
        # list_item — the head scatters roles) does NOT end a sentence AND the
        # next block starts lowercase; (c) is itself mid-clause. Leaving it
        # unclaimed routes it to the paragraph pass, which restitches the split
        # body paragraph. Relies on the deterministic flow signal, not the
        # (here unreliable) merge head; styled/numbered headings are untouched.
        # Verified clean win (opinion_11298455 3→2) with NO regression once a
        # stale-baseline measurement error was corrected (the council is
        # deterministic; compare against the immediate prior state).
        _seed_bold = bool(getattr(getattr(fb, "raw", None), "is_bold", False))
        _seed_fr = float(getattr(fb, "relative_font_ratio", 1.0) or 1.0)
        _plain = (not _seed_bold) and _seed_fr < 1.15 and _numbered_section_depth(text) is None
        if (
            _plain
            and i > 0
            and (i - 1) not in claimed
            and _is_prose_role(state, i - 1)
            and not _ends_sentence(_fb_text(feature_blocks, i - 1))
            and not _ends_sentence(text)
            and (i + 1) < n_fb
            and _fb_text(feature_blocks, i + 1)[:1].islower()
        ):
            continue
        # Forward-merge demotion (2026-06-17): a LONG heading candidate that
        # MergeOrSplit ties to the NEXT block as a strong "same", AND whose
        # next block starts lowercase, is a sentence-like paragraph LEAD-IN
        # flowing into body — not a standalone heading. Claiming it splits the
        # paragraph and orphans the continuation as a lowercase fragment.
        # (Whitepaper regression under a heavier-calibration head: a bold
        # "Inclusive workplaces and schools are built on this foundation:
        # accommodating …" lead-in fired is_heading and orphaned "an individual
        # requests …".) The >=7-word gate is the discriminator: genuine headings
        # are short noun phrases ("Conditions of Confinement", "A. Historically
        # the Eighth Amendment banned" — whose wrapped tails the absorption pass
        # below correctly keeps), whereas a finite-clause lead-in runs long.
        # Combined with not-ending-a-sentence + the high 0.85 same gate + the
        # lowercase-next orphan signature, real long legal headings (which carry
        # not_same on the forward side) are spared. Runs BEFORE absorption so a
        # lead-in is never used as an absorption seed.
        if (
            (i + 1) < n_fb
            and (i + 1) not in claimed
            and len(text.split()) >= 7
            and not _ends_sentence(text)
            and _merge_present_same(state, i, 0.85)
            and _fb_text(feature_blocks, i + 1)[:1].islower()
        ):
            continue
        # Multi-line heading absorption (2026-06-16): a heading that WRAPS
        # across visual lines (legal "A. Historically, the Eighth Amendment
        # banned / intentional sentences, not prison conditions") fires
        # is_heading only on its FIRST line; the 2nd+ lines (is_heading low,
        # lowercase start) otherwise leak into the FOLLOWING body paragraph as
        # a fragment. Extend the claim forward over lines that MergeOrSplit
        # ties to the heading as "same" AND that share the heading's style
        # (bold flag + font ratio), stopping at the first not_same (the
        # heading→body boundary) or a sentence end. The style match prevents
        # absorbing body even if the merge head erroneously says "same".
        head_idx = [i]
        seed_bold = bool(getattr(getattr(fb, "raw", None), "is_bold", False))
        seed_fr = float(getattr(fb, "relative_font_ratio", 1.0) or 1.0)
        # Only absorb wrapped continuation lines when the heading is
        # stylistically DISTINCT from body (bold or enlarged font). A "plain"
        # heading — same font as body, distinguished only by numbering /
        # position, common in legal opinions ("C. Vidal's SHU Confinement") —
        # is indistinguishable from the body line that follows, and the merge
        # head frequently ties heading→body as "same"; absorbing there
        # swallows body text and orphans the next line as a fragment. Bold /
        # large headings (the validated whitepaper + opinion case) are safe.
        seed_distinct = seed_bold or seed_fr >= 1.15
        j = i + 1
        while (
            seed_distinct
            and j < n_fb
            and j not in claimed
            and (j - i) <= 3
            and _merge_present_same(state, j - 1, same_logical_block_threshold)
            and not _ends_sentence(_fb_text(feature_blocks, j - 1))
        ):
            cand_fb = feature_blocks[j]
            cand_bold = bool(getattr(getattr(cand_fb, "raw", None), "is_bold", False))
            cand_fr = float(getattr(cand_fb, "relative_font_ratio", 1.0) or 1.0)
            # Centered multi-line TITLE exception (2026-06-17): a stacked title
            # often sets only its first line bold ("Shades of Accessibility" /
            # "Why ADA Sunglasses Accommodations" / "Do Not Require Disability
            # Disclosure"), so the bold/font style-match below would refuse to
            # absorb the continuation lines and strand them as stray paragraphs.
            # When BOTH seed and candidate are centered AND the role head itself
            # calls the candidate a heading, treat it as a title continuation and
            # absorb regardless of the bold mismatch. Gated on ``is_centered`` so
            # it can't fire on (left-aligned) legal/body text.
            cand_role, _ = _top1(_get_signal(state, "structure", "structural_role", j))
            title_continuation = (
                bool(getattr(fb, "is_centered", False))
                and bool(getattr(cand_fb, "is_centered", False))
                and cand_role == "heading"
            )
            if not title_continuation and (cand_bold != seed_bold or abs(cand_fr - seed_fr) > 0.15):
                break
            head_idx.append(j)
            j += 1
        head_text = (
            _join_dehyphenated(feature_blocks, head_idx, state) if len(head_idx) > 1 else text
        )
        regions_out.append(
            Region(
                kind="heading",
                feature_block_indices=tuple(head_idx),
                payload={
                    "level_hint": _level_hint(fb),
                    "text": head_text,
                    "confidence": conf,
                },
                provenance={"pass": "heading", "is_heading_conf": conf},
            )
        )
        claimed.update(head_idx)

    # Structural heading promotion (2026-06-17): recover NUMBERED-section
    # headings whose binary is_heading dipped below the claim threshold but
    # whose role head still says "heading" and whose structure is unambiguous.
    # A block that is numbered ("4.1", "II.", "A."), role=heading (confident),
    # heading-shaped, not a full sentence, and SET OFF from the previous block
    # (not a same-merge continuation) IS a heading even when is_heading < the
    # threshold. This hardens Pass 2 against is_heading operating-point shifts —
    # e.g. a heavier-calibration head dropping "4.1 Keyhani v. Trustees …" to
    # 0.65 — and firms up legal "I./II./A." section headings. The role=heading
    # gate is the discriminator (numbered BODY lines carry role=paragraph), and
    # ``_plausible_heading`` still rejects TOC dotted-leader entries.
    for i in range(n_fb):
        if i in claimed:
            continue
        fb = feature_blocks[i]
        text = (getattr(getattr(fb, "raw", None), "text", "") or "").strip()
        if _numbered_section_depth(text) is None:
            continue
        role_label, role_conf = _top1(_get_signal(state, "structure", "structural_role", i))
        if role_label != "heading" or role_conf < 0.6:
            continue
        if not _plausible_heading(text) or _ends_sentence(text):
            continue
        # Set off on the BACKWARD side: previous block is not a same-merge
        # continuation into this one (else this is mid-paragraph).
        if (
            i > 0
            and (i - 1) not in claimed
            and _is_paragraph_role(state, i - 1)
            and _merge_present_same(state, i - 1, same_logical_block_threshold)
        ):
            continue
        # Set off on the FORWARD side too: a genuine section heading is a
        # standalone break (MergeOrSplit says not_same to the next block, like
        # shades "4.1 Keyhani …" → body). Requiring this prevents promoting a
        # numbered BODY line that the role head mislabels "heading" mid-clause —
        # which would split the paragraph and orphan its continuation as a
        # lowercase fragment (observed on a legal citation "… 2023 REMS."). The
        # last block has no successor and is trivially set off.
        if not (i + 1 >= n_fb or _merge_present_not_same(state, i, same_logical_block_threshold)):
            continue
        regions_out.append(
            Region(
                kind="heading",
                feature_block_indices=(i,),
                payload={
                    "level_hint": _level_hint(fb),
                    "text": text,
                    "confidence": role_conf,
                },
                provenance={"pass": "heading_promote", "role_conf": role_conf},
            )
        )
        claimed.add(i)

    # ------------------------------------------------------------------
    # Pass 3 — Table & Math passthrough.
    # ------------------------------------------------------------------
    for region_idx, decision, cand in table_math_decisions:
        flags = tuple(getattr(decision, "flags", ()) or ())
        route = getattr(decision, "route", None)
        all_member_idx = [int(i) for i in (cand.member_block_indices or [])]
        # Drop members already emitted by an earlier passthrough so
        # overlapping candidates don't produce duplicate-claim regions.
        member_idx = tuple(i for i in all_member_idx if i not in emitted_passthrough)
        if route == "table" and "detector_demoted" not in flags:
            if not isinstance(cand, TableCandidate):
                # Defensive — Stage 4 should never route a non-table
                # to "table", but if it does we skip the passthrough
                # (the FBs are already claimed; they won't fall into
                # prose, so we lose them — fix the planner if so).
                continue
            if not member_idx:
                # All members already emitted by an earlier passthrough
                # (overlapping candidates). Skip this duplicate.
                continue
            tc: TableCandidate = cand
            n_rows = len(tc.cell_grid or [])
            n_cols = max(
                (len(row) for row in (tc.cell_grid or [])),
                default=0,
            )
            payload: dict[str, Any] = {
                "cell_grid": list(tc.cell_grid or []),
                "header_row_indices": list(tc.header_row_indices or []),
                "bordered": bool(tc.bordered),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "caption_fb_index": _find_caption_neighbor(
                    tc,
                    feature_blocks,
                    state,
                    doc_role_threshold=doc_role_threshold,
                ),
                # Per-cell role labels from BERT-TableSpecialist, if it
                # ran. ``None`` means the BERT didn't run (or didn't
                # cover this candidate); Stage 6's table prompt builder
                # falls back to "data" for every cell when None.
                "cell_roles": table_candidate_to_cell_roles.get(region_idx),
            }
            aria_hints: tuple[str, ...] = ("scope=col",) if tc.header_row_indices else ()
            regions_out.append(
                Region(
                    kind="table",
                    feature_block_indices=member_idx,
                    payload=payload,
                    aria_hints=aria_hints,
                    provenance={
                        "pass": "table_passthrough",
                        "decision_flags": flags,
                    },
                    source_region_id=region_idx,
                )
            )
            emitted_passthrough.update(member_idx)
        elif route == "math" and "math_detector_stand_in_demoted" not in flags:
            if not isinstance(cand, MathCandidate):
                continue
            if not member_idx:
                # Overlapping math candidate already covered by an
                # earlier emission — skip.
                continue
            mc: MathCandidate = cand
            # MathCandidate.src_text isn't a defined field — the
            # parent RegionCandidate dataclass doesn't carry it.
            # Pull from evidence_features / glyph features which
            # ARE populated; flag if missing.
            src_text = (
                getattr(mc, "src_text", None) or (mc.evidence_features or {}).get("src_text") or ""
            )
            payload = {
                "src_text": src_text,
                "display": True,  # v1: math detector pre-gates display only.
                "glyph_density_features": dict(mc.glyph_density_features or {}),
            }
            # T4 — deterministic MathML reconstruction (gated, default OFF ->
            # byte-identical: no stamp). When SEMANTIK_MATH_RECONSTRUCT is on,
            # recover per-glyph geometry (pdfplumber chars when ``pdf_path`` is
            # threaded; else the flat src_text) and stamp a gate-valid
            # deterministic <math> candidate on the payload. Stage-6's
            # local-by-default specialist + the assembler fallback own the
            # generative / accessible-uniform tiers (DETERMINISTIC-FIRST).
            try:
                from .math_reconstruct.policy import (
                    resolve_math_reconstruct_mode,
                    stamp_math_reconstruction,
                )

                if resolve_math_reconstruct_mode():
                    _page = (mc.pages or [None])[0]
                    _bbox = tuple(float(x) for x in mc.bbox) if mc.bbox else None
                    stamp_math_reconstruction(
                        payload,
                        page_num=_page,
                        bbox=_bbox,
                        pdf_path=pdf_path,
                    )
            except Exception:  # opt-in pass must never crash the cascade
                pass
            regions_out.append(
                Region(
                    kind="math",
                    feature_block_indices=member_idx,
                    payload=payload,
                    provenance={
                        "pass": "math_passthrough",
                        "decision_flags": flags,
                    },
                    source_region_id=region_idx,
                )
            )
            emitted_passthrough.update(member_idx)
        else:
            # Demoted (detector_demoted / math_detector_stand_in_demoted).
            # Release the member FBs so prose passes pick them up — but
            # only if they weren't already emitted by an overlapping
            # confirmed passthrough.
            for fb_idx in all_member_idx:
                if fb_idx not in emitted_passthrough:
                    claimed.discard(fb_idx)

    # ------------------------------------------------------------------
    # Pass 3a — Figure formation from ImageCandidates (LOAD-BEARING).
    # An ImageCandidate routed to "figure" by the deterministic arbiter
    # (Part F) is a real PDF IMAGE page-object. The geometric detector is
    # load-bearing here: we emit ONE figure Region per candidate, claiming
    # its synthetic image FeatureBlock, BEFORE the legacy is_image_block
    # demote pass (Pass-3b) and BEFORE the prose passes so the image FB
    # never leaks into a paragraph. The legacy Pass-3b stays as the
    # secondary path (the ``claimed`` set prevents double-emit). Records
    # the nearest "Figure N:" caption FB below for the assembler.
    # ------------------------------------------------------------------
    for region_idx, decision in enumerate(decisions):
        route = getattr(decision, "route", None)
        if route != "figure":
            continue
        cand = regions[region_idx]
        if not isinstance(cand, ImageCandidate):
            continue
        members = [int(i) for i in (cand.member_block_indices or [])]
        if not members:
            # No claimable synthetic FB (degenerate bbox) — skip rather
            # than emit a figure with no FB (would break the invariant).
            continue
        image_fb = members[0]
        if image_fb in claimed:
            continue
        caption_idx = _find_figure_caption_below(image_fb, feature_blocks)
        regions_out.append(
            Region(
                kind="figure",
                feature_block_indices=(image_fb,),
                payload={
                    "alt_hint": None,
                    "caption_fb_index": caption_idx,
                    "px_size": tuple(getattr(cand, "px_size", (0, 0))),
                },
                provenance={
                    "pass": "figure_image_candidate",
                    "decision_flags": tuple(
                        getattr(decision, "flags", ()) or ()
                    ),
                },
                source_region_id=region_idx,
            )
        )
        claimed.add(image_fb)

    # ------------------------------------------------------------------
    # Pass 3b — Figure (image_block_demoted from Stage 4).
    # We do this BEFORE the prose runs so the FBs don't leak into a
    # paragraph or list run.
    # ------------------------------------------------------------------
    image_demoted_fbs: set[int] = set()
    for region_idx, decision in enumerate(decisions):
        flags = tuple(getattr(decision, "flags", ()) or ())
        if "image_block_demoted" not in flags:
            continue
        cand = regions[region_idx]
        for fb_idx in cand.member_block_indices or []:
            image_demoted_fbs.add(int(fb_idx))

    for i in sorted(image_demoted_fbs):
        if i in claimed:
            continue
        regions_out.append(
            Region(
                kind="figure",
                feature_block_indices=(i,),
                payload={
                    "alt_hint": None,
                    "caption_fb_index": None,
                },
                provenance={"pass": "figure_image_block_demoted"},
            )
        )
        claimed.add(i)

    # ------------------------------------------------------------------
    # Pass 4 — List. Seeds (bullet marker or role=list_item) start a run;
    # wrapped continuation lines are absorbed into the same item so a list
    # item that wraps to a second line ("• … hats," + "and other eyewear.")
    # stays whole instead of orphaning its tail as a paragraph fragment.
    # ------------------------------------------------------------------
    def _list_seed(idx: int) -> bool:
        # Explicit bullet marker wins outright (the role head sometimes tags
        # bullets as paragraph). Ordered markers are NOT used as seeds — those
        # patterns recur in prose and would shatter paragraphs.
        if _BULLET_LIST_RE.match(_fb_text(feature_blocks, idx)):
            return True
        label, conf = _top1(_get_signal(state, "structure", "structural_role", idx))
        if label != "list_item" or conf < _LIST_ITEM_CONFIDENCE_FLOOR:
            return False
        # A markerless "list_item" that starts LOWERCASE, sitting right after a
        # paragraph, is a wrapped prose line the role head mis-fired on (a real
        # list item starts uppercase / a number / a bullet). Reject so the
        # paragraph pass absorbs it. This catches legal-prose continuations
        # ("he sold controlled substances…") the merge head split as not_same.
        if (
            idx > 0
            and (idx - 1) not in claimed
            and _is_paragraph_role(state, idx - 1)
            and _fb_text(feature_blocks, idx)[:1].islower()
        ):
            return False
        # MergeOrSplit override: a "list_item" the merge head ties to an
        # adjacent PARAGRAPH block is a mis-fired role on a wrapped body line,
        # not a real list — reject so the paragraph pass absorbs it.
        if (
            idx > 0
            and (idx - 1) not in claimed
            and _is_paragraph_role(state, idx - 1)
            and _merge_present_same(state, idx - 1, same_logical_block_threshold)
        ):
            return False
        if (
            (idx + 1) < len(feature_blocks)
            and _is_paragraph_role(state, idx + 1)
            and _merge_present_same(state, idx, same_logical_block_threshold)
        ):
            return False
        return True

    def _has_marker(idx: int) -> bool:
        return _has_list_marker(_fb_text(feature_blocks, idx))

    i = 0
    while i < n_fb:
        if i in claimed or not _list_seed(i):
            i += 1
            continue

        # Extend the run: subsequent seeds, or continuation lines (the wrapped
        # tail of an item) — recognised either by MergeOrSplit tying them to
        # the previous block, or, as a deterministic backstop, by the line
        # starting lowercase while NOT being a list seed itself ("…edu-" →
        # "cation."). The lowercase backstop rescues wrapped tails the merge
        # head wrongly split off, which would otherwise orphan as a fragment.
        def _is_wrapped_tail(k: int) -> bool:
            if _has_marker(k) or _list_seed(k):
                return False
            if _merge_present_same(state, k - 1, same_logical_block_threshold):
                return True
            text = _fb_text(feature_blocks, k)
            prev_text = _fb_text(feature_blocks, k - 1)
            # A lowercase-starting line or a hyphen-ending predecessor are
            # DEFINITIVE soft-wrap signals — the line is a wrapped tail even
            # when the merge head wrongly emitted not_same ("…justifi-" →
            # "cation.", "…disclosing specific" → "diagnoses."). New list
            # items and new paragraphs start uppercase, so this does not steal
            # them. Keep these unconditional (they are why FRAGMENT→0).
            if text[:1].islower() or prev_text.endswith("-"):
                return True
            # The short-fragment backstop, by contrast, IS speculative (a
            # short UPPERCASE block could be an unrelated next item/paragraph),
            # so it must not overrule a CONFIDENT not_same from the merge head
            # (the authority on boundaries — rank 7). Only apply it when the
            # boundary signal is absent or not confidently a split.
            if _merge_present_not_same(state, k - 1, same_logical_block_threshold):
                return False
            # Short fragment closing an incomplete previous item, e.g. a
            # trailing citation "[5]." after a bullet that ended mid-clause.
            return len(text.split()) <= 4 and not _ends_sentence(prev_text)

        end = i + 1
        while end < n_fb and end not in claimed:
            if _list_seed(end) or _is_wrapped_tail(end):
                end += 1
            else:
                break
        # Group [i, end) into items: each block starts a new item EXCEPT a
        # wrapped-tail continuation, which appends. So markerless role-list
        # runs keep one item per line, while a wrapped bullet stays one item.
        item_groups: list[list[int]] = []
        for j in range(i, end):
            if item_groups and _is_wrapped_tail(j):
                item_groups[-1].append(j)
            else:
                item_groups.append([j])
        # A single markerless item is almost always a role mis-fire — leave
        # it for the paragraph pass rather than mint a bogus 1-item list.
        if len(item_groups) == 1 and not _has_marker(i):
            i = end
            continue
        items = [
            {
                "fb_index": g[0],
                "text": _join_dehyphenated(feature_blocks, g, state),
                "depth": _list_depth(state, g[0]),
            }
            for g in item_groups
        ]
        regions_out.append(
            Region(
                kind="list",
                feature_block_indices=tuple(range(i, end)),
                payload={
                    "ordered": _is_ordered_list(_fb_text(feature_blocks, i)),
                    "items": items,
                },
                aria_hints=("aria-posinset",),
                provenance={"pass": "list", "n_items": len(items)},
            )
        )
        for j in range(i, end):
            claimed.add(j)
        i = end

    # ------------------------------------------------------------------
    # Pass 5 — Paragraph. Role=paragraph runs gated by MergeOrSplit.
    # ------------------------------------------------------------------
    def _is_paragraph(idx: int) -> bool:
        if _is_paragraph_role(state, idx):
            return True
        # Bridge (2026-06-16): MergeOrSplit ties this block to an adjacent
        # block as the same logical block → treat it as prose even when
        # Structure mislabeled the line-fragment's role. MergeOrSplit is the
        # AUTHORITY on logical-block continuity, so it overrules a spurious
        # is_heading here too (the head alternates paragraph/heading across
        # consecutive lines of ONE paragraph — see Pass 2, which already
        # declined to claim these as headings for the same reason). Genuine
        # headings were claimed in Pass 2 and are skipped by _iter_runs; a
        # real heading also carries merge=not_same on both sides, so it is
        # not bridged. The real paragraph boundary is still enforced below
        # by _should_merge_pair splitting on not_same.
        if (
            idx > 0
            and (idx - 1) not in claimed
            and _merge_present_same(state, idx - 1, same_logical_block_threshold)
        ):
            return True
        if (idx + 1) < len(feature_blocks) and _merge_present_same(
            state, idx, same_logical_block_threshold
        ):
            # Forward bridge only: do NOT demote a heading-shaped, plausible
            # block into prose merely because the noisy merge head ties it to
            # the FOLLOWING body line (rank 5). A genuine heading that reaches
            # here unclaimed (Pass 2's backward override declined it) would
            # otherwise be absorbed into the next paragraph and lost.
            txt = _fb_text(feature_blocks, idx)
            if not (
                _is_heading_shaped(state, idx, is_heading_threshold) and _plausible_heading(txt)
            ):
                return True
        # Blockquote-FP flow-through override (2026-06-16): the head sometimes
        # labels a mid-paragraph BODY line "blockquote" ("…she did not identify
        # much, if / any, speech protected by the First Amendment to which the /
        # curriculum limitation applies."). If prose clearly flows THROUGH this
        # plain block — the previous unclaimed prose block does NOT end a
        # sentence and is not a colon intro, this block is itself mid-clause,
        # and the next block starts lowercase — treat it as paragraph so the
        # body restitches. A REAL blockquote breaks the flow (prev ends a
        # sentence or a colon intro; or its tail/next is set off) and is
        # preserved — verified on opinion_11320152 fb440 (real quote, prev
        # ends a sentence) and opinion_11298455 fb111 (footnote, next uppercase).
        blk_label, _ = _top1(_get_signal(state, "structure", "structural_role", idx))
        if blk_label == "blockquote" and idx > 0 and (idx + 1) < len(feature_blocks):
            blk_bold = bool(getattr(getattr(feature_blocks[idx], "raw", None), "is_bold", False))
            blk_fr = float(getattr(feature_blocks[idx], "relative_font_ratio", 1.0) or 1.0)
            prev_txt = _fb_text(feature_blocks, idx - 1)
            if (
                not blk_bold
                and blk_fr < 1.15
                and (idx - 1) not in claimed
                and _is_prose_role(state, idx - 1)
                and not _ends_sentence(prev_txt)
                and not prev_txt.rstrip().endswith(":")
                and not _ends_sentence(_fb_text(feature_blocks, idx))
                and _fb_text(feature_blocks, idx + 1)[:1].islower()
            ):
                return True
        return False

    def _should_merge_pair(prev_idx: int) -> bool:
        """MergeOrSplit's region_id k corresponds to pair (k, k+1).
        Defensive default: merge when signal is absent (v1)."""
        # MergeOrSplit builds adjacent pairs WITHIN a page only, so at a page
        # boundary the signal is structurally absent and the defensive
        # missing→merge default would fuse genuinely-separate paragraphs (a
        # sentence-ending line on page N with an uppercase new paragraph on
        # page N+1). Force a split here; _merge_continuation_paragraphs owns
        # the cross-page rejoin via its deterministic continuation cue.
        if feature_blocks[prev_idx].raw.page != feature_blocks[prev_idx + 1].raw.page:
            return False
        return _merge_says_same(state, prev_idx, same_logical_block_threshold)

    # First find role=paragraph maximal runs, then split each by merge gates.
    for run_start, run_end in list(_iter_runs(_is_paragraph, feature_blocks, claimed)):
        seg_start = run_start
        for k in range(run_start, run_end - 1):
            if not _should_merge_pair(k):
                _emit_paragraph(
                    seg_start,
                    k + 1,
                    feature_blocks,
                    state,
                    regions_out,
                    claimed,
                    doc_role_threshold,
                )
                seg_start = k + 1
        _emit_paragraph(
            seg_start,
            run_end,
            feature_blocks,
            state,
            regions_out,
            claimed,
            doc_role_threshold,
        )

    # ------------------------------------------------------------------
    # Pass 6 — Code, blockquote, metadata_drop, form, fallback.
    # ------------------------------------------------------------------
    def _is_code(idx: int) -> bool:
        sig = _get_signal(state, "structure", "structural_role", idx)
        label, conf = _top1(sig)
        return label == "code_block" and conf >= _CODE_CONFIDENCE_FLOOR

    def _is_blockquote(idx: int) -> bool:
        sig = _get_signal(state, "structure", "structural_role", idx)
        label, conf = _top1(sig)
        return label == "blockquote" and conf >= _BLOCKQUOTE_CONFIDENCE_FLOOR

    for start, end in list(_iter_runs(_is_code, feature_blocks, claimed)):
        text = "\n".join(
            (getattr(getattr(feature_blocks[j], "raw", None), "text", "") or "").rstrip()
            for j in range(start, end)
        )
        regions_out.append(
            Region(
                kind="code_block",
                feature_block_indices=tuple(range(start, end)),
                payload={"text": text},
                provenance={"pass": "code_block"},
            )
        )
        for j in range(start, end):
            claimed.add(j)

    for start, end in list(_iter_runs(_is_blockquote, feature_blocks, claimed)):
        text = " ".join(
            (getattr(getattr(feature_blocks[j], "raw", None), "text", "") or "").strip()
            for j in range(start, end)
        )
        regions_out.append(
            Region(
                kind="blockquote",
                feature_block_indices=tuple(range(start, end)),
                payload={"text": text},
                provenance={"pass": "blockquote"},
            )
        )
        for j in range(start, end):
            claimed.add(j)

    # AXIS-1: definition_list — collapse a contiguous run of FBs whose
    # structural_role top-1 is a definition-list leaf (definition_term /
    # definition_def, the new AXIS-1 roles) into one ``definition_list``
    # Region (the previously-dead REGION_KINDS member). The payload carries
    # the per-FB (role, text) items so the assembler can emit <dl><dt>/<dd>.
    # A pre-AXIS-1 council (no definition_* logits) never fires this — the
    # run is empty and this pass is a no-op (byte-stable on old adapters).
    def _is_definition(idx: int) -> bool:
        sig = _get_signal(state, "structure", "structural_role", idx)
        label, conf = _top1(sig)
        return (
            label in ("definition_term", "definition_def")
            and conf >= _DEFINITION_CONFIDENCE_FLOOR
        )

    for start, end in list(_iter_runs(_is_definition, feature_blocks, claimed)):
        items = []
        for j in range(start, end):
            jlabel, _ = _top1(_get_signal(state, "structure", "structural_role", j))
            items.append(
                {
                    "role": jlabel,
                    "text": (
                        getattr(getattr(feature_blocks[j], "raw", None), "text", "")
                        or ""
                    ).strip(),
                }
            )
        text = " ".join(it["text"] for it in items)
        regions_out.append(
            Region(
                kind="definition_list",
                feature_block_indices=tuple(range(start, end)),
                payload={"text": text, "items": items},
                provenance={"pass": "definition_list"},
            )
        )
        for j in range(start, end):
            claimed.add(j)

    # metadata_drop — single-FB by Semantic doc_role.
    # Page-furniture (footers/page numbers) lives in the top/bottom margin
    # band; precompute per-page heights so a body-column block Semantic
    # mislabels as footer/metadata is NOT dropped (that loses content — a
    # sentence tail "diagnoses." or a reference fragment "Rep., 2002.").
    _page_h: dict[int, float] = {}
    for fb in feature_blocks:
        # Authoritative page height (see _detect_running_headers) — the
        # content-derived max(bbox[3]) under-estimates on short pages and
        # pulls the bottom band up into real body text.
        ph = float(getattr(fb.raw, "page_height", 0.0) or 0.0)
        _page_h[fb.raw.page] = (
            ph if ph > 0 else max(_page_h.get(fb.raw.page, 0.0), float(fb.raw.bbox[3]))
        )
    for i in range(n_fb):
        if i in claimed:
            continue
        sig = _get_signal(state, "semantic", "doc_role", i)
        label, conf = _top1(sig)
        if label in _METADATA_DOC_ROLES and conf >= doc_role_threshold:
            fb = feature_blocks[i]
            H = _page_h.get(fb.raw.page) or 1.0
            y0, y1 = float(fb.raw.bbox[1]), float(fb.raw.bbox[3])
            in_margin = (y0 < 0.12 * H) or (y1 > 0.88 * H)
            # MergeOrSplit ties it to adjacent prose → paragraph continuation.
            # This DELETES content, so a missing signal must err toward KEEP:
            # MergeOrSplit pairs are within-page only, so a real body line that
            # is the LAST line on a page (its continuation begins the next
            # page) has no within-page same-signal — add deterministic text
            # cues so it is not dropped as a "footer" (the 'diagnoses.' /
            # 'Rep., 2002.' content-loss class). Footers start uppercase /
            # numeric and follow a sentence end, so they still drop.
            prev_txt = _fb_text(feature_blocks, i - 1) if i > 0 else ""
            cur_txt = _fb_text(feature_blocks, i)
            text_continuation = (
                (bool(cur_txt) and cur_txt[:1].islower())
                or prev_txt.endswith("-")
                or (bool(prev_txt) and not _ends_sentence(prev_txt))
            )
            continues = (
                (i > 0 and _merge_present_same(state, i - 1, same_logical_block_threshold))
                or ((i + 1) < n_fb and _merge_present_same(state, i, same_logical_block_threshold))
                or text_continuation
            )
            # Only drop genuine page furniture: in a margin band AND not a
            # prose continuation. Body-positioned "footer/metadata" is a
            # misclassification — leave it for the paragraph pass.
            if not in_margin or continues:
                continue
            text = (getattr(getattr(fb, "raw", None), "text", "") or "").strip()
            regions_out.append(
                Region(
                    kind="metadata_drop",
                    feature_block_indices=(i,),
                    payload={"text": text, "doc_role": label},
                    provenance={
                        "pass": "metadata_drop",
                        "doc_role_conf": conf,
                    },
                )
            )
            claimed.add(i)

    # form — runs of in_widget=True FBs.
    def _is_widget(idx: int) -> bool:
        return bool(getattr(feature_blocks[idx], "in_widget", False))

    for start, end in list(_iter_runs(_is_widget, feature_blocks, claimed)):
        regions_out.append(
            Region(
                kind="form",
                feature_block_indices=tuple(range(start, end)),
                payload={
                    "widgets": [
                        {
                            "fb_index": j,
                            "kind": getattr(feature_blocks[j], "widget_kind", None),
                        }
                        for j in range(start, end)
                    ],
                },
                provenance={"pass": "form"},
            )
        )
        for j in range(start, end):
            claimed.add(j)

    # ------------------------------------------------------------------
    # Defensive fallback — unclaimed FBs become 1-FB paragraphs.
    # ------------------------------------------------------------------
    for i in range(n_fb):
        if i in claimed:
            continue
        fb = feature_blocks[i]
        text = (getattr(getattr(fb, "raw", None), "text", "") or "").strip()
        regions_out.append(
            Region(
                kind="paragraph",
                feature_block_indices=(i,),
                payload={
                    "text": text,
                    "doc_role": _doc_role_override(state, i, doc_role_threshold),
                },
                provenance={"pass": "paragraph_fallback"},
            )
        )
        claimed.add(i)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Post-pass — continuation merge. Stitch paragraph regions that were
    # split mid-prose: across a dropped running header at a page boundary,
    # or by a noisy per-line not_same. The deterministic continuation
    # signal (next line starts lowercase, or prev ends with a hyphen)
    # overrules the split; structural boundaries (a heading/list/table FB
    # between the two) always block the merge.
    # ------------------------------------------------------------------
    # Bridge paragraph halves across ANY dropped block (running header,
    # decorative rule, OCR speck), not just running headers — a stray "."
    # between two halves of a sentence should not block the rejoin.
    dropped_idx = {
        idx for r in regions_out if r.kind == "metadata_drop" for idx in r.feature_block_indices
    }
    regions_out = _merge_continuation_paragraphs(regions_out, feature_blocks, state, dropped_idx)

    # ------------------------------------------------------------------
    # Post-pass — bibliography grouping. In the References section each
    # "[N] ..." entry is one logical block, but its continuation lines
    # (URLs, "[Online]", trailing years) wrap onto their own FBs and the
    # generic prose merge can't restitch them (they start uppercase / "//"
    # / digits). Regroup deterministically by the "[N]" marker.
    # ------------------------------------------------------------------
    regions_out = _group_references(regions_out, feature_blocks, state)

    # Coverage invariant — every FB index appears in exactly one
    # Region's feature_block_indices. Documented at module top.
    # ------------------------------------------------------------------
    seen: dict[int, int] = {}
    for r_idx, region in enumerate(regions_out):
        for fb_idx in region.feature_block_indices:
            if fb_idx in seen:
                raise AssertionError(
                    f"Stage-5 coverage invariant: FB {fb_idx} appears in "
                    f"both region {seen[fb_idx]} ({regions_out[seen[fb_idx]].kind}) "
                    f"and region {r_idx} ({region.kind})"
                )
            seen[fb_idx] = r_idx
    if len(seen) != n_fb:
        missing = sorted(set(range(n_fb)) - set(seen))
        raise AssertionError(
            f"Stage-5 coverage invariant: {len(seen)} FBs covered but "
            f"{n_fb} expected; missing={missing[:10]}{'…' if len(missing) > 10 else ''}"
        )

    # BERT-authoritative precedence flip (gated, default OFF). When on, the
    # council Structure head's structural_role argmax IS the region kind and
    # role_confidence / pedagogical semantic_class are stamped. Runs AFTER the
    # coverage invariant (the FB partition is immutable, so coverage holds) and
    # BEFORE the reading-order sort (which keys off feature_block_indices,
    # unaffected by a kind/payload-only rewrite). Byte-identical when off.
    if resolve_bert_authoritative():
        regions_out = _apply_bert_authoritative_kinds(regions_out, state)

    # Reading-order restoration (gated). The six passes append regions in
    # PASS order (headings, then tables/math, then lists, then paragraphs,
    # then code) and never re-sort, so the region list is segregated by kind
    # rather than monotone in document order. Restore reading order with a
    # final STABLE sort by the region's first owned FeatureBlock index. The
    # coverage invariant just verified guarantees disjoint FB sets, so the
    # sort key is a TOTAL order with no ties -> deterministic. Off by default
    # -> byte-identical to the legacy segregated list.
    if resolve_reading_order_fix():
        _pre_ids = {id(r) for r in regions_out}
        regions_out.sort(
            key=lambda r: min(r.feature_block_indices)
            if r.feature_block_indices else len(feature_blocks)
        )
        assert {id(r) for r in regions_out} == _pre_ids, (
            "reading-order sort dropped/added a region"
        )

    return regions_out


def _fb_text(feature_blocks: list[FeatureBlock], j: int) -> str:
    return (getattr(getattr(feature_blocks[j], "raw", None), "text", "") or "").strip()


def _join_dehyphenated(
    feature_blocks: list[FeatureBlock],
    indices: list[int],
    state: Any,
    *,
    threshold: float = 0.5,
) -> str:
    """Concatenate the FB texts at ``indices`` (in order) into one paragraph
    string, repairing line-break hyphenation ("ac-" + "cessibility" →
    "accessibility").

    A trailing hyphen is repaired (glued with no space, hyphen removed) when
    MergeOrSplit's ``hyphen_repair`` head says ``repair`` for the pair, or —
    if that signal is absent — when the next line starts lowercase (the
    deterministic soft-wrap heuristic). Otherwise blocks join with a space.
    ``indices`` need not be contiguous (cross-gap paragraph merges skip the
    dropped running-header blocks between two halves).
    """
    out = ""
    glue = False  # current ``out`` ends in a hyphen to repair into next block
    for pos, j in enumerate(indices):
        t = _fb_text(feature_blocks, j)
        if not t:
            continue
        if out == "":
            out = t
        elif glue:
            out = out[:-1] + t  # drop the trailing hyphen, glue with no space
        else:
            out = out + " " + t
        glue = False
        if t.endswith("-") and (pos + 1) < len(indices):
            sig = _get_signal(state, "merge_or_split", "hyphen_repair", j)
            if sig is not None:
                lbl, conf = _top1(sig)
                glue = lbl == "repair" and conf >= threshold
            else:
                nxt = _fb_text(feature_blocks, indices[pos + 1])
                glue = bool(nxt[:1].islower())
    return out.strip()


def _emit_paragraph(
    start: int,
    end: int,
    feature_blocks: list[FeatureBlock],
    state: Any,
    regions_out: list[Region],
    claimed: set[int],
    doc_role_threshold: float,
) -> None:
    """Emit one paragraph Region covering FBs in ``[start, end)``.

    Concatenates text with single spaces. doc_role override pulled from
    the FIRST FB in the segment — it's the head of the logical block
    in MergeOrSplit terms.
    """
    if start >= end:
        return
    text = _join_dehyphenated(feature_blocks, list(range(start, end)), state)
    regions_out.append(
        Region(
            kind="paragraph",
            feature_block_indices=tuple(range(start, end)),
            payload={
                "text": text,
                "doc_role": _doc_role_override(
                    state,
                    start,
                    doc_role_threshold,
                ),
            },
            provenance={"pass": "paragraph", "n_blocks": end - start},
        )
    )
    for j in range(start, end):
        claimed.add(j)


def _ends_sentence(text: str) -> bool:
    """True if ``text`` ends a sentence (terminal punctuation, allowing a
    trailing quote/bracket/footnote-bracket)."""
    t = text.rstrip()
    if not t:
        return False
    # strip a trailing footnote/cite bracket like "[12]" or quotes/parens
    t = re.sub(r"[\)\]\"”’]+$", "", t).rstrip()
    t = re.sub(r"\[\d+(?:[,–-]\d+)*\]$", "", t).rstrip()
    return t[-1:] in ".!?"


_REFERENCES_HEADINGS = frozenset({"references", "bibliography", "works cited", "reference list"})
_REF_MARKER_RE = re.compile(r"^\s*\[\d+\]")


def _group_references(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    state: Any,
) -> list[Region]:
    """Regroup the References section so each "[N] ..." entry is one block.

    Scoped to FBs AFTER a ``References``/``Bibliography`` heading: gather the
    prose/list FBs there, sort by reading order, and start a new citation
    region at every FB that opens with a ``[N]`` marker (continuation lines
    attach to the current entry). No-op when there is no references heading.
    """
    ref_fb: int | None = None
    for r in regions:
        if r.kind == "heading":
            txt = " ".join(((r.payload or {}).get("text") or "").split()).strip().lower()
            if txt in _REFERENCES_HEADINGS:
                if r.feature_block_indices:
                    ref_fb = min(r.feature_block_indices)
                break
    if ref_fb is None:
        return regions

    zone_ids = {
        id(r)
        for r in regions
        if r.kind in ("paragraph", "list")
        and r.feature_block_indices
        and min(r.feature_block_indices) > ref_fb
    }
    if not zone_ids:
        return regions

    kept = [r for r in regions if id(r) not in zone_ids]
    fb_idx = sorted(i for r in regions if id(r) in zone_ids for i in r.feature_block_indices)
    groups: list[list[int]] = []
    cur: list[int] = []
    for i in fb_idx:
        if _REF_MARKER_RE.match(_fb_text(feature_blocks, i)) and cur:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        groups.append(cur)

    for g in groups:
        kept.append(
            Region(
                kind="paragraph",
                feature_block_indices=tuple(g),
                payload={
                    "text": _join_dehyphenated(feature_blocks, g, state),
                    "doc_role": "citation",
                },
                provenance={"pass": "references", "n_blocks": len(g)},
            )
        )
    return kept


_LEADING_OPENERS = " \t\"'“”‘’([{"


def _starts_lower_unquoted(text: str) -> bool:
    """True if the first alphabetic char — AFTER stripping leading quotes,
    brackets and parens — is lowercase. This is the mid-sentence continuation
    signature even when a wrapped line opens with a quoted term ('"assistant,"
    Defendant Carroll …') or a citation parenthetical ('(quoting Mayer v. …)'),
    which a bare ``text[:1].islower()`` misses because the first glyph is the
    quote/paren, not the letter. Genuinely-new paragraphs that open with a quote
    start with a CAPITAL ('"The court held …'), so this still won't fuse them.
    """
    s = (text or "").lstrip(_LEADING_OPENERS)
    return bool(s) and s[0].islower()


def _merge_continuation_paragraphs(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    state: Any,
    running_header_idx: set[int],
) -> list[Region]:
    """Stitch paragraph regions split mid-prose back together.

    Two paragraph regions adjacent in reading order are merged when (a) the
    only FBs between them are dropped running headers (or none) — i.e. no
    heading/list/table/figure separates them — AND (b) the prose clearly
    continues: the later region starts lowercase, or the earlier region's
    last line ends with a hyphen. This repairs both page-boundary splits
    (across a dropped header) and noisy within-page not_same splits, without
    fusing genuinely-separate paragraphs (those start uppercase / the prior
    one ends a sentence).
    """
    paras = sorted(
        (r for r in regions if r.kind == "paragraph"),
        key=lambda r: min(r.feature_block_indices),
    )
    others = [r for r in regions if r.kind != "paragraph"]
    if not paras:
        return regions

    merged_idx: list[list[int]] = []
    roles: list[Any] = []
    cur = list(paras[0].feature_block_indices)
    cur_role = (paras[0].payload or {}).get("doc_role")
    for nxt in paras[1:]:
        a_max = max(cur)
        b_min = min(nxt.feature_block_indices)
        gap = set(range(a_max + 1, b_min))
        # A mergeable gap is either a page boundary (non-empty, only dropped
        # running headers between the halves) OR a directly-adjacent pair
        # whose boundary actually carried a MergeOrSplit signal (i.e. a real
        # document, not the synthetic no-signal fallback): there we trust the
        # deterministic continuation cue over a noisy per-line not_same.
        header_gap = bool(gap) and gap.issubset(running_header_idx)
        boundary_has_signal = (
            _get_signal(state, "merge_or_split", "same_logical_block", a_max) is not None
        )
        empty_gap = not gap
        # MergeOrSplit builds adjacent pairs WITHIN a page, so a paragraph that
        # wraps across a page boundary has NO signal at the boundary. Treat a
        # page change as the missing signal (the synthetic no-signal tests are
        # all same-page, so they stay unmerged).
        cross_page = feature_blocks[a_max].raw.page != feature_blocks[b_min].raw.page
        mergeable_gap = header_gap or (empty_gap and (boundary_has_signal or cross_page))
        a_last = _fb_text(feature_blocks, a_max)
        b_first = _fb_text(feature_blocks, b_min)
        # Continuation cue: next line starts lowercase (mid-sentence wrap),
        # the prior line ends with a hyphen, OR the prior line does not end a
        # sentence (it is mid-clause, e.g. "…Sutton v. United" → "Air Lines,
        # Inc. [20]."). Genuinely-separate paragraphs both end a sentence AND
        # start uppercase, so they are never fused. References that end in a
        # URL/no-punct are protected: _group_references re-splits the
        # bibliography by its [N] markers afterwards.
        if cross_page:
            # At a true page boundary there is NO MergeOrSplit signal, so
            # require a STRONG continuation cue (next line starts lowercase,
            # or the prior line is hyphenated). The weak "prior line does not
            # end a sentence" cue is reserved for intra-page boundaries where
            # a real merge signal existed — at a page break it would fuse
            # genuinely-separate paragraphs whose prior line merely ends in a
            # colon/dash or lost its terminal punctuation in extraction.
            continues = a_last.endswith("-") or _starts_lower_unquoted(b_first)
        else:
            continues = (
                a_last.endswith("-")
                or _starts_lower_unquoted(b_first)
                or not _ends_sentence(a_last)
            )
        if mergeable_gap and continues:
            cur = sorted(cur + list(nxt.feature_block_indices))
        else:
            merged_idx.append(cur)
            roles.append(cur_role)
            cur = list(nxt.feature_block_indices)
            cur_role = (nxt.payload or {}).get("doc_role")
    merged_idx.append(cur)
    roles.append(cur_role)

    out = list(others)
    for idx, role in zip(merged_idx, roles):
        out.append(
            Region(
                kind="paragraph",
                feature_block_indices=tuple(idx),
                payload={
                    "text": _join_dehyphenated(feature_blocks, idx, state),
                    "doc_role": role,
                },
                provenance={"pass": "paragraph", "n_blocks": len(idx)},
            )
        )
    return out


__all__ = [
    "REGION_KINDS",
    "Region",
    "RegionKind",
    "build_structure_graph",
    "resolve_bert_authoritative",
    "resolve_reading_order_fix",
]
