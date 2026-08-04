"""HTML + resource helpers used by the Trainforge chunker.

Lifted from ``Trainforge/pipeline/process_course.py``::CourseProcessor as
standalone module functions so the chunker proper can call them
without a ``CourseProcessor`` instance.

Lifted helpers:
    - :func:`extract_plain_text` (was ``CourseProcessor._extract_plain_text``)
    - :func:`strip_assessment_feedback` (was ``CourseProcessor._strip_assessment_feedback``)
    - :func:`strip_feedback_from_text` (was ``CourseProcessor._strip_feedback_from_text``)
    - :func:`extract_section_html` (was ``CourseProcessor._extract_section_html``)
    - :func:`type_from_resource` (was ``CourseProcessor._type_from_resource``)

All five were ``@staticmethod`` on the class — none touched ``self`` —
so the lift is a straight rename + indent shift.

``HTMLTextExtractor`` import: was lazy when the chunker lived in the
sibling ``ed4all-chunker`` package (Phase 7a) to dodge a hypothetical
module-load cycle. Now that the chunker lives inside Trainforge the
import-cycle risk is gone — ``Trainforge.parsers.html_content_parser``
imports stdlib + bs4 + ``lib.ontology`` only (no
``Trainforge.pipeline.process_course`` reach-through).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from Trainforge.parsers.html_content_parser import HTMLTextExtractor
from lib.ontology.learning_objectives import LO_ID_PATTERN, LO_ID_SCAN_PATTERN

__all__ = [
    "build_semantik_block_offset_index",
    "extract_learning_outcome_refs",
    "extract_plain_text",
    "extract_plain_text_with_curies",
    "extract_section_html",
    "harvest_semantik_source_refs",
    "parse_dart_confidence_attr",
    "parse_dart_page_kind_attr",
    "parse_dart_pages_attr",
    "resolve_dart_refs_for_chunk",
    "strip_assessment_feedback",
    "strip_feedback_from_text",
    "type_from_resource",
    "union_source_pages",
]


# ---------------------------------------------------------------------------
# DART source-provenance harvest (data-dart-block-id / data-dart-pages)
# ---------------------------------------------------------------------------
#
# DART-produced HTML carries per-block source attribution on every
# ``<section>`` / component wrapper (never on leaf ``<p>``/``<li>`` — see
# DART/CLAUDE.md "Provenance attribute placement rule"):
#
#   <section data-dart-block-id="s3_c0" data-dart-pages="3-5" ...>
#
# The chunker harvests these so the emitted chunk's
# ``source.source_references[]`` can name the DART block(s) + PDF page(s)
# each chunk derives from. The sourceId itself (``dart:{slug}#{block_id}``)
# is minted by the DART-side ``_create_chunk`` callback (it owns the slug);
# this helper returns the raw ``{block_id, pages}`` pairs the callback
# folds into the canonical SourceReference shape.
#
# Additive contract: HTML without any ``data-dart-block-id`` attribute
# yields ``[]`` (legacy / IMSCC / Courseforge corpora stay byte-identical —
# the callback emits no ``source_references`` and the chunk is unchanged).

#: ``data-dart-block-id`` / ``data-semantik-block-id`` attribute value (the block
#: this wrapper carries). DART->semantik purge Stage 1 (dual-READ): both attr
#: spellings are harvested identically; emitters still stamp ``data-dart-*``.
_DATA_DART_BLOCK_ID_RE = re.compile(
    r'data-(?:dart|semantik)-block-id\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

#: ``data-dart-pages`` attribute value, emitted on the SAME element as the
#: block-id (DART/converter/block_templates._provenance_attrs +
#: DART/multi_source_interpreter._build_dart_attrs both stamp them together).
#: Shape is ``"3"`` (single), ``"3-5"`` (contiguous range), or ``"3,5,7"``
#: (non-contiguous list) per ``_format_pages_attr``.
_DATA_DART_PAGES_RE = re.compile(
    r'data-(?:dart|semantik)-pages\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

#: ``data-dart-page-kind`` attribute value — emitted by DART on the SAME element
#: as ``data-dart-pages``, with values ``printed`` | ``interpolated`` |
#: ``physical`` (PINNED CONTRACT). Records whether ``data-dart-pages`` is the
#: book's PRINTED page or the PDF/physical page. An ABSENT attribute MUST be
#: treated as ``physical`` (back-compat — the whole existing corpus has no kind
#: attr yet). Harvested additively into the ref dict's ``pages_kind`` field so a
#: downstream consumer (grounded-answer) can eventually be honest about
#: printed-vs-physical too — without breaking the existing ``pages[]`` shape.
_DATA_DART_PAGE_KIND_RE = re.compile(
    r'data-(?:dart|semantik)-page-kind\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

#: The canonical DART page-kind values; anything else (incl. an absent attr) is
#: normalized to ``"physical"``.
_DART_PAGE_KINDS = frozenset({"printed", "interpolated", "physical"})

#: ``data-dart-block-role`` attribute value — the SemantiK council/Qwen role
#: label (e.g. ``body`` / ``figure`` / ``heading``) emitted on the SAME element
#: as ``data-dart-block-id`` (SemantiK migration §4: chunk-accompanying data).
#: Harvested additively into the ref dict's ``block_role`` field. ABSENT means
#: ``None`` (legacy HTML without the attr still chunks — back-compat); the field
#: is simply OMITTED from the ref dict so existing harvest shapes stay
#: byte-identical.
_DATA_DART_BLOCK_ROLE_RE = re.compile(
    r'data-(?:dart|semantik)-block-role\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

#: ``data-dart-confidence`` attribute value — the SemantiK per-region cascade
#: confidence on the pinned 5-point band (``1.0/0.8/0.6/0.4/0.2``; 1.0 is
#: OMITTED by the adapter), emitted on the SAME element as
#: ``data-dart-block-id``. Harvested as a float into the ref dict's
#: ``confidence`` field. ABSENT / unparseable -> field omitted (back-compat).
_DATA_DART_CONFIDENCE_RE = re.compile(
    r'data-(?:dart|semantik)-confidence\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

#: ``data-dart-wcag`` attribute value — the SemantiK Stage-7 per-region WCAG
#: gate verdict (``passed`` / ``flagged`` / ``skipped``), emitted on the SAME
#: element as ``data-dart-block-id``. Harvested into the ref dict's
#: ``wcag_status`` field. ABSENT -> field omitted (back-compat).
_DATA_DART_WCAG_RE = re.compile(
    r'data-(?:dart|semantik)-wcag\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)


def parse_dart_confidence_attr(value: Optional[str]) -> Optional[float]:
    """Parse a ``data-dart-confidence`` attribute value into a float.

    Returns ``None`` (=> the consumer omits the field) when the value is
    absent / empty / unparseable. No band-snapping here — the adapter already
    emits a banded value; the chunker harvests it verbatim.
    """
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_dart_page_kind_attr(value: Optional[str]) -> str:
    """Normalize a ``data-dart-page-kind`` attribute value.

    Returns one of ``"printed"`` / ``"interpolated"`` / ``"physical"``. An
    absent / empty / unknown value normalizes to ``"physical"`` (the back-compat
    default — every existing kind-less corpus harvests as ``physical``).
    """
    kind = (value or "").strip().lower()
    return kind if kind in _DART_PAGE_KINDS else "physical"

def _harvest_dart_enrichment(tag: str) -> Dict[str, Any]:
    """Harvest the optional SemantiK enrichment attrs off a single opening tag.

    Returns a dict carrying only the keys whose attribute is PRESENT on the
    tag (``block_role`` / ``confidence`` / ``wcag_status``). Absent attributes
    are OMITTED (not ``None``-filled) so a legacy DART block's harvested ref
    stays byte-identical to its pre-SemantiK shape — the additive back-compat
    contract (mirrors how ``data-dart-*`` enrichment never perturbs the
    existing ``{block_id, pages, pages_kind}`` shape on kind-less corpora).

    All three pair to the SAME element as ``data-dart-block-id`` (the
    SemantiK adapter stamps them on the section wrapper, §3.2 placement rule),
    so this is called inside the same-element pairing loop with the matched
    block tag.
    """
    out: Dict[str, Any] = {}
    role_match = _DATA_DART_BLOCK_ROLE_RE.search(tag)
    if role_match:
        role = role_match.group(2).strip()
        if role:
            out["block_role"] = role
    conf_match = _DATA_DART_CONFIDENCE_RE.search(tag)
    if conf_match:
        conf = parse_dart_confidence_attr(conf_match.group(2))
        if conf is not None:
            out["confidence"] = conf
    wcag_match = _DATA_DART_WCAG_RE.search(tag)
    if wcag_match:
        wcag = wcag_match.group(2).strip()
        if wcag:
            out["wcag_status"] = wcag
    return out


#: One full opening tag carrying a ``data-dart-block-id`` — captured so we
#: can pair each block-id with the ``data-dart-pages`` attribute on the
#: *same* element (rather than a document-global scan that would mis-pair a
#: block-id on one wrapper with a pages value on another).
_DART_BLOCK_TAG_RE = re.compile(
    r"<[a-zA-Z][^>]*\bdata-(?:dart|semantik)-block-id\s*=\s*[\"'][^\"']*[\"'][^>]*>",
    re.IGNORECASE,
)


def parse_dart_pages_attr(value: Optional[str]) -> List[int]:
    """Parse a ``data-dart-pages`` attribute value into a sorted int list.

    Handles the three emitted forms (see
    ``DART/multi_source_interpreter._format_pages_attr`` /
    ``DART/converter/block_templates._provenance_attrs``):

      - ``"3"``      -> ``[3]``
      - ``"3-5"``    -> ``[3, 4, 5]`` (inclusive contiguous range)
      - ``"3,5,7"``  -> ``[3, 5, 7]``

    Tolerates whitespace, mixed forms (``"3, 5-7"``), and discards
    non-positive / non-integer tokens defensively. Returns a sorted,
    deduped list; an empty / unparseable value yields ``[]`` so the
    caller omits the optional ``pages`` field (SourceReference schema:
    ``pages`` items are ``minimum: 1``).
    """
    if not value:
        return []
    pages: set = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_hi = token.split("-", 1)
            try:
                lo = int(lo_hi[0].strip())
                hi = int(lo_hi[1].strip())
            except (ValueError, IndexError):
                continue
            if lo > hi:
                lo, hi = hi, lo
            for p in range(lo, hi + 1):
                if p > 0:
                    pages.add(p)
        else:
            try:
                p = int(token)
            except ValueError:
                continue
            if p > 0:
                pages.add(p)
    return sorted(pages)


def union_source_pages(
    refs: Optional[List[Dict[str, Any]]],
) -> List[int]:
    """Sorted, deduped union of the ``pages`` across a source-ref list.

    ``refs`` is a chunker source-ref list — the ``{block_id, pages, ...}``
    dicts :func:`harvest_semantik_source_refs` / :func:`resolve_dart_refs_for_chunk`
    produce off ``data-dart-pages`` / ``data-semantik-pages`` block attrs. This
    folds every ref's positive-int ``pages`` entries into one ascending,
    deduped list — the page-number union for a chunk composed of one or more
    such blocks, suitable for the additive top-level ``source_pages`` chunk
    field (retrieval-citation display).

    Defensive by construction so a malformed ref list never aborts chunk emit:
    non-dict entries are skipped, a missing / ``None`` / non-iterable ``pages``
    is treated as empty, and each page token is coerced through ``int()`` with
    non-positive / non-integer tokens dropped (mirrors
    :func:`parse_dart_pages_attr`, whose output is ``minimum: 1``).

    Returns ``[]`` for ``None`` / empty input or when no ref carries a usable
    page — the caller then OMITS ``source_pages`` (legacy corpora byte-stable).
    """
    if not refs:
        return []
    pages: set = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        raw_pages = ref.get("pages")
        if not raw_pages:
            continue
        try:
            iterator = iter(raw_pages)
        except TypeError:
            continue
        for token in iterator:
            if isinstance(token, bool):
                # bool is an int subclass — never a real page number.
                continue
            try:
                p = int(token)
            except (TypeError, ValueError):
                continue
            if p > 0:
                pages.add(p)
    return sorted(pages)


def harvest_semantik_source_refs(html: str) -> List[Dict[str, Any]]:
    """Harvest DART ``{block_id, pages}`` provenance pairs from ``html``.

    Scans every opening tag that carries a ``data-dart-block-id`` and
    pairs it with the ``data-dart-pages`` (+ sibling ``data-dart-page-kind``)
    attribute on the *same* element. Returns an ordered, deduped (by
    ``block_id``) list of ``{"block_id": str, "pages": List[int], "pages_kind":
    str}`` dicts — document order preserved so a chunk built from several merged
    DART sections lists its blocks in reading order.

    ``pages_kind`` is additive: one of ``"printed"`` / ``"interpolated"`` /
    ``"physical"`` per the DART ``data-dart-page-kind`` contract, normalized to
    ``"physical"`` when the attribute is absent (back-compat — the existing
    corpus carries no kind attr). The ``pages[]`` shape is untouched, so legacy
    consumers ignore the new field without breaking.

    Returns ``[]`` when ``html`` carries no ``data-dart-block-id`` (the
    additive contract — legacy / non-DART HTML emits no source references
    and the chunk stays byte-identical to its pre-harvest shape).

    The empty-string ``data-dart-block-id=""`` case (a defensive sentinel)
    is skipped: an empty block id can't form a valid
    ``dart:{slug}#{block_id}`` sourceId.
    """
    # DART->semantik purge Stage 1 (dual-READ): short-circuit only when NEITHER
    # attr spelling is present.
    if not html or (
        "data-dart-block-id" not in html
        and "data-semantik-block-id" not in html
    ):
        return []

    refs: List[Dict[str, Any]] = []
    seen: set = set()
    for tag in _DART_BLOCK_TAG_RE.findall(html):
        bid_match = _DATA_DART_BLOCK_ID_RE.search(tag)
        if not bid_match:
            continue
        block_id = bid_match.group(2).strip()
        if not block_id or block_id in seen:
            continue
        seen.add(block_id)
        pages_match = _DATA_DART_PAGES_RE.search(tag)
        pages = (
            parse_dart_pages_attr(pages_match.group(2))
            if pages_match
            else []
        )
        kind_match = _DATA_DART_PAGE_KIND_RE.search(tag)
        pages_kind = parse_dart_page_kind_attr(
            kind_match.group(2) if kind_match else None
        )
        ref: Dict[str, Any] = {
            "block_id": block_id, "pages": pages, "pages_kind": pages_kind
        }
        # SemantiK enrichment (block_role / confidence / wcag_status), paired
        # off the SAME opening tag. Keys present only when their attr is —
        # legacy DART HTML without them harvests byte-identically.
        ref.update(_harvest_dart_enrichment(tag))
        refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# Char-span-scoped DART block resolution (per-chunk provenance)
# ---------------------------------------------------------------------------
#
# The historical chunker scoped DART provenance per-section, then fell back
# to the WHOLE-DOCUMENT block list when a heading-derived HTML slice carried
# no ``data-dart-block-id`` (the common case: DART stamps the attribute on the
# block wrapper, and ``extract_section_html`` returns only the heading element
# for headings outside a ``<section>``). That fallback gave most chunks of a
# document every block in the file, so the first ref — and the user-visible
# "View original source" deep link + cited page — was identical across chunks.
#
# These helpers replace that fallback with char-span-scoped containment: each
# DART block's plain-text is located in the chunk's *container text* (the same
# text the chunk's ``char_span`` is measured against), and only blocks whose
# located offset falls inside the chunk's char span are attributed to the
# chunk — in document order, with per-block pages preserved. No precise match
# falls back to AT MOST ONE enclosing-section reference, never the whole
# document.

#: How many leading plain-text words of a block to use as its locate-probe.
#: Short enough to survive minor whitespace / entity normalisation between the
#: raw HTML and the extracted container text, long enough to disambiguate
#: near-duplicate block openings.
_BLOCK_PROBE_WORDS = 8


def _block_element_spans(html: str) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Return ``(html_start, html_next_start, ref)`` for each DART block.

    ``html_start`` is the byte offset of the block's opening tag in ``html``;
    ``html_next_start`` is the start of the *next* block's opening tag (or
    ``len(html)``) so ``html[html_start:html_next_start]`` brackets the block's
    own content (DART blocks are emitted as flat document-order siblings).
    ``ref`` is the ``{"block_id", "pages"}`` dict (same shape as
    :func:`harvest_semantik_source_refs`). Document order preserved; duplicate
    block-ids are kept here (offset disambiguation happens in
    :func:`resolve_dart_refs_for_chunk`) but the canonical dedupe still applies
    downstream.
    """
    spans: List[Tuple[int, int, Dict[str, Any]]] = []
    # DART->semantik purge Stage 1 (dual-READ): short-circuit only when NEITHER
    # attr spelling is present.
    if not html or (
        "data-dart-block-id" not in html
        and "data-semantik-block-id" not in html
    ):
        return spans
    matches = list(_DART_BLOCK_TAG_RE.finditer(html))
    for i, m in enumerate(matches):
        tag = m.group(0)
        bid_match = _DATA_DART_BLOCK_ID_RE.search(tag)
        if not bid_match:
            continue
        block_id = bid_match.group(2).strip()
        if not block_id:
            continue
        pages_match = _DATA_DART_PAGES_RE.search(tag)
        pages = (
            parse_dart_pages_attr(pages_match.group(2)) if pages_match else []
        )
        kind_match = _DATA_DART_PAGE_KIND_RE.search(tag)
        pages_kind = parse_dart_page_kind_attr(
            kind_match.group(2) if kind_match else None
        )
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(html)
        ref: Dict[str, Any] = {
            "block_id": block_id, "pages": pages, "pages_kind": pages_kind
        }
        # SemantiK enrichment paired off the SAME opening tag (omit-when-absent).
        ref.update(_harvest_dart_enrichment(tag))
        spans.append((m.start(), next_start, ref))
    return spans


def build_semantik_block_offset_index(
    html: str, container_text: str = ""
) -> List[Tuple[int, Dict[str, Any], str]]:
    """Build an ordered per-block probe index for char-span / text resolution.

    For every ``data-dart-block-id`` element in ``html`` we extract the block's
    own plain text and take its first :data:`_BLOCK_PROBE_WORDS` words as a
    locate-probe. Returns an ordered list of ``(doc_order, ref, probe)`` tuples
    in document order, where ``doc_order`` is the block's monotonic position in
    the document (0, 1, 2, ...). The probe is the short text signature used by
    :func:`resolve_dart_refs_for_chunk` to test whether the block's content
    appears inside a chunk's own text. ``container_text`` is accepted for
    backward signature compatibility but is no longer used for offset
    computation — resolution is now done against the chunk text directly, which
    is robust to per-section vs whole-body container scoping (the historical
    offset approach mis-scoped when the chunk text spanned multiple sections
    while ``container_text`` was a single section).

    Returns ``[]`` when ``html`` carries no DART blocks — the caller then emits
    no per-chunk refs (additive legacy contract).
    """
    spans = _block_element_spans(html)
    if not spans:
        return []
    index: List[Tuple[int, Dict[str, Any], str]] = []
    for doc_order, (html_start, html_next, ref) in enumerate(spans):
        block_html = html[html_start:html_next]
        block_text = " ".join(extract_plain_text(block_html).split())
        index.append((doc_order, ref, block_text))
    return index


def _block_probes(block_text_collapsed: str) -> List[str]:
    """Return the literal leading-words locate-probe for a block's text.

    The probe is simply the block's first :data:`_BLOCK_PROBE_WORDS`
    whitespace-collapsed words — no chrome-skipping, no content-run
    detection (an earlier draft skipped leading page-number/heading chrome;
    see the empirical note below for why the literal leading words win).

    A single leading probe (rather than a sliding window across the
    whole block) is deliberate: long early blocks — a table of contents, a
    preface listing every section title — contain interior windows that recur
    verbatim in many later chunks, so a multi-window match re-inflates the ref
    count toward the all-document shape this fix kills. The single
    content-anchored opening probe is specific to one block.

    Returns a one-element list (parity with the caller's ``any(...)`` loop) or
    ``[]`` for an empty block.

    The literal leading probe maximises primary-block diversity: a
    content-run-anchored probe that
    skips leading chrome collapses many chunks onto a few TOC/preface blocks
    whose first *content* run recurs, while the literal leading words are
    specific to each block. Chunks whose leading probe doesn't match (the chunk
    text starts mid-block) fall through to the caller's bounded enclosing-
    section ≤1-ref fallback — never the whole-document list.
    """
    words = block_text_collapsed.split()
    if not words:
        return []
    probe = " ".join(words[:_BLOCK_PROBE_WORDS])
    return [probe] if probe else []


def resolve_dart_refs_for_chunk(
    block_probe_index: List[Tuple[int, Dict[str, Any], str]],
    chunk_text: str,
) -> List[Dict[str, Any]]:
    """Select the DART blocks whose content overlaps ``chunk_text``.

    ``block_probe_index`` is the output of
    :func:`build_semantik_block_offset_index` — ``(doc_order, ref, block_text)``
    tuples in document order, where ``block_text`` is the block's
    whitespace-collapsed plain text. A block is attributed to the chunk when
    the block's leading-words probe (:func:`_block_probes` — its first
    :data:`_BLOCK_PROBE_WORDS` words) appears verbatim in the chunk's text.
    Matching is done against whitespace-collapsed projections so raw-HTML
    spacing and extracted spacing agree.

    The direction is deliberately one-way (a block window must appear in the
    chunk, NOT a chunk window in the block): an 8-word window of a short chunk
    recurs across many unrelated blocks (boilerplate phrasing, shared
    instructional language), so the chunk-in-block direction over-matches and
    re-inflates the ref count toward the all-document shape this fix kills.

    Returns the matching ``{"block_id", "pages"}`` dicts in document order,
    deduped by ``block_id`` (first occurrence wins). Returns ``[]`` when nothing
    overlaps (the caller then emits at most one enclosing-section reference
    rather than the whole-document list) or when the index / chunk text is
    empty.
    """
    if not block_probe_index or not chunk_text:
        return []
    chunk_collapsed = " ".join(chunk_text.split())
    if not chunk_collapsed:
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for _doc_order, ref, block_text in block_probe_index:
        if not block_text:
            continue
        matched = any(
            probe and probe in chunk_collapsed
            for probe in _block_probes(block_text)
        )
        if not matched:
            continue
        block_id = ref.get("block_id")
        if not block_id or block_id in seen:
            continue
        seen.add(block_id)
        resolved: Dict[str, Any] = {
            "block_id": block_id,
            "pages": list(ref.get("pages") or []),
            "pages_kind": ref.get("pages_kind") or "physical",
        }
        # Carry the SemantiK enrichment forward (omit-when-absent — these keys
        # are present on ``ref`` only when the source attr was).
        for _k in ("block_role", "confidence", "wcag_status"):
            if _k in ref:
                resolved[_k] = ref[_k]
        out.append(resolved)
    return out


# ---------------------------------------------------------------------------
# Learning-outcome reference extraction
# ---------------------------------------------------------------------------
#
# Courseforge content pages carry learning-outcome (LO) references in two
# forms the page-objective validator reads (see
# ``Courseforge/scripts/validation/validate_page_objectives.py``):
#
#   1. JSON-LD ``learningObjectives[].id`` — surfaced by the parser onto
#      ``ParsedHTMLModule.learning_objectives`` (each
#      ``LearningObjective.id`` is a canonical ``TO-NN`` / ``CO-NN`` id).
#      This lands in ``item["learning_objectives"]`` for the chunker.
#   2. ``data-cf-objective-id`` / ``data-cf-objective-ref`` HTML attributes
#      on ``<section>`` / activity / self-check elements. Surfaced by the
#      parser onto ``ContentSection.objective_refs`` (page-level union on
#      ``ParsedHTMLModule.objective_refs``).
#
# The shared chunk-emit callbacks in
# ``MCP/tools/pipeline_tools.py`` (``_run_dart_chunking`` /
# ``_run_imscc_chunking``) hardcoded ``learning_outcome_refs: []`` and so
# never anchored chunks to the LOs the page carried. This helper closes
# that gap: it harvests both forms, prefers the section-scoped attribute
# refs (so a multi-LO page anchors each chunk to its own section's LO),
# and falls back to the page-level structured ids. Every emitted ref is
# validated against the canonical LO id pattern
# (``^[A-Z]{2,}-\d{2,}$``) so non-LO noise (week prefixes, free-text
# activity labels) never reaches a chunk. When the source HTML carries no
# LO metadata (legacy / DART corpora without JSON-LD or data-cf-* LO
# attributes) the helper returns ``[]`` — byte-identical to the prior
# hardcoded behaviour.


def _normalize_lo_ref(raw: Any, *, preserve_case: bool) -> List[str]:
    """Split, strip, week-prefix-fold, and case-normalise one raw ref.

    Mirrors the normalization arms of
    ``CourseProcessor._extract_objective_refs`` (comma-splitting of
    subagent-emitted ``"co-01,co-02"`` strings; ``WEEK_PREFIX_RE``
    stripping; ``TRAINFORGE_PRESERVE_LO_CASE`` case policy) so the
    chunker path and the in-process wrapper produce identical refs.
    Only ids matching the canonical LO pattern survive — validation is
    applied after case-normalisation against the upper-cased candidate
    so a lowercased ``to-01`` still validates.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        raw = str(raw)
    parts = [p.strip() for p in raw.split(",")] if "," in raw else [raw.strip()]
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        # Strip a leading week prefix (w01-, W01-, ...) to align with the
        # course.json LO id form. Case-insensitive.
        base = _WEEK_PREFIX_RE.sub("", part)
        if not base:
            continue
        # Validate against the canonical LO pattern (always upper-cased so
        # a lowercased emit still passes the [A-Z] class check).
        if not LO_ID_PATTERN.match(base.upper()):
            continue
        out.append(base if preserve_case else base.lower())
    return out


#: Week-prefix stripper, mirrors ``CourseProcessor.WEEK_PREFIX_RE``.
_WEEK_PREFIX_RE = re.compile(r"^w\d{1,2}-", re.IGNORECASE)


def extract_learning_outcome_refs(
    item: Dict[str, Any],
    section_heading: Optional[str] = None,
) -> List[str]:
    """Extract canonical LO ids a chunk should anchor to.

    Resolution order (deterministic, dedup-preserving):

      1. Section-scoped ``ContentSection.objective_refs`` for the section
         matching ``section_heading`` (harvested by the parser from
         ``data-cf-objective-id`` / ``data-cf-objective-ref``). Preferred
         so a multi-LO page anchors each chunk to its own section's LO.
      2. Structured page-level ``item["learning_objectives"]`` ids
         (parser-extracted from JSON-LD ``learningObjectives[].id``).
      3. Page-level ``item["objective_refs"]`` union (data-cf-* fallback
         for the no-sections / heading-drift code path).

    Returns ``[]`` when the source HTML carried no LO metadata — the
    additive, backward-compatible contract: legacy / DART corpora emit
    byte-identically to the prior hardcoded ``[]``.
    """
    preserve_case = (
        os.getenv("TRAINFORGE_PRESERVE_LO_CASE", "").lower() == "true"
    )
    refs: List[str] = []

    def _extend(values: List[str]) -> None:
        for value in values:
            if value and value not in refs:
                refs.append(value)

    # (1) Section-scoped attribute refs (data-cf-objective-id / -ref).
    if section_heading:
        chunk_heading = re.sub(
            r"\s*\(part\s+\d+\)\s*$", "", section_heading
        ).strip().lower()
        for section in item.get("sections", []) or []:
            sec_heading = (
                section.heading
                if hasattr(section, "heading")
                else (section.get("heading") if isinstance(section, dict) else None)
            )
            if sec_heading is None:
                continue
            if sec_heading.strip().lower() != chunk_heading:
                continue
            sec_refs = (
                section.objective_refs
                if hasattr(section, "objective_refs")
                else (
                    section.get("objective_refs")
                    if isinstance(section, dict)
                    else None
                )
            ) or []
            for raw in sec_refs:
                _extend(_normalize_lo_ref(raw, preserve_case=preserve_case))
            break

    # Objective-catalog pages are a special provenance surface: their JSON-LD
    # intentionally enumerates the entire course, so copying that page-wide
    # list onto every split chunk falsely claims that each title/summary
    # fragment delivers every objective. Prefer IDs actually visible in the
    # chunk heading; title/summary fragments correctly remain unanchored. The
    # canonical content pages still carry the same IDs, preserving course-wide
    # objective coverage without fabricating per-chunk coverage.
    _catalog_identity = " ".join(
        str(item.get(key) or "")
        for key in ("item_id", "title", "module_id", "module_title")
    ).lower()
    _is_objective_catalog = (
        "learning objectives map" in _catalog_identity
        or str(item.get("module_id") or "").lower() == "learning_objectives"
    )
    if not refs and _is_objective_catalog:
        for raw in LO_ID_SCAN_PATTERN.findall(
            str(section_heading or "").upper()
        ):
            _extend(_normalize_lo_ref(raw, preserve_case=preserve_case))
        return refs

    # (2) Structured page-level LO ids from the parsed objectives list.
    if not refs:
        for lo in item.get("learning_objectives", []) or []:
            obj_id = lo.id if hasattr(lo, "id") else (
                lo.get("id") if isinstance(lo, dict) else None
            )
            if obj_id:
                _extend(_normalize_lo_ref(obj_id, preserve_case=preserve_case))

    # (3) Page-level data-cf-* union fallback (no-sections / heading drift).
    if not refs:
        for raw in item.get("objective_refs", []) or []:
            _extend(_normalize_lo_ref(raw, preserve_case=preserve_case))

    return refs


# ---------------------------------------------------------------------------
# Resource → chunk-type mapping
# ---------------------------------------------------------------------------

_RESOURCE_TYPE_TO_CHUNK_TYPE = {
    "quiz": "assessment_item",
    "overview": "overview",
    "summary": "summary",
    "discussion": "exercise",
    "application": "exercise",
}


def type_from_resource(resource_type: str) -> str:
    """Map an IMSCC ``resource_type`` to a canonical chunk type.

    Default for unmapped types is ``"explanation"`` (legacy parity
    with ``CourseProcessor._type_from_resource``).
    """

    return _RESOURCE_TYPE_TO_CHUNK_TYPE.get(resource_type, "explanation")


# ---------------------------------------------------------------------------
# Plain-text extraction
# ---------------------------------------------------------------------------


def extract_plain_text(html: str) -> str:
    """Return the plain-text projection of ``html``.

    Wraps :class:`Trainforge.parsers.html_content_parser.HTMLTextExtractor`,
    which already implements the Worker-Q template-chrome skip and the
    canonical ``<script>`` / ``<style>`` subtree drop.
    """

    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def extract_plain_text_with_curies(
    html: str,
) -> Tuple[str, List[str], List[str]]:
    """Return the plain-text projection of ``html`` plus harvested CURIEs.

    Sibling of :func:`extract_plain_text` that additionally surfaces the
    ``data-cf-curie`` tokens harvested by
    :class:`Trainforge.parsers.html_content_parser.HTMLTextExtractor`.
    The 376b64f contract still holds — force-injected curie-anchor
    subtrees are skipped, so the returned ``text`` is byte-identical to
    :func:`extract_plain_text`'s output — but the CURIE values are no
    longer discarded: the downstream ``curie_anchoring`` gate consumes
    them.

    Returns ``(text, curies, forced_curies)``:
      - ``text``    — the plain-text projection (same as
        :func:`extract_plain_text`).
      - ``curies``  — ordered append log of every ``data-cf-curie``
        token across the document.
      - ``forced_curies`` — sorted CURIE tokens whose anchoring element
        also carried ``data-cf-curie-forced="true"``.

    :func:`extract_plain_text`'s signature is intentionally left
    untouched — it has many callers and a tuple-return break is too
    wide a blast radius.
    """

    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return (
        extractor.get_text(),
        extractor.get_curies(),
        extractor.get_forced_curies(),
    )


# ---------------------------------------------------------------------------
# Assessment-feedback stripping (HTML + plain text)
# ---------------------------------------------------------------------------


def strip_assessment_feedback(html: str) -> str:
    """Remove answer feedback from quiz/self-check HTML before text extraction.

    Courseforge quizzes embed correct/incorrect feedback in
    ``<div class="sc-feedback">`` blocks and ``data-correct`` attributes
    on labels. This strips that content so assessment chunks contain
    only question stems and answer options without revealing
    correctness.
    """

    # Remove feedback divs (Courseforge self-check pattern)
    cleaned = re.sub(
        r'<div\s+class="sc-feedback"[^>]*>.*?</div>',
        '', html, flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove data-correct attributes from labels
    cleaned = re.sub(
        r'\s+data-correct="[^"]*"',
        '', cleaned,
    )
    return cleaned


def strip_feedback_from_text(text: str) -> str:
    """Remove residual feedback markers from plain text extraction.

    Handles both line-level and inline feedback patterns since the text
    extractor often concatenates feedback inline with answer options.
    """

    # Remove inline feedback: "Correct. <explanation>" or
    # "Incorrect. <explanation>". These appear after answer option text,
    # running to the next answer option or end.
    text = re.sub(
        r'\s*(?:Correct|Incorrect)\.\s+[^.]*(?:\.[^A-Z])*\.?',
        '', text,
    )
    # Also remove standalone lines
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Correct.") or stripped.startswith("Incorrect."):
            continue
        filtered.append(line)
    return '\n'.join(filtered)


# ---------------------------------------------------------------------------
# Section-aware HTML slicing
# ---------------------------------------------------------------------------


def extract_section_html(html: str, heading: str) -> str:
    """Return the HTML fragment for ``heading``, respecting section boundaries.

    Wave 83 (Phase B) replaces the legacy heading-to-heading regex slice
    with a section-boundary-aware extractor. The legacy slice ran from
    ``<hN>{heading}`` to the next ``<h[1-6]`` and ignored
    ``<section>...</section>`` wrappers entirely. When two adjacent
    headings lived in different ``<section>`` blocks (the canonical
    Courseforge layout), the slice would clip a closing ``</section>``
    from one fragment and an opening ``<section>`` from the next —
    leaving every chunk's HTML unbalanced. This was the load-bearing
    cause of the RDF/SHACL calibration corpus audit's 203/295
    unbalanced-section chunks.

    New behavior:
      - If the heading lives **inside** a ``<section>``, return the
        full enclosing section (open tag → matching close tag).
        Balanced HTML guaranteed.
      - If the heading lives **outside** any ``<section>`` (e.g. a
        page-title ``<h1>``), return just the heading element itself.
        No spurious section tags.
      - If the heading isn't found, return ``""`` (legacy behavior).

    Implementation: regex-locate the heading, regex-locate every
    ``<section>``/``</section>`` event, walk events to determine
    enclosure depth at the heading's position, slice the input string
    accordingly. No new dependencies. The output is always a
    substring of the input HTML, so any well-formedness in the source
    is preserved verbatim.
    """

    if not heading or not html:
        return ""

    # Step 1: locate the heading element. Match tolerantly across
    # whitespace and inline content within the heading body.
    heading_re = re.compile(
        r"<h([1-6])\b[^>]*>\s*" + re.escape(heading) + r"\s*</h\1>",
        re.DOTALL | re.IGNORECASE,
    )
    h_match = heading_re.search(html)
    if not h_match:
        return ""
    h_start, h_end = h_match.span()

    # Step 2: collect every <section> open/close event with its byte span.
    # ``</section\s*>`` allows whitespace before the closing bracket but
    # nothing else (per HTML5).
    section_event_re = re.compile(
        r"<section\b[^>]*>|</section\s*>", re.IGNORECASE
    )

    # Step 3: walk events to determine section enclosure at h_start.
    # Events strictly before the heading update an open-section stack;
    # events strictly after are saved for the close-search step.
    open_stack: List[int] = []  # stack of section-open START offsets
    events_after_heading: List[Tuple[int, int, bool]] = []  # (start, end, is_open)
    for m in section_event_re.finditer(html):
        is_open = m.group(0).lower().startswith("<section")
        s, e = m.span()
        if e <= h_start:
            if is_open:
                open_stack.append(s)
            elif open_stack:
                open_stack.pop()
            # Mismatched </section> with empty stack: ignore (defensive).
        elif s >= h_end:
            events_after_heading.append((s, e, is_open))
        # Events overlapping the heading (e.g. heading nested inside a
        # section's start tag) are exotic and ignored.

    if not open_stack:
        # Heading is outside any section → return just the heading element.
        return html[h_start:h_end]

    # Step 4: the innermost enclosing section starts at open_stack[-1].
    # Walk forward through events_after_heading tracking depth (we're
    # already 1-deep inside the enclosing section). Stop when we hit
    # the matching close.
    section_start = open_stack[-1]
    depth = 1
    for s, e, is_open in events_after_heading:
        if is_open:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return html[section_start:e]

    # Source HTML is unbalanced (section never closes). Best-effort:
    # return from section_start to end. Caller's HTML balance check
    # will flag this as a real source defect, distinct from the
    # legacy clipping bug.
    return html[section_start:]
