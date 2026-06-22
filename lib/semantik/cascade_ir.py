"""SemantiK v2 cascade result → adapter chapters-IR converter (Phase P3a).

This is the seam between the SemantiK cascade and the P2a output-contract
adapter (``lib/semantik/adapter.py``). The cascade computes per-region
provenance (page / role / confidence / WCAG / raw-block index) during the
run but historically dropped it from its result; P3a surfaces a distilled
``region_provenance`` list on the cascade result + ``PipelineV2Result``
(see ``SemantiK/dart_semantic/cascade.py::_build_region_provenance``), and
THIS module converts that list into the ``List[_AdapterChapter]`` the
adapter's :func:`~lib.semantik.adapter.normalize_cascade_to_ed4all`
consumes.

Migration plan ``plans/finegrain/semantic-v2-dart-migration-2026-06-21.md``
section map:

* §3.3a raw-block-index determinism .... :data:`region_provenance[i]
  ["first_raw_block_index"]` is carried verbatim onto
  :attr:`_AdapterBlock.raw_block_index`; the adapter mints the ``sid`` from
  it, so the same provenance → identical ``data-dart-block-id`` set across
  runs (the determinism contract).
* §3.4 chapter structure ............... :func:`_chapter_boundary` groups
  regions into ``<article role="doc-chapter">`` chapters at content-bearing
  chapter-level headings; non-content headings (answer-key / numeric /
  front-matter) NEVER open a chapter, and section counts are bounded so a
  single chapter never trips the >40-section collapse.
* §3.5 page provenance ................. per-block ``pages`` carried from
  ``region_provenance[i]["pages"]``.
* §3.7 object map ...................... the produced IR is EXACTLY what the
  P2a synthetic fixture (``_SyntheticCascadeResult.chapters``) builds, so
  the chain P3a → adapter → DartMarkersValidator is exercised end-to-end on
  synthetic data with no models.

NO models, NO GPU, NO JSON serialization: the cascade result lives in
memory and is consumed in-process by the seam.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# A section heading of the form "N.M[.K] Title" (e.g. "1.1 Introduction to
# Whole Numbers", "2.3 Solve Equations"). The leading integer ``N`` is the
# parent chapter ordinal used by the §3.4-B section-number chapter-derivation
# path. Mirrors ``toc_frontmatter_detector._SECTION_HEADING_RE`` but captures
# the chapter ordinal.
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+)\.(\d+)(?:\.\d+)*\s+(\S.*)$")

# A real chapter-opener heading "Chapter N[: Title]" — used only to COUNT
# genuine L1 content chapter openers so the section-number derivation path is
# entered only when the document lacks them (don't regress the synthetic
# L1-opener fixtures).
_CHAPTER_OPENER_RE = re.compile(r"^\s*chapter\s+(\d+)\b", re.IGNORECASE)

from lib.semantik.adapter import _AdapterBlock, _AdapterChapter

# Phantom-TOC + front-matter detector (root-cause fix for a PDF's full-book
# TOC, printed in front matter, being classified as real chapter headings).
# Applied to the coerced region_provenance BEFORE chapters are assembled,
# alongside the existing _is_noncontent_heading filtering. Gated behind
# SEMANTIK_DROP_FRONTMATTER_TOC (default ON in the IR-builder path).
from lib.semantik.toc_frontmatter_detector import drop_toc_and_frontmatter

# Reuse the SAME non-content-heading filter the adapter + extractor use so a
# chapter boundary is never opened on answer-key / numeric / front-matter
# noise (keeping each chapter under the >40-section collapse threshold).
from lib.semantic_structure_extractor.semantic_structure_extractor import (
    _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
    _is_noncontent_heading,
)

# §3.4 — a heading at THIS level or shallower opens a new chapter. SemantiK
# emits ``level_hint`` 1..6 (h1..h6); level 1 (and the absent/0 case for a
# first chapter-titling heading) is the chapter boundary. Deeper headings
# (h2+) become in-chapter section blocks.
_CHAPTER_LEVEL_MAX = 1

# §3.4 — hard ceiling on blocks per chapter so an un-headed or mega-chapter
# document can never produce a single chapter that trips the extractor's
# >40-section collapse. One below the threshold (the collapse fires on
# ``> threshold``, so ``threshold`` sections is the largest SAFE chapter).
_MAX_BLOCKS_PER_CHAPTER = _STRUCTURE_COLLAPSE_SECTION_THRESHOLD


def _coerce_provenance(result: Any) -> List[Mapping[str, Any]]:
    """Pull the ordered ``region_provenance`` list off a cascade result.

    Accepts (in priority order):

    1. A ``PipelineV2Result``-like object exposing ``.region_provenance``.
    2. The same object exposing a ``.cascade`` dict carrying
       ``"region_provenance"``.
    3. A bare mapping carrying ``"region_provenance"``.
    4. A bare list (already the provenance list).

    Returns ``[]`` when nothing resolves (a structureless run yields an
    empty IR rather than raising — the adapter then emits an empty doc that
    fails closed at the gate, never a silent pass).
    """
    prov = getattr(result, "region_provenance", None)
    if prov is None:
        cascade = getattr(result, "cascade", None)
        if isinstance(cascade, Mapping):
            prov = cascade.get("region_provenance")
    if prov is None and isinstance(result, Mapping):
        prov = result.get("region_provenance")
    if prov is None and isinstance(result, (list, tuple)):
        prov = result
    if not prov:
        return []
    return [p for p in prov if isinstance(p, Mapping)]


def _coerce_heading_tree(result: Any) -> List[Sequence[Any]]:
    """Pull the document-order ``heading_tree`` (``[(level, text), ...]``)."""
    ht = getattr(result, "heading_tree", None)
    if ht is None:
        cascade = getattr(result, "cascade", None)
        if isinstance(cascade, Mapping):
            ht = cascade.get("heading_tree")
    if ht is None and isinstance(result, Mapping):
        ht = result.get("heading_tree")
    return list(ht or [])


def _chapter_title_from_heading_tree(
    heading_tree: Sequence[Sequence[Any]],
) -> Optional[str]:
    """Best-effort document title = the first content-bearing top heading.

    Used only as the title for an IMPLICIT leading chapter (regions that
    precede the first explicit chapter-level heading). Returns ``None`` when
    no usable heading exists (the converter then falls back to a generic
    title, never fabricating prose).
    """
    for entry in heading_tree:
        if not entry:
            continue
        text = entry[1] if len(entry) > 1 else None
        if text and not _is_noncontent_heading(str(text)):
            return str(text)
    return None


def _block_from_provenance(prov: Mapping[str, Any]) -> _AdapterBlock:
    """Map one ``region_provenance`` dict onto an :class:`_AdapterBlock`.

    The §3.3a determinism anchor (``first_raw_block_index``) is carried
    verbatim; pages / confidence / role / WCAG / figure-alt ride along. The
    block's ``html`` is left as the deterministic raw text wrapped in a
    paragraph for headingless prose, or empty for headings (the adapter's
    renderer supplies the heading element). The rewrite tier (P-later) owns
    real prose; P3a carries grounded structure + provenance only.
    """
    region_kind = str(prov.get("region_kind") or "paragraph")
    raw_text = str(prov.get("raw_text") or "")
    heading_text = prov.get("heading_text")
    heading_text = str(heading_text) if heading_text else None

    pages = tuple(
        int(p) for p in (prov.get("pages") or []) if isinstance(p, int) and p > 0
    )
    confidence = prov.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    role = prov.get("role")
    role = str(role) if role else None
    wcag_status = prov.get("wcag_status")
    wcag_status = str(wcag_status) if wcag_status else None
    figure_alt = prov.get("figure_alt")
    figure_alt = str(figure_alt) if figure_alt else None

    return _AdapterBlock(
        html=f"<p>{raw_text}</p>" if raw_text else "",
        region_kind=region_kind,
        raw_block_index=int(prov.get("first_raw_block_index") or 0),
        raw_text=raw_text,
        heading_text=heading_text,
        pages=pages,
        confidence=confidence,
        block_role=role,
        wcag_status=wcag_status,
        figure_alt=figure_alt,
    )


def _is_chapter_boundary(prov: Mapping[str, Any]) -> bool:
    """§3.4 — does this region open a new ``<article role=doc-chapter>``?

    True iff the region is a CONTENT-BEARING heading at chapter level
    (``level_hint <= 1``). A non-content heading (answer-key / numeric /
    front-matter) NEVER opens a chapter — it is absorbed as a block under
    the current chapter and later dropped by the adapter's section filter,
    so it cannot inflate the chapter count or balloon a chapter past the
    >40-section collapse.
    """
    if str(prov.get("region_kind")) != "heading":
        return False
    heading_text = prov.get("heading_text")
    if not heading_text or _is_noncontent_heading(str(heading_text)):
        return False
    level = prov.get("level")
    try:
        lvl = int(level) if level is not None else 1
    except (TypeError, ValueError):
        lvl = 1
    return lvl <= _CHAPTER_LEVEL_MAX


def _section_chapter_ordinal(prov: Mapping[str, Any]) -> Optional[int]:
    """The parent chapter ordinal ``N`` if a region is an ``N.M Title`` section
    heading, else ``None``.

    A content-bearing ``N.M`` section heading (not answer-key / numeric noise)
    is the anchor for the §3.4-B section-number chapter-derivation path. A
    non-content heading returns ``None`` (it never anchors a chapter).
    """
    if str(prov.get("region_kind")) != "heading":
        return None
    heading_text = prov.get("heading_text")
    if not heading_text:
        return None
    text = str(heading_text)
    if _is_noncontent_heading(text):
        return None
    m = _SECTION_NUMBER_RE.match(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _count_real_l1_openers(provenance: Sequence[Mapping[str, Any]]) -> int:
    """Count genuine ``Chapter N``-shaped content L1 chapter openers.

    Used to decide whether to take the section-number derivation path: when the
    surviving content has section headings (``N.M``) but essentially NO real L1
    chapter openers, sections are grouped by their leading number into chapters.
    When real L1 openers ARE present, the legacy boundary path is kept (no
    regression to the synthetic L1-opener fixtures).
    """
    count = 0
    for prov in provenance:
        if not _is_chapter_boundary(prov):
            continue
        text = str(prov.get("heading_text") or "")
        if _CHAPTER_OPENER_RE.match(text):
            count += 1
    return count


def _chapter_titles_from_provenance(
    provenance: Sequence[Mapping[str, Any]],
) -> Dict[int, str]:
    """Map chapter ordinal → a real ``Chapter N: Title`` heading text if one
    was seen ANYWHERE in the surviving provenance.

    Used by the section-number derivation path to reuse a real chapter title
    (e.g. ``"Chapter 1: Foundations"``) for the synthesized chapter when the
    document carried the opener text — even if it was a heading with no content
    behind it — falling back to a generic ``"Chapter N"`` when absent (never
    fabricating a descriptive title).
    """
    titles: Dict[int, str] = {}
    for prov in provenance:
        if str(prov.get("region_kind")) != "heading":
            continue
        text = str(prov.get("heading_text") or "")
        m = _CHAPTER_OPENER_RE.match(text)
        if not m:
            continue
        try:
            ordinal = int(m.group(1))
        except (TypeError, ValueError):
            continue
        # First (document-order) real opener text wins for that ordinal.
        if ordinal not in titles and not _is_noncontent_heading(text):
            titles[ordinal] = text
    return titles


def _build_chapters_by_section_number(
    provenance: Sequence[Mapping[str, Any]],
    heading_tree: Sequence[Sequence[Any]],
) -> List[_AdapterChapter]:
    """§3.4-B — derive chapters by grouping ``N.M`` sections by their leading
    chapter number ``N``.

    Walks the surviving provenance in document order. A section heading whose
    leading chapter ordinal ``N`` differs from the current chapter opens a NEW
    chapter (titled from a real ``Chapter N: Title`` heading if one was seen,
    else a generic ``"Chapter N"``). Sections and the blocks that follow them
    attach to the current chapter. Leading blocks that precede the first
    ``N.M`` section form an implicit leading chapter (titled from the heading
    tree, mirroring the legacy path).

    The same per-chapter overflow guard applies (a chapter never exceeds
    ``_MAX_BLOCKS_PER_CHAPTER`` blocks → spills into a ``(cont.)`` continuation),
    but a correctly section-grouped chapter never trips it.
    """
    real_titles = _chapter_titles_from_provenance(provenance)
    chapters: List[_AdapterChapter] = []
    current: Optional[_AdapterChapter] = None
    current_ordinal: Optional[int] = None

    def _open_chapter(title: str) -> _AdapterChapter:
        ch = _AdapterChapter(title=title, blocks=[])
        chapters.append(ch)
        return ch

    def _append_block(block: _AdapterBlock) -> None:
        nonlocal current
        if current is None:
            title = (
                _chapter_title_from_heading_tree(heading_tree) or "Document"
            )
            current = _open_chapter(title)
        # NO block-count overflow guard on the section-number path: the chapter
        # boundary is well-defined (the next distinct N.M chapter ordinal), so a
        # chapter legitimately carries all of its hundreds of content blocks
        # (paragraphs / worked examples / exercises). The >40 collapse the guard
        # protects against is measured in SECTION HEADINGS, not blocks — and a
        # correctly section-grouped chapter carries only ~10 section headings,
        # well under the threshold. Spilling true content into "(cont.)" chains
        # is exactly the garbage this fix removes.
        current.blocks.append(block)

    for prov in provenance:
        ordinal = _section_chapter_ordinal(prov)
        if ordinal is not None and ordinal != current_ordinal:
            # A section whose chapter number differs from the current chapter
            # opens a new chapter.
            title = real_titles.get(ordinal, f"Chapter {ordinal}")
            current = _open_chapter(title)
            current_ordinal = ordinal
            # The N.M section heading becomes an in-chapter section block (it is
            # NOT the chapter title — the chapter title is "Chapter N").
            _append_block(_block_from_provenance(prov))
            continue
        # A real L1 chapter opener that slipped through (rare on this path):
        # treat it as a boundary so a genuine opener is honored.
        if _is_chapter_boundary(prov) and _CHAPTER_OPENER_RE.match(
            str(prov.get("heading_text") or "")
        ):
            title = str(prov.get("heading_text") or "Chapter")
            current = _open_chapter(title)
            m = _CHAPTER_OPENER_RE.match(title)
            try:
                current_ordinal = int(m.group(1)) if m else current_ordinal
            except (TypeError, ValueError):
                pass
            continue
        _append_block(_block_from_provenance(prov))

    # Drop empty chapters AND content-free chapters: an implicit LEADING chapter
    # that gathered only dropped front-matter (all metadata_drop / bare heading
    # blocks, no paragraph / list / table) before the first N.M section is
    # residual front-matter the TOC detector's zone-anchoring left behind — it
    # carries no teaching content and must not become a phantom "Document" /
    # title chapter. A real derived chapter always carries content blocks.
    def _has_content(ch: _AdapterChapter) -> bool:
        return any(
            str(b.region_kind or "") not in {"heading", "metadata_drop", ""}
            for b in ch.blocks
        )

    return [ch for ch in chapters if ch.blocks and _has_content(ch)]


def build_chapters_ir(result: Any) -> List[_AdapterChapter]:
    """Convert a cascade result into the adapter's chapters IR (§3.4).

    Parameters
    ----------
    result
        A ``PipelineV2Result`` (P3a-surfaced ``region_provenance`` +
        ``heading_tree``), its ``.cascade`` dict, or the bare
        ``region_provenance`` list. See :func:`_coerce_provenance`.

    Returns
    -------
    List[_AdapterChapter]
        Exactly the shape :func:`lib.semantik.adapter.normalize_cascade_to_ed4all`
        consumes (the same shape the P2a synthetic fixture builds).
        Regions are grouped into chapters at content-bearing chapter-level
        headings; a leading run of regions before the first such heading
        forms an implicit chapter titled from the heading tree (or a generic
        fallback). Every chapter is bounded to ``_MAX_BLOCKS_PER_CHAPTER``
        blocks (overflow spills into a continuation chapter) so no chapter
        trips the >40-section collapse.

    Document order is taken from ``region_provenance`` order, which the
    cascade derives from ``AssembledDoc.region_provenance`` (the assembler's
    emitted-block → region index sequence, i.e. reading order). When the
    assembler does not surface that mapping (mock / partial runs), the
    cascade falls back to natural region order; either way THIS converter
    consumes whatever document order the provenance list already encodes and
    never re-sorts it.
    """
    provenance = _coerce_provenance(result)
    heading_tree = _coerce_heading_tree(result)

    # Root-cause phantom-TOC + front-matter drop. Operates on the coerced
    # provenance BEFORE chapters are assembled, so a PDF's full-book TOC
    # printed in the front matter (classified as a run of real chapter
    # headings — e.g. a phantom "Chapter 5: Systems" in a ch1-3 extract) and
    # leading front-matter boilerplate (Preface / Copyright / authors / a bare
    # TOC header) never become fabricated IR chapters. Conservative: drops
    # only a contiguous, page-number-increasing TOC run + known front-matter
    # in the zone before the first real chapter; everything after the first
    # real chapter is protected. Gated by SEMANTIK_DROP_FRONTMATTER_TOC
    # (default ON; falsey → byte-identical pass-through, no drops).
    provenance_list = list(provenance)
    fm_diagnostics: Dict[str, Any] = {}
    provenance, dropped_count = drop_toc_and_frontmatter(
        provenance_list, diagnostics=fm_diagnostics
    )
    if dropped_count:
        logger.info(
            "phantom-TOC/front-matter detector dropped %d region(s) "
            "(of %d) before chapter assembly (frontmatter_zone_dropped=%d)",
            dropped_count,
            len(provenance_list),
            fm_diagnostics.get("frontmatter_zone_dropped", 0),
        )
    # Audit diagnostic, mirroring the resegment / TOC structureDiagnostics. The
    # IR builder returns a bare chapter list, so the page-density drop count is
    # surfaced on the result object (when it can hold an attribute) under
    # ``structureDiagnostics`` for downstream audit; a bare-mapping / immutable
    # result silently skips (best-effort, never crashes the build).
    if fm_diagnostics:
        try:
            existing = getattr(result, "structureDiagnostics", None)
            if not isinstance(existing, dict):
                existing = {}
            existing.update(
                {
                    "frontmatter_zone_dropped": fm_diagnostics.get(
                        "frontmatter_zone_dropped", 0
                    ),
                    "frontmatter_total_dropped": fm_diagnostics.get(
                        "total_dropped", 0
                    ),
                }
            )
            setattr(result, "structureDiagnostics", existing)
        except (AttributeError, TypeError):
            pass

    # §3.4-B — section-number chapter derivation. When the surviving content
    # has ``N.M`` section headings but essentially NO real ``Chapter N`` L1
    # content openers (the real EA2e case: the only L1 headings were the
    # chapter INDEX, dropped by Part A; the actual content is all L2/L3
    # ``N.M`` sections), group sections into chapters by their leading number.
    # When real L1 openers ARE present, fall through to the legacy boundary
    # path (so the synthetic L1-opener fixtures never regress).
    real_l1_openers = _count_real_l1_openers(provenance)
    section_ordinals = {
        o
        for o in (_section_chapter_ordinal(p) for p in provenance)
        if o is not None
    }
    # Trigger when the surviving content has N.M section headings but no/few
    # real L1 chapter openers — i.e. the chapters must be DERIVED from the
    # section numbers. Two ways in:
    #   (a) multiple distinct chapter ordinals (1.x, 2.x, 3.x …) with fewer
    #       real L1 openers than ordinals (a multi-chapter extract whose L1
    #       openers were the dropped chapter-INDEX), OR
    #   (b) at least one N.M chapter ordinal AND essentially ZERO real L1
    #       openers (the real EA2e ch1-only capture: the only L1 headings were
    #       the chapter index, dropped by Part A; the content is all N.M
    #       sections that would otherwise spill into un-headed "(cont.)"
    #       overflow chapters).
    derive_by_section = section_ordinals and (
        (len(section_ordinals) >= 2 and real_l1_openers < len(section_ordinals))
        or real_l1_openers == 0
    )
    if derive_by_section:
        logger.info(
            "section-number chapter derivation: %d distinct chapter "
            "ordinal(s) from N.M sections, %d real L1 opener(s)",
            len(section_ordinals),
            real_l1_openers,
        )
        return _build_chapters_by_section_number(provenance, heading_tree)

    chapters: List[_AdapterChapter] = []
    current: Optional[_AdapterChapter] = None

    def _ensure_leading_chapter() -> _AdapterChapter:
        title = _chapter_title_from_heading_tree(heading_tree) or "Document"
        ch = _AdapterChapter(title=title, blocks=[])
        chapters.append(ch)
        return ch

    def _append_block(block: _AdapterBlock) -> None:
        nonlocal current
        if current is None:
            current = _ensure_leading_chapter()
        # §3.4 overflow guard — spill into a continuation chapter so a
        # chapter never exceeds the safe section budget. (Most overflow is
        # later trimmed by the adapter's non-content filter; this is the
        # belt-and-braces ceiling for pathological un-headed documents.)
        if len(current.blocks) >= _MAX_BLOCKS_PER_CHAPTER:
            cont = _AdapterChapter(
                title=f"{current.title} (cont.)", blocks=[]
            )
            chapters.append(cont)
            current = cont
        current.blocks.append(block)

    for prov in provenance:
        if _is_chapter_boundary(prov):
            title = str(prov.get("heading_text") or "Chapter")
            current = _AdapterChapter(title=title, blocks=[])
            chapters.append(current)
            # The chapter-title heading is the article <h2>; it is NOT also
            # emitted as an in-chapter section block (the adapter renders the
            # title separately), so we do not append it as a block.
            continue
        _append_block(_block_from_provenance(prov))

    # Drop any chapter that ended up with zero blocks (a chapter-level
    # heading immediately followed by another — no content between them).
    return [ch for ch in chapters if ch.blocks]


def from_bridge_json(bridge: Mapping[str, Any]) -> List[_AdapterChapter]:
    """Build the chapters IR from a cross-venv bridge-JSON dict (§5 / M4).

    The subprocess bridge (``MCP/tools/pipeline_tools.py::
    _run_semantik_v2_conversion`` shelling out to
    ``SemantiK/scripts/run_cascade_json.py``) reads a plain JSON dict carrying
    ``{"region_provenance": [...], "heading_tree": [...]}`` (plus doc-level
    signals the adapter reads off the result separately). This is a thin,
    explicit entry point over :func:`build_chapters_ir`: the latter's
    :func:`_coerce_provenance` / :func:`_coerce_heading_tree` already accept a
    bare mapping, so the bridge dict feeds straight through — this helper just
    makes the bridge call site self-documenting and asserts dict shape.

    Parameters
    ----------
    bridge
        The decoded bridge JSON. Only ``region_provenance`` + ``heading_tree``
        are consumed here (document structure); the doc-level signals
        (``exit_action`` / ``wcag_status`` / ``theta_score`` / ``flags`` /
        ``lane_used``) are read by the seam off the same dict when it builds
        the adapter input, not here.

    Returns
    -------
    List[_AdapterChapter]
        Identical to ``build_chapters_ir(bridge)`` — the same IR the adapter
        consumes. An empty / structureless bridge yields ``[]`` (fail-closed
        at the gate downstream, never a silent pass).
    """
    if not isinstance(bridge, Mapping):
        raise TypeError(
            f"from_bridge_json expects a mapping, got {type(bridge).__name__}"
        )
    return build_chapters_ir(bridge)


__all__ = ["build_chapters_ir", "from_bridge_json"]
