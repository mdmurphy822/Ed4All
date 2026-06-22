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

from typing import Any, Dict, List, Mapping, Optional, Sequence

from lib.semantik.adapter import _AdapterBlock, _AdapterChapter

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
