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

# One-or-more trailing " (cont.)" continuation suffixes. Used to STRIP any
# existing continuation suffix off a chapter title before appending a single
# one, so an N-times-overflowing chapter reads "Title (cont.)" — never the
# accumulating "Title (cont.) (cont.) (cont.)" cascade (defect 2).
_CONT_SUFFIX_RE = re.compile(r"(?:\s*\(cont\.\))+\s*$")

from lib.semantik.adapter import _AdapterBlock, _AdapterChapter, _esc_text

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


# Inline bullet / list-item markers the council leaves embedded in a
# ``list``-kind region's deterministic ``raw_text`` (the structure graph
# types the region but the provenance carries only the flattened text). A run
# of ≥2 of these markers is the deterministic signal that the region is a
# genuine bullet list we can render as ``<ul><li>`` rather than flatten to a
# single ``<p>`` (the adapter's historical behaviour). ``◦``/``▪``/``‣`` are
# OpenStax's nested-bullet glyphs; ``•``/``·`` the top-level bullet.
_BULLET_SPLIT_RE = re.compile(r"\s*[•◦▪‣·]\s+")

# A ``definition_list``-kind region whose raw_text carries "term – definition"
# / "term: definition" pairs renders as ``<dl>``; we keep this conservative
# (only the explicit em-dash / colon term separator one-per-line shape) and
# otherwise fall back to ``<p>`` so we never fabricate a malformed ``<dl>``.


def _render_list_html(raw_text: str) -> Optional[str]:
    """Render a council ``list``-kind region's flattened ``raw_text`` as a
    semantic ``<ul>`` IFF it carries ≥2 inline bullet markers.

    Returns ``None`` when the text is NOT a genuine multi-item bullet list
    (the council's ``list`` head is noisy — most ``list``-typed regions on the
    real corpus are mis-typed headings / numbered exercise answers / single
    sentences with no bullet glyph). The caller then falls back to ``<p>`` so
    a mis-typed region degrades gracefully rather than producing a one-``<li>``
    pseudo-list. Anti-fabrication: items are SPLIT from the existing text only;
    no content is invented or dropped.
    """
    text = (raw_text or "").strip()
    if not text:
        return None
    # Strip a leading bullet so the first split segment is not empty.
    parts = [p.strip() for p in _BULLET_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return None
    items = "".join(f"<li>{_esc_text(p)}</li>" for p in parts)
    return f"<ul>{items}</ul>"


def _esc_attr(value: str) -> str:
    """Escape a string for use inside a double-quoted HTML attribute."""
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Honest type-level accessible name for a figure with no resolvable caption /
# model alt. Byte-frozen to the assembler's
# ``figure_captioner.TYPE_LEVEL_ALT`` so the cascade_ir adapter path and the
# Stage-9 assembler path agree on the screen-reader name.
_TYPE_LEVEL_ALT = "Figure."


def _render_figure_html(
    raw_text: str,
    figure_alt: Optional[str],
    image_src: Optional[str] = None,
    caption_text: Optional[str] = None,
) -> str:
    """Render a council ``figure``-kind region.

    Part F — when ``image_src`` is present (the Stage-F sidecar PNG was
    written), emit a real ``<figure><img src=… alt=…>`` with the SmolVLM2
    caption (or honest type-level alt) as the accessible name, plus a
    ``<figcaption>`` when a caption resolved. When NO ``image_src`` resolved
    (figure path off, render deferred, or write failed), keep the historic
    anti-broken-``<img>`` contract: a text-only ``<figure>`` with the caption
    as ``<figcaption>`` and NEVER a broken ``<img src="">``.

    A synthetic image FB carries empty ``raw_text``, so the caption is sourced
    in priority order: the SmolVLM2 ``figure_alt`` → the resolved neighbor
    ``caption_text`` (the "Figure N:" FB the structure graph linked) →
    ``raw_text``. A figure region NEVER degrades to an empty ``<figure></figure>``
    — when nothing resolves it still ships ``<figure><figcaption>`` with the
    honest type-level ``"Figure."`` accessible name, so the figure remains
    visible and labelled to a screen reader (the drop this fix closes).
    """
    caption = (figure_alt or caption_text or raw_text or "").strip()
    if image_src:
        alt = caption or _TYPE_LEVEL_ALT
        # The visible <figcaption> and the <img alt> may carry the same
        # text (one is the visible caption, the other the accessible name);
        # rendering both is standard and not a WCAG duplication issue. Emit
        # the figcaption whenever a caption resolved.
        figcap = (
            f"<figcaption>{_esc_text(caption)}</figcaption>" if caption else ""
        )
        return (
            f'<figure><img src="{_esc_attr(image_src)}" '
            f'alt="{_esc_attr(alt)}">{figcap}</figure>'
        )
    # No image src: a text-only <figure>. Carry the caption as the visible
    # <figcaption>; when no caption resolved, the type-level alt is the honest
    # accessible name so the figure is never a silent empty element.
    figcap = (
        f"<figcaption>{_esc_text(caption)}</figcaption>"
        if caption
        else f"<figcaption>{_esc_text(_TYPE_LEVEL_ALT)}</figcaption>"
    )
    return f"<figure>{figcap}</figure>"


def _render_table_html(
    raw_text: str,
    cell_grid: Optional[Sequence[Sequence[Any]]],
    header_row_indices: Optional[Sequence[int]] = None,
    cell_roles: Optional[Sequence[Sequence[str]]] = None,
    caption_text: Optional[str] = None,
) -> Optional[str]:
    """Render a council ``table``-kind region as an accessible ``<table>``.

    Reconstructs a WCAG ``<table>`` from the deterministic ``cell_grid`` the
    structure graph carries (the assembler's ``fallback_table`` does the same
    on its own path; this is the cascade_ir-side mirror so the Ed4All adapter —
    which consumes ONLY the distilled provenance, never the assembler HTML —
    can emit a real table instead of flattening the cells to a ``<p>``).

    Per-cell ``cell_roles`` (4-class ``data | header_col | header_row | span``
    from BERT-TableSpecialist), when present AND shape-matched, drives
    ``<th scope=>`` emission; otherwise the first row (or ``header_row_indices``)
    is the header. Returns ``None`` when no grid is available so the caller can
    fall back to the flat-text ``<p>`` (anti-fabrication: never invent a grid).
    """
    grid = [list(row or []) for row in (cell_grid or [])]
    if not grid:
        return None

    use_roles = (
        cell_roles is not None
        and len(cell_roles) == len(grid)
        and all(
            len(cell_roles[i] or []) == len(grid[i] or [])
            for i in range(len(grid))
        )
    )
    if use_roles:
        header_rows = {
            i
            for i, rr in enumerate(cell_roles)
            if rr and sum(1 for r in rr if r == "header_col") * 2 >= len(rr)
        }
    else:
        header_rows = {int(i) for i in (header_row_indices or [])}
    if not header_rows:
        header_rows = {0}

    def _cell(role: Optional[str], text: str, *, in_thead: bool) -> str:
        s = _esc_text(str(text or ""))
        if role == "header_row":
            return f'<th scope="row">{s}</th>'
        if role == "header_col":
            return f'<th scope="col">{s}</th>'
        if in_thead:
            return f'<th scope="col">{s}</th>'
        return f"<td>{s}</td>"

    parts: List[str] = ["<table>"]
    caption = (caption_text or "").strip()
    if caption:
        parts.append(f"<caption>{_esc_text(caption)}</caption>")
    parts.append("<thead>")
    for i in sorted(header_rows):
        if i >= len(grid):
            continue
        row = grid[i] or []
        roles = cell_roles[i] if use_roles else [None] * len(row)
        parts.append(
            "<tr>"
            + "".join(
                _cell(role, c, in_thead=True) for c, role in zip(row, roles)
            )
            + "</tr>"
        )
    parts.append("</thead><tbody>")
    body_emitted = False
    for i, row in enumerate(grid):
        if i in header_rows:
            continue
        body_emitted = True
        row = row or []
        roles = cell_roles[i] if use_roles else [None] * len(row)
        parts.append(
            "<tr>"
            + "".join(
                _cell(role, c, in_thead=False) for c, role in zip(row, roles)
            )
            + "</tr>"
        )
    if not body_emitted:
        parts.append("<tr><td></td></tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _block_html_for_kind(
    region_kind: str,
    raw_text: str,
    figure_alt: Optional[str],
    image_src: Optional[str] = None,
    caption_text: Optional[str] = None,
    cell_grid: Optional[Sequence[Sequence[Any]]] = None,
    header_row_indices: Optional[Sequence[int]] = None,
    cell_roles: Optional[Sequence[Sequence[str]]] = None,
) -> str:
    """Deterministically render the inner HTML for a region by its kind.

    The region_provenance carries the flattened ``raw_text`` PLUS the
    structured rendering data the cascade now distils onto a figure/table
    region (``caption_text`` resolved from the caption-neighbor FB; the
    deterministic ``cell_grid`` + header/role hints for tables — see
    ``cascade.py::_build_region_provenance``). This reconstructs the best
    available semantics deterministically:

    * ``figure`` → ``<figure><img>`` when an ``image_src`` resolved, else a
      text-only ``<figure><figcaption>``; NEVER an empty ``<figure></figure>``
      (the type-level ``"Figure."`` accessible name is the floor).
    * ``table`` → a real accessible ``<table>`` from ``cell_grid`` (with
      ``<th scope=>`` from ``cell_roles`` / ``header_row_indices``); falls back
      to ``<p>`` only when the grid is absent (anti-fabrication).
    * ``list`` / ``definition_list`` → ``<ul>`` when ≥2 bullet markers are
      present, else ``<p>`` (the council ``list`` head is noisy).
    * everything else (``paragraph`` / ``math`` / …) → ``<p>``.
    """
    text = (raw_text or "").strip()
    if region_kind == "figure":
        return _render_figure_html(text, figure_alt, image_src, caption_text)
    if region_kind == "table":
        table_html = _render_table_html(
            text, cell_grid, header_row_indices, cell_roles, caption_text
        )
        if table_html is not None:
            return table_html
        # No structured grid carried → fall through to the flat-text <p>
        # (a table region whose cells the structure graph didn't capture).
    if region_kind in {"list", "definition_list"}:
        ul = _render_list_html(text)
        if ul is not None:
            return ul
    if not text:
        return ""
    return f"<p>{_esc_text(text)}</p>"


def _replay_repaired_text(
    original: str, repaired: str, edits: Sequence[Mapping[str, Any]]
) -> bool:
    """Strict replay check: ``repaired`` must be EXACTLY ``original`` with the
    recorded ``edits`` applied in order (each replacing the FIRST occurrence of
    ``before`` with ``after``).

    The render-time fail-closed guard the design pins on the adapter/downstream
    seam (mirroring ``qwen_specialists.ocr_repair.assert_repair_conservation``,
    re-implemented locally so this Ed4All-venv seam never imports the SemantiK
    cascade package). ANY other drift → ``False`` and the caller reverts the
    block to verbatim ``original`` (the ``data-dart-repair`` exemption is only
    honored when this passes).
    """
    text = original
    for edit in edits:
        if not isinstance(edit, Mapping):
            return False
        before = str(edit.get("before", ""))
        after = str(edit.get("after", ""))
        if not before:
            return False
        text = text.replace(before, after, 1)
    return text == repaired


def _resolve_ocr_repair(
    prov: Mapping[str, Any], raw_text: str
) -> tuple[Optional[str], int]:
    """Resolve the render-time (repaired_text, n_edits) for a provenance dict.

    Returns ``(None, 0)`` — the byte-stable no-repair path — when the additive
    ``repaired_text`` / ``ocr_repair`` keys are absent, malformed, or FAIL the
    :func:`_replay_repaired_text` conservation check (fail-closed: a drifted
    repaired string is discarded and the block ships verbatim ``raw_text``).
    """
    repaired = prov.get("repaired_text")
    ocr = prov.get("ocr_repair")
    if not isinstance(repaired, str) or not repaired:
        return None, 0
    if not isinstance(ocr, Mapping):
        return None, 0
    edits = ocr.get("edits")
    if not isinstance(edits, (list, tuple)) or not edits:
        return None, 0
    if not _replay_repaired_text(raw_text, repaired, edits):
        logger.warning(
            "ocr-repair replay check failed on block %s — reverting to verbatim",
            prov.get("first_raw_block_index"),
        )
        return None, 0
    return repaired, len(edits)


def _block_from_provenance(prov: Mapping[str, Any]) -> _AdapterBlock:
    """Map one ``region_provenance`` dict onto an :class:`_AdapterBlock`.

    The §3.3a determinism anchor (``first_raw_block_index``) is carried
    verbatim; pages / confidence / role / WCAG / figure-alt ride along. The
    block's ``html`` is rendered deterministically by region kind
    (:func:`_block_html_for_kind`) — a genuine bullet ``list`` becomes
    ``<ul>``, a ``figure`` a text-only ``<figure>``, everything else a ``<p>``;
    a heading region carries empty ``html`` (the adapter renders the heading
    element). The rewrite tier (P-later) owns real prose; P3a carries grounded
    structure + provenance only.
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
    # Part F — relative sidecar PNG path (present only on figure regions
    # with a written sidecar). Absent → text-only figure (byte-stable).
    image_src = prov.get("image_src")
    image_src = str(image_src) if image_src else None
    # Resolved figure/table caption text (the "Figure N:" / "Table N:" neighbor
    # FB the structure graph linked). Absent → no <figcaption> / <caption>.
    caption_text = prov.get("caption_text")
    caption_text = str(caption_text) if caption_text else None
    # Structured table grid (table regions only). Absent → the table flattens
    # to <p> (anti-fabrication; never invent a grid).
    cell_grid = prov.get("cell_grid")
    cell_grid = cell_grid if isinstance(cell_grid, list) and cell_grid else None
    header_row_indices = prov.get("header_row_indices")
    header_row_indices = (
        header_row_indices
        if isinstance(header_row_indices, (list, tuple))
        else None
    )
    cell_roles = prov.get("cell_roles")
    cell_roles = cell_roles if isinstance(cell_roles, list) and cell_roles else None

    # SEMANTIK_OCR_CONFUSABLE_REPAIR — the additive repaired text (replay-checked
    # fail-closed). When it survives, the emitted body renders the REPAIRED text
    # (retrieval fidelity), while ``raw_text`` stays verbatim (the sid basis).
    repaired_text, ocr_repair_count = _resolve_ocr_repair(prov, raw_text)
    render_text = repaired_text if repaired_text is not None else raw_text

    # A heading region carries no inner body HTML (the adapter renders the
    # heading element from heading_text); content regions render their kind.
    inner_html = (
        ""
        if region_kind == "heading"
        else _block_html_for_kind(
            region_kind,
            render_text,
            figure_alt,
            image_src,
            caption_text=caption_text,
            cell_grid=cell_grid,
            header_row_indices=header_row_indices,
            cell_roles=cell_roles,
        )
    )

    return _AdapterBlock(
        html=inner_html,
        region_kind=region_kind,
        raw_block_index=int(prov.get("first_raw_block_index") or 0),
        raw_text=raw_text,
        heading_text=heading_text,
        pages=pages,
        confidence=confidence,
        block_role=role,
        wcag_status=wcag_status,
        figure_alt=figure_alt,
        image_src=image_src,
        repaired_text=repaired_text,
        ocr_repair_count=ocr_repair_count,
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
            # Strip any existing "(cont.)" suffix off the base title before
            # appending ONE, so a chapter that overflows the block budget
            # several times reads "Title (cont.)" every time — NOT the
            # accumulating "Title (cont.) (cont.) (cont.)" heading cascade
            # (defect 2: a page-spanning section re-emitting an ever-longer
            # "(cont.)"-chained heading per overflow).
            base_title = _CONT_SUFFIX_RE.sub("", current.title).rstrip()
            # continuation=True — the adapter renders this spill WITHOUT a
            # visible <h2> (an aria-hidden presentation <div> instead), so the
            # repeated overflow does not mint a phantom pseudo-section heading
            # that the downstream SemanticStructureExtractor reads as a new
            # section (the 8× "Chapter Outline (cont.)" hierarchy-pollution
            # defect). The SPLIT itself is retained — it is the belt-and-braces
            # >40-section-collapse ceiling for pathological un-headed documents.
            cont = _AdapterChapter(
                title=f"{base_title} (cont.)", blocks=[], continuation=True
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
