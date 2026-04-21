"""Phase 3 (Wave 13): role -> ontology-layered HTML template registry.

Every ``BlockRole`` has exactly one template function. Wave 13 expands
the Wave 12 placeholders into full DPUB-ARIA + schema.org microdata +
semantic HTML5 markup per role. Every template:

    * preserves the Wave 12 provenance contract: top-level element keeps
      ``data-dart-block-role`` + ``data-dart-block-id`` attributes.
    * HTML-escapes every interpolated text value via ``html.escape``.
    * uses ``.get()`` fallbacks on optional attributes so missing fields
      never crash rendering (e.g. a FIGURE with no ``src`` renders just
      the caption).
    * picks stable IDs from ``block.raw.block_id`` (unique per block) to
      avoid duplicate-id collisions when the same heading text appears
      twice in a document.

Wave 14 adds an LLM classifier source; templates stay classifier-agnostic
so heuristic and LLM-classified blocks render identically.
"""

from __future__ import annotations

import html
import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from DART.converter.block_roles import BlockRole, ClassifiedBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _role_attr(block: ClassifiedBlock) -> str:
    """``data-dart-block-role="..."`` provenance attribute string."""
    return f'data-dart-block-role="{block.role.value}"'


# ---------------------------------------------------------------------------
# Wave 19: data-dart-source mapping
# ---------------------------------------------------------------------------
#
# Maps (classifier_source, upstream extractor) -> canonical data-dart-source
# enum value emitted on every section/component wrapper. Kept in sync with
# ``lib/validators/dart_markers.py`` and ``DART/CLAUDE.md`` § "Source
# provenance".
#
# Enum values:
#   pdftotext    - raw prose from pdftotext (default for heuristic blocks)
#   pdfplumber   - pdfplumber table extractor
#   pymupdf      - PyMuPDF figure / TOC / table fallback extractor
#   claude_llm   - Claude-backed LLM classifier
#   dart_converter - generic converter-output source (heuristic/no upstream)


def _data_dart_source_value(block: ClassifiedBlock) -> str:
    """Return the canonical ``data-dart-source`` enum value for ``block``.

    Routing (first match wins):

    * ``classifier_source == "extractor_hint"`` -> upstream ``raw.extractor``
      (pdfplumber / pymupdf / pdftotext); default ``dart_converter``.
    * ``classifier_source == "llm"`` -> ``claude_llm``.
    * otherwise (``classifier_source == "heuristic"`` or anything else) ->
      ``dart_converter``.
    """
    cs = getattr(block, "classifier_source", "heuristic") or "heuristic"
    if cs == "extractor_hint":
        extra_source = ""
        if isinstance(block.attributes, dict):
            extra_source = str(block.attributes.get("source") or "").strip()
        if extra_source in {"pdfplumber", "pymupdf", "pdftotext"}:
            return extra_source
        up = getattr(block.raw, "extractor", "") or ""
        if up in {"pdfplumber", "pymupdf"}:
            return up
        if up == "pdftotext":
            return "pdftotext"
        return "dart_converter"
    if cs == "llm":
        return "claude_llm"
    # Heuristic / default fallback.
    return "dart_converter"


def _provenance_attrs(block: ClassifiedBlock) -> str:
    """Emit DART provenance attributes common to every wrapper template.

    Includes: ``data-dart-block-role``, ``data-dart-block-id``, optional
    ``data-dart-pages``, ``data-dart-confidence``, and ``data-dart-source``
    (Wave 19 restoration). All values are safe-escaped (they derive from
    enums, ints, floats, or a 16-hex string produced by the segmenter).

    Wave 20: the page attribute is now emitted in the plural form
    ``data-dart-pages`` to align with the Wave 8 multi-source contract
    (``DART/multi_source_interpreter.py`` has always emitted the plural
    form). When the segmenter stamped ``extra["page_label"]`` from the
    page-chrome detector, that label takes precedence over the raw
    form-feed-derived page number (the chrome detector's extracted
    page number is often the book's printed page, not the PDF's
    physical page, and is what downstream consumers want to cite).
    The attribute is omitted entirely when no page is known.
    """
    parts = [_role_attr(block), f'data-dart-block-id="{block.raw.block_id}"']
    page_label = ""
    extra = getattr(block.raw, "extra", None) or {}
    if isinstance(extra, dict):
        page_label = str(extra.get("page_label") or "").strip()
    page_value = page_label or (
        str(block.raw.page) if block.raw.page is not None else ""
    )
    if page_value:
        parts.append(f'data-dart-pages="{page_value}"')
    parts.append(f'data-dart-confidence="{block.confidence:.2f}"')
    parts.append(f'data-dart-source="{_data_dart_source_value(block)}"')
    return " ".join(parts)


def _section_class(existing: str = "") -> str:
    """Return a ``class="dart-section ..."`` attribute string.

    Wave 19 restoration: the ``dart-section`` semantic class is required
    by the ``dart_markers`` validator (``lib/validators/dart_markers.py``)
    on every top-level ``<section>`` / ``<article>`` / ``<aside>`` wrapper.
    When ``existing`` carries other classes (e.g. ``callout callout-info``)
    they're preserved — ``dart-section`` is prepended.
    """
    existing = (existing or "").strip()
    if existing:
        return f'class="dart-section {existing}"'
    return 'class="dart-section"'


# Roles whose template emits a leaf element (``<p>``, ``<span>``,
# ``<h1>``, ``<h3>``, ``<cite>``, ``<a>``, ``<li>``, ``<figcaption>``).
# Per the Wave 8 P2 rule, leaf elements never carry ``data-dart-*``
# attributes — the enclosing section/component wrapper does.
_WAVE19_LEAF_ROLES = frozenset({
    BlockRole.PARAGRAPH,
    BlockRole.SUBSECTION_HEADING,
    BlockRole.PAGE_BREAK,
    BlockRole.CITATION,
    BlockRole.CROSS_REFERENCE,
    BlockRole.FIGURE_CAPTION,
    BlockRole.BIBLIOGRAPHY_ENTRY,
    BlockRole.TITLE,
    BlockRole.AUTHOR_AFFILIATION,
    BlockRole.COPYRIGHT_LICENSE,
    BlockRole.KEYWORDS,
    BlockRole.BIBLIOGRAPHIC_METADATA,
})
# Note (Wave 21): LIST_ITEM is NOT listed as a leaf role because the
# fallback template :func:`_tpl_list_item` defensively emits a
# single-item ``<ul>`` / ``<ol>`` wrapper — wrappers carry the Wave 13
# provenance contract. Grouped list items render inside
# LIST_UNORDERED / LIST_ORDERED templates as attribute-free ``<li>``.


# Wave 21: map Unicode bullet variants to a CSS-friendly class so
# styling can distinguish "dot" (•) from "square" (▪) etc. while the
# semantic markup stays ``<ul>`` regardless.
_BULLET_CLASS_BY_CHAR = {
    "\u2022": "list-dot",       # • BULLET
    "\u00b7": "list-middot",    # · MIDDLE DOT
    "\u25aa": "list-square",    # ▪ BLACK SMALL SQUARE
    "\u25a0": "list-square",    # ■ BLACK SQUARE
    "\u25a1": "list-square-open",  # □ WHITE SQUARE
    "\u25cf": "list-disc",      # ● BLACK CIRCLE
    "\u25e6": "list-circle",    # ◦ WHITE BULLET
    "\u25cb": "list-circle",    # ○ WHITE CIRCLE
    "\u25b8": "list-triangle",  # ▸ BLACK SMALL RIGHT-POINTING TRIANGLE
    "\u25ba": "list-triangle",  # ► BLACK RIGHT-POINTING POINTER
    "-": "list-dash",
    "*": "list-asterisk",
}


def _stable_id(block: ClassifiedBlock, prefix: str) -> str:
    """Derive a document-unique id from ``block.raw.block_id``.

    The segmenter guarantees ``block_id`` uniqueness per run, so using it
    as the id-suffix avoids all duplicate-id collisions even when two
    headings carry identical text.
    """
    return f"{prefix}-{block.raw.block_id}"


def _escape(text: str | None) -> str:
    """Null-safe ``html.escape`` wrapper."""
    if text is None:
        return ""
    return html.escape(str(text))


def _render_list_items(items: Iterable[str]) -> str:
    """Render a list of strings as ``<li>`` children (escaped)."""
    return "".join(f"<li>{_escape(item)}</li>" for item in items)


def _split_lines(text: str) -> List[str]:
    """Return non-empty lines from a multi-line string, stripped."""
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_list_items(block: ClassifiedBlock) -> List[str]:
    """Pull list items from attributes, falling back to per-line split."""
    items = block.attributes.get("items")
    if isinstance(items, (list, tuple)) and items:
        return [str(item) for item in items]
    return _split_lines(block.raw.text)


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def _tpl_chapter_opener(block: ClassifiedBlock) -> str:
    """``<article role="doc-chapter">`` with schema.org Chapter microdata.

    Wave 15: emits ``id="chap-{number}"`` whenever a chapter number can
    be derived (explicit attribute or scraped from raw text) so the
    cross-reference resolver and document-level JSON-LD ``hasPart`` can
    target stable anchors. Falls back to the ``block_id``-based id when
    no number is available.
    """
    title = block.attributes.get("heading_text") or block.raw.text
    body = block.attributes.get("body_html", "")
    number = (
        block.attributes.get("chapter_number")
        or block.attributes.get("number")
    )
    if not number:
        # Scrape "Chapter N" from heading_text or raw text. The heuristic
        # classifier strips the "Chapter N:" prefix into heading_text, so
        # raw.text is usually where the number survives.
        import re as _re
        for candidate in (block.raw.text or "", block.attributes.get("heading_text") or ""):
            m = _re.search(r"\bchapter\s+(\d+)\b", candidate, _re.IGNORECASE)
            if m:
                number = m.group(1)
                break
    chap_id = f"chap-{number}" if number else _stable_id(block, "chap")
    return (
        f'<article {_section_class()} role="doc-chapter" id="{_escape(chap_id)}" itemscope '
        f'itemtype="https://schema.org/Chapter" {_provenance_attrs(block)}>'
        f"<header><h2 itemprop=\"name\">{_escape(title)}</h2></header>"
        f"{body}"
        f"</article>"
    )


def _numeric_section_anchor(block: ClassifiedBlock) -> Optional[str]:
    """Return a ``sec-{a}-{b}[-{c}...]`` id when the block carries a numeric hierarchy.

    Wave 30 Gap 2: TOC entries whose titles start with ``N.M`` emit
    ``<a href="#sec-N-M">`` targets (see ``_render_toc_entries``). Pre-
    Wave-30, section / subsection headings emitted a block-id-based
    stable anchor (``sec-<hash8>``) that never matched those TOC
    refs, so every dotted-numeric TOC link was dead.

    This helper extracts the dotted-numeric identifier from whichever
    attribute the classifier recorded:

    * ``dotted_number`` (Wave 25 Fix 5 — ``"1.1"``, ``"4.8.1.1"``, ...)
    * ``section_number`` / ``number`` (legacy ``_PAPER_SECTION_NUMBERED``
      path — ``"2"`` / ``"2.1"``).

    The emitted id replaces each ``.`` with ``-`` so the anchor matches
    the href format the cross-reference resolver + TOC renderer
    already use: ``#sec-4-8-1-1`` for heading ``4.8.1.1``.
    Alphabetic prefixes (``A2.3``) are kept verbatim (``sec-A2-3``).
    Returns ``None`` when no numeric hierarchy is available so callers
    can fall back to the stable-hash id.
    """
    import re as _re

    attrs = block.attributes or {}
    number: Optional[str] = None
    for key in ("dotted_number", "section_number", "number"):
        raw_val = attrs.get(key)
        if raw_val is None:
            continue
        candidate = str(raw_val).strip().rstrip(".")
        if candidate and _re.match(r"^[A-Za-z]?\d+(?:\.\d+){0,5}$", candidate):
            number = candidate
            break

    if number is None:
        # Last-resort: scrape a leading dotted-numeric prefix out of the
        # heading text itself (handles blocks where the classifier
        # didn't persist ``dotted_number`` but the raw text still
        # carries ``"1.1 Title"``).
        for candidate_text in (
            str(attrs.get("heading_text") or ""),
            str(block.raw.text or ""),
        ):
            m = _re.match(
                r"^\s*([A-Za-z]?\d+(?:\.\d+){0,5})\.?\s+\S",
                candidate_text,
            )
            if m:
                number = m.group(1)
                break

    if number is None:
        return None

    return "sec-" + number.replace(".", "-")


def _tpl_section_heading(block: ClassifiedBlock) -> str:
    """``<section role="region">`` with ``aria-labelledby`` h2.

    Wave 30 Gap 2: when the block carries a dotted-numeric hierarchy
    (``1``, ``1.1``, ``4.8.1.1``, ...) we emit ``id="sec-{A}-{B}..."``
    so TOC links of the form ``#sec-A-B`` resolve. Non-numbered
    section headings still fall back to the stable-hash id
    (``sec-<block_id>``) so every section has *some* anchor to jump
    to. The ``aria-labelledby`` target continues to use the
    block-id-derived suffix so it stays unique even when two
    sections share the same numbering (duplicate TOC entries in a
    PDF outline).
    """
    text = block.attributes.get("heading_text") or block.raw.text
    numeric_id = _numeric_section_anchor(block)
    sid = numeric_id if numeric_id else _stable_id(block, "sec")
    heading_sid = f"{_stable_id(block, 'sec')}-h"
    return (
        f'<section {_section_class()} id="{sid}" role="region" '
        f'aria-labelledby="{heading_sid}" {_provenance_attrs(block)}>'
        f'<h2 id="{heading_sid}">{_escape(text)}</h2>'
        f"</section>"
    )


def _tpl_subsection_heading(block: ClassifiedBlock) -> str:
    """Leaf ``<hN>`` with id — no section wrapper, no data-dart-* attrs.

    Wave 19: subsection headings are leaf nodes (Wave 8 P2 rule — attributes
    stop at the section/component wrapper level). The pre-Wave-13
    pipeline emitted just an ``<h3>`` here; reverting to that shape
    trims the ``<section>`` inflation the validator review surfaced.

    Wave 25 Fix 5: the dotted-numeric heading promoter emits a
    ``level`` attribute (3–6) derived from the hierarchy depth.
    The template picks the matching heading tag (``h3``/``h4``/
    ``h5``/``h6``) when present; legacy callers without ``level``
    see the original ``<h3>`` output.

    Wave 30 Gap 2: numbered subsection headings (``1.1``, ``4.8.1.1``)
    emit ``id="sec-{A}-{B}..."`` so TOC entries of the form
    ``#sec-A-B`` resolve. Non-numbered subsection headings keep the
    stable-hash id so downstream consumers (TOC validator,
    aria-labelledby targets) can always find a grounding anchor.
    """
    text = block.attributes.get("heading_text") or block.raw.text
    numeric_id = _numeric_section_anchor(block)
    sid = numeric_id if numeric_id else _stable_id(block, "sub")
    level = block.attributes.get("level")
    if isinstance(level, int) and 3 <= level <= 6:
        tag = f"h{level}"
    else:
        tag = "h3"
    return f'<{tag} id="{sid}">{_escape(text)}</{tag}>'


def _tpl_paragraph(block: ClassifiedBlock) -> str:
    """Leaf ``<p>`` with no data-dart-* attributes.

    Wave 19: strips provenance attributes from every ``<p>`` — per the
    Wave 8 P2 rule, attributes stop at the section/component wrapper
    level. The enclosing section/article carries the provenance; the
    paragraph is a leaf.
    """
    return f"<p>{_escape(block.raw.text)}</p>"


def _render_toc_entries(entries: list) -> str:
    """Render a list of ``{level, title, page, anchor?}`` dicts as nested ``<ol>``.

    Wave 18: when the extractor supplies structured TOC entries (via
    ``doc.toc``), the segmenter passes them through as
    ``block.attributes["entries"]``. Each entry carries:

    * ``level`` (1-based)
    * ``title``
    * ``page`` (1-indexed, for display + fallback anchor)
    * ``anchor`` (optional, ``#chap-N`` / ``#sec-N-M`` when known)

    We emit a single top-level ``<ol>`` and push deeper levels as nested
    ``<ol>`` children of the previous ``<li>``. Entries resolve to:

    * ``#chap-{N}`` when the title starts with ``"Chapter N"`` (matches
      the chapter-opener template's stable id).
    * ``#sec-{N}-{M}`` when the title starts with ``"N.M "`` (matches
      numbered section headings).
    * Fallback: ``#page-{N}`` so at minimum the screen reader can jump
      by page even when no block-level anchor exists.
    """
    if not entries:
        return ""

    import re as _re

    def _href_for(entry: dict) -> str:
        explicit = entry.get("anchor")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        title = str(entry.get("title") or "")
        chap = _re.match(r"^\s*Chapter\s+(\d+)\b", title, _re.IGNORECASE)
        if chap:
            return f"#chap-{chap.group(1)}"
        sec = _re.match(r"^\s*(\d+)\.(\d+)\b", title)
        if sec:
            return f"#sec-{sec.group(1)}-{sec.group(2)}"
        page = entry.get("page")
        if page:
            return f"#page-{page}"
        return "#"

    # Walk entries maintaining a level stack. Each level > 1 opens a
    # nested ``<ol>`` inside the previous ``<li>``; drops close nested
    # lists cleanly.
    pieces: List[str] = []
    level_stack: List[int] = []

    def _close_to(target_level: int) -> None:
        while level_stack and level_stack[-1] >= target_level:
            pieces.append("</li></ol>")
            level_stack.pop()

    pieces.append("<ol>")
    level_stack.append(1)
    first = True
    for entry in entries:
        level = max(1, int(entry.get("level", 1)))
        title = str(entry.get("title") or "")
        href = _href_for(entry)

        if first:
            pieces.append(
                f'<li><a href="{_escape(href)}">{_escape(title)}</a>'
            )
            first = False
            # Depth 1 li is now "open" — track at level 1 in the stack.
            level_stack[-1] = level
            continue

        if level > level_stack[-1]:
            # Open a new nested ol inside the previous (still-open) li.
            pieces.append("<ol>")
            level_stack.append(level)
            pieces.append(
                f'<li><a href="{_escape(href)}">{_escape(title)}</a>'
            )
        elif level == level_stack[-1]:
            # Close the previous li, open a new one at same level.
            pieces.append("</li>")
            pieces.append(
                f'<li><a href="{_escape(href)}">{_escape(title)}</a>'
            )
        else:
            # Dropping to a shallower level: close nested ols + the li
            # we're now leaving, and open a new sibling at target.
            while level_stack and level_stack[-1] > level:
                pieces.append("</li></ol>")
                level_stack.pop()
            # Now close the current li at the target level.
            pieces.append("</li>")
            pieces.append(
                f'<li><a href="{_escape(href)}">{_escape(title)}</a>'
            )
            if not level_stack:
                level_stack.append(level)
            else:
                level_stack[-1] = level

    # Close any still-open nested ols + the outermost one.
    while level_stack:
        pieces.append("</li></ol>")
        level_stack.pop()

    return "".join(pieces)


def _tpl_toc_nav(block: ClassifiedBlock) -> str:
    """``<nav role="doc-toc">`` wrapping a nested-ordered TOC list.

    Wave 18: when the block carries structured ``entries`` in its
    attributes (from :func:`DART.converter.block_segmenter.segment_extracted_document`
    seeded by ``doc.toc``), we render them as a nested ``<ol>`` /
    ``<li>`` tree with ``<a href="#...">`` anchors keyed by level.
    Entries resolve to ``#chap-N`` / ``#sec-N-M`` when the matching
    block emits those ids; otherwise fall back to ``#page-N``.

    Backward-compat: when no ``entries`` attribute is present (legacy
    callers), we fall back to the pre-Wave-18 flat ``<ol>`` rendering
    of per-line text items.
    """
    sid = _stable_id(block, "toc")
    entries = block.attributes.get("entries") if block.attributes else None
    if isinstance(entries, (list, tuple)) and entries:
        list_html = _render_toc_entries(list(entries))
    else:
        items = _extract_list_items(block)
        items_html = (
            _render_list_items(items)
            if items
            else f"<li>{_escape(block.raw.text)}</li>"
        )
        list_html = f"<ol>{items_html}</ol>"
    return (
        f'<nav role="doc-toc" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h2 id="{sid}-h">Contents</h2>'
        f"{list_html}"
        f"</nav>"
    )


def _tpl_page_break(block: ClassifiedBlock) -> str:
    """Leaf ``<span role="doc-pagebreak">`` with ``aria-label``.

    Wave 19: no data-dart-* attributes — span is a leaf, not a wrapper.

    Wave 25 Fix 6: when the block carries an integer ``page`` attribute
    (emitted by the segmenter for TOC-referenced pages), the span
    takes an ``id="page-N"`` anchor so TOC ``#page-N`` links can
    resolve. Legacy blocks without the numeric page stay anchor-less.
    """
    page_label = block.attributes.get("page") or block.raw.page or block.raw.text or ""
    label_text = f"page {page_label}" if page_label else "page break"
    id_attr = ""
    # Only emit an id when we know the integer page number. This is
    # what the TOC template writes into ``href="#page-N"``.
    page_num = block.attributes.get("page") or block.raw.page
    if isinstance(page_num, int) and page_num > 0:
        id_attr = f' id="page-{page_num}"'
    return (
        f'<span class="page-break"{id_attr} role="doc-pagebreak" '
        f'aria-label="{_escape(label_text)}"></span>'
    )


# ---------------------------------------------------------------------------
# Wave 21: list templates
# ---------------------------------------------------------------------------


def _render_list_li(item, ordered_parent: bool) -> str:
    """Render a single item as ``<li>{text}{<ul/ol>{sub_items}?}</li>``.

    Item shape (from :mod:`DART.converter.heuristic_classifier`):

    * ``text`` (required)
    * ``marker`` — informational, not emitted directly; the ``<ul>`` /
      ``<ol>`` semantic already encodes the bullet/number.
    * ``marker_type`` — ``"unordered"`` / ``"ordered"``.
    * ``sub_items`` (optional) — list of child items for nested lists.

    When a caller hands in a bare string (legacy ``attributes.items =
    ["foo", "bar"]`` shape from callers that predate the Wave 21
    dict form), it renders as a plain ``<li>`` with no nesting. This
    keeps the template compatible with the generic
    :func:`_extract_list_items` helper used by pre-Wave-21 templates
    (LEARNING_OBJECTIVES, KEY_TAKEAWAYS) + the xss invariants test.

    Per the Wave 8 P2 rule, ``<li>`` never carries ``data-dart-*`` —
    the enclosing ``<ul>`` / ``<ol>`` wrapper already does.
    """
    if isinstance(item, str):
        return f"<li>{_escape(item)}</li>"
    if not isinstance(item, dict):
        return f"<li>{_escape(str(item))}</li>"
    text = str(item.get("text") or "").strip()
    sub_items = item.get("sub_items") or []
    inner = _escape(text)

    if sub_items and isinstance(sub_items, (list, tuple)):
        # Determine child list type by scanning first nested marker_type.
        first = sub_items[0] if sub_items else None
        child_ordered = isinstance(first, dict) and first.get("marker_type") == "ordered"
        child_tag = "ol" if child_ordered else "ul"
        child_class = _bullet_class_for_items(sub_items, unordered=not child_ordered)
        class_attr = f' class="{child_class}"' if child_class else ""
        nested_lis = "".join(
            _render_list_li(sub, ordered_parent=child_ordered) for sub in sub_items
        )
        inner = f"{inner}<{child_tag}{class_attr}>{nested_lis}</{child_tag}>"

    return f"<li>{inner}</li>"


def _bullet_class_for_items(items, *, unordered: bool) -> str:
    """Pick a CSS-friendly list class when every item shares a bullet
    variant. Returns an empty string when markers are mixed, the
    items list is mixed string/dict, or the list is ordered (numbered
    lists don't need bullet-variant classes).
    """
    if not unordered or not items:
        return ""
    markers = set()
    for itm in items:
        if not isinstance(itm, dict):
            return ""
        markers.add(itm.get("marker"))
    markers.discard(None)
    if len(markers) != 1:
        return ""
    marker = next(iter(markers))
    if not isinstance(marker, str):
        return ""
    first_char = marker[:1]
    return _BULLET_CLASS_BY_CHAR.get(first_char, "")


def _tpl_list_unordered(block: ClassifiedBlock) -> str:
    """``<ul class="dart-section ...">`` wrapper with ``<li>`` children.

    Wave 21: emitted by the assembler's ``_group_consecutive_lists`` pass
    which folds a run of consecutive ``LIST_ITEM`` classified blocks
    (same ``marker_type``) into one synthesized ``LIST_UNORDERED`` block
    with ``attributes.items = [{text, marker, sub_items?}, ...]``.

    Per the Wave 19 P2 rule, the ``<ul>`` wrapper carries all
    provenance attributes (``class="dart-section"``, ``data-dart-*``);
    ``<li>`` children stay attribute-free. When every item in the list
    shares the same bullet glyph, a style-hint class (``list-dot`` /
    ``list-square`` / ...) is appended so CSS can reflect the author's
    original marker choice while keeping the markup semantic.

    Missing ``items`` (stray block fed straight to the template without
    going through grouping) falls back to a single-item list keyed off
    ``block.raw.text`` so the wrapper + provenance attrs still emit.
    """
    items = (block.attributes or {}).get("items") or []
    if not items:
        fallback_text = (block.raw.text or "").strip()
        if not fallback_text:
            return ""
        items = [{"text": fallback_text, "marker_type": "unordered"}]
    bullet_class = _bullet_class_for_items(items, unordered=True)
    class_snippet = (
        _section_class(bullet_class) if bullet_class else _section_class()
    )
    lis = "".join(_render_list_li(itm, ordered_parent=False) for itm in items)
    return f"<ul {class_snippet} {_provenance_attrs(block)}>{lis}</ul>"


def _tpl_list_ordered(block: ClassifiedBlock) -> str:
    """``<ol class="dart-section">`` wrapper with ``<li>`` children.

    Wave 21: emitted by the assembler's grouping pass for numbered /
    alpha / roman list runs. When the first item's marker is not
    ``"1."`` / ``"1)"`` / ``"a."`` / ``"a)"`` / ``"i."`` / ``"i)"``,
    the wrapper emits a ``start="N"`` attribute so screen readers +
    visual renderers preserve the authored numbering.

    Mirrors the unordered template's provenance emission and
    Wave 19 P2 leaf rule for ``<li>``.
    """
    items = (block.attributes or {}).get("items") or []
    if not items:
        fallback_text = (block.raw.text or "").strip()
        if not fallback_text:
            return ""
        items = [{"text": fallback_text, "marker_type": "ordered"}]
    # Derive ``start`` from the first marker when it's a non-default
    # start value. We only emit start= when it demonstrably differs
    # from 1 / a / i, to avoid cluttering the output.
    start_attr = ""
    first = items[0] if items else None
    first_marker = first.get("marker") if isinstance(first, dict) else None
    if isinstance(first_marker, str):
        import re as _re
        m = _re.match(r"^(\d+)", first_marker)
        if m:
            try:
                val = int(m.group(1))
                if val != 1:
                    start_attr = f' start="{val}"'
            except ValueError:  # pragma: no cover — regex guarantees int
                pass
    lis = "".join(_render_list_li(itm, ordered_parent=True) for itm in items)
    return (
        f"<ol {_section_class()}{start_attr} "
        f"{_provenance_attrs(block)}>{lis}</ol>"
    )


def _tpl_list_item(block: ClassifiedBlock) -> str:
    """Fallback: render a stray ``LIST_ITEM`` as a single-item ``<ul>``.

    This template should almost never fire — the assembler's grouping
    pass collapses consecutive ``LIST_ITEM`` blocks into
    ``LIST_UNORDERED`` / ``LIST_ORDERED``. Emitting a valid ``<ul>`` +
    ``<li>`` instead of a naked ``<li>`` keeps the HTML valid when a
    stray item escapes grouping (e.g. a list of exactly one item
    followed by a non-list block).
    """
    marker_type = (block.attributes or {}).get("marker_type") or "unordered"
    text = (block.attributes or {}).get("text") or block.raw.text
    item = {
        "text": text,
        "marker": (block.attributes or {}).get("marker"),
        "marker_type": marker_type,
    }
    sub_items = (block.attributes or {}).get("sub_items")
    if sub_items:
        item["sub_items"] = sub_items
    tag = "ol" if marker_type == "ordered" else "ul"
    bullet_class = (
        _bullet_class_for_items([item], unordered=True)
        if marker_type != "ordered"
        else ""
    )
    class_snippet = (
        _section_class(bullet_class) if bullet_class else _section_class()
    )
    return (
        f"<{tag} {class_snippet} {_provenance_attrs(block)}>"
        f"{_render_list_li(item, ordered_parent=(marker_type == 'ordered'))}"
        f"</{tag}>"
    )


# ---------------------------------------------------------------------------
# Educational
# ---------------------------------------------------------------------------


def _tpl_learning_objectives(block: ClassifiedBlock) -> str:
    """``schema.org/LearningResource`` section with ``<ul>`` of objectives."""
    items = _extract_list_items(block)
    items_html = _render_list_items(items) if items else f"<li>{_escape(block.raw.text)}</li>"
    sid = _stable_id(block, "lo")
    return (
        f'<section {_section_class()} itemscope '
        f'itemtype="https://schema.org/LearningResource" '
        f'aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h3 id="{sid}-h" itemprop="learningResourceType">Learning Objectives</h3>'
        f"<ul>{items_html}</ul>"
        f"</section>"
    )


def _tpl_key_takeaways(block: ClassifiedBlock) -> str:
    """``<aside role="doc-tip">`` for Key Takeaways."""
    sid = _stable_id(block, "kt")
    items = _extract_list_items(block)
    if items:
        content = f"<ul>{_render_list_items(items)}</ul>"
    else:
        content = f"<p>{_escape(block.raw.text)}</p>"
    return (
        f'<aside {_section_class()} role="doc-tip" '
        f'aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h4 id="{sid}-h">Key Takeaways</h4>'
        f"{content}"
        f"</aside>"
    )


def _tpl_activity(block: ClassifiedBlock) -> str:
    """``<section role="doc-example">`` for pedagogical activities."""
    sid = _stable_id(block, "act")
    title = block.attributes.get("title", "Activity")
    body = block.attributes.get("body_html") or f"<p>{_escape(block.raw.text)}</p>"
    return (
        f'<section {_section_class()} role="doc-example" '
        f'aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h4 id="{sid}-h">{_escape(title)}</h4>'
        f"{body}"
        f"</section>"
    )


def _tpl_self_check(block: ClassifiedBlock) -> str:
    """``<section role="doc-example">`` with ``aria-label="Self-check"``."""
    items = _extract_list_items(block)
    if items:
        questions = f"<ol>{_render_list_items(items)}</ol>"
    else:
        questions = f"<p>{_escape(block.raw.text)}</p>"
    return (
        f'<section {_section_class()} role="doc-example" '
        f'aria-label="Self-check" {_provenance_attrs(block)}>'
        f"<h4>Self-check</h4>"
        f"{questions}"
        f"</section>"
    )


def _tpl_example(block: ClassifiedBlock) -> str:
    """``<section role="doc-example">`` with worked example body."""
    sid = _stable_id(block, "ex")
    title = block.attributes.get("title", "")
    body = block.attributes.get("body_html") or f"<p>{_escape(block.raw.text)}</p>"
    heading = f"Example: {_escape(title)}" if title else "Example"
    return (
        f'<section {_section_class()} role="doc-example" '
        f'aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h4 id="{sid}-h">{heading}</h4>'
        f"{body}"
        f"</section>"
    )


def _tpl_exercise(block: ClassifiedBlock) -> str:
    """``<section role="doc-example">`` for student-facing exercises."""
    sid = _stable_id(block, "exe")
    title = block.attributes.get("title", "")
    body = block.attributes.get("body_html") or f"<p>{_escape(block.raw.text)}</p>"
    heading = f"Exercise: {_escape(title)}" if title else "Exercise"
    return (
        f'<section {_section_class()} role="doc-example" '
        f'aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h4 id="{sid}-h">{heading}</h4>'
        f"{body}"
        f"</section>"
    )


def _tpl_glossary_entry(block: ClassifiedBlock) -> str:
    """``<dl role="doc-glossary">`` with ``<dt>``/``<dd>`` microdata."""
    term = block.attributes.get("term", "")
    definition = block.attributes.get("definition", "")
    if not term or not definition:
        # Fallback: split on "—" / ":" / first " - " if available.
        text = block.raw.text
        for sep in (" — ", " - ", ": "):
            if sep in text:
                head, _, tail = text.partition(sep)
                term = term or head.strip()
                definition = definition or tail.strip()
                break
        if not term:
            term = block.raw.text
        if not definition:
            definition = ""
    return (
        f'<dl role="doc-glossary" {_provenance_attrs(block)}>'
        f'<dt itemprop="name">{_escape(term)}</dt>'
        f'<dd itemprop="description">{_escape(definition)}</dd>'
        f"</dl>"
    )


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _tpl_abstract(block: ClassifiedBlock) -> str:
    """``<section role="doc-abstract">`` with schema.org ``abstract`` itemprop."""
    sid = _stable_id(block, "abs")
    body = block.attributes.get("body_html") or f"<p>{_escape(block.raw.text)}</p>"
    return (
        f'<section {_section_class()} role="doc-abstract" '
        f'aria-labelledby="{sid}-h" '
        f'itemprop="abstract" {_provenance_attrs(block)}>'
        f'<h2 id="{sid}-h">Abstract</h2>'
        f"{body}"
        f"</section>"
    )


def _tpl_bibliography_entry(block: ClassifiedBlock) -> str:
    """``<li role="doc-endnote">`` with schema.org CreativeWork microdata.

    The assembler wraps the set in ``<ol role="doc-bibliography">``; each
    entry emits only the ``<li>`` node so the wrapping stays the
    assembler's concern.

    Wave 15: when no explicit ``number`` / ``ref_id`` attribute is set,
    scrape ``[N]`` or ``(N)`` from the raw text so the ``ref-N`` anchor
    matches the target the cross-reference resolver is looking for.
    """
    number = block.attributes.get("number") or block.attributes.get("ref_id")
    if not number:
        import re as _re
        match = _re.match(r"^\s*[\[\(](\d{1,3})[\]\)]\s+", block.raw.text or "")
        if match:
            number = match.group(1)
    anchor = f"ref-{number}" if number else f"ref-{block.raw.block_id}"
    # Wave 19: <li> is a leaf inside the <ol role="doc-bibliography"> the
    # assembler wraps around the entry set — no data-dart-* attributes
    # on the inner list item per the Wave 8 P2 rule.
    return (
        f'<li id="{anchor}" role="doc-endnote" itemscope '
        f'itemtype="https://schema.org/CreativeWork">'
        f'<cite itemprop="citation">{_escape(block.raw.text)}</cite>'
        f"</li>"
    )


def _tpl_footnote(block: ClassifiedBlock) -> str:
    """``<aside role="doc-footnote">`` with numeric marker + backref anchor."""
    number = block.attributes.get("number") or block.attributes.get("ref_id")
    sup = f"<sup>{_escape(number)}</sup> " if number else ""
    anchor_id = f"fn-{number}" if number else f"fn-{block.raw.block_id}"
    backref_target = f"#ref-fn{number}" if number else f"#ref-{block.raw.block_id}"
    return (
        f'<aside {_section_class()} id="{anchor_id}" role="doc-footnote" '
        f'{_provenance_attrs(block)}>'
        f"<p>{sup}{_escape(block.raw.text)} "
        f'<a href="{_escape(backref_target)}">\u21a9</a></p>'
        f"</aside>"
    )


def _tpl_citation(block: ClassifiedBlock) -> str:
    """Inline ``<cite>`` — leaf, no data-dart-* attrs (Wave 19)."""
    return f"<cite>{_escape(block.raw.text)}</cite>"


def _tpl_cross_reference(block: ClassifiedBlock) -> str:
    """Inline ``<a role="doc-cross-reference">`` anchor — leaf, no data-dart-*.

    Wave 19: anchor is an inline leaf; the enclosing section/paragraph
    wrapper carries the provenance. Target resolution done by
    :func:`DART.converter.cross_refs.resolve_cross_references`.
    """
    target_id = block.attributes.get("target_id", "")
    href = f"#{target_id}" if target_id else "#"
    return (
        f'<a href="{_escape(href)}" role="doc-cross-reference">'
        f"{_escape(block.raw.text)}</a>"
    )


# ---------------------------------------------------------------------------
# Content-rich
# ---------------------------------------------------------------------------


def _tpl_figure(block: ClassifiedBlock) -> str:
    """``<figure itemtype="schema.org/ImageObject">`` with optional ``<img>``.

    Accepts both ``src`` (legacy) and ``image_path`` (Wave 16 structured
    extractor output) for the image source. When no image source is
    available, the figure still renders with just a caption — useful
    for text-only figure references that survived pdftotext.

    Wave 17 rules:

    * Populated ``image_path`` + missing ``alt`` → emit ``alt=""
      role="presentation"`` (WCAG-acceptable decorative fallback)
      rather than omitting the attribute.
    * Empty caption → emit no ``<figcaption>`` at all. Never emit the
      literal placeholder string ``"(figure)"``; the schema.org
      microdata plus (where present) the ``alt`` attribute already
      describe the figure.
    """
    number = block.attributes.get("number")
    fig_id = f"fig-{number}" if number else f"fig-{block.raw.block_id}"
    # Prefer the explicit caption attribute; fall back to raw.text only
    # when it carries genuine content (segmenter leaves raw.text empty
    # when neither caption nor alt was available — gates the fallback).
    caption = block.attributes.get("caption") or ""
    if not caption and block.raw.text:
        caption = block.raw.text
    src = block.attributes.get("src") or block.attributes.get("image_path")
    alt = block.attributes.get("alt", "")

    img_html = ""
    if src:
        if alt:
            img_html = (
                f'<img src="{_escape(src)}" alt="{_escape(alt)}" '
                f'itemprop="contentUrl">'
            )
        else:
            # WCAG 2.2: decorative fallback when no alt is available.
            # ``role="presentation"`` tells AT to ignore the image so
            # absence of alt never surfaces as "image image image".
            img_html = (
                f'<img src="{_escape(src)}" alt="" role="presentation" '
                f'itemprop="contentUrl">'
            )

    caption_html = ""
    if caption.strip():
        caption_html = (
            f'<figcaption itemprop="caption">{_escape(caption)}</figcaption>'
        )

    return (
        f'<figure id="{fig_id}" itemscope '
        f'itemtype="https://schema.org/ImageObject" {_provenance_attrs(block)}>'
        f"{img_html}"
        f"{caption_html}"
        f"</figure>"
    )


def _tpl_figure_caption(block: ClassifiedBlock) -> str:
    """Standalone ``<figcaption>`` — leaf, no data-dart-* attrs (Wave 19).

    Rare — emitted when a caption detaches from its ``<figure>``. Per the
    Wave 8 P2 rule, captions are leaf nodes; the surrounding figure (or
    the enclosing section) carries the provenance.
    """
    return (
        f'<figcaption itemprop="caption">'
        f"{_escape(block.raw.text)}"
        f"</figcaption>"
    )


def _render_table_row(row: Iterable, tag: str = "td") -> str:
    """Render an iterable of cell values as a ``<tr>`` of ``<tag>`` cells."""
    cells = "".join(f"<{tag}>{_escape(cell)}</{tag}>" for cell in row)
    return f"<tr>{cells}</tr>"


def _render_table_head_row(row: Iterable) -> str:
    """Render a header row with ``scope="col"`` on every ``<th>``."""
    cells = "".join(
        f'<th scope="col">{_escape(cell)}</th>' for cell in row
    )
    return f"<tr>{cells}</tr>"


def _render_table_body_row(row: Iterable) -> str:
    """Render a body row with ``scope="row"`` on the first cell only.

    The first cell of each body row acts as a row header — this is the
    standard screen-reader-friendly pattern for data tables where each
    row represents a labelled record.
    """
    cells_list = list(row)
    if not cells_list:
        return "<tr></tr>"
    first = cells_list[0]
    rest = cells_list[1:]
    head = f'<th scope="row">{_escape(first)}</th>'
    tail = "".join(f"<td>{_escape(cell)}</td>" for cell in rest)
    return f"<tr>{head}{tail}</tr>"


def _tpl_table(block: ClassifiedBlock) -> str:
    """``<table role="grid">`` with caption, ``<thead>``, ``<tbody>``.

    Accepts both legacy and Wave-16 attribute shapes:

    * Legacy: ``headers`` (flat list or list-of-rows) + ``rows``
      (list-of-rows). Each row becomes a ``<tr>`` with ``<td>`` cells.
    * Wave 16 (structured extractor): ``header_rows`` (list-of-rows) +
      ``body_rows`` (list-of-rows). Header cells get ``scope="col"``;
      the first body cell of every row gets ``scope="row"`` so screen
      readers can associate row labels with every cell.

    When the attributes carry no structure (older heuristic calls
    without ``extra``), renders an empty ``<thead>`` / ``<tbody>`` and
    keeps the caption — guarantees we never regress below the Wave 13
    minimal shape.
    """
    tid = _stable_id(block, "tbl")
    title = (
        block.attributes.get("title")
        or block.attributes.get("caption")
        or block.raw.text
    )

    # Prefer the Wave 16 ``header_rows`` / ``body_rows`` pair.
    header_rows_attr = block.attributes.get("header_rows")
    body_rows_attr = block.attributes.get("body_rows")

    if header_rows_attr or body_rows_attr:
        header_rows: List = list(header_rows_attr or [])
        body_rows: List = list(body_rows_attr or [])
        thead_html = "".join(_render_table_head_row(r) for r in header_rows if r)
        tbody_html = "".join(_render_table_body_row(r) for r in body_rows if r)
    else:
        # Legacy path: ``headers`` + ``rows``.
        headers = block.attributes.get("headers") or []
        rows = block.attributes.get("rows") or []
        if headers and not isinstance(headers[0], (list, tuple)):
            header_row_list: List = [headers]
        else:
            header_row_list = list(headers)
        thead_html = "".join(_render_table_row(r, tag="th") for r in header_row_list)
        tbody_html = "".join(_render_table_row(r, tag="td") for r in rows)

    # Wave 18: surface the upstream extractor label on the <table> so
    # debug consumers can see whether the block came from pdfplumber or
    # PyMuPDF. Defaults to "pdfplumber" so legacy callers stay shape-
    # compatible.
    extractor_attr = ""
    source_label = block.attributes.get("source")
    if isinstance(source_label, str) and source_label.strip():
        extractor_attr = (
            f' data-dart-table-extractor="{_escape(source_label.strip())}"'
        )

    return (
        f'<table role="grid" aria-labelledby="{tid}-caption" '
        f"{_provenance_attrs(block)}{extractor_attr}>"
        f'<caption id="{tid}-caption">{_escape(title)}</caption>'
        f"<thead>{thead_html}</thead>"
        f"<tbody>{tbody_html}</tbody>"
        f"</table>"
    )


def _tpl_code_block(block: ClassifiedBlock) -> str:
    """``<pre role="region"><code>`` with optional caption paragraph.

    When ``attributes.caption`` is supplied, a sibling ``<p>`` carries the
    ``aria-labelledby`` target. When absent, the ``<pre>`` renders with
    no ``aria-labelledby`` to avoid a dangling id reference.
    """
    number = block.attributes.get("number")
    base_id = f"code-{number}" if number else f"code-{block.raw.block_id}"
    caption = block.attributes.get("caption")
    caption_html = ""
    aria_labelledby = ""
    if caption:
        caption_html = f'<p id="{base_id}-h"><strong>{_escape(caption)}</strong></p>'
        aria_labelledby = f' aria-labelledby="{base_id}-h"'
    return (
        f"{caption_html}"
        f'<pre role="region"{aria_labelledby} {_provenance_attrs(block)}>'
        f"<code>{_escape(block.raw.text)}</code>"
        f"</pre>"
    )


def _tpl_formula_math(block: ClassifiedBlock) -> str:
    """``<math>`` with ``<semantics>`` + ``<annotation>`` fallback.

    Wave 16: delegates to :mod:`DART.converter.mathml` so LaTeX-style
    delimiters (``$...$``, ``\\(...\\)``, ``\\[...\\]``) and plain
    equation-on-a-line patterns (``E = mc^2``) all route through the
    same minimal-accessible MathML shape. The ``<annotation>`` arm
    always carries the raw plain-text source so screen readers without
    a MathML reader still narrate the formula.

    Provenance attributes survive on the ``<math>`` wrapper so the
    downstream validator can trace the block even inside a structural
    element.
    """
    from DART.converter.mathml import detect_formulas, render_mathml

    # Prefer explicit attribute-carried body (LLM classifier can emit
    # a clean ``{"body": "..."}``); fall back to the raw text.
    explicit_body = block.attributes.get("body") or block.attributes.get("latex")
    fallback = block.attributes.get("fallback") or block.raw.text

    if explicit_body:
        body = str(explicit_body)
    else:
        detected = detect_formulas(block.raw.text or "")
        body = detected[0].body if detected else (block.raw.text or "")

    mathml = render_mathml(body, fallback=fallback, display="block")

    # Inject provenance attributes onto the outer ``<math>``.
    return mathml.replace(
        'display="block">',
        f'display="block" {_provenance_attrs(block)}>',
        1,
    )


def _tpl_blockquote(block: ClassifiedBlock) -> str:
    """``<blockquote>`` with optional ``cite`` URL + ``<footer>`` attribution."""
    cite_url = block.attributes.get("cite_url") or block.attributes.get("cite")
    attribution = block.attributes.get("attribution", "")
    cite_attr = f' cite="{_escape(cite_url)}"' if cite_url else ""
    footer_html = (
        f"<footer>\u2014 {_escape(attribution)}</footer>" if attribution else ""
    )
    return (
        f"<blockquote{cite_attr} {_provenance_attrs(block)}>"
        f"<p>{_escape(block.raw.text)}</p>"
        f"{footer_html}"
        f"</blockquote>"
    )


def _tpl_epigraph(block: ClassifiedBlock) -> str:
    """``<section role="doc-epigraph">`` with attribution footer."""
    attribution = block.attributes.get("attribution", "")
    footer_html = (
        f"<footer>\u2014 {_escape(attribution)}</footer>" if attribution else ""
    )
    return (
        f'<section {_section_class()} role="doc-epigraph" '
        f'{_provenance_attrs(block)}>'
        f"<blockquote>"
        f"<p>{_escape(block.raw.text)}</p>"
        f"{footer_html}"
        f"</blockquote>"
        f"</section>"
    )


def _tpl_pullquote(block: ClassifiedBlock) -> str:
    """``<aside role="doc-pullquote">`` with decorative styling class."""
    return (
        f'<aside {_section_class("pullquote")} role="doc-pullquote" '
        f'{_provenance_attrs(block)}>'
        f"<blockquote>"
        f"<p>{_escape(block.raw.text)}</p>"
        f"</blockquote>"
        f"</aside>"
    )


# ---------------------------------------------------------------------------
# Notice / Callout
# ---------------------------------------------------------------------------


def _callout(
    block: ClassifiedBlock,
    *,
    label: str,
    dpub_role: str,
    css_class: str,
    icon: str,
    sid_prefix: str,
) -> str:
    """Shared builder for the four callout templates.

    Each callout follows the Unicode-icon + sr-only label + ``aria-labelledby``
    pattern so screen readers announce the callout type while sighted
    readers see a compact glyph.
    """
    sid = _stable_id(block, sid_prefix)
    title = block.attributes.get("title", "")
    body = block.attributes.get("body_html") or f"<p>{_escape(block.raw.text)}</p>"
    return (
        f'<aside role="{dpub_role}" {_section_class(css_class)} '
        f'aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h4 id="{sid}-h">'
        f'<span aria-hidden="true">{icon}</span> '
        f'<span class="sr-only">{_escape(label)}:</span> '
        f"{_escape(title)}"
        f"</h4>"
        f"{body}"
        f"</aside>"
    )


def _tpl_callout_info(block: ClassifiedBlock) -> str:
    """Info callout: ``role="note"``, \u24d8 icon."""
    return _callout(
        block,
        label="Information",
        dpub_role="note",
        css_class="callout callout-info",
        icon="\u24d8",
        sid_prefix="ci",
    )


def _tpl_callout_warning(block: ClassifiedBlock) -> str:
    """Warning callout: ``role="doc-notice"``, \u26a0 icon."""
    return _callout(
        block,
        label="Warning",
        dpub_role="doc-notice",
        css_class="callout callout-warning",
        icon="\u26a0",
        sid_prefix="cw",
    )


def _tpl_callout_tip(block: ClassifiedBlock) -> str:
    """Tip callout: ``role="doc-tip"``, lightbulb icon."""
    return _callout(
        block,
        label="Tip",
        dpub_role="doc-tip",
        css_class="callout callout-tip",
        icon="\U0001f4a1",
        sid_prefix="ct",
    )


def _tpl_callout_danger(block: ClassifiedBlock) -> str:
    """Danger callout: ``role="doc-notice"``, \u26d4 icon."""
    return _callout(
        block,
        label="Danger",
        dpub_role="doc-notice",
        css_class="callout callout-danger",
        icon="\u26d4",
        sid_prefix="cd",
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _tpl_title(block: ClassifiedBlock) -> str:
    """Leaf ``<h1 itemprop="name">`` — no data-dart-* attrs (Wave 19).

    Normally the assembler emits the main H1; this template renders
    embedded TITLE blocks, which per the Wave 8 P2 rule are leaf nodes.
    """
    return (
        f'<h1 itemprop="name">'
        f"{_escape(block.raw.text)}"
        f"</h1>"
    )


def _tpl_author_affiliation(block: ClassifiedBlock) -> str:
    """Leaf ``<p>`` with ``schema.org/Person`` microdata — no data-dart-*."""
    name = block.attributes.get("name")
    affiliation = block.attributes.get("affiliation")
    if name or affiliation:
        body = ""
        if name:
            body += f'<span itemprop="name">{_escape(name)}</span>'
        if affiliation:
            body += f' <span itemprop="affiliation">{_escape(affiliation)}</span>'
    else:
        body = f'<span itemprop="name">{_escape(block.raw.text)}</span>'
    return (
        f'<p class="authors" itemprop="author" itemscope '
        f'itemtype="https://schema.org/Person">'
        f"{body}"
        f"</p>"
    )


def _tpl_copyright_license(block: ClassifiedBlock) -> str:
    """Leaf ``<p itemprop="license">`` — no data-dart-* attrs (Wave 19)."""
    return (
        f'<p class="license" itemprop="license">'
        f"{_escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_keywords(block: ClassifiedBlock) -> str:
    """Leaf ``<p itemprop="keywords">`` — no data-dart-* attrs (Wave 19)."""
    return (
        f'<p class="keywords" itemprop="keywords">'
        f"{_escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_bibliographic_metadata(block: ClassifiedBlock) -> str:
    """Leaf fallback ``<p>`` for bibliographic metadata — no data-dart-*."""
    return (
        f'<p class="biblio-metadata">'
        f"{_escape(block.raw.text)}"
        f"</p>"
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: Dict[BlockRole, Callable[[ClassifiedBlock], str]] = {
    # Structural
    BlockRole.CHAPTER_OPENER: _tpl_chapter_opener,
    BlockRole.SECTION_HEADING: _tpl_section_heading,
    BlockRole.SUBSECTION_HEADING: _tpl_subsection_heading,
    BlockRole.PARAGRAPH: _tpl_paragraph,
    BlockRole.TOC_NAV: _tpl_toc_nav,
    BlockRole.PAGE_BREAK: _tpl_page_break,
    # Wave 21: lists
    BlockRole.LIST_UNORDERED: _tpl_list_unordered,
    BlockRole.LIST_ORDERED: _tpl_list_ordered,
    BlockRole.LIST_ITEM: _tpl_list_item,
    # Educational
    BlockRole.LEARNING_OBJECTIVES: _tpl_learning_objectives,
    BlockRole.KEY_TAKEAWAYS: _tpl_key_takeaways,
    BlockRole.ACTIVITY: _tpl_activity,
    BlockRole.SELF_CHECK: _tpl_self_check,
    BlockRole.EXAMPLE: _tpl_example,
    BlockRole.EXERCISE: _tpl_exercise,
    BlockRole.GLOSSARY_ENTRY: _tpl_glossary_entry,
    # Reference
    BlockRole.ABSTRACT: _tpl_abstract,
    BlockRole.BIBLIOGRAPHY_ENTRY: _tpl_bibliography_entry,
    BlockRole.FOOTNOTE: _tpl_footnote,
    BlockRole.CITATION: _tpl_citation,
    BlockRole.CROSS_REFERENCE: _tpl_cross_reference,
    # Content-rich
    BlockRole.FIGURE: _tpl_figure,
    BlockRole.FIGURE_CAPTION: _tpl_figure_caption,
    BlockRole.TABLE: _tpl_table,
    BlockRole.CODE_BLOCK: _tpl_code_block,
    BlockRole.FORMULA_MATH: _tpl_formula_math,
    BlockRole.BLOCKQUOTE: _tpl_blockquote,
    BlockRole.EPIGRAPH: _tpl_epigraph,
    BlockRole.PULLQUOTE: _tpl_pullquote,
    # Notice
    BlockRole.CALLOUT_INFO: _tpl_callout_info,
    BlockRole.CALLOUT_WARNING: _tpl_callout_warning,
    BlockRole.CALLOUT_TIP: _tpl_callout_tip,
    BlockRole.CALLOUT_DANGER: _tpl_callout_danger,
    # Metadata
    BlockRole.TITLE: _tpl_title,
    BlockRole.AUTHOR_AFFILIATION: _tpl_author_affiliation,
    BlockRole.COPYRIGHT_LICENSE: _tpl_copyright_license,
    BlockRole.KEYWORDS: _tpl_keywords,
    BlockRole.BIBLIOGRAPHIC_METADATA: _tpl_bibliographic_metadata,
}


def render_block(block: ClassifiedBlock) -> str:
    """Render a classified block using the role's template.

    Raises ``KeyError`` if the role has no registered template — this is
    intentional: the registry is exhaustive by construction, so a miss
    indicates a coding error, not a user-facing failure.
    """
    tpl = TEMPLATE_REGISTRY[block.role]
    return tpl(block)


__all__: Tuple[str, ...] = ("TEMPLATE_REGISTRY", "render_block")
