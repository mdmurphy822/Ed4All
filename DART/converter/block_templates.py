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
from typing import Callable, Dict, Iterable, List, Tuple

from DART.converter.block_roles import BlockRole, ClassifiedBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _role_attr(block: ClassifiedBlock) -> str:
    """``data-dart-block-role="..."`` provenance attribute string."""
    return f'data-dart-block-role="{block.role.value}"'


def _provenance_attrs(block: ClassifiedBlock) -> str:
    """Emit DART provenance attributes common to every template.

    Includes: ``data-dart-block-role``, ``data-dart-block-id``, optional
    ``data-dart-page``, and ``data-dart-confidence``. All values are
    safe-escaped (they derive from enums, ints, floats, or a 16-hex
    string produced by the segmenter).
    """
    parts = [_role_attr(block), f'data-dart-block-id="{block.raw.block_id}"']
    if block.raw.page is not None:
        parts.append(f'data-dart-page="{block.raw.page}"')
    parts.append(f'data-dart-confidence="{block.confidence:.2f}"')
    return " ".join(parts)


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
        f'<article role="doc-chapter" id="{_escape(chap_id)}" itemscope '
        f'itemtype="https://schema.org/Chapter" {_provenance_attrs(block)}>'
        f"<header><h2 itemprop=\"name\">{_escape(title)}</h2></header>"
        f"{body}"
        f"</article>"
    )


def _tpl_section_heading(block: ClassifiedBlock) -> str:
    """``<section role="region">`` with ``aria-labelledby`` h2."""
    text = block.attributes.get("heading_text") or block.raw.text
    sid = _stable_id(block, "sec")
    return (
        f'<section id="{sid}" role="region" aria-labelledby="{sid}-h" '
        f"{_provenance_attrs(block)}>"
        f'<h2 id="{sid}-h">{_escape(text)}</h2>'
        f"</section>"
    )


def _tpl_subsection_heading(block: ClassifiedBlock) -> str:
    """``<section>`` wrapped ``<h3>`` with ``aria-labelledby``."""
    text = block.attributes.get("heading_text") or block.raw.text
    sid = _stable_id(block, "sub")
    return (
        f'<section id="{sid}" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h3 id="{sid}-h">{_escape(text)}</h3>'
        f"</section>"
    )


def _tpl_paragraph(block: ClassifiedBlock) -> str:
    """Plain ``<p>`` with provenance attributes."""
    return f"<p {_provenance_attrs(block)}>{_escape(block.raw.text)}</p>"


def _tpl_toc_nav(block: ClassifiedBlock) -> str:
    """``<nav role="navigation">`` wrapping an ordered TOC list."""
    items = _extract_list_items(block)
    items_html = _render_list_items(items) if items else f"<li>{_escape(block.raw.text)}</li>"
    sid = _stable_id(block, "toc")
    return (
        f'<nav role="navigation" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
        f'<h2 id="{sid}-h">Contents</h2>'
        f"<ol>{items_html}</ol>"
        f"</nav>"
    )


def _tpl_page_break(block: ClassifiedBlock) -> str:
    """``<span role="doc-pagebreak">`` with ``aria-label``."""
    page_label = block.attributes.get("page") or block.raw.page or block.raw.text or ""
    label_text = f"page {page_label}" if page_label else "page break"
    return (
        f'<span class="page-break" role="doc-pagebreak" '
        f'aria-label="{_escape(label_text)}" {_provenance_attrs(block)}></span>'
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
        f'<section itemscope itemtype="https://schema.org/LearningResource" '
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
        f'<aside role="doc-tip" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
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
        f'<section role="doc-example" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
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
        f'<section role="doc-example" aria-label="Self-check" {_provenance_attrs(block)}>'
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
        f'<section role="doc-example" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
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
        f'<section role="doc-example" aria-labelledby="{sid}-h" {_provenance_attrs(block)}>'
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
        f'<section role="doc-abstract" aria-labelledby="{sid}-h" '
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
    return (
        f'<li id="{anchor}" role="doc-endnote" itemscope '
        f'itemtype="https://schema.org/CreativeWork" {_provenance_attrs(block)}>'
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
        f'<aside id="{anchor_id}" role="doc-footnote" {_provenance_attrs(block)}>'
        f"<p>{sup}{_escape(block.raw.text)} "
        f'<a href="{_escape(backref_target)}">\u21a9</a></p>'
        f"</aside>"
    )


def _tpl_citation(block: ClassifiedBlock) -> str:
    """Inline ``<cite>`` with provenance attributes."""
    return f"<cite {_provenance_attrs(block)}>{_escape(block.raw.text)}</cite>"


def _tpl_cross_reference(block: ClassifiedBlock) -> str:
    """``<a role="doc-cross-reference">`` anchor; target resolved in Wave 15."""
    target_id = block.attributes.get("target_id", "")
    href = f"#{target_id}" if target_id else "#"
    return (
        f'<a href="{_escape(href)}" role="doc-cross-reference" '
        f"{_provenance_attrs(block)}>{_escape(block.raw.text)}</a>"
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
    """
    number = block.attributes.get("number")
    fig_id = f"fig-{number}" if number else f"fig-{block.raw.block_id}"
    caption = block.attributes.get("caption") or block.raw.text
    src = block.attributes.get("src") or block.attributes.get("image_path")
    alt = block.attributes.get("alt", "")
    img_html = ""
    if src:
        img_html = (
            f'<img src="{_escape(src)}" alt="{_escape(alt)}" itemprop="contentUrl">'
        )
    return (
        f'<figure id="{fig_id}" itemscope '
        f'itemtype="https://schema.org/ImageObject" {_provenance_attrs(block)}>'
        f"{img_html}"
        f'<figcaption itemprop="caption">{_escape(caption)}</figcaption>'
        f"</figure>"
    )


def _tpl_figure_caption(block: ClassifiedBlock) -> str:
    """Standalone ``<figcaption>`` — rare, emitted when a caption detaches.

    The provenance attributes still go on the ``<figcaption>`` itself so
    the downstream validator can trace the block even when it isn't
    nested in a ``<figure>``.
    """
    return (
        f'<figcaption itemprop="caption" {_provenance_attrs(block)}>'
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

    return (
        f'<table role="grid" aria-labelledby="{tid}-caption" {_provenance_attrs(block)}>'
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
        f'<section role="doc-epigraph" {_provenance_attrs(block)}>'
        f"<blockquote>"
        f"<p>{_escape(block.raw.text)}</p>"
        f"{footer_html}"
        f"</blockquote>"
        f"</section>"
    )


def _tpl_pullquote(block: ClassifiedBlock) -> str:
    """``<aside role="doc-pullquote">`` with decorative styling class."""
    return (
        f'<aside role="doc-pullquote" class="pullquote" {_provenance_attrs(block)}>'
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
        f'<aside role="{dpub_role}" class="{css_class}" '
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
    """``<h1 itemprop="name">`` — normally the assembler emits the main H1."""
    return (
        f'<h1 itemprop="name" {_provenance_attrs(block)}>'
        f"{_escape(block.raw.text)}"
        f"</h1>"
    )


def _tpl_author_affiliation(block: ClassifiedBlock) -> str:
    """``schema.org/Person`` microdata with inline name + affiliation."""
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
        f'itemtype="https://schema.org/Person" {_provenance_attrs(block)}>'
        f"{body}"
        f"</p>"
    )


def _tpl_copyright_license(block: ClassifiedBlock) -> str:
    """``<p itemprop="license">`` copyright / license notice."""
    return (
        f'<p class="license" itemprop="license" {_provenance_attrs(block)}>'
        f"{_escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_keywords(block: ClassifiedBlock) -> str:
    """``<p itemprop="keywords">`` keyword list."""
    return (
        f'<p class="keywords" itemprop="keywords" {_provenance_attrs(block)}>'
        f"{_escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_bibliographic_metadata(block: ClassifiedBlock) -> str:
    """Fallback bucket for misc bibliographic metadata lines."""
    return (
        f'<p class="biblio-metadata" {_provenance_attrs(block)}>'
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
