"""Phase 3 (Wave 12 minimal): role -> HTML template registry.

Every ``BlockRole`` has exactly one template function. Wave 12 ships
**placeholder** templates that preserve the current ``_raw_text_to_accessible_html``
HTML shape (``<section>`` / ``<h2>`` / ``<p>``) plus a new
``data-dart-block-role`` provenance attribute. Wave 13 will rewrite
each function to emit DPUB-ARIA + schema.org + richer structural HTML
(``<article role="doc-chapter">``, ``<blockquote>``, ``<figure>`` /
``<figcaption>``, ``<dl role="doc-glossary">``, etc.).

Templates receive a ``ClassifiedBlock`` and return an HTML string. All
inserted text is escaped via ``html.escape`` to prevent injection. No
template depends on classifier-source — a heuristic and an LLM block
with the same role render identically.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Callable, Dict

from DART.converter.block_roles import BlockRole, ClassifiedBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slug(text: str, prefix: str = "blk") -> str:
    """Return an id-safe slug. Falls back to block-hash style when empty."""
    cleaned = _SLUG_STRIP.sub("-", (text or "").lower()).strip("-")
    cleaned = cleaned[:60]
    if not cleaned:
        return prefix
    return cleaned


def _role_attr(block: ClassifiedBlock) -> str:
    """``data-dart-block-role="..."`` provenance attribute string."""
    return f'data-dart-block-role="{block.role.value}"'


def _provenance_attrs(block: ClassifiedBlock) -> str:
    """Emit DART provenance attributes common to every template."""
    parts = [_role_attr(block), f'data-dart-block-id="{block.raw.block_id}"']
    if block.raw.page is not None:
        parts.append(f'data-dart-page="{block.raw.page}"')
    parts.append(f'data-dart-confidence="{block.confidence:.2f}"')
    return " ".join(parts)


def _wrap_section(block: ClassifiedBlock, inner: str, heading: str, level: int = 2) -> str:
    """Standard section wrapper used by most structural templates.

    Wave 13 will replace this with role-specific containers (article,
    aside, section with DPUB-ARIA role etc.).
    """
    slug = _slug(heading, prefix=block.raw.block_id)
    safe_heading = html.escape(heading)
    tag = f"h{min(max(level, 2), 6)}"
    return (
        f'<section id="{slug}" aria-labelledby="{slug}-heading" {_provenance_attrs(block)}>'
        f'<{tag} id="{slug}-heading">{safe_heading}</{tag}>'
        f"{inner}"
        f"</section>"
    )


# ---------------------------------------------------------------------------
# Template functions (Wave 12 placeholders — minimal HTML shape)
# ---------------------------------------------------------------------------


def _tpl_chapter_opener(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role="doc-chapter" + schema.org Chapter microdata
    heading = block.attributes.get("heading_text") or block.raw.text
    return _wrap_section(block, inner="", heading=heading, level=2)


def _tpl_section_heading(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata
    heading = block.attributes.get("heading_text") or block.raw.text
    return _wrap_section(block, inner="", heading=heading, level=2)


def _tpl_subsection_heading(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata
    heading = block.attributes.get("heading_text") or block.raw.text
    return _wrap_section(block, inner="", heading=heading, level=3)


def _tpl_paragraph(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata
    return f"<p {_provenance_attrs(block)}>{html.escape(block.raw.text)}</p>"


def _tpl_toc_nav(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <nav role="navigation"> + aria-labelledby + <ol>
    return (
        f'<nav aria-label="Contents" {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</nav>"
    )


def _tpl_page_break(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <div role="doc-pagebreak" aria-label="Page N"/>
    return f'<div role="doc-pagebreak" {_provenance_attrs(block)}></div>'


def _tpl_learning_objectives(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata + <ol>
    return _wrap_section(
        block, inner=f"<p>{html.escape(block.raw.text)}</p>",
        heading="Learning Objectives",
    )


def _tpl_key_takeaways(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata + <ul>
    return _wrap_section(
        block, inner=f"<p>{html.escape(block.raw.text)}</p>",
        heading="Key Takeaways",
    )


def _tpl_activity(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata
    return _wrap_section(
        block, inner=f"<p>{html.escape(block.raw.text)}</p>",
        heading="Activity",
    )


def _tpl_self_check(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata
    return _wrap_section(
        block, inner=f"<p>{html.escape(block.raw.text)}</p>",
        heading="Self-Check",
    )


def _tpl_example(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role="doc-example" + schema.org microdata
    return _wrap_section(
        block, inner=f"<p>{html.escape(block.raw.text)}</p>",
        heading="Example",
    )


def _tpl_exercise(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role + schema.org microdata
    return _wrap_section(
        block, inner=f"<p>{html.escape(block.raw.text)}</p>",
        heading="Exercise",
    )


def _tpl_glossary_entry(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <dl role="doc-glossary"> + <dt>/<dd> pairs
    return (
        f'<div {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</div>"
    )


def _tpl_abstract(block: ClassifiedBlock) -> str:
    # Wave 13: expand with role="doc-abstract" + schema.org CreativeWork microdata
    return _wrap_section(block, inner="", heading="Abstract", level=2)


def _tpl_bibliography_entry(block: ClassifiedBlock) -> str:
    # Wave 13: expand with role="doc-endnote" + schema.org citation microdata
    return f'<li {_provenance_attrs(block)}>{html.escape(block.raw.text)}</li>'


def _tpl_footnote(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <aside role="doc-footnote"> + backref anchor
    return (
        f'<aside role="doc-footnote" {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</aside>"
    )


def _tpl_citation(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <cite> + schema.org citation microdata
    return f'<cite {_provenance_attrs(block)}>{html.escape(block.raw.text)}</cite>'


def _tpl_cross_reference(block: ClassifiedBlock) -> str:
    # Wave 15: resolve "See Chapter N" / "Figure M.N" into anchor links
    return f'<span {_provenance_attrs(block)}>{html.escape(block.raw.text)}</span>'


def _tpl_figure(block: ClassifiedBlock) -> str:
    # Wave 13/16: expand with <figure> + <figcaption> + schema.org ImageObject
    return (
        f'<figure {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</figure>"
    )


def _tpl_figure_caption(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <figcaption> inside parent <figure>
    return f'<figcaption {_provenance_attrs(block)}>{html.escape(block.raw.text)}</figcaption>'


def _tpl_table(block: ClassifiedBlock) -> str:
    # Wave 13/16: expand with <table><thead><tbody> + scope attributes + caption
    return (
        f'<div {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</div>"
    )


def _tpl_code_block(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <pre role="region"><code> + aria-labelledby
    return (
        f'<pre {_provenance_attrs(block)}>'
        f"<code>{html.escape(block.raw.text)}</code>"
        f"</pre>"
    )


def _tpl_formula_math(block: ClassifiedBlock) -> str:
    # Wave 16: expand with <math> + <annotation encoding="text/plain">
    return f'<p {_provenance_attrs(block)}><code>{html.escape(block.raw.text)}</code></p>'


def _tpl_blockquote(block: ClassifiedBlock) -> str:
    # Wave 13: expand with <blockquote cite="..."><p>...</p><footer>...</footer>
    return (
        f'<blockquote {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</blockquote>"
    )


def _tpl_epigraph(block: ClassifiedBlock) -> str:
    # Wave 13: expand with role="doc-epigraph" + attribution <footer>
    return (
        f'<blockquote role="doc-epigraph" {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</blockquote>"
    )


def _tpl_pullquote(block: ClassifiedBlock) -> str:
    # Wave 13: expand with role="doc-pullquote" + styled callout
    return (
        f'<aside role="doc-pullquote" {_provenance_attrs(block)}>'
        f"<p>{html.escape(block.raw.text)}</p>"
        f"</aside>"
    )


def _callout(block: ClassifiedBlock, label: str, dpub_role: str) -> str:
    # Wave 13: expand with Unicode icon + sr-only label + aria-labelledby
    return (
        f'<aside role="{dpub_role}" {_provenance_attrs(block)}>'
        f"<p><strong>{html.escape(label)}:</strong> {html.escape(block.raw.text)}</p>"
        f"</aside>"
    )


def _tpl_callout_info(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role="doc-notice" + icon + sr-only label
    return _callout(block, "Info", "doc-notice")


def _tpl_callout_warning(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role="doc-notice" + icon + sr-only label
    return _callout(block, "Warning", "doc-notice")


def _tpl_callout_tip(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role="doc-tip" + icon + sr-only label
    return _callout(block, "Tip", "doc-tip")


def _tpl_callout_danger(block: ClassifiedBlock) -> str:
    # Wave 13: expand with DPUB-ARIA role="doc-notice" + icon + sr-only label
    return _callout(block, "Danger", "doc-notice")


def _tpl_title(block: ClassifiedBlock) -> str:
    # Wave 13: expand with schema.org title microdata on assembler <h1>
    return f'<p {_provenance_attrs(block)}>{html.escape(block.raw.text)}</p>'


def _tpl_author_affiliation(block: ClassifiedBlock) -> str:
    # Wave 13: expand with schema.org Person microdata + <address>
    return (
        f'<p class="author-affiliation" {_provenance_attrs(block)}>'
        f"{html.escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_copyright_license(block: ClassifiedBlock) -> str:
    # Wave 15: expand with Dublin Core DC.rights meta emission
    return (
        f'<p class="copyright-license" {_provenance_attrs(block)}>'
        f"{html.escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_keywords(block: ClassifiedBlock) -> str:
    # Wave 13: expand with schema.org keywords microdata
    return (
        f'<p class="keywords" {_provenance_attrs(block)}>'
        f"<strong>Keywords:</strong> {html.escape(block.raw.text)}"
        f"</p>"
    )


def _tpl_bibliographic_metadata(block: ClassifiedBlock) -> str:
    # Wave 15: expand with Dublin Core meta emission on <head>
    return (
        f'<p class="bibliographic-metadata" {_provenance_attrs(block)}>'
        f"{html.escape(block.raw.text)}"
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


__all__ = ["TEMPLATE_REGISTRY", "render_block"]
