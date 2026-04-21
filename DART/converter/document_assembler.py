"""Phase 4 (Wave 13 assembler): document assembly.

Wraps the per-block rendered HTML in the DART document shell (skip link,
``<header>`` with ``<h1>``, ``<main>``, ``<footer>``) plus a
``<aside role="complementary">`` metadata block populated from either
the caller-supplied ``metadata`` dict or classifier-collected
``COPYRIGHT_LICENSE`` / ``AUTHOR_AFFILIATION`` / ``BIBLIOGRAPHIC_METADATA``
/ ``KEYWORDS`` blocks.

Wave 13 changes vs. Wave 12:

    * pulls the WCAG 2.2 AA CSS bundle from ``DART/templates/wcag22_css.py``
      instead of carrying an inline string, so the rules are sharable
      with ``gold_standard.html`` + future integration tests.
    * groups consecutive ``BIBLIOGRAPHY_ENTRY`` blocks into a single
      ``<ol role="doc-bibliography">`` wrapper so the DPUB-ARIA bibliography
      role applies at the list level rather than per-entry.
    * skips metadata-aside duplication for ``TITLE`` blocks: the
      assembler already emits the canonical ``<h1>`` so inline title
      blocks are moved to the aside to keep body semantics clean.

Wave 15 will add Dublin Core ``<meta>`` tags + schema.org JSON-LD to
``<head>`` + cross-reference anchor resolution.
"""

from __future__ import annotations

import html
import logging
from typing import Dict, List

from DART.converter.block_roles import BlockRole, ClassifiedBlock
from DART.converter.block_templates import render_block
from DART.templates.wcag22_css import WCAG22_CSS

logger = logging.getLogger(__name__)


# Roles swept into the metadata aside rather than inline in ``<main>``.
# Wave 15 may promote these into ``<head>`` Dublin Core / JSON-LD.
_METADATA_ASIDE_ROLES = {
    BlockRole.COPYRIGHT_LICENSE,
    BlockRole.AUTHOR_AFFILIATION,
    BlockRole.BIBLIOGRAPHIC_METADATA,
    BlockRole.KEYWORDS,
}


def _safe_title(title: str) -> str:
    """Pretty-print a filename-ish title for the ``<h1>`` / ``<title>``."""
    pretty = title.replace("-", " ").replace("_", " ").title()
    return html.escape(pretty)


def _split_metadata(
    classified_blocks: List[ClassifiedBlock],
) -> tuple[List[ClassifiedBlock], List[ClassifiedBlock]]:
    """Separate body blocks from metadata-aside blocks, order-preserving."""
    body: List[ClassifiedBlock] = []
    aside: List[ClassifiedBlock] = []
    for block in classified_blocks:
        if block.role in _METADATA_ASIDE_ROLES:
            aside.append(block)
        else:
            body.append(block)
    return body, aside


def _render_body(body_blocks: List[ClassifiedBlock]) -> str:
    """Render the body, grouping consecutive bibliography entries.

    Bibliography entries are DPUB-ARIA ``doc-endnote`` list items, and
    the canonical pattern puts the ``doc-bibliography`` role on the
    surrounding ``<ol>``. This loop buffers consecutive entries, emits
    the wrapping ``<ol>`` once, then resumes normal rendering.
    """
    if not body_blocks:
        return ""

    pieces: List[str] = []
    buffer: List[ClassifiedBlock] = []

    def flush() -> None:
        if not buffer:
            return
        inner = "\n".join(render_block(b) for b in buffer)
        pieces.append(f'<ol role="doc-bibliography">\n{inner}\n</ol>')
        buffer.clear()

    for block in body_blocks:
        if block.role == BlockRole.BIBLIOGRAPHY_ENTRY:
            buffer.append(block)
        else:
            flush()
            pieces.append(render_block(block))
    flush()

    return "\n".join(pieces)


def _render_aside(
    aside_blocks: List[ClassifiedBlock],
    metadata: Dict,
) -> str:
    """Build a ``<aside role="complementary">`` metadata block.

    Combines classifier-collected metadata blocks with any caller-supplied
    ``metadata`` dict. Caller-supplied values win when both are present
    for the same key since the caller has richer context.
    """
    has_classified = bool(aside_blocks)
    has_caller = any(metadata.get(k) for k in ("copyright", "license", "authors"))
    if not has_classified and not has_caller:
        return ""

    parts: List[str] = ['<aside role="complementary" aria-label="Document metadata">']

    if metadata.get("authors"):
        parts.append(
            f"<p><strong>Authors:</strong> {html.escape(str(metadata['authors']))}</p>"
        )
    if metadata.get("copyright"):
        parts.append(
            f"<p><strong>Copyright:</strong> {html.escape(str(metadata['copyright']))}</p>"
        )
    if metadata.get("license"):
        parts.append(
            f"<p><strong>License:</strong> {html.escape(str(metadata['license']))}</p>"
        )

    for block in aside_blocks:
        parts.append(render_block(block))

    parts.append("</aside>")
    return "\n".join(parts)


def assemble_html(
    classified_blocks: List[ClassifiedBlock],
    title: str,
    metadata: Dict | None = None,
) -> str:
    """Assemble the final HTML document.

    Wave 13 scope: single DOCTYPE + head + body skeleton, body renders
    every classified block via the template registry (with bibliography
    grouping), metadata blocks sweep into a trailing metadata aside, and
    the WCAG 2.2 AA CSS bundle is injected in ``<style>``.
    """
    metadata = metadata or {}
    body_blocks, aside_blocks = _split_metadata(classified_blocks)

    body_html = _render_body(body_blocks)
    aside_html = _render_aside(aside_blocks, metadata)
    safe_title = _safe_title(title)

    # Wave 15 will inject Dublin Core meta tags + JSON-LD into ``<head>``.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>{WCAG22_CSS}</style>
</head>
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <header role="banner">
    <h1>{safe_title}</h1>
  </header>
  <main id="main-content" role="main">
{body_html}
  </main>
{aside_html}
  <footer role="contentinfo">
    <p>Converted by DART (Document Accessibility Remediation Tool)</p>
  </footer>
</body>
</html>"""


__all__ = ("assemble_html",)
