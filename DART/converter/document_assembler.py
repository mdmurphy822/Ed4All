"""Phase 4 (Wave 12 minimal): document assembly.

Wraps the per-block rendered HTML in the current DART document shell
(skip link, ``<header>`` with ``<h1>``, ``<main>``, ``<footer>``) plus a
``<aside role="complementary">`` metadata block populated from either
the caller-supplied ``metadata`` dict or classifier-collected
``COPYRIGHT_LICENSE`` / ``AUTHOR_AFFILIATION`` / ``BIBLIOGRAPHIC_METADATA``
blocks.

Wave 15 will replace this module with a decoration-heavy assembler that
emits Dublin Core ``<meta>`` tags, schema.org JSON-LD, an accessibility
summary, WCAG 2.2 AA CSS, and resolved cross-reference anchors. For
Wave 12 the assembler stays intentionally thin so the foundation stays
readable.
"""

from __future__ import annotations

import html
import logging
from typing import Dict, List

from DART.converter.block_roles import BlockRole, ClassifiedBlock
from DART.converter.block_templates import render_block

logger = logging.getLogger(__name__)


# Roles the assembler sweeps into the ``<aside>`` metadata block rather
# than inline in ``<main>``. Wave 15 may move these into ``<head>``
# Dublin Core / JSON-LD instead.
_METADATA_ASIDE_ROLES = {
    BlockRole.COPYRIGHT_LICENSE,
    BlockRole.AUTHOR_AFFILIATION,
    BlockRole.BIBLIOGRAPHIC_METADATA,
    BlockRole.KEYWORDS,
}


_DOC_CSS = """
  body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; max-width: 50em; margin: 0 auto; padding: 1em; color: #1a1a1a; }
  .skip-link { position: absolute; left: -9999px; top: auto; width: 1px; height: 1px; overflow: hidden; }
  .skip-link:focus { position: static; width: auto; height: auto; }
  h1 { font-size: 2em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h2 { font-size: 1.5em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
  h3 { font-size: 1.25em; margin-top: 1.5em; }
  section { margin-bottom: 1.5em; }
  p { margin: 0.8em 0; }
  aside[role="complementary"] { border-left: 3px solid #888; padding-left: 1em; margin-top: 2em; font-size: 0.95em; }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #e0e0e0; }
    h1, h2 { border-color: #555; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def _safe_title(title: str) -> str:
    pretty = title.replace("-", " ").replace("_", " ").title()
    return html.escape(pretty)


def _split_metadata(
    classified_blocks: List[ClassifiedBlock],
) -> tuple[List[ClassifiedBlock], List[ClassifiedBlock]]:
    """Separate body blocks from metadata-aside blocks."""
    body: List[ClassifiedBlock] = []
    aside: List[ClassifiedBlock] = []
    for block in classified_blocks:
        if block.role in _METADATA_ASIDE_ROLES:
            aside.append(block)
        else:
            body.append(block)
    return body, aside


def _render_body(body_blocks: List[ClassifiedBlock]) -> str:
    if not body_blocks:
        return ""
    return "\n".join(render_block(block) for block in body_blocks)


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
        parts.append(f"<p><strong>Authors:</strong> {html.escape(str(metadata['authors']))}</p>")
    if metadata.get("copyright"):
        parts.append(f"<p><strong>Copyright:</strong> {html.escape(str(metadata['copyright']))}</p>")
    if metadata.get("license"):
        parts.append(f"<p><strong>License:</strong> {html.escape(str(metadata['license']))}</p>")

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

    Wave 12 scope: single DOCTYPE + head + body skeleton, body renders
    every classified block via the template registry, metadata blocks
    (copyright, author affiliation, keywords, bibliographic metadata)
    sweep into a trailing ``<aside role="complementary">``.
    """
    metadata = metadata or {}
    body_blocks, aside_blocks = _split_metadata(classified_blocks)

    body_html = _render_body(body_blocks)
    aside_html = _render_aside(aside_blocks, metadata)
    safe_title = _safe_title(title)

    # Wave 15 will inject Dublin Core meta tags + JSON-LD + WCAG 2.2 CSS.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>{_DOC_CSS}  </style>
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


__all__ = ["assemble_html"]
