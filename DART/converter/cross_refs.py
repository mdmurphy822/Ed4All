"""Wave 15: post-assembly cross-reference resolution.

Rewrites in-text references to chapters, figures, sections, and citations
into real ``<a href="#...">`` anchors, but only when the target anchor
actually exists in the classified block list. Orphan references are left
as plain text so missing targets never surface as dead links.

The resolver runs as the last step inside
:func:`DART.converter.document_assembler.assemble_html`, operating on the
already-assembled HTML string plus the list of classified blocks (which
is the source of truth for what IDs exist: ``chap-N`` from
``CHAPTER_OPENER`` attributes, ``fig-N-M`` from ``FIGURE`` attributes,
``sec-N-M`` from ``SECTION_HEADING`` / ``SUBSECTION_HEADING`` attributes,
and ``ref-N`` from ``BIBLIOGRAPHY_ENTRY`` attributes).

Design rules (per wave spec):

* **Skip inside existing anchors.** Already-linked spans
  (``<a>See Chapter 1</a>``) are not double-wrapped. We detect them by
  segmenting the HTML at ``<a ... /a>`` boundaries and only running the
  rewriter on the non-anchor chunks.
* **Silently skip orphans.** References whose targets are not present in
  the block list remain untouched — no warnings, no broken links.
* **No new dependencies.** Pure regex substitution over the emitted
  HTML string.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

from DART.converter.block_roles import BlockRole, ClassifiedBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target collection
# ---------------------------------------------------------------------------

# Chapter heading attributes often carry ``chapter_number`` or ``number``;
# fall back to the assembler's stable id convention (``chap-{N}``) when
# the attribute is missing but the heading text encodes it.
_CHAPTER_NUMBER_RE = re.compile(r"\bchapter\s+(\d+)\b", re.IGNORECASE)
# Numbered figure / section attribute shapes: "3.2" or "3-2".
_NUMBER_DOT_RE = re.compile(r"^\s*(\d+)[\.\-](\d+)\s*$")


def _collect_targets(blocks: List[ClassifiedBlock]) -> Dict[str, Set[str]]:
    """Walk ``blocks`` and build a map of reference-kind -> target-id set.

    Keys are ``"chapter"``, ``"figure"``, ``"section"``, ``"citation"``
    so the rewriter can look up only the targets it cares about per
    match kind.
    """
    targets: Dict[str, Set[str]] = {
        "chapter": set(),
        "figure": set(),
        "section": set(),
        "citation": set(),
    }

    for block in blocks:
        attrs = block.attributes or {}

        if block.role == BlockRole.CHAPTER_OPENER:
            # Prefer explicit attribute, fall back to heading text scrape
            # AND the raw block text (heuristic classifier may have
            # stripped the "Chapter N:" prefix into heading_text).
            number = (
                attrs.get("chapter_number")
                or attrs.get("number")
            )
            if not number:
                candidates = [
                    str(attrs.get("heading_text") or ""),
                    str(block.raw.text or ""),
                ]
                for candidate in candidates:
                    m = _CHAPTER_NUMBER_RE.search(candidate)
                    if m:
                        number = m.group(1)
                        break
            if number:
                targets["chapter"].add(str(number))

        elif block.role == BlockRole.FIGURE:
            number = attrs.get("number")
            if number:
                m = _NUMBER_DOT_RE.match(str(number))
                if m:
                    targets["figure"].add(f"{m.group(1)}-{m.group(2)}")
                else:
                    # Single-number figure ("Figure 3") still reaches a
                    # ``fig-3`` anchor emitted by the template.
                    targets["figure"].add(str(number).strip())

        elif block.role in (BlockRole.SECTION_HEADING, BlockRole.SUBSECTION_HEADING):
            number = attrs.get("number") or attrs.get("section_number")
            if number:
                m = _NUMBER_DOT_RE.match(str(number))
                if m:
                    targets["section"].add(f"{m.group(1)}-{m.group(2)}")
                else:
                    targets["section"].add(str(number).strip())
            # Also scrape "2.1" / "2-1" from heading_text or raw text so
            # headings classified without an explicit number attribute
            # still register their canonical ID.
            heading = str(attrs.get("heading_text") or block.raw.text or "")
            hm = re.match(r"^\s*(\d+)\.(\d+)\b", heading)
            if hm:
                targets["section"].add(f"{hm.group(1)}-{hm.group(2)}")

        elif block.role == BlockRole.BIBLIOGRAPHY_ENTRY:
            number = attrs.get("number") or attrs.get("ref_id")
            if not number:
                # Scrape "[N]" or "(N)" from the raw bibliography text —
                # the heuristic classifier doesn't populate the number
                # attribute even when the regex matched.
                match = re.match(r"^\s*[\[\(](\d{1,3})[\]\)]\s+", block.raw.text or "")
                if match:
                    number = match.group(1)
            if number:
                targets["citation"].add(str(number).strip())

    return targets


# ---------------------------------------------------------------------------
# Rewriters
# ---------------------------------------------------------------------------

# Match phrases while excluding anything already inside an anchor.
# We strip out anchors via _split_on_anchors() before running these.
_CHAPTER_RE = re.compile(r"\b(See\s+)?Chapter\s+(\d+)\b")
_FIGURE_RE = re.compile(r"\bFigure\s+(\d+)[\.\-](\d+)\b")
_SECTION_RE = re.compile(r"\bSection\s+(\d+)[\.\-](\d+)\b")
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _rewrite_chunk(chunk: str, targets: Dict[str, Set[str]]) -> str:
    """Rewrite reference phrases in a non-anchor HTML chunk."""

    def _sub_chapter(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        number = match.group(2)
        if number not in targets["chapter"]:
            return match.group(0)
        anchor = f'<a href="#chap-{number}">Chapter {number}</a>'
        return f"{prefix}{anchor}"

    def _sub_figure(match: re.Match[str]) -> str:
        n, m = match.group(1), match.group(2)
        key = f"{n}-{m}"
        # Permit "Figure 3" single-number form in the target set too.
        if key not in targets["figure"] and n not in targets["figure"]:
            return match.group(0)
        # Separator preserved (either '.' or '-' in the source text).
        sep = match.group(0)[len(f"Figure {n}")]  # char between groups
        return f'<a href="#fig-{key}">Figure {n}{sep}{m}</a>'

    def _sub_section(match: re.Match[str]) -> str:
        n, m = match.group(1), match.group(2)
        key = f"{n}-{m}"
        if key not in targets["section"]:
            return match.group(0)
        sep = match.group(0)[len(f"Section {n}")]
        return f'<a href="#sec-{key}">Section {n}{sep}{m}</a>'

    def _sub_citation(match: re.Match[str]) -> str:
        number = match.group(1)
        if number not in targets["citation"]:
            return match.group(0)
        return f'<a href="#ref-{number}">[{number}]</a>'

    chunk = _CHAPTER_RE.sub(_sub_chapter, chunk)
    chunk = _FIGURE_RE.sub(_sub_figure, chunk)
    chunk = _SECTION_RE.sub(_sub_section, chunk)
    chunk = _CITATION_RE.sub(_sub_citation, chunk)
    return chunk


# ---------------------------------------------------------------------------
# Anchor-aware segmentation
# ---------------------------------------------------------------------------

# Capture-group regex: chunk between anchors (may be empty) or a full anchor
# element (anchor text included). Running the rewriter only on the non-
# anchor chunks avoids double-wrapping links that already exist.
_ANCHOR_SEGMENT = re.compile(r"(<a\b[^>]*>.*?</a>)", re.IGNORECASE | re.DOTALL)


def _resolve_in_body(body_html: str, targets: Dict[str, Set[str]]) -> str:
    """Apply the rewriters to every non-anchor chunk of ``body_html``."""
    segments = _ANCHOR_SEGMENT.split(body_html)
    rewritten: List[str] = []
    for segment in segments:
        if not segment:
            rewritten.append(segment)
            continue
        # If this chunk itself is a full <a>...</a> element, leave it.
        if segment.startswith("<a") and segment.lower().startswith("<a") \
                and segment.endswith("</a>"):
            rewritten.append(segment)
        else:
            rewritten.append(_rewrite_chunk(segment, targets))
    return "".join(rewritten)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_HEAD_TAIL_SPLIT = re.compile(r"(<head\b[^>]*>.*?</head>)", re.IGNORECASE | re.DOTALL)


def resolve_cross_references(
    html_text: str,
    classified_blocks: List[ClassifiedBlock],
) -> str:
    """Return ``html_text`` with cross-references rewritten to anchors.

    Behaviour (all guarded):

    * ``See Chapter N`` / ``Chapter N`` -> ``<a href="#chap-N">Chapter N</a>``
      only when a ``CHAPTER_OPENER`` with number ``N`` exists.
    * ``Figure N.M`` (or ``Figure N-M``) -> ``<a href="#fig-N-M">...</a>``
      only when a ``FIGURE`` with that number exists.
    * ``Section N.M`` -> ``<a href="#sec-N-M">...</a>`` only when a
      matching section heading exists.
    * ``[N]`` -> ``<a href="#ref-N">[N]</a>`` only when a
      ``BIBLIOGRAPHY_ENTRY`` with that number exists.

    The ``<head>`` block is passed through unchanged so metadata /
    ``<title>`` text isn't accidentally rewritten.
    """
    if not html_text or not classified_blocks:
        return html_text

    targets = _collect_targets(classified_blocks)
    if not any(targets.values()):
        return html_text

    # Split into [before, head, after] so we never rewrite inside <head>.
    parts = _HEAD_TAIL_SPLIT.split(html_text, maxsplit=1)
    if len(parts) == 3:
        before, head, after = parts
        return before + head + _resolve_in_body(after, targets)
    # Fallback: no <head> block found (partial document), rewrite the lot.
    return _resolve_in_body(html_text, targets)


__all__ = ["resolve_cross_references"]
