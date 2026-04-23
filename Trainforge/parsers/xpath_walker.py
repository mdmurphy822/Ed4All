"""
XPath Walker — minimal absolute-xpath resolver for IMSCC HTML provenance.

Built on stdlib ``html.parser`` so the chunker can stamp an audit trail on
every chunk without pulling ``lxml`` into the dependency graph. Supports two
read paths:

1. ``find_heading_xpath(html, heading_text)``: returns the absolute xpath to
   the first ``<h1>``-``<h6>`` element whose text content matches
   ``heading_text`` (case-insensitive, whitespace-collapsed). Used to anchor
   sectioned chunks to the heading element that bounds their content.

2. ``resolve_xpath(html, xpath)``: returns the plain-text content of the
   element at ``xpath`` (i.e., the descendant-text concatenation the parser
   would produce for that subtree). Used by the round-trip test to verify
   ``chunk.text`` is recoverable from ``html_xpath + char_span``.

XPath format (locked):
    - Absolute, starts with ``/html/`` (or ``/<root>/`` if the document has
      no ``<html>`` shell).
    - Each step is ``tag[i]`` where ``i`` is the 1-based index among
      same-tag siblings, mirroring XPath 1.0 predicate semantics.
    - No ``//`` shortcuts, no wildcards, no namespaces.

The walker is deliberately minimal — it does not attempt to reproduce full
XPath 1.0. The round-trip contract is:

    element_text = resolve_xpath(raw_html, chunk.source.html_xpath)
    start, end = chunk.source.char_span
    assert element_text[start:end] starts with the first sentence of
           chunk.text (modulo whitespace normalization; see
           docs/compliance/audit-trail.md for tolerance details).
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import List, Optional, Tuple

# Tags whose content is intentionally dropped from the plain-text
# representation (matches HTMLTextExtractor in html_content_parser.py).
_DROP_TAGS = {"script", "style"}

# Void elements per HTML5 — no end tag, do not push to the stack.
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def _normalize(text: str) -> str:
    """Collapse whitespace the way the section-heading regex does."""
    return " ".join(text.split()).strip().lower()


class _XPathIndexer(HTMLParser):
    """Walk HTML, maintain an ancestor stack, record xpath for every element.

    After ``feed()``, ``self.elements`` holds one entry per opened element:
        (xpath, tag, attrs_dict, text_content)
    where ``text_content`` is the concatenated descendant text (same joining
    semantics as HTMLTextExtractor: whitespace-stripped tokens joined by a
    single space).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Stack of (tag, sibling_index_for_tag, text_parts) — the live
        # ancestor path from root to current open element. text_parts on
        # each frame accumulates data seen while that element is open.
        self._stack: List[Tuple[str, int, List[str]]] = []
        # Per-parent, per-tag sibling counter. Key is the depth of the
        # parent on the stack (0 = document root). Value is a dict
        # tag -> count-so-far.
        self._sibling_counts: List[dict] = [{}]
        # In-drop-tag flag (script/style): data is discarded.
        self._drop_depth = 0
        # Records: list of (xpath, tag, attrs_dict, joined_text).
        self.elements: List[Tuple[str, str, dict, str]] = []
        # Parallel list: element index -> xpath, for end-of-stream lookup.
        self._open_record_indices: List[int] = []

    # ------------------------------------------------------------------
    def _current_xpath(self) -> str:
        if not self._stack:
            return ""
        parts = []
        for tag, idx, _ in self._stack:
            parts.append(f"{tag}[{idx}]")
        return "/" + "/".join(parts)

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        tag = tag.lower()
        # Void element: record it with current xpath + sibling index, do not
        # push to the ancestor stack.
        parent_counts = self._sibling_counts[-1]
        parent_counts[tag] = parent_counts.get(tag, 0) + 1
        idx = parent_counts[tag]

        attrs_dict = {k: v for k, v in attrs if v is not None}

        if tag in _VOID_TAGS:
            # Void element xpath lives one level deeper than the current
            # stack frame but does not nest.
            if self._stack:
                parts = [f"{t}[{i}]" for t, i, _ in self._stack]
                parts.append(f"{tag}[{idx}]")
                xpath = "/" + "/".join(parts)
            else:
                xpath = f"/{tag}[{idx}]"
            self.elements.append((xpath, tag, attrs_dict, ""))
            return

        # Push: new frame starts its own sibling counter.
        self._stack.append((tag, idx, []))
        self._sibling_counts.append({})
        self._open_record_indices.append(len(self.elements))
        # Reserve the record; we fill ``text`` at end-tag time.
        self.elements.append(("", tag, attrs_dict, ""))
        if tag in _DROP_TAGS:
            self._drop_depth += 1

    def handle_endtag(self, tag: str):  # type: ignore[override]
        tag = tag.lower()
        # Tolerate sloppy HTML: only pop if the top of the stack matches.
        # If it doesn't, search for a matching frame; if none, ignore.
        if not self._stack:
            return
        if self._stack[-1][0] != tag:
            # Find nearest matching ancestor; close everything above it.
            match_idx = None
            for i in range(len(self._stack) - 1, -1, -1):
                if self._stack[i][0] == tag:
                    match_idx = i
                    break
            if match_idx is None:
                return
            # Close everything above match_idx (they'll get their xpath too).
            while len(self._stack) - 1 > match_idx:
                self._close_top()
        self._close_top()

    def _close_top(self) -> None:
        tag, idx, parts = self._stack[-1]
        rec_idx = self._open_record_indices.pop()
        xpath = self._current_xpath()
        joined = " ".join(parts)
        _, rec_tag, rec_attrs, _ = self.elements[rec_idx]
        self.elements[rec_idx] = (xpath, rec_tag, rec_attrs, joined)
        # Bubble this element's text up to its parent's text_parts, so
        # ancestors see the concatenated descendant text.
        self._stack.pop()
        self._sibling_counts.pop()
        if self._stack and rec_tag not in _DROP_TAGS:
            self._stack[-1][2].append(joined)
        if tag in _DROP_TAGS:
            self._drop_depth -= 1

    def handle_data(self, data: str):  # type: ignore[override]
        if self._drop_depth > 0:
            return
        stripped = data.strip()
        if not stripped or not self._stack:
            return
        self._stack[-1][2].append(stripped)

    def close(self):  # type: ignore[override]
        # Flush any un-closed tags so their xpaths are recorded.
        while self._stack:
            self._close_top()
        super().close()


def build_index(html: str) -> List[Tuple[str, str, dict, str]]:
    """Walk ``html`` and return [(xpath, tag, attrs, plain_text), ...].

    The list is in document order (start-tag order). ``plain_text`` on each
    entry is the concatenated descendant text, whitespace-stripped tokens
    joined by single spaces — matches ``HTMLTextExtractor.get_text()``.
    """
    indexer = _XPathIndexer()
    indexer.feed(html)
    indexer.close()
    return list(indexer.elements)


def find_heading_xpath(html: str, heading_text: str) -> Optional[str]:
    """Return absolute xpath to the first ``<hN>`` whose text matches.

    Matching is whitespace-collapsed and case-insensitive. Returns ``None``
    if no heading matches — caller falls back to the body-level xpath.
    """
    if not heading_text:
        return None
    target = _normalize(heading_text)
    # Also strip the "(part N)" suffix the chunker adds to split blocks.
    import re as _re
    target = _re.sub(r"\s*\(part\s+\d+\)\s*$", "", target).strip()
    if not target:
        return None
    for xpath, tag, _attrs, text in build_index(html):
        if tag not in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            continue
        if _normalize(text) == target:
            return xpath
    return None


def find_section_container_xpath(html: str, heading_text: str) -> Optional[str]:
    """Return xpath to the parent element that *contains* a heading.

    Sections in Courseforge HTML aren't wrapped in a semantic container —
    they're bounded implicitly by consecutive ``<hN>`` siblings. For the
    audit-trail round-trip to work, the xpath must anchor to a container
    whose descendant-text concatenation includes the whole section body,
    not just the heading glyph. That container is the heading's parent:
    typically ``<main>``, ``<article>``, ``<section>``, or ``<body>``.

    Returns ``None`` if no matching heading is found.
    """
    heading_xpath = find_heading_xpath(html, heading_text)
    if not heading_xpath:
        return None
    # Strip the last step: parent xpath is everything up to the last ``/``.
    parent = heading_xpath.rsplit("/", 1)[0]
    return parent or "/"


def find_body_xpath(html: str) -> str:
    """Return the xpath to ``<body>`` if present, else to the document root.

    Used as the fall-back anchor for items that have no section headings
    (whole-page chunks, no-sections items).
    """
    for xpath, tag, _attrs, _text in build_index(html):
        if tag == "body":
            return xpath
    # Degenerate document (no body): anchor to the first open element.
    for xpath, _tag, _attrs, _text in build_index(html):
        if xpath:
            return xpath
    return "/"


def resolve_xpath(html: str, xpath: str) -> Optional[str]:
    """Return the plain-text content of the element at ``xpath``.

    Returns ``None`` if the xpath is not found. The text uses the same
    whitespace-collapsed joining as ``HTMLTextExtractor.get_text()``, so a
    round-trip slice ``element_text[start:end]`` can be compared against
    ``chunk.text`` with the documented tolerance.
    """
    if not xpath:
        return None
    for element_xpath, _tag, _attrs, text in build_index(html):
        if element_xpath == xpath:
            return text
    return None
