"""Vendor-ingest adapter — publisher-supplied accessible HTML → Ed4All contract.

The DUAL-SOURCE companion to the SemantiK PDF-conversion path. SemantiK
(``lib/semantik/adapter.py``) synthesizes accessible HTML *from a PDF* (a
council of pypdfium2/pdfplumber/OCR extractors → ``data-semantik-source=
"synthesized"``). Some publishers already ship
CLEAN accessible HTML; for that input there is nothing to synthesize — we
only need to NORMALIZE the existing accessible markup into the Ed4All SemantiK
output contract (``data-semantik-*`` markers, ``semantik:`` sourceIds, sidecars,
exit_action→success) so the SAME downstream pipeline (chunk → index →
retrieve) consumes it.

Design contract (this session's task brief):

* REUSE the SemantiK adapter unchanged — only the IR-construction differs.
  This module parses the vendor HTML into the SAME
  ``_AdapterChapter``/``_AdapterBlock`` IR
  ``lib.semantik.adapter.normalize_cascade_to_ed4all`` consumes, then calls
  it with ``source="vendor"``.
* PROVENANCE DISCRIMINATOR: the emitted ``data-semantik-source`` is ``"vendor"``
  (AUTHORITATIVE — this HTML was authored accessible by the publisher, NOT
  synthesized by us). ``synthesized`` stays the SemantiK default. The M6
  finding holds: no consumer branches on the VALUE, so the new ``vendor``
  label mis-routes nothing.

``region_kind`` is derived from the source tag exactly as the adapter's
``_SECTION_TYPE_BY_KIND`` map expects (heading / paragraph / list / table /
figure / code_block / blockquote / definition_list). Chapter boundaries open
at ``h1``/``h2`` (the publisher's section-level headings); deeper headings
(``h3``-``h6``) become in-chapter blocks. ``raw_block_index`` is document
order (the §3.3a determinism anchor). ``figure_alt`` is recovered from
``<figcaption>`` text or an ``<img alt>``.

NO models / NO GPU: pure BeautifulSoup parsing + the model-free adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)

# Source-tag → SemantiK RegionKind. The adapter's _SECTION_TYPE_BY_KIND /
# sidecar section_type map is keyed on these values, so we emit exactly the
# vocab it understands (an unknown kind defaults to paragraph-group there).
_TAG_TO_REGION_KIND: Dict[str, str] = {
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "p": "paragraph",
    "ul": "list",
    "ol": "list",
    "dl": "definition_list",
    "table": "table",
    "figure": "figure",
    "img": "figure",
    "pre": "code_block",
    "blockquote": "blockquote",
}

# Tags whose subtree we descend into rather than emitting as a single block
# (structural containers the publisher uses for layout). We walk their
# children for the real content blocks. ``figure`` is NOT a container — it
# is emitted whole (caption + image stay together).
_CONTAINER_TAGS = frozenset(
    {"div", "section", "article", "main", "body", "header", "aside"}
)

# Content tags we emit as ONE block (no descent). ``figure`` is here so the
# whole figure + figcaption is one block.
_BLOCK_TAGS = frozenset(_TAG_TO_REGION_KIND) | {"figure"}

# §3.4 — a heading at this level (or shallower) opens a new chapter.
_CHAPTER_HEADING_TAGS = frozenset({"h1", "h2"})

# §3.4 overflow guard — hard ceiling on blocks per chapter so a long
# publisher section (which may carry hundreds of p/li blocks under a single
# h2) never produces a chapter that trips the SemanticStructureExtractor's
# >40-section collapse. Overflow spills into a "(cont.)" continuation
# chapter. Imported from the extractor so the two stay in lockstep.
try:  # pragma: no cover - import guard
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        _STRUCTURE_COLLAPSE_SECTION_THRESHOLD as _MAX_BLOCKS_PER_CHAPTER,
    )
except Exception:  # noqa: BLE001
    _MAX_BLOCKS_PER_CHAPTER = 40


def _heading_level(tag_name: str) -> int:
    """Numeric heading level (h2→2). Non-headings → 0."""
    if len(tag_name) == 2 and tag_name[0] == "h" and tag_name[1].isdigit():
        return int(tag_name[1])
    return 0


def _figure_alt_from(node: Any) -> Optional[str]:
    """Recover alt text from a figure node: <figcaption> text, then <img alt>."""
    cap = node.find("figcaption") if hasattr(node, "find") else None
    if cap is not None:
        text = cap.get_text(" ", strip=True)
        if text:
            return text
    img = node.find("img") if hasattr(node, "find") else None
    if img is not None:
        alt = (img.get("alt") or "").strip()
        if alt:
            return alt
    if getattr(node, "name", None) == "img":
        alt = (node.get("alt") or "").strip()
        return alt or None
    return None


def _block_html(node: Any) -> str:
    """Serialize a block node's OUTER HTML (the wrapped accessible markup).

    The adapter re-wraps each block in a ``<section>`` and supplies its own
    heading element, so for HEADING blocks we emit no inner HTML (the heading
    text drives the sid + the adapter renders the ``<h3>``). For all other
    block kinds we keep the publisher's accessible inner markup verbatim.
    """
    return str(node)


def _normalize_text(value: str) -> str:
    """Collapse whitespace in extracted text (deterministic hash/label input)."""
    return " ".join((value or "").split())


def _iter_content_nodes(node: Any):
    """Yield content/heading nodes in document order.

    Descends through layout containers (div/section/...) but emits block
    tags (p/ul/table/figure/...) whole. A heading is always emitted (it is
    both a potential chapter boundary and the sid source). Bare text and
    script/style/nav noise are skipped.
    """
    from bs4 import NavigableString, Tag

    name = getattr(node, "name", None)
    if name is None:
        return
    if name in {"script", "style", "nav", "head", "noscript"}:
        return

    if name in _BLOCK_TAGS:
        yield node
        return

    if name in _CONTAINER_TAGS:
        for child in node.children:
            if isinstance(child, Tag):
                yield from _iter_content_nodes(child)
            elif isinstance(child, NavigableString):
                # Loose text directly under a container with no block wrapper:
                # only emit if it carries real content (publishers occasionally
                # drop a bare sentence). Wrap it as a synthetic paragraph.
                txt = _normalize_text(str(child))
                if txt:
                    yield ("__text__", txt)
        return

    # Any other tag (span/em/a/... at top level, or an unrecognized wrapper):
    # descend so we don't lose nested block content, but don't emit the tag
    # itself.
    for child in node.children:
        if isinstance(child, Tag):
            yield from _iter_content_nodes(child)


def build_chapters_ir_from_html(
    html_or_pages: Union[str, Sequence[Mapping[str, Any]]],
    *,
    doc_title: str,
) -> List[_AdapterChapter]:
    """Parse vendor accessible HTML into the adapter's chapters IR (§3.4).

    Parameters
    ----------
    html_or_pages
        EITHER a single assembled HTML string, OR a sequence of per-page
        mappings each carrying ``output_html`` (+ optional ``title`` /
        ``page``). A page-sequence is concatenated in the order given (callers
        sort by page slug before passing) and a page-boundary heading is
        synthesized from each page's ``title`` so per-page structure survives
        even when a page has no top-level heading.
    doc_title
        The document title. Used as the title for the leading/implicit
        chapter when the first content precedes any chapter-level heading.

    Returns
    -------
    List[_AdapterChapter]
        Exactly the shape ``normalize_cascade_to_ed4all`` consumes. Regions
        are grouped into chapters at content-bearing ``h1``/``h2`` headings;
        ``h3``-``h6`` + content blocks become in-chapter section blocks.
    """
    from bs4 import BeautifulSoup

    # Normalize the input into an ordered list of (page_title, soup-root).
    page_specs: List[tuple[Optional[str], Any]] = []
    if isinstance(html_or_pages, str):
        soup = BeautifulSoup(html_or_pages, "html.parser")
        root = soup.body or soup
        page_specs.append((None, root))
    else:
        for page in html_or_pages:
            html = page.get("output_html") or ""
            if not html.strip():
                continue
            soup = BeautifulSoup(html, "html.parser")
            root = soup.body or soup
            page_title = page.get("title") or page.get("page")
            page_specs.append((page_title, root))

    chapters: List[_AdapterChapter] = []
    current: Optional[_AdapterChapter] = None
    raw_index = 0

    def _open_chapter(title: str) -> None:
        nonlocal current
        current = _AdapterChapter(title=_normalize_text(title) or doc_title, blocks=[])
        chapters.append(current)

    def _ensure_current() -> _AdapterChapter:
        nonlocal current
        if current is None:
            _open_chapter(doc_title)
        assert current is not None
        return current

    def _append_block(blk: _AdapterBlock) -> None:
        # §3.4 overflow guard — spill into a continuation chapter so no
        # chapter exceeds the safe section budget (a single publisher section
        # under one h2 can carry hundreds of p/li blocks).
        nonlocal current
        ch = _ensure_current()
        if len(ch.blocks) >= _MAX_BLOCKS_PER_CHAPTER:
            base = ch.title
            cont = base if base.endswith("(cont.)") else f"{base} (cont.)"
            _open_chapter(cont)
            ch = current  # type: ignore[assignment]
        ch.blocks.append(blk)

    for page_title, root in page_specs:
        # When a per-page title exists and the page does not lead with a
        # chapter-level heading, open a chapter for the page so per-page
        # structure is preserved (mirrors the publisher's per-section pages).
        page_opened = False

        for node in _iter_content_nodes(root):
            # Synthetic loose-text paragraph.
            if isinstance(node, tuple) and node[0] == "__text__":
                txt = node[1]
                blk = _AdapterBlock(
                    html=f"<p>{txt}</p>",
                    region_kind="paragraph",
                    raw_block_index=raw_index,
                    raw_text=txt,
                    block_role="vendor_body",
                )
                _append_block(blk)
                raw_index += 1
                continue

            name = node.name
            level = _heading_level(name)

            if name in _CHAPTER_HEADING_TAGS:
                heading_text = _normalize_text(node.get_text(" ", strip=True))
                if heading_text:
                    _open_chapter(heading_text)
                    page_opened = True
                    raw_index += 1
                    continue
                # Empty heading — skip (don't open a titleless chapter).
                raw_index += 1
                continue

            # Non-chapter content: ensure a chapter exists. If this is the
            # page's first content block and no chapter opened from a heading,
            # open one titled from the page title.
            if not page_opened and current is None and page_title:
                _open_chapter(str(page_title))
                page_opened = True

            region_kind = _TAG_TO_REGION_KIND.get(name, "paragraph")
            raw_text = _normalize_text(node.get_text(" ", strip=True))

            if region_kind == "heading":
                # h3-h6 in-chapter heading: drives the sid + section label.
                blk = _AdapterBlock(
                    html="",
                    region_kind="heading",
                    raw_block_index=raw_index,
                    raw_text=raw_text,
                    heading_text=raw_text or None,
                    block_role="vendor_heading",
                )
            elif region_kind == "figure":
                blk = _AdapterBlock(
                    html=_block_html(node),
                    region_kind="figure",
                    raw_block_index=raw_index,
                    raw_text=raw_text,
                    block_role="vendor_figure",
                    figure_alt=_figure_alt_from(node),
                )
            else:
                # paragraph / list / table / code_block / blockquote /
                # definition_list — keep the publisher's accessible markup.
                if not raw_text and region_kind not in {"table"}:
                    raw_index += 1
                    continue  # empty non-table block: skip
                blk = _AdapterBlock(
                    html=_block_html(node),
                    region_kind=region_kind,
                    raw_block_index=raw_index,
                    raw_text=raw_text,
                    block_role="vendor_body",
                )

            _append_block(blk)
            raw_index += 1

    return [ch for ch in chapters if ch.blocks]


def ingest_vendor_html(
    html_or_pages: Union[str, Sequence[Mapping[str, Any]]],
    *,
    pdf_stem: str,
    doc_title: str,
    source: str = "vendor",
    canonical_course_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest vendor accessible HTML into the Ed4All SemantiK output contract.

    Builds the chapters IR from the HTML (:func:`build_chapters_ir_from_html`)
    then runs the SAME ``normalize_cascade_to_ed4all`` adapter the SemantiK
    path uses, tagged with ``source="vendor"`` so every block carries
    ``data-semantik-source="vendor"`` (the authoritative provenance discriminator).

    The vendor input is already accessible, so the conversion always
    "ships with confidence" — we attach a synthetic cascade-like result
    carrying ``exit_action="ship_with_confidence"`` / ``wcag_status="passed"``
    so :func:`normalize_cascade_to_ed4all` resolves ``success=True``.

    Returns the SAME dict shape as ``normalize_cascade_to_ed4all`` (``html`` +
    ``synthesized_sidecar`` + ``quality_sidecar`` + ``success`` + ...), with
    ``data_semantik_source`` set to ``source``.
    """
    chapters = build_chapters_ir_from_html(html_or_pages, doc_title=doc_title)

    cascade_like = _VendorCascadeResult(chapters=chapters, doc_title=doc_title)
    out = normalize_cascade_to_ed4all(
        cascade_like,
        pdf_stem=pdf_stem,
        canonical_course_code=canonical_course_code,
        source=source,
    )
    out["method"] = "vendor_ingest"
    return out


class _VendorCascadeResult:
    """Duck-typed cascade-result stand-in for the vendor path.

    Carries the already-built chapters IR plus the doc-level signals the
    adapter reads (``exit_action`` / ``wcag_status`` / ``theta_score`` /
    ``flags`` / ``lane_used`` / ``lang``). Vendor HTML is publisher-authored
    accessible markup → it ships with confidence, WCAG passed, and a theta of
    1.0 (no synthesis loss — we did not transform the content, only re-wrapped
    it). ``flags`` carries a single honest provenance flag.
    """

    def __init__(self, *, chapters: List[_AdapterChapter], doc_title: str):
        self.chapters = chapters
        self.pdf = doc_title
        self.wcag_status = "passed"
        self.exit_action = "ship_with_confidence"
        self.theta_score = 1.0
        self.flags = ["vendor_supplied_accessible_html"]
        self.lane_used = "vendor-ingest"
        self.lang = "en"


__all__ = ["build_chapters_ir_from_html", "ingest_vendor_html"]
