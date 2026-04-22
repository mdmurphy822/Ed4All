"""Phase 1: block segmentation.

Splits raw ``pdftotext`` output into ``RawBlock`` instances on blank-line
boundaries (plus form-feed page boundaries when present). Performs
light-touch textual normalisation (soft-hyphen rejoining, whitespace
collapse) but does NOT classify. Classification is the next phase's job.

Stable block IDs are produced from a short hash of the block's
normalised text plus its positional index, so rerunning the segmenter
on the same input yields identical IDs. Downstream phases rely on this
for provenance attributes (``data-dart-block-id``) and caching.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, List

from DART.converter.block_roles import BlockRole, RawBlock

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from DART.converter.extractor import ExtractedDocument

logger = logging.getLogger(__name__)

# Form feed character signals a pdftotext page break in ``-layout`` mode.
_FORM_FEED = "\x0c"

# pdftotext represents broken-word soft hyphens as ``word-\n``. We
# rejoin them inside a block before computing its hash.
_SOFT_HYPHEN_JOIN = re.compile(r"(\w+)-\n(\w+)")

# Collapse runs of whitespace inside a block to single spaces so that
# visually identical content yields the same block hash regardless of
# layout-induced spacing differences.
_WHITESPACE = re.compile(r"\s+")


def _normalise_block_text(text: str) -> str:
    """Collapse whitespace + rejoin soft hyphens for hash / neighbour use.

    The normalised form is used both for ID computation and to populate
    ``RawBlock.text``. The raw whitespace layout is not preserved here
    because the classifier and templates operate on logical text, not
    column runs. Column-layout detection is a classifier concern and
    is deferred to Wave 14.
    """
    rejoined = _SOFT_HYPHEN_JOIN.sub(r"\1\2", text)
    collapsed = _WHITESPACE.sub(" ", rejoined).strip()
    return collapsed


def _compute_block_id(normalised_text: str, index: int) -> str:
    """Return a stable 16-hex block identifier.

    The identifier combines the block's normalised content hash with
    its positional index so two blocks that happen to carry identical
    text (e.g. a repeated header) still receive distinct IDs.
    """
    digest = hashlib.sha256(f"{index}:{normalised_text}".encode("utf-8"))
    return digest.hexdigest()[:16]


def _split_on_blank_lines(text: str) -> List[str]:
    """Split text on runs of blank lines, preserving non-empty blocks."""
    pieces = re.split(r"\n\s*\n+", text)
    return [piece for piece in pieces if piece.strip()]


def segment_pdftotext_output(raw_text: str) -> List[RawBlock]:
    """Phase 1 entry point: raw pdftotext -> ``List[RawBlock]``.

    Splits on blank-line boundaries within each form-feed-delimited
    page. Each produced ``RawBlock`` carries:

        * text        - whitespace-collapsed, soft-hyphen-joined content
        * block_id    - stable 16-hex positional hash
        * page        - 1-indexed page number when form-feeds are present
        * extractor   - ``"pdftotext"``
        * neighbors   - dict with ``prev`` / ``next`` sibling text

    If ``raw_text`` carries no form-feeds, page numbers are left as
    ``None`` and every block is treated as part of a single page.
    """
    if not raw_text:
        return []

    if _FORM_FEED in raw_text:
        page_texts = raw_text.split(_FORM_FEED)
    else:
        page_texts = [raw_text]

    blocks: List[RawBlock] = []
    for page_index, page_text in enumerate(page_texts):
        page_number = page_index + 1 if _FORM_FEED in raw_text else None
        for piece in _split_on_blank_lines(page_text):
            normalised = _normalise_block_text(piece)
            if not normalised:
                continue
            blocks.append(
                RawBlock(
                    text=normalised,
                    block_id=_compute_block_id(normalised, len(blocks)),
                    page=page_number,
                    extractor="pdftotext",
                )
            )

    # Populate neighbour context in a second pass so every block sees
    # its committed siblings. The classifier leans on this for
    # disambiguation (e.g. an ``Abstract`` heading followed by prose).
    for idx, block in enumerate(blocks):
        prev_text = blocks[idx - 1].text if idx > 0 else ""
        next_text = blocks[idx + 1].text if idx + 1 < len(blocks) else ""
        block.neighbors = {"prev": prev_text, "next": next_text}

    logger.debug("Segmented %d raw blocks from pdftotext output", len(blocks))
    return blocks


def segment_extracted_document(doc: "ExtractedDocument") -> List[RawBlock]:
    """Phase 1 Wave-16/18/20 entry point: ``ExtractedDocument`` -> ``List[RawBlock]``.

    Produces the Wave 12 baseline paragraph blocks from ``doc.raw_text``
    (via :func:`segment_pdftotext_output`) and then appends dedicated
    structured blocks for every ``doc.tables`` entry and every
    ``doc.figures`` entry. Structured blocks carry:

    * ``extractor_hint`` set to the hinted :class:`BlockRole`
      (``TABLE``, ``FIGURE``, or ``TOC_NAV``), which downstream
      classifiers honour by skipping text classification.
    * ``extra`` populated with the structured payload (rows / header /
      caption for tables; image_path / alt / caption for figures;
      entries list for TOC navigation).
    * ``extractor`` set to ``"pdfplumber"``, ``"pymupdf"`` so
      provenance attributes can distinguish structure-sourced blocks
      from pdftotext prose.

    Wave 18: when ``doc.toc`` is non-empty, a synthetic ``TOC_NAV``
    block is prepended at the start of the block list so the native
    PDF outline renders as a ``<nav role="doc-toc">`` block ahead of
    everything else. Table blocks also carry their upstream extractor
    label (``source="pdfplumber"`` vs ``source="pymupdf"``) into
    ``extra`` so the template can emit a ``data-dart-table-extractor``
    attribute for debuggability.

    Block IDs stay positional-hashed and unique across the combined
    sequence so the downstream assembler never produces duplicate
    ``id=`` attributes.
    """
    text_blocks = segment_pdftotext_output(doc.raw_text)

    # Wave 20: when page-chrome detection preserved a per-page page-
    # number label (e.g. a chrome line "<Book Title> 164" yielded
    # page_number_lines[164] = "... 164"), stamp that label into
    # every block on that page as ``extra["page_label"]`` so the
    # template emitter can surface ``data-dart-pages="164"`` even after
    # the chrome line itself has been stripped from the content stream.
    page_chrome = getattr(doc, "page_chrome", None)
    page_labels = (
        getattr(page_chrome, "page_number_lines", None) or {}
        if page_chrome is not None
        else {}
    )
    if page_labels:
        for block in text_blocks:
            if block.page is not None and block.page in page_labels:
                # Derive the numeric label — the value stored in
                # page_number_lines is the raw chrome line; we want
                # just the page number.
                from DART.converter.page_chrome import (
                    _normalise,
                    _strip_trailing_digits,
                )

                raw_line = page_labels[block.page]
                norm = _normalise(raw_line)
                _prefix, page_num = _strip_trailing_digits(norm)
                if page_num is not None:
                    block.extra["page_label"] = str(page_num)

    # Wave 18: build the TOC_NAV block first so it prepends the list.
    toc_blocks: List[RawBlock] = []
    raw_toc = getattr(doc, "toc", None) or []
    toc_referenced_pages: set = set()
    if raw_toc:
        entries: List[dict] = []
        for entry in raw_toc:
            title = getattr(entry, "title", "")
            if not title:
                continue
            level = int(getattr(entry, "level", 1) or 1)
            page = int(getattr(entry, "page", 0) or 0)
            entries.append({
                "level": max(1, level),
                "title": str(title),
                "page": max(1, page),
            })
            if page > 0:
                toc_referenced_pages.add(page)
        if entries:
            toc_text = " ".join(e["title"] for e in entries)
            normalised = _normalise_block_text(toc_text)
            toc_blocks.append(
                RawBlock(
                    text=normalised or "Contents",
                    block_id=_compute_block_id(
                        f"toc:{normalised}", 0
                    ),
                    page=entries[0]["page"] if entries else None,
                    extractor="pymupdf",
                    extractor_hint=BlockRole.TOC_NAV,
                    extra={"entries": entries},
                )
            )

    # Wave 25 Fix 6: emit synthetic ``PAGE_BREAK`` blocks at form-feed
    # boundaries for pages that are TOC targets. This populates the
    # ``#page-N`` anchors that the TOC template emits so ``<a
    # href="#page-5">`` links have real destinations. We only emit
    # for TOC-referenced pages because every physical page otherwise
    # would bloat the document with hundreds of anchors on a long
    # textbook.
    #
    # Each PAGE_BREAK block is inserted at the START of its target
    # page so the anchor resolves to the top of the page when the
    # reader clicks the TOC entry. The block is placed immediately
    # before the first text block on that page.
    page_break_blocks: List[RawBlock] = []
    if toc_referenced_pages:
        seen_pages: set = set()
        for block in text_blocks:
            if block.page is None or block.page in seen_pages:
                continue
            seen_pages.add(block.page)
            if block.page not in toc_referenced_pages:
                continue
            pb_text = f"page-{block.page}"
            page_break_blocks.append(
                RawBlock(
                    text=pb_text,
                    block_id=_compute_block_id(pb_text, -block.page),
                    page=block.page,
                    extractor="pdftotext",
                    extractor_hint=BlockRole.PAGE_BREAK,
                    extra={
                        "page": block.page,
                        "toc_target": True,
                    },
                )
            )

    structured: List[RawBlock] = []
    running_index = len(toc_blocks) + len(text_blocks)

    for table in doc.tables:
        header_text = " | ".join(" ".join(row) for row in table.header_rows if row)
        body_preview = " | ".join(" ".join(row) for row in table.body_rows[:3] if row)
        caption = table.caption or ""
        text_summary = (caption + "\n" if caption else "") + (
            header_text + "\n" + body_preview if (header_text or body_preview) else ""
        )
        if not text_summary.strip():
            text_summary = caption or "(table)"
        normalised = _normalise_block_text(text_summary)
        block_id = _compute_block_id(normalised, running_index)
        running_index += 1
        table_source = getattr(table, "source", "pdfplumber") or "pdfplumber"
        structured.append(
            RawBlock(
                text=normalised,
                block_id=block_id,
                page=table.page,
                bbox=table.bbox,
                extractor=table_source,
                extractor_hint=BlockRole.TABLE,
                extra={
                    "header_rows": list(table.header_rows),
                    "body_rows": list(table.body_rows),
                    "caption": caption,
                    "source": table_source,
                },
            )
        )

    for figure in doc.figures:
        caption = figure.caption or ""
        alt = figure.alt_text or ""
        # Descriptor picks the best available visible text (caption or
        # alt-text). When neither is present we use a ``figure-<page>``
        # hash seed so re-running the pipeline on the same PDF produces
        # stable block IDs without emitting the literal "(figure)"
        # string downstream (the template layer gates on empty raw.text
        # to drop the placeholder figcaption — see Wave 17).
        descriptor = caption or alt
        if descriptor:
            normalised = _normalise_block_text(descriptor)
        else:
            normalised = ""
        hash_seed = normalised or f"figure-page-{figure.page}-{running_index}"
        block_id = _compute_block_id(hash_seed, running_index)
        running_index += 1
        structured.append(
            RawBlock(
                text=normalised,
                block_id=block_id,
                page=figure.page,
                bbox=figure.bbox,
                extractor="pymupdf",
                extractor_hint=BlockRole.FIGURE,
                extra={
                    "image_path": figure.image_path or "",
                    "alt": alt,
                    "caption": caption,
                },
            )
        )

    # Wave 25 Fix 6: interleave PAGE_BREAK blocks at the start of each
    # TOC-referenced page so the TOC's ``#page-N`` anchors have real
    # destinations. Each PAGE_BREAK lands immediately before the first
    # text block on its page.
    text_blocks_with_pagebreaks: List[RawBlock] = []
    pagebreak_by_page = {pb.page: pb for pb in page_break_blocks}
    emitted_pages: set = set()
    for block in text_blocks:
        if (
            block.page is not None
            and block.page in pagebreak_by_page
            and block.page not in emitted_pages
        ):
            text_blocks_with_pagebreaks.append(pagebreak_by_page[block.page])
            emitted_pages.add(block.page)
        text_blocks_with_pagebreaks.append(block)

    combined = toc_blocks + text_blocks_with_pagebreaks + structured

    # Re-populate neighbour context across the combined sequence so
    # classifiers that use neighbours (the LLM classifier's prompt
    # builder) see the same window regardless of whether structured
    # blocks are present.
    for idx, block in enumerate(combined):
        prev_text = combined[idx - 1].text if idx > 0 else ""
        next_text = combined[idx + 1].text if idx + 1 < len(combined) else ""
        block.neighbors = {"prev": prev_text, "next": next_text}

    return combined


__all__ = ["segment_extracted_document", "segment_pdftotext_output"]
