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
    """Phase 1 Wave-16 entry point: ``ExtractedDocument`` -> ``List[RawBlock]``.

    Produces the Wave 12 baseline paragraph blocks from ``doc.raw_text``
    (via :func:`segment_pdftotext_output`) and then appends dedicated
    structured blocks for every ``doc.tables`` entry and every
    ``doc.figures`` entry. Structured blocks carry:

    * ``extractor_hint`` set to the hinted :class:`BlockRole`
      (``TABLE`` or ``FIGURE``), which downstream classifiers honour
      by skipping text classification.
    * ``extra`` populated with the structured payload (rows / header /
      caption for tables; image_path / alt / caption for figures).
    * ``extractor`` set to ``"pdfplumber"`` or ``"pymupdf"`` so
      provenance attributes can distinguish structure-sourced blocks
      from pdftotext prose.

    Block IDs stay positional-hashed and unique across the combined
    sequence so the downstream assembler never produces duplicate
    ``id=`` attributes.
    """
    text_blocks = segment_pdftotext_output(doc.raw_text)

    structured: List[RawBlock] = []
    running_index = len(text_blocks)

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
        structured.append(
            RawBlock(
                text=normalised,
                block_id=block_id,
                page=table.page,
                bbox=table.bbox,
                extractor="pdfplumber",
                extractor_hint=BlockRole.TABLE,
                extra={
                    "header_rows": list(table.header_rows),
                    "body_rows": list(table.body_rows),
                    "caption": caption,
                },
            )
        )

    for figure in doc.figures:
        caption = figure.caption or ""
        alt = figure.alt_text or ""
        descriptor = caption or alt or "(figure)"
        normalised = _normalise_block_text(descriptor)
        block_id = _compute_block_id(normalised, running_index)
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

    combined = text_blocks + structured

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
