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
from typing import List

from DART.converter.block_roles import RawBlock

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


__all__ = ["segment_pdftotext_output"]
