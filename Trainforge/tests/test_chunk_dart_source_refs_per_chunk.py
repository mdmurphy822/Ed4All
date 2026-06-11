"""Regression: per-chunk DART provenance is char-span-scoped, not all-document.

The historical chunker scoped DART ``data-dart-block-id`` provenance per
section, then fell back to harvesting the WHOLE DOCUMENT's block list whenever
a heading-derived HTML slice carried no attribute (the common leaf-paragraph
layout). That gave most chunks of a multi-block page every block in the file,
so each chunk's FIRST ref — and the downstream "View original source" deep
link + cited page — was identical across the page (all pointing at the file's
first block).

Contract under test (the fix):
  - A chunk's ``dart_source_refs`` contains ONLY the blocks whose text genuinely
    overlaps the chunk's char span, in document order, with per-block pages.
  - Different chunks of the same multi-block document get DIFFERENT primary
    (first) block ids + pages — no all-document collapse.
  - The no-match case (a section whose own HTML slice carried no block) emits AT
    MOST ONE enclosing-section reference, never the whole-document list.
  - The old all-refs fallback shape (every chunk carrying every block) is gone.

Driven through ``Trainforge.chunker.chunk_content`` with a recording callback
and a synthetic multi-block fixture (no real course identifiers).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker import ChunkerContext, chunk_content  # noqa: E402
from Trainforge.chunker.helpers import (  # noqa: E402
    build_dart_block_offset_index,
    resolve_dart_refs_for_chunk,
)


def _recording_ctx(records: List[Dict[str, Any]]) -> ChunkerContext:
    def create_chunk(*, chunk_id, text, html, item, section_heading,
                     chunk_type, follows_chunk_id=None, position_in_module=0,
                     html_xpath=None, char_span=None, section_source_ids=None,
                     merged_headings=None, **kwargs):
        refs = kwargs.get("dart_source_refs") or []
        records.append({
            "chunk_id": chunk_id,
            "heading": section_heading,
            "text": text,
            "refs": refs,
            "n_refs": len(refs),
            "first_block": refs[0]["block_id"] if refs else None,
            "first_pages": refs[0].get("pages") if refs else None,
            "block_ids": [r["block_id"] for r in refs],
        })
        return {"id": chunk_id, "text": text, "source": {}}

    return ChunkerContext(create_chunk=create_chunk)


def _multi_block_item() -> Dict[str, Any]:
    """A synthetic DART page: three distinct headed blocks on different pages.

    Each block is a flat sibling ``<section data-dart-block-id=...>`` with an
    ``<hN>`` heading inside, mirroring the real DART leaf layout where the
    block attribute lives on the wrapper, not the heading. The blocks teach
    distinct prose so char-span resolution can tell them apart.
    """
    def _block(bid: str, pages: str, heading: str, body_words: List[str]) -> str:
        body = " ".join(body_words)
        return (
            f'<section class="dart-section" data-dart-block-id="{bid}" '
            f'data-dart-pages="{pages}" data-dart-block-role="paragraph">'
            f"<h2>{heading}</h2><p>{body}</p></section>"
        )

    # ~450 words per block: each clears the 100-word merge floor and any two
    # together exceed the 800-word merge ceiling, so the three blocks stay
    # three separate chunks (no merge into a single chunk). Each block's prose
    # is distinct so text-containment resolution attributes the right block.
    alpha = _block(
        "block_alpha", "2", "Cellular Respiration",
        ["Mitochondria"] + [f"alpha{n}" for n in range(450)],
    )
    beta = _block(
        "block_beta", "5-6", "Photosynthesis Overview",
        ["Chloroplasts"] + [f"beta{n}" for n in range(450)],
    )
    gamma = _block(
        "block_gamma", "9", "Enzyme Kinetics",
        ["Catalysis"] + [f"gamma{n}" for n in range(450)],
    )
    raw_html = "<html><body><main>" + alpha + beta + gamma + "</main></body></html>"

    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parsed = HTMLContentParser().parse(raw_html)
    return {
        "module_id": "mod1",
        "module_title": "Mod 1",
        "item_id": "synthetic-page",
        "title": parsed.title or "Synthetic Page",
        "resource_type": "page",
        "item_path": "synthetic-page.html",
        "raw_html": raw_html,
        "sections": parsed.sections,
        "learning_objectives": [],
    }


# ---------------------------------------------------------------------------
# Pure-helper unit tests for the char-span scoping primitives
# ---------------------------------------------------------------------------


def test_build_probe_index_orders_blocks_by_document_order() -> None:
    raw_html = (
        '<section data-dart-block-id="a" data-dart-pages="1"><p>'
        "Alpha distinct opening prose here for locating purposes.</p></section>"
        '<section data-dart-block-id="b" data-dart-pages="2"><p>'
        "Beta separate opening prose here for locating purposes.</p></section>"
    )
    index = build_dart_block_offset_index(raw_html)
    assert [ref["block_id"] for _, ref, _ in index] == ["a", "b"]
    # doc_order is monotonic (a before b).
    assert index[0][0] < index[1][0]
    # Each block carries non-empty (collapsed) text.
    assert all(block_text for _, _, block_text in index)


def test_resolve_refs_selects_only_blocks_present_in_chunk_text() -> None:
    raw_html = (
        '<section data-dart-block-id="a" data-dart-pages="1"><p>'
        "Alpha distinct opening prose here for locating purposes.</p></section>"
        '<section data-dart-block-id="b" data-dart-pages="2"><p>'
        "Beta separate opening prose here for locating purposes.</p></section>"
    )
    index = build_dart_block_offset_index(raw_html)
    # Chunk text containing only block a's prose.
    only_a = "Alpha distinct opening prose here for locating purposes."
    assert resolve_dart_refs_for_chunk(index, only_a) == [
        {"block_id": "a", "pages": [1]}
    ]
    # Chunk text containing only block b's prose.
    only_b = "Beta separate opening prose here for locating purposes."
    assert resolve_dart_refs_for_chunk(index, only_b) == [
        {"block_id": "b", "pages": [2]}
    ]
    # Chunk text containing both (document order preserved).
    both = resolve_dart_refs_for_chunk(index, only_a + " " + only_b)
    assert [r["block_id"] for r in both] == ["a", "b"]
    # Empty text / empty index -> [].
    assert resolve_dart_refs_for_chunk(index, "") == []
    assert resolve_dart_refs_for_chunk([], only_a) == []


# ---------------------------------------------------------------------------
# End-to-end chunker contract
# ---------------------------------------------------------------------------


def test_per_chunk_refs_are_narrow_and_ordered() -> None:
    records: List[Dict[str, Any]] = []
    chunk_content([_multi_block_item()], "TEST_101", ctx=_recording_ctx(records))
    assert records, "expected at least one chunk"

    total_blocks = 3
    # No chunk carries ALL blocks (the old all-document fallback shape).
    assert all(r["n_refs"] < total_blocks or r["n_refs"] == 0 for r in records), (
        "a chunk carried the whole document's block list — all-refs "
        f"fallback regression: {[r['block_ids'] for r in records]}"
    )
    # Every chunk that carries refs is narrow (1-2 blocks for these distinct
    # single-block sections).
    for r in records:
        assert r["n_refs"] <= 2, (
            f"chunk {r['chunk_id']} carried {r['n_refs']} refs: {r['block_ids']}"
        )
        # Block ids are unique within a chunk (dedupe holds).
        assert len(r["block_ids"]) == len(set(r["block_ids"]))

    # Distinct chunks resolve to distinct primary block ids (the user-visible
    # deep-link no longer collapses onto one block).
    primaries = {r["first_block"] for r in records if r["first_block"]}
    assert len(primaries) >= 2, (
        f"expected distinct primary blocks across chunks, got {primaries}"
    )


def test_pages_propagate_per_matched_block() -> None:
    records: List[Dict[str, Any]] = []
    chunk_content([_multi_block_item()], "TEST_101", ctx=_recording_ctx(records))
    # Map first_block -> first_pages and assert each block keeps its own pages.
    by_block = {r["first_block"]: r["first_pages"] for r in records if r["first_block"]}
    assert by_block.get("block_alpha") == [2]
    assert by_block.get("block_beta") == [5, 6]
    assert by_block.get("block_gamma") == [9]


def test_no_match_emits_at_most_one_section_ref() -> None:
    """A heading whose section HTML slice carries no block (the leaf layout)
    still gets at most ONE enclosing-section reference, never the document list.

    Construct a doc whose first heading sits OUTSIDE any block wrapper (so its
    section slice is the bare heading, carrying no data-dart-block-id), with a
    single block wrapping the prose. The chunk for that heading must carry at
    most one ref.
    """
    body = " ".join(f"word{n}" for n in range(80))
    raw_html = (
        "<html><body><main>"
        "<h2>Bare Heading Outside Block</h2>"
        '<section class="dart-section" data-dart-block-id="only_block" '
        f'data-dart-pages="3"><p>{body}</p></section>'
        "</main></body></html>"
    )
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parsed = HTMLContentParser().parse(raw_html)
    item = {
        "module_id": "mod1", "module_title": "Mod 1",
        "item_id": "bare-page", "title": parsed.title or "Bare Page",
        "resource_type": "page", "item_path": "bare-page.html",
        "raw_html": raw_html, "sections": parsed.sections,
        "learning_objectives": [],
    }
    records: List[Dict[str, Any]] = []
    chunk_content([item], "TEST_101", ctx=_recording_ctx(records))
    assert records
    for r in records:
        assert r["n_refs"] <= 1, (
            f"no-match chunk {r['chunk_id']} carried {r['n_refs']} refs "
            f"(expected <=1 section-level fallback): {r['block_ids']}"
        )
