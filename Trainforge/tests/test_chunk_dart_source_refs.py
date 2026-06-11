"""B1 regression: the canonical chunker harvests ``data-dart-block-id`` /
``data-dart-pages`` provenance off DART HTML and threads it to the
``ctx.create_chunk`` callback as the additive ``dart_source_refs`` kwarg.

Contract under test:
  - HTML carrying a ``<section data-dart-block-id="..." data-dart-pages="...">``
    wrapper -> the chunk's callback receives ``dart_source_refs`` with the
    ``{block_id, pages}`` pair (pages parsed from the attribute's
    ``"3"`` / ``"3-5"`` / ``"3,5,7"`` forms).
  - HTML WITHOUT any ``data-dart-*`` attribute -> the callback is invoked
    WITHOUT a ``dart_source_refs`` kwarg (additive contract; legacy /
    IMSCC / Courseforge corpora stay byte-identical).
  - The pure parse helpers (``parse_dart_pages_attr`` /
    ``harvest_dart_source_refs``) handle every emitted page form + the
    empty-attribute / multi-wrapper cases.

Driven through ``Trainforge.chunker.chunk_content`` with a recording
callback (rather than CourseProcessor) so the B1 threading
(harvest -> chunk_content -> chunk_text_block -> _dispatch_create_chunk
additive-kwarg) is exercised in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker import (  # noqa: E402
    ChunkerContext,
    chunk_content,
    harvest_dart_source_refs,
    parse_dart_pages_attr,
)


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------


def test_parse_dart_pages_attr_forms() -> None:
    assert parse_dart_pages_attr("3") == [3]
    assert parse_dart_pages_attr("3-5") == [3, 4, 5]
    assert parse_dart_pages_attr("3,5,7") == [3, 5, 7]
    # Mixed + whitespace + reversed range tolerated.
    assert parse_dart_pages_attr("7, 3-4") == [3, 4, 7]
    assert parse_dart_pages_attr("5-3") == [3, 4, 5]
    # Empty / junk -> [].
    assert parse_dart_pages_attr("") == []
    assert parse_dart_pages_attr(None) == []
    assert parse_dart_pages_attr("0,-1,abc") == []


def test_harvest_pairs_block_id_with_same_element_pages() -> None:
    html = (
        '<section data-dart-block-id="s1" data-dart-pages="2-4">a</section>'
        '<section data-dart-block-id="s2" data-dart-pages="9">b</section>'
    )
    refs = harvest_dart_source_refs(html)
    assert refs == [
        {"block_id": "s1", "pages": [2, 3, 4]},
        {"block_id": "s2", "pages": [9]},
    ]


def test_harvest_block_without_pages_attr() -> None:
    html = '<section data-dart-block-id="s3_c0">contact</section>'
    assert harvest_dart_source_refs(html) == [{"block_id": "s3_c0", "pages": []}]


def test_harvest_empty_block_id_skipped() -> None:
    html = '<section data-dart-block-id="" data-dart-pages="3">boilerplate</section>'
    assert harvest_dart_source_refs(html) == []


def test_harvest_dedupes_repeated_block_id() -> None:
    html = (
        '<section data-dart-block-id="s1" data-dart-pages="1">a</section>'
        '<ul data-dart-block-id="s1" data-dart-pages="1">b</ul>'
    )
    assert harvest_dart_source_refs(html) == [{"block_id": "s1", "pages": [1]}]


def test_harvest_non_dart_html_returns_empty() -> None:
    html = '<section data-cf-source-ids="dart:doc#s1"><p>prose</p></section>'
    assert harvest_dart_source_refs(html) == []


# ---------------------------------------------------------------------------
# Chunker threading tests
# ---------------------------------------------------------------------------


def _recording_ctx(records: List[Dict[str, Any]]) -> ChunkerContext:
    """A ChunkerContext whose create_chunk records the kwargs it received
    and whether ``dart_source_refs`` was passed at all."""

    def create_chunk(*, chunk_id, text, html, item, section_heading,
                     chunk_type, follows_chunk_id=None, position_in_module=0,
                     html_xpath=None, char_span=None, section_source_ids=None,
                     merged_headings=None, **kwargs):
        records.append({
            "chunk_id": chunk_id,
            "text": text,
            "dart_source_refs_passed": "dart_source_refs" in kwargs,
            "dart_source_refs": kwargs.get("dart_source_refs"),
        })
        return {"id": chunk_id, "text": text, "source": {}}

    return ChunkerContext(create_chunk=create_chunk)


def _item(raw_html: str) -> Dict[str, Any]:
    return {
        "module_id": "mod1",
        "module_title": "Mod 1",
        "item_id": "page-one",
        "title": "Page One",
        "resource_type": "page",
        "item_path": "page-one.html",
        "raw_html": raw_html,
        "sections": [],
        "learning_objectives": [],
    }


def test_chunker_threads_dart_refs_no_sections_path() -> None:
    body = " ".join(f"word{n}" for n in range(60))
    raw_html = (
        "<html><body>"
        '<section class="dart-section" data-dart-block-id="main-content" '
        f'data-dart-pages="1-3"><p>Photosynthesis basics. {body}</p></section>'
        "</body></html>"
    )
    records: List[Dict[str, Any]] = []
    chunk_content([_item(raw_html)], "TEST_101", ctx=_recording_ctx(records))
    assert len(records) == 1
    rec = records[0]
    assert rec["dart_source_refs_passed"] is True
    assert rec["dart_source_refs"] == [
        {"block_id": "main-content", "pages": [1, 2, 3]}
    ]


def test_chunker_omits_kwarg_for_non_dart_html() -> None:
    """Byte-stability: non-DART HTML -> the kwarg is never passed, so a
    legacy callback that doesn't accept it keeps working unchanged."""
    body = " ".join(f"word{n}" for n in range(60))
    raw_html = f"<html><body><p>Ordinary prose. {body}</p></body></html>"
    records: List[Dict[str, Any]] = []
    chunk_content([_item(raw_html)], "TEST_101", ctx=_recording_ctx(records))
    assert len(records) == 1
    assert records[0]["dart_source_refs_passed"] is False
    assert records[0]["dart_source_refs"] is None


def test_legacy_callback_without_kwarg_still_works() -> None:
    """A create_chunk callback with NO ``dart_source_refs`` param (and no
    **kwargs catch-all) keeps working on DART HTML via the chunker's
    TypeError-fallback — the refs are simply not stamped."""

    def legacy_create_chunk(*, chunk_id, text, html, item, section_heading,
                            chunk_type, follows_chunk_id=None,
                            position_in_module=0, html_xpath=None,
                            char_span=None, section_source_ids=None,
                            merged_headings=None):
        return {"id": chunk_id, "text": text, "source": {}}

    raw_html = (
        "<html><body>"
        '<section data-dart-block-id="s1" data-dart-pages="4">'
        f'<p>{" ".join("w" + str(n) for n in range(60))}</p></section>'
        "</body></html>"
    )
    result = chunk_content(
        [_item(raw_html)], "TEST_101",
        ctx=ChunkerContext(create_chunk=legacy_create_chunk),
    )
    assert len(result.chunks) == 1  # no crash; ref simply not stamped
