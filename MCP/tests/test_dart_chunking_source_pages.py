"""Task #26 end-to-end: ``_run_dart_chunking`` stamps a top-level
``source_pages`` union onto emitted chunks from the ``data-semantik-pages`` /
``data-dart-pages`` attributes the SemantiK converter stamps on staged HTML.

The canonical chunker harvests each block's ``{block_id, pages}`` pair; the
DART-side ``_create_chunk`` callback unions their ``pages`` via
``Trainforge.chunker.union_source_pages`` into the additive top-level
``source_pages`` field. SemantiK HTML without those attributes keeps
``source_pages`` unset — legacy corpora stay byte-stable.

Exercises the ``data-semantik-*`` attribute spelling (the CURRENT emit) so a
future purge Stage-4 that drops ``data-dart-*`` doesn't orphan the test. The
per-file / per-chunk layout mirrors the sibling
``test_dart_chunking_source_refs.py`` — one provenance-attributed block per
staged file, so the char-span resolver attributes each block to its own chunk
(rather than merging two small sections and attributing only the first's ref,
the documented "at most one enclosing-section ref" fallback).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture
def dart_chunking_tool(monkeypatch, tmp_path):
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    registry = _build_tool_registry()
    return registry["run_dart_chunking"]


# Distinct synthetic prose per file so a chunk's page provenance is checkably
# scoped to the file it came from.
_PROSE_A = (
    "The mitochondrion is the powerhouse of the cell. Cells rely on "
    "mitochondria to produce ATP through oxidative phosphorylation, the "
    "primary mechanism by which cells generate metabolic energy across nearly "
    "every tissue type found in the organism under study."
)
_PROSE_B = (
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose. Chloroplasts capture photons and drive the Calvin cycle, fixing "
    "atmospheric carbon dioxide into sugars that sustain the plant and the "
    "wider food web depending on that captured solar input."
)


def _write_semantik_html(
    path: Path, *, block_id: str, pages: str, title: str, prose: str
) -> None:
    """Write SemantiK-shaped HTML: one section carrying ``data-semantik-*``."""
    path.write_text(
        "<!doctype html><html><head><title>" + title + "</title></head><body>"
        '<main class="dart-document">'
        f'<section class="dart-section" role="region" '
        f'data-semantik-source="semantik" '
        f'data-semantik-block-id="{block_id}" data-semantik-pages="{pages}">'
        f"<h2>{title}</h2><p>{prose}</p></section>"
        "</main></body></html>",
        encoding="utf-8",
    )


def _write_no_attr_html(path: Path, *, title: str, prose: str) -> None:
    path.write_text(
        "<!doctype html><html><head><title>" + title + "</title></head><body>"
        '<main class="dart-document"><section class="dart-section">'
        f"<h2>{title}</h2><p>{prose}</p></section></main></body></html>",
        encoding="utf-8",
    )


def _load_chunks(result_str: str):
    result = json.loads(result_str)
    assert result.get("success"), f"chunking errored: {result}"
    chunks_path = Path(result["semantik_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_range_and_single_pages_surface_as_union(dart_chunking_tool, tmp_path):
    """A block whose ``data-semantik-pages`` is a 2-page range surfaces the full
    [62, 63] union on its chunk; a single-page block surfaces [70]."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_semantik_html(
        staging / "lesson_01.html",
        block_id="s1", pages="62-63", title="Cells", prose=_PROSE_A,
    )
    _write_semantik_html(
        staging / "lesson_02.html",
        block_id="s2", pages="70", title="Energy", prose=_PROSE_B,
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="PAGES_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected at least one chunk emitted"

    # Every emitted source_pages list is sorted, deduped, positive-int.
    for chunk in chunks:
        if "source_pages" not in chunk:
            continue
        pages = chunk["source_pages"]
        assert pages == sorted(set(pages)), f"not sorted/deduped: {pages}"
        assert all(isinstance(p, int) and p >= 1 for p in pages), pages

    all_pages = set()
    for chunk in chunks:
        all_pages.update(chunk.get("source_pages", []))
    # The range block's full union AND the single-page block both surfaced.
    assert {62, 63}.issubset(all_pages), all_pages
    assert 70 in all_pages, all_pages
    # The range block surfaced its full [62, 63] union on some chunk.
    assert any(chunk.get("source_pages") == [62, 63] for chunk in chunks), (
        "expected a chunk carrying the full [62, 63] range union"
    )


def test_no_page_attrs_omit_source_pages(dart_chunking_tool, tmp_path):
    """Legacy SemantiK HTML carrying no page attrs: no chunk carries
    ``source_pages`` (legacy-safe omission)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_no_attr_html(
        staging / "legacy.html", title="Legacy", prose=_PROSE_A
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="PAGES_LEGACY_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected at least one chunk emitted"
    for chunk in chunks:
        assert "source_pages" not in chunk, (
            f"chunk {chunk.get('id')!r} unexpectedly carries source_pages"
        )


def test_per_chunk_page_scoping_not_document_union(dart_chunking_tool, tmp_path):
    """Blocks resolved to DIFFERENT chunks carry only their own block's pages —
    the lesson_01 chunk (pages 62-63) must NOT carry lesson_02's page 70, i.e.
    ``source_pages`` is per-chunk-scoped, not a document-wide union."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_semantik_html(
        staging / "lesson_01.html",
        block_id="s1", pages="62-63", title="Cells", prose=_PROSE_A,
    )
    _write_semantik_html(
        staging / "lesson_02.html",
        block_id="s2", pages="70", title="Energy", prose=_PROSE_B,
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="PAGES_SCOPE_TEST",
        staging_dir=str(staging),
    )))

    saw_a = saw_b = False
    for chunk in chunks:
        pages = set(chunk.get("source_pages", []))
        if not pages:
            continue
        text = chunk.get("text", "")
        if "mitochondrion" in text and "Photosynthesis" not in text:
            saw_a = True
            assert 70 not in pages, (
                f"lesson_01 chunk leaked lesson_02 page: {sorted(pages)}"
            )
        if "Photosynthesis" in text and "mitochondrion" not in text:
            saw_b = True
            assert not ({62, 63} & pages), (
                f"lesson_02 chunk leaked lesson_01 pages: {sorted(pages)}"
            )
    assert saw_a and saw_b, (
        f"expected page-carrying chunks from both files (a={saw_a}, b={saw_b})"
    )
