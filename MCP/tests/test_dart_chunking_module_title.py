"""Chapter-ladder reconcile (SEMANTIK_CHAPTER_LADDER_RECONCILE) — chunk-side.

``_run_dart_chunking``'s ``_create_chunk`` stamps ``source.module_title``
from the FILE-level document title, which flattens a whole-book single-file
corpus onto one module (the whole-book reference corpus: 112/112 chunks stamped with
the defective "Chapter 0" h1). With the reconcile flag ON (the default), a
file carrying ≥2 ``<article role="doc-chapter">`` chapters resolves each
chunk's module_title to its ENCLOSING chapter-article ``<h2>`` via the
chunk's own block-id refs. A single-article file — the historical per-chapter
corpus shape — never builds a map and stays byte-identical; the explicit
falsey flag restores the legacy file-level title everywhere.

Synthetic fixtures only (invented placeholder text, NO real course content).
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
from MCP.tools.pipeline_tools import (  # noqa: E402
    _article_module_titles_by_block_id,
    _build_tool_registry,
)


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


def _prose(topic: str) -> str:
    """~500 words of varied placeholder prose per chapter, so the two chapter
    sections exceed the merge ceiling (MAX_CHUNK_SIZE=800 combined) and can
    never merge into one cross-chapter chunk."""
    cycles = [
        f"Measurement cycle {i} for the {topic.lower()} widget recorded a "
        f"spindle deflection of zero point zero {i} millimetres at the "
        "reference speed, which the supervising controller logged against "
        "the maintenance ledger for later trend comparison."
        for i in range(1, 11)
    ]
    parts = [
        f"{topic} widgets are assembled from placeholder components that "
        "rotate around a central spindle inside a sealed outer casing.",
        "The spindle transfers torque to the casing, which distributes the "
        "load evenly across every mounting point in the assembly.",
        f"Operators calibrate each {topic.lower()} widget by measuring the "
        "spindle deflection at three reference speeds and recording the "
        "values in the maintenance ledger for later comparison.",
        "When the deflection drifts beyond the tolerance band, the casing "
        "is opened and the bearing preload is adjusted with a torque wrench "
        "until the reading returns to the nominal range.",
        f"A well-maintained {topic.lower()} widget survives sustained "
        "operation across the full duty cycle without measurable wear on "
        "either the spindle or the casing interface surfaces.",
        "Inspection intervals depend on the operating environment, the "
        "ambient temperature, and the duty profile recorded by the "
        "supervising controller over the preceding maintenance period.",
    ]
    return " ".join(parts + cycles)


_PROSE_A = _prose("Alpha")
_PROSE_B = _prose("Beta")


def _article(block_id: str, title: str, prose: str, pages: str) -> str:
    return (
        f'<article role="doc-chapter" id="chap-{block_id}">'
        f"<h2>{title}</h2>"
        f'<section class="semantik-section" data-semantik-source="synthesized" '
        f'data-semantik-block-id="{block_id}" data-semantik-pages="{pages}">'
        f"<h3>{title} Basics</h3><p>{prose}</p></section>"
        "</article>"
    )


def _write_book_html(path: Path) -> None:
    """A whole-book single file: TWO doc-chapter articles."""
    path.write_text(
        "<!doctype html><html><head><title>SAMPLE WIDGET COMPENDIUM</title>"
        "</head><body><main><h1>SAMPLE WIDGET COMPENDIUM</h1>"
        + _article("s1", "Chapter 1", _PROSE_A, "3")
        + _article("s2", "Chapter 2", _PROSE_B, "9")
        + "</main></body></html>",
        encoding="utf-8",
    )


def _write_single_chapter_html(path: Path) -> None:
    """The historical per-chapter shape: exactly ONE doc-chapter article."""
    path.write_text(
        "<!doctype html><html><head><title>Chapter 3 Gamma Widgets</title>"
        "</head><body><main><h1>Chapter 3 Gamma Widgets</h1>"
        + _article("s1", "Chapter 3 Gamma Widgets", _PROSE_A, "1")
        + "</main></body></html>",
        encoding="utf-8",
    )


def _load_chunks(result_str: str):
    result = json.loads(result_str)
    assert result.get("success"), f"chunking errored: {result}"
    chunks_path = Path(result["semantik_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── unit: the block-id → article-title map ──────────────────────────────────


def test_article_module_map_multi_article():
    html = (
        "<main>"
        + _article("s1", "Chapter 1", _PROSE_A, "3")
        + _article("s2", "Chapter 2", _PROSE_B, "9")
        + "</main>"
    )
    mapping = _article_module_titles_by_block_id(html)
    assert mapping == {"s1": "Chapter 1", "s2": "Chapter 2"}


def test_article_module_map_single_article_is_none():
    html = "<main>" + _article("s1", "Chapter 3", _PROSE_A, "1") + "</main>"
    assert _article_module_titles_by_block_id(html) is None


def test_article_module_map_legacy_block_id_spelling():
    html = (
        '<article role="doc-chapter"><h2>Chapter 1</h2>'
        f'<section data-dart-block-id="s1"><p>{_PROSE_A}</p></section>'
        "</article>"
        '<article role="doc-chapter"><h2>Chapter 2</h2>'
        f'<section data-dart-block-id="s2"><p>{_PROSE_B}</p></section>'
        "</article>"
    )
    mapping = _article_module_titles_by_block_id(html)
    assert mapping == {"s1": "Chapter 1", "s2": "Chapter 2"}


# ── end-to-end: chunks resolve module_title per chapter article ─────────────


def test_whole_book_chunks_module_title_per_chapter(dart_chunking_tool, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_book_html(staging / "sample_book.html")

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="LADDER_MOD_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected chunks emitted"
    titles = {c["source"]["module_title"] for c in chunks}
    assert titles <= {"Chapter 1", "Chapter 2"}, titles
    assert "Chapter 1" in titles and "Chapter 2" in titles
    # each chunk's module matches the chapter its block-id ref belongs to
    for chunk in chunks:
        refs = chunk["source"].get("source_references") or []
        assert refs, "fixture chunks must carry refs"
        block_id = refs[0]["sourceId"].rsplit("#", 1)[-1]
        expected = {"s1": "Chapter 1", "s2": "Chapter 2"}[block_id]
        assert chunk["source"]["module_title"] == expected


def test_single_chapter_file_keeps_file_level_title(dart_chunking_tool, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_single_chapter_html(staging / "chapter_03.html")

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="LADDER_SINGLE_TEST",
        staging_dir=str(staging),
    )))
    assert chunks
    for chunk in chunks:
        assert chunk["source"]["module_title"] == "Chapter 3 Gamma Widgets"


def test_off_flag_keeps_file_level_title(
    dart_chunking_tool, tmp_path, monkeypatch
):
    monkeypatch.setenv("SEMANTIK_CHAPTER_LADDER_RECONCILE", "0")
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_book_html(staging / "sample_book.html")

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="LADDER_OFF_TEST",
        staging_dir=str(staging),
    )))
    assert chunks
    for chunk in chunks:
        assert chunk["source"]["module_title"] == "SAMPLE WIDGET COMPENDIUM"
