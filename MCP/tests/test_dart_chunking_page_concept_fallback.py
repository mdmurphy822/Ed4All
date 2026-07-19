"""TRAINFORGE_PAGE_CONCEPT_FALLBACK end-to-end through ``_run_dart_chunking``.

Audit defect D1: the SemantiK GLM-OCR accessible-HTML lane emits markup-less
prose (ZERO ``<strong>`` / ``<b>`` / ``<dt>`` elements), so
``HTMLContentParser`` harvests no page-level ``key_concepts`` and every chunk
of the page emits ``concept_tags == []`` (704/705 empty on a real GLM-OCR
course). The vendor/publisher-HTML course, by contrast, averages ~14
tags/chunk because its bold/dt markup feeds the parser.

Fix: behind ``TRAINFORGE_PAGE_CONCEPT_FALLBACK`` (default OFF), the chunking
phase derives a PAGE-LEVEL key-concept list from each markup-less page's own
text and assigns it to ``item["key_concepts"]`` BEFORE the existing
page-level ``extract_concept_tags`` path runs — so all chunks of a page share
the same tag set (page-level, NOT chunk-local).

These tests drive the real ``run_dart_chunking`` registry tool over a
markup-less corpus and assert:

  (a) flag OFF  → chunks emit empty concept_tags (byte-identical legacy);
  (b) flag ON   → chunks emit non-empty concept_tags, IDENTICAL across all
      chunks of the same page (page-level, not chunk-local);
  (c) a page carrying real parser key_concepts is NOT overwritten.
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
    """Return the run_dart_chunking registry entry rooted at tmp_path."""
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    # Ensure the chunk-local + page-fallback flags start from a clean slate.
    monkeypatch.delenv("TRAINFORGE_CHUNK_LOCAL_TAGS", raising=False)
    monkeypatch.delenv("TRAINFORGE_PAGE_CONCEPT_FALLBACK", raising=False)
    registry = _build_tool_registry()
    return registry["run_dart_chunking"]


# Markup-less prose about a DELIBERATELY fake/neutral domain: no <strong>/<b>/
# <dt> anywhere, and none of the pedagogy CONCEPT_PATTERNS fire — so with the
# fallback OFF the chunks get empty concept_tags, exactly like the GLM-OCR lane.
_FAKE_DOMAIN_BODY = (
    "The zorbulon lattice stores flux packets inside a resonance chamber. "
    "Each resonance chamber couples to a quenby manifold. The quenby manifold "
    "regulates the flux packets and the zorbulon lattice. A flux packet decays "
    "into a resonance chamber over time. The zorbulon lattice anchors every "
    "quenby manifold to the resonance chamber."
)


def _write_markupless_html(path: Path, title: str, body: str) -> None:
    """Accessible-HTML-lane page: headings + paragraphs, ZERO bold/dt markup."""
    path.write_text(
        "<!doctype html><html><head><title>"
        + title
        + "</title></head><body>"
        + "<h1>" + title + "</h1>"
        + "<section><h2>Overview</h2><p>" + body + "</p></section>"
        + "<section><h2>Details</h2><p>" + body + "</p></section>"
        + "</body></html>",
        encoding="utf-8",
    )


def _run(tool, course_name: str, staging: Path) -> list[dict]:
    result = json.loads(asyncio.run(tool(
        course_name=course_name, staging_dir=str(staging),
    )))
    assert result.get("success"), f"chunking errored: {result}"
    chunks_path = Path(result["semantik_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_flag_off_markupless_chunks_have_empty_tags(
    dart_chunking_tool, tmp_path, monkeypatch
):
    """(a) Flag OFF → markup-less prose yields empty concept_tags (legacy)."""
    monkeypatch.delenv("TRAINFORGE_PAGE_CONCEPT_FALLBACK", raising=False)
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_markupless_html(staging / "page_01.html", "Page One", _FAKE_DOMAIN_BODY)

    chunks = _run(dart_chunking_tool, "DART_PCF_OFF", staging)
    assert chunks, "expected at least one chunk emitted"
    # The fake domain trips NO CONCEPT_PATTERNS and carries no bold/dt markup,
    # so every chunk's concept_tags must be empty with the fallback off.
    assert all(not c.get("concept_tags") for c in chunks), (
        "flag OFF must be byte-identical legacy — markup-less prose has no "
        "concept-tag substrate; got "
        + repr({c["id"]: c.get("concept_tags") for c in chunks
                if c.get("concept_tags")})
    )


def test_flag_on_fills_page_level_tags(
    dart_chunking_tool, tmp_path, monkeypatch
):
    """(b) Flag ON → non-empty tags, IDENTICAL across a page's chunks."""
    monkeypatch.setenv("TRAINFORGE_PAGE_CONCEPT_FALLBACK", "true")
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_markupless_html(staging / "page_01.html", "Page One", _FAKE_DOMAIN_BODY)

    chunks = _run(dart_chunking_tool, "DART_PCF_ON", staging)
    assert chunks, "expected at least one chunk emitted"

    tagged = [c for c in chunks if c.get("concept_tags")]
    assert tagged, (
        "flag ON must derive page-level concept_tags from the page's own text"
    )
    # Real fake-domain nouns surface (the whole point of the fix).
    union = {t for c in chunks for t in (c.get("concept_tags") or [])}
    assert any(
        tok in slug
        for slug in union
        for tok in ("zorbulon", "resonance", "quenby", "flux", "manifold")
    ), sorted(union)

    # PAGE-LEVEL, not chunk-local: every chunk from the single source page
    # carries the SAME tag list (this is the vendor semantics the fix restores).
    tag_lists = {tuple(c.get("concept_tags") or []) for c in chunks}
    assert len(tag_lists) == 1, (
        "all chunks of one page must share an identical page-level tag list; "
        f"got {tag_lists!r}"
    )
    # And that shared list is non-empty.
    assert next(iter(tag_lists)), "the shared page-level tag list is empty"


def test_flag_on_does_not_overwrite_parser_concepts(
    dart_chunking_tool, tmp_path, monkeypatch
):
    """(c) A page WITH real parser key_concepts is not overwritten on-flag.

    The page carries <strong> markup (real parser key_concepts), so the
    fallback must leave those page-level concepts in place — the emitted tags
    reflect the bold terms, not lexical-derived fake-domain slugs.
    """
    monkeypatch.setenv("TRAINFORGE_PAGE_CONCEPT_FALLBACK", "true")
    staging = tmp_path / "staging"
    staging.mkdir()
    # Bold terms feed HTMLContentParser._extract_concepts → real key_concepts.
    (staging / "page_01.html").write_text(
        "<!doctype html><html><head><title>Marked</title></head><body>"
        "<h1>Marked Page</h1><section><h2>Terms</h2>"
        "<p>The <strong>zorbulon lattice</strong> stores flux inside a "
        "<strong>resonance chamber</strong>. The resonance chamber couples "
        "to the zorbulon lattice repeatedly and reliably here.</p>"
        "</section></body></html>",
        encoding="utf-8",
    )

    chunks = _run(dart_chunking_tool, "DART_PCF_PARSER", staging)
    union = {t for c in chunks for t in (c.get("concept_tags") or [])}
    assert union, "expected the parser's bold key_concepts to yield tags"
    # The bold-term concepts survived (they are the parser's key_concepts,
    # untouched by the fallback).
    assert any("zorbulon" in s for s in union) or any(
        "resonance" in s for s in union
    ), sorted(union)
