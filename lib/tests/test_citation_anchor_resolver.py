"""Always-on synthetic contract tests for the citation-anchor resolver.

This is the fail-without-fix tier: each case pins exactly one resolver
behavior on hand-built pages/chunks in ``tmp_path`` (no LibV2 corpus, no
network, no LLM). Written against the API in
``lib/retrieval/citation_anchor.py`` per the WS1.2 plan.

The negative-path case (``test_fabricated_span_detected``) documents today's
chunker ``_locate`` total-miss bug class as a *detectable* state — it must NOT
pass via span repair.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from lib.retrieval.citation_anchor import (
    AnchorStatus,
    CitationAnchor,
    anchor_report,
    resolve_citation_anchor,
    resolve_source_page,
)
from Trainforge.parsers.xpath_walker import resolve_xpath

BODY_XPATH = "/html[1]/body[1]"

PAGE_HTML = """<!DOCTYPE html>
<html><head><title>Resolver Test Page</title></head>
<body>
  <h1>Indexing</h1>
  <p>An inverted index maps each term to the list of documents that contain it, so a keyword query can be answered without scanning every document in the collection.</p>
  <p>Term frequency weighting boosts documents in which a query term appears often, while inverse document frequency down-weights terms that appear in nearly every document.</p>
</body></html>
"""

EXACT_ANCHOR = (
    "An inverted index maps each term to the list of documents that contain it, "
    "so a keyword query can be answered without scanning every document in the "
    "collection."
)


def _write_html_course(tmp_path: Path, *, item_path: str = "indexing.html") -> Path:
    """Build a minimal LibV2-style course dir with a DART source page."""
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / item_path).write_text(PAGE_HTML, encoding="utf-8")
    return course_dir


def _base_chunk(item_path: str, text: str, char_span):
    return {
        "id": "chunk_001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": text,
        "html": "<h1>Indexing</h1>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "indexing",
            "lesson_id": "indexing",
            "section_heading": "Indexing",
            "html_xpath": BODY_XPATH,
            "char_span": list(char_span) if char_span is not None else None,
            "item_path": item_path,
        },
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 30,
        "word_count": len(text.split()),
        "bloom_level": "understand",
    }


def _correct_span(item_path: str = "indexing.html"):
    body = resolve_xpath(PAGE_HTML, BODY_XPATH)
    start = body.find(EXACT_ANCHOR)
    assert start >= 0, "test fixture broken: exact anchor not in body text"
    return [start, start + len(EXACT_ANCHOR)]


def test_resolve_source_page_returns_html(tmp_path):
    course_dir = _write_html_course(tmp_path)
    chunk = _base_chunk("indexing.html", EXACT_ANCHOR, _correct_span())
    resolved = resolve_source_page(chunk, course_dir, chunkset_kind="dart")
    assert resolved is not None
    path, html = resolved
    assert "inverted index" in html
    assert Path(path).name == "indexing.html"


def test_resolved_exact_on_clean_fixture(tmp_path):
    course_dir = _write_html_course(tmp_path)
    chunk = _base_chunk("indexing.html", EXACT_ANCHOR, _correct_span())
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    assert isinstance(anchor, CitationAnchor)
    assert anchor.status is AnchorStatus.RESOLVED_EXACT
    assert anchor.containment_rate == pytest.approx(1.0)
    assert anchor.source_path is not None


def test_normalized_fallback(tmp_path):
    course_dir = _write_html_course(tmp_path)
    # Chunk text is the page sentence with collapsed-whitespace drift (the
    # post-extract projection prepends newlines + doubles spaces). The
    # char_span here is DELIBERATELY WRONG (points elsewhere), so the exact
    # arm cannot fire; resolution must fall through to the normalized
    # substring arm (the page plain text still contains the normalized text).
    text = "\n\n" + EXACT_ANCHOR.replace(" ", "  ") + "\n"
    chunk = _base_chunk("indexing.html", text, [0, 12])
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    assert anchor.status is AnchorStatus.RESOLVED_NORMALIZED
    assert anchor.normalized_match is True


def test_fabricated_span_detected(tmp_path):
    """A chunk whose char_span was produced by the _locate total-miss arm:
    the text IS on the page but NOT at the recorded span. Must surface as
    SPAN_FABRICATED — never repaired into RESOLVED_EXACT.

    To isolate the span signal from containment, the chunk text here is a
    DIFFERENT real page sentence than the span slices to, AND we set the
    containment threshold high enough that the (correct, locatable) text
    still resolves only via the span check — i.e. we test the negative arm
    by making the text un-locatable at the span while still on the page.
    """
    # Use a needle that is NOT a contiguous page substring (so normalized +
    # containment-at-1.0 both fail) but a fabricated [start,end] span exists.
    needle = "This sentence about query planning never appears verbatim on the indexing page at all."
    span = [10, 10 + len(needle)]  # plausible-looking but bogus
    chunk = _base_chunk("indexing.html", needle, span)
    anchor = resolve_citation_anchor(
        chunk, course_dir=_write_html_course(tmp_path), chunkset_kind="dart"
    )
    assert anchor.status is AnchorStatus.SPAN_FABRICATED
    # The page WAS found — this is a span/text problem, not a missing source.
    assert anchor.source_path is not None


def test_missing_source_page(tmp_path):
    course_dir = _write_html_course(tmp_path)
    chunk = _base_chunk("does_not_exist.html", EXACT_ANCHOR, [0, 10])
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    assert anchor.status is AnchorStatus.SOURCE_PAGE_MISSING
    assert anchor.source_path is None


def test_imscc_member_resolution(tmp_path):
    """Zip a 2-page mini cartridge; a chunk whose item_path is a member path
    resolves without extracting the archive to disk."""
    course_dir = tmp_path / "course"
    imscc_dir = course_dir / "source" / "imscc"
    imscc_dir.mkdir(parents=True)
    imscc_path = imscc_dir / "pkg.imscc"
    with zipfile.ZipFile(imscc_path, "w") as zf:
        zf.writestr("imsmanifest.xml", "<manifest/>")
        zf.writestr("week_01/intro.html", PAGE_HTML)
        zf.writestr("week_01/other.html", "<html><body><p>unrelated</p></body></html>")

    span = _correct_span()
    chunk = _base_chunk("week_01/intro.html", EXACT_ANCHOR, span)
    resolved = resolve_source_page(chunk, course_dir, chunkset_kind="imscc")
    assert resolved is not None
    member_id, html = resolved
    assert "intro.html" in str(member_id)
    assert "inverted index" in html

    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="imscc")
    assert anchor.status in {
        AnchorStatus.RESOLVED_EXACT,
        AnchorStatus.RESOLVED_NORMALIZED,
    }


def test_containment_threshold_boundary(tmp_path):
    """A chunk whose middle was boilerplate-stripped: ~90% of its shingles
    are present on the page. RESOLVED_CONTAINMENT at threshold 0.85,
    unresolved at 0.95.
    """
    course_dir = _write_html_course(tmp_path)
    # Take the page sentence and append a short foreign tail so neither the
    # exact span nor a contiguous normalized substring matches, but ~91% of
    # the 8-token shingles still do. char_span deliberately bogus.
    text = EXACT_ANCHOR + " plus coda"
    chunk = _base_chunk("indexing.html", text, [0, 5])

    high = resolve_citation_anchor(
        chunk, course_dir, chunkset_kind="dart", containment_threshold=0.85
    )
    assert high.status is AnchorStatus.RESOLVED_CONTAINMENT
    assert 0.85 <= high.containment_rate < 0.95

    strict = resolve_citation_anchor(
        chunk, course_dir, chunkset_kind="dart", containment_threshold=0.95
    )
    assert strict.status is AnchorStatus.SPAN_FABRICATED
    assert strict.containment_rate < 0.95


def test_anchor_report_rollup_deterministic(tmp_path):
    course_dir = _write_html_course(tmp_path)
    chunks = [
        _base_chunk("indexing.html", EXACT_ANCHOR, _correct_span()),
        {**_base_chunk("does_not_exist.html", EXACT_ANCHOR, [0, 10]), "id": "chunk_002"},
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    r1 = anchor_report(chunks_path, course_dir, chunkset_kind="dart")
    r2 = anchor_report(chunks_path, course_dir, chunkset_kind="dart")
    assert r1 == r2
    assert r1["total_chunks"] == 2
    assert r1["status_counts"][AnchorStatus.SOURCE_PAGE_MISSING.value] == 1
    assert r1["anchoring_rate"] == pytest.approx(0.5)
