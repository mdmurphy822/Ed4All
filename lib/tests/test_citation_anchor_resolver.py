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


# --------------------------------------------------------------------------- #
# Marketable-v1 B3 regression: normalization fixes for the citation-anchor
# resolver. Each case pins exactly one fixed failure class on a hand-built
# page/chunk pair (no LibV2 corpus). Without the resolver fix the chunk would
# fail to anchor (SPAN_FABRICATED) and the fail-closed citation gate would
# block a correct gold answer.
# --------------------------------------------------------------------------- #

# A full content page whose <body> carries data-cf-role="template-chrome"
# (Courseforge stamps this on the page body). HTMLTextExtractor opens a
# chrome-skip scope on <body> that its fixed end-tag set can never close, so
# the OLD path swallowed the entire body. Genuine chrome (header/footer) keeps
# its mark and must still be skipped.
BODY_CHROME_PAGE_HTML = """<!DOCTYPE html>
<html><head><title>Validation Report</title></head>
<body data-cf-role="template-chrome">
  <header data-cf-role="template-chrome"><nav>Week 7 navigation skip-link clutter</nav></header>
  <main>
    <h1>The Validation Report</h1>
    <p>An inverted index maps each term to the list of documents that contain it, so a keyword query can be answered without scanning every document in the collection.</p>
    <p>Term frequency weighting boosts documents in which a query term appears often, while inverse document frequency down-weights terms that appear in nearly every document.</p>
  </main>
  <footer data-cf-role="template-chrome">Footer boilerplate that repeats on every page in the course.</footer>
</body></html>
"""


def test_body_chrome_mark_does_not_swallow_page(tmp_path):
    """A page whose <body> carries data-cf-role=template-chrome must still
    yield its full body text — the resolver neutralizes the un-closeable
    root-level chrome mark before re-extracting. Without the fix the body is
    swallowed, containment collapses to ~0, and the chunk reports
    SPAN_FABRICATED.
    """
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / "report.html").write_text(BODY_CHROME_PAGE_HTML, encoding="utf-8")

    # char_span deliberately bogus so resolution must come from the body text,
    # not the exact-span arm.
    chunk = _base_chunk("report.html", EXACT_ANCHOR, [0, 5])
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    assert anchor.status in {
        AnchorStatus.RESOLVED_NORMALIZED,
        AnchorStatus.RESOLVED_CONTAINMENT,
    }
    assert anchor.containment_rate == pytest.approx(1.0)


def test_genuine_chrome_still_skipped(tmp_path):
    """The root-mark neutralization is surgical: header/footer chrome keeps its
    mark and must NOT leak into the matched page text. A chunk whose text is
    ONLY the footer boilerplate must NOT anchor.
    """
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / "report.html").write_text(BODY_CHROME_PAGE_HTML, encoding="utf-8")

    footer_text = "Footer boilerplate that repeats on every page in the course."
    chunk = _base_chunk("report.html", footer_text, [0, 5])
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    # Footer chrome was skipped, so its text is absent from the page text.
    assert anchor.status is AnchorStatus.SPAN_FABRICATED


# A page whose entities HTMLParser will decode (&rsquo; -> ', &mdash; -> —),
# paired with a chunk that retained the RAW entities verbatim.
ENTITY_PAGE_HTML = """<!DOCTYPE html>
<html><head><title>Scenario</title></head>
<body>
  <main>
    <p>Acme Corp&rsquo;s HR system stores the org chart as a flat table &mdash; each employee has a single manager column pointing to a direct supervisor, and the validator must walk that chain to the top.</p>
  </main>
</body></html>
"""

# Same sentence, but with the raw entities the chunk text retained.
ENTITY_CHUNK_TEXT = (
    "Acme Corp&rsquo;s HR system stores the org chart as a flat table &mdash; "
    "each employee has a single manager column pointing to a direct supervisor, "
    "and the validator must walk that chain to the top."
)


def test_entity_drift_normalized_match(tmp_path):
    """Chunk retained raw HTML entities while the page got them decoded by the
    parser. After symmetric html.unescape on both sides the chunk is a
    normalized substring of the page. Without the fix the shingles straddling
    each entity mismatch and containment lands just below 0.85.
    """
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / "scenario.html").write_text(ENTITY_PAGE_HTML, encoding="utf-8")

    chunk = _base_chunk("scenario.html", ENTITY_CHUNK_TEXT, [0, 5])
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    assert anchor.status in {
        AnchorStatus.RESOLVED_NORMALIZED,
        AnchorStatus.RESOLVED_CONTAINMENT,
    }
    assert anchor.containment_rate == pytest.approx(1.0)


def test_normalization_fixes_compose(tmp_path):
    """Body-chrome mark AND entity drift on the same page/chunk pair — both
    resolver fixes must compose so the chunk anchors at full containment.
    """
    page = """<!DOCTYPE html>
<html><head><title>Combined</title></head>
<body data-cf-role="template-chrome">
  <main>
    <p>The reasoner derives a triple when the rule&rsquo;s antecedent matches &mdash; otherwise the closure stays fixed and no new statements appear in the graph.</p>
  </main>
</body></html>
"""
    chunk_text = (
        "The reasoner derives a triple when the rule&rsquo;s antecedent matches "
        "&mdash; otherwise the closure stays fixed and no new statements appear "
        "in the graph."
    )
    course_dir = tmp_path / "course"
    html_dir = course_dir / "sources" / "textbooks"
    html_dir.mkdir(parents=True)
    (html_dir / "combined.html").write_text(page, encoding="utf-8")

    chunk = _base_chunk("combined.html", chunk_text, [0, 5])
    anchor = resolve_citation_anchor(chunk, course_dir, chunkset_kind="dart")
    assert anchor.status in {
        AnchorStatus.RESOLVED_NORMALIZED,
        AnchorStatus.RESOLVED_CONTAINMENT,
    }
    assert anchor.containment_rate == pytest.approx(1.0)


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
