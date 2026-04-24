"""Wave 74 Session 3 semantic_structure_extractor heading-fallback tests.

DART's live pipeline was caught emitting HTML that lacked both
``<article role="doc-chapter">`` wrappers AND ``<section>`` wrappers
for many files (notably the W3C spec family: rdf11-primer,
shacl-advanced-features, owl2-primer). The pre-Wave-74-S3 extractor
keyed chapter detection off ``<section>``/doc-chapter wrappers alone,
so those inputs yielded a single "Contents" chapter with no sections
or, worse, zero chapters. Downstream ``source_module_map.json`` ended
up with 0 entries per week.

These tests lock in the heading-hierarchy fallback that activates
when the primary section-based and doc-chapter paths produce trivial
output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lib.semantic_structure_extractor import SemanticStructureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_HEADING_ONLY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Heading Only Doc</title></head>
<body>
  <main>
    <h1>Heading Only Doc</h1>
    <h2 id="intro">Introduction</h2>
    <p>Intro paragraph.</p>
    <h3 id="intro-scope">Scope</h3>
    <p>Scope paragraph.</p>
    <h3 id="intro-audience">Audience</h3>
    <p>Audience paragraph.</p>
    <h2 id="body">Body</h2>
    <p>Body paragraph.</p>
    <h3 id="body-concepts">Concepts</h3>
    <p>Concepts paragraph.</p>
    <h3 id="body-examples">Examples</h3>
    <p>Examples paragraph.</p>
    <h2 id="closing">Closing</h2>
    <p>Closing paragraph.</p>
    <h3 id="closing-recap">Recap</h3>
    <p>Recap paragraph.</p>
    <h3 id="closing-next">Next Steps</h3>
    <p>Next steps paragraph.</p>
  </main>
</body>
</html>
"""


_H3_ONLY_WITH_TOC_H2_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>W3C Spec Shape</title></head>
<body>
  <main>
    <h1>W3C Spec Shape</h1>
    <h2>Contents</h2>
    <ul>
      <li>Introduction</li><li>Data Model</li><li>Concepts</li>
    </ul>
    <h3 id="intro">Introduction</h3>
    <p>Introduction paragraph with substantive content.</p>
    <h3 id="data-model">Data Model</h3>
    <p>Data model paragraph with substantive content.</p>
    <h3 id="concepts">Concepts</h3>
    <p>Concepts paragraph with substantive content.</p>
  </main>
</body>
</html>
"""


_SECTION_WRAPPED_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Wrapped</title></head>
<body>
  <main>
    <h1>Wrapped Textbook</h1>
    <section aria-labelledby="ch1-h">
      <h2 id="ch1-h">Chapter One</h2>
      <p>Chapter one body.</p>
      <section aria-labelledby="ch1-s1-h">
        <h3 id="ch1-s1-h">Section 1.1</h3>
        <p>Subsection body.</p>
      </section>
    </section>
    <section aria-labelledby="ch2-h">
      <h2 id="ch2-h">Chapter Two</h2>
      <p>Chapter two body.</p>
    </section>
  </main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Heading-only synthesis path
# ---------------------------------------------------------------------------


def test_heading_only_html_synthesizes_three_chapters_with_two_sections_each():
    """Given heading-only HTML (no <section> wrappers, no doc-chapter
    articles) with 3 h2s and 6 h3s, emit 3 chapters each with 2
    sections. This is the canonical third-party/W3C shape the
    pre-fix pipeline silently degraded on."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_HEADING_ONLY_HTML)
    chapters = result["chapters"]
    assert len(chapters) == 3, (
        f"expected 3 chapters, got {len(chapters)}: "
        f"{[c.get('headingText') for c in chapters]}"
    )
    titles = [c["headingText"] for c in chapters]
    assert titles == ["Introduction", "Body", "Closing"]

    for chapter in chapters:
        assert len(chapter["sections"]) == 2, (
            f"chapter {chapter['headingText']!r} has "
            f"{len(chapter['sections'])} sections, expected 2"
        )
        for section in chapter["sections"]:
            assert section["headingText"], (
                f"section under {chapter['headingText']!r} has empty title"
            )


def test_heading_only_titles_are_non_empty():
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_HEADING_ONLY_HTML)
    for chapter in result["chapters"]:
        assert chapter["headingText"], (
            f"chapter {chapter['id']} has empty title"
        )


def test_toc_h2_is_demoted_and_h3s_promote_to_chapters():
    """Real-world W3C shape: single h2 ``Contents`` (TOC), real
    structure in h3s. The fallback must skip the TOC h2 and promote
    the h3s to chapter-level."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_H3_ONLY_WITH_TOC_H2_HTML)
    chapters = result["chapters"]
    # Expect 3 chapters (one per real h3), NOT a single "Contents"
    # chapter with 3 demoted sections.
    assert len(chapters) == 3, (
        f"expected 3 chapters, got {len(chapters)}: "
        f"{[c.get('headingText') for c in chapters]}"
    )
    titles = {c["headingText"] for c in chapters}
    assert titles == {"Introduction", "Data Model", "Concepts"}
    # The TOC "Contents" heading must never appear as a chapter title.
    assert "Contents" not in titles


# ---------------------------------------------------------------------------
# Section-wrapped regression guard
# ---------------------------------------------------------------------------


def test_section_wrapped_html_still_uses_primary_path():
    """Regression guard: the primary section-based path must remain
    the winner when <section> wrappers are present. The fallback
    activates only when the primary paths degenerate."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_SECTION_WRAPPED_HTML)
    chapters = result["chapters"]
    assert len(chapters) == 2
    titles = [c["headingText"] for c in chapters]
    assert titles == ["Chapter One", "Chapter Two"]
    # Chapter one should have at least one section from the nested
    # <section aria-labelledby="ch1-s1-h">.
    assert len(chapters[0]["sections"]) >= 1
    assert "Section 1.1" in chapters[0]["sections"][0]["headingText"]


# ---------------------------------------------------------------------------
# Warning observability
# ---------------------------------------------------------------------------


def test_warning_fires_when_fallback_activates(caplog):
    """The fallback must log a WARNING so DART output quality is
    observable. Callers (the textbook-to-course pipeline) rely on this
    signal to flag upstream DART regressions.

    The TOC-h2 fixture is the canonical case where the primary path
    degenerates (picks up "Contents" as the only chapter) and the
    fallback has to rescue the real hierarchy from the h3s.
    """
    caplog.set_level(
        logging.WARNING,
        logger="lib.semantic_structure_extractor.semantic_structure_extractor",
    )
    extractor = SemanticStructureExtractor()
    extractor.extract(
        _H3_ONLY_WITH_TOC_H2_HTML, source_path="toc_h2.html"
    )
    fallback_warnings = [
        rec for rec in caplog.records
        if "falling back to heading-hierarchy synthesis" in rec.message
    ]
    assert fallback_warnings, (
        "expected at least one warning when fallback activates, got: "
        f"{[r.message for r in caplog.records]}"
    )


def test_warning_does_not_fire_for_section_wrapped_html(caplog):
    """Inverse: when the primary path succeeds, the fallback must
    stay silent."""
    caplog.set_level(
        logging.WARNING,
        logger="lib.semantic_structure_extractor.semantic_structure_extractor",
    )
    extractor = SemanticStructureExtractor()
    extractor.extract(_SECTION_WRAPPED_HTML, source_path="wrapped.html")
    fallback_warnings = [
        rec for rec in caplog.records
        if "falling back to heading-hierarchy synthesis" in rec.message
    ]
    assert not fallback_warnings, (
        f"fallback fired unexpectedly: "
        f"{[r.message for r in fallback_warnings]}"
    )


# ---------------------------------------------------------------------------
# data-dart-* provenance carries through the fallback
# ---------------------------------------------------------------------------


def test_dart_provenance_content_survives_fallback():
    """``data-dart-*``-tagged content elements must survive the
    fallback path intact. The `ContentBlock` schema doesn't serialize
    raw HTML attributes (that's parity with the primary path), but the
    block's originating `element` reference must still point at the
    tag so downstream consumers can walk the DOM for provenance.
    Regression guard: the fallback must not drop the paragraphs
    entirely, since that would amputate the PDF-origin chain.
    """
    # Construct HTML that forces the fallback (only h3 real headings
    # plus a TOC h2) so we can assert the fallback path specifically.
    html = """
<!DOCTYPE html>
<html lang="en">
<head><title>Provenance Doc</title></head>
<body>
  <main>
    <h1>Provenance Doc</h1>
    <h2>Contents</h2>
    <h3 id="ch1">Chapter One</h3>
    <p data-dart-block-id="p1" data-dart-source="pdf"
       data-dart-pages="1-2">
      A paragraph with provenance attributes on chapter one.
    </p>
    <h3 id="ch2">Chapter Two</h3>
    <p data-dart-block-id="p2" data-dart-source="pdf"
       data-dart-pages="3-4">
      Another paragraph with provenance attributes on chapter two.
    </p>
  </main>
</body>
</html>
"""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(html)
    chapters = result["chapters"]
    assert len(chapters) == 2
    # Each chapter must carry at least one content block whose
    # originating element still has the data-dart-* attributes.
    for chapter_dict, expected_block_id in zip(
        chapters, ["p1", "p2"], strict=True
    ):
        assert chapter_dict.get("contentBlocks"), (
            f"chapter {chapter_dict['headingText']!r} has no content "
            f"blocks — fallback dropped the provenance-carrying body"
        )
        # We also verify via a direct re-extraction of the typed
        # chapter objects that the underlying `element` reference is
        # intact — this is what downstream DART-provenance walkers
        # read.
    # Re-extract using the internal path so we can see ContentBlock
    # `element` references (which aren't serialized in to_dict).
    from bs4 import BeautifulSoup

    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor as _E,
    )
    soup = BeautifulSoup(html, "html.parser")
    ex = _E()
    typed_chapters = ex._build_chapters_from_headings(soup)
    assert len(typed_chapters) == 2
    expected_ids = ["p1", "p2"]
    for chapter, expected in zip(typed_chapters, expected_ids, strict=True):
        found = False
        # Paragraphs may attach directly to the chapter (no sections
        # since there were no h4s) OR to a section.
        buckets = [chapter.content_blocks]
        for sec in chapter.sections:
            buckets.append(sec.content_blocks)
            for sub in sec.subsections:
                buckets.append(sub.content_blocks)
        for blocks in buckets:
            for block in blocks:
                elem = block.element
                if elem is not None and elem.get("data-dart-block-id") == expected:
                    found = True
                    break
            if found:
                break
        assert found, (
            f"chapter {chapter.heading_text!r} has no ContentBlock with "
            f"data-dart-block-id={expected!r} — provenance lost"
        )


# ---------------------------------------------------------------------------
# Live DART output smoke
# ---------------------------------------------------------------------------


_LIVE_DART_SAMPLES = [
    "DART/output/rdf11_primer_accessible.html",
    "DART/output/shacl_advanced_features_accessible.html",
    "DART/output/rdf_schema_accessible.html",
]


@pytest.mark.parametrize("sample_path", _LIVE_DART_SAMPLES)
def test_live_dart_sample_produces_titled_chapters(sample_path):
    """Pointed at a real DART output file from the 2026-04-24 pipeline
    run, the extractor must now emit at least one chapter with a
    non-empty title. Auto-skips when the file isn't materialised so
    CI doesn't require live DART artefacts to stay green."""
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / sample_path
    if not target.exists():
        pytest.skip(f"Live DART sample not available: {sample_path}")

    extractor = SemanticStructureExtractor()
    result = extractor.extract(
        target.read_text(encoding="utf-8"),
        source_path=str(target),
    )
    chapters = result["chapters"]
    assert len(chapters) >= 1, f"got zero chapters from {sample_path}"
    titled = [c for c in chapters if c.get("headingText")]
    assert len(titled) == len(chapters), (
        f"{sample_path}: "
        f"{len(chapters) - len(titled)}/{len(chapters)} chapters have "
        f"empty titles"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
