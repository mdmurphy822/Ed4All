"""Wave 19 semantic_structure_extractor ``doc-chapter`` path tests.

DART's Wave 13+ converter emits every chapter as an
``<article role="doc-chapter">`` wrapper. The pre-Wave-19 extractor
only knew how to group chapters by heading hierarchy — fed the new
shape, it emitted 90 chapters with ``title=None`` and 0 sections
each on Bates. These tests lock in the restored grouping path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lib.semantic_structure_extractor import SemanticStructureExtractor


_BATES_PATH = (
    "/home/mdmur/Projects/Ed4All/.claude/worktrees/agent-a0855638/"
    "DART/output/bates_wave17/bates_teaching_digital_age_accessible.html"
)


_DOC_CHAPTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Test Doc</title></head>
<body>
  <header><h1 id="main-content-heading">Test Doc</h1></header>
  <main id="main-content" role="main" class="dart-document">
    <article class="dart-section" role="doc-chapter" id="chap-1">
      <header><h2>Introduction to Biology</h2></header>
    </article>
    <p>Biology is the study of life.</p>
    <section class="dart-section" role="region" aria-labelledby="sec-1-h"
             data-dart-source="dart_converter" data-dart-block-id="s1">
      <h2 id="sec-1-h">Cells</h2>
      <p>Cells are the basic unit of life.</p>
    </section>
    <article class="dart-section" role="doc-chapter" id="chap-2">
      <header><h2>Genetics</h2></header>
    </article>
    <p>Genetics is the study of heredity.</p>
  </main>
</body>
</html>
"""


_LEGACY_H2_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>Legacy</title></head>
<body>
  <h1>Legacy Textbook</h1>
  <h2 id="ch1">Chapter One: Getting Started</h2>
  <p>Welcome to the textbook.</p>
  <h2 id="ch2">Chapter Two: Going Further</h2>
  <p>Advanced topics.</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# doc-chapter primary path
# ---------------------------------------------------------------------------


def test_extractor_finds_doc_chapter_articles():
    """Wave 19 primary path: every ``<article role="doc-chapter">`` becomes
    one chapter."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_DOC_CHAPTER_HTML)
    assert len(result["chapters"]) == 2
    titles = [c["headingText"] for c in result["chapters"]]
    assert "Introduction to Biology" in titles
    assert "Genetics" in titles


def test_extractor_preserves_chapter_id_from_article():
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_DOC_CHAPTER_HTML)
    assert result["chapters"][0]["id"] == "chap-1"
    assert result["chapters"][1]["id"] == "chap-2"


def test_extractor_non_none_titles_on_doc_chapter_articles():
    """Regression guard: the stale extractor emitted title=None for
    every article. Wave 19 path must deliver a real string."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_DOC_CHAPTER_HTML)
    for chapter in result["chapters"]:
        assert chapter["headingText"], (
            f"chapter {chapter['id']} has empty title"
        )
        assert chapter["headingText"] is not None


def test_extractor_picks_up_sibling_sections_as_chapter_sections():
    """Wave 13 DART emits sections as siblings of the chapter article
    (not as children). The sibling-walk fallback must collect them
    until the next ``<article role="doc-chapter">``.
    """
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_DOC_CHAPTER_HTML)
    chapters = result["chapters"]
    # Chapter 1 has one sibling <section>; Chapter 2 has none before EOD.
    sections_ch1 = chapters[0].get("sections", [])
    assert len(sections_ch1) == 1
    assert "Cells" in sections_ch1[0]["headingText"]


# ---------------------------------------------------------------------------
# Legacy h2 fallback path
# ---------------------------------------------------------------------------


def test_legacy_h2_hierarchy_path_still_works():
    """When no doc-chapter articles are present, the extractor falls
    back to the pre-Wave-19 h2-grouping heuristic."""
    extractor = SemanticStructureExtractor()
    result = extractor.extract(_LEGACY_H2_HTML)
    assert len(result["chapters"]) == 2
    titles = [c["headingText"] for c in result["chapters"]]
    assert "Chapter One: Getting Started" in titles
    assert "Chapter Two: Going Further" in titles


# ---------------------------------------------------------------------------
# Bates smoke test — slow + skippable when the artifact isn't present
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bates_has_at_least_30_real_chapter_titles():
    """Bates real-world smoke: the full textbook renders at least 30
    chapters with non-None titles through the Wave 19 extractor."""
    if not Path(_BATES_PATH).exists():
        # Also try the worktree-local output (if the orchestrator
        # re-ran the pipeline against Bates for verification).
        alt = Path("/tmp/bates_wave19/bates.html")
        if not alt.exists():
            pytest.skip("Bates HTML not available")
        path = alt
    else:
        path = Path(_BATES_PATH)

    html = path.read_text(encoding="utf-8")
    extractor = SemanticStructureExtractor()
    result = extractor.extract(html)
    non_none_titles = [
        c for c in result["chapters"] if c.get("headingText")
    ]
    assert len(non_none_titles) >= 30, (
        f"expected >= 30 chapter titles, got {len(non_none_titles)} "
        f"out of {len(result['chapters'])} chapters"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
