"""SEMANTIK_BOX_TITLE_HEADINGS exclusion contract — a presentational callout box
title (``data-semantik-box-title``) is UNCONDITIONALLY invisible to
chapter/section structure derivation (anti-re-poisoning).

The skip is NOT gated on ``ED4ALL_STRUCTURE_EXTRACT_GUARDS`` — a presentational
box title is never a section, on any lane.
"""
from __future__ import annotations

from lib.semantic_structure_extractor.core.heading_parser import HeadingParser

# A section (h2/h3) tree with worked-example / solution / definition box titles
# emitted as data-semantik-box-title <h4>/<h5> INSIDE the callout sections.
_WITH_BOX_TITLES = """
<h1>Whole Numbers</h1>
<article role="doc-chapter" id="chap-1">
<h2>Whole Numbers</h2>
<section class="semantik-section" id="s0"><h3 id="s0h">1.1 Add Whole Numbers</h3></section>
<section class="semantik-section" id="s1" aria-labelledby="example-1-1-s1">
  <h4 data-semantik-box-title="1" id="example-1-1-s1">Example 1.1</h4>
  <p>Add 3 and 4.</p>
</section>
<section class="semantik-section" id="s2" aria-labelledby="solution-s2">
  <h5 data-semantik-box-title="1" id="solution-s2">Solution</h5>
  <p>The sum is 7.</p>
</section>
<section class="semantik-section" id="s3"><h3 id="s3h">1.2 Subtract Whole Numbers</h3></section>
</article>
"""

# The SAME document with the box-title headings removed entirely.
_WITHOUT_BOX_TITLES = """
<h1>Whole Numbers</h1>
<article role="doc-chapter" id="chap-1">
<h2>Whole Numbers</h2>
<section class="semantik-section" id="s0"><h3 id="s0h">1.1 Add Whole Numbers</h3></section>
<section class="semantik-section" id="s1">
  <p>Add 3 and 4.</p>
</section>
<section class="semantik-section" id="s2">
  <p>The sum is 7.</p>
</section>
<section class="semantik-section" id="s3"><h3 id="s3h">1.2 Subtract Whole Numbers</h3></section>
</article>
"""


def _heading_texts(html: str):
    hierarchy = HeadingParser().parse(html)
    # ``children`` holds child IDs; ``all_nodes`` is the document-order node map.
    return [(n.level, n.text) for n in hierarchy.all_nodes.values()]


def test_box_titles_excluded_from_heading_hierarchy(monkeypatch):
    # Skip is unconditional — assert with the structure-extract guards OFF.
    monkeypatch.delenv("ED4ALL_STRUCTURE_EXTRACT_GUARDS", raising=False)
    headings = _heading_texts(_WITH_BOX_TITLES)
    texts = [t for _lvl, t in headings]
    # The real sections survive; the box titles never enter the hierarchy.
    assert "1.1 Add Whole Numbers" in texts
    assert "1.2 Subtract Whole Numbers" in texts
    assert "Example 1.1" not in texts
    assert "Solution" not in texts
    # No h4/h5 box-title level leaked in.
    assert all(lvl <= 3 for lvl, _t in headings), headings


def test_structure_identical_with_and_without_box_titles(monkeypatch):
    monkeypatch.delenv("ED4ALL_STRUCTURE_EXTRACT_GUARDS", raising=False)
    assert _heading_texts(_WITH_BOX_TITLES) == _heading_texts(_WITHOUT_BOX_TITLES)


def test_skip_holds_with_guards_enabled(monkeypatch):
    monkeypatch.setenv("ED4ALL_STRUCTURE_EXTRACT_GUARDS", "1")
    texts = [t for _lvl, t in _heading_texts(_WITH_BOX_TITLES)]
    assert "Example 1.1" not in texts
    assert "Solution" not in texts
