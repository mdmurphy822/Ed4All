"""SEMANTIK_BOX_TITLE_HEADINGS exclusion contract — the chunker section source.

``HTMLContentParser._extract_sections`` regexes ``<hN>`` boundaries into
``ContentSection`` objects the chunker consumes (incl. the
``ED4ALL_CHUNK_SECTION_HARD_BREAK`` path). A presentational callout box title
(``data-semantik-box-title`` <h4>/<h5>) must NEVER open a ContentSection — its
verbatim text folds into the enclosing section's body instead, so no false
chunk/section boundary is minted and no content is lost.
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLContentParser

_HTML = """
<h2>1.1 Add Whole Numbers</h2>
<p>Whole numbers are the counting numbers plus zero.</p>
<section aria-labelledby="example-1-1">
  <h4 data-semantik-box-title="1" id="example-1-1">Example 1.1</h4>
  <p>Add 3 and 4.</p>
</section>
<section aria-labelledby="solution-1">
  <h5 data-semantik-box-title="1" id="solution-1">Solution</h5>
  <p>The sum is 7.</p>
</section>
<h2>1.2 Subtract Whole Numbers</h2>
<p>Subtraction is the inverse of addition.</p>
"""


def _sections(html: str):
    return HTMLContentParser()._extract_sections(html)


def test_box_titles_do_not_open_sections():
    headings = [s.heading for s in _sections(_HTML)]
    # The two real section headings are the ONLY boundaries.
    assert headings == ["1.1 Add Whole Numbers", "1.2 Subtract Whole Numbers"]
    assert "Example 1.1" not in headings
    assert "Solution" not in headings


def test_box_title_content_is_not_lost():
    sections = _sections(_HTML)
    first = next(s for s in sections if s.heading == "1.1 Add Whole Numbers")
    # The example / solution box titles + their bodies fold into the enclosing
    # section's content (no content dropped — only the false boundary removed).
    assert "Example 1.1" in first.content
    assert "Add 3 and 4" in first.content
    assert "Solution" in first.content
    assert "The sum is 7" in first.content


def test_only_box_titles_removed_normal_headings_kept():
    # A normal h4 (no box-title marker) still opens its own section.
    html = _HTML.replace(
        '<h4 data-semantik-box-title="1" id="example-1-1">Example 1.1</h4>',
        '<h4 id="real">Real Subheading</h4>',
    )
    headings = [s.heading for s in _sections(html)]
    assert "Real Subheading" in headings
    assert "Solution" not in headings  # the remaining box title is still skipped
