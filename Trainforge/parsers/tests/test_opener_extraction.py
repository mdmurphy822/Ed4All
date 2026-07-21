"""A7 — HTMLContentParser harvests the opener role the SemantiK adapter stamps
on a promoted pedagogical-opener heading.

The parser dual-reads both the current ``data-semantik-opener`` spelling and the
legacy ``data-dart-*`` spelling; the harvested value lands on the
``ContentSection.data_dart_opener`` field. Synthetic HTML strings only (no
corpus files).
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLContentParser

# The adapter emits the opener role on the <section> WRAPPER (which precedes the
# <h4>). The parser must walk back to that wrapper to resolve it.
_OPENER_HTML = """
<main>
<article role="doc-chapter">
<h2>Chapter 9 Roots and Radicals</h2>
<section class="semantik-section" aria-labelledby="s1" data-semantik-block-id="s1"
         data-semantik-source="synthesized">
  <h3 id="s1">9.1 Simplify Square Roots</h3>
</section>
<section class="semantik-section" aria-labelledby="try-it-9-1"
         data-semantik-block-id="try-it-9-1" data-semantik-source="synthesized"
         data-semantik-opener="try_it">
  <h4 id="try-it-9-1">Try It 9.1</h4>
</section>
<section class="semantik-section" id="p2" data-semantik-block-id="p2"
         data-semantik-source="synthesized">
  <p>Ordinary body prose with no opener.</p>
</section>
</article>
</main>
"""

# Legacy-compat: the pre-purge ``data-dart-*`` spelling still harvests (dual-read).
_LEGACY_OPENER_HTML = """
<main>
<article role="doc-chapter">
<h2>Chapter 9 Roots and Radicals</h2>
<section class="dart-section" aria-labelledby="try-it-9-1"
         data-dart-block-id="try-it-9-1" data-dart-source="synthesized"
         data-dart-opener="try_it">
  <h4 id="try-it-9-1">Try It 9.1</h4>
</section>
</article>
</main>
"""


def _sections_by_heading(html):
    parsed = HTMLContentParser().parse(html)
    return {s.heading: s for s in parsed.sections}


def test_opener_role_harvested_from_section_wrapper():
    by_heading = _sections_by_heading(_OPENER_HTML)
    assert "Try It 9.1" in by_heading
    opener_sec = by_heading["Try It 9.1"]
    assert opener_sec.data_dart_opener == "try_it"
    assert opener_sec.level == 4


def test_non_opener_sections_have_none():
    by_heading = _sections_by_heading(_OPENER_HTML)
    assert by_heading["9.1 Simplify Square Roots"].data_dart_opener is None


def test_legacy_data_dart_opener_still_harvested():
    # Legacy-compat read path for the pre-purge attribute spelling.
    by_heading = _sections_by_heading(_LEGACY_OPENER_HTML)
    assert by_heading["Try It 9.1"].data_dart_opener == "try_it"
