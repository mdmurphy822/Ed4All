"""Wave #22 Tier-2 — the parser harvests ``data-dart-unit`` off the composite-
unit wrapper onto the unit's LEAD section, and the ``<section class="dart-unit">``
wrapper is transparent to text extraction (no attribute/markup leak).

Synthetic HTML strings only (no corpus files).
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import (
    HTMLContentParser,
    HTMLTextExtractor,
)

# A worked-example composite unit: the <section class="dart-unit"> wrapper folds
# the Example callout-group + the Solution callout-group. Only the LEAD block
# (Example 9.1, the wrapper's first child) is the unit EDGE.
_UNIT_HTML = """
<main><article role="doc-chapter"><h2>Ch 9</h2>
<section class="dart-unit dart-unit-worked_example" data-dart-unit="worked_example"
         role="group" aria-labelledby="ex-9-1">
<div class="dart-callout-group" data-dart-opener-group="worked_example">
<section class="dart-section" aria-labelledby="ex-9-1" data-dart-block-id="ex-9-1"
         data-dart-source="synthesized" data-dart-opener="worked_example">
<h4 id="ex-9-1">Example 9.1</h4>
</section>
<section class="dart-section" id="p1" data-dart-block-id="p1" data-dart-source="synthesized">
<p>Simplify the radical expression fully.</p>
</section>
</div>
<div class="dart-callout-group" data-dart-opener-group="solution">
<section class="dart-section" aria-labelledby="sol-1" data-dart-block-id="sol-1"
         data-dart-source="synthesized" data-dart-opener="solution">
<h4 id="sol-1">Solution</h4>
</section>
</div>
</section>
</article></main>
"""


def _sections_by_heading(html):
    parsed = HTMLContentParser().parse(html)
    return {s.heading: s for s in parsed.sections}


def test_unit_type_harvested_onto_lead_section():
    by_heading = _sections_by_heading(_UNIT_HTML)
    assert by_heading["Example 9.1"].data_dart_unit == "worked_example"


def test_non_lead_section_has_no_unit():
    # The Solution opener is INSIDE the unit but is not its edge.
    by_heading = _sections_by_heading(_UNIT_HTML)
    assert by_heading["Solution"].data_dart_unit is None


def test_unit_wrapper_transparent_to_text_extraction():
    ex = HTMLTextExtractor()
    ex.feed(_UNIT_HTML)
    text = ex.get_text()
    # The unit <section> class / role / aria attributes never leak into text.
    assert "dart-unit" not in text
    assert "worked_example" not in text
    tokens = text.split()
    assert "group" not in tokens
    # The real body content is intact and in reading order.
    assert "Simplify the radical expression fully." in text
    assert text.index("Example 9.1") < text.index("Solution")
