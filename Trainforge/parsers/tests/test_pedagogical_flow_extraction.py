"""Wave #22 quick-wins — the parser harvests the per-block ``flow`` role values
off the BLOCKS in a section's body onto ``ContentSection.data_dart_flows``.

The parser dual-reads both the current ``data-semantik-flow`` spelling and the
legacy ``data-dart-flow`` spelling. Synthetic HTML strings only (no corpus
files). Backward-compat: HTML without any flow attribute yields an empty list.
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLContentParser

# A worked-example unit: the statement block carries data-semantik-flow="statement",
# the solution block carries data-semantik-flow="solution-steps".
_FLOW_HTML = """
<main><article role="doc-chapter"><h2>Ch 9</h2>
<section class="semantik-unit semantik-unit-worked_example" data-semantik-unit="worked_example"
         role="group">
<section class="semantik-section" data-semantik-block-id="ex-1" data-semantik-source="synthesized"
         data-semantik-opener="worked_example">
<h4>Example 9.1</h4>
</section>
<section class="semantik-section" data-semantik-block-id="p1" data-semantik-flow="statement">
<p>Simplify the radical expression fully.</p>
</section>
<section class="semantik-section" data-semantik-block-id="sol-1" data-semantik-opener="solution">
<h4>Solution</h4>
</section>
<section class="semantik-section" data-semantik-block-id="s1" data-semantik-flow="solution-steps">
<p>First factor out the perfect square, then reduce.</p>
</section>
</section>
</article></main>
"""

# Legacy-compat: the pre-purge ``data-dart-*`` spelling still harvests (dual-read).
# Structurally identical to _FLOW_HTML — only the attribute spelling differs.
_LEGACY_FLOW_HTML = _FLOW_HTML.replace("semantik-", "dart-")

# Legacy Courseforge-style HTML with no data-dart-flow anywhere.
_LEGACY_HTML = """
<main><article><h2>Overview</h2>
<section class="cf-section"><h3>Intro</h3><p>Some prose about the topic.</p></section>
</article></main>
"""


def _sections_by_heading(html):
    parsed = HTMLContentParser().parse(html)
    return {s.heading: s for s in parsed.sections}


def test_statement_flow_harvested_onto_section():
    by_heading = _sections_by_heading(_FLOW_HTML)
    assert by_heading["Example 9.1"].data_dart_flows == ["statement"]


def test_solution_steps_flow_harvested_onto_section():
    by_heading = _sections_by_heading(_FLOW_HTML)
    assert by_heading["Solution"].data_dart_flows == ["solution-steps"]


def test_flows_sorted_and_deduped():
    # Two blocks with the same flow + one different → sorted distinct list.
    html = (
        "<main><article><h2>Ch</h2>"
        '<section data-semantik-block-id="a" data-semantik-flow="procedure-steps">'
        "<p>step one here now</p></section>"
        '<section data-semantik-block-id="b" data-semantik-flow="procedure-steps">'
        "<p>step two here now</p></section>"
        '<section data-semantik-block-id="c" data-semantik-flow="statement">'
        "<p>the claim here now</p></section>"
        "</article></main>"
    )
    by_heading = _sections_by_heading(html)
    assert by_heading["Ch"].data_dart_flows == ["procedure-steps", "statement"]


def test_legacy_data_dart_flow_still_harvested():
    # Legacy-compat read path for the pre-purge attribute spelling.
    by_heading = _sections_by_heading(_LEGACY_FLOW_HTML)
    assert by_heading["Example 9.1"].data_dart_flows == ["statement"]
    assert by_heading["Solution"].data_dart_flows == ["solution-steps"]


def test_legacy_html_has_no_flows():
    by_heading = _sections_by_heading(_LEGACY_HTML)
    for sec in by_heading.values():
        assert sec.data_dart_flows == []
