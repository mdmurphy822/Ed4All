"""Build #23 Tier-3 — parser harvest of ``data-dart-subclass`` off unit wrappers.

Mirrors ``test_data_dart_unit_extraction.py``: the subclass rides the SAME
``<section class="dart-unit">`` wrapper as ``data-dart-unit`` and is harvested
onto the lead ContentSection. Legacy HTML (no attribute) yields None.
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLContentParser


def _unit_html(with_subclass: bool) -> str:
    sub = ' data-dart-subclass="symbolic-manipulation"' if with_subclass else ""
    return (
        "<html><body>"
        f'<section class="dart-unit dart-unit-worked_example" '
        f'data-dart-unit="worked_example"{sub} role="group" data-dart-pages="3-4">'
        '<section class="dart-section" data-dart-block-id="s1">'
        "<h3>Example 1.1</h3><p>Solve 2x plus 3 equals 7 by isolating x here now.</p>"
        "</section></section>"
        "</body></html>"
    )


def test_harvest_subclass_onto_lead_section():
    mod = HTMLContentParser().parse(_unit_html(True))
    sections = [s for s in mod.sections if s.heading]
    assert sections
    lead = sections[0]
    assert lead.data_dart_unit == "worked_example"
    assert lead.data_dart_subclass == "symbolic-manipulation"


def test_legacy_html_without_subclass_is_none():
    mod = HTMLContentParser().parse(_unit_html(False))
    sections = [s for s in mod.sections if s.heading]
    assert sections
    assert sections[0].data_dart_unit == "worked_example"
    assert sections[0].data_dart_subclass is None
