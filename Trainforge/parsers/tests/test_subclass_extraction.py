"""Build #23 Tier-3 — parser harvest of the unit ``subclass`` off unit wrappers.

Mirrors ``test_unit_extraction.py``: the subclass rides the SAME
``<section class="semantik-unit">`` wrapper as the unit type and is harvested
onto the lead ContentSection (``ContentSection.data_dart_subclass``). The parser
dual-reads both the current ``data-semantik-*`` spelling and the legacy
``data-dart-*`` spelling. Legacy HTML (no attribute) yields None.
"""
from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLContentParser


def _unit_html(with_subclass: bool, attr: str = "semantik") -> str:
    sub = f' data-{attr}-subclass="symbolic-manipulation"' if with_subclass else ""
    return (
        "<html><body>"
        f'<section class="{attr}-unit {attr}-unit-worked_example" '
        f'data-{attr}-unit="worked_example"{sub} role="group" data-{attr}-pages="3-4">'
        f'<section class="{attr}-section" data-{attr}-block-id="s1">'
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


def test_legacy_data_dart_subclass_still_harvested():
    # Legacy-compat read path for the pre-purge attribute spelling.
    mod = HTMLContentParser().parse(_unit_html(True, attr="dart"))
    sections = [s for s in mod.sections if s.heading]
    assert sections
    assert sections[0].data_dart_unit == "worked_example"
    assert sections[0].data_dart_subclass == "symbolic-manipulation"
