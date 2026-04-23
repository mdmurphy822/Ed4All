"""REC-JSL-03 (Wave 3, Worker M) — activity / self-check objective_ref ingest.

Courseforge emits ``data-cf-objective-ref`` on ``.activity-card`` and
``.self-check`` elements at ``generate_course.py:378,491`` when a
curriculum JSON entry carries an ``objective_ref``. Worker M extends
Trainforge's HTML parser (``Trainforge/parsers/html_content_parser.py``)
to harvest those attrs into ``ContentSection.objective_refs`` /
``ParsedHTMLModule.objective_refs`` and extends
``process_course._extract_objective_refs`` to merge the section-scoped
refs (with page-level fallback) into each chunk's
``learning_outcome_refs``. This materializes the Activity→LO edge in
the KG for the first time.

Tests exercise two layers:
  1. Parser: ``HTMLContentParser.parse(html)`` returns the refs on the
     matching section and on the module.
  2. ``_extract_objective_refs``: the merged ref appears in the chunk's
     outcome list (with and without the ``TRAINFORGE_PRESERVE_LO_CASE``
     case-preservation flag).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Project root (Ed4All/). This file lives at
# Ed4All/Trainforge/tests/test_activity_objective_ref.py → parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402
from Trainforge.process_course import CourseProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_with_activity(objective_ref: str) -> str:
    """Emit a minimal page with one section containing one activity-card."""
    return f"""<!DOCTYPE html>
<html>
  <head><title>Sample Page</title></head>
  <body>
    <h2>Practice Section</h2>
    <p>Intro text for the section.</p>
    <div class="activity-card"
         data-cf-component="activity"
         data-cf-purpose="practice"
         data-cf-teaching-role="transfer"
         data-cf-bloom-level="apply"
         data-cf-objective-ref="{objective_ref}">
      <h3>Activity 1: Apply the concept</h3>
      <p>Work through a short scenario.</p>
    </div>
  </body>
</html>"""


def _page_with_self_check(objective_ref: str) -> str:
    """Emit a minimal page with one section containing one self-check."""
    return f"""<!DOCTYPE html>
<html>
  <head><title>Sample Page</title></head>
  <body>
    <h2>Check Your Understanding</h2>
    <p>Answer the following questions.</p>
    <div class="self-check"
         data-cf-component="self-check"
         data-cf-purpose="formative-assessment"
         data-cf-teaching-role="assess"
         data-cf-bloom-level="remember"
         data-cf-objective-ref="{objective_ref}">
      <h3>Question 1</h3>
      <p>What is REST?</p>
    </div>
  </body>
</html>"""


def _page_with_two_activities_same_ref(objective_ref: str) -> str:
    """Emit a page with two activity-cards citing the same objective ref."""
    return f"""<!DOCTYPE html>
<html>
  <head><title>Sample Page</title></head>
  <body>
    <h2>Practice Section</h2>
    <p>Intro.</p>
    <div class="activity-card"
         data-cf-component="activity"
         data-cf-purpose="practice"
         data-cf-objective-ref="{objective_ref}">
      <h3>Activity 1</h3>
      <p>First exercise.</p>
    </div>
    <div class="activity-card"
         data-cf-component="activity"
         data-cf-purpose="practice"
         data-cf-objective-ref="{objective_ref}">
      <h3>Activity 2</h3>
      <p>Second exercise.</p>
    </div>
  </body>
</html>"""


def _build_item(parsed, heading: str) -> dict:
    """Assemble the minimal item dict ``_extract_objective_refs`` reads.

    Mirrors the fields set in ``CourseProcessor._parse_html`` at
    ``process_course.py:934``.
    """
    return {
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "sections": parsed.sections,
        "objective_refs": parsed.objective_refs,
    }


# ---------------------------------------------------------------------------
# Layer 1 — Parser extracts objective_refs onto ContentSection + module
# ---------------------------------------------------------------------------

def _find_section(parsed, heading: str):
    """Return the first section whose heading matches ``heading``.

    The parser splits sections on every heading level (h1–h6), so the
    activity-card's inner ``<h3>`` creates an extra (empty) section
    after the outer ``<h2>``. Tests care only about the outer section
    where the ref was harvested.
    """
    for sec in parsed.sections:
        if sec.heading == heading:
            return sec
    raise AssertionError(
        f"No section with heading {heading!r} in {[s.heading for s in parsed.sections]}"
    )


def test_parser_extracts_activity_objective_ref_onto_section():
    """Parser: activity-card's data-cf-objective-ref surfaces on section."""
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_activity("CO-05"))

    section = _find_section(parsed, "Practice Section")
    assert section.objective_refs == ["CO-05"], (
        f"expected ['CO-05'] on section, got {section.objective_refs}"
    )
    # Page-level union must include the same ref.
    assert parsed.objective_refs == ["CO-05"]


def test_parser_extracts_self_check_objective_ref_onto_section():
    """Parser: self-check's data-cf-objective-ref surfaces on section."""
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_self_check("TO-02"))

    section = _find_section(parsed, "Check Your Understanding")
    assert section.objective_refs == ["TO-02"]
    assert parsed.objective_refs == ["TO-02"]


def test_parser_deduplicates_repeated_objective_refs():
    """Parser: same ref on two activities in one section → deduped list."""
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_two_activities_same_ref("CO-05"))

    section = _find_section(parsed, "Practice Section")
    # Sorted unique only — no duplicates even though two activity-cards
    # carried the same data-cf-objective-ref.
    assert section.objective_refs == ["CO-05"]
    assert parsed.objective_refs == ["CO-05"]


# ---------------------------------------------------------------------------
# Layer 2 — _extract_objective_refs attaches refs to chunk outcome list
# ---------------------------------------------------------------------------

def test_activity_objective_ref_parses_into_learning_outcome_refs(monkeypatch):
    """JSL-03: activity-card ref appears in chunk's learning_outcome_refs."""
    # Default env → lowercased ref (backward-compat).
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_activity("CO-05"))

    assert parsed.sections, "parser must return at least one section"
    item = _build_item(parsed, parsed.sections[0].heading)
    stub = SimpleNamespace(
        WEEK_PREFIX_RE=CourseProcessor.WEEK_PREFIX_RE,
        OBJECTIVE_CODE_RE=CourseProcessor.OBJECTIVE_CODE_RE,
    )
    refs = CourseProcessor._extract_objective_refs(
        stub, item, section_heading=parsed.sections[0].heading
    )
    # Default (flag off) → lowercased.
    assert "co-05" in refs, f"expected 'co-05' in refs; got {refs}"


def test_self_check_objective_ref_parses_into_learning_outcome_refs(monkeypatch):
    """JSL-03: self-check ref appears in chunk's learning_outcome_refs."""
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_self_check("TO-02"))

    assert parsed.sections
    item = _build_item(parsed, parsed.sections[0].heading)
    stub = SimpleNamespace(
        WEEK_PREFIX_RE=CourseProcessor.WEEK_PREFIX_RE,
        OBJECTIVE_CODE_RE=CourseProcessor.OBJECTIVE_CODE_RE,
    )
    refs = CourseProcessor._extract_objective_refs(
        stub, item, section_heading=parsed.sections[0].heading
    )
    assert "to-02" in refs, f"expected 'to-02' in refs; got {refs}"


def test_activity_objective_ref_deduped(monkeypatch):
    """JSL-03: same ref on multiple activities → single entry on chunk."""
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_two_activities_same_ref("CO-05"))

    assert parsed.sections
    item = _build_item(parsed, parsed.sections[0].heading)
    stub = SimpleNamespace(
        WEEK_PREFIX_RE=CourseProcessor.WEEK_PREFIX_RE,
        OBJECTIVE_CODE_RE=CourseProcessor.OBJECTIVE_CODE_RE,
    )
    refs = CourseProcessor._extract_objective_refs(
        stub, item, section_heading=parsed.sections[0].heading
    )
    # Exactly one occurrence of the ref even though two activity-cards
    # cited it.
    assert refs.count("co-05") == 1, (
        f"expected exactly one 'co-05' in refs; got {refs}"
    )


def test_activity_objective_ref_fallback_to_page_level_when_no_section_match(monkeypatch):
    """No section match → fall back to page-level objective_refs."""
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    parser = HTMLContentParser()
    parsed = parser.parse(_page_with_activity("CO-05"))

    item = _build_item(parsed, parsed.sections[0].heading)
    stub = SimpleNamespace(
        WEEK_PREFIX_RE=CourseProcessor.WEEK_PREFIX_RE,
        OBJECTIVE_CODE_RE=CourseProcessor.OBJECTIVE_CODE_RE,
    )
    # Pass a heading that doesn't match any section → the function must
    # still pick up the page-level objective_refs fallback.
    refs = CourseProcessor._extract_objective_refs(
        stub, item, section_heading="A heading that does not exist"
    )
    assert "co-05" in refs, (
        f"fallback to page-level objective_refs failed; got {refs}"
    )
