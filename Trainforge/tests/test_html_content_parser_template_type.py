"""Wave 81: HTMLContentParser propagates ``data-cf-template-type`` to sections.

Courseforge Wave 79 C content-generator emits ``<section
data-cf-template-type="...">`` on every page root. The parser must surface
that value as ``ContentSection.template_type`` so the chunker
(``Trainforge.process_course._merge_small_sections``) can carry it through
to ``chunk.chunk_type`` instead of falling back to the heading-keyword
heuristic. Falls back to ``None`` when the attribute is absent (legacy
Courseforge corpora and non-Courseforge IMSCC packages) so the existing
``_type_from_heading`` pathway keeps working.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402


# ---------------------------------------------------------------------------
# Section-level template_type propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template_value",
    [
        "explanation",
        "example",
        "procedure",
        "real_world_scenario",
        "common_pitfall",
        "problem_solution",
        "summary",
        "overview",
    ],
)
def test_section_template_type_extracted(template_value):
    html = f"""<!DOCTYPE html><html><body>
        <section data-cf-template-type="{template_value}"
                 data-cf-objective-id="CO-01">
            <h1>{template_value} title</h1>
            <h3>Body Heading</h3>
            <p>Body content for the {template_value} section.</p>
        </section>
    </body></html>"""
    parsed = HTMLContentParser().parse(html)
    assert parsed.sections, "expected at least one section to be extracted"
    # Both the H1 and H3 sit inside the same section root, so both inherit
    # the template_type.
    for sec in parsed.sections:
        assert sec.template_type == template_value, (
            f"section heading={sec.heading!r} did not inherit "
            f"template_type from <section> root"
        )


def test_section_without_template_type_falls_back_to_none():
    """Legacy Courseforge / non-CF IMSCC: no data-cf-template-type → None.

    Downstream chunker must continue to use ``_type_from_heading`` for these
    sections (existing behavior).
    """
    html = """<!DOCTYPE html><html><body>
        <section data-cf-objective-id="CO-01">
            <h1>Legacy section</h1>
            <h3>Worked Example</h3>
            <p>Walkthrough text for the example heading.</p>
        </section>
    </body></html>"""
    parsed = HTMLContentParser().parse(html)
    assert parsed.sections
    for sec in parsed.sections:
        assert sec.template_type is None, (
            f"section heading={sec.heading!r} should not have a "
            f"template_type when the section root lacks the attribute"
        )


def test_pitfall_template_type_propagates_through_subsections():
    """Mirrors the rdf-shacl-551-2 ``pitfall_01.html`` shape.

    The Wave 79 C ``common_pitfall`` template emits exactly one
    ``<section data-cf-template-type="common_pitfall">`` whose body contains
    multiple H3/H4 subsections (``What looks like the right answer``,
    ``Why it's wrong``, ``The right approach``). Every parsed
    ``ContentSection`` produced from those subsections must inherit the
    parent section's template_type.
    """
    html = """<!DOCTYPE html><html><body>
        <section data-cf-template-type="common_pitfall"
                 data-cf-pitfall-concept="sh-not"
                 data-cf-objective-id="CO-19">
            <h1>Common Pitfall: sh:not vs Negated Value-List</h1>
            <h3>Common Pitfall: treating sh:not as 'value not in list'</h3>
            <p>Intro paragraph explaining the pitfall.</p>
            <h4>What looks like the right answer</h4>
            <p>Misconception statement.</p>
            <h4>Why it's wrong</h4>
            <p>Reason text.</p>
            <h4>The right approach</h4>
            <p>Correct approach text.</p>
        </section>
    </body></html>"""
    parsed = HTMLContentParser().parse(html)
    assert len(parsed.sections) >= 4
    template_types = {sec.template_type for sec in parsed.sections}
    assert template_types == {"common_pitfall"}, (
        f"every subsection must inherit common_pitfall, got {template_types}"
    )


# ---------------------------------------------------------------------------
# Schema accepts all 10 canonical chunk types
# ---------------------------------------------------------------------------


def test_content_type_schema_accepts_all_ten_canonical_chunk_types():
    """schemas/taxonomies/content_type.json::ChunkType must list all ten
    Wave 81 canonical values.
    """
    schema_path = PROJECT_ROOT / "schemas" / "taxonomies" / "content_type.json"
    with schema_path.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    chunk_types = set(schema["$defs"]["ChunkType"]["enum"])
    expected = {
        # Six legacy values
        "assessment_item",
        "overview",
        "summary",
        "exercise",
        "explanation",
        "example",
        # Four Wave 81 additions
        "procedure",
        "real_world_scenario",
        "common_pitfall",
        "problem_solution",
    }
    assert chunk_types == expected, (
        f"ChunkType enum mismatch: got {sorted(chunk_types)}, "
        f"expected {sorted(expected)}"
    )


# ---------------------------------------------------------------------------
# Process_course chunker integration
# ---------------------------------------------------------------------------


def test_chunker_honors_template_type_over_heading_heuristic():
    """When a section carries data-cf-template-type, the chunker uses it
    even if the heading text would have triggered the legacy heuristic.

    Heading "Worked Example" would resolve to chunk_type=example via
    ``_type_from_heading``. With ``data-cf-template-type="procedure"`` on the
    enclosing section, the chunk must instead carry chunk_type=procedure.
    """
    from Trainforge.process_course import CourseProcessor
    from Trainforge.parsers.html_content_parser import ContentSection

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    proc.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    sections = [
        ContentSection(
            heading="Worked Example",
            level=3,
            content="A worked walkthrough that would be ``example`` by heading.",
            word_count=11,
            template_type="procedure",
        ),
    ]
    merged = proc._merge_small_sections(sections)
    assert merged
    _, _, chunk_type, _, _ = merged[0]
    assert chunk_type == "procedure", (
        f"expected template_type to override heading heuristic, got {chunk_type}"
    )


def test_chunker_falls_back_to_heading_heuristic_when_template_type_absent():
    from Trainforge.process_course import CourseProcessor
    from Trainforge.parsers.html_content_parser import ContentSection

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    proc.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    sections = [
        ContentSection(
            heading="Worked Example",
            level=3,
            content="Without template_type, the legacy heading heuristic fires.",
            word_count=10,
            template_type=None,
        ),
    ]
    merged = proc._merge_small_sections(sections)
    assert merged
    _, _, chunk_type, _, _ = merged[0]
    assert chunk_type == "example", (
        f"expected legacy heading heuristic when template_type is None, "
        f"got {chunk_type}"
    )


def test_chunker_rejects_non_canonical_template_type():
    """Off-spec template_type values fall back to the heading heuristic."""
    from Trainforge.process_course import CourseProcessor
    from Trainforge.parsers.html_content_parser import ContentSection

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    proc.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    sections = [
        ContentSection(
            heading="Worked Example",
            level=3,
            content="Text body.",
            word_count=2,
            template_type="off-spec-not-canonical",
        ),
    ]
    merged = proc._merge_small_sections(sections)
    _, _, chunk_type, _, _ = merged[0]
    assert chunk_type == "example"


def test_chunker_first_section_template_type_wins_in_merge_group():
    from Trainforge.process_course import CourseProcessor
    from Trainforge.parsers.html_content_parser import ContentSection

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    proc.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    sections = [
        ContentSection(
            heading="Pitfall",
            level=3,
            content="First small section.",
            word_count=3,
            template_type="common_pitfall",
        ),
        ContentSection(
            heading="Why it's wrong",
            level=4,
            content="Second small section.",
            word_count=3,
            template_type=None,
        ),
    ]
    merged = proc._merge_small_sections(sections)
    assert len(merged) == 1, (
        "small sections should collapse into one merge group"
    )
    _, _, chunk_type, _, _ = merged[0]
    assert chunk_type == "common_pitfall"
