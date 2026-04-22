"""Wave 28: verify overview pages emit one <li data-cf-objective-id="..."> per
LO, never an empty <ul></ul> block.

The pre-Wave-28 bug: every week's Overview carried a literal ``<ul></ul>``
under "Learning Objectives" because objective synthesis never ran or
produced empty output. These tests lock in the populated <ul> invariant
and check that the per-LO attributes match the schema-expected pattern.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402
from Courseforge.scripts import generate_course as _gen  # noqa: E402


def _topic(heading: str, paragraph: str, chapter_id: str = "ch1") -> dict:
    return {
        "heading": heading,
        "paragraphs": [paragraph],
        "key_terms": [],
        "source_file": "synth",
        "word_count": len(paragraph.split()),
        "chapter_id": chapter_id,
        "dart_block_ids": [],
        "extracted_lo_statements": [],
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


def test_overview_ul_has_one_li_per_objective(tmp_path: Path):
    topic = _topic(
        "Formative vs Summative Assessment",
        "Paragraph prose about assessment purposes and timing in a course.",
    )
    objectives = [
        {
            "id": "TO-01",
            "statement": "Distinguish formative and summative assessment.",
            "bloom_level": "understand",
            "bloom_verb": "distinguish",
        },
        {
            "id": "CO-01",
            "statement": "Identify appropriate assessment timing within a module.",
            "bloom_level": "apply",
            "bloom_verb": "identify",
        },
    ]
    wd = _cgh.build_week_data(
        week_num=1,
        duration_weeks=2,
        week_topics=[topic],
        week_objectives=objectives,
        all_objectives=objectives,
        course_code="SYNTH_101",
    )
    _gen.generate_week(wd, tmp_path, "SYNTH_101")
    overview = (tmp_path / "week_01" / "week_01_overview.html").read_text(
        encoding="utf-8"
    )
    # Every LO must be represented as exactly one <li data-cf-objective-id="...">.
    for obj in objectives:
        pattern = (
            rf'<li[^>]*data-cf-objective-id="{re.escape(obj["id"])}"'
        )
        assert re.search(pattern, overview), (
            f"Expected <li data-cf-objective-id={obj['id']!r}> in overview"
        )


def test_overview_ul_not_empty(tmp_path: Path):
    """The <ul> under Learning Objectives must contain at least one child
    when objectives were supplied."""
    topic = _topic(
        "Cognitive Load Theory",
        "Paragraph prose about working memory and instructional design.",
    )
    objectives = [{
        "id": "TO-01",
        "statement": "Explain the three types of cognitive load.",
        "bloom_level": "understand",
        "bloom_verb": "explain",
    }]
    wd = _cgh.build_week_data(
        week_num=1,
        duration_weeks=1,
        week_topics=[topic],
        week_objectives=objectives,
        all_objectives=objectives,
        course_code="SYNTH_101",
    )
    _gen.generate_week(wd, tmp_path, "SYNTH_101")
    overview = (tmp_path / "week_01" / "week_01_overview.html").read_text(
        encoding="utf-8"
    )
    # Must not have <ul></ul> (possibly with whitespace) under a
    # "Learning Objectives" heading.
    assert not re.search(
        r"Learning Objectives</h2>[\s\S]{0,80}?<ul>\s*</ul>",
        overview,
        re.IGNORECASE,
    ), "Empty <ul></ul> leaked into overview under Learning Objectives"


def test_per_objective_bloom_attributes_attached(tmp_path: Path):
    topic = _topic("Metacognition", "Paragraph prose about self-regulation.")
    objectives = [{
        "id": "TO-01",
        "statement": "Evaluate your own understanding using a reflection prompt.",
        "bloom_level": "evaluate",
        "bloom_verb": "evaluate",
    }]
    wd = _cgh.build_week_data(
        week_num=1,
        duration_weeks=1,
        week_topics=[topic],
        week_objectives=objectives,
        all_objectives=objectives,
        course_code="SYNTH_101",
    )
    _gen.generate_week(wd, tmp_path, "SYNTH_101")
    overview = (tmp_path / "week_01" / "week_01_overview.html").read_text(
        encoding="utf-8"
    )
    li_match = re.search(
        r'<li[^>]*data-cf-objective-id="TO-01"[^>]*>',
        overview,
    )
    assert li_match, "Objective LI not found"
    assert 'data-cf-bloom-level="evaluate"' in li_match.group(0)
    assert 'data-cf-bloom-verb="evaluate"' in li_match.group(0)


def test_overview_objectives_include_statement_text(tmp_path: Path):
    topic = _topic(
        "Group Work in Online Courses",
        "Paragraph about synchronous and asynchronous group configurations.",
    )
    objectives = [{
        "id": "TO-01",
        "statement": "Design a small-group activity for asynchronous delivery.",
        "bloom_level": "create",
        "bloom_verb": "design",
    }]
    wd = _cgh.build_week_data(
        week_num=1,
        duration_weeks=1,
        week_topics=[topic],
        week_objectives=objectives,
        all_objectives=objectives,
        course_code="SYNTH_101",
    )
    _gen.generate_week(wd, tmp_path, "SYNTH_101")
    overview = (tmp_path / "week_01" / "week_01_overview.html").read_text(
        encoding="utf-8"
    )
    # The LO statement (or a distinctive substring) must appear in the overview.
    assert "small-group activity" in overview, (
        "Expected LO statement substring in rendered overview"
    )
