"""Dynamic content-page count tests.

Verifies the user directive:
    "number of html files per week should be dynamic based on learning
    objectives identified"

:func:`MCP.tools._content_gen_helpers.build_week_data` must return a
``content_modules`` list whose length grows with the number of learning
objectives / distinct source topics for that week.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _mk_topic(heading: str, source_file: str = "ch1") -> dict:
    return {
        "heading": heading,
        "paragraphs": [
            f"Body text for {heading}. " * 6,
            "Second paragraph with additional detail.",
        ],
        "key_terms": [heading.split()[0]],
        "source_file": source_file,
        "word_count": 60,
        "extracted_lo_statements": [],
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


def _mk_obj(obj_id: str, statement: str) -> dict:
    return {
        "id": obj_id,
        "statement": statement,
        "bloom_level": "understand",
        "bloom_verb": "describe",
        "key_concepts": [],
    }


# ---------------------------------------------------------------------- #
# Direct build_week_data tests
# ---------------------------------------------------------------------- #


class TestDynamicContentPageCount:
    def test_three_los_yields_three_content_modules(self):
        """User directive: 3 LOs → 3 content modules."""
        week_topics = [
            _mk_topic("Introduction"),
            _mk_topic("Stages"),
            _mk_topic("Applications"),
        ]
        week_objectives = [
            _mk_obj("TO-01", "Describe introductory concepts."),
            _mk_obj("TO-02", "Explain the stages."),
            _mk_obj("CO-01", "Apply concepts to new examples."),
        ]
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=week_topics,
            week_objectives=week_objectives,
            all_objectives=week_objectives,
            course_code="BIO_101",
        )
        assert len(wd["content_modules"]) == 3

    def test_one_lo_yields_one_content_module(self):
        """User directive: minimal week → 1 content module."""
        week_topics = [_mk_topic("Introduction")]
        week_objectives = [_mk_obj("TO-01", "Describe introductory concepts.")]
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=week_topics,
            week_objectives=week_objectives,
            all_objectives=week_objectives,
            course_code="BIO_101",
        )
        assert len(wd["content_modules"]) == 1

    def test_zero_los_and_zero_topics_yields_one_module_floor(self):
        """Edge: empty corpus still needs 1 content module so the
        integration test's 5-page floor has a chance to be met."""
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=[],
            week_objectives=[],
            all_objectives=[],
            course_code="BIO_101",
        )
        assert len(wd["content_modules"]) == 1

    def test_more_topics_than_los_counts_from_topics(self):
        """When topic count exceeds LO count, the module count uses the
        larger value so no topic gets dropped."""
        week_topics = [
            _mk_topic("Introduction"),
            _mk_topic("Stages"),
            _mk_topic("Applications"),
            _mk_topic("Beyond"),
        ]
        week_objectives = [_mk_obj("TO-01", "Describe all.")]
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=week_topics,
            week_objectives=week_objectives,
            all_objectives=week_objectives,
            course_code="BIO_101",
        )
        assert len(wd["content_modules"]) == 4

    def test_module_titles_come_from_source(self):
        """Every module title must be a real topic heading or LO
        statement (no fabricated prose)."""
        week_topics = [
            _mk_topic("Introduction to Photosynthesis"),
            _mk_topic("The Calvin Cycle"),
        ]
        week_objectives = [
            _mk_obj("TO-01", "Describe photosynthesis."),
            _mk_obj("TO-02", "Explain the Calvin cycle."),
        ]
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=week_topics,
            week_objectives=week_objectives,
            all_objectives=week_objectives,
            course_code="BIO_101",
        )
        titles = [m["title"] for m in wd["content_modules"]]
        assert titles == [
            "Introduction to Photosynthesis",
            "The Calvin Cycle",
        ]


# ---------------------------------------------------------------------- #
# End-to-end check via generate_week: content module count → file count.
# ---------------------------------------------------------------------- #


class TestDynamicContentPagesEmitted:
    def test_three_modules_produces_three_content_html_files(self, tmp_path):
        """Feed 3 content_modules into generate_week — confirm 3 content
        HTML files land on disk."""
        from Courseforge.scripts import generate_course as _gen

        week_topics = [
            _mk_topic("Alpha Topic"),
            _mk_topic("Beta Topic"),
            _mk_topic("Gamma Topic"),
        ]
        week_objectives = [
            _mk_obj("TO-01", "Objective alpha."),
            _mk_obj("TO-02", "Objective beta."),
            _mk_obj("CO-01", "Objective gamma."),
        ]
        week_data = _cgh.build_week_data(
            week_num=3,
            duration_weeks=3,
            week_topics=week_topics,
            week_objectives=week_objectives,
            all_objectives=week_objectives,
            course_code="TST_101",
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        _gen.generate_week(week_data, output_dir, course_code="TST_101")
        week_dir = output_dir / "week_03"
        content_files = sorted(week_dir.glob("week_03_content_*.html"))
        assert len(content_files) == 3, [p.name for p in content_files]

    def test_single_module_produces_single_content_html_file(self, tmp_path):
        from Courseforge.scripts import generate_course as _gen

        week_topics = [_mk_topic("Only Topic")]
        week_objectives = [_mk_obj("TO-01", "Only objective.")]
        week_data = _cgh.build_week_data(
            week_num=1,
            duration_weeks=1,
            week_topics=week_topics,
            week_objectives=week_objectives,
            all_objectives=week_objectives,
            course_code="TST_101",
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        _gen.generate_week(week_data, output_dir, course_code="TST_101")
        week_dir = output_dir / "week_01"
        content_files = sorted(week_dir.glob("week_01_content_*.html"))
        assert len(content_files) == 1
