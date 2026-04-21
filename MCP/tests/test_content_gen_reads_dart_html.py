"""Wave 28: verify content-gen reads real DART HTML into every week's pages.

Pre-Wave-28, the full textbook-to-course pipeline could run end-to-end and
emit weekly pages whose Content body was just an H1/H2 skeleton — because the
week had no bound topic. These tests lock in the invariant that when DART
HTML carries ``<article role="doc-chapter">`` wrappers with real paragraph
prose, the corresponding Courseforge week pages contain that prose (not a
template skeleton).

All fixtures are synthetic; no references to any specific textbook, author,
or publisher are embedded. When adding new assertions, keep fixture prose
neutral and generic.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


# ---------------------------------------------------------------------- #
# Synthetic DART HTML fixtures
# ---------------------------------------------------------------------- #


_CHAPTER_A_PROSE_A = (
    "This chapter examines how digital tools reshape instructional design. "
    "Interactive modules now supplement lectures, and instructors must "
    "evaluate when synchronous or asynchronous formats best fit the learning "
    "objectives. Research suggests that blended approaches outperform pure "
    "face-to-face delivery when feedback loops are short and authentic "
    "tasks are included."
)
_CHAPTER_A_PROSE_B = (
    "Faculty development programs can bridge the gap between content expertise "
    "and pedagogical fluency. Case studies from several institutions show "
    "consistent improvement in course completion rates when faculty received "
    "structured support in course design practice over multiple semesters."
)

_CHAPTER_B_PROSE_A = (
    "Conceptual knowledge is built through comparison, categorization, and "
    "the construction of mental models. Concept maps, paired comparisons, "
    "and worked examples all give learners explicit structure to hang new "
    "ideas on. Without this scaffolding, factual recall tends to fade quickly."
)
_CHAPTER_B_PROSE_B = (
    "Procedural knowledge develops through deliberate practice. Students need "
    "repeated opportunities to perform a skill in authentic contexts with "
    "immediate feedback. Fluency emerges when procedures become automatic, "
    "freeing working memory for higher-order reasoning tasks."
)


def _build_dart_html() -> str:
    """Return a two-chapter synthetic DART-shaped HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en"><body>
<article id="chA" role="doc-chapter">
  <section data-dart-block-id="sa_c0">
    <h2>Instructional Design in the Digital Classroom</h2>
    <p>{_CHAPTER_A_PROSE_A}</p>
    <p>{_CHAPTER_A_PROSE_B}</p>
  </section>
</article>
<article id="chB" role="doc-chapter">
  <section data-dart-block-id="sb_c0">
    <h2>Types of Knowledge and Pedagogical Choice</h2>
    <p>{_CHAPTER_B_PROSE_A}</p>
    <p>{_CHAPTER_B_PROSE_B}</p>
  </section>
</article>
</body></html>
"""


@pytest.fixture
def dart_html_path():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "synthetic_source.html"
        p.write_text(_build_dart_html(), encoding="utf-8")
        yield p


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


class TestParseDartHtmlFiles:
    def test_parses_both_chapter_topics(self, dart_html_path: Path):
        topics = _cgh.parse_dart_html_files([dart_html_path])
        # Two <article role="doc-chapter"> wrappers → two topics.
        assert len(topics) == 2
        headings = {t["heading"] for t in topics}
        assert "Instructional Design in the Digital Classroom" in headings
        assert "Types of Knowledge and Pedagogical Choice" in headings

    def test_topics_carry_chapter_ids_for_week_binding(
        self, dart_html_path: Path,
    ):
        topics = _cgh.parse_dart_html_files([dart_html_path])
        chapter_ids = {t["chapter_id"] for t in topics}
        # Both topics must carry chapter_id so _group_topics_by_week can
        # respect chapter boundaries when distributing across weeks.
        assert chapter_ids == {"chA", "chB"}

    def test_topics_include_real_paragraph_prose(self, dart_html_path: Path):
        topics = _cgh.parse_dart_html_files([dart_html_path])
        all_paragraphs = [p for t in topics for p in t["paragraphs"]]
        # At least one paragraph must contain a distinctive fragment from
        # each synthetic chapter's prose.
        assert any("instructional design" in p.lower() for p in all_paragraphs)
        assert any("procedural knowledge" in p.lower() for p in all_paragraphs)


class TestContentGenerationWeekProse:
    def test_week_content_module_heading_is_real_topic_heading(
        self, dart_html_path: Path,
    ):
        topics = _cgh.parse_dart_html_files([dart_html_path])
        by_week = _cgh._group_topics_by_week(topics, duration_weeks=4)
        # Week 1 should contain one of the parsed topics.
        assert by_week[0], "Week 1 must receive at least one DART topic"
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=4,
            week_topics=by_week[0],
            week_objectives=[],
            all_objectives=[],
            course_code="SYNTH_101",
        )
        # build_week_data's content_modules must carry the topic heading as
        # its title (not "Week 1 Concepts" or a template skeleton).
        titles = [m["title"] for m in wd["content_modules"]]
        assert any(
            t == "Instructional Design in the Digital Classroom"
            or t == "Types of Knowledge and Pedagogical Choice"
            for t in titles
        )

    def test_week_content_sections_contain_real_paragraph_prose(
        self, dart_html_path: Path,
    ):
        topics = _cgh.parse_dart_html_files([dart_html_path])
        by_week = _cgh._group_topics_by_week(topics, duration_weeks=4)
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=4,
            week_topics=by_week[0],
            week_objectives=[],
            all_objectives=[],
            course_code="SYNTH_101",
        )
        # At least one content section must include paragraph text from
        # the synthetic DART source, not a template placeholder.
        all_paragraphs: list = []
        for m in wd["content_modules"]:
            for s in m["sections"]:
                all_paragraphs.extend(s.get("paragraphs") or [])
        joined = " ".join(all_paragraphs).lower()
        # Either chapter's distinctive phrasing must be present.
        assert ("instructional design" in joined
                or "procedural knowledge" in joined), (
            f"Expected real DART prose in content sections; got: {joined!r}"
        )


class TestContentBodyNotSkeletonOnly:
    def test_content_page_not_just_two_headings(self, dart_html_path: Path):
        """The bug that motivated Wave 28: content pages rendered as just
        <h1>...</h1><h2>...</h2> with no body. Locking in that build_week_data
        always contributes at least one non-empty paragraph when a topic is
        bound to the week.
        """
        topics = _cgh.parse_dart_html_files([dart_html_path])
        by_week = _cgh._group_topics_by_week(topics, duration_weeks=4)
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=4,
            week_topics=by_week[0],
            week_objectives=[],
            all_objectives=[],
            course_code="SYNTH_101",
        )
        non_empty_paragraph_count = sum(
            1
            for m in wd["content_modules"]
            for s in m["sections"]
            for p in (s.get("paragraphs") or [])
            if p and len(re.sub(r"\s+", " ", str(p)).strip()) > 30
        )
        assert non_empty_paragraph_count >= 1, (
            "Expected at least one non-empty paragraph in content modules "
            "when a DART topic is bound to the week; got a skeleton only."
        )
