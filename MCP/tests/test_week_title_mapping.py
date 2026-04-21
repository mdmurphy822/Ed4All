"""Wave 28: week-title mapping end-to-end through build_week_data +
generate_week.

Verifies:
  * Weeks bound to a real topic emit ``"Week {N} Overview: {Chapter Title}"``
    as the page H1 — not ``"Week {N} Concepts"``.
  * Weeks with NO bound topic emit a neutral ``"Week {N} Overview: Overview"``
    (or simpler) — never the tautological ``"Week {N} Overview: Week {N}
    Concepts"`` observed on pre-Wave-28 runs.
  * The IMSCC packager's manifest helper threads the real chapter title
    into the week item label.
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
from Courseforge.scripts import generate_course as _gen  # noqa: E402


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _topic(heading: str, paragraph: str, chapter_id: str) -> dict:
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


def _extract_h1(html: str) -> str:
    m = re.search(r"<h1[^>]*>\s*(.*?)\s*</h1>", html, re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else "")


# ---------------------------------------------------------------------- #
# Tests — build_week_data + generate_week H1 threading
# ---------------------------------------------------------------------- #


class TestWeekTitleReflectsChapter:
    def test_bound_topic_week_uses_real_chapter_title(self, tmp_path: Path):
        """When a week has a bound topic, the rendered overview H1 must
        include the topic's heading — not the placeholder 'Week N Concepts'."""
        topic = _topic(
            heading="Theories of Conceptual Change",
            paragraph=(
                "Learners revise their mental models when a new concept "
                "clashes with an existing one, provided they recognize the "
                "contradiction and can rebuild the model around the new idea."
            ),
            chapter_id="ch1",
        )
        wd = _cgh.build_week_data(
            week_num=1,
            duration_weeks=3,
            week_topics=[topic],
            week_objectives=[],
            all_objectives=[],
            course_code="SYNTH_101",
        )
        assert wd["title"] == "Theories of Conceptual Change"
        _gen.generate_week(wd, tmp_path, "SYNTH_101")
        overview = (tmp_path / "week_01" / "week_01_overview.html").read_text(
            encoding="utf-8"
        )
        h1 = _extract_h1(overview)
        assert "Theories of Conceptual Change" in h1
        # Must NOT contain the tautological "Week 1 Concepts".
        assert "Week 1 Concepts" not in h1

    def test_empty_week_avoids_tautology(self, tmp_path: Path):
        """When no topic is bound, the rendered H1 must NOT be
        'Week N Overview: Week N Concepts'."""
        wd = _cgh.build_week_data(
            week_num=3,
            duration_weeks=6,
            week_topics=[],
            week_objectives=[],
            all_objectives=[],
            course_code="SYNTH_101",
        )
        _gen.generate_week(wd, tmp_path, "SYNTH_101")
        overview = (tmp_path / "week_03" / "week_03_overview.html").read_text(
            encoding="utf-8"
        )
        h1 = _extract_h1(overview)
        assert "Week 3 Concepts" not in h1, (
            f"Tautological 'Week 3 Concepts' leaked into H1: {h1!r}"
        )
        # The neutral fallback title is 'Overview', so the H1 becomes
        # 'Week 3 Overview: Overview'. We accept either that or a plain
        # 'Week 3 Overview' — both avoid the tautology.
        assert "Week 3" in h1


class TestChapterToWeekDistribution:
    def test_equal_counts_one_chapter_per_week(self):
        topics = [
            _topic("Chapter A Title", "A paragraph about topic A.", "ch1"),
            _topic("Chapter B Title", "A paragraph about topic B.", "ch2"),
            _topic("Chapter C Title", "A paragraph about topic C.", "ch3"),
        ]
        by_week = _cgh._group_topics_by_week(topics, duration_weeks=3)
        # Each week should receive exactly one chapter in order.
        assert len(by_week) == 3
        assert by_week[0][0]["heading"] == "Chapter A Title"
        assert by_week[1][0]["heading"] == "Chapter B Title"
        assert by_week[2][0]["heading"] == "Chapter C Title"

    def test_more_weeks_than_chapters_leaves_later_weeks_empty(self):
        """Current contract: later weeks simply receive no topic. The
        fallback title must still be neutral (verified elsewhere)."""
        topics = [
            _topic("Only Chapter", "Paragraph.", "ch1"),
        ]
        by_week = _cgh._group_topics_by_week(topics, duration_weeks=3)
        assert len(by_week) == 3
        assert by_week[0] and by_week[0][0]["heading"] == "Only Chapter"
        assert by_week[1] == []
        assert by_week[2] == []

    def test_more_chapters_than_weeks_does_not_lose_topics(self):
        topics = [
            _topic(f"Chapter {i}", "Paragraph prose.", f"ch{i}")
            for i in range(1, 6)  # 5 chapters
        ]
        by_week = _cgh._group_topics_by_week(topics, duration_weeks=3)
        assert len(by_week) == 3
        # No topic must be dropped on the floor.
        total = sum(len(bucket) for bucket in by_week)
        assert total == 5


class TestPackagerManifestWeekTitle:
    """The IMSCC manifest packager derives the week item title from the
    overview H1. Real chapter title in → "Week N: {title}" out; bare
    "Overview" or missing overview → "Week N" fallback.
    """

    def test_extracts_chapter_title_from_overview_h1(self, tmp_path: Path):
        import sys as _sys
        _sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[2]
                / "Courseforge" / "scripts"),
        )
        import package_multifile_imscc as pkg

        wdir = tmp_path / "week_01"
        wdir.mkdir()
        (wdir / "week_01_overview.html").write_text(
            "<html><body><h1>Week 1 Overview: Formative Assessment</h1></body></html>",
            encoding="utf-8",
        )
        assert pkg._extract_week_title(wdir, 1) == (
            "Week 1: Formative Assessment"
        )

    def test_bare_overview_fallback(self, tmp_path: Path):
        import sys as _sys
        _sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[2]
                / "Courseforge" / "scripts"),
        )
        import package_multifile_imscc as pkg

        wdir = tmp_path / "week_02"
        wdir.mkdir()
        (wdir / "week_02_overview.html").write_text(
            "<html><body><h1>Week 2 Overview: Overview</h1></body></html>",
            encoding="utf-8",
        )
        # "Overview" bare title → neutral "Week 2" fallback, no tautology.
        assert pkg._extract_week_title(wdir, 2) == "Week 2"

    def test_missing_overview_fallback(self, tmp_path: Path):
        import sys as _sys
        _sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[2]
                / "Courseforge" / "scripts"),
        )
        import package_multifile_imscc as pkg

        wdir = tmp_path / "week_03"
        wdir.mkdir()
        # No overview HTML file — helper must not raise.
        assert pkg._extract_week_title(wdir, 3) == "Week 3"
