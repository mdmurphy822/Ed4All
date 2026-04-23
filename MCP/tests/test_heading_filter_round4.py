"""Round-4 heading-filter tests.

A real-world corpus run exposed residual leaked headings after rounds
1-3:

  1. Colon-ended prompts:
       "This chapter covers the following topics:"
       "The functional syntax equivalent is as follows:"
  2. Author bylines:
       "Ada-Lee Researcher Ben Otherwriter"
       "Cover design by Author Name"
  3. (Ambiguous, kept per documented decision) formula / notation
     fragments:
       "C v \u2200R.D"
       "FirstYearCourse SubClassOf isTaughtBy only Professor"

This module asserts the negative cases (filter rejects colon-prompts and
bylines) AND the positive cases that must still be preserved (real
chapter titles, formula fragments).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402

# ---------------------------------------------------------------------- #
# Headings that MUST be filtered out.
# ---------------------------------------------------------------------- #


class TestHeadingFilterRejectsRound4Artifacts:
    def test_rejects_colon_prompt_chapter_topics(self):
        heading = "This chapter covers the following topics:"
        assert _cgh._is_low_signal_heading(heading) is True

    def test_rejects_colon_prompt_functional_syntax(self):
        heading = "The functional syntax equivalent is as follows:"
        assert _cgh._is_low_signal_heading(heading) is True

    def test_rejects_author_byline_two_names(self):
        heading = "Ada-Lee Researcher Ben Otherwriter"
        assert _cgh._is_low_signal_heading(heading) is True

    def test_rejects_author_byline_with_leadin(self):
        heading = "Cover design by Author Name"
        assert _cgh._is_low_signal_heading(heading) is True

    def test_rejects_author_byline_edited_by(self):
        heading = "Edited by Robert Hanneman Mark Riddle"
        assert _cgh._is_low_signal_heading(heading) is True


# ---------------------------------------------------------------------- #
# Headings that MUST be kept (real source content).
# ---------------------------------------------------------------------- #


class TestHeadingFilterKeepsLegitimateHeadings:
    def test_keeps_short_real_chapter_title(self):
        heading = "Introduction to Photosynthesis"
        assert _cgh._is_low_signal_heading(heading) is False

    def test_keeps_formula_fragment_camelcase(self):
        heading = "FirstYearCourse SubClassOf isTaughtBy only Professor"
        # Not rejected: CamelCase identifiers + formal keywords look
        # unusual but represent real chapter examples in ontology textbooks.
        assert _cgh._is_low_signal_heading(heading) is False

    def test_keeps_title_case_with_common_nouns(self):
        # 2-3 Title-Case words but one is a common noun (not a name).
        # Must NOT be misclassified as an author byline.
        heading = "European Union Policy"
        assert _cgh._is_low_signal_heading(heading) is False

    def test_keeps_short_colon_title_prefix(self):
        # Short colon title (<= 3 words) — treat as title-prefix, not a
        # prompt. Keeps headings like "Introduction:" if they occur.
        heading = "Introduction:"
        assert _cgh._is_low_signal_heading(heading) is False


# ---------------------------------------------------------------------- #
# Round-3 regression: make sure the new filters didn't re-break old ones.
# ---------------------------------------------------------------------- #


class TestHeadingFilterRegression:
    def test_still_rejects_all_caps_short_chrome(self):
        assert _cgh._is_low_signal_heading("REFERENCES") is True

    def test_still_rejects_city_abbrev(self):
        assert _cgh._is_low_signal_heading("VANCOUVER BC") is True

    def test_still_rejects_blocklist_phrase(self):
        assert _cgh._is_low_signal_heading("Table of Contents") is True

    def test_still_rejects_hyphen_truncated_word(self):
        assert _cgh._is_low_signal_heading(
            "Some headings can have an inconsis-"
        ) is True

    def test_still_keeps_real_chapter_title(self):
        assert _cgh._is_low_signal_heading(
            "The Calvin Cycle and Carbon Fixation"
        ) is False
