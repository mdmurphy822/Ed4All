"""Wave 25 Fix 4: activity-prompt false-positive filter for CHAPTER_OPENER.

Audit evidence: on a textbook with many reflective activities, the
permissive chapter-heading regex promoted dozens of activity-prompt
phrases ("What are your reasons?", "Determine which is a medium...",
"Do you find the distinction helpful?") to
``<article role="doc-chapter">`` — tripling the emitted chapter count
relative to the book's real chapter count.

The guard rejects blocks whose text opens with an interrogative /
directive starter UNLESS a strong chapter-number pattern (``"Chapter
5"``, ``"Part II"``, ``"5 Introduction"``) wins first.
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, RawBlock
from DART.converter.heuristic_classifier import (
    HeuristicClassifier,
    _looks_like_activity_prompt,
)


def _classify(text: str):
    block = RawBlock(
        text=text, block_id="b1", page=1, extractor="pdftotext"
    )
    clf = HeuristicClassifier()
    return clf.classify_sync([block])[0]


@pytest.mark.unit
@pytest.mark.dart
class TestActivityPromptStartersHelper:
    def test_what_starter_flagged(self):
        assert _looks_like_activity_prompt("What are your reasons?")

    def test_do_you_starter_flagged(self):
        assert _looks_like_activity_prompt(
            "Do you find the distinction between media and technology helpful?"
        )

    def test_determine_starter_flagged(self):
        assert _looks_like_activity_prompt(
            "Determine which is a medium and which a technology"
        )

    def test_consider_starter_flagged(self):
        assert _looks_like_activity_prompt("Consider the following scenario")

    def test_chapter_prefix_wins(self):
        # "Chapter N" beats the activity guard.
        assert not _looks_like_activity_prompt("Chapter 1: What are the basics?")

    def test_bare_number_prefix_wins(self):
        # "5 Introduction to digital pedagogy"
        assert not _looks_like_activity_prompt(
            "5 Introduction to digital pedagogy"
        )


@pytest.mark.unit
@pytest.mark.dart
class TestChapterPromotionGuards:
    def test_real_chapter_still_classified(self):
        result = _classify("Chapter 1: Fundamentals of Digital Pedagogy")
        assert result.role == BlockRole.CHAPTER_OPENER

    def test_what_prompt_not_chapter(self):
        result = _classify("What are your reasons?")
        assert result.role != BlockRole.CHAPTER_OPENER

    def test_determine_prompt_not_chapter(self):
        # Text starts with capital D + no number → falls under the
        # permissive chapter regex pre-guard; must be rejected.
        # Note: the current _CHAPTER_HEADING regex only fires on
        # "Chapter N:" / "N." / roman / etc. prefixes — not "Determine".
        # But the block may still be classified as SUBSECTION_HEADING
        # if it fits short title-case pattern. The invariant is just:
        # it is NOT CHAPTER_OPENER.
        result = _classify(
            "Determine which is a medium and which a technology."
        )
        assert result.role != BlockRole.CHAPTER_OPENER

    def test_consider_scenario_not_chapter(self):
        result = _classify("Consider the following scenario")
        assert result.role != BlockRole.CHAPTER_OPENER

    def test_do_you_prompt_not_chapter(self):
        result = _classify(
            "Do you find the distinction between media and technology helpful?"
        )
        assert result.role != BlockRole.CHAPTER_OPENER

    def test_real_chapter_with_you_in_body_still_chapter(self):
        # Body uses "you" but the chapter prefix wins the strong match.
        result = _classify(
            "Chapter 3: What You Need to Know About Online Learning"
        )
        assert result.role == BlockRole.CHAPTER_OPENER

    def test_bare_number_chapter_matches(self):
        # "5 Introduction to digital pedagogy" via the bare-number
        # path — would also match the chapter regex at "5. " but
        # the lead-bare-number shape has no dot; ensure this form
        # still classifies correctly.
        result = _classify("5. Introduction to Digital Pedagogy")
        # "5. Foo" matches _CHAPTER_HEADING via the "\d{1,2}\. " arm.
        assert result.role == BlockRole.CHAPTER_OPENER

    def test_how_question_rejected(self):
        result = _classify("How might you apply this to your course?")
        assert result.role != BlockRole.CHAPTER_OPENER
