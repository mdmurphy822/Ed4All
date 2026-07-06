"""Leading-text apparatus retype (FIX 2 + FIX 3) — apparatus-typing fidelity.

The apparatus-heading demotion moved per-section exercise banners out of the
heading stream into leading prose; the chunker's heading-driven ``chunk_type``
therefore stopped typing real problem sets ``"exercise"``. These deterministic,
corpus-agnostic passes restore fidelity from a chunk's OWN leading window.

Synthetic fixtures only — no course slugs / publisher names / data paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from Trainforge.chunker.chunker import (
    ChunkerContext,
    chunk_content,
    leads_with_exercise_banner,
    leads_with_lo_boilerplate,
    retype_from_leading_text,
)


@dataclass
class _Section:
    heading: str
    content: str
    word_count: int
    level: int = 3


def _item(sections):
    return {
        "module_id": "m1",
        "item_id": "i1",
        "item_path": "week_01/i1.html",
        "title": "Page",
        "resource_type": "page",
        "raw_html": "<html></html>",
        "sections": sections,
        "misconceptions": [],
    }


def _capturing_ctx(captured):
    def _create_chunk(*, chunk_id, text, html, item, section_heading,
                      chunk_type, **_ignored):
        captured.append({"id": chunk_id, "type": chunk_type, "text": text})
        return {"id": chunk_id, "text": text, "chunk_type": chunk_type}

    return ChunkerContext(create_chunk=_create_chunk)


@pytest.mark.parametrize(
    "text",
    [
        "2.7 Exercises\n1. Simplify the expression below.",       # ordinal + bare
        "2.7 Section Exercises\n1. Simplify.",                    # ordinal + phrase
        "2.7 EXERCISES Practice Makes Perfect\nⓐ solve for x.",   # OCR-fused banner
        "In the following exercises, simplify each expression.",  # phrase banner
        "Practice Makes Perfect\nWork each problem in order.",    # phrase banner
        "Review Exercises\nComplete the following.",              # phrase banner
    ],
)
def test_banner_leading_types_exercise(text):
    # FIX 2: whatever the heading-driven type, a banner-leading chunk is exercise.
    assert leads_with_exercise_banner(text)
    assert retype_from_leading_text(text, "example") == "exercise"
    assert retype_from_leading_text(text, "explanation") == "exercise"


def test_banner_leading_within_window_but_not_char_zero():
    # A short pedagogical lead-in ahead of the banner (still inside the ~200-char
    # / first-2-lines window) is caught.
    text = "Example 2.4 wrapped up.\n2.7 Exercises\n1. Add the integers."
    assert leads_with_exercise_banner(text)
    assert retype_from_leading_text(text, "example") == "exercise"


def test_lo_boilerplate_leading_vetoes_exercise():
    # FIX 3: a section-intro boilerplate chunk mis-typed exercise (its heading was
    # an exercise-ish banner) is demoted to overview — never an exercise.
    text = "By the end of this section, you will be able to add and subtract."
    assert leads_with_lo_boilerplate(text)
    assert not leads_with_exercise_banner(text)
    assert retype_from_leading_text(text, "exercise") == "overview"


def test_lo_boilerplate_leading_leaves_non_exercise_untouched():
    # The veto only fires against an exercise type; other types are unchanged.
    text = "By the end of this section, you will be able to add and subtract."
    assert retype_from_leading_text(text, "explanation") == "explanation"
    assert retype_from_leading_text(text, "overview") == "overview"


def test_plain_prose_unchanged():
    # No banner, no boilerplate -> byte-identical typing (deterministic no-op).
    text = "The distributive property lets us multiply a sum term by term."
    assert not leads_with_exercise_banner(text)
    assert not leads_with_lo_boilerplate(text)
    for ct in ("example", "explanation", "overview", "summary", "assessment_item"):
        assert retype_from_leading_text(text, ct) == ct


def test_real_numbered_title_not_retyped():
    # A genuine "N.M Exercises in <topic>" section title (lowercase continuation)
    # is NOT a drill banner and must not be forced to exercise.
    text = "3.4 Exercises in Measure Theory motivate the constructions ahead."
    assert not leads_with_exercise_banner(text)
    assert retype_from_leading_text(text, "explanation") == "explanation"


def test_composition_banner_wins_over_veto():
    # "In the following exercises" is BOTH an exercise banner AND lexicon
    # LO-boilerplate; the banner rule wins so it types exercise, not overview.
    text = "In the following exercises, determine whether each equation is true."
    assert leads_with_exercise_banner(text)
    assert leads_with_lo_boilerplate(text)  # shared lexicon lists it under both
    assert retype_from_leading_text(text, "example") == "exercise"


def test_empty_text_is_noop():
    assert retype_from_leading_text("", "explanation") == "explanation"
    assert not leads_with_exercise_banner("")
    assert not leads_with_lo_boilerplate("")


# ---------------------------------------------------------------------------
# Integration — the retype flows through chunk_content -> create_chunk. A
# demoted "N.M Exercises" banner heading has become leading prose whose section
# now inherits an example-ish heading; the leading-text pass restores exercise.
# ---------------------------------------------------------------------------


def test_chunk_content_restores_exercise_type_from_leading_banner():
    captured: List[Dict[str, Any]] = []
    # Heading is a benign section title (post-demotion the exercise banner is no
    # longer the heading); the exercise banner leads the section BODY prose.
    sections = [
        _Section(
            heading="Add and Subtract Integers",
            content=(
                "2.7 Exercises Practice Makes Perfect. "
                "In the following exercises simplify each expression fully and "
                "show every step of the work you performed to reach the answer."
            ),
            word_count=28,
        ),
    ]
    chunk_content([_item(sections)], "TEST_101", ctx=_capturing_ctx(captured))
    assert captured
    assert captured[0]["type"] == "exercise"


def test_chunk_content_lo_boilerplate_intro_not_exercise():
    captured: List[Dict[str, Any]] = []
    # A section-intro whose heading reads exercise-ish ("... Exercises Review")
    # but whose body leads with LO boilerplate must not type exercise.
    sections = [
        _Section(
            heading="Exercises Review",  # would key 'exercise' off the heading
            content=(
                "By the end of this section, you will be able to add and "
                "subtract integers, multiply and divide integers, and simplify "
                "expressions using the order of operations correctly."
            ),
            word_count=30,
        ),
    ]
    chunk_content([_item(sections)], "TEST_101", ctx=_capturing_ctx(captured))
    assert captured
    assert captured[0]["type"] != "exercise"
    assert captured[0]["type"] == "overview"
