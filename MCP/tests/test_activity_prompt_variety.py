"""Wave 28: verify per-week activity prompts vary in body, not just week number.

Pre-Wave-28 each week's application page carried the exact same prompt:
    "Respond in your own words (150 words) or with a diagram
    demonstrating the concept from the week's material."

With week-specific topic content and objective statements now available,
the prompt must reference the week's own material. These tests pin the
contract: the prompt body references (a) the week's objective statement
OR (b) the week's key terms — and varies across weeks with distinct
topics.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


def _topic(
    heading: str,
    paragraphs,
    key_terms,
    chapter_id: str = "ch1",
) -> dict:
    return {
        "heading": heading,
        "paragraphs": list(paragraphs),
        "key_terms": list(key_terms),
        "source_file": "synth",
        "word_count": sum(len(p.split()) for p in paragraphs),
        "chapter_id": chapter_id,
        "dart_block_ids": [],
        "extracted_lo_statements": [],
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }


def _activity_for(week_num: int, topic: dict, obj: dict) -> dict:
    wd = _cgh.build_week_data(
        week_num=week_num,
        duration_weeks=2,
        week_topics=[topic],
        week_objectives=[obj],
        all_objectives=[obj],
        course_code="SYNTH_101",
    )
    return wd["activities"][0]


def test_prompt_references_week_specific_key_terms():
    topic = _topic(
        "Assessment Design",
        ["Paragraph about formative and summative assessment planning."],
        key_terms=["Formative Assessment", "Rubric"],
    )
    obj = {
        "id": "TO-01",
        "statement": "Design a formative assessment using a rubric.",
        "bloom_level": "create",
        "bloom_verb": "design",
    }
    act = _activity_for(1, topic, obj)
    desc = act["description"]
    # Must reference at least one of the week's actual key terms.
    assert "Formative Assessment" in desc or "Rubric" in desc, (
        f"Activity description missed both week-specific terms: {desc!r}"
    )


def test_prompt_references_objective_statement():
    topic = _topic(
        "Rubric Construction",
        ["Paragraph about criteria-based rubric construction."],
        key_terms=[],
    )
    obj = {
        "id": "TO-01",
        "statement": "Write rubric criteria for a discussion assignment.",
        "bloom_level": "apply",
        "bloom_verb": "write",
    }
    act = _activity_for(1, topic, obj)
    # The objective statement (or a distinctive substring) must appear.
    assert "rubric criteria" in act["description"]


def test_prompt_does_not_carry_generic_tautology():
    """The phrase 'demonstrating the concept from the week's material'
    used to appear in every prompt body — regardless of week content.
    Lock it out going forward."""
    topic = _topic(
        "Course Pacing",
        ["Paragraph about term length and student workload distribution."],
        key_terms=["Pacing"],
    )
    obj = {
        "id": "TO-01",
        "statement": "Plan pacing for a 12-week graduate seminar.",
        "bloom_level": "create",
        "bloom_verb": "plan",
    }
    act = _activity_for(1, topic, obj)
    assert (
        "demonstrating the concept from the week's material"
        not in act["description"]
    )


def test_prompts_differ_across_weeks_with_distinct_topics():
    topic_a = _topic(
        "Accessibility Standards",
        ["Paragraph about WCAG criteria and captioning requirements."],
        key_terms=["WCAG", "Captioning"],
    )
    topic_b = _topic(
        "Discussion Forum Norms",
        ["Paragraph about asynchronous discussion etiquette and moderation."],
        key_terms=["Moderation", "Etiquette"],
    )
    obj_a = {
        "id": "TO-01",
        "statement": "Apply WCAG standards to a course page.",
        "bloom_level": "apply",
        "bloom_verb": "apply",
    }
    obj_b = {
        "id": "TO-02",
        "statement": "Compare discussion forum moderation strategies.",
        "bloom_level": "analyze",
        "bloom_verb": "compare",
    }
    act_a = _activity_for(1, topic_a, obj_a)
    act_b = _activity_for(2, topic_b, obj_b)
    # Different body text — week-specific terms must differentiate them.
    assert act_a["description"] != act_b["description"], (
        "Expected per-week variation in activity description body"
    )
