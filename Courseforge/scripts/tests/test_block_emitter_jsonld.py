"""Phase 2 Subtask 8 — Block.to_jsonld_entry() regression suite.

Per-block-type assertions on the JSON-LD entry shape Block emits, matched
against the existing ``_build_*_metadata`` helpers in ``generate_course.py``:

    objective       -> _build_objectives_metadata :1364-1420
    section blocks  -> _build_sections_metadata :1467-1490
    misconception   -> _build_misconceptions_metadata :1571-1578

Other Phase-2-introduced block types (flip_card_grid, self_check_question,
activity, chrome, prereq_set, summary_takeaway, reflection_prompt,
discussion_prompt, recap, assessment_item) emit a minimal Phase-2 entry
shape (``blockId`` / ``blockType`` / ``sequence`` plus optional
``touchedBy`` / ``contentHash``) consumed by the new top-level
``blocks[]`` array.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block, Touch  # noqa: E402


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def test_objective_jsonld_basic_camelcase_keys() -> None:
    b = Block(
        block_id="x",
        block_type="objective",
        page_id="p",
        sequence=0,
        content="Define X",
        objective_ids=("TO-01",),
        bloom_level="remember",
        bloom_verb="define",
        cognitive_domain="factual",
    )
    e = b.to_jsonld_entry()
    assert e == {
        "id": "TO-01",
        "statement": "Define X",
        "bloomLevel": "remember",
        "bloomVerb": "define",
        "cognitiveDomain": "factual",
    }


def test_objective_jsonld_emits_multi_bloom_arrays() -> None:
    b = Block(
        block_id="x",
        block_type="objective",
        page_id="p",
        sequence=0,
        content="Analyze and evaluate Y",
        objective_ids=("TO-02",),
        bloom_level="analyze",
        bloom_verb="analyze",
        cognitive_domain="conceptual",
        bloom_levels=("analyze", "evaluate"),
        bloom_verbs=("analyze", "evaluate"),
    )
    e = b.to_jsonld_entry()
    assert e["bloomLevels"] == ["analyze", "evaluate"]
    assert e["bloomVerbs"] == ["analyze", "evaluate"]


def test_objective_jsonld_emits_targeted_concepts_when_terms_and_bloom() -> None:
    b = Block(
        block_id="x",
        block_type="objective",
        page_id="p",
        sequence=0,
        content="Define X",
        objective_ids=("TO-01",),
        bloom_level="remember",
        key_terms=("alpha", "beta"),
    )
    e = b.to_jsonld_entry()
    assert e["keyConcepts"] == ["alpha", "beta"]
    assert e["targetedConcepts"] == [
        {"concept": "alpha", "bloomLevel": "remember"},
        {"concept": "beta", "bloomLevel": "remember"},
    ]


def test_objective_jsonld_omits_targeted_concepts_when_no_bloom() -> None:
    b = Block(
        block_id="x",
        block_type="objective",
        page_id="p",
        sequence=0,
        content="Identify Y",
        objective_ids=("TO-03",),
        key_terms=("alpha",),
    )
    e = b.to_jsonld_entry()
    assert e["keyConcepts"] == ["alpha"]
    assert "targetedConcepts" not in e


# ---------------------------------------------------------------------------
# Section blocks (explanation, example, concept, summary_takeaway)
# ---------------------------------------------------------------------------


def test_explanation_jsonld_section_shape() -> None:
    b = Block(
        block_id="x",
        block_type="explanation",
        page_id="p",
        sequence=0,
        content="Section heading",
        content_type_label="explanation",
        key_terms=("term_one",),
        teaching_role="reinforce",
        bloom_range="understand-apply",
    )
    e = b.to_jsonld_entry()
    assert e["heading"] == "Section heading"
    assert e["contentType"] == "explanation"
    assert e["keyTerms"] == ["term_one"]
    assert e["teachingRole"] == ["reinforce"]
    assert e["bloomRange"] == ["understand-apply"]


def test_section_jsonld_omits_optional_keys_when_unset() -> None:
    b = Block(
        block_id="x",
        block_type="example",
        page_id="p",
        sequence=0,
        content="Example heading",
        content_type_label="example",
    )
    e = b.to_jsonld_entry()
    assert e == {"heading": "Example heading", "contentType": "example"}


def test_section_jsonld_emits_source_references_when_present() -> None:
    refs = ({"sourceId": "dart:ch1#b1", "role": "primary"},)
    b = Block(
        block_id="x",
        block_type="explanation",
        page_id="p",
        sequence=0,
        content="Heading",
        content_type_label="explanation",
        source_references=refs,
    )
    e = b.to_jsonld_entry()
    assert e["sourceReferences"] == [{"sourceId": "dart:ch1#b1", "role": "primary"}]


def test_concept_jsonld_section_shape() -> None:
    b = Block(
        block_id="x",
        block_type="concept",
        page_id="p",
        sequence=0,
        content="Key concept",
        content_type_label="overview",
    )
    e = b.to_jsonld_entry()
    assert e["heading"] == "Key concept"
    assert e["contentType"] == "overview"


def test_summary_takeaway_jsonld_section_shape() -> None:
    b = Block(
        block_id="x",
        block_type="summary_takeaway",
        page_id="p",
        sequence=0,
        content="Recap heading",
        content_type_label="summary",
    )
    e = b.to_jsonld_entry()
    assert e["heading"] == "Recap heading"
    assert e["contentType"] == "summary"


# ---------------------------------------------------------------------------
# Misconception
# ---------------------------------------------------------------------------


def test_misconception_jsonld_shape() -> None:
    b = Block(
        block_id="x",
        block_type="misconception",
        page_id="p",
        sequence=0,
        content={
            "misconception": "Belief X is true.",
            "correction": "Actually X is false.",
        },
        bloom_level="understand",
        cognitive_domain="conceptual",
    )
    e = b.to_jsonld_entry()
    assert e == {
        "misconception": "Belief X is true.",
        "correction": "Actually X is false.",
        "bloomLevel": "understand",
        "cognitiveDomain": "conceptual",
    }


def test_misconception_jsonld_elides_bloom_when_unknown() -> None:
    b = Block(
        block_id="x",
        block_type="misconception",
        page_id="p",
        sequence=0,
        content={"misconception": "wrong", "correction": "right"},
    )
    e = b.to_jsonld_entry()
    assert e == {"misconception": "wrong", "correction": "right"}


# ---------------------------------------------------------------------------
# Phase-2 minimal block_type entries
# ---------------------------------------------------------------------------


def test_flip_card_grid_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="page_01#flip_card_grid_term_one_0",
        block_type="flip_card_grid",
        page_id="page_01",
        sequence=2,
        content={"terms": []},
        key_terms=("term_one",),
    )
    e = b.to_jsonld_entry()
    assert e == {
        "blockId": "page_01#flip_card_grid_term_one_0",
        "blockType": "flip_card_grid",
        "sequence": 2,
    }


def test_self_check_question_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="page_01#self_check_question_q1_0",
        block_type="self_check_question",
        page_id="page_01",
        sequence=3,
        content={"question": "?", "options": []},
    )
    e = b.to_jsonld_entry()
    assert e == {
        "blockId": "page_01#self_check_question_q1_0",
        "blockType": "self_check_question",
        "sequence": 3,
    }


def test_activity_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="page_01#activity_practice_0",
        block_type="activity",
        page_id="page_01",
        sequence=4,
        content={"title": "T", "description": "D"},
    )
    e = b.to_jsonld_entry()
    assert e["blockType"] == "activity"
    assert e["sequence"] == 4
    assert e["blockId"] == "page_01#activity_practice_0"


def test_chrome_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="page_01#chrome_skip_link_0",
        block_type="chrome",
        page_id="page_01",
        sequence=0,
        content="Skip to main content",
    )
    e = b.to_jsonld_entry()
    assert e["blockType"] == "chrome"
    assert "blockId" in e


def test_prereq_set_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="page_01#prereq_set_0",
        block_type="prereq_set",
        page_id="page_01",
        sequence=0,
        content={"prerequisitePages": ["page_00"]},
    )
    e = b.to_jsonld_entry()
    assert e["blockType"] == "prereq_set"


def test_reflection_prompt_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="x",
        block_type="reflection_prompt",
        page_id="p",
        sequence=0,
        content="Reflect on Y",
    )
    e = b.to_jsonld_entry()
    assert e == {"blockId": "x", "blockType": "reflection_prompt", "sequence": 0}


def test_discussion_prompt_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="x",
        block_type="discussion_prompt",
        page_id="p",
        sequence=1,
        content="Discuss Z",
    )
    e = b.to_jsonld_entry()
    assert e == {"blockId": "x", "blockType": "discussion_prompt", "sequence": 1}


def test_recap_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="x",
        block_type="recap",
        page_id="p",
        sequence=0,
        content="Chapter recap text",
    )
    e = b.to_jsonld_entry()
    assert e == {"blockId": "x", "blockType": "recap", "sequence": 0}


def test_assessment_item_jsonld_minimal_entry() -> None:
    b = Block(
        block_id="x",
        block_type="assessment_item",
        page_id="p",
        sequence=0,
        content={"question": "?"},
    )
    e = b.to_jsonld_entry()
    assert e == {"blockId": "x", "blockType": "assessment_item", "sequence": 0}


# ---------------------------------------------------------------------------
# Touched-by chain + content-hash projection
# ---------------------------------------------------------------------------


def test_minimal_jsonld_emits_touched_by_when_chain_present() -> None:
    touch = Touch(
        model="claude-sonnet-4",
        provider="local",
        tier="outline",
        timestamp="2026-05-02T00:00:00Z",
        decision_capture_id="decisions:0",
        purpose="draft",
    )
    b = Block(
        block_id="x",
        block_type="recap",
        page_id="p",
        sequence=0,
        content="Recap",
        touched_by=(touch,),
    )
    e = b.to_jsonld_entry()
    assert "touchedBy" in e
    assert e["touchedBy"] == [
        {
            "model": "claude-sonnet-4",
            "provider": "local",
            "tier": "outline",
            "timestamp": "2026-05-02T00:00:00Z",
            "decisionCaptureId": "decisions:0",
            "purpose": "draft",
        }
    ]


def test_minimal_jsonld_emits_content_hash_when_set() -> None:
    b = Block(
        block_id="x",
        block_type="recap",
        page_id="p",
        sequence=0,
        content="Recap text",
        content_hash="a" * 64,
    )
    e = b.to_jsonld_entry()
    assert e["contentHash"] == "a" * 64


def test_render_touched_by_returns_camelcase_dicts() -> None:
    touch_a = Touch(
        model="m1",
        provider="local",
        tier="outline",
        timestamp="2026-05-02T00:00:00Z",
        decision_capture_id="cap:0",
        purpose="draft",
    )
    touch_b = Touch(
        model="m2",
        provider="anthropic",
        tier="validation",
        timestamp="2026-05-02T01:00:00Z",
        decision_capture_id="cap:1",
        purpose="validate",
    )
    b = Block(
        block_id="x",
        block_type="recap",
        page_id="p",
        sequence=0,
        content="Recap",
        touched_by=(touch_a, touch_b),
    )
    rendered = b._render_touched_by()
    assert len(rendered) == 2
    assert rendered[0]["decisionCaptureId"] == "cap:0"
    assert rendered[1]["tier"] == "validation"
