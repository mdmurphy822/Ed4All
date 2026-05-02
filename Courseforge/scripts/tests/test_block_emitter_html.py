"""Phase 2 Subtask 8 + 9 — Block.to_html_attrs() regression suite.

Per-block-type assertions on the exact substring the emitter produces for
representative inputs. Cross-checks against the literal string format in
the legacy renderer functions in ``generate_course.py``:

    objective         -> :854-860
    flip_card_grid    -> :887-889
    self_check        -> :929-944
    activity          -> :1126-1140
    content_section   -> :1018-1035
    callout           -> :1071-1073
    chrome            -> :796-797, :804
    wrapper-only      -> _source_attr_string :812-830

Subtask 9 also adds byte-stable snapshot tests asserting the legacy
``_render_*`` output's ``<li ...>`` / ``<div ...>`` / ``<h2 ...>``
substring contains EXACTLY the bytes returned by ``Block.to_html_attrs()``.
Skips when ``COURSEFORGE_EMIT_BLOCKS`` is set (the new
``data-cf-block-id`` attribute would break byte equality).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Per-block-type emit assertions (Subtask 8)
# ---------------------------------------------------------------------------


def test_objective_attrs_full_shape() -> None:
    b = Block(
        block_id="page_01#objective_TO-01_0",
        block_type="objective",
        page_id="page_01",
        sequence=0,
        content="Define X",
        objective_ids=("TO-01",),
        bloom_level="remember",
        bloom_verb="define",
        cognitive_domain="factual",
    )
    s = b.to_html_attrs()
    assert s == (
        ' data-cf-objective-id="TO-01"'
        ' data-cf-bloom-level="remember"'
        ' data-cf-bloom-verb="define"'
        ' data-cf-cognitive-domain="factual"'
    )


def test_objective_attrs_elide_when_no_bloom() -> None:
    b = Block(
        block_id="x",
        block_type="objective",
        page_id="p",
        sequence=0,
        content="Identify Y",
        objective_ids=("CO-02",),
    )
    s = b.to_html_attrs()
    assert ' data-cf-objective-id="CO-02"' in s
    assert "data-cf-bloom-level" not in s
    assert "data-cf-cognitive-domain" not in s


def test_objective_attrs_html_escapes_objective_id() -> None:
    b = Block(
        block_id="x",
        block_type="objective",
        page_id="p",
        sequence=0,
        content="Define ampersand",
        objective_ids=("TO-A&B",),
    )
    assert ' data-cf-objective-id="TO-A&amp;B"' in b.to_html_attrs()


def test_flip_card_grid_attrs_shape() -> None:
    b = Block(
        block_id="page_01#flip_card_grid_term_one_0",
        block_type="flip_card_grid",
        page_id="page_01",
        sequence=0,
        content={"terms": [{"term": "Term One", "definition": "Def"}]},
        key_terms=("term_one",),
        teaching_role="reinforce",
    )
    s = b.to_html_attrs()
    assert s == (
        ' data-cf-component="flip-card"'
        ' data-cf-purpose="term-definition"'
        ' data-cf-teaching-role="reinforce"'
        ' data-cf-term="term_one"'
    )


def test_flip_card_grid_attrs_omits_role_when_unset() -> None:
    b = Block(
        block_id="x",
        block_type="flip_card_grid",
        page_id="p",
        sequence=0,
        content={"terms": []},
        key_terms=("alpha",),
    )
    s = b.to_html_attrs()
    assert "data-cf-teaching-role" not in s
    assert ' data-cf-component="flip-card"' in s
    assert ' data-cf-term="alpha"' in s


def test_self_check_question_attrs_shape() -> None:
    b = Block(
        block_id="page_01#self_check_question_q1_0",
        block_type="self_check_question",
        page_id="page_01",
        sequence=0,
        content={"question": "Q1?", "options": []},
        bloom_level="apply",
        teaching_role="assess",
        objective_ids=("CO-03",),
    )
    s = b.to_html_attrs()
    assert s == (
        ' data-cf-component="self-check"'
        ' data-cf-purpose="formative-assessment"'
        ' data-cf-teaching-role="assess"'
        ' data-cf-bloom-level="apply"'
        ' data-cf-objective-ref="CO-03"'
    )


def test_self_check_question_default_bloom_remember() -> None:
    b = Block(
        block_id="x",
        block_type="self_check_question",
        page_id="p",
        sequence=0,
        content={"question": "?", "options": []},
    )
    assert ' data-cf-bloom-level="remember"' in b.to_html_attrs()


def test_self_check_question_attrs_with_source_ids() -> None:
    b = Block(
        block_id="x",
        block_type="self_check_question",
        page_id="p",
        sequence=0,
        content={"question": "?", "options": []},
        bloom_level="understand",
        source_ids=("dart:ch1#b1", "dart:ch1#b2"),
        source_primary="dart:ch1#b1",
    )
    s = b.to_html_attrs()
    assert ' data-cf-source-ids="dart:ch1#b1,dart:ch1#b2"' in s
    assert ' data-cf-source-primary="dart:ch1#b1"' in s


def test_activity_attrs_shape() -> None:
    b = Block(
        block_id="page_01#activity_practice_0",
        block_type="activity",
        page_id="page_01",
        sequence=0,
        content={"title": "T", "description": "D"},
        bloom_level="apply",
        teaching_role="practice",
        objective_ids=("CO-04",),
    )
    s = b.to_html_attrs()
    assert s == (
        ' data-cf-component="activity"'
        ' data-cf-purpose="practice"'
        ' data-cf-teaching-role="practice"'
        ' data-cf-bloom-level="apply"'
        ' data-cf-objective-ref="CO-04"'
    )


def test_activity_attrs_default_bloom_apply() -> None:
    b = Block(
        block_id="x",
        block_type="activity",
        page_id="p",
        sequence=0,
        content={"title": "T", "description": "D"},
    )
    assert ' data-cf-bloom-level="apply"' in b.to_html_attrs()


def test_explanation_section_attrs_shape() -> None:
    b = Block(
        block_id="page_01#explanation_explain_x_0",
        block_type="explanation",
        page_id="page_01",
        sequence=0,
        content="Explain X",
        content_type_label="explanation",
        key_terms=("term_one", "term_two"),
        bloom_range="understand-apply",
    )
    s = b.to_html_attrs()
    assert s == (
        ' data-cf-content-type="explanation"'
        ' data-cf-key-terms="term_one,term_two"'
        ' data-cf-bloom-range="understand-apply"'
    )


def test_explanation_section_attrs_no_terms_no_range() -> None:
    b = Block(
        block_id="x",
        block_type="explanation",
        page_id="p",
        sequence=0,
        content="text",
        content_type_label="explanation",
    )
    s = b.to_html_attrs()
    assert s == ' data-cf-content-type="explanation"'


def test_section_attrs_with_source_ids() -> None:
    b = Block(
        block_id="x",
        block_type="example",
        page_id="p",
        sequence=0,
        content="Section heading",
        content_type_label="example",
        source_ids=("dart:ch3#b9",),
        source_primary="dart:ch3#b9",
    )
    s = b.to_html_attrs()
    assert ' data-cf-content-type="example"' in s
    assert ' data-cf-source-ids="dart:ch3#b9"' in s
    assert ' data-cf-source-primary="dart:ch3#b9"' in s


def test_callout_attrs_shape() -> None:
    b = Block(
        block_id="x",
        block_type="callout",
        page_id="p",
        sequence=0,
        content={"items": []},
        content_type_label="application-note",
    )
    s = b.to_html_attrs()
    assert s == ' data-cf-content-type="application-note"'


def test_callout_attrs_default_when_label_unset() -> None:
    b = Block(
        block_id="x",
        block_type="callout",
        page_id="p",
        sequence=0,
        content={"items": []},
    )
    assert b.to_html_attrs() == ' data-cf-content-type="note"'


def test_chrome_attrs_shape() -> None:
    b = Block(
        block_id="x",
        block_type="chrome",
        page_id="p",
        sequence=0,
        content="Skip to main content",
    )
    assert b.to_html_attrs() == ' data-cf-role="template-chrome"'


def test_prereq_set_wrapper_only_source_attrs() -> None:
    b = Block(
        block_id="x",
        block_type="prereq_set",
        page_id="p",
        sequence=0,
        content={"prerequisitePages": []},
        source_ids=("dart:ch1#b1",),
        source_primary="dart:ch1#b1",
    )
    s = b.to_html_attrs()
    assert s == ' data-cf-source-ids="dart:ch1#b1" data-cf-source-primary="dart:ch1#b1"'


def test_summary_takeaway_section_attrs() -> None:
    b = Block(
        block_id="x",
        block_type="summary_takeaway",
        page_id="p",
        sequence=0,
        content="Recap text",
        content_type_label="summary",
        source_ids=("dart:ch9#b1",),
    )
    s = b.to_html_attrs()
    assert ' data-cf-content-type="summary"' in s
    assert ' data-cf-source-ids="dart:ch9#b1"' in s


def test_reflection_prompt_wrapper_only() -> None:
    b = Block(
        block_id="x",
        block_type="reflection_prompt",
        page_id="p",
        sequence=0,
        content="Reflect on Y",
        source_ids=("dart:ch2#b3",),
    )
    s = b.to_html_attrs()
    assert s == ' data-cf-source-ids="dart:ch2#b3"'


def test_discussion_prompt_wrapper_only() -> None:
    b = Block(
        block_id="x",
        block_type="discussion_prompt",
        page_id="p",
        sequence=0,
        content="Discuss Z",
        source_ids=(),
    )
    # Empty source_ids -> empty wrapper attr (matches _source_attr_string).
    assert b.to_html_attrs() == ""


def test_recap_wrapper_only_with_sources() -> None:
    b = Block(
        block_id="x",
        block_type="recap",
        page_id="p",
        sequence=0,
        content="Chapter recap text",
        source_ids=("dart:ch5#b9", "dart:ch5#b10"),
        source_primary="dart:ch5#b9",
    )
    assert (
        b.to_html_attrs()
        == ' data-cf-source-ids="dart:ch5#b9,dart:ch5#b10" data-cf-source-primary="dart:ch5#b9"'
    )


def test_misconception_attrs_empty_when_flag_off() -> None:
    b = Block(
        block_id="x",
        block_type="misconception",
        page_id="p",
        sequence=0,
        content={"misconception": "wrong", "correction": "right"},
        bloom_level="understand",
    )
    # Misconceptions emit only via JSON-LD, so default attrs are empty.
    assert b.to_html_attrs() == ""


def test_concept_block_treated_as_section() -> None:
    b = Block(
        block_id="x",
        block_type="concept",
        page_id="p",
        sequence=0,
        content="Key concept heading",
        content_type_label="overview",
    )
    s = b.to_html_attrs()
    assert s == ' data-cf-content-type="overview"'


def test_assessment_item_attrs_empty() -> None:
    # Assessment items live in QTI XML, not HTML — emit empty.
    b = Block(
        block_id="x",
        block_type="assessment_item",
        page_id="p",
        sequence=0,
        content={"question": "?"},
    )
    assert b.to_html_attrs() == ""


# ---------------------------------------------------------------------------
# Phase-2 emit-flag (gates the new ``data-cf-block-id`` attribute)
# ---------------------------------------------------------------------------


def test_to_html_attrs_includes_data_cf_block_id_when_emit_blocks_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``COURSEFORGE_EMIT_BLOCKS`` is truthy, every emit appends
    ``data-cf-block-id="<block_id>"``. Off by default — see Subtask 25."""
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    b = Block(
        block_id="page_01#objective_TO-01_0",
        block_type="objective",
        page_id="page_01",
        sequence=0,
        content="Define X",
        objective_ids=("TO-01",),
        bloom_level="remember",
        bloom_verb="define",
        cognitive_domain="factual",
    )
    s = b.to_html_attrs()
    assert s.endswith(' data-cf-block-id="page_01#objective_TO-01_0"')


def test_to_html_attrs_omits_block_id_when_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    b = Block(
        block_id="page_01#objective_TO-01_0",
        block_type="objective",
        page_id="page_01",
        sequence=0,
        content="Define X",
        objective_ids=("TO-01",),
        bloom_level="remember",
    )
    s = b.to_html_attrs()
    assert "data-cf-block-id" not in s


def test_to_html_attrs_block_id_emit_flag_recognises_truthy_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = Block(
        block_id="b1",
        block_type="chrome",
        page_id="p",
        sequence=0,
        content="text",
    )
    for value in ("1", "true", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", value)
        assert ' data-cf-block-id="b1"' in b.to_html_attrs()
    for value in ("", "0", "false", "off", "no", "bogus"):
        monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", value)
        assert "data-cf-block-id" not in b.to_html_attrs()


# ---------------------------------------------------------------------------
# Subtask 9 — byte-stable snapshot tests against legacy renderers
# ---------------------------------------------------------------------------

# All snapshot tests skip when COURSEFORGE_EMIT_BLOCKS is set since the new
# trailing attribute would break byte equality.

_SKIP_REASON = (
    "COURSEFORGE_EMIT_BLOCKS is set; new data-cf-block-id attribute breaks "
    "byte equality vs. legacy renderer output."
)


def _legacy_emit_blocks_flag_set() -> bool:
    return os.environ.get("COURSEFORGE_EMIT_BLOCKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.skipif(_legacy_emit_blocks_flag_set(), reason=_SKIP_REASON)
def test_block_to_html_attrs_byte_equal_to_legacy_render_objectives() -> None:
    """Byte-stability gate: legacy ``_render_objectives`` output's
    ``<li ...>`` substring contains exactly the bytes returned by
    ``Block.to_html_attrs()`` for an equivalent objective."""
    from generate_course import _render_objectives  # noqa: E402

    objs = [
        {
            "id": "TO-01",
            "statement": "Define X",
            "bloom_level": "remember",
            "bloom_verb": "define",
        }
    ]
    legacy_html = _render_objectives(objs)
    block = Block(
        block_id="page_01#objective_TO-01_0",
        block_type="objective",
        page_id="page_01",
        sequence=0,
        content="Define X",
        objective_ids=("TO-01",),
        bloom_level="remember",
        bloom_verb="define",
        cognitive_domain="factual",
    )
    block_attrs = block.to_html_attrs()
    # Expect a `<li ...>` substring whose attributes exactly match.
    assert f"<li{block_attrs}>" in legacy_html


@pytest.mark.skipif(_legacy_emit_blocks_flag_set(), reason=_SKIP_REASON)
def test_block_to_html_attrs_byte_equal_to_legacy_render_flip_cards() -> None:
    from generate_course import (  # noqa: E402
        _map_teaching_role,
        _render_flip_cards,
        _slugify,
    )

    terms = [{"term": "Term One", "definition": "Definition one"}]
    legacy_html = _render_flip_cards(terms)
    block = Block(
        block_id="page_01#flip_card_grid_term_one_0",
        block_type="flip_card_grid",
        page_id="page_01",
        sequence=0,
        content={"terms": terms},
        key_terms=(_slugify("Term One"),),
        teaching_role=_map_teaching_role("flip-card", "term-definition"),
    )
    block_attrs = block.to_html_attrs()
    # `flip_card_grid` legacy emit splits the per-card attrs across two
    # text lines (the wrapper `<div class="flip-card" ...>` opens on one
    # line and continues `data-cf-*` on the next, indented). Byte-stable
    # comparison: every space-leading data-cf-* attribute the Block
    # emits must appear in the legacy output. Whitespace between attrs
    # is renderer-controlled, not Block-controlled, so we don't assert
    # on a contiguous substring here.
    for attr in (
        ' data-cf-component="flip-card"',
        ' data-cf-purpose="term-definition"',
        ' data-cf-teaching-role="introduce"',
        ' data-cf-term="term-one"',
    ):
        assert attr in legacy_html
        assert attr in block_attrs


@pytest.mark.skipif(_legacy_emit_blocks_flag_set(), reason=_SKIP_REASON)
def test_block_to_html_attrs_byte_equal_to_legacy_render_self_check() -> None:
    from generate_course import _map_teaching_role, _render_self_check  # noqa: E402

    questions = [
        {
            "question": "What is X?",
            "options": [
                {"text": "A", "correct": True, "feedback": "good"},
                {"text": "B", "correct": False, "feedback": "no"},
            ],
            "bloom_level": "remember",
            "objective_ref": "CO-01",
        }
    ]
    legacy_html = _render_self_check(questions)
    block = Block(
        block_id="page_01#self_check_question_q1_0",
        block_type="self_check_question",
        page_id="page_01",
        sequence=0,
        content={"question": "What is X?", "options": questions[0]["options"]},
        bloom_level="remember",
        teaching_role=_map_teaching_role("self-check", "formative-assessment"),
        objective_ids=("CO-01",),
    )
    block_attrs = block.to_html_attrs()
    assert f'<div class="self-check"{block_attrs}>' in legacy_html


@pytest.mark.skipif(_legacy_emit_blocks_flag_set(), reason=_SKIP_REASON)
def test_block_to_html_attrs_byte_equal_to_legacy_render_activities() -> None:
    from generate_course import _map_teaching_role, _render_activities  # noqa: E402

    acts = [
        {
            "title": "Practice",
            "description": "<p>Do thing</p>",
            "bloom_level": "apply",
            "objective_ref": "CO-02",
        }
    ]
    legacy_html = _render_activities(acts)
    block = Block(
        block_id="page_01#activity_practice_0",
        block_type="activity",
        page_id="page_01",
        sequence=0,
        content={"title": "Practice", "description": "<p>Do thing</p>"},
        bloom_level="apply",
        teaching_role=_map_teaching_role("activity", "practice"),
        objective_ids=("CO-02",),
    )
    block_attrs = block.to_html_attrs()
    assert f'<div class="activity-card"{block_attrs}>' in legacy_html


@pytest.mark.skipif(_legacy_emit_blocks_flag_set(), reason=_SKIP_REASON)
def test_block_to_html_attrs_byte_equal_to_legacy_render_content_section_heading() -> None:
    from generate_course import _render_content_sections  # noqa: E402

    sections = [
        {
            "heading": "Explanation",
            "level": 2,
            "content_type": "explanation",
            "key_terms": [],
            "paragraphs": ["body."],
            "bloom_range": "understand-apply",
        }
    ]
    legacy_html = _render_content_sections(sections)
    block = Block(
        block_id="page_01#explanation_explanation_0",
        block_type="explanation",
        page_id="page_01",
        sequence=0,
        content="Explanation",
        content_type_label="explanation",
        bloom_range="understand-apply",
    )
    block_attrs = block.to_html_attrs()
    # legacy emits `<h2 data-cf-content-type="..." data-cf-bloom-range="...">`
    assert f"<h2{block_attrs}>" in legacy_html


@pytest.mark.skipif(_legacy_emit_blocks_flag_set(), reason=_SKIP_REASON)
def test_block_to_html_attrs_byte_equal_to_legacy_render_callout() -> None:
    from generate_course import _render_content_sections  # noqa: E402

    sections = [
        {
            "heading": "Heading",
            "level": 2,
            "content_type": "explanation",
            "key_terms": [],
            "paragraphs": [],
            "callout": {
                "type": "callout-warning",
                "label": "Warning",
                "heading": "Heads up",
                "items": ["Be careful."],
            },
        }
    ]
    legacy_html = _render_content_sections(sections)
    block = Block(
        block_id="page_01#callout_warning_0",
        block_type="callout",
        page_id="page_01",
        sequence=0,
        content={"items": ["Be careful."]},
        content_type_label="application-note",
    )
    block_attrs = block.to_html_attrs()
    # `<div class="callout callout-warning" role="region" aria-label="Warning"`
    # then the data-cf-content-type attr
    assert block_attrs.lstrip() in legacy_html
