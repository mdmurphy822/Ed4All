"""Regression — the QTI harvest recovers the answer key already in the XML.

Before this landed, ``_parse_qti_assessment_items`` collected only the
``response_label`` MATTEXT strings and threw away each label's ``@ident``; it
never looked at ``resprocessing`` / ``respcondition`` / ``varequal`` at all. The
harvested ``assessment_item`` chunks therefore carried none of the four fields
``Trainforge/synthesis/synthesis_eligibility.py::pair_eligibility`` accepts
(``correct_answer`` / ``answer_key`` / ``reference_answer`` /
``assessment_answer``), so EVERY harvested assessment chunk hard-failed with
``assessment_answer_key_missing`` — on a real course that was 43% of the whole
chunkset.

These tests pin, against XML built by the REAL emitter
(``Courseforge/scripts/packaging/qti_emitter.py``) wherever possible:

* the key resolves through ``varequal`` -> ``response_label/@ident`` -> choice
  text, and the RIGHT choice wins even with per-distractor ``itemfeedback``
  respconditions present (those name distractor idents and must not be read as
  the key);
* a genuinely keyless item (essay / short answer, no ``resprocessing``) stays
  keyless — no answer is invented for it;
* a fill-in item, which has no ``response_label`` at all, takes the literal
  ``varequal`` string as its answer;
* a ``<not>``-negated ``varequal`` (a WRONG answer by construction) is never
  taken as the key;
* an item that DECLARES response labels but whose scoring ``varequal`` matches
  none of them is reported LOUDLY and left keyless, never guessed;
* the resulting chunk clears ``pair_eligibility``'s assessment answer-key check.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _harvest_qti_assessment_chunks,
    _parse_qti_assessment_items,
)


def _fake_create_chunk(**kwargs: Any) -> Dict[str, Any]:
    """Minimal stand-in for ``_run_imscc_chunking``'s v4 emit callback."""
    return {
        "id": kwargs["chunk_id"],
        "text": kwargs["text"],
        "chunk_type": kwargs["chunk_type"],
        "learning_outcome_refs": kwargs["item"]["objective_refs"],
    }


def _harvest(xml: str) -> List[Dict[str, Any]]:
    return _harvest_qti_assessment_chunks(
        [{"path": "06_assessments/quiz.xml", "content": xml}],
        create_chunk=_fake_create_chunk,
        existing_chunks=[],
        course_code="DEMO",
    )


def _quiz(questions: List[Dict[str, Any]]) -> str:
    from Courseforge.scripts.packaging.qti_emitter import assessment_to_qti

    return assessment_to_qti({
        "assessment_id": "ASM-1",
        "title": "Sample Quiz",
        "questions": questions,
    })


# --------------------------------------------------------------------------- #
# choice items — ident -> answer text resolution
# --------------------------------------------------------------------------- #

def test_multiple_choice_answer_resolves_to_the_correct_choice_text() -> None:
    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "multiple_choice",
        "stem": "<p>Which term names the rate of change?</p>",
        "objective_id": "CO-01",
        "bloom_level": "understand",
        "choices": [
            {"id": "A", "text": "Derivative", "is_correct": True},
            {"id": "B", "text": "Antiderivative", "is_correct": False},
            {"id": "C", "text": "Residual", "is_correct": False},
        ],
        "correct_answer": "A",
    }])

    _title, items = _parse_qti_assessment_items(xml)
    assert len(items) == 1
    item = items[0]
    assert item["answer_idents"] == ["A"]
    assert [c["ident"] for c in item["choices"]] == ["A", "B", "C"]
    assert item["answer_match_rule"] == "any_of"
    assert not item["unresolved_answer_idents"]

    chunks = _harvest(xml)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["correct_answer"] == "Derivative"
    assert chunk["correct_answers"] == ["Derivative"]
    assert chunk["correct_answer_idents"] == ["A"]
    assert chunk["answer_key_source"] == "qti_resprocessing"


def test_distractor_itemfeedback_respconditions_are_not_read_as_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-distractor feedback respconditions must not poison the key.

    With ``itemfeedback`` on (the default), the emitter appends one extra
    ``<respcondition continue="Yes">`` per distractor carrying a ``varequal``
    naming THAT distractor. Reading ``varequal`` from the whole
    ``<resprocessing>`` subtree would harvest wrong answers alongside the right
    one; only the ``<setvar>``-carrying scoring condition is the key.
    """
    monkeypatch.setenv("ED4ALL_ASSESSMENT_ITEMFEEDBACK", "1")
    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "multiple_choice",
        "stem": "<p>Which term names the rate of change?</p>",
        "objective_id": "CO-01",
        "bloom_level": "understand",
        "feedback": "The derivative is the instantaneous rate of change.",
        "choices": [
            {"id": "A", "text": "Derivative", "is_correct": True},
            {"id": "B", "text": "Antiderivative", "is_correct": False,
             "misconception_note": "Confuses integration with differentiation."},
            {"id": "C", "text": "Residual", "is_correct": False,
             "misconception_note": "Confuses error terms with slopes."},
        ],
        "correct_answer": "A",
    }])
    # Guard the premise: the feedback respconditions really are present.
    assert xml.count("respcondition") > 2
    assert "displayfeedback" in xml

    chunks = _harvest(xml)
    assert len(chunks) == 1
    assert chunks[0]["correct_answer"] == "Derivative"
    assert chunks[0]["correct_answer_idents"] == ["A"]


def test_multiple_response_key_is_all_of_and_keeps_every_ident() -> None:
    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "multiple_response",
        "stem": "<p>Select every prime.</p>",
        "objective_id": "CO-01",
        "bloom_level": "apply",
        "choices": [
            {"id": "A", "text": "Two", "is_correct": True},
            {"id": "B", "text": "Four", "is_correct": False},
            {"id": "C", "text": "Five", "is_correct": True},
        ],
    }])

    chunks = _harvest(xml)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["correct_answers"] == ["Two", "Five"]
    assert chunk["correct_answer_idents"] == ["A", "C"]
    assert chunk["answer_match_rule"] == "all_of"
    assert chunk["correct_answer"] == "Two; Five"


# --------------------------------------------------------------------------- #
# fill-in items — the varequal text IS the answer (no response labels)
# --------------------------------------------------------------------------- #

def test_fill_in_blank_takes_the_literal_varequal_answer() -> None:
    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "fill_in_blank",
        "stem": "<p>The derivative of a constant is ___ .</p>",
        "objective_id": "CO-01",
        "bloom_level": "remember",
        "correct_answer": "0",
    }])

    _title, items = _parse_qti_assessment_items(xml)
    assert items[0]["choices"] == []
    assert items[0]["answer_idents"] == []
    assert items[0]["answer_html"] == ["0"]
    assert not items[0]["unresolved_answer_idents"]

    chunks = _harvest(xml)
    assert chunks[0]["correct_answer"] == "0"
    # No response labels means nothing to resolve an ident THROUGH, so the
    # ident field is correctly absent rather than fabricated.
    assert "correct_answer_idents" not in chunks[0]


def test_pattern_match_accepted_forms_are_any_of() -> None:
    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "fill_in_blank",
        "stem": "<p>Write one half as a number.</p>",
        "objective_id": "CO-01",
        "bloom_level": "remember",
        "correct_answers": ["1/2", "0.5"],
    }])

    chunks = _harvest(xml)
    assert chunks[0]["correct_answers"] == ["1/2", "0.5"]
    assert chunks[0]["answer_match_rule"] == "any_of"


# --------------------------------------------------------------------------- #
# anti-fabrication
# --------------------------------------------------------------------------- #

def test_essay_item_stays_keyless() -> None:
    """An ungraded item emits no resprocessing and is legitimately keyless."""
    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "essay",
        "stem": (
            "<p>Explain, in your own words, why the derivative of a constant "
            "function is zero everywhere on its domain.</p>"
        ),
        "objective_id": "CO-01",
        "bloom_level": "evaluate",
    }])
    assert "resprocessing" not in xml

    chunks = _harvest(xml)
    assert len(chunks) == 1
    for field in (
        "correct_answer", "correct_answers", "correct_answer_idents",
        "answer_key_source",
    ):
        assert field not in chunks[0], f"invented {field} on an ungraded item"


def _hand_built_quiz(item_body: str) -> str:
    """A minimal namespace-naked QTI document wrapping one raw ``<item>``.

    Used only for shapes the production emitter cannot produce (a broken key,
    a negated condition) — everything the emitter CAN produce is tested against
    the emitter itself.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<questestinterop>"
        '<assessment ident="ASM-1" title="Sample Quiz"><section ident="root">'
        f"{item_body}"
        "</section></assessment></questestinterop>"
    )


_CHOICE_PRESENTATION = (
    "<presentation>"
    "<material><mattext texttype=\"text/html\">"
    "&lt;p&gt;Which term names the rate of change?&lt;/p&gt;"
    "</mattext></material>"
    '<response_lid ident="R1" rcardinality="Single"><render_choice>'
    '<response_label ident="A"><material><mattext texttype="text/html">'
    "Derivative</mattext></material></response_label>"
    '<response_label ident="B"><material><mattext texttype="text/html">'
    "Antiderivative</mattext></material></response_label>"
    "</render_choice></response_lid>"
    "</presentation>"
)


def test_unresolvable_varequal_on_a_choice_item_is_loud_and_keyless(
    caplog: pytest.LogCaptureFixture,
) -> None:
    xml = _hand_built_quiz(
        '<item ident="Q-1" title="CO-01">'
        + _CHOICE_PRESENTATION
        + "<resprocessing><outcomes>"
        '<decvar varname="SCORE" vartype="Decimal" minvalue="0" maxvalue="1"/>'
        "</outcomes><respcondition><conditionvar>"
        '<varequal respident="R1">ZZ</varequal>'
        "</conditionvar>"
        '<setvar varname="SCORE" action="Set">1</setvar>'
        "</respcondition></resprocessing></item>"
    )

    _title, items = _parse_qti_assessment_items(xml)
    assert items[0]["unresolved_answer_idents"] == ["ZZ"]
    assert items[0]["answer_idents"] == []
    assert items[0]["answer_html"] == []

    with caplog.at_level(logging.ERROR):
        chunks = _harvest(xml)

    assert len(chunks) == 1
    assert "correct_answer" not in chunks[0], "guessed an answer from a bad key"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an unresolvable answer key must be reported loudly"
    joined = " ".join(r.getMessage() for r in errors)
    assert "ZZ" in joined
    assert "Q-1" in joined


def test_negated_varequal_is_never_taken_as_the_answer() -> None:
    """``<not><varequal>B</varequal></not>`` names a WRONG choice."""
    xml = _hand_built_quiz(
        '<item ident="Q-1" title="CO-01">'
        + _CHOICE_PRESENTATION
        + "<resprocessing><outcomes>"
        '<decvar varname="SCORE" vartype="Decimal" minvalue="0" maxvalue="1"/>'
        "</outcomes><respcondition><conditionvar>"
        '<varequal respident="R1">A</varequal>'
        '<not><varequal respident="R1">B</varequal></not>'
        "</conditionvar>"
        '<setvar varname="SCORE" action="Set">1</setvar>'
        "</respcondition></resprocessing></item>"
    )

    _title, items = _parse_qti_assessment_items(xml)
    assert items[0]["answer_idents"] == ["A"]
    assert items[0]["unresolved_answer_idents"] == []

    chunks = _harvest(xml)
    assert chunks[0]["correct_answer"] == "Derivative"
    assert chunks[0]["correct_answers"] == ["Derivative"]


# --------------------------------------------------------------------------- #
# the point of the whole exercise
# --------------------------------------------------------------------------- #

def test_harvested_chunk_clears_the_pair_eligibility_answer_key_check() -> None:
    """The harvested chunk no longer trips ``assessment_answer_key_missing``.

    Asserted through the production ``pair_eligibility`` entry point (not a leaf
    predicate), on a chunk carrying the rest of the shape that gate needs.
    """
    from Trainforge.synthesis.synthesis_eligibility import pair_eligibility

    xml = _quiz([{
        "question_id": "q-1",
        "question_type": "multiple_choice",
        "stem": (
            "<p>A skydiver reaches terminal velocity when the upward drag "
            "force on the body exactly balances the downward gravitational "
            "force. Which quantity is zero at that moment?</p>"
        ),
        "objective_id": "CO-01",
        "bloom_level": "understand",
        "choices": [
            {"id": "A", "text": "The net acceleration", "is_correct": True},
            {"id": "B", "text": "The gravitational force", "is_correct": False},
            {"id": "C", "text": "The drag force", "is_correct": False},
        ],
        "correct_answer": "A",
    }])

    focus = {
        "id": "CO-01",
        "statement": (
            "Explain why a skydiver at terminal velocity has zero net "
            "acceleration."
        ),
        "bloom_level": "understand",
    }

    def _focused(chunk: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(chunk)
        out["concept_tags"] = ["terminal velocity", "drag force"]
        out["learning_outcome_refs"] = ["CO-01"]
        out["synthesis_focus_objective"] = focus
        return out

    chunk = _harvest(xml)[0]
    assert chunk["correct_answer"]

    keyless = _focused(chunk)
    keyless.pop("correct_answer", None)
    keyless.pop("correct_answers", None)
    before = pair_eligibility(keyless, kind="instruction")
    assert before.reason == "assessment_answer_key_missing", (
        "premise check: without the harvested key this chunk must fail on the "
        f"answer-key gate, got {before.reason!r}"
    )

    after = pair_eligibility(_focused(chunk), kind="instruction")
    assert after.reason != "assessment_answer_key_missing"
