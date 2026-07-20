"""Regression net for TRACK ASSESS-A (backlog A1, A2, A3, A5).

Covers, all offline with mocked/injected clients (no network, no model load —
LAW 1) and NO course slugs (LAW 2):

- A1 LLM apply-arm (``ED4ALL_ASSESSMENT_APPLY_ARM``): the mandatory verify
  chain (sympy re-check fail-closed → groundedness entailment → Bloom trivote
  clamp), the license guard on the provider seat, and the DecisionCapture that
  MUST fire per draft call (LAW 3).
- A2 item-linter Bloom-trivote seam (``ED4ALL_ASSESSMENT_ITEM_TRIVOTE``):
  structural ceiling stays the default; the trivote arm fires only when the
  flag is on.
- A3 apparatus-guard numeric recovery
  (``ED4ALL_ASSESSMENT_NUMERIC_RECOVERY``): default off is byte-identical; on,
  a plain-text ``Solution:`` equation yields a sympy-verified numeric item.
- A5 discussion/assignment NLI text-grounding arm
  (``ED4ALL_DISCUSSION_GROUNDING_NLI``): default legacy refs-only; on, the NLI
  verdict is authoritative for a resolvable item.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment_generator import (  # noqa: E402
    AssessmentGenerator,
    _apply_arm_enabled,
    _apply_arm_max,
    _apply_arm_provider_allowed,
)
from Trainforge.generators.content_extractor import (  # noqa: E402
    ContentExtractor,
    _numeric_recovery_enabled,
)


# --------------------------------------------------------------------------- #
# Helpers: an injected fake groundedness report + fake provider.
# --------------------------------------------------------------------------- #

def _report(*, available=True, contradicted=0, rate=1.0):
    return SimpleNamespace(
        available=available,
        contradicted_count=contradicted,
        groundedness_rate=rate,
    )


def _entailed_scorer(answer, passages, **_kw):
    return _report(rate=1.0)


def _unsupported_scorer(answer, passages, **_kw):
    return _report(rate=0.0)


class _FakeProvider:
    """Stand-in for AssessmentGeneratorProvider — returns canned questions."""

    def __init__(self, questions):
        self._questions = questions
        self._provider = "local"
        self._model = "fake-qwen"

    def generate_assessments(self, **_kw):
        return {"questions": list(self._questions), "skipped_items": []}


class _RecordingCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kw):
        self.events.append(kw)


# A cited chunk whose text supports the drafted stem AND carries the equation
# the sympy re-check solves. No course slug anywhere.
_CITED_CHUNK = {
    "id": "c_apply_1",
    "text": (
        "A ticket costs the same for every rider. When 3 tickets are bought "
        "for a total of 3x = 12 dollars, the price x satisfies 3x = 12."
    ),
}


def _apply_generator(questions, *, scorer=_entailed_scorer, zshot=None):
    return AssessmentGenerator(
        capture=_RecordingCapture(),
        check_leaks=False,
        assessment_provider=_FakeProvider(questions),
        groundedness_scorer=scorer,
        bloom_zero_shot_fn=zshot,
    )


# =========================================================================== #
# A1 — apply-arm verify chain
# =========================================================================== #

def test_apply_arm_flag_default_off():
    assert _apply_arm_enabled() is False


def test_apply_arm_max_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_APPLY_ARM_MAX", raising=False)
    assert _apply_arm_max() == 4
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM_MAX", "garbage")
    assert _apply_arm_max() == 4
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM_MAX", "-3")
    assert _apply_arm_max() == 4
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM_MAX", "6")
    assert _apply_arm_max() == 6


def test_apply_arm_provider_license_guard():
    assert _apply_arm_provider_allowed("local") is True
    assert _apply_arm_provider_allowed("together") is True
    assert _apply_arm_provider_allowed("anthropic") is False
    assert _apply_arm_provider_allowed("claude_session") is False
    assert _apply_arm_provider_allowed("") is False
    # Roster verdict flows through.
    roster = {"barredteacher": {"verdict": "barred"}}
    assert _apply_arm_provider_allowed("barredteacher", roster) is False
    assert _apply_arm_provider_allowed("local", roster) is True


def test_apply_arm_ships_verified_item_and_fires_capture(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM", "1")
    q = {
        "stem": "A rider buys 3 equal tickets for 3x = 12 dollars; find x.",
        "bloom_level": "apply",
        "objective_id": "TO-01",
        "question_type": "fill_in_blank",
        "correct_answer": "4",
        "source_chunks": ["c_apply_1"],
        "feedback": "x = 4 because 3x = 12.",
    }
    gen = _apply_generator([q])
    out = gen.generate_diversified(
        course_code="APPLYARM",
        objective_ids=["TO-01"],
        bloom_levels=["apply"],
        question_count=1,
        source_chunks=[_CITED_CHUNK],
    )
    apply_items = [x for x in out.questions if x.item_subtype == "apply_word_problem"]
    assert len(apply_items) == 1
    assert apply_items[0].bloom_level == "apply"
    # LAW 3: exactly one apply-arm DecisionCapture event fired with the runtime
    # provider + dynamic verify counts.
    arm_events = [
        e for e in gen.capture.events
        if str(e.get("decision", "")).startswith("apply_arm:")
    ]
    assert len(arm_events) == 1
    rationale = arm_events[0]["rationale"]
    assert "provider=local" in rationale
    assert "shipped=1" in rationale
    assert len(rationale) >= 20


def test_apply_arm_drops_numeric_key_that_fails_sympy(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM", "1")
    # Declared numeric key 99 does NOT solve the cited equation 3x = 12 → drop.
    q = {
        "stem": "A rider buys 3 equal tickets for 3x = 12 dollars; find x.",
        "bloom_level": "apply",
        "objective_id": "TO-01",
        "question_type": "fill_in_blank",
        "correct_answer": "99",
        "source_chunks": ["c_apply_1"],
    }
    gen = _apply_generator([q])
    out = gen.generate_diversified(
        course_code="APPLYARM", objective_ids=["TO-01"],
        bloom_levels=["apply"], question_count=1, source_chunks=[_CITED_CHUNK],
    )
    assert not [x for x in out.questions if x.item_subtype == "apply_word_problem"]


def test_apply_arm_drops_unsupported_grounding(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM", "1")
    q = {
        "stem": "A conceptual apply question with no numeric key.",
        "bloom_level": "apply",
        "objective_id": "TO-01",
        "question_type": "multiple_choice",
        "choices": [
            {"text": "right", "is_correct": True},
            {"text": "wrong", "is_correct": False},
        ],
        "source_chunks": ["c_apply_1"],
    }
    gen = _apply_generator([q], scorer=_unsupported_scorer)
    out = gen.generate_diversified(
        course_code="APPLYARM", objective_ids=["TO-01"],
        bloom_levels=["apply"], question_count=1, source_chunks=[_CITED_CHUNK],
    )
    assert not [x for x in out.questions if x.item_subtype == "apply_word_problem"]


def test_apply_arm_clamps_bloom_when_trivote_unavailable(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPLY_ARM", "1")
    # Claim analyze, but no zero-shot voter and a stem with no canonical verb →
    # only the asserted voter is present → trivote skipped → clamp to the
    # apply_word_problem structural ceiling (apply).
    q = {
        "stem": "The quantity in the pooled fund for 3x = 12 dollars.",
        "bloom_level": "analyze",
        "objective_id": "TO-01",
        "question_type": "multiple_choice",
        "choices": [{"text": "a", "is_correct": True}, {"text": "b", "is_correct": False}],
        "source_chunks": ["c_apply_1"],
    }
    gen = _apply_generator([q])
    out = gen.generate_diversified(
        course_code="APPLYARM", objective_ids=["TO-01"],
        bloom_levels=["apply"], question_count=1, source_chunks=[_CITED_CHUNK],
    )
    items = [x for x in out.questions if x.item_subtype == "apply_word_problem"]
    assert len(items) == 1
    assert items[0].bloom_level == "apply"  # clamped from analyze


def test_apply_arm_off_produces_no_apply_items(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_APPLY_ARM", raising=False)
    q = {
        "stem": "3x = 12; find x.", "bloom_level": "apply",
        "objective_id": "TO-01", "question_type": "fill_in_blank",
        "correct_answer": "4", "source_chunks": ["c_apply_1"],
    }
    gen = _apply_generator([q])
    out = gen.generate_diversified(
        course_code="APPLYARM", objective_ids=["TO-01"],
        bloom_levels=["apply"], question_count=1, source_chunks=[_CITED_CHUNK],
    )
    assert not [x for x in out.questions if x.item_subtype == "apply_word_problem"]


# =========================================================================== #
# A3 — apparatus-guard numeric recovery
# =========================================================================== #

# A plain-text (no LaTeX/<code>) Solution equation the marked-math harvester
# misses — the GLM-OCR scan shape.
_PLAINTEXT_SOLUTION_CHUNK = {
    "id": "c_scan_1",
    "text": "<p>Solution: 3x + 5 = 20 so x = 5.</p>",
    "chunk_type": "worked_example",
}


def test_numeric_recovery_default_off_byte_identical(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_NUMERIC_RECOVERY", raising=False)
    assert _numeric_recovery_enabled() is False
    cx = ContentExtractor()
    assert cx.recover_numeric_equation_candidates([_PLAINTEXT_SOLUTION_CHUNK]) == []
    # And the numeric-FIB builder finds nothing in a plain-text scan chunk.
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    r = gen.build_numeric_fib("Q1", "TO-01", "apply", [_PLAINTEXT_SOLUTION_CHUNK])
    from Trainforge.generators.assessment_generator import SkippedItem
    assert isinstance(r, SkippedItem)


def test_numeric_recovery_on_yields_verified_item(monkeypatch):
    pytest.importorskip("sympy")
    monkeypatch.setenv("ED4ALL_ASSESSMENT_NUMERIC_RECOVERY", "1")
    cx = ContentExtractor()
    cands = cx.recover_numeric_equation_candidates([_PLAINTEXT_SOLUTION_CHUNK])
    assert cands, "recovery should harvest the Solution: equation"
    assert any("=" in frag for frag, _cid in cands)
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    q = gen.build_numeric_fib("Q1", "TO-01", "apply", [_PLAINTEXT_SOLUTION_CHUNK])
    from Trainforge.generators.assessment_generator import QuestionData
    assert isinstance(q, QuestionData)
    assert q.correct_answer == "5"
    assert q.item_subtype == "fib_numeric"


# =========================================================================== #
# A2 — item-linter Bloom-trivote seam
# =========================================================================== #

def _item(bloom, stem, subtype="mc_single"):
    # Dict path in AssessmentItemWritingValidator reads the "stem" + "choices"
    # keys (the canonical QuestionData.to_dict() shape).
    return {
        "question_id": "item-1",
        "item_subtype": subtype,
        "bloom_level": bloom,
        "question_type": "multiple_choice",
        "stem": f"<p>{stem}</p>",
        "choices": [
            {"text": "the correct choice here", "is_correct": True},
            {"text": "a distractor option here", "is_correct": False},
            {"text": "another distractor option", "is_correct": False},
        ],
    }


def _codes(result):
    return {i.code for i in result.issues}


def test_item_trivote_off_no_trivote_code(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_ITEM_TRIVOTE", raising=False)
    from lib.validators.assessment_item_writing import (
        AssessmentItemWritingValidator,
    )
    # Assert analyze but the verb says "define" (remember) — the trivote WOULD
    # flag, but the arm is off so it must not.
    res = AssessmentItemWritingValidator().validate(
        {"assessment_items": [_item("analyze", "Define the term slope.")]}
    )
    assert "ITEM_BLOOM_TRIVOTE_UNSUPPORTED" not in _codes(res)


def test_item_trivote_on_flags_unsupported_claim(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_ITEM_TRIVOTE", "1")
    from lib.validators.assessment_item_writing import (
        AssessmentItemWritingValidator,
    )
    # asserted=analyze, verb voter reads "define" → remember → disagreement.
    res = AssessmentItemWritingValidator().validate(
        {"assessment_items": [_item("analyze", "Define the term slope.")]}
    )
    assert "ITEM_BLOOM_TRIVOTE_UNSUPPORTED" in _codes(res)


def test_item_trivote_on_agreeing_claim_not_flagged(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_ITEM_TRIVOTE", "1")
    from lib.validators.assessment_item_writing import (
        AssessmentItemWritingValidator,
    )
    # asserted=analyze, verb voter reads "differentiate" → analyze → supported.
    res = AssessmentItemWritingValidator().validate(
        {"assessment_items": [
            _item("analyze", "Differentiate the two solution methods.",
                  "error_analysis")
        ]}
    )
    assert "ITEM_BLOOM_TRIVOTE_UNSUPPORTED" not in _codes(res)


# =========================================================================== #
# A5 — discussion/assignment NLI text-grounding arm
# =========================================================================== #

def _write_chunks(tmp_path, rows):
    import json
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_discussion_grounding_nli_default_off_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_DISCUSSION_GROUNDING_NLI", raising=False)
    from lib.validators.discussion_assignment_grounding import (
        DiscussionAssignmentGroundingValidator,
        _grounding_nli_enabled,
    )
    assert _grounding_nli_enabled() is False
    chunks = _write_chunks(tmp_path, [
        {"id": "k1", "text": "Photosynthesis converts light to chemical energy.",
         "learning_outcome_refs": ["to-01"]},
    ])
    # Legacy path: objective_id resolves to a chunk → grounded (no NLI even if
    # the prompt text is off-topic).
    res = DiscussionAssignmentGroundingValidator().validate({
        "discussion_items": [
            {"discussion_id": "d1", "objective_id": "TO-01",
             "text": "Totally unrelated prompt about tax law."}
        ],
        "chunks_path": chunks,
    })
    assert "DISCUSSION_UNGROUNDED" not in {i.code for i in res.issues}


def test_discussion_grounding_nli_on_refutes_offtopic(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_DISCUSSION_GROUNDING_NLI", "1")
    from lib.validators.discussion_assignment_grounding import (
        DiscussionAssignmentGroundingValidator,
    )
    chunks = _write_chunks(tmp_path, [
        {"id": "k1", "text": "Photosynthesis converts light to chemical energy.",
         "learning_outcome_refs": ["to-01"]},
    ])
    res = DiscussionAssignmentGroundingValidator().validate({
        "discussion_items": [
            {"discussion_id": "d1", "objective_id": "TO-01",
             "text": "Discuss the photosynthesis prompt."}
        ],
        "chunks_path": chunks,
        # Injected scorer says NOT entailed → NLI arm flags ungrounded even
        # though the objective_id resolves (legacy would pass).
        "groundedness_scorer": _unsupported_scorer,
    })
    assert "DISCUSSION_UNGROUNDED" in {i.code for i in res.issues}


def test_discussion_grounding_nli_on_confirms_entailed(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_DISCUSSION_GROUNDING_NLI", "1")
    from lib.validators.discussion_assignment_grounding import (
        DiscussionAssignmentGroundingValidator,
    )
    chunks = _write_chunks(tmp_path, [
        {"id": "k1", "text": "Photosynthesis converts light to chemical energy.",
         "learning_outcome_refs": ["to-01"]},
    ])
    res = DiscussionAssignmentGroundingValidator().validate({
        "discussion_items": [
            {"discussion_id": "d1", "objective_id": "TO-01",
             "text": "Discuss the photosynthesis prompt.",
             "source_chunk_ids": ["k1"]}
        ],
        "chunks_path": chunks,
        "groundedness_scorer": _entailed_scorer,
    })
    assert "DISCUSSION_UNGROUNDED" not in {i.code for i in res.issues}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-x", "-q"]))
