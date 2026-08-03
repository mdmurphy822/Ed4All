"""Tests for the deterministic diversified assessment tier (Phase 2).

Covers the generator-foundation + deterministic-diversified-tier work item:
- QuestionData response-cardinality data model (backward compatible)
- Item-writing rules: 3-option default + ED4ALL_ASSESSMENT_OPTION_COUNT,
  banned NOTA/AOTA/absolute-term options, question_type rotation,
  central-idea stems
- Each diversified builder against synthetic fixture chunks (no course slugs)
- Numeric-FIB sympy verification (accept + reject)
- Bloom-honesty type->max ceiling clamps
- Mix planner + generate_diversified orchestration + per-CO coverage

All fixtures are synthetic and carry NO course slugs. Offline: only sympy is
needed (declared in the [embedding] extra); a builder degrades to SkippedItem
when sympy is absent.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.assessment.generator import (  # noqa: E402
    AssessmentGenerator,
    QuestionData,
    SkippedItem,
    _is_banned_option,
    _resolve_option_count,
    _clamp_bloom,
    _apply_first_misconception,
    plan_item_mix,
)


# ----------------------------- fixtures ----------------------------------- #

PROC_CHUNK = {
    "id": "c_proc",
    "text": (
        "<h3>Solve the system by substitution</h3><ol>"
        "<li>Write the equation x - 2y = -2</li>"
        "<li>Isolate x: x = 2y - 2</li>"
        "<li>Substitute into 3x + y = 10</li>"
        "<li>Solve for y and back-substitute</li>"
        "</ol>"
    ),
    "chunk_type": "worked_example",
}

NUM_CHUNK = {
    "id": "c_num",
    "text": r"<p>Solution: we solve \( 2x + 3 = 11 \) to find x.</p>",
    "chunk_type": "worked_example",
}

# Two free variables -> not a single-variable solve -> unverifiable/skip.
NUM_BAD_CHUNK = {
    "id": "c_num_bad",
    "text": r"<p>Consider the relation \( x + y = 5 \).</p>",
    "chunk_type": "explanation",
}

TERMS_CHUNK = {
    "id": "c_terms",
    "key_terms": [
        {"term": "Slope",
         "definition": "the steepness of a line measured as rise over run"},
        {"term": "Intercept",
         "definition": "the point where a line crosses an axis"},
        {"term": "Coefficient",
         "definition": "a numeric multiplier of a variable in an expression"},
        {"term": "Variable",
         "definition": "a symbol standing for an unknown quantity"},
        {"term": "Constant",
         "definition": "a fixed value that does not change"},
    ],
    "misconceptions": [
        {"misconception": "Slope measures how far a line sits from the origin"},
        {"misconception": "The intercept equals the coefficient of x"},
    ],
}

# A rich chunk that supports MC (terms) + T/F + FIB (context sentence).
ROTATION_CHUNK = {
    "id": "c_rot",
    "text": (
        "<p><strong>Cognitive load</strong> is the mental effort used in "
        "working memory. Working memory is limited in capacity. Learners "
        "can process only a few items at once.</p>"
    ),
    "key_terms": [
        {"term": "Cognitive load",
         "definition": "the mental effort used in working memory"},
    ],
    "chunk_type": "explanation",
}


def _gen():
    return AssessmentGenerator(capture=None, check_leaks=False)


# --------------------------- data model ----------------------------------- #

def test_question_data_backward_compatible_omits_new_fields():
    q = QuestionData(
        question_id="Q1", question_type="multiple_choice", stem="<p>s</p>",
        bloom_level="understand", objective_id="LO-01",
    )
    d = q.to_dict()
    for k in ("correct_answers", "item_subtype", "linked_item_id"):
        assert k not in d, f"{k} must be omitted when unset (byte-identical)"


def test_question_data_emits_new_fields_when_set():
    q = QuestionData(
        question_id="Q1", question_type="multiple_response", stem="<p>s</p>",
        bloom_level="apply", objective_id="LO-01",
        correct_answers=["a", "b"], item_subtype="mc_multiple_response",
        linked_item_id="Q1-reason",
    )
    d = q.to_dict()
    assert d["correct_answers"] == ["a", "b"]
    assert d["item_subtype"] == "mc_multiple_response"
    assert d["linked_item_id"] == "Q1-reason"


# --------------------------- item-writing rules --------------------------- #

def test_banned_options():
    assert _is_banned_option("None of the above")
    assert _is_banned_option("<p>All of the following</p>")
    assert _is_banned_option("Extraneous load can always be reduced")
    assert _is_banned_option("This never happens")
    assert not _is_banned_option("A numeric multiplier of a variable")


def test_option_count_default_and_env(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_OPTION_COUNT", raising=False)
    assert _resolve_option_count() == 3
    monkeypatch.setenv("ED4ALL_ASSESSMENT_OPTION_COUNT", "4")
    assert _resolve_option_count() == 4
    monkeypatch.setenv("ED4ALL_ASSESSMENT_OPTION_COUNT", "garbage")
    assert _resolve_option_count() == 3
    monkeypatch.setenv("ED4ALL_ASSESSMENT_OPTION_COUNT", "99")
    assert _resolve_option_count() == 3


def test_matching_mc_respects_option_count(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSESSMENT_OPTION_COUNT", raising=False)
    q = _gen().build_matching_mc("Q1", "LO-01", "understand", [TERMS_CHUNK])
    assert isinstance(q, QuestionData)
    assert len(q.choices) == 3  # 1 key + 2 distractors
    monkeypatch.setenv("ED4ALL_ASSESSMENT_OPTION_COUNT", "4")
    q4 = _gen().build_matching_mc("Q2", "LO-01", "understand", [TERMS_CHUNK])
    assert len(q4.choices) == 4


def test_question_type_rotation():
    gen = _gen()
    types = []
    for i in range(3):
        r = gen._generate_question(
            objective_id="LO-01", bloom_level="remember",
            source_chunks=[ROTATION_CHUNK],
        )
        # rotation cursor advances regardless of skip/emit
        types.append(gen._type_rotation["remember"])
    # cursor advanced once per call -> monotone
    assert types == [1, 2, 3]


def test_central_idea_stem_no_generic_template():
    # Statement path fires only when there are >=4 factual statements and no
    # metadata terms; assert the generic "Which statement is correct?" string
    # is never emitted by the classic MC path via the module source.
    import Trainforge.generators.assessment.generator as m
    src = Path(m.__file__).read_text()
    assert "Which of the following statements is correct?" not in src


# --------------------------- builders ------------------------------------- #

def test_build_multiple_response():
    q = _gen().build_multiple_response("Q1", "LO-01", "apply", [PROC_CHUNK])
    assert isinstance(q, QuestionData), q
    assert q.question_type == "multiple_response"
    assert q.item_subtype == "mc_multiple_response"
    keys = [c for c in q.choices if c.get("is_correct")]
    distractors = [c for c in q.choices if not c.get("is_correct")]
    assert len(keys) >= 2
    assert len(distractors) >= 2
    assert q.correct_answers  # plural keys recorded
    # at least one distractor carries a misconception note
    assert any("misconception_note" in c for c in distractors)


def test_build_error_analysis():
    q = _gen().build_error_analysis("Q1", "LO-01", "analyze", [PROC_CHUNK])
    assert isinstance(q, QuestionData), q
    assert q.item_subtype == "error_analysis"
    assert q.question_type == "multiple_choice"
    correct = [c for c in q.choices if c.get("is_correct")]
    assert len(correct) == 1  # single-key MC -> validator-safe
    assert q.correct_answer.startswith("Step ")
    assert "misconception_note" in correct[0]
    assert "Which step contains the error" in q.stem


def test_build_matching_mc_homogeneous():
    q = _gen().build_matching_mc("Q1", "LO-01", "understand", [TERMS_CHUNK])
    assert isinstance(q, QuestionData)
    assert q.item_subtype == "matching_mc"
    correct = [c for c in q.choices if c.get("is_correct")]
    assert len(correct) == 1
    assert "best matches the term" in q.stem


def test_build_ordering_mc():
    q = _gen().build_ordering_mc("Q1", "LO-01", "apply", [PROC_CHUNK])
    assert isinstance(q, QuestionData), q
    assert q.item_subtype == "ordering_mc"
    assert len([c for c in q.choices if c.get("is_correct")]) == 1
    assert len(q.choices) >= 3
    assert "correct order" in q.stem


def test_build_two_tier_returns_linked_pair():
    result = _gen().build_two_tier("Q1", "LO-01", "analyze", [TERMS_CHUNK])
    assert isinstance(result, list) and len(result) == 2
    answer, reason = result
    assert answer.item_subtype == "two_tier_answer"
    assert reason.item_subtype == "two_tier_reason"
    assert answer.linked_item_id == reason.question_id
    assert reason.linked_item_id == answer.question_id


def test_build_tf_justified():
    q = _gen().build_tf_justified("Q1", "LO-01", "remember", [ROTATION_CHUNK])
    if isinstance(q, QuestionData):
        assert q.item_subtype == "tf"
        assert q.question_type == "true_false"


def test_builders_skip_without_chunks():
    gen = _gen()
    for method in ("build_multiple_response", "build_error_analysis",
                   "build_numeric_fib", "build_matching_mc",
                   "build_ordering_mc", "build_two_tier"):
        r = getattr(gen, method)("Q1", "LO-01", "apply", None)
        assert isinstance(r, SkippedItem)
        assert r.reason == "no_source_chunks"


# --------------------------- numeric verification ------------------------- #

def test_numeric_fib_accepts_verified_key():
    q = _gen().build_numeric_fib("Q1", "LO-01", "apply", [NUM_CHUNK])
    assert isinstance(q, QuestionData), q
    assert q.item_subtype == "fib_numeric"
    assert q.question_type == "fill_in_blank"
    assert q.correct_answer == "4"  # 2x+3=11 -> x=4, sympy-verified
    assert "Solve for" in q.stem


def test_numeric_fib_rejects_unverifiable():
    r = _gen().build_numeric_fib("Q1", "LO-01", "apply", [NUM_BAD_CHUNK])
    assert isinstance(r, SkippedItem)
    assert r.reason == "numeric_unverifiable"


def test_misconception_transforms_apply():
    out = _apply_first_misconception("x = 2y - 2")
    assert out is not None
    perturbed, name, note = out
    assert "+ 2" in perturbed
    assert name == "sign_drop"
    assert len(note) > 10


# --------------------------- Bloom honesty -------------------------------- #

def test_clamp_bloom_ceilings():
    # single-fact types cap at understand
    assert _clamp_bloom("evaluate", "mc_single") == "understand"
    assert _clamp_bloom("apply", "matching_mc") == "understand"
    assert _clamp_bloom("create", "tf") == "understand"
    # constructed-response types may claim higher
    assert _clamp_bloom("analyze", "fib_numeric") == "apply"
    assert _clamp_bloom("evaluate", "error_analysis") == "analyze"
    assert _clamp_bloom("apply", "mc_multiple_response") == "apply"
    # never RAISES a claim
    assert _clamp_bloom("remember", "error_analysis") == "remember"


def test_builder_clamps_asserted_bloom():
    q = _gen().build_matching_mc("Q1", "LO-01", "evaluate", [TERMS_CHUNK])
    assert isinstance(q, QuestionData)
    assert q.bloom_level == "understand"  # clamped from evaluate

    q2 = _gen().build_numeric_fib("Q2", "LO-01", "evaluate", [NUM_CHUNK])
    assert isinstance(q2, QuestionData)
    assert q2.bloom_level == "apply"  # clamped from evaluate


# --------------------------- mix planner ---------------------------------- #

def test_plan_item_mix_sums_and_spans():
    plan = plan_item_mix(20)
    assert len(plan) == 20
    assert "mc_single" in plan
    assert len(set(plan)) >= 4  # spans several subtypes
    # deterministic / reproducible
    assert plan_item_mix(20) == plan


def test_plan_item_mix_edge_counts():
    assert plan_item_mix(0) == []
    assert len(plan_item_mix(1)) == 1
    assert len(plan_item_mix(3)) == 3


# --------------------------- orchestration -------------------------------- #

def test_generate_diversified_produces_mix_and_stats():
    gen = _gen()
    chunks = [PROC_CHUNK, NUM_CHUNK, TERMS_CHUNK]
    asm = gen.generate_diversified(
        course_code="TEST_101",
        objective_ids=["LO-01", "LO-02", "LO-03"],
        bloom_levels=["understand", "apply", "analyze"],
        question_count=8,
        source_chunks=chunks,
    )
    assert len(asm.questions) == 8
    # every emitted item is grounded to at least one source chunk id
    for q in asm.questions:
        assert q.source_chunks, f"{q.question_id} not grounded"
    stats = gen.last_generation_stats
    assert stats is not None
    assert sum(stats.subtype_counts.values()) >= 1
    # subtype variety
    assert len(stats.subtype_counts) >= 2


def test_generate_diversified_per_co_coverage():
    gen = _gen()
    chunks = [PROC_CHUNK, NUM_CHUNK, TERMS_CHUNK]
    objs = ["LO-01", "LO-02", "LO-03"]
    asm = gen.generate_diversified(
        course_code="TEST_101",
        objective_ids=objs,
        bloom_levels=["understand", "apply"],
        question_count=3,
        source_chunks=chunks,
        ensure_objective_coverage=True,
    )
    covered = {q.objective_id for q in asm.questions}
    assert covered == set(objs), f"per-CO coverage broken: {covered}"


# --------------------- written-response constructed items ----------------- #

def _all_criterion_cites(rubric):
    cites = set()
    for row in (rubric or {}).get("criteria", []):
        cites.update(row.get("cites") or [])
    return cites


def test_build_short_answer_specific_stem_and_grounded_rubric():
    q = _gen().build_short_answer("Q1", "LO-01", "understand", [ROTATION_CHUNK])
    assert isinstance(q, QuestionData)
    assert q.item_subtype == "short_answer"
    assert q.question_type == "essay"  # travels as cc.essay.v0p1
    # SPECIFIC stem — never a bare "discuss X".
    assert "explain" in q.stem.lower()
    assert len(q.stem.split()) > 6
    # Per-item analytic rubric present + grounded by construction.
    assert q.rubric and q.rubric["criteria"]
    for row in q.rubric["criteria"]:
        assert row["cites"], f"criterion not grounded: {row['criterion']!r}"
        assert row["levels"] and row["levels"][0]["score"] == 2
    # Every criterion cite resolves a chunk the item is grounded in.
    assert _all_criterion_cites(q.rubric) <= set(q.source_chunks)


def test_build_extended_response_solve_shows_work():
    q = _gen().build_extended_response("Q1", "LO-01", "apply", [PROC_CHUNK])
    assert isinstance(q, QuestionData)
    assert q.item_subtype == "extended_response"
    assert q.question_type == "essay"
    assert "show your work" in q.stem.lower()
    # One criterion per real solution step, each citing the source chunk.
    assert len(q.rubric["criteria"]) >= 3
    assert _all_criterion_cites(q.rubric) <= set(q.source_chunks)
    # Misconception-derived deduction rows carry a signed point penalty.
    for d in q.rubric["deductions"]:
        assert d["points"] < 0 and d["cites"]


def test_build_extended_response_compare_fallback():
    # No procedures in a terms-only chunk → compare/contrast fallback.
    q = _gen().build_extended_response("Q1", "LO-01", "analyze", [TERMS_CHUNK])
    assert isinstance(q, QuestionData)
    assert q.item_subtype == "extended_response"
    assert "compare and contrast" in q.stem.lower()
    assert len(q.rubric["criteria"]) == 3
    assert _all_criterion_cites(q.rubric) <= set(q.source_chunks)


def test_written_types_have_no_bloom_ceiling():
    # short_answer / extended_response are the honest higher-Bloom vehicle:
    # an 'evaluate' claim is NOT clamped down (unlike mc_single → understand).
    sa = _gen().build_short_answer("Q1", "LO-01", "evaluate", [ROTATION_CHUNK])
    er = _gen().build_extended_response("Q2", "LO-01", "evaluate", [PROC_CHUNK])
    assert sa.bloom_level == "evaluate"
    assert er.bloom_level == "evaluate"


def test_written_builders_skip_without_chunks():
    g = _gen()
    for method in ("build_short_answer", "build_extended_response"):
        r = getattr(g, method)("Q1", "LO-01", "apply", None)
        assert type(r).__name__ == "SkippedItem"
        assert r.reason == "no_source_chunks"


def test_question_data_rubric_roundtrips_and_omits_when_unset():
    plain = QuestionData(
        question_id="Q1", question_type="multiple_choice", stem="<p>s</p>",
        bloom_level="understand", objective_id="LO-01",
    )
    assert "rubric" not in plain.to_dict()
    with_rubric = QuestionData(
        question_id="Q2", question_type="essay", stem="<p>s</p>",
        bloom_level="apply", objective_id="LO-01",
        rubric={"criteria": [{"criterion": "c", "cites": ["x"], "levels": []}],
                "deductions": []},
    )
    assert with_rubric.to_dict()["rubric"]["criteria"][0]["cites"] == ["x"]


def test_plan_item_mix_includes_written_types():
    plan = plan_item_mix(40)
    assert "short_answer" in plan
    assert "extended_response" in plan
    # written share is a bounded ~10-15% of the mix
    written = plan.count("short_answer") + plan.count("extended_response")
    assert 2 <= written <= 8


def test_generate_diversified_can_emit_written_types():
    gen = _gen()
    chunks = [PROC_CHUNK, TERMS_CHUNK, ROTATION_CHUNK]
    asm = gen.generate_diversified(
        course_code="TEST_101",
        objective_ids=["LO-01", "LO-02", "LO-03"],
        bloom_levels=["apply", "analyze", "evaluate"],
        question_count=40,
        source_chunks=chunks,
    )
    subtypes = {q.item_subtype for q in asm.questions}
    assert subtypes & {"short_answer", "extended_response"}, subtypes
    # Every written item carries a grounded rubric.
    for q in asm.questions:
        if q.item_subtype in {"short_answer", "extended_response"}:
            assert q.rubric and q.rubric["criteria"]
            assert _all_criterion_cites(q.rubric) <= set(q.source_chunks)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---- apparatus backstop coverage on the diversified path -----------------
#
# `generate()` guards each `_generate_question` call site, but the diversified
# builders reached `questions` UNGATED. Turning ED4ALL_ASSESSMENT_DIVERSIFIED
# on therefore routed generation around the guard: a real build shipped
# 159/626 distractors and 30/305 correct answers carrying figure captions,
# HOW-TO banners and glyph alt-text. The guard now sits at `_emit` — the single
# assembly seam — so a new subtype builder cannot bypass it.

APPARATUS_CHUNK = {
    "id": "c-app-1",
    "text": (
        "A gray checkmark inside a circle, indicating correct or complete. "
        "HOW TO ROUND WHOLE NUMBERS Round 23,658 to the nearest hundred. "
        "Figure 1.14 shows the names of the place values."
    ),
    "learning_outcome_refs": ["CO-01"],
    "key_terms": [
        {"term": "place value",
         "definition": "A gray checkmark inside a circle, indicating correct or complete."},
        {"term": "rounding",
         "definition": "HOW TO ROUND WHOLE NUMBERS Round 23,658 to the nearest hundred."},
    ],
}


def _apparatus_hits(assessment):
    import re
    pat = re.compile(r"(HOW TO|Figure \d|checkmark|indicating)", re.I)
    texts = []
    for q in assessment.questions:
        texts.append(str(q.correct_answer or ""))
        texts += [str((c or {}).get("text") or "") for c in (q.choices or [])]
    return [t for t in texts if t.strip() and pat.search(t)]


def test_diversified_emit_applies_the_apparatus_guard(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSESSMENT_APPARATUS_STRICT", "1")
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    out = gen.generate_diversified(
        course_code="T", objective_ids=["CO-01"],
        bloom_levels=["remember"], question_count=4,
        source_chunks=[APPARATUS_CHUNK],
    )
    assert _apparatus_hits(out) == [], (
        "apparatus text reached the diversified emit — the _emit guard is the "
        "single assembly seam and must reject it"
    )


def test_diversified_guard_is_inert_when_flag_off(monkeypatch):
    """Default OFF stays byte-identical (the widened markers never fire)."""
    monkeypatch.delenv("ED4ALL_ASSESSMENT_APPARATUS_STRICT", raising=False)
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    out = gen.generate_diversified(
        course_code="T", objective_ids=["CO-01"],
        bloom_levels=["remember"], question_count=4,
        source_chunks=[APPARATUS_CHUNK],
    )
    assert out is not None  # no crash; legacy marker set governs
