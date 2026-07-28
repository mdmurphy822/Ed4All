"""Regressions for objective-index chunk fan-out during pair synthesis."""

from __future__ import annotations

import pytest

from Trainforge.generators._synthesis_provider import SynthesisProvider
from Trainforge.synthesize_training import (
    _checkpoint_pair_matches_focus,
    _focus_chunk_on_objective,
)
from Trainforge.synthesis_eligibility import pair_eligibility
from lib.validators.pair.claim_support import (
    PairClaimSupportValidator,
    _decompose_sentences,
    _localized_source_premises,
    _normalize_nli_text,
)
from lib.classifiers.nli_classifier import NliScore


@pytest.fixture(autouse=True)
def _enable_staged_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    """These regressions exercise the opt-in staged-v4 focus contract."""
    monkeypatch.setenv("TRAINFORGE_STAGED_SYNTHESIS_V4", "true")


def _objective_index_chunk() -> dict:
    return {
        "id": "course_chunk_00002",
        "text": (
            "Supporting chapter objectives (2): "
            "CO-01 — Identify the place value of each digit in a whole number. "
            "Bloom: remember Chapter Week 1 "
            "CO-02 — Write a whole number in standard form using digits. "
            "Bloom: apply Chapter Week 1"
        ),
        "learning_outcome_refs": ["to-01", "co-01", "co-02", "co-99"],
        "bloom_level": "understand",
    }


def _objectives() -> dict:
    return {
        "co-01": {
            "statement": "Identify the place value of each digit in a whole number.",
            "bloom_level": "remember",
        },
        "co-02": {
            "statement": "Write a whole number in standard form using digits.",
            "bloom_level": "apply",
        },
        "co-03": {
            "statement": "Compare fractions using a common denominator.",
            "bloom_level": "analyze",
        },
        "co-10": {
            "statement": "Add fractions by rewriting them with a common denominator.",
            "bloom_level": "apply",
        },
        "co-30": {
            "statement": "Define absolute value as distance from zero.",
            "bloom_level": "remember",
        },
        "co-31": {
            "statement": "Calculate absolute value for a given integer.",
            "bloom_level": "apply",
        },
    }


def test_objective_index_chunk_is_focused_on_one_source_derived_objective() -> None:
    chunk = _objective_index_chunk()

    focused = _focus_chunk_on_objective(
        chunk, seed=17, objectives=_objectives(),
    )

    assert focused is not chunk
    assert len(focused["learning_outcome_refs"]) == 1
    focus = focused["synthesis_focus_objective"]
    assert focused["learning_outcome_refs"] == [focus["id"]]
    assert focus["id"] in {"co-01", "co-02"}
    assert focus["statement"] in chunk["text"]
    assert focused["bloom_level"] == focus["bloom_level"]
    assert focused["text"] == chunk["text"]
    # The authoritative source chunk remains untouched.
    assert len(chunk["learning_outcome_refs"]) == 4


def test_objective_focus_is_stable_for_resume_seed() -> None:
    chunk = _objective_index_chunk()
    first = _focus_chunk_on_objective(
        chunk, seed=123, objectives=_objectives(),
    )
    second = _focus_chunk_on_objective(
        chunk, seed=123, objectives=_objectives(),
    )

    assert first["synthesis_focus_objective"] == second[
        "synthesis_focus_objective"
    ]


def test_regular_content_chunk_is_canonicalized() -> None:
    chunk = {
        "id": "ordinary",
        "text": (
            "A worked example adds fractions after rewriting them with a "
            "common denominator."
        ),
        "learning_outcome_refs": ["co-10"],
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=1, objectives=_objectives(),
    )
    assert focused["learning_outcome_refs"] == ["co-10"]
    assert focused["bloom_level"] == "apply"
    assert focused["synthesis_focus_objective"]["statement"].startswith(
        "Add fractions"
    )


def test_canonical_focus_preserves_bloom_verb_and_abcd_behavior() -> None:
    chunk = {
        "id": "transaction",
        "text": "A failed transaction rolls back atomic and partial updates.",
        "learning_outcome_refs": ["co-77"],
    }
    objectives = {
        "co-77": {
            "statement": "Differentiate atomic and partial updates.",
            "bloom_level": "analyze",
            "bloom_verb": "differentiate",
            "behavior": {
                "action_object": "atomic and partial updates",
                "condition": "given a failed transaction",
                "degree": "without omitting rollback effects",
            },
        },
    }
    focus = _focus_chunk_on_objective(
        chunk, seed=1, objectives=objectives,
    )["synthesis_focus_objective"]
    assert focus["bloom_verb"] == "differentiate"
    assert focus["action_object"] == "atomic and partial updates"
    assert focus["condition"] == "given a failed transaction"
    assert focus["degree"] == "without omitting rollback effects"
    assert focus["behavior"] == {
        "verb": "differentiate",
        "action_object": "atomic and partial updates",
        "condition": "given a failed transaction",
        "degree": "without omitting rollback effects",
    }


def test_canonical_focus_is_legacy_identical_without_abcd_fields() -> None:
    focused = _focus_chunk_on_objective(
        {
            "id": "legacy",
            "text": "A worked example adds fractions with a denominator.",
            "learning_outcome_refs": ["co-10"],
        },
        seed=1,
        objectives=_objectives(),
    )
    assert focused["synthesis_focus_objective"] == {
        "id": "co-10",
        "statement": (
            "Add fractions by rewriting them with a common denominator."
        ),
        "bloom_level": "apply",
    }


def test_regular_multi_ref_chunk_chooses_semantically_aligned_ref() -> None:
    chunk = {
        "id": "worked-example",
        "text": (
            "A worked example adds fractions after rewriting them with a "
            "common denominator."
        ),
        "learning_outcome_refs": ["co-10", "co-03"],
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=1, objectives=_objectives(),
    )

    assert focused["learning_outcome_refs"] == ["co-10"]
    assert chunk["learning_outcome_refs"] == ["co-10", "co-03"]


def test_broad_mapping_chunk_without_statements_is_not_trainable() -> None:
    chunk = {
        "id": "mapping",
        "text": "Chapter Objective | Chapter | Terminal Objective",
        "learning_outcome_refs": [f"co-{i:02d}" for i in range(1, 40)],
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=1, objectives=_objectives(),
    )

    assert focused["learning_outcome_refs"] == []
    assert focused["synthesis_focus_skip_reason"] == (
        "broad_objective_index_not_instructional_content"
    )


def test_objective_list_without_inline_bloom_is_focused() -> None:
    chunk = {
        "id": "list",
        "text": (
            "CO-30 — Define absolute value as distance from zero. "
            "CO-31 — Calculate absolute value for a given integer."
        ),
        "learning_outcome_refs": ["co-30", "co-31"],
        "bloom_level": "apply",
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=4, objectives=_objectives(),
    )

    assert focused["learning_outcome_refs"][0] in {"co-30", "co-31"}
    assert focused["synthesis_focus_objective"]["statement"] in chunk["text"]


def test_leading_objective_statement_becomes_provider_focus() -> None:
    chunk = {
        "id": "content",
        "text": (
            "CO-01: Identify the place value of each digit in a whole number.\n"
            "Key Idea: A digit's position determines its value."
        ),
        "learning_outcome_refs": ["co-01"],
        "bloom_level": "remember",
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=2, objectives=_objectives(),
    )

    assert focused["learning_outcome_refs"] == ["co-01"]
    assert focused["synthesis_focus_objective"] == {
        "id": "co-01",
        "statement": "Identify the place value of each digit in a whole number.",
        "bloom_level": "remember",
    }


def test_stale_broad_checkpoint_pair_is_not_replayed_after_focus() -> None:
    focused = _focus_chunk_on_objective(
        _objective_index_chunk(), seed=17, objectives=_objectives(),
    )


def test_canonical_bloom_overrides_contradictory_chunk_metadata() -> None:
    chunk = {
        "id": "parabola-direction",
        "text": (
            "CO-40: Identify whether a parabola opens upward or downward.\n"
            "The sign of coefficient a determines its direction."
        ),
        "learning_outcome_refs": ["co-40"],
        "bloom_level": "remember",
        "chunk_type": "explanation",
    }
    objectives = {
        "co-40": {
            "statement": "Explain how coefficient a determines whether a parabola opens upward or downward.",
            "bloom_level": "understand",
        },
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=7, objectives=objectives,
    )

    assert focused["bloom_level"] == "understand"
    assert focused["synthesis_original_bloom_level"] == "remember"
    assert focused["synthesis_focus_objective"]["bloom_level"] == "understand"


def test_unrelated_first_ref_does_not_override_aligned_ref() -> None:
    chunk = {
        "id": "graph-example",
        "text": (
            "Graph a parabola by plotting its vertex, axis of symmetry, "
            "intercepts, and the reflected point."
        ),
        "learning_outcome_refs": ["co-10", "co-20"],
        "chunk_type": "example",
    }
    objectives = {
        "co-10": {
            "statement": "Solve a quadratic equation by completing the square.",
            "bloom_level": "apply",
        },
        "co-20": {
            "statement": "Graph a parabola using its vertex, axis of symmetry, and intercepts.",
            "bloom_level": "apply",
        },
    }

    focused = _focus_chunk_on_objective(
        chunk, seed=3, objectives=objectives,
    )

    assert focused["learning_outcome_refs"] == ["co-20"]


def test_malformed_assessment_is_ineligible_for_both_pair_kinds() -> None:
    focused = {
        "id": "assessment-fragment",
        "text": (
            "Which definition best matches the term Because of this, it ? "
            "one unrelated phrase another unrelated phrase"
        ),
        "chunk_type": "assessment_item",
        "learning_outcome_refs": ["co-01"],
        "synthesis_focus_objective": {
            "id": "co-01",
            "statement": "Identify place value.",
            "bloom_level": "remember",
        },
    }

    assert pair_eligibility(
        focused, kind="instruction",
    ).reason == "malformed_assessment_item"
    assert pair_eligibility(
        focused, kind="preference",
    ).reason == "malformed_assessment_item"


def test_dpo_requires_misconception_affordance_beyond_sft_evidence() -> None:
    focused = {
        "id": "plain-summary",
        "text": (
            "A polynomial contains terms joined by addition. Its degree is "
            "determined by the greatest exponent present in those terms."
        ),
        "chunk_type": "summary",
        "learning_outcome_refs": ["co-01"],
        "synthesis_focus_objective": {
            "id": "co-01",
            "statement": "Describe polynomial degree.",
            "bloom_level": "understand",
        },
    }

    assert pair_eligibility(focused, kind="instruction").eligible
    assert pair_eligibility(
        focused, kind="preference",
    ).reason == "preference_misconception_candidate_missing"
    stale = {
        "lo_refs": ["to-01", "co-01", "co-02", "co-99"],
        "prompt": "old broad pair",
    }

    assert not _checkpoint_pair_matches_focus(stale, focused)
    assert _checkpoint_pair_matches_focus(
        {"lo_refs": focused["learning_outcome_refs"]},
        focused,
    )


def test_provider_prompts_pin_factual_focus_without_changing_json_shape() -> None:
    focus = {
        "id": "co-02",
        "statement": "Write a whole number in standard form using digits.",
        "bloom_level": "apply",
    }
    provider = object.__new__(SynthesisProvider)
    provider._local_user_directives = True

    instruction = provider._render_instruction_user(
        {
            "prompt": "Explain the objective.",
            "completion": "A grounded answer.",
            "bloom_level": "apply",
            "content_type": "explanation",
            "template_id": "apply._default",
        },
        "chunk_2",
        focus=focus,
    )
    preference = provider._render_preference_user(
        {
            "prompt": "Explain the objective.",
            "chosen": "A grounded answer.",
            "rejected": "An incorrect answer.",
            "rejected_source": "rule_synthesized",
        },
        "chunk_2",
        focus=focus,
    )

    assert "teach only co-02" in instruction
    assert focus["statement"] in instruction
    assert "Every factual claim in the completion" in instruction
    assert "teach only co-02" in preference
    assert "Every factual claim in the chosen completion" in preference
    assert '{"prompt": "<paraphrased prompt>", "completion":' in instruction
    assert '"chosen": "<paraphrased chosen>"' in preference


def test_claim_decomposition_splits_independent_semicolon_clauses() -> None:
    claims = _decompose_sentences(
        "Place value depends on position in a number; "
        "a common mistake is confusing face value with place value."
    )

    assert claims == [
        "Place value depends on position in a number;",
        "a common mistake is confusing face value with place value.",
    ]


def test_nli_typography_normalization_preserves_words_and_ascii_quotes() -> None:
    assert _normalize_nli_text(
        "A digit\u2019s \u201cface value\u201d\u00a0\u2014 not its position."
    ) == "A digit's \"face value\" - not its position."


def test_long_source_localization_retrieves_support_beyond_truncation() -> None:
    source = (
        ("Unrelated introductory material. " * 80)
        + "Name each three-digit group as a standalone number. "
        + "Attach the appropriate period name except for the ones period. "
        + ("Unrelated closing material. " * 80)
    )
    claim = (
        "Name each three-digit group as a standalone number, then attach "
        "the appropriate period name except for the ones period."
    )

    windows = _localized_source_premises(source, claim)

    assert windows
    assert "Attach the appropriate period name" in windows[0]


def test_lexical_window_collision_cannot_count_as_claim_support() -> None:
    """Lexical retrieval selects evidence; it never supplies entailment."""

    class _NeutralNli:
        def score_batch(self, *, pairs):
            assert any("interest rate" in premise for premise, _ in pairs)
            return [
                NliScore(entailment=0.08, neutral=0.88, contradiction=0.04)
                for _ in pairs
            ]

    source = (
        ("Introductory material about financial terminology. " * 40)
        + "A bank interest rate describes the price of borrowing money. "
        + ("Closing material about financial terminology. " * 40)
    )
    pair = {
        "chunk_id": "collision",
        "completion": (
            "Student interest rates determine how quickly a learner reads."
        ),
    }

    status, reason, fields = PairClaimSupportValidator(
        nli=_NeutralNli()
    ).validate_pair(
        pair,
        kind="instruction",
        chunk={"id": "collision", "text": source},
    )

    assert status == "rejected"
    assert reason == "unsupported_claim"
    assert fields["claim_support_rate"] == 0.0
