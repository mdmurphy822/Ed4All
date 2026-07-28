"""Generic adjudication regressions for staged synthesis."""
from __future__ import annotations

import json

from Trainforge.generators.staged_synthesis_provider import (
    StagedSynthesisProvider,
    _coverage_units,
    _explicit_relation_conflict,
    _first_distinctive_verbatim_span,
    _relation_preserving_support,
    _scenario_numeric_literals,
)
from Trainforge.tests.test_staged_synthesis_provider import (
    _Base,
    _chunk,
    _sft_plan,
)


def test_only_exponent_position_numerals_are_structural():
    assert _scenario_numeric_literals("Factor 5x² + 3x + c.") == {"5", "3"}
    assert _scenario_numeric_literals("Use x^2 and y³.") == set()
    assert _scenario_numeric_literals("A container holds 17 liters.") == {"17"}


def test_numeric_closed_world_covers_coefficients_units_subscripts_and_rhs():
    # Zero is a value under the closed-world policy, including on an equation
    # RHS; it therefore needs the same provenance as every other constant.
    assert _scenario_numeric_literals("Solve 19x^2+x=0.") == {"19", "0"}
    assert _scenario_numeric_literals("Use 23m distance.") == {"23"}
    assert _scenario_numeric_literals("Carry 5 kg.") == {"5"}
    assert _scenario_numeric_literals("Set x=41.") == {"41"}
    assert _scenario_numeric_literals("Compare x₂ with x.") == {"2"}


def test_unicode_multidigit_and_negative_exponents_are_structural():
    assert _scenario_numeric_literals("Use x¹² and y⁻³.") == set()
    assert _scenario_numeric_literals("Use x^12 and y^-3.") == set()


def test_source_backed_constant_remains_a_closed_world_value():
    source_values = _scenario_numeric_literals("The threshold is 12 units.")
    assert _scenario_numeric_literals("Use the 12-unit threshold.") <= source_values
    assert not _scenario_numeric_literals("Use a 19-unit threshold.") <= source_values


def test_relation_signature_accepts_faithful_form_and_rejects_swaps():
    premise = "Multiply alpha by beta when gamma is positive."
    assert _relation_preserving_support(
        premise, "Multiplication alpha by beta when gamma is positive.",
    )
    assert not _relation_preserving_support(
        premise, "Multiply beta by alpha when gamma is positive.",
    )


def test_contradiction_requires_explicit_polarity_or_direction_conflict():
    premise = "Multiply alpha by beta."
    assert _explicit_relation_conflict(premise, "Do not multiply alpha by beta.")
    assert _explicit_relation_conflict(premise, "Multiply beta by alpha.")
    assert not _explicit_relation_conflict(
        premise, "Discuss alpha without giving a multiplication relation.",
    )


def test_canonical_contract_exemption_is_exact_and_narrative_stays_blocked():
    canonical = (
        "Analyze the relationship between symbolic factors under the stated "
        "condition."
    )
    narrative = (
        "A distinctive account then follows one unusual sequence whose words "
        "belong only to this source."
    )
    assert _first_distinctive_verbatim_span(
        canonical, canonical, canonical_contract_text=[canonical],
    ) is None
    assert _first_distinctive_verbatim_span(
        f"{canonical} {narrative}",
        f"{canonical} {narrative}",
        canonical_contract_text=[canonical],
    ) is not None


def test_prompt_contract_does_not_require_answer_bearing_relation():
    class _Base:
        _model = "test"
        _provider_name = "test"

    provider = StagedSynthesisProvider(_Base())
    focus = {
        "bloom_verb": "analyze",
        "action_object": "factor pairs satisfying product and sum conditions",
        "conditions": ["given a symbolic trinomial"],
        "content_obligations": [
            "factor pairs satisfying product and sum conditions",
        ],
        "content_obligation_anchors": [],
    }
    assert provider._objective_contract_error(
        "Given a symbolic trinomial, analyze its factor pairs.",
        focus,
        answer_bearing=False,
    ) is None


def test_current_plan_failures_are_aggregated_into_one_repair(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768")
    invalid = json.loads(_sft_plan())
    invalid["objective_id"] = "wrong-objective"
    invalid["bloom_level"] = "create"
    corrected = json.loads(_sft_plan())
    realization = {
        "prompt": "Analyze how atomicity prevents partial updates.",
        "completion": "Atomicity prevents partially applied updates.",
        "covered_claim_indices": [0],
    }
    base = _Base([
        json.dumps(invalid), json.dumps(corrected), json.dumps(realization),
    ])
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    repair = base.prompts[1][-1]["content"]
    assert "objective_id differs from canonical objective" in repair
    assert "bloom_level differs from canonical Bloom" in repair
    assert len(base.prompts) == 3


def test_omission_is_not_relabelled_as_contradiction():
    premise = "Divide the quantity by the nonzero scale factor."
    omission = "Discuss the quantity and report an observation."
    assert not _explicit_relation_conflict(premise, omission)


def test_introductory_subordinate_clause_stays_with_governing_clause():
    assert _coverage_units(
        "When objects move in opposite directions, their separation increases."
    ) == [
        "When objects move in opposite directions, their separation increases."
    ]
    assert _coverage_units(
        "When objects move apart, separation increases, but speed can differ."
    ) == [
        "When objects move apart, separation increases",
        "speed can differ.",
    ]


def test_independent_coordinate_clauses_remain_separate():
    assert _coverage_units(
        "The product matches the constant, and the sum matches the coefficient."
    ) == [
        "The product matches the constant",
        "the sum matches the coefficient.",
    ]


def test_prompt_bloom_obligation_accepts_semantic_paraphrase_and_rejects_omission():
    class Score:
        def __init__(self, entailment, contradiction):
            self.entailment = entailment
            self.contradiction = contradiction

    class Scorer:
        def score_pair(self, *, premise, hypothesis):
            if "examine" in premise and "analyze" in hypothesis:
                return Score(0.91, 0.01)
            return Score(0.05, 0.02)

    base = type("_SemanticBase", (), {
        "_model": "test",
        "_provider_name": "test",
        "_plan_nli_scorer": Scorer(),
    })()
    provider = StagedSynthesisProvider(base)
    focus = {
        "bloom_verb": "analyze",
        "action_object": "transaction rollback behavior",
        "conditions": [],
    }
    assert provider._objective_contract_error(
        "Examine transaction rollback behavior and explain its effect.",
        focus,
        answer_bearing=False,
    ) is None
    assert provider._objective_contract_error(
        "Write a note about transaction rollback behavior.",
        focus,
        answer_bearing=False,
    ) == "learner task omits canonical bloom_verb obligation: 'analyze'"
