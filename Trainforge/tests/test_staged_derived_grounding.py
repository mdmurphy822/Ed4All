"""Closed-world numeric and symbolic grounding for staged realizations."""
from __future__ import annotations

import json

import pytest

from Trainforge.generators.staged.provider import (
    StagedSynthesisProvider,
    _deterministic_derivation_ledger,
)
from Trainforge.tests.test_staged_synthesis_provider import _Base, _chunk, _sft_plan


MOTION_PROMPT = (
    "Two vehicles leave the same point in opposite directions. "
    "The eastbound vehicle travels at 140 miles per hour and the westbound "
    "vehicle travels at 100 miles per hour. Analyze their separation rate."
)
CLAIM = (
    "For objects moving in opposite directions, total distance is the sum "
    "of their individual distances."
)
FIVE_SENTENCE_PROOF = (
    "The vehicles begin together and move in opposite directions. "
    "The eastbound rate is 140 miles per hour. "
    "The westbound rate is 100 miles per hour. "
    "The combined separation rate is 140 + 100 = 240 miles per hour. "
    "Thus their individual distances add to determine how quickly they separate."
)


class _LexicalNli:
    @staticmethod
    def score_pair(*, premise, hypothesis):
        if (
            "collide" in premise.lower()
            and premise.lower().strip().startswith("therefore")
        ):
            supported = False
        else:
            stop = {"a", "an", "the", "is", "are", "to", "of", "their"}
            premise_words = {
                word for word in premise.lower().replace(".", "").split()
                if word not in stop
            }
            hypothesis_words = {
                word for word in hypothesis.lower().replace(".", "").split()
                if word not in stop
            }
            supported = bool(premise_words & hypothesis_words)
        return type("_Score", (), {
            "entailment": 0.99 if supported else 0.01,
            "contradiction": 0.001,
        })()


def _provider():
    base = _Base([])
    base._plan_nli_scorer = _LexicalNli()
    return StagedSynthesisProvider(base)


def _plan():
    return {
        "supported_claims": [{"claim": CLAIM}],
        "generated_givens": [
            {
                "symbol": "east_rate", "value": "140", "unit": "miles per hour",
                "role": "rate", "synthetic": True, "provenance": "generated",
            },
            {
                "symbol": "west_rate", "value": "100", "unit": "miles per hour",
                "role": "rate", "synthetic": True, "provenance": "generated",
            },
        ],
    }


def test_five_sentence_numeric_proof_is_grounded():
    provider = _provider()
    assert provider._semantic_coverage_error(
        FIVE_SENTENCE_PROOF, _plan(), prompt=MOTION_PROMPT,
    ) is None
    ledger = provider._validation_audit.derivations
    assert ledger[0]["kind"] == "numeric_arithmetic_relation"
    assert ledger[0]["unit_index"] == 4
    assert all(len(ledger[0][key]) == 64 for key in (
        "premise_sha256", "clause_span_sha256",
        "relation_signature_sha256", "result_sha256",
    ))
    assert not any("140 + 100" in str(value) for value in ledger[0].values())


def test_wrong_arithmetic_is_rejected_exactly():
    output = FIVE_SENTENCE_PROOF.replace("140 + 100 = 240", "140 + 100 = 250")
    _ledger, error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT, output=output, claims=[CLAIM],
    )
    assert "has false arithmetic relation" in error


def test_hallucinated_extra_number_is_rejected():
    output = FIVE_SENTENCE_PROOF + " A third vehicle travels at 75 miles per hour."
    _ledger, error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT, output=output, claims=[CLAIM],
    )
    assert "introduces ungrounded numeric literal(s): ['75']" in error


def test_unsupported_prose_conclusion_is_rejected():
    output = FIVE_SENTENCE_PROOF + " Therefore the vehicles will collide."
    error = _provider()._semantic_coverage_error(
        output, _plan(), prompt=MOTION_PROMPT,
    )
    assert "will collide" in error


def test_false_arithmetic_rejects_even_when_result_is_seeded():
    _ledger, error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT + " A dashboard also displays 250.",
        output="The displayed calculation is 140 + 100 = 250.",
        claims=[CLAIM],
    )
    assert error == "output clause 0 has false arithmetic relation"


def test_valid_numeric_clause_cannot_bless_unsupported_symbolic_conjunction():
    _ledger, error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT,
        output="The rate calculation is 140 + 100 = 240 and x = y.",
        claims=[CLAIM],
    )
    assert error == "output clause 1 introduces an ungrounded symbolic equation"


def test_valid_numeric_and_supported_symbolic_clauses_are_both_recorded():
    ledger, error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT + " Let x = y.",
        output="The rate calculation is 140 + 100 = 240, and y = x.",
        claims=[CLAIM],
    )
    assert error is None
    assert [step["kind"] for step in ledger] == [
        "numeric_arithmetic_relation", "symbolic_equivalence",
    ]
    assert ledger[0]["unit_index"] == 0
    assert ledger[1]["unit_index"] == 1


def test_reused_derived_number_does_not_ground_an_unrelated_conclusion():
    output = FIVE_SENTENCE_PROOF + " Therefore the account balance is 240 dollars."
    error = _provider()._semantic_coverage_error(
        output, _plan(), prompt=MOTION_PROMPT,
    )
    assert "account balance" in error


def test_exact_inequality_is_validated_and_false_inequality_is_rejected():
    ledger, error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT,
        output="The combined rate satisfies 140 + 100 >= 240.",
        claims=[CLAIM],
    )
    assert error is None
    assert ledger[0]["kind"] == "numeric_arithmetic_relation"
    _ledger, false_error = _deterministic_derivation_ledger(
        prompt=MOTION_PROMPT,
        output="The combined rate satisfies 140 + 100 < 240.",
        claims=[CLAIM],
    )
    assert false_error == "output clause 0 has false arithmetic relation"


def test_rounding_is_not_accepted_as_exact_arithmetic():
    _ledger, error = _deterministic_derivation_ledger(
        prompt="A quantity is divided into 3 equal parts from a total of 140.",
        output="The exact quotient is 140 / 3 = 46.67 units.",
        claims=["Division determines the size of each equal part."],
    )
    assert error == "output clause 0 has false arithmetic relation"


class _OrderedClaimNli:
    @staticmethod
    def score_pair(*, premise, hypothesis):
        premise_l = premise.lower()
        hypothesis_l = hypothesis.lower()
        contradiction = (
            ("claim alpha" in hypothesis_l and "not alpha" in premise_l)
            or ("claim beta" in hypothesis_l and "not beta" in premise_l)
        )
        if "claim alpha" in hypothesis_l:
            entailed = "alpha" in premise_l
        elif "claim beta" in hypothesis_l:
            entailed = "beta" in premise_l
        else:
            entailed = "unsupported" not in premise_l
        return type("_Score", (), {
            "entailment": 0.01 if contradiction else (0.99 if entailed else 0.01),
            "contradiction": 0.99 if contradiction else 0.001,
        })()


def _ordered_provider():
    base = _Base([])
    base._plan_nli_scorer = _OrderedClaimNli()
    return StagedSynthesisProvider(base)


def _ordered_plan():
    return {
        "supported_claims": [
            {"claim": "claim alpha"},
            {"claim": "claim beta"},
        ],
        "generated_givens": [],
    }


def test_full_response_and_monotonic_adjacent_claim_spans_pass():
    assert _ordered_provider()._semantic_coverage_error(
        "Alpha is explained in one clause, and alpha is completed here. "
        "Beta follows in the final clause.",
        _ordered_plan(),
        prompt="Explain alpha before beta.",
    ) is None


@pytest.mark.parametrize("output", [
    "Alpha is explained without the other claim.",
    "Beta is explained first. Alpha is explained second.",
    "Alpha is explained. Not beta is the conclusion.",
    "Alpha is explained and unsupported folklore is appended. Beta is explained.",
])
def test_partial_reordered_contradictory_or_unsupported_neighbors_fail(output):
    assert _ordered_provider()._semantic_coverage_error(
        output, _ordered_plan(), prompt="Explain alpha before beta.",
    ) is not None


def _plan_with_given(*, provenance="generated"):
    plan = json.loads(_sft_plan())
    plan["generated_givens"] = [{
        "symbol": "v",
        "value": "7",
        "unit": "km/h",
        "role": "rate",
        "synthetic": True,
        "provenance": provenance,
    }]
    return json.dumps(plan)


def test_typed_synthetic_given_is_shared_with_prompt_and_ledger(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768")
    base = _Base([
        _plan_with_given(),
        json.dumps({
            "prompt": (
                "Analyze transaction atomicity using synthetic rate v = 7 km/h."
            ),
            "completion": (
                "Atomicity prevents partial updates; v remains 7 km/h."
            ),
            "covered_claim_indices": [0],
        }),
    ])
    result = StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert result["completion"].endswith("v remains 7 km/h.")
    assert "generated_givens" in base.schemas[0]["required"]
    given_schema = base.schemas[0]["properties"]["generated_givens"]["items"]
    assert given_schema["properties"]["synthetic"]["const"] is True
    assert given_schema["properties"]["provenance"]["const"] == "generated"


def test_generated_given_rejects_unapproved_provenance(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768")
    base = _Base([_plan_with_given(provenance="source")] * 3)
    with pytest.raises(Exception) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert "typed synthetic provenance contract" in str(caught.value)


def test_prompt_number_without_source_or_typed_given_is_rejected(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768")
    realization = json.dumps({
        "prompt": "Analyze transaction atomicity using an invented value of 7.",
        "completion": "Atomicity prevents partial updates at value 7.",
        "covered_claim_indices": [0],
    })
    base = _Base([_sft_plan(), realization, realization, realization])
    with pytest.raises(Exception) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert "ungrounded numeric literal" in str(caught.value)
