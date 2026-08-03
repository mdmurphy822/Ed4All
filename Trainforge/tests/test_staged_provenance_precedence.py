"""Deterministic provenance precedence and objective relation integrity."""
from __future__ import annotations

import pytest

from Trainforge.generators.staged.provider import (
    StagedSynthesisProvider,
    _claim_repair_diff,
    _claim_repair_guard,
    _coverage_units,
)
from Trainforge.generators.staged.window_contract import objective_card
from Trainforge.tests.test_staged_synthesis_provider import _Base


class _RejectEverythingNli:
    @staticmethod
    def score_pair(*, premise, hypothesis):
        return type("_Score", (), {
            "entailment": 0.0,
            "contradiction": 1.0,
        })()


class _AcceptEverythingNli:
    @staticmethod
    def score_pair(*, premise, hypothesis):
        return type("_Score", (), {
            "entailment": 1.0,
            "contradiction": 0.0,
        })()


def _provider():
    base = _Base([])
    base._plan_nli_scorer = _RejectEverythingNli()
    return StagedSynthesisProvider(base)


@pytest.mark.parametrize(
    "claim",
    [
        "The product of the selected integers equals the constant term.",
        "A proportional relationship has a constant ratio.",
        "Parallel lines have equal slopes.",
        "A valid factor pair has the required sum.",
    ],
)
def test_accepted_atomic_claim_exact_rendering_grounds_before_nli(claim):
    provider = _provider()
    plan = {
        "supported_claims": [{
            "claim": claim,
            "evidence_quote": claim,
        }],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(claim, plan) is None
    row = provider._validation_audit.deterministic_provenance[0]
    assert row["basis"] == "accepted_plan_claim"
    assert all(len(row[key]) == 64 for key in (
        "claim_sha256", "evidence_quote_sha256", "unit_sha256",
    ))


def test_independently_accepted_exact_quote_grounds_before_nli():
    provider = _provider()
    quote = "A constant ratio characterizes every proportional relationship."
    plan = {
        "supported_claims": [{
            "claim": "Proportional relationships have a constant ratio.",
            "evidence_quote": quote,
        }],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(quote, plan) is None
    assert provider._validation_audit.deterministic_provenance[0]["basis"] == (
        "exact_evidence_quote"
    )


@pytest.mark.parametrize(
    "output",
    [
        "The product is not the selected integers' constant term.",
        "The selected integers have a product-like label.",
        "The product of the selected integers does not equal the constant term.",
    ],
)
def test_substrings_and_reversed_polarity_do_not_receive_provenance(output):
    provider = _provider()
    claim = "The product of the selected integers equals the constant term."
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(output, plan) is not None
    assert provider._validation_audit.deterministic_full_claim_indices == []


@pytest.mark.parametrize(
    "output",
    [
        "It is not true that parallel lines have equal slopes.",
        "It cannot be established that parallel lines have equal slopes.",
        "Contrary to the evidence, parallel lines have equal slopes.",
        "Parallel lines have equal slopes or the Moon is cheese.",
        "Parallel lines have equal slopes while the Moon is cheese.",
        "Parallel lines have equal slopes although the Moon is cheese.",
        "Parallel lines have equal slopes, which proves the Moon is cheese.",
        "Parallel lines have equal slopes plus the Moon is cheese.",
        "Parallel lines don't have equal slopes.",
        "Parallel lines aren't known to have equal slopes.",
    ],
)
def test_whole_scope_negation_and_extra_clauses_never_receive_provenance(output):
    provider = _provider()
    claim = "Parallel lines have equal slopes."
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(output, plan) is not None
    assert provider._validation_audit.deterministic_full_claim_indices == []


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("Reportedly ", ""),
        ("Moon cheese; ", ""),
        ("", " in every universe"),
        ("", " and the Moon is cheese"),
        ("Unverified: ", " according to nobody"),
    ],
)
def test_arbitrary_prefix_or_suffix_cannot_create_deterministic_full(prefix, suffix):
    provider = _provider()
    claim = "A proportional relationship has a constant ratio."
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(f"{prefix}{claim}{suffix}", plan) is not None
    assert provider._validation_audit.deterministic_full_claim_indices == []


@pytest.mark.parametrize(
    "output",
    [
        "Parallel lines have equal slopes.",
        "  PARALLEL lines have equal slopes ! ",
        "The parallel lines have the equal slopes.",
        "Ｔｈｅ parallel lines have equal slopes.",
    ],
)
def test_only_safe_normalized_whole_proposition_variants_are_deterministic(output):
    provider = _provider()
    claim = "Parallel lines have equal slopes."
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(output, plan) is None
    assert provider._validation_audit.deterministic_provenance
    assert provider._validation_audit.deterministic_full_claim_indices == [0]


@pytest.mark.parametrize(
    "claim",
    [
        "The selected pair has the required product, and its sum matches the coefficient.",
        "A constant ratio is preserved and corresponding values remain proportional.",
        "Parallel lines have equal slopes, which keeps their direction unchanged.",
        "The first relation fixes the product, and the second relation fixes the sum.",
    ],
)
def test_exact_multiclause_full_output_inherits_one_root_proposition(claim):
    provider = _provider()
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(claim, plan) is None
    rows = provider._validation_audit.deterministic_provenance
    assert len(rows) == len(_coverage_units(claim))
    assert provider._validation_audit.deterministic_full_claim_indices == [0]
    assert len({row["root_proposition_sha256"] for row in rows}) == 1
    assert all(
        row["clause_span_sha256"] == row["unit_sha256"] for row in rows
    )
    assert all(
        row["basis"].endswith("_full_output_inheritance") for row in rows
    )


@pytest.mark.parametrize(
    "claim",
    [
        "The selected pair has the required product, and its sum matches the coefficient.",
        "A constant ratio is preserved and corresponding values remain proportional.",
        "Parallel lines have equal slopes, which keeps their direction unchanged.",
        "The first relation fixes the product, and the second relation fixes the sum.",
    ],
)
@pytest.mark.parametrize("affix", ["Unsupported premise: {}", "{} and an unsupported result"])
def test_multiclause_inheritance_rejects_any_unsupported_affix(claim, affix):
    provider = _provider()
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(affix.format(claim), plan) is not None
    assert provider._validation_audit.deterministic_full_claim_indices == []
    assert not any(
        row["basis"].endswith("_full_output_inheritance")
        for row in provider._validation_audit.deterministic_provenance
    )


def test_paraphrase_routes_to_nli_without_deterministic_provenance():
    base = _Base([])
    base._plan_nli_scorer = _AcceptEverythingNli()
    provider = StagedSynthesisProvider(base)
    claim = "Parallel lines have equal slopes."
    plan = {
        "supported_claims": [{"claim": claim, "evidence_quote": claim}],
        "generated_givens": [],
    }

    assert provider._semantic_coverage_error(
        "The slopes of parallel lines are equal.", plan,
    ) is None
    assert provider._validation_audit.deterministic_provenance == []
    assert provider._validation_audit.deterministic_full_claim_indices == []
    assert any(
        row["decision_type"] == "semantic_full_response_claim"
        for row in provider._validation_audit.nli
    )


def _objective():
    return objective_card({
        "id": "objective-synthetic",
        "statement": "Select an integer pair using two relations.",
        "bloom_level": "apply",
        "bloom_verb": "select",
        "action_object": (
            "product equals constant term; sum equals linear coefficient; "
            "factor-pair selection"
        ),
    })


@pytest.mark.parametrize(
    "task",
    [
        (
            "Select the appropriate integer pair whose product equals the "
            "given constant term and whose sum equals the stated linear "
            "coefficient."
        ),
        (
            "Select a suitable pair: its product must be the positive constant "
            "term; its sum must be the supplied linear coefficient."
        ),
    ],
)
def test_objective_relations_allow_articles_and_modifiers(task):
    assert _provider()._objective_contract_error(task, _objective()) is None


@pytest.mark.parametrize(
    "task,reason",
    [
        (
            "Select a pair whose sum is the linear coefficient.",
            "omits canonical content_obligation anchor",
        ),
        (
            "Select a pair whose sum is the linear coefficient and whose "
            "product is the constant term.",
            "reorders canonical content_obligation relations",
        ),
        (
            "Choose a pair whose product is the linear coefficient and whose "
            "sum is the constant term.",
            "changes canonical content_obligation relation target",
        ),
        (
            "Describe a pair whose product is the constant term and whose sum "
            "is the linear coefficient.",
            "omits canonical bloom_verb",
        ),
    ],
)
def test_objective_relations_fail_closed(task, reason):
    assert reason in _provider()._objective_contract_error(task, _objective())


@pytest.mark.parametrize("left,right", [("product", "sum"), ("sum", "product")])
def test_relation_order_property_accepts_only_canonical_permutation(left, right):
    task = (
        f"Select a pair whose {left} is the "
        f"{'constant term' if left == 'product' else 'linear coefficient'} "
        f"and whose {right} is the "
        f"{'constant term' if right == 'product' else 'linear coefficient'}."
    )
    error = _provider()._objective_contract_error(task, _objective())
    assert (error is None) == (left == "product")


def test_multi_claim_repair_uses_stable_ids_and_freezes_valid_claim():
    valid = {"claim": "A valid relation holds.", "evidence_quote": "evidence"}
    plan = {
        "supported_claims": [
            {"claim": "duplicate", "evidence_quote": "wrong"},
            valid,
            {"claim": "duplicate", "evidence_quote": "wrong"},
        ],
        "learner_task": "Analyze the relation.",
        "generated_givens": [],
    }
    guard = _claim_repair_guard(plan, [0, 2])
    repaired = {
        **plan,
        "supported_claims": [
            {"claim": "replacement one", "evidence_quote": "evidence one"},
            valid,
            {"claim": "replacement two", "evidence_quote": "evidence two"},
        ],
    }
    assert len(set(guard["invalid_stable_ids"])) == 1
    assert _claim_repair_diff(repaired, guard) is None
    repaired["supported_claims"][1]["claim"] = "mutated"
    assert _claim_repair_diff(repaired, guard)["reason"]


def test_json_pointer_permissions_allow_joint_task_and_typed_given_repair():
    plan = {
        "supported_claims": [{
            "claim": "A rate is constant.", "evidence_quote": "A rate is constant.",
        }],
        "learner_task": "Analyze a rate of 9 units.",
        "generated_givens": [],
        "objective_id": "objective-synthetic",
    }
    guard = _claim_repair_guard(
        plan, [],
        editable_pointers={"/learner_task", "/generated_givens"},
    )
    repaired = {
        **plan,
        "learner_task": "Analyze a rate of 8 units.",
        "generated_givens": [{
            "symbol": "rate", "value": "8", "unit": "units",
            "role": "rate", "synthetic": True, "provenance": "generated",
        }],
    }
    assert _claim_repair_diff(repaired, guard) is None
    repaired["objective_id"] = "changed"
    assert _claim_repair_diff(repaired, guard)["reason"]
