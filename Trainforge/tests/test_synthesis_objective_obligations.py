from __future__ import annotations

import pytest

from Trainforge.generators.staged_synthesis_provider import StagedSynthesisProvider
from Trainforge.generators.synthesis_window_contract import (
    build_evidence_window,
    objective_card,
)


def _focus(*, broad: bool = False):
    return {
        "id": "obj-broad" if broad else "obj-specific",
        "statement": "Canonical objective.",
        "bloom_level": "analyze",
        "bloom_verb": "analyze" if broad else "determine",
        "action_object": (
            "linear relationships in two variables"
            if broad
            else "the number of solutions of a system of linear equations"
        ),
        "condition": (
            "given equations, inequalities, graphs, or real-world scenarios"
            if broad
            else "by comparing slopes and y-intercepts"
        ),
        "degree": (
            "to model, graph, and interpret solutions with accuracy and relevance"
            if broad
            else "with 100% accuracy"
        ),
    }


def test_objective_card_types_content_condition_and_performance():
    broad = objective_card(_focus(broad=True))
    assert broad["content_obligations"] == [
        "linear relationships in two variables",
        "model",
        "graph",
        "interpret solutions",
    ]
    assert broad["conditions"] == [
        "given equations, inequalities, graphs, or real-world scenarios",
    ]
    assert broad["performance_criteria"] == ["accuracy and relevance"]

    scored = objective_card(_focus())
    assert scored["performance_criteria"] == ["with 100% accuracy"]
    assert "with 100% accuracy" not in scored["conditions"]


@pytest.mark.parametrize("chunk_id", ["synthetic-case-a", "synthetic-case-b"])
def test_incomplete_broad_objective_evidence_fails_before_generation(chunk_id):
    chunk = {
        "id": chunk_id,
        "text": (
            "A linear equation in two variables can be represented by a "
            "straight line and solved for an ordered pair."
        ),
    }
    with pytest.raises(ValueError, match="every objective content obligation"):
        build_evidence_window(chunk, _focus(broad=True))


def test_corrected_evidence_satisfies_content_not_literal_score():
    focus = _focus()
    chunk = {
        "id": "synthetic-scored",
        "text": (
            "Convert both linear equations to slope-intercept form. Equal "
            "slopes with different y-intercepts give no solution; equal "
            "slopes and equal intercepts give infinitely many solutions; "
            "different slopes give one solution."
        ),
    }
    window = build_evidence_window(chunk, focus)
    assert window["objective"]["performance_criteria"] == ["with 100% accuracy"]

    provider = object.__new__(StagedSynthesisProvider)
    provider._plan_nli = type("_Nli", (), {
        "score_pair": staticmethod(
            lambda **_: type(
                "_Score", (), {"entailment": 0.99, "contradiction": 0.0},
            )()
        ),
    })()
    provider._validation_audit = type(
        "_Audit", (), {"nli": [], "stage": "test", "attempt": 1},
    )()
    task = (
        "Determine how many solutions the system has by comparing its slopes "
        "and y-intercepts."
    )
    assert provider._objective_contract_error(
        task, objective_card(focus),
    ) is None


class _StrictAnchorNli:
    @staticmethod
    def score_pair(*, premise, hypothesis):
        anchor = next(
            (
                item for item in ("product", "sum", "select")
                if item in hypothesis.lower()
            ),
            "",
        )
        supported = anchor and anchor in premise.lower()
        return type("_Score", (), {
            "entailment": 0.99 if supported else 0.01,
            "contradiction": 0.0 if supported else 0.90,
        })()


def _anchor_provider():
    provider = object.__new__(StagedSynthesisProvider)
    provider._plan_nli = _StrictAnchorNli()
    provider._validation_audit = type(
        "_Audit", (), {"nli": [], "stage": "test", "attempt": 1},
    )()
    return provider


def _factor_pair_focus():
    return objective_card({
        "id": "objective-factor-pair",
        "statement": "Factor a monic quadratic using an integer factor pair.",
        "bloom_level": "apply",
        "bloom_verb": "select",
        "action_object": (
            "factor-pair selection: product equals the constant term; "
            "sum equals the linear coefficient"
        ),
    })


def test_telegraphic_factor_pair_contract_normalizes_required_relations():
    focus = _factor_pair_focus()

    assert focus["content_obligation_anchors"] == [{
        "obligation": (
            "factor-pair selection: product equals the constant term; "
            "sum equals the linear coefficient"
        ),
        "anchors": [
            {
                "lemma": "multiply",
                "relation": "product",
                "hypothesis": (
                    "The learner task requires using the product relation."
                ),
            },
            {
                "lemma": "add",
                "relation": "sum",
                "hypothesis": "The learner task requires using the sum relation.",
            },
            {
                "lemma": "select",
                "relation": "selection",
                "hypothesis": (
                    "The learner task requires selecting the correct candidate."
                ),
            },
        ],
    }]
    assert _anchor_provider()._objective_contract_error(
        (
            "Select the integer pair whose product is the constant term and "
            "whose sum is the linear coefficient."
        ),
        focus,
    ) is None


@pytest.mark.parametrize(
    "task,missing",
    [
        (
            "Select the integer pair whose sum is the linear coefficient.",
            "product",
        ),
        (
            "Select the integer pair whose product is the constant term.",
            "sum",
        ),
    ],
)
def test_factor_pair_contract_rejects_each_missing_relation(task, missing):
    error = _anchor_provider()._objective_contract_error(
        task, _factor_pair_focus(),
    )

    assert error == (
        "learner task omits canonical content_obligation anchor: "
        f"{missing!r}"
    )
