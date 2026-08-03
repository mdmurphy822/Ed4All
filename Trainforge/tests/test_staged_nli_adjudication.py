"""Regressions for exact-quote and symbolic algebra adjudication."""
from __future__ import annotations

import json

from Trainforge.generators.staged.provider import (
    StagedSynthesisProvider,
    _symbolic_math_supports,
)
from Trainforge.tests.test_staged_synthesis_provider import (
    _Base,
    _chunk,
    _sft_plan,
)


def test_equivalent_algebraic_transformation_is_supported():
    premise = (
        "Factoring out 5 gives 5(y² + 2y - 3) = 0, and dividing both "
        "sides by 5 yields y² + 2y - 3 = 0."
    )
    hypothesis = (
        "Dividing both sides by 5 after factoring the GCF yields "
        "y² + 2y - 3 = 0."
    )
    assert _symbolic_math_supports(premise, hypothesis)
    assert not _symbolic_math_supports(
        premise, "Dividing by 5 incorrectly yields y² + 2y + 3 = 0.",
    )


def test_plan_scores_exact_canonical_quote_before_local_context(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768")
    class Recorder:
        def __init__(self):
            self.calls = []

        def score_pair(self, *, premise, hypothesis):
            self.calls.append((premise, hypothesis))
            return type("_Score", (), {
                "entailment": 0.99,
                "contradiction": 0.001,
            })()

    plan = json.loads(_sft_plan())
    realization = json.dumps({
        "prompt": "Analyze how atomicity prevents partially applied updates.",
        "completion": (
            "Atomicity makes all operations succeed or fail together, "
            "preventing partially applied updates."
        ),
        "covered_claim_indices": [0],
    })
    scorer = Recorder()
    base = _Base([json.dumps(plan), realization])
    base._plan_nli_scorer = scorer
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    exact_quote = plan["supported_claims"][0]["evidence_quote"]
    assert scorer.calls[0][0] == exact_quote
    assert scorer.calls[0][1] == plan["supported_claims"][0]["claim"]


def test_monic_specialization_rejects_overbroad_leading_coefficient(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768")
    chunk = {
        "id": "synthetic-monic",
        "text": (
            "For a monic trinomial x² + bx + c, list factor pairs of c. "
            "The correct pair has product c and sum b; testing only product "
            "is insufficient."
        ),
        "learning_outcome_refs": ["obj-monic"],
        "bloom_level": "analyze",
        "content_type_label": "explanation",
        "concept_tags": ["monic trinomial factoring"],
        "misconceptions": [],
        "synthesis_focus_objective": {
            "id": "obj-monic",
            "statement": (
                "Analyze factor pairs to select the correct pair satisfying "
                "the product and sum conditions for a monic trinomial."
            ),
            "bloom_level": "analyze",
            "abcd": {
                "behavior": {
                    "action_object": (
                        "factor pairs to select the correct pair satisfying "
                        "the product and sum conditions"
                    ),
                },
                "condition": (
                    "given a trinomial x² + bx + c and factor pairs of c"
                ),
                "degree": "with logical justification",
            },
        },
    }
    plan = {
        "objective_id": "obj-monic",
        "bloom_level": "analyze",
        "supported_claims": [{
            "claim": (
                "For x² + bx + c, the selected factor pair must have "
                "product c and sum b."
            ),
            "evidence_quote": (
                "The correct pair has product c and sum b; testing only "
                "product is insufficient."
            ),
        }],
        "learner_task": (
            "Given x² + bx + c and factor pairs of c, analyze which pair "
            "has product c and sum b, and justify the selection."
        ),
        "misconception_affordance": {
            "error_mechanism": "",
            "faulty_step": "",
            "causal_rationale": "",
        },
    }
    assert "ax²" not in json.dumps(plan)
    response = {
        "prompt": (
            "Given the trinomial x² + bx + c and factor pairs of c, select "
            "the pair satisfying product c and sum b, with justification."
        ),
        "completion": (
            "Choose the pair whose product is c and whose sum is b. "
            "A pair satisfying only the product condition is insufficient."
        ),
        "covered_claim_indices": [0],
    }
    base = _Base([json.dumps(plan), json.dumps(response)])
    StagedSynthesisProvider(base).paraphrase_instruction({}, chunk)
    assert "ax²" not in json.dumps(base.prompts)
