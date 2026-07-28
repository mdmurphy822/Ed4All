"""Eligibility must apply the production evidence-window contract pre-dispatch."""

from Trainforge.synthesis_eligibility import pair_eligibility


def _focused_chunk(
    *,
    chunk_id: str,
    text: str,
    action_object: str,
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "chunk_type": "explanation",
        "learning_outcome_refs": ["co-01"],
        "synthesis_focus_objective": {
            "id": "co-01",
            "statement": f"Explain {action_object}.",
            "bloom_level": "understand",
            "action_object": action_object,
            "behavior": {
                "verb": "explain",
                "action_object": action_object,
            },
        },
    }


def test_broad_conjunctive_glossary_objective_rejects_partial_evidence() -> None:
    focused = _focused_chunk(
        chunk_id="glossary-entry",
        action_object="domain, range, and codomain",
        text=(
            "Domain is the set of permitted input values for a function. "
            "A glossary may describe those inputs with interval notation and "
            "give several representative examples for learners to inspect."
        ),
    )

    eligibility = pair_eligibility(focused, kind="instruction")

    assert not eligibility.eligible
    assert eligibility.reason == "objective_evidence_window_not_viable"


def test_broad_conjunctive_line_objective_rejects_slope_only_evidence() -> None:
    focused = _focused_chunk(
        chunk_id="line-explanation",
        action_object="slope and vertical intercept",
        text=(
            "Slope measures the rate of change of a straight line. It can be "
            "calculated as rise divided by run, and its sign determines "
            "whether the line increases or decreases from left to right."
        ),
    )

    eligibility = pair_eligibility(focused, kind="instruction")

    assert not eligibility.eligible
    assert eligibility.reason == "objective_evidence_window_not_viable"


def test_broad_conjunctive_objective_accepts_complete_window() -> None:
    focused = _focused_chunk(
        chunk_id="complete-line-explanation",
        action_object="slope and vertical intercept",
        text=(
            "Slope measures the rate of change of a straight line and can be "
            "calculated as rise divided by run. The vertical intercept is the "
            "point where that line crosses the vertical axis, so both values "
            "describe distinct features of the same linear relationship."
        ),
    )

    assert pair_eligibility(focused, kind="instruction").eligible
