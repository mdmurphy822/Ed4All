"""Regression tests for the shared synthesis runtime-focus helper."""

import pytest

from Trainforge.scripts.maintenance.runtime_focus import apply_runtime_focus


def test_runtime_focus_uses_the_canonical_objective():
    objectives = {
        "co-01": {
            "id": "co-01",
            "statement": "Analyze the supplied evidence.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "abcd": {
                "behavior": {"action_object": "supplied evidence"},
                "condition": "given a case",
                "degree": "without unsupported claims",
            },
        },
    }
    focused = apply_runtime_focus(
        {
            "id": "fixture-chunk",
            "text": "The supplied evidence supports a bounded conclusion.",
            "learning_outcome_refs": ["CO-01"],
        },
        objectives,
    )
    assert focused["synthesis_focus_objective"] == objectives["co-01"]
    assert focused["learning_outcome_refs"] == ["co-01"]
    assert focused["bloom_level"] == "analyze"


def test_runtime_focus_fails_loudly_when_no_objective_resolves():
    with pytest.raises(ValueError, match="objective"):
        apply_runtime_focus(
            {
                "id": "fixture-chunk",
                "text": "A neutral fixture statement.",
                "learning_outcome_refs": ["co-missing"],
            },
            {},
        )
