"""Wave 1 Worker W1.E deliverable + Wave 1 end-of-wave Test 5.

Confirms that ``schemas/events/decision_event.schema.json::decision_type.enum``
contains the four new strings Wave 2 will emit. Catches the regression class
where Wave 2 lands the emit-side code, fires under
``DECISION_VALIDATION_STRICT=true``, and fails closed because the enum was
never extended. Catches it at Wave 1 end-of-wave time, not at Wave 2
first-CI-run time.

See ``plans/gpt-feedback-2-wave1-schemas-2026-05.md`` Section "Worker W1.E".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "events" / "decision_event.schema.json"
)

# The four strings Wave 2 will emit. Each is paired with the validator that
# fires it so a failure message can point the operator at the right
# downstream consumer.
WAVE2_DECISION_TYPES: tuple[tuple[str, str], ...] = (
    (
        "claim_support_check",
        "ClaimSupportValidator (Wave 2 W2.F)",
    ),
    (
        "training_pair_promotion_check",
        "TrainingPairPromotionValidator (Wave 2 W2.E / P0c Fix 7b)",
    ),
    (
        "padded_distractor_check",
        "PaddedDistractorValidator (Wave 2 W2.D / P0b Fix 6)",
    ),
    (
        "distractor_structural_check",
        "DistractorStructuralValidator / extended"
        " BlockAssessmentItemPayloadValidator (Wave 2 W2.C)",
    ),
)


@pytest.fixture(scope="module")
def decision_type_enum() -> list[str]:
    """Load the decision_type enum from the canonical decision-event schema."""
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    return schema["properties"]["decision_type"]["enum"]


@pytest.mark.parametrize("decision_type,emitter", WAVE2_DECISION_TYPES)
def test_wave2_decision_type_present(
    decision_type: str, emitter: str, decision_type_enum: list[str]
) -> None:
    """Every Wave 2 decision_type string MUST be a member of the enum.

    Failing here means Worker W1.E's deliverable was not landed (or was
    reverted). Wave 2's ``DECISION_VALIDATION_STRICT=true`` runs will fail
    closed when ``%(emitter)s`` fires.
    """
    assert decision_type in decision_type_enum, (
        f"Wave 2 decision_type {decision_type!r} is missing from "
        "schemas/events/decision_event.schema.json::decision_type.enum. "
        f"Fired by {emitter}. Add the string (alphabetically sorted) per "
        "Worker W1.E in plans/gpt-feedback-2-wave1-schemas-2026-05.md."
    )


def test_decision_type_enum_alphabetically_sorted(
    decision_type_enum: list[str],
) -> None:
    """Anti-regression: the enum is alphabetically sorted.

    Free check that catches a future PR dropping new entries in arbitrary
    positions. The list is canonically sorted today (audited 2026-05-06
    pre-Wave-1; preserved post-Wave-1 by Worker W1.E inserting each new
    entry at its sorted slot).
    """
    assert decision_type_enum == sorted(decision_type_enum), (
        "decision_type.enum is not alphabetically sorted. The first "
        "out-of-order pair must be re-inserted at its sorted slot."
    )
