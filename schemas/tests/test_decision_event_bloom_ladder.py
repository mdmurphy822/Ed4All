"""WI-03 — bloom-ladder decision_type enum extension regression tests.

Confirms that ``schemas/events/decision_event.schema.json::decision_type.enum``
contains the four bloom-ladder / misconception strings the ladder + DPO
emit-side code fires, and that a well-formed event carrying each new type
validates against the canonical schema (fixture:
``schemas/tests/fixtures/bloom_ladder/decision_event_bloom_ladder_accept.json``).
Additive extension — no schema-version bump; the alphabetical-sort invariant is
enforced by ``test_decision_event_wave2_enum_coverage.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "events" / "decision_event.schema.json"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "bloom_ladder"
    / "decision_event_bloom_ladder_accept.json"
)

# The four strings WI-03 adds. Each is paired with the emitter that fires it
# so a failure message can point the operator at the right consumer.
BLOOM_LADDER_DECISION_TYPES: tuple[tuple[str, str], ...] = (
    (
        "bloom_ladder_block_selection",
        "bloom-ladder block selection (ED4ALL_BLOOM_LADDER)",
    ),
    (
        "bloom_ladder_ceiling_enforcement",
        "bloom-ladder objective-ceiling clamp (ED4ALL_BLOOM_LADDER)",
    ),
    (
        "misconception_card_authoring",
        "misconception-card authoring (training synthesis)",
    ),
    (
        "dpo_pair_from_misconception",
        "misconception-derived DPO pair admission (training synthesis)",
    ),
)


def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_fixture_events() -> list[dict]:
    with FIXTURE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["events"]


@pytest.fixture(scope="module")
def decision_type_enum() -> list[str]:
    """Load the decision_type enum from the canonical decision-event schema."""
    return _load_schema()["properties"]["decision_type"]["enum"]


@pytest.mark.parametrize("decision_type,emitter", BLOOM_LADDER_DECISION_TYPES)
def test_bloom_ladder_decision_type_present(
    decision_type: str, emitter: str, decision_type_enum: list[str]
) -> None:
    """Every WI-03 decision_type string MUST be a member of the enum.

    Failing here means the enum extension was not landed (or was reverted);
    a ``DECISION_VALIDATION_STRICT=true`` run fails closed when
    ``%(emitter)s`` fires.
    """
    assert decision_type in decision_type_enum, (
        f"Bloom-ladder decision_type {decision_type!r} is missing from "
        "schemas/events/decision_event.schema.json::decision_type.enum. "
        f"Fired by {emitter}. Add the string at its alphabetically sorted slot."
    )


def test_fixture_covers_every_new_decision_type() -> None:
    """The fixture carries exactly one well-formed event per new type."""
    fixture_types = [event["decision_type"] for event in _load_fixture_events()]
    expected = [decision_type for decision_type, _ in BLOOM_LADDER_DECISION_TYPES]
    assert sorted(fixture_types) == sorted(expected), (
        "decision_event_bloom_ladder_accept.json must hold one event per "
        f"WI-03 decision_type; got {fixture_types!r}"
    )


def test_fixture_events_validate_against_schema() -> None:
    """Each fixture event validates against the canonical decision-event schema."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    for event in _load_fixture_events():
        errors = list(validator.iter_errors(event))
        assert not errors, (
            f"fixture event {event['decision_type']!r} failed schema "
            f"validation: {[error.message for error in errors]}"
        )


def test_bogus_decision_type_rejected() -> None:
    """Negative control: the enum is actually enforced, not vacuously passing."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    bogus = dict(_load_fixture_events()[0], decision_type="not_a_real_decision_type")
    assert list(validator.iter_errors(bogus)), (
        "a bogus decision_type validated — the enum constraint is not enforced"
    )
