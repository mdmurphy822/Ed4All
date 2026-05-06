"""GPT Feedback v2 Wave 1 (W1.D) — PhaseOutput schema.

Pins ``schemas/governance/phase_output.schema.json``:

  * PhaseOutput dict without ``promotion_decision`` validates (legacy shape).
  * PhaseOutput dict with each of the four enum values validates.
  * PhaseOutput dict with ``promotion_decision: "regenerate"`` (a
    ``GateResult.action`` value, NOT a ``PhaseOutput.promotion_decision``
    value) FAILS — confirms the enum scope is constrained correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_OUTPUT_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "governance" / "phase_output.schema.json"
)

EXPECTED_PROMOTION_DECISION_ENUM = ["pass", "warn", "fail", "escalate"]


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _load_schema() -> Dict[str, Any]:
    return json.loads(PHASE_OUTPUT_SCHEMA_PATH.read_text())


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    return Draft202012Validator(schema)


def _legacy_phase_output_dict() -> Dict[str, Any]:
    """A PhaseOutput dict in the pre-Wave-1 shape (no promotion_decision)."""
    return {
        "run_id": "WF-20260506-test1234",
        "phase_name": "content_generation_outline",
        "outputs": {"blocks_emitted": 12},
        "artifacts": ["/tmp/run_id/01_outline/page_001.json"],
        "gate_results": {
            "content_structure": {
                "gate_id": "content_structure",
                "severity": "critical",
                "passed": True,
                "issues": [],
                "details": {},
            }
        },
        "decision_captures_path": "/tmp/run_id/captures/decisions.jsonl",
        "status": "ok",
        "error": None,
        "metrics": {"elapsed_seconds": 42.0},
    }


# ---------------------------------------------------------------------------
# Schema-level invariants
# ---------------------------------------------------------------------------


def test_schema_file_exists():
    assert PHASE_OUTPUT_SCHEMA_PATH.exists(), (
        f"PhaseOutput schema missing at {PHASE_OUTPUT_SCHEMA_PATH}"
    )


def test_schema_has_canonical_id_and_draft():
    schema = _load_schema()
    assert (
        schema["$id"]
        == "https://ed4all.dev/ns/governance/v1/phase_output.schema.json"
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"


def test_promotion_decision_registered_but_not_required():
    """The whole point of the W1.D wave: the field is registered in
    `properties` so the schema validates the enum, but is NOT in
    `required[]` so legacy PhaseOutput dicts validate untouched."""
    schema = _load_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    assert "promotion_decision" in properties, (
        "Expected promotion_decision to be registered in properties"
    )
    assert "promotion_decision" not in required, (
        "promotion_decision must remain OPTIONAL — Wave 1 ships the field "
        "unwired; legacy PhaseOutput dicts must validate without it."
    )


def test_promotion_decision_enum_scope():
    schema = _load_schema()
    promo = schema["properties"]["promotion_decision"]
    enum_values = promo.get("enum", [])
    # Enum allows null because the field is optional — a missing value or
    # an explicit null must both validate.
    assert set(EXPECTED_PROMOTION_DECISION_ENUM).issubset(set(enum_values)), (
        f"Expected enum to contain {EXPECTED_PROMOTION_DECISION_ENUM}, "
        f"got {enum_values}"
    )
    # A common GateResult.action value MUST NOT have leaked into the
    # PhaseOutput.promotion_decision enum.
    assert "regenerate" not in enum_values, (
        "promotion_decision enum must not contain 'regenerate' — that value "
        "lives on GateResult.action, not PhaseOutput.promotion_decision."
    )


# ---------------------------------------------------------------------------
# Value-level validation
# ---------------------------------------------------------------------------


def test_phase_output_without_promotion_decision_validates():
    validator = _build_validator()
    legacy = _legacy_phase_output_dict()
    errors = list(validator.iter_errors(legacy))
    assert errors == [], (
        f"Legacy PhaseOutput dict (no promotion_decision) must validate; "
        f"got {errors}"
    )


@pytest.mark.parametrize("value", EXPECTED_PROMOTION_DECISION_ENUM)
def test_phase_output_with_each_promotion_decision_value_validates(value):
    validator = _build_validator()
    payload = _legacy_phase_output_dict()
    payload["promotion_decision"] = value
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"Expected promotion_decision='{value}' to validate; got {errors}"
    )


def test_phase_output_with_explicit_null_promotion_decision_validates():
    validator = _build_validator()
    payload = _legacy_phase_output_dict()
    payload["promotion_decision"] = None
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"Expected explicit null promotion_decision to validate; got {errors}"
    )


def test_phase_output_with_regenerate_value_fails():
    """`regenerate` is a GateResult.action value, not a
    PhaseOutput.promotion_decision value — the schema must reject it to
    keep the enum scopes disjoint."""
    validator = _build_validator()
    payload = _legacy_phase_output_dict()
    payload["promotion_decision"] = "regenerate"
    errors = list(validator.iter_errors(payload))
    assert errors, (
        "Expected promotion_decision='regenerate' to FAIL — confirms enum "
        "scope is constrained to {pass, warn, fail, escalate}."
    )


@pytest.mark.parametrize(
    "value",
    [
        "PASS",
        "block",
        "promoted",
        "rejected",
        "",
    ],
)
def test_phase_output_with_other_invalid_promotion_decision_fails(value):
    validator = _build_validator()
    payload = _legacy_phase_output_dict()
    payload["promotion_decision"] = value
    errors = list(validator.iter_errors(payload))
    assert errors, f"Expected promotion_decision='{value}' to fail validation"
