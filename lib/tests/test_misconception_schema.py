"""Worker R -- REC-LNK-02 first-class Misconception schema.

Schema-conformance tests for ``schemas/knowledge/misconception.schema.json``.
The schema itself must be a valid draft-2020-12 JSON Schema; the required
fields (``id``, ``misconception``, ``correction``) must be enforced; the
``id`` pattern ``^mc_[0-9a-f]{16}$`` must reject all malformed variants; and
strict ``additionalProperties: false`` must reject unknown fields so that
forward-compat additions remain deliberate schema changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Project root (Ed4All/). This file lives at
# Ed4All/lib/tests/test_misconception_schema.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "knowledge" / "misconception.schema.json"


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    return Draft202012Validator(_load_schema())


def _valid_misconception(**overrides: Any) -> Dict[str, Any]:
    base = {
        "id": "mc_0123456789abcdef",
        "misconception": "Accessibility is a legal checklist.",
        "correction": "Accessibility is a design mindset that improves outcomes for all users.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Meta: the schema itself is a valid draft-2020-12 JSON Schema.
# ---------------------------------------------------------------------------


def test_schema_valid_json_schema():
    """``Draft202012Validator.check_schema`` passes on our schema."""
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    # Raises jsonschema.exceptions.SchemaError if the schema itself is invalid.
    Draft202012Validator.check_schema(schema)


def test_schema_has_expected_top_level_keys():
    """Sanity-check the schema file to catch accidental shape drift."""
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://ed4all.dev/ns/knowledge/v1/misconception.schema.json"
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"id", "misconception", "correction"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["id"]["pattern"] == r"^mc_[0-9a-f]{16}$"


# ---------------------------------------------------------------------------
# Positive: valid misconceptions validate.
# ---------------------------------------------------------------------------


def test_valid_misconception_validates():
    """A minimal valid misconception (required fields only) validates."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_misconception()))
    assert errors == [], [e.message for e in errors]


def test_valid_misconception_with_optional_links_validates():
    """Optional concept_id and lo_id are accepted when present."""
    validator = _validator()
    mc = _valid_misconception(
        concept_id="accessibility",
        lo_id="WCAG_201_D1_1.1",
    )
    errors = list(validator.iter_errors(mc))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# Negative: required fields.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", ["id", "misconception", "correction"])
def test_missing_required_fails(missing_field):
    """Dropping any required field produces a validation error."""
    validator = _validator()
    mc = _valid_misconception()
    del mc[missing_field]
    errors = list(validator.iter_errors(mc))
    assert errors, f"Expected failure when {missing_field!r} is missing"
    # At least one error should reference the missing required property.
    assert any(
        missing_field in err.message or err.validator == "required"
        for err in errors
    )


def test_empty_misconception_text_fails():
    """``misconception`` with empty string fails minLength=1."""
    validator = _validator()
    mc = _valid_misconception(misconception="")
    errors = list(validator.iter_errors(mc))
    assert errors, "Empty misconception text must fail validation"


def test_empty_correction_text_fails():
    """``correction`` with empty string fails minLength=1."""
    validator = _validator()
    mc = _valid_misconception(correction="")
    errors = list(validator.iter_errors(mc))
    assert errors, "Empty correction text must fail validation"


# ---------------------------------------------------------------------------
# Negative: id pattern.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "mc_xyz",                      # too short, non-hex
        "misconception_abcdefghijklmnop",  # wrong prefix
        "mc_0123456789abcdef0",        # 17 hex chars (too long)
        "mc_0123456789abcde",          # 15 hex chars (too short)
        "mc_0123456789ABCDEF",         # uppercase hex (pattern is lowercase-only)
        "MC_0123456789abcdef",         # uppercase prefix
        "mc-0123456789abcdef",         # wrong separator
        "mc_0123456789abcdeg",         # 'g' not a hex char
        "",                            # empty
        "0123456789abcdef",            # no prefix at all
    ],
)
def test_invalid_id_pattern_fails(bad_id):
    """Non-conforming IDs must fail pattern validation."""
    validator = _validator()
    mc = _valid_misconception(id=bad_id)
    errors = list(validator.iter_errors(mc))
    assert errors, f"Malformed id {bad_id!r} should fail validation"


# ---------------------------------------------------------------------------
# Negative: strict additionalProperties.
# ---------------------------------------------------------------------------


def test_additional_properties_rejected():
    """Unknown fields are rejected (schema is strict)."""
    validator = _validator()
    mc = _valid_misconception()
    mc["extra_field"] = "not-in-schema"
    errors = list(validator.iter_errors(mc))
    assert errors, "Additional properties must be rejected by strict schema"
    assert any(
        err.validator == "additionalProperties" or "additional" in err.message.lower()
        for err in errors
    )
