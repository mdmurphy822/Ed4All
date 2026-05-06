"""GPT Feedback v2 Wave 1 (W1.C) — course-status enum schema.

Pins the GPT 5-value enum at
``schemas/governance/course_status.schema.json``:

  * All five enum values validate.
  * Non-enum strings fail (``"trainable"``, ``"certified"``).
  * The empty string fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_STATUS_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "governance" / "course_status.schema.json"
)

EXPECTED_ENUM = [
    "certified_accessible",
    "certified_instructional",
    "certified_trainable",
    "non_certified_archive",
    "failed",
]


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _load_schema() -> Dict[str, Any]:
    return json.loads(COURSE_STATUS_SCHEMA_PATH.read_text())


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# Schema-level invariants
# ---------------------------------------------------------------------------


def test_schema_file_exists():
    assert COURSE_STATUS_SCHEMA_PATH.exists(), (
        f"Course-status schema missing at {COURSE_STATUS_SCHEMA_PATH}"
    )


def test_schema_has_canonical_id_and_draft():
    schema = _load_schema()
    assert (
        schema["$id"]
        == "https://ed4all.dev/ns/governance/v1/course_status.schema.json"
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "string"


def test_schema_enum_matches_gpt_five_value_list():
    schema = _load_schema()
    assert schema["enum"] == EXPECTED_ENUM


# ---------------------------------------------------------------------------
# Value-level validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", EXPECTED_ENUM)
def test_each_enum_value_validates(value):
    validator = _build_validator()
    errors = list(validator.iter_errors(value))
    assert errors == [], (
        f"Expected enum value '{value}' to validate, got {errors}"
    )


@pytest.mark.parametrize(
    "value",
    [
        "trainable",
        "certified",
        "CERTIFIED_TRAINABLE",  # case-sensitive enum
        "certified-trainable",  # hyphen vs underscore
        "passed",
    ],
)
def test_non_enum_strings_fail(value):
    validator = _build_validator()
    errors = list(validator.iter_errors(value))
    assert errors, f"Expected '{value}' to fail validation"


def test_empty_string_fails():
    validator = _build_validator()
    errors = list(validator.iter_errors(""))
    assert errors, "Expected empty string to fail validation"
