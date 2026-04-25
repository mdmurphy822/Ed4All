"""Wave 8 Worker — Schema-conformance tests for the canonical SourceReference.

The ``schemas/knowledge/source_reference.schema.json`` file is the shared
shape consumed by DART provenance sidecars (Wave 8), Courseforge JSON-LD
(Wave 9), and Trainforge chunks + evidence arms (Waves 10-11). This suite
locks down:

- the schema itself is a valid draft-2020-12 JSON Schema
- required fields (``sourceId``, ``role``) are enforced
- ``sourceId`` pattern accepts positional + content-hash forms and rejects
  malformed variants
- ``role`` enum is exactly ``primary | contributing | corroborating``
- optional fields (``weight``, ``confidence``, ``pages``, ``extractor``)
  accept valid values and reject out-of-range / wrong-type inputs
- strict ``additionalProperties: false`` blocks unknown keys
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Project root (Ed4All/). This file lives at
# Ed4All/schemas/tests/test_source_reference_schema.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "source_reference.schema.json"
)


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


def _valid_ref(**overrides: Any) -> Dict[str, Any]:
    base = {
        "sourceId": "dart:science_of_learning#s3_c0",
        "role": "primary",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Meta: schema is a valid draft-2020-12 JSON Schema
# ---------------------------------------------------------------------------


def test_schema_valid_json_schema():
    """Draft202012Validator.check_schema passes on our schema."""
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    Draft202012Validator.check_schema(schema)


def test_schema_has_expected_top_level_keys():
    """Sanity-check the schema shape to catch accidental drift."""
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://ed4all.dev/schemas/knowledge/source_reference.schema.json"
    )
    assert schema["title"] == "SourceReference"
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"sourceId", "role"}
    assert schema["additionalProperties"] is False


def test_source_id_pattern_is_declared():
    """The sourceId pattern must match the design doc's canonical shape."""
    schema = _load_schema()
    assert (
        schema["properties"]["sourceId"]["pattern"]
        == r"^dart:[a-z0-9_-]+#[a-z0-9_-]+$"
    )


def test_role_enum_is_locked():
    """Exactly three role values are allowed."""
    schema = _load_schema()
    assert schema["properties"]["role"]["enum"] == [
        "primary", "contributing", "corroborating",
    ]


def test_extractor_enum_is_locked():
    """Extractor enum is locked to canonical extractors + Wave 74 agent identifiers.

    First five are DART per-block envelope extractors; remaining three are
    Wave 9 / Wave 74 courseforge subagents that synthesize or re-cite source
    references (source-router from Wave 9 TF-IDF; content-generator family
    from Wave 74 mailbox-dispatched content subagents).
    """
    schema = _load_schema()
    assert schema["properties"]["extractor"]["enum"] == [
        "pdftotext", "pdfplumber", "ocr", "claude", "synthesized",
        "source-router", "content-generator", "content-generator-v1",
    ]


# ---------------------------------------------------------------------------
# Positive: minimal + fully-populated valid references
# ---------------------------------------------------------------------------


def test_minimal_valid_reference_validates():
    """Required-only shape validates."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref()))
    assert errors == [], [e.message for e in errors]


def test_fully_populated_reference_validates():
    """All optional fields at valid values validate."""
    validator = _validator()
    ref = _valid_ref(
        weight=0.75,
        confidence=0.87,
        pages=[3, 4],
        extractor="pdfplumber",
    )
    errors = list(validator.iter_errors(ref))
    assert errors == [], [e.message for e in errors]


@pytest.mark.parametrize(
    "source_id",
    [
        "dart:science_of_learning#s3_c0",          # positional
        "dart:foo#a3f9d812ac04bbc1",               # 16-hex content hash
        "dart:x#y",                                # minimal
        "dart:slug-with-dash#block-with-dash",     # dashes allowed
        "dart:slug_with_underscore#block_id",      # underscores allowed
    ],
)
def test_valid_source_id_shapes(source_id):
    """Representative valid sourceId strings validate."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(sourceId=source_id)))
    assert errors == [], f"{source_id!r} should be valid: {[e.message for e in errors]}"


# ---------------------------------------------------------------------------
# Negative: required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", ["sourceId", "role"])
def test_missing_required_field_fails(missing_field):
    """Dropping any required field fails validation."""
    validator = _validator()
    ref = _valid_ref()
    del ref[missing_field]
    errors = list(validator.iter_errors(ref))
    assert errors, f"Expected failure when {missing_field!r} is missing"


# ---------------------------------------------------------------------------
# Negative: sourceId pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_source_id",
    [
        "",                                   # empty
        "science_of_learning#s3_c0",          # missing dart: prefix
        "dart:SCIENCE#s3_c0",                 # uppercase in slug
        "dart:science#S3_C0",                 # uppercase in block id
        "dart:science",                       # missing # separator
        "dart:science#",                      # empty block id
        "dart:#s3_c0",                        # empty slug
        "courseforge:page#s3_c0",             # wrong prefix
        "dart:science_of learning#s3_c0",     # whitespace in slug
        "dart:science#s3 c0",                 # whitespace in block id
    ],
)
def test_invalid_source_id_pattern_fails(bad_source_id):
    """Malformed sourceId strings fail pattern validation."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(sourceId=bad_source_id)))
    assert errors, f"Malformed sourceId {bad_source_id!r} should fail"


# ---------------------------------------------------------------------------
# Negative: role enum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_role",
    ["", "Primary", "supporting", "PRIMARY", "main", "source"],
)
def test_invalid_role_fails(bad_role):
    """Unknown role values fail enum validation."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(role=bad_role)))
    assert errors, f"Unknown role {bad_role!r} should fail"


# ---------------------------------------------------------------------------
# Negative: optional-field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5, "high", None])
def test_invalid_weight_fails(bad_weight):
    """Weight outside [0, 1] or wrong type fails."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(weight=bad_weight)))
    assert errors, f"Invalid weight {bad_weight!r} should fail"


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, "0.8", None])
def test_invalid_confidence_fails(bad_confidence):
    """Confidence outside [0, 1] or wrong type fails."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(confidence=bad_confidence)))
    assert errors, f"Invalid confidence {bad_confidence!r} should fail"


@pytest.mark.parametrize(
    "bad_pages",
    [
        [0, 1],        # page 0 invalid
        [-1],          # negative
        "3,4",         # wrong type (string)
        [1.5],         # float
    ],
)
def test_invalid_pages_fails(bad_pages):
    """Pages array rejects non-positive / non-integer elements."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(pages=bad_pages)))
    assert errors, f"Invalid pages {bad_pages!r} should fail"


@pytest.mark.parametrize(
    "bad_extractor",
    ["PDFTOTEXT", "regex", "", "manual", "tesseract"],
)
def test_invalid_extractor_fails(bad_extractor):
    """Unknown extractor values fail enum validation."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_ref(extractor=bad_extractor)))
    assert errors, f"Unknown extractor {bad_extractor!r} should fail"


# ---------------------------------------------------------------------------
# Negative: additionalProperties strict
# ---------------------------------------------------------------------------


def test_additional_properties_rejected():
    """Unknown top-level keys are rejected (schema is strict)."""
    validator = _validator()
    ref = _valid_ref()
    ref["extra_field"] = "not-in-schema"
    errors = list(validator.iter_errors(ref))
    assert errors, "Additional properties must be rejected"


# ---------------------------------------------------------------------------
# Cross-check: DART emits IDs that validate against this schema
# ---------------------------------------------------------------------------


def test_dart_positional_block_id_validates():
    """Positional IDs like s3_c0 from DART validate."""
    validator = _validator()
    ref = _valid_ref(sourceId="dart:science_of_learning#s3_c0")
    errors = list(validator.iter_errors(ref))
    assert errors == [], [e.message for e in errors]


def test_dart_content_hash_block_id_validates():
    """Content-hash IDs from DART (16-hex) validate."""
    validator = _validator()
    ref = _valid_ref(sourceId="dart:science_of_learning#a3f9d812ac04bbc1")
    errors = list(validator.iter_errors(ref))
    assert errors == [], [e.message for e in errors]
