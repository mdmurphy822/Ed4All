"""Wave 10 — chunk_v4 Source.source_references[] schema extension tests.

Wave 10 extends ``$defs/Source`` with an optional ``source_references[]``
array, threading Courseforge's ``sourceReferences`` +
``data-cf-source-ids`` through Trainforge chunks. Every entry conforms
to the canonical ``schemas/knowledge/source_reference.schema.json``
shape.

Contract locked by this suite:

- chunk with ``source.source_references[]`` populated validates (strict)
- chunk without ``source.source_references`` still validates (legacy /
  pre-Wave-9 corpora must pass — absence = "unknown")
- malformed entries (missing required fields, bad sourceId pattern, bad
  role enum, unknown properties on the ref) are rejected
- the ``$ref`` to source_reference.schema.json resolves properly
- Source sub-schema stays strict (``additionalProperties: false``) —
  arbitrary keys must still be rejected
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


# --------------------------------------------------------------------- #
# Validator construction — mirrors the pattern used in test_chunk_validation
# --------------------------------------------------------------------- #


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator, RefResolver

    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    store: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    return Draft202012Validator(schema, resolver=resolver)


# --------------------------------------------------------------------- #
# Chunk fixture helpers
# --------------------------------------------------------------------- #


def _base_source() -> Dict[str, Any]:
    return {
        "course_id": "SAMPLE_101",
        "module_id": "week_01",
        "lesson_id": "lesson_01",
    }


def _base_chunk(source_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    source = _base_source()
    if source_overrides:
        source.update(source_overrides)
    return {
        "id": "sample_101_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk content for testing.",
        "html": "<p>Sample chunk content for testing.</p>",
        "follows_chunk": None,
        "source": source,
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 10,
        "word_count": 6,
        "bloom_level": "understand",
    }


def _valid_ref(**overrides: Any) -> Dict[str, Any]:
    base = {
        "sourceId": "dart:science_of_learning#s3_c0",
        "role": "primary",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- #
# Meta: schema remains valid JSON Schema after Wave 10 extension
# --------------------------------------------------------------------- #


def test_chunk_schema_remains_valid_draft_2020_12():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)


def test_source_sub_schema_still_strict():
    """Source sub-schema must keep additionalProperties: false intact."""
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    source_def = schema["$defs"]["Source"]
    assert source_def["additionalProperties"] is False


def test_source_references_property_declared():
    """source_references must be declared in Source.properties (so the
    strict path allows it when present)."""
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    source_def = schema["$defs"]["Source"]
    assert "source_references" in source_def["properties"]
    prop = source_def["properties"]["source_references"]
    assert prop["type"] == "array"
    assert "$ref" in prop["items"]
    assert prop["items"]["$ref"].endswith("source_reference.schema.json")


# --------------------------------------------------------------------- #
# Positive cases: chunks with + without source_references both validate
# --------------------------------------------------------------------- #


def test_legacy_chunk_without_source_references_validates():
    """Pre-Wave-9 corpora have no source_references — must still pass."""
    validator = _build_validator()
    chunk = _base_chunk()
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


def test_chunk_with_single_source_reference_validates():
    validator = _build_validator()
    chunk = _base_chunk({"source_references": [_valid_ref()]})
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


def test_chunk_with_multiple_source_references_validates():
    """Merged chunks carry multiple refs (one per merged section)."""
    validator = _build_validator()
    refs = [
        _valid_ref(sourceId="dart:slug#s1_c0", role="primary"),
        _valid_ref(sourceId="dart:slug#s2_c0", role="contributing"),
        _valid_ref(sourceId="dart:slug#s3_c0", role="corroborating"),
    ]
    chunk = _base_chunk({"source_references": refs})
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


def test_chunk_with_empty_source_references_validates():
    """Empty array is valid — the field is optional."""
    validator = _build_validator()
    chunk = _base_chunk({"source_references": []})
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


def test_chunk_with_fully_populated_refs_validates():
    """All optional SourceReference fields at valid values."""
    validator = _build_validator()
    refs = [
        _valid_ref(
            weight=0.85,
            confidence=0.92,
            pages=[3, 4, 5],
            extractor="pdfplumber",
        ),
    ]
    chunk = _base_chunk({"source_references": refs})
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


# --------------------------------------------------------------------- #
# Role-precedence fixture — multi-role mix that mirrors merge output
# --------------------------------------------------------------------- #


def test_chunk_with_role_precedence_mix_validates():
    """Merged sections can produce all three roles in one chunk."""
    validator = _build_validator()
    refs = [
        _valid_ref(sourceId="dart:a#s0_c0", role="primary"),
        _valid_ref(sourceId="dart:a#s1_c0", role="contributing"),
        _valid_ref(sourceId="dart:b#s0_c0", role="corroborating"),
    ]
    chunk = _base_chunk({"source_references": refs})
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


# --------------------------------------------------------------------- #
# Negative cases: malformed entries rejected
# --------------------------------------------------------------------- #


def test_chunk_with_missing_sourceId_rejected():
    validator = _build_validator()
    bad_ref = {"role": "primary"}  # no sourceId
    chunk = _base_chunk({"source_references": [bad_ref]})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Missing sourceId should be rejected"


def test_chunk_with_missing_role_rejected():
    validator = _build_validator()
    bad_ref = {"sourceId": "dart:slug#s0_c0"}  # no role
    chunk = _base_chunk({"source_references": [bad_ref]})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Missing role should be rejected"


@pytest.mark.parametrize(
    "bad_source_id",
    [
        "",
        "no_dart_prefix",
        "dart:NO_UPPERCASE#s0_c0",
        "dart:slug#",
        "dart:#s0_c0",
    ],
)
def test_chunk_with_malformed_source_id_rejected(bad_source_id):
    validator = _build_validator()
    bad_ref = {"sourceId": bad_source_id, "role": "primary"}
    chunk = _base_chunk({"source_references": [bad_ref]})
    errors = list(validator.iter_errors(chunk))
    assert errors, f"Malformed sourceId {bad_source_id!r} should be rejected"


@pytest.mark.parametrize("bad_role", ["", "PRIMARY", "supporting", "main"])
def test_chunk_with_bad_role_rejected(bad_role):
    validator = _build_validator()
    bad_ref = {"sourceId": "dart:slug#s0_c0", "role": bad_role}
    chunk = _base_chunk({"source_references": [bad_ref]})
    errors = list(validator.iter_errors(chunk))
    assert errors, f"Bad role {bad_role!r} should be rejected"


def test_chunk_with_unknown_source_ref_field_rejected():
    """SourceReference sub-schema has additionalProperties: false."""
    validator = _build_validator()
    bad_ref = _valid_ref(unknown_extra_key="should_fail")
    chunk = _base_chunk({"source_references": [bad_ref]})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Unknown keys on SourceReference should be rejected"


def test_chunk_with_arbitrary_source_key_still_rejected():
    """Adding an unknown top-level key to source.* must still fail."""
    validator = _build_validator()
    chunk = _base_chunk({"unknown_source_field": "should_fail"})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Unknown keys on Source must be rejected"


def test_chunk_with_ref_weight_out_of_range_rejected():
    validator = _build_validator()
    chunk = _base_chunk({"source_references": [_valid_ref(weight=1.5)]})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Weight > 1 should be rejected"


def test_chunk_with_ref_pages_non_integer_rejected():
    validator = _build_validator()
    chunk = _base_chunk({"source_references": [_valid_ref(pages=[1.5])]})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Float pages should be rejected"


def test_chunk_with_ref_bad_extractor_rejected():
    validator = _build_validator()
    chunk = _base_chunk({"source_references": [_valid_ref(extractor="unknown")]})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Unknown extractor should be rejected"


# --------------------------------------------------------------------- #
# Ref resolution — the $ref resolves to the source_reference schema
# --------------------------------------------------------------------- #


def test_source_references_items_ref_resolves():
    """The $ref chain (chunk.Source → source_reference) must resolve."""
    validator = _build_validator()
    # Negative case leveraging the $ref: rejection proves resolution happened.
    bad_ref = {"sourceId": "dart:slug#s0_c0", "role": "not-in-enum"}
    chunk = _base_chunk({"source_references": [bad_ref]})
    errors: List[Any] = list(validator.iter_errors(chunk))
    assert errors, (
        "Bad role value must fail via $ref resolution to "
        "source_reference.schema.json"
    )


# --------------------------------------------------------------------- #
# Heterogeneous mix: valid refs alongside audit-trail fields
# --------------------------------------------------------------------- #


def test_chunk_with_source_references_and_audit_trail_validates():
    """source_references coexists with html_xpath / char_span / item_path."""
    validator = _build_validator()
    chunk = _base_chunk(
        {
            "html_xpath": "/html/body/main[1]/section[1]",
            "char_span": [0, 32],
            "item_path": "content/week_01/lesson_01.html",
            "source_references": [_valid_ref()],
        }
    )
    errors = list(validator.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


def test_chunk_source_references_always_array_not_scalar():
    """source_references is always array — scalar refs must be rejected."""
    validator = _build_validator()
    chunk = _base_chunk({"source_references": _valid_ref()})
    errors = list(validator.iter_errors(chunk))
    assert errors, "Scalar source_references should be rejected (array-only)"
