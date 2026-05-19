"""chunk_v4 schema test for the new optional ``curies`` + ``forced_curies``
fields (curie_anchoring-gate fix, unit U1).

The Trainforge ``curie_anchoring`` pre-training gate is blind to force-injected
CURIEs. The fix threads a structured CURIE set onto each chunk so the gate has
an authoritative source-CURIE set. This test pins the SCHEMA contract for that
field pair:

  * Chunks WITH ``curies`` + ``forced_curies`` populated validate.
  * ``forced_curies`` NOT a subset of ``curies`` is still schema-VALID — the
    subset invariant is a validator concern (lands in a separate unit), NOT a
    JSON Schema constraint.
  * Legacy chunks OMITTING both fields validate (back-compat — the chunk root
    carries ``additionalProperties: true`` and neither field is required).
  * An empty-string CURIE token fails (``minLength: 1`` on the array items).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schema = json.loads(CHUNK_SCHEMA_PATH.read_text())
    id_to_schema: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s

    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry)


def _base_chunk() -> Dict[str, Any]:
    """Minimal chunk dict carrying every key in the schema's ``required`` list.

    Built field-by-field against schemas/knowledge/chunk_v4.schema.json's
    ``required`` array: id, schema_version, chunk_type, text, html,
    follows_chunk, source, concept_tags, learning_outcome_refs, difficulty,
    tokens_estimate, word_count, bloom_level.
    """
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "week_01_overview",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "apply",
    }


def test_chunk_with_curies_and_forced_curies_validates():
    """(a) A chunk carrying populated non-empty ``curies`` + ``forced_curies``
    string arrays validates."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["curies"] = ["wcag:1.1.1", "wcag:2.4.6", "aria:landmark"]
    chunk["forced_curies"] = ["wcag:2.4.6"]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_forced_curies_not_subset_of_curies_is_schema_valid():
    """(b) ``forced_curies`` need NOT be a subset of ``curies`` at the SCHEMA
    layer. The subset-of-``curies`` invariant is enforced by the curie_anchoring
    validator (a separate implementation unit), not by JSON Schema — JSON Schema
    has no cross-property set-containment construct. This test pins that the
    schema deliberately does NOT reject the violation, so the validator remains
    the single authoritative owner of the invariant."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["curies"] = ["wcag:1.1.1"]
    # "aria:landmark" is absent from curies — a subset-invariant violation.
    chunk["forced_curies"] = ["aria:landmark"]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        "schema must NOT enforce the forced_curies ⊆ curies invariant; "
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_legacy_chunk_without_curie_fields_validates():
    """(c) Pre-this-wave chunks omitting both ``curies`` and ``forced_curies``
    must validate untouched (back-compat — neither field is required)."""
    validator = _build_validator()
    chunk = _base_chunk()
    assert "curies" not in chunk
    assert "forced_curies" not in chunk
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_chunk_with_empty_string_curie_token_fails():
    """(d) An empty-string CURIE token violates ``minLength: 1`` on the array
    items and must fail validation."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["curies"] = ["wcag:1.1.1", ""]
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for empty-string curie token"


def test_forced_curies_with_empty_string_token_fails():
    """``forced_curies`` items carry the same ``minLength: 1`` constraint."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["curies"] = ["wcag:1.1.1"]
    chunk["forced_curies"] = [""]
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for empty-string forced_curies token"
