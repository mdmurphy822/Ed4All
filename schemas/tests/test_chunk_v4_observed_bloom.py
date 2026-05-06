"""GPT Feedback v2 Wave 1 / W1.A — chunk_v4 schema test for the new optional
``observed_bloom_level`` + ``bloom_alignment`` fields.

The chunk root carries ``additionalProperties: true`` so legacy chunks accept
the new fields silently — but this test pins:

  * Chunks WITHOUT the fields validate (legacy path).
  * Chunks WITH the fields populated validate.
  * ``observed_bloom_level`` is enum-constrained to the canonical six BloomLevel
    values (or null) — non-canonical strings fail.
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


def test_legacy_chunk_without_observed_bloom_validates():
    """Pre-Wave-1 chunks (no observed_bloom_level / bloom_alignment) must
    validate untouched."""
    validator = _build_validator()
    chunk = _base_chunk()
    assert "observed_bloom_level" not in chunk
    assert "bloom_alignment" not in chunk
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


@pytest.mark.parametrize(
    "level",
    ["remember", "understand", "apply", "analyze", "evaluate", "create"],
)
def test_chunk_with_canonical_observed_bloom_level_validates(level):
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["observed_bloom_level"] = level
    chunk["bloom_alignment"] = (level == chunk["bloom_level"])
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"level={level} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_chunk_with_null_observed_bloom_level_validates():
    """observed_bloom_level is nullable — classifier-not-run path."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["observed_bloom_level"] = None
    chunk["bloom_alignment"] = None
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


@pytest.mark.parametrize("alignment", [True, False, None])
def test_chunk_with_canonical_bloom_alignment_validates(alignment):
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["observed_bloom_level"] = "apply"
    chunk["bloom_alignment"] = alignment
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"alignment={alignment!r} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_chunk_with_non_enum_observed_bloom_level_fails():
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["observed_bloom_level"] = "synthesize"  # not canonical
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for non-canonical observed_bloom_level"


def test_chunk_with_non_boolean_bloom_alignment_fails():
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["bloom_alignment"] = "yes"  # not bool/null
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for non-boolean bloom_alignment"
