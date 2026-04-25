"""Wave 76 — chunk_v4 schema test for the optional ``bloom_level_secondary`` field.

Compound bloom_level values (``"remember-apply"``, ``"understand-analyze"``)
are split at chunk emit time into a primary (HIGHER) Bloom level + a
secondary (LOWER) Bloom level. The secondary is stored in a new optional
``bloom_level_secondary`` field that mirrors the canonical six-value
BloomLevel enum.

These tests guard the schema:

  * The field is optional (chunks without it must validate).
  * When present, it accepts any of the six canonical Bloom levels.
  * It rejects values outside the canonical enum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    id_to_schema: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s

    try:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        resources = [
            (sid, Resource.from_contents(s, default_specification=DRAFT202012))
            for sid, s in id_to_schema.items()
        ]
        registry = Registry().with_resources(resources)
        return Draft202012Validator(schema, registry=registry)
    except ImportError:
        from jsonschema import RefResolver  # type: ignore

        resolver = RefResolver.from_schema(schema, store=dict(id_to_schema))
        return Draft202012Validator(schema, resolver=resolver)


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


def test_chunk_without_bloom_secondary_validates():
    validator = _build_validator()
    chunk = _base_chunk()
    assert "bloom_level_secondary" not in chunk
    errors = list(validator.iter_errors(chunk))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


@pytest.mark.parametrize(
    "level",
    ["remember", "understand", "apply", "analyze", "evaluate", "create"],
)
def test_chunk_with_canonical_bloom_secondary_validates(level):
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["bloom_level"] = "create"
    chunk["bloom_level_secondary"] = level
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"level={level} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_chunk_with_invalid_bloom_secondary_rejected():
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["bloom_level_secondary"] = "synthesize"  # not canonical
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for non-canonical bloom_level_secondary"


def test_chunk_with_compound_bloom_secondary_rejected():
    """The split lands a single canonical level; compound values stay banned."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["bloom_level_secondary"] = "remember-apply"
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for compound bloom_level_secondary"
