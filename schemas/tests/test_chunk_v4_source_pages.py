"""chunk_v4 schema test for the new optional top-level ``source_pages`` field
(Task #26 — source page-number provenance on SemantiK chunks).

``source_pages`` is the sorted, deduped union of the page numbers the composing
SemantiK blocks carried (``data-semantik-pages``), emitted at the chunk TOP
LEVEL (deliberately NOT inside ``source``, whose
``$defs.Source`` is ``additionalProperties: false``). This test pins the SCHEMA
contract:

  * A chunk WITH ``source_pages=[62, 63]`` validates.
  * A legacy chunk OMITTING the field validates (back-compat — root carries
    ``additionalProperties: true`` and the field is not required).
  * ``source_pages=[0]`` fails (``minimum: 1``).
  * ``source_pages=[]`` fails (``minItems: 1`` — the emitter omits the field
    rather than writing an empty array).
  * A non-integer item fails.
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
    """Minimal chunk carrying every key in the schema's ``required`` list."""
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


def test_chunk_with_source_pages_validates():
    """A chunk carrying a populated ``source_pages`` int array validates."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["source_pages"] = [62, 63]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_legacy_chunk_without_source_pages_validates():
    """A pre-Task-#26 chunk omitting ``source_pages`` validates untouched."""
    validator = _build_validator()
    chunk = _base_chunk()
    assert "source_pages" not in chunk
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_source_pages_with_zero_fails():
    """``source_pages=[0]`` violates ``minimum: 1`` (page numbers are 1-based)."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["source_pages"] = [0]
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for a page number < 1"


def test_empty_source_pages_fails():
    """``source_pages=[]`` violates ``minItems: 1`` — the emitter OMITS the
    field when the union is empty rather than writing an empty array."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["source_pages"] = []
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for an empty source_pages array"


def test_non_integer_source_pages_item_fails():
    """A non-integer ``source_pages`` item fails the ``integer`` item type."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["source_pages"] = [62, "63"]
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for a non-integer page token"
