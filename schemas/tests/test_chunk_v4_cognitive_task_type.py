"""GPT Feedback (May 12) item 5 — chunk_v4 schema test for the new optional
``cognitive_task_type`` field at both chunk top-level and inside the
misconception sub-schema.

The chunk root carries ``additionalProperties: true`` so legacy chunks accept
the new field silently — but this test pins:

  * Chunks WITHOUT the field validate (legacy / back-compat path).
  * Chunks WITH the field populated validate (each of the 15 canonical task
    verbs).
  * Misconception sub-objects accept ``cognitive_task_type`` on the same
    canonical enum.
  * Non-canonical task verbs (e.g. ``"frobnicate"``) fail validation — proves
    the enum is wired through the $ref to the taxonomy schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"

CANONICAL_TASK_VERBS = [
    "define",
    "identify",
    "explain",
    "summarize",
    "classify",
    "compare",
    "apply",
    "compute",
    "analyze",
    "debug",
    "predict",
    "infer",
    "critique",
    "evaluate",
    "construct",
]


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


def test_legacy_chunk_without_cognitive_task_type_validates():
    """Pre-cognitive-task-type chunks (no field at all) must validate."""
    validator = _build_validator()
    chunk = _base_chunk()
    assert "cognitive_task_type" not in chunk
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


@pytest.mark.parametrize("verb", CANONICAL_TASK_VERBS)
def test_chunk_with_canonical_cognitive_task_type_validates(verb):
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["cognitive_task_type"] = verb
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"verb={verb!r} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_chunk_with_non_enum_cognitive_task_type_fails():
    """Invalid task verb (not in 15-value canonical list) must fail."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["cognitive_task_type"] = "frobnicate"
    errors = list(validator.iter_errors(chunk))
    assert errors, "expected validation error for non-canonical cognitive_task_type"


def test_misconception_with_cognitive_task_type_validates():
    """Misconception sub-object accepts cognitive_task_type."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["misconceptions"] = [
        {
            "misconception": "All birds can fly.",
            "correction": "Penguins and ostriches are birds that cannot fly.",
            "bloom_level": "understand",
            "cognitive_task_type": "classify",
        }
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_misconception_without_cognitive_task_type_validates():
    """Legacy misconception (no cognitive_task_type field) must still pass."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["misconceptions"] = [
        {
            "misconception": "All birds can fly.",
            "correction": "Penguins and ostriches are birds that cannot fly.",
        }
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


def test_misconception_with_non_enum_cognitive_task_type_fails():
    """Invalid task verb on misconception sub-object must fail."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["misconceptions"] = [
        {
            "misconception": "All birds can fly.",
            "correction": "Penguins and ostriches are birds.",
            "cognitive_task_type": "frobnicate",
        }
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors, (
        "expected validation error for non-canonical misconception "
        "cognitive_task_type"
    )
