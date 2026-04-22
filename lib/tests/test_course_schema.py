"""Tests for schemas/knowledge/course.schema.json (Wave 24).

Covers the canonical shape of Trainforge-emitted course.json:
  * Real shapes produced by _build_course_json validate.
  * Missing required keys fail schema validation.
  * Drift shapes (e.g. old {COURSE}_OBJ_N IDs) fail validation.
  * bloom_level enum rejects unknown levels.
  * hierarchy_level enum rejects unknown values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "schemas" / "knowledge" / "course.schema.json"


def _load_schema() -> dict:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _valid_minimal_course() -> dict:
    return {
        "course_code": "PHYS_101",
        "title": "Intro Physics",
        "learning_outcomes": [
            {
                "id": "to-01",
                "statement": "Describe the laws of motion.",
                "hierarchy_level": "terminal",
                "bloom_level": "understand",
            },
            {
                "id": "co-01",
                "statement": "Define the concept of inertia.",
                "hierarchy_level": "chapter",
                "bloom_level": "remember",
            },
        ],
    }


def test_minimal_valid_course_passes():
    schema = _load_schema()
    jsonschema.validate(_valid_minimal_course(), schema)


def test_missing_course_code_fails():
    schema = _load_schema()
    data = _valid_minimal_course()
    del data["course_code"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)


def test_learning_outcomes_can_be_empty_list():
    """Empty LO list is acceptable — schema doesn't require populated ones."""
    schema = _load_schema()
    data = _valid_minimal_course()
    data["learning_outcomes"] = []
    jsonschema.validate(data, schema)


def test_phantom_obj_id_shape_rejected():
    """Pre-Wave-24 {COURSE}_OBJ_N phantoms fail the id pattern."""
    schema = _load_schema()
    data = _valid_minimal_course()
    data["learning_outcomes"][0]["id"] = "PHYS_101_OBJ_1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)


def test_unknown_hierarchy_rejected():
    schema = _load_schema()
    data = _valid_minimal_course()
    data["learning_outcomes"][0]["hierarchy_level"] = "module"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)


def test_unknown_bloom_level_rejected():
    schema = _load_schema()
    data = _valid_minimal_course()
    data["learning_outcomes"][0]["bloom_level"] = "memorize"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)
