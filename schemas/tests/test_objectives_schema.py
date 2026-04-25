"""Wave 75 — regression tests for objectives_v1.schema.json.

The schema is the canonical contract for the ``objectives.json``
sidecar emitted alongside ``course.json`` in LibV2 archives. It
declares the full TO-/CO- hierarchy so chunk
``learning_outcome_refs`` can resolve against ALL outcomes (not just
the 7 terminal ones declared on course.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "objectives_v1.schema.json"
)


def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _minimal_doc() -> dict:
    return {
        "schema_version": "v1",
        "course_code": "TEST_101",
        "terminal_outcomes": [
            {"id": "to-01", "statement": "Recall the basics."}
        ],
        "component_objectives": [
            {
                "id": "co-01",
                "parent_terminal": "to-01",
                "statement": "Identify foundational terms.",
                "week": 1,
            }
        ],
        "objective_count": {"terminal": 1, "component": 1},
    }


@pytest.mark.unit
def test_schema_file_exists_and_is_valid_json():
    """The schema itself must be parseable JSON Schema."""
    assert SCHEMA_PATH.exists(), f"schema missing: {SCHEMA_PATH}"
    schema = _load_schema()
    assert schema.get("title") == "Objectives (LibV2 objectives.json)"
    assert schema["properties"]["schema_version"]["const"] == "v1"


@pytest.mark.unit
def test_minimal_objectives_doc_validates():
    """A minimal Wave-75 doc with one TO + one CO must validate."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.validate(_minimal_doc(), schema)


@pytest.mark.unit
def test_schema_rejects_missing_required_field():
    """Missing ``objective_count`` is a hard schema failure."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_doc()
    doc.pop("objective_count")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_schema_rejects_bad_lo_id_pattern():
    """LO IDs that don't match ``^[a-zA-Z]{2,}-\\d{2,}$`` fail."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_doc()
    doc["component_objectives"][0]["id"] = "co_01"  # underscore not hyphen
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_schema_accepts_optional_fields():
    """Optional fields (bloom_level, source_refs, weeks) all validate."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_doc()
    doc["terminal_outcomes"][0].update({
        "bloom_level": "remember",
        "bloom_verb": "recall",
        "cognitive_domain": "factual",
        "weeks": [1, 2],
    })
    doc["component_objectives"][0].update({
        "bloom_level": "remember",
        "bloom_verb": "identify",
        "cognitive_domain": "factual",
        "source_refs": ["dart:foo#s1"],
    })
    jsonschema.validate(doc, schema)


@pytest.mark.unit
def test_schema_rejects_unknown_bloom_level():
    """``bloom_level`` enum is constrained — typos fail closed."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    doc = _minimal_doc()
    doc["terminal_outcomes"][0]["bloom_level"] = "memorise"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
