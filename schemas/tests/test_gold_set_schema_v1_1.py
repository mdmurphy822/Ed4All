"""retrieval-answer-eval-set P0 — regression tests for
schemas/retrieval/gold_set.schema.v1_1.json.

Accept/reject snapshots per the schema-fixture policy
(schemas/tests/fixtures/ws1-retrieval/). v1.0 stays untouched/const-pinned;
these tests pin the v1.1 additive shape: procedural + multi_part question_types,
the parts[]-required-iff-multi_part conditional, absence_note-required-iff-
uncovered, the new optional fields, and the anchor.content_sha256 pattern.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "retrieval" / "gold_set.schema.v1_1.json"
V1_0_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "retrieval" / "gold_set.schema.json"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ws1-retrieval"
VALID_FIXTURE = FIXTURE_DIR / "valid_minimal_gold_set_v1_1.json"


def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _valid_doc() -> dict:
    with VALID_FIXTURE.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_schema_file_exists_and_is_valid_draft202012():
    assert SCHEMA_PATH.exists(), f"schema missing: {SCHEMA_PATH}"
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema.get("title") == "Retrieval Eval Gold Set (v1.1)"
    assert schema["properties"]["schema_version"]["const"] == "1.1"


def test_v1_0_schema_unchanged_const_pinned():
    """The v1.0 schema file remains const-pinned to 1.0 (untouched)."""
    with V1_0_SCHEMA_PATH.open(encoding="utf-8") as fh:
        v1_0 = json.load(fh)
    assert v1_0["properties"]["schema_version"]["const"] == "1.0"
    # v1.0 must NOT carry the new question types.
    assert set(v1_0["properties"]["questions"]["items"]["properties"]
               ["question_type"]["enum"]) == {
        "factual_recall", "conceptual_synthesis", "where_covered"
    }


def test_valid_minimal_doc_validates():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_valid_doc(), _load_schema())


def test_procedural_and_multi_part_accepted():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    types = {q["question_type"] for q in doc["questions"]}
    assert {"procedural", "multi_part"} <= types
    jsonschema.validate(doc, _load_schema())


# ---------------------------------------------------------------- reject cases


def test_multi_part_without_parts_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    # question[1] is the multi_part one — drop its parts.
    del doc["questions"][1]["parts"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_uncovered_part_without_absence_note_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    # part 'c' is covered:false — strip its absence_note.
    parts = doc["questions"][1]["parts"]
    uncovered = next(p for p in parts if p["covered"] is False)
    del uncovered["absence_note"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_content_sha256_pattern_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["relevant_passages"][0]["anchor"]["content_sha256"] = "NOTAHEX"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_difficulty_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["difficulty"] = "trivial"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_population_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["expected_citation_population"] = "external"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_expected_key_points_too_many_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["expected_key_points"] = [f"point {i}" for i in range(7)]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_wrong_schema_version_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["schema_version"] = "1.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_objective_ref_pattern_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["objective_refs"] = ["NOT_AN_LO"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_additional_top_level_property_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["unexpected"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_synthesis_scope_enum_enforced():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][1]["synthesis_scope"] = "between_chapters"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())
