"""Phase IA WS1.4 — regression tests for schemas/retrieval/gold_set.schema.json.

Accept/reject snapshots per the schema-fixture policy
(schemas/tests/fixtures/ws1-retrieval/). The schema is the canonical contract
for the per-course retrieval-eval gold set archived at
LibV2/courses/<slug>/retrieval_eval/gold_set.json. Reject cases mutate the
known-good minimal fixture so each test pins exactly one constraint.

Also validates Executor A's mini-course fixture gold set
(tests/fixtures/retrieval/mini_course/retrieval_eval/gold_set.json) against
this schema — the cross-executor handoff check.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "retrieval" / "gold_set.schema.json"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ws1-retrieval"
VALID_FIXTURE = FIXTURE_DIR / "valid_minimal_gold_set.json"
MINI_COURSE_GOLD = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
    / "retrieval_eval"
    / "gold_set.json"
)


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
    assert schema.get("title") == "Retrieval Eval Gold Set"


def test_valid_minimal_doc_validates():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_valid_doc(), _load_schema())


def test_mini_course_fixture_gold_set_validates():
    """Cross-executor handoff: Executor A's hand-authored mini-course gold set
    must validate against Executor B's authoritative schema."""
    jsonschema = pytest.importorskip("jsonschema")
    assert MINI_COURSE_GOLD.exists(), f"mini-course gold set missing: {MINI_COURSE_GOLD}"
    with MINI_COURSE_GOLD.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    jsonschema.validate(doc, _load_schema())


# ---------------------------------------------------------------- reject cases


def test_unknown_question_type_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["question_type"] = "trivia"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_empty_relevant_passages_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["relevant_passages"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_no_primary_passage_rejected():
    """At least one passage must have relevance:primary (contains constraint)."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["relevant_passages"][0]["relevance"] = "supporting"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_short_text_quote_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["relevant_passages"][0]["anchor"]["text_quote"] = "too short"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_question_id_pattern_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["question_id"] = "q-demo-1"  # wrong prefix + not 4 digits
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_missing_chunkset_pin_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    del doc["chunkset"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_chunks_sha256_pattern_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["chunkset"]["chunks_sha256"] = "NOTAHEX"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_invalid_chunkset_kind_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["chunkset"]["kind"] = "pdf"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_wrong_schema_version_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["schema_version"] = "2.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_empty_questions_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_additional_top_level_property_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["unexpected_field"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_short_question_text_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["questions"][0]["question_text"] = "short"  # < 10 chars
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_operator_note_optional_string_accepted():
    """_operator_note is permitted (seed files carry the scale-up flag)."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["_operator_note"] = "scale-up to 50 + freeze is an operator task"
    jsonschema.validate(doc, _load_schema())
