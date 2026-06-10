"""Phase IA WS3 (D4.1) — regression tests for refusal_probes.schema.json.

Accept/reject snapshots per the schema-fixture policy
(schemas/tests/fixtures/ws3-retrieval/). The schema is the canonical contract
for the per-course refusal-probe set archived at
LibV2/courses/<slug>/retrieval_eval/refusal_probes.json — UNANSWERABLE
questions a correct grounded-answer pipeline must refuse, each with VERIFIED
(not assumed) retrieval-dry-run evidence. Reject cases mutate the known-good
minimal fixture so each test pins exactly one constraint.

Also validates Executor E6's mini-course fixture probe set
(tests/fixtures/retrieval/mini_course/retrieval_eval/refusal_probes.json)
against this schema — the cross-executor / fixture handoff check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "retrieval" / "refusal_probes.schema.json"
)
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ws3-retrieval"
VALID_FIXTURE = FIXTURE_DIR / "valid_minimal_refusal_probes.json"
MINI_COURSE_PROBES = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
    / "retrieval_eval"
    / "refusal_probes.json"
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
    assert schema.get("title") == "Retrieval Refusal Probe Set"


def test_valid_minimal_doc_validates():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_valid_doc(), _load_schema())


def test_mini_course_fixture_probes_validate():
    """The hand-authored mini-course probe set must validate against the schema."""
    jsonschema = pytest.importorskip("jsonschema")
    assert MINI_COURSE_PROBES.exists(), f"mini-course probes missing: {MINI_COURSE_PROBES}"
    with MINI_COURSE_PROBES.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    jsonschema.validate(doc, _load_schema())


# ---------------------------------------------------------------- reject cases


def test_unknown_category_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["category"] = "trivia"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_probe_id_pattern_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["probe_id"] = "p-demo-1"  # wrong prefix + not 4 digits
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_short_why_unanswerable_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["why_unanswerable"] = "too short"  # < 20 chars
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_missing_dry_run_rejected():
    """A probe without dry-run evidence is unverified — rejected (verified-not-assumed)."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    del doc["probes"][0]["dry_run"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_dry_run_verified_false_rejected():
    """verified must be const true — an unrun dry-run is not evidence."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_run"]["verified"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_dry_run_top_passage_answers_true_rejected():
    """top_passage_answers=true contradicts 'unanswerable' — schema forbids it."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_run"]["top_passage_answers"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_dry_run_missing_top_score_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    del doc["probes"][0]["dry_run"]["top_score"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_negative_top_score_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_run"]["top_score"] = -1.0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_short_question_text_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["question_text"] = "short"  # < 10 chars
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_empty_probes_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_wrong_schema_version_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["schema_version"] = "2.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_additional_top_level_property_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["unexpected_field"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_additional_probe_property_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["unexpected"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_bad_derived_from_gold_id_pattern_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["derived_from_gold_question_id"] = "q-1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_optional_fields_omittable():
    """engine, _operator_note, derived_from_gold_question_id, authoring optional."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc.pop("engine", None)
    doc.pop("_operator_note", None)
    doc["probes"][0].pop("derived_from_gold_question_id", None)
    doc["probes"][0].pop("authoring", None)
    jsonschema.validate(doc, _load_schema())
