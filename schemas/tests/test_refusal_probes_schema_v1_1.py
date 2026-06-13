"""retrieval-answer-eval-set P0 — regression tests for
schemas/retrieval/refusal_probes.schema.v1_1.json.

v1.0 stays const-pinned/untouched; these tests pin the v1.1 additive shape:
the optional plural dry_runs[] (per-engine evidence) alongside the still-
required single dry_run, with the same verified/top_passage_answers const
contract on each entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "retrieval" / "refusal_probes.schema.v1_1.json"
V1_0_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "retrieval" / "refusal_probes.schema.json"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ws3-retrieval"
VALID_FIXTURE = FIXTURE_DIR / "valid_minimal_refusal_probes_v1_1.json"


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
    assert schema.get("title") == "Retrieval Refusal Probe Set (v1.1)"
    assert schema["properties"]["schema_version"]["const"] == "1.1"


def test_v1_0_schema_unchanged_const_pinned():
    with V1_0_SCHEMA_PATH.open(encoding="utf-8") as fh:
        v1_0 = json.load(fh)
    assert v1_0["properties"]["schema_version"]["const"] == "1.0"
    # v1.0 has no plural dry_runs on the probe item.
    assert "dry_runs" not in v1_0["properties"]["probes"]["items"]["properties"]


def test_valid_minimal_doc_validates():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_valid_doc(), _load_schema())


def test_dry_runs_has_three_engines():
    doc = _valid_doc()
    engines = {dr["engine"] for dr in doc["probes"][0]["dry_runs"]}
    assert engines == {"lexical", "semantic", "hybrid-rrf"}


# ---------------------------------------------------------------- reject cases


def test_single_dry_run_still_required():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    del doc["probes"][0]["dry_run"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_dry_runs_entry_top_passage_answers_true_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_runs"][1]["top_passage_answers"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_dry_runs_entry_unverified_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_runs"][0]["verified"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_dry_runs_entry_bad_engine_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_runs"][0]["engine"] = "tfidf"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_empty_dry_runs_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["probes"][0]["dry_runs"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_wrong_schema_version_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["schema_version"] = "1.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())


def test_additional_top_level_property_rejected():
    jsonschema = pytest.importorskip("jsonschema")
    doc = _valid_doc()
    doc["unexpected"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _load_schema())
