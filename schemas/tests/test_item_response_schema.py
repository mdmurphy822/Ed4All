"""item_response.schema.json — response-ingestion seam contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "assessment" / "item_response.schema.json"


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_valid_record_with_chunk_id():
    v = _validator()
    rec = {"chunk_id": "c1", "learner_id": "L1", "correct": True}
    assert list(v.iter_errors(rec)) == []


def test_valid_record_with_item_id_and_optionals():
    v = _validator()
    rec = {"item_id": "q1", "learner_id": "L2", "correct": False,
           "timestamp": "2026-06-30T00:00:00Z", "score": 0.5}
    assert list(v.iter_errors(rec)) == []


def test_missing_correct_rejected():
    v = _validator()
    rec = {"chunk_id": "c1", "learner_id": "L1"}
    assert list(v.iter_errors(rec))


def test_missing_both_ids_rejected():
    v = _validator()
    rec = {"learner_id": "L1", "correct": True}
    assert list(v.iter_errors(rec))


def test_non_boolean_correct_rejected():
    v = _validator()
    rec = {"chunk_id": "c1", "learner_id": "L1", "correct": "yes"}
    assert list(v.iter_errors(rec))


def test_additional_properties_rejected():
    v = _validator()
    rec = {"chunk_id": "c1", "learner_id": "L1", "correct": True, "rogue": 1}
    assert list(v.iter_errors(rec))
