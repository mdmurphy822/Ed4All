"""Wave 93 — Schema-conformance tests for model_pointers.schema.json.

The pointers file at ``courses/<slug>/models/_pointers.json`` records
the currently-promoted ``model_id`` and an append-only promotion
history. This suite locks down:

- the schema is a valid Draft202012 JSON Schema
- ``current`` is required and may be ``null`` (no model promoted yet)
- ``history`` is required (defaults to ``[]`` on a fresh file)
- each history entry requires ``model_id`` + ``promoted_at``
- ``additionalProperties: false`` is enforced at every nesting level
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "models" / "model_pointers.schema.json"
)


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    return Draft202012Validator(_load_schema())


def _valid_pointers(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "current": "qwen2-5-1-5b-tst-101-3a4f8c92",
        "history": [
            {
                "model_id": "qwen2-5-1-5b-tst-101-3a4f8c92",
                "promoted_at": "2026-04-26T18:30:00Z",
                "promoted_by": "cli:libv2-import-model",
                "demoted_at": None,
            },
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Meta: schema validity
# ---------------------------------------------------------------------------


def test_schema_is_valid_draft_2020_12():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    Draft202012Validator.check_schema(schema)


def test_schema_top_level_shape():
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://ed4all.dev/schemas/models/model_pointers.schema.json"
    )
    assert schema["title"] == "ModelPointers"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"current", "history"}


def test_history_item_additional_properties_false():
    schema = _load_schema()
    items = schema["properties"]["history"]["items"]
    assert items["additionalProperties"] is False
    assert set(items["required"]) == {"model_id", "promoted_at"}


# ---------------------------------------------------------------------------
# Positive: round-trips + nullable current
# ---------------------------------------------------------------------------


def test_fully_populated_round_trips():
    validator = _validator()
    data = _valid_pointers()
    serialized = json.dumps(data)
    rehydrated = json.loads(serialized)
    errors = list(validator.iter_errors(rehydrated))
    assert errors == [], [e.message for e in errors]
    assert rehydrated == data


def test_current_may_be_null_with_empty_history():
    """A fresh pointers file before any model is promoted."""
    validator = _validator()
    data = {"current": None, "history": []}
    errors = list(validator.iter_errors(data))
    assert errors == [], [e.message for e in errors]


def test_history_entry_can_omit_optional_fields():
    """``promoted_by`` + ``demoted_at`` are optional."""
    validator = _validator()
    data = {
        "current": "m1",
        "history": [
            {"model_id": "m1", "promoted_at": "2026-04-26T18:30:00Z"},
        ],
    }
    errors = list(validator.iter_errors(data))
    assert errors == [], [e.message for e in errors]


def test_demoted_history_entry_round_trips():
    validator = _validator()
    data = {
        "current": "m2",
        "history": [
            {
                "model_id": "m1",
                "promoted_at": "2026-04-20T10:00:00Z",
                "promoted_by": "user@host",
                "demoted_at": "2026-04-26T18:30:00Z",
            },
            {
                "model_id": "m2",
                "promoted_at": "2026-04-26T18:30:00Z",
                "promoted_by": "user@host",
                "demoted_at": None,
            },
        ],
    }
    errors = list(validator.iter_errors(data))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# Negative: required fields
# ---------------------------------------------------------------------------


def test_missing_current_field_fails():
    validator = _validator()
    data = _valid_pointers()
    del data["current"]
    errors = list(validator.iter_errors(data))
    assert errors, "Missing current must fail validation"


def test_missing_history_field_fails():
    validator = _validator()
    data = _valid_pointers()
    del data["history"]
    errors = list(validator.iter_errors(data))
    assert errors, "Missing history must fail validation"


def test_history_entry_missing_model_id_fails():
    validator = _validator()
    data = {
        "current": "m1",
        "history": [{"promoted_at": "2026-04-26T18:30:00Z"}],
    }
    errors = list(validator.iter_errors(data))
    assert errors, "history entry missing model_id must fail"


def test_history_entry_missing_promoted_at_fails():
    validator = _validator()
    data = {
        "current": "m1",
        "history": [{"model_id": "m1"}],
    }
    errors = list(validator.iter_errors(data))
    assert errors, "history entry missing promoted_at must fail"


# ---------------------------------------------------------------------------
# Negative: types + extras
# ---------------------------------------------------------------------------


def test_current_string_must_be_non_empty():
    validator = _validator()
    data = _valid_pointers(current="")
    errors = list(validator.iter_errors(data))
    assert errors, "Empty string for current must fail (minLength=1)"


def test_extra_top_level_field_rejected():
    validator = _validator()
    data = _valid_pointers()
    data["unexpected_extra"] = 123
    errors = list(validator.iter_errors(data))
    assert errors, "Extra top-level fields must be rejected"


def test_extra_history_entry_field_rejected():
    validator = _validator()
    data = _valid_pointers()
    data["history"][0]["mystery_param"] = "nope"
    errors = list(validator.iter_errors(data))
    assert errors, "Extra fields on history entries must be rejected"


def test_history_must_be_array():
    validator = _validator()
    data = _valid_pointers()
    data["history"] = {"not": "an array"}
    errors = list(validator.iter_errors(data))
    assert errors, "history must be an array"


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------


def test_load_does_not_mutate_schema():
    a = _load_schema()
    b = _load_schema()
    assert a == b
    snapshot = copy.deepcopy(a)
    list(_validator().iter_errors(_valid_pointers()))
    assert a == snapshot
