"""Worker T — regression tests for REC-VOC-03 Phase 2.

Tests the opt-in content_type enforcement layer behind
``TRAINFORGE_ENFORCE_CONTENT_TYPE=true``:

- Default (flag off): any non-empty string passes — backward-compat with the
  existing free-string instruction_pair.content_type field.
- Flag on: values must be members of the ChunkType enum from
  ``schemas/taxonomies/content_type.json#/$defs/ChunkType``.
- Strict schema variant (``instruction_pair.strict.schema.json``) rejects
  free-string values; the default schema continues to accept them.

Follows the established pattern from ``Trainforge/tests/test_chunk_validation.py``
for env-var gated behavior.
"""

from __future__ import annotations

import json

import pytest

from lib.paths import SCHEMAS_PATH
from lib.validators.content_type import (
    assert_chunk_type,
    get_valid_chunk_types,
    get_valid_section_content_types,
    validate_chunk_type,
    validate_section_content_type,
)

ENV_VAR = "TRAINFORGE_ENFORCE_CONTENT_TYPE"


# ---------------------------------------------------------------------------
# Flag-off (default) — backward-compat: any string accepted.
# ---------------------------------------------------------------------------


def test_flag_off_accepts_any_string(monkeypatch):
    """Default behavior: validate_chunk_type returns True for any string."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert validate_chunk_type("foobar") is True
    assert validate_chunk_type("") is True
    assert validate_chunk_type("not-in-any-enum") is True
    # assert_chunk_type is a no-op when flag off
    assert_chunk_type("foobar")
    assert_chunk_type("")


def test_flag_explicitly_false_accepts_any_string(monkeypatch):
    """Explicit false value also disables enforcement."""
    monkeypatch.setenv(ENV_VAR, "false")
    assert validate_chunk_type("foobar") is True
    assert_chunk_type("definitely_not_a_chunk_type")


def test_flag_off_accepts_section_content_type_anything(monkeypatch):
    """SectionContentType helper is also opt-in."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert validate_section_content_type("anything") is True


# ---------------------------------------------------------------------------
# Flag-on — ChunkType enum enforcement.
# ---------------------------------------------------------------------------


def test_flag_on_accepts_valid_chunk_type(monkeypatch):
    """All ChunkType enum members pass when flag on."""
    monkeypatch.setenv(ENV_VAR, "true")
    for value in get_valid_chunk_types():
        assert validate_chunk_type(value) is True, f"{value!r} should be valid"
        # assert variant does not raise
        assert_chunk_type(value)


def test_flag_on_rejects_invalid_chunk_type(monkeypatch):
    """Non-enum values fail when flag on."""
    monkeypatch.setenv(ENV_VAR, "true")
    assert validate_chunk_type("foobar") is False
    assert validate_chunk_type("") is False
    with pytest.raises(ValueError, match="foobar"):
        assert_chunk_type("foobar")


def test_flag_on_assert_chunk_type_includes_context(monkeypatch):
    """assert_chunk_type surfaces the context hint in the error."""
    monkeypatch.setenv(ENV_VAR, "true")
    with pytest.raises(ValueError, match="chunk_id=foo_42"):
        assert_chunk_type("bogus", context="chunk_id=foo_42")


def test_flag_on_rejects_callout_content_types(monkeypatch):
    """CalloutContentType values are NOT valid ChunkTypes (union discriminator)."""
    monkeypatch.setenv(ENV_VAR, "true")
    # `application-note` and `note` are CalloutContentType-only.
    assert validate_chunk_type("application-note") is False
    assert validate_chunk_type("note") is False


def test_flag_on_rejects_section_only_content_types(monkeypatch):
    """SectionContentType members that aren't in ChunkType also reject."""
    monkeypatch.setenv(ENV_VAR, "true")
    # `definition`, `procedure`, `comparison` are SectionContentType-only.
    assert validate_chunk_type("definition") is False
    assert validate_chunk_type("procedure") is False
    assert validate_chunk_type("comparison") is False


def test_flag_on_section_content_type_enforcement(monkeypatch):
    """validate_section_content_type enforces SectionContentType enum."""
    monkeypatch.setenv(ENV_VAR, "true")
    assert validate_section_content_type("definition") is True
    assert validate_section_content_type("explanation") is True
    # `assessment_item` is ChunkType-only, NOT in SectionContentType.
    assert validate_section_content_type("assessment_item") is False
    assert validate_section_content_type("foobar") is False


# ---------------------------------------------------------------------------
# Enum accessors — guard against upstream taxonomy drift.
# ---------------------------------------------------------------------------


def test_get_valid_chunk_types_returns_expected_set():
    """ChunkType enum matches Worker F's Wave 1 taxonomy publication."""
    assert get_valid_chunk_types() == frozenset({
        "assessment_item",
        "overview",
        "summary",
        "exercise",
        "explanation",
        "example",
    })


def test_get_valid_section_content_types_returns_expected_set():
    """SectionContentType enum matches Worker F's Wave 1 taxonomy publication."""
    assert get_valid_section_content_types() == frozenset({
        "definition",
        "example",
        "procedure",
        "comparison",
        "exercise",
        "overview",
        "summary",
        "explanation",
    })


def test_enum_accessors_return_frozensets():
    """Immutable return type prevents accidental mutation at call sites."""
    assert isinstance(get_valid_chunk_types(), frozenset)
    assert isinstance(get_valid_section_content_types(), frozenset)


# ---------------------------------------------------------------------------
# Schema-level validation — strict variant rejects free strings;
# default schema stays permissive (legacy records still validate).
# ---------------------------------------------------------------------------


def _load_schema(name: str) -> dict:
    path = SCHEMAS_PATH / "knowledge" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_instruction_pair_record(content_type: str) -> dict:
    """Construct a minimal valid instruction_pair record."""
    return {
        "prompt": (
            "Describe the primary pedagogical purpose of this concept in your own words, "
            "grounding your answer in the key terms introduced in the source material."
        ),
        "completion": (
            "The concept functions as a foundational scaffold: students who master it can apply "
            "it to new contexts, which is why the module emphasizes it early."
        ),
        "chunk_id": "test_course_chunk_00001",
        "lo_refs": ["LO-1.1"],
        "bloom_level": "understand",
        "content_type": content_type,
        "seed": 17,
        "decision_capture_id": "event_000001",
    }


def test_legacy_instruction_pairs_still_validate_default():
    """Existing free-string content_type values pass the default schema."""
    pytest.importorskip("jsonschema")
    from jsonschema import validate

    schema = _load_schema("instruction_pair.schema.json")
    record = _valid_instruction_pair_record(content_type="random_free_string")
    # Default schema: free-string, just minLength=1. Passes.
    validate(record, schema)


def test_strict_schema_rejects_free_string_content_type():
    """Strict schema rejects free-string values not in the ChunkType enum."""
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError, validate

    schema = _load_schema("instruction_pair.strict.schema.json")
    record = _valid_instruction_pair_record(content_type="random_free_string")
    with pytest.raises(ValidationError):
        validate(record, schema)


def test_strict_schema_accepts_valid_chunk_type():
    """Strict schema accepts any ChunkType enum member."""
    pytest.importorskip("jsonschema")
    from jsonschema import validate

    schema = _load_schema("instruction_pair.strict.schema.json")
    for value in get_valid_chunk_types():
        record = _valid_instruction_pair_record(content_type=value)
        validate(record, schema)  # should not raise


def test_strict_schema_enum_matches_chunk_type_source_of_truth():
    """Inlined enum in strict schema stays in sync with content_type.json."""
    strict_schema = _load_schema("instruction_pair.strict.schema.json")
    strict_enum = set(strict_schema["properties"]["content_type"]["enum"])
    assert strict_enum == set(get_valid_chunk_types()), (
        "instruction_pair.strict.schema.json content_type enum drifted from "
        "schemas/taxonomies/content_type.json#/$defs/ChunkType. Sync them."
    )
