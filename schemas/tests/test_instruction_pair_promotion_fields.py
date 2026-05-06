"""GPT Feedback v2 Wave 1 / W1.B — instruction_pair promotion-fields tests.

Pins:

  * Legacy pairs (no promotion fields) validate on lenient AND strict.
  * New pair with all 7 optional promotion fields validates.
  * promotion_status == "trainable" WITHOUT source_chunk_id fails (the if/then
    clause's whole point).
  * promotion_status == "trainable" WITH source_chunk_id validates.
  * promotion_status == "pending" WITHOUT source_chunk_id validates (the
    conditional doesn't fire).
  * Invalid enum values fail (promotion_status == "promoted",
    observed_bloom == "synthesize").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
LENIENT_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "instruction_pair.schema.json"
STRICT_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "instruction_pair.strict.schema.json"


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _load_validator(schema_path: Path):
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = json.loads(schema_path.read_text())
    return Draft202012Validator(schema)


def _base_pair() -> Dict[str, Any]:
    return {
        "prompt": "Explain how RDF triples encode subject-predicate-object relationships in 2-3 sentences.",
        "completion": (
            "An RDF triple binds a subject to an object via a named predicate, encoding a directed edge "
            "in a knowledge graph. The subject and predicate are IRIs; the object is an IRI or literal."
        ),
        "chunk_id": "test_course_chunk_00042",
        "lo_refs": ["TO-01", "CO-03"],
        "bloom_level": "understand",
        "content_type": "explanation",
        "seed": 12345,
        "decision_capture_id": "evt_abcdef0123456789",
    }


# ---------------------------------------------------------------------------
# Legacy compatibility (no promotion fields).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_legacy_pair_without_promotion_fields_validates(schema_path):
    """Pre-Wave-1 instruction pairs (no promotion fields) must validate on
    BOTH lenient AND strict schemas."""
    validator = _load_validator(schema_path)
    pair = _base_pair()
    for key in (
        "promotion_status",
        "rejection_reason",
        "observed_bloom",
        "bloom_alignment",
        "answer_support_score",
        "rationale_richness_score",
        "source_chunk_id",
    ):
        assert key not in pair
    errors = list(validator.iter_errors(pair))
    assert errors == [], (
        f"unexpected errors on {schema_path.name}: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------------
# All seven new optional fields populated.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_pair_with_all_seven_promotion_fields_validates(schema_path):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair.update(
        {
            "promotion_status": "validated",
            "rejection_reason": None,
            "observed_bloom": "understand",
            "bloom_alignment": True,
            "answer_support_score": 0.82,
            "rationale_richness_score": 0.71,
            "source_chunk_id": "test_course_chunk_00042",
        }
    )
    errors = list(validator.iter_errors(pair))
    assert errors == [], (
        f"unexpected errors on {schema_path.name}: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------------
# if/then clause — source_chunk_id required when promotion_status == "trainable".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_trainable_without_source_chunk_id_fails(schema_path):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["promotion_status"] = "trainable"
    # source_chunk_id intentionally omitted — the if/then clause must fire.
    errors = list(validator.iter_errors(pair))
    assert errors, (
        f"expected if/then to require source_chunk_id when promotion_status=='trainable' "
        f"on {schema_path.name}"
    )


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_trainable_with_source_chunk_id_validates(schema_path):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["promotion_status"] = "trainable"
    pair["source_chunk_id"] = "test_course_chunk_00042"
    errors = list(validator.iter_errors(pair))
    assert errors == [], (
        f"unexpected errors on {schema_path.name}: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_pending_without_source_chunk_id_validates(schema_path):
    """promotion_status == 'pending' MUST NOT trigger the if/then clause."""
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["promotion_status"] = "pending"
    # source_chunk_id intentionally omitted; the conditional must not fire.
    errors = list(validator.iter_errors(pair))
    assert errors == [], (
        f"unexpected errors on {schema_path.name}: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------------
# Enum scope — invalid values fail.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_invalid_promotion_status_fails(schema_path):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["promotion_status"] = "promoted"  # NOT in enum
    errors = list(validator.iter_errors(pair))
    assert errors, (
        f"expected validation error for promotion_status=='promoted' on {schema_path.name}"
    )


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_invalid_observed_bloom_fails(schema_path):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["observed_bloom"] = "synthesize"  # NOT in enum
    errors = list(validator.iter_errors(pair))
    assert errors, (
        f"expected validation error for observed_bloom=='synthesize' on {schema_path.name}"
    )


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
@pytest.mark.parametrize(
    "level",
    ["remember", "understand", "apply", "analyze", "evaluate", "create"],
)
def test_canonical_observed_bloom_validates(schema_path, level):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["observed_bloom"] = level
    errors = list(validator.iter_errors(pair))
    assert errors == [], (
        f"level={level} unexpectedly errored on {schema_path.name}: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
def test_null_observed_bloom_validates(schema_path):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["observed_bloom"] = None
    pair["bloom_alignment"] = None
    errors = list(validator.iter_errors(pair))
    assert errors == [], (
        f"unexpected errors on {schema_path.name}: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------------
# Numeric range constraints.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_out_of_range_answer_support_score_fails(schema_path, score):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["answer_support_score"] = score
    errors = list(validator.iter_errors(pair))
    assert errors, (
        f"expected validation error for answer_support_score={score} on {schema_path.name}"
    )


@pytest.mark.parametrize("schema_path", [LENIENT_SCHEMA_PATH, STRICT_SCHEMA_PATH])
@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_out_of_range_rationale_richness_score_fails(schema_path, score):
    validator = _load_validator(schema_path)
    pair = _base_pair()
    pair["rationale_richness_score"] = score
    errors = list(validator.iter_errors(pair))
    assert errors, (
        f"expected validation error for rationale_richness_score={score} on {schema_path.name}"
    )
