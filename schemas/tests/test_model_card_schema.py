"""Wave 89 — Schema-conformance tests for the canonical ModelCard.

The ``schemas/models/model_card.schema.json`` file is the shared
shape consumed by Trainforge training-run output (Wave 90) and the
LibV2ModelValidator gate (Wave 89). This suite locks down:

- the schema itself is a valid draft-2020-12 JSON Schema
- required fields are enforced (model_id, course_slug, base_model,
  adapter_format, training_config, provenance, created_at)
- ``model_id`` pattern accepts canonical kebab-case and rejects
  malformed variants
- ``adapter_format`` enum is exactly the three allowed values
- ``training_config`` enforces the seven canonical hyperparameters
  with strict additionalProperties=false
- ``provenance`` requires all six SHA-256 hashes and validates each
- optional ``eval_scores`` numbers stay in their range bands
- strict ``additionalProperties: false`` blocks unknown keys at every
  nesting level
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Project root (Ed4All/). This file lives at
# Ed4All/schemas/tests/test_model_card_schema.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "models" / "model_card.schema.json"
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


_HASH64 = "a" * 64
_OTHER_HASH64 = "b" * 64


def _valid_card(**overrides: Any) -> Dict[str, Any]:
    """Return a fully-populated, valid ModelCard. Override any top-level field."""
    base: Dict[str, Any] = {
        "model_id": "qwen2-5-1-5b-tst-101-v1",
        "course_slug": "tst-101",
        "base_model": {
            "name": "qwen2.5-1.5b",
            "revision": "main",
            "huggingface_repo": "Qwen/Qwen2.5-1.5B",
        },
        "adapter_format": "safetensors",
        "training_config": {
            "seed": 42,
            "learning_rate": 2e-4,
            "epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
            "max_seq_length": 2048,
            "batch_size": 4,
        },
        "provenance": {
            "chunks_hash": _HASH64,
            "pedagogy_graph_hash": _OTHER_HASH64,
            "instruction_pairs_hash": _HASH64,
            "preference_pairs_hash": _OTHER_HASH64,
            "concept_graph_hash": _HASH64,
            "vocabulary_ttl_hash": _OTHER_HASH64,
            "holdout_graph_hash": _HASH64,
        },
        "created_at": "2026-04-26T18:30:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Meta: schema is a valid draft-2020-12 JSON Schema
# ---------------------------------------------------------------------------


def test_schema_is_valid_draft_2020_12():
    """Draft202012Validator.check_schema passes on our schema."""
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    Draft202012Validator.check_schema(schema)


def test_schema_top_level_shape():
    """Sanity-check the schema shape to catch accidental drift."""
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == (
        "https://ed4all.dev/schemas/models/model_card.schema.json"
    )
    assert schema["title"] == "ModelCard"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "model_id",
        "course_slug",
        "base_model",
        "adapter_format",
        "training_config",
        "provenance",
        "created_at",
    }


def test_adapter_format_enum_is_locked():
    """Exactly three adapter formats are allowed."""
    schema = _load_schema()
    assert schema["properties"]["adapter_format"]["enum"] == [
        "safetensors", "gguf", "merged_safetensors",
    ]


def test_provenance_requires_six_hashes():
    """All six provenance fields are required (no hash is optional)."""
    schema = _load_schema()
    prov = schema["properties"]["provenance"]
    assert prov["additionalProperties"] is False
    assert set(prov["required"]) == {
        "chunks_hash",
        "pedagogy_graph_hash",
        "instruction_pairs_hash",
        "preference_pairs_hash",
        "concept_graph_hash",
        "vocabulary_ttl_hash",
        "holdout_graph_hash",
    }


def test_training_config_strict_no_extras():
    """training_config blocks unknown keys."""
    schema = _load_schema()
    tc = schema["properties"]["training_config"]
    assert tc["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Positive: minimal valid card + fully populated round-trip
# ---------------------------------------------------------------------------


def test_minimal_valid_card_validates():
    """A required-only card validates."""
    validator = _validator()
    errors = list(validator.iter_errors(_valid_card()))
    assert errors == [], [e.message for e in errors]


def test_fully_populated_card_round_trips():
    """A card with every optional field populated round-trips through
    JSON serialization and re-validates. Catches schema/encoder drift."""
    validator = _validator()
    card = _valid_card(
        eval_scores={
            "faithfulness": 0.83,
            "coverage": 0.91,
            "baseline_delta": 0.12,
            "scoring_commit": "f" * 40,
            "tolerance_band": {
                "accuracy": 0.0,
                "faithfulness": 0.05,
                "hallucination_rate": 0.05,
                "source_match": 0.0,
            },
        },
        license="apache-2.0",
        description="QLoRA SFT+DPO adapter for tst-101 trained on Qwen2.5-1.5B.",
        tags=["qlora", "stem", "physics"],
    )
    serialized = json.dumps(card)
    rehydrated = json.loads(serialized)
    errors = list(validator.iter_errors(rehydrated))
    assert errors == [], [e.message for e in errors]
    # Round-trip also preserves the whole shape
    assert rehydrated == card


@pytest.mark.parametrize(
    "model_id",
    [
        "x",
        "qwen2-5-1-5b-tst-101-v1",
        "smollm2-1-7b-bio-201-v3",
        "phi-3-5-mini-chem-101",
    ],
)
def test_valid_model_id_shapes(model_id):
    validator = _validator()
    errors = list(validator.iter_errors(_valid_card(model_id=model_id)))
    assert errors == [], (
        f"{model_id!r} should be valid: {[e.message for e in errors]}"
    )


@pytest.mark.parametrize(
    "adapter_format",
    ["safetensors", "gguf", "merged_safetensors"],
)
def test_valid_adapter_format_values(adapter_format):
    validator = _validator()
    errors = list(validator.iter_errors(_valid_card(adapter_format=adapter_format)))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# Negative: required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_field",
    [
        "model_id",
        "course_slug",
        "base_model",
        "adapter_format",
        "training_config",
        "provenance",
        "created_at",
    ],
)
def test_missing_required_top_level_field_fails(missing_field):
    validator = _validator()
    card = _valid_card()
    del card[missing_field]
    errors = list(validator.iter_errors(card))
    assert errors, f"Expected failure when {missing_field!r} is missing"


def test_missing_provenance_block_fails():
    """Explicit case: dropping the entire provenance block fails."""
    validator = _validator()
    card = _valid_card()
    del card["provenance"]
    errors = list(validator.iter_errors(card))
    assert errors, "Missing provenance must fail validation"


@pytest.mark.parametrize(
    "missing_hash",
    [
        "chunks_hash",
        "pedagogy_graph_hash",
        "instruction_pairs_hash",
        "preference_pairs_hash",
        "concept_graph_hash",
        "vocabulary_ttl_hash",
        "holdout_graph_hash",
    ],
)
def test_missing_provenance_hash_fails(missing_hash):
    """Each individual provenance hash is required."""
    validator = _validator()
    card = _valid_card()
    del card["provenance"][missing_hash]
    errors = list(validator.iter_errors(card))
    assert errors, f"Missing provenance.{missing_hash} must fail"


# ---------------------------------------------------------------------------
# Negative: enums and patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_format",
    ["", "ggml", "pytorch", "Safetensors", "merged-safetensors"],
)
def test_invalid_adapter_format_fails(bad_format):
    validator = _validator()
    errors = list(validator.iter_errors(_valid_card(adapter_format=bad_format)))
    assert errors, f"Invalid adapter_format {bad_format!r} must fail"


@pytest.mark.parametrize(
    "bad_model_id",
    [
        "",
        "Qwen-2-5",            # uppercase
        "qwen2.5",             # dot not allowed
        "-leading-hyphen",
        "trailing-hyphen-",
        "double--hyphen",      # implicit by pattern (empty token)
        "has space",
        "underscore_id",
    ],
)
def test_invalid_model_id_pattern_fails(bad_model_id):
    validator = _validator()
    errors = list(validator.iter_errors(_valid_card(model_id=bad_model_id)))
    assert errors, f"Malformed model_id {bad_model_id!r} must fail"


@pytest.mark.parametrize(
    "bad_hash",
    [
        "",
        "abcd",                       # too short
        "a" * 63,                     # one short
        "a" * 65,                     # one long
        "g" * 64,                     # not hex
        ("A" * 64),                   # uppercase rejected
    ],
)
def test_invalid_provenance_hash_fails(bad_hash):
    validator = _validator()
    card = _valid_card()
    card["provenance"]["chunks_hash"] = bad_hash
    errors = list(validator.iter_errors(card))
    assert errors, f"Malformed hash {bad_hash!r} must fail"


@pytest.mark.parametrize(
    "bad_repo",
    [
        "",
        "no-slash",
        "double/slash/here",
        "has space/repo",
        "org/has space",
    ],
)
def test_invalid_huggingface_repo_pattern_fails(bad_repo):
    validator = _validator()
    card = _valid_card()
    card["base_model"]["huggingface_repo"] = bad_repo
    errors = list(validator.iter_errors(card))
    assert errors, f"Malformed huggingface_repo {bad_repo!r} must fail"


# ---------------------------------------------------------------------------
# Negative: additionalProperties strict at every level
# ---------------------------------------------------------------------------


def test_extra_top_level_field_rejected():
    validator = _validator()
    card = _valid_card()
    card["unexpected_extra"] = "nope"
    errors = list(validator.iter_errors(card))
    assert errors, "Extra top-level fields must be rejected"


def test_extra_field_on_training_config_rejected():
    """training_config must not accept unknown keys (per spec)."""
    validator = _validator()
    card = _valid_card()
    card["training_config"]["mystery_param"] = 0.5
    errors = list(validator.iter_errors(card))
    assert errors, "Extra fields on training_config must be rejected"


def test_extra_field_on_base_model_rejected():
    validator = _validator()
    card = _valid_card()
    card["base_model"]["family"] = "qwen"
    errors = list(validator.iter_errors(card))
    assert errors, "Extra fields on base_model must be rejected"


def test_extra_field_on_provenance_rejected():
    validator = _validator()
    card = _valid_card()
    card["provenance"]["extra_hash"] = _HASH64
    errors = list(validator.iter_errors(card))
    assert errors, "Extra fields on provenance must be rejected"


def test_extra_field_on_eval_scores_rejected():
    validator = _validator()
    card = _valid_card(
        eval_scores={
            "faithfulness": 0.5,
            "coverage": 0.5,
            "baseline_delta": 0.0,
            "scoring_commit": "a" * 40,
            "tolerance_band": {"faithfulness": 0.05},
            "extra_metric": 0.9,
        },
    )
    errors = list(validator.iter_errors(card))
    assert errors, "Extra fields on eval_scores must be rejected"


# ---------------------------------------------------------------------------
# Wave 102: scoring_commit + tolerance_band reproducibility surface
# ---------------------------------------------------------------------------


def test_eval_scores_requires_scoring_commit():
    """When eval_scores is present, scoring_commit must be supplied."""
    validator = _validator()
    card = _valid_card(eval_scores={
        "faithfulness": 0.8,
        "tolerance_band": {"faithfulness": 0.05},
    })
    errors = list(validator.iter_errors(card))
    assert errors, "Missing eval_scores.scoring_commit must fail"


def test_eval_scores_requires_tolerance_band():
    """When eval_scores is present, tolerance_band must be supplied."""
    validator = _validator()
    card = _valid_card(eval_scores={
        "faithfulness": 0.8,
        "scoring_commit": "a" * 40,
    })
    errors = list(validator.iter_errors(card))
    assert errors, "Missing eval_scores.tolerance_band must fail"


def test_scoring_commit_pattern_enforced():
    """scoring_commit must be a 40-char lowercase-hex SHA."""
    validator = _validator()
    for bad in ("", "abc", "Z" * 40, "a" * 39, "a" * 41, "g" * 40):
        card = _valid_card(eval_scores={
            "faithfulness": 0.5,
            "scoring_commit": bad,
            "tolerance_band": {"faithfulness": 0.05},
        })
        errors = list(validator.iter_errors(card))
        assert errors, f"scoring_commit={bad!r} must fail"


def test_eval_scores_accepts_headline_table():
    """Optional headline_table validates row shape."""
    validator = _validator()
    card = _valid_card(eval_scores={
        "faithfulness": 0.8,
        "scoring_commit": "a" * 40,
        "tolerance_band": {"faithfulness": 0.05},
        "headline_table": [
            {"setup": "base", "accuracy": 0.4, "faithfulness": 0.5,
             "hallucination_rate": 0.5, "source_match": 0.1,
             "qualitative_score": None},
            {"setup": "adapter+rag", "accuracy": 0.85, "faithfulness": 0.9,
             "hallucination_rate": 0.1, "source_match": 0.6,
             "qualitative_score": 4.5},
        ],
    })
    errors = list(validator.iter_errors(card))
    assert errors == [], [e.message for e in errors]


def test_eval_scores_retrieval_method_enum_locked():
    """retrieval_method_table.method must be one of the five presets."""
    validator = _validator()
    card = _valid_card(eval_scores={
        "scoring_commit": "a" * 40,
        "tolerance_band": {"faithfulness": 0.05},
        "retrieval_method_table": [
            {"method": "not-a-method", "accuracy": 0.5},
        ],
    })
    errors = list(validator.iter_errors(card))
    assert errors, "Unknown retrieval method must fail enum check"


# ---------------------------------------------------------------------------
# Negative: numeric ranges on training_config + eval_scores
# ---------------------------------------------------------------------------


def test_training_config_seed_negative_fails():
    validator = _validator()
    card = _valid_card()
    card["training_config"]["seed"] = -1
    errors = list(validator.iter_errors(card))
    assert errors


def test_training_config_zero_lr_fails():
    """exclusiveMinimum on learning_rate."""
    validator = _validator()
    card = _valid_card()
    card["training_config"]["learning_rate"] = 0
    errors = list(validator.iter_errors(card))
    assert errors


@pytest.mark.parametrize("field", ["epochs", "lora_rank", "lora_alpha", "max_seq_length", "batch_size"])
def test_training_config_zero_in_positive_int_field_fails(field):
    validator = _validator()
    card = _valid_card()
    card["training_config"][field] = 0
    errors = list(validator.iter_errors(card))
    assert errors


@pytest.mark.parametrize(
    "score, value",
    [
        ("faithfulness", -0.1),
        ("faithfulness", 1.1),
        ("coverage", -0.1),
        ("coverage", 1.1),
        ("baseline_delta", -1.1),
        ("baseline_delta", 1.1),
    ],
)
def test_eval_scores_out_of_range_fails(score, value):
    validator = _validator()
    card = _valid_card(eval_scores={
        "faithfulness": 0.5,
        "coverage": 0.5,
        "baseline_delta": 0.0,
        "scoring_commit": "a" * 40,
        "tolerance_band": {"faithfulness": 0.05},
    })
    card["eval_scores"][score] = value
    errors = list(validator.iter_errors(card))
    assert errors, f"eval_scores.{score}={value} must fail range check"


# ---------------------------------------------------------------------------
# Defensive: schema is immutable from the test's perspective
# ---------------------------------------------------------------------------


def test_load_does_not_mutate_schema():
    """Two independent loads produce the same dict (no global mutation)."""
    a = _load_schema()
    b = _load_schema()
    assert a == b
    # And our local copy stays untouched
    snapshot = copy.deepcopy(a)
    list(_validator().iter_errors(_valid_card()))
    assert a == snapshot
