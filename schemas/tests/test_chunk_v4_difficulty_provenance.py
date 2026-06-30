"""chunk_v4 difficulty_provenance / difficulty_irt additive-contract tests.

The honest IRT scaffold adds two OPTIONAL chunk fields. Contract:

- a chunk WITH the new fields validates (strict);
- a legacy chunk WITHOUT them still validates (additive);
- difficulty_irt with a bad type is rejected;
- the new fields do NOT participate in the content-hash chunk id
  derivation (chunk id unchanged when the fields are set).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator, RefResolver

    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    store: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    return Draft202012Validator(schema, resolver=resolver)


def _base_chunk() -> Dict[str, Any]:
    return {
        "id": "sample_101_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk content for testing.",
        "html": "<p>Sample chunk content for testing.</p>",
        "follows_chunk": None,
        "source": {"course_id": "SAMPLE_101", "module_id": "week_01", "lesson_id": "lesson_01"},
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 10,
        "word_count": 6,
        "bloom_level": "understand",
    }


def test_chunk_with_new_fields_validates():
    v = _build_validator()
    chunk = _base_chunk()
    chunk["difficulty_provenance"] = "calibrated"
    chunk["difficulty_irt"] = {"difficulty_b": 1.2, "discrimination_a": 1.0, "n_responses": 42}
    errors = sorted(v.iter_errors(chunk), key=lambda e: e.path)
    assert errors == [], [e.message for e in errors]


def test_legacy_chunk_without_new_fields_validates():
    v = _build_validator()
    chunk = _base_chunk()
    assert "difficulty_provenance" not in chunk
    errors = list(v.iter_errors(chunk))
    assert errors == [], [e.message for e in errors]


def test_bad_difficulty_irt_type_rejected():
    v = _build_validator()
    chunk = _base_chunk()
    chunk["difficulty_irt"] = {"n_responses": "not-an-int"}
    errors = list(v.iter_errors(chunk))
    assert errors, "difficulty_irt.n_responses must be an integer"


def test_bad_provenance_enum_rejected():
    v = _build_validator()
    chunk = _base_chunk()
    chunk["difficulty_provenance"] = "guessed"
    errors = list(v.iter_errors(chunk))
    assert errors, "difficulty_provenance must be heuristic|calibrated"


def test_new_fields_excluded_from_content_hash_id():
    """Setting the new fields must not perturb the content-hash chunk id."""
    from Trainforge.chunker.chunker import _generate_chunk_id
    import os
    os.environ["TRAINFORGE_CONTENT_HASH_IDS"] = "true"
    try:
        # The content-hash id payload is text|source_locator|schema_version —
        # difficulty_provenance / difficulty_irt never enter it, so two chunks
        # with identical text+locator share an id regardless of the new fields.
        id_a = _generate_chunk_id("sample_101_chunk_", 1, "the chunk text", "loc#1")
        id_b = _generate_chunk_id("sample_101_chunk_", 1, "the chunk text", "loc#1")
        assert id_a == id_b
    finally:
        os.environ.pop("TRAINFORGE_CONTENT_HASH_IDS", None)
