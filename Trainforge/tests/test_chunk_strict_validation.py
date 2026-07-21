"""GPT Feedback v2 Wave 5 / W5.C — chunk_v4 strict-mode regression for the
new optional ``key_claims`` (Wave 1.5 W1.5.A) + ``objective_alignment``
(Wave 1.7 W1.7.A) fields.

The chunk root carries ``additionalProperties: true`` so legacy chunks accept
the new fields silently — but this test pins three contracts under
``TRAINFORGE_VALIDATE_CHUNKS=true`` strict-mode validation:

  (a) chunk WITHOUT the new fields (legacy / pre-Wave-5 corpus) MUST validate
      cleanly — back-compat day-1.
  (b) chunk WITH structured ``key_claims=[{"claim": "X is Y",
      "source_chunk_ids": ["semantik:chunk_A"]}]`` AND ``objective_alignment=[
      {"objective_id": "TO-05", "declared_bloom": "apply",
      "status": "delivered"}]`` MUST validate cleanly.
  (c) chunk WITH malformed ``key_claims=[{"claim": 5,
      "source_chunk_ids": "not-a-list"}]`` MUST fail with a clear schema
      error — both the integer claim and the non-list source_chunk_ids
      should violate the structured arm of the oneOf.

Mirror surface: ``Trainforge/tests/test_chunk_validation.py`` carries the
production hook tests (env-gate, write-time validation, etc.); this file
focuses on the W5.C fields specifically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _build_validator():
    """Build a Draft202012Validator with offline ref resolution against every
    $id in schemas/ so the new ObjectiveAlignment $ref into
    courseforge_jsonld_v1.schema.json resolves without network.

    Mirrors the loader in ``Trainforge/tests/test_chunk_validation.py``
    (see ``_build_validator``) — both use ``referencing.Registry`` because
    jsonschema's deprecated RefResolver hits ``Unresolvable JSON pointer``
    on cross-schema $refs.
    """
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schema = json.loads(CHUNK_SCHEMA_PATH.read_text())
    id_to_schema: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s

    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry)


def _base_chunk() -> Dict[str, Any]:
    """Minimal chunk_v4-compliant record. No new W5.C fields populated."""
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "week_01_overview",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "apply",
    }


# ---------------------------------------------------------------------------
# (a) Legacy chunk (no new fields) — strict-mode-clean.
# ---------------------------------------------------------------------------


def test_legacy_chunk_without_new_fields_validates():
    """Pre-Wave-5 chunks (no key_claims, no objective_alignment) MUST
    validate untouched. Day-1 back-compat contract."""
    validator = _build_validator()
    chunk = _base_chunk()
    assert "key_claims" not in chunk
    assert "objective_alignment" not in chunk
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"legacy chunk unexpectedly errored: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------------
# (b) Structured chunk (both new fields populated) — strict-mode-clean.
# ---------------------------------------------------------------------------


def test_chunk_with_structured_key_claims_and_objective_alignment_validates():
    """Wave-5-shape chunks MUST validate cleanly under strict mode.

    Pins both new fields together so a single test exercises the full
    W5.C contract: structured key_claims + objective_alignment $ref to
    the courseforge JSON-LD schema's ObjectiveAlignment def.
    """
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        {
            "claim": "X is Y",
            "source_chunk_ids": ["semantik:chunk_A"],
        }
    ]
    chunk["objective_alignment"] = [
        {
            "objective_id": "TO-05",
            "declared_bloom": "apply",
            "status": "delivered",
        }
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"structured-shape chunk unexpectedly errored: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


def test_chunk_with_legacy_string_key_claims_validates():
    """The W1.5 oneOf preserves the legacy List[str] shape — pre-Wave-1.5
    Courseforge emits and IMSCC chunks built from non-Wave-1.5 corpora
    still validate. This is the back-compat arm of the oneOf."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        "X relates to Y",
        "Y relates to Z",
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"legacy List[str] key_claims unexpectedly errored: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


def test_chunk_with_structured_key_claims_null_source_chunk_ids_validates():
    """The structured arm admits source_chunk_ids: null (graceful-degrade
    when the originating block didn't yet have per-claim attribution)."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        {"claim": "Standalone claim with no per-claim attribution"},
        {"claim": "Explicit-null arm", "source_chunk_ids": None},
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"null source_chunk_ids unexpectedly errored: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------------
# (c) Malformed chunk — strict-mode MUST fail with a clear schema error.
# ---------------------------------------------------------------------------


def test_chunk_with_malformed_key_claims_fails():
    """A malformed structured key_claims entry (integer claim + non-list
    source_chunk_ids) MUST fail strict validation. Closes the silent-
    drift class W5.C is built to catch."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        {"claim": 5, "source_chunk_ids": "not-a-list"}
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors, (
        "malformed key_claims (int claim + non-list source_chunk_ids) "
        "MUST fail strict validation; the oneOf must reject both the "
        "List[str] arm (object is not a string) and the structured arm "
        "(claim is not a string AND source_chunk_ids is not an array/null)"
    )
    # The error MUST surface against the key_claims field — pin the path
    # so a future schema refactor can't silently shift error attribution.
    paths = [list(e.absolute_path) for e in errors]
    assert any("key_claims" in p for p in paths), (
        f"expected error path to include 'key_claims'; got {paths}"
    )


def test_chunk_with_learning_outcome_source_refs_validates():
    """Wave 5 W5.G — strict-mode positive case for the new optional
    ``learning_outcome_source_refs`` field. Each value MUST be a non-
    empty array of non-empty strings; the field is wholly optional so
    its absence on legacy chunks is unaffected (covered by the
    ``test_legacy_chunk_without_new_fields_validates`` predecessor)."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["learning_outcome_source_refs"] = {
        "TO-05": ["semantik:rdf-primer-ch3#sec-2"],
        "TO-06": ["semantik:chunk_alpha", "semantik:chunk_beta"],
    }
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        f"learning_outcome_source_refs unexpectedly errored: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


def test_chunk_with_malformed_learning_outcome_source_refs_fails():
    """Empty-string source-chunk IDs MUST fail strict validation.
    Closes the silent-drift class W5.G is built to catch."""
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["learning_outcome_source_refs"] = {
        "TO-05": [""],  # empty string violates minLength: 1
    }
    errors = list(validator.iter_errors(chunk))
    assert errors, (
        "empty-string sourceId in learning_outcome_source_refs MUST fail "
        "strict validation; minLength: 1 on the inner items must reject"
    )


def test_chunk_with_malformed_objective_alignment_fails():
    """A malformed objective_alignment entry (missing required
    objective_id) MUST fail strict validation via the cross-schema $ref
    to courseforge_jsonld_v1.schema.json#/$defs/ObjectiveAlignment.

    Pins the $ref resolution path: if the registry doesn't load the
    courseforge schema, this test would falsely pass (no validator =
    no errors); failing here proves the cross-schema reference resolves.
    """
    validator = _build_validator()
    chunk = _base_chunk()
    chunk["objective_alignment"] = [
        {
            # missing required 'objective_id' + 'declared_bloom' + 'status'
            "action_verb_present": True,
        }
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors, (
        "objective_alignment entry missing required objective_id / "
        "declared_bloom / status MUST fail strict validation via the "
        "cross-schema $ref to ObjectiveAlignment"
    )
