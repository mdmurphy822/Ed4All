"""W-D11 T11.0 — schema contract tests for evidence-quote storage.

GPT Feedback v3 item 15 (P1) raised "evidence_spans[] with verbatim
quote per question". T11.0 lands the schema foundation: the per-claim
attribution surface (chunk_v4 ``key_claims`` structured arm and
``pair_audit_fields.PerClaimSupport``) gains two optional, additive
fields — ``evidence_quote`` (verbatim substring of the cited chunk's
text) and ``evidence_char_span`` ([start, end) char offsets).

This file pins the schema contract for those two fields:

  * Test 1: structured ``key_claims`` entry WITH the new fields
    validates against ``chunk_v4.schema.json``.
  * Test 2: structured ``key_claims`` entry WITHOUT the new fields
    (legacy / pre-W-D11) still validates.
  * Test 3: legacy ``List[str]`` ``key_claims`` validates — W1.5.A
    ``oneOf`` migration regression guard.
  * Test 4: ``PerClaimSupport`` WITH the new fields validates.
  * Test 5: ``PerClaimSupport`` WITHOUT the new fields validates.
  * Test 6: empty-string ``evidence_quote`` fails ``minLength: 1``.
  * Test 7: 1-element ``evidence_char_span`` fails ``minItems: 2``.

T11.1 + T11.2 will land the validator-side substring enforcement
(``evidence_quote ∈ chunk.text`` and
``chunk.text[start:end] == evidence_quote``) in follow-on commits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"
PAIR_AUDIT_SCHEMA_PATH = (
    SCHEMAS_DIR / "knowledge" / "pair_audit_fields.schema.json"
)


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _build_registry():
    """Build a referencing.Registry over every $id-bearing schema under
    ``schemas/`` so $ref resolution succeeds for cross-file references."""
    _require_jsonschema()
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

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
    return Registry().with_resources(resources)


def _build_chunk_validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = json.loads(CHUNK_SCHEMA_PATH.read_text())
    return Draft202012Validator(schema, registry=_build_registry())


def _build_per_claim_support_validator():
    """Build a Draft202012Validator that points at
    ``$defs/PerClaimSupport`` inside ``pair_audit_fields.schema.json``.
    Mirrors the inline ``$ref`` shape every consumer schema uses."""
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    # Wrapper schema that $refs into the canonical $defs path — same
    # shape that instruction_pair / instruction_pair.strict /
    # preference_pair use to consume PerClaimSupport.
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": (
            "https://ed4all.dev/ns/knowledge/v1/pair_audit_fields.schema.json"
            "#/$defs/PerClaimSupport"
        ),
    }
    return Draft202012Validator(wrapper, registry=_build_registry())


def _base_chunk() -> Dict[str, Any]:
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": (
            "A SHACL node shape declares constraints that fire on "
            "target nodes."
        ),
        "html": "<p>...</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "week_01_overview",
            "lesson_id": "l1",
        },
        "concept_tags": ["shacl"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 12,
        "word_count": 12,
        "bloom_level": "understand",
    }


def _base_per_claim_support() -> Dict[str, Any]:
    return {
        "sentence": "A SHACL node shape declares constraints.",
        "entailment": 0.92,
        "contradiction": 0.02,
        "outcome": "entailed",
    }


# --------------------------------------------------------------------------- #
# Test 1 — structured key_claims WITH evidence_quote + evidence_char_span
# --------------------------------------------------------------------------- #


def test_structured_key_claims_with_evidence_fields_validates():
    """Structured ``key_claims`` entry carrying the W-D11 fields validates."""
    validator = _build_chunk_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        {
            "claim": "A SHACL node shape declares constraints on RDF nodes.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
            "evidence_quote": "A SHACL node shape declares constraints",
            "evidence_char_span": [0, 39],
        },
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        "structured key_claims with W-D11 evidence fields unexpectedly "
        f"failed: {[(list(e.absolute_path), e.message) for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Test 2 — structured key_claims WITHOUT evidence fields (legacy)
# --------------------------------------------------------------------------- #


def test_structured_key_claims_without_evidence_fields_validates():
    """Pre-W-D11 structured ``key_claims`` (no evidence_quote) still validates."""
    validator = _build_chunk_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        {
            "claim": "A SHACL node shape declares constraints on RDF nodes.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
        },
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        "pre-W-D11 structured key_claims unexpectedly failed: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — legacy List[str] key_claims arm (W1.5.A oneOf preservation)
# --------------------------------------------------------------------------- #


def test_legacy_list_of_str_key_claims_still_validates():
    """W1.5.A migration contract: legacy ``List[str]`` arm preserved."""
    validator = _build_chunk_validator()
    chunk = _base_chunk()
    chunk["key_claims"] = [
        "A SHACL node shape declares constraints on RDF nodes.",
        "Constraints fire when a target node fails a property predicate.",
    ]
    errors = list(validator.iter_errors(chunk))
    assert errors == [], (
        "legacy List[str] key_claims arm regressed against the W-D11 "
        "schema bump — W1.5.A oneOf migration contract violated: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — PerClaimSupport WITH evidence_quote + evidence_char_span
# --------------------------------------------------------------------------- #


def test_per_claim_support_with_evidence_fields_validates():
    validator = _build_per_claim_support_validator()
    record = _base_per_claim_support()
    record["evidence_quote"] = "A SHACL node shape declares constraints"
    record["evidence_char_span"] = [0, 39]
    errors = list(validator.iter_errors(record))
    assert errors == [], (
        "PerClaimSupport with W-D11 evidence fields unexpectedly failed: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — PerClaimSupport WITHOUT evidence fields (legacy)
# --------------------------------------------------------------------------- #


def test_per_claim_support_without_evidence_fields_validates():
    """Pre-W-D11 PerClaimSupport records still validate."""
    validator = _build_per_claim_support_validator()
    record = _base_per_claim_support()
    errors = list(validator.iter_errors(record))
    assert errors == [], (
        "pre-W-D11 PerClaimSupport unexpectedly failed: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — empty evidence_quote fails minLength: 1
# --------------------------------------------------------------------------- #


def test_per_claim_support_empty_evidence_quote_fails():
    validator = _build_per_claim_support_validator()
    record = _base_per_claim_support()
    record["evidence_quote"] = ""  # violates minLength: 1
    record["evidence_char_span"] = [0, 0]
    errors = list(validator.iter_errors(record))
    assert errors, (
        "expected validation error for empty evidence_quote "
        "(minLength: 1 violation); got none"
    )
    # Surface the relevant error at least once.
    assert any(
        "evidence_quote" in str(list(e.absolute_path)) or
        "minLength" in (e.validator or "")
        for e in errors
    ), f"unexpected error shape: {[(list(e.absolute_path), e.message) for e in errors]}"


# --------------------------------------------------------------------------- #
# Test 7 — evidence_char_span with one element fails minItems: 2
# --------------------------------------------------------------------------- #


def test_per_claim_support_one_element_evidence_char_span_fails():
    validator = _build_per_claim_support_validator()
    record = _base_per_claim_support()
    record["evidence_quote"] = "A SHACL node shape"
    record["evidence_char_span"] = [0]  # violates minItems: 2
    errors = list(validator.iter_errors(record))
    assert errors, (
        "expected validation error for 1-element evidence_char_span "
        "(minItems: 2 violation); got none"
    )
