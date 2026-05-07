#!/usr/bin/env python3
"""
Wave 112 Task 5 regression test.

Wave 107 added the `ClaudeSessionProvider` which sets
`out["provider"] = "claude_session"` on every emitted pair, but the
instruction- and preference-pair schemas only allowed
`["mock", "anthropic"]`. Strict consumers therefore rejected every
Wave 107+ pair.

This test pins the contract: a minimal-but-schema-valid pair dict
carrying `provider="claude_session"` must validate against both
`schemas/knowledge/instruction_pair.schema.json` and
`schemas/knowledge/preference_pair.schema.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.utils.jsonschema import validate_pair_record  # noqa: E402

SCHEMAS_ROOT = PROJECT_ROOT / "schemas"

# W-D4 follow-up: cross-schema $ref to pair_audit_fields.schema.json
# now lands inline in instruction_pair{,.strict} + preference_pair.
# Use ``validate_pair_record`` (registry-aware) instead of bare
# ``jsonschema.validate`` so the cross-schema $ref resolves offline.
INSTRUCTION_PAIR_SCHEMA = (
    SCHEMAS_ROOT / "knowledge" / "instruction_pair.schema.json"
)
PREFERENCE_PAIR_SCHEMA = (
    SCHEMAS_ROOT / "knowledge" / "preference_pair.schema.json"
)


def _load_schema(name: str) -> dict:
    with (SCHEMAS_ROOT / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _minimal_instruction_pair(provider: str) -> dict:
    """Return a minimal instruction-pair dict that satisfies every required
    field plus length constraints. Only the `provider` value varies."""
    return {
        "prompt": (
            "Explain the difference between sh:datatype and sh:class "
            "in a SHACL property shape, with a concrete example."
        ),  # >= 40 chars, <= 400
        "completion": (
            "sh:datatype constrains a literal's XSD datatype, while "
            "sh:class constrains an IRI value to be an instance of a "
            "specific class. Use sh:datatype xsd:string for literals; "
            "use sh:class ex:Person for object references."
        ),  # >= 50 chars, <= 600
        "chunk_id": "sample_course_chunk_00042",
        "lo_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "evt-0000-0001",
        "provider": provider,
        "schema_version": "v1",
    }


def _minimal_preference_pair(provider: str) -> dict:
    return {
        "prompt": (
            "Which SHACL constraint targets the datatype of a literal "
            "value: sh:datatype or sh:class? Justify in one sentence."
        ),  # >= 40 chars
        "chosen": (
            "sh:datatype targets the literal's XSD datatype because it "
            "constrains the lexical form, whereas sh:class constrains "
            "node identity for IRIs."
        ),  # >= 50, <= 600
        "rejected": (
            "sh:class targets the literal's datatype because every value "
            "in RDF is implicitly typed by its rdfs:Class membership in "
            "the closed-world view, so datatype is redundant."
        ),  # >= 50, token-Jaccard distinct enough from chosen
        "chunk_id": "sample_course_chunk_00042",
        "lo_refs": ["TO-01"],
        "seed": 17,
        "decision_capture_id": "evt-0000-0002",
        "rejected_source": "rule_synthesized",
        "provider": provider,
        "schema_version": "v1",
    }


def test_minimal_pairs_with_known_providers_validate_baseline():
    """Sanity check: the minimal-pair builders pass schema validation
    under the pre-existing provider values. Catches authoring errors
    in this test file independent of the enum extension."""
    for provider in ("mock", "anthropic"):
        validate_pair_record(
            _minimal_instruction_pair(provider), INSTRUCTION_PAIR_SCHEMA
        )
        validate_pair_record(
            _minimal_preference_pair(provider), PREFERENCE_PAIR_SCHEMA
        )


def test_instruction_pair_schema_accepts_claude_session_provider():
    pair = _minimal_instruction_pair("claude_session")
    # Must not raise. Pre-fix: ValidationError "'claude_session' is not
    # one of ['mock', 'anthropic']".
    validate_pair_record(pair, INSTRUCTION_PAIR_SCHEMA)


def test_preference_pair_schema_accepts_claude_session_provider():
    pair = _minimal_preference_pair("claude_session")
    validate_pair_record(pair, PREFERENCE_PAIR_SCHEMA)


# ---------------------------------------------------------------------------
# Wave 113 prep — Together provider enum extension
# ---------------------------------------------------------------------------


def test_instruction_pair_schema_accepts_together_provider():
    """TogetherSynthesisProvider sets ``provider="together"`` on every
    emitted pair; the instruction_pair schema enum must admit it or
    strict consumers reject every Together-produced row."""
    pair = _minimal_instruction_pair("together")
    validate_pair_record(pair, INSTRUCTION_PAIR_SCHEMA)


def test_preference_pair_schema_accepts_together_provider():
    pair = _minimal_preference_pair("together")
    validate_pair_record(pair, PREFERENCE_PAIR_SCHEMA)


# ---------------------------------------------------------------------------
# Wave 113 — LocalSynthesisProvider enum extension
# ---------------------------------------------------------------------------


def test_instruction_pair_schema_accepts_local_provider():
    """LocalSynthesisProvider sets ``provider="local"`` on every emitted
    pair; the instruction_pair schema enum must admit it or strict
    consumers reject every locally-paraphrased row."""
    pair = _minimal_instruction_pair("local")
    validate_pair_record(pair, INSTRUCTION_PAIR_SCHEMA)


def test_preference_pair_schema_accepts_local_provider():
    pair = _minimal_preference_pair("local")
    validate_pair_record(pair, PREFERENCE_PAIR_SCHEMA)
