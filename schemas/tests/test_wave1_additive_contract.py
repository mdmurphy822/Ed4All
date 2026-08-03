"""GPT Feedback v2 — Wave 1 end-of-wave additive-contract gate.

This is Tests 2 + 3 of the Wave 1 validation gate. The single test file pins:

  * Test 2 (``test_no_required_field_added_without_intent``) — every Wave-1
    touched schema's root ``required[]`` array (and key sub-shape ``required[]``
    on the courseforge_jsonld_v1 ``$defs.AssessmentItem`` / ``$defs.Block``
    definitions) is byte-equal to the audited Wave 1 baseline. Catches silent
    promotion of an optional field to required.
  * Test 3 (``test_property_count_deltas``) — ``len(properties)`` /
    ``len(enum)`` per touched schema is at least the audited post-Wave-1
    count. Uses ``>=`` (not ``==``) so subsequent waves that ADD more fields
    don't break this gate; catches REMOVAL.
  * ``test_legacy_fixtures_validate`` — neutral tracked and inline legacy
    shapes validate against the post-Wave-1 schemas.
  * ``test_new_fixtures_validate`` — three new fixtures under
    ``schemas/tests/fixtures/wave1/`` validate against their target schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "wave1"

CHUNK_V4_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"
COURSEFORGE_JSONLD_PATH = SCHEMAS_DIR / "knowledge" / "courseforge_jsonld_v1.schema.json"
INSTRUCTION_PAIR_PATH = SCHEMAS_DIR / "knowledge" / "instruction_pair.schema.json"
INSTRUCTION_PAIR_STRICT_PATH = (
    SCHEMAS_DIR / "knowledge" / "instruction_pair.strict.schema.json"
)
PREFERENCE_PAIR_PATH = SCHEMAS_DIR / "knowledge" / "preference_pair.schema.json"
COURSE_STATUS_PATH = SCHEMAS_DIR / "governance" / "course_status.schema.json"
PHASE_OUTPUT_PATH = SCHEMAS_DIR / "governance" / "phase_output.schema.json"
DECISION_EVENT_PATH = SCHEMAS_DIR / "events" / "decision_event.schema.json"

# --------------------------------------------------------------------------- #
# Audited Wave-1 baselines (literal snapshots).
# --------------------------------------------------------------------------- #

# Root-level required[] arrays for each touched schema.  Snapshotted by reading
# the schema file at audit time (2026-05-06, post-W1.A-E).  A future refactor
# that promotes a Wave-1 field to required without re-baselining this dict
# fails Test 2.
EXPECTED_REQUIRED: Dict[str, List[str]] = {
    "chunk_v4": [
        "id",
        "schema_version",
        "chunk_type",
        "text",
        "html",
        "follows_chunk",
        "source",
        "concept_tags",
        "learning_outcome_refs",
        "difficulty",
        "tokens_estimate",
        "word_count",
        "bloom_level",
    ],
    "instruction_pair": [
        "prompt",
        "completion",
        "chunk_id",
        "lo_refs",
        "bloom_level",
        "content_type",
        "seed",
        "decision_capture_id",
    ],
    "instruction_pair_strict": [
        "prompt",
        "completion",
        "chunk_id",
        "lo_refs",
        "bloom_level",
        "content_type",
        "seed",
        "decision_capture_id",
    ],
    "preference_pair": [
        "prompt",
        "chosen",
        "rejected",
        "chunk_id",
        "lo_refs",
        "seed",
        "decision_capture_id",
    ],
    # course_status is a root-level enum; no properties block, no required[].
    "course_status": [],
    # phase_output: only run_id + phase_name (W1.D's deliverable).
    "phase_output": ["run_id", "phase_name"],
    "decision_event": [
        "run_id",
        "timestamp",
        "operation",
        "decision_type",
        "decision",
        "rationale",
    ],
}

# courseforge_jsonld_v1 sub-shape required[] arrays (W1.A-touched defs).
EXPECTED_COURSEFORGE_DEFS_REQUIRED: Dict[str, List[str]] = {
    "AssessmentItem": ["stem", "answer_key", "distractors", "correct_answer_index"],
    "Block": ["blockId", "blockType", "sequence"],
}

# Audited post-Wave-1 property / enum counts.  ``>=`` semantics: tests fail
# loudly only when a subsequent change REMOVES a property below the floor.
EXPECTED_PROPERTY_FLOORS: Dict[str, int] = {
    # courseforge_jsonld_v1::AssessmentItem.properties (was 4, +3 → 7)
    "courseforge_jsonld_v1::$defs.AssessmentItem.properties": 7,
    # courseforge_jsonld_v1::Block.properties (was 18, +2 → 20).  Plan said
    # pre-wave 17; W1.A worker report audited 18.
    "courseforge_jsonld_v1::$defs.Block.properties": 20,
    # chunk_v4 root properties (was 24, +2 → 26)
    "chunk_v4::properties": 26,
    # instruction_pair root properties (was 11, +7 → 18).  Plan said 10/17;
    # W1.B worker report audited 11/18 (schema_version was uncounted in plan).
    "instruction_pair::properties": 18,
    "instruction_pair_strict::properties": 18,
    # preference_pair adds distractor_quality (was 11, +8 → 19).
    "preference_pair::properties": 19,
    # course_status enum (file new)
    "course_status::enum": 5,
    # phase_output properties (file new — 9 dataclass-mirror fields + the
    # W1.D promotion_decision = 10).
    "phase_output::properties": 10,
    # decision_event::decision_type.enum (was 142, +4 → 146).
    "decision_event::decision_type.enum": 146,
}


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _build_registry():
    """Build a referencing registry covering every schema we ship.

    Mirrors the loader pattern in test_courseforge_jsonld_observed_bloom.py;
    needed so cross-schema $refs (BloomLevel, source_reference, content_type)
    resolve when validating fixtures against the chunk / courseforge schemas.
    """
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


def _build_validator(schema: Dict[str, Any]):
    """Return a Draft 2020-12 validator with the cross-schema registry.

    W-D9 migrated the pair schemas (instruction / instruction_strict /
    preference) to draft-2020-12, so the legacy draft-07 dispatch branch
    is now dead. Every schema in the tree uses 2020-12; this helper
    always returns a Draft202012Validator.
    """
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    return Draft202012Validator(schema, registry=_build_registry())


def _build_validator_for_courseforge_def(def_name: str):
    """Return a validator targeting courseforge_jsonld_v1::$defs/<def_name>."""
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    page = _load(COURSEFORGE_JSONLD_PATH)
    sub_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": page["$id"] + "#/$defs/" + def_name,
    }
    return Draft202012Validator(sub_schema, registry=_build_registry())


# --------------------------------------------------------------------------- #
# Test 2 — required-field baselines (additive contract).
# --------------------------------------------------------------------------- #


def test_no_required_field_added_without_intent():
    """Each Wave-1 touched schema's required[] matches the audited baseline.

    Catches silent promotion of an optional field to required — for example,
    a future refactor that flips ``promotion_status`` to required on
    instruction_pair would break legacy artifact loading.
    """
    schemas = {
        "chunk_v4": _load(CHUNK_V4_PATH),
        "instruction_pair": _load(INSTRUCTION_PAIR_PATH),
        "instruction_pair_strict": _load(INSTRUCTION_PAIR_STRICT_PATH),
        "preference_pair": _load(PREFERENCE_PAIR_PATH),
        "course_status": _load(COURSE_STATUS_PATH),
        "phase_output": _load(PHASE_OUTPUT_PATH),
        "decision_event": _load(DECISION_EVENT_PATH),
    }
    for name, schema in schemas.items():
        actual = list(schema.get("required", []))
        expected = EXPECTED_REQUIRED[name]
        assert actual == expected, (
            f"{name}.required[] drifted from Wave-1 baseline. "
            f"expected={expected!r}, actual={actual!r}. "
            "If this is intentional, re-baseline EXPECTED_REQUIRED in "
            "schemas/tests/test_wave1_additive_contract.py and document the "
            "rationale in the commit message."
        )

    # courseforge_jsonld_v1 is a meta-shape; pin sub-defs that W1.A touched.
    cf = _load(COURSEFORGE_JSONLD_PATH)
    for def_name, expected_required in EXPECTED_COURSEFORGE_DEFS_REQUIRED.items():
        actual = list(cf["$defs"][def_name].get("required", []))
        assert actual == expected_required, (
            f"courseforge_jsonld_v1::$defs.{def_name}.required[] drifted. "
            f"expected={expected_required!r}, actual={actual!r}."
        )


# --------------------------------------------------------------------------- #
# Test 3 — property / enum count floors (anti-silent-degradation).
# --------------------------------------------------------------------------- #


def test_property_count_deltas():
    """Each touched schema's property / enum count is at least the audited floor.

    ``>=`` semantics so future waves that legitimately ADD fields don't
    break the gate.  Catches REMOVAL: a merge conflict that truncates a
    properties block, or a refactor that flips a property from
    schema-declared to schema-inferred.
    """
    actual_counts: Dict[str, int] = {}

    cf = _load(COURSEFORGE_JSONLD_PATH)
    actual_counts["courseforge_jsonld_v1::$defs.AssessmentItem.properties"] = len(
        cf["$defs"]["AssessmentItem"]["properties"]
    )
    actual_counts["courseforge_jsonld_v1::$defs.Block.properties"] = len(
        cf["$defs"]["Block"]["properties"]
    )

    chunk = _load(CHUNK_V4_PATH)
    actual_counts["chunk_v4::properties"] = len(chunk["properties"])

    ip = _load(INSTRUCTION_PAIR_PATH)
    actual_counts["instruction_pair::properties"] = len(ip["properties"])

    ips = _load(INSTRUCTION_PAIR_STRICT_PATH)
    actual_counts["instruction_pair_strict::properties"] = len(ips["properties"])

    pp = _load(PREFERENCE_PAIR_PATH)
    actual_counts["preference_pair::properties"] = len(pp["properties"])

    cs = _load(COURSE_STATUS_PATH)
    actual_counts["course_status::enum"] = len(cs["enum"])

    po = _load(PHASE_OUTPUT_PATH)
    actual_counts["phase_output::properties"] = len(po["properties"])

    de = _load(DECISION_EVENT_PATH)
    actual_counts["decision_event::decision_type.enum"] = len(
        de["properties"]["decision_type"]["enum"]
    )

    failures = []
    for key, floor in EXPECTED_PROPERTY_FLOORS.items():
        actual = actual_counts[key]
        if actual < floor:
            failures.append(f"  {key}: {actual} < floor {floor}")
    assert not failures, (
        "Wave-1 property-count floor regressions detected:\n"
        + "\n".join(failures)
        + "\n\nIf the floor needs to drop, re-baseline EXPECTED_PROPERTY_FLOORS "
        "in schemas/tests/test_wave1_additive_contract.py and document the "
        "rationale in the commit message."
    )


# --------------------------------------------------------------------------- #
# Legacy-shape round-trip using neutral fixtures.
# --------------------------------------------------------------------------- #


def test_legacy_fixtures_validate():
    """Neutral legacy-shaped records validate against post-Wave-1 schemas."""
    _require_jsonschema()
    ip_records = [_load(FIXTURES_DIR / "instruction_pair_trainable.json")]
    ip_validator = _build_validator(_load(INSTRUCTION_PAIR_PATH))
    for idx, rec in enumerate(ip_records):
        errors = list(ip_validator.iter_errors(rec))
        assert errors == [], (
            f"legacy instruction_pair record {idx} fails post-Wave-1 schema: "
            f"{[(list(e.absolute_path), e.message) for e in errors]}"
        )

    pp_records = [{
        "prompt": "Which explanation is supported by the supplied neutral context and why?",
        "chosen": "The supported explanation cites the supplied context.",
        "rejected": "An unsupported explanation invents a result that the supplied context never states.",
        "chunk_id": "neutral_chunk_001",
        "lo_refs": ["co-01"],
        "seed": 42,
        "decision_capture_id": "dc_neutral_001",
    }]
    pp_validator = _build_validator(_load(PREFERENCE_PAIR_PATH))
    for idx, rec in enumerate(pp_records):
        errors = list(pp_validator.iter_errors(rec))
        assert errors == [], (
            f"legacy preference_pair record {idx} fails post-Wave-1 schema: "
            f"{[(list(e.absolute_path), e.message) for e in errors]}"
        )

    chunk_records = [_load(FIXTURES_DIR.parent / "wave_d11" / "chunk_with_evidence_quote_key_claims.json")]
    chunk_validator = _build_validator(_load(CHUNK_V4_PATH))
    for idx, rec in enumerate(chunk_records):
        errors = list(chunk_validator.iter_errors(rec))
        assert errors == [], (
            f"legacy chunk record {idx} fails post-Wave-1 chunk_v4 schema: "
            f"{[(list(e.absolute_path), e.message) for e in errors]}"
        )


# --------------------------------------------------------------------------- #
# New-fixture round-trip.
# --------------------------------------------------------------------------- #


def test_new_fixtures_validate():
    """Three new wave1 fixtures validate against their target schemas."""
    _require_jsonschema()

    # 1. AssessmentItem with all three optional Bloom fields populated.
    ai_fixture = _load(FIXTURES_DIR / "assessment_item_with_observed_bloom.json")
    ai_validator = _build_validator_for_courseforge_def("AssessmentItem")
    ai_errors = list(ai_validator.iter_errors(ai_fixture))
    assert ai_errors == [], (
        "assessment_item_with_observed_bloom.json failed AssessmentItem validation: "
        f"{[(list(e.absolute_path), e.message) for e in ai_errors]}"
    )
    # Pin that the new fields are actually present (defends against
    # accidental fixture truncation).
    for key in ("bloomLevel", "observedBloomLevel", "bloomAlignment"):
        assert key in ai_fixture, f"{key} missing from assessment_item fixture"

    # 2. Instruction pair with promotion_status="trainable" + source_chunk_id.
    ip_fixture = _load(FIXTURES_DIR / "instruction_pair_trainable.json")
    ip_validator = _build_validator(_load(INSTRUCTION_PAIR_PATH))
    ip_errors = list(ip_validator.iter_errors(ip_fixture))
    assert ip_errors == [], (
        "instruction_pair_trainable.json failed instruction_pair validation: "
        f"{[(list(e.absolute_path), e.message) for e in ip_errors]}"
    )
    assert ip_fixture["promotion_status"] == "trainable"
    assert ip_fixture["source_chunk_id"], (
        "fixture must carry non-empty source_chunk_id (if/then clause requires it)"
    )

    # 3. PhaseOutput with promotion_decision="pass".
    po_fixture = _load(FIXTURES_DIR / "phase_output_with_promotion_decision.json")
    po_validator = _build_validator(_load(PHASE_OUTPUT_PATH))
    po_errors = list(po_validator.iter_errors(po_fixture))
    assert po_errors == [], (
        "phase_output_with_promotion_decision.json failed phase_output validation: "
        f"{[(list(e.absolute_path), e.message) for e in po_errors]}"
    )
    assert po_fixture["promotion_decision"] == "pass"
