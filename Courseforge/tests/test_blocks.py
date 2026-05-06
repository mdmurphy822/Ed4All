"""GPT Feedback v2 Wave 1 / W1.A — Block dataclass tests for the new optional
``observed_bloom_level`` and ``bloom_alignment`` fields.

Pins three contracts:

  1. The fields default to ``None`` and round-trip through
     ``Block.to_jsonld_entry()`` only when populated (camelCase keys
     ``observedBloomLevel`` / ``bloomAlignment``).
  2. ``Block.compute_content_hash()`` is byte-identical between a Block
     constructed without the new fields and the same Block with them
     populated. The two new fields are AUDIT-only and intentionally
     excluded from the canonical hash payload — this guarantees that
     wiring the BERT classifier in Wave 2 doesn't drift every existing
     block hash on rebuild (cache invalidation, broken LibV2 archive
     integrity).
  3. The emitted JSON-LD entry validates against the post-Wave-1
     ``$defs.Block`` shape in courseforge_jsonld_v1.schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Courseforge.scripts.blocks import Block

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
PAGE_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "courseforge_jsonld_v1.schema.json"


def _make_block(
    *,
    observed_bloom_level=None,
    bloom_alignment=None,
) -> Block:
    """Construct a minimal Block at a non-legacy block_type so
    ``to_jsonld_entry()`` dispatches to ``_minimal_block_jsonld()`` (the
    Phase-2 default shape that mirrors $defs.Block)."""
    return Block(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="Prereq set content.",
        bloom_level="apply",
        objective_ids=("TO-01",),
        observed_bloom_level=observed_bloom_level,
        bloom_alignment=bloom_alignment,
    )


# ---------------------------------------------------------------------- #
# Default values + round-trip
# ---------------------------------------------------------------------- #


def test_block_observed_bloom_fields_default_to_none():
    block = Block(
        block_id="week_01_overview#prereq_set_x_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="x",
    )
    assert block.observed_bloom_level is None
    assert block.bloom_alignment is None


def test_block_to_jsonld_entry_omits_unset_observed_bloom_fields():
    """Legacy emit is byte-stable: the new keys are absent when fields
    are None."""
    block = _make_block()
    entry = block.to_jsonld_entry()
    assert "observedBloomLevel" not in entry
    assert "bloomAlignment" not in entry


def test_block_to_jsonld_entry_emits_observed_bloom_when_populated():
    block = _make_block(observed_bloom_level="apply", bloom_alignment=True)
    entry = block.to_jsonld_entry()
    assert entry["observedBloomLevel"] == "apply"
    assert entry["bloomAlignment"] is True


def test_block_to_jsonld_entry_emits_observed_bloom_false_alignment():
    """A False alignment value is NOT treated as missing — emit when
    the classifier ran but disagreed."""
    block = _make_block(observed_bloom_level="understand", bloom_alignment=False)
    entry = block.to_jsonld_entry()
    assert entry["observedBloomLevel"] == "understand"
    assert entry["bloomAlignment"] is False


# ---------------------------------------------------------------------- #
# Schema validation of the emitted JSON-LD entry
# ---------------------------------------------------------------------- #


def test_block_jsonld_entry_validates_against_schema():
    """Round-trip: dataclass → to_jsonld_entry() → JSON → schema-validate
    against $defs.Block."""
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    page = json.loads(PAGE_SCHEMA_PATH.read_text())
    id_to_schema = {}
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
    sub_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": page["$id"] + "#/$defs/Block",
    }
    validator = Draft202012Validator(sub_schema, registry=registry)

    block = _make_block(observed_bloom_level="apply", bloom_alignment=True)
    entry = block.to_jsonld_entry()
    # The minimal block emit lacks bloomLevel / pageId on $defs.Block —
    # those are optional. Round-trip through JSON to mimic actual emit.
    serialised = json.loads(json.dumps(entry))
    errors = list(validator.iter_errors(serialised))
    assert errors == [], (
        f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"
    )


# ---------------------------------------------------------------------- #
# Hash-stability invariant (the load-bearing test for Wave 1)
# ---------------------------------------------------------------------- #


def test_block_hash_excludes_observed_bloom_fields():
    """compute_content_hash() must be byte-identical between a Block with
    the new fields populated and the same Block without them.

    Wave 2 will populate observed_bloom_level + bloom_alignment at
    validation time — this test guarantees that retro-fit doesn't drift
    every existing block's content hash on rebuild.
    """
    base = Block(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="Prereq set content.",
        bloom_level="apply",
        objective_ids=("TO-01",),
    )
    extended = Block(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="Prereq set content.",
        bloom_level="apply",
        objective_ids=("TO-01",),
        observed_bloom_level="apply",
        bloom_alignment=True,
    )
    assert base.compute_content_hash() == extended.compute_content_hash()


def test_block_hash_excludes_observed_bloom_fields_when_misaligned():
    """A False alignment + different observed level still doesn't drift
    the hash. Pins that BOTH fields are excluded from the payload."""
    base = Block(
        block_id="week_02_overview#concept_x_0",
        block_type="concept",
        page_id="week_02_overview",
        sequence=0,
        content="concept text",
        bloom_level="understand",
    )
    extended = Block(
        block_id="week_02_overview#concept_x_0",
        block_type="concept",
        page_id="week_02_overview",
        sequence=0,
        content="concept text",
        bloom_level="understand",
        observed_bloom_level="remember",
        bloom_alignment=False,
    )
    assert base.compute_content_hash() == extended.compute_content_hash()


# ---------------------------------------------------------------------- #
# GPT Feedback v2 Wave 1.7 / W1.7.A — objective_alignment field
# ---------------------------------------------------------------------- #


def _alignment_entry(
    *,
    objective_id: str = "TO-01",
    declared_bloom: str = "apply",
    observed_bloom=None,
    statement_entailment_score=None,
    action_verb_present: bool = True,
    status: str = "delivered",
):
    """Build a canonical-shape ObjectiveAlignment entry for tests."""
    return {
        "objective_id": objective_id,
        "declared_bloom": declared_bloom,
        "observed_bloom": observed_bloom,
        "statement_entailment_score": statement_entailment_score,
        "action_verb_present": action_verb_present,
        "status": status,
    }


def test_block_objective_alignment_default_empty():
    """Default empty tuple, no JSON-LD emit when empty."""
    block = Block(
        block_id="week_01_overview#prereq_set_x_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="x",
    )
    assert block.objective_alignment == ()
    entry = block.to_jsonld_entry()
    assert "objectiveAlignment" not in entry


def test_block_objective_alignment_emits_when_populated():
    """Round-trip through to_jsonld_entry() produces the camelCase array."""
    alignments = (
        _alignment_entry(
            objective_id="CO-08",
            declared_bloom="create",
            observed_bloom="remember",
            statement_entailment_score=0.32,
            action_verb_present=False,
            status="underdelivered",
        ),
        _alignment_entry(
            objective_id="TO-02",
            declared_bloom="analyze",
            observed_bloom="analyze",
            statement_entailment_score=0.91,
            action_verb_present=True,
            status="delivered",
        ),
    )
    block = Block(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="content",
        objective_ids=("CO-08", "TO-02"),
        objective_alignment=alignments,
    )
    entry = block.to_jsonld_entry()
    assert "objectiveAlignment" in entry
    assert isinstance(entry["objectiveAlignment"], list)
    assert len(entry["objectiveAlignment"]) == 2
    assert entry["objectiveAlignment"][0]["objective_id"] == "CO-08"
    assert entry["objectiveAlignment"][0]["status"] == "underdelivered"
    assert entry["objectiveAlignment"][1]["status"] == "delivered"
    # Round-trip JSON serialisation works (no tuples, no non-JSON values).
    json.dumps(entry)


def test_block_objective_alignment_excluded_from_hash():
    """Same Block with and without objective_alignment produces identical
    compute_content_hash().

    Wave 1.7 will populate objective_alignment at validation time — this
    test guarantees that retro-fit doesn't drift every existing block's
    content hash on rebuild.
    """
    base = Block(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="Prereq set content.",
        bloom_level="apply",
        objective_ids=("TO-01",),
    )
    extended = Block(
        block_id="week_01_overview#prereq_set_intro_0",
        block_type="prereq_set",
        page_id="week_01_overview",
        sequence=0,
        content="Prereq set content.",
        bloom_level="apply",
        objective_ids=("TO-01",),
        objective_alignment=(
            _alignment_entry(
                objective_id="TO-01",
                declared_bloom="apply",
                observed_bloom="remember",
                statement_entailment_score=0.41,
                action_verb_present=False,
                status="underdelivered",
            ),
        ),
    )
    assert base.compute_content_hash() == extended.compute_content_hash()


def test_block_objective_alignment_status_enum_validates():
    """An entry with non-canonical status (e.g. "unknown") FAILS the
    JSON-LD schema's enum constraint."""
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    page = json.loads(PAGE_SCHEMA_PATH.read_text())
    id_to_schema = {}
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
    sub_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": page["$id"] + "#/$defs/ObjectiveAlignment",
    }
    validator = Draft202012Validator(sub_schema, registry=registry)

    invalid_entry = {
        "objective_id": "TO-01",
        "declared_bloom": "apply",
        "status": "unknown",
    }
    errors = list(validator.iter_errors(invalid_entry))
    assert errors, "expected schema validation to fail on non-canonical status"
    assert any("status" in str(e.absolute_path) or "unknown" in e.message for e in errors)


def test_block_objective_alignment_min_required_fields():
    """An entry missing 'status' FAILS schema validation; an entry with
    only the 3 required fields PASSES."""
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    page = json.loads(PAGE_SCHEMA_PATH.read_text())
    id_to_schema = {}
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
    sub_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": page["$id"] + "#/$defs/ObjectiveAlignment",
    }
    validator = Draft202012Validator(sub_schema, registry=registry)

    # Missing 'status' fails.
    missing_status = {
        "objective_id": "TO-01",
        "declared_bloom": "apply",
    }
    errors = list(validator.iter_errors(missing_status))
    assert errors, "expected schema validation to fail when 'status' is missing"

    # Three required fields only — passes.
    minimal_valid = {
        "objective_id": "TO-01",
        "declared_bloom": "apply",
        "status": "unverifiable",
    }
    errors = list(validator.iter_errors(minimal_valid))
    assert errors == [], (
        f"unexpected errors on minimal-valid entry: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_block_objective_alignment_observed_bloom_nullable():
    """observed_bloom: null is valid (it's nullable in the enum)."""
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    page = json.loads(PAGE_SCHEMA_PATH.read_text())
    id_to_schema = {}
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
    sub_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": page["$id"] + "#/$defs/ObjectiveAlignment",
    }
    validator = Draft202012Validator(sub_schema, registry=registry)

    null_observed = {
        "objective_id": "TO-01",
        "declared_bloom": "apply",
        "observed_bloom": None,
        "statement_entailment_score": None,
        "action_verb_present": False,
        "status": "unverifiable",
    }
    errors = list(validator.iter_errors(null_observed))
    assert errors == [], (
        f"unexpected errors on null observed_bloom: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_block_legacy_load_round_trip_clean():
    """Legacy blocks (no objective_alignment field on disk) still load
    and round-trip clean. Pin the back-compat invariant: a Block
    constructor call from a pre-Wave-1.7 corpus that doesn't pass
    objective_alignment defaults it to an empty tuple, and JSON-LD emit
    omits the key — byte-stable with pre-Wave-1.7 emit shape."""
    block = Block(
        block_id="week_01_overview#concept_intro_0",
        block_type="concept",
        page_id="week_01_overview",
        sequence=0,
        content="concept text",
        bloom_level="understand",
        objective_ids=("TO-01",),
    )
    assert block.objective_alignment == ()
    entry = block.to_jsonld_entry()
    assert "objectiveAlignment" not in entry
    # Hash matches a Block where the field is explicitly defaulted.
    explicit_default = Block(
        block_id="week_01_overview#concept_intro_0",
        block_type="concept",
        page_id="week_01_overview",
        sequence=0,
        content="concept text",
        bloom_level="understand",
        objective_ids=("TO-01",),
        objective_alignment=(),
    )
    assert block.compute_content_hash() == explicit_default.compute_content_hash()
