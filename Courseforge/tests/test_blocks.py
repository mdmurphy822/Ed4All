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
