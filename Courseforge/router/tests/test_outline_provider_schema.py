"""Wave 1.5 W1.5.A — outline-tier ``key_claims`` schema bump regression.

Pins the back-compat ``oneOf`` contract on
``Courseforge.generators._outline_provider._BLOCK_TYPE_JSON_SCHEMAS``:

- legacy ``List[str]`` (every existing fixture + every existing corpus
  emit) MUST keep validating clean against the bumped schema.
- new structured ``List[{claim, source_chunk_ids[]}]`` validates.
- mixed-shape arrays (one string + one object in the same list) FAIL
  via the canonical ``jsonschema`` ``oneOf`` rejection.
- empty ``source_chunk_ids: []`` FAILS the inner ``minItems: 1`` bound.
- Block content-hash round-trips deterministically for the legacy
  shape — addresses the §6.5 hash-stability concern in the plan.

Plan: ``plans/gpt-feedback-2-wave1.5-per-claim-attribution-2026-05.md``
§4 Tests 1 + 3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import jsonschema
import pytest

# Block lives at Courseforge/scripts/blocks.py — mirror the import
# bridge used by the rest of the router test suite.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from Courseforge.generators._outline_provider import (  # noqa: E402
    _BLOCK_TYPE_JSON_SCHEMAS,
    _CONTENT_TYPE_ENUM,
    _OUTLINE_KIND_BOUNDS,
)
from lib.validators.content_type import (  # noqa: E402
    get_valid_chunk_types,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _valid_payload(
    *,
    block_type: str,
    key_claims: List[Any],
) -> Dict[str, Any]:
    """Build a structurally-valid outline payload around the supplied
    ``key_claims`` shape so the only contract under test is the
    ``key_claims`` arm of the ``oneOf``.

    Each block_type-specific extra (assessment_item distractors+stem,
    prereq_set prerequisitePages) is filled in just enough to satisfy
    the per-block-type ``required`` list — these tests are scoped to
    the ``key_claims`` shape, not to the wider per-block-type schema.
    """
    # Honour the per-block-type ``section_skeleton`` bound: atomic
    # block_types (e.g. callout) declare ``(0, 0)`` so the section
    # skeleton MUST be the empty list. Block_types whose minItems
    # >= 1 get a single placeholder section.
    section_min, _section_max = (
        _OUTLINE_KIND_BOUNDS.get(block_type, {}).get("section_skeleton", (0, 0))
    )
    section_skeleton: List[Dict[str, Any]]
    if section_min <= 0:
        section_skeleton = []
    else:
        section_skeleton = [{"heading": "Definition"}]

    base: Dict[str, Any] = {
        "block_id": f"page_01#{block_type}_x_0",
        "block_type": block_type,
        "content_type": "explanation",
        "bloom_level": "understand",
        "objective_refs": ["TO-01"],
        "curies": ["sh:NodeShape"],
        "key_claims": key_claims,
        "section_skeleton": section_skeleton,
        "source_refs": [
            {"sourceId": "semantik:shacl-spec#sec1", "role": "primary"},
        ],
        "structural_warnings": [],
    }
    if block_type == "assessment_item":
        base["stem"] = "Which of the following describes a SHACL node shape?"
        base["answer_key"] = "A"
        base["distractors"] = [
            {"text": "An OWL class declaration."},
            {"text": "A SHACL node shape declaring constraints."},
        ]
        base["correct_answer_index"] = 1
        # assessment_item bounds are (1, 4) so legacy/structured arms
        # need at least 1 item — the caller passes that via key_claims.
        # section_skeleton bounds: (1, 2) — the default [{...}] satisfies.
    elif block_type == "prereq_set":
        base["prerequisitePages"] = ["page_00"]
    return base


# Representative legacy ``List[str]`` ``key_claims`` values per
# block_type, sized to satisfy each type's ``minItems`` bound.
_LEGACY_CLAIMS_BY_TYPE: Dict[str, List[str]] = {
    "concept": [
        "A SHACL node shape declares constraints that fire on target nodes.",
        "Each constraint references a property predicate.",
    ],
    "assessment_item": [
        "Stem references the SHACL node shape concept.",
    ],
    "prereq_set": [
        "Learners must know RDF triples before SHACL.",
    ],
    "summary_takeaway": [
        "SHACL gates structural validity of an RDF graph.",
        "Node shapes are composable with property shapes.",
    ],
    "callout": [
        "Note: SHACL targets are not the same as OWL ranges.",
    ],
}


# Structured-shape ``key_claims`` mirroring the same per-type bounds.
_STRUCTURED_CLAIMS_BY_TYPE: Dict[str, List[Dict[str, Any]]] = {
    "concept": [
        {
            "claim": "A SHACL node shape declares constraints "
            "that fire on target nodes.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
        },
        {
            "claim": "Each constraint references a property predicate.",
            "source_chunk_ids": [
                "semantik:shacl-spec#sec2",
                "semantik:shacl-spec#sec3",
            ],
        },
    ],
    "assessment_item": [
        {
            "claim": "Stem references the SHACL node shape concept.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
        },
    ],
    "prereq_set": [
        {
            "claim": "Learners must know RDF triples before SHACL.",
            "source_chunk_ids": ["semantik:rdf-primer#triples"],
        },
    ],
    "summary_takeaway": [
        {
            "claim": "SHACL gates structural validity of an RDF graph.",
            "source_chunk_ids": ["semantik:shacl-spec#sec0"],
        },
        {
            "claim": "Node shapes are composable with property shapes.",
            "source_chunk_ids": ["semantik:shacl-spec#sec5"],
        },
    ],
    "callout": [
        {
            "claim": "Note: SHACL targets are not the same as OWL ranges.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
        },
    ],
}


# --------------------------------------------------------------------------- #
# Test 1 — legacy List[str] validates against the bumped schema
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "block_type",
    ["concept", "assessment_item", "prereq_set", "summary_takeaway"],
)
def test_legacy_list_str_validates(block_type: str) -> None:
    """Legacy ``List[str]`` shape MUST keep validating after the
    Wave 1.5 W1.5.A ``oneOf`` bump — every existing fixture and every
    existing corpus emit relies on this back-compat arm."""
    schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
    payload = _valid_payload(
        block_type=block_type,
        key_claims=_LEGACY_CLAIMS_BY_TYPE[block_type],
    )
    # Should not raise.
    jsonschema.Draft202012Validator(schema).validate(payload)


# --------------------------------------------------------------------------- #
# Test 2 — structured List[{claim, source_chunk_ids}] validates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "block_type",
    ["concept", "assessment_item", "prereq_set", "summary_takeaway"],
)
def test_structured_list_objects_validates(block_type: str) -> None:
    """New structured shape — list of objects each carrying ``claim``
    + non-empty ``source_chunk_ids`` — MUST validate against the
    bumped schema for every representative block_type."""
    schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
    payload = _valid_payload(
        block_type=block_type,
        key_claims=_STRUCTURED_CLAIMS_BY_TYPE[block_type],
    )
    # Should not raise.
    jsonschema.Draft202012Validator(schema).validate(payload)


# --------------------------------------------------------------------------- #
# Test 3 — mixed-shape arrays FAIL
# --------------------------------------------------------------------------- #


def test_mixed_shape_fails() -> None:
    """An array containing one string + one structured object MUST
    FAIL ``oneOf`` validation. Mixed-shape is the contract violation
    plan §6.1 explicitly rejects."""
    schema = _BLOCK_TYPE_JSON_SCHEMAS["concept"]
    payload = _valid_payload(
        block_type="concept",
        key_claims=[
            "A bare-string legacy claim.",
            {
                "claim": "A structured claim mixed in.",
                "source_chunk_ids": ["semantik:shacl-spec#sec1"],
            },
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(payload)


# --------------------------------------------------------------------------- #
# Test 4 — empty source_chunk_ids: [] fails (minItems: 1)
# --------------------------------------------------------------------------- #


def test_empty_source_chunk_ids_fails() -> None:
    """Inner ``source_chunk_ids`` MUST carry at least one chunk_id —
    a per-claim attribution with zero chunks is structurally
    meaningless and rejected by the inner ``minItems: 1`` bound."""
    schema = _BLOCK_TYPE_JSON_SCHEMAS["concept"]
    payload = _valid_payload(
        block_type="concept",
        key_claims=[
            {
                "claim": "A claim with no attribution.",
                "source_chunk_ids": [],
            },
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(payload)


# --------------------------------------------------------------------------- #
# Test 5 — Block hash round-trips deterministically for legacy shape
# --------------------------------------------------------------------------- #


def test_legacy_block_hash_round_trip_stable() -> None:
    """Construct a Block whose ``content["key_claims"]`` is the legacy
    ``List[str]`` shape; compute hash; serialise the entire content
    dict via ``json.dumps(..., sort_keys=True, ensure_ascii=False)``;
    re-deserialise; recompute hash. The two hashes MUST be byte-
    identical.

    This pins the §6.5 hash-stability promise: re-emitting a
    legacy-shape block under the bumped schema does not shift its
    canonical content hash."""
    legacy_content = {
        "curies": ["sh:NodeShape"],
        "key_claims": _LEGACY_CLAIMS_BY_TYPE["concept"],
        "content_type": "explanation",
    }
    block_a = Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content=legacy_content,
        objective_ids=("TO-01",),
    )
    hash_a = block_a.compute_content_hash()

    # Round-trip through json.dumps + json.loads with the canonical
    # serialisation flags.
    serialised = json.dumps(
        legacy_content, sort_keys=True, ensure_ascii=False
    )
    deserialised = json.loads(serialised)
    block_b = Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content=deserialised,
        objective_ids=("TO-01",),
    )
    hash_b = block_b.compute_content_hash()

    assert hash_a == hash_b, (
        f"Block content-hash drifted across round-trip: "
        f"{hash_a!r} != {hash_b!r}"
    )


# --------------------------------------------------------------------------- #
# content_type enum constraint (2026-05-15 Qwen off-vocab regression)
# --------------------------------------------------------------------------- #


def test_content_type_property_is_enum_constrained() -> None:
    """Every per-block-type outline schema must constrain ``content_type``
    to the canonical chunk-type enum.

    Observed 2026-05-15 on Qwen-7B: the schema had ``content_type:
    {"type": "string", "minLength": 1}`` (free-form), and Ollama
    structured-output decoding let the model drift off-vocabulary
    ('text', 'definition_and_example', 'place value and rounding') —
    every one of which fails ``BlockContentTypeValidator`` at the
    inter-tier seam. Pinning the enum into the schema forces a canonical
    value at decode time.
    """
    canonical = sorted(get_valid_chunk_types())
    assert _CONTENT_TYPE_ENUM == canonical, (
        "_CONTENT_TYPE_ENUM drifted from the canonical chunk-type "
        f"taxonomy: {_CONTENT_TYPE_ENUM!r} != {canonical!r}"
    )
    for block_type, schema in _BLOCK_TYPE_JSON_SCHEMAS.items():
        ct_prop = schema["properties"]["content_type"]
        assert ct_prop.get("enum") == canonical, (
            f"{block_type}: content_type schema is not enum-constrained "
            f"to the canonical taxonomy. Got {ct_prop!r}. A free-form "
            f"content_type lets the outline model drift off-vocabulary."
        )


# --------------------------------------------------------------------------- #
# Test 6 — plan §4 Test 3: legacy fixtures all validate clean
# --------------------------------------------------------------------------- #


def test_legacy_fixtures_validate_against_bumped_schema() -> None:
    """Plan §4 Test 3: the first 5 distinct legacy ``List[str]``
    ``key_claims`` values across the canonical block_types MUST each
    validate clean against the bumped per-block-type schema. Catches
    a future schema author tightening the legacy arm of the ``oneOf``
    and silently breaking every existing fixture + existing rebuild.

    The plan permits back-fill with synthetic legacy-shape examples
    when fixtures don't yield 5 distinct values — that's exactly what
    ``_LEGACY_CLAIMS_BY_TYPE`` carries (5 entries: concept,
    assessment_item, prereq_set, callout, summary_takeaway).
    """
    canonical_types = [
        "concept",
        "assessment_item",
        "prereq_set",
        "callout",
        "summary_takeaway",
    ]
    assert len(canonical_types) >= 5, (
        "Plan §4 Test 3 requires at least 5 distinct legacy "
        "fixture values — back-fill list is too short."
    )
    seen_claims: List[List[str]] = []
    for block_type in canonical_types:
        claims = _LEGACY_CLAIMS_BY_TYPE[block_type]
        # Distinct: assert each block_type's claim list is novel.
        assert claims not in seen_claims, (
            f"Legacy fixture for {block_type!r} duplicates a prior "
            f"block_type's value — Plan §4 Test 3 requires distinct "
            f"values."
        )
        seen_claims.append(claims)
        schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
        payload = _valid_payload(block_type=block_type, key_claims=claims)
        # Should not raise.
        jsonschema.Draft202012Validator(schema).validate(payload)
