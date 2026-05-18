"""Phase 3 Subtask 53 — inter-tier gate adapter regression tests.

Pins the per-adapter behavior contract for the Block-input validator
shims in Courseforge.router.inter_tier_gates. Each test exercises a
single adapter end-to-end with a minimal Block fixture and asserts
the GateResult.action signal the router consumes.

Action contract (Phase 4 §1):
- regenerate: content-side semantic miss the rewrite tier could fix
  on a re-roll (curie / content_type adapters).
- block: structural miss the rewrite tier can't fix because the
  outline references something that doesn't exist downstream
  (objective / source_id adapters).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Block lives at Courseforge/scripts/blocks.py — mirror the import
# bridge used by the rest of the router test suite.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    BlockCurieAnchoringValidator,
    BlockPageObjectivesValidator,
    BlockSourceRefValidator,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _outline_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    page_id: str = "page_01",
    sequence: int = 0,
    curies: Tuple[str, ...] = ("ed4all:Foo",),
    key_claims: Optional[List[Any]] = None,
    content_type: str = "definition",
    objective_ids: Tuple[str, ...] = ("TO-01",),
    source_references: Tuple[Dict[str, Any], ...] = (),
    source_ids: Tuple[str, ...] = (),
) -> Block:
    """Minimal outline-tier Block fixture.

    Outline tier: ``block.content`` is a dict carrying curies +
    key_claims + content_type. Phase-3 inter-tier adapters audit this
    shape only; rewrite-tier (HTML string) blocks are silently
    skipped per Phase 3.5 scope split.

    Wave 1.5 W1.5.A: ``key_claims`` accepts either the legacy
    ``List[str]`` shape (default — preserves every existing call-site)
    OR the structured ``List[{claim, source_chunk_ids[]}]`` shape
    (new authoring under the Wave 1.5 schema bump). The helper does
    NOT enforce shape — callers pass whichever shape the test
    fixture needs.
    """
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content={
            "curies": list(curies),
            "key_claims": (
                list(key_claims)
                if key_claims is not None
                else ["The ed4all:Foo predicate marks anchoring."]
            ),
            "content_type": content_type,
        },
        objective_ids=tuple(objective_ids),
        source_references=tuple(source_references),
        source_ids=tuple(source_ids),
    )


# --------------------------------------------------------------------------- #
# 1. CURIE anchoring
# --------------------------------------------------------------------------- #


def test_block_curie_anchoring_passes_when_curies_present():
    """Outline-tier Block declares CURIEs and at least one is present
    in key_claims → gate passes, action remains None."""
    blocks = [
        _outline_block(
            block_id="page_01#concept_a_0",
            curies=("ed4all:Foo", "ed4all:Bar"),
            key_claims=[
                "The ed4all:Foo predicate marks the anchoring relation.",
                "Examples include ed4all:Bar usage in citations.",
            ],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.action is None
    assert result.score == 1.0
    # No critical issues for a passing run.
    assert all(i.severity != "critical" for i in result.issues)


def test_block_curie_anchoring_returns_regenerate_when_missing():
    """Outline-tier Block with empty curies list fails closed with
    action='regenerate' so the router can re-roll the outline tier."""
    blocks = [
        _outline_block(
            block_id="page_01#concept_b_0",
            curies=(),  # Empty CURIE list — the regenerate trigger.
            key_claims=["Some text without CURIEs."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_MISSING_CURIES" in codes


# --------------------------------------------------------------------------- #
# 1b. Minted-CURIE concept-aware anchoring (v0.3.0)
# --------------------------------------------------------------------------- #


def test_minted_curie_anchored_via_surface_form():
    """A minted CURIE is anchored when one of its vocabulary surface
    forms appears in the block text — the synthetic CURIE token itself
    need NOT appear literally."""
    minted_map = {
        "samplecoursea:slope": {
            "canonical": "slope",
            "surface_forms": ["slope", "gradient"],
        },
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_slope_0",
            curies=("samplecoursea:slope",),
            # The minted CURIE token never appears; the surface form
            # "gradient" does.
            key_claims=["The gradient of a line measures its steepness."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is True
    assert result.action is None


def test_minted_curie_not_anchored_when_no_surface_form_in_text():
    """A minted CURIE whose surface forms are absent from the block
    text fails the anchoring gate (action=regenerate)."""
    minted_map = {
        "samplecoursea:slope": {
            "canonical": "slope",
            "surface_forms": ["slope", "gradient"],
        },
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_slope_0",
            curies=("samplecoursea:slope",),
            key_claims=["This sentence mentions nothing about lines."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_CURIE_NOT_ANCHORED" in codes


def test_rdf_curie_still_uses_literal_check_with_map():
    """A CURIE NOT in the minted map (an RDF CURIE) keeps the legacy
    literal-token anchoring even when a minted_curie_map is supplied."""
    minted_map = {
        "samplecoursea:slope": {
            "canonical": "slope",
            "surface_forms": ["slope"],
        },
    }
    # The RDF CURIE token must appear literally in the text to anchor.
    blocks = [
        _outline_block(
            block_id="page_01#concept_a_0",
            curies=("sh:NodeShape",),
            key_claims=["The sh:NodeShape constrains the focus node."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is True

    # And an RDF CURIE absent from the text fails — surface forms of
    # an unrelated minted CURIE do not rescue it.
    blocks_fail = [
        _outline_block(
            block_id="page_01#concept_b_0",
            curies=("sh:NodeShape",),
            key_claims=["The slope of a line is its steepness."],
        ),
    ]
    result_fail = BlockCurieAnchoringValidator().validate({
        "blocks": blocks_fail,
        "minted_curie_map": minted_map,
    })
    assert result_fail.passed is False
    assert result_fail.action == "regenerate"


def test_no_minted_map_byte_identical_to_legacy():
    """Without a minted_curie_map, anchoring behaviour is byte-identical
    to the pre-minting contract: a minted-style CURIE that does not
    appear literally fails (no surface-form rescue)."""
    blocks = [
        _outline_block(
            block_id="page_01#concept_slope_0",
            curies=("samplecoursea:slope",),
            key_claims=["The gradient of a line measures its steepness."],
        ),
    ]
    # No minted_curie_map → the legacy literal check applies; "slope"
    # token absent → fails.
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"


# --------------------------------------------------------------------------- #
# 1c. R4 — vacuous-anchoring guard (word-boundary + short/common-token)
# --------------------------------------------------------------------------- #


def test_short_surface_form_does_not_substring_anchor():
    """R4.1: a surface form ``"ion"`` must NOT anchor a block whose text
    only says "definition" — the pre-R4 unbounded substring match
    false-passed here."""
    minted_map = {
        "bio:ion": {"canonical": "ion", "surface_forms": ["ion"]},
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_ion_0",
            curies=("bio:ion",),
            key_claims=["The definition of a charged particle is given."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is False
    assert result.action == "regenerate"


def test_surface_form_word_boundary_match_anchors():
    """R4.1: the same ``"ion"`` surface form DOES anchor when it appears
    as a whole word."""
    minted_map = {
        "bio:ion": {"canonical": "ion", "surface_forms": ["ion"]},
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_ion_0",
            curies=("bio:ion",),
            key_claims=["An ion carries a net electric charge."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is True


def test_stopword_surface_form_does_not_anchor():
    """R4.2: a stopword surface form (``"is"``) must not anchor
    arbitrary prose — it is filtered by the short/common-token guard."""
    minted_map = {
        "x:is": {"canonical": "is", "surface_forms": ["is"]},
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_is_0",
            curies=("x:is",),
            key_claims=["This sentence is entirely unrelated prose."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is False
    assert result.action == "regenerate"


def test_minted_literal_token_arm_anchors():
    """R4: the minted arm anchors on literal-token OR surface-form. A
    block carrying the minted CURIE token literally in its text (e.g.
    the R1 force-injection) passes via the literal arm even when no
    surface form is present."""
    minted_map = {
        "samplecoursea:slope": {
            "canonical": "slope",
            "surface_forms": ["gradient"],
        },
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_slope_0",
            curies=("samplecoursea:slope",),
            # Neither "slope" nor "gradient" in prose, but the minted
            # CURIE token itself is present (force-injected shape).
            key_claims=["A discussion referencing samplecoursea:slope here."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is True


def test_short_domain_term_still_anchors():
    """R4.2: the guard is a stoplist, not a blunt length cut — a legit
    short domain term like ``"pH"`` still anchors via word boundary."""
    minted_map = {
        "chem:ph": {"canonical": "pH", "surface_forms": ["pH"]},
    }
    blocks = [
        _outline_block(
            block_id="page_01#concept_ph_0",
            curies=("chem:ph",),
            key_claims=["Measure the pH of the solution carefully."],
        ),
    ]
    result = BlockCurieAnchoringValidator().validate({
        "blocks": blocks,
        "minted_curie_map": minted_map,
    })
    assert result.passed is True


# --------------------------------------------------------------------------- #
# 2. Content-type taxonomy
# --------------------------------------------------------------------------- #


def test_block_content_type_validates_against_taxonomy(monkeypatch):
    """Outline-tier Block declaring a content_type outside the
    canonical ChunkType enum fails closed with action='regenerate'.
    """
    # Force enforcement so get_valid_chunk_types is non-trivial.
    monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "true")
    # Drop the lru_cache on get_valid_chunk_types so the env-var flip
    # is honoured. The taxonomy file is bundled.
    from lib.validators import content_type as _ct
    _ct.get_valid_chunk_types.cache_clear()

    valid_types = sorted(_ct.get_valid_chunk_types())
    assert valid_types, "ChunkType taxonomy must be non-empty for this test"
    # Pick a value that's actually in the taxonomy for the positive
    # arm.
    good_value = valid_types[0]
    bad_value = "definitelynotacontenttype_xyz"
    blocks_good = [
        _outline_block(
            block_id="page_01#good_0",
            content_type=good_value,
        ),
    ]
    blocks_bad = [
        _outline_block(
            block_id="page_01#bad_0",
            content_type=bad_value,
        ),
    ]
    good_result = BlockContentTypeValidator().validate({"blocks": blocks_good})
    bad_result = BlockContentTypeValidator().validate({"blocks": blocks_bad})

    assert good_result.passed is True
    assert good_result.action is None

    assert bad_result.passed is False
    assert bad_result.action == "regenerate"
    codes = [i.code for i in bad_result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_INVALID_CONTENT_TYPE" in codes


# --------------------------------------------------------------------------- #
# 3. Page objective coverage
# --------------------------------------------------------------------------- #


def test_block_page_objectives_returns_block_action_when_objective_unmatched(
    tmp_path,
):
    """Outline-tier Block referencing an objective_id that doesn't
    resolve against the canonical objectives JSON fails closed with
    action='block' (structural miss — re-roll won't help)."""
    objectives_path = tmp_path / "synthesized_objectives.json"
    objectives_path.write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Stub TO 1"},
                {"id": "TO-02", "statement": "Stub TO 2"},
            ],
            "chapter_objectives": [
                {"id": "CO-01", "statement": "Stub CO 1"},
            ],
        }),
        encoding="utf-8",
    )
    blocks = [
        # Positive: references TO-01 which is in the canonical set.
        _outline_block(
            block_id="page_01#good_0",
            objective_ids=("TO-01",),
        ),
        # Negative: references TO-99 which is unknown.
        _outline_block(
            block_id="page_01#bad_1",
            objective_ids=("TO-99",),
        ),
    ]
    result = BlockPageObjectivesValidator().validate({
        "blocks": blocks,
        "objectives_path": str(objectives_path),
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNKNOWN_OBJECTIVE" in codes


# --------------------------------------------------------------------------- #
# 4. Source-ref manifest resolution
# --------------------------------------------------------------------------- #


def test_block_source_ref_returns_block_action_when_sourceid_unknown():
    """Outline-tier Block declaring a sourceId that doesn't resolve
    against the staging manifest fails closed with action='block'."""
    valid_ids = {"dart:textbook_a#chap1_para3"}
    blocks = [
        # Positive: sid resolves.
        _outline_block(
            block_id="page_01#good_0",
            source_references=(
                {"sourceId": "dart:textbook_a#chap1_para3"},
            ),
        ),
        # Negative: sid doesn't resolve against the manifest.
        _outline_block(
            block_id="page_01#bad_1",
            source_references=(
                {"sourceId": "dart:textbook_a#nonexistent_block"},
            ),
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID" in codes


# --------------------------------------------------------------------------- #
# 4b. Wave 1.5 W1.5.C: per-claim source attribution consistency
# --------------------------------------------------------------------------- #


class _RecordingCapture:
    """Minimal capture stub that records every log_decision payload."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def test_block_source_ref_per_claim_passes_when_chunk_ids_in_block_refs():
    """Structured key_claims shape — every source_chunk_id appears in
    the block's resolved source_refs[] → action=None, passed=True."""
    valid_ids = {"dart:textbook_a#chap1_para3"}
    blocks = [
        _outline_block(
            block_id="page_01#concept_0",
            source_references=(
                {"sourceId": "dart:textbook_a#chap1_para3"},
            ),
            key_claims=[
                {
                    "claim": "An RDF graph carries node-level constraints.",
                    "source_chunk_ids": ["dart:textbook_a#chap1_para3"],
                },
            ],
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
    })
    assert result.passed is True
    assert result.action is None
    # No claim-level miss issues should fire.
    codes = [i.code for i in result.issues]
    assert "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS" not in codes


def test_block_source_ref_per_claim_regenerates_when_chunk_id_missing():
    """Structured key_claims shape — one source_chunk_id NOT in
    block.source_refs[] → action='regenerate', warning issue with code
    OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS, decision-capture event
    surfaces claim_misses_count >= 1.

    Mirrors Plan §4 Test 2 fixture verbatim:
      block.source_references = [{"sourceId": "dart:baz#qux"}]
      block.content["key_claims"] = [
        {"claim": "foo", "source_chunk_ids": ["dart:foo#bar"]}
      ]
    Expectation: passed=False, action="regenerate" (NOT "block" — per-
    claim only), per-claim miss code, capture event surfaces
    claim_misses_count >= 1.
    """
    capture = _RecordingCapture()
    valid_ids = {"dart:baz#qux"}
    blocks = [
        _outline_block(
            block_id="page_01#concept_5",
            source_references=({"sourceId": "dart:baz#qux"},),
            key_claims=[
                {
                    "claim": "foo",
                    "source_chunk_ids": ["dart:foo#bar"],
                },
            ],
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
        "decision_capture": capture,
    })
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS" in codes
    # Per-claim miss must be warning severity (not critical).
    per_claim = [
        i for i in result.issues
        if i.code == "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS"
    ]
    assert all(i.severity == "warning" for i in per_claim)
    # Decision-capture event must surface claim_misses_count >= 1 in
    # the rationale (the per-block emit serialises signals as
    # ``key=value`` pairs into the rationale tail).
    assert capture.events, "expected at least one decision-capture event"
    relevant = [
        e for e in capture.events
        if e.get("decision_type") == "block_source_ref_check"
    ]
    assert relevant, "expected block_source_ref_check event"
    rationales = " ".join(
        str(e.get("rationale", "")) for e in relevant
    )
    assert "claim_misses_count" in rationales
    assert "claims_with_attribution_count" in rationales
    assert "claims_total_count" in rationales
    # claim_misses_count must be at least 1 (per the plan §4 Test 2
    # contract). The renderer uses repr() so the integer is stringified
    # without quoting; substring search for "claim_misses_count=1" or
    # higher.
    assert any(
        "claim_misses_count=" in str(e.get("rationale", ""))
        and "claim_misses_count=0" not in str(e.get("rationale", ""))
        for e in relevant
    )


def test_block_source_ref_block_level_dominates_claim_level():
    """Both block-level structural miss AND per-claim attribution miss
    fire on the same block → action='block' (block-level dominates).
    Both issue codes appear in the issue list."""
    valid_ids = {"dart:textbook_a#chap1_para3"}
    blocks = [
        _outline_block(
            block_id="page_01#bad_block",
            # Block-level miss: sourceId not in valid_ids
            source_references=({"sourceId": "dart:textbook_a#missing_block"},),
            key_claims=[
                {
                    "claim": "A claim citing a chunk not in source_refs[].",
                    # Claim-level miss: chunk_id not in block.source_refs[]
                    "source_chunk_ids": ["dart:textbook_a#unrelated_chunk"],
                },
            ],
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
    })
    assert result.passed is False
    # Block-level dominance: action MUST be "block" even though a
    # claim-level miss is also present.
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    assert "OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID" in codes
    assert "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS" in codes


def test_block_source_ref_legacy_list_str_skips_per_claim_walk():
    """Legacy List[str] key_claims shape skips the per-claim walk —
    no new claim-level issues emitted (back-compat preserved per
    Wave 1.5 §6.1)."""
    valid_ids = {"dart:textbook_a#chap1_para3"}
    blocks = [
        _outline_block(
            block_id="page_01#legacy_0",
            source_references=(
                {"sourceId": "dart:textbook_a#chap1_para3"},
            ),
            # Legacy flat string list (the pre-Wave-1.5 shape every
            # existing fixture emits).
            key_claims=[
                "An RDF graph carries node-level constraints.",
                "SHACL targets fire on property predicate failure.",
            ],
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
    })
    assert result.passed is True
    assert result.action is None
    codes = [i.code for i in result.issues]
    assert "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS" not in codes


def test_block_source_ref_per_claim_mixed_shape_skips_string_entries():
    """Mixed shape (one string + one object claim) — the schema validator
    rejects this upstream (oneOf semantics), but the validator's
    claim-level walk treats string entries as skip (defensive). The
    object entry is still walked — when its source_chunk_ids miss the
    block.source_refs[], the warning fires."""
    valid_ids = {"dart:textbook_a#chap1_para3"}
    blocks = [
        _outline_block(
            block_id="page_01#mixed_0",
            source_references=(
                {"sourceId": "dart:textbook_a#chap1_para3"},
            ),
            key_claims=[
                # String entry — silently skipped by the per-claim walk.
                "A legacy-shape claim string.",
                # Object entry with a chunk_id NOT in source_refs[] —
                # surfaces the warning.
                {
                    "claim": "An object claim citing an outside chunk.",
                    "source_chunk_ids": ["dart:textbook_a#stranger_chunk"],
                },
            ],
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
    })
    # Object entry's miss surfaces — confirms string-entry skip is
    # defensive (the walk doesn't crash on the mixed shape).
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS" in codes


def test_block_source_ref_validator_version_bumped_to_1_2_0():
    """W1.5.C bumps BlockSourceRefValidator.version from 1.1.0 → 1.2.0
    so the audit trail records the contract change."""
    assert BlockSourceRefValidator.version == "1.2.0"


# --------------------------------------------------------------------------- #
# Cross-cutting coverage (sanity)
# --------------------------------------------------------------------------- #


def test_missing_blocks_input_fails_closed_for_every_adapter():
    """Every adapter MUST fail closed when inputs['blocks'] is absent
    (defensive — protects against orchestrator misconfiguration)."""
    for cls in (
        BlockCurieAnchoringValidator,
        BlockContentTypeValidator,
        BlockPageObjectivesValidator,
        BlockSourceRefValidator,
    ):
        result = cls().validate({})
        assert result.passed is False, (
            f"{cls.__name__} should fail closed on missing blocks"
        )
        codes = [i.code for i in result.issues if i.severity == "critical"]
        assert "MISSING_BLOCKS_INPUT" in codes
