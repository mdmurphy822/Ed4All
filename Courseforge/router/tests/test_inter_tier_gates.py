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
    key_claims: Optional[List[str]] = None,
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
