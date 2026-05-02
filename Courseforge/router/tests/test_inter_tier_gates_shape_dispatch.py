"""Phase 3.5 Subtask 9 — shape-dispatch regression for inter-tier gate
adapters.

Pins the Phase-3.5 extension contract that the four ``Block*Validator``
adapters in ``Courseforge.router.inter_tier_gates`` discriminate on the
``Block.content`` shape and produce equivalent gate signals on both the
outline-tier (dict content) and rewrite-tier (HTML string content)
surfaces.

Coverage matrix (4 validators × 2 tiers × 2 outcomes = 16 tests):
- BlockCurieAnchoringValidator     × {dict, str} × {pass, fail}
- BlockContentTypeValidator        × {dict, str} × {pass, fail}
- BlockPageObjectivesValidator     × {dict, str} × {pass, fail}
- BlockSourceRefValidator          × {dict, str} × {pass, fail}

Plus 1 regression test asserting the legacy dict-content path is
byte-stable (same code paths the existing ``test_inter_tier_gates.py``
exercises continue to produce identical GateResult fields).

Worker M's flagged inconsistencies (Phase 3 review) are documented
inline:
- ``BlockPageObjectivesValidator`` requires ``valid_objective_ids``
  input key (asymmetric with the other three Block validators that
  take only ``blocks``).
- ``BlockContentTypeValidator`` enforces the chunk-type taxonomy
  (Trainforge-side enum), NOT the section-content-type taxonomy.
"""
from __future__ import annotations

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
# Fixture factories
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
    """Outline-tier (dict-content) Block fixture."""
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


def _rewrite_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    page_id: str = "page_01",
    sequence: int = 0,
    html: str = "<p>The ed4all:Foo predicate marks anchoring.</p>",
    objective_ids: Tuple[str, ...] = ("TO-01",),
    source_references: Tuple[Dict[str, Any], ...] = (),
    source_ids: Tuple[str, ...] = (),
) -> Block:
    """Rewrite-tier (str-content) Block fixture."""
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content=html,
        objective_ids=tuple(objective_ids),
        source_references=tuple(source_references),
        source_ids=tuple(source_ids),
    )


# --------------------------------------------------------------------------- #
# 1. CURIE anchoring — dict + str, pass + fail
# --------------------------------------------------------------------------- #


def test_curie_anchoring_dict_pass():
    """Dict-tier: declared CURIEs anchored in key_claims → pass."""
    blocks = [_outline_block(
        curies=("ed4all:Foo", "ed4all:Bar"),
        key_claims=[
            "The ed4all:Foo predicate marks the anchoring relation.",
            "Examples include ed4all:Bar usage in citations.",
        ],
    )]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.action is None


def test_curie_anchoring_dict_fail_empty_curies():
    """Dict-tier: empty curies list → action='regenerate'."""
    blocks = [_outline_block(
        curies=(),
        key_claims=["Plain text without CURIEs."],
    )]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_MISSING_CURIES" in codes


def test_curie_anchoring_str_pass():
    """Rewrite-tier (HTML): CURIE present in HTML body → pass."""
    blocks = [_rewrite_block(
        html=(
            "<section>"
            "<p>The <code>ed4all:Foo</code> predicate marks anchoring.</p>"
            "<p>See also ed4all:Bar for citation usage.</p>"
            "</section>"
        ),
    )]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.action is None


def test_curie_anchoring_str_fail_no_curies_in_html():
    """Rewrite-tier (HTML): no CURIEs in HTML body → action='regenerate'."""
    blocks = [_rewrite_block(
        html=(
            "<section><p>Plain prose with no CURIEs at all.</p></section>"
        ),
    )]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_MISSING_CURIES" in codes


# --------------------------------------------------------------------------- #
# 2. Content-type taxonomy — dict + str, pass + fail
# --------------------------------------------------------------------------- #


def _force_chunk_taxonomy(monkeypatch) -> List[str]:
    """Helper: force enforcement and clear the get_valid_chunk_types
    LRU cache so the env-var flip is honoured."""
    monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "true")
    from lib.validators import content_type as _ct
    _ct.get_valid_chunk_types.cache_clear()
    valid = sorted(_ct.get_valid_chunk_types())
    assert valid, "ChunkType taxonomy must be non-empty for this test"
    return valid


def test_content_type_dict_pass(monkeypatch):
    """Dict-tier: content['content_type'] is in chunk-type taxonomy → pass."""
    valid = _force_chunk_taxonomy(monkeypatch)
    good = valid[0]
    blocks = [_outline_block(content_type=good)]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.action is None


def test_content_type_dict_fail(monkeypatch):
    """Dict-tier: content_type outside taxonomy → action='regenerate'."""
    _force_chunk_taxonomy(monkeypatch)
    blocks = [_outline_block(content_type="bogusvalue_xyz")]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_INVALID_CONTENT_TYPE" in codes


def test_content_type_str_pass(monkeypatch):
    """Rewrite-tier (HTML): data-cf-content-type attribute is in chunk-type
    taxonomy → pass. Worker M's flag: chunk-type, NOT section-content-type."""
    valid = _force_chunk_taxonomy(monkeypatch)
    good = valid[0]
    blocks = [_rewrite_block(
        html=(
            f'<section data-cf-content-type="{good}">'
            "<p>Body content with ed4all:Foo anchor.</p>"
            "</section>"
        ),
    )]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.action is None


def test_content_type_str_fail(monkeypatch):
    """Rewrite-tier: data-cf-content-type attribute outside taxonomy →
    action='regenerate'."""
    _force_chunk_taxonomy(monkeypatch)
    blocks = [_rewrite_block(
        html=(
            '<section data-cf-content-type="not_a_real_type">'
            "<p>Body with ed4all:Foo.</p>"
            "</section>"
        ),
    )]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_INVALID_CONTENT_TYPE" in codes


# --------------------------------------------------------------------------- #
# 3. Page objectives — dict + str, pass + fail
# --------------------------------------------------------------------------- #
#
# Per Worker M's Phase 3 review flag: this validator's input contract
# uses ``valid_objective_ids`` (NOT ``objective_ids`` or
# ``page_objectives``). Documented inline so the contract is pinned by
# tests.


def test_page_objectives_dict_pass():
    """Dict-tier: block.objective_ids resolves against valid_objective_ids
    → pass."""
    blocks = [_outline_block(objective_ids=("TO-01",))]
    result = BlockPageObjectivesValidator().validate({
        "blocks": blocks,
        "valid_objective_ids": ["TO-01", "TO-02", "CO-01"],
    })
    assert result.passed is True
    assert result.action is None


def test_page_objectives_dict_fail_unknown_id():
    """Dict-tier: block.objective_ids includes an unknown id →
    action='block' (structural — re-roll won't help)."""
    blocks = [_outline_block(objective_ids=("TO-99",))]
    result = BlockPageObjectivesValidator().validate({
        "blocks": blocks,
        "valid_objective_ids": ["TO-01", "TO-02"],
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNKNOWN_OBJECTIVE" in codes


def test_page_objectives_str_pass():
    """Rewrite-tier (HTML): data-cf-objective-id attribute resolves
    against valid_objective_ids → pass. Helper falls back to scraping
    HTML when block.objective_ids is empty."""
    blocks = [_rewrite_block(
        html=(
            '<section><ul>'
            '<li data-cf-objective-id="TO-01">First objective</li>'
            '<li data-cf-objective-id="CO-02">Second objective</li>'
            '</ul></section>'
        ),
        # Empty structural field forces the str-path scraper to fire.
        objective_ids=(),
    )]
    result = BlockPageObjectivesValidator().validate({
        "blocks": blocks,
        "valid_objective_ids": ["TO-01", "CO-02"],
    })
    assert result.passed is True
    assert result.action is None


def test_page_objectives_str_fail_unknown_id():
    """Rewrite-tier (HTML): data-cf-objective-id attribute references an
    unknown objective → action='block'."""
    blocks = [_rewrite_block(
        html=(
            '<section><ul>'
            '<li data-cf-objective-id="TO-99">Bogus objective</li>'
            '</ul></section>'
        ),
        objective_ids=(),
    )]
    result = BlockPageObjectivesValidator().validate({
        "blocks": blocks,
        "valid_objective_ids": ["TO-01", "TO-02"],
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNKNOWN_OBJECTIVE" in codes


# --------------------------------------------------------------------------- #
# 4. Source-ref manifest — dict + str, pass + fail
# --------------------------------------------------------------------------- #


def test_source_ref_dict_pass():
    """Dict-tier: block.source_references resolves against the
    valid_source_ids universe → pass."""
    blocks = [_outline_block(
        source_references=({"sourceId": "dart:textbook_a#chap1_para3"},),
    )]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": ["dart:textbook_a#chap1_para3"],
    })
    assert result.passed is True
    assert result.action is None


def test_source_ref_dict_fail_unknown_sourceid():
    """Dict-tier: block.source_references includes an unknown sourceId →
    action='block'."""
    blocks = [_outline_block(
        source_references=({"sourceId": "dart:textbook_a#nonexistent_block"},),
    )]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": ["dart:textbook_a#chap1_para3"],
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID" in codes


def test_source_ref_str_pass():
    """Rewrite-tier (HTML): data-cf-source-ids attribute resolves against
    the valid_source_ids universe → pass. Helper merges scraped ids
    with the structural field (deduplicated)."""
    blocks = [_rewrite_block(
        html=(
            '<section data-cf-source-ids="dart:textbook_a#chap1_para3">'
            "<p>Body</p></section>"
        ),
        source_references=(),
        source_ids=(),
    )]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": ["dart:textbook_a#chap1_para3"],
    })
    assert result.passed is True
    assert result.action is None


def test_source_ref_str_fail_unknown_sourceid():
    """Rewrite-tier: data-cf-source-ids references an id that doesn't
    resolve against the manifest universe → action='block'."""
    blocks = [_rewrite_block(
        html=(
            '<section data-cf-source-ids="dart:textbook_a#missing_block">'
            "<p>Body</p></section>"
        ),
        source_references=(),
        source_ids=(),
    )]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": ["dart:textbook_a#chap1_para3"],
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID" in codes


# --------------------------------------------------------------------------- #
# Regression: legacy dict-content path is byte-stable
# --------------------------------------------------------------------------- #


def test_legacy_dict_path_byte_stable_curie_anchoring():
    """Regression — the dict-tier (legacy outline-tier) GateResult shape
    for BlockCurieAnchoringValidator is byte-stable across Phase 3.5's
    shape-dispatch refactor.

    Pins: passed/action/score/issue codes for the same input that
    ``test_inter_tier_gates.py::test_block_curie_anchoring_passes_when_curies_present``
    exercised. Phase 3.5 must not silently shift behavior on the legacy
    dict-content path.
    """
    blocks = [_outline_block(
        block_id="page_01#concept_a_0",
        curies=("ed4all:Foo", "ed4all:Bar"),
        key_claims=[
            "The ed4all:Foo predicate marks the anchoring relation.",
            "Examples include ed4all:Bar usage in citations.",
        ],
    )]
    result = BlockCurieAnchoringValidator().validate({"blocks": blocks})
    # Legacy (Phase 3) contract — every field pinned.
    assert result.passed is True
    assert result.action is None
    assert result.score == 1.0
    # No critical issues for a passing run on the legacy dict path.
    assert all(i.severity != "critical" for i in result.issues)
    # validator_name preserved (the gate_id seam the router consumes).
    assert result.validator_name == "outline_curie_anchoring"
