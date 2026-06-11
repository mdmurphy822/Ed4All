"""Escalation-skip predicate regression tests for the four Block-input
inter-tier validators (A4 gate-calibration review, MAJOR finding fix).

The skip predicate
``Courseforge.router.inter_tier_gates._is_unshipped_escalation_tombstone``
must mirror the packager's ship-exclusion filter at
``MCP/tools/pipeline_tools.py:5226``:

    ``escalation_marker is not None and not (content or "").strip()``

Three classes per validator:

(a) **Tombstone** (marker + empty content) — the packager filters it out
    of the shipped IMSCC, so the validator must SKIP it: not counted
    toward ``audited``, no issue emitted from it.
(b) **Salvaged** (marker + non-empty content) — the escalated-rewrite
    salvage path ships it, so the validator must AUDIT it: a
    deliberately-bad salvaged block produces its expected GateIssue.
(c) **No marker** — audited as always (control).

The original ``_is_escalated`` skipped on the marker alone, which
silently waived salvaged-with-content blocks the packager DOES ship —
the false premise this test fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Block lives at Courseforge/scripts/blocks.py — mirror the import bridge.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    BlockCurieAnchoringValidator,
    BlockPageObjectivesValidator,
    BlockSourceRefValidator,
    _is_unshipped_escalation_tombstone,
)

_MARKER = "outline_budget_exhausted"


# --------------------------------------------------------------------------- #
# Block builders — rewrite-tier (str content) shape so the salvage/tombstone
# distinction is exercised on the path the packager actually ships.
# --------------------------------------------------------------------------- #


def _str_block(
    *,
    block_id: str = "page_01#concept_x_0",
    block_type: str = "concept",
    content: str = "",
    escalation_marker: Optional[str] = None,
    objective_ids: Tuple[str, ...] = (),
    source_ids: Tuple[str, ...] = (),
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
        objective_ids=objective_ids,
        source_ids=source_ids,
        escalation_marker=escalation_marker,
    )


# A deliberately-bad rewrite-tier emit: no CURIE tokens, no
# data-cf-content-type attribute, no objective ids, a malformed sourceId.
# Each validator fails on it.
_BAD_HTML = (
    '<section data-cf-source-ids="not-a-valid-source-id">'
    "<p>plain prose with no anchored identifiers and no metadata</p>"
    "</section>"
)
# A clean rewrite-tier concept emit so the no-marker control passes the
# curie/content-type checks.
_GOOD_HTML = (
    '<section data-cf-content-type="explanation">'
    "<p>The ex:Thing predicate marks the anchoring relation here.</p>"
    "</section>"
)


# --------------------------------------------------------------------------- #
# Predicate-level direct tests
# --------------------------------------------------------------------------- #


def test_predicate_tombstone_marker_plus_empty_is_skipped():
    blk = _str_block(content="   ", escalation_marker=_MARKER)
    assert _is_unshipped_escalation_tombstone(blk) is True


def test_predicate_salvaged_marker_plus_content_is_not_skipped():
    blk = _str_block(content=_GOOD_HTML, escalation_marker=_MARKER)
    assert _is_unshipped_escalation_tombstone(blk) is False


def test_predicate_no_marker_is_not_skipped():
    blk = _str_block(content=_GOOD_HTML, escalation_marker=None)
    assert _is_unshipped_escalation_tombstone(blk) is False


def test_predicate_dict_outline_content_never_tombstone():
    # Outline-tier dict content can't be a shipped-HTML tombstone.
    blk = Block(
        block_id="page_01#concept_d_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={"curies": [], "key_claims": [], "content_type": "definition"},
        escalation_marker=_MARKER,
    )
    assert _is_unshipped_escalation_tombstone(blk) is False


# --------------------------------------------------------------------------- #
# Per-validator (a) tombstone / (b) salvaged / (c) no-marker matrix
# --------------------------------------------------------------------------- #


def _curie_inputs(blocks: List[Block]) -> Dict[str, Any]:
    return {"blocks": blocks}


def test_curie_anchoring_tombstone_skipped():
    """(a) tombstone → not audited, no issue from it (score==1.0, pass)."""
    blocks = [_str_block(content="", escalation_marker=_MARKER)]
    result = BlockCurieAnchoringValidator().validate(_curie_inputs(blocks))
    assert result.passed is True
    assert result.score == 1.0  # audited==0 → vacuous pass
    assert result.issues == []


def test_curie_anchoring_salvaged_audited_and_fails():
    """(b) salvaged (marker + curie-less content) → audited → fails."""
    blocks = [_str_block(content=_BAD_HTML, escalation_marker=_MARKER)]
    result = BlockCurieAnchoringValidator().validate(_curie_inputs(blocks))
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_MISSING_CURIES" in codes


def test_curie_anchoring_no_marker_audited():
    """(c) no marker → audited as always (clean content passes)."""
    blocks = [_str_block(content=_GOOD_HTML, escalation_marker=None)]
    result = BlockCurieAnchoringValidator().validate(_curie_inputs(blocks))
    assert result.passed is True


def test_content_type_tombstone_skipped():
    blocks = [_str_block(content="", escalation_marker=_MARKER)]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []


def test_content_type_salvaged_audited_and_fails():
    # _BAD_HTML carries no data-cf-content-type attribute → missing.
    blocks = [_str_block(content=_BAD_HTML, escalation_marker=_MARKER)]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_MISSING_CONTENT_TYPE" in codes


def test_content_type_no_marker_audited():
    blocks = [_str_block(content=_GOOD_HTML, escalation_marker=None)]
    result = BlockContentTypeValidator().validate({"blocks": blocks})
    assert result.passed is True


def test_page_objectives_tombstone_skipped():
    blocks = [_str_block(content="", escalation_marker=_MARKER)]
    result = BlockPageObjectivesValidator().validate(
        {"blocks": blocks, "valid_objective_ids": ["TO-01"]}
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []


def test_page_objectives_salvaged_audited_and_fails():
    # Salvaged block references an unknown objective id → UNKNOWN_OBJECTIVE.
    blocks = [
        _str_block(
            content=_GOOD_HTML,
            escalation_marker=_MARKER,
            objective_ids=("TO-99",),
        )
    ]
    result = BlockPageObjectivesValidator().validate(
        {"blocks": blocks, "valid_objective_ids": ["TO-01"]}
    )
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_UNKNOWN_OBJECTIVE" in codes


def test_page_objectives_no_marker_audited():
    blocks = [
        _str_block(
            content=_GOOD_HTML,
            escalation_marker=None,
            objective_ids=("TO-01",),
        )
    ]
    result = BlockPageObjectivesValidator().validate(
        {"blocks": blocks, "valid_objective_ids": ["TO-01"]}
    )
    assert result.passed is True


def test_source_refs_tombstone_skipped():
    blocks = [
        _str_block(
            content="",
            escalation_marker=_MARKER,
            source_ids=("not-a-valid-source-id",),
        )
    ]
    result = BlockSourceRefValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []


def test_source_refs_salvaged_audited_and_fails():
    # Salvaged block declares a malformed sourceId → INVALID_SOURCE_ID_SHAPE.
    blocks = [
        _str_block(
            content=_GOOD_HTML,
            escalation_marker=_MARKER,
            source_ids=("not-a-valid-source-id",),
        )
    ]
    result = BlockSourceRefValidator().validate({"blocks": blocks})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE" in codes


def test_source_refs_no_marker_audited():
    # Clean canonical sourceId, no manifest → structural shape check passes.
    blocks = [
        _str_block(
            content=_GOOD_HTML,
            escalation_marker=None,
            source_ids=("dart:slug#blk1",),
        )
    ]
    result = BlockSourceRefValidator().validate({"blocks": blocks})
    assert result.passed is True
