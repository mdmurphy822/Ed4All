"""Regression tests for the rewrite_content_type backstop fill + repair fix.

After the IB5 content-type work a real ``post_rewrite_validation`` run
was down to 5/66 blocks failing — ALL on the critical ``rewrite_content_type``
gate (``BlockContentTypeValidator``), on STANDARD block types, with two roots:

1. ``lib/ontology/content_types.py::BLOCK_TYPE_CONTENT_TYPE`` (consumed via
   ``MCP/tools/pipeline_tools.py::_REWRITE_BLOCK_TYPE_CONTENT_TYPE``) was
   INCOMPLETE — ``scenario`` / ``problem`` / ``concept`` / ``example`` / … had
   no backstop default, so a rewrite-tier block with NO ``data-cf-content-type``
   fired ``OUTLINE_BLOCK_MISSING_CONTENT_TYPE``.
2. The backstop only FILLED a missing attribute; it never REPAIRED an INVALID
   value, so a rewrite-LLM-authored ``data-cf-content-type="expression"`` (a
   domain term in the content-type slot — not a ChunkType member) was left to
   fail with ``OUTLINE_BLOCK_INVALID_CONTENT_TYPE``.

This suite pins:
* the drift-guard: every non-scaffolding ``BLOCK_TYPES`` member has a VALID
  backstop entry (mirrors the CLI enum drift-guard).
* fill-missing for scenario / problem -> PASS through the real gate.
* repair-invalid for concept's ``expression`` -> PASS through the real gate.
* idempotence + valid-value preservation + scaffolding no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    _CONTENT_TYPE_SCAFFOLDING_TYPES,
)
from Courseforge.scripts.blocks import BLOCK_TYPES, Block  # noqa: E402
from lib.ontology.content_types import (  # noqa: E402
    BLOCK_TYPE_CONTENT_TYPE,
    get_valid_content_types,
    is_valid_content_type,
    resolve_block_content_type,
)
from MCP.tools import pipeline_tools as _pt  # noqa: E402


def _mk(block_type: str, html: str) -> Block:
    return Block(
        block_id=f"week_01_application#{block_type}_x_1",
        block_type=block_type,
        page_id="week_01_application",
        sequence=1,
        content=html,
    )


def _gate(block: Block):
    return BlockContentTypeValidator().validate({"blocks": [block]})


# ---------------------------------------------------------------------------
# Drift-guard: completeness + validity of the backstop map.
# ---------------------------------------------------------------------------

def test_backstop_map_covers_every_non_scaffolding_block_type():
    """Every BLOCK_TYPES member the gate audits has a backstop default.

    The content_type gate skips ``objective`` / ``chrome`` / ``recap``; every
    OTHER block type must have a key so the backstop can always fill / repair.
    """
    auditable = set(BLOCK_TYPES) - set(_CONTENT_TYPE_SCAFFOLDING_TYPES)
    missing = sorted(auditable - set(BLOCK_TYPE_CONTENT_TYPE))
    assert not missing, f"block types with no content_type backstop: {missing}"


def test_backstop_map_values_are_all_valid_chunk_types():
    valid = get_valid_content_types()
    bad = {k: v for k, v in BLOCK_TYPE_CONTENT_TYPE.items() if v not in valid}
    assert not bad, f"backstop values not in ChunkType enum: {bad}"


def test_scaffolding_types_have_no_backstop_default():
    """Gate-skipped scaffolding types resolve to None (never stamped)."""
    for t in _CONTENT_TYPE_SCAFFOLDING_TYPES:
        assert resolve_block_content_type(t) is None
        assert t not in BLOCK_TYPE_CONTENT_TYPE


# ---------------------------------------------------------------------------
# Fill-missing: scenario + problem with NO content_type -> injected -> PASS.
# ---------------------------------------------------------------------------

def test_scenario_missing_content_type_filled_and_passes():
    raw = '<section class="scenario-card" data-cf-block-id="x">A car travels.</section>'
    # Pre-fix: gate fails MISSING.
    assert not _gate(_mk("scenario", raw)).passed
    fixed, filled, repaired = _pt._apply_content_type_backstop(raw, "scenario")
    assert filled and not repaired
    assert 'data-cf-content-type="scenario"' in fixed
    assert _gate(_mk("scenario", fixed)).passed


def test_problem_missing_content_type_filled_and_passes():
    raw = '<section class="problem" data-cf-block-id="x">Solve 2x+1=5.</section>'
    assert not _gate(_mk("problem", raw)).passed
    fixed, filled, repaired = _pt._apply_content_type_backstop(raw, "problem")
    assert filled and not repaired
    assert 'data-cf-content-type="problem"' in fixed
    assert _gate(_mk("problem", fixed)).passed


# ---------------------------------------------------------------------------
# Repair-invalid: concept content_type="expression" -> repaired -> PASS.
# ---------------------------------------------------------------------------

def test_concept_invalid_content_type_repaired_and_passes():
    raw = '<section data-cf-content-type="expression" data-cf-block-id="x">body</section>'
    # Pre-fix: gate fails INVALID.
    bad = _gate(_mk("concept", raw))
    assert not bad.passed
    assert "OUTLINE_BLOCK_INVALID_CONTENT_TYPE" in [i.code for i in bad.issues]

    fixed, filled, repaired = _pt._apply_content_type_backstop(raw, "concept")
    assert repaired and not filled
    assert 'data-cf-content-type="explanation"' in fixed
    assert "expression" not in fixed
    assert _gate(_mk("concept", fixed)).passed


def test_repair_is_idempotent():
    raw = '<section data-cf-content-type="expression">b</section>'
    once, _, _ = _pt._apply_content_type_backstop(raw, "concept")
    twice, filled, repaired = _pt._apply_content_type_backstop(once, "concept")
    assert not filled and not repaired
    assert twice == once


def test_valid_value_is_preserved_verbatim():
    raw = '<section data-cf-content-type="example">b</section>'
    out, filled, repaired = _pt._apply_content_type_backstop(raw, "example")
    assert not filled and not repaired
    assert out == raw


def test_scaffolding_block_is_a_noop():
    raw = '<section data-cf-content-type="expression">b</section>'
    out, filled, repaired = _pt._apply_content_type_backstop(raw, "objective")
    assert not filled and not repaired
    assert out == raw  # gate skips objective; backstop never touches it


# ---------------------------------------------------------------------------
# SoT helper sanity.
# ---------------------------------------------------------------------------

def test_is_valid_content_type():
    assert is_valid_content_type("explanation")
    assert not is_valid_content_type("expression")
    assert not is_valid_content_type("")
    assert not is_valid_content_type(None)
