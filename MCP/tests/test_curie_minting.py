"""Regression tests for the v0.3.0 dynamic CURIE-minting wiring.

Exercises ``MCP/tools/pipeline_tools.py::_mint_outline_curies`` — the
outline-handler step that stamps per-course minted CURIEs onto
curie-less outline blocks when a Stage-3 ``domain_concept_vocabulary.json``
is present.

Contract assertions:

1. **With a vocabulary present**, an outline block with empty ``curies``
   whose ``key_claims`` text mentions a vocab concept gets the minted
   CURIE stamped into ``content["curies"]``.
2. **Without a vocabulary**, blocks are byte-identical to the input
   (the backward-compat no-op contract — RDF / legacy corpora).
3. **A block that already carries CURIEs** is never overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Block lives at Courseforge/scripts/blocks.py.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from MCP.tools.pipeline_tools import _mint_outline_curies  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _vocab_file(tmp_path: Path) -> Path:
    """Write a minimal domain_concept_vocabulary.json under a fake LibV2
    course tree and return its path."""
    course_dir = tmp_path / "courses" / "openstax-alg-9" / "concept_graph"
    course_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "schema_version": "v1",
        "course_id": "OPENSTAX_ALG_9",
        "course_slug": "openstax-alg-9",
        "concept_count": 2,
        "concepts": [
            {"canonical": "slope", "aliases": ["gradient"]},
            {"canonical": "y-intercept", "aliases": []},
        ],
    }
    path = course_dir / "domain_concept_vocabulary.json"
    path.write_text(json.dumps(vocab), encoding="utf-8")
    return path


def _outline_block(*, block_id: str, curies, key_claims) -> Block:
    return Block(
        block_id=block_id,
        block_type="objective",
        page_id="week_01_content_01",
        sequence=0,
        content={
            "curies": list(curies),
            "key_claims": list(key_claims),
            "content_type": "definition",
        },
        objective_ids=("TO-01",),
    )


# ---------------------------------------------------------------------------
# 1. With a vocabulary present — minting fires.
# ---------------------------------------------------------------------------


def test_minting_stamps_curies_when_vocabulary_present(tmp_path):
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_slope_0",
        curies=[],  # empty — the minting trigger
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="OPENSTAX_ALG_9",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    # The block was replaced with a minted-CURIE-carrying copy.
    minted = blocks[0].content["curies"]
    assert minted, "expected minted CURIEs to be stamped"
    assert "openstaxalg9:slope" in minted


def test_minting_matches_via_alias_surface_form(tmp_path):
    """The block text mentions the alias 'gradient' (not 'slope') — the
    concept-seed match still fires and the minted CURIE lands."""
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_a_0",
        curies=[],
        key_claims=["A line's gradient describes how steep it is."],
    )
    blocks = [block]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="OPENSTAX_ALG_9",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    assert "openstaxalg9:slope" in blocks[0].content["curies"]


# ---------------------------------------------------------------------------
# 2. Without a vocabulary — no-op (backward-compat contract).
# ---------------------------------------------------------------------------


def test_no_vocabulary_leaves_blocks_untouched(tmp_path):
    # tmp_path has NO domain_concept_vocabulary.json.
    block = _outline_block(
        block_id="week_01_content_01#objective_a_0",
        curies=[],
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    before = blocks[0]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="OPENSTAX_ALG_9",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    # Same object, untouched — byte-identical to the input.
    assert blocks[0] is before
    assert blocks[0].content["curies"] == []


# ---------------------------------------------------------------------------
# 3. Existing CURIEs are never overwritten.
# ---------------------------------------------------------------------------


def test_existing_curies_not_overwritten(tmp_path):
    _vocab_file(tmp_path)
    block = _outline_block(
        block_id="week_01_content_01#objective_a_0",
        curies=["sh:NodeShape"],  # already populated
        key_claims=["The slope of a line measures its steepness."],
    )
    blocks = [block]
    before = blocks[0]
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="OPENSTAX_ALG_9",
        kwargs={"libv2_root": str(tmp_path)},
        capture=None,
    )
    # Untouched — real CURIEs win.
    assert blocks[0] is before
    assert blocks[0].content["curies"] == ["sh:NodeShape"]
