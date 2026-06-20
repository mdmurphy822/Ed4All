"""Unit tests for the extracted ``_build_outline_curie_minter`` factory.

The factory (loads ``domain_concept_vocabulary.json`` + compiles concept
seeds ONCE) backs BOTH the in-loop ``pre_validate_hook`` threaded into
``route_with_self_consistency`` AND the post-loop ``_mint_outline_curies``
list pass. These tests pin its single-block minting contract without any
LLM / pipeline run.

Covered:
- ``minter is None`` (complete no-op) when no vocabulary exists — the
  RDF / legacy-corpus backward-compat gate.
- ``mint_block`` stamps the per-course minted CURIE onto a curie-less
  prose block whose ``key_claims`` discusses a vocabulary concept, and
  the resulting block then ANCHORS through the real
  ``BlockCurieAnchoringValidator`` (the end-to-end point of the fix).
- ``mint_block`` is idempotent: a block already carrying the minted CURIE
  is left untouched (so the post-loop pass is a safe no-op after the
  in-loop hook minted the same block).
- ``mint_block`` fabricates nothing: an ungrounded block with no
  surface-form match mints no CURIE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from MCP.tools.pipeline_tools import _build_outline_curie_minter  # noqa: E402
from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockCurieAnchoringValidator,
)
from blocks import Block  # noqa: E402


_VOCAB = {
    "course_id": "introbio101",
    "concepts": [
        {"canonical": "slope", "aliases": ["gradient", "steepness"]},
        {"canonical": "intercept", "aliases": ["y-intercept"]},
    ],
}


def _write_vocab(tmp_path: Path) -> Path:
    p = tmp_path / "domain_concept_vocabulary.json"
    p.write_text(json.dumps(_VOCAB), encoding="utf-8")
    return p


def _kwargs_with_vocab(vocab_path: Path) -> dict:
    # ``_vocabulary_threaded_path`` reads kwargs["phase_outputs"]
    # ["concept_extraction"]["domain_concept_vocabulary_path"].
    return {
        "phase_outputs": {
            "concept_extraction": {
                "domain_concept_vocabulary_path": str(vocab_path),
            }
        }
    }


def _prose_block(curies, key_claims) -> Block:
    return Block(
        block_id="page_01#concept_slope_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "curies": list(curies),
            "key_claims": list(key_claims),
            "content_type": "definition",
        },
        objective_ids=("CO-01",),
    )


def test_minter_is_none_without_vocabulary():
    """No vocabulary file → factory returns None (complete no-op)."""
    minter = _build_outline_curie_minter(
        course_code="introbio101", kwargs={},
    )
    assert minter is None


def test_mint_block_stamps_and_then_anchors(tmp_path):
    """A curie-less prose block gets the minted CURIE, then ANCHORS.

    End-to-end point of the fix: the minted block passes the real
    ``BlockCurieAnchoringValidator`` (with the minter's ``minted_map``
    threaded) on the FIRST validation — no regenerate.
    """
    vocab_path = _write_vocab(tmp_path)
    minter = _build_outline_curie_minter(
        course_code="introbio101",
        kwargs=_kwargs_with_vocab(vocab_path),
    )
    assert minter is not None

    blk = _prose_block(
        curies=[],
        key_claims=["The gradient of a line measures how steeply it rises."],
    )
    new_block, meta = minter.mint_block(blk, None)
    assert new_block is not None
    assert meta is not None
    minted = new_block.content["curies"]
    assert minted, "expected a minted CURIE"
    # The minted CURIE keys into the minter's minted_map.
    assert any(c in minter.minted_map for c in minted)

    # And the minted block now ANCHORS through the real validator.
    result = BlockCurieAnchoringValidator().validate({
        "blocks": [new_block],
        "minted_curie_map": minter.minted_map,
    })
    assert result.passed is True
    assert result.action is None


def test_mint_block_idempotent(tmp_path):
    """A block already carrying the minted CURIE is left untouched."""
    vocab_path = _write_vocab(tmp_path)
    minter = _build_outline_curie_minter(
        course_code="introbio101",
        kwargs=_kwargs_with_vocab(vocab_path),
    )
    assert minter is not None

    blk = _prose_block(
        curies=[],
        key_claims=["The gradient of a line measures how steeply it rises."],
    )
    minted_block, _ = minter.mint_block(blk, None)
    assert minted_block is not None
    # Second pass over the already-minted block is a no-op.
    again, again_meta = minter.mint_block(minted_block, None)
    assert again is None
    assert again_meta is None


def test_mint_block_no_fabrication(tmp_path):
    """An ungrounded block with no surface-form match mints nothing."""
    vocab_path = _write_vocab(tmp_path)
    minter = _build_outline_curie_minter(
        course_code="introbio101",
        kwargs=_kwargs_with_vocab(vocab_path),
    )
    assert minter is not None

    blk = _prose_block(
        curies=[],
        key_claims=["This sentence is about nothing in the vocabulary."],
    )
    new_block, meta = minter.mint_block(blk, None)
    assert new_block is None
    assert meta is None
