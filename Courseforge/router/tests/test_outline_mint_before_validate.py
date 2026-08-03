"""Regression: outline-tier CURIE minting runs BEFORE the in-loop validator.

Root cause (diagnosed pre-fix): the two-pass ``content_generation_outline``
tier dispatched each block through
``CourseforgeRouter.route_with_self_consistency`` WITHOUT (a) stamping the
per-course minted CURIEs onto the freshly-routed candidate before the
in-loop ``BlockCurieAnchoringValidator`` runs, or (b) threading the
``minted_curie_map`` / ``source_chunks_by_block_id`` into the validator
chain inputs. On a PROSE corpus every outline block lands with
``content["curies"] == []``; the in-loop curie-anchoring gate fired
``OUTLINE_BLOCK_MISSING_CURIES`` → ``regenerate`` and the block re-rolled
IDENTICALLY up to the regen budget (~3× the calls). Minting ran only
AFTER the route loop — too late to prevent the churn.

These tests pin the fix:

1. A prose outline block with empty ``curies`` + a resolvable minted
   concept, routed through ``route_with_self_consistency`` with the
   per-block ``pre_validate_hook`` + ``minted_curie_map`` threaded, gets
   its minted CURIE stamped BEFORE the in-loop validator so
   ``outline_curie_anchoring`` PASSES on the FIRST attempt — the route
   makes 1 dispatch, not 3.
2. The threaded ``minted_curie_map`` + ``source_chunks_by_block_id`` reach
   the validator's inputs dict (so the anchoring runs the same way the
   standalone ``inter_tier_validation`` phase already does).
3. Without the hook (RDF / legacy behaviour) the same prose block re-rolls
   the full budget — proving the hook is what eliminates the churn.

No LLM, no pipeline run — a stub outline provider returns the canned
prose block and the REAL ``BlockCurieAnchoringValidator`` runs.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockCurieAnchoringValidator,
)
from blocks import Block  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# A minted (prose-corpus) CURIE whose vocabulary surface form "gradient"
# appears in the block's prose — but the literal CURIE token never does.
_MINTED_MAP = {
    "tstcourse101:slope": {
        "canonical": "slope",
        "surface_forms": ["slope", "gradient"],
    },
}

_PROSE_BLOCK_ID = "page_01#concept_slope_0"


def _prose_block_no_curies() -> Block:
    """A prose outline block the LLM emitted with EMPTY ``curies``.

    Its ``key_claims`` discusses the "gradient" — a surface form of the
    minted ``tstcourse101:slope`` concept — so a minted CURIE, once
    stamped on, anchors via surface form. Until stamped, ``curies`` is
    empty → the validator's ``OUTLINE_BLOCK_MISSING_CURIES`` branch.
    """
    return Block(
        block_id=_PROSE_BLOCK_ID,
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "curies": [],  # prose corpus → LLM emits no CURIEs
            "key_claims": [
                "The gradient of a line measures how steeply it rises."
            ],
            "content_type": "definition",
        },
        objective_ids=("CO-01",),
    )


class _ConstantOutlineProvider:
    """Stub OutlineProvider: every dispatch returns a fresh prose block.

    Mirrors the real outline tier's contract that ``content["curies"]``
    is REGENERATED from scratch on each dispatch (so a re-roll never
    "remembers" a prior mint). The stub records every call so the test
    can count dispatches.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any = None,
        objectives: Any = None,
        **kwargs: Any,
    ) -> Block:
        self.calls.append({"block_id": block.block_id})
        # Always return a fresh curie-less prose block — emulates the
        # prose-corpus failure mode the fix targets.
        return _prose_block_no_curies()


class _RecordingCurieValidator:
    """Wraps the REAL BlockCurieAnchoringValidator, recording inputs.

    Lets the test assert (a) the real anchoring verdict AND (b) that the
    threaded ``minted_curie_map`` / ``source_chunks_by_block_id`` reached
    the validator's inputs dict.
    """

    name = "outline_curie_anchoring"
    version = "1.0.0"

    def __init__(self) -> None:
        self._inner = BlockCurieAnchoringValidator()
        self.seen_inputs: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> Any:
        self.seen_inputs.append(inputs)
        return self._inner.validate(inputs)


def _mint_hook_factory(source_chunks: List[Any]):
    """Per-block ``pre_validate_hook`` that stamps the minted CURIE.

    Mirrors the real ``_run_content_generation_outline`` hook: it appends
    the minted ``tstcourse101:slope`` CURIE onto a candidate whose
    ``content["curies"]`` is empty (never overwrites an existing real
    minted CURIE — idempotent). ``source_chunks`` is unused here (the
    block's key_claims already carry the surface form) but threaded to
    mirror the real signature.
    """

    def _hook(candidate: Block) -> Block:
        content = candidate.content
        if not isinstance(content, dict):
            return candidate
        existing = [c for c in (content.get("curies") or []) if c]
        if any(c in _MINTED_MAP for c in existing):
            return candidate  # already minted — idempotent no-op
        new_content = dict(content)
        new_content["curies"] = existing + ["tstcourse101:slope"]
        return dataclasses.replace(candidate, content=new_content)

    return _hook


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_mint_before_validate_passes_on_first_attempt(monkeypatch):
    """Minting before the in-loop validator → 1 dispatch, not 3.

    The hook stamps the minted CURIE onto the freshly-routed prose
    candidate; the in-loop curie-anchoring gate (with the threaded
    ``minted_curie_map``) anchors it via surface form on attempt #1, so
    the self-consistency loop returns immediately. No regenerate churn.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _prose_block_no_curies()
    provider = _ConstantOutlineProvider()
    validator = _RecordingCurieValidator()
    router = CourseforgeRouter(outline_provider=provider, n_candidates=3)

    source_chunks: List[Any] = []
    out = router.route_with_self_consistency(
        blk,
        validators=[validator],
        source_chunks=source_chunks,
        minted_curie_map=_MINTED_MAP,
        pre_validate_hook=_mint_hook_factory(source_chunks),
    )

    # 1 dispatch (passed on the first attempt) — NOT the 3-candidate budget.
    assert len(provider.calls) == 1, (
        f"expected 1 outline dispatch, got {len(provider.calls)} "
        "(re-roll churn not eliminated)"
    )
    # The returned block carries the minted CURIE (stamped pre-validate).
    assert "tstcourse101:slope" in out.content["curies"]
    # And it won (self_consistency_winner Touch appended).
    assert any(
        t.purpose == "self_consistency_winner" for t in out.touched_by
    )
    # The validator saw the threaded minted map in its inputs.
    assert validator.seen_inputs, "validator never ran"
    assert validator.seen_inputs[0].get("minted_curie_map") == _MINTED_MAP


def test_source_chunks_threaded_into_validator_inputs(monkeypatch):
    """``source_chunks`` → ``source_chunks_by_block_id`` keyed by block_id.

    When ``source_chunks`` is non-empty the router builds a single-entry
    ``{block_id: source_chunks}`` map and threads it into the validator
    inputs (so the anchoring runs over grounded provenance, the same way
    the standalone ``inter_tier_validation`` phase does).
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _prose_block_no_curies()
    provider = _ConstantOutlineProvider()
    validator = _RecordingCurieValidator()
    router = CourseforgeRouter(outline_provider=provider, n_candidates=3)

    # Grounded source chunk carrying the surface form in its text.
    source_chunks = [
        {"id": "chunk-1", "text": "The gradient describes the slope of a line."}
    ]
    router.route_with_self_consistency(
        blk,
        validators=[validator],
        source_chunks=source_chunks,
        minted_curie_map=_MINTED_MAP,
        pre_validate_hook=_mint_hook_factory(source_chunks),
    )

    assert validator.seen_inputs, "validator never ran"
    sc_map = validator.seen_inputs[0].get("source_chunks_by_block_id")
    assert sc_map == {_PROSE_BLOCK_ID: source_chunks}


def test_without_hook_prose_block_rerolls_full_budget(monkeypatch):
    """Control: NO hook → the prose block re-rolls the full budget.

    Proves the ``pre_validate_hook`` (mint-before-validate) is the lever
    that eliminates the re-roll churn — without it the empty-``curies``
    prose block fails ``OUTLINE_BLOCK_MISSING_CURIES`` every attempt and
    burns all 3 dispatches.
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    blk = _prose_block_no_curies()
    provider = _ConstantOutlineProvider()
    validator = _RecordingCurieValidator()
    router = CourseforgeRouter(outline_provider=provider, n_candidates=3)

    out = router.route_with_self_consistency(
        blk,
        validators=[validator],
        source_chunks=[],
        minted_curie_map=_MINTED_MAP,
        pre_validate_hook=None,  # the old path
    )

    # All 3 candidates burned — the regression the fix removes.
    assert len(provider.calls) == 3
    # The surviving block never got a CURIE.
    assert out.content["curies"] == []
