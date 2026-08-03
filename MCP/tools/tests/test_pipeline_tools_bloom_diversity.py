"""Bloom-diversity fix regression — outline-tier per-template target Bloom.

The 7B outline tier produces Bloom-skewed blocks (an observed 70
``understand`` / 19 ``apply`` / 1 ``remember`` distribution) because
``bloom_level`` is left to the model, which defaults to the floor. The fix
makes Bloom a DETERMINISTIC, per-template declaration carried on the page
block plan AND enforces it as a FLOOR after the LLM emits.

This module asserts the two load-bearing contracts:

1. ``MCP.tools.pipeline_tools._PAGE_BLOCK_PLAN`` is a sequence of
   ``(block_type, target_bloom)`` pairs whose block_types are members of
   the canonical ``Courseforge.scripts.blocks.BLOCK_TYPES`` frozenset and
   whose target_blooms are members of ``lib.ontology.bloom.BLOOM_LEVELS``.

2. The outline provider's FLOOR logic
   (``Courseforge.generators.outline._outline_provider._max_bloom_level``) LIFTS a
   below-target emitted bloom up to the target but PRESERVES a higher
   objective-declared bloom (never lowers below the existing ≥-objective
   rule). The ordering is sourced from ``lib.ontology.bloom.BLOOM_LEVELS``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import BLOCK_TYPES  # noqa: E402
from lib.ontology.bloom import BLOOM_LEVELS  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _PAGE_BLOCK_PLAN,
    _PAGE_TYPE_BLOCK_PLAN,
)
from Courseforge.generators.outline._outline_provider import (  # noqa: E402
    _max_bloom_level,
)


# ---------------------------------------------------------------------- #
# (1) Plan-shape contract
# ---------------------------------------------------------------------- #


def test_page_block_plan_entries_are_blocktype_bloom_pairs() -> None:
    """Every plan entry is a ``(block_type, target_bloom)`` pair with valid
    members."""
    assert len(_PAGE_BLOCK_PLAN) >= 1
    for entry in _PAGE_BLOCK_PLAN:
        assert isinstance(entry, tuple), f"entry is not a tuple: {entry!r}"
        assert len(entry) == 2, f"entry is not a 2-tuple pair: {entry!r}"
        block_type, target_bloom = entry
        assert block_type in BLOCK_TYPES, (
            f"plan block_type {block_type!r} not in canonical BLOCK_TYPES"
        )
        assert target_bloom in BLOOM_LEVELS, (
            f"plan target_bloom {target_bloom!r} not a canonical Bloom level"
        )


def test_page_block_plan_widened_beyond_objective_prose() -> None:
    """The plan widens past the original objective/explanation/example trio.

    Guards against a silent regression to the pre-fix narrow plan (which is
    what produced the Bloom skew).

    Re-pinned 2026-07-19 (drift from 6bc16dc76): ``_PAGE_BLOCK_PLAN`` is now
    the back-compat alias for the ``content`` entry of the page-type-keyed
    ``_PAGE_TYPE_BLOCK_PLAN``. The objective STUB (which carries the page LO
    statement) migrated to the ``overview`` page type in that split, so the
    ``content`` plan now LEADS with ``concept`` — the objective-first
    invariant is asserted against the ``overview`` plan below, and the
    content plan is asserted to lead with ``concept``. The widening guard
    (at least one higher-order Bloom target) is preserved.
    """
    block_types = [bt for bt, _ in _PAGE_BLOCK_PLAN]
    # Content pages open with the concept block post-split (was "objective"
    # pre-6bc16dc76, before the objective stub moved to the overview page).
    assert block_types[0] == "concept", (
        f"content plan no longer leads with 'concept' (got {block_types[0]!r}); "
        "re-pin this assertion to the current _PAGE_TYPE_BLOCK_PLAN['content'] "
        "shape if the split changed again"
    )
    # The objective stub now lives on the overview page type — assert it
    # remains first there so the LO-statement-carrying stub isn't lost.
    overview_types = [bt for bt, _ in _PAGE_TYPE_BLOCK_PLAN["overview"]]
    assert overview_types[0] == "objective"
    # Bloom diversity requires at least one above-``apply`` target so the
    # plan can't collapse back to a two-level distribution.
    targets = {tb for _, tb in _PAGE_BLOCK_PLAN}
    high_rank = {"analyze", "evaluate", "create"}
    assert targets & high_rank, (
        "plan carries no higher-order Bloom target — the diversity fix is "
        "defeated"
    )


# ---------------------------------------------------------------------- #
# (2) Floor logic
# ---------------------------------------------------------------------- #


def test_floor_lifts_below_target_emitted_bloom() -> None:
    """A lazy 'understand' emit is LIFTED to the template target."""
    # emitted=understand, target=analyze, no objective floor -> analyze
    assert _max_bloom_level("understand", "analyze", None) == "analyze"
    # emitted=understand, target=apply -> apply
    assert _max_bloom_level("understand", "apply", None) == "apply"


def test_floor_preserves_higher_objective_declared_bloom() -> None:
    """A higher objective-declared bloom is NEVER lowered to the target."""
    # emitted=understand, target=apply, objective floor=create -> create
    assert _max_bloom_level("understand", "apply", "create") == "create"
    # objective floor outranks an already-higher-than-target emit too
    assert _max_bloom_level("apply", "apply", "evaluate") == "evaluate"


def test_floor_never_lowers_a_higher_emit() -> None:
    """An emit already above the target is preserved (target is a FLOOR)."""
    # emitted=create, target=understand -> create (target never lowers)
    assert _max_bloom_level("create", "understand", None) == "create"


def test_floor_ignores_invalid_inputs_and_fails_closed() -> None:
    """Unknown / None / empty inputs are ignored; all-invalid returns None."""
    assert _max_bloom_level(None, None, None) is None
    assert _max_bloom_level("", "bogus", None) is None
    # A valid value among garbage still resolves.
    assert _max_bloom_level("bogus", "analyze", "") == "analyze"


def test_floor_uses_canonical_bloom_ordering() -> None:
    """The lift respects the canonical BLOOM_LEVELS rank ordering."""
    # Walk adjacent canonical levels: the higher one always wins.
    for lower, higher in zip(BLOOM_LEVELS, BLOOM_LEVELS[1:]):
        assert _max_bloom_level(lower, higher, None) == higher
        assert _max_bloom_level(higher, lower, None) == higher


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-x", "-q"]))
