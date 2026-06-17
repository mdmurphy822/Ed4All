"""FIX 2 regression — outline-tier hallucinated objective-ref repair.

Pins ``MCP/tools/pipeline_tools.py::_repair_outline_objective_refs`` —
the outline-handler pass that drops objective refs the 7B outline tier
emits that are NOT in the canonical synthesized-objective set (e.g.
``TO-08`` when only four terminal objectives exist, or a stray CURIE that
leaked into ``content["objective_refs"]``). The repair never invents a
ref; valid (canonical) week refs are preserved verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _repair_outline_objective_refs,
)

CANON = {"TO-01", "TO-02", "TO-03", "TO-04", "CO-01", "CO-23", "CO-24"}


def _block(*, block_id="p#objective_x_0", objective_ids=("TO-01",),
           objective_refs=None) -> Block:
    content = {
        "curies": ["crs:foo"],
        "key_claims": ["something"],
        "content_type": "definition",
    }
    if objective_refs is not None:
        content["objective_refs"] = list(objective_refs)
    return Block(
        block_id=block_id,
        block_type="objective",
        page_id="p",
        sequence=0,
        content=content,
        objective_ids=tuple(objective_ids),
    )


def test_hallucinated_terminal_ref_dropped_from_content():
    """A hallucinated TO-08 (only TO-01..TO-04 exist) is dropped from
    content['objective_refs']; valid refs are kept."""
    blocks = [_block(objective_refs=["CO-23", "TO-08", "CO-24"])]
    _repair_outline_objective_refs(
        outline_blocks=blocks, canonical_ids=CANON, capture=None,
    )
    refs = blocks[0].content["objective_refs"]
    assert "TO-08" not in refs
    assert refs == ["CO-23", "CO-24"]


def test_leaked_curie_in_objective_refs_dropped():
    """A stray CURIE (cf:CO-36) that leaked into objective_refs is not in
    the canonical set and is dropped."""
    blocks = [_block(objective_refs=["cf:CO-36", "TO-02", "CO-23"])]
    _repair_outline_objective_refs(
        outline_blocks=blocks, canonical_ids=CANON, capture=None,
    )
    refs = blocks[0].content["objective_refs"]
    assert "cf:CO-36" not in refs
    assert refs == ["TO-02", "CO-23"]


def test_valid_week_refs_preserved_verbatim():
    """A block whose refs are all canonical is left byte-identical
    (no spurious mutation)."""
    before = _block(objective_ids=("TO-01",), objective_refs=["TO-01", "CO-01"])
    blocks = [before]
    _repair_outline_objective_refs(
        outline_blocks=blocks, canonical_ids=CANON, capture=None,
    )
    # Untouched object — nothing to repair.
    assert blocks[0] is before
    assert blocks[0].content["objective_refs"] == ["TO-01", "CO-01"]
    assert blocks[0].objective_ids == ("TO-01",)


def test_hallucinated_ref_dropped_from_structural_field():
    """A hallucinated id in the structural block.objective_ids tuple is
    dropped; the surviving canonical week ref is kept."""
    blocks = [_block(objective_ids=("TO-02", "TO-99"), objective_refs=None)]
    _repair_outline_objective_refs(
        outline_blocks=blocks, canonical_ids=CANON, capture=None,
    )
    assert blocks[0].objective_ids == ("TO-02",)


def test_no_invention_when_all_refs_hallucinated():
    """The repair NEVER invents a ref: a block whose refs are ALL
    hallucinated ends with empty refs (it then fails the missing-ref gate
    downstream — honest fail-closed, not a fabricated pass)."""
    blocks = [_block(objective_ids=("TO-77",), objective_refs=["TO-88"])]
    _repair_outline_objective_refs(
        outline_blocks=blocks, canonical_ids=CANON, capture=None,
    )
    assert blocks[0].objective_ids == ()
    assert blocks[0].content["objective_refs"] == []


def test_empty_canonical_set_is_noop():
    """No canonical universe → no-op (preserves the validator's
    skip-when-no-universe contract)."""
    before = _block(objective_refs=["TO-08", "ANYTHING"])
    blocks = [before]
    _repair_outline_objective_refs(
        outline_blocks=blocks, canonical_ids=set(), capture=None,
    )
    assert blocks[0] is before
    assert blocks[0].content["objective_refs"] == ["TO-08", "ANYTHING"]
