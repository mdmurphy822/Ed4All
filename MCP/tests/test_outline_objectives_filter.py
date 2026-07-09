"""outline-overflow-fix-2026-07 regression tests for the per-block
objectives filter.

The outline tier previously passed the ENTIRE course objectives list to
EVERY per-block LLM call, bloating the prompt into the served window so a
small-ctx local server silently head-truncated it. The filter passes each
outline call only the objectives relevant to its block — its own TO(s) +
rolled-up COs (and a directly-cited CO's parent TO + siblings) — falling back
to the full list on an empty filter (never an empty objectives section).

Covers the module-level helpers in ``MCP/tools/pipeline_tools.py``:
``_build_objective_rollup`` / ``_relevant_objective_ids`` /
``_filter_objectives_payload_for_block``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_objective_rollup,
    _filter_objectives_payload_for_block,
    _relevant_objective_ids,
)


# ---------------------------------------------------------------------------
# Fixtures: 2 TOs, 4 COs (2 per TO). TO-01 uses child_co_ids; TO-02 relies
# purely on the COs' parent back-pointers — exercises BOTH roll-up paths.
# ---------------------------------------------------------------------------


def _tos() -> List[Dict[str, Any]]:
    return [
        {"id": "TO-01", "statement": "Terminal one", "child_co_ids": ["CO-01", "CO-02"]},
        {"id": "TO-02", "statement": "Terminal two"},
    ]


def _cos() -> List[Dict[str, Any]]:
    return [
        {"id": "CO-01", "statement": "Chapter one", "parent_to": "TO-01"},
        {"id": "CO-02", "statement": "Chapter two", "parent_to": "TO-01"},
        {"id": "CO-03", "statement": "Chapter three", "parent_to": "TO-02"},
        {"id": "CO-04", "statement": "Chapter four", "parent_terminal": "TO-02"},
    ]


def _payload() -> List[Dict[str, Any]]:
    return [
        {"id": o["id"], "statement": o["statement"]}
        for o in _tos() + _cos()
    ]


def _rollup() -> Dict[str, Any]:
    return _build_objective_rollup(_tos(), _cos())


# ---------------------------------------------------------------------------
# Relevance resolution
# ---------------------------------------------------------------------------


def test_to_target_pulls_to_plus_its_cos_child_co_ids():
    """A block whose objective_ids=[TO-01] resolves TO-01 + its COs (via
    ``child_co_ids``)."""
    assert _relevant_objective_ids(["TO-01"], _rollup()) == {
        "TO-01", "CO-01", "CO-02"
    }


def test_to_target_pulls_cos_via_parent_backpointer():
    """TO-02 has no ``child_co_ids`` — its COs roll up purely by the COs'
    parent back-pointers (``parent_to`` / ``parent_terminal``)."""
    assert _relevant_objective_ids(["TO-02"], _rollup()) == {
        "TO-02", "CO-03", "CO-04"
    }


def test_co_target_pulls_parent_to_and_siblings():
    """A block citing a CO directly resolves the CO + its parent TO + the
    TO's sibling COs (that TO/week)."""
    assert _relevant_objective_ids(["CO-03"], _rollup()) == {
        "TO-02", "CO-03", "CO-04"
    }


# ---------------------------------------------------------------------------
# (a) Payload filter — the primary contract from the spec.
# ---------------------------------------------------------------------------


def test_block_with_to02_receives_only_to02_and_its_cos():
    """A block whose objective_ids=[TO-02] receives ONLY TO-02 + its COs —
    not the full 6-entry list."""
    filtered = _filter_objectives_payload_for_block(
        ["TO-02"], _payload(), _rollup()
    )
    ids = [item["id"] for item in filtered]
    assert set(ids) == {"TO-02", "CO-03", "CO-04"}
    # Original payload order is preserved (TO before its COs).
    assert ids == ["TO-02", "CO-03", "CO-04"]
    # It is a strict subset — the other TO + its COs are excluded.
    assert "TO-01" not in ids and "CO-01" not in ids


def test_empty_objective_ids_falls_back_to_full_list():
    """An empty filter (block carries no objective_ids) → the FULL list
    (never an empty objectives section)."""
    payload = _payload()
    filtered = _filter_objectives_payload_for_block([], payload, _rollup())
    assert filtered == payload


def test_unknown_id_that_matches_nothing_falls_back_to_full_list():
    """A target id absent from the loaded doc matches no payload entry →
    the fallback returns the full list rather than an empty section."""
    payload = _payload()
    filtered = _filter_objectives_payload_for_block(
        ["ZZ-99"], payload, _rollup()
    )
    assert filtered == payload


def test_payload_item_shape_preserved():
    """The filter preserves the exact payload item shape (id + statement)."""
    filtered = _filter_objectives_payload_for_block(
        ["TO-01"], _payload(), _rollup()
    )
    for item in filtered:
        assert set(item.keys()) == {"id", "statement"}
