"""Tests for ``lib/ontology/terminal_coverage.py`` (roll-up model).

Covers the shared helper used by both Layer A (synthesis-side prune in
``_plan_course_structure``) and Layer B (the ``course_planning``
fail-fast gate). The two layers MUST agree on what "orphan terminal"
means, so the helper is tested in isolation here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.ontology.terminal_coverage import (  # noqa: E402
    attach_terminal_chapter_refs,
    co_parent_terminal,
    find_orphan_terminals,
    is_co_bearing,
    prune_orphan_terminals,
    rolled_up_terminal_ids,
    terminal_chapter_ref,
)


# --- roll-up key tolerance -------------------------------------------------

@pytest.mark.parametrize(
    "key", ["parent_to", "parent_terminal", "parent_terminal_id", "terminal_id"]
)
def test_co_parent_terminal_key_tolerance(key: str) -> None:
    assert co_parent_terminal({key: "TO-03"}) == "to-03"


def test_co_parent_terminal_none_when_absent() -> None:
    assert co_parent_terminal({"id": "CO-01"}) is None


def test_terminal_chapter_ref_int_and_str() -> None:
    assert terminal_chapter_ref({"chapter": 6}) == "6"
    assert terminal_chapter_ref({"chapter_id": "ch6"}) == "ch6"


# --- CO-bearing detection + rollup ----------------------------------------

def test_is_co_bearing() -> None:
    assert is_co_bearing(
        [{"chapter": "W1", "objectives": [{"id": "CO-01"}]}]
    )
    assert not is_co_bearing([])
    assert not is_co_bearing(None)


def test_rolled_up_terminal_ids() -> None:
    cos = [
        {"chapter": "W1", "objectives": [{"id": "CO-01", "parent_to": "TO-01"}]},
        {"chapter": "W2", "objectives": [{"id": "CO-02", "parent_to": "TO-02"}]},
    ]
    assert rolled_up_terminal_ids(cos) == {"to-01", "to-02"}


# --- orphan detection ------------------------------------------------------

def test_find_orphan_co_bearing() -> None:
    terminals = [{"id": "TO-01"}, {"id": "TO-02"}, {"id": "TO-99"}]
    cos = [
        {"chapter": "W1", "objectives": [{"id": "CO-01", "parent_to": "TO-01"}]},
        {"chapter": "W2", "objectives": [{"id": "CO-02", "parent_to": "TO-02"}]},
    ]
    orphans = find_orphan_terminals(terminals, cos)
    assert [o["id"] for o in orphans] == ["TO-99"]
    assert orphans[0]["reason"] == "no_chapter_objective"


def test_find_orphan_co_less_chapter_missing() -> None:
    terminals = [{"id": "TO-01", "chapter": "1"}, {"id": "TO-02", "chapter": "9"}]
    structure = {"chapters": [{"id": "ch1"}, {"id": "ch2"}]}
    orphans = find_orphan_terminals(terminals, [], textbook_structure=structure)
    assert [o["id"] for o in orphans] == ["TO-02"]
    assert orphans[0]["reason"] == "chapter_not_in_structure"


def test_find_orphan_co_less_chapter_label_normalisation() -> None:
    # "Chapter 1" / "Week 2" must reconcile with "ch1" / "ch2".
    terminals = [
        {"id": "TO-01", "chapter": "Chapter 1"},
        {"id": "TO-02", "chapter": "Week 2"},
    ]
    structure = {"chapters": [{"id": "ch1"}, {"id": "ch2"}]}
    assert find_orphan_terminals(terminals, [], textbook_structure=structure) == []


def test_find_orphan_co_less_no_structure_no_orphans() -> None:
    terminals = [{"id": "TO-01", "chapter": "1"}]
    assert find_orphan_terminals(terminals, []) == []


# --- prune (Layer A) -------------------------------------------------------

def test_prune_co_bearing_drops_orphan() -> None:
    terminals = [{"id": "TO-01"}, {"id": "TO-99"}]
    cos = [{"chapter": "W1", "objectives": [{"id": "CO-01", "parent_to": "TO-01"}]}]
    kept, pruned = prune_orphan_terminals(terminals, cos)
    assert [t["id"] for t in kept] == ["TO-01"]
    assert pruned == ["TO-99"]


def test_prune_co_bearing_consistent_keeps_all() -> None:
    """Backward-compat: an RDF-style consistent set is untouched."""
    terminals = [{"id": "TO-01"}, {"id": "TO-02"}]
    cos = [
        {"chapter": "W1", "objectives": [{"id": "CO-01", "parent_to": "TO-01"}]},
        {"chapter": "W2", "objectives": [{"id": "CO-02", "parent_to": "TO-02"}]},
    ]
    kept, pruned = prune_orphan_terminals(terminals, cos)
    assert [t["id"] for t in kept] == ["TO-01", "TO-02"]
    assert pruned == []


def test_prune_co_less_is_noop() -> None:
    """CO-less courses are NEVER pruned by Layer A (content-driven)."""
    terminals = [{"id": "TO-01", "chapter": "1"}, {"id": "TO-99", "chapter": "9"}]
    kept, pruned = prune_orphan_terminals(terminals, [])
    assert [t["id"] for t in kept] == ["TO-01", "TO-99"]
    assert pruned == []


def test_prune_co_bearing_no_backpointers_is_noop() -> None:
    """CO-bearing but zero COs carry a parent-terminal key ⟹ prune nothing.

    Regression for the Layer-A orphan-terminal prune that silently deleted
    EVERY terminal on the deterministic-synthesizer / Stage-2 /
    _normalize_objective_entry paths (none emit parent-terminal keys).
    The roll-up model is not in use, so orphanhood is unprovable and the
    prune must be a no-op (Layer-B gate adjudicates loudly).
    """
    terminals = [{"id": "TO-01"}, {"id": "TO-02"}, {"id": "TO-99"}]
    cos = [
        {"chapter": "W1", "objectives": [{"id": "CO-01"}, {"id": "CO-02"}]},
    ]
    kept, pruned = prune_orphan_terminals(terminals, cos)
    assert [t["id"] for t in kept] == ["TO-01", "TO-02", "TO-99"]
    assert pruned == []


def test_find_orphans_co_bearing_no_backpointers_returns_empty() -> None:
    """Layer A/B byte-agreement: no back-pointers ⟹ no orphans surfaced.

    find_orphan_terminals must mirror prune_orphan_terminals so the two
    layers agree byte-for-byte (the module's stated contract).
    """
    terminals = [{"id": "TO-01"}, {"id": "TO-02"}, {"id": "TO-99"}]
    cos = [
        {"chapter": "W1", "objectives": [{"id": "CO-01"}, {"id": "CO-02"}]},
    ]
    assert find_orphan_terminals(terminals, cos) == []


# --- attach_terminal_chapter_refs (CO-less Layer-A guarantee) --------------


def test_attach_resolves_co_less_orphans() -> None:
    """A CO-less course's chapter-less terminals get a resolvable ``chapter``
    back-pointer (round-robin over structure chapters) so the gate's
    no_chapter_ref orphan case can't fire. This is the deterministic
    synthesizer's bug: it mints terminals with no chapter back-pointer even
    on a corpus whose extractor synthesized real ch1/ch2/... chapters."""
    ts = {"chapters": [{"id": "ch1"}, {"id": "ch2"}, {"id": "ch3"}]}
    terminals = [{"id": "TO-01"}, {"id": "TO-02"}, {"id": "TO-03"}, {"id": "TO-04"}]
    out, assigned = attach_terminal_chapter_refs(terminals, [], textbook_structure=ts)
    assert assigned == ["TO-01", "TO-02", "TO-03", "TO-04"]
    assert [t["chapter"] for t in out] == ["ch1", "ch2", "ch3", "ch1"]
    # The fix makes the objectives set internally consistent: zero orphans.
    assert find_orphan_terminals(out, [], textbook_structure=ts) == []


def test_attach_does_not_mutate_input() -> None:
    ts = {"chapters": [{"id": "ch1"}]}
    terminals = [{"id": "TO-01"}]
    attach_terminal_chapter_refs(terminals, [], textbook_structure=ts)
    assert "chapter" not in terminals[0]


def test_attach_preserves_already_resolvable_backpointer() -> None:
    """A terminal that already carries a resolvable ``chapter`` is left
    verbatim — only chapter-less / unresolvable terminals are stamped."""
    ts = {"chapters": [{"id": "ch1"}, {"id": "ch2"}]}
    terminals = [{"id": "TO-01", "chapter": "ch2"}, {"id": "TO-02"}]
    out, assigned = attach_terminal_chapter_refs(terminals, [], textbook_structure=ts)
    assert out[0]["chapter"] == "ch2"  # untouched
    assert assigned == ["TO-02"]
    assert out[1]["chapter"] in {"ch1", "ch2"}


def test_attach_noop_for_co_bearing_course() -> None:
    ts = {"chapters": [{"id": "ch1"}]}
    terminals = [{"id": "TO-01"}]
    cos = [{"chapter": "W1", "objectives": [{"id": "CO-01", "parent_to": "TO-01"}]}]
    out, assigned = attach_terminal_chapter_refs(terminals, cos, textbook_structure=ts)
    assert assigned == []
    assert "chapter" not in out[0]


def test_attach_noop_without_resolvable_structure() -> None:
    """No chapters in structure ⟹ nothing to anchor to ⟹ no-op (the gate's
    TERMINAL_COVERAGE_UNVERIFIED graceful-degrade path owns that case)."""
    terminals = [{"id": "TO-01"}]
    out, assigned = attach_terminal_chapter_refs(terminals, [], textbook_structure={})
    assert assigned == []
    assert "chapter" not in out[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
