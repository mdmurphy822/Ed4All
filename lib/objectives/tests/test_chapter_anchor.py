"""Unit tests for the Defect-A chapter-anchor grouping core (PURE).

Synthetic ``mod-a`` / ``mod-b`` / ``mod-c`` modules — no corpus vocabulary, no
course slugs, no data paths. Exercises plurality voting, both deterministic
tie-breaks, the zero-resolve last-module default, the partition invariant, the
graceful-degrade signal properties, and the book-order reorder helper.
"""
from __future__ import annotations

from typing import Any, Dict, List

from lib.objectives.chapter_anchor import assign_cos_to_modules


# ---------------------------------------------------------------------------
# Synthetic corpus builders
# ---------------------------------------------------------------------------
def _chunk(cid: str, module_id: str, *, title: str = "", pos: int = 0) -> Dict[str, Any]:
    return {
        "id": cid,
        "source": {
            "module_id": module_id,
            "module_title": title or module_id.upper(),
            "position_in_module": pos,
        },
    }


def _three_module_chunks() -> List[Dict[str, Any]]:
    """Book order a → b → c; a has 3 chunks, b has 3, c has 2."""
    return [
        _chunk("a1", "mod-a", title="Alpha", pos=0),
        _chunk("a2", "mod-a", title="Alpha", pos=1),
        _chunk("a3", "mod-a", title="Alpha", pos=2),
        _chunk("b1", "mod-b", title="Bravo", pos=0),
        _chunk("b2", "mod-b", title="Bravo", pos=1),
        _chunk("b3", "mod-b", title="Bravo", pos=2),
        _chunk("c1", "mod-c", title="Charlie", pos=0),
        _chunk("c2", "mod-c", title="Charlie", pos=1),
    ]


def _by_id(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {c["id"]: c for c in chunks}


def _co(stmt: str, chunk_ids: List[str], **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"statement": stmt, "source_chunk_ids": list(chunk_ids)}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Module map + book order
# ---------------------------------------------------------------------------
def test_module_map_first_occurrence_order():
    chunks = _three_module_chunks()
    res = assign_cos_to_modules([], _by_id(chunks), chunks)
    # Empty CO list still builds the module map? No — empty cos short-circuits.
    assert res.module_order == []

    cos = [_co("x", ["a1"])]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.module_order == ["mod-a", "mod-b", "mod-c"]
    assert res.n_modules == 3
    assert res.module_title_by_id["mod-b"] == "Bravo"


# ---------------------------------------------------------------------------
# Plurality
# ---------------------------------------------------------------------------
def test_plurality_majority_module_wins():
    chunks = _three_module_chunks()
    # 2 votes mod-a, 1 vote mod-b → mod-a.
    cos = [_co("obj", ["a1", "a2", "b1"])]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.assignment[0] == "mod-a"
    assert res.multi_module_cos == 1  # spanned 2 modules
    assert res.ties == 0
    assert res.co_coverage == 1.0


# ---------------------------------------------------------------------------
# Tie-break (i) — entailing chunk's module
# ---------------------------------------------------------------------------
def test_tiebreak_entailing_chunk_module():
    chunks = _three_module_chunks()
    # 1 vote mod-a, 1 vote mod-b (tie); entailing chunk is in mod-b → mod-b.
    cos = [_co("obj", ["a1", "b1"], entailing_chunk_id="b1")]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.assignment[0] == "mod-b"
    assert res.ties == 1


def test_tiebreak_entailing_not_among_tied_falls_to_order():
    chunks = _three_module_chunks()
    # Tie mod-b/mod-c; entailing points at mod-a (NOT tied) → ignored → lowest
    # order among {mod-b, mod-c} = mod-b.
    cos = [_co("obj", ["b1", "c1"], entailing_chunk_id="a1")]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.assignment[0] == "mod-b"


# ---------------------------------------------------------------------------
# Tie-break (ii) — lowest module_order
# ---------------------------------------------------------------------------
def test_tiebreak_lowest_module_order():
    chunks = _three_module_chunks()
    # Tie mod-b/mod-c, no entailing → lowest book-order = mod-b.
    cos = [_co("obj", ["c1", "b1"])]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.assignment[0] == "mod-b"
    assert res.ties == 1


# ---------------------------------------------------------------------------
# Zero-resolve → last module, recorded, never dropped
# ---------------------------------------------------------------------------
def test_zero_resolve_defaults_to_last_module():
    chunks = _three_module_chunks()
    cos = [
        _co("resolves", ["a1"]),
        _co("dangling", ["nonexistent-1", "nonexistent-2"]),
    ]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.assignment[1] == "mod-c"  # last module
    assert res.unresolved_co_indices == [1]
    assert res.co_coverage == 0.5


def test_empty_source_chunk_ids_is_zero_resolve():
    chunks = _three_module_chunks()
    cos = [_co("no chunks", [])]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.unresolved_co_indices == [0]
    assert res.assignment[0] == "mod-c"


# ---------------------------------------------------------------------------
# Partition invariant — every CO in exactly one group; groups in book order
# ---------------------------------------------------------------------------
def test_partition_invariant_and_group_order():
    chunks = _three_module_chunks()
    cos = [
        _co("o0", ["c1"]),   # mod-c
        _co("o1", ["a1"]),   # mod-a
        _co("o2", ["b1"]),   # mod-b
        _co("o3", ["a2"]),   # mod-a
        _co("o4", ["nope"]),  # zero-resolve → mod-c
    ]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)

    # Groups are ordered mod-a, mod-b, mod-c (book order), non-empty only.
    assert [mid for mid, _ in res.groups] == ["mod-a", "mod-b", "mod-c"]
    # Partition: union of all group members == all CO indices, disjoint.
    seen: List[int] = []
    for _mid, members in res.groups:
        seen.extend(members)
    assert sorted(seen) == list(range(len(cos)))
    assert len(seen) == len(set(seen))  # disjoint
    # mod-a has o1 + o3 (input order preserved), mod-c has o0 + o4.
    groups = dict(res.groups)
    assert groups["mod-a"] == [1, 3]
    assert groups["mod-c"] == [0, 4]


# ---------------------------------------------------------------------------
# Degrade signal properties (the eligibility probe reads these)
# ---------------------------------------------------------------------------
def test_single_module_yields_n_modules_one():
    chunks = [_chunk("a1", "mod-a"), _chunk("a2", "mod-a")]
    cos = [_co("o", ["a1"]), _co("o2", ["a2"])]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.n_modules == 1  # caller degrades (min_modules default 2)


def test_low_coverage_reported():
    chunks = _three_module_chunks()
    cos = [_co("o0", ["a1"])] + [_co(f"o{i}", ["missing"]) for i in range(1, 5)]
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.co_coverage == 0.2  # 1 of 5 resolves → caller degrades at 0.80


# ---------------------------------------------------------------------------
# min_position_by_co — within-module reorder tie-break signal
# ---------------------------------------------------------------------------
def test_min_position_over_cited_chunks():
    chunks = _three_module_chunks()
    cos = [_co("o", ["a3", "a1"])]  # positions 2 and 0 → min 0
    res = assign_cos_to_modules(cos, _by_id(chunks), chunks)
    assert res.min_position_by_co[0] == 0


# ---------------------------------------------------------------------------
# Reorder helper — book-order stable sort (real pipeline code)
# ---------------------------------------------------------------------------
def test_reorder_cos_by_anchor_book_order_stable(monkeypatch):
    import MCP.tools.pipeline_tools as pt

    chunks = _three_module_chunks()
    # Deliberately out-of-book-order input; two mod-a COs at differing positions.
    cos = [
        _co("c-obj", ["c1"]),               # mod-c, pos 0
        _co("a-late", ["a3"]),              # mod-a, pos 2
        _co("b-obj", ["b1"]),               # mod-b, pos 0
        _co("a-early", ["a1"]),             # mod-a, pos 0
    ]
    reordered = pt._reorder_cos_by_anchor(cos, _by_id(chunks), chunks)
    # Expect mod-a (early then late), mod-b, mod-c.
    assert [c["statement"] for c in reordered] == [
        "a-early", "a-late", "b-obj", "c-obj",
    ]


def test_reorder_noop_when_single_module():
    import MCP.tools.pipeline_tools as pt

    chunks = [_chunk("a1", "mod-a"), _chunk("a2", "mod-a")]
    cos = [_co("second", ["a2"]), _co("first", ["a1"])]
    reordered = pt._reorder_cos_by_anchor(cos, _by_id(chunks), chunks)
    # n_modules < 2 → degrade → input order unchanged (same list object contents).
    assert [c["statement"] for c in reordered] == ["second", "first"]
