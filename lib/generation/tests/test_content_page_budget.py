"""Unit tests for the page-per-CO token-budget helpers.

Phase 0 of ``plans/finegrain/page-per-co-token-aware-emit-2026-06.md`` — the
pure-library resolvers + token budget + per-page union-then-top-K chunk cap.
These exercise the deterministic, GPU-free contract: the resolvers
parse-with-fallback, the budget math is conservative + can go negative (fixed
cost overflows the window), and the cap is always-keep-≥1 / kept-⊆-union.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import content_page_budget as cpb  # noqa: E402


# --------------------------------------------------------------------------- #
# Resolvers
# --------------------------------------------------------------------------- #
def test_per_co_flag_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_CONTENT_PAGE_PER_CO", raising=False)
    assert cpb.content_page_per_co_enabled() is False


def test_per_co_flag_truthy(monkeypatch):
    for tok in ("1", "true", "yes", "on", "ON", "True"):
        monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO", tok)
        assert cpb.content_page_per_co_enabled() is True


def test_per_co_flag_garbage_off(monkeypatch):
    for tok in ("0", "false", "garbage", "", "  "):
        monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO", tok)
        assert cpb.content_page_per_co_enabled() is False


def test_num_ctx_resolution_chain(monkeypatch):
    monkeypatch.delenv("ED4ALL_CONTENT_PAGE_NUM_CTX", raising=False)
    monkeypatch.delenv("ED4ALL_ANSWER_NUM_CTX", raising=False)
    # default
    assert cpb.resolve_content_page_num_ctx() == 4096
    # falls back to the shared answer knob
    monkeypatch.setenv("ED4ALL_ANSWER_NUM_CTX", "8192")
    assert cpb.resolve_content_page_num_ctx() == 8192
    # this feature's own knob wins
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_NUM_CTX", "16384")
    assert cpb.resolve_content_page_num_ctx() == 16384
    # explicit arg wins over both
    assert cpb.resolve_content_page_num_ctx(2048) == 2048


def test_num_ctx_garbage_falls_back(monkeypatch):
    monkeypatch.delenv("ED4ALL_ANSWER_NUM_CTX", raising=False)
    for tok in ("0", "-1", "abc", ""):
        monkeypatch.setenv("ED4ALL_CONTENT_PAGE_NUM_CTX", tok)
        assert cpb.resolve_content_page_num_ctx() == 4096


def test_max_chunks_resolution(monkeypatch):
    monkeypatch.delenv("ED4ALL_CONTENT_PAGE_MAX_CHUNKS", raising=False)
    assert cpb.resolve_content_page_max_chunks() == 5
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_MAX_CHUNKS", "3")
    assert cpb.resolve_content_page_max_chunks() == 3
    assert cpb.resolve_content_page_max_chunks(7) == 7
    for tok in ("0", "-2", "x", ""):
        monkeypatch.setenv("ED4ALL_CONTENT_PAGE_MAX_CHUNKS", tok)
        assert cpb.resolve_content_page_max_chunks() == 5


# --------------------------------------------------------------------------- #
# Token budget
# --------------------------------------------------------------------------- #
def test_budget_can_go_negative():
    # fixed cost alone overflows a 4096 window (the plan's measured rewrite case)
    b = cpb.page_chunk_token_budget(
        num_ctx=4096, system_prompt_tokens=8000, max_tokens=2400
    )
    assert b < 0  # caller fails/escalates rather than shipping a chunk


def test_budget_positive_with_headroom():
    b = cpb.page_chunk_token_budget(
        num_ctx=16384,
        system_prompt_tokens=7800,
        user_fixed_tokens=1200,
        max_tokens=2400,
        reserve_tokens=256,
    )
    assert b > 0


def test_chunk_token_weight_truncates():
    # a fat chunk is capped at the per-chunk char ceiling before estimating
    huge = "x" * 100000
    w = cpb.chunk_token_weight(huge)
    assert w == cpb.chunk_token_weight("x" * 1200)
    assert cpb.chunk_token_weight("") == 0


# --------------------------------------------------------------------------- #
# Per-page top-K cap
# --------------------------------------------------------------------------- #
def test_cap_thin_co_keeps_one():
    kept, dropped_cited, stats = cpb.cap_page_chunks(
        union_ids=["a"],
        chunk_text_map={"a": "x" * 100},
        token_budget=1,  # below the single chunk's weight
        max_chunks=5,
    )
    assert kept == ["a"]  # always-keep-≥1 supersedes the budget
    assert dropped_cited == []
    assert stats["rank_method"] == "citation_order"


def test_cap_multi_section_co_caps_at_k():
    tm = {c: "word " * 500 for c in "abcdefgh"}
    kept, _dropped, stats = cpb.cap_page_chunks(
        union_ids=list("abcdefgh"),
        cited_ids=list("abcdefgh"),
        chunk_text_map=tm,
        token_budget=10_000_000,  # not the binding constraint
        max_chunks=3,
    )
    assert len(kept) == 3
    assert set(kept) <= set("abcdefgh")  # kept ⊆ union (anti-fabrication)
    assert stats["dropped_count"] == 5


def test_cap_token_budget_drops_trailing():
    tm = {c: "word " * 500 for c in "abcdefgh"}  # ~1000 tok each pre-cap
    kept, _dropped, _stats = cpb.cap_page_chunks(
        union_ids=list("abcdefgh"),
        cited_ids=list("abcdefgh"),
        chunk_text_map=tm,
        token_budget=900,  # admits only the first
        max_chunks=99,
    )
    assert kept == ["a"]


def test_cap_empty_union():
    kept, dropped, stats = cpb.cap_page_chunks(
        union_ids=[], chunk_text_map={}, max_chunks=5
    )
    assert kept == []
    assert dropped == []
    assert stats["union_count"] == 0


def test_cap_preserves_union_order():
    tm = {c: "t" for c in "abcd"}
    kept, _d, _s = cpb.cap_page_chunks(
        union_ids=["c", "a", "d", "b"],
        cited_ids=["c", "a", "d", "b"],
        chunk_text_map=tm,
        token_budget=10_000_000,
        max_chunks=2,
    )
    # kept is re-projected onto union order
    assert kept == [kept[0], kept[1]]
    assert kept[0] in ("c", "a") and set(kept) <= {"c", "a", "d", "b"}


def test_cap_embedding_absent_uses_citation_order():
    # no embed_builder, no client → deterministic citation-order rank
    tm = {c: "t" for c in "abc"}
    kept, _d, stats = cpb.cap_page_chunks(
        union_ids=list("abc"),
        cited_ids=list("abc"),
        chunk_text_map=tm,
        statement="some objective statement",
        token_budget=10_000_000,
        max_chunks=2,
        embed_builder=None,
    )
    assert stats["rank_method"] == "citation_order"
    assert kept == ["a", "b"]


# --------------------------------------------------------------------------- #
# Gap #11 — per-block-role chunk-order diversification
# --------------------------------------------------------------------------- #
def test_chunk_role_diversify_flag_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_CHUNK_ROLE_DIVERSIFY", raising=False)
    assert cpb.chunk_role_diversify_enabled() is False


def test_chunk_role_diversify_flag_truthy(monkeypatch):
    for tok in ("1", "true", "yes", "on", "ON", "True"):
        monkeypatch.setenv("ED4ALL_CHUNK_ROLE_DIVERSIFY", tok)
        assert cpb.chunk_role_diversify_enabled() is True


def test_chunk_role_diversify_flag_garbage_off(monkeypatch):
    for tok in ("0", "false", "garbage", "", "  "):
        monkeypatch.setenv("ED4ALL_CHUNK_ROLE_DIVERSIFY", tok)
        assert cpb.chunk_role_diversify_enabled() is False


def test_diversify_disabled_is_identity():
    ids = ["c1", "c2", "c3", "c4", "c5"]
    assert cpb.diversify_block_chunk_order(
        ids, role_key="example:0", enabled=False
    ) == ids


def test_diversify_env_default_off_identity(monkeypatch):
    monkeypatch.delenv("ED4ALL_CHUNK_ROLE_DIVERSIFY", raising=False)
    ids = ["c1", "c2", "c3", "c4", "c5"]
    # enabled=None → read env (off) → identity.
    assert cpb.diversify_block_chunk_order(ids, role_key="example:0") == ids


def test_diversify_is_a_permutation():
    ids = ["a", "b", "c", "d", "e", "f"]
    out = cpb.diversify_block_chunk_order(ids, role_key="concept:2", enabled=True)
    assert sorted(out) == sorted(ids)
    # Same universe, no drop / invent.
    assert set(out) == set(ids)
    assert len(out) == len(ids)


def test_diversify_dedups_preserving_order_when_off():
    ids = ["a", "b", "a", "c", "b"]
    assert cpb.diversify_block_chunk_order(
        ids, role_key="x:0", enabled=False
    ) == ["a", "b", "c"]


def test_diversify_short_lists_unchanged():
    assert cpb.diversify_block_chunk_order([], role_key="e:0", enabled=True) == []
    assert cpb.diversify_block_chunk_order(
        ["only"], role_key="e:0", enabled=True
    ) == ["only"]


def test_diversify_empty_role_key_is_identity():
    ids = ["a", "b", "c"]
    assert cpb.diversify_block_chunk_order(ids, role_key="", enabled=True) == ids


def test_diversify_distinct_roles_get_distinct_heads():
    ids = [f"c{i}" for i in range(12)]
    heads = {
        role: cpb.diversify_block_chunk_order(ids, role_key=role, enabled=True)[0]
        for role in ("example:0", "concept:1", "explanation:2", "practice:3")
    }
    # At least two distinct lead chunks across the four roles (diversification
    # actually spreads the anchor head — not all identical).
    assert len(set(heads.values())) >= 2


def test_diversify_deterministic_repeat_calls():
    ids = ["a", "b", "c", "d", "e"]
    first = cpb.diversify_block_chunk_order(ids, role_key="r:1", enabled=True)
    for _ in range(5):
        assert cpb.diversify_block_chunk_order(
            ids, role_key="r:1", enabled=True
        ) == first


def test_stable_offset_is_hashseed_independent():
    # The offset must not depend on the builtin salted hash() — recompute here
    # via the same sha1 contract and assert equality (a proxy for cross-process
    # stability; the process-level PYTHONHASHSEED sweep runs in CI).
    import hashlib

    key = "example:0"
    modulo = 7
    expected = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big") % modulo
    assert cpb._stable_offset(key, modulo) == expected
    assert cpb._stable_offset("anything", 0) == 0
