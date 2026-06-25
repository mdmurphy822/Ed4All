"""Unit tests for the rewrite-tier fit-window resolvers + chunk selector +
the system-prompt-trim equivalence (rewrite-overflow-fix-2026-06)."""
from __future__ import annotations

import Courseforge.generators._rewrite_provider as rp
from Courseforge.generators._rewrite_fit_window import (
    cited_chunk_ids_from_content,
    resolve_fit_window,
    resolve_rewrite_num_ctx,
    resolve_truncation_tripwire,
    select_chunks_under_budget,
)


# ---------------------------------------------------------------------------
# Resolvers — parse-with-fallback
# ---------------------------------------------------------------------------
def test_resolve_fit_window_parse_with_fallback():
    assert resolve_fit_window({}) is False
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "1"}) is True
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "true"}) is True
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "ON"}) is True
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "yes"}) is True
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "garbage"}) is False
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "0"}) is False
    assert resolve_fit_window({"ED4ALL_REWRITE_FIT_WINDOW": "off"}) is False


def test_resolve_rewrite_num_ctx_parse_with_fallback():
    assert resolve_rewrite_num_ctx({}) == 8192
    assert resolve_rewrite_num_ctx({"ED4ALL_REWRITE_NUM_CTX": "16384"}) == 16384
    assert resolve_rewrite_num_ctx({"ED4ALL_REWRITE_NUM_CTX": "0"}) == 8192
    assert resolve_rewrite_num_ctx({"ED4ALL_REWRITE_NUM_CTX": "-9"}) == 8192
    assert resolve_rewrite_num_ctx({"ED4ALL_REWRITE_NUM_CTX": "x"}) == 8192


def test_resolve_truncation_tripwire_default_on():
    assert resolve_truncation_tripwire({}) is True
    assert resolve_truncation_tripwire(
        {"ED4ALL_REWRITE_TRUNCATION_TRIPWIRE": "1"}
    ) is True
    assert resolve_truncation_tripwire(
        {"ED4ALL_REWRITE_TRUNCATION_TRIPWIRE": "garbage"}
    ) is True
    # Escape hatch.
    for falsey in ("0", "false", "no", "off"):
        assert resolve_truncation_tripwire(
            {"ED4ALL_REWRITE_TRUNCATION_TRIPWIRE": falsey}
        ) is False


# ---------------------------------------------------------------------------
# select_chunks_under_budget
# ---------------------------------------------------------------------------
def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "text": text}


def test_selector_fits_budget_drops_trailing_whole():
    # Large window keeps all; tiny window keeps only what fits, dropping the
    # trailing chunks whole (never head-truncating).
    chunks = [_chunk(f"c{i}", "word " * 200) for i in range(8)]
    selected, dropped = select_chunks_under_budget(
        chunks, num_ctx=100000, sys_tokens=100, scaffold_tokens=100,
        max_tokens=2400,
    )
    assert len(selected) == 8 and dropped == []

    selected2, _ = select_chunks_under_budget(
        chunks, num_ctx=1000, sys_tokens=100, scaffold_tokens=100,
        max_tokens=400,
    )
    # Only a prefix fits; the rest are dropped whole.
    assert 1 <= len(selected2) < 8


def test_selector_always_keeps_at_least_one():
    # A single chunk that alone exceeds the budget is still emitted (the
    # per-chunk cap bounds it; empty grounding is worse).
    chunks = [_chunk("big", "word " * 5000)]
    selected, dropped = select_chunks_under_budget(
        chunks, num_ctx=10, sys_tokens=100, scaffold_tokens=100,
        max_tokens=4000,
    )
    assert len(selected) == 1 and dropped == []


def test_selector_cited_first_and_records_dropped_cited():
    # Cited chunks are ordered first; a cited chunk dropped under budget is
    # recorded in dropped_cited_chunk_ids.
    chunks = [
        _chunk("a", "word " * 300),
        _chunk("cited1", "word " * 300),
        _chunk("cited2", "word " * 300),
        _chunk("b", "word " * 300),
    ]
    cited = ["cited1", "cited2"]
    # Generous budget → both cited fit first; a/b also fit. No cited dropped.
    selected, dropped = select_chunks_under_budget(
        chunks, num_ctx=100000, sys_tokens=50, scaffold_tokens=50,
        max_tokens=200, cited_chunk_ids=cited,
    )
    sel_ids = [c["id"] for c in selected]
    # Cited come first (before the non-cited a/b).
    assert sel_ids[0] in cited and sel_ids[1] in cited
    # No cited dropped here (they're prioritized and all fit).
    assert dropped == []

    # Now squeeze so a cited chunk is dropped.
    selected2, dropped2 = select_chunks_under_budget(
        chunks, num_ctx=600, sys_tokens=50, scaffold_tokens=50,
        max_tokens=200, cited_chunk_ids=cited,
    )
    # The kept set is cited-first; some cited dropped under the tight budget.
    assert set(dropped2).issubset(set(cited))


def test_selector_embedding_absent_falls_back_to_input_order():
    # No embedder / no query → non-cited chunks keep input order (citation-
    # order fallback, NOT a fail-closed dep).
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta"), _chunk("c", "gamma")]
    selected, _ = select_chunks_under_budget(
        chunks, num_ctx=100000, sys_tokens=10, scaffold_tokens=10,
        max_tokens=10, embedder=None, rank_query=None,
    )
    assert [c["id"] for c in selected] == ["a", "b", "c"]


def test_selector_never_invents_a_chunk():
    chunks = [_chunk("a", "x"), _chunk("b", "y")]
    selected, _ = select_chunks_under_budget(
        chunks, num_ctx=100000, sys_tokens=0, scaffold_tokens=0,
        max_tokens=0,
    )
    assert {c["id"] for c in selected}.issubset({"a", "b"})


def test_cited_chunk_ids_from_content():
    content = {
        "key_claims": [
            {"text": "c1", "source_chunk_ids": ["x", "y"]},
            {"text": "c2", "source_chunk_ids": ["y", "z"]},
            "bare-string-claim",
        ]
    }
    assert cited_chunk_ids_from_content(content) == ["x", "y", "z"]
    # Non-dict content → no cited ids.
    assert cited_chunk_ids_from_content("<p>html</p>") == []
    assert cited_chunk_ids_from_content({}) == []


# ---------------------------------------------------------------------------
# System-prompt-trim equivalence
# ---------------------------------------------------------------------------
def test_system_prompt_off_is_byte_identical():
    # OFF → the untrimmed prompt, byte-identical to the constant.
    assert rp._resolve_system_prompt(False) == rp._REWRITE_SYSTEM_PROMPT


def test_trimmed_prompt_is_strictly_shorter_but_loses_no_kept_text():
    full = rp._REWRITE_SYSTEM_PROMPT
    trimmed = rp._resolve_system_prompt(True)
    assert len(trimmed) < len(full)
    # Every KEEP segment appears verbatim in both.
    segs = rp._REWRITE_SYSTEM_PROMPT_SEGMENTS
    for i, seg in enumerate(segs):
        if i not in rp._RELOCATED_SEGMENT_INDICES:
            assert seg in trimmed
            assert seg in full


def test_no_directive_text_is_lost_overall():
    # The union of (trimmed system prompt) + (every relocated per-type
    # suffix) must contain every original segment — no directive vanishes.
    trimmed = rp._resolve_system_prompt(True)
    # Every relocated segment must land in SOME per-type contract.
    placed = set()
    for indices in rp._RELOCATED_SEGMENTS_BY_BLOCK_TYPE.values():
        placed.update(indices)
    # Every relocated-out index is placed into at least one contract.
    assert rp._RELOCATED_SEGMENT_INDICES.issubset(placed)
    # And each placed segment's text appears in its contract.
    for bt, indices in rp._RELOCATED_SEGMENTS_BY_BLOCK_TYPE.items():
        contract = rp._block_type_output_contract(bt, fit_window=True)
        for i in indices:
            assert rp._REWRITE_SYSTEM_PROMPT_SEGMENTS[i] in contract
    # Sanity: trimmed prompt still carries the Wave-27 MANDATORY contract.
    assert "MANDATORY OUTPUT CONTRACT" in trimmed


def test_contract_off_byte_identical_on_appends_only():
    for bt in ("assessment_item", "concept", "self_check_question", "example"):
        off = rp._block_type_output_contract(bt, fit_window=False)
        on = rp._block_type_output_contract(bt, fit_window=True)
        assert on.startswith(off)
        assert len(on) > len(off)


def test_block_type_without_relocation_unchanged_on():
    # A block type with no relocated segments → contract identical ON/OFF.
    off = rp._block_type_output_contract("summary_takeaway", fit_window=False)
    on = rp._block_type_output_contract("summary_takeaway", fit_window=True)
    assert off == on
