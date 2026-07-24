"""NORMALIZED-WINDOW architecture tests for the GLM-OCR heading judge
(``SEMANTIK_HEADING_JUDGE_NORMALIZE``) — CPU-only, deterministic, mock post_fn.

Covers: adaptive W resolution (NEAREST-RANK + fixed override + clamps +
degenerate); oversized-chapter splitting into overlapping <= W slices (overlap +
coverage invariant); the no-split single-window path; the skeleton-only chapter
reviewer + the deterministic interior-slice-wins fallback (exact pick + reviewer
error); the byte-identical default-OFF guarantee (chapter_slices is None); and
the end-to-end run (every pending verdicted, zero finish=length, chapter review
fired ONLY for the oversized chapter, whole-doc final review still runs).

Reuses ``test_heading_judge.py``'s post_fn stub pattern + the legacy-seat-context
fixture. NO GPU, NO live seat — every seat call is a mock.
"""

from __future__ import annotations

import json
import re

import pytest

from semantik_structure.glmocr import heading_judge as hj


@pytest.fixture(autouse=True)
def _normalize_env(monkeypatch):
    """Force the LEGACY hardcoded budgets (seat-context probe off) + LEGACY
    char-heuristic token counting (``SEMANTIK_HEADING_JUDGE_TOKENIZER=off``) so a
    live host seat / tokenizer can never perturb the digest budget the W clamp
    reads or the slice boundaries the ``<= W`` assertions pin, and clear the
    probe + tokenizer caches. Each test sets its own NORMALIZE_* envs."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "off")
    # Thinking-ON: the budget resolvers are now thinking-aware; these tests pin
    # the legacy digest-budget / W behaviour (unchanged by mode) — the explicit
    # setting keeps them immune to the thinking-off default flip.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "1")
    hj._reset_seat_context_cache()
    hj._reset_tokenizer_cache()
    yield
    hj._reset_seat_context_cache()
    hj._reset_tokenizer_cache()


# ── Synthetic fixtures (no real corpus). ─────────────────────────────────────
def _chapter(opener_n, n_units, *, content_len=120, start=0):
    """A chapter: fixed level-1 ``Chapter N`` opener + ``n_units`` pending
    heading/paragraph pairs. Returns (regions, next_index)."""
    regions = []
    idx = start
    regions.append({"region_kind": "heading", "heading_text": f"Chapter {opener_n}",
                    "level": 1, "first_raw_block_index": idx, "source_page": 1})
    idx += 1
    for _ in range(n_units):
        regions.append({"region_kind": "heading", "heading_text": f"Topic {idx}",
                        "level": 3, "first_raw_block_index": idx, "source_page": 1,
                        "heading_level_pending": True})
        idx += 1
        regions.append({"region_kind": "paragraph", "raw_text": "y" * content_len,
                        "first_raw_block_index": idx, "source_page": 1})
        idx += 1
    return regions, idx


def _multi_chapter_book():
    """One oversized chapter (8 units) + two small chapters (2 + 1 units)."""
    prov = []
    r0, nxt = _chapter(1, 8, start=0)
    prov += r0
    r1, nxt = _chapter(2, 2, start=nxt)
    prov += r1
    r2, nxt = _chapter(3, 1, start=nxt)
    prov += r2
    return prov


def _level_all_post(level=2):
    """post_fn that assigns ``level`` to every pending region id it sees in the
    window user message (``R<id> ... hN*``). finish='stop' (never length)."""
    def _post(messages, max_tokens):
        user = messages[-1]["content"]
        ids = [int(m) for m in re.findall(r"R(\d+) p\S+ h\d+\*", user)]
        return json.dumps({"levels": {str(i): level for i in ids}}), "stop"
    return _post


# ── 1. W resolution. ─────────────────────────────────────────────────────────
def test_w_nearest_rank_p75(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN", "1")
    # P=75 is now OPT-IN (the default is 100 — overflow-only); pin it to exercise
    # the NEAREST-RANK rule.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", "75")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", raising=False)
    # NEAREST-RANK: k = ceil(0.75 * 4) = 3 → sorted[2] = 300 (interpolation would
    # give 325 — the assertion pins the exact rule implemented).
    assert hj.resolve_normalized_window_tokens([100, 200, 300, 400]) == 300


def test_w_default_percentile_is_100_overflow_only(monkeypatch):
    """DEFAULT (percentile unset) → P=100 → W == the MAX chapter size, so a book
    whose chapters all fit the digest budget targets its own largest chapter and
    NEVER proactively splits (the A/B-driven overflow-only scoping)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN", "1")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", raising=False)
    assert hj.resolve_normalize_percentile() == 100
    # p100 nearest-rank of [100,200,300,400] == 400 (the max), not p75's 300.
    assert hj.resolve_normalized_window_tokens([100, 200, 300, 400]) == 400


def test_w_fixed_override_wins(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "777")
    # Override wins regardless of the sizes / percentile / clamps.
    assert hj.resolve_normalized_window_tokens([1, 2, 3, 999999]) == 777


def test_w_upper_clamps_to_digest_budget(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN", "1")
    budget = hj.resolve_digest_budget_tokens()
    # >1 element so it's not the degenerate path; percentile far above budget.
    assert hj.resolve_normalized_window_tokens([10 ** 9, 10 ** 9]) == budget


def test_w_lower_clamps_to_window_min(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN", "5000")
    # percentile of [10,10,10] is 10 → floored up to W_MIN 5000 (< budget).
    assert hj.resolve_normalized_window_tokens([10, 10, 10]) == 5000


def test_w_degenerate_zero_or_one_chapter_is_digest_budget(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    budget = hj.resolve_digest_budget_tokens()
    assert hj.resolve_normalized_window_tokens([]) == budget
    assert hj.resolve_normalized_window_tokens([123]) == budget


def test_percentile_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", "garbage")
    assert hj.resolve_normalize_percentile() == 100  # default flipped 75 -> 100
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", "500")
    assert hj.resolve_normalize_percentile() == 100  # clamped into [1, 100]
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", "-5")
    assert hj.resolve_normalize_percentile() == 1
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", "75")
    assert hj.resolve_normalize_percentile() == 75  # lower P still reachable


# ── 2. Split into overlapping <= W slices. ───────────────────────────────────
def test_oversized_chapter_splits_with_overlap_and_full_coverage(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "150")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SLICE_OVERLAP", "2")
    prov, _ = _chapter(1, 10)  # one big chapter, forced to split by a small W
    plan = hj.build_heading_skeleton(prov)

    assert plan.chapter_slices is not None
    assert len(plan.windows) >= 2  # genuinely split
    W = 150
    # every emitted window fits W (truncation impossible by construction)
    for digest, _pend in plan.windows:
        assert hj._estimate_tokens(digest) <= W
    # every split slice is flagged split
    assert all(cs["split"] for cs in plan.chapter_slices)
    # adjacent slices share EXACTLY `overlap` (2) heading ids
    for a, b in zip(plan.chapter_slices, plan.chapter_slices[1:]):
        shared = set(a["heading_ids"]) & set(b["heading_ids"])
        assert len(shared) == 2
    # COVERAGE invariant: union of slice CORES == the chapter's pendings
    core_union = set()
    for cs in plan.chapter_slices:
        core_union |= set(cs["core_pending_ids"])
    assert core_union == set(plan.pending_ids)


def test_split_boundary_heading_appears_in_two_slices(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "150")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SLICE_OVERLAP", "2")
    prov, _ = _chapter(1, 10)
    plan = hj.build_heading_skeleton(prov)
    # an overlap pending appears both as a later slice's overlap AND an earlier
    # slice's window pending set (judged twice).
    overlap_ids = set()
    for cs in plan.chapter_slices:
        overlap_ids |= set(cs["overlap_pending_ids"])
    assert overlap_ids  # at least one boundary heading is re-included
    for oid in overlap_ids:
        windows_with = [w for (w, cs) in zip(plan.windows, plan.chapter_slices)
                        if oid in cs["core_pending_ids"] or oid in cs["overlap_pending_ids"]]
        assert len(windows_with) >= 2


_SENTINEL = "LAST_REGION_SENTINEL_ZZ"


def _oversized_chapter(n_units=40, body_chars=8500):
    """A single chapter whose TRUE (unbudgeted) content far exceeds a realistic
    digest budget, with a sentinel token at the end of the last region so a
    truncated render is detectable."""
    regions = []
    idx = 0
    regions.append({"region_kind": "heading", "heading_text": "Chapter 1",
                    "level": 1, "first_raw_block_index": idx, "source_page": 1})
    idx += 1
    for u in range(n_units):
        regions.append({"region_kind": "heading", "heading_text": f"Topic {u}",
                        "level": 3, "first_raw_block_index": idx, "source_page": 1,
                        "heading_level_pending": True})
        idx += 1
        body = "w" * body_chars
        if u == n_units - 1:
            body = body + " " + _SENTINEL
        regions.append({"region_kind": "paragraph", "raw_text": body,
                        "first_raw_block_index": idx, "source_page": 1})
        idx += 1
    return regions


def test_oversized_chapter_splits_under_realistic_budget(monkeypatch):
    """REGRESSION: with a REALISTIC digest budget (not 1), an ~85k-token chapter
    must SPLIT (>=2 slices, split=True, W==budget, every window <= W) and each
    slice must carry FULL, un-truncated content. Fails against the pre-fix code,
    where the budgeted render capped the measured size at ~digest_budget so the
    ``size <= W`` trigger was trivially true and it shipped one TRUNCATED
    window."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 38092)
    prov = _oversized_chapter()

    # the TRUE chapter size is far over budget (the sizer must see this).
    full, _ = hj._render_chapter_window(prov, "", budgeted=False)
    assert hj._estimate_tokens(full) > 38092

    plan = hj.build_heading_skeleton(prov)
    W = plan.chapter_slices[0]["W"]
    assert W == 38092                       # adaptive W == the digest budget
    assert len(plan.windows) >= 2           # genuinely split (pre-fix: 1)
    assert all(cs["split"] for cs in plan.chapter_slices)
    for digest, _pend in plan.windows:
        assert hj._estimate_tokens(digest) <= W  # truncation impossible
    # FULL content preserved — the last region's sentinel survives in the last
    # slice (a budget-truncated slice would have dropped it).
    assert _SENTINEL in plan.windows[-1][0]
    # coverage invariant still holds on the real-budget split.
    core_union = set()
    for cs in plan.chapter_slices:
        core_union |= set(cs["core_pending_ids"])
    assert core_union == set(plan.pending_ids)


def test_default_percentile_no_split_when_all_chapters_fit(monkeypatch):
    """Task-3 overflow-only scoping: at the DEFAULT percentile (100, unset) a
    multi-chapter book whose chapters ALL fit the digest budget produces ZERO
    splits — no proactive balancing of the typical-outlier chapter."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", raising=False)
    prov = _multi_chapter_book()          # chapters 8/2/1 units — all tiny
    plan = hj.build_heading_skeleton(prov)
    assert plan.chapter_slices is not None
    # one window per chapter, NONE split.
    assert all(cs["n_slices"] == 1 for cs in plan.chapter_slices)
    assert not any(cs["split"] for cs in plan.chapter_slices)
    # p75 WOULD have split the 8-unit outlier under a small W_MIN; p100 does not.


def test_default_percentile_splits_over_budget_chapter(monkeypatch):
    """Task-3: even at the DEFAULT percentile (100) a chapter whose one-window
    size genuinely EXCEEDS the seat digest budget still splits (overflow =
    correctness), while its book-mates that fit do not."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", raising=False)
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 38092)
    # chapter 0 = the oversized (>38092-token) fixture; chapters 1,2 tiny.
    big = _oversized_chapter()                       # indices 0..80
    small1, nxt = _chapter(2, 2, start=len(big))
    small2, _ = _chapter(3, 1, start=nxt)
    prov = big + small1 + small2
    # the over-budget chapter is genuinely over budget (the sizer sees it).
    assert hj._estimate_tokens(hj._render_chapter_window(big, "", budgeted=False)[0]) > 38092

    plan = hj.build_heading_skeleton(prov)
    by_ch = {}
    for cs in plan.chapter_slices:
        by_ch.setdefault(cs["chapter_id"], []).append(cs)
    # W clamps to the digest budget (upper clamp wins over the p100 max).
    assert plan.chapter_slices[0]["W"] == 38092
    # the oversized chapter (id 0) SPLIT into >=2 slices …
    assert len(by_ch[0]) >= 2 and all(cs["split"] for cs in by_ch[0])
    # … while the two fitting chapters are each ONE unsplit window.
    for cid in (1, 2):
        assert len(by_ch[cid]) == 1 and by_ch[cid][0]["split"] is False


def test_budgeted_default_truncates_chapter_mode_unbudgeted_normalized_does_not(monkeypatch):
    """The ``budgeted=True`` DEFAULT keeps chapter-mode's truncating render
    (byte-identical), while normalized mode's ``budgeted=False`` preserves full
    content — the two renders DIFFER on an over-budget span."""
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 38092)
    prov = _oversized_chapter()
    budgeted, _ = hj._render_chapter_window(prov, "", budgeted=True)
    default, _ = hj._render_chapter_window(prov, "")   # default is budgeted=True
    unbudgeted, _ = hj._render_chapter_window(prov, "", budgeted=False)

    assert budgeted == default                         # default == budgeted=True
    assert _SENTINEL not in budgeted                   # chapter-mode truncates
    assert _SENTINEL in unbudgeted                     # normalized keeps it all
    assert hj._estimate_tokens(budgeted) <= 38092      # budgeted fits the budget
    assert hj._estimate_tokens(unbudgeted) > 38092     # true size exceeds it


def test_chapter_mode_plan_untouched_uses_budgeted_render(monkeypatch):
    """CHAPTER-MODE (not normalized) still budget-truncates an over-budget
    chapter — the fix left ``_build_chapter_mode_plan`` byte-identical."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", raising=False)
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 38092)
    prov = _oversized_chapter()
    plan = hj.build_heading_skeleton(prov)
    assert plan.chapter_slices is None                 # not the normalized path
    assert len(plan.windows) == 1                       # chapter-mode: one window
    assert _SENTINEL not in plan.windows[0][0]          # budget-truncated


# ── 3. No-split single-window path. ──────────────────────────────────────────
def test_small_chapter_is_one_window_not_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    # a large W so a small chapter fits in one window
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "100000")
    prov, _ = _chapter(1, 2)
    plan = hj.build_heading_skeleton(prov)
    assert len(plan.windows) == 1
    assert plan.chapter_slices is not None
    assert plan.chapter_slices[0]["n_slices"] == 1
    assert plan.chapter_slices[0]["split"] is False


def test_no_chapter_review_call_for_unsplit_chapter(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "100000")
    prov, _ = _chapter(1, 2)
    tree, esc = [], []
    rep = hj.run_heading_judge(prov, tree, esc, post_fn=_level_all_post(),
                               use_cache=False, emit_capture=False)
    assert rep["meta"].get("chapter_reviews") == 0


# ── 4. Chapter reviewer dedup + deterministic interior-slice-wins fallback. ──
def test_deterministic_interior_slice_wins_exact_pick():
    # heading 5 is judged by TWO slices at DIFFERENT levels; it sits INTERIOR in
    # slice A (middle of [3,5,7]) and at the EDGE of slice B (start of [5,7,9]).
    slice_a = {"heading_ids": [3, 5, 7], "verdicts": {3: 2, 5: 4, 7: 3}}
    slice_b = {"heading_ids": [5, 7, 9], "verdicts": {5: 6, 7: 3, 9: 4}}
    out = hj._deterministic_consolidate([slice_a, slice_b])
    # 5: slice A min-edge-dist = min(1, 1) = 1 (interior) beats slice B dist 0 →
    # slice A's verdict (4) wins.
    assert out[5] == 4
    # 7: dist 0 in A (edge), dist 1 in B (interior) → B's verdict (3) wins.
    assert out[7] == 3
    # singly-judged ids keep their only verdict.
    assert out[3] == 2 and out[9] == 4


def test_interior_tie_breaks_to_earlier_slice():
    # heading 5 at the SAME min-edge-distance in both slices → the EARLIER slice
    # wins the tie.
    slice_a = {"heading_ids": [4, 5, 6], "verdicts": {5: 2}}   # dist(5)=1
    slice_b = {"heading_ids": [1, 5, 9], "verdicts": {5: 5}}   # dist(5)=1
    out = hj._deterministic_consolidate([slice_a, slice_b])
    assert out[5] == 2  # earlier slice (A)


def test_chapter_review_off_uses_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW", "0")
    slice_a = {"heading_ids": [3, 5, 7], "verdicts": {5: 4}}
    slice_b = {"heading_ids": [5, 7, 9], "verdicts": {5: 6}}
    entries = [hj.SkeletonEntry(rid, 3, f"T{rid}", 1, True, None)
               for rid in (3, 5, 7, 9)]

    def _boom(messages, max_tokens):  # must never be called when review is off
        raise AssertionError("reviewer POST fired with CHAPTER_REVIEW off")

    out = hj.run_chapter_review([], entries, [slice_a, slice_b], post_fn=_boom)
    assert out[5] == 4  # interior-slice-wins deterministic pick


def test_chapter_review_error_falls_back_complete(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW", "1")
    slice_a = {"heading_ids": [3, 5], "verdicts": {3: 2, 5: 4}}
    slice_b = {"heading_ids": [5, 7], "verdicts": {5: 6, 7: 3}}
    entries = [hj.SkeletonEntry(rid, 3, f"T{rid}", 1, True, None)
               for rid in (3, 5, 7)]

    def _raise(messages, max_tokens):
        raise hj._JudgeTransportError("seat down", transient=False)

    out = hj.run_chapter_review([], entries, [slice_a, slice_b], post_fn=_raise)
    # complete map over every judged pending, overlap conflict resolved.
    assert set(out) == {3, 5, 7}
    assert out[5] == 4  # slice A interior wins (min-dist 1 vs 0)


def test_chapter_review_llm_refines_but_keeps_map_complete(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW", "1")
    slice_a = {"heading_ids": [3, 5], "verdicts": {3: 2, 5: 4}}
    slice_b = {"heading_ids": [5, 7], "verdicts": {5: 6, 7: 3}}
    entries = [hj.SkeletonEntry(rid, 3, f"T{rid}", 1, True, None)
               for rid in (3, 5, 7)]

    def _review(messages, max_tokens):  # reviewer re-levels only heading 5
        return json.dumps({"levels": {"5": 2}}), "stop"

    out = hj.run_chapter_review([], entries, [slice_a, slice_b], post_fn=_review)
    assert out[5] == 2                 # reviewer verdict wins for named id
    assert set(out) == {3, 5, 7}       # every other pending kept (complete)


# ── 5. Byte-identical default-OFF. ───────────────────────────────────────────
def test_normalize_off_chapter_slices_none_and_windows_legacy(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", raising=False)
    prov, _ = _chapter(1, 3)
    plan_unset = hj.build_heading_skeleton(prov)
    assert plan_unset.chapter_slices is None

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "0")
    plan_off = hj.build_heading_skeleton(prov)
    assert plan_off.chapter_slices is None
    # explicit-falsey and unset produce byte-identical windows (legacy path).
    assert plan_off.windows == plan_unset.windows
    assert plan_off.digest == plan_unset.digest


def test_normalize_off_report_not_normalized(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", raising=False)
    prov, _ = _chapter(1, 2)
    tree, esc = [], []
    rep = hj.run_heading_judge(prov, tree, esc, post_fn=_level_all_post(),
                               use_cache=False, emit_capture=False)
    assert rep["normalized"] is False
    assert "chapter_reviews" not in rep["meta"]


# ── 6. End-to-end integration. ───────────────────────────────────────────────
def test_run_normalized_book_all_pendings_verdicted_review_only_oversized(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "160")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov = _multi_chapter_book()
    tree, esc = [], []
    rep = hj.run_heading_judge(prov, tree, esc, post_fn=_level_all_post(),
                               use_cache=False, emit_capture=False)

    assert rep["normalized"] is True
    # every pending got a verdict; NONE left unjudged; ZERO finish=length.
    assert rep["unjudged"] == 0
    assert rep["meta"].get("length_exhausted") == 0
    assert not any(r.get("heading_level_pending") for r in prov
                   if r.get("region_kind") == "heading")
    # chapter review fired ONLY for the oversized chapter (id 0).
    assert rep["meta"].get("chapter_reviews") == 1
    tele = {c["chapter_id"]: c["split"] for c in rep["meta"]["chapters"]}
    assert tele[0] is True and tele[1] is False and tele[2] is False
    # whole-doc final review still runs once (separate tier).
    assert rep["final_review"] is not None
    assert rep["meta"].get("W") == 160


# ── 7. Redundant whole-doc FINAL REVIEW skip for single-chapter normalized. ──
def test_resolve_final_review_min_chapters(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS",
                       raising=False)
    assert hj.resolve_final_review_min_chapters() == 2
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS", "1")
    assert hj.resolve_final_review_min_chapters() == 1
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS", "garbage")
    assert hj.resolve_final_review_min_chapters() == 2
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS", "0")
    assert hj.resolve_final_review_min_chapters() == 2   # non-positive → default


def _spy_final_review(monkeypatch):
    calls = []
    monkeypatch.setattr(hj, "run_final_review",
                        lambda *a, **k: calls.append(1))
    return calls


def test_final_review_skipped_for_single_chapter_normalized(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS",
                       raising=False)
    calls = _spy_final_review(monkeypatch)
    prov, _ = _chapter(1, 3)             # ONE chapter
    hj.run_heading_judge(prov, [], [], post_fn=_level_all_post(),
                         use_cache=False, emit_capture=False)
    # default MIN_CHAPTERS=2 → a single-chapter normalized plan skips the
    # redundant whole-doc review (the per-chapter reviewer's verdicts stand).
    assert calls == []


def test_final_review_runs_for_multi_chapter_normalized(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS",
                       raising=False)
    calls = _spy_final_review(monkeypatch)
    prov = _multi_chapter_book()          # 3 distinct chapters
    hj.run_heading_judge(prov, [], [], post_fn=_level_all_post(),
                         use_cache=False, emit_capture=False)
    assert calls == [1]                   # >= 2 chapters → review runs


def test_final_review_min_chapters_1_forces_single_chapter(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS", "1")
    calls = _spy_final_review(monkeypatch)
    prov, _ = _chapter(1, 3)
    hj.run_heading_judge(prov, [], [], post_fn=_level_all_post(),
                         use_cache=False, emit_capture=False)
    assert calls == [1]                   # MIN_CHAPTERS=1 → runs even single-chapter


def test_final_review_skip_never_touches_non_normalized_path(monkeypatch):
    # NORMALIZE off → chapter_slices is None → the skip logic never engages, so
    # the non-normalized final-review path is byte-identical (still invoked).
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", raising=False)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS", "99")
    calls = _spy_final_review(monkeypatch)
    prov, _ = _chapter(1, 3)
    hj.run_heading_judge(prov, [], [], post_fn=_level_all_post(),
                         use_cache=False, emit_capture=False)
    assert calls == [1]                   # non-normalized → always invoked
