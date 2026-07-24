"""Deterministic offline GLM-OCR heading-judge book-profiler / seat-tier
right-sizer tests (NO GPU / no seat / no network / no LLM).

Covers: profile_book stats + forced-unbounded budget (no pre-clip) + offline
determinism + ZERO os.environ leakage; resolve_seat_tiers default table /
malformed-token skip / KV-frontier seqs re-derivation; select_seat_tier tier
bands + overflow + the digest-budget consistency invariant; and bucket_profiles
grouping / ordering / empty-bucket omission.

All fixtures are SYNTHETIC hand-authored ``glmocr_layout`` dicts — no real book
or corpus file is ever referenced.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from semantik_structure.glmocr import heading_judge as hj
from semantik_structure.glmocr import seat_profile as sp
from semantik_structure.glmocr.seat_profile import (
    BookProfile,
    SeatTier,
    bucket_profiles,
    profile_book,
    resolve_seat_tiers,
    select_seat_tier,
)


@pytest.fixture(autouse=True)
def _legacy_tokenizer(monkeypatch):
    """Pin the LEGACY char-heuristic token counter
    (``SEMANTIK_HEADING_JUDGE_TOKENIZER=off``) so the deterministic offline
    profiler never depends on whether the real model tokenizer is present on the
    host (these tests assume the chars//4 sizing). Cleared per test."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "off")
    hj._reset_tokenizer_cache()
    yield
    hj._reset_tokenizer_cache()


# ── Synthetic layout fixtures. ───────────────────────────────────────────────
def _region(idx, native_label, content, page=1):
    return {
        "index": idx,
        "native_label": native_label,
        "content": content,
        "bbox_2d": [0, idx * 20, 400, idx * 20 + 18],
    }


def _write_layout(tmp_path: Path, pages, stem="synthbook") -> Path:
    """Serialize a synthetic ``{stem}.glmocr_layout.json`` sidecar."""
    path = tmp_path / f"{stem}.glmocr_layout.json"
    path.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return path


def _two_chapter_pages(body_reps=3):
    """Two chapter openers (fixed L1), each with pending (non-numbered) section
    titles → two chapter-mode judge windows with pendings."""
    body = "Place value describes the value of a digit by position. " * body_reps
    body2 = "To add fractions you first find a common denominator. " * body_reps
    return [
        {"page_no": 1, "regions": [
            _region(0, "paragraph_title", "Chapter 1: Whole Numbers"),
            _region(1, "paragraph_title", "Understanding Place Value"),
            _region(2, "text", body),
            _region(3, "paragraph_title", "Rounding Numbers"),
            _region(4, "text", "Rounding approximates a number to a place. " * body_reps),
        ]},
        {"page_no": 2, "regions": [
            _region(5, "paragraph_title", "Chapter 2: Fractions", page=2),
            _region(6, "paragraph_title", "Adding Fractions", page=2),
            _region(7, "text", body2, page=2),
        ]},
    ]


def _big_chapter_pages(body_reps):
    """One chapter opener + one pending section whose body is deliberately huge
    (drives the largest-window measurement)."""
    huge = "The distributive property lets you multiply a sum term by term. " * body_reps
    return [
        {"page_no": 1, "regions": [
            _region(0, "paragraph_title", "Chapter 1: Algebra"),
            _region(1, "paragraph_title", "The Distributive Property"),
            _region(2, "text", huge),
        ]},
    ]


# ── profile_book. ────────────────────────────────────────────────────────────
def test_profile_book_stats_and_shape(tmp_path):
    path = _write_layout(tmp_path, _two_chapter_pages())
    prof = profile_book(path)

    assert isinstance(prof, BookProfile)
    assert prof.n_chapters == 2
    assert prof.n_pending == 3  # 2 pending titles in ch1 + 1 in ch2
    assert len(prof.window_tokens) == prof.n_chapters
    assert all(t > 0 for t in prof.window_tokens)
    assert prof.max_window_tokens == max(prof.window_tokens)
    assert prof.total_window_tokens == sum(prof.window_tokens)
    # largest_window_index points at the max entry.
    assert prof.window_tokens[prof.largest_window_index] == prof.max_window_tokens
    # deterministic nearest-rank percentiles.
    assert prof.p50_window_tokens == sp._percentile(prof.window_tokens, 50)
    assert prof.p95_window_tokens == sp._percentile(prof.window_tokens, 95)
    assert prof.max_window_tokens >= prof.p95_window_tokens >= prof.p50_window_tokens
    # source is the opaque basename, never book content.
    assert prof.source == path.name


def test_profile_book_forced_unbounded_budget_not_clipped(tmp_path, monkeypatch):
    """A deliberately large chapter is measured at its TRUE full size — NOT the
    truncated size a default/small digest budget would produce."""
    path = _write_layout(tmp_path, _big_chapter_pages(body_reps=4000))
    prof = profile_book(path)

    # Now emulate what the judge would feed under a SMALL digest budget: the
    # chapter content is truncated by _chapter_content_budgeted, so the window
    # is strictly SMALLER. profile_book must have captured the full size.
    from semantik_structure.glmocr.transform import transform_document
    from semantik_structure.glmocr.heading_judge_standalone import _load_layout_pages
    import copy

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET", "500")
    hj._reset_seat_context_cache()
    prov = copy.deepcopy(transform_document(_load_layout_pages(path)).region_provenance)
    clipped_plan = hj.build_heading_skeleton(prov)
    clipped_tokens = max(
        hj._estimate_tokens(wd) for wd, _ in clipped_plan.windows)

    assert prof.max_window_tokens > clipped_tokens, (
        "profiler must capture the FULL un-truncated window size")


def test_profile_book_offline_determinism(tmp_path):
    path = _write_layout(tmp_path, _two_chapter_pages())
    a = profile_book(path)
    b = profile_book(path)
    assert a == b


def test_profile_book_no_env_leakage(tmp_path):
    path = _write_layout(tmp_path, _two_chapter_pages())
    before = dict(os.environ)
    profile_book(path)
    after = dict(os.environ)
    assert after == before
    # explicitly assert the overridden keys are untouched (incl. unset→unset).
    for key in (
        "SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET",
        "SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT",
        "SEMANTIK_HEADING_JUDGE_CHAPTER_MODE",
        "SEMANTIK_HEADING_JUDGE_DOC_SCHEMA",
    ):
        assert (key in after) == (key in before)
        assert after.get(key) == before.get(key)


# ── resolve_seat_tiers. ──────────────────────────────────────────────────────
def test_resolve_seat_tiers_default(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_TIERS, raising=False)
    monkeypatch.delenv(sp.ENV_SEAT_KV_FRONTIER, raising=False)
    tiers = resolve_seat_tiers()
    assert [t.name for t in tiers] == ["super-32k", "super-128k", "super-250k"]
    assert [t.context for t in tiers] == [32768, 131072, 262144]
    assert [t.seqs for t in tiers] == [32, 8, 4]
    # sorted by context ascending.
    assert [t.context for t in tiers] == sorted(t.context for t in tiers)


def test_resolve_seat_tiers_malformed_token_skipped(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_KV_FRONTIER, raising=False)
    monkeypatch.setenv(
        sp.ENV_SEAT_TIERS,
        "tiny:16384:48, GARBAGE, bad:notanint:4, huge:262144:4")
    tiers = resolve_seat_tiers()
    assert [t.name for t in tiers] == ["tiny", "huge"]
    assert [(t.context, t.seqs) for t in tiers] == [(16384, 48), (262144, 4)]


def test_resolve_seat_tiers_all_malformed_falls_back_to_default(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_KV_FRONTIER, raising=False)
    monkeypatch.setenv(sp.ENV_SEAT_TIERS, "GARBAGE, also:bad")
    tiers = resolve_seat_tiers()
    assert [t.name for t in tiers] == ["super-32k", "super-128k", "super-250k"]


def test_resolve_seat_tiers_kv_frontier_derives_seqs(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_TIERS, raising=False)
    monkeypatch.delenv(sp.ENV_SEAT_MAX_SEQS, raising=False)
    monkeypatch.setenv(sp.ENV_SEAT_KV_FRONTIER, "1048576")
    tiers = {t.name: t for t in resolve_seat_tiers()}
    assert tiers["super-32k"].seqs == 32   # 1048576 // 32768 = 32
    assert tiers["super-128k"].seqs == 8    # 1048576 // 131072 = 8
    assert tiers["super-250k"].seqs == 4    # 1048576 // 262144 = 4


def test_resolve_seat_tiers_kv_frontier_capped_by_max_seqs(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_TIERS, raising=False)
    monkeypatch.setenv(sp.ENV_SEAT_MAX_SEQS, "16")
    monkeypatch.setenv(sp.ENV_SEAT_KV_FRONTIER, "1048576")
    tiers = {t.name: t for t in resolve_seat_tiers()}
    # 32k would derive 32 but is capped at max_seqs=16.
    assert tiers["super-32k"].seqs == 16
    assert tiers["super-250k"].seqs == 4


def test_resolve_seat_tiers_kv_frontier_floored_at_one(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_TIERS, raising=False)
    monkeypatch.delenv(sp.ENV_SEAT_MAX_SEQS, raising=False)
    monkeypatch.setenv(sp.ENV_SEAT_KV_FRONTIER, "100000")  # < 262144
    tiers = {t.name: t for t in resolve_seat_tiers()}
    assert tiers["super-250k"].seqs == 1  # max(1, 100000 // 262144) == 1


# ── select_seat_tier. ────────────────────────────────────────────────────────
def _profile_with_max(max_tok):
    return BookProfile(
        n_chapters=1, n_pending=1, window_tokens=[max_tok],
        max_window_tokens=max_tok, p50_window_tokens=max_tok,
        p95_window_tokens=max_tok, total_window_tokens=max_tok,
        largest_window_index=0, source="synth")


def _default_tiers():
    return [SeatTier("super-32k", 32768, 32),
            SeatTier("super-128k", 131072, 8),
            SeatTier("super-250k", 262144, 4)]


def _invariant(tier, max_tok, safety):
    """The consistency invariant: the picked tier's judge digest budget holds
    the largest chapter × safety."""
    return hj._derive_budgets(tier.context)[0] >= math.ceil(max_tok * safety)


def test_select_seat_tier_small_book_picks_smallest(monkeypatch):
    # 32k digest budget is 8601; need = ceil(5000*1.3)=6500 <= 8601.
    tier, overflow = select_seat_tier(
        _profile_with_max(5000), _default_tiers(), safety=1.3)
    assert tier.name == "super-32k"
    assert overflow is False
    assert _invariant(tier, 5000, 1.3)


def test_select_seat_tier_medium_book_picks_middle(monkeypatch):
    # need = ceil(20000*1.3)=26000; 32k(8601) too small, 128k(38092) holds.
    tier, overflow = select_seat_tier(
        _profile_with_max(20000), _default_tiers(), safety=1.3)
    assert tier.name == "super-128k"
    assert overflow is False
    assert _invariant(tier, 20000, 1.3)


def test_select_seat_tier_large_book_picks_biggest_no_overflow(monkeypatch):
    # need = ceil(45000*1.3)=58500; only 250k(77414) holds.
    tier, overflow = select_seat_tier(
        _profile_with_max(45000), _default_tiers(), safety=1.3)
    assert tier.name == "super-250k"
    assert overflow is False
    assert _invariant(tier, 45000, 1.3)


def test_select_seat_tier_monster_chapter_overflows(monkeypatch):
    # need = ceil(100000*1.3)=130000 > 77414 (biggest digest budget).
    tier, overflow = select_seat_tier(
        _profile_with_max(100000), _default_tiers(), safety=1.3)
    assert tier.name == "super-250k"  # biggest, never a silent too-small pick
    assert overflow is True
    # the invariant is DELIBERATELY violated on overflow (that is what the flag
    # signals) — the judge will truncate that one chapter.
    assert not _invariant(tier, 100000, 1.3)


def test_select_seat_tier_arg_safety_beats_env(monkeypatch):
    monkeypatch.setenv(sp.ENV_SEAT_SELECT_SAFETY, "1.3")
    # explicit safety=1.0 shrinks the need: ceil(8000*1.0)=8000 <= 8601 → 32k.
    tier, _ = select_seat_tier(
        _profile_with_max(8000), _default_tiers(), safety=1.0)
    assert tier.name == "super-32k"


def test_select_seat_tier_env_safety_used_when_arg_none(monkeypatch):
    monkeypatch.setenv(sp.ENV_SEAT_SELECT_SAFETY, "2.0")
    # need = ceil(5000*2.0)=10000 > 8601 → not 32k, → 128k.
    tier, _ = select_seat_tier(_profile_with_max(5000), _default_tiers())
    assert tier.name == "super-128k"


# ── bucket_profiles. ─────────────────────────────────────────────────────────
def test_bucket_profiles_groups_and_orders(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_TIERS, raising=False)
    monkeypatch.delenv(sp.ENV_SEAT_KV_FRONTIER, raising=False)
    monkeypatch.setenv(sp.ENV_SEAT_SELECT_SAFETY, "1.3")
    items = [
        ("book_small_a", _profile_with_max(5000)),    # -> super-32k
        ("book_big", _profile_with_max(45000)),       # -> super-250k
        ("book_small_b", _profile_with_max(4000)),    # -> super-32k
        ("book_medium", _profile_with_max(20000)),    # -> super-128k
    ]
    buckets = bucket_profiles(items)
    # ordered smallest -> largest context, empty buckets omitted.
    assert list(buckets.keys()) == ["super-32k", "super-128k", "super-250k"]
    assert buckets["super-32k"] == ["book_small_a", "book_small_b"]
    assert buckets["super-128k"] == ["book_medium"]
    assert buckets["super-250k"] == ["book_big"]


def test_bucket_profiles_omits_empty_buckets(monkeypatch):
    monkeypatch.delenv(sp.ENV_SEAT_TIERS, raising=False)
    monkeypatch.delenv(sp.ENV_SEAT_KV_FRONTIER, raising=False)
    monkeypatch.setenv(sp.ENV_SEAT_SELECT_SAFETY, "1.3")
    # all small → only the 32k bucket exists.
    items = [("a", _profile_with_max(3000)), ("b", _profile_with_max(4000))]
    buckets = bucket_profiles(items)
    assert list(buckets.keys()) == ["super-32k"]
    assert buckets["super-32k"] == ["a", "b"]


def test_bucket_profiles_empty_input(monkeypatch):
    assert bucket_profiles([]) == {}
