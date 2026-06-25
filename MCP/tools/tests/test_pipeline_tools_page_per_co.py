"""Page-per-CO content-emit wiring regressions.

Covers the three review-fixes folded into the page-per-CO driver
(``plans/finegrain/page-per-co-token-aware-emit-2026-06.md``):

* Review fix 2 — ``_chapter_label_to_week_num`` returns an UNBOUNDED week int,
  so a "Chapter 12" group on a ``--weeks 8`` run lands in a never-iterated week
  and its COs silently vanish. The page-per-CO path CLAMPS every CO group's
  week into ``[1, duration_weeks]`` (``_clamp_week``).
* Review fix 2 — the two per-CO index builders must be PREDICATE-IDENTICAL
  (one skipped id-less objectives, the other did not → positional misalignment).
  On the page-per-CO path (``require_id=True``) both skip id-less objectives.
* OFF byte-stability — with the default (legacy) signature, both builders are
  byte-identical to before (no clamp, chunk index keeps id-less objectives).

Pure-helper unit tests — no LLM client, no router, no filesystem.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools as _pt  # noqa: E402


def _groups_chapters_gt_weeks():
    """Two groups: Chapter 1 (week 1) + Chapter 12 (overflows --weeks 8)."""
    return [
        {
            "chapter": "Chapter 1",
            "objectives": [
                {"id": "CO-01", "statement": "co1",
                 "source_refs": [{"chunk_ids": ["c1", "c2"]}]},
                {"id": "CO-02", "statement": "co2",
                 "source_refs": [{"chunk_ids": ["c3"]}]},
            ],
        },
        {
            "chapter": "Chapter 12",
            "objectives": [
                {"id": "CO-12", "statement": "co12",
                 "source_refs": [{"chunk_ids": ["c9"]}]},
            ],
        },
    ]


def _groups_with_idless_obj():
    return [
        {
            "chapter": "Week 1",
            "objectives": [
                {"id": "CO-01", "statement": "co1",
                 "source_refs": [{"chunk_ids": ["c1"]}]},
                {"statement": "id-less",  # no id
                 "source_refs": [{"chunk_ids": ["c2"]}]},
            ],
        },
    ]


# --------------------------------------------------------------------------- #
# _clamp_week
# --------------------------------------------------------------------------- #
def test_clamp_week_identity_when_no_duration():
    assert _pt._clamp_week(12, None) == 12
    assert _pt._clamp_week(12, 0) == 12
    assert _pt._clamp_week(12, -3) == 12


def test_clamp_week_folds_overflow_into_last_week():
    assert _pt._clamp_week(12, 8) == 8
    assert _pt._clamp_week(3, 8) == 3  # in range → unchanged
    assert _pt._clamp_week(0, 8) == 1  # below floor → 1


# --------------------------------------------------------------------------- #
# Review fix 2 — clamp on the page-per-CO path
# --------------------------------------------------------------------------- #
def test_chapters_gt_weeks_clamped_on_path():
    groups = _groups_chapters_gt_weeks()
    ci = _pt._build_week_co_chunk_index(groups, duration_weeks=8, require_id=True)
    oi = _pt._build_week_co_objectives_index(groups, duration_weeks=8)
    # Chapter 12 was clamped into week 8 (an iterated week), NOT week 12.
    assert 12 not in ci and 12 not in oi
    assert set(ci) == {1, 8}
    assert [o["id"] for o in oi.get(8, [])] == ["CO-12"]
    # Every valid CO id lands in SOME iterated week (zero stranded).
    all_obj_ids = {
        o["id"] for w in range(1, 9) for o in oi.get(w, [])
    }
    assert all_obj_ids == {"CO-01", "CO-02", "CO-12"}


def test_off_path_does_not_clamp():
    """Legacy signature: Chapter 12 stays week 12 (byte-identical to before)."""
    groups = _groups_chapters_gt_weeks()
    ci = _pt._build_week_co_chunk_index(groups)
    oi = _pt._build_week_co_objectives_index(groups)
    assert 12 in ci and 12 in oi  # unclamped — the legacy behaviour


# --------------------------------------------------------------------------- #
# Review fix 2 — predicate-identical builders
# --------------------------------------------------------------------------- #
def test_builders_predicate_identical_on_path():
    """require_id=True → both skip id-less objs → per-week lengths align."""
    groups = _groups_with_idless_obj()
    ci = _pt._build_week_co_chunk_index(groups, duration_weeks=8, require_id=True)
    oi = _pt._build_week_co_objectives_index(groups, duration_weeks=8)
    for w in set(ci) | set(oi):
        assert len(ci.get(w, [])) == len(oi.get(w, [])), (
            f"week {w} misaligned: chunk={len(ci.get(w, []))} "
            f"obj={len(oi.get(w, []))}"
        )
    # The id-less obj's chunk (c2) is dropped from the chunk index on the path.
    assert ci.get(1) == [["c1"]]


def test_off_path_chunk_index_keeps_idless():
    """Legacy signature: chunk index keeps the id-less obj's slice."""
    groups = _groups_with_idless_obj()
    ci = _pt._build_week_co_chunk_index(groups)
    oi = _pt._build_week_co_objectives_index(groups)
    # chunk index has BOTH objs; objectives index has only the id-bearing one
    # → the historical positional MISALIGNMENT this fix closes on the path.
    assert ci.get(1) == [["c1"], ["c2"]]
    assert [o["id"] for o in oi.get(1, [])] == ["CO-01"]


# --------------------------------------------------------------------------- #
# _resolve_content_page_count — the never-increase guard (review fix 3)
# --------------------------------------------------------------------------- #
def test_page_count_off_is_topic_driven():
    # OFF → max(topic_count, 1) verbatim regardless of CO count
    assert _pt._resolve_content_page_count(
        topic_count=7, co_count=3, page_per_co=False) == 7
    assert _pt._resolve_content_page_count(
        topic_count=0, co_count=3, page_per_co=False) == 1


def test_page_count_on_is_co_driven():
    # ON, topic-rich week → CO count drives (fewer pages)
    assert _pt._resolve_content_page_count(
        topic_count=7, co_count=3, page_per_co=True) == 3


def test_page_count_never_increase_guard():
    # ON, CO-rich / topic-thin week → capped at topic_count (never increases)
    assert _pt._resolve_content_page_count(
        topic_count=2, co_count=9, page_per_co=True) == 2


def test_page_count_topicless_week_keeps_co_count():
    # ON, topic-less week → not capped to zero; falls through to CO count
    assert _pt._resolve_content_page_count(
        topic_count=0, co_count=4, page_per_co=True) == 4
    # no COs either → still emits ≥1 content page
    assert _pt._resolve_content_page_count(
        topic_count=0, co_count=0, page_per_co=True) == 1


# --------------------------------------------------------------------------- #
# Flag gate (master flag AND two-pass)
# --------------------------------------------------------------------------- #
def test_content_page_per_co_gate_requires_two_pass(monkeypatch):
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO", "1")
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)
    assert _pt._content_page_per_co_enabled() is False
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    assert _pt._content_page_per_co_enabled() is True
    monkeypatch.setenv("ED4ALL_CONTENT_PAGE_PER_CO", "0")
    assert _pt._content_page_per_co_enabled() is False
