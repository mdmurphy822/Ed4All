"""W2 §5.2 — chunk-window budget + join invariants (hermetic, no model).

Asserts the headline budget invariant (no window exceeds num_ctx), the
no-cross-chapter-boundary partition, overlap == 0, the single-over-budget-chunk
1-chunk-window rule, and the sourceId-join / order-fallback ``join_method``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.chunk_window import (  # noqa: E402
    MAX_CHUNK_BODY_CHARS,
    chunks_for_chapter,
    group_chunks_into_windows,
)


def _chunk(cid: str, slug: str, body: str) -> dict:
    return {
        "id": cid,
        "text": body,
        "source": {
            "source_references": [
                {"sourceId": f"dart:{slug}#{cid}", "role": "primary"}
            ]
        },
    }


def _chapter(cid: str, source_file: str, chapter_text: str = "x") -> dict:
    return {
        "id": cid,
        "source_file": source_file,
        "chapter_text": chapter_text,
    }


_SYS_PROMPT = "You are a synthesis author."
_DRAFT = "  (none)"


def test_no_window_exceeds_num_ctx():
    """The headline invariant: no MULTI-chunk window exceeds the budget.

    Bodies are small enough that no SINGLE chunk overflows the budget, so the
    budget genuinely bounds the greedy packing (the single-over-budget-chunk
    exception is its own test).
    """
    num_ctx = 768
    chapter = _chapter("ch1", "chapter-one.html")
    # ~30 words/chunk → a single chunk fits, but 8 of them must split.
    chunks = [
        _chunk(f"c{i}", "chapter-one", "word " * 30) for i in range(8)
    ]
    windows = group_chunks_into_windows(
        chapter,
        chunks,
        num_ctx=num_ctx,
        system_prompt=_SYS_PROMPT,
        draft_block=_DRAFT,
        max_tokens=128,
    )
    assert windows, "expected ≥1 window"
    for w in windows:
        assert w.estimated_prompt_tokens <= num_ctx, (
            f"window {w.window_index} estimate {w.estimated_prompt_tokens} "
            f"exceeds num_ctx {num_ctx}"
        )
    # Multiple chunks at a small num_ctx must split into >1 window.
    assert len(windows) >= 2


def test_windows_never_cross_chapter_boundary():
    """Every window carries the SAME chapter_id (partition by chapter)."""
    chapter = _chapter("ch7", "seven.html")
    chunks = [_chunk(f"c{i}", "seven", "alpha beta gamma " * 30) for i in range(6)]
    windows = group_chunks_into_windows(
        chapter, chunks, num_ctx=400, system_prompt=_SYS_PROMPT,
        draft_block=_DRAFT, max_tokens=64,
    )
    assert {w.chapter_id for w in windows} == {"ch7"}


def test_overlap_is_zero():
    """No chunk_id appears in two windows of the same chapter."""
    chapter = _chapter("ch1", "one.html")
    chunks = [_chunk(f"c{i}", "one", "lorem ipsum dolor " * 25) for i in range(9)]
    windows = group_chunks_into_windows(
        chapter, chunks, num_ctx=380, system_prompt=_SYS_PROMPT,
        draft_block=_DRAFT, max_tokens=64,
    )
    seen: set = set()
    for w in windows:
        for cid in w.chunk_ids:
            assert cid not in seen, f"chunk {cid} appears in two windows"
            seen.add(cid)
    # Every input chunk landed in some window (none lost).
    assert seen == {f"c{i}" for i in range(9)}


def test_single_over_budget_chunk_gets_own_window():
    """A chunk whose truncated body alone exceeds the budget → 1-chunk window."""
    chapter = _chapter("ch1", "one.html")
    big_body = "x" * (MAX_CHUNK_BODY_CHARS + 5000)  # huge; truncated to the cap
    chunks = [
        _chunk("big1", "one", big_body),
        _chunk("big2", "one", big_body),
    ]
    # Tiny num_ctx so even one capped chunk overflows; each must be its own window.
    windows = group_chunks_into_windows(
        chapter, chunks, num_ctx=200, system_prompt=_SYS_PROMPT,
        draft_block=_DRAFT, max_tokens=32,
    )
    assert len(windows) == 2
    for w in windows:
        assert len(w.chunk_ids) == 1
        # Body never head-truncated beyond the per-chunk cap.
        assert len(w.chunks[0]["text"]) <= MAX_CHUNK_BODY_CHARS


def test_chunks_for_chapter_sourceid_join():
    """sourceId-slug join resolves; join_method == 'sourceid'."""
    chapter = _chapter("ch1", "/abs/path/intro-to-algebra.html")
    chunks = [
        _chunk("a1", "intro-to-algebra", "body"),
        _chunk("b1", "other-chapter", "body"),
    ]
    matched, method = chunks_for_chapter(chapter, chunks, all_chapters=[chapter])
    assert method == "sourceid"
    assert [c["id"] for c in matched] == ["a1"]


def test_chunks_for_chapter_order_fallback():
    """No sourceId join (no source_file) → proportional order fallback."""
    ch1 = {"id": "ch1", "chapter_text": "a" * 100}
    ch2 = {"id": "ch2", "chapter_text": "b" * 100}
    chunks = [
        # No source_references → no slug join possible.
        {"id": f"c{i}", "text": "body"} for i in range(4)
    ]
    m1, method1 = chunks_for_chapter(ch1, chunks, all_chapters=[ch1, ch2])
    m2, method2 = chunks_for_chapter(ch2, chunks, all_chapters=[ch1, ch2])
    assert method1 == "order_fallback"
    assert method2 == "order_fallback"
    # Equal chapter_text lengths → 2 chunks each, disjoint, in order.
    assert [c["id"] for c in m1] == ["c0", "c1"]
    assert [c["id"] for c in m2] == ["c2", "c3"]
