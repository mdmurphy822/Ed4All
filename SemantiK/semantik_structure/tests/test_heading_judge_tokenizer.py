"""Real-tokenizer token-counting tests for the GLM-OCR heading judge
(``SEMANTIK_HEADING_JUDGE_TOKENIZER``) — CPU-only, deterministic, the real HF
tokenizer MOCKED (no model load / no GPU / no network).

Covers: the ``off`` mode byte-identical char heuristics (chars//4 sizer,
chars//3 guard); the ``auto`` mode consuming a MOCKED tokenizer's real count in
BOTH the sizer and the ctx guard; the conservative ceil(len/3) fallback that
logs ONCE and never retries the failed load; and the O(n) (not O(n^2))
tokenization performance guard on a ~250-region normalized plan (locks in the
per-region memoization).
"""

from __future__ import annotations

import math

import pytest

from semantik_structure.glmocr import heading_judge as hj


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Reset the process-singleton tokenizer + seat-context caches around every
    test and pin the seat-context off (legacy budgets). Each test picks its own
    tokenizer mode."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    # Thinking-ON so the (now thinking-aware) budget resolvers keep their legacy
    # values; these tests exercise token COUNTING, not the completion budgets.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "1")
    hj._reset_seat_context_cache()
    hj._reset_tokenizer_cache()
    yield
    hj._reset_seat_context_cache()
    hj._reset_tokenizer_cache()


class _FakeTokenizer:
    """A stand-in HF tokenizer: one token per character (so a count is exactly
    ``len(text)``), with an encode-call counter for the perf guard."""

    def __init__(self):
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return list(text)  # len == len(text)


# ── off mode: byte-identical char heuristics. ────────────────────────────────
def test_off_mode_count_tokens_is_chars_over_4(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "off")
    t = "y" * 40
    assert hj._count_tokens(t) == len(t) // 4      # 10
    # the sizer routes through _count_tokens → byte-identical legacy
    assert hj._estimate_tokens(t) == len(t) // 4


def test_off_mode_prompt_guard_is_chars_over_3(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "off")
    msgs = [{"role": "system", "content": "x" * 30},
            {"role": "user", "content": "y" * 60}]
    # legacy guard: sum(len)//3 == len(concat)//3
    assert hj._estimate_prompt_tokens(msgs) == (30 + 60) // 3


def test_off_mode_never_loads_a_tokenizer(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "off")

    def _boom(_tok_id):
        raise AssertionError("off mode must never load the real tokenizer")

    monkeypatch.setattr(hj, "_load_tokenizer", _boom)
    assert hj._count_tokens("hello world") == len("hello world") // 4


# ── auto mode: the MOCKED real tokenizer drives the counts. ──────────────────
def test_auto_mode_uses_the_real_tokenizer(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "auto")
    monkeypatch.setattr(hj, "_load_tokenizer", lambda _tid: fake)
    hj._reset_tokenizer_cache()

    t = "hello"
    assert hj._count_tokens(t) == len(t)        # fake = one token / char
    assert hj._estimate_tokens(t) == len(t)     # sizer consumes the real count


def test_auto_mode_prompt_guard_adds_role_overhead(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "auto")
    monkeypatch.setattr(hj, "_load_tokenizer", lambda _tid: fake)
    hj._reset_tokenizer_cache()

    msgs = [{"role": "system", "content": "ab"},
            {"role": "user", "content": "cd"}]
    # real count of the concatenated text + ~4 tokens/message framing overhead
    assert hj._estimate_prompt_tokens(msgs) == 4 + hj._PROMPT_ROLE_OVERHEAD_TOKENS * 2


def test_explicit_id_mode_uses_that_tokenizer(monkeypatch):
    fake = _FakeTokenizer()
    seen = []

    def _load(tok_id):
        seen.append(tok_id)
        return fake

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "my/custom-model")
    monkeypatch.setattr(hj, "_load_tokenizer", _load)
    hj._reset_tokenizer_cache()

    assert hj.resolve_tokenizer_mode() == "my/custom-model"
    assert hj._count_tokens("abcd") == 4
    assert seen == ["my/custom-model"]          # the explicit id was loaded


def test_tokenizer_singleton_loads_once(monkeypatch):
    loads = []
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "auto")
    monkeypatch.setattr(hj, "_load_tokenizer",
                        lambda tid: loads.append(tid) or _FakeTokenizer())
    hj._reset_tokenizer_cache()

    hj._count_tokens("a")
    hj._count_tokens("bb")
    hj._count_tokens("ccc")
    assert len(loads) == 1                       # loaded ONCE, then cached


# ── fallback: a failed load → conservative ceil(len/3), logged once, no retry.
def test_fallback_ceil_len_over_3(monkeypatch, caplog):
    load_calls = []

    def _raises(tok_id):
        load_calls.append(tok_id)
        raise RuntimeError("offline miss")

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "auto")
    monkeypatch.setattr(hj, "_load_tokenizer", _raises)
    hj._reset_tokenizer_cache()

    t = "y" * 40
    with caplog.at_level("WARNING"):
        first = hj._count_tokens(t)
        second = hj._count_tokens(t)

    assert first == math.ceil(len(t) / 3.0)      # 14 (conservative, OVER chars//4=10)
    assert second == first
    # NEVER the chars//4 legacy in the fallback path.
    assert first != len(t) // 4
    # loaded ONCE (the failed load is cached; no retry).
    assert len(load_calls) == 1
    # logged ONCE.
    warnings = [r for r in caplog.records
                if "tokenizer" in r.getMessage() and "unavailable" in r.getMessage()]
    assert len(warnings) == 1


# ── performance: O(n), not O(n^2), tokenizer calls on a normalized plan. ─────
def _big_chapter(n_units, content_len=120):
    """A single chapter of ``n_units`` heading/paragraph pairs (~2*n_units+1
    regions) — forced to SPLIT under a small fixed W."""
    regions = [{"region_kind": "heading", "heading_text": "Chapter 1", "level": 1,
                "first_raw_block_index": 0, "source_page": 1}]
    idx = 1
    for u in range(n_units):
        regions.append({"region_kind": "heading", "heading_text": f"Topic {u}",
                        "level": 3, "first_raw_block_index": idx, "source_page": 1,
                        "heading_level_pending": True})
        idx += 1
        regions.append({"region_kind": "paragraph", "raw_text": "y" * content_len,
                        "first_raw_block_index": idx, "source_page": 1})
        idx += 1
    return regions


def test_normalized_plan_tokenizes_linearly(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "auto")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW", "2000")
    monkeypatch.setattr(hj, "_load_tokenizer", lambda _tid: fake)
    hj._reset_tokenizer_cache()

    regions = _big_chapter(125)          # 251 regions
    n_regions = len(regions)
    assert n_regions >= 250

    plan = hj.build_heading_skeleton(regions)
    assert plan.chapter_slices is not None
    assert len(plan.windows) >= 2        # genuinely split (the sizer ran)

    # O(n): each region tokenized ~once (memoized), NOT re-tokenized per growing
    # window step. An O(n^2) slicer would fire thousands of encode calls here
    # (~n * avg-window); the memoized path is a small multiple of the region
    # count.
    assert fake.encode_calls <= 3 * n_regions
    assert fake.encode_calls < n_regions * n_regions // 4
