"""Tests for the multi-block BATCHED rewrite primitive (rate-limit defeat).

Exercises the Courseforge-native batch envelope + packer + flag resolvers in
``Courseforge/generators/_rewrite_batch.py`` (mirrors the SemantiK Stage-6
batch shape but with CF_BLOCK keys + per-block rewrite max_tokens budget).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._rewrite_batch import (  # noqa: E402
    CF_BLOCK_CLOSE,
    CF_BLOCK_OPEN,
    DEFAULT_OUTPUT_TOKEN_CAP,
    pack_rewrite_batches,
    parse_rewrite_batch_envelope,
    resolve_rewrite_batch_k,
    resolve_rewrite_batch_mode,
    resolve_rewrite_batch_size,
    resolve_rewrite_batched_concurrency,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(block_id: str, *, max_tokens: Optional[int] = None) -> Block:
    blk = Block(
        block_id=block_id,
        block_type="concept",
        page_id="page1",
        sequence=0,
        content={"key_claims": ["x"]},
    )
    if max_tokens is not None:
        object.__setattr__(blk, "_rewrite_max_tokens", max_tokens)
    return blk


def _envelope(block_id: str, body: str) -> str:
    return (
        CF_BLOCK_OPEN.format(block_id=block_id)
        + "\n" + body + "\n"
        + CF_BLOCK_CLOSE.format(block_id=block_id)
    )


# ---------------------------------------------------------------------------
# Envelope round-trip
# ---------------------------------------------------------------------------


def test_envelope_round_trip_two_blocks():
    ids = ["page1#concept_a_0", "page1#concept_b_1"]
    resp = "\n".join(
        [_envelope(ids[0], "<p>Alpha</p>"), _envelope(ids[1], "<p>Beta</p>")]
    )
    out = parse_rewrite_batch_envelope(resp, ids)
    assert out[ids[0]] == "<p>Alpha</p>"
    assert out[ids[1]] == "<p>Beta</p>"


def test_envelope_id_with_hash_and_underscore_round_trips():
    bid = "week_03#assessment_item_pemdas_2"
    out = parse_rewrite_batch_envelope(_envelope(bid, "<div>Q</div>"), [bid])
    assert out[bid] == "<div>Q</div>"


def test_dotall_multiline_body_preserved():
    bid = "p#c_0"
    body = "<section>\n  <h2>T</h2>\n  <p>line1</p>\n  <p>line2</p>\n</section>"
    out = parse_rewrite_batch_envelope(_envelope(bid, body), [bid])
    assert out[bid] == body.strip()


# ---------------------------------------------------------------------------
# Missing-slot → None fail-soft sentinel
# ---------------------------------------------------------------------------


def test_missing_block_slot_maps_to_none():
    ids = ["a", "b", "c"]
    # Only emit a + c; b is absent.
    resp = "\n".join([_envelope("a", "<p>A</p>"), _envelope("c", "<p>C</p>")])
    out = parse_rewrite_batch_envelope(resp, ids)
    assert out["a"] == "<p>A</p>"
    assert out["b"] is None
    assert out["c"] == "<p>C</p>"
    # Every requested id is present in the result.
    assert set(out.keys()) == set(ids)


def test_mismatched_end_tag_yields_none():
    # OPEN id "a" but END id "b" → no backreference match for "a".
    resp = (
        CF_BLOCK_OPEN.format(block_id="a")
        + "\n<p>orphan</p>\n"
        + CF_BLOCK_CLOSE.format(block_id="b")
    )
    out = parse_rewrite_batch_envelope(resp, ["a"])
    assert out["a"] is None


def test_empty_response_all_none():
    out = parse_rewrite_batch_envelope("", ["a", "b"])
    assert out == {"a": None, "b": None}


# ---------------------------------------------------------------------------
# Fenced response tolerated
# ---------------------------------------------------------------------------


def test_fenced_response_is_stripped():
    bid = "p#c_0"
    inner = _envelope(bid, "<p>X</p>")
    fenced = "```html\n" + inner + "\n```"
    out = parse_rewrite_batch_envelope(fenced, [bid])
    assert out[bid] == "<p>X</p>"


def test_bare_fence_tolerated():
    bid = "p#c_0"
    fenced = "```\n" + _envelope(bid, "<p>Y</p>") + "\n```"
    out = parse_rewrite_batch_envelope(fenced, [bid])
    assert out[bid] == "<p>Y</p>"


# ---------------------------------------------------------------------------
# Packer: count + token-cap split + oversized single
# ---------------------------------------------------------------------------


def test_packer_splits_on_batch_size():
    blocks = [_block(f"b{i}") for i in range(5)]
    batches = pack_rewrite_batches(blocks, batch_size=2)
    assert [len(b) for b in batches] == [2, 2, 1]
    # Order preserved.
    flat = [b.block_id for batch in batches for b in batch]
    assert flat == [f"b{i}" for i in range(5)]


def test_packer_splits_on_token_cap():
    # 3 blocks each budgeting 5000 tokens; cap 12000 → batches of [2, 1]
    # even though batch_size=10 would otherwise keep them together.
    blocks = [_block(f"b{i}", max_tokens=5000) for i in range(3)]
    batches = pack_rewrite_batches(blocks, batch_size=10, output_token_cap=12000)
    assert [len(b) for b in batches] == [2, 1]


def test_packer_oversized_single_gets_own_batch():
    # A single block whose own budget exceeds the cap still gets its own batch.
    blocks = [
        _block("small", max_tokens=1000),
        _block("huge", max_tokens=99999),
        _block("after", max_tokens=1000),
    ]
    batches = pack_rewrite_batches(blocks, batch_size=10, output_token_cap=4000)
    # small alone (next would overflow), huge alone, after alone.
    assert [[b.block_id for b in batch] for batch in batches] == [
        ["small"], ["huge"], ["after"],
    ]


def test_packer_default_token_cap_constant():
    assert DEFAULT_OUTPUT_TOKEN_CAP == 16384 - 512


def test_packer_block_without_max_tokens_uses_default():
    # No stamped budget → 4096 default. cap 5000 → only 1 fits per batch.
    blocks = [_block(f"b{i}") for i in range(2)]
    batches = pack_rewrite_batches(blocks, batch_size=10, output_token_cap=5000)
    assert [len(b) for b in batches] == [1, 1]


# ---------------------------------------------------------------------------
# Flag resolvers — parse-with-fallback
# ---------------------------------------------------------------------------


def test_batch_mode_default_on(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_REWRITE_BATCH", raising=False)
    assert resolve_rewrite_batch_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", "False"])
def test_batch_mode_explicit_falsey_off(monkeypatch, val):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", val)
    assert resolve_rewrite_batch_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "on", "garbage", ""])
def test_batch_mode_truthy_or_garbage_on(monkeypatch, val):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", val)
    assert resolve_rewrite_batch_mode() is True


def test_batch_size_default(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_REWRITE_BATCH_SIZE", raising=False)
    assert resolve_rewrite_batch_size() == 3


@pytest.mark.parametrize("val,expected", [("5", 5), ("1", 1), ("12", 12)])
def test_batch_size_pinned(monkeypatch, val, expected):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH_SIZE", val)
    assert resolve_rewrite_batch_size() == expected


@pytest.mark.parametrize("val", ["0", "-2", "garbage", ""])
def test_batch_size_garbage_falls_back(monkeypatch, val):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH_SIZE", val)
    assert resolve_rewrite_batch_size() == 3


def test_batch_k_default_clamps_to_one(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_REWRITE_BATCH_K", raising=False)
    assert resolve_rewrite_batch_k(4) == 1
    assert resolve_rewrite_batch_k(1) == 1


def test_batch_k_pinned_never_exceeds_requested(monkeypatch):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH_K", "3")
    assert resolve_rewrite_batch_k(5) == 3
    # Caller asked for 2 → cap of 3 cannot raise it.
    assert resolve_rewrite_batch_k(2) == 2


@pytest.mark.parametrize("val", ["0", "-1", "garbage"])
def test_batch_k_garbage_falls_back_to_one(monkeypatch, val):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH_K", val)
    assert resolve_rewrite_batch_k(4) == 1


def test_batched_concurrency_default(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_REWRITE_CONCURRENCY", raising=False)
    assert resolve_rewrite_batched_concurrency() == 2


def test_batched_concurrency_explicit_override(monkeypatch):
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "6")
    assert resolve_rewrite_batched_concurrency() == 6


@pytest.mark.parametrize("val", ["0", "garbage", "-3"])
def test_batched_concurrency_garbage_falls_back(monkeypatch, val):
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", val)
    assert resolve_rewrite_batched_concurrency() == 2
