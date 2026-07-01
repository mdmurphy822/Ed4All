"""W1b.1 — embed-overflow guard unit tests (stub tokenizer, no model load)."""
from __future__ import annotations

from lib.embedding.providers import (
    DEFAULT_MAX_SEQ_TOKENS,
    count_overflow_records,
    estimate_token_count,
    resolve_embed_max_seq_tokens,
    resolve_embed_overflow_guard,
    resolve_embed_overflow_split,
    split_overflow_records,
)

# Stub tokenizer: one token per whitespace word (deterministic, no weights).
_STUB = lambda t: len(t.split())  # noqa: E731


def test_resolvers_default_off_and_parse_with_fallback():
    assert resolve_embed_overflow_guard({}) is False
    assert resolve_embed_overflow_split({}) is False
    assert resolve_embed_overflow_guard({"ED4ALL_EMBED_OVERFLOW_GUARD": "true"}) is True
    assert resolve_embed_overflow_guard({"ED4ALL_EMBED_OVERFLOW_GUARD": "garbage"}) is False
    assert resolve_embed_overflow_split({"ED4ALL_EMBED_OVERFLOW_SPLIT": "on"}) is True

    assert resolve_embed_max_seq_tokens({}) == DEFAULT_MAX_SEQ_TOKENS
    assert resolve_embed_max_seq_tokens({"ED4ALL_EMBED_MAX_SEQ_TOKENS": "128"}) == 128
    assert resolve_embed_max_seq_tokens({"ED4ALL_EMBED_MAX_SEQ_TOKENS": "-4"}) == DEFAULT_MAX_SEQ_TOKENS
    assert resolve_embed_max_seq_tokens({"ED4ALL_EMBED_MAX_SEQ_TOKENS": "x"}) == DEFAULT_MAX_SEQ_TOKENS


def test_estimate_token_count_scales_with_words():
    assert estimate_token_count("") == 0
    assert estimate_token_count("one two three four") >= 4  # >= word count


def test_count_overflow_records_counts_over_window():
    records = [
        {"id": "c1", "text": "a b c"},              # 3 tokens - fits
        {"id": "c2", "text": " ".join(["w"] * 12)},  # 12 tokens - overflow
        {"id": "c3", "text": " ".join(["w"] * 20)},  # 20 tokens - overflow
    ]
    block = count_overflow_records(records, 5, count_tokens=_STUB)
    assert block["records_scanned"] == 3
    assert block["overflow_count"] == 2
    assert block["max_observed_tokens"] == 20
    assert set(block["overflow_chunk_ids"]) == {"c2", "c3"}
    assert 0.66 < block["overflow_rate"] < 0.67


def test_split_overflow_records_parent_resolving_and_anti_fabrication():
    parent_words = [f"w{i}" for i in range(12)]
    records = [
        {"id": "keep", "text": "short one two"},
        {"id": "big", "text": " ".join(parent_words)},
    ]
    out, stats = split_overflow_records(records, 5, count_tokens=_STUB)

    # Non-overflow record passes through unchanged (same object identity).
    assert out[0] is records[0]

    subs = [r for r in out if r.get("parent_chunk_id") == "big"]
    assert stats["overflow_count"] == 1
    assert stats["sub_pieces_created"] == len(subs) >= 2

    # Every sub-piece is within the window and id resolves to the parent.
    for n, sub in enumerate(subs):
        assert sub["id"] == f"big#p{n}"
        assert sub["parent_chunk_id"] == "big"
        assert sub["overflow_split"] is True
        assert _STUB(sub["text"]) <= 5

    # Anti-fabrication: concatenating the sub-piece words reproduces the parent
    # word sequence exactly (strict contiguous slices, nothing synthesized).
    rejoined = " ".join(sub["text"] for sub in subs).split()
    assert rejoined == parent_words


def test_split_no_overflow_is_identity():
    records = [{"id": "a", "text": "one two three"}]
    out, stats = split_overflow_records(records, 5, count_tokens=_STUB)
    assert out == records
    assert stats["overflow_count"] == 0
    assert stats["sub_pieces_created"] == 0
