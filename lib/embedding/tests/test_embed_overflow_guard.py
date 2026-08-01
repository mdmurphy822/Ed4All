"""W1b.1 — embed-overflow guard unit tests (stub tokenizer, no model load).

Zero weights, zero network, zero GPU — the encoder is a stub object with a
``max_seq_length`` attribute. Run CPU-pinned:

    ED4ALL_EMBEDDING_DEVICE=cpu ED4ALL_NLI_DEVICE=cpu \
      pytest lib/embedding/tests/test_embed_overflow_guard.py
"""
from __future__ import annotations

import pytest

from lib.embedding.providers import (
    DEFAULT_MAX_SEQ_TOKENS,
    EmbeddingClient,
    count_overflow_records,
    estimate_token_count,
    resolve_embed_max_seq_pin,
    resolve_embed_max_seq_tokens,
    resolve_embed_overflow_guard,
    resolve_embed_overflow_split,
    split_overflow_records,
)

# Stub tokenizer: one token per whitespace word (deterministic, no weights).
_STUB = lambda t: len(t.split())  # noqa: E731

#: Verified from the on-disk checkpoint config of the product embedding pin
#: ``BAAI/bge-large-en-v1.5`` (``sentence_bert_config.json`` →
#: ``max_seq_length: 512``; ``config.json`` → ``max_position_embeddings: 512``).
_PRODUCT_NATIVE_WINDOW = 512


def test_report_arm_defaults_on_and_parses_with_fallback():
    """The REPORT arm is on by default (W11): over-window truncation is
    invisible unless it is counted, and counting mutates nothing."""
    assert resolve_embed_overflow_guard({}) is True
    assert resolve_embed_overflow_guard({"ED4ALL_EMBED_OVERFLOW_GUARD": "true"}) is True
    # Garbage keeps the documented default rather than silently disabling it.
    assert resolve_embed_overflow_guard({"ED4ALL_EMBED_OVERFLOW_GUARD": "garbage"}) is True
    for off in ("0", "false", "no", "off", "OFF", " Off "):
        assert resolve_embed_overflow_guard({"ED4ALL_EMBED_OVERFLOW_GUARD": off}) is False


def test_split_arm_stays_default_off():
    """Splitting changes chunk identity and the chunkset hash — a separate
    decision from reporting, and still opt-in."""
    assert resolve_embed_overflow_split({}) is False
    assert resolve_embed_overflow_split({"ED4ALL_EMBED_OVERFLOW_SPLIT": "garbage"}) is False
    assert resolve_embed_overflow_split({"ED4ALL_EMBED_OVERFLOW_SPLIT": "on"}) is True


def test_max_seq_tokens_parse_with_fallback():
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


# ---------------------------------------------------------------------------
# Guard/pin decoupling.
#
# The report arm defaults ON and promises it "never changes a vector, and never
# changes embeddings_sha256". It used to also gate the encoder
# ``max_seq_length`` pin, so flipping the guard on silently clamped every model
# with a native window wider than the ceiling. These tests hold the two apart.
# ---------------------------------------------------------------------------
class _StubModel:
    """Minimal stand-in for a loaded SentenceTransformer (no weights, no GPU)."""

    def __init__(self, native):
        self.max_seq_length = native


@pytest.fixture(autouse=True)
def _clean_pin_env(monkeypatch):
    """Neither knob leaks in from the ambient environment."""
    monkeypatch.delenv("ED4ALL_EMBED_MAX_SEQ_PIN", raising=False)
    monkeypatch.delenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", raising=False)
    monkeypatch.delenv("ED4ALL_EMBED_OVERFLOW_GUARD", raising=False)


def test_max_seq_pin_parse_with_fallback():
    """Unset / garbage / non-positive → None ("no pin"), never some clamp.

    The fallback cannot be a number: every clamp value produces a different
    corpus, so guessing one would be exactly the silent re-embed this split
    exists to prevent.
    """
    assert resolve_embed_max_seq_pin({}) is None
    assert resolve_embed_max_seq_pin({"ED4ALL_EMBED_MAX_SEQ_PIN": "256"}) == 256
    for bad in ("0", "-4", "x", "", "512.0"):
        assert resolve_embed_max_seq_pin({"ED4ALL_EMBED_MAX_SEQ_PIN": bad}) is None


def test_report_guard_on_by_default_does_not_touch_the_encoder():
    """The crux: guard ON (default) + pin unset → the window is untouched.

    A long-context encoder is the case that used to be silently clamped to the
    512 accounting ceiling.
    """
    assert resolve_embed_overflow_guard({}) is True  # default ON, as documented
    for native in (_PRODUCT_NATIVE_WINDOW, 8192, 32768):
        model = _StubModel(native)
        EmbeddingClient._pin_max_seq_length(model)
        assert model.max_seq_length == native


def test_accounting_ceiling_never_reaches_the_encoder(monkeypatch):
    """Tightening the REPORT ceiling re-scores the manifest, nothing more.

    ED4ALL_EMBED_MAX_SEQ_TOKENS used to double as the pin value, so lowering it
    to sharpen the overflow report also re-embedded the corpus — including the
    product model, whose 512 native window is otherwise never clamped.
    """
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "1")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_TOKENS", "128")
    assert resolve_embed_max_seq_tokens() == 128  # report arm sees it

    model = _StubModel(_PRODUCT_NATIVE_WINDOW)
    EmbeddingClient._pin_max_seq_length(model)
    assert model.max_seq_length == _PRODUCT_NATIVE_WINDOW  # encoder does not


def test_pin_is_independently_controllable(monkeypatch):
    """The pin fires on its own knob even with the report guard explicitly OFF."""
    monkeypatch.setenv("ED4ALL_EMBED_OVERFLOW_GUARD", "0")
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_PIN", "256")
    assert resolve_embed_overflow_guard() is False

    model = _StubModel(8192)
    EmbeddingClient._pin_max_seq_length(model)
    assert model.max_seq_length == 256


def test_product_model_vectors_unchanged_under_defaults_and_at_512():
    """``BAAI/bge-large-en-v1.5`` emits identical vectors before and after.

    Its native window is 512, so the historical ``min(native, 512)`` clamp was
    already a no-op on the product pin — the defect bit wider-window models.
    Both the new default (no pin) and an explicit 512 pin leave it at 512, so
    this change moves no product vector and no ``embeddings_sha256``.
    """
    default_env = _StubModel(_PRODUCT_NATIVE_WINDOW)
    EmbeddingClient._pin_max_seq_length(default_env)
    assert default_env.max_seq_length == _PRODUCT_NATIVE_WINDOW


def test_pin_at_512_is_a_verified_noop_for_the_product_model(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_PIN", str(_PRODUCT_NATIVE_WINDOW))
    model = _StubModel(_PRODUCT_NATIVE_WINDOW)
    EmbeddingClient._pin_max_seq_length(model)
    assert model.max_seq_length == _PRODUCT_NATIVE_WINDOW


def test_pin_only_lowers_never_raises(monkeypatch):
    """min(native, pin) — a pin wider than the checkpoint window is ignored."""
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_PIN", "4096")
    model = _StubModel(256)
    EmbeddingClient._pin_max_seq_length(model)
    assert model.max_seq_length == 256


def test_pin_below_native_really_does_change_the_product_window(monkeypatch):
    """The knob is real — which is exactly why it must be opt-in.

    A pin under 512 truncates the product encoder earlier and moves its
    vectors. Nothing default-ON may be able to do this.
    """
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_PIN", "128")
    model = _StubModel(_PRODUCT_NATIVE_WINDOW)
    EmbeddingClient._pin_max_seq_length(model)
    assert model.max_seq_length == 128


def test_pin_is_a_noop_on_a_model_without_the_attribute(monkeypatch):
    """Best-effort: an encoder exposing no window is skipped, never crashed."""
    monkeypatch.setenv("ED4ALL_EMBED_MAX_SEQ_PIN", "256")

    class _NoWindow:
        pass

    model = _NoWindow()
    EmbeddingClient._pin_max_seq_length(model)  # must not raise
    assert not hasattr(model, "max_seq_length")
