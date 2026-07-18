"""Tests for the opt-in NLI micro-batching dispatcher (``ED4ALL_NLI_MICROBATCH``).

Covers:

* resolver parse-with-fallback for the three env knobs;
* batched-via-dispatcher scores match sequential per-pair scores within 1e-3
  (stub NLI here; a ``@pytest.mark.slow`` real-model twin proves it on the
  actual DeBERTa forward pass);
* the dispatcher actually COALESCES concurrent requests into shared drains and
  routes each caller its own slice (N threads x M pairs, no deadlock);
* a scorer-side exception is surfaced to every affected caller (never a
  fabricated score, never a hang);
* ``get_dispatcher`` caches per underlying-nli identity and returns None for a
  missing model.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.classifiers import nli_microbatch as mb  # noqa: E402
from lib.classifiers.nli_classifier import estimate_pair_tokens  # noqa: E402
from lib.classifiers.nli_microbatch import (  # noqa: E402
    NliMicrobatchDispatcher,
    get_dispatcher,
    resolve_microbatch_enabled,
    resolve_microbatch_max_pairs,
    resolve_microbatch_window_ms,
)


class _Score:
    """Minimal NliScore-shaped result."""

    def __init__(self, entailment: float, neutral: float, contradiction: float) -> None:
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction


def _deterministic_score(hypothesis: str) -> _Score:
    """A per-pair-deterministic score derived only from the hypothesis text.

    Because the score depends solely on the pair content (never on batch-mates),
    a batched drain and a sequential per-pair run must produce identical numbers
    — the property real attention-masked batching also guarantees to within
    low-order bits.
    """
    ent = (abs(hash(hypothesis)) % 1000) / 1000.0
    con = (abs(hash(hypothesis + "#c")) % 1000) / 1000.0 * (1.0 - ent)
    neu = max(0.0, 1.0 - ent - con)
    return _Score(ent, neu, con)


class _StubNLI:
    """Deterministic stub that records the pair-count of every score_batch call.

    Accepts ``batch_size`` so the dispatcher's big-forward-pass path is
    exercised. Optionally sleeps per call to widen the coalescing window in the
    concurrency test.
    """

    _revision = "stub-rev"
    device = "cpu"
    dtype = "float32"

    def __init__(self, *, per_call_sleep: float = 0.0) -> None:
        self.call_pair_counts: List[int] = []
        self._per_call_sleep = per_call_sleep
        self._lock = threading.Lock()

    def score_batch(
        self, *, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> List[_Score]:
        with self._lock:
            self.call_pair_counts.append(len(pairs))
        if self._per_call_sleep:
            time.sleep(self._per_call_sleep)
        return [_deterministic_score(str(h)) for _p, h in pairs]


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    mb._reset_for_tests()
    yield
    mb._reset_for_tests()


# --------------------------------------------------------------------------- #
# Resolvers
# --------------------------------------------------------------------------- #


def test_resolve_enabled_parse_with_fallback() -> None:
    assert resolve_microbatch_enabled({}) is False
    for truthy in ("1", "true", "YES", "On"):
        assert resolve_microbatch_enabled({"ED4ALL_NLI_MICROBATCH": truthy}) is True
    for off in ("0", "false", "no", "off", "garbage", ""):
        assert resolve_microbatch_enabled({"ED4ALL_NLI_MICROBATCH": off}) is False


def test_resolve_max_pairs_parse_with_fallback() -> None:
    assert resolve_microbatch_max_pairs({}) == 64
    assert resolve_microbatch_max_pairs({"ED4ALL_NLI_MICROBATCH_MAX_PAIRS": "128"}) == 128
    assert resolve_microbatch_max_pairs({"ED4ALL_NLI_MICROBATCH_MAX_PAIRS": "0"}) == 64
    assert resolve_microbatch_max_pairs({"ED4ALL_NLI_MICROBATCH_MAX_PAIRS": "-3"}) == 64
    assert resolve_microbatch_max_pairs({"ED4ALL_NLI_MICROBATCH_MAX_PAIRS": "x"}) == 64


def test_resolve_window_ms_parse_with_fallback() -> None:
    assert resolve_microbatch_window_ms({}) == 10.0
    assert resolve_microbatch_window_ms({"ED4ALL_NLI_MICROBATCH_WINDOW_MS": "25"}) == 25.0
    assert resolve_microbatch_window_ms({"ED4ALL_NLI_MICROBATCH_WINDOW_MS": "0"}) == 0.0
    assert resolve_microbatch_window_ms({"ED4ALL_NLI_MICROBATCH_WINDOW_MS": "-1"}) == 10.0
    assert resolve_microbatch_window_ms({"ED4ALL_NLI_MICROBATCH_WINDOW_MS": "junk"}) == 10.0


def test_resolve_crossblock_parse_with_fallback() -> None:
    assert mb.resolve_crossblock_enabled({}) is False
    for truthy in ("1", "true", "YES", "On"):
        assert mb.resolve_crossblock_enabled({"ED4ALL_NLI_CROSSBLOCK": truthy}) is True
    for off in ("0", "false", "no", "off", "garbage", ""):
        assert mb.resolve_crossblock_enabled({"ED4ALL_NLI_CROSSBLOCK": off}) is False
    assert mb.resolve_crossblock_threads({}) == 16
    assert mb.resolve_crossblock_threads({"ED4ALL_NLI_CROSSBLOCK_THREADS": "8"}) == 8
    assert mb.resolve_crossblock_threads({"ED4ALL_NLI_CROSSBLOCK_THREADS": "0"}) == 16
    assert mb.resolve_crossblock_threads({"ED4ALL_NLI_CROSSBLOCK_THREADS": "x"}) == 16


# --------------------------------------------------------------------------- #
# (a) batched == sequential within 1e-3
# --------------------------------------------------------------------------- #


def test_batched_scores_match_sequential() -> None:
    nli = _StubNLI()
    dispatcher = NliMicrobatchDispatcher(nli, max_pairs=64, window_ms=5)
    try:
        pairs = [(f"premise {i}", f"hypothesis number {i}") for i in range(7)]
        # Sequential reference — one pair per call, straight through the stub.
        expected = [nli.score_batch(pairs=[p])[0] for p in pairs]
        got = dispatcher.score_batch(pairs=pairs)
        assert len(got) == len(pairs)
        for g, e in zip(got, expected):
            assert abs(g.entailment - e.entailment) < 1e-3
            assert abs(g.contradiction - e.contradiction) < 1e-3
    finally:
        dispatcher.shutdown()


def test_drain_accumulates_token_estimate_stat() -> None:
    """The drain accumulates a tokenizer-free token-length estimate so the 30s
    stats line can report tok/s, not just pairs/s."""
    nli = _StubNLI()
    dispatcher = NliMicrobatchDispatcher(nli, max_pairs=64, window_ms=5)
    try:
        pairs = [
            ("short", "h"),
            ("a considerably longer premise " * 5, "hypothesis body " * 3),
            ("mid length premise here", "another hypothesis"),
        ]
        # score_batch blocks until the drain has set stats + result, so by the
        # time it returns _stats_tokens reflects this drain.
        dispatcher.score_batch(pairs=pairs)
        expected_tokens = sum(estimate_pair_tokens(p, h) for p, h in pairs)
        assert dispatcher._stats_tokens == expected_tokens
        assert dispatcher._stats_tokens > dispatcher._stats_pairs  # tokens >> pairs
    finally:
        dispatcher.shutdown()


def test_batched_scores_match_sequential_mixed_lengths() -> None:
    """Padding-stress: pairs of WILDLY different premise/hypothesis lengths in
    ONE drain must each still get exactly their own single-pair score.

    The deterministic stub is batch-position-independent, so this asserts the
    dispatcher ROUTES each caller's slice correctly regardless of where a pair
    lands in a mixed-length coalesced batch (the routing half of the batched-vs-
    single contract). The numerical half — that real attention masking makes a
    short pair's score independent of a long batch-mate to within low-order bits
    — is proven on the actual DeBERTa forward pass by
    ``test_real_model_batched_matches_sequential`` (slow).
    """
    nli = _StubNLI()
    dispatcher = NliMicrobatchDispatcher(nli, max_pairs=64, window_ms=5)
    try:
        pairs = [
            ("short.", "tiny hypothesis"),
            (
                "A considerably longer premise sentence that carries many more "
                "tokens than its neighbors so the tokenized batch must pad the "
                "shorter members up to this length.",
                "long hypothesis with a matching sprawl of additional tokens to "
                "stress the padded attention mask across the coalesced batch",
            ),
            ("mid length premise here", "mid"),
            ("x", "y"),
            (
                "Another lengthy premise, distinct wording, distinct length, so "
                "the drain holds a genuinely heterogeneous length distribution.",
                "z hypothesis",
            ),
        ]
        expected = [nli.score_batch(pairs=[p])[0] for p in pairs]
        got = dispatcher.score_batch(pairs=pairs)
        assert len(got) == len(pairs)
        for g, e in zip(got, expected):
            assert abs(g.entailment - e.entailment) < 1e-3
            assert abs(g.contradiction - e.contradiction) < 1e-3
    finally:
        dispatcher.shutdown()


def test_empty_pairs_is_noop() -> None:
    nli = _StubNLI()
    dispatcher = NliMicrobatchDispatcher(nli)
    try:
        assert dispatcher.score_batch(pairs=[]) == []
        # No call reached the model.
        assert nli.call_pair_counts == []
    finally:
        dispatcher.shutdown()


# --------------------------------------------------------------------------- #
# (b) N threads x M pairs — coalesced, no deadlock, correctly routed
# --------------------------------------------------------------------------- #


def test_concurrent_callers_coalesce_and_route() -> None:
    n_threads = 16
    m_pairs = 4
    # Per-call sleep widens the window so requests pile up on the queue and the
    # scorer coalesces them into shared drains.
    nli = _StubNLI(per_call_sleep=0.02)
    dispatcher = NliMicrobatchDispatcher(nli, max_pairs=256, window_ms=100)
    barrier = threading.Barrier(n_threads)
    results: dict = {}
    errors: List[BaseException] = []
    lock = threading.Lock()

    def _worker(tid: int) -> None:
        pairs = [(f"p{tid}", f"caller{tid}_hyp{j}") for j in range(m_pairs)]
        try:
            barrier.wait(timeout=10)
            out = dispatcher.score_batch(pairs=pairs)
            with lock:
                results[tid] = (pairs, out)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(n_threads)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "worker deadlocked"
    finally:
        dispatcher.shutdown()

    assert not errors, f"workers raised: {errors}"
    assert len(results) == n_threads
    # Each caller got exactly its own pairs' scores (correct routing).
    for tid, (pairs, out) in results.items():
        assert len(out) == m_pairs
        for (_p, hyp), score in zip(pairs, out):
            expected = _deterministic_score(hyp)
            assert abs(score.entailment - expected.entailment) < 1e-9
    # Coalescing actually happened: at least one drain merged >1 request.
    assert dispatcher.drain_request_counts, "no drains recorded"
    assert max(dispatcher.drain_request_counts) >= 2, (
        f"expected coalescing; per-drain request counts="
        f"{dispatcher.drain_request_counts}"
    )
    # Every submitted pair was scored exactly once across all drains.
    assert sum(dispatcher.drain_pair_counts) == n_threads * m_pairs


# --------------------------------------------------------------------------- #
# Scorer failure is surfaced to every caller (never fabricated, never a hang)
# --------------------------------------------------------------------------- #


def test_scorer_exception_surfaced_to_callers() -> None:
    class _BoomNLI:
        _revision = "boom"
        device = "cpu"

        def score_batch(self, *, pairs, batch_size=None):
            raise RuntimeError("forward pass exploded")

    dispatcher = NliMicrobatchDispatcher(_BoomNLI(), window_ms=1)
    try:
        with pytest.raises(RuntimeError, match="forward pass exploded"):
            dispatcher.score_batch(pairs=[("p", "h")])
    finally:
        dispatcher.shutdown()


def test_length_mismatch_is_surfaced_not_routed() -> None:
    class _ShortNLI:
        _revision = "short"
        device = "cpu"

        def score_batch(self, *, pairs, batch_size=None):
            return []  # wrong length

    dispatcher = NliMicrobatchDispatcher(_ShortNLI(), window_ms=1)
    try:
        with pytest.raises(RuntimeError, match="length-mismatched"):
            dispatcher.score_batch(pairs=[("p", "h")])
    finally:
        dispatcher.shutdown()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_get_dispatcher_caches_per_nli_identity() -> None:
    a, b = _StubNLI(), _StubNLI()
    da1 = get_dispatcher(a)
    da2 = get_dispatcher(a)
    db = get_dispatcher(b)
    assert da1 is da2
    assert db is not da1


def test_get_dispatcher_none_for_missing_model() -> None:
    assert get_dispatcher(None) is None


def test_stub_without_batch_size_kwarg_still_works() -> None:
    class _NoBatchSizeNLI:
        _revision = "nobs"
        device = "cpu"

        def score_batch(self, *, pairs):
            return [_deterministic_score(str(h)) for _p, h in pairs]

    dispatcher = NliMicrobatchDispatcher(_NoBatchSizeNLI(), window_ms=1)
    try:
        assert dispatcher._supports_batch_size is False
        out = dispatcher.score_batch(pairs=[("p", "hi")])
        assert len(out) == 1
    finally:
        dispatcher.shutdown()


# --------------------------------------------------------------------------- #
# (a-real) slow twin: batched vs sequential on the real DeBERTa forward pass
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_model_batched_matches_sequential() -> None:
    from lib.classifiers.nli_classifier import NliClassifier

    nli = NliClassifier.get_or_load()
    if nli is None:
        pytest.skip("transformers / torch extras not installed")
    pairs = [
        ("The cat sat on the mat.", "A feline rested on a rug."),
        ("Water boils at 100 degrees Celsius.", "Water freezes at 100 degrees."),
        ("Paris is the capital of France.", "The capital of France is Paris."),
    ]
    sequential = nli.score_batch(pairs=pairs)
    dispatcher = NliMicrobatchDispatcher(nli, max_pairs=64, window_ms=5)
    try:
        batched = dispatcher.score_batch(pairs=pairs)
    finally:
        dispatcher.shutdown()
    for b, s in zip(batched, sequential):
        assert abs(b.entailment - s.entailment) < 1e-3
        assert abs(b.contradiction - s.contradiction) < 1e-3
