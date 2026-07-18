"""BUILDER B — token-length-bucketed batching + per-bucket batch sizes.

Covers :meth:`NliClassifier._run_forward`'s bucketing path and the
``ED4ALL_NLI_BUCKET_BATCHING`` / ``ED4ALL_NLI_BUCKET_BATCH`` resolvers.

No GPU / no real model load — a purpose-built recording harness (fake
tokenizer + model + torch) captures the EXACT sequence of forward-pass
batches the classifier issues, so the flag-off path is asserted
byte-identical (same batch slicing) to the historical single-size loop and
the flag-on path is asserted to (a) preserve result order + count and (b)
keep each forward batch inside a single token bucket at its configured size.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.classifiers.nli_classifier import (  # noqa: E402
    ENV_BUCKET_BATCH,
    ENV_BUCKET_BATCHING,
    NliClassifier,
    NliScore,
    _BUCKET_TOKEN_BOUNDS,
    _CHARS_PER_TOKEN,
    _DEFAULT_BUCKET_BATCH,
    _MAX_SEQUENCE_LENGTH,
    estimate_pair_tokens,
    resolve_bucket_batch_sizes,
    resolve_bucket_batching_enabled,
)


# --------------------------------------------------------------------- #
# Env hygiene: the operator box pins these; clear so defaults hold.
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_bucket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BUCKET_BATCHING, raising=False)
    monkeypatch.delenv(ENV_BUCKET_BATCH, raising=False)
    monkeypatch.delenv("ED4ALL_NLI_DEVICE", raising=False)
    monkeypatch.delenv("ED4ALL_NLI_MIN_FREE_VRAM_MIB", raising=False)
    monkeypatch.delenv("ED4ALL_NLI_EVICT_FOR_CUDA", raising=False)
    NliClassifier._reset_for_tests()
    yield
    NliClassifier._reset_for_tests()


# --------------------------------------------------------------------- #
# Recording harness — a fake torch/model/tokenizer that records the exact
# per-forward batch composition and returns per-pair scores derived from
# the premise so result order/identity is checkable.
# --------------------------------------------------------------------- #


def _pair_id(premise: str) -> int:
    """The 4-digit id embedded at the head of a harness premise."""
    return int(premise[:4])


class _FakeRow:
    def __init__(self, premise: str) -> None:
        self._premise = premise

    def __getitem__(self, axis: int) -> Any:
        # entailment axis carries the pair id (unique per pair) so a
        # restored result can be traced back to its input pair; the other
        # axes are 0.0. Values need not be a real softmax for these tests.
        val = float(_pair_id(self._premise)) if axis == 0 else 0.0

        class _Item:
            def __init__(self, v: float) -> None:
                self._v = v

            def item(self) -> float:
                return self._v

        return _Item(val)


class _FakeProbs:
    def __init__(self, premises: Tuple[str, ...]) -> None:
        self._premises = premises
        self.shape = (len(premises), 3)

    def __getitem__(self, i: int) -> _FakeRow:
        return _FakeRow(self._premises[i])


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> bool:
        return False


class _FakeTorch:
    def no_grad(self) -> _NoGrad:
        return _NoGrad()

    def softmax(self, logits: Any, dim: int) -> _FakeProbs:
        # ``logits`` is the (premises, hypotheses) tuple threaded through the
        # fake model; build a probs view whose rows carry each premise.
        premises, _hyps = logits
        return _FakeProbs(premises)


class _FakeOutputs:
    def __init__(self, logits: Any) -> None:
        self.logits = logits


def _make_recording_classifier() -> Tuple[NliClassifier, List[List[str]]]:
    """Build a classifier whose forward passes are recorded.

    Returns ``(classifier, batches)`` where ``batches`` is appended one
    ``[premise, ...]`` list per forward call, in issue order.
    """
    batches: List[List[str]] = []

    def _tokenizer(
        premises: List[str],
        hypotheses: List[str],
        **_kwargs: Any,
    ) -> dict:
        batches.append(list(premises))
        # Thread the batch's premises/hypotheses through as the "encoded"
        # dict; the fake model reassembles them into logits.
        return {"premises": tuple(premises), "hypotheses": tuple(hypotheses)}

    def _model(**encoded: Any) -> _FakeOutputs:
        return _FakeOutputs((encoded["premises"], encoded["hypotheses"]))

    classifier = NliClassifier(
        model=_model,
        tokenizer=_tokenizer,
        torch_module=_FakeTorch(),
        revision="test-rev",
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    return classifier, batches


def _pair(idx: int, char_len: int) -> Tuple[str, str]:
    """A ``(premise, hypothesis)`` whose id is ``idx`` and premise length
    is exactly ``char_len`` (>= 4). Hypothesis is empty so char length is
    fully controlled by the premise."""
    assert char_len >= 4
    head = f"{idx:04d}"
    premise = head + ("a" * (char_len - len(head)))
    return premise, ""


# --------------------------------------------------------------------- #
# Resolver tests
# --------------------------------------------------------------------- #


def test_bucket_batching_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_bucket_batching_enabled() is False


@pytest.mark.parametrize("token", ["1", "true", "TRUE", "yes", "on", "On"])
def test_bucket_batching_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv(ENV_BUCKET_BATCHING, token)
    assert resolve_bucket_batching_enabled() is True


@pytest.mark.parametrize("token", ["0", "false", "no", "off", "garbage", ""])
def test_bucket_batching_falsey_tokens(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv(ENV_BUCKET_BATCHING, token)
    assert resolve_bucket_batching_enabled() is False


def test_bucket_batching_explicit_arg_wins() -> None:
    assert resolve_bucket_batching_enabled(True) is True
    assert resolve_bucket_batching_enabled(False) is False


def test_bucket_batch_sizes_default_when_unset() -> None:
    assert resolve_bucket_batch_sizes() == _DEFAULT_BUCKET_BATCH
    # Returns a fresh list (not the module constant) so a caller can't mutate it.
    got = resolve_bucket_batch_sizes()
    got.append(999)
    assert resolve_bucket_batch_sizes() == _DEFAULT_BUCKET_BATCH


def test_bucket_batch_sizes_valid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BUCKET_BATCH, "200, 100, 50, 20")
    assert resolve_bucket_batch_sizes() == [200, 100, 50, 20]


@pytest.mark.parametrize(
    "raw",
    [
        "200,100,50",  # too few (3 != 4 bounds)
        "200,100,50,20,10",  # too many
        "200,100,x,20",  # non-int member
        "200,100,0,20",  # non-positive member
        "200,100,-5,20",  # negative member
        "   ",  # blank
    ],
)
def test_bucket_batch_sizes_garbage_falls_back(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(ENV_BUCKET_BATCH, raw)
    assert resolve_bucket_batch_sizes() == _DEFAULT_BUCKET_BATCH


def test_bucket_batch_sizes_explicit_arg() -> None:
    assert resolve_bucket_batch_sizes("64,64,64,64") == [64, 64, 64, 64]


def test_estimate_pair_tokens_formula() -> None:
    assert estimate_pair_tokens("", "") == 0
    assert estimate_pair_tokens("a" * 7, "") == math.ceil(7 / _CHARS_PER_TOKEN)
    # Cap at the model's max sequence length.
    huge = "a" * (10 * _MAX_SEQUENCE_LENGTH)
    assert estimate_pair_tokens(huge, "") == _MAX_SEQUENCE_LENGTH


# --------------------------------------------------------------------- #
# Flag-OFF: byte-identical batch slicing to the historical single-size loop
# --------------------------------------------------------------------- #


def test_flag_off_batch_slicing_is_legacy_single_size() -> None:
    """Default (flag off) → sort by char length, chunk by the single global
    ``batch_size``: assert the EXACT batch sequence the stub sees."""
    classifier, batches = _make_recording_classifier()

    # 10 pairs, deliberately supplied in a scrambled order with distinct
    # char lengths so the char-length sort is observable.
    specs = [
        (0, 20), (1, 5), (2, 400), (3, 8), (4, 600),
        (5, 30), (6, 1000), (7, 12), (8, 250), (9, 1500),
    ]
    pairs = [_pair(idx, clen) for idx, clen in specs]

    scores = classifier.score_batch(pairs=pairs, batch_size=4)

    # Result contract: same length, restored to INPUT order (entailment axis
    # carries the pair id).
    assert len(scores) == len(pairs)
    for i, s in enumerate(scores):
        assert isinstance(s, NliScore)
        assert s.entailment == float(i)

    # Expected batch slicing: sort input positions by char length, chunk by 4.
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    sorted_ids = [i for i in order]  # id == input index by construction
    expected = [sorted_ids[b : b + 4] for b in range(0, len(sorted_ids), 4)]

    seen = [[_pair_id(p) for p in batch] for batch in batches]
    assert seen == expected


def test_flag_off_with_env_set_but_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """ED4ALL_NLI_BUCKET_BATCH set but the master flag off → still legacy."""
    monkeypatch.setenv(ENV_BUCKET_BATCH, "256,128,64,32")
    classifier, batches = _make_recording_classifier()
    pairs = [_pair(i, 10 + i) for i in range(6)]
    classifier.score_batch(pairs=pairs, batch_size=3)
    # One global size of 3 → batches of 3, 3 (not bucketed).
    assert [len(b) for b in batches] == [3, 3]


# --------------------------------------------------------------------- #
# Flag-ON: bucketing preserves order + count; batches stay within a bucket
# --------------------------------------------------------------------- #


def _bucket_of(premise: str, hypothesis: str) -> int:
    est = estimate_pair_tokens(premise, hypothesis)
    for bi, bound in enumerate(_BUCKET_TOKEN_BOUNDS):
        if est <= bound:
            return bi
    raise AssertionError("est exceeded final bucket bound")


def test_flag_on_preserves_order_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BUCKET_BATCHING, "1")
    monkeypatch.setenv(ENV_BUCKET_BATCH, "8,4,2,2")
    classifier, batches = _make_recording_classifier()

    # Char lengths chosen to spread across all four token buckets
    # (bounds 128/256/384/512 tokens => ~<=448 / <=896 / <=1344 / >1344 chars),
    # supplied scrambled.
    specs = [
        (0, 100),   # bucket 0
        (1, 1500),  # bucket 3 (capped)
        (2, 600),   # bucket 1
        (3, 100),   # bucket 0
        (4, 1000),  # bucket 2
        (5, 600),   # bucket 1
        (6, 1500),  # bucket 3
        (7, 100),   # bucket 0
        (8, 1000),  # bucket 2
        (9, 600),   # bucket 1
    ]
    pairs = [_pair(idx, clen) for idx, clen in specs]

    scores = classifier.score_batch(pairs=pairs)

    # Count + input-order restoration.
    assert len(scores) == len(pairs)
    for i, s in enumerate(scores):
        assert s.entailment == float(i)

    # Every recorded forward batch must (a) be homogeneous by bucket and
    # (b) not exceed that bucket's configured size.
    bucket_sizes = [8, 4, 2, 2]
    for batch in batches:
        ids = [_pair_id(p) for p in batch]
        pair_buckets = {_bucket_of(*pairs[i]) for i in ids}
        assert len(pair_buckets) == 1, f"batch spans buckets: {ids}"
        bi = pair_buckets.pop()
        assert len(batch) <= bucket_sizes[bi]

    # Sanity: all four buckets were actually populated (test is meaningful).
    all_ids = [_pair_id(p) for batch in batches for p in batch]
    assert sorted(all_ids) == list(range(len(pairs)))
    used_buckets = {_bucket_of(*pairs[i]) for i in all_ids}
    assert used_buckets == {0, 1, 2, 3}


def test_flag_on_bucket_batch_sizes_split_within_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bucket with more pairs than its batch size splits into ceil(n/size)
    forwards, each <= size."""
    monkeypatch.setenv(ENV_BUCKET_BATCHING, "1")
    monkeypatch.setenv(ENV_BUCKET_BATCH, "3,3,3,3")
    classifier, batches = _make_recording_classifier()

    # 7 pairs all in bucket 0 (short) → 3 + 3 + 1.
    pairs = [_pair(i, 100 + i) for i in range(7)]
    scores = classifier.score_batch(pairs=pairs)
    assert len(scores) == 7
    assert [len(b) for b in batches] == [3, 3, 1]
    # Order preserved.
    for i, s in enumerate(scores):
        assert s.entailment == float(i)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
