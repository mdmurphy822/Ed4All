"""Tests for the opt-in validator-chain NLI micro-batching path
(``ED4ALL_NLI_MICROBATCH_VALIDATORS``) on ``BlockProseEntailmentValidator``.

The ``post_rewrite_validation`` prose-entailment gate historically scored every
block's premise/hypothesis pairs SERIALLY (one DeBERTa forward pass at a time).
``ED4ALL_NLI_MICROBATCH_VALIDATORS`` routes the per-block ``score_groundedness``
forward passes through the shared coalescing micro-batch dispatcher
(`lib/classifiers/nli_microbatch.py`) so concurrent block scoring COALESCES into
shared batched forward passes.

CORRECTNESS IS PARAMOUNT (this is behind a load-bearing quality gate). These
tests use a deterministic per-pair NLI stub (no DeBERTa load / GPU) and assert:

* (a) **verdict identity** — flag ON produces a GateResult identical (codes,
  severities, order, passed/action/score) to the serial flag-OFF path, on a
  fixture with varied prose lengths (padding stress at the pair level);
* (b) **flag OFF → serial path unchanged** — no dispatcher is even constructed;
* (c) **concurrency / coalescing** — N blocks scored under the flag produce N
  in-order decision events, and the dispatcher actually merged >1 request into a
  shared drain (the batching win), with every pair scored exactly once.

A ``@pytest.mark.slow`` real-DeBERTa twin in
``lib/classifiers/tests/test_nli_microbatch.py`` proves the batched-vs-single
numerical equivalence on the actual attention-masked forward pass.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers import nli_microbatch as mb  # noqa: E402
from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.classifiers.nli_microbatch import get_dispatcher  # noqa: E402
from lib.validators.block_prose_entailment import (  # noqa: E402
    _CODE_BLOCK_PROSE_CONTRADICTED,
    _CODE_BLOCK_PROSE_UNGROUNDED,
    _ENV_MICROBATCH_VALIDATORS,
    BlockProseEntailmentValidator,
    _microbatch_validators_enabled,
    _resolve_microbatch_validators_concurrency,
)


# --------------------------------------------------------------------- #
# Deterministic per-pair NLI stub (batch-position independent → batched and
# serial scores are identical, the property real attention-masked batching also
# guarantees to within low-order bits).
# --------------------------------------------------------------------- #


class _FakeNli:
    """Deterministic NLI stub; verdict keyed on markers embedded in the prose.

    ``[ENTAIL]`` → high entailment, ``[CONTRA]`` → high contradiction, else
    unsupported. Optionally sleeps per ``score_batch`` call to widen the
    dispatcher's coalescing window in the concurrency test.
    """

    _revision = "fake-nli-rev-mb"
    device = "cpu"

    def __init__(self, *, per_call_sleep: float = 0.0) -> None:
        self.batch_calls: List[int] = []
        self._per_call_sleep = per_call_sleep
        self._lock = threading.Lock()

    def score_batch(
        self, *, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None
    ) -> List[NliScore]:
        with self._lock:
            self.batch_calls.append(len(pairs))
        if self._per_call_sleep:
            time.sleep(self._per_call_sleep)
        out: List[NliScore] = []
        for _premise, hypothesis in pairs:
            if "[ENTAIL]" in hypothesis:
                out.append(NliScore(entailment=0.95, neutral=0.03, contradiction=0.02))
            elif "[CONTRA]" in hypothesis:
                out.append(NliScore(entailment=0.05, neutral=0.10, contradiction=0.85))
            else:
                out.append(NliScore(entailment=0.10, neutral=0.85, contradiction=0.05))
        return out


_CHUNK = (
    "The mitochondrion is the powerhouse of the cell and produces ATP "
    "through oxidative phosphorylation across the inner membrane."
)


def _make_block(*, block_id: str, content: str) -> Block:
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id=block_id.split("#")[0],
        sequence=0,
        content=content,
        source_ids=("dart:slug#blk_0",),
        source_references=({"sourceId": "dart:slug#blk_0"},),
    )


# Varied-length prose bodies → the (premise, hypothesis) pairs the scorer sees
# span a wide length range (padding stress at the pair level). Mix of entailed /
# fabricated-unsupported / contradicted so the fixture exercises every issue
# code and both pass and fail blocks.
_ENTAILED = (
    "<p>The mitochondrion is the cell's powerhouse and generates ATP "
    "through oxidative phosphorylation across its inner membrane [ENTAIL]. "
    "This energy production sustains the metabolic activity of the entire "
    "cell continuously [ENTAIL].</p>"
)
_FABRICATED = (
    "<p>The mitochondrion secretly manufactures cellular gold deep within "
    "its folded cristae structures every passing moment. It also quietly "
    "encodes ancient memories into the surrounding cytoplasm of the cell for "
    "safekeeping.</p>"
)
_CONTRADICTED = (
    "<p>The mitochondrion is the cell's powerhouse producing ATP through "
    "oxidative phosphorylation across its inner membrane [ENTAIL]. The "
    "mitochondrion actually destroys all cellular ATP and halts every "
    "metabolic process within the cell [CONTRA].</p>"
)
_SHORT_ENTAILED = (
    "<p>The mitochondrion generates ATP across its inner membrane [ENTAIL].</p>"
)


def _fixture_blocks() -> List[Block]:
    # Deliberately interleave block types + lengths so the 50-issue-cap order
    # and the per-block scoring both get exercised in a non-trivial sequence.
    return [
        _make_block(block_id="p01#a", content=_ENTAILED),
        _make_block(block_id="p02#b", content=_FABRICATED),
        _make_block(block_id="p03#c", content=_CONTRADICTED),
        _make_block(block_id="p04#d", content=_SHORT_ENTAILED),
        _make_block(block_id="p05#e", content=_FABRICATED),
        _make_block(block_id="p06#f", content=_ENTAILED),
    ]


def _result_signature(result: Any) -> Tuple[Any, ...]:
    """Order-stable, comparable projection of a GateResult."""
    return (
        result.passed,
        result.action,
        result.score,
        tuple(
            (i.code, i.severity, i.location) for i in result.issues
        ),
    )


class _CaptureSpy:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def log_decision(self, **kwargs: Any) -> None:
        with self._lock:
            self.events.append(dict(kwargs))


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_ENV_MICROBATCH_VALIDATORS, raising=False)
    monkeypatch.delenv(
        "ED4ALL_NLI_MICROBATCH_VALIDATORS_CONCURRENCY", raising=False
    )
    mb._reset_for_tests()
    yield
    mb._reset_for_tests()


# --------------------------------------------------------------------- #
# Resolvers
# --------------------------------------------------------------------- #


def test_resolver_parse_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _microbatch_validators_enabled() is False
    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, truthy)
        assert _microbatch_validators_enabled() is True
    for off in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, off)
        assert _microbatch_validators_enabled() is False


def test_concurrency_resolver_parse_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _resolve_microbatch_validators_concurrency() == 16
    for raw, expect in (("32", 32), ("1", 1), ("0", 16), ("-5", 16), ("x", 16)):
        monkeypatch.setenv(
            "ED4ALL_NLI_MICROBATCH_VALIDATORS_CONCURRENCY", raw
        )
        assert _resolve_microbatch_validators_concurrency() == expect


# --------------------------------------------------------------------- #
# (a) Verdict identity — flag ON == serial flag OFF
# --------------------------------------------------------------------- #


def test_microbatch_verdict_identical_to_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_chunks = {"dart:slug#blk_0": _CHUNK}

    # Serial reference (flag off).
    off_validator = BlockProseEntailmentValidator(nli=_FakeNli())
    off = off_validator.validate(
        {"blocks": _fixture_blocks(), "source_chunks": source_chunks}
    )

    # Micro-batch (flag on) — fresh nli instance so a fresh dispatcher is used.
    mb._reset_for_tests()
    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    on_validator = BlockProseEntailmentValidator(nli=_FakeNli())
    on = on_validator.validate(
        {"blocks": _fixture_blocks(), "source_chunks": source_chunks}
    )

    assert _result_signature(on) == _result_signature(off)
    # Sanity: the fixture actually produced BOTH a failure and a pass so the
    # identity assertion is meaningful (not comparing two empty results).
    codes = {i.code for i in off.issues}
    assert _CODE_BLOCK_PROSE_UNGROUNDED in codes
    assert _CODE_BLOCK_PROSE_CONTRADICTED in codes
    assert off.passed is False


def test_microbatch_decision_events_match_serial_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-block decision-capture events fire in the SAME (block) order
    under the flag — issue/counter/capture assembly stays serial."""
    source_chunks = {"dart:slug#blk_0": _CHUNK}

    off_cap = _CaptureSpy()
    BlockProseEntailmentValidator(nli=_FakeNli()).validate(
        {
            "blocks": _fixture_blocks(),
            "source_chunks": source_chunks,
            "decision_capture": off_cap,
        }
    )

    mb._reset_for_tests()
    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    on_cap = _CaptureSpy()
    BlockProseEntailmentValidator(nli=_FakeNli()).validate(
        {
            "blocks": _fixture_blocks(),
            "source_chunks": source_chunks,
            "decision_capture": on_cap,
        }
    )

    def _block_order(cap: _CaptureSpy) -> List[str]:
        # The rationale interpolates ``Block '<id>'`` — pull the ids in order.
        ids: List[str] = []
        for evt in cap.events:
            rat = evt["rationale"]
            marker = "Block '"
            if marker in rat:
                ids.append(rat.split(marker)[1].split("'")[0])
        return ids

    assert _block_order(on_cap) == _block_order(off_cap)
    assert len(on_cap.events) == len(off_cap.events) == len(_fixture_blocks())


# --------------------------------------------------------------------- #
# (b) Flag OFF → serial path, no dispatcher constructed
# --------------------------------------------------------------------- #


def test_flag_off_never_constructs_dispatcher() -> None:
    nli = _FakeNli()
    validator = BlockProseEntailmentValidator(nli=nli)
    validator.validate(
        {"blocks": _fixture_blocks(), "source_chunks": {"dart:slug#blk_0": _CHUNK}}
    )
    # No dispatcher was registered for this nli identity (serial inline path).
    assert get_dispatcher.__module__  # sanity import guard
    assert id(nli) not in mb._REGISTRY
    # And the model was called directly, once per block's score_groundedness
    # (>=1 forward pass per scorable block; here 6 blocks, each ≥1 call).
    assert len(nli.batch_calls) >= len(_fixture_blocks())


# --------------------------------------------------------------------- #
# (c) Concurrency / coalescing — N blocks, coalesced drains, N results in order
# --------------------------------------------------------------------- #


def test_concurrent_blocks_coalesce_into_shared_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_MICROBATCH_VALIDATORS, "1")
    monkeypatch.setenv("ED4ALL_NLI_MICROBATCH_VALIDATORS_CONCURRENCY", "16")
    # A per-call sleep widens the dispatcher window so concurrent block scorers
    # pile up and get coalesced into shared drains.
    monkeypatch.setenv("ED4ALL_NLI_MICROBATCH_WINDOW_MS", "80")
    monkeypatch.setenv("ED4ALL_NLI_MICROBATCH_MAX_PAIRS", "256")
    nli = _FakeNli(per_call_sleep=0.02)

    n_blocks = 16
    blocks = [
        _make_block(block_id=f"pg{i:02d}#c", content=_ENTAILED)
        for i in range(n_blocks)
    ]
    capture = _CaptureSpy()
    validator = BlockProseEntailmentValidator(nli=nli)
    result = validator.validate(
        {
            "blocks": blocks,
            "source_chunks": {"dart:slug#blk_0": _CHUNK},
            "decision_capture": capture,
        }
    )

    # N looped blocks → N in-order decision events (all passed, entailed prose).
    assert result.passed is True
    assert len(capture.events) == n_blocks

    dispatcher = get_dispatcher(nli)
    assert dispatcher is not None
    # Coalescing actually happened: at least one drain merged >1 block request.
    assert dispatcher.drain_request_counts, "no drains recorded"
    assert max(dispatcher.drain_request_counts) >= 2, (
        f"expected coalescing; per-drain request counts="
        f"{dispatcher.drain_request_counts}"
    )
    # Every submitted pair scored exactly once across all drains equals the sum
    # of the per-block forward-pass pair counts (no double-scoring / no drops).
    assert sum(dispatcher.drain_pair_counts) == sum(nli.batch_calls)
