"""Cross-block NLI (``ED4ALL_NLI_CROSSBLOCK``) verdict-identity + cache tests.

Stub-only (deterministic ``_FakeNli`` via the ``nli=`` seam — no DeBERTa load).
Asserts:

* the threaded cross-block path (dispatcher fan-out) produces a GateResult
  verdict-identical + issue-set-identical to the serial path,
* order determinism (issues emitted in original block order regardless of thread
  completion order),
* flag-OFF is byte-identical to the legacy serial path,
* the resume sidecar + feature-cache cooperate (a warm run is a cache HIT).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

import lib.classifiers.nli_microbatch as mb  # noqa: E402
from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.block_prose_entailment import (  # noqa: E402
    BlockProseEntailmentValidator,
)
from lib.validators.feature_cache import BlockFeatureCache  # noqa: E402


class _FakeNli:
    """Deterministic NLI stub. Marker-driven per-hypothesis verdict.

    A single scorer thread owns this via the dispatcher, so ``score_batch`` is
    only ever called from one thread — no lock needed. Accepts ``batch_size``
    so the dispatcher's big-forward-pass path is exercised.
    """

    _revision = "fake-rev"
    device = "cpu"
    dtype = "float32"

    def score_batch(
        self, *, pairs: List[Tuple[str, str]], batch_size: int = None
    ) -> List[NliScore]:
        out: List[NliScore] = []
        for _p, h in pairs:
            if "[ENTAIL]" in h:
                out.append(NliScore(entailment=0.95, neutral=0.03, contradiction=0.02))
            elif "[CONTRA]" in h:
                out.append(NliScore(entailment=0.05, neutral=0.10, contradiction=0.85))
            else:
                out.append(NliScore(entailment=0.10, neutral=0.85, contradiction=0.05))
        return out


def _blocks() -> List[Block]:
    specs = [
        ("page_01#concept_a_0", "The alpha concept is grounded and produces energy [ENTAIL]. "
         "This sustains the whole process continuously and reliably [ENTAIL]."),
        ("page_01#concept_b_1", "The beta claim invents cellular gold in its cristae folds. "
         "It also encodes ancient memories into the cytoplasm for safe keeping."),
        ("page_02#example_c_2", "The gamma worked example computes the sum across the domain [ENTAIL]. "
         "It demonstrates the exact procedure step by careful step [ENTAIL]."),
        ("page_02#concept_d_3", "The delta statement directly contradicts the cited source badly [CONTRA]. "
         "It asserts the precise opposite of the grounded premise entirely [CONTRA]."),
        ("page_03#concept_e_4", "The epsilon concept mixes one grounded and true supported idea [ENTAIL]. "
         "But then it fabricates a wholly unsupported second sweeping claim."),
    ]
    blocks: List[Block] = []
    for i, (bid, content) in enumerate(specs):
        sid = f"dart:slug#blk_{i}"
        blocks.append(
            Block(
                block_id=bid,
                block_type="concept" if "concept" in bid else "example",
                page_id=bid.split("#")[0],
                sequence=i,
                content=f"<p>{content}</p>",
                source_ids=(sid,),
                source_references=({"sourceId": sid},),
            )
        )
    return blocks


def _source_chunks(blocks: List[Block]) -> Dict[str, str]:
    return {
        f"dart:slug#blk_{i}": (
            "The grounded premise body for this block covers energy production, "
            "the computed sum across the domain, and the exact procedure."
        )
        for i in range(len(blocks))
    }


def _issue_signature(result: Any) -> List[Tuple[str, str, Any]]:
    return [(i.code, i.severity, i.location) for i in result.issues]


@pytest.fixture(autouse=True)
def _reset_dispatcher() -> None:
    mb._reset_for_tests()
    yield
    mb._reset_for_tests()


def test_crossblock_verdict_identical_to_serial(monkeypatch: Any) -> None:
    blocks = _blocks()
    source_chunks = _source_chunks(blocks)

    # Serial (flag off).
    monkeypatch.delenv("ED4ALL_NLI_CROSSBLOCK", raising=False)
    serial = BlockProseEntailmentValidator(nli=_FakeNli()).validate(
        {"blocks": blocks, "source_chunks": source_chunks}
    )

    # Cross-block threaded (flag on).
    monkeypatch.setenv("ED4ALL_NLI_CROSSBLOCK", "1")
    monkeypatch.setenv("ED4ALL_NLI_CROSSBLOCK_THREADS", "8")
    threaded = BlockProseEntailmentValidator(nli=_FakeNli()).validate(
        {"blocks": blocks, "source_chunks": source_chunks}
    )

    assert threaded.passed == serial.passed
    assert threaded.action == serial.action
    # Issue set + ORDER identical (issues assembled in original block order in
    # the serial validate loop regardless of thread completion order).
    assert _issue_signature(threaded) == _issue_signature(serial)


def test_crossblock_with_feature_cache_identical(monkeypatch: Any) -> None:
    blocks = _blocks()
    source_chunks = _source_chunks(blocks)

    monkeypatch.delenv("ED4ALL_NLI_CROSSBLOCK", raising=False)
    baseline = BlockProseEntailmentValidator(nli=_FakeNli()).validate(
        {"blocks": blocks, "source_chunks": source_chunks}
    )

    monkeypatch.setenv("ED4ALL_NLI_CROSSBLOCK", "1")
    cache = BlockFeatureCache({}, {})
    withcache = BlockProseEntailmentValidator(nli=_FakeNli()).validate(
        {
            "blocks": blocks,
            "source_chunks": source_chunks,
            "feature_cache": cache,
        }
    )
    assert withcache.passed == baseline.passed
    assert _issue_signature(withcache) == _issue_signature(baseline)


def test_sidecar_warm_run_is_cache_hit(monkeypatch: Any, tmp_path: Path) -> None:
    blocks = _blocks()
    source_chunks = _source_chunks(blocks)
    inputs = {
        "blocks": blocks,
        "source_chunks": source_chunks,
        "prose_entailment_cache_dir": str(tmp_path),
    }
    monkeypatch.setenv("ED4ALL_VALIDATION_CHECKPOINT", "1")
    monkeypatch.delenv("ED4ALL_NLI_CROSSBLOCK", raising=False)

    cold = BlockProseEntailmentValidator(nli=_FakeNli()).validate(dict(inputs))
    # Warm run: a scorer that would RAISE if invoked proves every scorable block
    # was restored from the content-addressed sidecar without re-scoring.
    class _ExplodingNli:
        _revision = "fake-rev"
        device = "cpu"

        def score_batch(self, *, pairs, batch_size=None):  # noqa: ANN001
            raise AssertionError("warm run must not re-score — sidecar HIT expected")

    warm = BlockProseEntailmentValidator(nli=_ExplodingNli()).validate(dict(inputs))
    assert _issue_signature(warm) == _issue_signature(cold)
    assert warm.passed == cold.passed


def test_dispatcher_recovers_from_cuda_oom_by_halving() -> None:
    """Risk #3: a drain larger than the fake card's limit OOMs; the dispatcher
    halves the forward-pass slice and retries rather than failing the caller
    (which the W2.1 handler would otherwise convert to a vacuous auto-pass)."""

    class _OomAt:
        """Raises a CUDA-OOM-shaped RuntimeError for any batch above ``limit``."""

        _revision = "oom-rev"
        device = "cuda:0"

        def __init__(self, limit: int) -> None:
            self.limit = limit
            self.max_seen = 0

        def score_batch(self, *, pairs, batch_size=None):  # noqa: ANN001
            self.max_seen = max(self.max_seen, len(pairs))
            if len(pairs) > self.limit:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            return [NliScore(entailment=0.5, neutral=0.4, contradiction=0.1) for _ in pairs]

    nli = _OomAt(limit=3)
    disp = mb.NliMicrobatchDispatcher(nli, max_pairs=8, window_ms=0)
    try:
        # 10 pairs > limit(3) at max_pairs(8) → first attempt OOMs; halving to
        # 4 still OOMs; halving to 2 succeeds. Every pair still scored.
        pairs = [(f"p{i}", f"h{i}") for i in range(10)]
        scores = disp.score_batch(pairs=pairs)
    finally:
        disp.shutdown()
    assert len(scores) == 10
    assert all(abs(s.entailment - 0.5) < 1e-9 for s in scores)
