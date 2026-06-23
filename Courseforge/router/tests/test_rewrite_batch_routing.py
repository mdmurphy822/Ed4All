"""Router-level tests for the multi-block BATCHED rewrite path.

Exercises:

- ``route_rewrite_batch`` round-based regen: a FAILED block is re-batched
  WITHOUT re-rolling its passing siblings (a passing block is dispatched
  exactly once).
- per-block-type validator isolation within a batch.
- the cloud-lane top-1 clamp on ``route_rewrite_with_remediation``.
- round-budget exhaustion stamps ``validator_consensus_fail``.
- total-failure (every slot None) handling.
- byte-stable contract: COURSEFORGE_REWRITE_BATCH=0 → the per-block path
  runs unchanged and ``generate_rewrite_batch`` is never called.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import (  # noqa: E402
    BlockProviderSpec,
    CourseforgeRouter,
)
from MCP.hardening.validation_gates import GateResult  # noqa: E402
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _block(block_id: str, block_type: str = "concept") -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content={"key_claims": ["x"], "curies": []},
    )


class _RecordingBatchProvider:
    """Fake RewriteProvider that records each batch call.

    ``generate_rewrite_batch`` returns a rendered-HTML Block per requested
    block (str content), recording which block_ids appeared in each batch so a
    test can assert no passing sibling is re-dispatched. ``fail_ids`` controls
    which block_ids the PAIRED validator will reject (the provider itself
    always emits valid HTML — failure is driven by the validator).
    """

    def __init__(self, *, none_ids: Optional[set] = None) -> None:
        self.batch_calls: List[List[str]] = []
        self.single_calls: List[str] = []
        self._none_ids = none_ids or set()

    def generate_rewrite_batch(
        self,
        blocks: Sequence[Block],
        *,
        source_chunks_by_id: Any = None,
        objectives: Any = None,
        remediation_suffix_by_id: Any = None,
    ) -> Dict[str, Optional[Block]]:
        self.batch_calls.append([b.block_id for b in blocks])
        out: Dict[str, Optional[Block]] = {}
        for b in blocks:
            if b.block_id in self._none_ids:
                out[b.block_id] = None
                continue
            out[b.block_id] = dataclasses.replace(
                b, content=f"<p>{b.block_id}</p>",
            )
        return out

    # Used to prove the per-block path is NOT taken on the batch path.
    def generate_rewrite(self, block: Block, **kwargs: Any) -> Block:
        self.single_calls.append(block.block_id)
        return dataclasses.replace(block, content=f"<p>{block.block_id}</p>")


class _SelectiveValidator:
    """Validator that fails for block_ids in ``fail_ids`` for ``fail_rounds``.

    Tracks per-block validate-call counts so it can pass a block after it has
    been re-batched N times (simulating remediation eventually succeeding).
    """

    def __init__(
        self, *, fail_ids: Optional[set] = None, fail_rounds: int = 99,
    ) -> None:
        self.name = "rewrite_concept_check"
        self.gate_id = "rewrite_concept_check"
        self._fail_ids = fail_ids or set()
        self._fail_rounds = fail_rounds
        self.seen: Dict[str, int] = {}

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        block = inputs.get("block")
        bid = getattr(block, "block_id", "")
        self.seen[bid] = self.seen.get(bid, 0) + 1
        should_fail = (
            bid in self._fail_ids and self.seen[bid] <= self._fail_rounds
        )
        return GateResult(
            gate_id=self.gate_id,
            validator_name=self.name,
            validator_version="1.0.0",
            passed=not should_fail,
            action=None if not should_fail else "regenerate",
        )


def _cloud_router(provider: _RecordingBatchProvider) -> CourseforgeRouter:
    return CourseforgeRouter(rewrite_provider=provider)


def _force_cloud_spec(monkeypatch):
    """Pin the rewrite lane to a cloud endpoint via env so _resolve_spec
    yields a non-loopback / hosted seat for every block_type."""
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "nvidia")
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", "1")


# ---------------------------------------------------------------------------
# route_rewrite_batch — round-based regen + sibling isolation
# ---------------------------------------------------------------------------


def test_failed_block_rebatches_without_rerolling_passing_siblings(monkeypatch):
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    blocks = [_block("a"), _block("b"), _block("c")]
    # "b" fails round 1 only, passes round 2.
    validator = _SelectiveValidator(fail_ids={"b"}, fail_rounds=1)

    results = router.route_rewrite_batch(
        blocks, validators=[validator], regen_budget=5,
    )

    # All three blocks finalize as passing.
    assert set(results.keys()) == {"a", "b", "c"}
    assert all(isinstance(results[k].content, str) for k in results)
    assert results["b"].escalation_marker is None

    # Round 1 batched all three; round 2 batched ONLY the failed "b".
    assert provider.batch_calls[0] == ["a", "b", "c"]
    assert provider.batch_calls[1] == ["b"]
    # "a" and "c" each dispatched exactly ONCE (no re-roll of passing sibs).
    a_dispatches = sum(1 for c in provider.batch_calls if "a" in c)
    c_dispatches = sum(1 for c in provider.batch_calls if "c" in c)
    assert a_dispatches == 1
    assert c_dispatches == 1
    # Single-block path never used.
    assert provider.single_calls == []


def test_none_slot_rebatches_next_round(monkeypatch):
    _force_cloud_spec(monkeypatch)
    # "x" returns None on EVERY batch (parse-miss) → re-batched until budget.
    provider = _RecordingBatchProvider(none_ids={"x"})
    router = _cloud_router(provider)
    blocks = [_block("x"), _block("y")]
    results = router.route_rewrite_batch(
        blocks, validators=[], regen_budget=3,
    )
    # "y" finalizes; "x" exhausts the budget and is consensus-fail stamped.
    assert results["y"].escalation_marker is None
    assert results["x"].escalation_marker == "validator_consensus_fail"
    # "x" was re-batched each round; "y" finalized round 1.
    x_calls = sum(1 for c in provider.batch_calls if "x" in c)
    assert x_calls == 3


def test_round_budget_exhaustion_stamps_consensus_fail(monkeypatch):
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    blocks = [_block("z")]
    # Always fails the validator → never converges.
    validator = _SelectiveValidator(fail_ids={"z"}, fail_rounds=99)
    results = router.route_rewrite_batch(
        blocks, validators=[validator], regen_budget=2,
    )
    assert results["z"].escalation_marker == "validator_consensus_fail"
    # Re-batched for both rounds.
    assert sum(1 for c in provider.batch_calls if "z" in c) == 2


def test_total_failure_all_none(monkeypatch):
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider(none_ids={"a", "b"})
    router = _cloud_router(provider)
    blocks = [_block("a"), _block("b")]
    results = router.route_rewrite_batch(
        blocks, validators=[], regen_budget=2,
    )
    assert results["a"].escalation_marker == "validator_consensus_fail"
    assert results["b"].escalation_marker == "validator_consensus_fail"


def test_per_block_type_validator_isolation_within_batch(monkeypatch):
    """A validator that rejects only one block_type re-batches only that
    block; the other block_type passes its OWN chain and finalizes."""
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    # assessment_item + concept share a (provider, model, max_tokens) group
    # only when their specs match; if not, they batch separately — either way
    # the failing one is isolated. Use a validator keyed on block_id.
    ai = _block("ai", block_type="assessment_item")
    co = _block("co", block_type="concept")
    validator = _SelectiveValidator(fail_ids={"ai"}, fail_rounds=1)
    results = router.route_rewrite_batch(
        [ai, co], validators=[validator], regen_budget=5,
    )
    assert results["co"].escalation_marker is None
    assert results["ai"].escalation_marker is None
    # "co" dispatched once; "ai" re-dispatched (fail round 1 then pass).
    co_calls = sum(1 for c in provider.batch_calls if "co" in c)
    assert co_calls == 1


# ---------------------------------------------------------------------------
# Cloud-lane top-1 clamp on route_rewrite_with_remediation
# ---------------------------------------------------------------------------


class _CountingRewriteProvider:
    """Counts generate_rewrite calls (per-candidate)."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_rewrite(self, block: Block, **kwargs: Any) -> Block:
        self.calls += 1
        return dataclasses.replace(block, content="<p>ok</p>")


def test_cloud_lane_clamps_n_candidates_to_one(monkeypatch):
    _force_cloud_spec(monkeypatch)
    provider = _CountingRewriteProvider()

    class _AlwaysFail:
        name = "x"
        gate_id = "x"

        def validate(self, inputs):
            return GateResult(
                gate_id="x", validator_name="x", validator_version="1",
                passed=False, action="regenerate",
            )

    # Ask for 4 candidates with a generous budget; without the cloud clamp the
    # loop would dispatch 4 times. The top-1 clamp bounds it to ONE dispatch.
    router = CourseforgeRouter(
        rewrite_provider=provider, n_candidates=4, regen_budget=10,
    )
    blk = _block("clamp")
    router.route_rewrite_with_remediation(blk, validators=[_AlwaysFail()])
    assert provider.calls == 1


def test_local_lane_does_not_clamp(monkeypatch):
    # Loopback base_url → NOT cloud → no clamp; with an always-fail validator
    # the loop runs all N candidates.
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "local")
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", "1")
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")
    provider = _CountingRewriteProvider()

    class _AlwaysFail:
        name = "x"
        gate_id = "x"

        def validate(self, inputs):
            return GateResult(
                gate_id="x", validator_name="x", validator_version="1",
                passed=False, action="regenerate",
            )

    router = CourseforgeRouter(rewrite_provider=provider, n_candidates=3,
                              regen_budget=10)
    blk = _block("noclamp")
    router.route_rewrite_with_remediation(blk, validators=[_AlwaysFail()])
    # No clamp → 3 candidate dispatches (N=3 < budget=10).
    assert provider.calls == 3


# ---------------------------------------------------------------------------
# Byte-stable: COURSEFORGE_REWRITE_BATCH=0 → per-block path, no batch call
# ---------------------------------------------------------------------------


def test_batch_off_uses_per_block_path_no_batch_call(monkeypatch):
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", "0")
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "nvidia")
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    blk = _block("legacy")
    # The per-block remediation path uses generate_rewrite, never the batch.
    out = router.route_rewrite_with_remediation(blk, validators=[])
    assert provider.single_calls == ["legacy"]
    assert provider.batch_calls == []
    assert isinstance(out.content, str)


def test_is_cloud_rewrite_lane_detection():
    s_loop = BlockProviderSpec(
        block_type="concept", tier="rewrite", provider="local", model="m",
        base_url="http://127.0.0.1:11434/v1",
    )
    s_cloud_url = BlockProviderSpec(
        block_type="concept", tier="rewrite", provider="local", model="m",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    s_cloud_prov = BlockProviderSpec(
        block_type="concept", tier="rewrite", provider="nvidia", model="m",
    )
    s_local_prov = BlockProviderSpec(
        block_type="concept", tier="rewrite", provider="local", model="m",
    )
    assert CourseforgeRouter._is_cloud_rewrite_lane(s_loop) is False
    assert CourseforgeRouter._is_cloud_rewrite_lane(s_cloud_url) is True
    assert CourseforgeRouter._is_cloud_rewrite_lane(s_cloud_prov) is True
    assert CourseforgeRouter._is_cloud_rewrite_lane(s_local_prov) is False
