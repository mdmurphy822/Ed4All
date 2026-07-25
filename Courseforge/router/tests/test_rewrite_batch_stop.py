"""Router-level graceful-stop ("checkpoint on command") for the batched lane.

``route_rewrite_batch`` (the rewrite-tier BATCHED cloud lane) is the worst
"worst-case-loss" violator in the system: today a kill mid-batch loses EVERY
freshly-authored block because the pipeline defers all checkpointing to a
post-dispatch Phase-C sweep. Plan P4.7 / Resolved-Decision D2 closes that gap
with TWO additions INSIDE the router:

- an optional ``on_block_resolved(block_id, block)`` checkpoint callback fired
  the instant a block PASSES its validator chain (its terminal resolution), and
- a between-round ``stop_control.check_stop("rewrite_batched", n_resolved)`` at
  the top of every regen round.

So a stop armed mid-run raises at the next round's top, by which point every
block resolved in rounds ``<= N`` has already been handed to the callback
(checkpointed) — the raise loses at most the in-flight round. Degraded
(``validator_consensus_fail``) survivors are NOT called back (they re-run on
resume, mirroring the per-block rewrite-checkpoint contract).

The named contract (plan AMENDMENT #10) exercised here:

- **stop after round N** → the callback was invoked for every block resolved in
  rounds ``<= N``; ``GracefulStopRequested`` raised; the still-failing
  (unresolved) blocks were NOT called back; the provider dispatched exactly N
  batches (one per round for a ``<=`` batch_size group).
- **resume** → re-running with the still-unresolved blocks (what the pipeline's
  sidecar filter passes) resolves them; total callbacks across both legs == the
  full block set.
- **pre-armed sentinel** → zero provider calls, zero callbacks.
- **callback=None** (the default) → byte-identical to the legacy path (regression
  guard; the broader legacy suite lives in ``test_rewrite_batch_routing.py``).

Sentinel isolation is fully hermetic: a per-test ``ED4ALL_STATE_RUNS_DIR`` +
synthetic ``ED4ALL_RUN_ID`` (no dependency on the repo ``state/runs/``).
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

from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from MCP.hardening.validation_gates import GateResult  # noqa: E402
from blocks import Block  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402

_RUN_ID = "STOP_ROUTER_BATCH_TESTRUN"


# ---------------------------------------------------------------------------
# Helpers / fakes (mirror test_rewrite_batch_routing.py)
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
    """Fake RewriteProvider: emits valid str-HTML per block, records batches."""

    def __init__(self, *, none_ids: Optional[set] = None) -> None:
        self.batch_calls: List[List[str]] = []
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
            out[b.block_id] = dataclasses.replace(b, content=f"<p>{b.block_id}</p>")
        return out


class _ArmAfterRoundProvider(_RecordingBatchProvider):
    """Arms the real run-scoped sentinel AFTER its ``arm_after_calls``-th batch.

    With a block set that fits one batch per round (<= batch_size remaining),
    one batch call == one round, so arming after the Nth call arms the sentinel
    once round N's dispatch has completed. The NEXT round's loop-top
    ``check_stop`` then raises — round N's PASSING blocks are already resolved +
    checkpointed by that point.
    """

    def __init__(
        self, *, arm_after_calls: int, none_ids: Optional[set] = None,
    ) -> None:
        super().__init__(none_ids=none_ids)
        self._arm_after_calls = arm_after_calls

    def generate_rewrite_batch(self, blocks, **kwargs):  # type: ignore[override]
        out = super().generate_rewrite_batch(blocks, **kwargs)
        if len(self.batch_calls) == self._arm_after_calls:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return out


class _SelectiveValidator:
    """Fails ``fail_ids`` for their first ``fail_rounds`` validate calls."""

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
        should_fail = bid in self._fail_ids and self.seen[bid] <= self._fail_rounds
        return GateResult(
            gate_id=self.gate_id,
            validator_name=self.name,
            validator_version="1.0.0",
            passed=not should_fail,
            action=None if not should_fail else "regenerate",
        )


def _cloud_router(provider: _RecordingBatchProvider) -> CourseforgeRouter:
    return CourseforgeRouter(rewrite_provider=provider)


def _force_cloud_spec(monkeypatch) -> None:
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "nvidia")
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", "1")
    # Pin the per-call cap so batch packing is decoupled from the shipped
    # rewrite max_tokens default (batches split when the summed per-block
    # output budget exceeds the 16384-512 cap); these tests assert on stop /
    # checkpoint semantics, not on the token cap.
    monkeypatch.setenv("COURSEFORGE_REWRITE_MAX_TOKENS", "2400")


class _Recorder:
    """Records ``on_block_resolved`` invocations (block_id + resolved block)."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __call__(self, block_id: str, block: Block) -> None:
        self.calls.append((block_id, block))

    @property
    def ids(self) -> List[str]:
        return [bid for bid, _ in self.calls]


@pytest.fixture
def _armed_env(tmp_path, monkeypatch):
    """Per-test sentinel isolation: tmp state/runs + a synthetic run_id."""
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "state" / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# ---------------------------------------------------------------------------
# Named test (AMENDMENT #10): stop after round N
# ---------------------------------------------------------------------------
def test_stop_after_round_n_checkpoints_resolved_only(monkeypatch, _armed_env):
    """Stop after round 2 → callbacks for the round-1 + round-2 winners only.

    - ``a`` passes round 1, ``b`` passes round 2, ``c`` never converges.
    - The provider arms the sentinel once round 2's dispatch completes.
    - Round 3's loop-top ``check_stop`` raises: ``a`` + ``b`` were already
      handed to the callback; ``c`` (still unresolved / in-flight) was NOT.
    """
    _force_cloud_spec(monkeypatch)
    provider = _ArmAfterRoundProvider(arm_after_calls=2)
    router = _cloud_router(provider)
    recorder = _Recorder()
    blocks = [_block("a"), _block("b"), _block("c")]

    class _ABCValidator(_SelectiveValidator):
        """a passes always; b passes from round 2; c never converges."""

        def validate(self, inputs):
            bid = getattr(inputs.get("block"), "block_id", "")
            self.seen[bid] = self.seen.get(bid, 0) + 1
            if bid == "a":
                fail = False
            elif bid == "b":
                fail = self.seen[bid] <= 1
            else:  # c
                fail = True
            return GateResult(
                gate_id=self.gate_id, validator_name=self.name,
                validator_version="1.0.0", passed=not fail,
                action=None if not fail else "regenerate",
            )

    validator = _ABCValidator()

    with pytest.raises(GracefulStopRequested) as exc_info:
        router.route_rewrite_batch(
            blocks, validators=[validator], regen_budget=10,
            on_block_resolved=recorder,
        )

    # Resolved-in-rounds<=2 blocks were checkpointed; the unresolved one wasn't.
    assert set(recorder.ids) == {"a", "b"}
    assert "c" not in recorder.ids
    # Every callback carried a validator-PASSING (marker-free, str) block.
    for _bid, blk in recorder.calls:
        assert blk.escalation_marker is None
        assert isinstance(blk.content, str)
    # units_completed reported on the stop == the two checkpointed winners.
    assert exc_info.value.units_completed == 2
    assert exc_info.value.site_id == "rewrite_batched"
    # Provider dispatched exactly 2 batches (round 1 [a,b,c], round 2 [b,c]).
    assert provider.batch_calls[0] == ["a", "b", "c"]
    assert provider.batch_calls[1] == ["b", "c"]
    assert len(provider.batch_calls) == 2


def test_stop_after_round_one(monkeypatch, _armed_env):
    """Minimal leg: stop after round 1 → only the round-1 winner is called back."""
    _force_cloud_spec(monkeypatch)
    provider = _ArmAfterRoundProvider(arm_after_calls=1)
    router = _cloud_router(provider)
    recorder = _Recorder()
    blocks = [_block("a"), _block("b")]
    # a passes round 1; b fails round 1 (would pass round 2, never reached).
    validator = _SelectiveValidator(fail_ids={"b"}, fail_rounds=1)

    with pytest.raises(GracefulStopRequested) as exc_info:
        router.route_rewrite_batch(
            blocks, validators=[validator], regen_budget=10,
            on_block_resolved=recorder,
        )

    assert recorder.ids == ["a"]
    assert exc_info.value.units_completed == 1
    assert len(provider.batch_calls) == 1  # round 2 never dispatched


def test_resume_after_stop_resolves_remaining(monkeypatch, _armed_env):
    """Resume leg: the interrupted run checkpoints N; a fresh run over the
    still-unresolved blocks (what the pipeline's sidecar filter passes) resolves
    them — total callbacks across both legs == the full block set."""
    _force_cloud_spec(monkeypatch)

    # Interrupted leg — stop after round 1 (a resolved, b unresolved).
    provider1 = _ArmAfterRoundProvider(arm_after_calls=1)
    router1 = _cloud_router(provider1)
    rec1 = _Recorder()
    validator1 = _SelectiveValidator(fail_ids={"b"}, fail_rounds=1)
    with pytest.raises(GracefulStopRequested):
        router1.route_rewrite_batch(
            [_block("a"), _block("b")], validators=[validator1],
            regen_budget=10, on_block_resolved=rec1,
        )
    assert rec1.ids == ["a"]

    # Resume leg — clear the sentinel; re-dispatch ONLY the unresolved block.
    stop_control.clear_stop(include_global=True)
    provider2 = _RecordingBatchProvider()
    router2 = _cloud_router(provider2)
    rec2 = _Recorder()
    results = router2.route_rewrite_batch(
        [_block("b")], validators=[], regen_budget=10, on_block_resolved=rec2,
    )
    assert rec2.ids == ["b"]
    assert results["b"].escalation_marker is None
    # Total distinct blocks resolved across both legs == the full set.
    assert set(rec1.ids) | set(rec2.ids) == {"a", "b"}


def test_pre_armed_zero_calls(monkeypatch, _armed_env):
    """A pre-armed sentinel raises on round 0 → zero provider calls, zero
    callbacks."""
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    recorder = _Recorder()
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested) as exc_info:
        router.route_rewrite_batch(
            [_block("a"), _block("b")], validators=[], regen_budget=10,
            on_block_resolved=recorder,
        )

    assert provider.batch_calls == []      # nothing dispatched
    assert recorder.calls == []            # nothing checkpointed
    assert exc_info.value.units_completed == 0


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------
def test_consensus_fail_survivors_not_called_back(monkeypatch, _armed_env):
    """A budget-exhausted (``validator_consensus_fail``) block is NOT handed to
    the callback — degraded units are never checkpointed (they re-run on
    resume). No sentinel is armed here, so the run completes normally."""
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    recorder = _Recorder()
    # z always fails → exhausts the budget → consensus-fail stamp; y passes.
    validator = _SelectiveValidator(fail_ids={"z"}, fail_rounds=99)
    results = router.route_rewrite_batch(
        [_block("y"), _block("z")], validators=[validator], regen_budget=2,
        on_block_resolved=recorder,
    )

    assert results["z"].escalation_marker == "validator_consensus_fail"
    assert results["y"].escalation_marker is None
    # Only the PASSING block was checkpointed.
    assert recorder.ids == ["y"]


def test_callback_none_default_unchanged(monkeypatch, _armed_env):
    """Backward compat: the default ``on_block_resolved=None`` is a pure no-op —
    the batched path resolves every block exactly as before (regression guard for
    the legacy callers that pass no callback)."""
    _force_cloud_spec(monkeypatch)
    provider = _RecordingBatchProvider()
    router = _cloud_router(provider)
    blocks = [_block("a"), _block("b"), _block("c")]
    validator = _SelectiveValidator(fail_ids={"b"}, fail_rounds=1)

    results = router.route_rewrite_batch(
        blocks, validators=[validator], regen_budget=5,
    )

    assert set(results.keys()) == {"a", "b", "c"}
    assert all(results[k].escalation_marker is None for k in results)
    assert all(isinstance(results[k].content, str) for k in results)
    # b re-batched (fail round 1, pass round 2); a + c dispatched once.
    assert provider.batch_calls[0] == ["a", "b", "c"]
    assert provider.batch_calls[1] == ["b"]
