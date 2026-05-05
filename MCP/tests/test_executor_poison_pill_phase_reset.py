"""Worker W2 regression test — poison-pill detector resets at phase boundary.

`MCP/core/executor.py::TaskExecutor` keeps a single
``PoisonPillDetector`` instance for the lifetime of the executor; the
detector groups failures by pattern hash and trips when N same-pattern
errors fall inside a 5-minute window. Pre-W2, ``execute_phase``
(``MCP/core/executor.py:1217+``) never called the existing
``reset_poison_detector()`` helper, so cross-phase pattern collisions
could trip a false-positive halt on phase N+1's very first failure
when phases N and earlier had already recorded N-1 same-pattern errors.

W2 inserts ``self.reset_poison_detector()`` at the top of
``execute_phase`` so each phase starts with a clean state hash window.
This module asserts the fix:

1. ``test_poison_pill_state_resets_between_phases`` — drive 2
   same-pattern errors via ``poison_detector.record_failure`` (NOT
   yet at threshold=3), call ``execute_phase`` (phase B), then
   record 1 more same-pattern error. Pre-W2 this would trip
   poison-pill (2 from phase A + 1 from phase B = 3); post-W2 the
   reset at phase B entry clears the window so the same-pattern
   counter restarts at 1.

2. ``test_reset_poison_detector_called_at_phase_entry`` — direct
   assertion that ``reset_poison_detector`` is invoked when
   ``execute_phase`` runs, via spy / ``call_count`` tracking.

3. ``test_within_phase_poison_detection_still_trips`` — semantic
   smoke test that the W2 reset doesn't accidentally disable
   intra-phase poison detection (Wave 38 semantics intact). Three
   same-pattern errors recorded inside a single phase still trip.

The tests bypass the workflow JSON / tool-registry plumbing entirely
by stubbing ``_execute_parallel`` to return ``{}`` immediately —
``execute_phase`` still runs to completion, exercising the reset
call site without needing real tasks on disk. ``checkpoint_manager``
is also disabled (set to ``None``) so the test doesn't write
phase-checkpoint sidecars to the run path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_classified_error(executor, message: str, task_id: str):
    """Build a ClassifiedError via the executor's own classifier so its
    pattern_hash is computed exactly the way runtime would compute it.
    """
    return executor.error_classifier.classify(
        TimeoutError(message), task_id=task_id
    )


def _build_executor(state_runs_isolated):
    """Construct a TaskExecutor wired for the W2 phase-reset tests.

    - ``state_runs_isolated`` redirects run-state writes into ``tmp_path``.
    - ``checkpoint_manager`` is disabled so ``execute_phase`` doesn't
      try to materialise phase checkpoints on disk.
    - ``_execute_parallel`` is stubbed to return ``{}`` so the phase
      runs to completion without needing real tasks on disk.
    """
    from MCP.core.executor import TaskExecutor

    executor = TaskExecutor(
        tool_registry={},
        run_id="W2_TEST_RUN",
        poison_pill_threshold=3,
    )
    # Disable checkpoint manager so execute_phase doesn't try to
    # materialise phase checkpoints on disk.
    executor.checkpoint_manager = None
    # Stub the parallel executor so execute_phase doesn't touch the
    # tool registry / workflow state file.
    async def _noop_parallel(workflow_id, tasks, max_concurrent):
        return {}

    executor._execute_parallel = _noop_parallel  # type: ignore[assignment]
    return executor


@pytest.mark.asyncio
async def test_poison_pill_state_resets_between_phases(state_runs_isolated):
    """Cross-phase same-pattern errors below threshold do NOT trip.

    Phase A records 2 same-pattern ``TimeoutError("ConnectTimeout: ...")``
    failures (just under threshold=3). Phase B then records 1 same-
    pattern failure. Without the W2 reset, the detector's pattern-hash
    counter would reach 3 (2 + 1) and trip ``triggered=True``. With the
    reset at phase B entry, the counter restarts at 1 and stays below
    threshold.
    """
    executor = _build_executor(state_runs_isolated)
    assert executor.poison_detector is not None, (
        "Hardening must be wired for this test to be meaningful"
    )

    # Phase A: record 2 same-pattern errors (below threshold=3).
    err_a1 = _make_classified_error(
        executor, "ConnectTimeout: upstream tool unreachable", "T-A1"
    )
    err_a2 = _make_classified_error(
        executor, "ConnectTimeout: upstream tool unreachable", "T-A2"
    )
    executor.poison_detector.record_failure(err_a1)
    executor.poison_detector.record_failure(err_a2)

    # Sanity: 2 errors recorded under one pattern hash, no trigger yet.
    pattern_hash = err_a1.pattern_hash
    assert err_a2.pattern_hash == pattern_hash, (
        "Same normalized message must collide on pattern_hash"
    )
    assert len(executor.poison_detector._errors[pattern_hash]) == 2

    # Phase B begins — execute_phase entry must reset the detector.
    results, gates_passed, gate_results = await executor.execute_phase(
        workflow_id="W-test-W2",
        phase_name="phase_b",
        phase_index=1,
        tasks=[],
        gate_configs=None,
        max_concurrent=1,
    )

    # After phase B starts, the detector must have a clean state.
    assert pattern_hash not in executor.poison_detector._errors or len(
        executor.poison_detector._errors[pattern_hash]
    ) == 0, (
        "W2 regression: execute_phase did NOT call reset_poison_detector(); "
        "phase A's pattern-hash bucket survived into phase B"
    )

    # Now phase B's first error fires. Threshold is 3 — 1 error must
    # NOT trip poison-pill.
    err_b1 = _make_classified_error(
        executor, "ConnectTimeout: upstream tool unreachable", "T-B1"
    )
    poison_result = executor.poison_detector.record_failure(err_b1)
    assert poison_result is None or not poison_result.triggered, (
        "W2 regression: phase B's first same-pattern error tripped poison-pill "
        "because detector state from phase A leaked across the boundary"
    )
    # And the per-pattern counter is back to 1 (just this phase's error).
    assert len(executor.poison_detector._errors[pattern_hash]) == 1


@pytest.mark.asyncio
async def test_reset_poison_detector_called_at_phase_entry(state_runs_isolated):
    """Spy on ``reset_poison_detector`` to assert it fires per phase entry."""
    executor = _build_executor(state_runs_isolated)

    call_count = {"n": 0}
    original_reset = executor.reset_poison_detector

    def _spy_reset():
        call_count["n"] += 1
        return original_reset()

    executor.reset_poison_detector = _spy_reset  # type: ignore[assignment]

    # Run two phases back-to-back.
    await executor.execute_phase(
        workflow_id="W-test-W2-spy",
        phase_name="phase_one",
        phase_index=0,
        tasks=[],
        max_concurrent=1,
    )
    await executor.execute_phase(
        workflow_id="W-test-W2-spy",
        phase_name="phase_two",
        phase_index=1,
        tasks=[],
        max_concurrent=1,
    )

    assert call_count["n"] == 2, (
        f"Expected reset_poison_detector to fire once per phase entry "
        f"(2 phases run), got {call_count['n']}"
    )


@pytest.mark.asyncio
async def test_within_phase_poison_detection_still_trips(state_runs_isolated):
    """Sanity check: W2's reset does NOT disable intra-phase detection.

    Three same-pattern errors recorded inside a single phase (i.e.
    after a single ``reset_poison_detector`` call) must still trip
    poison-pill at the threshold boundary. This pins Wave 38 semantics.
    """
    executor = _build_executor(state_runs_isolated)
    assert executor.poison_detector is not None

    # Simulate phase entry (the executor would normally do this inside
    # execute_phase). After this, the detector starts clean.
    executor.reset_poison_detector()

    pattern_msg = "ConnectTimeout: upstream tool unreachable"
    err_1 = _make_classified_error(executor, pattern_msg, "T-1")
    err_2 = _make_classified_error(executor, pattern_msg, "T-2")
    err_3 = _make_classified_error(executor, pattern_msg, "T-3")

    r1 = executor.poison_detector.record_failure(err_1)
    r2 = executor.poison_detector.record_failure(err_2)
    r3 = executor.poison_detector.record_failure(err_3)

    assert r1 is None or not r1.triggered, "1st error must not trip"
    assert r2 is None or not r2.triggered, "2nd error must not trip"
    assert r3 is not None and r3.triggered, (
        "3rd same-pattern error within one phase MUST trip poison-pill "
        "(Wave 38 semantics — W2 reset must not disable intra-phase detection)"
    )
    assert r3.occurrence_count == 3
