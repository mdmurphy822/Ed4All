"""P3.M / AMENDMENT #6 — graceful-stop race in the TaskMailbox waiters.

A stopped orchestrator must not sit out the full mailbox timeout
(``ED4ALL_AGENT_TIMEOUT_SECONDS`` / ``MailboxBrokeredBackend`` 120s) waiting
for a completion that will never arrive. Both the sync and async waiters race
the completion-poll against the stop sentinel at every wakeup: on stop they
raise ``GracefulStopRequested`` (the pause channel) — never ``TimeoutError``
(``MAILBOX_TIMEOUT``). Instrumented at the mailbox level so every waiter,
including ``MailboxBrokeredBackend``, inherits it.

The sentinel is resolved through ``stop_control`` (``get_state_runs_dir()``),
so these tests pin the mailbox dir + sentinel dir together via the
``state_runs_isolated`` fixture (sets ``ED4ALL_STATE_RUNS_DIR``).
"""
from __future__ import annotations

import threading
import time

import pytest

from lib.generation import stop_control
from lib.generation.stop_control import GracefulStopRequested
from MCP.orchestrator.task_mailbox import TaskMailbox


def _mailbox(run_id: str) -> TaskMailbox:
    # base_dir=None → TaskMailbox reads ED4ALL_STATE_RUNS_DIR (set by the
    # state_runs_isolated fixture), so the mailbox and the sentinel resolve
    # to the same state/runs parent.
    return TaskMailbox(run_id=run_id)


# ------------------------------------------------------------ pre-armed


async def test_async_wait_pre_armed_sentinel_raises_immediately(
    state_runs_isolated,
):
    """A sentinel already up when the wait starts → GracefulStopRequested
    on the first poll, well before the (large) timeout, and NOT a
    TimeoutError."""
    run_id = "RUN_PREARM"
    mb = _mailbox(run_id)
    mb.put_pending("never", {"k": "v"})
    stop_control.request_stop(run_id=run_id, reason="test", source="test")

    t0 = time.monotonic()
    with pytest.raises(GracefulStopRequested) as ei:
        await mb.await_completion_async(
            "never", timeout_seconds=30.0, poll_interval=0.02
        )
    elapsed = time.monotonic() - t0

    assert ei.value.site_id == "mailbox_wait:never"
    assert elapsed < 1.0, f"paused only after {elapsed:.3f}s; expected ~immediate"


def test_sync_wait_pre_armed_global_sentinel_raises(state_runs_isolated):
    """The global STOP_ALL sentinel (no run_id) trips the sync waiter too."""
    run_id = "RUN_GLOBAL"
    mb = _mailbox(run_id)
    mb.put_pending("never", {"k": "v"})
    stop_control.request_stop(scope="all", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        mb.wait_for_completion("never", timeout_seconds=30.0, poll_interval=0.02)


# ------------------------------------------------------------ mid-wait race


async def test_async_wait_sentinel_written_mid_wait_returns_paused(
    state_runs_isolated,
):
    """Sentinel armed mid-wait → the waiter pauses within ~one poll interval
    of the write, far below the large timeout (poll-race, not timeout)."""
    run_id = "RUN_MID"
    mb = _mailbox(run_id)
    mb.put_pending("slow", {"k": "v"})

    arm_delay = 0.15

    def arm_later() -> None:
        time.sleep(arm_delay)
        stop_control.request_stop(run_id=run_id, reason="mid", source="test")

    th = threading.Thread(target=arm_later, daemon=True)
    th.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(GracefulStopRequested):
            await mb.await_completion_async(
                "slow", timeout_seconds=30.0, poll_interval=0.02
            )
        elapsed = time.monotonic() - t0
    finally:
        th.join(timeout=2.0)

    # Paused shortly after the sentinel appeared — nowhere near the 30s
    # timeout. Generous upper bound for CI slack.
    assert arm_delay - 0.05 <= elapsed <= 5.0, (
        f"paused at {elapsed:.3f}s; expected ~{arm_delay}s (mid-wait race)"
    )


# --------------------------------------------------- completion still wins


async def test_completion_wins_over_pending_stop(state_runs_isolated):
    """If the completion file is already present, the waiter returns it even
    when a stop sentinel is also up — a finished task is never discarded."""
    run_id = "RUN_DONE"
    mb = _mailbox(run_id)
    mb.put_pending("done", {"k": "v"})
    mb.complete("done", {"success": True, "result": "payload"})
    stop_control.request_stop(run_id=run_id, reason="late", source="test")

    envelope = await mb.await_completion_async(
        "done", timeout_seconds=5.0, poll_interval=0.02
    )
    assert envelope["success"] is True
    assert envelope["result"] == "payload"


# ------------------------------------------------------ not a MAILBOX_TIMEOUT


async def test_paused_outcome_is_not_timeout(state_runs_isolated):
    """The paused outcome is a distinct exception type — TimeoutError is NOT
    raised (so downstream never classifies it as MAILBOX_TIMEOUT)."""
    run_id = "RUN_NOTTO"
    mb = _mailbox(run_id)
    mb.put_pending("never", {"k": "v"})
    stop_control.request_stop(run_id=run_id, reason="t", source="test")

    with pytest.raises(GracefulStopRequested):
        await mb.await_completion_async(
            "never", timeout_seconds=0.2, poll_interval=0.02
        )
    # And with no sentinel + no completion, the ordinary timeout still fires.
    stop_control.clear_stop(run_id=run_id)
    with pytest.raises(TimeoutError):
        await mb.await_completion_async(
            "never", timeout_seconds=0.2, poll_interval=0.02
        )
