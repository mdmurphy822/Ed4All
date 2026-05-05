"""Wave W5 regression: ``TaskMailbox.await_completion_async`` does not
saturate the asyncio default thread pool under 10-way fanout.

The dispatcher previously wrapped the synchronous
``wait_for_completion`` in ``loop.run_in_executor``; under 10-way
phase fanout, every blocking ``time.sleep`` held a thread-pool slot
for the entire wait window. The async-native variant must yield the
event loop between polls so unrelated coroutines (and unrelated
``run_in_executor`` calls) make progress.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402


async def test_async_wait_does_not_block_event_loop(tmp_path: Path) -> None:
    """Spawn 10 concurrent waiters; assert a sentinel coroutine
    interleaves with them rather than queuing behind a thread-pool
    backlog.

    With the legacy ``run_in_executor`` path, 10 concurrent waiters
    each holding a default-pool slot for the full poll interval
    starved unrelated executor work. The async-native variant uses
    ``asyncio.sleep`` and never touches the executor, so a sentinel
    coroutine scheduled alongside the waiters runs within one poll
    interval of dispatch.
    """
    mb = TaskMailbox(run_id="RUN_NB", base_dir=tmp_path)
    poll_interval = 0.05
    timeout = 2.0

    # Pre-fill the default thread pool with blocking work to make the
    # starvation regression detectable: if the async waiter regressed
    # to using ``run_in_executor`` internally, it would queue behind
    # these blockers and miss the sentinel deadline.
    loop = asyncio.get_running_loop()
    pool_blockers = [
        loop.run_in_executor(None, lambda: time.sleep(timeout))
        for _ in range(16)
    ]

    task_ids = [f"task_{i}" for i in range(10)]
    for tid in task_ids:
        mb.put_pending(tid, {"k": "v"})

    sentinel_dispatched: float = 0.0
    sentinel_ran: float = 0.0

    async def sentinel() -> None:
        nonlocal sentinel_ran
        # Yield once so the gather()'d waiters get a chance to start
        # polling first; then record when we actually run.
        await asyncio.sleep(0)
        sentinel_ran = time.monotonic()

    async def writer_after(delay: float) -> None:
        # Resolves all waiters after ``delay`` so the test terminates.
        await asyncio.sleep(delay)
        for tid in task_ids:
            mb.complete(tid, {"success": True, "result": tid})

    sentinel_dispatched = time.monotonic()
    waiters = [
        mb.await_completion_async(
            tid, timeout_seconds=timeout, poll_interval=poll_interval
        )
        for tid in task_ids
    ]
    results = await asyncio.gather(
        sentinel(),
        writer_after(poll_interval * 3),
        *waiters,
    )

    # Cancel the pre-filled pool blockers so the test exits cleanly.
    for fut in pool_blockers:
        fut.cancel()

    # Sentinel must have run within ~1 poll interval — well before
    # the writer resolves the waiters. If we regressed to executor-
    # backed blocking, the sentinel would either queue behind the
    # 16 pool blockers (~timeout seconds) or interleave only after
    # the waiters returned.
    sentinel_latency = sentinel_ran - sentinel_dispatched
    assert sentinel_latency < poll_interval * 2, (
        f"sentinel latency {sentinel_latency:.4f}s exceeds "
        f"{poll_interval * 2:.4f}s — event loop appears blocked"
    )

    # All 10 waiters returned their respective completion envelopes.
    waiter_results = results[2:]
    assert len(waiter_results) == 10
    for tid, env in zip(task_ids, waiter_results):
        assert env["success"] is True
        assert env["result"] == tid


async def test_async_wait_returns_on_completion(tmp_path: Path) -> None:
    """The async waiter resolves promptly once the completion file
    is written, mirroring ``wait_for_completion``'s contract."""
    mb = TaskMailbox(run_id="RUN_OK", base_dir=tmp_path)
    mb.put_pending("hello", {"k": "v"})

    def late_writer() -> None:
        time.sleep(0.1)
        mb.complete("hello", {"success": True, "result": "world"})

    th = threading.Thread(target=late_writer)
    th.start()
    try:
        t0 = time.monotonic()
        envelope = await mb.await_completion_async(
            "hello", timeout_seconds=5.0, poll_interval=0.02
        )
        elapsed = time.monotonic() - t0
    finally:
        th.join()

    assert envelope["success"] is True
    assert envelope["result"] == "world"
    # Should resolve within a few poll intervals after the writer runs.
    assert elapsed < 1.0, f"async waiter took {elapsed:.3f}s; expected <1.0s"


async def test_async_wait_respects_timeout(tmp_path: Path) -> None:
    """Without a completion file, the async waiter raises
    ``TimeoutError`` near the deadline (not later)."""
    mb = TaskMailbox(run_id="RUN_TO", base_dir=tmp_path)
    mb.put_pending("never", {"k": "v"})

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        await mb.await_completion_async(
            "never", timeout_seconds=0.2, poll_interval=0.02
        )
    elapsed = time.monotonic() - t0

    # Allow generous wall-clock slack for CI: deadline is 0.2s,
    # accept up to ~0.6s before flagging a regression.
    assert 0.15 <= elapsed <= 0.6, (
        f"timeout fired at {elapsed:.3f}s; expected ~0.2s"
    )
