"""Wave 34 tests: TaskMailbox file-based task handoff primitives."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.task_mailbox import (  # noqa: E402
    MailboxError,
    TaskClaimConflict,
    TaskMailbox,
    TaskNotFoundError,
)


class TestRoundTrip:
    def test_put_list_claim_complete_round_trip(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_RT", base_dir=tmp_path)
        assert mb.pending_count() == 0

        mb.put_pending(
            "task_one",
            {"prompt": "do the thing", "phase": "content_generation"},
        )
        assert mb.list_pending() == ["task_one"]
        assert mb.in_progress_count() == 0

        spec = mb.claim("task_one")
        assert spec["prompt"] == "do the thing"
        assert spec["task_id"] == "task_one"
        assert mb.list_pending() == []
        assert mb.list_in_progress() == ["task_one"]

        mb.complete("task_one", {"success": True, "result": {"emitted": 4}})
        assert mb.list_completed() == ["task_one"]
        assert mb.list_in_progress() == []

        payload = mb.read_completion("task_one")
        assert payload["success"] is True
        assert payload["result"] == {"emitted": 4}


class TestConcurrentClaim:
    def test_two_claimers_one_wins(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_CC", base_dir=tmp_path)
        mb.put_pending("race_task", {"payload": "x"})

        winners: list[str] = []
        errors: list[Exception] = []

        def try_claim():
            try:
                spec = mb.claim("race_task")
                winners.append(spec["task_id"])
            except (TaskNotFoundError, TaskClaimConflict) as exc:
                errors.append(exc)

        t1 = threading.Thread(target=try_claim)
        t2 = threading.Thread(target=try_claim)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one claimer should succeed; the other must raise a
        # recognised mailbox error (not crash the process).
        assert len(winners) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], (TaskNotFoundError, TaskClaimConflict))


class TestWaitForCompletion:
    def test_completion_written_returns_immediately(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_W", base_dir=tmp_path)
        mb.put_pending("t", {"k": "v"})
        mb.complete("t", {"success": True, "result": "ok"})
        t0 = time.monotonic()
        payload = mb.wait_for_completion("t", timeout_seconds=5.0)
        assert payload["success"] is True
        assert time.monotonic() - t0 < 1.0  # should not have polled long

    def test_timeout_raises(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_T", base_dir=tmp_path)
        mb.put_pending("never", {"k": "v"})
        with pytest.raises(TimeoutError):
            mb.wait_for_completion(
                "never", timeout_seconds=0.1, poll_interval=0.02
            )

    def test_completion_written_after_delay(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_D", base_dir=tmp_path)
        mb.put_pending("later", {"k": "v"})

        def late_writer():
            time.sleep(0.1)
            mb.complete("later", {"success": True, "result": "late"})

        th = threading.Thread(target=late_writer)
        th.start()
        try:
            payload = mb.wait_for_completion(
                "later", timeout_seconds=5.0, poll_interval=0.02
            )
        finally:
            th.join()
        assert payload["result"] == "late"


class TestCleanup:
    def test_cleanup_removes_all_state(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_CL", base_dir=tmp_path)
        mb.put_pending("gone", {"k": "v"})
        mb.claim("gone")
        mb.complete("gone", {"success": True, "result": "done"})
        assert mb.list_completed() == ["gone"]

        mb.cleanup("gone")
        assert mb.list_pending() == []
        assert mb.list_in_progress() == []
        assert mb.list_completed() == []


class TestAtomicWrites:
    def test_partial_files_never_visible(self, tmp_path: Path):
        """The put_pending write must not leave a readable partial file.

        We can't easily simulate a crash mid-write, but we can verify the
        temp-file convention: no ``.tmp`` files remain after a clean put.
        """
        mb = TaskMailbox(run_id="RUN_A", base_dir=tmp_path)
        mb.put_pending("atomic", {"payload": "x" * 1024})
        leftover = list(mb.pending_dir.glob(".*.tmp"))
        assert leftover == []

        # list_pending must ignore dotfiles even if one appears mid-write.
        (mb.pending_dir / ".inflight.json.tmp").write_text("{}", encoding="utf-8")
        assert "inflight" not in mb.list_pending()


class TestTaskIdValidation:
    def test_rejects_empty_and_path_separators(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_V", base_dir=tmp_path)
        with pytest.raises(ValueError):
            mb.put_pending("", {"x": 1})
        with pytest.raises(ValueError):
            mb.put_pending("../escape", {"x": 1})
        with pytest.raises(ValueError):
            mb.put_pending("a/b", {"x": 1})

    def test_missing_completion_raises(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_M", base_dir=tmp_path)
        with pytest.raises(TaskNotFoundError):
            mb.read_completion("never_existed")
