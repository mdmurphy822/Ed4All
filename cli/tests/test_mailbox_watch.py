"""Tests for the ``ed4all mailbox watch`` CLI command (Wave 34)."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cli.commands.mailbox_watch import MailboxWatcher  # noqa: E402
from cli.main import cli  # noqa: E402
from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402


class TestHelpWiring:
    def test_mailbox_appears_in_top_level_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "mailbox" in result.output

    def test_mailbox_watch_help_lists_flags(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["mailbox", "watch", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output
        assert "--exit-when-idle" in result.output


class TestExitWhenIdle:
    def test_empty_queue_with_flag_exits_cleanly(self, tmp_path: Path):
        """With no pending tasks and --exit-when-idle, the watcher should
        emit a header + idle event and exit 0 without blocking."""
        watcher = MailboxWatcher(
            run_id="RUN_W34_IDLE",
            base_dir=tmp_path,
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
            poll_interval=0.05,
            exit_when_idle=True,
        )
        exit_code = watcher.run()
        assert exit_code == 0
        lines = [
            json.loads(line)
            for line in watcher.stdout.getvalue().splitlines()
            if line
        ]
        kinds = [ev["kind"] for ev in lines]
        assert "header" in kinds
        assert "idle" in kinds


class TestTaskPickup:
    def test_three_pending_all_claimed_and_completed(self, tmp_path: Path):
        """Seed three pending tasks and stage three completion lines on
        stdin. Watcher must claim, emit a task event for each, consume
        matching completion, and write three completed/ files."""
        mb = TaskMailbox(run_id="RUN_W34_PICKUP", base_dir=tmp_path)
        task_ids = []
        for i in range(3):
            tid = f"task_{i:02d}"
            mb.put_pending(
                tid,
                {
                    "subagent_type": "content-generator",
                    "prompt": f"prompt {i}",
                    "phase_input": {
                        "phase_name": f"phase_{i}",
                        "run_id": "RUN_W34_PICKUP",
                    },
                },
            )
            task_ids.append(tid)

        # Completions: success envelopes that match each task_id.
        stdin_lines = "\n".join(
            json.dumps(
                {
                    "kind": "completion",
                    "task_id": tid,
                    "success": True,
                    "result": {
                        "run_id": "RUN_W34_PICKUP",
                        "phase_name": f"phase_{i}",
                        "status": "ok",
                        "outputs": {"completed_by": "test_harness"},
                    },
                }
            )
            for i, tid in enumerate(task_ids)
        ) + "\n"

        watcher = MailboxWatcher(
            run_id="RUN_W34_PICKUP",
            base_dir=tmp_path,
            stdin=io.StringIO(stdin_lines),
            stdout=io.StringIO(),
            poll_interval=0.02,
            exit_when_idle=True,
        )
        exit_code = watcher.run()
        assert exit_code == 0

        # Each task must have produced a completion file.
        assert sorted(mb.list_completed()) == sorted(task_ids)
        assert mb.list_pending() == []
        assert mb.list_in_progress() == []

        # Each envelope carries the success flag + result payload.
        for tid in task_ids:
            env = mb.read_completion(tid)
            assert env["success"] is True
            assert env["result"]["status"] == "ok"

        # Stdout stream carried one task event per task + header + idle.
        lines = [
            json.loads(line)
            for line in watcher.stdout.getvalue().splitlines()
            if line
        ]
        task_events = [ev for ev in lines if ev["kind"] == "task"]
        assert sorted(e["task_id"] for e in task_events) == sorted(task_ids)
        # Task events should carry the spec fields the runner needs.
        for ev in task_events:
            assert ev["subagent_type"] == "content-generator"
            assert ev["prompt"].startswith("prompt ")
            assert ev["phase_input"]["run_id"] == "RUN_W34_PICKUP"


class TestSigtermExitsCleanly:
    def test_request_stop_breaks_loop(self, tmp_path: Path):
        """Simulating SIGTERM via request_stop() while the watcher is
        polling an idle mailbox should exit cleanly."""
        watcher = MailboxWatcher(
            run_id="RUN_W34_SIG",
            base_dir=tmp_path,
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
            poll_interval=0.05,
            exit_when_idle=False,
        )

        def late_stop():
            time.sleep(0.15)
            watcher.request_stop()

        th = threading.Thread(target=late_stop)
        th.start()
        try:
            exit_code = watcher.run()
        finally:
            th.join(timeout=2.0)

        assert exit_code == 0
        lines = [
            json.loads(line)
            for line in watcher.stdout.getvalue().splitlines()
            if line
        ]
        # At minimum the header event is emitted before the loop exits.
        assert any(ev["kind"] == "header" for ev in lines)


class TestMalformedStdinHandling:
    def test_non_json_lines_skipped_without_dropping_task(self, tmp_path: Path):
        """Garbage lines on stdin should not crash the watcher or
        acknowledge a pending task. The real completion line that comes
        after must still be consumed."""
        mb = TaskMailbox(run_id="RUN_W34_JUNK", base_dir=tmp_path)
        mb.put_pending(
            "only_task",
            {
                "subagent_type": "content-generator",
                "prompt": "p",
                "phase_input": {
                    "phase_name": "p_only",
                    "run_id": "RUN_W34_JUNK",
                },
            },
        )

        stdin_lines = "\n".join(
            [
                "not json at all",
                json.dumps({"kind": "progress", "msg": "still thinking"}),
                json.dumps(
                    {
                        "kind": "completion",
                        "task_id": "only_task",
                        "success": True,
                        "result": {
                            "run_id": "RUN_W34_JUNK",
                            "phase_name": "p_only",
                            "status": "ok",
                            "outputs": {},
                        },
                    }
                ),
            ]
        ) + "\n"

        watcher = MailboxWatcher(
            run_id="RUN_W34_JUNK",
            base_dir=tmp_path,
            stdin=io.StringIO(stdin_lines),
            stdout=io.StringIO(),
            poll_interval=0.02,
            exit_when_idle=True,
        )
        exit_code = watcher.run()
        assert exit_code == 0
        env = mb.read_completion("only_task")
        assert env["success"] is True
