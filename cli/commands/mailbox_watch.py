"""
``ed4all mailbox watch`` — outer-session watcher for the TaskMailbox bridge
(Wave 34).

When the orchestrator runs in ``--mode local`` without an ``agent_tool``
callable, ``LocalDispatcher`` writes each phase task to a file-based
``TaskMailbox`` under ``state/runs/{run_id}/mailbox/pending/`` and blocks
on ``wait_for_completion``. The outer Claude Code session (or any cooperating
process) needs to:

  1. Poll ``pending/`` for new task files.
  2. Claim a task (atomic move into ``in_progress/``).
  3. Dispatch a real subagent via the MCP ``Agent`` tool using the
     ``prompt`` + ``subagent_type`` carried in the task spec.
  4. Write a completion envelope to ``completed/``.

This CLI implements a generic watcher with a **stdin/stdout JSON
protocol** so the operator-facing "runner" (whoever has the ``Agent``
tool available — typically a Claude Code session) can plug in without
depending on this module's internals.

Protocol
--------

Every pending task causes the watcher to print a single JSON line to
stdout (with a ``"kind": "task"`` tag). The runner reads that line,
dispatches the subagent, and feeds a JSON completion line back on stdin
(``"kind": "completion"``). The watcher writes it to the mailbox.

  STDOUT (watcher -> runner):
      {"kind": "task", "task_id": "...", "subagent_type": "...",
       "prompt": "...", "phase_input": {...}}

  STDIN (runner -> watcher):
      {"kind": "completion", "task_id": "...", "success": true,
       "result": {...}}   # or "error": "...", "error_code": "..."

Alternative API use
-------------------

Callers that prefer to drive the mailbox directly (no stdio protocol)
can import :class:`TaskMailbox` from ``MCP.orchestrator.task_mailbox``
and call ``list_pending`` / ``claim`` / ``complete`` themselves. This
CLI is only one of several valid outer-session shapes.

Exit conditions
---------------

The watcher loop exits when:

  * SIGTERM / SIGINT is received.
  * ``--run-id`` is not supplied.
  * ``--exit-when-idle`` is passed and the pending + in_progress
    queues are both empty (useful for CI / one-shot smokes).

The watcher does NOT attempt to detect "workflow complete" on its own —
the orchestrator knows that. Operators typically scope one watcher
per run and stop it when the run finishes.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Dict, Optional

import click

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402


class MailboxWatcher:
    """Loop body for the ``ed4all mailbox watch`` command.

    Split out of the Click handler so it's unit-testable.
    """

    def __init__(
        self,
        run_id: str,
        *,
        base_dir: Optional[Path] = None,
        stdin: Optional[IO[str]] = None,
        stdout: Optional[IO[str]] = None,
        poll_interval: float = 1.0,
        exit_when_idle: bool = False,
    ):
        self.run_id = run_id
        self.mailbox = TaskMailbox(
            run_id=run_id,
            base_dir=base_dir,
        )
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.poll_interval = float(poll_interval)
        self.exit_when_idle = bool(exit_when_idle)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ api

    def request_stop(self) -> None:
        """Signal the watch loop to exit at the next opportunity."""
        self._stop.set()

    def run(self) -> int:
        """Main loop. Returns an exit code (0 = clean, 1 = aborted)."""
        self._emit_header()
        while not self._stop.is_set():
            pending = self.mailbox.list_pending()
            for task_id in pending:
                if self._stop.is_set():
                    break
                self._handle_task(task_id)

            if self._stop.is_set():
                break

            if self.exit_when_idle and not pending and not self.mailbox.list_in_progress():
                self._emit({"kind": "idle", "run_id": self.run_id})
                return 0

            time.sleep(self.poll_interval)
        return 0

    # ----------------------------------------------------------- internals

    def _handle_task(self, task_id: str) -> None:
        try:
            spec = self.mailbox.claim(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("watcher: could not claim %s: %s", task_id, exc)
            return

        task_event = {
            "kind": "task",
            "task_id": task_id,
            "run_id": self.run_id,
            "subagent_type": spec.get("subagent_type"),
            "prompt": spec.get("prompt"),
            "phase_input": spec.get("phase_input"),
        }
        self._emit(task_event)

        envelope = self._read_completion(task_id)
        if envelope is None:
            envelope = {
                "success": False,
                "error": "watcher stdin closed before completion arrived",
                "error_code": "WATCHER_STDIN_EOF",
            }
        try:
            self.mailbox.complete(task_id, envelope)
        except Exception as exc:  # noqa: BLE001
            logger.exception("watcher: failed to write completion for %s", task_id)
            # Try to surface the error so callers aren't silent on disk failures
            sys.stderr.write(
                f"mailbox_watch: failed to write completion for {task_id}: {exc}\n"
            )

    def _read_completion(self, expected_task_id: str) -> Optional[Dict[str, Any]]:
        """Read JSON lines from stdin until a completion for the given
        task arrives. Lines that don't parse or don't match are skipped
        with a warning so the runner can stream progress events safely.
        """
        while not self._stop.is_set():
            line = self.stdin.readline()
            if not line:
                return None  # EOF
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stderr.write(
                    f"mailbox_watch: dropping non-JSON stdin line "
                    f"({exc}): {line[:80]!r}\n"
                )
                continue
            if not isinstance(payload, dict):
                sys.stderr.write("mailbox_watch: dropping non-object stdin payload\n")
                continue
            if payload.get("kind") != "completion":
                continue
            task_id = payload.get("task_id")
            if task_id != expected_task_id:
                sys.stderr.write(
                    f"mailbox_watch: ignoring completion for unexpected task "
                    f"{task_id!r} (waiting on {expected_task_id!r})\n"
                )
                continue
            # Strip the kind/task_id wrapper before passing to mailbox.complete.
            envelope = {k: v for k, v in payload.items() if k not in ("kind", "task_id")}
            return envelope
        return None

    def _emit_header(self) -> None:
        self._emit({
            "kind": "header",
            "run_id": self.run_id,
            "mailbox_root": str(self.mailbox.root),
            "exit_when_idle": self.exit_when_idle,
        })

    def _emit(self, payload: Dict[str, Any]) -> None:
        self.stdout.write(json.dumps(payload, default=str) + "\n")
        self.stdout.flush()


# -------------------------------------------------------------- click wiring


@click.group(name="mailbox")
def mailbox_group():
    """TaskMailbox operator commands (Wave 34)."""


@mailbox_group.command("watch")
@click.option(
    "--run-id",
    required=True,
    help="Workflow run id; determines the state/runs/{run_id}/mailbox/ path.",
)
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Override state/runs parent dir (tests only).",
)
@click.option(
    "--poll-interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between pending/ scans when idle.",
)
@click.option(
    "--exit-when-idle",
    is_flag=True,
    default=False,
    help="Exit when pending + in_progress are both empty (one-shot mode).",
)
def mailbox_watch(
    run_id: str,
    base_dir: Optional[str],
    poll_interval: float,
    exit_when_idle: bool,
):
    """Watch a TaskMailbox and route subagent tasks via stdio.

    For each pending task the watcher prints a JSON task line to stdout
    and waits for a JSON completion line on stdin. See module docstring
    for the wire format. SIGTERM / SIGINT cleanly terminate the loop.
    """
    watcher = MailboxWatcher(
        run_id=run_id,
        base_dir=Path(base_dir) if base_dir else None,
        poll_interval=poll_interval,
        exit_when_idle=exit_when_idle,
    )

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info("mailbox_watch: signal %s received, stopping", signum)
        watcher.request_stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):  # pragma: no cover
            pass  # not in main thread (tests) — they drive stop() directly

    exit_code = watcher.run()
    sys.exit(exit_code)


def register_mailbox_command(cli_group: click.Group) -> None:
    """Attach the ``mailbox`` subgroup to the top-level ``ed4all`` group."""
    cli_group.add_command(mailbox_group)


__all__ = [
    "MailboxWatcher",
    "mailbox_group",
    "mailbox_watch",
    "register_mailbox_command",
]
