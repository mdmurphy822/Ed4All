"""Wave 73 operator-side helper for the ``MailboxBrokeredBackend`` bridge.

Running ``ed4all run ... --mode local`` starts a pipeline subprocess whose
LLM call sites (DART alt-text, block classifier, Trainforge align_chunks)
write pending ``kind="llm_call"`` tasks to
``state/runs/{run_id}/mailbox/pending/``.  A Claude Code session drives
the other side of the bridge: it reads each pending task, dispatches a
Claude completion, and writes the completion envelope to
``completed/{task_id}.json``.

This module exposes two small helpers designed to be called from a
Claude Code session via Bash tool calls — keeping the LLM dispatch on
the caller side (where the Claude Code ``Agent`` tool lives) and the
file-system plumbing on the Python side:

``ed4all mailbox peek --run-id RUN [--max N]``
    Claim up to N pending tasks from the mailbox and emit their specs
    as a JSON array to stdout. Exits 0 with ``[]`` when no pending
    tasks exist. Sets an upper bound (``--max``, default 1) so the
    caller can drive parallelism by looping and calling peek+complete
    in small batches rather than grabbing everything at once.

``ed4all mailbox complete --run-id RUN --task-id T --text-file F``
    Write a success completion envelope for ``T`` with the contents of
    ``F`` as ``response_text``.  Use ``--text`` for short inline
    responses instead of ``--text-file``.  Use ``--error`` to write a
    failure envelope instead.

The separation of peek / complete (instead of a single long-running
``ed4all mailbox watch`` process) lets the caller perform the LLM
dispatch using whatever mechanism they have available — typically a
Claude Code ``Agent`` tool call in the outer session — without having
to negotiate a live JSON-over-stdin protocol.  The ``watch`` command
from Wave 34 remains for automated operator scripts.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MCP.orchestrator.task_mailbox import (  # noqa: E402
    TaskClaimConflict,
    TaskMailbox,
    TaskNotFoundError,
)


def _mailbox(run_id: str, base_dir: Optional[str]) -> TaskMailbox:
    base_path = Path(base_dir) if base_dir else None
    return TaskMailbox(run_id=run_id, base_dir=base_path)


@click.group(name="mailbox-bridge", hidden=True)
def mailbox_bridge_group():
    """Wave 73 peek/complete helpers for the MailboxBrokeredBackend bridge."""


@mailbox_bridge_group.command("peek")
@click.option(
    "--run-id", required=True,
    help="Workflow run id — determines the state/runs/{run_id}/mailbox/ path.",
)
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Override state/runs parent dir (tests only).",
)
@click.option(
    "--max",
    "max_tasks",
    type=int,
    default=1,
    show_default=True,
    help="Maximum number of pending tasks to claim + return in this call.",
)
def mailbox_peek(run_id: str, base_dir: Optional[str], max_tasks: int):
    """Claim up to ``--max`` pending tasks and print them as a JSON array.

    Each element has shape:

        {"task_id": "...", "spec": {"kind": "...", ...}}

    Prints ``[]`` when nothing is pending. Tasks successfully claimed
    are moved into ``in_progress/`` and must be completed (or re-queued)
    by a later ``mailbox-bridge complete`` call; otherwise they pin
    the pipeline's ``wait_for_completion`` until its timeout fires.
    """
    mb = _mailbox(run_id, base_dir)
    claimed = []
    pending = mb.list_pending()
    for task_id in pending[:max(1, max_tasks)]:
        try:
            spec = mb.claim(task_id)
        except (TaskNotFoundError, TaskClaimConflict):
            # Another bridge instance or a stale completion beat us to
            # the claim — just skip and surface what we have so the
            # caller can decide whether to loop.
            continue
        claimed.append({"task_id": task_id, "spec": spec})

    click.echo(json.dumps(claimed, indent=2, default=str))


@mailbox_bridge_group.command("complete")
@click.option(
    "--run-id", required=True,
    help="Workflow run id.",
)
@click.option(
    "--task-id", required=True,
    help="Task id previously emitted by peek.",
)
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Override state/runs parent dir (tests only).",
)
@click.option(
    "--text",
    default=None,
    help="Inline completion text. Mutually exclusive with --text-file.",
)
@click.option(
    "--text-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a file whose contents become the response_text.",
)
@click.option(
    "--error",
    default=None,
    help="When set, writes a failure envelope with this error message instead "
         "of a success envelope.",
)
@click.option(
    "--error-code",
    default=None,
    help="Optional classifier tag to pair with --error.",
)
def mailbox_complete(
    run_id: str,
    task_id: str,
    base_dir: Optional[str],
    text: Optional[str],
    text_file: Optional[str],
    error: Optional[str],
    error_code: Optional[str],
):
    """Write a completion envelope for ``task_id`` to ``mailbox/completed/``.

    Success envelope shape (matches ``MailboxBrokeredBackend._text_from_envelope``):

        {"success": true, "result": {"response_text": "<str>"}}

    Failure envelope shape (triggers backend-side RuntimeError):

        {"success": false, "error": "<msg>", "error_code": "<tag>"}
    """
    if error is not None and (text is not None or text_file is not None):
        raise click.UsageError(
            "--error is mutually exclusive with --text / --text-file",
        )
    if error is None and text is None and text_file is None:
        raise click.UsageError(
            "One of --text, --text-file, or --error is required",
        )

    mb = _mailbox(run_id, base_dir)

    if error is not None:
        envelope = {"success": False, "error": error}
        if error_code:
            envelope["error_code"] = error_code
    else:
        if text_file is not None:
            response_text = Path(text_file).read_text(encoding="utf-8")
        else:
            response_text = text or ""
        envelope = {
            "success": True,
            "result": {"response_text": response_text},
        }

    mb.complete(task_id, envelope)
    click.echo(json.dumps({"ok": True, "task_id": task_id}))


def register_mailbox_bridge_command(cli_group: click.Group) -> None:
    """Attach the ``mailbox-bridge`` subgroup to the top-level CLI."""
    cli_group.add_command(mailbox_bridge_group)


__all__ = [
    "mailbox_bridge_group",
    "mailbox_peek",
    "mailbox_complete",
    "register_mailbox_bridge_command",
]
