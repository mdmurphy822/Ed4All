"""Wave 73 / Wave 74 operator-side helper for the mailbox bridge.

Running ``ed4all run ... --mode local`` starts a pipeline subprocess
whose LLM call sites and per-task subagent dispatches write pending
specs to ``state/runs/{run_id}/mailbox/pending/``. A Claude Code
session drives the other side of the bridge: it reads each pending
task, dispatches a completion, and writes the result envelope to
``completed/{task_id}.json``.

There are two shapes of pending tasks:

* ``kind="llm_call"`` (Wave 73) — emitted by ``MailboxBrokeredBackend``
  when DART alt-text / block classifier / Trainforge align_chunks
  need an LLM completion. Operators reply with a ``response_text``
  string.
* ``kind="agent_task"`` (Wave 74) — emitted by
  ``LocalDispatcher.dispatch_task`` when a subagent-classified phase
  task is routed through per-task dispatch. Operators reply with a
  tool-shape result (``{"success": bool, "artifacts": [...],
  "outputs": {...}, ...}``) wrapped in a ``{"success": True,
  "result": ...}`` envelope.

This module exposes four helpers callable via ``ed4all
mailbox-bridge`` subcommands. Pair them by kind — ``peek``/``complete``
for ``llm_call`` items and ``peek-agent``/``complete-agent`` for
``agent_task`` items. The kind filter is enforced on the peek side so
a ``peek`` call never accidentally claims an agent task (and
vice-versa).

``ed4all mailbox-bridge peek --run-id RUN [--max N]``
    Claim up to N pending ``kind="llm_call"`` tasks and emit their
    specs as a JSON array. Non-llm_call pendings are left untouched.

``ed4all mailbox-bridge complete --run-id RUN --task-id T --text-file F``
    Write a success completion envelope for an llm_call task.

``ed4all mailbox-bridge peek-agent --run-id RUN [--max N]``
    Claim up to N pending ``kind="agent_task"`` tasks and emit their
    specs (flattened, no nested ``spec`` key — each element is the
    task fields plus ``task_id``). Non-agent-task pendings are left
    untouched.

``ed4all mailbox-bridge complete-agent --run-id RUN --task-id T --result-file F``
    Write a success completion envelope for an agent_task. F must be a
    JSON object with at minimum ``success``, ``artifacts``, and
    ``outputs`` keys. Use ``--error MSG [--error-code CODE]`` to write
    a failure envelope instead.

The separation of peek / complete (instead of a single long-running
``ed4all mailbox watch`` process) lets the caller perform the LLM /
subagent dispatch using whatever mechanism they have available —
typically a Claude Code ``Agent`` tool call in the outer session —
without having to negotiate a live JSON-over-stdin protocol.  The
``watch`` command from Wave 34 remains for automated operator scripts.
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


def _read_pending_spec(mb: TaskMailbox, task_id: str) -> Optional[dict]:
    """Read a pending task's on-disk spec WITHOUT claiming it.

    Used by kind-filtered peek helpers so we can skip non-matching
    items instead of claiming + releasing them (``TaskMailbox`` has no
    first-class "release" operation — once claimed, an item is locked
    until a completion envelope lands).

    Returns ``None`` if the file disappeared (winner raced us) or if
    the contents aren't JSON.
    """
    target = mb.pending_dir / f"{task_id}.json"
    try:
        text = target.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable pending spec: %s", target)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _peek_by_kind(
    mb: TaskMailbox,
    *,
    want_kind: str,
    accept_missing: bool,
    max_tasks: int,
):
    """Claim up to ``max_tasks`` pending items whose spec kind matches.

    Items whose kind does not match are left pending (no claim, no
    release) so the complementary peek subcommand can pick them up.
    Returns the canonical ``peek`` shape: list of
    ``{"task_id": ..., "spec": ...}`` dicts.
    """
    claimed = []
    limit = max(1, max_tasks)
    for task_id in mb.list_pending():
        if len(claimed) >= limit:
            break
        spec_preview = _read_pending_spec(mb, task_id)
        if spec_preview is None:
            continue
        kind = spec_preview.get("kind")
        if kind != want_kind:
            if kind is None and accept_missing:
                pass  # treat as legacy llm_call
            else:
                continue
        try:
            spec = mb.claim(task_id)
        except (TaskNotFoundError, TaskClaimConflict):
            # Another bridge instance or a stale completion beat us to
            # the claim — just skip and surface what we have so the
            # caller can decide whether to loop.
            continue
        claimed.append({"task_id": task_id, "spec": spec})
    return claimed


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
    """Claim up to ``--max`` pending llm_call tasks and print them.

    Each element has shape:

        {"task_id": "...", "spec": {"kind": "llm_call", ...}}

    Only items whose pending spec has ``kind="llm_call"`` are claimed —
    agent-task items are left untouched for ``peek-agent`` to pick up.
    Prints ``[]`` when nothing matching is pending. Tasks successfully
    claimed are moved into ``in_progress/`` and must be completed (or
    re-queued) by a later ``mailbox-bridge complete`` call; otherwise
    they pin the pipeline's ``wait_for_completion`` until its timeout
    fires.
    """
    mb = _mailbox(run_id, base_dir)
    claimed = _peek_by_kind(
        mb,
        want_kind="llm_call",
        # llm_call is the historical default; treat specs missing a
        # ``kind`` field as llm_call so pre-Wave-74 producers keep
        # working without needing a schema migration.
        accept_missing=True,
        max_tasks=max_tasks,
    )
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


# ------------------------------------------------------------------- agent

# Wave 74 Session 2: per-task subagent dispatch bridge. When
# ``ED4ALL_AGENT_DISPATCH=true`` and an outer Claude Code session is
# driving the mailbox, ``LocalDispatcher.dispatch_task`` writes
# ``kind="agent_task"`` specs that these subcommands drive.


@mailbox_bridge_group.command("peek-agent")
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
    help="Maximum number of pending agent tasks to claim + return in this call.",
)
def mailbox_peek_agent(run_id: str, base_dir: Optional[str], max_tasks: int):
    """Claim up to ``--max`` pending ``kind="agent_task"`` items.

    Emits a JSON array of flattened task dicts — each element carries
    the fields an outer operator needs to dispatch the right subagent:

        {"task_id": "<agent_type>-<uuid12>",
         "agent_type": "content-generator",
         "tool_name": "generate_course_content",
         "task_params": {...},
         "agent_spec_path": "Courseforge/agents/content-generator.md",
         "run_id": "...",
         "created_at": <unix_ts>,
         "phase_context": {...}}

    Non-agent-task pendings (e.g. ``kind="llm_call"``) are left
    untouched. Prints ``[]`` when no matching pending tasks exist.
    Claimed tasks move into ``in_progress/`` and must be completed (or
    re-queued) by a later ``complete-agent`` call; otherwise they pin
    ``LocalDispatcher.dispatch_task``'s ``wait_for_completion`` until
    the ``ED4ALL_AGENT_TIMEOUT_SECONDS`` timeout fires.
    """
    mb = _mailbox(run_id, base_dir)
    claimed = _peek_by_kind(
        mb,
        want_kind="agent_task",
        accept_missing=False,
        max_tasks=max_tasks,
    )

    # Flatten the spec shape for easier operator scripting — they just
    # want to read task_id + agent_type + task_params + agent_spec_path
    # without navigating nested ``spec`` keys.
    flattened = []
    for entry in claimed:
        spec = entry["spec"]
        out = {
            "task_id": entry["task_id"],
            "agent_type": spec.get("agent_type"),
            "tool_name": spec.get("tool_name"),
            "task_params": spec.get("task_params", {}),
            "agent_spec_path": spec.get("agent_spec_path"),
            "run_id": spec.get("run_id", run_id),
            "created_at": spec.get("created_at"),
            "phase_context": spec.get("phase_context", {}),
        }
        flattened.append(out)

    click.echo(json.dumps(flattened, indent=2, default=str))


@mailbox_bridge_group.command("complete-agent")
@click.option(
    "--run-id", required=True,
    help="Workflow run id.",
)
@click.option(
    "--task-id", required=True,
    help="Task id previously emitted by peek-agent.",
)
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Override state/runs parent dir (tests only).",
)
@click.option(
    "--result-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a JSON file whose contents become the tool-shape result. "
         "Must be a JSON object with at least 'success', 'artifacts', "
         "'outputs' keys.",
)
@click.option(
    "--error",
    default=None,
    help="When set, writes a failure envelope with this error message "
         "instead of a success envelope. Mutually exclusive with --result-file.",
)
@click.option(
    "--error-code",
    default=None,
    help="Optional classifier tag to pair with --error. "
         "Defaults to OPERATOR_ERROR when --error is set.",
)
def mailbox_complete_agent(
    run_id: str,
    task_id: str,
    base_dir: Optional[str],
    result_file: Optional[str],
    error: Optional[str],
    error_code: Optional[str],
):
    """Write a completion envelope for a ``kind="agent_task"`` item.

    Success envelope shape (matches
    ``LocalDispatcher._tool_dict_from_envelope``):

        {"success": true,
         "result": {"success": true,
                    "artifacts": [...],
                    "outputs": {...},
                    "dispatch_mode": "operator"}}

    Failure envelope shape (passed through to the executor's retry path):

        {"success": false,
         "error": "<msg>",
         "error_code": "<tag>"}
    """
    if error is not None and result_file is not None:
        raise click.UsageError(
            "--error is mutually exclusive with --result-file",
        )
    if error is None and result_file is None:
        raise click.UsageError(
            "One of --result-file or --error is required",
        )

    mb = _mailbox(run_id, base_dir)

    if error is not None:
        envelope = {
            "success": False,
            "error": error,
            "error_code": error_code or "OPERATOR_ERROR",
        }
    else:
        # result_file is guaranteed non-None by the usage check above.
        try:
            raw = Path(result_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise click.UsageError(
                f"could not read --result-file {result_file!r}: {exc}",
            )
        try:
            tool_result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise click.UsageError(
                f"--result-file {result_file!r} is not valid JSON: {exc}",
            )
        if not isinstance(tool_result, dict):
            raise click.UsageError(
                f"--result-file must contain a JSON object, got "
                f"{type(tool_result).__name__}",
            )
        # Minimal agent-task shape validation — match what
        # ``_tool_dict_from_envelope`` + the executor retry path
        # expect. We don't constrain the ``outputs`` payload (it's
        # tool-specific), just enforce that the contract keys exist.
        missing = [
            key for key in ("success", "artifacts", "outputs")
            if key not in tool_result
        ]
        if missing:
            raise click.UsageError(
                f"--result-file missing required agent-task keys: "
                f"{missing!r}. Expected at minimum "
                f"{{'success': bool, 'artifacts': [...], 'outputs': {{...}}}}.",
            )
        tool_result.setdefault("dispatch_mode", "operator")
        envelope = {"success": True, "result": tool_result}

    mb.complete(task_id, envelope)
    click.echo(json.dumps({"ok": True, "task_id": task_id}))


def register_mailbox_bridge_command(cli_group: click.Group) -> None:
    """Attach the ``mailbox-bridge`` subgroup to the top-level CLI."""
    cli_group.add_command(mailbox_bridge_group)


__all__ = [
    "mailbox_bridge_group",
    "mailbox_peek",
    "mailbox_complete",
    "mailbox_peek_agent",
    "mailbox_complete_agent",
    "register_mailbox_bridge_command",
]
