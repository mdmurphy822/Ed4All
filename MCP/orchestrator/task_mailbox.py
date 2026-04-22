"""
TaskMailbox — file-based task handoff between LocalDispatcher and an outer
Claude Code session (Wave 34).

The orchestrator runs in a Python subprocess but needs to delegate individual
content-generation tasks to subagents (the ``Agent`` tool) which only exist
inside an enclosing Claude Code session. This module provides the bridge:
a directory of JSON files under ``state/runs/{run_id}/mailbox/`` that the
dispatcher uses as a task queue and the outer session (or a wrapper script)
uses as a work list.

Directory layout
----------------

``state/runs/{run_id}/mailbox/``

    ``pending/``      — task specs the dispatcher has written and is waiting on
    ``in_progress/``  — tasks an outer watcher has claimed (atomic move from ``pending/``)
    ``completed/``    — completion envelopes the outer watcher has written

Atomicity is preserved via ``os.replace`` (atomic rename within a filesystem)
for state transitions and temp-file-plus-rename for writes.

Example usage (dispatcher side)
-------------------------------

    mb = TaskMailbox(run_id="RUN_123")
    mb.put_pending("content_gen_week_1", {"prompt": ..., ...})
    result = mb.wait_for_completion("content_gen_week_1", timeout_seconds=600)
    mb.cleanup("content_gen_week_1")

Example usage (watcher side)
----------------------------

    mb = TaskMailbox(run_id="RUN_123")
    for task_id in mb.list_pending():
        spec = mb.claim(task_id)
        # ... dispatch subagent, await result ...
        mb.complete(task_id, {"success": True, "result": ...})
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MailboxError(Exception):
    """Base class for TaskMailbox-specific errors."""


class TaskNotFoundError(MailboxError):
    """Raised when a task_id has no file in the expected state directory."""


class TaskClaimConflict(MailboxError):
    """Raised when two watchers try to claim the same task_id simultaneously."""


class TaskMailbox:
    """File-based task mailbox under ``state/runs/{run_id}/mailbox/``.

    Args:
        run_id: The workflow run identifier. Used as the subdirectory key.
        base_dir: Parent directory under which ``{run_id}/mailbox/`` lives.
            Defaults to ``Path("state/runs")``.
    """

    def __init__(self, run_id: str, base_dir: Optional[Path] = None):
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        base = Path(base_dir) if base_dir is not None else Path("state/runs")
        self.root = base / run_id / "mailbox"
        self.pending_dir = self.root / "pending"
        self.in_progress_dir = self.root / "in_progress"
        self.completed_dir = self.root / "completed"
        for directory in (self.pending_dir, self.in_progress_dir, self.completed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ api

    def put_pending(self, task_id: str, task_spec: Dict[str, Any]) -> Path:
        """Atomically write ``task_spec`` to ``pending/{task_id}.json``.

        Uses write-to-temp + rename so readers never see a partial file.

        Raises:
            MailboxError: when the write fails.
        """
        self._validate_task_id(task_id)
        if not isinstance(task_spec, dict):
            raise ValueError("task_spec must be a dict")

        spec = dict(task_spec)
        spec.setdefault("task_id", task_id)
        spec.setdefault("run_id", self.run_id)
        spec.setdefault("created_at", time.time())

        target = self.pending_dir / f"{task_id}.json"
        tmp = self.pending_dir / f".{task_id}.json.tmp"
        try:
            tmp.write_text(
                json.dumps(spec, indent=2, default=str), encoding="utf-8"
            )
            os.replace(tmp, target)
        except OSError as exc:
            # Clean up partial temp
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise MailboxError(f"failed to write pending task {task_id}: {exc}") from exc
        return target

    def list_pending(self) -> List[str]:
        """Return task_ids currently in the pending directory (sorted)."""
        if not self.pending_dir.exists():
            return []
        names = []
        for path in self.pending_dir.iterdir():
            if path.is_file() and path.suffix == ".json" and not path.name.startswith("."):
                names.append(path.stem)
        return sorted(names)

    def list_in_progress(self) -> List[str]:
        """Return task_ids currently in the in_progress directory (sorted)."""
        if not self.in_progress_dir.exists():
            return []
        names = [
            path.stem
            for path in self.in_progress_dir.iterdir()
            if path.is_file() and path.suffix == ".json" and not path.name.startswith(".")
        ]
        return sorted(names)

    def list_completed(self) -> List[str]:
        """Return task_ids currently in the completed directory (sorted)."""
        if not self.completed_dir.exists():
            return []
        names = [
            path.stem
            for path in self.completed_dir.iterdir()
            if path.is_file() and path.suffix == ".json" and not path.name.startswith(".")
        ]
        return sorted(names)

    def claim(self, task_id: str) -> Dict[str, Any]:
        """Atomically move ``pending/{task_id}.json`` to ``in_progress/``.

        Returns the task spec (as a dict) after the move. Raises
        ``TaskNotFoundError`` if the pending file is missing and
        ``TaskClaimConflict`` if someone else already claimed it.
        """
        self._validate_task_id(task_id)
        pending = self.pending_dir / f"{task_id}.json"
        in_progress = self.in_progress_dir / f"{task_id}.json"
        if not pending.exists():
            if in_progress.exists():
                raise TaskClaimConflict(
                    f"task {task_id!r} already claimed (in_progress file exists)"
                )
            raise TaskNotFoundError(f"no pending task named {task_id!r}")

        # The actual atomic move. On POSIX, ``os.replace`` atomically
        # overwrites the destination. Under concurrent claim attempts
        # only one rename succeeds; the loser will then observe that
        # ``pending`` is gone on its re-check.
        try:
            os.replace(pending, in_progress)
        except OSError as exc:
            raise MailboxError(
                f"failed to claim {task_id}: {exc}"
            ) from exc

        # Double-check we still own this — a racing claimer may have
        # moved a re-posted task in the meantime, but we've at least got
        # an in_progress file to read.
        try:
            text = in_progress.read_text(encoding="utf-8")
            return json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            raise MailboxError(
                f"claimed {task_id} but could not read spec: {exc}"
            ) from exc

    def complete(self, task_id: str, result: Dict[str, Any]) -> Path:
        """Write a completion envelope to ``completed/{task_id}.json``.

        The in_progress entry (if any) is removed after the completion
        file is successfully written. ``result`` should at minimum carry
        ``success: bool`` and a ``result`` or ``error`` field.
        """
        self._validate_task_id(task_id)
        if not isinstance(result, dict):
            raise ValueError("result must be a dict")

        payload = dict(result)
        payload.setdefault("task_id", task_id)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("completed_at", time.time())

        target = self.completed_dir / f"{task_id}.json"
        tmp = self.completed_dir / f".{task_id}.json.tmp"
        try:
            tmp.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            os.replace(tmp, target)
        except OSError as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise MailboxError(f"failed to write completion {task_id}: {exc}") from exc

        # Best-effort cleanup of the in_progress entry; completion is the
        # source of truth for "done" state regardless.
        in_progress = self.in_progress_dir / f"{task_id}.json"
        try:
            if in_progress.exists():
                in_progress.unlink()
        except OSError as exc:
            logger.debug(
                "complete(%s): could not remove in_progress file: %s",
                task_id,
                exc,
            )

        return target

    def read_completion(self, task_id: str) -> Dict[str, Any]:
        """Read the completion envelope for ``task_id``.

        Raises ``TaskNotFoundError`` if no completion file exists.
        """
        self._validate_task_id(task_id)
        target = self.completed_dir / f"{task_id}.json"
        if not target.exists():
            raise TaskNotFoundError(f"no completion for {task_id!r}")
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MailboxError(
                f"could not read completion {task_id}: {exc}"
            ) from exc

    def wait_for_completion(
        self,
        task_id: str,
        timeout_seconds: float = 600.0,
        poll_interval: float = 0.25,
    ) -> Dict[str, Any]:
        """Block until ``completed/{task_id}.json`` exists, then return its contents.

        Raises ``TimeoutError`` if ``timeout_seconds`` elapses first.
        """
        self._validate_task_id(task_id)
        target = self.completed_dir / f"{task_id}.json"
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        first_check = True
        while True:
            if target.exists():
                return self.read_completion(task_id)
            if not first_check and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout_seconds}s waiting for {task_id!r}"
                )
            first_check = False
            time.sleep(poll_interval)

    def cleanup(self, task_id: str) -> None:
        """Remove any pending/in_progress/completed files for ``task_id``.

        Silently ignores missing files.
        """
        self._validate_task_id(task_id)
        for directory in (self.pending_dir, self.in_progress_dir, self.completed_dir):
            path = directory / f"{task_id}.json"
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.debug(
                    "cleanup(%s) could not remove %s: %s", task_id, path, exc
                )

    def pending_count(self) -> int:
        return len(self.list_pending())

    def in_progress_count(self) -> int:
        return len(self.list_in_progress())

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty string")
        # Paranoia: avoid path separators sneaking into task_ids. We
        # don't try to do general slug sanitization, just block
        # filesystem escapes.
        if os.sep in task_id or "/" in task_id or ".." in task_id:
            raise ValueError(f"task_id may not contain path separators: {task_id!r}")


__all__ = [
    "TaskMailbox",
    "MailboxError",
    "TaskNotFoundError",
    "TaskClaimConflict",
]
