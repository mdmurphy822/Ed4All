"""Best-effort incremental progress writer for the three-arm eval harness.

The three-arm eval (base / retrieval / grounded over the gold set + refusal
probes, with CPU NLI scoring) takes ~100-160 min and historically wrote NOTHING
to disk until the very end, so an operator could only infer progress from
GPU / process state. :class:`EvalProgressWriter` lays down two RUN-SCOPED scratch
files in the course's ``retrieval_eval/`` dir so a long run can be checked
intermittently by tailing a file:

  * ``eval_progress.jsonl`` — one append-only line PER ITEM (arm, item_id,
    index, status, ts, a few cheap running metrics), each with an explicit
    ``flush()`` + ``os.fsync()`` so a concurrent ``tail -f`` sees it the instant
    the item completes.
  * ``eval_progress.json`` — an atomically-rewritten SNAPSHOT (write ``.tmp`` +
    ``os.replace``) holding the run id, timestamps, per-arm rollups, and an
    ``overall`` block with ``done`` / ``total`` / ``percent`` / ``elapsed_s`` /
    ``items_per_s`` / ``eta_s`` (ETA guarded ``None`` until ≥1 item is done).

CONTRACT — these files are ADVISORY, never the source of truth. The eval's real
artifacts (``eval_scorecard_<ts>.json`` / ``grounded_answer_eval_<ts>.json`` /
the review sample) are unchanged and authoritative. Every write here is wrapped
so any ``OSError`` / ``IOError`` is swallowed with a ONE-TIME logged warning —
a progress-write failure must NEVER crash or alter the eval. The distinct
filenames (``eval_progress.*``) keep these scratch files from ever being mistaken
for a canonical artifact, and both are truncated/overwritten at construction so
a new run never appends onto a stale one.

The harness constructs ONE writer for the whole run (knowing the arms + their
planned item counts up front, so total work is known), threads it through each
arm as an optional ``progress`` parameter (default ``None`` ⇒ no file written,
byte-identical to the legacy behavior), and finalizes it via :meth:`close` (or
``with`` — it is a context manager) which marks the snapshot ``"complete"`` or
``"failed"``.

This is production code, so ``datetime.now(timezone.utc)`` is used directly for
timestamps (only workflow scripts forbid wall-clock reads).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

#: Run-scoped scratch filenames. Distinct from every canonical artifact
#: (``eval_scorecard_*`` / ``grounded_answer_eval_*`` / the review sample) so a
#: consumer never mistakes progress for a result.
PROGRESS_JSONL_FILENAME = "eval_progress.jsonl"
PROGRESS_SNAPSHOT_FILENAME = "eval_progress.json"


def _utcnow_iso() -> str:
    """UTC ISO-8601 timestamp (production code may read wall-clock)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EvalProgressWriter:
    """Lay down per-item eval progress to ``retrieval_eval/eval_progress.*``.

    Best-effort: every disk write is guarded so an IO failure logs ONCE and is
    then swallowed — the eval is never aborted by a progress-logging error.

    Parameters
    ----------
    eval_dir:
        The course's ``retrieval_eval/`` directory. Created if absent (guarded).
    run_id:
        A run identifier / timestamp string stamped into both files so a tailer
        can tell which run the progress belongs to.
    arm_totals:
        Mapping of ``arm name -> planned item count`` for every arm that will
        run, so total work is known up front (the ``overall`` rollup + ETA need
        the denominator before the first item completes).
    """

    def __init__(
        self,
        eval_dir: Path,
        *,
        run_id: str,
        arm_totals: Mapping[str, int],
    ) -> None:
        self._eval_dir = Path(eval_dir)
        self._run_id = str(run_id)
        self._started_monotonic = time.monotonic()
        self._started_at = _utcnow_iso()
        self._warned = False  # one-time warning latch
        self._closed = False

        # Per-arm rollup state. ``done`` counts items recorded; the running
        # metric sums are kept cheap (only what record_item is handed).
        self._arms: Dict[str, Dict[str, Any]] = {}
        total = 0
        for arm, planned in arm_totals.items():
            planned_int = int(planned) if planned is not None else 0
            self._arms[str(arm)] = {
                "total": planned_int,
                "done": 0,
                "started": False,
                "finished": False,
                # Running rollups (advisory; only the cheap signals the arm
                # already computes per item are summed here).
                "answered": 0,
                "declined": 0,
                "errored": 0,
                "unsupported": 0,
                "key_points_total": 0,
                "key_points_covered": 0,
            }
            total += planned_int
        self._total = total
        self._done = 0

        self._jsonl_path = self._eval_dir / PROGRESS_JSONL_FILENAME
        self._snapshot_path = self._eval_dir / PROGRESS_SNAPSHOT_FILENAME

        # RUN-SCOPED reset: truncate any prior progress files so a new run never
        # appends onto a stale one. Done in __init__ so the truncation happens
        # exactly once, at run start.
        self._reset_files()
        # Emit an initial snapshot so a tailer sees the planned totals + state
        # before the first item lands.
        self._write_snapshot("running")

    # -- lifecycle ------------------------------------------------------- #

    def start_arm(self, arm: str, total_items: Optional[int] = None) -> None:
        """Mark an arm as started; optionally correct its planned item count.

        ``total_items`` lets the arm refine the up-front estimate once it has
        loaded its real work set (e.g. only gold questions carrying expected
        key points are scored). Best-effort: a snapshot rewrite failure is
        swallowed.
        """
        entry = self._arms.get(str(arm))
        if entry is None:
            entry = self._new_arm_entry()
            self._arms[str(arm)] = entry
        entry["started"] = True
        if total_items is not None:
            new_total = int(total_items)
            self._total += new_total - int(entry.get("total", 0) or 0)
            entry["total"] = new_total
        self._write_snapshot("running")

    def record_item(
        self,
        arm: str,
        item_id: str,
        status: str,
        **metrics: Any,
    ) -> None:
        """Record one completed eval item (gold question or refusal probe).

        Appends one line to ``eval_progress.jsonl`` (flushed + fsync'd so a tail
        sees it immediately) and atomically rewrites the snapshot. ``metrics``
        carries only cheap, already-computed signals the arm hands in (e.g.
        ``answered=True``, ``key_points_total`` / ``key_points_covered``,
        ``unsupported=1``); nothing is computed here solely for progress.

        Best-effort: any IO error is swallowed with a one-time warning.
        """
        entry = self._arms.get(str(arm))
        if entry is None:
            entry = self._new_arm_entry()
            self._arms[str(arm)] = entry

        entry["done"] = int(entry.get("done", 0)) + 1
        self._done += 1
        index = entry["done"]

        # Cheap running rollups — only the signals the caller handed us.
        self._roll_up(entry, status, metrics)

        elapsed = max(time.monotonic() - self._started_monotonic, 0.0)
        line = {
            "run_id": self._run_id,
            "ts": _utcnow_iso(),
            "arm": str(arm),
            "item_id": str(item_id),
            "index": index,
            "arm_total": int(entry.get("total", 0) or 0),
            "status": str(status),
            "overall_done": self._done,
            "overall_total": self._total,
            "elapsed_s": round(elapsed, 3),
        }
        # Cheap running metrics for the tail line (where present).
        for k in (
            "key_point_coverage_so_far",
            "answered_so_far",
            "declined_so_far",
            "errored_so_far",
            "unsupported_so_far",
        ):
            v = self._running_metric(entry, k)
            if v is not None:
                line[k] = v

        self._append_jsonl(line)
        self._write_snapshot("running")

    def finish_arm(self, arm: str) -> None:
        """Mark an arm finished (best-effort snapshot rewrite)."""
        entry = self._arms.get(str(arm))
        if entry is None:
            entry = self._new_arm_entry()
            self._arms[str(arm)] = entry
        entry["finished"] = True
        self._write_snapshot("running")

    def close(self, *, failed: bool = False) -> None:
        """Finalize the snapshot, marking it ``"complete"`` (or ``"failed"``).

        Idempotent: a second call is a no-op. Best-effort: a write failure is
        swallowed.
        """
        if self._closed:
            return
        self._closed = True
        self._write_snapshot("failed" if failed else "complete")

    # Context-manager sugar: ``with EvalProgressWriter(...) as p:`` finalizes on
    # exit, marking the snapshot "failed" iff the body raised.
    def __enter__(self) -> "EvalProgressWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close(failed=exc_type is not None)
        return False  # never suppress an exception

    # -- introspection (tests / callers) --------------------------------- #

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def snapshot_path(self) -> Path:
        return self._snapshot_path

    # -- internals ------------------------------------------------------- #

    @staticmethod
    def _new_arm_entry() -> Dict[str, Any]:
        return {
            "total": 0,
            "done": 0,
            "started": True,
            "finished": False,
            "answered": 0,
            "declined": 0,
            "errored": 0,
            "unsupported": 0,
            "key_points_total": 0,
            "key_points_covered": 0,
        }

    @staticmethod
    def _roll_up(
        entry: Dict[str, Any], status: str, metrics: Mapping[str, Any]
    ) -> None:
        """Fold one item's cheap signals into the arm rollup (no computation)."""
        status = str(status)
        # Status-driven answered / declined / errored tallies.
        if status in ("error", "errored"):
            entry["errored"] = int(entry.get("errored", 0)) + 1
        elif metrics.get("answered") is True or status == "answered":
            entry["answered"] = int(entry.get("answered", 0)) + 1
        elif metrics.get("answered") is False or status in (
            "declined",
            "refused",
        ):
            entry["declined"] = int(entry.get("declined", 0)) + 1
        # Explicit declined/errored signals (override / augment status).
        if metrics.get("declined") is True:
            entry["declined"] = int(entry.get("declined", 0)) + 1
        # Key-point coverage running totals (cheap counts the arm already has).
        kp_total = metrics.get("key_points_total")
        kp_covered = metrics.get("key_points_covered")
        if isinstance(kp_total, (int, float)):
            entry["key_points_total"] = int(entry["key_points_total"]) + int(
                kp_total
            )
        if isinstance(kp_covered, (int, float)):
            entry["key_points_covered"] = int(
                entry["key_points_covered"]
            ) + int(kp_covered)
        # Unsupported-claim running count (hallucination axis), where handed in.
        unsupported = metrics.get("unsupported")
        if isinstance(unsupported, (int, float)):
            entry["unsupported"] = int(entry["unsupported"]) + int(unsupported)

    @staticmethod
    def _running_metric(entry: Dict[str, Any], key: str) -> Any:
        if key == "answered_so_far":
            return int(entry.get("answered", 0))
        if key == "declined_so_far":
            return int(entry.get("declined", 0))
        if key == "errored_so_far":
            return int(entry.get("errored", 0))
        if key == "unsupported_so_far":
            return int(entry.get("unsupported", 0))
        if key == "key_point_coverage_so_far":
            kpt = int(entry.get("key_points_total", 0))
            kpc = int(entry.get("key_points_covered", 0))
            return (kpc / kpt) if kpt else None
        return None

    def _arm_snapshot(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        kpt = int(entry.get("key_points_total", 0))
        kpc = int(entry.get("key_points_covered", 0))
        return {
            "done": int(entry.get("done", 0)),
            "total": int(entry.get("total", 0)),
            "started": bool(entry.get("started", False)),
            "finished": bool(entry.get("finished", False)),
            "answered": int(entry.get("answered", 0)),
            "declined": int(entry.get("declined", 0)),
            "errored": int(entry.get("errored", 0)),
            "unsupported": int(entry.get("unsupported", 0)),
            "key_point_coverage_so_far": (kpc / kpt) if kpt else None,
        }

    def _overall(self) -> Dict[str, Any]:
        elapsed = max(time.monotonic() - self._started_monotonic, 0.0)
        done = self._done
        total = self._total
        percent = (100.0 * done / total) if total else None
        # Rate + ETA: None until ≥1 item is done (guard div-by-zero); ETA also
        # None when total is unknown / already complete.
        items_per_s: Optional[float] = None
        eta_s: Optional[float] = None
        if done > 0 and elapsed > 0:
            items_per_s = done / elapsed
            remaining = max(total - done, 0)
            if items_per_s > 0 and remaining > 0:
                eta_s = remaining / items_per_s
        return {
            "done": done,
            "total": total,
            "percent": round(percent, 2) if percent is not None else None,
            "elapsed_s": round(elapsed, 3),
            "items_per_s": (
                round(items_per_s, 4) if items_per_s is not None else None
            ),
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
        }

    def _build_snapshot(self, state: str) -> Dict[str, Any]:
        return {
            "run_id": self._run_id,
            "state": state,
            "started_at": self._started_at,
            "last_update": _utcnow_iso(),
            "arms": {
                arm: self._arm_snapshot(entry)
                for arm, entry in self._arms.items()
            },
            "overall": self._overall(),
        }

    # -- guarded IO ------------------------------------------------------ #

    def _warn_once(self, msg: str, exc: BaseException) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "EvalProgressWriter: progress write failed (advisory, "
                "swallowed; eval artifacts are the source of truth): %s: %s",
                msg,
                exc,
            )

    def _ensure_dir(self) -> bool:
        try:
            self._eval_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:  # best-effort
            self._warn_once("mkdir", exc)
            return False

    def _reset_files(self) -> None:
        """Truncate/overwrite any prior progress files at run start."""
        if not self._ensure_dir():
            return
        try:
            # Truncate the jsonl (open 'w' then close).
            with open(self._jsonl_path, "w", encoding="utf-8"):
                pass
        except OSError as exc:  # best-effort
            self._warn_once("reset_jsonl", exc)
        # The snapshot is overwritten by the first _write_snapshot; remove any
        # stale .tmp left by a crashed prior run.
        try:
            tmp = self._snapshot_path.with_suffix(".json.tmp")
            if tmp.exists():
                tmp.unlink()
        except OSError as exc:  # best-effort
            self._warn_once("reset_snapshot_tmp", exc)

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:  # best-effort — never abort the eval
            self._warn_once("append_jsonl", exc)

    def _write_snapshot(self, state: str) -> None:
        snapshot = self._build_snapshot(state)
        tmp = self._snapshot_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._snapshot_path)
        except OSError as exc:  # best-effort — never abort the eval
            self._warn_once("write_snapshot", exc)
            # Clean up a partial .tmp so it is not mistaken for a snapshot.
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


__all__ = [
    "EvalProgressWriter",
    "PROGRESS_JSONL_FILENAME",
    "PROGRESS_SNAPSHOT_FILENAME",
]
