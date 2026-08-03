"""Durable, concurrent per-generation resume journal."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from lib.file_lock import file_lock


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
MAX_TRANSIENT_RESUME_ATTEMPTS = 3
UnitKey = Tuple[str, str, int]


def load_generation_journal(
    path: Optional[Path],
    *,
    expected_fresh_start_id: Optional[str] = None,
    expected_marker_digest: Optional[str] = None,
    expected_holdout_identity: Optional[Mapping[str, str]] = None,
    expected_run_contract_sha256: Optional[str] = None,
) -> Dict[UnitKey, Dict[str, Any]]:
    """Load the latest durable disposition for every generation unit."""

    if path is None or not path.exists():
        return {}
    cache: Dict[UnitKey, Dict[str, Any]] = {}
    terminal_seen: set[UnitKey] = set()
    skipped_count: int = 0
    with path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                if expected_fresh_start_id is not None:
                    # Skip malformed rows instead of raising the whole file.
                    # Log and count so the skip is visible.
                    print(
                        f"load_generation_journal: skipping malformed row at line {line_num} "
                        f"in fresh-start mode",
                        file=sys.stderr,
                    )
                    skipped_count += 1
                    continue
                continue
            if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
                if expected_fresh_start_id is not None:
                    # Skip incompatible rows instead of raising the whole file.
                    print(
                        f"load_generation_journal: skipping incompatible row at line {line_num} "
                        f"in fresh-start mode",
                        file=sys.stderr,
                    )
                    skipped_count += 1
                    continue
                continue
            if expected_fresh_start_id is not None and (
                row.get("fresh_start_id") != expected_fresh_start_id
                or row.get("fresh_start_marker_digest") != expected_marker_digest
            ):
                # Skip fresh-start identity mismatch instead of raising the whole file.
                # Cache miss at row level, not cascade rejection.
                print(
                    f"load_generation_journal: skipping fresh-start identity mismatch "
                    f"at line {line_num}; row identity differs from expected",
                    file=sys.stderr,
                )
                skipped_count += 1
                continue
            if (
                expected_holdout_identity is not None
                and row.get("synthesis_holdout_identity")
                != dict(expected_holdout_identity)
            ):
                # Skip holdout identity mismatch instead of raising the whole file.
                print(
                    f"load_generation_journal: skipping holdout identity mismatch "
                    f"at line {line_num}; row holdout differs from expected",
                    file=sys.stderr,
                )
                skipped_count += 1
                continue
            if (
                expected_run_contract_sha256 is not None
                and row.get("synthesis_run_contract_sha256")
                != expected_run_contract_sha256
            ):
                # Skip run-contract mismatch instead of raising the whole file.
                # This is a row-level cache miss: the changed rows will re-run.
                print(
                    f"load_generation_journal: skipping run-contract mismatch "
                    f"at line {line_num}; row contract differs, will re-run",
                    file=sys.stderr,
                )
                skipped_count += 1
                continue
            chunk_id = str(row.get("chunk_id") or "")
            kind = str(row.get("kind") or "")
            try:
                variant = int(row.get("variant_index", 0))
                attempt = int(row.get("attempt", 0))
            except (TypeError, ValueError):
                continue
            if (
                chunk_id
                and kind in {"instruction", "preference"}
                and variant >= 0
                and attempt > 0
                and row.get("disposition")
                in {"success", "transient", "fatal"}
            ):
                key = (chunk_id, kind, variant)
                if row.get("disposition") in {"success", "fatal"}:
                    if (
                        expected_run_contract_sha256 is not None
                        and key in terminal_seen
                    ):
                        # Skip duplicate terminal key instead of raising the whole file.
                        # Log the skip so it's visible in audit.
                        print(
                            f"load_generation_journal: skipping duplicate terminal semantic key "
                            f"{key!r} at line {line_num}; keeping first occurrence",
                            file=sys.stderr,
                        )
                        skipped_count += 1
                        continue
                    terminal_seen.add(key)
                cache[key] = row
    if skipped_count:
        # The per-row prints above name each skip; this names the TOTAL.
        # Without it ``skipped_count`` was computed and thrown away, so a pass
        # that silently discarded its whole resume cache (every row skipped)
        # looked identical to a clean load in any aggregated log.
        print(
            f"load_generation_journal: skipped {skipped_count} row(s) in "
            f"{path}; those units re-run this pass",
            file=sys.stderr,
        )
    return cache


def summarize_generation_journal(path: Optional[Path]) -> Dict[str, int]:
    """Return authoritative lifetime attempt/recovery counts from all rows."""

    summary = {
        "provider_results": 0,
        "transient_pending_units": 0,
        "transient_attempts": 0,
        "recovered_units": 0,
        "exhausted_units": 0,
        "fatal_units": 0,
    }
    if path is None or not path.exists():
        return summary
    histories: Dict[UnitKey, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
                continue
            disposition = row.get("disposition")
            if disposition not in {"success", "transient", "fatal"}:
                continue
            try:
                key = (
                    str(row.get("chunk_id") or ""),
                    str(row.get("kind") or ""),
                    int(row.get("variant_index", 0)),
                )
                attempt = int(row.get("attempt", 0))
            except (TypeError, ValueError):
                continue
            if not key[0] or key[1] not in {"instruction", "preference"}:
                continue
            summary["provider_results"] += 1
            histories.setdefault(key, []).append(str(disposition))
            if disposition == "transient" or row.get("was_transient") is True:
                summary["transient_attempts"] += 1
            if disposition == "fatal":
                summary["fatal_units"] += 1
                if (
                    row.get("was_transient") is True
                    and attempt >= MAX_TRANSIENT_RESUME_ATTEMPTS
                ):
                    summary["exhausted_units"] += 1
    summary["recovered_units"] = sum(
        1
        for dispositions in histories.values()
        if "transient" in dispositions and dispositions[-1] == "success"
    )
    latest = load_generation_journal(path)
    summary["transient_pending_units"] = sum(
        row.get("disposition") == "transient" for row in latest.values()
    )
    return summary


class GenerationJournal:
    """Append unit outcomes durably from worker threads.

    A process-local lock serializes threads and ``file_lock`` serializes
    processes. Each row is flushed and fsynced before the worker reports
    completion, so ordered emission may safely block behind an earlier unit.
    """

    def __init__(
        self,
        path: Optional[Path],
        *,
        fresh_start_id: Optional[str] = None,
        marker_digest: Optional[str] = None,
        holdout_identity: Optional[Mapping[str, str]] = None,
        run_contract_sha256: Optional[str] = None,
        component_manifest_sha256: Optional[str] = None,
    ) -> None:
        self.path = path
        self.fresh_start_id = fresh_start_id
        self.marker_digest = marker_digest
        self.holdout_identity = (
            dict(holdout_identity) if holdout_identity is not None else None
        )
        self.run_contract_sha256 = run_contract_sha256
        self.component_manifest_sha256 = component_manifest_sha256
        self._thread_lock = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        payload = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **dict(row),
        }
        if self.fresh_start_id is not None:
            payload["fresh_start_id"] = self.fresh_start_id
            payload["fresh_start_marker_digest"] = self.marker_digest
        if self.holdout_identity is not None:
            payload["synthesis_holdout_identity"] = self.holdout_identity
        if self.run_contract_sha256 is not None:
            payload["synthesis_run_contract_sha256"] = self.run_contract_sha256
        if self.component_manifest_sha256 is not None:
            payload["synthesis_contract_components_sha256"] = (
                self.component_manifest_sha256
            )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self._thread_lock:
            with file_lock(lock_path, timeout=30.0):
                if self.fresh_start_id is not None:
                    current = load_generation_journal(
                        self.path,
                        expected_fresh_start_id=self.fresh_start_id,
                        expected_marker_digest=self.marker_digest,
                        expected_holdout_identity=self.holdout_identity,
                        expected_run_contract_sha256=self.run_contract_sha256,
                    )
                    key = (
                        str(payload.get("chunk_id") or ""),
                        str(payload.get("kind") or ""),
                        int(payload.get("variant_index", 0)),
                    )
                    prior = current.get(key)
                    if (
                        prior is not None
                        and prior.get("disposition") in {"success", "fatal"}
                    ):
                        # A unit reaching a terminal disposition twice is an
                        # invariant violation, not a cache hit: exactly one
                        # source-order writer is supposed to close each unit, so
                        # a second terminal means double dispatch, a resume that
                        # re-queued completed work, or two writers racing. Any of
                        # those silently corrupts the accepted set — and the
                        # second write is where it becomes visible.
                        #
                        # Downgrading this to a log-and-return was tried and
                        # reverted: it made the bug invisible at the one moment
                        # it announces itself, and left the resume cache holding
                        # whichever disposition happened to land first.
                        raise RuntimeError(
                            "duplicate generation terminal for semantic key "
                            f"{key!r}: already recorded as "
                            f"{prior.get('disposition')!r}, now appending "
                            f"{payload.get('disposition')!r}; exactly one writer "
                            "must close each unit"
                        )
                existed = self.path.exists()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                if not existed:
                    try:
                        directory_fd = os.open(self.path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except OSError:
                        # Windows does not permit fsync on directory handles;
                        # the row's file handle itself is still fsynced.
                        pass


__all__ = [
    "GenerationJournal",
    "MAX_TRANSIENT_RESUME_ATTEMPTS",
    "SCHEMA_VERSION",
    "UnitKey",
    "load_generation_journal",
    "summarize_generation_journal",
]
