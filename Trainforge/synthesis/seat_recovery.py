"""Bounded, durable recovery for a dead local synthesis seat."""

from __future__ import annotations

import json
import hashlib
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from lib.file_lock import file_lock


def eligible_engine_transport_failure(exc: BaseException) -> bool:
    """Return true only for exhausted local transport/server failures."""

    text = str(exc).lower()
    code = str(getattr(exc, "code", "") or "").lower()
    if any(token in text for token in ("429", "malformed", "truncat", "out of memory")):
        return False
    return (
        "failed after 3 transport attempts: timed out" in text
        or code in {"max_retries_exceeded", "500", "502", "503", "504"}
    )


class SynthesisSeatRecoveryCoordinator:
    """One cold-recovery leader shared by all generation worker threads."""

    def __init__(
        self,
        *,
        base_url: str,
        run_dir: Optional[Path],
        marker_path: Optional[Path],
    ) -> None:
        self.base_url = str(base_url).rstrip("/").removesuffix("/v1")
        self.run_dir = run_dir
        self.run_id = run_dir.name if run_dir is not None else None
        self.marker_path = marker_path
        self.recovery_id = ""
        self._condition = threading.Condition()
        self._state = "idle"
        self._result = False

    def _append_marker(self, state: str, **extra: Any) -> None:
        if self.marker_path is None:
            return
        row = {
            "schema_version": 1,
            "recovery_id": self.recovery_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "base_url": self.base_url,
            "state": state,
            **extra,
        }
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.marker_path.with_suffix(self.marker_path.suffix + ".lock")
        with file_lock(lock, timeout=30.0):
            existed = self.marker_path.exists()
            with self.marker_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if not existed:
                try:
                    directory_fd = os.open(self.marker_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass

    def _incident_key(self, context: dict[str, Any]) -> str:
        identity = {
            "run_id": self.run_id,
            "base_url": self.base_url,
            "task": context,
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _prior_incident_state(self, incident_key: str) -> Optional[str]:
        if self.marker_path is None or not self.marker_path.exists():
            return None
        latest = None
        try:
            lock = self.marker_path.with_suffix(
                self.marker_path.suffix + ".lock"
            )
            with file_lock(lock, timeout=None):
                with self.marker_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except (TypeError, ValueError):
                            return "marker_unreadable"
                        if (
                            not isinstance(row, dict)
                            or row.get("schema_version") != 1
                        ):
                            return "marker_unreadable"
                        state = str(row.get("state") or "")
                        if state not in {
                            "started",
                            "recovered",
                            "recovered_after_crash_probe",
                            "failed",
                        }:
                            return "marker_unreadable"
                        if row.get("incident_key") == incident_key:
                            latest = state
        except OSError:
            return "marker_unreadable"
        return latest

    def recover(
        self,
        exc: BaseException,
        *,
        incident_context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Recover once; followers wait and reuse the leader's verdict."""

        if not eligible_engine_transport_failure(exc):
            return False
        context = dict(incident_context or {})
        incident_key = self._incident_key(context)
        self.recovery_id = incident_key
        prior_state = self._prior_incident_state(incident_key)
        if prior_state in {
            "started",
            "recovered",
            "recovered_after_crash_probe",
        }:
            # A process crash must not grant the same run/task/seat another
            # destructive recycle. Reuse only if a fresh deep probe proves the
            # prior window healed the seat; otherwise pause loud.
            from lib.vllm_container_lifecycle import (
                _seat_for_base_url,
                coherence_probe,
            )
            healthy = coherence_probe(self.base_url, max_tokens=32)
            if healthy:
                self._append_marker(
                    "recovered_after_crash_probe",
                    incident_key=incident_key,
                    run_id=self.run_id,
                    seat_name=_seat_for_base_url(self.base_url),
                    workflow_phase=context.get("workflow_phase"),
                    task_id=context.get("task_id"),
                    task=context,
                    prior_state=prior_state,
                )
            return healthy
        if prior_state in {"failed", "marker_unreadable"}:
            return False
        with self._condition:
            if self._state == "running":
                self._condition.wait_for(lambda: self._state != "running")
                return self._result
            if self._state == "failed":
                return False
            if self._state == "recovered":
                # A late follower observes the now-healthy seat. A genuinely
                # later incident observes a failed deep probe and receives the
                # exhausted recovery budget (False → pause loud).
                from lib.vllm_container_lifecycle import coherence_probe
                return coherence_probe(self.base_url)
            self._state = "running"
        ok = False
        try:
            from lib.vllm_container_lifecycle import _seat_for_base_url

            self._append_marker(
                "started",
                incident_key=incident_key,
                run_id=self.run_id,
                seat_name=_seat_for_base_url(self.base_url),
                workflow_phase=context.get("workflow_phase"),
                task_id=context.get("task_id"),
                task=context,
                thread_name=threading.current_thread().name,
                triggering_error_code=str(getattr(exc, "code", "") or ""),
                triggering_error=str(exc)[:1000],
            )
            from lib.vllm_container_lifecycle import recover_seat_for_base_url

            result = recover_seat_for_base_url(
                self.base_url,
                run_dir=self.run_dir,
                reason=(
                    "synthesis_engine_dead:"
                    f"phase={context.get('workflow_phase')}:"
                    f"task={context.get('task_id')}:"
                    f"unit={context.get('kind')}:{context.get('variant_index')}:"
                    f"fingerprint={context.get('fingerprint')}:"
                    f"incident={self.recovery_id}"
                ),
            )
            ok = bool(result.ok)
            self._append_marker(
                "recovered" if ok else "failed",
                incident_key=incident_key,
                seat_name=result.seat_name,
                run_id=self.run_id,
                workflow_phase=context.get("workflow_phase"),
                task_id=context.get("task_id"),
                task=context,
                reason=result.reason,
                recreated=result.recreated,
                engine_signals=list(result.engine_signals),
                load_seconds=result.load_seconds,
            )
        except Exception as error:  # noqa: BLE001 - fail loud via false verdict
            ok = False
            try:
                self._append_marker(
                    "failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            except Exception:
                # Marker durability failure is itself a failed recovery, but it
                # must never strand sibling workers in ``running``.
                pass
        finally:
            with self._condition:
                self._result = ok
                self._state = "recovered" if ok else "failed"
                self._condition.notify_all()
        return ok


__all__ = [
    "SynthesisSeatRecoveryCoordinator",
    "eligible_engine_transport_failure",
]
