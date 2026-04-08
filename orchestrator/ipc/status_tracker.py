"""
Status Tracker for Multi-Terminal Coordination

File-based IPC for coordinating parallel processing across multiple terminals.
Adapted from INTEGRATOR CURRICULUM patterns.
"""

import fcntl
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
_IPC_DIR = Path(__file__).resolve().parent
_ORCHESTRATOR_DIR = _IPC_DIR.parent
_PROJECT_ROOT = _ORCHESTRATOR_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import PROJECT_ROOT, STATE_PATH  # noqa: E402

PROJECT_DIR = PROJECT_ROOT
STATE_DIR = STATE_PATH
HEARTBEAT_DIR = STATE_PATH / "heartbeats"

# Task timeout configuration (in minutes)
DEFAULT_TASK_TIMEOUT_MINUTES = 60

# Phase-specific timeouts
PHASE_TIMEOUTS: Dict[str, int] = {
    "content-generator": 120,
    "dart-conversion": 90,
    "validation": 30,
    "packaging": 45,
}


@dataclass
class StatusUpdate:
    """A status update record."""
    component: str
    status: str  # PENDING, IN_PROGRESS, COMPLETE, ERROR
    updated_at: str
    worker_id: Optional[str] = None
    details: Dict[str, Any] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class StatusTracker:
    """
    File-based status tracking for multi-terminal coordination.

    Uses file locking for safe concurrent access.
    """

    # Status constants
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize status tracker.

        Args:
            state_dir: Directory for state files (default: PROJECT_DIR/state)
        """
        self.state_dir = state_dir or STATE_DIR
        self.status_dir = self.state_dir / "status"
        self.locks_dir = self.state_dir / "locks"
        self.logs_dir = self.state_dir / "logs"

        # Ensure directories exist
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_status_path(self, component: str) -> Path:
        """Get path to status file for a component."""
        safe_name = component.replace("/", "_").replace("\\", "_")
        return self.status_dir / f"{safe_name}.json"

    def _get_lock_path(self, resource: str) -> Path:
        """Get path to lock file for a resource."""
        safe_name = resource.replace("/", "_").replace("\\", "_")
        return self.locks_dir / f"{safe_name}.lock"

    def get_status(self, component: str) -> Optional[StatusUpdate]:
        """
        Get current status of a component.

        Args:
            component: Component identifier

        Returns:
            StatusUpdate or None if not found
        """
        status_path = self._get_status_path(component)

        if not status_path.exists():
            return None

        try:
            with open(status_path, 'r') as f:
                data = json.load(f)
                return StatusUpdate(**data)
        except (json.JSONDecodeError, IOError):
            return None

    def update_status(
        self,
        component: str,
        status: str,
        worker_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ) -> StatusUpdate:
        """
        Update status of a component.

        Args:
            component: Component identifier
            status: New status (PENDING, IN_PROGRESS, COMPLETE, ERROR)
            worker_id: ID of worker making update
            details: Additional status details
            error_message: Error message if status is ERROR
            duration_seconds: Processing duration if COMPLETE

        Returns:
            The created StatusUpdate
        """
        status_path = self._get_status_path(component)

        update = StatusUpdate(
            component=component,
            status=status,
            updated_at=datetime.now().isoformat(),
            worker_id=worker_id,
            details=details or {},
            error_message=error_message,
            duration_seconds=duration_seconds
        )

        # Atomic write with file locking
        temp_path = status_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            # Try to acquire lock
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Lock unavailable, wait briefly
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            try:
                json.dump(asdict(update), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        os.rename(temp_path, status_path)

        # Update progress markdown
        self.update_progress_md()

        return update

    def claim_task(self, component: str, worker_id: str) -> bool:
        """
        Attempt to claim a task for processing.

        Only succeeds if task is PENDING. Uses atomic file locking
        to prevent race conditions between multiple workers.

        Args:
            component: Component to claim
            worker_id: ID of claiming worker

        Returns:
            True if claim successful, False otherwise
        """
        status_path = self._get_status_path(component)
        claim_lock_path = status_path.with_suffix('.claim')

        try:
            # Create claim lock file for atomic check-and-update
            with open(claim_lock_path, 'w') as lock_file:
                try:
                    # Non-blocking exclusive lock attempt
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # Another worker is claiming this task
                    return False

                try:
                    current = self.get_status(component)

                    if current is None:
                        # Create new status
                        self.update_status(component, self.IN_PROGRESS, worker_id)
                        return True

                    if current.status == self.PENDING:
                        self.update_status(component, self.IN_PROGRESS, worker_id)
                        return True

                    return False
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except IOError:
            return False
        finally:
            # Clean up claim lock file
            try:
                claim_lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def complete_task(
        self,
        component: str,
        worker_id: str,
        duration_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a task as complete.

        Args:
            component: Component to complete
            worker_id: ID of completing worker
            duration_seconds: How long processing took
            details: Additional completion details

        Returns:
            True if update successful
        """
        current = self.get_status(component)

        if current and current.status == self.IN_PROGRESS:
            self.update_status(
                component,
                self.COMPLETE,
                worker_id,
                details,
                duration_seconds=duration_seconds
            )
            return True

        return False

    def fail_task(
        self,
        component: str,
        worker_id: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a task as failed.

        Args:
            component: Component that failed
            worker_id: ID of worker reporting failure
            error_message: Description of error
            details: Additional error details

        Returns:
            True if update successful
        """
        self.update_status(
            component,
            self.ERROR,
            worker_id,
            details,
            error_message=error_message
        )
        return True

    def acquire_lock(
        self,
        resource: str,
        owner: str,
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Acquire exclusive lock on a resource.

        Uses atomic file locking to prevent race conditions.

        Args:
            resource: Resource to lock
            owner: Lock owner identifier
            ttl_seconds: Lock time-to-live

        Returns:
            True if lock acquired, False if already locked
        """
        lock_path = self._get_lock_path(resource)
        acquisition_lock_path = lock_path.with_suffix('.acquiring')

        try:
            # Use a separate file lock for atomic acquisition
            with open(acquisition_lock_path, 'w') as acq_lock:
                try:
                    fcntl.flock(acq_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # Another process is acquiring this lock
                    return False

                try:
                    # Check existing lock
                    if lock_path.exists():
                        try:
                            with open(lock_path, 'r') as f:
                                existing = json.load(f)

                            expires = datetime.fromisoformat(existing["expires"])
                            if datetime.now() < expires:
                                return False  # Still locked
                        except (json.JSONDecodeError, IOError, KeyError):
                            pass  # Corrupted lock, allow override

                    # Create lock
                    lock_data = {
                        "resource": resource,
                        "owner": owner,
                        "acquired": datetime.now().isoformat(),
                        "expires": (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat(),
                        "pid": os.getpid()
                    }

                    temp_path = lock_path.with_suffix('.tmp')
                    with open(temp_path, 'w') as f:
                        json.dump(lock_data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())

                    os.rename(temp_path, lock_path)
                    self.update_progress_md()
                    return True
                finally:
                    fcntl.flock(acq_lock.fileno(), fcntl.LOCK_UN)
        except IOError:
            return False
        finally:
            # Clean up acquisition lock file
            try:
                acquisition_lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def release_lock(self, resource: str, owner: str) -> bool:
        """
        Release a lock.

        Args:
            resource: Resource to unlock
            owner: Lock owner (must match)

        Returns:
            True if released, False if not owner
        """
        lock_path = self._get_lock_path(resource)

        if not lock_path.exists():
            return True

        try:
            with open(lock_path, 'r') as f:
                existing = json.load(f)

            if existing.get("owner") != owner:
                return False

            lock_path.unlink()
            self.update_progress_md()
            return True

        except (json.JSONDecodeError, IOError):
            return False

    def get_all_status(self) -> List[StatusUpdate]:
        """Get status of all components."""
        statuses = []

        for status_file in self.status_dir.glob("*.json"):
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
                    statuses.append(StatusUpdate(**data))
            except (json.JSONDecodeError, IOError):
                continue

        return statuses

    def get_pending_tasks(self) -> List[str]:
        """Get list of pending task components."""
        return [
            s.component for s in self.get_all_status()
            if s.status == self.PENDING
        ]

    def get_progress_summary(self) -> Dict[str, int]:
        """Get summary count by status."""
        summary = {
            self.PENDING: 0,
            self.IN_PROGRESS: 0,
            self.COMPLETE: 0,
            self.ERROR: 0,
            self.SKIPPED: 0
        }

        for status in self.get_all_status():
            if status.status in summary:
                summary[status.status] += 1

        return summary

    def update_progress_md(self):
        """Update GENERATION_PROGRESS.md from current state."""
        progress_path = self.state_dir / "GENERATION_PROGRESS.md"

        lines = [
            "# Ed4All Generation Progress\n\n",
            f"Last Updated: {datetime.now().isoformat()}\n\n"
        ]

        # Summary
        summary = self.get_progress_summary()
        total = sum(summary.values())
        completed = summary[self.COMPLETE]

        lines.append("## Summary\n\n")
        lines.append(f"- **Total Tasks**: {total}\n")
        pct = (completed / total * 100) if total > 0 else 0
        lines.append(f"- **Completed**: {completed} ({pct:.1f}%)\n")
        lines.append(f"- **In Progress**: {summary[self.IN_PROGRESS]}\n")
        lines.append(f"- **Pending**: {summary[self.PENDING]}\n")
        lines.append(f"- **Errors**: {summary[self.ERROR]}\n\n")

        # Active Workflows Table
        workflows_dir = self.state_dir / "workflows"
        lines.append("## Active Workflows\n\n")
        lines.append("| Workflow ID | Type | Status | Started | Progress |\n")
        lines.append("|-------------|------|--------|---------|----------|\n")

        if workflows_dir.exists():
            for wf_file in workflows_dir.glob("*.json"):
                try:
                    with open(wf_file, 'r') as f:
                        wf = json.load(f)
                    progress = wf.get("progress", {})
                    wf_total = progress.get("total", 0)
                    wf_completed = progress.get("completed", 0)
                    lines.append(
                        f"| {wf.get('id', '-')} | {wf.get('type', '-')} | {wf.get('status', '-')} | "
                        f"{wf.get('created_at', '-')[:16]} | {wf_completed}/{wf_total} |\n"
                    )
                except (json.JSONDecodeError, IOError):
                    continue

        # Component Status Table
        lines.append("\n## Component Status\n\n")
        lines.append("| Component | Status | Worker | Updated |\n")
        lines.append("|-----------|--------|--------|--------|\n")

        for status in sorted(self.get_all_status(), key=lambda s: s.component):
            lines.append(
                f"| {status.component} | {status.status} | "
                f"{status.worker_id or '-'} | {status.updated_at[:16]} |\n"
            )

        # Active Locks
        lines.append("\n## Active Locks\n\n")
        lines.append("| Resource | Owner | Acquired | Expires |\n")
        lines.append("|----------|-------|----------|--------|\n")

        for lock_file in self.locks_dir.glob("*.lock"):
            try:
                with open(lock_file, 'r') as f:
                    lock = json.load(f)

                # Check if expired
                expires = datetime.fromisoformat(lock["expires"])
                if datetime.now() < expires:
                    lines.append(
                        f"| {lock['resource']} | {lock['owner']} | "
                        f"{lock.get('acquired', '-')[:16]} | {lock.get('expires', '-')[:16]} |\n"
                    )
            except (json.JSONDecodeError, IOError):
                continue

        # Error Log
        errors = [s for s in self.get_all_status() if s.status == self.ERROR]
        if errors:
            lines.append("\n## Recent Errors\n\n")
            lines.append("| Component | Error | Time |\n")
            lines.append("|-----------|-------|------|\n")

            for err in errors[:10]:  # Last 10 errors
                msg = (err.error_message or "Unknown error")[:50]
                lines.append(f"| {err.component} | {msg} | {err.updated_at[:16]} |\n")

        with open(progress_path, 'w') as f:
            f.writelines(lines)

    def update_heartbeat(self, component: str, worker_id: str) -> None:
        """Update heartbeat timestamp for a worker.

        Args:
            component: Component the worker is processing
            worker_id: Worker identifier
        """
        HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
        heartbeat_path = HEARTBEAT_DIR / f"{component}_{worker_id}.heartbeat"

        heartbeat_data = {
            "component": component,
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid()
        }

        with open(heartbeat_path, 'w') as f:
            json.dump(heartbeat_data, f)

    def get_task_timeout_for_phase(self, phase: str) -> int:
        """Get timeout in minutes for a phase.

        Args:
            phase: Phase name

        Returns:
            Timeout in minutes
        """
        return PHASE_TIMEOUTS.get(phase, DEFAULT_TASK_TIMEOUT_MINUTES)

    def detect_dead_workers(self, timeout_multiplier: float = 2.0) -> List[Dict]:
        """Detect workers that have stopped sending heartbeats.

        Args:
            timeout_multiplier: Multiply phase timeout by this factor

        Returns:
            List of dead worker info dicts
        """
        if not HEARTBEAT_DIR.exists():
            return []

        dead_workers = []
        now = datetime.now()

        for heartbeat_file in HEARTBEAT_DIR.glob("*.heartbeat"):
            try:
                with open(heartbeat_file, 'r') as f:
                    data = json.load(f)

                component = data.get("component", "")
                timestamp_str = data.get("timestamp", "")

                if not timestamp_str:
                    continue

                last_heartbeat = datetime.fromisoformat(timestamp_str)
                timeout_mins = self.get_task_timeout_for_phase(component)
                threshold = timedelta(minutes=timeout_mins * timeout_multiplier)

                if now - last_heartbeat > threshold:
                    dead_workers.append({
                        "component": component,
                        "worker_id": data.get("worker_id"),
                        "last_heartbeat": timestamp_str,
                        "timeout_minutes": timeout_mins,
                        "dead_for_minutes": (now - last_heartbeat).total_seconds() / 60
                    })
            except (json.JSONDecodeError, ValueError, IOError):
                continue

        return dead_workers

    def cleanup_stale_tasks(self, dry_run: bool = True) -> List[Dict]:
        """Clean up tasks from dead workers.

        Args:
            dry_run: If True, only report what would be cleaned up

        Returns:
            List of cleaned up task info
        """
        dead_workers = self.detect_dead_workers()
        cleaned = []

        for worker in dead_workers:
            component = worker.get("component")
            if not component:
                continue

            status = self.get_status(component)
            if status and status.status == self.IN_PROGRESS:
                cleaned.append({
                    "component": component,
                    "previous_worker": worker.get("worker_id"),
                    "action": "reset_to_pending" if not dry_run else "would_reset"
                })

                if not dry_run:
                    self.update_status(component, self.PENDING, worker_id=None)
                    # Clean up heartbeat file
                    heartbeat_path = HEARTBEAT_DIR / f"{component}_{worker.get('worker_id')}.heartbeat"
                    if heartbeat_path.exists():
                        heartbeat_path.unlink()

        return cleaned
