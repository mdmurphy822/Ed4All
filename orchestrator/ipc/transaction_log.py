"""
Transaction logging for orchestrator audit trail.

Provides append-only JSONL logging with crash-safe writes for
tracking workflow events and enabling recovery.
"""

import fcntl
import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Transaction log directory (relative to this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSACTION_DIR = _PROJECT_ROOT / "state" / "transactions"


class EventType(Enum):
    """Types of audit events."""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    PHASE_ERROR = "phase_error"
    TASK_CLAIM = "task_claim"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
    LOCK_ACQUIRE = "lock_acquire"
    LOCK_RELEASE = "lock_release"
    RECOVERY = "recovery"


class TransactionLog:
    """Append-only transaction log for audit trail and recovery."""

    def __init__(self, workflow_id: str):
        """Initialize transaction log for a workflow.

        Args:
            workflow_id: Unique workflow identifier
        """
        self.workflow_id = workflow_id
        TRANSACTION_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = TRANSACTION_DIR / f"{workflow_id}.jsonl"

    def log_event(
        self,
        event_type: EventType,
        details: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None,
        worker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log an event with crash-safe writes.

        Args:
            event_type: Type of event
            details: Additional event details
            component: Component/phase name
            worker_id: Worker identifier

        Returns:
            The logged event record
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": self.workflow_id,
            "event_type": event_type.value,
            "component": component,
            "worker_id": worker_id,
            "details": details or {}
        }

        # Crash-safe append
        with open(self.log_path, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + '\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        component: Optional[str] = None,
        since: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events matching filters.

        Args:
            event_type: Filter by event type
            component: Filter by component
            since: Filter events after this ISO timestamp

        Returns:
            List of matching events
        """
        if not self.log_path.exists():
            return []

        events = []
        with open(self.log_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        if event_type and event.get("event_type") != event_type.value:
                            continue
                        if component and event.get("component") != component:
                            continue
                        if since and event.get("timestamp", "") < since:
                            continue
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        return events

    def recover_from_log(self) -> Dict[str, Any]:
        """Analyze log to determine recovery state.

        Returns:
            Dict with recovery information including incomplete phases
        """
        events = self.get_events()

        started_phases = set()
        completed_phases = set()
        failed_phases = set()

        for event in events:
            component = event.get("component")
            if not component:
                continue

            event_type = event.get("event_type")
            if event_type == "phase_start":
                started_phases.add(component)
            elif event_type == "phase_complete":
                completed_phases.add(component)
            elif event_type == "phase_error":
                failed_phases.add(component)

        incomplete = started_phases - completed_phases - failed_phases

        return {
            "workflow_id": self.workflow_id,
            "total_events": len(events),
            "started_phases": list(started_phases),
            "completed_phases": list(completed_phases),
            "failed_phases": list(failed_phases),
            "incomplete_phases": list(incomplete),
            "needs_recovery": len(incomplete) > 0
        }

    def get_workflow_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological workflow timeline.

        Returns:
            List of events sorted by timestamp
        """
        events = self.get_events()
        return sorted(events, key=lambda e: e.get("timestamp", ""))
