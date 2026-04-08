"""
Phase Checkpointing for Crash Recovery

Saves execution state at phase boundaries for resumption.

Phase 0 Hardening - Requirement 2: Execution Model Hardening
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PhaseCheckpoint:
    """Checkpoint state for a workflow phase."""
    run_id: str
    workflow_id: str
    phase_name: str
    phase_index: int
    status: str  # "started", "completed", "failed"
    started_at: str
    completed_at: Optional[str] = None
    tasks_completed: List[str] = field(default_factory=list)
    tasks_failed: List[str] = field(default_factory=list)
    tasks_pending: List[str] = field(default_factory=list)
    last_event_seq: int = 0
    artifacts_produced: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PhaseCheckpoint":
        """Create from dictionary."""
        return cls(**data)

    @property
    def is_complete(self) -> bool:
        """Check if phase is complete."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if phase failed."""
        return self.status == "failed"

    @property
    def can_resume(self) -> bool:
        """Check if phase can be resumed."""
        return self.status == "started" and len(self.tasks_pending) > 0

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        total = len(self.tasks_completed) + len(self.tasks_failed) + len(self.tasks_pending)
        if total == 0:
            return 0.0
        completed = len(self.tasks_completed)
        return (completed / total) * 100


class CheckpointManager:
    """Manages phase checkpoints for crash recovery."""

    def __init__(self, run_path: Path):
        """
        Initialize checkpoint manager.

        Args:
            run_path: Path to run directory (state/runs/{run_id}/)
        """
        self.run_path = run_path
        self.checkpoints_dir = run_path / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, phase_name: str) -> Path:
        """Get path for checkpoint file."""
        safe_name = phase_name.replace("/", "_").replace("\\", "_")
        return self.checkpoints_dir / f"{safe_name}_checkpoint.json"

    def start_phase(
        self,
        run_id: str,
        workflow_id: str,
        phase_name: str,
        phase_index: int,
        task_ids: List[str]
    ) -> PhaseCheckpoint:
        """
        Record phase start checkpoint.

        Args:
            run_id: Run identifier
            workflow_id: Workflow identifier
            phase_name: Name of the phase
            phase_index: Index of phase in workflow (0-based)
            task_ids: List of task IDs to execute in this phase

        Returns:
            Created PhaseCheckpoint
        """
        checkpoint = PhaseCheckpoint(
            run_id=run_id,
            workflow_id=workflow_id,
            phase_name=phase_name,
            phase_index=phase_index,
            status="started",
            started_at=datetime.now().isoformat(),
            tasks_pending=task_ids.copy()
        )

        self._write_checkpoint(checkpoint)
        logger.info(f"Phase checkpoint started: {phase_name} with {len(task_ids)} tasks")

        return checkpoint

    def complete_task(
        self,
        phase_name: str,
        task_id: str,
        success: bool,
        artifacts: Optional[List[Dict]] = None,
        event_seq: Optional[int] = None
    ) -> PhaseCheckpoint:
        """
        Update checkpoint after task completion.

        Args:
            phase_name: Name of the phase
            task_id: ID of completed task
            success: Whether task succeeded
            artifacts: List of artifact dictionaries produced
            event_seq: Latest event sequence number

        Returns:
            Updated PhaseCheckpoint

        Raises:
            ValueError: If no checkpoint exists for phase
        """
        checkpoint = self.load_checkpoint(phase_name)
        if not checkpoint:
            raise ValueError(f"No checkpoint for phase: {phase_name}")

        # Move task from pending to appropriate list
        if task_id in checkpoint.tasks_pending:
            checkpoint.tasks_pending.remove(task_id)

        if success:
            if task_id not in checkpoint.tasks_completed:
                checkpoint.tasks_completed.append(task_id)
            if artifacts:
                checkpoint.artifacts_produced.extend(artifacts)
        else:
            if task_id not in checkpoint.tasks_failed:
                checkpoint.tasks_failed.append(task_id)

        if event_seq is not None:
            checkpoint.last_event_seq = max(checkpoint.last_event_seq, event_seq)

        self._write_checkpoint(checkpoint)
        return checkpoint

    def complete_phase(
        self,
        phase_name: str,
        validation_results: Optional[Dict] = None
    ) -> PhaseCheckpoint:
        """
        Mark phase as completed.

        Args:
            phase_name: Name of the phase
            validation_results: Optional validation gate results

        Returns:
            Updated PhaseCheckpoint

        Raises:
            ValueError: If no checkpoint exists for phase
        """
        checkpoint = self.load_checkpoint(phase_name)
        if not checkpoint:
            raise ValueError(f"No checkpoint for phase: {phase_name}")

        checkpoint.status = "completed"
        checkpoint.completed_at = datetime.now().isoformat()
        if validation_results:
            checkpoint.validation_results = validation_results

        self._write_checkpoint(checkpoint)
        logger.info(
            f"Phase checkpoint completed: {phase_name} "
            f"({len(checkpoint.tasks_completed)} tasks, "
            f"{len(checkpoint.artifacts_produced)} artifacts)"
        )

        return checkpoint

    def fail_phase(
        self,
        phase_name: str,
        error: str,
        error_details: Optional[Dict] = None
    ) -> PhaseCheckpoint:
        """
        Mark phase as failed.

        Args:
            phase_name: Name of the phase
            error: Error message
            error_details: Additional error context

        Returns:
            Updated PhaseCheckpoint

        Raises:
            ValueError: If no checkpoint exists for phase
        """
        checkpoint = self.load_checkpoint(phase_name)
        if not checkpoint:
            raise ValueError(f"No checkpoint for phase: {phase_name}")

        checkpoint.status = "failed"
        checkpoint.completed_at = datetime.now().isoformat()
        checkpoint.error = error
        checkpoint.error_details = error_details

        self._write_checkpoint(checkpoint)
        logger.error(f"Phase checkpoint failed: {phase_name} - {error}")

        return checkpoint

    def load_checkpoint(self, phase_name: str) -> Optional[PhaseCheckpoint]:
        """
        Load checkpoint for a phase.

        Args:
            phase_name: Name of the phase

        Returns:
            PhaseCheckpoint or None if not found
        """
        path = self._checkpoint_path(phase_name)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return PhaseCheckpoint.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to load checkpoint {phase_name}: {e}")
            return None

    def get_resumable_phase(self) -> Optional[PhaseCheckpoint]:
        """
        Find the most recent incomplete phase for resumption.

        Returns:
            PhaseCheckpoint of resumable phase or None
        """
        checkpoints = []
        for path in self.checkpoints_dir.glob("*_checkpoint.json"):
            phase_name = path.stem.replace("_checkpoint", "")
            cp = self.load_checkpoint(phase_name)
            if cp and cp.can_resume:
                checkpoints.append(cp)

        if not checkpoints:
            return None

        # Return highest phase index that's incomplete
        return max(checkpoints, key=lambda c: c.phase_index)

    def get_all_checkpoints(self) -> List[PhaseCheckpoint]:
        """
        Get all checkpoints for this run.

        Returns:
            List of PhaseCheckpoint objects sorted by phase_index
        """
        checkpoints = []
        for path in self.checkpoints_dir.glob("*_checkpoint.json"):
            phase_name = path.stem.replace("_checkpoint", "")
            cp = self.load_checkpoint(phase_name)
            if cp:
                checkpoints.append(cp)

        return sorted(checkpoints, key=lambda c: c.phase_index)

    def get_last_completed_phase(self) -> Optional[PhaseCheckpoint]:
        """
        Get the most recently completed phase.

        Returns:
            PhaseCheckpoint of last completed phase or None
        """
        checkpoints = [cp for cp in self.get_all_checkpoints() if cp.is_complete]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda c: c.phase_index)

    def get_phase_summary(self) -> Dict[str, Any]:
        """
        Get summary of all phase checkpoints.

        Returns:
            Dictionary with summary statistics
        """
        checkpoints = self.get_all_checkpoints()

        return {
            "total_phases": len(checkpoints),
            "completed": len([c for c in checkpoints if c.is_complete]),
            "failed": len([c for c in checkpoints if c.is_failed]),
            "in_progress": len([c for c in checkpoints if c.can_resume]),
            "total_tasks_completed": sum(len(c.tasks_completed) for c in checkpoints),
            "total_tasks_failed": sum(len(c.tasks_failed) for c in checkpoints),
            "total_artifacts": sum(len(c.artifacts_produced) for c in checkpoints),
            "phases": [
                {
                    "name": c.phase_name,
                    "status": c.status,
                    "progress": c.progress_percent,
                    "tasks_completed": len(c.tasks_completed),
                    "tasks_failed": len(c.tasks_failed),
                    "tasks_pending": len(c.tasks_pending),
                }
                for c in checkpoints
            ]
        }

    def _write_checkpoint(self, checkpoint: PhaseCheckpoint) -> None:
        """Atomically write checkpoint."""
        path = self._checkpoint_path(checkpoint.phase_name)
        temp_path = path.with_suffix('.tmp')

        with open(temp_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        temp_path.rename(path)


def get_resume_point(run_path: Path) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get resume point for a run.

    Args:
        run_path: Path to run directory

    Returns:
        Dictionary with resume information or None if not resumable
    """
    manager = CheckpointManager(run_path)
    resumable = manager.get_resumable_phase()

    if not resumable:
        return None

    return {
        "phase_name": resumable.phase_name,
        "phase_index": resumable.phase_index,
        "tasks_pending": resumable.tasks_pending,
        "tasks_completed": resumable.tasks_completed,
        "last_event_seq": resumable.last_event_seq,
        "started_at": resumable.started_at
    }
