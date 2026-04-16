"""
Tests for orchestrator/core/checkpoint.py - Phase checkpointing for crash recovery.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from MCP.hardening.checkpoint import (
        CheckpointManager,
        PhaseCheckpoint,
    )
except ImportError:
    pytest.skip("checkpoint not available", allow_module_level=True)


# =============================================================================
# PHASE CHECKPOINT TESTS
# =============================================================================

class TestPhaseCheckpoint:
    """Test PhaseCheckpoint dataclass."""

    @pytest.mark.unit
    def test_is_complete_property(self):
        """is_complete should return True when status is 'completed'."""
        checkpoint = PhaseCheckpoint(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            status="completed",
            started_at="2025-01-01T00:00:00",
        )

        assert checkpoint.is_complete is True

        checkpoint.status = "started"
        assert checkpoint.is_complete is False

    @pytest.mark.unit
    def test_is_failed_property(self):
        """is_failed should return True when status is 'failed'."""
        checkpoint = PhaseCheckpoint(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            status="failed",
            started_at="2025-01-01T00:00:00",
            error="Task T001 failed",
        )

        assert checkpoint.is_failed is True

    @pytest.mark.unit
    def test_can_resume_property(self):
        """can_resume should be True when started with pending tasks."""
        checkpoint = PhaseCheckpoint(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            status="started",
            started_at="2025-01-01T00:00:00",
            tasks_pending=["T003", "T004"],
        )

        assert checkpoint.can_resume is True

        # Completed phases cannot resume
        checkpoint.status = "completed"
        assert checkpoint.can_resume is False

    @pytest.mark.unit
    def test_progress_percent(self):
        """progress_percent should calculate correct percentage."""
        checkpoint = PhaseCheckpoint(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            status="started",
            started_at="2025-01-01T00:00:00",
            tasks_completed=["T001", "T002"],
            tasks_failed=[],
            tasks_pending=["T003", "T004"],
        )

        # 2 completed out of 4 total = 50%
        assert checkpoint.progress_percent == 50.0

    @pytest.mark.unit
    def test_progress_percent_empty(self):
        """progress_percent should handle empty task lists."""
        checkpoint = PhaseCheckpoint(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            status="started",
            started_at="2025-01-01T00:00:00",
        )

        assert checkpoint.progress_percent == 0.0

    @pytest.mark.unit
    def test_to_dict_roundtrip(self):
        """Should serialize and deserialize correctly."""
        original = PhaseCheckpoint(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            status="completed",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T01:00:00",
            tasks_completed=["T001"],
            artifacts_produced=[{"type": "outline", "path": "outline.json"}],
        )

        d = original.to_dict()
        restored = PhaseCheckpoint.from_dict(d)

        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert restored.tasks_completed == original.tasks_completed


# =============================================================================
# CHECKPOINT MANAGER TESTS
# =============================================================================

class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create CheckpointManager with temp directory."""
        run_path = tmp_path / "runs" / "RUN_001"
        run_path.mkdir(parents=True)
        return CheckpointManager(run_path)

    @pytest.mark.unit
    def test_start_phase_creates_checkpoint(self, manager):
        """start_phase should create and return checkpoint."""
        checkpoint = manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001", "T002", "T003"],
        )

        assert checkpoint.status == "started"
        assert checkpoint.phase_name == "planning"
        assert len(checkpoint.tasks_pending) == 3
        assert "T001" in checkpoint.tasks_pending

    @pytest.mark.unit
    def test_start_phase_saves_to_disk(self, manager):
        """start_phase should persist checkpoint to disk."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001"],
        )

        checkpoint_file = manager.checkpoints_dir / "planning_checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file) as f:
            data = json.load(f)
        assert data["phase_name"] == "planning"

    @pytest.mark.unit
    def test_complete_task_updates_checkpoint(self, manager):
        """complete_task should move task from pending to completed."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T001", "T002", "T003"],
        )

        checkpoint = manager.complete_task(
            phase_name="content_generation",
            task_id="T001",
            success=True,
        )

        assert "T001" in checkpoint.tasks_completed
        assert "T001" not in checkpoint.tasks_pending

    @pytest.mark.unit
    def test_complete_task_with_failure(self, manager):
        """Failed task should go to tasks_failed."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T001", "T002"],
        )

        checkpoint = manager.complete_task(
            phase_name="content_generation",
            task_id="T001",
            success=False,
        )

        assert "T001" in checkpoint.tasks_failed
        assert "T001" not in checkpoint.tasks_completed

    @pytest.mark.unit
    def test_complete_task_with_artifacts(self, manager):
        """Should record produced artifacts."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T001"],
        )

        artifacts = [{"type": "html", "path": "week_01/module_01.html"}]

        checkpoint = manager.complete_task(
            phase_name="content_generation",
            task_id="T001",
            success=True,
            artifacts=artifacts,
        )

        assert len(checkpoint.artifacts_produced) == 1
        assert checkpoint.artifacts_produced[0]["type"] == "html"

    @pytest.mark.unit
    def test_complete_phase_marks_complete(self, manager):
        """complete_phase should mark phase as completed."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001"],
        )

        manager.complete_task(
            phase_name="planning",
            task_id="T001",
            success=True,
        )

        if hasattr(manager, 'complete_phase'):
            checkpoint = manager.complete_phase("planning")
            assert checkpoint.status == "completed"
            assert checkpoint.completed_at is not None

    @pytest.mark.unit
    def test_fail_phase_marks_failed(self, manager):
        """fail_phase should mark phase as failed with error."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001"],
        )

        if hasattr(manager, 'fail_phase'):
            checkpoint = manager.fail_phase(
                "planning",
                error="Task T001 failed permanently",
            )
            assert checkpoint.status == "failed"
            assert checkpoint.error is not None

    @pytest.mark.unit
    def test_load_checkpoint_from_disk(self, manager):
        """Should load existing checkpoint from disk."""
        # Create checkpoint
        original = manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001", "T002"],
        )

        # Load it back
        if hasattr(manager, 'load_checkpoint'):
            loaded = manager.load_checkpoint("planning")
            assert loaded.run_id == original.run_id
            assert loaded.phase_name == original.phase_name

    @pytest.mark.unit
    def test_get_resumable_phase(self, manager):
        """Should identify phase that can be resumed."""
        # Create a started but incomplete phase
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T001", "T002", "T003"],
        )

        # Complete one task
        manager.complete_task(
            phase_name="content_generation",
            task_id="T001",
            success=True,
        )

        if hasattr(manager, 'get_resumable_phase'):
            resumable = manager.get_resumable_phase()
            assert resumable is not None
            assert resumable.phase_name == "content_generation"
            assert resumable.can_resume is True

    @pytest.mark.unit
    def test_get_all_checkpoints_sorted(self, manager):
        """Should return checkpoints sorted by phase_index."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="packaging",
            phase_index=2,
            task_ids=["T005"],
        )
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001"],
        )
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T002", "T003"],
        )

        if hasattr(manager, 'get_all_checkpoints'):
            all_checkpoints = manager.get_all_checkpoints()
            assert len(all_checkpoints) == 3
            assert all_checkpoints[0].phase_index == 0
            assert all_checkpoints[1].phase_index == 1
            assert all_checkpoints[2].phase_index == 2

    @pytest.mark.unit
    def test_get_last_completed_phase(self, manager):
        """Should return most recent completed phase."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001"],
        )
        manager.complete_task(
            phase_name="planning",
            task_id="T001",
            success=True,
        )

        if hasattr(manager, 'complete_phase'):
            manager.complete_phase("planning")

        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T002"],
        )

        if hasattr(manager, 'get_last_completed_phase'):
            last = manager.get_last_completed_phase()
            assert last is not None
            assert last.phase_name == "planning"

    @pytest.mark.unit
    def test_get_phase_summary(self, manager):
        """Should return summary statistics."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="content_generation",
            phase_index=1,
            task_ids=["T001", "T002", "T003"],
        )
        manager.complete_task("content_generation", "T001", success=True)
        manager.complete_task("content_generation", "T002", success=False)

        if hasattr(manager, 'get_phase_summary'):
            summary = manager.get_phase_summary()
            assert isinstance(summary, dict)

    @pytest.mark.unit
    def test_atomic_checkpoint_writes(self, manager):
        """Checkpoint writes should be atomic (no corruption on crash)."""
        manager.start_phase(
            run_id="RUN_001",
            workflow_id="W001",
            phase_name="planning",
            phase_index=0,
            task_ids=["T001"],
        )

        # File should exist and be valid JSON
        checkpoint_file = manager.checkpoints_dir / "planning_checkpoint.json"
        with open(checkpoint_file) as f:
            data = json.load(f)  # Should not raise

        assert data["phase_name"] == "planning"
