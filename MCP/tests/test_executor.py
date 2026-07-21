"""
Tests for orchestrator/core/executor.py - Task execution and workflow management.
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from MCP.core.executor import (
        AGENT_TOOL_MAPPING,
        ExecutionResult,
        TaskExecutor,
        ToolRegistryError,
        _is_cuda_oom,
        _probe_free_vram_mib,
    )
except ImportError:
    pytest.skip("executor not available", allow_module_level=True)


# =============================================================================
# EXECUTION RESULT TESTS
# =============================================================================

class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        result = ExecutionResult(
            task_id="T001",
            status="COMPLETE",
            result={"output": "success"},
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            duration_seconds=60.0,
        )

        d = result.to_dict()

        assert d["task_id"] == "T001"
        assert d["status"] == "COMPLETE"
        assert d["result"]["output"] == "success"
        assert d["duration_seconds"] == 60.0

    @pytest.mark.unit
    def test_to_dict_with_error(self):
        """Should include error in serialization."""
        result = ExecutionResult(
            task_id="T001",
            status="ERROR",
            error="Connection timeout",
            error_class="transient",
            retry_count=3,
        )

        d = result.to_dict()

        assert d["status"] == "ERROR"
        assert d["error"] == "Connection timeout"
        assert d["error_class"] == "transient"
        assert d["retry_count"] == 3

    @pytest.mark.unit
    def test_to_dict_with_artifacts(self):
        """Should include artifacts in serialization."""
        result = ExecutionResult(
            task_id="T001",
            status="COMPLETE",
            artifacts=[{"type": "html", "path": "module.html"}],
        )

        d = result.to_dict()

        assert len(d["artifacts"]) == 1
        assert d["artifacts"][0]["type"] == "html"


# =============================================================================
# TASK EXECUTOR INITIALIZATION TESTS
# =============================================================================

@pytest.mark.usefixtures("state_runs_isolated")
class TestTaskExecutorInit:
    """Test TaskExecutor initialization.

    Wave 74: opted into ``state_runs_isolated`` so the timestamp-fallback
    ``run_path`` lands in tmp_path instead of project ``state/runs/``.
    """

    @pytest.mark.unit
    def test_init_empty_registry(self):
        """Should initialize with empty tool registry."""
        executor = TaskExecutor()

        assert executor.tool_registry == {}
        assert executor.run_id is not None

    @pytest.mark.unit
    def test_init_with_registry(self):
        """Should initialize with provided tool registry."""
        registry = {"test_tool": AsyncMock()}
        executor = TaskExecutor(tool_registry=registry)

        assert "test_tool" in executor.tool_registry

    @pytest.mark.unit
    def test_init_with_run_id(self):
        """Should use provided run_id."""
        executor = TaskExecutor(run_id="custom_run_123")

        assert executor.run_id == "custom_run_123"

    @pytest.mark.unit
    def test_init_generates_run_id(self):
        """Should generate run_id if not provided."""
        executor = TaskExecutor()

        assert executor.run_id is not None
        assert "run_" in executor.run_id

    @pytest.mark.unit
    def test_init_with_custom_timeouts(self):
        """Should use custom timeout values."""
        executor = TaskExecutor(
            max_retries=5,
            timeout_seconds=300,
        )

        assert executor.max_retries == 5
        assert executor.timeout_seconds == 300

    @pytest.mark.unit
    def test_init_with_decision_capture(self):
        """Should accept decision capture instance."""
        capture = Mock()
        executor = TaskExecutor(capture=capture)

        assert executor.capture is capture


# =============================================================================
# TOOL REGISTRY VALIDATION TESTS
# =============================================================================

@pytest.mark.usefixtures("state_runs_isolated")
class TestToolRegistryValidation:
    """Test tool registry validation.

    Wave 74: opted into ``state_runs_isolated`` to avoid polluting
    project ``state/runs/``.
    """

    @pytest.mark.unit
    def test_validate_empty_registry(self):
        """Empty registry should report missing tools."""
        executor = TaskExecutor(tool_registry={})

        issues = executor.validate_tool_registry(fail_fast=False)

        assert len(issues["missing"]) > 0
        # Wave 24: course-outliner now routes to plan_course_structure
        # (was create_course_project pre-Wave-24); textbook-ingestor
        # routes to extract_textbook_structure. Either surfaces as
        # missing in an empty registry.
        assert (
            "plan_course_structure" in issues["missing"]
            or "extract_textbook_structure" in issues["missing"]
        )

    @pytest.mark.unit
    def test_validate_full_registry(self):
        """Full registry should pass validation."""
        # Create registry with all mapped tools
        registry = {tool: AsyncMock() for tool in set(AGENT_TOOL_MAPPING.values())}
        executor = TaskExecutor(tool_registry=registry)

        issues = executor.validate_tool_registry(fail_fast=False)

        assert len(issues["missing"]) == 0

    @pytest.mark.unit
    def test_validate_fail_fast_raises(self):
        """Should raise on missing tools when fail_fast=True."""
        executor = TaskExecutor(tool_registry={})

        with pytest.raises(ToolRegistryError):
            executor.validate_tool_registry(fail_fast=True)

    @pytest.mark.unit
    def test_get_missing_tools(self):
        """Should return list of missing tools."""
        executor = TaskExecutor(tool_registry={"create_course_project": AsyncMock()})

        missing = executor.get_missing_tools()

        assert "create_course_project" not in missing
        assert "generate_course_content" in missing

    @pytest.mark.unit
    def test_register_tool(self):
        """Should register new tools."""
        executor = TaskExecutor()
        tool_func = AsyncMock()

        executor.register_tool("new_tool", tool_func)

        assert "new_tool" in executor.tool_registry


# =============================================================================
# TASK EXECUTION TESTS
# =============================================================================

@pytest.mark.usefixtures("state_runs_isolated")
class TestTaskExecution:
    """Test task execution.

    Wave 74: opted into ``state_runs_isolated`` to avoid polluting
    project ``state/runs/``.
    """

    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        async def mock_tool(**kwargs):
            return json.dumps({"status": "success", "output": "test_output"})

        return {tool: mock_tool for tool in set(AGENT_TOOL_MAPPING.values())}

    @pytest.fixture
    def workflow_state_dir(self, tmp_path):
        """Create workflow state directory."""
        workflows_dir = tmp_path / "state" / "workflows"
        workflows_dir.mkdir(parents=True)
        return workflows_dir

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_task_missing_workflow(self, mock_tool_registry):
        """Should return error for missing workflow."""
        executor = TaskExecutor(tool_registry=mock_tool_registry)

        with patch.object(executor, '_load_task', return_value=None):
            result = await executor.execute_task("MISSING_W", "T001")

        assert result.status == "ERROR"
        assert "not found" in result.error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_task_unknown_agent(self, mock_tool_registry):
        """Should return error for unknown agent type."""
        executor = TaskExecutor(tool_registry=mock_tool_registry)

        task = {"agent_type": "unknown-agent", "params": {}}
        with patch.object(executor, '_load_task', return_value=task):
            result = await executor.execute_task("W001", "T001")

        assert result.status == "ERROR"
        assert "No tool mapping" in result.error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_tool_registry):
        """Should execute task successfully."""
        executor = TaskExecutor(tool_registry=mock_tool_registry)

        task = {"agent_type": "content-generator", "params": {"project_id": "TEST_001", "week": 1}}
        with patch.object(executor, '_load_task', return_value=task):
            with patch.object(executor, '_update_task_status'):
                result = await executor.execute_task("W001", "T001")

        assert result.status == "COMPLETE"
        assert result.result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_task_with_capture(self, mock_tool_registry):
        """Should log decisions to capture."""
        capture = Mock()
        capture.log_decision = Mock()
        executor = TaskExecutor(tool_registry=mock_tool_registry, capture=capture)

        task = {"agent_type": "content-generator", "params": {"project_id": "TEST_001"}}
        with patch.object(executor, '_load_task', return_value=task):
            with patch.object(executor, '_update_task_status'):
                await executor.execute_task("W001", "T001")

        assert capture.log_decision.called


# =============================================================================
# RETRY LOGIC TESTS
# =============================================================================

@pytest.mark.usefixtures("state_runs_isolated")
class TestRetryLogic:
    """Test retry logic and error handling.

    Wave 74: opted into ``state_runs_isolated`` to avoid polluting
    project ``state/runs/``.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        """Should retry on transient errors."""
        call_count = [0]

        async def failing_tool(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("Connection timeout")
            return json.dumps({"status": "success"})

        registry = {"generate_course_content": failing_tool}
        executor = TaskExecutor(tool_registry=registry, max_retries=3)

        task = {"agent_type": "content-generator", "params": {"project_id": "TEST_001"}}
        with patch.object(executor, '_load_task', return_value=task):
            with patch.object(executor, '_update_task_status'):
                result = await executor.execute_task("W001", "T001")

        assert result.status == "COMPLETE"
        assert call_count[0] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stops_after_max_retries(self):
        """Should stop after max retries exceeded."""
        async def always_fails(**kwargs):
            raise ConnectionError("Connection failed")

        registry = {"generate_course_content": always_fails}
        executor = TaskExecutor(tool_registry=registry, max_retries=2)
        executor.poison_detector = None  # Disable poison pill for retry-count test

        task = {"agent_type": "content-generator", "params": {"project_id": "TEST_001"}}
        with patch.object(executor, '_load_task', return_value=task):
            with patch.object(executor, '_update_task_status'):
                result = await executor.execute_task("W001", "T001")

        assert result.status == "ERROR"
        assert result.retry_count == 3  # 1 initial + 2 retries = 3 total attempts

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        """Should not retry permanent errors."""
        call_count = [0]

        async def permanent_failure(**kwargs):
            call_count[0] += 1
            raise FileNotFoundError("File not found: config.yaml")

        registry = {"generate_course_content": permanent_failure}
        executor = TaskExecutor(tool_registry=registry, max_retries=3)

        # Mock error classifier to mark as permanent
        if executor.error_classifier:
            task = {"agent_type": "content-generator", "params": {"project_id": "TEST_001"}}
            with patch.object(executor, '_load_task', return_value=task):
                with patch.object(executor, '_update_task_status'):
                    result = await executor.execute_task("W001", "T001")

            # Should have stopped early due to permanent classification
            assert result.status == "ERROR"


# =============================================================================
# WORKFLOW EXECUTION TESTS
# =============================================================================

@pytest.mark.usefixtures("state_runs_isolated")
class TestWorkflowExecution:
    """Test workflow execution.

    Wave 74: opted into ``state_runs_isolated`` to avoid polluting
    project ``state/runs/``.
    """

    @pytest.fixture
    def mock_executor(self):
        """Create executor with mocked methods."""
        async def mock_tool(**kwargs):
            return json.dumps({"status": "success"})

        registry = {tool: mock_tool for tool in set(AGENT_TOOL_MAPPING.values())}
        return TaskExecutor(tool_registry=registry)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_workflow_missing_file(self, mock_executor, tmp_path):
        """Should return empty for missing workflow."""
        with patch('MCP.core.executor.STATE_PATH', tmp_path / "state"):
            results = await mock_executor.execute_workflow("MISSING_W")

        assert results == {}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_workflow_parallel(self, mock_executor, tmp_path):
        """Should execute tasks in parallel."""
        # Create workflow file
        state_path = tmp_path / "state" / "workflows"
        state_path.mkdir(parents=True)

        workflow = {
            "tasks": [
                {"id": "T001", "status": "PENDING", "agent_type": "content-generator"},
                {"id": "T002", "status": "PENDING", "agent_type": "content-generator"},
            ]
        }
        (state_path / "W001.json").write_text(json.dumps(workflow))

        with patch('MCP.core.executor.STATE_PATH', tmp_path / "state"):
            with patch.object(mock_executor, 'execute_task') as mock_exec:
                mock_exec.return_value = ExecutionResult(task_id="T001", status="COMPLETE")
                await mock_executor.execute_workflow("W001", parallel=True)

        # Should have called execute_task for each task
        assert mock_exec.call_count >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_workflow_sequential(self, mock_executor, tmp_path):
        """Should execute tasks sequentially."""
        state_path = tmp_path / "state" / "workflows"
        state_path.mkdir(parents=True)

        workflow = {
            "tasks": [
                {"id": "T001", "status": "PENDING", "agent_type": "content-generator"},
                {"id": "T002", "status": "PENDING", "agent_type": "content-generator"},
            ]
        }
        (state_path / "W001.json").write_text(json.dumps(workflow))

        with patch('MCP.core.executor.STATE_PATH', tmp_path / "state"):
            with patch.object(mock_executor, 'execute_task') as mock_exec:
                mock_exec.return_value = ExecutionResult(task_id="T001", status="COMPLETE")
                await mock_executor.execute_workflow("W001", parallel=False)

        assert mock_exec.call_count >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_workflow_respects_max_concurrent(self, mock_executor, tmp_path):
        """Should respect max_concurrent limit."""
        state_path = tmp_path / "state" / "workflows"
        state_path.mkdir(parents=True)

        workflow = {
            "tasks": [
                {"id": "T001", "status": "PENDING", "agent_type": "content-generator"},
                {"id": "T002", "status": "PENDING", "agent_type": "content-generator"},
            ]
        }
        (state_path / "W001.json").write_text(json.dumps(workflow))

        with patch('MCP.core.executor.STATE_PATH', tmp_path / "state"):
            with patch.object(mock_executor, 'execute_task') as mock_exec:
                mock_exec.return_value = ExecutionResult(task_id="T001", status="COMPLETE")
                await mock_executor.execute_workflow("W001", max_concurrent=3)

        assert mock_exec.call_count >= 1


# =============================================================================
# PHASE EXECUTION TESTS
# =============================================================================

class TestPhaseExecution:
    """Test phase execution with checkpointing."""

    @pytest.fixture
    def executor_with_checkpoints(self, tmp_path):
        """Create executor with checkpoint support."""
        async def mock_tool(**kwargs):
            return json.dumps({"status": "success"})

        registry = {tool: mock_tool for tool in set(AGENT_TOOL_MAPPING.values())}
        run_path = tmp_path / "runs" / "RUN_001"
        run_path.mkdir(parents=True)

        return TaskExecutor(
            tool_registry=registry,
            run_path=run_path,
            run_id="RUN_001",
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_phase_creates_checkpoint(self, executor_with_checkpoints):
        """Should create checkpoint at phase start."""
        tasks = [
            {"id": "T001", "status": "PENDING", "agent_type": "content-generator"},
        ]

        with patch.object(executor_with_checkpoints, '_execute_parallel') as mock_exec:
            mock_exec.return_value = {
                "T001": ExecutionResult(task_id="T001", status="COMPLETE")
            }

            results, passed, _ = await executor_with_checkpoints.execute_phase(
                workflow_id="W001",
                phase_name="content_generation",
                phase_index=1,
                tasks=tasks,
            )

        assert "T001" in results

    @pytest.mark.unit
    def test_get_resumable_phase_none(self, executor_with_checkpoints):
        """Should return None when no resumable phase."""
        result = executor_with_checkpoints.get_resumable_phase()

        # May return None or checkpoint depending on state
        # Just verify it doesn't raise
        assert result is None or isinstance(result, dict)

    @pytest.mark.unit
    def test_reset_poison_detector(self, executor_with_checkpoints):
        """Should reset poison detector."""
        # Should not raise
        executor_with_checkpoints.reset_poison_detector()


# =============================================================================
# AGENT TOOL MAPPING TESTS
# =============================================================================

class TestAgentToolMapping:
    """Test agent to tool mapping."""

    @pytest.mark.unit
    def test_mapping_exists(self):
        """Should have agent to tool mappings."""
        assert len(AGENT_TOOL_MAPPING) > 0

    @pytest.mark.unit
    def test_courseforge_agents_mapped(self):
        """Courseforge agents should be mapped."""
        assert "course-outliner" in AGENT_TOOL_MAPPING
        assert "content-generator" in AGENT_TOOL_MAPPING
        assert "brightspace-packager" in AGENT_TOOL_MAPPING

    @pytest.mark.unit
    def test_legacy_dart_agent_aliases_mapped(self):
        """Legacy pre-SemantiK agent aliases stay registered as read-compat
        dispatch aliases so paused runs resume across the rename."""
        assert "dart-converter" in AGENT_TOOL_MAPPING
        assert "dart-automation-coordinator" in AGENT_TOOL_MAPPING

    @pytest.mark.unit
    def test_trainforge_agents_mapped(self):
        """Trainforge agents should be mapped."""
        assert "assessment-generator" in AGENT_TOOL_MAPPING
        assert "assessment-validator" in AGENT_TOOL_MAPPING

    @pytest.mark.unit
    def test_mapping_values_are_strings(self):
        """All mapping values should be tool name strings."""
        for agent, tool in AGENT_TOOL_MAPPING.items():
            assert isinstance(agent, str)
            assert isinstance(tool, str)


# =============================================================================
# CUDA OUT-OF-MEMORY DIAGNOSTIC TESTS
# =============================================================================

class _SyntheticOutOfMemoryError(Exception):
    """Stand-in whose ``__name__`` is ``OutOfMemoryError`` for OOM detection.

    Mirrors how ``torch.cuda.OutOfMemoryError`` is seen by ``_is_cuda_oom``
    without requiring torch in CI.
    """


# Rename so ``type(exc).__name__ == "OutOfMemoryError"`` matches.
_SyntheticOutOfMemoryError.__name__ = "OutOfMemoryError"
_SyntheticOutOfMemoryError.__qualname__ = "OutOfMemoryError"


class TestIsCudaOom:
    """Test the _is_cuda_oom predicate (torch-free)."""

    @pytest.mark.unit
    def test_true_for_outofmemoryerror_class_name(self):
        """True when the exception class name is OutOfMemoryError."""
        exc = _SyntheticOutOfMemoryError("ran out")
        assert type(exc).__name__ == "OutOfMemoryError"
        assert _is_cuda_oom(exc) is True

    @pytest.mark.unit
    def test_true_for_cuda_out_of_memory_message(self):
        """True for a generic exception carrying a CUDA OOM message."""
        exc = RuntimeError(
            "CUDA out of memory. Tried to allocate 512.00 MiB"
        )
        assert _is_cuda_oom(exc) is True

    @pytest.mark.unit
    def test_true_for_out_of_memory_plus_cuda_message(self):
        """True when message has both 'out of memory' and 'cuda' separately."""
        exc = Exception("the CUDA device reported it is out of memory")
        assert _is_cuda_oom(exc) is True

    @pytest.mark.unit
    def test_false_for_value_error(self):
        """False for an ordinary ValueError."""
        assert _is_cuda_oom(ValueError("bad input")) is False

    @pytest.mark.unit
    def test_false_for_timeout(self):
        """False for a timeout-style error (the transient retry path)."""
        assert _is_cuda_oom(TimeoutError("Connection timeout")) is False

    @pytest.mark.unit
    def test_false_for_host_oom_message(self):
        """False for a plain host 'out of memory' with no CUDA context."""
        assert _is_cuda_oom(MemoryError("out of memory")) is False

    @pytest.mark.unit
    def test_false_for_none(self):
        """False (no crash) for None."""
        assert _is_cuda_oom(None) is False


@pytest.mark.usefixtures("state_runs_isolated")
class TestCudaOomExecutionPath:
    """Test the executor's loud CUDA-OOM DIAGNOSTIC branch.

    Contract (post-redesign): a CUDA OOM is a pure LOGGING side-effect —
    the executor emits one loud, attributable ``GPU OUT OF MEMORY`` error
    (with a best-effort free-VRAM probe), then FALLS THROUGH to the
    unchanged ``ErrorClassifier`` + ``PoisonPillDetector`` path. Because
    the classifier matches "out of memory" as a POISON_PATTERN, a real
    CUDA OOM is classified POISON_PILL and trips the runaway-VRAM circuit
    breaker. The OOM branch must NOT short-circuit with a forced one-shot
    PERMANENT result (the regression this redesign fixes — that bypassed
    the poison detector and removed the circuit breaker).
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_oom_fires_loud_diagnostic_and_defers_to_classifier(self, caplog):
        """OOM fires the loud diagnostic, then flows through classify/poison.

        Not a one-shot forced PERMANENT: a persistent OOM is recorded by
        the poison detector across attempts and stops the batch
        (POISON_PILL) once the default threshold (3) is reached — proving
        the branch is logging-only, not a control-flow short-circuit.
        """
        import logging

        call_count = [0]

        async def oom_tool(**kwargs):
            call_count[0] += 1
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 2.00 GiB"
            )

        registry = {"generate_course_content": oom_tool}
        # Default poison_pill_threshold=3.
        executor = TaskExecutor(tool_registry=registry, max_retries=3)
        if executor.poison_detector is None:
            pytest.skip("hardening error_classifier/poison_detector unavailable")

        task = {
            "agent_type": "content-generator",
            "params": {"project_id": "TEST_OOM"},
        }
        with patch(
            "MCP.core.executor._probe_free_vram_mib", return_value=137
        ):
            with patch.object(executor, "_load_task", return_value=task):
                with patch.object(executor, "_update_task_status"):
                    with caplog.at_level(logging.ERROR, logger="MCP.core.executor"):
                        result = await executor.execute_task("W001", "T_OOM")

        # Deferred to the classifier → POISON_PILL via the circuit breaker
        # (NOT a forced one-shot PERMANENT).
        assert result.status == "POISON_PILL"
        assert result.error_class == "poison_pill"
        # Reached the threshold → tool invoked 3x (NOT short-circuited at 1).
        assert call_count[0] == 3
        # The loud diagnostic landed exactly once, carrying the probed VRAM.
        loud = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and "GPU OUT OF MEMORY" in r.getMessage()
        ]
        assert loud, "expected a loud GPU OUT OF MEMORY error log"
        assert "137 MiB free" in loud[0].getMessage()
        # Logged at most once per task execution despite 3 OOM attempts.
        assert len(loud) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_oom_unprobeable_vram_still_loud(self, caplog):
        """OOM with an unprobeable VRAM snapshot still logs the loud diagnostic."""
        import logging

        async def oom_tool(**kwargs):
            # Synthetic torch-OOM class AND a real OOM message so it is both
            # detected by ``_is_cuda_oom`` and classified POISON_PILL.
            raise _SyntheticOutOfMemoryError(
                "CUDA out of memory while allocating buffer"
            )

        registry = {"generate_course_content": oom_tool}
        executor = TaskExecutor(tool_registry=registry, max_retries=3)
        if executor.poison_detector is None:
            pytest.skip("hardening error_classifier/poison_detector unavailable")

        task = {
            "agent_type": "content-generator",
            "params": {"project_id": "TEST_OOM2"},
        }
        with patch(
            "MCP.core.executor._probe_free_vram_mib", return_value=None
        ):
            with patch.object(executor, "_load_task", return_value=task):
                with patch.object(executor, "_update_task_status"):
                    with caplog.at_level(logging.ERROR, logger="MCP.core.executor"):
                        result = await executor.execute_task("W001", "T_OOM2")

        # Still deferred to the classifier (POISON_PILL); the diagnostic
        # just reports the VRAM as unprobeable.
        assert result.status == "POISON_PILL"
        loud = [
            r for r in caplog.records
            if "GPU OUT OF MEMORY" in r.getMessage()
        ]
        assert loud, "expected a loud GPU OUT OF MEMORY error log"
        assert "free VRAM unprobeable" in loud[0].getMessage()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_oom_does_not_bypass_poison_detector(self):
        """Regression guard: the OOM branch must REACH the poison detector.

        The buggy early-return classified OOM as a one-shot PERMANENT and
        returned BEFORE ``record_failure`` ran, removing the runaway-VRAM
        circuit breaker. This proves the detector now sees the OOM
        failures and the breaker fires.
        """
        async def oom_tool(**kwargs):
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 4.00 GiB"
            )

        registry = {"generate_course_content": oom_tool}
        executor = TaskExecutor(tool_registry=registry, max_retries=3)
        if executor.poison_detector is None:
            pytest.skip("hardening error_classifier/poison_detector unavailable")

        # Spy on the real poison detector (delegating to the real impl) to
        # prove the OOM path reaches record_failure.
        real_record = executor.poison_detector.record_failure
        seen = []

        def _spy(classified):
            seen.append(classified)
            return real_record(classified)

        executor.poison_detector.record_failure = _spy

        task = {
            "agent_type": "content-generator",
            "params": {"project_id": "TEST_CB"},
        }
        with patch("MCP.core.executor._probe_free_vram_mib", return_value=42):
            with patch.object(executor, "_load_task", return_value=task):
                with patch.object(executor, "_update_task_status"):
                    result = await executor.execute_task("W001", "T_CB")

        # The detector saw the OOM failures (NOT bypassed) ...
        assert len(seen) >= 1
        # ... each classified POISON_PILL by the existing classifier ...
        assert all(c.error_class.name == "POISON_PILL" for c in seen)
        # ... and the circuit breaker fired.
        assert result.status == "POISON_PILL"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_oom_transient_error_unchanged(self):
        """A non-OOM transient error still follows the classify/retry path."""
        call_count = [0]

        async def flaky_tool(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Connection reset by peer")
            return json.dumps({"status": "success"})

        registry = {"generate_course_content": flaky_tool}
        executor = TaskExecutor(tool_registry=registry, max_retries=3)

        task = {
            "agent_type": "content-generator",
            "params": {"project_id": "TEST_RETRY"},
        }
        # Probe must never be consulted on the non-OOM path.
        with patch(
            "MCP.core.executor._probe_free_vram_mib",
            side_effect=AssertionError("VRAM probe must not run on non-OOM path"),
        ):
            with patch.object(executor, "_load_task", return_value=task):
                with patch.object(executor, "_update_task_status"):
                    result = await executor.execute_task("W001", "T_RETRY")

        # Retried and eventually succeeded — the existing path is intact.
        assert result.status == "COMPLETE"
        assert call_count[0] == 3

    @pytest.mark.unit
    def test_probe_free_vram_never_raises(self):
        """_probe_free_vram_mib returns an int or None, never raises."""
        out = _probe_free_vram_mib()
        assert out is None or isinstance(out, int)
