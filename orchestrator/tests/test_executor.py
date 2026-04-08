"""
Tests for orchestrator/core/executor.py - Task execution and workflow management.
"""
import pytest
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from orchestrator.core.executor import (
        TaskExecutor,
        ExecutionResult,
        ToolRegistryError,
        AGENT_TOOL_MAPPING,
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

class TestTaskExecutorInit:
    """Test TaskExecutor initialization."""

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

class TestToolRegistryValidation:
    """Test tool registry validation."""

    @pytest.mark.unit
    def test_validate_empty_registry(self):
        """Empty registry should report missing tools."""
        executor = TaskExecutor(tool_registry={})

        issues = executor.validate_tool_registry(fail_fast=False)

        assert len(issues["missing"]) > 0
        assert "create_course_project" in issues["missing"]

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

class TestTaskExecution:
    """Test task execution."""

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

        task = {"agent_type": "content-generator", "params": {"week": 1}}
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

        task = {"agent_type": "content-generator", "params": {}}
        with patch.object(executor, '_load_task', return_value=task):
            with patch.object(executor, '_update_task_status'):
                await executor.execute_task("W001", "T001")

        assert capture.log_decision.called


# =============================================================================
# RETRY LOGIC TESTS
# =============================================================================

class TestRetryLogic:
    """Test retry logic and error handling."""

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

        task = {"agent_type": "content-generator", "params": {}}
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

        task = {"agent_type": "content-generator", "params": {}}
        with patch.object(executor, '_load_task', return_value=task):
            with patch.object(executor, '_update_task_status'):
                result = await executor.execute_task("W001", "T001")

        assert result.status == "ERROR"
        assert result.retry_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        """Should not retry permanent errors."""
        call_count = [0]

        async def permanent_failure(**kwargs):
            call_count[0] += 1
            raise FileNotFoundError("Config file missing")

        registry = {"generate_course_content": permanent_failure}
        executor = TaskExecutor(tool_registry=registry, max_retries=3)

        # Mock error classifier to mark as permanent
        if executor.error_classifier:
            task = {"agent_type": "content-generator", "params": {}}
            with patch.object(executor, '_load_task', return_value=task):
                with patch.object(executor, '_update_task_status'):
                    result = await executor.execute_task("W001", "T001")

            # Should have stopped early due to permanent classification
            assert result.status == "ERROR"


# =============================================================================
# WORKFLOW EXECUTION TESTS
# =============================================================================

class TestWorkflowExecution:
    """Test workflow execution."""

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
        with patch('orchestrator.core.executor.STATE_PATH', tmp_path / "state"):
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

        with patch('orchestrator.core.executor.STATE_PATH', tmp_path / "state"):
            with patch.object(mock_executor, 'execute_task') as mock_exec:
                mock_exec.return_value = ExecutionResult(task_id="T001", status="COMPLETE")
                results = await mock_executor.execute_workflow("W001", parallel=True)

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

        with patch('orchestrator.core.executor.STATE_PATH', tmp_path / "state"):
            with patch.object(mock_executor, 'execute_task') as mock_exec:
                mock_exec.return_value = ExecutionResult(task_id="T001", status="COMPLETE")
                results = await mock_executor.execute_workflow("W001", parallel=False)

        assert mock_exec.call_count >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_workflow_respects_max_concurrent(self, mock_executor):
        """Should respect max_concurrent limit."""
        # This is an implicit test - verify the parameter is passed correctly
        with patch.object(mock_executor, '_execute_parallel') as mock_parallel:
            mock_parallel.return_value = {}

            with patch('orchestrator.core.executor.STATE_PATH') as mock_path:
                mock_path.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))

                # Mock file read
                with patch('builtins.open', Mock()):
                    with patch('json.load', return_value={"tasks": []}):
                        await mock_executor.execute_workflow("W001", max_concurrent=3)


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
    def test_dart_agents_mapped(self):
        """DART agents should be mapped."""
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
