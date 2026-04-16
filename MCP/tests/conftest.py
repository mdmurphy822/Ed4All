"""
Ed4All orchestrator/ test configuration and shared fixtures.
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MCP CLIENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_mcp_client():
    """AsyncMock MCP client for executor tests."""
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value={"status": "success", "data": {}})
    return client


@pytest.fixture
def mock_mcp_client_with_errors():
    """MCP client that raises various errors."""
    client = AsyncMock()

    # Will raise different errors based on call count
    call_count = [0]

    async def call_tool_with_errors(tool_name, params):
        call_count[0] += 1
        if call_count[0] == 1:
            raise TimeoutError("Connection timeout")
        elif call_count[0] == 2:
            raise ConnectionError("Connection reset")
        else:
            return {"status": "success"}

    client.call_tool = AsyncMock(side_effect=call_tool_with_errors)
    return client


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def sample_workflows_yaml():
    """Sample workflows.yaml content."""
    return """
workflows:
  course_generation:
    description: "Generate new course content"
    max_concurrent: 10
    phases:
      - name: planning
        agents: [course-outliner]
        timeout_minutes: 30
      - name: content_generation
        agents: [content-generator]
        timeout_minutes: 120
      - name: packaging
        agents: [brightspace-packager]
        timeout_minutes: 30

  batch_dart:
    description: "Batch PDF conversion"
    max_concurrent: 4
    phases:
      - name: conversion
        agents: [dart-converter]
        timeout_minutes: 60
"""


@pytest.fixture
def sample_agents_yaml():
    """Sample agents.yaml content."""
    return """
agents:
  course-outliner:
    description: "Creates course structure and learning objectives"
    tool: create_course_project
    capabilities:
      - course_planning
      - objective_mapping

  content-generator:
    description: "Generates educational content for modules"
    tool: generate_course_content
    capabilities:
      - content_generation
      - wcag_compliance

  brightspace-packager:
    description: "Packages course as IMSCC"
    tool: package_imscc
    capabilities:
      - imscc_packaging

  dart-converter:
    description: "Converts PDF to accessible HTML"
    tool: convert_pdf_multi_source
    capabilities:
      - pdf_conversion
      - accessibility
"""


@pytest.fixture
def sample_config_dir(tmp_path, sample_workflows_yaml, sample_agents_yaml):
    """Create config directory with sample YAML files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "workflows.yaml").write_text(sample_workflows_yaml)
    (config_dir / "agents.yaml").write_text(sample_agents_yaml)

    return config_dir


# =============================================================================
# WORKFLOW STATE FIXTURES
# =============================================================================

@pytest.fixture
def sample_workflow_state():
    """Sample workflow state dictionary."""
    return {
        "workflow_id": "W001",
        "workflow_type": "course_generation",
        "status": "running",
        "created_at": "2025-01-01T00:00:00",
        "phases": {
            "planning": {"status": "completed"},
            "content_generation": {"status": "running"},
            "packaging": {"status": "pending"},
        },
        "tasks": {
            "T001": {
                "task_id": "T001",
                "agent_type": "course-outliner",
                "status": "complete",
                "params": {"course_name": "TEST_101"},
            },
            "T002": {
                "task_id": "T002",
                "agent_type": "content-generator",
                "status": "running",
                "params": {"week": 1, "module": 1},
            },
            "T003": {
                "task_id": "T003",
                "agent_type": "content-generator",
                "status": "pending",
                "params": {"week": 1, "module": 2},
            },
        },
    }


@pytest.fixture
def workflow_state_file(tmp_path, sample_workflow_state):
    """Create workflow state JSON file."""
    workflows_dir = tmp_path / "state" / "workflows"
    workflows_dir.mkdir(parents=True)

    state_path = workflows_dir / "W001.json"
    state_path.write_text(json.dumps(sample_workflow_state, indent=2))

    return state_path


# =============================================================================
# CHECKPOINT FIXTURES
# =============================================================================

@pytest.fixture
def checkpoint_manager(tmp_path):
    """CheckpointManager with temp state directory."""
    try:
        from MCP.hardening.checkpoint import CheckpointManager
        run_path = tmp_path / "runs" / "RUN_001"
        run_path.mkdir(parents=True)
        return CheckpointManager(run_path)
    except ImportError:
        pytest.skip("checkpoint module not available")


@pytest.fixture
def sample_checkpoint_data():
    """Sample checkpoint data dictionary."""
    return {
        "run_id": "RUN_001",
        "workflow_id": "W001",
        "phase_name": "content_generation",
        "phase_index": 1,
        "status": "started",
        "started_at": "2025-01-01T10:00:00",
        "tasks_completed": ["T001", "T002"],
        "tasks_failed": [],
        "tasks_pending": ["T003", "T004", "T005"],
        "last_event_seq": 42,
    }


# =============================================================================
# ERROR CLASSIFIER FIXTURES
# =============================================================================

@pytest.fixture
def error_classifier():
    """ErrorClassifier instance."""
    try:
        from MCP.hardening.error_classifier import ErrorClassifier
        return ErrorClassifier()
    except ImportError:
        pytest.skip("error_classifier module not available")


@pytest.fixture
def poison_pill_detector():
    """PoisonPillDetector instance."""
    try:
        from MCP.hardening.error_classifier import PoisonPillDetector
        return PoisonPillDetector(threshold=3, time_window_seconds=60)
    except ImportError:
        pytest.skip("error_classifier module not available")


@pytest.fixture
def sample_errors():
    """Collection of sample errors for classification testing."""
    return {
        "timeout": TimeoutError("Connection timeout"),
        "connection": ConnectionError("Connection reset by peer"),
        "rate_limit": Exception("Rate limit exceeded, retry after 60s"),
        "file_not_found": FileNotFoundError("No such file: config.yaml"),
        "permission": PermissionError("Access denied"),
        "validation": ValueError("Invalid input: expected string"),
        "memory": MemoryError("Out of memory"),
        "auth": Exception("Authentication failed: invalid API key"),
        "generic": Exception("Something went wrong"),
    }


# =============================================================================
# EXECUTOR FIXTURES
# =============================================================================

@pytest.fixture
def mock_tool_registry():
    """Mock tool registry for executor tests."""
    try:
        from lib.tool_registry import SandboxLevel, ToolCapability, ToolRegistry

        registry = ToolRegistry()

        # Register test tools
        registry.register(ToolCapability(
            tool_name="create_course_project",
            version="1.0.0",
            description="Create course project",
            sandbox_level=SandboxLevel.RESTRICTED,
        ))
        registry.register(ToolCapability(
            tool_name="generate_course_content",
            version="1.0.0",
            description="Generate content",
            sandbox_level=SandboxLevel.RESTRICTED,
        ))
        registry.register(ToolCapability(
            tool_name="package_imscc",
            version="1.0.0",
            description="Package IMSCC",
            sandbox_level=SandboxLevel.RESTRICTED,
        ))

        return registry

    except ImportError:
        # Return mock if not available
        registry = Mock()
        registry.get.return_value = Mock(tool_name="test_tool")
        registry.list_tools.return_value = ["create_course_project", "generate_course_content"]
        return registry


# =============================================================================
# TASK FIXTURES
# =============================================================================

@pytest.fixture
def sample_task():
    """Sample task dictionary."""
    return {
        "task_id": "T001",
        "agent_type": "content-generator",
        "status": "pending",
        "params": {
            "course_name": "TEST_101",
            "week": 1,
            "module": 1,
            "content_type": "introduction",
        },
        "dependencies": [],
    }


@pytest.fixture
def batch_tasks():
    """Batch of tasks for parallel execution testing."""
    return [
        {
            "task_id": f"T{i:03d}",
            "agent_type": "content-generator",
            "status": "pending",
            "params": {"week": (i // 3) + 1, "module": (i % 3) + 1},
            "dependencies": [],
        }
        for i in range(10)
    ]
