"""
Ed4All lib/ test configuration and shared fixtures.
"""
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PATH AND FILESYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def temp_root(tmp_path):
    """Isolated root directory for path validation tests."""
    root = tmp_path / "root"
    root.mkdir()
    return root


@pytest.fixture
def temp_file_in_root(temp_root):
    """Create a test file within temp_root."""
    test_file = temp_root / "test_file.txt"
    test_file.write_text("test content")
    return test_file


@pytest.fixture
def nested_dir_structure(temp_root):
    """Create nested directory structure for testing."""
    dirs = ["subdir1", "subdir1/nested", "subdir2"]
    for d in dirs:
        (temp_root / d).mkdir(parents=True, exist_ok=True)

    # Create some files
    (temp_root / "file1.txt").write_text("content1")
    (temp_root / "subdir1/file2.txt").write_text("content2")
    (temp_root / "subdir1/nested/file3.txt").write_text("content3")

    return temp_root


# =============================================================================
# ZIP FILE FIXTURES
# =============================================================================

@pytest.fixture
def valid_zip(tmp_path):
    """Create a valid ZIP file for extraction tests."""
    zip_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("file1.txt", "content1")
        zf.writestr("subdir/file2.txt", "content2")
        zf.writestr("subdir/nested/file3.html", "<html></html>")
    return zip_path


@pytest.fixture
def zip_with_traversal(tmp_path):
    """Create a malicious ZIP with path traversal."""
    zip_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("../escape.txt", "malicious")
    return zip_path


@pytest.fixture
def zip_with_absolute_path(tmp_path):
    """Create a ZIP with absolute path entry."""
    zip_path = tmp_path / "absolute.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("/etc/passwd", "fake")
    return zip_path


@pytest.fixture
def large_zip(tmp_path):
    """Create a ZIP with many files for limit testing."""
    zip_path = tmp_path / "large.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for i in range(100):
            zf.writestr(f"file_{i}.txt", f"content {i}")
    return zip_path


# =============================================================================
# STATE AND JSON FIXTURES
# =============================================================================

@pytest.fixture
def sample_json_data():
    """Sample JSON-serializable data."""
    return {
        "name": "test_workflow",
        "status": "running",
        "tasks": [
            {"id": "T001", "status": "complete"},
            {"id": "T002", "status": "pending"}
        ],
        "metadata": {
            "created": "2025-01-01T00:00:00",
            "version": "1.0.0"
        }
    }


@pytest.fixture
def temp_json_file(tmp_path, sample_json_data):
    """Create a temporary JSON file."""
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(sample_json_data, indent=2))
    return json_path


# =============================================================================
# TOOL REGISTRY FIXTURES
# =============================================================================

@pytest.fixture
def sample_tool_capability():
    """Pre-configured ToolCapability for registry tests."""
    try:
        from lib.tool_registry import SandboxLevel, ToolCapability
        return ToolCapability(
            name="test_tool",
            description="A test tool for unit tests",
            schema_version="1.0.0",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "integer"}
                },
                "required": ["param1"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            },
            sandbox_level=SandboxLevel.RESTRICTED,
            allowed_paths=["/home/bacon/Desktop/Ed4All/state"],
            blocked_paths=["/home/bacon/Desktop/Ed4All/.env"]
        )
    except ImportError:
        pytest.skip("tool_registry not available")


@pytest.fixture
def empty_registry():
    """Create an empty ToolRegistry instance."""
    try:
        from lib.tool_registry import ToolRegistry
        return ToolRegistry()
    except ImportError:
        pytest.skip("tool_registry not available")


# =============================================================================
# DECISION CAPTURE FIXTURES
# =============================================================================

@pytest.fixture
def mock_run_context():
    """Mocked RunContext for decision capture tests."""
    context = Mock()
    context.run_id = "RUN_20250101_120000"
    context.course_id = "TEST_101"
    context.workflow_id = "W001"
    context.run_dir = Path("/tmp/test_run")
    return context


@pytest.fixture
def capture_output_dir(tmp_path):
    """Directory for decision capture output."""
    output_dir = tmp_path / "training-captures" / "test"
    output_dir.mkdir(parents=True)
    return output_dir


# =============================================================================
# ERROR FIXTURES
# =============================================================================

@pytest.fixture
def sample_exceptions():
    """Collection of sample exceptions for error taxonomy testing."""
    return {
        "file_not_found": FileNotFoundError("No such file: test.txt"),
        "permission": PermissionError("Access denied"),
        "timeout": TimeoutError("Operation timed out"),
        "memory": MemoryError("Out of memory"),
        "type": TypeError("Expected string, got int"),
        "value": ValueError("Invalid value: -1"),
        "key": KeyError("missing_key"),
        "io": OSError("Disk full"),
        "generic": Exception("Something went wrong"),
    }


# =============================================================================
# WORKFLOW FIXTURES
# =============================================================================

@pytest.fixture
def sample_workflow_json(tmp_path):
    """Sample workflow state file."""
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(parents=True)

    workflow_data = {
        "workflow_id": "W001",
        "workflow_type": "course_generation",
        "status": "running",
        "created_at": "2025-01-01T00:00:00",
        "tasks": {
            "T001": {
                "task_id": "T001",
                "agent_type": "course-outliner",
                "status": "complete",
                "params": {"course_name": "TEST_101"}
            },
            "T002": {
                "task_id": "T002",
                "agent_type": "content-generator",
                "status": "pending",
                "params": {"week": 1, "module": 1}
            }
        }
    }

    workflow_path = workflow_dir / "W001.json"
    workflow_path.write_text(json.dumps(workflow_data, indent=2))

    return workflow_path
