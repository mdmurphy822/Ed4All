"""Smoke tests for MCP server module."""

import sys
from pathlib import Path

import pytest

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.secure_paths import validate_path_within_root, is_safe_path
from lib.tool_registry import ToolRegistry, get_registry


@pytest.mark.unit
@pytest.mark.mcp
class TestMCPToolModules:
    """Test MCP tool module imports (no FastMCP dependency required)."""

    def test_secure_paths_import(self):
        assert callable(validate_path_within_root)
        assert callable(is_safe_path)


@pytest.mark.unit
@pytest.mark.mcp
class TestToolRegistryIntegration:
    """Test tool registry used by MCP server."""

    def test_tool_registry_import(self):
        registry = get_registry()
        assert registry is not None

    def test_registry_list_tools(self):
        registry = get_registry()
        tools = registry.list_tools()
        assert isinstance(tools, list)

    def test_registry_snapshot(self):
        registry = get_registry()
        snapshot = registry.snapshot()
        assert "tool_count" in snapshot
        assert "snapshot_hash" in snapshot
