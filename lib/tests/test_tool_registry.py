"""
Tests for lib/tool_registry.py - Tool capabilities and sandbox enforcement.
"""
import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.tool_registry import (
        ToolCapability,
        ToolRegistry,
        ValidationResult,
        SandboxLevel,
        get_registry,
        register_tool,
    )
except ImportError:
    pytest.skip("tool_registry not available", allow_module_level=True)


# =============================================================================
# TOOL CAPABILITY TESTS
# =============================================================================

class TestToolCapability:
    """Test ToolCapability dataclass."""

    @pytest.mark.unit
    def test_to_dict_roundtrip(self):
        """ToolCapability should serialize and deserialize correctly."""
        original = ToolCapability(
            tool_name="test_tool",
            version="1.0.0",
            description="A test tool",
            sandbox_level=SandboxLevel.RESTRICTED,
            allowed_paths=["/home/user/safe"],
            can_write_files=True,
        )

        as_dict = original.to_dict()
        restored = ToolCapability.from_dict(as_dict)

        assert restored.tool_name == original.tool_name
        assert restored.version == original.version
        assert restored.sandbox_level == original.sandbox_level
        assert restored.allowed_paths == original.allowed_paths
        assert restored.can_write_files == original.can_write_files

    @pytest.mark.unit
    def test_capability_hash_stable(self):
        """Same capability should produce same hash."""
        cap1 = ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
            sandbox_level=SandboxLevel.NONE,
        )
        cap2 = ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
            sandbox_level=SandboxLevel.NONE,
        )

        assert cap1.get_capability_hash() == cap2.get_capability_hash()

    @pytest.mark.unit
    def test_capability_hash_changes_on_modification(self):
        """Different capabilities should produce different hashes."""
        cap1 = ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
            sandbox_level=SandboxLevel.NONE,
        )
        cap2 = ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
            sandbox_level=SandboxLevel.ISOLATED,  # Different sandbox
        )

        assert cap1.get_capability_hash() != cap2.get_capability_hash()

    @pytest.mark.unit
    def test_default_values(self):
        """Default values should be set correctly."""
        cap = ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
        )

        assert cap.schema_version == "1.0.0"
        assert cap.min_supported_version == "1.0.0"
        assert cap.sandbox_level == SandboxLevel.RESTRICTED
        assert cap.can_write_files is False
        assert cap.can_network is False


# =============================================================================
# TOOL REGISTRATION TESTS
# =============================================================================

class TestToolRegistration:
    """Test tool registration functionality."""

    @pytest.fixture
    def registry(self):
        """Create empty registry for each test."""
        return ToolRegistry()

    @pytest.fixture
    def sample_capability(self):
        """Sample tool capability."""
        return ToolCapability(
            tool_name="sample_tool",
            version="1.0.0",
            description="Sample tool",
        )

    @pytest.mark.unit
    def test_register_tool(self, registry, sample_capability):
        """Should register a tool successfully."""
        registry.register(sample_capability)

        assert "sample_tool" in registry.list_tools()
        assert registry.get("sample_tool") is sample_capability

    @pytest.mark.unit
    def test_unregister_tool(self, registry, sample_capability):
        """Should unregister a tool successfully."""
        registry.register(sample_capability)
        result = registry.unregister("sample_tool")

        assert result is True
        assert "sample_tool" not in registry.list_tools()

    @pytest.mark.unit
    def test_unregister_nonexistent_tool(self, registry):
        """Unregistering non-existent tool should return False."""
        result = registry.unregister("nonexistent")
        assert result is False

    @pytest.mark.unit
    def test_get_registered_tool(self, registry, sample_capability):
        """Should retrieve registered tool."""
        registry.register(sample_capability)
        retrieved = registry.get("sample_tool")

        assert retrieved is sample_capability

    @pytest.mark.unit
    def test_get_missing_returns_none(self, registry):
        """Getting non-existent tool should return None."""
        assert registry.get("nonexistent") is None

    @pytest.mark.unit
    def test_list_tools(self, registry):
        """Should list all registered tools."""
        registry.register(ToolCapability(
            tool_name="tool1", version="1.0", description="Tool 1"
        ))
        registry.register(ToolCapability(
            tool_name="tool2", version="1.0", description="Tool 2"
        ))

        tools = registry.list_tools()

        assert "tool1" in tools
        assert "tool2" in tools
        assert len(tools) == 2


# =============================================================================
# SNAPSHOT TESTS
# =============================================================================

class TestSnapshots:
    """Test registry snapshot functionality."""

    @pytest.fixture
    def populated_registry(self):
        """Registry with tools registered."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            tool_name="tool1", version="1.0", description="Tool 1"
        ))
        registry.register(ToolCapability(
            tool_name="tool2", version="2.0", description="Tool 2"
        ))
        return registry

    @pytest.mark.unit
    def test_create_snapshot(self, populated_registry):
        """Snapshot should include all registered tools."""
        snapshot = populated_registry.create_snapshot()

        assert "snapshot_hash" in snapshot
        assert "snapshot_time" in snapshot
        assert snapshot["tool_count"] == 2
        assert "tool1" in snapshot["tools"]
        assert "tool2" in snapshot["tools"]

    @pytest.mark.unit
    def test_save_snapshot_atomic(self, populated_registry, tmp_path):
        """Snapshot should be saved atomically."""
        snapshot_path = tmp_path / "snapshot.json"

        hash_value = populated_registry.save_snapshot(snapshot_path)

        assert snapshot_path.exists()
        assert len(hash_value) == 64  # SHA-256 hash

        # Verify it's valid JSON
        with open(snapshot_path) as f:
            loaded = json.load(f)
        assert loaded["snapshot_hash"] == hash_value

    @pytest.mark.unit
    def test_load_snapshot(self, populated_registry, tmp_path):
        """Should load snapshot and restore registry state."""
        snapshot_path = tmp_path / "snapshot.json"
        populated_registry.save_snapshot(snapshot_path)

        # Create new empty registry
        new_registry = ToolRegistry()
        result = new_registry.load_snapshot(snapshot_path)

        assert result is True
        assert len(new_registry.list_tools()) == 2
        assert "tool1" in new_registry.list_tools()

    @pytest.mark.unit
    def test_load_snapshot_missing_file(self):
        """Loading non-existent snapshot should return False."""
        registry = ToolRegistry()
        result = registry.load_snapshot(Path("/nonexistent/snapshot.json"))
        assert result is False

    @pytest.mark.unit
    def test_verify_snapshot_matches(self, populated_registry, tmp_path):
        """Verify should return True when registry matches snapshot."""
        snapshot_path = tmp_path / "snapshot.json"
        populated_registry.save_snapshot(snapshot_path)

        # Same registry should match
        result = populated_registry.verify_snapshot(snapshot_path)
        assert result is True

    @pytest.mark.unit
    def test_verify_snapshot_diverged(self, populated_registry, tmp_path):
        """Verify should return False when registry has changed."""
        snapshot_path = tmp_path / "snapshot.json"
        populated_registry.save_snapshot(snapshot_path)

        # Modify registry
        populated_registry.register(ToolCapability(
            tool_name="tool3", version="1.0", description="New tool"
        ))

        result = populated_registry.verify_snapshot(snapshot_path)
        assert result is False


# =============================================================================
# SANDBOX COMPLIANCE TESTS
# =============================================================================

class TestSandboxCompliance:
    """Test sandbox enforcement."""

    @pytest.fixture
    def registry_with_tools(self):
        """Registry with tools of various sandbox levels."""
        registry = ToolRegistry()

        # Isolated tool (no file access)
        registry.register(ToolCapability(
            tool_name="isolated_tool",
            version="1.0",
            description="Isolated",
            sandbox_level=SandboxLevel.ISOLATED,
        ))

        # Read-only tool
        registry.register(ToolCapability(
            tool_name="readonly_tool",
            version="1.0",
            description="Read only",
            sandbox_level=SandboxLevel.READ_ONLY,
        ))

        # Restricted tool with allowlist
        registry.register(ToolCapability(
            tool_name="restricted_tool",
            version="1.0",
            description="Restricted",
            sandbox_level=SandboxLevel.RESTRICTED,
            allowed_paths=["/home/user/safe"],
            blocked_paths=["/home/user/safe/blocked"],
            can_write_files=True,
        ))

        # No restrictions
        registry.register(ToolCapability(
            tool_name="unrestricted_tool",
            version="1.0",
            description="Unrestricted",
            sandbox_level=SandboxLevel.NONE,
            can_write_files=True,
        ))

        return registry

    @pytest.mark.unit
    @pytest.mark.security
    def test_isolated_blocks_all_paths(self, registry_with_tools):
        """Isolated sandbox should block all file access."""
        allowed, reason = registry_with_tools.check_sandbox_compliance(
            "isolated_tool",
            Path("/any/path"),
            "read"
        )

        assert allowed is False
        assert "isolated" in reason.lower()

    @pytest.mark.unit
    @pytest.mark.security
    def test_read_only_blocks_writes(self, registry_with_tools):
        """Read-only sandbox should block write operations."""
        allowed, reason = registry_with_tools.check_sandbox_compliance(
            "readonly_tool",
            Path("/any/path"),
            "write"
        )

        assert allowed is False
        assert "read" in reason.lower()

    @pytest.mark.unit
    @pytest.mark.security
    def test_read_only_allows_reads(self, registry_with_tools):
        """Read-only sandbox should allow read operations."""
        allowed, reason = registry_with_tools.check_sandbox_compliance(
            "readonly_tool",
            Path("/any/path"),
            "read"
        )

        assert allowed is True
        assert reason is None

    @pytest.mark.unit
    @pytest.mark.security
    def test_restricted_checks_allowlist(self, registry_with_tools):
        """Restricted sandbox should check allowed paths."""
        # Allowed path
        allowed, _ = registry_with_tools.check_sandbox_compliance(
            "restricted_tool",
            Path("/home/user/safe/file.txt"),
            "read"
        )
        assert allowed is True

        # Not in allowed list
        allowed, reason = registry_with_tools.check_sandbox_compliance(
            "restricted_tool",
            Path("/other/path/file.txt"),
            "read"
        )
        assert allowed is False
        assert "not in allowed" in reason.lower()

    @pytest.mark.unit
    @pytest.mark.security
    def test_restricted_checks_blocklist(self, registry_with_tools):
        """Restricted sandbox should check blocked paths."""
        allowed, reason = registry_with_tools.check_sandbox_compliance(
            "restricted_tool",
            Path("/home/user/safe/blocked/secret.txt"),
            "read"
        )

        assert allowed is False
        assert "blocked" in reason.lower()

    @pytest.mark.unit
    @pytest.mark.security
    def test_none_allows_all(self, registry_with_tools):
        """None sandbox should allow all operations."""
        allowed, _ = registry_with_tools.check_sandbox_compliance(
            "unrestricted_tool",
            Path("/any/path/anywhere"),
            "write"
        )

        assert allowed is True

    @pytest.mark.unit
    @pytest.mark.security
    def test_unknown_tool_denied(self, registry_with_tools):
        """Unknown tool should be denied."""
        allowed, reason = registry_with_tools.check_sandbox_compliance(
            "unknown_tool",
            Path("/any/path"),
            "read"
        )

        assert allowed is False
        assert "Unknown" in reason


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test input/output validation and required tools."""

    @pytest.fixture
    def registry_with_schemas(self):
        """Registry with tools that have schemas."""
        registry = ToolRegistry(required_tools=["required_tool"])

        registry.register(ToolCapability(
            tool_name="schema_tool",
            version="1.0",
            description="Tool with schema",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name"],
            },
        ))

        return registry

    @pytest.mark.unit
    def test_validate_tool_input_valid(self, registry_with_schemas):
        """Valid input should pass validation."""
        valid, issues = registry_with_schemas.validate_tool_input(
            "schema_tool",
            {"name": "test", "count": 5}
        )

        assert valid is True
        assert len(issues) == 0

    @pytest.mark.unit
    def test_validate_tool_input_unknown_tool(self, registry_with_schemas):
        """Unknown tool should fail validation."""
        valid, issues = registry_with_schemas.validate_tool_input(
            "unknown_tool",
            {"name": "test"}
        )

        assert valid is False
        assert any("Unknown" in issue for issue in issues)

    @pytest.mark.unit
    def test_validate_required_tools_all_present(self):
        """Should pass when all required tools are registered."""
        registry = ToolRegistry(required_tools=["tool1", "tool2"])
        registry.register(ToolCapability(
            tool_name="tool1", version="1.0", description="Tool 1"
        ))
        registry.register(ToolCapability(
            tool_name="tool2", version="1.0", description="Tool 2"
        ))

        result = registry.validate_required_tools()

        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.unit
    def test_validate_required_tools_missing(self):
        """Should fail when required tools are missing."""
        registry = ToolRegistry(required_tools=["tool1", "tool2", "tool3"])
        registry.register(ToolCapability(
            tool_name="tool1", version="1.0", description="Tool 1"
        ))

        result = registry.validate_required_tools()

        assert result.valid is False
        assert len(result.errors) == 2  # tool2 and tool3 missing

    @pytest.mark.unit
    def test_schema_compatibility_valid(self):
        """Compatible versions should pass."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
            schema_version="2.0.0",
            min_supported_version="1.0.0",
        ))

        result = registry.check_schema_compatibility("test", "1.5.0")

        assert result.valid is True

    @pytest.mark.unit
    def test_schema_compatibility_invalid(self):
        """Incompatible versions should fail."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            tool_name="test",
            version="1.0.0",
            description="Test",
            schema_version="1.0.0",
            min_supported_version="1.0.0",
        ))

        result = registry.check_schema_compatibility("test", "2.0.0")

        assert result.valid is False


# =============================================================================
# RATE LIMIT TESTS
# =============================================================================

class TestRateLimiting:
    """Test rate limit functionality."""

    @pytest.mark.unit
    def test_rate_limit_not_exceeded(self):
        """Should allow invocations within limit."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            tool_name="limited_tool",
            version="1.0",
            description="Limited",
            rate_limit_per_minute=10,
        ))

        allowed, _ = registry.check_rate_limit("limited_tool")
        assert allowed is True

    @pytest.mark.unit
    def test_rate_limit_exceeded(self):
        """Should deny invocations beyond limit."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            tool_name="limited_tool",
            version="1.0",
            description="Limited",
            rate_limit_per_minute=2,
        ))

        # Record invocations
        registry.record_invocation("limited_tool")
        registry.record_invocation("limited_tool")

        allowed, reason = registry.check_rate_limit("limited_tool")

        assert allowed is False
        assert "Rate limit" in reason

    @pytest.mark.unit
    def test_reset_rate_counts(self):
        """Reset should clear invocation counts."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            tool_name="limited_tool",
            version="1.0",
            description="Limited",
            rate_limit_per_minute=1,
        ))

        registry.record_invocation("limited_tool")
        registry.reset_rate_counts()

        allowed, _ = registry.check_rate_limit("limited_tool")
        assert allowed is True


# =============================================================================
# GLOBAL REGISTRY TESTS
# =============================================================================

class TestGlobalRegistry:
    """Test global registry functions."""

    @pytest.mark.unit
    def test_get_registry_returns_singleton(self):
        """get_registry should return same instance."""
        reg1 = get_registry()
        reg2 = get_registry()

        assert reg1 is reg2

    @pytest.mark.unit
    def test_register_tool_convenience(self):
        """register_tool should add to global registry."""
        cap = register_tool(
            tool_name="convenience_tool",
            version="1.0.0",
            description="Convenience test",
        )

        assert cap.tool_name == "convenience_tool"
        assert get_registry().get("convenience_tool") is not None
