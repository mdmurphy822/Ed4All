"""
Tool Registry for MCP Contract Hardening

Captures tool capabilities, input/output schemas, and sandbox requirements.
Provides registry snapshot for immutable tool configuration per run.

Phase 0 Hardening - Requirement 7: MCP Contract Hardening
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class SandboxLevel(Enum):
    """Tool sandbox requirements."""
    NONE = "none"              # No restrictions
    READ_ONLY = "read_only"    # Can only read files
    RESTRICTED = "restricted"  # Limited file access paths
    ISOLATED = "isolated"      # No file system access


@dataclass
class ToolCapability:
    """
    Capability declaration for a tool.

    Phase 0.5 Enhanced with schema versioning.
    """
    tool_name: str
    version: str
    description: str

    # Phase 0.5: Schema versioning
    schema_version: str = "1.0.0"
    min_supported_version: str = "1.0.0"

    # Schema definitions (JSON Schema format)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # Sandbox requirements
    sandbox_level: SandboxLevel = SandboxLevel.RESTRICTED
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)

    # Capability flags
    can_write_files: bool = False
    can_execute_subprocess: bool = False
    can_network: bool = False
    requires_auth: bool = False

    # Resource limits
    rate_limit_per_minute: Optional[int] = None
    max_input_size_bytes: Optional[int] = None
    max_output_size_bytes: Optional[int] = None
    timeout_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['sandbox_level'] = self.sandbox_level.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolCapability":
        """Create from dictionary."""
        # Handle sandbox_level enum
        if 'sandbox_level' in data:
            data['sandbox_level'] = SandboxLevel(data['sandbox_level'])
        return cls(**data)

    def get_capability_hash(self) -> str:
        """Get hash of capability declaration for change detection."""
        d = self.to_dict()
        # Remove mutable fields
        d.pop('version', None)
        content = json.dumps(d, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ValidationResult:
    """Result of validation check."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolRegistry:
    """
    Registry of tool capabilities with snapshot support.

    Phase 0.5 Enhanced with:
    - Required tools validation
    - Schema version compatibility checking
    """

    def __init__(self, required_tools: Optional[List[str]] = None):
        """
        Initialize registry.

        Args:
            required_tools: List of tool names that must be registered
        """
        self._tools: Dict[str, ToolCapability] = {}
        self._snapshot_hash: Optional[str] = None
        self._snapshot_time: Optional[str] = None
        self._invocation_counts: Dict[str, int] = {}

        # Phase 0.5: Required tools
        self.required_tools: List[str] = required_tools or []

    def register(self, capability: ToolCapability) -> None:
        """
        Register a tool capability.

        Args:
            capability: Tool capability to register
        """
        self._tools[capability.tool_name] = capability
        self._snapshot_hash = None  # Invalidate snapshot
        logger.debug(f"Registered tool: {capability.tool_name} v{capability.version}")

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._snapshot_hash = None
            logger.debug(f"Unregistered tool: {tool_name}")
            return True
        return False

    def get(self, tool_name: str) -> Optional[ToolCapability]:
        """
        Get capability for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolCapability or None if not found
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create immutable snapshot of registry state.

        Returns:
            Dictionary with snapshot data and hash
        """
        snapshot_data = {
            tool_name: cap.to_dict()
            for tool_name, cap in sorted(self._tools.items())
        }

        # Compute hash
        snapshot_json = json.dumps(snapshot_data, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_json.encode()).hexdigest()

        self._snapshot_hash = snapshot_hash
        self._snapshot_time = datetime.now().isoformat()

        return {
            "snapshot_hash": snapshot_hash,
            "snapshot_time": self._snapshot_time,
            "tool_count": len(self._tools),
            "tools": snapshot_data
        }

    def save_snapshot(self, path: Path) -> str:
        """
        Save snapshot to file and return hash.

        Args:
            path: Path to save snapshot

        Returns:
            Snapshot hash
        """
        snapshot = self.create_snapshot()

        # Atomic write
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        temp_path.rename(path)

        logger.info(f"Saved registry snapshot: {snapshot['snapshot_hash'][:12]}...")
        return snapshot['snapshot_hash']

    def load_snapshot(self, path: Path) -> bool:
        """
        Load registry state from snapshot.

        Args:
            path: Path to snapshot file

        Returns:
            True if loaded successfully
        """
        if not path.exists():
            return False

        try:
            with open(path) as f:
                snapshot = json.load(f)

            self._tools.clear()
            for tool_name, tool_data in snapshot.get('tools', {}).items():
                self._tools[tool_name] = ToolCapability.from_dict(tool_data)

            self._snapshot_hash = snapshot.get('snapshot_hash')
            self._snapshot_time = snapshot.get('snapshot_time')

            logger.info(f"Loaded registry snapshot with {len(self._tools)} tools")
            return True
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Failed to load snapshot: {e}")
            return False

    def verify_snapshot(self, path: Path) -> bool:
        """
        Verify current registry matches saved snapshot.

        Args:
            path: Path to snapshot file

        Returns:
            True if registry matches snapshot
        """
        if not path.exists():
            return False

        with open(path) as f:
            saved = json.load(f)

        current = self.create_snapshot()
        return current['snapshot_hash'] == saved.get('snapshot_hash')

    def validate_tool_input(
        self,
        tool_name: str,
        inputs: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate inputs against tool's input schema.

        Args:
            tool_name: Name of the tool
            inputs: Input data to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        cap = self._tools.get(tool_name)
        if not cap:
            return False, [f"Unknown tool: {tool_name}"]

        if not cap.input_schema:
            return True, []  # No schema = no validation

        try:
            import jsonschema
            jsonschema.validate(inputs, cap.input_schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [f"Input validation failed: {e.message}"]
        except ImportError:
            # jsonschema not installed, skip validation
            logger.warning("jsonschema not installed, skipping input validation")
            return True, []

    def validate_tool_output(
        self,
        tool_name: str,
        outputs: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate outputs against tool's output schema.

        Args:
            tool_name: Name of the tool
            outputs: Output data to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        cap = self._tools.get(tool_name)
        if not cap:
            return False, [f"Unknown tool: {tool_name}"]

        if not cap.output_schema:
            return True, []  # No schema = no validation

        try:
            import jsonschema
            jsonschema.validate(outputs, cap.output_schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [f"Output validation failed: {e.message}"]
        except ImportError:
            return True, []

    def check_sandbox_compliance(
        self,
        tool_name: str,
        requested_path: Path,
        operation: str = "read"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if path access is allowed by tool's sandbox.

        Args:
            tool_name: Name of the tool
            requested_path: Path being accessed
            operation: "read" or "write"

        Returns:
            Tuple of (allowed, reason if denied)
        """
        cap = self._tools.get(tool_name)
        if not cap:
            return False, f"Unknown tool: {tool_name}"

        # Check sandbox level
        if cap.sandbox_level == SandboxLevel.ISOLATED:
            return False, "Tool has no file system access (isolated)"

        if cap.sandbox_level == SandboxLevel.READ_ONLY and operation != "read":
            return False, "Tool only has read access"

        # Resolve to absolute path for comparison
        try:
            path_str = str(requested_path.resolve())
        except (OSError, RuntimeError):
            path_str = str(requested_path)

        # Check blocked paths first
        for blocked in cap.blocked_paths:
            if path_str.startswith(blocked):
                return False, f"Path blocked by policy: {blocked}"

        # Check allowed paths (if specified)
        if cap.allowed_paths:
            allowed = False
            for allowed_path in cap.allowed_paths:
                if path_str.startswith(allowed_path):
                    allowed = True
                    break
            if not allowed:
                return False, f"Path not in allowed list"

        # Check write permission
        if operation == "write" and not cap.can_write_files:
            return False, "Tool does not have write permission"

        return True, None

    def check_rate_limit(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if tool invocation is within rate limit.

        Args:
            tool_name: Name of the tool

        Returns:
            Tuple of (allowed, reason if denied)
        """
        cap = self._tools.get(tool_name)
        if not cap or not cap.rate_limit_per_minute:
            return True, None

        # Simple in-memory rate tracking (would need persistence for production)
        count = self._invocation_counts.get(tool_name, 0)
        if count >= cap.rate_limit_per_minute:
            return False, f"Rate limit exceeded: {cap.rate_limit_per_minute}/min"

        return True, None

    def record_invocation(self, tool_name: str) -> None:
        """Record a tool invocation for rate limiting."""
        self._invocation_counts[tool_name] = self._invocation_counts.get(tool_name, 0) + 1

    def reset_rate_counts(self) -> None:
        """Reset rate limit counters (call periodically)."""
        self._invocation_counts.clear()

    # ========================================================================
    # Phase 0.5: Required Tools Validation
    # ========================================================================

    def validate_required_tools(self) -> ValidationResult:
        """
        Ensure all required tools are registered.

        Returns:
            ValidationResult with any missing tools as errors
        """
        errors = []
        warnings = []

        for tool_name in self.required_tools:
            if tool_name not in self._tools:
                errors.append(f"Required tool not registered: {tool_name}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def check_schema_compatibility(
        self,
        tool_name: str,
        required_version: str,
    ) -> ValidationResult:
        """
        Check if tool schema version is compatible.

        Args:
            tool_name: Tool to check
            required_version: Minimum required version

        Returns:
            ValidationResult
        """
        cap = self._tools.get(tool_name)
        if not cap:
            return ValidationResult(
                valid=False,
                errors=[f"Tool not found: {tool_name}"]
            )

        # Simple version comparison (semver-like)
        def parse_version(v: str) -> tuple:
            parts = v.split(".")
            return tuple(int(p) for p in parts[:3])

        try:
            current = parse_version(cap.schema_version)
            required = parse_version(required_version)
            minimum = parse_version(cap.min_supported_version)

            if current < required:
                return ValidationResult(
                    valid=False,
                    errors=[
                        f"Tool {tool_name} schema version {cap.schema_version} "
                        f"is below required {required_version}"
                    ]
                )

            if required < minimum:
                return ValidationResult(
                    valid=False,
                    errors=[
                        f"Required version {required_version} is below "
                        f"minimum supported {cap.min_supported_version}"
                    ]
                )

            return ValidationResult(valid=True)

        except (ValueError, AttributeError) as e:
            return ValidationResult(
                valid=False,
                errors=[f"Version comparison failed: {e}"]
            )

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a lightweight snapshot for RunContext.

        Returns:
            Dictionary with registry state
        """
        return {
            "snapshot_hash": self._snapshot_hash or self._compute_hash(),
            "snapshot_time": self._snapshot_time or datetime.now().isoformat(),
            "tool_count": len(self._tools),
            "tools": list(self._tools.keys()),
            "required_tools": self.required_tools,
        }

    def _compute_hash(self) -> str:
        """Compute current registry hash."""
        snapshot_data = {
            tool_name: cap.to_dict()
            for tool_name, cap in sorted(self._tools.items())
        }
        snapshot_json = json.dumps(snapshot_data, sort_keys=True)
        return hashlib.sha256(snapshot_json.encode()).hexdigest()


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    tool_name: str,
    version: str = "1.0.0",
    description: str = "",
    input_schema: Optional[Dict] = None,
    output_schema: Optional[Dict] = None,
    sandbox_level: str = "restricted",
    allowed_paths: Optional[List[str]] = None,
    can_write_files: bool = False,
    can_network: bool = False,
    **kwargs
) -> ToolCapability:
    """
    Convenience function to register a tool.

    Args:
        tool_name: Name of the tool
        version: Version string
        description: Tool description
        input_schema: JSON Schema for inputs
        output_schema: JSON Schema for outputs
        sandbox_level: "none", "read_only", "restricted", "isolated"
        allowed_paths: List of allowed path prefixes
        can_write_files: Whether tool can write files
        can_network: Whether tool can make network requests
        **kwargs: Additional ToolCapability fields

    Returns:
        Created ToolCapability
    """
    capability = ToolCapability(
        tool_name=tool_name,
        version=version,
        description=description,
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        sandbox_level=SandboxLevel(sandbox_level),
        allowed_paths=allowed_paths or [],
        can_write_files=can_write_files,
        can_network=can_network,
        **kwargs
    )

    get_registry().register(capability)
    return capability
