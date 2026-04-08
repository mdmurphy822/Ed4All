"""
Ed4All MCP Server

Unified MCP server for DART, Courseforge, Trainforge, and Orchestration operations.
Provides tools for:
- PDF to accessible HTML conversion (DART)
- Course generation and IMSCC packaging (Courseforge)
- Assessment-based RAG training (Trainforge)
- Workflow orchestration and state management

Phase 0 Hardening:
- Tool registry snapshot at startup
- Production mode sandbox enforcement
- Structured error responses
- Audit logging for security events

Run: python server.py
"""

from mcp.server.fastmcp import FastMCP
import os
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Configure logging to stderr (required for stdio servers)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import secure path utilities
from lib.secure_paths import validate_path_within_root, PathTraversalError, is_safe_path

# Phase 0 Hardening: Import tool registry and error taxonomy
try:
    from lib.tool_registry import ToolRegistry, ToolCapability, SandboxLevel, get_registry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    logger.warning("Tool registry not available - running without capability tracking")

try:
    from lib.error_taxonomy import (
        StructuredError, ErrorCategory, ErrorCode,
        input_error, security_error, processing_error
    )
    ERROR_TAXONOMY_AVAILABLE = True
except ImportError:
    ERROR_TAXONOMY_AVAILABLE = False

try:
    from lib.secrets_filter import SecretsFilter, get_secrets_filter
    SECRETS_FILTER_AVAILABLE = True
except ImportError:
    SECRETS_FILTER_AVAILABLE = False

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Allowed root for file operations (defaults to project root, can override via env)
ALLOWED_FILE_ROOT = Path(os.environ.get("ED4ALL_ROOT", _PROJECT_ROOT))

# Audit log location
AUDIT_LOG_DIR = ALLOWED_FILE_ROOT / "runtime" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "file_access.jsonl"
SECURITY_LOG_FILE = AUDIT_LOG_DIR / "security_events.jsonl"

# Dev mode disables write restrictions (set DEV_MODE=1 to enable)
DEV_MODE = os.environ.get("DEV_MODE", "0") == "1"

# Phase 0 Hardening: Production mode enforces strict sandbox
PRODUCTION_MODE = os.environ.get("ED4ALL_PRODUCTION", "0") == "1"

# Phase 0 Hardening: Tool registry snapshot location
REGISTRY_SNAPSHOT_DIR = ALLOWED_FILE_ROOT / "runtime" / "registry"


# =============================================================================
# AUDIT LOGGING
# =============================================================================

def log_file_access(
    tool: str,
    path: str,
    allowed: bool,
    error: Optional[str] = None
) -> None:
    """Log file access to audit trail."""
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tool": tool,
            "path": path,
            "allowed": allowed,
            "error": error
        }
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")


def log_security_event(
    event_type: str,
    tool: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Phase 0 Hardening: Log security events.

    Args:
        event_type: Type of security event (sandbox_violation, capability_error, etc.)
        tool: Tool that triggered the event
        message: Human-readable message
        details: Additional details
    """
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "tool": tool,
            "message": message,
            "details": details or {},
            "production_mode": PRODUCTION_MODE
        }
        with open(SECURITY_LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write security log: {e}")


def make_error_response(
    category: str,
    code: str,
    message: str,
    details: Optional[Dict] = None
) -> str:
    """
    Phase 0 Hardening: Create structured error response.

    Args:
        category: Error category
        code: Error code
        message: Error message
        details: Additional details

    Returns:
        JSON string with structured error
    """
    if ERROR_TAXONOMY_AVAILABLE:
        error = StructuredError(
            category=ErrorCategory(category) if hasattr(ErrorCategory, category.upper()) else ErrorCategory.PROCESSING_ERROR,
            code=ErrorCode(code) if code in [e.value for e in ErrorCode] else ErrorCode.INTERNAL_ERROR,
            message=message,
            details=details
        )
        return error.to_json()
    else:
        return json.dumps({
            "error": True,
            "category": category,
            "code": code,
            "message": message,
            "details": details
        })

# Create server instance
mcp = FastMCP("ed4all-orchestrator")

# =============================================================================
# PHASE 0 HARDENING: TOOL REGISTRY
# =============================================================================

# Initialize tool registry
_tool_registry: Optional[Any] = None

def init_tool_registry() -> None:
    """Initialize the tool registry and register core tools."""
    global _tool_registry

    if not TOOL_REGISTRY_AVAILABLE:
        logger.warning("Tool registry not available - skipping initialization")
        return

    _tool_registry = get_registry()

    # Register core file operation tools
    core_tools = [
        ToolCapability(
            tool_name="list_directory",
            version="1.0.0",
            description="List contents of a directory",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            output_schema={"type": "string"},
            sandbox_level=SandboxLevel.READ_ONLY,
            allowed_paths=[str(ALLOWED_FILE_ROOT)],
            can_write_files=False,
            can_execute_subprocess=False,
            can_network=False
        ),
        ToolCapability(
            tool_name="read_file",
            version="1.0.0",
            description="Read contents of a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            output_schema={"type": "string"},
            sandbox_level=SandboxLevel.READ_ONLY,
            allowed_paths=[str(ALLOWED_FILE_ROOT)],
            can_write_files=False,
            can_execute_subprocess=False,
            can_network=False
        ),
        ToolCapability(
            tool_name="write_file",
            version="1.0.0",
            description="Write content to a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
            output_schema={"type": "string"},
            sandbox_level=SandboxLevel.RESTRICTED,
            allowed_paths=[str(ALLOWED_FILE_ROOT / "runtime"), str(ALLOWED_FILE_ROOT / "state")],
            can_write_files=True,
            can_execute_subprocess=False,
            can_network=False
        ),
        ToolCapability(
            tool_name="file_info",
            version="1.0.0",
            description="Get file or directory information",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            output_schema={"type": "string"},
            sandbox_level=SandboxLevel.READ_ONLY,
            allowed_paths=[str(ALLOWED_FILE_ROOT)],
            can_write_files=False,
            can_execute_subprocess=False,
            can_network=False
        ),
    ]

    for cap in core_tools:
        _tool_registry.register(cap)

    logger.info(f"Tool registry initialized with {len(_tool_registry.list_tools())} core tools")


def save_registry_snapshot() -> Optional[str]:
    """
    Save tool registry snapshot at server startup.

    Returns:
        Snapshot hash if successful, None otherwise
    """
    if not TOOL_REGISTRY_AVAILABLE or _tool_registry is None:
        return None

    try:
        REGISTRY_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = REGISTRY_SNAPSHOT_DIR / f"tools_{timestamp}.json"
        snapshot_hash = _tool_registry.save_snapshot(snapshot_path)
        logger.info(f"Tool registry snapshot saved: {snapshot_path} (hash: {snapshot_hash[:12]}...)")
        return snapshot_hash
    except Exception as e:
        logger.error(f"Failed to save registry snapshot: {e}")
        return None


def validate_tool_capability(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """
    Phase 0 Hardening: Validate tool call against registry.

    In production mode, returns error for unregistered tools.
    In dev mode, logs warning but allows execution.

    Args:
        tool_name: Name of tool being called
        args: Arguments being passed

    Returns:
        Error message if validation fails, None if OK
    """
    if not TOOL_REGISTRY_AVAILABLE or _tool_registry is None:
        return None

    cap = _tool_registry.get(tool_name)

    if cap is None:
        msg = f"Tool not registered: {tool_name}"
        log_security_event(
            event_type="unregistered_tool",
            tool=tool_name,
            message=msg,
            details={"args_keys": list(args.keys())}
        )

        if PRODUCTION_MODE:
            return make_error_response(
                category="security_error",
                code="E5003",
                message=msg
            )
        else:
            logger.warning(f"[DEV MODE] {msg}")
            return None

    # Validate inputs against schema
    is_valid, issues = _tool_registry.validate_tool_input(tool_name, args)
    if not is_valid:
        msg = f"Input validation failed for {tool_name}: {issues}"
        log_security_event(
            event_type="input_validation_failed",
            tool=tool_name,
            message=msg,
            details={"issues": issues}
        )

        if PRODUCTION_MODE:
            return make_error_response(
                category="input_error",
                code="E1004",
                message=msg,
                details={"issues": issues}
            )

    return None


def check_sandbox_compliance(tool_name: str, path: Path, operation: str = "read") -> Optional[str]:
    """
    Phase 0 Hardening: Check if path access is allowed by tool's sandbox.

    Args:
        tool_name: Name of tool making the access
        path: Path being accessed
        operation: Type of operation (read/write)

    Returns:
        Error message if sandbox violated, None if OK
    """
    if not TOOL_REGISTRY_AVAILABLE or _tool_registry is None:
        return None

    is_allowed, reason = _tool_registry.check_sandbox_compliance(tool_name, path, operation)

    if not is_allowed:
        log_security_event(
            event_type="sandbox_violation",
            tool=tool_name,
            message=f"Sandbox violation: {reason}",
            details={"path": str(path), "operation": operation}
        )

        if PRODUCTION_MODE:
            return make_error_response(
                category="security_error",
                code="E5003",
                message=f"Sandbox violation: {reason}",
                details={"path": str(path)}
            )
        else:
            logger.warning(f"[DEV MODE] Sandbox violation by {tool_name}: {reason}")

    return None

# =============================================================================
# CORE FILE OPERATIONS (secured with path validation)
# =============================================================================

@mcp.tool()
async def list_directory(path: str) -> str:
    """
    List contents of a directory.

    Security: Path must be within project root (ED4ALL_ROOT).
    """
    try:
        # Validate path is within allowed root
        safe_path = validate_path_within_root(
            Path(path), ALLOWED_FILE_ROOT, must_exist=True
        )
        log_file_access("list_directory", path, allowed=True)

        items = os.listdir(safe_path)
        return "\n".join(items) if items else "(empty directory)"
    except PathTraversalError as e:
        log_file_access("list_directory", path, allowed=False, error=str(e))
        return f"Security error: Path not allowed - {e}"
    except Exception as e:
        log_file_access("list_directory", path, allowed=False, error=str(e))
        return f"Error: {e}"

@mcp.tool()
async def read_file(path: str) -> str:
    """
    Read contents of a file.

    Security: Path must be within project root (ED4ALL_ROOT).
    """
    try:
        # Validate path is within allowed root
        safe_path = validate_path_within_root(
            Path(path), ALLOWED_FILE_ROOT, must_exist=True
        )
        log_file_access("read_file", path, allowed=True)

        with open(safe_path, 'r') as f:
            return f.read()
    except PathTraversalError as e:
        log_file_access("read_file", path, allowed=False, error=str(e))
        return f"Security error: Path not allowed - {e}"
    except Exception as e:
        log_file_access("read_file", path, allowed=False, error=str(e))
        return f"Error: {e}"

@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """
    Write content to a file.

    Security: Path must be within project root (ED4ALL_ROOT).
    Note: Write operations require DEV_MODE=1 unless writing to runtime/.
    """
    try:
        # Validate path is within allowed root
        safe_path = validate_path_within_root(Path(path), ALLOWED_FILE_ROOT)

        # Additional restriction: writes only to runtime/ unless DEV_MODE
        runtime_dir = ALLOWED_FILE_ROOT / "runtime"
        if not DEV_MODE and not is_safe_path(safe_path, runtime_dir):
            log_file_access("write_file", path, allowed=False, error="Write outside runtime/ requires DEV_MODE")
            return "Security error: Write operations outside runtime/ require DEV_MODE=1"

        log_file_access("write_file", path, allowed=True)

        # Ensure parent directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        with open(safe_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except PathTraversalError as e:
        log_file_access("write_file", path, allowed=False, error=str(e))
        return f"Security error: Path not allowed - {e}"
    except Exception as e:
        log_file_access("write_file", path, allowed=False, error=str(e))
        return f"Error: {e}"

@mcp.tool()
async def file_info(path: str) -> str:
    """
    Get file or directory information.

    Security: Path must be within project root (ED4ALL_ROOT).
    """
    try:
        # Validate path is within allowed root
        safe_path = validate_path_within_root(
            Path(path), ALLOWED_FILE_ROOT, must_exist=True
        )
        log_file_access("file_info", path, allowed=True)

        stat = os.stat(safe_path)
        return f"Size: {stat.st_size} bytes\nModified: {stat.st_mtime}"
    except PathTraversalError as e:
        log_file_access("file_info", path, allowed=False, error=str(e))
        return f"Security error: Path not allowed - {e}"
    except Exception as e:
        log_file_access("file_info", path, allowed=False, error=str(e))
        return f"Error: {e}"

# =============================================================================
# RESOURCES (secured with path validation)
# =============================================================================

@mcp.resource("file://{path}")
def get_file_content(path: str) -> str:
    """
    Expose file contents as a resource.

    Security: Path must be within project root (ED4ALL_ROOT).
    """
    try:
        # Validate path is within allowed root
        safe_path = validate_path_within_root(
            Path(path), ALLOWED_FILE_ROOT, must_exist=True
        )
        log_file_access("resource:file", path, allowed=True)

        with open(safe_path, 'r') as f:
            return f.read()
    except PathTraversalError as e:
        log_file_access("resource:file", path, allowed=False, error=str(e))
        return f"Security error: Path not allowed - {e}"
    except Exception as e:
        log_file_access("resource:file", path, allowed=False, error=str(e))
        return f"Error reading file: {e}"

# =============================================================================
# REGISTER TOOL MODULES (Ed4All: DART + Courseforge + Trainforge only)
# =============================================================================

try:
    from tools.dart_tools import register_dart_tools
    register_dart_tools(mcp)
    logger.info("DART tools registered")
except ImportError as e:
    logger.warning(f"Could not register DART tools: {e}")

try:
    from tools.courseforge_tools import register_courseforge_tools
    register_courseforge_tools(mcp)
    logger.info("Courseforge tools registered")
except ImportError as e:
    logger.warning(f"Could not register Courseforge tools: {e}")

try:
    from tools.orchestrator_tools import register_orchestrator_tools
    register_orchestrator_tools(mcp)
    logger.info("Orchestrator tools registered")
except ImportError as e:
    logger.warning(f"Could not register Orchestrator tools: {e}")

try:
    from tools.trainforge_tools import register_trainforge_tools
    register_trainforge_tools(mcp)
    logger.info("Trainforge tools registered")
except ImportError as e:
    logger.warning(f"Could not register Trainforge tools: {e}")

try:
    from tools.analysis_tools import register_analysis_tools
    register_analysis_tools(mcp)
    logger.info("Analysis tools registered")
except ImportError as e:
    logger.warning(f"Could not register Analysis tools: {e}")

try:
    from tools.learning_science_tools import register_learning_science_tools
    register_learning_science_tools(mcp)
    logger.info("Learning Science tools registered")
except ImportError as e:
    logger.warning(f"Could not register Learning Science tools: {e}")

try:
    from tools.pipeline_tools import register_pipeline_tools
    register_pipeline_tools(mcp)
    logger.info("Pipeline tools registered")
except ImportError as e:
    logger.warning(f"Could not register Pipeline tools: {e}")

# =============================================================================
# SERVER STARTUP
# =============================================================================

def startup_hardening() -> None:
    """
    Phase 0 Hardening: Server startup initialization.

    - Initialize tool registry
    - Save registry snapshot
    - Log startup security configuration
    """
    # Initialize tool registry
    init_tool_registry()

    # Save snapshot after all tools are registered
    snapshot_hash = save_registry_snapshot()

    # Log security configuration
    logger.info(f"Security configuration:")
    logger.info(f"  - Production mode: {PRODUCTION_MODE}")
    logger.info(f"  - Dev mode: {DEV_MODE}")
    logger.info(f"  - Allowed root: {ALLOWED_FILE_ROOT}")
    if snapshot_hash:
        logger.info(f"  - Registry snapshot: {snapshot_hash[:12]}...")

    if PRODUCTION_MODE:
        logger.info("Running in PRODUCTION mode - strict sandbox enforcement enabled")
    elif DEV_MODE:
        logger.info("Running in DEV mode - write restrictions relaxed")


if __name__ == "__main__":
    logger.info("Starting Ed4All MCP Server")
    logger.info("Available tool categories: file_ops, dart, courseforge, orchestrator, trainforge, analysis, learning_science")

    # Phase 0 Hardening: Initialize security components
    startup_hardening()

    mcp.run()
