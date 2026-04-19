"""
Centralized path configuration for Ed4All project.

This module provides the single source of truth for all project paths.
All other modules should import paths from here rather than defining their own.

The project root can be configured via the ED4ALL_ROOT environment variable.
If not set, it defaults to the parent directory of this module.

Usage:
    from lib.paths import PROJECT_ROOT, LIBV2_PATH, DART_PATH

Environment:
    ED4ALL_ROOT: Override the project root path (e.g., /opt/ed4all)
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================

# Single source of truth for project root
# Priority: ED4ALL_ROOT env var > parent of this file's directory
PROJECT_ROOT = Path(os.environ.get(
    "ED4ALL_ROOT",
    Path(__file__).resolve().parents[1]
))

# ============================================================================
# TOP-LEVEL COMPONENT PATHS
# ============================================================================

# Core pipeline components (Ed4All: DART + Courseforge + Trainforge + LibV2)
DART_PATH = PROJECT_ROOT / "DART"
COURSEFORGE_PATH = PROJECT_ROOT / "Courseforge"
TRAINFORGE_PATH = PROJECT_ROOT / "Trainforge"
LIBV2_PATH = PROJECT_ROOT / "LibV2"

# Infrastructure components
MCP_PATH = PROJECT_ROOT / "MCP"
ORCHESTRATOR_PATH = PROJECT_ROOT / "orchestrator"

# Shared resources
LIB_PATH = PROJECT_ROOT / "lib"
CONFIG_PATH = PROJECT_ROOT / "config"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
SCHEMAS_PATH = PROJECT_ROOT / "schemas"
STATE_PATH = PROJECT_ROOT / "state"

# ============================================================================
# LIBV2 SUBDIRECTORIES
# ============================================================================

LIBV2_CATALOG = LIBV2_PATH / "catalog"
LIBV2_COURSES = LIBV2_PATH / "courses"
# Library schemas and ontology are unified under project-root /schemas/
# (Formerly LibV2/ontology/ and LibV2/schema/; migrated in Worker S PR.)
LIBV2_ONTOLOGY = SCHEMAS_PATH / "taxonomies"
LIBV2_SCHEMA = SCHEMAS_PATH / "library"
LIBV2_TOOLS = LIBV2_PATH / "tools"

# ============================================================================
# STATE MANAGEMENT PATHS
# ============================================================================

# Workflow state files
STATE_WORKFLOWS = STATE_PATH / "workflows"
STATE_PROGRESS = STATE_PATH / "progress"
STATE_LOCKS = STATE_PATH / "locks"

# ============================================================================
# TRAINING CAPTURES
# ============================================================================

TRAINING_DIR = PROJECT_ROOT / "training-captures"
TRAINING_DIR_LEGACY = TRAINING_DIR

# ============================================================================
# RUNS PATH
# ============================================================================

RUNS_PATH = STATE_PATH / "runs"


# ============================================================================
# PATH RESOLUTION HELPERS (Phase 0.5)
# ============================================================================

def get_project_root() -> Path:
    """
    Resolve project root using 3-tier resolution.

    Priority:
    1. ED4ALL_ROOT environment variable
    2. Git repository root (if in a git repo)
    3. Parent of a marker file (CLAUDE.md, .ed4all_root, pyproject.toml)

    Returns:
        Resolved project root path

    Raises:
        RuntimeError: If project root cannot be determined
    """
    import subprocess

    # Tier 1: Environment variable
    env_root = os.environ.get("ED4ALL_ROOT")
    if env_root:
        path = Path(env_root).resolve()
        if path.exists():
            return path

    # Tier 2: Git repository root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_root = Path(result.stdout.strip()).resolve()
            if git_root.exists():
                return git_root
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Tier 3: Marker file search
    marker_files = ["CLAUDE.md", ".ed4all_root", "pyproject.toml"]

    # Start from current file's parent directory
    current = Path(__file__).resolve().parent

    # Walk up directory tree looking for markers
    for _ in range(10):  # Limit search depth
        for marker in marker_files:
            if (current / marker).exists():
                return current

        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # Fallback to original PROJECT_ROOT
    return PROJECT_ROOT


def resolve_in_project(rel_path: str) -> Path:
    """
    Safely resolve a relative path within the project.

    Validates that the path doesn't escape project boundaries.

    Args:
        rel_path: Relative path string

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path escapes project boundaries or is invalid
    """
    from .path_constants import (
        DISALLOW_PARENT_TRAVERSAL,
        MAX_PATH_LENGTH,
    )

    # Check path length
    if len(rel_path) > MAX_PATH_LENGTH:
        raise ValueError(f"Path exceeds maximum length: {len(rel_path)} > {MAX_PATH_LENGTH}")

    # Check for parent traversal
    if DISALLOW_PARENT_TRAVERSAL and ".." in rel_path:
        raise ValueError(f"Parent directory traversal not allowed: {rel_path}")

    # Resolve path
    project_root = get_project_root()
    resolved = (project_root / rel_path).resolve()

    # Verify path is within project
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"Path escapes project boundaries: {rel_path}") from None

    return resolved


def resolve_run_path(run_id: str) -> Path:
    """
    Get canonical run directory path.

    Args:
        run_id: Run identifier

    Returns:
        Path to run directory

    Raises:
        ValueError: If run_id is invalid
    """
    # Validate run_id format
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError(f"Invalid run_id: {run_id}")

    return RUNS_PATH / run_id


def resolve_run_artifact_path(
    run_id: str,
    component: str,
    filename: str = "",
) -> Path:
    """
    Get canonical artifact path within a run.

    Args:
        run_id: Run identifier
        component: Component name (dart, courseforge, trainforge)
        filename: Optional filename

    Returns:
        Path to artifact directory or file
    """
    run_path = resolve_run_path(run_id)
    artifact_path = run_path / "artifacts" / component

    if filename:
        # Validate filename
        if "/" in filename or ".." in filename:
            raise ValueError(f"Invalid filename: {filename}")
        artifact_path = artifact_path / filename

    return artifact_path


def ensure_run_directories(run_path: Path) -> None:
    """
    Create standard run directory structure.

    Args:
        run_path: Path to run directory
    """
    directories = [
        run_path,
        run_path / "config_snapshot",
        run_path / "artifacts" / "dart",
        run_path / "artifacts" / "courseforge",
        run_path / "artifacts" / "trainforge",
        run_path / "decisions",
        run_path / "audit",
        run_path / "checkpoints",
        run_path / "state",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def is_path_within(path: Path, parent: Path) -> bool:
    """
    Check if a path is within a parent directory.

    Args:
        path: Path to check
        parent: Parent directory

    Returns:
        True if path is within parent
    """
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ============================================================================
# PATH VALIDATION
# ============================================================================

def validate_paths() -> dict[str, bool]:
    """
    Validate that critical project paths exist.

    Returns:
        Dict mapping path names to existence status
    """
    return {
        "PROJECT_ROOT": PROJECT_ROOT.exists(),
        "DART_PATH": DART_PATH.exists(),
        "COURSEFORGE_PATH": COURSEFORGE_PATH.exists(),
        "TRAINFORGE_PATH": TRAINFORGE_PATH.exists(),
        "LIBV2_PATH": LIBV2_PATH.exists(),
        "MCP_PATH": MCP_PATH.exists(),
        "CONFIG_PATH": CONFIG_PATH.exists(),
        "SCHEMAS_PATH": SCHEMAS_PATH.exists(),
    }


def ensure_state_dirs() -> None:
    """Create state management directories if they don't exist."""
    for path in [STATE_PATH, STATE_WORKFLOWS, STATE_PROGRESS, STATE_LOCKS]:
        path.mkdir(parents=True, exist_ok=True)


def get_agent_prompt_path(component: str, agent_name: str) -> Path:
    """
    Get the path to an agent prompt file.

    Args:
        component: Component name (DART, Courseforge, Trainforge)
        agent_name: Agent name (e.g., content-generator)

    Returns:
        Path to the agent markdown file
    """
    component_map = {
        "dart": DART_PATH,
        "courseforge": COURSEFORGE_PATH,
        "trainforge": TRAINFORGE_PATH,
    }

    component_path = component_map.get(component.lower())
    if component_path is None:
        raise ValueError(f"Unknown component: {component}")

    return component_path / "agents" / f"{agent_name}.md"


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Project root
    "PROJECT_ROOT",

    # Component paths
    "DART_PATH",
    "COURSEFORGE_PATH",
    "TRAINFORGE_PATH",
    "LIBV2_PATH",
    "MCP_PATH",
    "ORCHESTRATOR_PATH",

    # Shared resources
    "LIB_PATH",
    "CONFIG_PATH",
    "SCRIPTS_PATH",
    "SCHEMAS_PATH",
    "STATE_PATH",

    # LibV2 subdirectories
    "LIBV2_CATALOG",
    "LIBV2_COURSES",
    "LIBV2_ONTOLOGY",
    "LIBV2_SCHEMA",
    "LIBV2_TOOLS",

    # State paths
    "STATE_WORKFLOWS",
    "STATE_PROGRESS",
    "STATE_LOCKS",
    "RUNS_PATH",

    # Training captures
    "TRAINING_DIR",
    "TRAINING_DIR_LEGACY",

    # Path validation
    "validate_paths",
    "ensure_state_dirs",
    "get_agent_prompt_path",

    # Path resolution helpers (Phase 0.5)
    "get_project_root",
    "resolve_in_project",
    "resolve_run_path",
    "resolve_run_artifact_path",
    "ensure_run_directories",
    "is_path_within",
]
