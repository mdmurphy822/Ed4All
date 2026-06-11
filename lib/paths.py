"""
Centralized path configuration for Ed4All project.

This module provides the single source of truth for all project paths.
All other modules should import paths from here rather than defining their own.

The project root can be configured via the ED4ALL_ROOT environment variable.
If not set, it defaults to the parent directory of this module.

Usage:
    from lib.paths import PROJECT_ROOT, LIBV2_PATH, DART_PATH

Environment:
    ED4ALL_ROOT: Override the project root path (e.g., /opt/ed4all). This is
        the *code* root (where DART/, Courseforge/, lib/, schemas/ live).
    ED4ALL_HOME: Relocatable *data* root. When set, every mutable data
        directory (state, libv2, exports, training-captures, uploads,
        dart-output) defaults to ``<ED4ALL_HOME>/<dirname>`` instead of being
        repo-relative. This unblocks non-editable (site-packages) installs and
        keeps Docker volumes tidy — the read-only code ships under ED4ALL_ROOT
        and the writable state lives under a single mounted ED4ALL_HOME.

        Precedence for every data dir is:
            explicit per-dir env (e.g. ED4ALL_LIBV2_ROOT, ED4ALL_STATE_RUNS_DIR,
                ED4ALL_TRAINING_CAPTURES_DIR)
            > ED4ALL_HOME/<dirname>
            > repo-relative default (ED4ALL_ROOT/<dirname>)

        When ED4ALL_HOME is unset every path is byte-identical to the legacy
        repo-relative default (no behavior change for the in-repo case).
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
# ED4ALL_HOME — relocatable data root
# ============================================================================
#
# The DATA-DIR basenames that relocate under ED4ALL_HOME. The basename is what
# the dir is called underneath the data root; for the repo-relative default it
# matches the in-tree layout exactly (state/, LibV2/, Courseforge/exports/,
# training-captures/, DART/output/). ``exports`` and ``dart-output`` are nested
# under their parent component dirs in-repo but flatten to a single level under
# ED4ALL_HOME (an ED4ALL_HOME deployment ships no code, only data).
_DATA_DIR_KEYS = ("state", "libv2", "exports", "training-captures", "dart-output")


def ed4all_home() -> Path | None:
    """Return the configured ``ED4ALL_HOME`` data root, or ``None`` if unset.

    Read at call time (not import time) so a test can monkeypatch the env var
    and exercise the relocated layout without re-importing this module. An
    empty / whitespace value is treated as unset.
    """
    raw = os.environ.get("ED4ALL_HOME", "").strip()
    return Path(raw) if raw else None


def _data_dir(basename: str, repo_relative_default: Path) -> Path:
    """Resolve a data dir honoring ED4ALL_HOME, else the repo-relative default.

    This is the shared chokepoint for the ED4ALL_HOME relocation. Per-dir env
    overrides (ED4ALL_LIBV2_ROOT etc.) are layered ON TOP of this by their own
    resolver functions, so the full precedence is:

        per-dir env (resolver-owned) > ED4ALL_HOME/<basename> > repo-relative

    When ED4ALL_HOME is unset this returns ``repo_relative_default`` verbatim,
    so the in-repo default is byte-stable.
    """
    home = ed4all_home()
    if home is not None:
        return home / basename
    return repo_relative_default


def ensure_data_dir(path: Path) -> Path:
    """Create ``path`` (and parents) on first use when ED4ALL_HOME is active.

    Bootstrap helper for the relocated layout: a fresh ED4ALL_HOME (e.g. a
    just-mounted Docker volume) has no subdirs yet, so the first consumer of a
    data dir creates it. A no-op when the dir already exists. When ED4ALL_HOME
    is unset this still mkdir's the repo-relative default, matching the existing
    ``ensure_state_dirs`` / ``uploads_dir`` lazy-create idiom; callers that want
    create-only-under-home semantics gate on ``ed4all_home() is not None``.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================================
# TOP-LEVEL COMPONENT PATHS
# ============================================================================

# Core pipeline components (Ed4All: DART + Courseforge + Trainforge + LibV2).
# DART / Courseforge / Trainforge are CODE roots (agents/, scripts/) and stay
# under PROJECT_ROOT regardless of ED4ALL_HOME — only their mutable *data*
# subdirs (DART/output/, Courseforge/exports/) relocate (see dart_output_dir /
# courseforge_exports_dir). LibV2 is a pure DATA root, so it relocates wholesale
# under ED4ALL_HOME (the per-dir ED4ALL_LIBV2_ROOT override still wins; see
# ``libv2_path``).
DART_PATH = PROJECT_ROOT / "DART"
COURSEFORGE_PATH = PROJECT_ROOT / "Courseforge"
TRAINFORGE_PATH = PROJECT_ROOT / "Trainforge"
LIBV2_PATH = _data_dir("libv2", PROJECT_ROOT / "LibV2")

# Infrastructure components
MCP_PATH = PROJECT_ROOT / "MCP"
ORCHESTRATOR_PATH = PROJECT_ROOT / "orchestrator"

# Shared resources
LIB_PATH = PROJECT_ROOT / "lib"
CONFIG_PATH = PROJECT_ROOT / "config"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
SCHEMAS_PATH = PROJECT_ROOT / "schemas"
# state/ is a pure DATA root: relocates under ED4ALL_HOME when set. The per-dir
# ED4ALL_STATE_RUNS_DIR override (read at call time in ``get_state_runs_dir``)
# still wins for the runs/ subtree specifically.
STATE_PATH = _data_dir("state", PROJECT_ROOT / "state")

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

# training-captures/ is a pure DATA root: relocates under ED4ALL_HOME. The
# per-dir ED4ALL_TRAINING_CAPTURES_DIR override (read at call time in
# ``get_training_captures_dir``) still wins.
TRAINING_DIR = _data_dir("training-captures", PROJECT_ROOT / "training-captures")
TRAINING_DIR_LEGACY = TRAINING_DIR


def libv2_path() -> Path:
    """Resolve the LibV2 root directory, honoring ``ED4ALL_LIBV2_ROOT``.

    Priority:
    1. ``ED4ALL_LIBV2_ROOT`` env var (documented in root CLAUDE.md;
       honored by the MCP / GUI / CLI layers and — as of this change —
       by ``lib/libv2_storage.py`` and ``lib/decision_capture.py`` so a
       test that threads ``libv2_root=tmp`` no longer leaks skeleton
       dirs into the real ``LibV2/`` tree).
    2. ``ED4ALL_HOME/libv2`` when ``ED4ALL_HOME`` is set (folded into the
       ``LIBV2_PATH`` constant at import time via ``_data_dir``).
    3. ``PROJECT_ROOT / "LibV2"`` — the canonical in-tree default. Default
       behavior is byte-identical when neither env var is set.

    Read at call time (not import time) so tests can monkeypatch the env
    var without re-importing modules.
    """
    env = os.environ.get("ED4ALL_LIBV2_ROOT", "").strip()
    if env:
        return Path(env)
    # Default leg: when ED4ALL_HOME is set, relocate under it; otherwise return
    # the module-global ``LIBV2_PATH`` (read at call time so the session-isolation
    # conftest's ``setattr(paths, "LIBV2_PATH", tmp)`` is honored). ``_data_dir``
    # returns its second arg verbatim when ED4ALL_HOME is unset, so the setattr
    # contract is preserved.
    return _data_dir("libv2", LIBV2_PATH)


def get_training_captures_dir() -> Path:
    """Resolve the ``training-captures/`` root, env-overridable.

    Priority:
    1. ``ED4ALL_TRAINING_CAPTURES_DIR`` env var. NOTE: ``ED4ALL_LIBV2_ROOT``
       intentionally does NOT govern ``training-captures/`` — the legacy
       decision-capture mirror lives at the project root, not under LibV2,
       so it needs its own override. Used by the repo-root ``conftest.py``
       autouse isolation fixture to redirect the legacy mirror into
       ``tmp_path`` and stop pytest runs from growing ``training-captures/``
       by thousands of files.
    2. ``ED4ALL_HOME/training-captures`` when ``ED4ALL_HOME`` is set.
    3. ``TRAINING_DIR`` (``PROJECT_ROOT / "training-captures"``) — the
       canonical in-tree default. Default behavior unchanged when unset.

    Read at call time so tests can monkeypatch without re-importing.
    """
    env = os.environ.get("ED4ALL_TRAINING_CAPTURES_DIR", "").strip()
    if env:
        return Path(env)
    # Default leg: relocate under ED4ALL_HOME when set, else the module-global
    # ``TRAINING_DIR`` (read at call time; honors the conftest setattr).
    return _data_dir("training-captures", TRAINING_DIR)

# ============================================================================
# RUNS PATH
# ============================================================================

RUNS_PATH = STATE_PATH / "runs"


# ============================================================================
# COURSEFORGE EXPORTS / DART OUTPUT (data subdirs of code roots)
# ============================================================================

def courseforge_exports_dir() -> Path:
    """Resolve the Courseforge ``exports/`` data dir.

    Precedence:
    1. ``ED4ALL_HOME/exports`` when ``ED4ALL_HOME`` is set (a relocated
       deployment ships no Courseforge code, so the export data flattens to a
       single level under the data root).
    2. ``COURSEFORGE_PATH / "exports"`` — the in-tree default.

    Read at call time so tests / Docker can flip ``ED4ALL_HOME`` without
    re-importing. Byte-stable to the in-tree path when ``ED4ALL_HOME`` is unset.
    """
    return _data_dir("exports", COURSEFORGE_PATH / "exports")


def dart_output_dir() -> Path:
    """Resolve the DART ``output/`` data dir.

    Precedence:
    1. ``ED4ALL_HOME/dart-output`` when ``ED4ALL_HOME`` is set.
    2. ``DART_PATH / "output"`` — the in-tree default.

    Byte-stable to the in-tree path when ``ED4ALL_HOME`` is unset.
    """
    return _data_dir("dart-output", DART_PATH / "output")


def get_state_runs_dir() -> Path:
    """Resolve the ``state/runs/`` parent directory.

    Priority:
    1. ``ED4ALL_STATE_RUNS_DIR`` env var (used by tests via the
       ``state_runs_isolated`` pytest fixture so unit tests don't
       pollute the real project ``state/runs/``).
    2. ``ED4ALL_HOME/state/runs`` when ``ED4ALL_HOME`` is set.
    3. ``STATE_PATH / "runs"`` — the canonical project location.

    The result is read at call time (not import time) so tests can
    monkeypatch the env var without re-importing modules.
    """
    env_override = os.environ.get("ED4ALL_STATE_RUNS_DIR")
    if env_override:
        return Path(env_override)
    return _data_dir("state", PROJECT_ROOT / "state") / "runs"


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
    "get_training_captures_dir",

    # ED4ALL_HOME relocatable data root
    "ed4all_home",
    "ensure_data_dir",
    "courseforge_exports_dir",
    "dart_output_dir",
    "get_state_runs_dir",

    # LibV2 root resolver
    "libv2_path",

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
