"""
Run Manager - Canonical Run Lifecycle Management

Manages the lifecycle of workflow runs, including:
- Run ID generation
- Immutable manifest creation
- Run directory structure
- Git commit capture
- Config snapshot with hashes

The canonical run folder is: state/runs/<run_id>/

Phase 0 Hardening: Requirement 0 (Run Manifest System)
Phase 0.5 Enhancement: Enhanced RunContext with finalization support (C1)
"""

import fnmatch
import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .paths import CONFIG_PATH, SCHEMAS_PATH, STATE_PATH
from .state_manager import atomic_read_json, atomic_write_json

# Type hints for optional dependencies (avoid circular imports)
if TYPE_CHECKING:
    from .audit_logger import AuditLogger
    from .sequence_manager import SequenceManager


# ============================================================================
# CONSTANTS
# ============================================================================

RUNS_PATH = STATE_PATH / "runs"
HARDENED_MODE = os.environ.get("ED4ALL_HARDENED_MODE", "false").lower() == "true"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class InputRef:
    """Reference to an input file with provenance."""
    path: str
    content_hash: str
    hash_algorithm: str = "sha256"
    size_bytes: int = 0


@dataclass
class RunManifest:
    """Immutable manifest capturing run initialization state."""
    run_id: str
    created_at: str
    workflow_type: str
    operator: str = ""
    git_commit: Optional[str] = None
    git_dirty: bool = False
    goals: List[str] = field(default_factory=list)
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    config_hashes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    immutable: bool = True
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RunContext:
    """
    Context for an active run, passed to other modules.

    Phase 0.5 Enhanced with:
    - Direct references to sequence_manager and audit_logger
    - Finalization state tracking
    - Timing information
    - Hardened mode flag
    """
    run_id: str
    run_path: Path
    manifest: RunManifest
    _active: bool = True

    # Phase 0.5: Direct references (not separate getters)
    hardened_mode: bool = False
    sequence_manager: Optional["SequenceManager"] = None
    audit_logger: Optional["AuditLogger"] = None
    registry_snapshot: Optional[Dict[str, Any]] = None

    # Phase 0.5: Timing
    started_at: datetime = field(default_factory=datetime.now)

    # Phase 0.5: Finalization state
    finalized: bool = False
    finalization_report: Optional[Dict[str, Any]] = None

    @property
    def decisions_path(self) -> Path:
        """Path to decisions directory."""
        return self.run_path / "decisions"

    @property
    def audit_path(self) -> Path:
        """Path to audit directory."""
        return self.run_path / "audit"

    @property
    def artifacts_path(self) -> Path:
        """Path to artifacts directory."""
        return self.run_path / "artifacts"

    @property
    def checkpoints_path(self) -> Path:
        """Path to checkpoints directory."""
        return self.run_path / "checkpoints"

    @property
    def state_path(self) -> Path:
        """Path to run-specific state directory."""
        return self.run_path / "state"

    @property
    def is_active(self) -> bool:
        """Check if run is still active (not finalized)."""
        return self._active and not self.finalized

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since run started."""
        return (datetime.now() - self.started_at).total_seconds()

    def get_artifact_path(self, component: str, filename: str = "") -> Path:
        """Get path for a component artifact."""
        path = self.artifacts_path / component
        if filename:
            path = path / filename
        return path

    def mark_finalized(self, report: Optional[Dict[str, Any]] = None) -> None:
        """Mark the run as finalized."""
        self.finalized = True
        self._active = False
        self.finalization_report = report


# ============================================================================
# GLOBAL RUN CONTEXT
# ============================================================================

_current_run: Optional[RunContext] = None


def get_current_run() -> Optional[RunContext]:
    """Get the current active run context."""
    return _current_run


def set_current_run(context: Optional[RunContext]) -> None:
    """Set the current active run context."""
    global _current_run
    _current_run = context


# ============================================================================
# RUN MANAGER CLASS
# ============================================================================

class RunManager:
    """Manages workflow run lifecycle."""

    def __init__(self, runs_path: Path = RUNS_PATH):
        """
        Initialize run manager.

        Args:
            runs_path: Base path for runs (default: state/runs/)
        """
        self.runs_path = runs_path
        self.runs_path.mkdir(parents=True, exist_ok=True)

    def generate_run_id(self) -> str:
        """
        Generate a unique run ID.

        Format: RUN_YYYYMMDD_HHMMSS_<8-char-uuid>
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"RUN_{timestamp}_{short_uuid}"

    def get_run_path(self, run_id: str) -> Path:
        """Get the canonical path for a run."""
        return self.runs_path / run_id

    def initialize_run(
        self,
        workflow_type: str,
        workflow_params: Optional[Dict[str, Any]] = None,
        operator: str = "",
        goals: Optional[List[str]] = None,
        inputs: Optional[List[InputRef]] = None,
    ) -> RunContext:
        """
        Initialize a new run.

        Creates the run directory structure and writes the immutable manifest.

        Args:
            workflow_type: Type of workflow (course_generation, intake_remediation, etc.)
            workflow_params: Parameters for the workflow
            operator: User or system initiating the run
            goals: High-level goals for this run
            inputs: Input files with provenance

        Returns:
            RunContext for the new run
        """
        run_id = self.generate_run_id()
        run_path = self.get_run_path(run_id)

        # Create directory structure
        self._create_run_directories(run_path)

        # Capture git state
        git_commit, git_dirty = self._capture_git_state()

        # Capture environment
        environment = self._capture_environment()

        # Hash and snapshot config files
        config_hashes = self._snapshot_configs(run_path)

        # Build manifest
        manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            workflow_type=workflow_type,
            operator=operator or os.environ.get("USER", "unknown"),
            git_commit=git_commit,
            git_dirty=git_dirty,
            goals=goals or [],
            workflow_params=workflow_params or {},
            config_hashes=config_hashes,
            environment=environment,
            inputs=[asdict(i) if isinstance(i, InputRef) else i for i in (inputs or [])],
        )

        # Write immutable manifest
        manifest_path = run_path / "run_manifest.json"
        atomic_write_json(manifest_path, manifest.to_dict())

        # Set restrictive permissions if in hardened mode
        if HARDENED_MODE:
            os.chmod(run_path, 0o700)

        # Create and set context
        context = RunContext(
            run_id=run_id,
            run_path=run_path,
            manifest=manifest,
        )
        set_current_run(context)

        return context

    def load_run(self, run_id: str) -> RunContext:
        """
        Load an existing run.

        Args:
            run_id: Run ID to load

        Returns:
            RunContext for the run

        Raises:
            FileNotFoundError: If run doesn't exist
        """
        run_path = self.get_run_path(run_id)
        manifest_path = run_path / "run_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        manifest_data = atomic_read_json(manifest_path)
        manifest = RunManifest.from_dict(manifest_data)

        context = RunContext(
            run_id=run_id,
            run_path=run_path,
            manifest=manifest,
        )
        set_current_run(context)

        return context

    def finalize_run(
        self,
        run_id: str,
        status: str = "completed",
        summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Finalize a run.

        Writes a finalization record (does not modify the immutable manifest).

        Args:
            run_id: Run ID to finalize
            status: Final status (completed, failed, aborted)
            summary: Summary data
        """
        run_path = self.get_run_path(run_id)
        finalization_path = run_path / "finalization.json"

        finalization = {
            "run_id": run_id,
            "status": status,
            "finalized_at": datetime.now().isoformat(),
            "summary": summary or {},
        }

        atomic_write_json(finalization_path, finalization)

        # Clear current run context if this was the active run
        current = get_current_run()
        if current and current.run_id == run_id:
            current._active = False

    def list_runs(
        self,
        workflow_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[str]:
        """
        List all runs, optionally filtered.

        Args:
            workflow_type: Filter by workflow type
            status: Filter by status (requires finalization.json)

        Returns:
            List of run IDs
        """
        runs = []
        for run_dir in self.runs_path.iterdir():
            if not run_dir.is_dir():
                continue
            if not run_dir.name.startswith("RUN_"):
                continue

            # Apply filters
            if workflow_type or status:
                try:
                    manifest_path = run_dir / "run_manifest.json"
                    manifest = atomic_read_json(manifest_path)

                    if workflow_type and manifest.get("workflow_type") != workflow_type:
                        continue

                    if status:
                        final_path = run_dir / "finalization.json"
                        if final_path.exists():
                            final = atomic_read_json(final_path)
                            if final.get("status") != status:
                                continue
                        elif status != "in_progress":
                            continue
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

            runs.append(run_dir.name)

        return sorted(runs, reverse=True)  # Most recent first

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _create_run_directories(self, run_path: Path) -> None:
        """Create the run directory structure."""
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

    def _capture_git_state(self) -> tuple[Optional[str], bool]:
        """Capture git commit hash and dirty state."""
        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit = result.stdout.strip() if result.returncode == 0 else None

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

            return commit, dirty
        except (subprocess.SubprocessError, FileNotFoundError):
            return None, False

    def _capture_environment(self) -> Dict[str, Any]:
        """
        Capture environment with allowlist/deny patterns.

        Phase 0.5: Enhanced environment capture with configurable filtering.
        """
        import platform
        import sys

        # Base environment info (always captured)
        env_info: Dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "platform_version": platform.release(),
            "hostname": platform.node(),
        }

        # Load environment capture config
        try:
            from .path_constants import get_environment_config
            config = get_environment_config()
        except ImportError:
            config = {}

        capture_mode = config.get("env_capture_mode", "allowlist")

        if capture_mode == "none":
            return env_info

        # Get filter patterns
        allowlist = config.get("env_allowlist", [
            "ED4ALL_*", "PATH", "HOME", "USER", "SHELL", "LANG"
        ])
        deny_patterns = config.get("env_deny_patterns", [
            "KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"
        ])
        max_vars = config.get("max_env_vars_captured", 50)

        # Filter environment variables
        captured_vars: Dict[str, str] = {}
        captured_count = 0

        for key, value in os.environ.items():
            if captured_count >= max_vars:
                break

            # Check deny patterns first
            if self._matches_any_pattern(key, deny_patterns):
                continue

            # Check allowlist
            if self._matches_any_pattern(key, allowlist):
                if capture_mode == "hash_only":
                    # Hash the value for privacy
                    captured_vars[key] = f"sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}"
                else:
                    captured_vars[key] = value
                captured_count += 1

        env_info["captured_env"] = captured_vars
        env_info["capture_mode"] = capture_mode

        return env_info

    def _matches_any_pattern(self, key: str, patterns: List[str]) -> bool:
        """Check if key matches any of the glob patterns."""
        for pattern in patterns:
            # Handle glob patterns
            if "*" in pattern:
                if fnmatch.fnmatch(key, pattern):
                    return True
            # Handle substring patterns (for deny list)
            elif pattern.upper() in key.upper():
                return True
            # Exact match
            elif key == pattern:
                return True
        return False

    def _snapshot_configs(self, run_path: Path) -> Dict[str, str]:
        """
        Snapshot configuration files with hashes.

        Copies config files to run_path/config_snapshot/ and returns hashes.
        """
        config_hashes = {}
        snapshot_dir = run_path / "config_snapshot"

        # Config files to snapshot
        config_files = [
            ("workflows.yaml", CONFIG_PATH / "workflows.yaml"),
            ("agents.yaml", CONFIG_PATH / "agents.yaml"),
        ]

        for name, source_path in config_files:
            if source_path.exists():
                # Read content
                content = source_path.read_bytes()

                # Compute hash
                file_hash = hashlib.sha256(content).hexdigest()
                config_hashes[name.replace(".", "_")] = f"sha256:{file_hash}"

                # Copy to snapshot
                dest_path = snapshot_dir / name
                dest_path.write_bytes(content)

        # Also hash schemas directory
        schemas_hash = self._hash_directory(SCHEMAS_PATH)
        if schemas_hash:
            config_hashes["schemas"] = f"sha256:{schemas_hash}"

        return config_hashes

    def _hash_directory(self, directory: Path) -> Optional[str]:
        """Compute aggregate hash of directory contents."""
        if not directory.exists():
            return None

        hasher = hashlib.sha256()
        for file_path in sorted(directory.rglob("*.json")):
            hasher.update(file_path.name.encode())
            hasher.update(file_path.read_bytes())

        return hasher.hexdigest()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def initialize_run(
    workflow_type: str,
    workflow_params: Optional[Dict[str, Any]] = None,
    operator: str = "",
    goals: Optional[List[str]] = None,
    inputs: Optional[List[InputRef]] = None,
) -> RunContext:
    """
    Initialize a new run (convenience function).

    See RunManager.initialize_run() for details.
    """
    manager = RunManager()
    return manager.initialize_run(
        workflow_type=workflow_type,
        workflow_params=workflow_params,
        operator=operator,
        goals=goals,
        inputs=inputs,
    )


def load_run(run_id: str) -> RunContext:
    """
    Load an existing run (convenience function).

    See RunManager.load_run() for details.
    """
    manager = RunManager()
    return manager.load_run(run_id)


def finalize_run(
    run_id: str,
    status: str = "completed",
    summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Finalize a run (convenience function).

    See RunManager.finalize_run() for details.
    """
    manager = RunManager()
    manager.finalize_run(run_id, status, summary)


def get_run_artifact_path(component: str, filename: str = "") -> Optional[Path]:
    """
    Get artifact path for current run.

    Falls back to legacy paths if no run is active.

    Args:
        component: Component name (dart, courseforge, trainforge)
        filename: Optional filename

    Returns:
        Path to artifact directory/file, or None if no run active
    """
    context = get_current_run()
    if context:
        return context.get_artifact_path(component, filename)
    return None


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "RUNS_PATH",
    "HARDENED_MODE",
    # Data classes
    "InputRef",
    "RunManifest",
    "RunContext",
    # Run manager
    "RunManager",
    # Context functions
    "get_current_run",
    "set_current_run",
    # Convenience functions
    "initialize_run",
    "load_run",
    "finalize_run",
    "get_run_artifact_path",
]
