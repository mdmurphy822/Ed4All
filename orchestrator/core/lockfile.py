"""
Config Lockfile Management

Creates immutable snapshots of configuration files at run start.
Prevents config drift during execution.

Phase 0 Hardening - Requirement 1: Deterministic Orchestration
"""

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConfigLockfile:
    """Immutable configuration snapshot."""
    run_id: str
    created_at: str
    config_hashes: Dict[str, str]  # filename -> sha256 hash
    config_paths: Dict[str, str]   # filename -> snapshot path
    locked: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ConfigLockfile":
        """Create from dictionary."""
        return cls(**data)


class LockfileManager:
    """Manages config lockfiles for runs."""

    def __init__(self, run_path: Path):
        """
        Initialize lockfile manager.

        Args:
            run_path: Path to the run directory (state/runs/{run_id}/)
        """
        self.run_path = run_path
        self.snapshot_dir = run_path / "config_snapshot"
        self.lockfile_path = run_path / "config_lockfile.json"

    def create_lockfile(
        self,
        run_id: str,
        config_files: List[Path]
    ) -> ConfigLockfile:
        """
        Create lockfile by copying and hashing config files.

        Args:
            run_id: Run identifier
            config_files: List of config file paths to lock

        Returns:
            ConfigLockfile with hashes and snapshot paths

        Raises:
            RuntimeError: If lockfile already exists (immutability violation)
        """
        if self.lockfile_path.exists():
            raise RuntimeError(
                f"Lockfile already exists for run {run_id}. "
                "Config lockfiles are immutable once created."
            )

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        config_hashes = {}
        config_paths = {}

        for config_path in config_files:
            if not config_path.exists():
                logger.warning(f"Config file not found, skipping: {config_path}")
                continue

            # Compute hash
            content = config_path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            config_hashes[config_path.name] = file_hash

            # Copy to snapshot directory
            snapshot_path = self.snapshot_dir / config_path.name
            shutil.copy2(config_path, snapshot_path)
            config_paths[config_path.name] = str(snapshot_path)

            logger.debug(f"Locked config: {config_path.name} -> {file_hash[:12]}...")

        lockfile = ConfigLockfile(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            config_hashes=config_hashes,
            config_paths=config_paths,
            locked=True
        )

        # Atomic write lockfile
        self._atomic_write(self.lockfile_path, lockfile.to_dict())

        logger.info(f"Created config lockfile for run {run_id} with {len(config_files)} files")
        return lockfile

    def verify_lockfile(self) -> Tuple[bool, List[str]]:
        """
        Verify config files haven't changed since lock.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if not self.lockfile_path.exists():
            return False, ["No lockfile found"]

        try:
            with open(self.lockfile_path) as f:
                lockfile_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return False, [f"Failed to read lockfile: {e}"]

        issues = []

        for filename, expected_hash in lockfile_data.get('config_hashes', {}).items():
            snapshot_path = self.snapshot_dir / filename
            if not snapshot_path.exists():
                issues.append(f"Missing snapshot: {filename}")
                continue

            actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                issues.append(f"Hash mismatch: {filename} (snapshot modified)")

        return len(issues) == 0, issues

    def verify_against_originals(self, config_dir: Path) -> Tuple[bool, List[str]]:
        """
        Verify original config files match locked snapshots.

        Args:
            config_dir: Directory containing original config files

        Returns:
            Tuple of (matches, list of differences)
        """
        if not self.lockfile_path.exists():
            return False, ["No lockfile found"]

        try:
            with open(self.lockfile_path) as f:
                lockfile_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return False, [f"Failed to read lockfile: {e}"]

        differences = []

        for filename, locked_hash in lockfile_data.get('config_hashes', {}).items():
            original_path = config_dir / filename
            if not original_path.exists():
                differences.append(f"Original config deleted: {filename}")
                continue

            current_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
            if current_hash != locked_hash:
                differences.append(
                    f"Config modified since lock: {filename} "
                    f"(locked: {locked_hash[:12]}..., current: {current_hash[:12]}...)"
                )

        return len(differences) == 0, differences

    def get_locked_config(self, filename: str) -> Optional[Path]:
        """
        Get path to locked config snapshot.

        Args:
            filename: Name of the config file

        Returns:
            Path to snapshot or None if not found
        """
        snapshot_path = self.snapshot_dir / filename
        return snapshot_path if snapshot_path.exists() else None

    def get_locked_config_content(self, filename: str) -> Optional[str]:
        """
        Get content of a locked config file.

        Args:
            filename: Name of the config file

        Returns:
            Content as string or None if not found
        """
        snapshot_path = self.get_locked_config(filename)
        if snapshot_path:
            return snapshot_path.read_text()
        return None

    def load_lockfile(self) -> Optional[ConfigLockfile]:
        """
        Load existing lockfile.

        Returns:
            ConfigLockfile instance or None if not found
        """
        if not self.lockfile_path.exists():
            return None

        try:
            with open(self.lockfile_path) as f:
                data = json.load(f)
            return ConfigLockfile.from_dict(data)
        except (json.JSONDecodeError, IOError, TypeError) as e:
            logger.error(f"Failed to load lockfile: {e}")
            return None

    def get_config_checksum(self) -> Optional[str]:
        """
        Get combined checksum of all locked configs.

        Returns:
            SHA-256 hash of all config hashes concatenated, or None if no lockfile
        """
        lockfile = self.load_lockfile()
        if not lockfile:
            return None

        # Sort by filename for determinism
        sorted_hashes = sorted(lockfile.config_hashes.items())
        combined = "".join(h for _, h in sorted_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _atomic_write(self, path: Path, data: Dict) -> None:
        """Atomically write JSON data to path."""
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            import os
            os.fsync(f.fileno())
        temp_path.rename(path)


def create_run_lockfile(
    run_id: str,
    run_path: Path,
    config_dir: Path,
    config_files: Optional[List[str]] = None
) -> ConfigLockfile:
    """
    Convenience function to create a lockfile for a run.

    Args:
        run_id: Run identifier
        run_path: Path to run directory
        config_dir: Directory containing config files
        config_files: List of config filenames to lock (defaults to common configs)

    Returns:
        Created ConfigLockfile
    """
    if config_files is None:
        config_files = ["workflows.yaml", "agents.yaml"]

    config_paths = [config_dir / f for f in config_files]

    manager = LockfileManager(run_path)
    return manager.create_lockfile(run_id, config_paths)


def verify_run_lockfile(run_path: Path) -> Tuple[bool, List[str]]:
    """
    Convenience function to verify a run's lockfile integrity.

    Args:
        run_path: Path to run directory

    Returns:
        Tuple of (is_valid, list of issues)
    """
    manager = LockfileManager(run_path)
    return manager.verify_lockfile()
