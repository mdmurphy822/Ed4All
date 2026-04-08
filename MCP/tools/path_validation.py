"""
Path validation utilities for MCP tools.

Provides validation functions to check required paths exist
at module load time and during runtime operations.
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class PathValidationError(Exception):
    """Raised when a required path is missing or invalid."""
    pass


def validate_required_paths(
    paths: Dict[str, Path],
    raise_on_missing: bool = False
) -> Dict[str, bool]:
    """Validate that required paths exist.

    Called at module load time to check installation paths.

    Args:
        paths: Dict mapping path names to Path objects
        raise_on_missing: If True, raise PathValidationError on first missing path

    Returns:
        Dict mapping path names to existence status

    Raises:
        PathValidationError: If raise_on_missing=True and a path is missing
    """
    results = {}

    for name, path in paths.items():
        exists = path.exists()
        results[name] = exists

        if not exists:
            msg = f"Required path not found: {name} = {path}"
            logger.warning(msg)

            if raise_on_missing:
                raise PathValidationError(msg)

    return results


def validate_runtime_path(
    path: Path,
    path_name: str = "path",
    must_be_file: bool = False,
    must_be_dir: bool = False
) -> Path:
    """Validate a path during runtime operations.

    Args:
        path: Path to validate
        path_name: Human-readable name for error messages
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory

    Returns:
        The validated Path object

    Raises:
        PathValidationError: If validation fails
    """
    if not path.exists():
        raise PathValidationError(f"{path_name} not found: {path}")

    if must_be_file and not path.is_file():
        raise PathValidationError(f"{path_name} is not a file: {path}")

    if must_be_dir and not path.is_dir():
        raise PathValidationError(f"{path_name} is not a directory: {path}")

    return path


def get_validation_summary(
    paths: Dict[str, Path]
) -> str:
    """Get a human-readable summary of path validation.

    Args:
        paths: Dict mapping path names to Path objects

    Returns:
        Multi-line summary string
    """
    results = validate_required_paths(paths)

    lines = ["Path Validation Summary:"]
    for name, exists in results.items():
        status = "✓" if exists else "✗ MISSING"
        lines.append(f"  {status} {name}: {paths[name]}")

    valid_count = sum(1 for v in results.values() if v)
    lines.append(f"Total: {valid_count}/{len(results)} paths valid")

    return "\n".join(lines)
