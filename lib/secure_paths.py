"""
Secure Path Utilities

Centralized security functions for path validation and safe file operations.
Prevents Zip Slip attacks, path traversal, and directory escapes.

Usage:
    from lib.secure_paths import safe_extract_zip, validate_path_within_root

    # Safe ZIP extraction
    safe_extract_zip(zip_path, extract_dir)

    # Validate user-supplied paths
    safe_path = validate_path_within_root(user_path, PROJECT_ROOT)

    # Sanitize path components from user input
    safe_name = sanitize_path_component(course_name)
"""

import os
import re
import zipfile
from pathlib import Path
from typing import Optional, Set


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""
    pass


class ZipSlipError(PathTraversalError):
    """Raised when a ZIP file contains path traversal entries."""
    pass


def validate_path_within_root(
    path: Path,
    allowed_root: Path,
    must_exist: bool = False,
) -> Path:
    """
    Validate that a path stays within the allowed root directory.

    Prevents path traversal attacks using ".." or symlinks.

    Args:
        path: Path to validate (can be relative or absolute)
        allowed_root: Root directory the path must stay within
        must_exist: If True, raise error if path doesn't exist

    Returns:
        Resolved absolute path that is confirmed within allowed_root

    Raises:
        PathTraversalError: If path escapes allowed_root
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    # Resolve both paths to absolute form
    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()

    # Check if path exists when required
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Check path is within root
    # Use os.path.commonpath for reliable comparison
    try:
        common = os.path.commonpath([str(resolved_path), str(resolved_root)])
        if common != str(resolved_root):
            raise PathTraversalError(
                f"Path escapes allowed root: {path} -> {resolved_path}"
            )
    except ValueError:
        # Paths are on different drives (Windows) or otherwise incompatible
        raise PathTraversalError(
            f"Path not within allowed root: {path}"
        )

    return resolved_path


def sanitize_path_component(
    name: str,
    allow_dots: bool = False,
    max_length: int = 255,
) -> str:
    """
    Sanitize a single path component (filename or directory name).

    Removes or replaces dangerous characters that could be used for
    path traversal or other attacks.

    Args:
        name: The path component to sanitize
        allow_dots: If False, reject names starting with "."
        max_length: Maximum allowed length

    Returns:
        Sanitized path component

    Raises:
        ValueError: If name contains path separators or traversal patterns
    """
    if not name or not name.strip():
        raise ValueError("Path component cannot be empty")

    # Reject obvious traversal attempts
    if '..' in name:
        raise ValueError(f"Path traversal pattern detected: {name}")

    # Reject path separators
    if '/' in name or '\\' in name:
        raise ValueError(f"Path separators not allowed: {name}")

    # Reject hidden files unless explicitly allowed
    if not allow_dots and name.startswith('.'):
        raise ValueError(f"Hidden files not allowed: {name}")

    # Replace dangerous characters with underscores
    # Allow: alphanumeric, underscore, hyphen, dot (if not at start)
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')

    # Ensure we still have something
    if not sanitized:
        raise ValueError(f"Sanitization resulted in empty name: {name}")

    # Enforce length limit
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def safe_extract_zip(
    zip_path: Path,
    extract_to: Path,
    allowed_extensions: Optional[Set[str]] = None,
    max_total_size_mb: int = 1000,
    max_file_count: int = 10000,
) -> int:
    """
    Safely extract a ZIP file with Zip Slip protection.

    Validates each entry in the ZIP to ensure:
    - No path traversal (../)
    - No absolute paths
    - Resolved path stays within extraction directory

    Args:
        zip_path: Path to the ZIP file
        extract_to: Destination directory for extraction
        allowed_extensions: Optional set of allowed file extensions (e.g., {'.html', '.xml'})
        max_total_size_mb: Maximum total size in MB (protection against ZIP bombs)
        max_file_count: Maximum number of files to extract

    Returns:
        Number of files extracted

    Raises:
        ZipSlipError: If any entry would extract outside the target directory
        ValueError: If ZIP exceeds size limits or contains blocked extensions
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Ensure extraction directory exists
    extract_to.mkdir(parents=True, exist_ok=True)

    total_size = 0
    max_total_bytes = max_total_size_mb * 1024 * 1024
    file_count = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # First pass: validate all entries
        for info in zf.infolist():
            member_name = info.filename

            # Skip directories (they end with /)
            if member_name.endswith('/'):
                continue

            file_count += 1
            if file_count > max_file_count:
                raise ValueError(
                    f"ZIP contains too many files: > {max_file_count}"
                )

            # Check for absolute paths
            if member_name.startswith('/') or member_name.startswith('\\'):
                raise ZipSlipError(
                    f"Absolute path in ZIP: {member_name}"
                )

            # Check for path traversal
            if '..' in member_name:
                raise ZipSlipError(
                    f"Path traversal in ZIP: {member_name}"
                )

            # Resolve the target path and verify it's within extract_to
            target_path = (extract_to / member_name).resolve()

            try:
                common = os.path.commonpath([str(target_path), str(extract_to)])
                if common != str(extract_to):
                    raise ZipSlipError(
                        f"ZIP entry escapes target directory: {member_name} -> {target_path}"
                    )
            except ValueError:
                raise ZipSlipError(
                    f"ZIP entry on different path: {member_name}"
                )

            # Check extension if whitelist provided
            if allowed_extensions is not None:
                ext = Path(member_name).suffix.lower()
                if ext not in allowed_extensions and ext != '':
                    raise ValueError(
                        f"Blocked file extension in ZIP: {member_name}"
                    )

            # Track total size
            total_size += info.file_size
            if total_size > max_total_bytes:
                raise ValueError(
                    f"ZIP exceeds maximum size: {total_size} bytes > {max_total_bytes} bytes"
                )

        # Second pass: safe to extract
        zf.extractall(extract_to)

    return file_count


def safe_join_path(
    base: Path,
    *parts: str,
    allowed_root: Optional[Path] = None,
) -> Path:
    """
    Safely join path components, validating against traversal.

    Args:
        base: Base directory
        *parts: Path components to join (will be sanitized)
        allowed_root: If provided, validate result stays within this root

    Returns:
        Safe joined path

    Raises:
        ValueError: If any component is invalid
        PathTraversalError: If result escapes allowed_root
    """
    result = base

    for part in parts:
        sanitized = sanitize_path_component(part, allow_dots=True)
        result = result / sanitized

    if allowed_root is not None:
        result = validate_path_within_root(result, allowed_root)

    return result


def is_safe_path(path: Path, allowed_root: Path) -> bool:
    """
    Check if a path is safe (within allowed root) without raising.

    Args:
        path: Path to check
        allowed_root: Root directory

    Returns:
        True if path is safe, False otherwise
    """
    try:
        validate_path_within_root(path, allowed_root)
        return True
    except (PathTraversalError, FileNotFoundError):
        return False
