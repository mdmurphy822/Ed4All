"""
Path Constants - Security and Structure Constants

Centralizes all path-related constants for consistent usage across the project.
These values can be overridden via config/workflows.yaml hardening section.

Phase 0.5 Enhancement: Centralized Path Constants (B2)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

# Maximum allowed path length (filesystem security)
MAX_PATH_LENGTH = 4096

# Maximum filename length
MAX_FILENAME_LENGTH = 255

# Whether to reject absolute paths in user inputs
DISALLOW_ABSOLUTE_PATHS = True

# Whether to reject paths with parent directory traversal (..)
DISALLOW_PARENT_TRAVERSAL = True

# Forbidden path components
FORBIDDEN_PATH_COMPONENTS = frozenset([
    "..",
    "~",
])

# Forbidden filename patterns (security)
FORBIDDEN_FILENAME_PATTERNS = frozenset([
    ".git",
    ".env",
    ".ssh",
    "__pycache__",
])


# ============================================================================
# RUN STRUCTURE CONSTANTS
# ============================================================================

# Standard run directory files
RUN_MANIFEST_FILE = "run_manifest.json"
FINALIZATION_FILE = "finalization.json"
FINALIZATION_REPORT_FILE = "finalization_report.json"
CHECKSUMS_FILE = "checksums.json"

# Standard run subdirectories
RUN_DECISIONS_DIR = "decisions"
RUN_AUDIT_DIR = "audit"
RUN_ARTIFACTS_DIR = "artifacts"
RUN_CHECKPOINTS_DIR = "checkpoints"
RUN_STATE_DIR = "state"
RUN_CONFIG_SNAPSHOT_DIR = "config_snapshot"

# Sequence manager files
SEQUENCE_FILE = "sequence.json"
SEQUENCE_LOCK_FILE = "sequence.lock"

# Hash chain files
AUDIT_CHAIN_FILE = "audit_chain.jsonl"
DECISIONS_CHAIN_FILE = "decisions_chain.jsonl"


# ============================================================================
# PROJECT STRUCTURE CONSTANTS
# ============================================================================

# Environment variable for project root override
PROJECT_ROOT_ENV_VAR = "ED4ALL_ROOT"

# Marker files that indicate project root
PROJECT_MARKER_FILES = [
    "CLAUDE.md",
    ".ed4all_root",
    "pyproject.toml",
]

# Standard project directories (relative to root)
STATE_DIR = "state"
CONFIG_DIR = "config"
SCHEMAS_DIR = "schemas"
LIB_DIR = "lib"
TRAINING_CAPTURES_DIR = "training-captures"
LIBV2_DIR = "LibV2"

# Standard state subdirectories
RUNS_DIR = "runs"
LOCKS_DIR = "locks"
STATUS_DIR = "status"


# ============================================================================
# LIBV2 STRUCTURE CONSTANTS
# ============================================================================

LIBV2_BLOBS_DIR = "blobs"
LIBV2_CATALOG_DIR = "catalog"
LIBV2_COURSES_DIR = "courses"
LIBV2_TOOLS_DIR = "tools"


# ============================================================================
# LOCK FILE CONSTANTS
# ============================================================================

# Default lock timeout in seconds
DEFAULT_LOCK_TIMEOUT_SECONDS = 30

# Default retry count for lock acquisition
DEFAULT_LOCK_RETRY_COUNT = 3

# Exponential backoff delays for lock retries (seconds)
DEFAULT_LOCK_RETRY_BACKOFF = [0.1, 0.5, 2.0]


# ============================================================================
# HASH CONSTANTS
# ============================================================================

# Default hash algorithm
DEFAULT_HASH_ALGORITHM = "sha256"

# Hash prefix format
HASH_PREFIX_FORMAT = "{algorithm}:{hash}"


# ============================================================================
# CONFIG LOADING
# ============================================================================

_config_cache: Optional[Dict[str, Any]] = None


def load_hardening_config() -> Dict[str, Any]:
    """
    Load hardening configuration from workflows.yaml.

    Returns:
        Hardening configuration dictionary
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    try:
        import yaml
        from .paths import CONFIG_PATH

        config_path = CONFIG_PATH / "workflows.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                _config_cache = config.get("hardening", {})
                return _config_cache
    except Exception:
        pass

    _config_cache = {}
    return _config_cache


def get_path_security_config() -> Dict[str, Any]:
    """Get path security configuration."""
    config = load_hardening_config()
    return config.get("path_security", {})


def get_concurrency_config() -> Dict[str, Any]:
    """Get concurrency configuration."""
    config = load_hardening_config()
    return config.get("concurrency", {})


def get_enforcement_config() -> Dict[str, Any]:
    """Get enforcement configuration."""
    config = load_hardening_config()
    return config.get("enforcement", {})


def get_environment_config() -> Dict[str, Any]:
    """Get environment capture configuration."""
    config = load_hardening_config()
    return config.get("environment", {})


# ============================================================================
# CONFIG-AWARE GETTERS
# ============================================================================

def get_max_path_length() -> int:
    """Get configured max path length."""
    config = get_path_security_config()
    return config.get("max_path_length", MAX_PATH_LENGTH)


def get_lock_timeout() -> float:
    """Get configured lock timeout."""
    config = get_concurrency_config()
    return config.get("lock_timeout_seconds", DEFAULT_LOCK_TIMEOUT_SECONDS)


def get_lock_retry_count() -> int:
    """Get configured lock retry count."""
    config = get_concurrency_config()
    return config.get("lock_retry_count", DEFAULT_LOCK_RETRY_COUNT)


def get_lock_retry_backoff() -> list:
    """Get configured lock retry backoff delays."""
    config = get_concurrency_config()
    return config.get("lock_retry_backoff", DEFAULT_LOCK_RETRY_BACKOFF)


def is_parent_traversal_allowed() -> bool:
    """Check if parent traversal is allowed."""
    config = get_path_security_config()
    return not config.get("disallow_parent_traversal", DISALLOW_PARENT_TRAVERSAL)


def is_write_facade_enforced() -> bool:
    """Check if write facade is enforced."""
    config = get_enforcement_config()
    return config.get("enforce_write_facade", False)


def is_finalization_enforced() -> bool:
    """Check if run finalization is enforced."""
    config = get_enforcement_config()
    return config.get("enforce_finalization", False)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Security constants
    "MAX_PATH_LENGTH",
    "MAX_FILENAME_LENGTH",
    "DISALLOW_ABSOLUTE_PATHS",
    "DISALLOW_PARENT_TRAVERSAL",
    "FORBIDDEN_PATH_COMPONENTS",
    "FORBIDDEN_FILENAME_PATTERNS",
    # Run structure
    "RUN_MANIFEST_FILE",
    "FINALIZATION_FILE",
    "FINALIZATION_REPORT_FILE",
    "CHECKSUMS_FILE",
    "RUN_DECISIONS_DIR",
    "RUN_AUDIT_DIR",
    "RUN_ARTIFACTS_DIR",
    "RUN_CHECKPOINTS_DIR",
    "RUN_STATE_DIR",
    "RUN_CONFIG_SNAPSHOT_DIR",
    "SEQUENCE_FILE",
    "SEQUENCE_LOCK_FILE",
    "AUDIT_CHAIN_FILE",
    "DECISIONS_CHAIN_FILE",
    # Project structure
    "PROJECT_ROOT_ENV_VAR",
    "PROJECT_MARKER_FILES",
    "STATE_DIR",
    "CONFIG_DIR",
    "SCHEMAS_DIR",
    "LIB_DIR",
    "TRAINING_CAPTURES_DIR",
    "LIBV2_DIR",
    "RUNS_DIR",
    "LOCKS_DIR",
    "STATUS_DIR",
    # LibV2 structure
    "LIBV2_BLOBS_DIR",
    "LIBV2_CATALOG_DIR",
    "LIBV2_COURSES_DIR",
    "LIBV2_TOOLS_DIR",
    # Lock constants
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_LOCK_RETRY_COUNT",
    "DEFAULT_LOCK_RETRY_BACKOFF",
    # Hash constants
    "DEFAULT_HASH_ALGORITHM",
    "HASH_PREFIX_FORMAT",
    # Config loading
    "load_hardening_config",
    "get_path_security_config",
    "get_concurrency_config",
    "get_enforcement_config",
    "get_environment_config",
    # Config-aware getters
    "get_max_path_length",
    "get_lock_timeout",
    "get_lock_retry_count",
    "get_lock_retry_backoff",
    "is_parent_traversal_allowed",
    "is_write_facade_enforced",
    "is_finalization_enforced",
]
