"""
Centralized configuration loader for all Ed4All components.

Provides unified configuration management with support for:
- Environment variables
- YAML configuration files
- Default values
- Runtime overrides

Configuration precedence (highest to lowest):
1. Runtime overrides (passed to load())
2. Environment variables
3. Configuration files
4. Default values

Usage:
    from lib.config import AppConfig

    # Load from environment
    config = AppConfig.from_env()

    # Load with file
    config = AppConfig.load(config_file="config/app.yaml")

    # Access values
    print(config.log_level)
    print(config.dev_mode)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from lib.paths import CONFIG_PATH, PROJECT_ROOT, STATE_PATH

# =============================================================================
# APP CONFIGURATION
# =============================================================================

@dataclass
class AppConfig:
    """
    Application-wide configuration.

    Attributes:
        project_root: Root directory of the Ed4All project
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        dev_mode: Development mode flag (allows writes outside runtime/)

        enable_audit_logging: Enable file access audit logging
        enable_decision_capture: Enable decision capture for training

        max_parallel_agents: Maximum concurrent agent tasks
        task_timeout_minutes: Timeout for individual tasks
        retry_attempts: Number of retry attempts for failed tasks
    """

    # Core paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    config_path: Path = field(default_factory=lambda: CONFIG_PATH)
    state_path: Path = field(default_factory=lambda: STATE_PATH)

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: Optional[Path] = None

    # Modes
    dev_mode: bool = False

    # Feature toggles
    enable_audit_logging: bool = True
    enable_decision_capture: bool = True

    # Execution limits
    max_parallel_agents: int = 10
    task_timeout_minutes: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 30

    # Batch settings
    max_batch_size: int = 10
    lock_ttl_seconds: int = 3600

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Load configuration from environment variables only.

        Environment variables:
            ED4ALL_ROOT: Project root path
            LOG_LEVEL: Logging level
            DEV_MODE: Development mode (0 or 1)
            ENABLE_AUDIT: Enable audit logging (0 or 1)
            ENABLE_CAPTURE: Enable decision capture (0 or 1)
            MAX_PARALLEL: Max parallel agents
            TASK_TIMEOUT: Task timeout in minutes

        Returns:
            AppConfig instance
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "1" if default else "0")
            return val.lower() in ("1", "true", "yes", "on")

        def get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, str(default)))
            except ValueError:
                return default

        project_root = Path(os.environ.get("ED4ALL_ROOT", PROJECT_ROOT))

        return cls(
            project_root=project_root,
            config_path=project_root / "config",
            state_path=project_root / "state",
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            dev_mode=get_bool("DEV_MODE", False),
            enable_audit_logging=get_bool("ENABLE_AUDIT", True),
            enable_decision_capture=get_bool("ENABLE_CAPTURE", True),
            max_parallel_agents=get_int("MAX_PARALLEL", 10),
            task_timeout_minutes=get_int("TASK_TIMEOUT", 60),
            retry_attempts=get_int("RETRY_ATTEMPTS", 3),
        )

    @classmethod
    def load(
        cls,
        config_file: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "AppConfig":
        """
        Load configuration with full precedence chain.

        Args:
            config_file: Optional YAML config file path
            overrides: Optional dict of runtime overrides

        Returns:
            AppConfig instance

        Raises:
            FileNotFoundError: If config_file specified but not found
            ValueError: If YAML parsing fails
        """
        # Start with environment config
        config = cls.from_env()

        # Load from file if specified
        if config_file:
            config = cls._merge_from_file(config, config_file)

        # Apply runtime overrides
        if overrides:
            config = cls._apply_overrides(config, overrides)

        return config

    @classmethod
    def _merge_from_file(cls, config: "AppConfig", config_file: Path) -> "AppConfig":
        """Merge configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required for file-based config. pip install pyyaml")

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file) as f:
            file_config = yaml.safe_load(f) or {}

        return cls._apply_overrides(config, file_config)

    @classmethod
    def _apply_overrides(
        cls,
        config: "AppConfig",
        overrides: Dict[str, Any]
    ) -> "AppConfig":
        """Apply dict overrides to config."""
        # Create new config with overridden values
        config_dict = {
            'project_root': config.project_root,
            'config_path': config.config_path,
            'state_path': config.state_path,
            'log_level': config.log_level,
            'log_to_file': config.log_to_file,
            'log_file': config.log_file,
            'dev_mode': config.dev_mode,
            'enable_audit_logging': config.enable_audit_logging,
            'enable_decision_capture': config.enable_decision_capture,
            'max_parallel_agents': config.max_parallel_agents,
            'task_timeout_minutes': config.task_timeout_minutes,
            'retry_attempts': config.retry_attempts,
            'retry_delay_seconds': config.retry_delay_seconds,
            'max_batch_size': config.max_batch_size,
            'lock_ttl_seconds': config.lock_ttl_seconds,
        }

        # Apply overrides
        for key, value in overrides.items():
            if key in config_dict:
                # Handle Path conversion
                if key.endswith('_path') or key.endswith('_root'):
                    config_dict[key] = Path(value) if value else None
                else:
                    config_dict[key] = value

        return cls(**config_dict)

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check log level
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}")

        # Check numeric ranges
        if self.max_parallel_agents < 1:
            raise ValueError("max_parallel_agents must be >= 1")

        if self.task_timeout_minutes < 1:
            raise ValueError("task_timeout_minutes must be >= 1")

        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be >= 0")

        # Check paths exist
        if not self.project_root.exists():
            raise ValueError(f"project_root does not exist: {self.project_root}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'project_root': str(self.project_root),
            'config_path': str(self.config_path),
            'state_path': str(self.state_path),
            'log_level': self.log_level,
            'log_to_file': self.log_to_file,
            'log_file': str(self.log_file) if self.log_file else None,
            'dev_mode': self.dev_mode,
            'enable_audit_logging': self.enable_audit_logging,
            'enable_decision_capture': self.enable_decision_capture,
            'max_parallel_agents': self.max_parallel_agents,
            'task_timeout_minutes': self.task_timeout_minutes,
            'retry_attempts': self.retry_attempts,
            'retry_delay_seconds': self.retry_delay_seconds,
            'max_batch_size': self.max_batch_size,
            'lock_ttl_seconds': self.lock_ttl_seconds,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_config() -> AppConfig:
    """
    Get the default application configuration.

    Loads from environment variables with defaults.

    Returns:
        AppConfig instance
    """
    return AppConfig.from_env()


def get_log_level() -> str:
    """Get the configured log level."""
    return os.environ.get("LOG_LEVEL", "INFO").upper()


def is_dev_mode() -> bool:
    """Check if development mode is enabled."""
    return os.environ.get("DEV_MODE", "0") == "1"
