"""
Centralized logging configuration for Ed4All components.

Provides consistent logging setup across all components with:
- Console output (stderr)
- Optional file logging to runtime/logs/
- Configurable log levels
- Structured formatting

Usage:
    from lib.logging_config import setup_logging, get_logger

    # Setup logging for a component
    setup_logging(level="INFO", component="courseforge")

    # Get a logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from lib.paths import PROJECT_ROOT


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEBUG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNTIME_LOGS_DIR = PROJECT_ROOT / "runtime" / "logs"


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    component: str = "ed4all",
    include_debug_info: bool = False,
) -> logging.Logger:
    """
    Configure logging for a component.

    Sets up both console (stderr) and optional file logging with
    consistent formatting across all Ed4All components.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path for file logging. If None and component
                 is specified, will use runtime/logs/{component}.log
        component: Component name for logger hierarchy
        include_debug_info: Include filename/line number in format

    Returns:
        Configured logger instance

    Example:
        # Basic setup
        logger = setup_logging(level="INFO", component="dart")

        # With file logging
        logger = setup_logging(
            level="DEBUG",
            log_file=Path("runtime/logs/dart.log"),
            component="dart"
        )
    """
    # Get or create logger
    logger = logging.getLogger(component)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Choose format based on debug info setting
    log_format = DEBUG_FORMAT if include_debug_info else DEFAULT_FORMAT
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_component_logging(
    component: str,
    level: Optional[str] = None,
    enable_file_logging: bool = False,
) -> logging.Logger:
    """
    Setup logging for a specific component with automatic file logging.

    This is the recommended way to setup logging for Ed4All components.
    It automatically:
    - Uses LOG_LEVEL from environment if not specified
    - Creates log files in runtime/logs/{component}.log if enabled
    - Uses consistent formatting

    Args:
        component: Component name (dart, courseforge, trainforge, orchestrator, mcp)
        level: Log level override. Uses LOG_LEVEL env var if not specified.
        enable_file_logging: Enable logging to runtime/logs/{component}.log

    Returns:
        Configured logger

    Example:
        from lib.logging_config import setup_component_logging

        logger = setup_component_logging("courseforge", enable_file_logging=True)
    """
    # Get level from environment if not specified
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    # Setup log file path if enabled
    log_file = None
    if enable_file_logging:
        RUNTIME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = RUNTIME_LOGS_DIR / f"{component}.log"

    return setup_logging(
        level=level,
        log_file=log_file,
        component=component,
        include_debug_info=(level.upper() == "DEBUG"),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    This is a simple wrapper around logging.getLogger() that ensures
    the logger inherits from the configured root logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger_name: str, level: str) -> None:
    """
    Change the log level for a specific logger.

    Args:
        logger_name: Name of the logger to modify
        level: New log level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# =============================================================================
# RUN ID SUPPORT
# =============================================================================

class RunIdFilter(logging.Filter):
    """
    Logging filter that adds run_id to log records.

    Usage:
        logger = setup_logging(...)
        logger.addFilter(RunIdFilter(run_id="RUN_20260107_123456"))

        # Now all log messages include run_id
        logger.info("Processing")  # Includes run_id in output
    """

    def __init__(self, run_id: str):
        """
        Initialize with a run ID.

        Args:
            run_id: Run identifier to include in log records
        """
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add run_id to the record."""
        record.run_id = self.run_id
        return True


def setup_logging_with_run_id(
    run_id: str,
    level: str = "INFO",
    component: str = "ed4all",
) -> logging.Logger:
    """
    Setup logging with run_id included in all messages.

    Args:
        run_id: Run identifier
        level: Log level
        component: Component name

    Returns:
        Configured logger with run_id filter
    """
    # Custom format with run_id
    format_str = "%(asctime)s [%(levelname)s] [%(run_id)s] %(name)s: %(message)s"

    logger = logging.getLogger(component)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(format_str, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add run_id filter
    logger.addFilter(RunIdFilter(run_id))
    logger.propagate = False

    return logger


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_root_logging(level: Optional[str] = None) -> None:
    """
    Initialize root logging with basic configuration.

    This should be called once at application startup.

    Args:
        level: Log level. Uses LOG_LEVEL env var if not specified.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=DEFAULT_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr,
    )


# =============================================================================
# GENERATE RUN ID
# =============================================================================

def generate_run_id(prefix: str = "run") -> str:
    """
    Generate a unique run ID.

    Format: {prefix}_{YYYYMMDD}_{HHMMSS}

    Args:
        prefix: Prefix for the run ID

    Returns:
        Generated run ID

    Example:
        run_id = generate_run_id("courseforge")
        # Returns: "courseforge_20260107_143022"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
