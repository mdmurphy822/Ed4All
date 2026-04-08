"""
Structured Error Types for MCP Tools

Defines error taxonomy for consistent error handling across tools.
Provides machine-readable error codes for automated handling.

Phase 0 Hardening - Requirement 7: MCP Contract Hardening
"""

import json
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCategory(Enum):
    """High-level error categories."""
    INPUT_ERROR = "input_error"           # Client-provided data issues
    PROCESSING_ERROR = "processing_error" # Internal processing failures
    OUTPUT_ERROR = "output_error"         # Result generation issues
    SYSTEM_ERROR = "system_error"         # Infrastructure/system issues
    SECURITY_ERROR = "security_error"     # Security policy violations
    TIMEOUT_ERROR = "timeout_error"       # Operation timeouts


class ErrorCode(Enum):
    """Specific error codes for programmatic handling."""
    # Input errors (E1xxx)
    MISSING_REQUIRED_PARAM = "E1001"
    INVALID_PARAM_TYPE = "E1002"
    INVALID_PARAM_VALUE = "E1003"
    SCHEMA_VALIDATION_FAILED = "E1004"
    INPUT_TOO_LARGE = "E1005"
    UNSUPPORTED_FORMAT = "E1006"

    # Processing errors (E2xxx)
    INTERNAL_ERROR = "E2001"
    DEPENDENCY_FAILED = "E2002"
    RESOURCE_EXHAUSTED = "E2003"
    OPERATION_ABORTED = "E2004"
    EXTERNAL_SERVICE_ERROR = "E2005"
    CONFIGURATION_ERROR = "E2006"
    STATE_INCONSISTENCY = "E2007"

    # Output errors (E3xxx)
    OUTPUT_VALIDATION_FAILED = "E3001"
    ARTIFACT_CREATION_FAILED = "E3002"
    OUTPUT_TOO_LARGE = "E3003"
    SERIALIZATION_ERROR = "E3004"

    # System errors (E4xxx)
    FILE_NOT_FOUND = "E4001"
    FILE_READ_ERROR = "E4002"
    FILE_WRITE_ERROR = "E4003"
    NETWORK_ERROR = "E4004"
    DATABASE_ERROR = "E4005"
    DISK_FULL = "E4006"
    OUT_OF_MEMORY = "E4007"

    # Security errors (E5xxx)
    PATH_TRAVERSAL = "E5001"
    PERMISSION_DENIED = "E5002"
    SANDBOX_VIOLATION = "E5003"
    SECRET_DETECTED = "E5004"
    UNAUTHORIZED = "E5005"
    RATE_LIMIT_EXCEEDED = "E5006"

    # Timeout errors (E6xxx)
    TASK_TIMEOUT = "E6001"
    BATCH_TIMEOUT = "E6002"
    CONNECTION_TIMEOUT = "E6003"


@dataclass
class StructuredError:
    """
    Structured error for MCP tool responses.

    Provides consistent error format across all tools for
    machine-readable error handling.
    """
    category: ErrorCategory
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = False
    retry_after_seconds: Optional[int] = None
    suggestion: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": True,
            "category": self.category.value,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after_seconds": self.retry_after_seconds,
            "suggestion": self.suggestion,
            "stack_trace": self.stack_trace
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        category: Optional[ErrorCategory] = None,
        code: Optional[ErrorCode] = None,
        include_trace: bool = False
    ) -> "StructuredError":
        """
        Create StructuredError from an exception.

        Args:
            exc: The exception to convert
            category: Override category (auto-detected if not provided)
            code: Override code (auto-detected if not provided)
            include_trace: Whether to include stack trace

        Returns:
            StructuredError instance
        """
        # Auto-detect category and code from exception type
        if category is None or code is None:
            auto_cat, auto_code = _classify_exception(exc)
            category = category or auto_cat
            code = code or auto_code

        return cls(
            category=category,
            code=code,
            message=str(exc),
            recoverable=_is_recoverable(exc),
            stack_trace=traceback.format_exc() if include_trace else None
        )


def _classify_exception(exc: Exception) -> tuple:
    """Auto-classify exception into category and code."""
    exc_type = type(exc).__name__
    message = str(exc).lower()

    # Timeout errors (must be checked before IOError since TimeoutError inherits from OSError)
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT_ERROR, ErrorCode.TASK_TIMEOUT

    # File errors
    if isinstance(exc, FileNotFoundError):
        return ErrorCategory.SYSTEM_ERROR, ErrorCode.FILE_NOT_FOUND
    if isinstance(exc, PermissionError):
        return ErrorCategory.SECURITY_ERROR, ErrorCode.PERMISSION_DENIED
    if isinstance(exc, IOError):
        if "no space" in message or "disk full" in message:
            return ErrorCategory.SYSTEM_ERROR, ErrorCode.DISK_FULL
        return ErrorCategory.SYSTEM_ERROR, ErrorCode.FILE_WRITE_ERROR

    # Memory errors
    if isinstance(exc, MemoryError):
        return ErrorCategory.SYSTEM_ERROR, ErrorCode.OUT_OF_MEMORY

    # Timeout errors (string-based fallback for non-builtin timeout exceptions)
    if "timeout" in exc_type.lower() or "timeout" in message:
        return ErrorCategory.TIMEOUT_ERROR, ErrorCode.TASK_TIMEOUT

    # Input validation errors
    if isinstance(exc, (TypeError, ValueError)):
        if "required" in message:
            return ErrorCategory.INPUT_ERROR, ErrorCode.MISSING_REQUIRED_PARAM
        if "invalid" in message or "expected" in message:
            return ErrorCategory.INPUT_ERROR, ErrorCode.INVALID_PARAM_VALUE
        return ErrorCategory.INPUT_ERROR, ErrorCode.INVALID_PARAM_TYPE

    if isinstance(exc, KeyError):
        return ErrorCategory.INPUT_ERROR, ErrorCode.MISSING_REQUIRED_PARAM

    # Security patterns
    if "permission" in message or "denied" in message:
        return ErrorCategory.SECURITY_ERROR, ErrorCode.PERMISSION_DENIED
    if "unauthorized" in message or "auth" in message:
        return ErrorCategory.SECURITY_ERROR, ErrorCode.UNAUTHORIZED
    if "rate limit" in message:
        return ErrorCategory.SECURITY_ERROR, ErrorCode.RATE_LIMIT_EXCEEDED

    # Network errors
    if "connection" in message or "network" in message:
        return ErrorCategory.SYSTEM_ERROR, ErrorCode.NETWORK_ERROR

    # Default to internal error
    return ErrorCategory.PROCESSING_ERROR, ErrorCode.INTERNAL_ERROR


def _is_recoverable(exc: Exception) -> bool:
    """Determine if exception is recoverable (retry-safe)."""
    message = str(exc).lower()

    # Recoverable patterns
    recoverable_patterns = [
        "timeout", "temporary", "rate limit", "too many requests",
        "service unavailable", "connection reset", "try again"
    ]

    return any(p in message for p in recoverable_patterns)


# Convenience constructors

def input_error(
    message: str,
    code: ErrorCode = ErrorCode.INVALID_PARAM_VALUE,
    **details
) -> StructuredError:
    """Create an input error."""
    return StructuredError(
        category=ErrorCategory.INPUT_ERROR,
        code=code,
        message=message,
        details=details or None,
        recoverable=False
    )


def processing_error(
    message: str,
    recoverable: bool = False,
    **details
) -> StructuredError:
    """Create a processing error."""
    return StructuredError(
        category=ErrorCategory.PROCESSING_ERROR,
        code=ErrorCode.INTERNAL_ERROR,
        message=message,
        details=details or None,
        recoverable=recoverable
    )


def output_error(
    message: str,
    code: ErrorCode = ErrorCode.ARTIFACT_CREATION_FAILED,
    **details
) -> StructuredError:
    """Create an output error."""
    return StructuredError(
        category=ErrorCategory.OUTPUT_ERROR,
        code=code,
        message=message,
        details=details or None,
        recoverable=False
    )


def system_error(
    message: str,
    code: ErrorCode = ErrorCode.INTERNAL_ERROR,
    recoverable: bool = False,
    **details
) -> StructuredError:
    """Create a system error."""
    return StructuredError(
        category=ErrorCategory.SYSTEM_ERROR,
        code=code,
        message=message,
        details=details or None,
        recoverable=recoverable
    )


def security_error(
    message: str,
    code: ErrorCode = ErrorCode.PERMISSION_DENIED,
    **details
) -> StructuredError:
    """Create a security error."""
    return StructuredError(
        category=ErrorCategory.SECURITY_ERROR,
        code=code,
        message=message,
        details=details or None,
        recoverable=False
    )


def timeout_error(
    message: str,
    retry_after: int = 60,
    **details
) -> StructuredError:
    """Create a timeout error."""
    return StructuredError(
        category=ErrorCategory.TIMEOUT_ERROR,
        code=ErrorCode.TASK_TIMEOUT,
        message=message,
        details=details or None,
        recoverable=True,
        retry_after_seconds=retry_after
    )


# Error response helpers

def error_response(error: StructuredError) -> Dict[str, Any]:
    """
    Create a standard error response for MCP tools.

    Args:
        error: The StructuredError

    Returns:
        Dictionary suitable for tool response
    """
    return {
        "success": False,
        "error": error.to_dict()
    }


def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """
    Create a standard success response for MCP tools.

    Args:
        data: The response data
        message: Success message

    Returns:
        Dictionary suitable for tool response
    """
    return {
        "success": True,
        "message": message,
        "data": data
    }


class Ed4AllError(Exception):
    """Base exception for Ed4All with structured error support."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PROCESSING_ERROR,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: Optional[Dict] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.structured = StructuredError(
            category=category,
            code=code,
            message=message,
            details=details,
            recoverable=recoverable
        )

    def to_response(self) -> Dict[str, Any]:
        """Convert to error response."""
        return error_response(self.structured)


class InputValidationError(Ed4AllError):
    """Input validation failed."""

    def __init__(self, message: str, **details):
        super().__init__(
            message=message,
            category=ErrorCategory.INPUT_ERROR,
            code=ErrorCode.SCHEMA_VALIDATION_FAILED,
            details=details or None
        )


class SandboxViolationError(Ed4AllError):
    """Sandbox policy violation."""

    def __init__(self, message: str, path: Optional[str] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY_ERROR,
            code=ErrorCode.SANDBOX_VIOLATION,
            details={"path": path} if path else None
        )


class RateLimitError(Ed4AllError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY_ERROR,
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            recoverable=True
        )
        self.structured.retry_after_seconds = retry_after
