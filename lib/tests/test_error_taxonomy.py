"""
Tests for lib/error_taxonomy.py - Structured error handling.
"""
import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.error_taxonomy import (
        ErrorCategory,
        ErrorCode,
        StructuredError,
        Ed4AllError,
        InputValidationError,
        SandboxViolationError,
        RateLimitError,
        input_error,
        processing_error,
        output_error,
        system_error,
        security_error,
        timeout_error,
        error_response,
        success_response,
        _classify_exception,
        _is_recoverable,
    )
except ImportError:
    pytest.skip("error_taxonomy not available", allow_module_level=True)


# =============================================================================
# ERROR CLASSIFICATION TESTS
# =============================================================================

class TestErrorClassification:
    """Test automatic exception classification."""

    @pytest.mark.unit
    def test_classifies_file_not_found(self):
        """FileNotFoundError should classify as SYSTEM_ERROR/FILE_NOT_FOUND."""
        exc = FileNotFoundError("No such file")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.SYSTEM_ERROR
        assert code == ErrorCode.FILE_NOT_FOUND

    @pytest.mark.unit
    def test_classifies_permission_error(self):
        """PermissionError should classify as SECURITY_ERROR/PERMISSION_DENIED."""
        exc = PermissionError("Access denied")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.SECURITY_ERROR
        assert code == ErrorCode.PERMISSION_DENIED

    @pytest.mark.unit
    def test_classifies_timeout_error(self):
        """TimeoutError should classify as TIMEOUT_ERROR/TASK_TIMEOUT."""
        exc = TimeoutError("Operation timed out")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.TIMEOUT_ERROR
        assert code == ErrorCode.TASK_TIMEOUT

    @pytest.mark.unit
    def test_classifies_memory_error(self):
        """MemoryError should classify as SYSTEM_ERROR/OUT_OF_MEMORY."""
        exc = MemoryError("Out of memory")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.SYSTEM_ERROR
        assert code == ErrorCode.OUT_OF_MEMORY

    @pytest.mark.unit
    def test_classifies_type_error_as_input(self):
        """TypeError should classify as INPUT_ERROR."""
        exc = TypeError("Expected string, got int")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.INPUT_ERROR

    @pytest.mark.unit
    def test_classifies_value_error_with_required(self):
        """ValueError with 'required' should be MISSING_REQUIRED_PARAM."""
        exc = ValueError("required field missing")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.INPUT_ERROR
        assert code == ErrorCode.MISSING_REQUIRED_PARAM

    @pytest.mark.unit
    def test_default_fallback_category(self):
        """Unknown exceptions should fall back to PROCESSING_ERROR/INTERNAL_ERROR."""
        exc = Exception("Unknown error")
        cat, code = _classify_exception(exc)

        assert cat == ErrorCategory.PROCESSING_ERROR
        assert code == ErrorCode.INTERNAL_ERROR


# =============================================================================
# RECOVERABILITY TESTS
# =============================================================================

class TestRecoverability:
    """Test recoverability detection."""

    @pytest.mark.unit
    def test_timeout_is_recoverable(self):
        """Timeout errors should be recoverable."""
        exc = Exception("Connection timeout")
        assert _is_recoverable(exc) is True

    @pytest.mark.unit
    def test_rate_limit_is_recoverable(self):
        """Rate limit errors should be recoverable."""
        exc = Exception("Rate limit exceeded, try again")
        assert _is_recoverable(exc) is True

    @pytest.mark.unit
    def test_temporary_is_recoverable(self):
        """Temporary errors should be recoverable."""
        exc = Exception("Temporary failure")
        assert _is_recoverable(exc) is True

    @pytest.mark.unit
    def test_file_not_found_not_recoverable(self):
        """FileNotFoundError should not be recoverable."""
        exc = FileNotFoundError("No such file")
        assert _is_recoverable(exc) is False

    @pytest.mark.unit
    def test_permission_denied_not_recoverable(self):
        """Permission denied should not be recoverable."""
        exc = PermissionError("Permission denied")
        assert _is_recoverable(exc) is False


# =============================================================================
# STRUCTURED ERROR TESTS
# =============================================================================

class TestStructuredError:
    """Test StructuredError dataclass."""

    @pytest.mark.unit
    def test_from_exception_basic(self):
        """Create StructuredError from exception."""
        exc = FileNotFoundError("test.txt not found")
        error = StructuredError.from_exception(exc)

        assert error.category == ErrorCategory.SYSTEM_ERROR
        assert error.code == ErrorCode.FILE_NOT_FOUND
        assert "test.txt" in error.message

    @pytest.mark.unit
    def test_from_exception_with_trace(self):
        """StructuredError should include trace when requested."""
        try:
            raise ValueError("Test error")
        except ValueError as exc:
            error = StructuredError.from_exception(exc, include_trace=True)

        assert error.stack_trace is not None
        assert "ValueError" in error.stack_trace

    @pytest.mark.unit
    def test_from_exception_with_overrides(self):
        """Category and code overrides should be respected."""
        exc = Exception("Generic error")
        error = StructuredError.from_exception(
            exc,
            category=ErrorCategory.SECURITY_ERROR,
            code=ErrorCode.UNAUTHORIZED
        )

        assert error.category == ErrorCategory.SECURITY_ERROR
        assert error.code == ErrorCode.UNAUTHORIZED

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        """to_dict should return serializable dictionary."""
        error = StructuredError(
            category=ErrorCategory.INPUT_ERROR,
            code=ErrorCode.INVALID_PARAM_VALUE,
            message="Invalid value",
            recoverable=False
        )

        d = error.to_dict()

        assert d["error"] is True
        assert d["category"] == "input_error"
        assert d["code"] == "E1003"
        assert d["message"] == "Invalid value"
        assert d["recoverable"] is False

    @pytest.mark.unit
    def test_to_json_valid(self):
        """to_json should produce valid JSON."""
        error = StructuredError(
            category=ErrorCategory.PROCESSING_ERROR,
            code=ErrorCode.INTERNAL_ERROR,
            message="Test error"
        )

        json_str = error.to_json()
        parsed = json.loads(json_str)

        assert parsed["error"] is True
        assert parsed["message"] == "Test error"


# =============================================================================
# CONVENIENCE CONSTRUCTOR TESTS
# =============================================================================

class TestConvenienceConstructors:
    """Test convenience error constructors."""

    @pytest.mark.unit
    def test_input_error(self):
        """input_error should create INPUT_ERROR category."""
        error = input_error("Invalid parameter")

        assert error.category == ErrorCategory.INPUT_ERROR
        assert error.message == "Invalid parameter"
        assert error.recoverable is False

    @pytest.mark.unit
    def test_processing_error(self):
        """processing_error should create PROCESSING_ERROR category."""
        error = processing_error("Processing failed", recoverable=True)

        assert error.category == ErrorCategory.PROCESSING_ERROR
        assert error.code == ErrorCode.INTERNAL_ERROR
        assert error.recoverable is True

    @pytest.mark.unit
    def test_output_error(self):
        """output_error should create OUTPUT_ERROR category."""
        error = output_error("Failed to create artifact")

        assert error.category == ErrorCategory.OUTPUT_ERROR
        assert error.code == ErrorCode.ARTIFACT_CREATION_FAILED

    @pytest.mark.unit
    def test_system_error(self):
        """system_error should create SYSTEM_ERROR category."""
        error = system_error("Disk full", code=ErrorCode.DISK_FULL)

        assert error.category == ErrorCategory.SYSTEM_ERROR
        assert error.code == ErrorCode.DISK_FULL

    @pytest.mark.unit
    def test_security_error(self):
        """security_error should create SECURITY_ERROR category."""
        error = security_error("Access denied")

        assert error.category == ErrorCategory.SECURITY_ERROR
        assert error.code == ErrorCode.PERMISSION_DENIED

    @pytest.mark.unit
    def test_timeout_error(self):
        """timeout_error should create TIMEOUT_ERROR with retry_after."""
        error = timeout_error("Task timed out", retry_after=120)

        assert error.category == ErrorCategory.TIMEOUT_ERROR
        assert error.code == ErrorCode.TASK_TIMEOUT
        assert error.recoverable is True
        assert error.retry_after_seconds == 120


# =============================================================================
# RESPONSE HELPER TESTS
# =============================================================================

class TestResponseHelpers:
    """Test error_response and success_response helpers."""

    @pytest.mark.unit
    def test_error_response_format(self):
        """error_response should include success=False and error dict."""
        error = input_error("Bad input")
        response = error_response(error)

        assert response["success"] is False
        assert "error" in response
        assert response["error"]["message"] == "Bad input"

    @pytest.mark.unit
    def test_success_response_format(self):
        """success_response should include success=True and data."""
        response = success_response({"result": "ok"}, message="Done")

        assert response["success"] is True
        assert response["message"] == "Done"
        assert response["data"]["result"] == "ok"


# =============================================================================
# EXCEPTION CLASS TESTS
# =============================================================================

class TestExceptionClasses:
    """Test custom exception classes."""

    @pytest.mark.unit
    def test_ed4all_error_has_structured(self):
        """Ed4AllError should have structured attribute."""
        error = Ed4AllError("Test error")

        assert hasattr(error, 'structured')
        assert error.structured.message == "Test error"

    @pytest.mark.unit
    def test_ed4all_error_to_response(self):
        """Ed4AllError.to_response should return error response dict."""
        error = Ed4AllError("Test error")
        response = error.to_response()

        assert response["success"] is False
        assert "error" in response

    @pytest.mark.unit
    def test_input_validation_error(self):
        """InputValidationError should have correct category/code."""
        error = InputValidationError("Invalid schema")

        assert error.structured.category == ErrorCategory.INPUT_ERROR
        assert error.structured.code == ErrorCode.SCHEMA_VALIDATION_FAILED

    @pytest.mark.unit
    def test_sandbox_violation_error(self):
        """SandboxViolationError should include path in details."""
        error = SandboxViolationError("Path escape", path="/etc/passwd")

        assert error.structured.category == ErrorCategory.SECURITY_ERROR
        assert error.structured.code == ErrorCode.SANDBOX_VIOLATION
        assert error.structured.details["path"] == "/etc/passwd"

    @pytest.mark.unit
    def test_rate_limit_error(self):
        """RateLimitError should be recoverable with retry_after."""
        error = RateLimitError("Too many requests", retry_after=60)

        assert error.structured.category == ErrorCategory.SECURITY_ERROR
        assert error.structured.code == ErrorCode.RATE_LIMIT_EXCEEDED
        assert error.structured.recoverable is True
        assert error.structured.retry_after_seconds == 60
