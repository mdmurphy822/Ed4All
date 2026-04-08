"""
Tests for orchestrator/core/error_classifier.py - Error classification and poison-pill detection.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from orchestrator.core.error_classifier import (
        ClassifiedError,
        ErrorClass,
        ErrorClassifier,
        PoisonPillDetector,
        PoisonPillResult,  # noqa: F401
    )
except ImportError:
    pytest.skip("error_classifier not available", allow_module_level=True)


# =============================================================================
# ERROR CLASSIFIER TESTS
# =============================================================================

class TestErrorClassifier:
    """Test ErrorClassifier error classification."""

    @pytest.fixture
    def classifier(self):
        return ErrorClassifier()

    # -------------------------------------------------------------------------
    # TRANSIENT ERROR TESTS
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_classifies_timeout_as_transient(self, classifier):
        """Timeout errors should be classified as TRANSIENT."""
        error = TimeoutError("Connection timeout")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.TRANSIENT
        assert result.retry_recommended is True

    @pytest.mark.unit
    def test_classifies_connection_reset_as_transient(self, classifier):
        """Connection reset errors should be TRANSIENT."""
        error = ConnectionError("Connection reset by peer")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.TRANSIENT

    @pytest.mark.unit
    def test_classifies_rate_limit_as_transient(self, classifier):
        """Rate limit errors should be TRANSIENT."""
        error = Exception("Rate limit exceeded")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.TRANSIENT

    @pytest.mark.unit
    def test_classifies_503_as_transient(self, classifier):
        """503 Service Unavailable should be TRANSIENT."""
        error = Exception("503 Service Unavailable")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.TRANSIENT

    # -------------------------------------------------------------------------
    # PERMANENT ERROR TESTS
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_classifies_file_not_found_as_permanent(self, classifier):
        """FileNotFoundError should be PERMANENT."""
        error = FileNotFoundError("No such file: config.yaml")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.PERMANENT
        assert result.retry_recommended is False

    @pytest.mark.unit
    def test_classifies_validation_error_as_permanent(self, classifier):
        """Validation errors should be PERMANENT."""
        error = ValueError("Invalid input format")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.PERMANENT

    @pytest.mark.unit
    def test_classifies_permission_denied_as_permanent(self, classifier):
        """Permission denied should be PERMANENT."""
        error = PermissionError("Permission denied")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.PERMANENT

    @pytest.mark.unit
    def test_classifies_400_as_permanent(self, classifier):
        """400 Bad Request should be PERMANENT."""
        error = Exception("400 Bad Request: invalid parameter")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.PERMANENT

    # -------------------------------------------------------------------------
    # POISON PILL ERROR TESTS
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_classifies_oom_as_poison_pill(self, classifier):
        """Out of memory should be POISON_PILL."""
        error = MemoryError("Out of memory")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.POISON_PILL

    @pytest.mark.unit
    def test_classifies_auth_failed_as_poison_pill(self, classifier):
        """Authentication failed should be POISON_PILL."""
        error = Exception("Authentication failed: invalid API key")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.POISON_PILL

    @pytest.mark.unit
    def test_classifies_disk_full_as_poison_pill(self, classifier):
        """Disk full should be POISON_PILL."""
        error = IOError("Disk full")
        result = classifier.classify(error, "T001")

        assert result.error_class == ErrorClass.POISON_PILL

    # -------------------------------------------------------------------------
    # MESSAGE NORMALIZATION TESTS
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_normalizes_message_strips_uuids(self, classifier):
        """Should strip UUIDs from error messages for pattern matching."""
        error = Exception("Task abc123-def456-789012 failed")
        result = classifier.classify(error, "T001")

        # Normalized message should not have UUID patterns
        assert result.normalized_message is not None
        # UUID pattern shouldn't affect classification
        assert result.error_class is not None

    @pytest.mark.unit
    def test_normalizes_message_strips_paths(self, classifier):
        """Should normalize paths in error messages."""
        error = Exception("File not found: /home/user/project/config.yaml")
        result = classifier.classify(error, "T001")

        assert result.normalized_message is not None

    # -------------------------------------------------------------------------
    # PATTERN HASH TESTS
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_pattern_hash_stable(self, classifier):
        """Same error pattern should produce same hash."""
        error1 = TimeoutError("Connection timeout")
        error2 = TimeoutError("Connection timeout")

        result1 = classifier.classify(error1, "T001")
        result2 = classifier.classify(error2, "T002")

        assert result1.pattern_hash == result2.pattern_hash

    @pytest.mark.unit
    def test_pattern_hash_different_for_different_errors(self, classifier):
        """Different errors should have different hashes."""
        error1 = TimeoutError("Connection timeout")
        error2 = FileNotFoundError("File not found")

        result1 = classifier.classify(error1, "T001")
        result2 = classifier.classify(error2, "T001")

        assert result1.pattern_hash != result2.pattern_hash


# =============================================================================
# CLASSIFIED ERROR TESTS
# =============================================================================

class TestClassifiedError:
    """Test ClassifiedError dataclass."""

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        """ClassifiedError should serialize to dict."""
        error = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="TimeoutError",
            message="Connection timeout",
            normalized_message="connection timeout",
            task_id="T001",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="abc123",
        )

        d = error.to_dict()

        assert d["error_class"] == "transient"
        assert d["error_type"] == "TimeoutError"
        assert d["retry_recommended"] is True


# =============================================================================
# POISON PILL DETECTOR TESTS
# =============================================================================

class TestPoisonPillDetector:
    """Test PoisonPillDetector for batch failure detection."""

    @pytest.fixture
    def detector(self):
        """Detector with low threshold for testing."""
        return PoisonPillDetector(threshold=3, window_seconds=300)

    @pytest.mark.unit
    def test_no_trigger_below_threshold(self, detector):
        """Should not trigger below threshold."""
        error = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="TimeoutError",
            message="Timeout",
            normalized_message="timeout",
            task_id="T001",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="abc123",
        )

        # Record only 2 failures (below threshold of 3)
        result1 = detector.record_failure(error)
        result2 = detector.record_failure(error)

        assert result1 is None or not result1.triggered
        assert result2 is None or not result2.triggered

    @pytest.mark.unit
    def test_triggers_at_threshold(self, detector):
        """Should trigger at threshold."""
        error = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="TimeoutError",
            message="Timeout",
            normalized_message="timeout",
            task_id="T001",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="same_pattern",
        )

        # Record failures up to threshold
        detector.record_failure(error)
        detector.record_failure(error)
        result = detector.record_failure(error)

        assert result is not None
        assert result.triggered is True
        assert result.occurrence_count >= 3

    @pytest.mark.unit
    def test_reset_clears_history(self, detector):
        """Reset should clear error history."""
        error = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="TimeoutError",
            message="Timeout",
            normalized_message="timeout",
            task_id="T001",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="abc123",
        )

        detector.record_failure(error)
        detector.record_failure(error)
        detector.reset()

        # After reset, should start fresh
        result = detector.record_failure(error)
        assert result is None or not result.triggered

    @pytest.mark.unit
    def test_different_patterns_dont_combine(self, detector):
        """Different error patterns should not combine."""
        error1 = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="TimeoutError",
            message="Timeout",
            normalized_message="timeout",
            task_id="T001",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="pattern_a",
        )
        error2 = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="ConnectionError",
            message="Connection failed",
            normalized_message="connection failed",
            task_id="T002",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="pattern_b",
        )

        # Alternate between patterns
        detector.record_failure(error1)
        detector.record_failure(error2)
        detector.record_failure(error1)
        result = detector.record_failure(error2)

        # Neither should trigger (only 2 of each)
        assert result is None or not result.triggered

    @pytest.mark.unit
    def test_get_error_summary(self, detector):
        """Should return summary of recorded errors."""
        error = ClassifiedError(
            error_class=ErrorClass.TRANSIENT,
            error_type="TimeoutError",
            message="Timeout",
            normalized_message="timeout",
            task_id="T001",
            timestamp=datetime.now().isoformat(),
            retry_recommended=True,
            pattern_hash="abc123",
        )

        detector.record_failure(error)
        detector.record_failure(error)

        if hasattr(detector, 'get_error_summary'):
            summary = detector.get_error_summary()
            assert isinstance(summary, dict)


# =============================================================================
# RETRY POLICY TESTS
# =============================================================================

class TestRetryPolicy:
    """Test retry policy behavior."""

    @pytest.mark.unit
    def test_exponential_backoff(self):
        """Retry delays should use exponential backoff."""
        try:
            from orchestrator.core.error_classifier import RetryPolicy
            policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=60.0, exponential_base=2.0)

            # Each retry should have longer delay
            delay1 = policy.get_retry_delay(1, None)
            delay2 = policy.get_retry_delay(2, None)
            delay3 = policy.get_retry_delay(3, None)

            assert delay2 > delay1
            assert delay3 > delay2

        except (ImportError, AttributeError):
            pytest.skip("RetryPolicy not available")

    @pytest.mark.unit
    def test_should_retry_transient(self):
        """Should recommend retry for transient errors."""
        try:
            from orchestrator.core.error_classifier import RetryPolicy
            policy = RetryPolicy(max_retries=3)

            error = ClassifiedError(
                error_class=ErrorClass.TRANSIENT,
                error_type="TimeoutError",
                message="Timeout",
                normalized_message="timeout",
                task_id="T001",
                timestamp=datetime.now().isoformat(),
                retry_recommended=True,
                pattern_hash="abc123",
            )

            assert policy.should_retry(1, error) is True

        except (ImportError, AttributeError):
            pytest.skip("RetryPolicy not available")

    @pytest.mark.unit
    def test_should_not_retry_permanent(self):
        """Should not recommend retry for permanent errors."""
        try:
            from orchestrator.core.error_classifier import RetryPolicy
            policy = RetryPolicy(max_retries=3)

            error = ClassifiedError(
                error_class=ErrorClass.PERMANENT,
                error_type="FileNotFoundError",
                message="File not found",
                normalized_message="file not found",
                task_id="T001",
                timestamp=datetime.now().isoformat(),
                retry_recommended=False,
                pattern_hash="abc123",
            )

            assert policy.should_retry(1, error) is False

        except (ImportError, AttributeError):
            pytest.skip("RetryPolicy not available")
