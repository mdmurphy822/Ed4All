"""
Error Classification and Poison-Pill Detection

Classifies errors as transient vs permanent for retry decisions.
Detects poison-pill patterns that should stop batch processing.

Phase 0 Hardening - Requirement 2: Execution Model Hardening
"""

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorClass(Enum):
    """Error classification for retry decisions."""
    TRANSIENT = "transient"      # Network, timeout, rate limit - retry
    PERMANENT = "permanent"       # Bad input, missing file - don't retry
    POISON_PILL = "poison_pill"  # Systemic issue - stop batch
    UNKNOWN = "unknown"          # Classify manually


@dataclass
class ClassifiedError:
    """Error with classification metadata."""
    error_class: ErrorClass
    error_type: str
    message: str
    normalized_message: str
    task_id: str
    timestamp: str
    retry_recommended: bool
    pattern_hash: str
    original_exception: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['error_class'] = self.error_class.value
        return d


@dataclass
class PoisonPillResult:
    """Result when poison-pill pattern detected."""
    triggered: bool
    pattern_hash: str
    error_pattern: str
    occurrence_count: int
    affected_tasks: List[str]
    recommendation: str
    time_window_seconds: int
    threshold: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ErrorClassifier:
    """Classifies errors for retry and batch decisions."""

    # Patterns indicating transient errors (retry-safe)
    TRANSIENT_PATTERNS = [
        r"timeout",
        r"timed?\s*out",
        r"connection\s*(refused|reset|closed)",
        r"rate\s*limit",
        r"too\s*many\s*requests",
        r"503\s*service\s*unavailable",
        r"504\s*gateway\s*timeout",
        r"502\s*bad\s*gateway",
        r"ECONNRESET",
        r"ETIMEDOUT",
        r"ECONNREFUSED",
        r"temporary\s*(failure|error)",
        r"temporarily\s*unavailable",
        r"retry\s*later",
        r"overloaded",
        r"busy",
        r"anthropic.*overloaded",
        r"openai.*rate",
    ]

    # Patterns indicating permanent errors (don't retry)
    PERMANENT_PATTERNS = [
        r"file\s*not\s*found",
        r"no\s*such\s*file",
        r"permission\s*denied",
        r"access\s*denied",
        r"invalid\s*(input|format|schema|argument|parameter)",
        r"missing\s*required",
        r"404\s*not\s*found",
        r"400\s*bad\s*request",
        r"401\s*unauthorized",
        r"403\s*forbidden",
        r"validation\s*(failed|error)",
        r"malformed",
        r"syntax\s*error",
        r"type\s*error",
        r"key\s*error",
        r"attribute\s*error",
        r"import\s*error",
        r"module\s*not\s*found",
        r"schema.*violation",
        r"constraint.*violation",
    ]

    # Patterns indicating systemic issues (stop batch)
    POISON_PATTERNS = [
        r"out\s*of\s*memory",
        r"disk\s*full",
        r"quota\s*exceeded",
        r"license\s*expired",
        r"api\s*key\s*(invalid|expired|revoked)",
        r"authentication\s*failed",
        r"credentials?\s*(invalid|expired)",
        r"database\s*(down|unavailable|connection\s*failed)",
        r"critical\s*system\s*error",
    ]

    def __init__(self):
        """Initialize error classifier with compiled patterns."""
        self._transient_re = re.compile(
            "|".join(self.TRANSIENT_PATTERNS), re.IGNORECASE
        )
        self._permanent_re = re.compile(
            "|".join(self.PERMANENT_PATTERNS), re.IGNORECASE
        )
        self._poison_re = re.compile(
            "|".join(self.POISON_PATTERNS), re.IGNORECASE
        )

    def classify(self, error: Exception, task_id: str) -> ClassifiedError:
        """
        Classify an error for retry decisions.

        Args:
            error: The exception to classify
            task_id: ID of the task that failed

        Returns:
            ClassifiedError with classification and metadata
        """
        message = str(error)
        error_type = type(error).__name__
        normalized = self._normalize_message(message)
        pattern_hash = self._compute_pattern_hash(error_type, normalized)

        # Check patterns in order of severity
        if self._poison_re.search(message):
            error_class = ErrorClass.POISON_PILL
            retry = False
        elif self._permanent_re.search(message):
            error_class = ErrorClass.PERMANENT
            retry = False
        elif self._transient_re.search(message):
            error_class = ErrorClass.TRANSIENT
            retry = True
        else:
            error_class = ErrorClass.UNKNOWN
            retry = True  # Default to retry for unknown

        classified = ClassifiedError(
            error_class=error_class,
            error_type=error_type,
            message=message,
            normalized_message=normalized,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            retry_recommended=retry,
            pattern_hash=pattern_hash,
            original_exception=repr(error)
        )

        logger.debug(
            f"Classified error for task {task_id}: "
            f"{error_class.value} (retry={retry})"
        )

        return classified

    def _normalize_message(self, message: str) -> str:
        """
        Normalize error message for pattern matching.

        Strips variable data like UUIDs, paths, timestamps, numbers
        to enable grouping of similar errors.
        """
        normalized = message

        # Strip UUIDs
        normalized = re.sub(
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            '<UUID>',
            normalized
        )

        # Strip hex IDs (like EVT_, RUN_, T-)
        normalized = re.sub(r'(EVT_|RUN_|T-)[a-f0-9]+', r'\1<ID>', normalized)

        # Strip file paths
        normalized = re.sub(r'/[^\s,\'"]+', '<PATH>', normalized)
        normalized = re.sub(r'[A-Z]:\\[^\s,\'"]+', '<PATH>', normalized)

        # Strip timestamps
        normalized = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:?\d{2}|Z)?',
            '<TIMESTAMP>',
            normalized
        )

        # Strip large numbers (keep small ones for context)
        normalized = re.sub(r'\b\d{4,}\b', '<NUM>', normalized)

        # Strip IP addresses
        normalized = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', normalized)

        # Lowercase and clean whitespace
        normalized = normalized.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    def _compute_pattern_hash(self, error_type: str, normalized_message: str) -> str:
        """Compute hash for error pattern grouping."""
        pattern = f"{error_type}:{normalized_message}"
        return hashlib.sha256(pattern.encode()).hexdigest()[:16]

    def should_retry(self, error: ClassifiedError) -> bool:
        """Check if error indicates retry is appropriate."""
        return error.retry_recommended and error.error_class in (
            ErrorClass.TRANSIENT,
            ErrorClass.UNKNOWN
        )


class PoisonPillDetector:
    """Detects poison-pill patterns that should stop batch processing."""

    def __init__(
        self,
        threshold: int = 3,
        window_seconds: int = 300
    ):
        """
        Initialize poison-pill detector.

        Args:
            threshold: Number of same-pattern failures to trigger (default 3)
            window_seconds: Time window in seconds (default 5 minutes)
        """
        self.threshold = threshold
        self.window = timedelta(seconds=window_seconds)
        self._errors: Dict[str, List[ClassifiedError]] = defaultdict(list)

    def record_failure(
        self,
        error: ClassifiedError
    ) -> Optional[PoisonPillResult]:
        """
        Record a failure and check for poison-pill pattern.

        Args:
            error: The classified error to record

        Returns:
            PoisonPillResult if threshold exceeded, None otherwise
        """
        pattern_hash = error.pattern_hash
        self._errors[pattern_hash].append(error)

        # Clean old errors outside window
        cutoff = datetime.now() - self.window
        self._errors[pattern_hash] = [
            e for e in self._errors[pattern_hash]
            if datetime.fromisoformat(e.timestamp) > cutoff
        ]

        recent_errors = self._errors[pattern_hash]

        if len(recent_errors) >= self.threshold:
            affected_tasks = [e.task_id for e in recent_errors]

            result = PoisonPillResult(
                triggered=True,
                pattern_hash=pattern_hash,
                error_pattern=recent_errors[0].normalized_message[:200],
                occurrence_count=len(recent_errors),
                affected_tasks=affected_tasks,
                recommendation=(
                    f"Stop batch: {len(recent_errors)} failures with same pattern "
                    f"in {self.window.total_seconds():.0f}s window"
                ),
                time_window_seconds=int(self.window.total_seconds()),
                threshold=self.threshold
            )

            logger.warning(
                f"Poison pill detected: {result.error_pattern[:100]}... "
                f"({result.occurrence_count} occurrences)"
            )

            return result

        return None

    def reset(self) -> None:
        """Clear all recorded errors."""
        self._errors.clear()

    def get_error_summary(self) -> Dict[str, Dict]:
        """
        Get summary of recorded errors by pattern.

        Returns:
            Dictionary mapping pattern_hash to error summary
        """
        summary = {}
        for pattern_hash, errors in self._errors.items():
            if errors:
                summary[pattern_hash] = {
                    "pattern": errors[0].normalized_message[:100],
                    "error_type": errors[0].error_type,
                    "count": len(errors),
                    "affected_tasks": [e.task_id for e in errors],
                    "first_seen": errors[0].timestamp,
                    "last_seen": errors[-1].timestamp
                }
        return summary


class RetryPolicy:
    """Configurable retry policy based on error classification."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 5.0,
        max_delay_seconds: float = 300.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay_seconds: Initial delay between retries
            max_delay_seconds: Maximum delay between retries
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay_seconds
        self.max_delay = max_delay_seconds
        self.exponential_base = exponential_base

    def get_retry_delay(self, attempt: int, error: ClassifiedError) -> float:
        """
        Calculate retry delay for a given attempt.

        Args:
            attempt: Current attempt number (0-indexed)
            error: The classified error

        Returns:
            Delay in seconds before next retry
        """
        if error.error_class == ErrorClass.TRANSIENT:
            # Exponential backoff for transient errors
            delay = self.base_delay * (self.exponential_base ** attempt)
            return min(delay, self.max_delay)
        else:
            # Fixed delay for unknown errors
            return self.base_delay

    def should_retry(self, attempt: int, error: ClassifiedError) -> bool:
        """
        Determine if task should be retried.

        Args:
            attempt: Current attempt number (0-indexed)
            error: The classified error

        Returns:
            True if retry is recommended
        """
        if attempt >= self.max_retries:
            return False

        if error.error_class == ErrorClass.PERMANENT:
            return False

        if error.error_class == ErrorClass.POISON_PILL:
            return False

        return error.retry_recommended


# Convenience functions

def classify_error(error: Exception, task_id: str) -> ClassifiedError:
    """
    Convenience function to classify an error.

    Args:
        error: Exception to classify
        task_id: Task identifier

    Returns:
        ClassifiedError instance
    """
    classifier = ErrorClassifier()
    return classifier.classify(error, task_id)


def is_transient_error(error: Exception) -> bool:
    """
    Quick check if error appears transient.

    Args:
        error: Exception to check

    Returns:
        True if error appears transient (retry-safe)
    """
    classifier = ErrorClassifier()
    classified = classifier.classify(error, "check")
    return classified.error_class == ErrorClass.TRANSIENT
