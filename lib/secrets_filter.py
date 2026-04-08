"""
Secret and PII Redaction Filter

Detects and redacts sensitive information before logging.
Supports configurable patterns and field-based redaction.

Phase 0 Hardening - Requirement 10: Security Posture
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    original_length: int
    redacted_length: int
    redactions_made: int
    redacted_fields: List[str] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "original_length": self.original_length,
            "redacted_length": self.redacted_length,
            "redactions_made": self.redactions_made,
            "redacted_fields": self.redacted_fields,
            "patterns_matched": self.patterns_matched
        }


class SecretsFilter:
    """Filter for detecting and redacting secrets/PII."""

    # Default patterns for secret detection
    DEFAULT_PATTERNS = [
        # API keys and tokens (general)
        (r'(?i)(api[_-]?key|secret|password|token|credential)["\']?\s*[:=]\s*["\']?([^"\'\s,}{]+)', "API_KEY"),
        (r'(?i)bearer\s+[a-zA-Z0-9_\-\.]+', "BEARER_TOKEN"),

        # OpenAI
        (r'sk-[a-zA-Z0-9]{48}', "OPENAI_KEY"),
        (r'sk-proj-[a-zA-Z0-9_\-]{80,}', "OPENAI_PROJECT_KEY"),

        # Anthropic
        (r'sk-ant-[a-zA-Z0-9-]{90,}', "ANTHROPIC_KEY"),

        # GitHub
        (r'ghp_[a-zA-Z0-9]{36}', "GITHUB_PAT"),
        (r'gho_[a-zA-Z0-9]{36}', "GITHUB_OAUTH"),
        (r'ghs_[a-zA-Z0-9]{36}', "GITHUB_APP"),
        (r'github_pat_[a-zA-Z0-9_]{82}', "GITHUB_FINE_PAT"),

        # AWS
        (r'AKIA[0-9A-Z]{16}', "AWS_ACCESS_KEY"),
        (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})', "AWS_SECRET"),

        # Google Cloud
        (r'AIza[0-9A-Za-z_-]{35}', "GOOGLE_API_KEY"),

        # Stripe
        (r'sk_live_[0-9a-zA-Z]{24}', "STRIPE_SECRET"),
        (r'pk_live_[0-9a-zA-Z]{24}', "STRIPE_PUBLISHABLE"),

        # Credentials in URLs
        (r'://([^:]+):([^@]+)@', "URL_CREDENTIALS"),

        # Private keys
        (r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----', "PRIVATE_KEY"),
        (r'-----BEGIN PGP PRIVATE KEY BLOCK-----', "PGP_PRIVATE_KEY"),

        # Database connection strings
        (r'(?i)(mongodb|postgres|mysql|redis)://[^\s]+', "DATABASE_URL"),

        # JWT tokens
        (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', "JWT_TOKEN"),

        # PII patterns
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL"),
        (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
        (r'\b(?:\d{4}[- ]?){3}\d{4}\b', "CREDIT_CARD"),
        (r'\b\d{3}[-.)\s]?\d{3}[-.)\s]?\d{4}\b', "PHONE_NUMBER"),
    ]

    # Sensitive field names (redact entire value)
    DEFAULT_SENSITIVE_FIELDS = {
        'password', 'secret', 'token', 'api_key', 'apikey', 'api-key',
        'access_token', 'refresh_token', 'private_key', 'privatekey',
        'credentials', 'credential', 'auth', 'authorization',
        'secret_key', 'secretkey', 'client_secret', 'clientsecret',
        'bearer', 'session', 'cookie', 'ssn', 'social_security',
        'credit_card', 'creditcard', 'card_number', 'cvv', 'cvc'
    }

    def __init__(
        self,
        extra_patterns: Optional[List[Tuple[str, str]]] = None,
        extra_sensitive_fields: Optional[Set[str]] = None,
        redaction_placeholder: str = "[REDACTED:{name}]"
    ):
        """
        Initialize secrets filter.

        Args:
            extra_patterns: Additional (pattern, name) tuples to detect
            extra_sensitive_fields: Additional field names to redact
            redaction_placeholder: Format string for redaction (use {name})
        """
        self.patterns: List[Tuple[Pattern, str]] = []
        self.redaction_placeholder = redaction_placeholder

        # Compile default patterns
        for pattern, name in self.DEFAULT_PATTERNS:
            try:
                self.patterns.append((re.compile(pattern), name))
            except re.error as e:
                logger.warning(f"Invalid pattern for {name}: {e}")

        # Add extra patterns
        if extra_patterns:
            for pattern, name in extra_patterns:
                try:
                    self.patterns.append((re.compile(pattern), name))
                except re.error as e:
                    logger.warning(f"Invalid extra pattern for {name}: {e}")

        # Build sensitive fields set
        self.sensitive_fields = self.DEFAULT_SENSITIVE_FIELDS.copy()
        if extra_sensitive_fields:
            self.sensitive_fields.update(extra_sensitive_fields)

    def redact_string(self, text: str) -> Tuple[str, int, List[str]]:
        """
        Redact secrets from a string.

        Args:
            text: String to redact

        Returns:
            Tuple of (redacted_text, redaction_count, pattern_names)
        """
        redaction_count = 0
        patterns_matched = []

        for pattern, name in self.patterns:
            matches = pattern.findall(text)
            if matches:
                placeholder = self.redaction_placeholder.format(name=name)
                text = pattern.sub(placeholder, text)
                redaction_count += len(matches)
                if name not in patterns_matched:
                    patterns_matched.append(name)

        return text, redaction_count, patterns_matched

    def redact_dict(
        self,
        data: Dict[str, Any],
        path: str = ""
    ) -> Tuple[Dict[str, Any], RedactionResult]:
        """
        Recursively redact secrets from a dictionary.

        Args:
            data: Dictionary to redact
            path: Current path (for tracking redacted fields)

        Returns:
            Tuple of (redacted_dict, RedactionResult)
        """
        result = RedactionResult(
            original_length=0,
            redacted_length=0,
            redactions_made=0
        )

        redacted = {}

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            key_lower = key.lower()

            # Check if field name is sensitive
            if key_lower in self.sensitive_fields:
                placeholder = self.redaction_placeholder.format(name="SENSITIVE_FIELD")
                redacted[key] = placeholder
                result.redactions_made += 1
                result.redacted_fields.append(current_path)
                continue

            # Recursively handle nested structures
            if isinstance(value, dict):
                redacted[key], sub_result = self.redact_dict(value, current_path)
                result.redactions_made += sub_result.redactions_made
                result.redacted_fields.extend(sub_result.redacted_fields)
                result.patterns_matched.extend(sub_result.patterns_matched)

            elif isinstance(value, list):
                redacted[key] = []
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, dict):
                        item_redacted, sub_result = self.redact_dict(item, item_path)
                        redacted[key].append(item_redacted)
                        result.redactions_made += sub_result.redactions_made
                        result.redacted_fields.extend(sub_result.redacted_fields)
                        result.patterns_matched.extend(sub_result.patterns_matched)
                    elif isinstance(item, str):
                        redacted_item, count, patterns = self.redact_string(item)
                        redacted[key].append(redacted_item)
                        if count > 0:
                            result.redactions_made += count
                            result.redacted_fields.append(item_path)
                            result.patterns_matched.extend(patterns)
                    else:
                        redacted[key].append(item)

            elif isinstance(value, str):
                redacted_value, count, patterns = self.redact_string(value)
                redacted[key] = redacted_value
                if count > 0:
                    result.redactions_made += count
                    result.redacted_fields.append(current_path)
                    result.patterns_matched.extend(patterns)
            else:
                redacted[key] = value

        # Deduplicate patterns
        result.patterns_matched = list(set(result.patterns_matched))

        return redacted, result

    def scan_for_secrets(self, text: str) -> List[Tuple[str, int]]:
        """
        Scan text for secrets without redacting.

        Args:
            text: Text to scan

        Returns:
            List of (pattern_name, occurrence_count) tuples
        """
        found = []
        for pattern, name in self.patterns:
            matches = pattern.findall(text)
            if matches:
                found.append((name, len(matches)))
        return found

    def contains_secrets(self, text: str) -> bool:
        """
        Quick check if text contains any secrets.

        Args:
            text: Text to check

        Returns:
            True if any secrets detected
        """
        for pattern, _ in self.patterns:
            if pattern.search(text):
                return True
        return False

    def is_sensitive_field(self, field_name: str) -> bool:
        """Check if field name indicates sensitive data."""
        return field_name.lower() in self.sensitive_fields


# Global filter instance
_global_filter: Optional[SecretsFilter] = None


def get_secrets_filter() -> SecretsFilter:
    """Get global secrets filter instance."""
    global _global_filter
    if _global_filter is None:
        _global_filter = SecretsFilter()
    return _global_filter


def redact(data: Any) -> Any:
    """
    Convenience function to redact data.

    Args:
        data: String or dict to redact

    Returns:
        Redacted data (same type as input)
    """
    filter = get_secrets_filter()

    if isinstance(data, str):
        redacted, _, _ = filter.redact_string(data)
        return redacted
    elif isinstance(data, dict):
        redacted, _ = filter.redact_dict(data)
        return redacted
    else:
        return data


def scan(text: str) -> List[Tuple[str, int]]:
    """
    Convenience function to scan for secrets.

    Args:
        text: Text to scan

    Returns:
        List of (pattern_name, count) tuples
    """
    return get_secrets_filter().scan_for_secrets(text)


def contains_secrets(text: str) -> bool:
    """
    Convenience function to check for secrets.

    Args:
        text: Text to check

    Returns:
        True if secrets found
    """
    return get_secrets_filter().contains_secrets(text)
