"""W-D7 T7.1 — Result dataclasses for the libv2 packet-integrity validator.

Extracted from :mod:`lib.validators.libv2.packet_integrity` so the
result shape lives in one auditable file. Re-exported through the
canonical module so existing imports
(``from lib.validators.libv2.packet_integrity import ValidationIssue``)
keep resolving.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = ["ValidationIssue", "ValidationResult"]


@dataclass
class ValidationIssue:
    """One issue raised by a packet integrity rule."""

    rule: str
    severity: str  # "critical" | "warning"
    issue_code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "issue_code": self.issue_code,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Aggregate result for ``PacketIntegrityValidator.validate``."""

    archive_root: str
    rules_run: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archive_root": self.archive_root,
            "rules_run": self.rules_run,
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }
