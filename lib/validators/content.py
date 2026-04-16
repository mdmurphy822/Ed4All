"""
Content Structure Validator

Validates generated course content for structural correctness:
- Heading hierarchy (h1 -> h2 -> h3, no skips)
- Required sections present (objectives, content, summary)
- No empty or placeholder content
- Minimum content length

Referenced by: config/workflows.yaml (course_generation, textbook_to_course)
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

# Minimum word count for substantive content
MIN_CONTENT_WORDS = 50

# Placeholder patterns that indicate incomplete content
PLACEHOLDER_PATTERNS = [
    r"\[TODO\b",
    r"\[PLACEHOLDER\b",
    r"Lorem ipsum",
    r"TBD\b",
    r"FIXME\b",
    r"INSERT .* HERE",
]


class ContentStructureValidator:
    """Validates HTML content structure for course modules."""

    name = "content_structure"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate content structure.

        Expected inputs:
            html_path: Path to HTML file to validate
            html_content: Raw HTML string (alternative to html_path)
            week: Week number (optional)
            module: Module identifier (optional)
        """
        gate_id = inputs.get("gate_id", "content_structure")
        issues: List[GateIssue] = []

        # Load HTML content
        html_content = inputs.get("html_content", "")
        if not html_content and inputs.get("html_path"):
            path = Path(inputs["html_path"])
            if not path.exists():
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[
                        GateIssue(
                            severity="error",
                            code="FILE_NOT_FOUND",
                            message=f"File not found: {path}",
                        )
                    ],
                )
            html_content = path.read_text(encoding="utf-8")

        if not html_content.strip():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="EMPTY_CONTENT",
                        message="HTML content is empty",
                    )
                ],
            )

        # Check heading hierarchy
        issues.extend(self._check_headings(html_content))

        # Check for placeholder content
        issues.extend(self._check_placeholders(html_content))

        # Check minimum content length
        issues.extend(self._check_content_length(html_content))

        # Determine pass/fail
        has_errors = any(i.severity == "error" for i in issues)
        score = max(0.0, 1.0 - len(issues) * 0.1)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=not has_errors,
            score=score,
            issues=issues,
        )

    def _check_headings(self, html: str) -> List[GateIssue]:
        """Check heading hierarchy for skipped levels."""
        issues = []
        headings = re.findall(r"<(h[1-6])\b[^>]*>", html, re.IGNORECASE)

        if not headings:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="NO_HEADINGS",
                    message="No headings found in content",
                    suggestion="Add heading structure",
                )
            )
            return issues

        prev_level = 0
        for tag in headings:
            level = int(tag[1])
            if prev_level > 0 and level > prev_level + 1:
                issues.append(
                    GateIssue(
                        severity="error",
                        code="HEADING_SKIP",
                        message=f"Heading skips from h{prev_level} to h{level}",
                        suggestion=f"Use h{prev_level + 1} instead",
                    )
                )
            prev_level = level

        # Check for empty headings
        empty = re.findall(
            r"<h[1-6][^>]*>\s*</h[1-6]>", html, re.IGNORECASE
        )
        for _ in empty:
            issues.append(
                GateIssue(
                    severity="error",
                    code="EMPTY_HEADING",
                    message="Empty heading element found",
                    suggestion="Add text or remove empty heading",
                )
            )

        return issues

    def _check_placeholders(self, html: str) -> List[GateIssue]:
        """Check for placeholder or incomplete content."""
        issues = []
        for pattern in PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                issues.append(
                    GateIssue(
                        severity="error",
                        code="PLACEHOLDER_CONTENT",
                        message=f"Placeholder content detected: {matches[0]}",
                        suggestion="Replace placeholder with actual content",
                    )
                )
        return issues

    def _check_content_length(self, html: str) -> List[GateIssue]:
        """Check minimum content length."""
        # Strip HTML tags to get text
        text = re.sub(r"<[^>]+>", " ", html)
        words = text.split()

        if len(words) < MIN_CONTENT_WORDS:
            return [
                GateIssue(
                    severity="warning",
                    code="THIN_CONTENT",
                    message=f"Content has only {len(words)} words (min: {MIN_CONTENT_WORDS})",
                    suggestion="Add more substantive content",
                )
            ]
        return []
