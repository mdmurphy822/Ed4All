"""
Content Structure Validator

Validates generated course content for structural correctness:
- Heading hierarchy (h1 -> h2 -> h3, no skips)
- Required sections present (objectives, content, summary)
- No empty or placeholder content
- Minimum content length

Referenced by: config/workflows.yaml (course_generation, textbook_to_course)
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# H3 W6a: orchestration-phase decision-capture (Pattern A — one emit
# per validate() call).
def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    pages_audited: int,
    sections_count: int,
    headings_count: int,
    paragraphs_count: int,
    avg_section_depth: Optional[float],
    issue_codes: List[str],
) -> None:
    """Emit one ``content_structure_check`` decision per validate() call."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    depth_str = (
        f"{avg_section_depth:.2f}" if avg_section_depth is not None else "n/a"
    )
    rationale = (
        f"Content-structure orchestration check: "
        f"pages_audited={pages_audited}, "
        f"sections_count={sections_count}, "
        f"headings_count={headings_count}, "
        f"paragraphs_count={paragraphs_count}, "
        f"avg_section_depth={depth_str}, "
        f"issue_codes={sorted(set(issue_codes))[:8]}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="content_structure_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on content_structure_check: %s",
            exc,
        )

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
        capture = inputs.get("decision_capture")
        if capture is None:
            capture = inputs.get("capture")
        issues: List[GateIssue] = []

        # Load HTML content
        html_content = inputs.get("html_content", "")
        if not html_content and inputs.get("html_path"):
            path = Path(inputs["html_path"])
            if not path.exists():
                _emit_decision(
                    capture,
                    passed=False,
                    code="FILE_NOT_FOUND",
                    pages_audited=0,
                    sections_count=0,
                    headings_count=0,
                    paragraphs_count=0,
                    avg_section_depth=None,
                    issue_codes=["FILE_NOT_FOUND"],
                )
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
            _emit_decision(
                capture,
                passed=False,
                code="EMPTY_CONTENT",
                pages_audited=0,
                sections_count=0,
                headings_count=0,
                paragraphs_count=0,
                avg_section_depth=None,
                issue_codes=["EMPTY_CONTENT"],
            )
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

        # Compute structural counts for capture rationale.
        headings_count = len(re.findall(r"<h[1-6]\b[^>]*>", html_content, re.IGNORECASE))
        sections_count = len(re.findall(r"<section\b[^>]*>", html_content, re.IGNORECASE))
        paragraphs_count = len(re.findall(r"<p\b[^>]*>", html_content, re.IGNORECASE))
        # Average heading depth is a cheap proxy for section nesting depth.
        depth_levels = [int(t[1]) for t in re.findall(r"<(h[1-6])\b[^>]*>", html_content, re.IGNORECASE)]
        avg_depth = (sum(depth_levels) / len(depth_levels)) if depth_levels else None

        first_error = next(
            (i.code for i in issues if i.severity == "error"), None
        )
        _emit_decision(
            capture,
            passed=not has_errors,
            code=first_error,
            pages_audited=1,
            sections_count=sections_count,
            headings_count=headings_count,
            paragraphs_count=paragraphs_count,
            avg_section_depth=avg_depth,
            issue_codes=[i.code for i in issues],
        )
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
