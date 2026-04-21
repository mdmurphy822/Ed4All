#!/usr/bin/env python3
"""
WCAG 2.2 AA Accessibility Validator

Validates HTML content for WCAG 2.2 AA compliance with automated checking
for images, headings, forms, contrast indicators, focus styling, and more.

Features:
- Alt text validation for all images (1.1.1)
- Heading hierarchy validation (1.3.1, 2.4.6)
- Form label association checking (1.3.1)
- ARIA landmark verification (2.4.1)
- Language declaration validation (3.1.1)
- Link text quality assessment (2.4.4)
- Skip link detection (2.4.1)
- Focus indicator validation (2.4.7)
- Focus not obscured detection (2.4.11, 2.4.12) - WCAG 2.2
- Focus appearance validation (2.4.13) - WCAG 2.2
- Target size validation (2.5.8) - WCAG 2.2

Usage:
    from pdf_converter.wcag_validator import WCAGValidator, ValidationReport

    validator = WCAGValidator()
    report = validator.validate(html_content)
    print(report.to_text())
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


class IssueSeverity(Enum):
    """Severity levels for accessibility issues"""
    CRITICAL = "critical"  # WCAG Level A failure
    HIGH = "high"          # WCAG Level AA failure
    MEDIUM = "medium"      # Best practice violation
    LOW = "low"            # Enhancement opportunity


class WCAGCriterion(Enum):
    """WCAG 2.2 Success Criteria relevant to document conversion"""
    # Level A
    SC_1_1_1 = "1.1.1"   # Non-text Content
    SC_1_3_1 = "1.3.1"   # Info and Relationships
    SC_1_3_2 = "1.3.2"   # Meaningful Sequence
    SC_2_1_1 = "2.1.1"   # Keyboard
    SC_2_4_1 = "2.4.1"   # Bypass Blocks
    SC_2_4_2 = "2.4.2"   # Page Titled
    SC_2_4_4 = "2.4.4"   # Link Purpose (In Context)
    SC_3_1_1 = "3.1.1"   # Language of Page
    SC_4_1_1 = "4.1.1"   # Parsing
    SC_4_1_2 = "4.1.2"   # Name, Role, Value
    # Level A (NEW WCAG 2.2)
    SC_2_4_11 = "2.4.11" # Focus Not Obscured (Minimum)
    # Level AA
    SC_1_4_3 = "1.4.3"   # Contrast (Minimum)
    SC_1_4_4 = "1.4.4"   # Resize Text
    SC_2_4_6 = "2.4.6"   # Headings and Labels
    SC_2_4_7 = "2.4.7"   # Focus Visible
    SC_3_1_2 = "3.1.2"   # Language of Parts
    # Level AA (NEW WCAG 2.2)
    SC_2_4_12 = "2.4.12" # Focus Not Obscured (Enhanced)
    SC_2_4_13 = "2.4.13" # Focus Appearance
    SC_2_5_8 = "2.5.8"   # Target Size (Minimum)


@dataclass
class WCAGIssue:
    """Represents a single accessibility issue"""
    criterion: str
    severity: IssueSeverity
    element: str
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete accessibility validation report"""
    file_path: str
    timestamp: str
    total_issues: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    wcag_aa_compliant: bool = True
    issues: List[WCAGIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        """Export report as JSON"""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        data = asdict(self)
        # Convert enums to strings
        for issue in data['issues']:
            if isinstance(issue['severity'], IssueSeverity):
                issue['severity'] = issue['severity'].value

        return json.dumps(data, indent=2, default=serialize)

    def to_text(self) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 70,
            "WCAG 2.2 AA ACCESSIBILITY VALIDATION REPORT",
            "=" * 70,
            f"File: {self.file_path}",
            f"Timestamp: {self.timestamp}",
            "-" * 70,
            f"Total Issues: {self.total_issues}",
            f"  Critical: {self.critical_count}",
            f"  High: {self.high_count}",
            f"  Medium: {self.medium_count}",
            f"  Low: {self.low_count}",
            "-" * 70,
            f"WCAG 2.2 AA Compliant: {'YES' if self.wcag_aa_compliant else 'NO'}",
            "=" * 70,
        ]

        if self.issues:
            lines.append("\nISSUES FOUND:\n")
            for i, issue in enumerate(self.issues, 1):
                severity = issue.severity.value if isinstance(issue.severity, IssueSeverity) else issue.severity
                lines.extend([
                    f"{i}. [{severity.upper()}] WCAG {issue.criterion}",
                    f"   Element: {issue.element}",
                    f"   Issue: {issue.message}",
                ])
                if issue.suggestion:
                    lines.append(f"   Fix: {issue.suggestion}")
                lines.append("")

        return "\n".join(lines)


class WCAGValidator:
    """
    WCAG 2.2 AA Accessibility Validator for HTML Content.

    Performs comprehensive accessibility checks including:
    - Image alt text validation
    - Heading hierarchy analysis
    - Form label associations
    - ARIA landmark validation
    - Language declarations
    - Link text quality
    - Skip link detection
    - Focus indicator validation
    - Focus not obscured detection (WCAG 2.2)
    - Focus appearance validation (WCAG 2.2)
    - Target size validation (WCAG 2.2)
    """

    # Generic link text patterns to flag
    GENERIC_LINK_TEXT = [
        'click here', 'read more', 'learn more', 'more',
        'here', 'link', 'this', 'page', 'info'
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the accessibility validator.

        Args:
            strict_mode: If True, treat warnings as failures
        """
        self.strict_mode = strict_mode
        self.issues: List[WCAGIssue] = []

    # Exposed for gate manager integration — set at class level so the
    # ValidationGateManager can introspect without instantiation.
    name = "wcag_validator"
    version = "2.0.0"  # Wave 31 semantic upgrade

    def validate(self, html=None, file_path: str = "inline"):
        """Validate HTML for WCAG 2.2 AA compliance.

        Dual-mode signature (Wave 31):

        * Legacy positional: ``validator.validate(html_string, file_path)``
          returns a ``ValidationReport``. Preserves pre-Wave-31 callers.
        * Gate-manager kwargs: ``validator.validate({"html_path": ...})``
          (or ``{"html_content": ...}``) returns a ``GateResult``. The
          ``ValidationGateManager`` always calls with a single dict, so we
          detect that and route to the gate adapter.

        Both modes run the same underlying checks; the only difference is
        the return type + input plumbing.
        """
        # Gate manager path — single-positional dict.
        if isinstance(html, dict) and file_path == "inline":
            return self._validate_gate(html)

        if html is None:
            raise ValueError("validate() requires either html string or input dict")

        return self._validate_html(str(html), file_path)

    def _validate_html(self, html: str, file_path: str = "inline") -> ValidationReport:
        """Run all semantic checks + generate a legacy ValidationReport."""
        self.issues = []
        soup = BeautifulSoup(html, 'html.parser')

        # Legacy tag-presence checks (retained — true positives still catch)
        self._check_language_declaration(soup)
        self._check_page_title(soup)
        self._check_images(soup)  # Wave 31: upgraded semantic
        self._check_headings(soup)  # Wave 31: upgraded for PDF artifacts
        self._check_links(soup)
        self._check_forms(soup)
        self._check_tables(soup)
        self._check_landmarks(soup)  # Wave 31: fix multiple-main false positive
        self._check_skip_links(soup)
        self._check_focus_indicators(soup, html)  # Wave 31: fix focus FP
        # WCAG 2.2 specific checks
        self._check_focus_not_obscured(soup, html)
        self._check_focus_appearance(soup, html)
        self._check_target_size(soup, html)  # Wave 31: skip-link exempt

        # Wave 31 new semantic checks
        self._check_empty_lists(soup)         # SC 1.3.1
        self._check_empty_doc_chapters(soup)  # SC 1.3.1
        self._check_toc_anchor_resolution(soup)  # SC 2.4.1 / 2.4.5
        self._check_pdf_artifact_headings(soup)  # SC 2.4.6

        # Generate report
        return self._generate_report(file_path)

    def _validate_gate(self, inputs):
        """Adapter: run validation from a gate manager input dict → GateResult.

        Produces a ``GateResult`` from ``MCP.hardening.validation_gates``
        with a ``score`` computed from the fail rate:
        * score = 1.0 when no CRITICAL / HIGH findings.
        * score drops proportionally with CRITICAL (weight 1.0) and
          HIGH (weight 0.5) counts.
        """
        from MCP.hardening.validation_gates import GateIssue, GateResult

        gate_id = inputs.get("gate_id", "wcag_compliance")
        html_content = inputs.get("html_content") or ""
        html_path = inputs.get("html_path")
        file_path_str = "inline"

        if not html_content and html_path:
            p = Path(html_path)
            if not p.exists():
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FILE_NOT_FOUND",
                        message=f"HTML file not found: {p}",
                    )],
                )
            try:
                html_content = p.read_text(encoding="utf-8", errors="ignore")
                file_path_str = str(p)
            except OSError as exc:
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[GateIssue(
                        severity="critical",
                        code="FILE_READ_ERROR",
                        message=f"Could not read {p}: {exc}",
                    )],
                )

        if not html_content:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="EMPTY_HTML",
                    message="No HTML content or path provided",
                )],
            )

        report = self._validate_html(html_content, file_path_str)

        # Translate severity — map WCAGIssue.severity (enum) → gate severity str.
        sev_map = {
            IssueSeverity.CRITICAL: "critical",
            IssueSeverity.HIGH: "critical",  # Wave 31: HIGH = real SC failure = critical
            IssueSeverity.MEDIUM: "warning",
            IssueSeverity.LOW: "warning",
        }
        gate_issues = []
        for wcag_issue in report.issues:
            sev_enum = wcag_issue.severity
            if not isinstance(sev_enum, IssueSeverity):
                # Defensive fallback.
                sev_enum = IssueSeverity.MEDIUM
            gate_issues.append(GateIssue(
                severity=sev_map.get(sev_enum, "warning"),
                code=f"WCAG_{wcag_issue.criterion.replace('.', '_')}",
                message=wcag_issue.message,
                location=wcag_issue.element,
                suggestion=wcag_issue.suggestion,
            ))

        critical_count = sum(1 for g in gate_issues if g.severity == "critical")
        high_count = report.high_count  # legacy count for scoring

        # Score formula (Wave 31): 1.0 clean; proportional drop with
        # CRITICAL + HIGH counts. High-severity weight 0.5 so many
        # warnings don't tank the score too aggressively.
        score = 1.0
        if report.critical_count + high_count > 0:
            # Normalize against issue count (report.total_issues) so a page
            # with 2/100 critical scores higher than 2/2.
            denominator = max(1, report.total_issues)
            weighted = report.critical_count + 0.5 * high_count
            score = max(0.0, 1.0 - weighted / denominator)

        passed = critical_count == 0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=gate_issues,
        )

    def validate_file(self, file_path: Path) -> ValidationReport:
        """
        Validate an HTML file for accessibility compliance.

        Args:
            file_path: Path to HTML file

        Returns:
            ValidationReport with all issues found
        """
        file_path = Path(file_path)

        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        return self.validate(content, str(file_path))

    def _check_language_declaration(self, soup: BeautifulSoup) -> None:
        """Check for language declaration on html element (WCAG 3.1.1)"""
        html_tag = soup.find('html')

        if not html_tag:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_3_1_1.value,
                severity=IssueSeverity.CRITICAL,
                element="<html>",
                message="Missing <html> element",
                suggestion="Add proper HTML document structure with <html> element"
            ))
            return

        lang = html_tag.get('lang')
        if not lang:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_3_1_1.value,
                severity=IssueSeverity.CRITICAL,
                element="<html>",
                message="Missing language declaration (lang attribute)",
                suggestion='Add lang attribute: <html lang="en">'
            ))

    def _check_page_title(self, soup: BeautifulSoup) -> None:
        """Check for descriptive page title (WCAG 2.4.2)"""
        title = soup.find('title')

        if not title or not title.string or not title.string.strip():
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_2.value,
                severity=IssueSeverity.HIGH,
                element="<title>",
                message="Missing or empty page title",
                suggestion="Add descriptive <title> element in <head>"
            ))

    def _check_images(self, soup: BeautifulSoup) -> None:
        """Check all images for alt text (WCAG 1.1.1) — Wave 31 semantic upgrade.

        Pre-Wave-31: ``alt=""`` was always a MEDIUM "verify decorative"
        warning — every informational figure with an empty alt was
        missed. Wave 31 flips the logic: a ``<figure>`` that has a
        ``<figcaption>`` is declaring itself as informational, so
        ``alt=""`` on the inner ``<img>`` is a real SC 1.1.1 failure
        (the screen-reader user gets no alternative text for a figure
        the author said was meaningful).

        True-decorative images (no figcaption, OR ``role="presentation"``
        on the figure / image, OR ``aria-hidden="true"``) still get a
        clean pass on ``alt=""``.
        """
        images = soup.find_all('img')

        informational_empty_alt_count = 0
        generic_alt_count = 0

        for img in images:
            src = img.get('src', 'unknown')
            alt = img.get('alt')

            if alt is None:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_1_1.value,
                    severity=IssueSeverity.CRITICAL,
                    element=f'<img src="{src[:50]}">',
                    message="Image missing alt attribute",
                    suggestion="Add alt attribute with descriptive text or alt='' for decorative images"
                ))
                continue

            if alt.strip() == '':
                # Determine informational vs decorative.
                role = img.get('role', '')
                aria_hidden = (img.get('aria-hidden') or '').lower() == 'true'
                # Climb to enclosing figure and inspect its caption.
                parent_figure = img.find_parent('figure')
                figure_role = (parent_figure.get('role') if parent_figure else '') or ''
                has_figcaption = False
                if parent_figure:
                    caption = parent_figure.find('figcaption')
                    if caption and caption.get_text(strip=True):
                        has_figcaption = True

                truly_decorative = (
                    role == 'presentation'
                    or role == 'none'
                    or figure_role == 'presentation'
                    or figure_role == 'none'
                    or aria_hidden
                )

                if has_figcaption and not truly_decorative:
                    # Informational figure with empty alt — real SC 1.1.1 fail.
                    informational_empty_alt_count += 1
                    # Avoid flooding the report with one-issue-per-image;
                    # emit one aggregate critical + representative location
                    # per 50 hits.
                    if informational_empty_alt_count <= 5:
                        self.issues.append(WCAGIssue(
                            criterion=WCAGCriterion.SC_1_1_1.value,
                            severity=IssueSeverity.CRITICAL,
                            element=f'<figure><img src="{src[:50]}">',
                            message=(
                                "Informational figure has empty alt but "
                                "a visible <figcaption> — screen-reader users "
                                "get no description of the image content."
                            ),
                            suggestion=(
                                "Either populate alt with a concise description, "
                                "or mark the image as decorative via "
                                "role=\"presentation\" if the caption is the "
                                "complete equivalent."
                            )
                        ))
                elif not truly_decorative and not parent_figure:
                    # Empty alt outside a figure — ambiguous. MEDIUM warning.
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_1_1.value,
                        severity=IssueSeverity.MEDIUM,
                        element=f'<img src="{src[:50]}">',
                        message="Empty alt text outside a figure — verify image is decorative",
                        suggestion="If decorative, add role='presentation'. If meaningful, add descriptive alt text"
                    ))
                # else: decorative image → no issue (pass).
                continue

            if alt.lower() in ['image', 'photo', 'picture', 'graphic', 'icon']:
                generic_alt_count += 1
                if generic_alt_count <= 5:
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_1_1.value,
                        severity=IssueSeverity.HIGH,
                        element=f'<img alt="{alt}">',
                        message="Generic alt text does not describe image content",
                        suggestion="Replace with specific description of what the image shows"
                    ))

        # Emit aggregate summary issues if we suppressed individual hits.
        if informational_empty_alt_count > 5:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_1_1.value,
                severity=IssueSeverity.CRITICAL,
                element="<img> aggregate",
                message=(
                    f"{informational_empty_alt_count - 5} additional "
                    "informational figures have empty alt text (suppressed)."
                ),
                suggestion="Populate alt text across all informational figures."
            ))
        if generic_alt_count > 5:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_1_1.value,
                severity=IssueSeverity.HIGH,
                element="<img> aggregate",
                message=(
                    f"{generic_alt_count - 5} additional generic-alt images "
                    "(suppressed)."
                )
            ))

    def _check_headings(self, soup: BeautifulSoup) -> None:
        """Check heading hierarchy (WCAG 1.3.1, 2.4.6)"""
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        if not headings:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_6.value,
                severity=IssueSeverity.MEDIUM,
                element="document",
                message="No headings found in document",
                suggestion="Add heading structure to organize content"
            ))
            return

        # Check for multiple h1s
        h1_count = len(soup.find_all('h1'))
        if h1_count > 1:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_3_1.value,
                severity=IssueSeverity.MEDIUM,
                element="<h1>",
                message=f"Multiple h1 elements found ({h1_count})",
                suggestion="Use single h1 for main page heading, use h2-h6 for subheadings"
            ))
        elif h1_count == 0:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_6.value,
                severity=IssueSeverity.HIGH,
                element="document",
                message="No h1 element found",
                suggestion="Add h1 element as main page heading"
            ))

        # Check heading hierarchy
        prev_level = 0
        for heading in headings:
            level = int(heading.name[1])
            if prev_level > 0 and level > prev_level + 1:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_3_1.value,
                    severity=IssueSeverity.HIGH,
                    element=f"<{heading.name}>",
                    message=f"Skipped heading level: h{prev_level} to h{level}",
                    suggestion=f"Use h{prev_level + 1} instead of h{level}"
                ))
            prev_level = level

            # Check for empty headings
            if not heading.get_text().strip():
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_4_6.value,
                    severity=IssueSeverity.HIGH,
                    element=f"<{heading.name}>",
                    message="Empty heading element",
                    suggestion="Add text content or remove empty heading"
                ))

    def _check_links(self, soup: BeautifulSoup) -> None:
        """Check link text quality (WCAG 2.4.4)"""
        links = soup.find_all('a')

        for link in links:
            href = link.get('href', '')
            text = link.get_text().strip().lower()

            # Skip skip-links
            if 'skip' in text.lower():
                continue

            # Check for generic link text
            if text in self.GENERIC_LINK_TEXT:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_4_4.value,
                    severity=IssueSeverity.HIGH,
                    element=f'<a href="{href[:30]}...">{text}</a>',
                    message=f"Generic link text: '{text}'",
                    suggestion="Use descriptive link text that indicates destination"
                ))

            # Check for links with no text
            if not text and not link.find('img'):
                aria_label = link.get('aria-label', '')
                if not aria_label:
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_2_4_4.value,
                        severity=IssueSeverity.CRITICAL,
                        element=f'<a href="{href[:30]}...">',
                        message="Link has no accessible text",
                        suggestion="Add link text or aria-label for accessible name"
                    ))

    def _check_forms(self, soup: BeautifulSoup) -> None:
        """Check form accessibility (WCAG 1.3.1, 4.1.2)"""
        inputs = soup.find_all(['input', 'select', 'textarea'])

        for input_elem in inputs:
            input_type = input_elem.get('type', 'text')
            input_id = input_elem.get('id')
            input_name = input_elem.get('name', 'unnamed')

            # Skip hidden and submit/button types
            if input_type in ['hidden', 'submit', 'button', 'reset']:
                continue

            # Check for associated label
            has_label = False

            if input_id:
                label = soup.find('label', {'for': input_id})
                if label:
                    has_label = True

            # Check for aria-label or aria-labelledby
            if input_elem.get('aria-label') or input_elem.get('aria-labelledby'):
                has_label = True

            # Check for wrapping label
            if input_elem.find_parent('label'):
                has_label = True

            if not has_label:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_3_1.value,
                    severity=IssueSeverity.CRITICAL,
                    element=f'<{input_elem.name} name="{input_name}">',
                    message="Form control missing associated label",
                    suggestion="Add <label for='id'> or aria-label attribute"
                ))

    def _check_tables(self, soup: BeautifulSoup) -> None:
        """Check data table accessibility (WCAG 1.3.1)"""
        tables = soup.find_all('table')

        for table in tables:
            # Check for caption or aria-label
            caption = table.find('caption')
            aria_label = table.get('aria-label')
            aria_labelledby = table.get('aria-labelledby')

            if not caption and not aria_label and not aria_labelledby:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_3_1.value,
                    severity=IssueSeverity.MEDIUM,
                    element="<table>",
                    message="Data table missing caption or accessible name",
                    suggestion="Add <caption> or aria-label to describe table purpose"
                ))

            # Check for header cells
            headers = table.find_all('th')
            if not headers:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_3_1.value,
                    severity=IssueSeverity.HIGH,
                    element="<table>",
                    message="Data table missing header cells (<th>)",
                    suggestion="Add <th> elements for column/row headers"
                ))
            else:
                # Check for scope attribute
                for th in headers:
                    if not th.get('scope'):
                        header_text = th.get_text()[:20]
                        self.issues.append(WCAGIssue(
                            criterion=WCAGCriterion.SC_1_3_1.value,
                            severity=IssueSeverity.MEDIUM,
                            element=f"<th>{header_text}...</th>",
                            message="Table header missing scope attribute",
                            suggestion='Add scope="col" or scope="row" to header cells'
                        ))

    def _check_landmarks(self, soup: BeautifulSoup) -> None:
        """Check for ARIA landmarks (WCAG 1.3.1, 2.4.1).

        Wave 31: fix the multiple-main false positive. Pre-Wave-31 the
        validator would count a single ``<main role="main">`` element
        as two landmarks (one for the HTML tag, one for the ARIA role
        override) and fire a HIGH finding. Wave 31 counts each *unique
        element* once — ``role="main"`` on a ``<main>`` tag is a
        redundant-but-valid authoring pattern, not a duplicate
        landmark.
        """
        # Check for main landmark
        main = soup.find('main') or soup.find(attrs={'role': 'main'})
        if not main:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_1.value,
                severity=IssueSeverity.MEDIUM,
                element="document",
                message="Missing main landmark",
                suggestion="Add <main> element to wrap primary content"
            ))

        # Deduplicate by element identity. A single <main role="main">
        # is exactly ONE landmark; the role="main" overlay on a <main>
        # tag is a valid (if redundant) authoring pattern.
        main_elements = set(id(el) for el in soup.find_all('main'))
        # Only count role="main" elements that are NOT already a <main>
        # tag — those are distinct landmarks.
        for el in soup.find_all(attrs={'role': 'main'}):
            if el.name != 'main':
                main_elements.add(id(el))

        if len(main_elements) > 1:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_3_1.value,
                severity=IssueSeverity.HIGH,
                element="<main>",
                message=f"Multiple main landmarks found ({len(main_elements)} unique elements)",
                suggestion="Use only one main landmark per page"
            ))

    def _check_skip_links(self, soup: BeautifulSoup) -> None:
        """Check for skip navigation links (WCAG 2.4.1)"""
        skip_patterns = ['skip', 'jump to', 'go to main', 'skip to main']
        links = soup.find_all('a')

        has_skip_link = False
        for link in links:
            text = link.get_text().lower()
            href = link.get('href', '')
            css_class = ' '.join(link.get('class', []))

            if any(pattern in text for pattern in skip_patterns) or \
               href.startswith('#main') or \
               'skip-link' in css_class:
                has_skip_link = True
                break

        if not has_skip_link:
            # Only flag if there's significant navigation before main content
            nav = soup.find('nav')
            header = soup.find('header')
            if nav or header:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_4_1.value,
                    severity=IssueSeverity.MEDIUM,
                    element="document",
                    message="Consider adding skip navigation link",
                    suggestion='Add <a href="#main" class="skip-link">Skip to main content</a>'
                ))

    def _check_focus_indicators(self, soup: BeautifulSoup, content: str) -> None:
        """Check for focus indicator removal (WCAG 2.4.7).

        Wave 31: fix false positive. Pre-Wave-31 any occurrence of
        ``outline: none`` would flag HIGH, even when paired with a
        ``:focus-visible`` rule that re-adds outline for
        keyboard-only users. The modern pattern is:

        .. code-block:: css

            :focus:not(:focus-visible) { outline: none; }
            :focus-visible { outline: 2px solid accent; }

        This is the canonical WCAG 2.2 technique (removes outline for
        mouse focus, keeps it for keyboard focus) — no warning.

        Wave 31 only flags ``outline:none`` when no ``:focus-visible``
        rule re-introduces outline anywhere in the stylesheet, regardless
        of selector-ordering.
        """
        # Extract CSS content out of <style> blocks only — running the
        # CSS rule parser on raw HTML is catastrophic (1.5MB pages).
        css_texts = re.findall(
            r'<style\b[^>]*>(.*?)</style>',
            content,
            re.IGNORECASE | re.DOTALL,
        )
        css_combined = "\n".join(css_texts)
        # Also accept inline style= attributes — skip those for focus
        # indicator analysis (they can't carry :focus pseudo-selectors).

        has_focus_visible_outline = False
        problematic_outline_none = False
        if css_combined:
            has_focus_visible_outline = bool(re.search(
                r':focus-visible\b[^{]*\{[^}]*outline\b',
                css_combined, re.IGNORECASE | re.DOTALL
            ))
            # Parse CSS rules only inside the style blocks.
            css_rule_iter = re.finditer(
                r'([^{}]+)\{([^{}]*)\}',
                css_combined,
                re.DOTALL,
            )
            for rule_match in css_rule_iter:
                selector = rule_match.group(1).strip().lower()
                body = rule_match.group(2)
                if re.search(r'outline\s*:\s*(none|0)\b', body, re.IGNORECASE):
                    if ':focus:not(:focus-visible)' in selector:
                        continue
                    problematic_outline_none = True
                    break

        if problematic_outline_none and not has_focus_visible_outline:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_7.value,
                severity=IssueSeverity.HIGH,
                element="<style>",
                message="Focus indicator may be removed (outline:none detected)",
                suggestion="Ensure custom focus styles are provided when removing default outline"
            ))

    def _check_focus_not_obscured(self, soup: BeautifulSoup, content: str) -> None:
        """Check for elements that might obscure focus (WCAG 2.4.11, 2.4.12)"""
        # Check for sticky/fixed position elements without scroll-margin
        has_fixed_sticky = re.search(r'position\s*:\s*(fixed|sticky)', content, re.IGNORECASE)
        has_scroll_margin = re.search(r'scroll-margin', content, re.IGNORECASE)

        if has_fixed_sticky and not has_scroll_margin:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_11.value,
                severity=IssueSeverity.MEDIUM,
                element="<style>",
                message="Fixed/sticky positioned elements detected without scroll-margin",
                suggestion="Add scroll-margin-top/bottom to focused elements to prevent obscuring by fixed headers/footers"
            ))

    def _check_focus_appearance(self, soup: BeautifulSoup, content: str) -> None:
        """Check focus indicator styling meets WCAG 2.2 requirements (WCAG 2.4.13)"""
        # Look for focus styling
        focus_pattern = re.search(
            r':focus[^{]*\{[^}]*outline\s*:\s*(\d+)px',
            content, re.IGNORECASE
        )

        if focus_pattern:
            outline_width = int(focus_pattern.group(1))
            if outline_width < 2:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_4_13.value,
                    severity=IssueSeverity.HIGH,
                    element=":focus",
                    message=f"Focus outline width ({outline_width}px) below 2px minimum",
                    suggestion="Set focus outline to minimum 2px solid with 3:1 contrast ratio"
                ))

    def _check_target_size(self, soup: BeautifulSoup, content: str) -> None:
        """Check interactive element target sizes (WCAG 2.5.8).

        Wave 31: fix the visually-hidden skip-link false positive.
        Pre-Wave-31 the validator fired a HIGH finding on every 1px-by-1px
        interactive element, including skip-links that are deliberately
        offscreen via the visually-hidden pattern (``position:absolute;
        left:-9999px``). Per SC 2.5.8 exceptions: elements outside the
        normal tab focus order or rendered offscreen are exempt from
        the 24×24 minimum.

        Wave 31 detection: when a small-target rule selector matches
        ``.skip-link``, ``.sr-only``, ``.visually-hidden``, ``.screen-reader-text``
        (the canonical visually-hidden class names), or the block contains
        ``position:absolute`` paired with ``left:-9999px`` / ``clip:rect(0,0,0,0)``,
        skip the flag.
        """
        # Extract <style> content only — avoid O(n²) regex on 1.5MB of HTML.
        css_texts = re.findall(
            r'<style\b[^>]*>(.*?)</style>',
            content,
            re.IGNORECASE | re.DOTALL,
        )
        css_combined = "\n".join(css_texts)

        # Pre-extract visually-hidden selectors by scanning CSS rules.
        visually_hidden_selector_names = {
            '.skip-link', '.sr-only', '.visually-hidden',
            '.screen-reader-text', '.a11y-hidden', '.visuallyhidden',
        }

        # Collect selectors whose body marks them as visually-hidden.
        visually_hidden_selectors = set()
        if css_combined:
            for rule_match in re.finditer(r'([^{}]+)\{([^{}]*)\}', css_combined, re.DOTALL):
                selector = rule_match.group(1).strip().lower()
                body = rule_match.group(2).lower()
                is_visually_hidden = (
                    ('position' in body and 'absolute' in body and (
                        '-9999px' in body or '-10000px' in body or
                        re.search(r'clip\s*:\s*rect\s*\(\s*0[^)]*\)', body)
                    ))
                    or re.search(r'clip\s*:\s*rect\s*\(\s*0[^)]*\)', body)
                    or re.search(r'\bwidth\s*:\s*1px\b', body) and re.search(r'\bheight\s*:\s*1px\b', body) and 'overflow' in body
                )
                if is_visually_hidden:
                    visually_hidden_selectors.add(selector)

        # Check for explicit small sizing on interactive elements (CSS only).
        small_size_patterns = [
            r'(button[^{]*)\{([^}]*(?:width|height)\s*:\s*(\d+)px[^}]*)\}',
            r'(\.btn[^{]*)\{([^}]*(?:width|height)\s*:\s*(\d+)px[^}]*)\}',
            r'(input\[type[^{]*)\{([^}]*(?:width|height)\s*:\s*(\d+)px[^}]*)\}',
        ]

        for pattern in small_size_patterns:
            for match in re.finditer(pattern, css_combined, re.IGNORECASE | re.DOTALL):
                selector = match.group(1).strip().lower()
                body = match.group(2).lower()
                size = int(match.group(3))
                if not (0 < size < 24):
                    continue
                # Wave 31 exemption: visually-hidden interactive elements.
                if any(sel in selector for sel in visually_hidden_selector_names):
                    continue
                if selector in visually_hidden_selectors:
                    continue
                if (('position' in body and 'absolute' in body and ('-9999px' in body or '-10000px' in body))
                        or re.search(r'clip\s*:\s*rect\s*\(\s*0[^)]*\)', body)):
                    continue
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_5_8.value,
                    severity=IssueSeverity.HIGH,
                    element="interactive element",
                    message=f"Interactive element size ({size}px) below 24px minimum",
                    suggestion="Ensure interactive elements are at least 24x24 CSS pixels"
                ))

        # Check for btn-xs class which typically produces small targets
        if 'btn-xs' in content.lower():
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_5_8.value,
                severity=IssueSeverity.MEDIUM,
                element=".btn-xs",
                message="Extra-small button class may not meet 24px target size",
                suggestion="Ensure clickable area is at least 24x24 CSS pixels or has adequate spacing"
            ))

    # ------------------------------------------------------------------ #
    # Wave 31 new semantic checks
    # ------------------------------------------------------------------ #

    def _check_empty_lists(self, soup: BeautifulSoup) -> None:
        """SC 1.3.1 — empty lists are info-relationship failures.

        Pre-Wave-31 tag-presence validation accepted ``<ul></ul>`` as
        "has a list". Screen readers announce an empty list as
        "list, 0 items" — real SC 1.3.1 failure.
        """
        empty_count = 0
        for list_el in soup.find_all(['ul', 'ol']):
            items = list_el.find_all('li', recursive=False)
            # Also count li descendants in case the list wraps them deeper.
            if not items:
                items = list_el.find_all('li')
            if not items:
                empty_count += 1
                if empty_count <= 5:
                    # Include the preceding heading text for context.
                    prev_heading = list_el.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    ctx = prev_heading.get_text(strip=True)[:50] if prev_heading else ''
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_3_1.value,
                        severity=IssueSeverity.CRITICAL,
                        element=f'<{list_el.name}>',
                        message=(
                            f"Empty {list_el.name} element"
                            + (f" under heading: {ctx!r}" if ctx else "")
                            + " — screen readers announce as 'list, 0 items'."
                        ),
                        suggestion="Populate list items or remove the empty list."
                    ))
        if empty_count > 5:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_3_1.value,
                severity=IssueSeverity.CRITICAL,
                element="<ul>/<ol> aggregate",
                message=f"{empty_count - 5} additional empty lists (suppressed).",
                suggestion="Populate or remove empty lists."
            ))

    def _check_empty_doc_chapters(self, soup: BeautifulSoup) -> None:
        """SC 1.3.1 — doc-chapter / doc-abstract containers must have content.

        A ``<article role="doc-chapter">`` wrapper that contains only a
        heading and no body text is structurally meaningless for
        assistive tech — declares a chapter landmark then provides
        nothing.
        """
        structural_roles = ('doc-chapter', 'doc-part', 'doc-abstract',
                            'doc-preface', 'doc-introduction')
        empty_count = 0
        for role in structural_roles:
            for el in soup.find_all(attrs={'role': role}):
                # Count body words after subtracting heading text.
                for heading in el.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    heading.extract()
                body_text = el.get_text(separator=' ', strip=True)
                body_words = len(body_text.split()) if body_text else 0
                if body_words < 20:
                    empty_count += 1
                    if empty_count <= 5:
                        self.issues.append(WCAGIssue(
                            criterion=WCAGCriterion.SC_1_3_1.value,
                            severity=IssueSeverity.CRITICAL,
                            element=f'<{el.name} role="{role}">',
                            message=(
                                f"Structural landmark role=\"{role}\" contains "
                                f"only {body_words} words of body content."
                            ),
                            suggestion=(
                                "Populate the landmark with real chapter body "
                                "content, or drop the role if the section is "
                                "empty."
                            )
                        ))
        if empty_count > 5:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_3_1.value,
                severity=IssueSeverity.CRITICAL,
                element="structural landmarks aggregate",
                message=f"{empty_count - 5} additional empty structural landmarks (suppressed)."
            ))

    def _check_toc_anchor_resolution(self, soup: BeautifulSoup) -> None:
        """SC 2.4.1 / 2.4.5 — TOC anchors must resolve to real IDs.

        Pre-Wave-31 we silently shipped TOCs where half the anchors
        pointed at IDs that don't exist in the document. Wave 31
        verifies each fragment anchor resolves to an ``id=`` in the
        same document.

        Severity rule: if ≥ 10% of TOC anchors are dead → CRITICAL;
        otherwise per-dead-link HIGH.
        """
        toc_navs = soup.find_all('nav', attrs={'role': 'doc-toc'})
        if not toc_navs:
            # Fall back to <nav class="toc"> / <nav id*=toc>.
            toc_navs = [
                nav for nav in soup.find_all('nav')
                if 'toc' in ' '.join(nav.get('class', [])).lower()
                or 'toc' in (nav.get('id') or '').lower()
            ]
        if not toc_navs:
            return

        all_ids = {el.get('id') for el in soup.find_all(attrs={'id': True})}
        for nav in toc_navs:
            anchors = nav.find_all('a', href=True)
            frag_anchors = [a for a in anchors if a['href'].startswith('#')]
            if not frag_anchors:
                continue
            dead = []
            for a in frag_anchors:
                target_id = a['href'][1:]
                if target_id and target_id not in all_ids:
                    dead.append(target_id)
            total = len(frag_anchors)
            if not dead:
                continue
            frac_dead = len(dead) / total
            # Report aggregate first.
            if frac_dead >= 0.10:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_4_1.value,
                    severity=IssueSeverity.CRITICAL,
                    element='<nav role="doc-toc">',
                    message=(
                        f"{len(dead)} of {total} TOC anchors are dead "
                        f"({frac_dead:.0%}) — bypass-blocks target IDs don't exist."
                    ),
                    suggestion=(
                        "Ensure every TOC link target has a matching id=\"...\" "
                        "in the document body."
                    )
                ))
            else:
                # Small number of dead anchors — HIGH each (up to 3).
                for target in dead[:3]:
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_2_4_5.value,
                        severity=IssueSeverity.HIGH,
                        element='<a href>',
                        message=f"Dead TOC anchor: href=#{target} does not resolve.",
                        suggestion="Remove or fix the anchor."
                    ))

    def _check_pdf_artifact_headings(self, soup: BeautifulSoup) -> None:
        """SC 2.4.6 — headings should not carry PDF page-number artifacts.

        pdftotext output often fuses a page number to the end of a
        heading line: ``"Chapter 3 47"`` or ``"Preface viii"``. When
        those land in an ``<h2>`` / ``<h3>`` they read awkwardly in
        the document outline (the page number isn't meaningful in
        HTML).
        """
        artifact_re = re.compile(
            r'^(.{2,}?)\s+([0-9]{1,4}|[ivxlcdm]+)$',
            re.IGNORECASE,
        )
        flagged = 0
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text(strip=True)
            if len(text) < 5:
                continue
            m = artifact_re.match(text)
            if not m:
                continue
            # Skip if the heading is legitimately numbered
            # ("Chapter 3", "Section 2.1") — those have no trailing
            # page-number token beyond the leading chapter number.
            # The artifact pattern requires TWO tokens + trailing number,
            # e.g. "Chapter 3 47" has prefix "Chapter 3" and suffix 47.
            prefix = m.group(1).strip()
            if len(prefix.split()) < 1:
                continue
            # Chapter/Preface/Appendix with ONLY a trailing number
            # (e.g. "Chapter 3") are legit — needs ≥2 prefix tokens OR
            # a mix of letters and digits in the prefix.
            if len(prefix.split()) < 2 and not re.search(r'[a-z]', prefix, re.IGNORECASE):
                continue
            # Additional filter: skip pure numeric headings.
            if prefix.isdigit():
                continue
            flagged += 1
            if flagged <= 3:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_4_6.value,
                    severity=IssueSeverity.MEDIUM,
                    element=f'<{heading.name}>',
                    message=f"Heading looks like PDF-extraction artifact: {text!r}",
                    suggestion="Strip trailing page numbers from heading text."
                ))

    # ------------------------------------------------------------------ #
    # Report generation
    # ------------------------------------------------------------------ #

    def _generate_report(self, file_path: str) -> ValidationReport:
        """Generate validation report from collected issues"""
        report = ValidationReport(
            file_path=file_path,
            timestamp=datetime.now().isoformat(),
            total_issues=len(self.issues),
            issues=self.issues
        )

        # Count by severity
        for issue in self.issues:
            if issue.severity == IssueSeverity.CRITICAL:
                report.critical_count += 1
            elif issue.severity == IssueSeverity.HIGH:
                report.high_count += 1
            elif issue.severity == IssueSeverity.MEDIUM:
                report.medium_count += 1
            else:
                report.low_count += 1

        # Determine WCAG AA compliance
        report.wcag_aa_compliant = (
            report.critical_count == 0 and
            (report.high_count == 0 or not self.strict_mode)
        )

        # Summary by criterion
        for issue in self.issues:
            criterion = issue.criterion
            report.summary[criterion] = report.summary.get(criterion, 0) + 1

        return report


def validate_html_wcag(html: str, strict: bool = False) -> ValidationReport:
    """
    Convenience function to validate HTML for WCAG 2.2 AA compliance.

    Args:
        html: HTML content string
        strict: If True, treat high severity issues as failures

    Returns:
        ValidationReport with all issues found
    """
    validator = WCAGValidator(strict_mode=strict)
    return validator.validate(html)


def validate_html_file(file_path: str, strict: bool = False) -> ValidationReport:
    """
    Convenience function to validate an HTML file for WCAG 2.2 AA compliance.

    Args:
        file_path: Path to HTML file
        strict: If True, treat high severity issues as failures

    Returns:
        ValidationReport with all issues found
    """
    validator = WCAGValidator(strict_mode=strict)
    return validator.validate_file(Path(file_path))


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Validate HTML content for WCAG 2.2 AA accessibility compliance'
    )
    parser.add_argument(
        'input',
        help='Input HTML file to validate'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for report (default: stdout)'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict mode: treat high severity as failures'
    )

    args = parser.parse_args()

    validator = WCAGValidator(strict_mode=args.strict)
    report = validator.validate_file(Path(args.input))

    if args.format == 'json':
        output = report.to_json()
    else:
        output = report.to_text()

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Report written to: {args.output}")
    else:
        print(output)

    # Exit with non-zero if not compliant
    sys.exit(0 if report.wcag_aa_compliant else 1)
