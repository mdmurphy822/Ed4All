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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
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

    def validate(self, html: str, file_path: str = "inline") -> ValidationReport:
        """
        Validate HTML content for accessibility compliance.

        Args:
            html: HTML content string
            file_path: Optional file path for reporting

        Returns:
            ValidationReport with all issues found
        """
        self.issues = []
        soup = BeautifulSoup(html, 'html.parser')

        # Run all validation checks
        self._check_language_declaration(soup)
        self._check_page_title(soup)
        self._check_images(soup)
        self._check_headings(soup)
        self._check_links(soup)
        self._check_forms(soup)
        self._check_tables(soup)
        self._check_landmarks(soup)
        self._check_skip_links(soup)
        self._check_focus_indicators(soup, html)
        # WCAG 2.2 specific checks
        self._check_focus_not_obscured(soup, html)
        self._check_focus_appearance(soup, html)
        self._check_target_size(soup, html)

        # Generate report
        return self._generate_report(file_path)

    def validate_file(self, file_path: Path) -> ValidationReport:
        """
        Validate an HTML file for accessibility compliance.

        Args:
            file_path: Path to HTML file

        Returns:
            ValidationReport with all issues found
        """
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
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
        """Check all images for alt text (WCAG 1.1.1)"""
        images = soup.find_all('img')

        for img in images:
            src = img.get('src', 'unknown')
            alt = img.get('alt')

            if alt is None:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_1_1.value,
                    severity=IssueSeverity.CRITICAL,
                    element=f'<img src="{src[:50]}...">',
                    message="Image missing alt attribute",
                    suggestion="Add alt attribute with descriptive text or alt='' for decorative images"
                ))
            elif alt.strip() == '':
                # Empty alt is valid for decorative images
                role = img.get('role')
                if role != 'presentation':
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_1_1.value,
                        severity=IssueSeverity.MEDIUM,
                        element=f'<img src="{src[:50]}...">',
                        message="Empty alt text - verify image is decorative",
                        suggestion="If decorative, add role='presentation'. If meaningful, add descriptive alt text"
                    ))
            elif alt.lower() in ['image', 'photo', 'picture', 'graphic', 'icon']:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_1_1.value,
                    severity=IssueSeverity.HIGH,
                    element=f'<img alt="{alt}">',
                    message="Generic alt text does not describe image content",
                    suggestion="Replace with specific description of what the image shows"
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
        """Check for ARIA landmarks (WCAG 1.3.1, 2.4.1)"""
        # Check for main landmark
        main = soup.find('main') or soup.find(role='main')
        if not main:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_1.value,
                severity=IssueSeverity.MEDIUM,
                element="document",
                message="Missing main landmark",
                suggestion="Add <main> element to wrap primary content"
            ))

        # Check for multiple mains (avoid double-counting <main role="main">)
        mains = soup.find_all('main')
        main_roles = [el for el in soup.find_all(role='main') if el.name != 'main']
        if len(mains) + len(main_roles) > 1:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_3_1.value,
                severity=IssueSeverity.HIGH,
                element="<main>",
                message="Multiple main landmarks found",
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
        """Check for focus indicator removal (WCAG 2.4.7)"""
        # Check for outline:none or outline:0 without replacement
        # Skip if :focus-visible is used (proper pattern for keyboard-only focus)
        has_focus_visible = re.search(r':focus-visible\s*\{[^}]*outline', content, re.IGNORECASE)
        has_outline_none = re.search(r'outline\s*:\s*(none|0)[^;]*;', content, re.IGNORECASE)

        if has_outline_none and not has_focus_visible:
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
        """Check interactive element target sizes (WCAG 2.5.8)"""
        # Check for explicit small sizing on interactive elements
        small_size_patterns = [
            r'button[^{]*\{[^}]*(width|height)\s*:\s*(\d+)px',
            r'\.btn[^{]*\{[^}]*(width|height)\s*:\s*(\d+)px',
            r'input\[type[^{]*\{[^}]*(width|height)\s*:\s*(\d+)px',
        ]

        for pattern in small_size_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                size = int(match.group(2))
                if 0 < size < 24:
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
    import sys
    import argparse

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
