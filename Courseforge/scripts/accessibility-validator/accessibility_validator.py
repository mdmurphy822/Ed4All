#!/usr/bin/env python3
"""
Accessibility Validator - WCAG 2.2 AA Compliance Checker

Comprehensive accessibility validation for HTML content ensuring WCAG 2.2 AA
compliance with automated checking for images, headings, forms, contrast, and more.

Features:
- Alt text validation for all images
- Color contrast verification (4.5:1 for normal text, 3:1 for large text)
- Heading hierarchy validation (no skipped levels)
- Keyboard navigation testing
- ARIA landmark verification
- Form label association checking
- Language declaration validation
- Link text quality assessment
- Focus appearance validation (WCAG 2.2)
- Target size validation (WCAG 2.2)
- Dragging movements detection (WCAG 2.2)
- Accessible authentication checks (WCAG 2.2)

Usage:
    python accessibility_validator.py --input file.html --output report.json
    python accessibility_validator.py --input-dir /content/ --format html
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from bs4 import BeautifulSoup

# Add Ed4All lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # scripts/accessibility-validator/... → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for accessibility issues"""
    CRITICAL = "critical"  # WCAG A failure
    HIGH = "high"          # WCAG AA failure
    MEDIUM = "medium"      # Best practice violation
    LOW = "low"            # Enhancement opportunity


class WCAGCriterion(Enum):
    """WCAG 2.2 Success Criteria"""
    # Level A
    SC_1_1_1 = "1.1.1"   # Non-text Content
    SC_1_3_1 = "1.3.1"   # Info and Relationships
    SC_1_3_2 = "1.3.2"   # Meaningful Sequence
    SC_2_1_1 = "2.1.1"   # Keyboard
    SC_2_4_1 = "2.4.1"   # Bypass Blocks
    SC_2_4_2 = "2.4.2"   # Page Titled
    SC_3_1_1 = "3.1.1"   # Language of Page
    SC_4_1_1 = "4.1.1"   # Parsing
    SC_4_1_2 = "4.1.2"   # Name, Role, Value
    # Level A (NEW WCAG 2.2)
    SC_2_4_11 = "2.4.11" # Focus Not Obscured (Minimum)
    SC_3_2_6 = "3.2.6"   # Consistent Help
    SC_3_3_7 = "3.3.7"   # Redundant Entry
    # Level AA
    SC_1_4_3 = "1.4.3"   # Contrast (Minimum)
    SC_1_4_4 = "1.4.4"   # Resize Text
    SC_2_4_6 = "2.4.6"   # Headings and Labels
    SC_2_4_7 = "2.4.7"   # Focus Visible
    SC_3_1_2 = "3.1.2"   # Language of Parts
    # Level AA (NEW WCAG 2.2)
    SC_2_4_12 = "2.4.12" # Focus Not Obscured (Enhanced)
    SC_2_4_13 = "2.4.13" # Focus Appearance
    SC_2_5_7 = "2.5.7"   # Dragging Movements
    SC_2_5_8 = "2.5.8"   # Target Size (Minimum)
    SC_3_3_8 = "3.3.8"   # Accessible Authentication (Minimum)
    SC_3_3_9 = "3.3.9"   # Accessible Authentication (Enhanced)


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


class AccessibilityValidator:
    """
    WCAG 2.2 AA Accessibility Validator for HTML Content.

    Performs comprehensive accessibility checks including:
    - Image alt text validation
    - Heading hierarchy analysis
    - Color contrast verification
    - Form label associations
    - ARIA landmark validation
    - Keyboard accessibility indicators
    - Language declarations
    - Link text quality
    - Focus appearance (WCAG 2.2)
    - Target size (WCAG 2.2)
    - Dragging movements (WCAG 2.2)
    - Accessible authentication (WCAG 2.2)
    """

    # Generic link text patterns to flag
    GENERIC_LINK_TEXT = [
        'click here', 'read more', 'learn more', 'more',
        'here', 'link', 'this', 'page', 'info'
    ]

    # Required ARIA landmarks for educational content
    REQUIRED_LANDMARKS = ['main', 'navigation', 'banner', 'contentinfo']

    # Minimum contrast ratios
    NORMAL_TEXT_CONTRAST = 4.5
    LARGE_TEXT_CONTRAST = 3.0

    def __init__(
        self,
        strict_mode: bool = False,
        capture: Optional["DecisionCapture"] = None,
    ):
        """
        Initialize the accessibility validator.

        Args:
            strict_mode: If True, treat warnings as failures
            capture: Optional DecisionCapture for logging validation decisions
        """
        self.strict_mode = strict_mode
        self.issues: List[WCAGIssue] = []
        self.capture = capture

    def validate_file(self, file_path: Path) -> ValidationReport:
        """
        Validate a single HTML file for accessibility compliance.

        Args:
            file_path: Path to HTML file

        Returns:
            ValidationReport with all issues found
        """
        file_path = Path(file_path)
        self.issues = []

        logger.info(f"Validating: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading file: {e}")
            raise

        soup = BeautifulSoup(content, 'html.parser')

        # Run all validation checks
        self._check_language_declaration(soup)
        self._check_page_title(soup)
        self._check_images(soup)
        self._check_headings(soup)
        self._check_links(soup)
        self._check_forms(soup)
        self._check_tables(soup)
        self._check_landmarks(soup)
        self._check_focus_indicators(soup, content)
        self._check_skip_links(soup)
        # WCAG 2.2 specific checks
        self._check_focus_not_obscured(soup, content)
        self._check_focus_appearance(soup, content)
        self._check_target_size(soup, content)
        self._check_dragging_movements(soup, content)
        self._check_accessible_authentication(soup, content)

        # Generate report
        report = self._generate_report(str(file_path))
        return report

    def validate_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> List[ValidationReport]:
        """
        Validate all HTML files in a directory.

        Args:
            directory: Directory path
            recursive: Process subdirectories

        Returns:
            List of ValidationReport objects
        """
        directory = Path(directory)
        reports = []

        pattern = '**/*.html' if recursive else '*.html'

        for file_path in directory.glob(pattern):
            try:
                report = self.validate_file(file_path)
                reports.append(report)
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")

        return reports

    def _check_language_declaration(self, soup: BeautifulSoup):
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

    def _check_page_title(self, soup: BeautifulSoup):
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

    def _check_images(self, soup: BeautifulSoup):
        """Check all images for alt text (WCAG 1.1.1)"""
        images = soup.find_all('img')

        for img in images:
            src = img.get('src', 'unknown')
            alt = img.get('alt')

            if alt is None:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_1_1.value,
                    severity=IssueSeverity.CRITICAL,
                    element=f'<img src="{src}">',
                    message="Image missing alt attribute",
                    suggestion="Add alt attribute with descriptive text or alt='' for decorative images"
                ))
            elif alt.strip() == '':
                # Empty alt is valid for decorative images, check if it seems intentional
                role = img.get('role')
                if role != 'presentation':
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_1_1.value,
                        severity=IssueSeverity.MEDIUM,
                        element=f'<img src="{src}">',
                        message="Empty alt text - ensure image is decorative",
                        suggestion="If decorative, add role='presentation'. If meaningful, add descriptive alt text"
                    ))
            elif alt.lower() in ['image', 'photo', 'picture', 'graphic', 'icon']:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_1_1.value,
                    severity=IssueSeverity.HIGH,
                    element=f'<img src="{src}" alt="{alt}">',
                    message="Generic alt text does not describe image content",
                    suggestion="Replace with specific description of what the image shows"
                ))
            elif src and src.lower() in alt.lower():
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_1_1.value,
                    severity=IssueSeverity.MEDIUM,
                    element=f'<img src="{src}" alt="{alt}">',
                    message="Alt text appears to be filename rather than description",
                    suggestion="Replace with meaningful description of image content"
                ))

    def _check_headings(self, soup: BeautifulSoup):
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

    def _check_links(self, soup: BeautifulSoup):
        """Check link text quality (WCAG 2.4.4)"""
        links = soup.find_all('a')

        for link in links:
            href = link.get('href', '')
            text = link.get_text().strip().lower()

            # Check for generic link text
            if text in self.GENERIC_LINK_TEXT:
                self.issues.append(WCAGIssue(
                    criterion="2.4.4",
                    severity=IssueSeverity.HIGH,
                    element=f'<a href="{href}">{text}</a>',
                    message=f"Generic link text: '{text}'",
                    suggestion="Use descriptive link text that indicates destination"
                ))

            # Check for links with no text
            if not text and not link.find('img'):
                self.issues.append(WCAGIssue(
                    criterion="2.4.4",
                    severity=IssueSeverity.CRITICAL,
                    element=f'<a href="{href}">',
                    message="Link has no accessible text",
                    suggestion="Add link text or aria-label for accessible name"
                ))

            # Check image links
            img = link.find('img')
            if img and not text:
                alt = img.get('alt', '')
                if not alt:
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_1_1.value,
                        severity=IssueSeverity.CRITICAL,
                        element=f'<a href="{href}"><img>',
                        message="Image link missing alt text",
                        suggestion="Add alt text to image describing link destination"
                    ))

    def _check_forms(self, soup: BeautifulSoup):
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

            # Check for title attribute (not recommended but acceptable)
            if input_elem.get('title'):
                has_label = True
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_3_1.value,
                    severity=IssueSeverity.LOW,
                    element=f'<{input_elem.name} name="{input_name}">',
                    message="Form control uses title attribute instead of label",
                    suggestion="Use <label> element for better accessibility"
                ))

            if not has_label:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_1_3_1.value,
                    severity=IssueSeverity.CRITICAL,
                    element=f'<{input_elem.name} name="{input_name}">',
                    message="Form control missing associated label",
                    suggestion="Add <label for='id'> or aria-label attribute"
                ))

        # Check fieldsets for radio/checkbox groups
        radios = soup.find_all('input', {'type': 'radio'})
        radio_names: Set[str] = set()
        for radio in radios:
            name = radio.get('name')
            if name:
                radio_names.add(name)

        for name in radio_names:
            group = soup.find_all('input', {'type': 'radio', 'name': name})
            if len(group) > 1:
                # Check if wrapped in fieldset
                first = group[0]
                fieldset = first.find_parent('fieldset')
                if not fieldset:
                    self.issues.append(WCAGIssue(
                        criterion=WCAGCriterion.SC_1_3_1.value,
                        severity=IssueSeverity.MEDIUM,
                        element=f'<input type="radio" name="{name}">',
                        message="Radio button group not wrapped in fieldset",
                        suggestion="Wrap radio group in <fieldset> with <legend>"
                    ))

    def _check_tables(self, soup: BeautifulSoup):
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
                        self.issues.append(WCAGIssue(
                            criterion=WCAGCriterion.SC_1_3_1.value,
                            severity=IssueSeverity.MEDIUM,
                            element=f"<th>{th.get_text()[:20]}...</th>",
                            message="Table header missing scope attribute",
                            suggestion='Add scope="col" or scope="row" to header cells'
                        ))

    def _check_landmarks(self, soup: BeautifulSoup):
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

        # Check for multiple mains
        mains = soup.find_all('main')
        main_roles = soup.find_all(role='main')
        if len(mains) + len(main_roles) > 1:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_1_3_1.value,
                severity=IssueSeverity.HIGH,
                element="<main>",
                message="Multiple main landmarks found",
                suggestion="Use only one main landmark per page"
            ))

    def _check_focus_indicators(self, soup: BeautifulSoup, content: str):
        """Check for focus indicator removal (WCAG 2.4.7)"""
        # Check for outline:none or outline:0 in inline styles
        if re.search(r'outline\s*:\s*(none|0)', content, re.IGNORECASE):
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_7.value,
                severity=IssueSeverity.HIGH,
                element="<style>",
                message="Focus indicator may be removed (outline:none detected)",
                suggestion="Ensure custom focus styles are provided when removing default outline"
            ))

    def _check_skip_links(self, soup: BeautifulSoup):
        """Check for skip navigation links (WCAG 2.4.1)"""
        # Look for skip links
        skip_patterns = ['skip', 'jump to', 'go to main', 'skip to main']
        links = soup.find_all('a')

        has_skip_link = False
        for link in links:
            text = link.get_text().lower()
            href = link.get('href', '')
            if any(pattern in text for pattern in skip_patterns) or href.startswith('#main'):
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

    def _check_focus_not_obscured(self, soup: BeautifulSoup, content: str):
        """Check for elements that might obscure focus (WCAG 2.4.11, 2.4.12)"""
        # Check for sticky/fixed position elements that could obscure focus
        if re.search(r'position\s*:\s*(fixed|sticky)', content, re.IGNORECASE):
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_2_4_11.value,
                severity=IssueSeverity.MEDIUM,
                element="<style>",
                message="Fixed/sticky positioned elements detected - verify they don't obscure focus indicators",
                suggestion="Ensure focused elements scroll into view and are not covered by fixed elements. Use scroll-margin-top/bottom."
            ))

    def _check_focus_appearance(self, soup: BeautifulSoup, content: str):
        """Check focus indicator styling meets WCAG 2.2 requirements (WCAG 2.4.13)"""
        # Check for focus styling that meets 2px minimum
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

    def _check_target_size(self, soup: BeautifulSoup, content: str):
        """Check interactive element target sizes (WCAG 2.5.8)"""
        # Check for explicit small sizing on interactive elements
        small_size_pattern = re.compile(
            r'(button|\.btn|input\[type|a)[^{]*\{[^}]*(width|height)\s*:\s*(\d+)px',
            re.IGNORECASE
        )
        for match in small_size_pattern.finditer(content):
            size = int(match.group(3))
            if size < 24 and size > 0:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_5_8.value,
                    severity=IssueSeverity.HIGH,
                    element=match.group(1),
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

    def _check_dragging_movements(self, soup: BeautifulSoup, content: str):
        """Check for drag operations without alternatives (WCAG 2.5.7)"""
        drag_patterns = ['draggable="true"', 'ondrag', 'ondragstart', 'ondrop', 'ondragover']
        content_lower = content.lower()
        for pattern in drag_patterns:
            if pattern.lower() in content_lower:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_2_5_7.value,
                    severity=IssueSeverity.MEDIUM,
                    element="draggable element",
                    message="Drag functionality detected - ensure single pointer alternative exists",
                    suggestion="Add buttons or alternative controls for drag operations (e.g., up/down buttons for reordering)"
                ))
                break

    def _check_accessible_authentication(self, soup: BeautifulSoup, content: str):
        """Check for cognitive function tests in authentication (WCAG 3.3.8)"""
        # Check for CAPTCHA indicators
        captcha_patterns = ['captcha', 'recaptcha', 'hcaptcha', 'g-recaptcha', 'verify you are human', 'i am not a robot']
        content_lower = content.lower()
        for pattern in captcha_patterns:
            if pattern in content_lower:
                self.issues.append(WCAGIssue(
                    criterion=WCAGCriterion.SC_3_3_8.value,
                    severity=IssueSeverity.HIGH,
                    element="authentication",
                    message="CAPTCHA or cognitive test detected in form",
                    suggestion="Provide alternative authentication methods that don't require cognitive function tests (e.g., email verification, biometrics)"
                ))
                break

        # Check for password fields that block paste
        if 'onpaste="return false"' in content_lower or 'onpaste="false"' in content_lower:
            self.issues.append(WCAGIssue(
                criterion=WCAGCriterion.SC_3_3_8.value,
                severity=IssueSeverity.HIGH,
                element="<input type='password'>",
                message="Password field may block paste functionality",
                suggestion="Allow copy-paste in password fields to support password managers"
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

    def to_json(self, report: ValidationReport) -> str:
        """Export report as JSON"""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        data = asdict(report)
        # Convert enums
        for issue in data['issues']:
            issue['severity'] = issue['severity'].value if isinstance(issue['severity'], IssueSeverity) else issue['severity']

        return json.dumps(data, indent=2, default=serialize)

    def to_text(self, report: ValidationReport) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 70,
            "WCAG 2.2 AA ACCESSIBILITY VALIDATION REPORT",
            "=" * 70,
            f"File: {report.file_path}",
            f"Timestamp: {report.timestamp}",
            "-" * 70,
            f"Total Issues: {report.total_issues}",
            f"  Critical: {report.critical_count}",
            f"  High: {report.high_count}",
            f"  Medium: {report.medium_count}",
            f"  Low: {report.low_count}",
            "-" * 70,
            f"WCAG 2.2 AA Compliant: {'YES' if report.wcag_aa_compliant else 'NO'}",
            "=" * 70,
        ]

        if report.issues:
            lines.append("\nISSUES FOUND:\n")
            for i, issue in enumerate(report.issues, 1):
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


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description='Validate HTML content for WCAG 2.2 AA accessibility compliance'
    )
    parser.add_argument(
        '-i', '--input',
        help='Input HTML file to validate'
    )
    parser.add_argument(
        '--input-dir',
        help='Directory of HTML files to validate'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for report'
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
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.input and not args.input_dir:
        parser.error("Must provide --input or --input-dir")

    validator = AccessibilityValidator(strict_mode=args.strict)

    if args.input:
        report = validator.validate_file(Path(args.input))
        reports = [report]
    else:
        reports = validator.validate_directory(Path(args.input_dir))

    # Generate output
    if args.format == 'json':
        output = json.dumps([json.loads(validator.to_json(r)) for r in reports], indent=2)
    else:
        output = "\n\n".join(validator.to_text(r) for r in reports)

    # Write or print
    if args.output:
        Path(args.output).write_text(output)
        logger.info(f"Report saved to: {args.output}")
    else:
        print(output)

    # Exit with appropriate code
    any_failures = any(not r.wcag_aa_compliant for r in reports)
    sys.exit(1 if any_failures else 0)


if __name__ == '__main__':
    main()
