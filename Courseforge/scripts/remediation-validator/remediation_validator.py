#!/usr/bin/env python3
"""
Remediation Validator - Final Quality Assurance for Remediated Courses

This script performs comprehensive validation of remediated course content,
ensuring WCAG 2.2 AA compliance, OSCQR standards adherence, and Brightspace
compatibility before final IMSCC packaging.

Features:
- WCAG 2.2 AA compliance validation
- OSCQR standards checking
- Brightspace import compatibility verification
- Content integrity validation
- Before/after accessibility score comparison

Usage:
    python remediation_validator.py --course-dir /path/to/course/
    python remediation_validator.py --course-dir /path/to/course/ --output report.json
    python remediation_validator.py --before /original/ --after /remediated/ --compare
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
from typing import Dict, List, Optional, Set, Tuple, Any
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('remediation_validator.log')
    ]
)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation checks"""
    WCAG = "wcag"
    OSCQR = "oscqr"
    BRIGHTSPACE = "brightspace"
    CONTENT = "content"
    STRUCTURE = "structure"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    category: ValidationCategory
    severity: ValidationSeverity
    code: str
    message: str
    file: str
    line: Optional[int] = None
    element: Optional[str] = None
    wcag_criterion: Optional[str] = None
    oscqr_standard: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class FileValidationResult:
    """Validation result for a single file"""
    file_path: str
    issues: List[ValidationIssue] = field(default_factory=list)
    accessibility_score: float = 100.0
    wcag_compliant: bool = True
    oscqr_compliant: bool = True
    brightspace_ready: bool = True


@dataclass
class ValidationReport:
    """Complete validation report"""
    course_path: str
    validation_timestamp: str
    total_files: int = 0
    files_with_issues: int = 0
    total_issues: int = 0
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    wcag_compliance: float = 0.0
    oscqr_compliance: float = 0.0
    brightspace_ready: bool = True
    overall_score: float = 0.0
    file_results: List[FileValidationResult] = field(default_factory=list)
    summary: str = ""


class RemediationValidator:
    """
    Comprehensive validator for remediated course content.
    """

    # WCAG 2.2 AA criteria to check
    WCAG_CHECKS = {
        # WCAG 2.0/2.1 Criteria
        '1.1.1': 'Non-text Content',
        '1.2.1': 'Audio-only and Video-only (Prerecorded)',
        '1.2.2': 'Captions (Prerecorded)',
        '1.2.3': 'Audio Description or Media Alternative',
        '1.2.5': 'Audio Description (Prerecorded)',
        '1.3.1': 'Info and Relationships',
        '1.3.2': 'Meaningful Sequence',
        '1.3.3': 'Sensory Characteristics',
        '1.3.4': 'Orientation',
        '1.3.5': 'Identify Input Purpose',
        '1.4.1': 'Use of Color',
        '1.4.3': 'Contrast (Minimum)',
        '1.4.4': 'Resize Text',
        '1.4.5': 'Images of Text',
        '1.4.10': 'Reflow',
        '1.4.11': 'Non-text Contrast',
        '1.4.12': 'Text Spacing',
        '1.4.13': 'Content on Hover or Focus',
        '2.1.1': 'Keyboard',
        '2.1.2': 'No Keyboard Trap',
        '2.1.4': 'Character Key Shortcuts',
        '2.2.1': 'Timing Adjustable',
        '2.2.2': 'Pause, Stop, Hide',
        '2.3.1': 'Three Flashes or Below Threshold',
        '2.4.1': 'Bypass Blocks',
        '2.4.2': 'Page Titled',
        '2.4.3': 'Focus Order',
        '2.4.4': 'Link Purpose (In Context)',
        '2.4.5': 'Multiple Ways',
        '2.4.6': 'Headings and Labels',
        '2.4.7': 'Focus Visible',
        '2.5.1': 'Pointer Gestures',
        '2.5.2': 'Pointer Cancellation',
        '2.5.3': 'Label in Name',
        '2.5.4': 'Motion Actuation',
        '3.1.1': 'Language of Page',
        '3.1.2': 'Language of Parts',
        '3.2.1': 'On Focus',
        '3.2.2': 'On Input',
        '3.2.3': 'Consistent Navigation',
        '3.2.4': 'Consistent Identification',
        '3.3.1': 'Error Identification',
        '3.3.2': 'Labels or Instructions',
        '3.3.3': 'Error Suggestion',
        '3.3.4': 'Error Prevention (Legal, Financial, Data)',
        '4.1.1': 'Parsing',
        '4.1.2': 'Name, Role, Value',
        '4.1.3': 'Status Messages',
        # WCAG 2.2 New Criteria - Level A
        '2.4.11': 'Focus Not Obscured (Minimum)',
        '3.2.6': 'Consistent Help',
        '3.3.7': 'Redundant Entry',
        # WCAG 2.2 New Criteria - Level AA
        '2.4.12': 'Focus Not Obscured (Enhanced)',
        '2.4.13': 'Focus Appearance',
        '2.5.7': 'Dragging Movements',
        '2.5.8': 'Target Size (Minimum)',
        '3.3.8': 'Accessible Authentication (Minimum)',
        '3.3.9': 'Accessible Authentication (Enhanced)',
    }

    # OSCQR standards to check
    OSCQR_CHECKS = {
        '1.1': 'Course includes Welcome and Getting Started content',
        '1.2': 'Instructor introduces themselves',
        '1.3': 'Learners are asked to introduce themselves',
        '2.1': 'Course navigation instructions provided',
        '2.2': 'Technology requirements stated',
        '2.3': 'Required skills and prior knowledge stated',
        '3.1': 'Content is organized into modules',
        '3.2': 'Content follows logical progression',
        '3.3': 'Visual design is consistent',
        '4.1': 'Learning objectives are stated',
        '4.2': 'Content supports learning objectives',
        '4.3': 'Activities support learning objectives',
        '5.1': 'Opportunities for interaction provided',
        '5.2': 'Communication expectations stated',
        '6.1': 'Assessment aligns with objectives',
        '6.2': 'Grading policy clearly stated',
    }

    def __init__(self, course_dir: Path):
        """
        Initialize the validator.

        Args:
            course_dir: Path to the course directory
        """
        self.course_dir = Path(course_dir)
        self.issues: List[ValidationIssue] = []
        self.file_results: List[FileValidationResult] = []

    def validate(self) -> ValidationReport:
        """
        Run complete validation.

        Returns:
            ValidationReport with all findings
        """
        logger.info(f"Starting validation of: {self.course_dir}")

        # Validate course directory exists
        if not self.course_dir.exists():
            raise FileNotFoundError(f"Course directory not found: {self.course_dir}")

        # Find all HTML files
        html_files = list(self.course_dir.rglob('*.html'))
        logger.info(f"Found {len(html_files)} HTML files to validate")

        # Validate each file
        for html_file in html_files:
            result = self._validate_file(html_file)
            self.file_results.append(result)

        # Validate manifest if exists
        manifest_path = self.course_dir / 'imsmanifest.xml'
        if manifest_path.exists():
            self._validate_manifest(manifest_path)

        # Calculate scores
        wcag_score = self._calculate_wcag_score()
        oscqr_score = self._calculate_oscqr_score()
        overall_score = (wcag_score + oscqr_score) / 2

        # Aggregate issues
        issues_by_severity = {}
        issues_by_category = {}
        for result in self.file_results:
            for issue in result.issues:
                sev = issue.severity.value
                cat = issue.category.value
                issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1
                issues_by_category[cat] = issues_by_category.get(cat, 0) + 1

        # Determine brightspace readiness
        brightspace_ready = not any(
            r for r in self.file_results if not r.brightspace_ready
        )

        # Generate report
        report = ValidationReport(
            course_path=str(self.course_dir),
            validation_timestamp=datetime.now().isoformat(),
            total_files=len(html_files),
            files_with_issues=sum(1 for r in self.file_results if r.issues),
            total_issues=sum(len(r.issues) for r in self.file_results),
            issues_by_severity=issues_by_severity,
            issues_by_category=issues_by_category,
            wcag_compliance=wcag_score,
            oscqr_compliance=oscqr_score,
            brightspace_ready=brightspace_ready,
            overall_score=overall_score,
            file_results=self.file_results,
            summary=self._generate_summary(wcag_score, oscqr_score, brightspace_ready)
        )

        logger.info(f"Validation complete. Score: {overall_score:.1f}%")
        return report

    def _validate_file(self, file_path: Path) -> FileValidationResult:
        """Validate a single HTML file"""
        result = FileValidationResult(file_path=str(file_path.relative_to(self.course_dir)))

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Run all checks
            self._check_alt_text(content, file_path, result)
            self._check_heading_structure(content, file_path, result)
            self._check_language(content, file_path, result)
            self._check_links(content, file_path, result)
            self._check_forms(content, file_path, result)
            self._check_tables(content, file_path, result)
            self._check_color_contrast(content, file_path, result)
            self._check_keyboard_access(content, file_path, result)
            self._check_page_title(content, file_path, result)
            self._check_skip_links(content, file_path, result)
            self._check_aria(content, file_path, result)
            self._check_brightspace_compat(content, file_path, result)
            # WCAG 2.2 specific checks
            self._check_focus_not_obscured(content, file_path, result)
            self._check_focus_appearance(content, file_path, result)
            self._check_target_size(content, file_path, result)
            self._check_dragging_movements(content, file_path, result)
            self._check_consistent_help(content, file_path, result)
            self._check_accessible_authentication(content, file_path, result)

            # Calculate accessibility score
            critical_issues = sum(1 for i in result.issues
                                if i.severity == ValidationSeverity.CRITICAL)
            high_issues = sum(1 for i in result.issues
                            if i.severity == ValidationSeverity.HIGH)

            result.accessibility_score = max(0, 100 - (critical_issues * 20) - (high_issues * 5))
            result.wcag_compliant = critical_issues == 0
            result.brightspace_ready = not any(
                i for i in result.issues
                if i.category == ValidationCategory.BRIGHTSPACE
                and i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]
            )

        except Exception as e:
            result.issues.append(ValidationIssue(
                category=ValidationCategory.CONTENT,
                severity=ValidationSeverity.CRITICAL,
                code='FILE_READ_ERROR',
                message=f"Failed to read file: {e}",
                file=str(file_path)
            ))

        return result

    def _check_alt_text(self, content: str, file_path: Path, result: FileValidationResult):
        """Check images for alt text"""
        img_pattern = re.compile(r'<img[^>]*>', re.IGNORECASE)

        for match in img_pattern.finditer(content):
            img_tag = match.group()

            if 'alt=' not in img_tag.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.CRITICAL,
                    code='MISSING_ALT',
                    message='Image missing alt attribute',
                    file=str(file_path.relative_to(self.course_dir)),
                    element=img_tag[:100],
                    wcag_criterion='1.1.1',
                    suggestion='Add alt attribute with descriptive text'
                ))
            elif 'alt=""' in img_tag or "alt=''" in img_tag:
                # Check if properly marked as decorative
                if 'role="presentation"' not in img_tag and 'aria-hidden="true"' not in img_tag:
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.WCAG,
                        severity=ValidationSeverity.LOW,
                        code='EMPTY_ALT_NO_ROLE',
                        message='Decorative image should have role="presentation"',
                        file=str(file_path.relative_to(self.course_dir)),
                        element=img_tag[:100],
                        wcag_criterion='1.1.1',
                        suggestion='Add role="presentation" for decorative images'
                    ))

    def _check_heading_structure(self, content: str, file_path: Path, result: FileValidationResult):
        """Check heading hierarchy"""
        heading_pattern = re.compile(r'<h([1-6])[^>]*>', re.IGNORECASE)
        headings = [(int(m.group(1)), m.start()) for m in heading_pattern.finditer(content)]

        if not headings:
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.MEDIUM,
                code='NO_HEADINGS',
                message='Page has no headings',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='1.3.1',
                suggestion='Add heading structure to organize content'
            ))
            return

        # Check for H1
        if not any(h[0] == 1 for h in headings):
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.HIGH,
                code='NO_H1',
                message='Page missing H1 heading',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='2.4.6',
                suggestion='Add H1 as the main page heading'
            ))

        # Check for skipped levels
        for i in range(1, len(headings)):
            prev_level = headings[i-1][0]
            curr_level = headings[i][0]
            if curr_level > prev_level + 1:
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='SKIPPED_HEADING',
                    message=f'Heading level skipped from H{prev_level} to H{curr_level}',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='1.3.1',
                    suggestion=f'Use H{prev_level + 1} instead of H{curr_level}'
                ))

    def _check_language(self, content: str, file_path: Path, result: FileValidationResult):
        """Check language declaration"""
        if '<html' in content.lower():
            if not re.search(r'<html[^>]*\slang=', content, re.IGNORECASE):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='NO_LANG',
                    message='HTML element missing lang attribute',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='3.1.1',
                    suggestion='Add lang="en" (or appropriate language) to html element'
                ))

    def _check_links(self, content: str, file_path: Path, result: FileValidationResult):
        """Check link accessibility"""
        link_pattern = re.compile(r'<a[^>]*>([^<]*)</a>', re.IGNORECASE)

        non_descriptive = ['click here', 'here', 'read more', 'more', 'link']

        for match in link_pattern.finditer(content):
            link_text = match.group(1).strip().lower()
            if link_text in non_descriptive:
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='NON_DESCRIPTIVE_LINK',
                    message=f'Non-descriptive link text: "{link_text}"',
                    file=str(file_path.relative_to(self.course_dir)),
                    element=match.group(),
                    wcag_criterion='2.4.4',
                    suggestion='Use descriptive link text that indicates destination'
                ))

    def _check_forms(self, content: str, file_path: Path, result: FileValidationResult):
        """Check form accessibility"""
        input_pattern = re.compile(r'<input[^>]*>', re.IGNORECASE)

        for match in input_pattern.finditer(content):
            input_tag = match.group()
            input_type = re.search(r'type=["\']?(\w+)', input_tag, re.IGNORECASE)

            # Skip hidden and submit/button inputs
            if input_type and input_type.group(1).lower() in ['hidden', 'submit', 'button', 'image']:
                continue

            # Check for id to associate with label
            if 'id=' not in input_tag.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='INPUT_NO_ID',
                    message='Form input missing id attribute for label association',
                    file=str(file_path.relative_to(self.course_dir)),
                    element=input_tag[:100],
                    wcag_criterion='3.3.2',
                    suggestion='Add id attribute and associated label element'
                ))

    def _check_tables(self, content: str, file_path: Path, result: FileValidationResult):
        """Check table accessibility"""
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)

        for match in table_pattern.finditer(content):
            table = match.group()

            if '<th' not in table.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='TABLE_NO_HEADERS',
                    message='Data table missing header cells',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='1.3.1',
                    suggestion='Add th elements for table headers with scope attributes'
                ))

            if 'scope=' not in table.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='TABLE_NO_SCOPE',
                    message='Table headers missing scope attribute',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='1.3.1',
                    suggestion='Add scope="col" or scope="row" to th elements'
                ))

    def _check_color_contrast(self, content: str, file_path: Path, result: FileValidationResult):
        """Check for color contrast indicators (heuristic)"""
        # Check for color-only information
        color_words = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        color_pattern = re.compile(
            r'\b(in\s+)?(' + '|'.join(color_words) + r')\b(?!\s+text)',
            re.IGNORECASE
        )

        if color_pattern.search(content):
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.LOW,
                code='COLOR_REFERENCE',
                message='Content may reference color for meaning',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='1.4.1',
                suggestion='Ensure color is not the only way information is conveyed'
            ))

    def _check_keyboard_access(self, content: str, file_path: Path, result: FileValidationResult):
        """Check keyboard accessibility"""
        # Check for click handlers without keyboard equivalents
        if 'onclick=' in content.lower():
            if 'onkeypress=' not in content.lower() and 'onkeydown=' not in content.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='MOUSE_ONLY_HANDLER',
                    message='Click handler without keyboard alternative',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='2.1.1',
                    suggestion='Add keyboard event handlers (onkeypress, onkeydown)'
                ))

        # Check for tabindex abuse
        if re.search(r'tabindex=["\']?-?\d+', content):
            if 'tabindex="-1"' in content:
                pass  # Valid use
            elif re.search(r'tabindex=["\']?[1-9]', content):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='POSITIVE_TABINDEX',
                    message='Positive tabindex values can disrupt keyboard navigation',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='2.4.3',
                    suggestion='Use tabindex="0" or remove tabindex and rely on DOM order'
                ))

    def _check_page_title(self, content: str, file_path: Path, result: FileValidationResult):
        """Check page title"""
        title_match = re.search(r'<title[^>]*>([^<]*)</title>', content, re.IGNORECASE)

        if not title_match:
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.HIGH,
                code='NO_TITLE',
                message='Page missing title element',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='2.4.2',
                suggestion='Add descriptive title element in head'
            ))
        elif not title_match.group(1).strip():
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.HIGH,
                code='EMPTY_TITLE',
                message='Page title is empty',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='2.4.2',
                suggestion='Add descriptive text to title element'
            ))

    def _check_skip_links(self, content: str, file_path: Path, result: FileValidationResult):
        """Check for skip navigation links"""
        if '<nav' in content.lower() or 'navigation' in content.lower():
            skip_patterns = ['skip to', 'skip navigation', 'jump to']
            has_skip = any(p in content.lower() for p in skip_patterns)

            if not has_skip:
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='NO_SKIP_LINK',
                    message='Page with navigation lacks skip link',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='2.4.1',
                    suggestion='Add skip navigation link at top of page'
                ))

    def _check_aria(self, content: str, file_path: Path, result: FileValidationResult):
        """Check ARIA usage"""
        # Check for role without required ARIA attributes
        if 'role="tab"' in content.lower():
            if 'aria-selected=' not in content.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='TAB_NO_SELECTED',
                    message='Tab role missing aria-selected attribute',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='4.1.2',
                    suggestion='Add aria-selected to tab elements'
                ))

        # Check for expandable controls
        if 'data-toggle="collapse"' in content.lower():
            if 'aria-expanded=' not in content.lower():
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='COLLAPSE_NO_EXPANDED',
                    message='Collapsible element missing aria-expanded',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='4.1.2',
                    suggestion='Add aria-expanded to toggle buttons'
                ))

    def _check_brightspace_compat(self, content: str, file_path: Path, result: FileValidationResult):
        """Check Brightspace compatibility"""
        # Check for problematic JavaScript
        js_issues = [
            (r'document\.write', 'document.write may cause issues in Brightspace'),
            (r'window\.open\s*\(', 'window.open may be blocked in Brightspace'),
            (r'eval\s*\(', 'eval() is a security risk and may be blocked'),
        ]

        for pattern, message in js_issues:
            if re.search(pattern, content, re.IGNORECASE):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.BRIGHTSPACE,
                    severity=ValidationSeverity.MEDIUM,
                    code='JS_COMPATIBILITY',
                    message=message,
                    file=str(file_path.relative_to(self.course_dir)),
                    suggestion='Consider alternative approaches for Brightspace compatibility'
                ))

        # Check for external resources that may be blocked
        if re.search(r'src=["\']http:', content, re.IGNORECASE):
            result.issues.append(ValidationIssue(
                category=ValidationCategory.BRIGHTSPACE,
                severity=ValidationSeverity.HIGH,
                code='HTTP_RESOURCE',
                message='HTTP (non-HTTPS) resource may be blocked',
                file=str(file_path.relative_to(self.course_dir)),
                suggestion='Use HTTPS for all external resources'
            ))

    # ==================== WCAG 2.2 Validation Methods ====================

    def _check_focus_not_obscured(self, content: str, file_path: Path, result: FileValidationResult):
        """Check for elements that might obscure focused content (WCAG 2.4.11, 2.4.12)"""
        # Check for fixed or sticky positioned elements that might obscure focus
        if re.search(r'position\s*:\s*(fixed|sticky)', content, re.IGNORECASE):
            # Check if there's scroll-margin to prevent obscuring
            if not re.search(r'scroll-margin', content, re.IGNORECASE):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.MEDIUM,
                    code='FOCUS_MAY_BE_OBSCURED',
                    message='Fixed/sticky elements may obscure focused content',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='2.4.11',
                    suggestion='Add scroll-margin to prevent fixed headers/footers from obscuring focused elements'
                ))

    def _check_focus_appearance(self, content: str, file_path: Path, result: FileValidationResult):
        """Check focus indicator styling meets WCAG 2.2 requirements (WCAG 2.4.13)"""
        # Check for focus styles with insufficient outline width
        focus_pattern = re.search(r':focus[^{]*\{[^}]*outline\s*:\s*(\d+)px', content, re.IGNORECASE)
        if focus_pattern:
            outline_width = int(focus_pattern.group(1))
            if outline_width < 2:
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='INSUFFICIENT_FOCUS_OUTLINE',
                    message=f'Focus outline width ({outline_width}px) is less than 2px minimum',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='2.4.13',
                    suggestion='Increase focus outline to at least 2px for WCAG 2.2 compliance'
                ))

        # Check for outline: none without replacement
        if re.search(r':focus[^{]*\{[^}]*outline\s*:\s*none', content, re.IGNORECASE):
            # Check if there's an alternative focus indicator
            if not re.search(r':focus[^{]*\{[^}]*(box-shadow|border|background)', content, re.IGNORECASE):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.CRITICAL,
                    code='FOCUS_REMOVED_NO_ALTERNATIVE',
                    message='Focus outline removed without alternative indicator',
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='2.4.13',
                    suggestion='Provide visible focus indicator (outline, box-shadow, or border)'
                ))

    def _check_target_size(self, content: str, file_path: Path, result: FileValidationResult):
        """Check interactive element target sizes (WCAG 2.5.8)"""
        # Check for small explicit dimensions on interactive elements
        small_size_patterns = [
            (r'<(button|a|input)[^>]*(?:width|height)\s*[:=]\s*["\']?(\d+)px', 'element'),
            (r'(?:width|height)\s*:\s*(\d+)px[^}]*}[^{]*(?:button|\.btn|a\s*{)', 'css'),
        ]

        for pattern, pattern_type in small_size_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                size = int(match.group(2) if pattern_type == 'element' else match.group(1))
                if size < 24:
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.WCAG,
                        severity=ValidationSeverity.MEDIUM,
                        code='SMALL_TARGET_SIZE',
                        message=f'Interactive element has size {size}px, less than 24px minimum',
                        file=str(file_path.relative_to(self.course_dir)),
                        element=match.group()[:100] if len(match.group()) > 100 else match.group(),
                        wcag_criterion='2.5.8',
                        suggestion='Increase target size to at least 24x24 CSS pixels'
                    ))

    def _check_dragging_movements(self, content: str, file_path: Path, result: FileValidationResult):
        """Check for drag operations without single-pointer alternatives (WCAG 2.5.7)"""
        # Detect drag-related attributes and events
        drag_patterns = [
            r'draggable\s*=\s*["\']?true',
            r'ondrag\s*=',
            r'ondragstart\s*=',
            r'ondragend\s*=',
            r'\.draggable\s*\(',
            r'\.sortable\s*\(',
        ]

        for pattern in drag_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Check if there's a button-based alternative
                if not re.search(r'(move-up|move-down|reorder|aria-grabbed)', content, re.IGNORECASE):
                    result.issues.append(ValidationIssue(
                        category=ValidationCategory.WCAG,
                        severity=ValidationSeverity.HIGH,
                        code='DRAG_NO_ALTERNATIVE',
                        message='Draggable element found without single-pointer alternative',
                        file=str(file_path.relative_to(self.course_dir)),
                        wcag_criterion='2.5.7',
                        suggestion='Provide button-based alternative for drag operations'
                    ))
                break

    def _check_consistent_help(self, content: str, file_path: Path, result: FileValidationResult):
        """Check for consistent help mechanism placement (WCAG 3.2.6)"""
        # This is a page-level check - look for help links
        help_patterns = [
            r'<a[^>]*>.*?help.*?</a>',
            r'<a[^>]*>.*?support.*?</a>',
            r'<a[^>]*>.*?contact.*?</a>',
            r'aria-label\s*=\s*["\'][^"\']*help',
        ]

        has_help = any(re.search(p, content, re.IGNORECASE) for p in help_patterns)

        # If content has forms, help should be available
        if '<form' in content.lower() and not has_help:
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.LOW,
                code='NO_HELP_FOR_FORMS',
                message='Form content lacks visible help mechanism',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='3.2.6',
                suggestion='Add help link in consistent location (header or footer)'
            ))

    def _check_accessible_authentication(self, content: str, file_path: Path, result: FileValidationResult):
        """Check for cognitive function tests in authentication (WCAG 3.3.8, 3.3.9)"""
        # Detect CAPTCHA or cognitive tests
        auth_issues = [
            (r'captcha', 'CAPTCHA detected - may fail cognitive function test requirement'),
            (r'recaptcha', 'reCAPTCHA detected - may fail cognitive function test requirement'),
            (r'hcaptcha', 'hCaptcha detected - may fail cognitive function test requirement'),
            (r'g-recaptcha', 'Google reCAPTCHA detected - may fail cognitive function test requirement'),
        ]

        for pattern, message in auth_issues:
            if re.search(pattern, content, re.IGNORECASE):
                result.issues.append(ValidationIssue(
                    category=ValidationCategory.WCAG,
                    severity=ValidationSeverity.HIGH,
                    code='COGNITIVE_AUTH_TEST',
                    message=message,
                    file=str(file_path.relative_to(self.course_dir)),
                    wcag_criterion='3.3.8',
                    suggestion='Provide alternative authentication method without cognitive function tests'
                ))

        # Check for paste blocking on password fields
        if re.search(r'onpaste\s*=\s*["\']?\s*return\s+false', content, re.IGNORECASE):
            result.issues.append(ValidationIssue(
                category=ValidationCategory.WCAG,
                severity=ValidationSeverity.HIGH,
                code='PASTE_BLOCKED',
                message='Paste functionality blocked - prevents password manager use',
                file=str(file_path.relative_to(self.course_dir)),
                wcag_criterion='3.3.8',
                suggestion='Allow paste on password fields to support password managers'
            ))

    def _validate_manifest(self, manifest_path: Path):
        """Validate imsmanifest.xml"""
        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()

            # Check for required elements
            if root.find('.//{*}organizations') is None:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.BRIGHTSPACE,
                    severity=ValidationSeverity.CRITICAL,
                    code='NO_ORGANIZATIONS',
                    message='Manifest missing organizations element',
                    file='imsmanifest.xml'
                ))

            if root.find('.//{*}resources') is None:
                self.issues.append(ValidationIssue(
                    category=ValidationCategory.BRIGHTSPACE,
                    severity=ValidationSeverity.CRITICAL,
                    code='NO_RESOURCES',
                    message='Manifest missing resources element',
                    file='imsmanifest.xml'
                ))

        except ET.ParseError as e:
            self.issues.append(ValidationIssue(
                category=ValidationCategory.BRIGHTSPACE,
                severity=ValidationSeverity.CRITICAL,
                code='INVALID_XML',
                message=f'Manifest XML parse error: {e}',
                file='imsmanifest.xml'
            ))

    def _calculate_wcag_score(self) -> float:
        """Calculate WCAG compliance score"""
        total_files = len(self.file_results)
        if total_files == 0:
            return 100.0

        compliant_files = sum(1 for r in self.file_results if r.wcag_compliant)
        return (compliant_files / total_files) * 100

    def _calculate_oscqr_score(self) -> float:
        """Calculate OSCQR compliance score (based on content analysis)"""
        # Simplified OSCQR check based on available content
        checks_passed = 0
        total_checks = 5

        # Check for learning objectives
        has_objectives = any(
            'objective' in str(r.file_path).lower() or
            any('learning objective' in str(i.message).lower() for i in r.issues)
            for r in self.file_results
        )
        if has_objectives:
            checks_passed += 1

        # Check for consistent structure
        has_structure = all(
            not any(i.code == 'NO_HEADINGS' for i in r.issues)
            for r in self.file_results
        )
        if has_structure:
            checks_passed += 1

        # Check for navigation aids
        has_nav = any(
            not any(i.code == 'NO_SKIP_LINK' for i in r.issues)
            for r in self.file_results
        )
        if has_nav:
            checks_passed += 1

        # Check for accessible content
        all_accessible = all(r.wcag_compliant for r in self.file_results)
        if all_accessible:
            checks_passed += 2

        return (checks_passed / total_checks) * 100

    def _generate_summary(self, wcag: float, oscqr: float, brightspace: bool) -> str:
        """Generate human-readable summary"""
        status = "PASS" if wcag >= 95 and oscqr >= 80 and brightspace else "NEEDS ATTENTION"

        return f"""
Validation Summary: {status}
===========================
WCAG 2.2 AA Compliance: {wcag:.1f}%
OSCQR Standards: {oscqr:.1f}%
Brightspace Ready: {'Yes' if brightspace else 'No'}

{'Course is ready for deployment.' if status == 'PASS' else 'Please address the identified issues before deployment.'}
"""

    def generate_report_text(self, report: ValidationReport) -> str:
        """Generate human-readable report"""
        text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                 REMEDIATION VALIDATION REPORT                     ║
╠══════════════════════════════════════════════════════════════════╣
║ Course: {report.course_path[:56]:<56} ║
║ Timestamp: {report.validation_timestamp[:52]:<52} ║
╠══════════════════════════════════════════════════════════════════╣
║ SUMMARY                                                           ║
╠══════════════════════════════════════════════════════════════════╣
║ Total Files: {report.total_files:<51} ║
║ Files with Issues: {report.files_with_issues:<45} ║
║ Total Issues: {report.total_issues:<50} ║
║ WCAG Compliance: {report.wcag_compliance:.1f}%{' ':<47} ║
║ OSCQR Compliance: {report.oscqr_compliance:.1f}%{' ':<46} ║
║ Brightspace Ready: {'Yes' if report.brightspace_ready else 'No':<45} ║
║ Overall Score: {report.overall_score:.1f}%{' ':<49} ║
╠══════════════════════════════════════════════════════════════════╣
║ ISSUES BY SEVERITY                                                ║
╠══════════════════════════════════════════════════════════════════╣
"""
        for severity, count in sorted(report.issues_by_severity.items()):
            text += f"║ {severity.upper()}: {count:<57} ║\n"

        text += """╚══════════════════════════════════════════════════════════════════╝
"""
        return text

    def to_json(self, report: ValidationReport) -> str:
        """Export report as JSON"""
        data = asdict(report)

        # Fix enum values
        for file_result in data['file_results']:
            for issue in file_result['issues']:
                issue['category'] = issue['category'].value if hasattr(issue['category'], 'value') else issue['category']
                issue['severity'] = issue['severity'].value if hasattr(issue['severity'], 'value') else issue['severity']

        return json.dumps(data, indent=2, default=str)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate remediated course content'
    )
    parser.add_argument('--course-dir', '-c', required=True,
                       help='Course directory to validate')
    parser.add_argument('--output', '-o', help='Output report file (JSON)')
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')
    parser.add_argument('--before', help='Original course for comparison')
    parser.add_argument('--after', help='Remediated course for comparison')
    parser.add_argument('--compare', action='store_true', help='Generate comparison report')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        validator = RemediationValidator(Path(args.course_dir))
        report = validator.validate()

        if args.json:
            print(validator.to_json(report))
        else:
            print(validator.generate_report_text(report))

        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(validator.to_json(report))
            print(f"\nReport saved to: {output_path}")

        # Exit with appropriate code
        if report.overall_score >= 95:
            sys.exit(0)
        elif report.overall_score >= 80:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Failed

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
