# Accessibility Validator Module
# WCAG 2.2 AA compliance validation for educational content

"""
This module provides comprehensive accessibility validation for HTML content,
ensuring WCAG 2.2 AA compliance with automated checking for alt text, color
contrast, heading hierarchy, keyboard navigation, ARIA landmarks, and form labels.
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ['AccessibilityValidator', 'WCAGIssue', 'IssueSeverity', 'ValidationReport']
