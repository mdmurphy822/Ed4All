# Remediation Validator Module
# Final quality assurance for remediated courses

"""
This module provides comprehensive validation of remediated course content
ensuring WCAG 2.2 AA compliance, OSCQR standards adherence, and Brightspace
compatibility before final IMSCC packaging.
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ['RemediationValidator', 'ValidationReport', 'ValidationSeverity']
