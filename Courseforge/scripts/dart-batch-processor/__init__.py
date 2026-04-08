# DART Batch Processor Module
# Automated document to accessible HTML conversion

"""
This module orchestrates batch conversion of PDF and Office documents
to WCAG 2.2 AA compliant accessible HTML using DART (Digital Accessibility
Remediation Tool).
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ['DARTBatchProcessor', 'ConversionStatus', 'DocumentType']
