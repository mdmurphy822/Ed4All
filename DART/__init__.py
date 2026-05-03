"""
DART - PDF to Accessible HTML Conversion

DART converts PDF textbooks into accessible, semantically-structured HTML
via multi-source synthesis with WCAG 2.2 AA compliance.

Core capabilities:
- Multi-source PDF extraction and synthesis
- Semantic HTML structure generation
- Accessibility remediation (alt text, headings, WCAG)
- Per-block source provenance for downstream consumers
"""

__version__ = "0.1.0"
__author__ = "Ed4All"

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
