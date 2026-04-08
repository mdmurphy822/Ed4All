#!/usr/bin/env python3
"""
PDF Accessibility Converter - Convenience CLI Script

Convert PDF documents to WCAG 2.1 AA compliant accessible HTML.

Usage:
    python convert.py input.pdf [options]

Options:
    -o, --output DIR    Output directory (default: ./output/)
    -n, --name NAME     Output filename (default: derived from PDF)
    --no-dark-mode      Disable dark mode CSS
    --no-ocr            Skip OCR for image-heavy PDFs
    --dpi N             DPI for OCR (default: 300)
    --lang CODE         Tesseract language (default: eng)
    -v, --verbose       Verbose output
    --version           Show version

Examples:
    python convert.py document.pdf
    python convert.py document.pdf -o ./accessible/
    python convert.py scanned.pdf --dpi 400 --lang deu
"""

import sys
from pathlib import Path

# Add package to path if running directly
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from pdf_converter.cli import main

if __name__ == '__main__':
    sys.exit(main())
