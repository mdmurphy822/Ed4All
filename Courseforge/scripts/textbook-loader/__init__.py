"""
Textbook Loader - DART to Courseforge integration.

Loads DART-processed HTML textbooks and extracts structured content
for course generation.
"""

from .textbook_loader import (
    TextbookLoader,
    TextbookContent,
    TextbookSection,
    load_textbooks,
    DEFAULT_TEXTBOOKS_DIR,
)

__all__ = [
    'TextbookLoader',
    'TextbookContent',
    'TextbookSection',
    'load_textbooks',
    'DEFAULT_TEXTBOOKS_DIR',
]
