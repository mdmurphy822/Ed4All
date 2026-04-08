"""
Textbook Loader - DART to Courseforge integration.

Loads DART-processed HTML textbooks and extracts structured content
for course generation.
"""

from .textbook_loader import (
    DEFAULT_TEXTBOOKS_DIR,
    TextbookContent,
    TextbookLoader,
    TextbookSection,
    load_textbooks,
)

__all__ = [
    'TextbookLoader',
    'TextbookContent',
    'TextbookSection',
    'load_textbooks',
    'DEFAULT_TEXTBOOKS_DIR',
]
