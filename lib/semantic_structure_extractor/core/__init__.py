"""
Core modules for semantic structure extraction.

These modules provide the foundational parsing capabilities:
- HeadingParser: Extract heading hierarchy from HTML
- ContentBlockClassifier: Classify content blocks by type
"""

from .heading_parser import HeadingParser, HeadingHierarchy, HeadingNode
from .content_block_classifier import (
    ContentBlockClassifier,
    ContentBlock,
    BlockType,
    Definition,
    KeyTerm
)

__all__ = [
    # Heading parser
    'HeadingParser',
    'HeadingHierarchy',
    'HeadingNode',
    # Content block classifier
    'ContentBlockClassifier',
    'ContentBlock',
    'BlockType',
    'Definition',
    'KeyTerm',
]
