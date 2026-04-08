"""
Core modules for semantic structure extraction.

These modules provide the foundational parsing capabilities:
- HeadingParser: Extract heading hierarchy from HTML
- ContentBlockClassifier: Classify content blocks by type
"""

from .content_block_classifier import (
    BlockType,
    ContentBlock,
    ContentBlockClassifier,
    Definition,
    KeyTerm,
)
from .heading_parser import HeadingHierarchy, HeadingNode, HeadingParser

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
