"""
Semantic Structure Extractor Library v2.0.0

Unified library for extracting semantic structure from HTML and Markdown content.
Consolidates functionality from Courseforge and Slideforge into a shared library.

Usage:
    from lib.semantic_structure_extractor import (
        SemanticStructureExtractor,
        extract_textbook_structure,
        extract_for_presentation,
    )

    # Basic extraction (Courseforge-style)
    extractor = SemanticStructureExtractor()
    result = extractor.extract_file('document.html')

    # With profiling (Slideforge-style)
    result = extractor.extract_with_profiling('document.html')

    # For presentations (Slideforge-style)
    result = extractor.extract_for_presentation('document.md')
"""

__version__ = "2.0.0"

# Core modules
from .core import (
    HeadingParser,
    HeadingHierarchy,
    HeadingNode,
    ContentBlockClassifier,
    ContentBlock,
    BlockType,
    Definition,
    KeyTerm,
)

# Format parsers
from .formats import (
    MarkdownParser,
    MarkdownDocument,
    detect_format,
)

# Analysis modules
from .analysis import (
    ContentProfiler,
    ContentProfile,
    ConceptGraphBuilder,
    ConceptGraph,
)

# Transformers
from .transformers import (
    PresentationTransformer,
)

# Main extractor and convenience functions
from .semantic_structure_extractor import (
    SemanticStructureExtractor,
    ChapterStructure,
    SectionStructure,
    ExtractedProcedure,
    ExtractedExample,
    ReviewQuestion,
    extract_textbook_structure,
)

__all__ = [
    # Version
    '__version__',
    # Core
    'HeadingParser',
    'HeadingHierarchy',
    'HeadingNode',
    'ContentBlockClassifier',
    'ContentBlock',
    'BlockType',
    'Definition',
    'KeyTerm',
    # Formats
    'MarkdownParser',
    'MarkdownDocument',
    'detect_format',
    # Analysis
    'ContentProfiler',
    'ContentProfile',
    'ConceptGraphBuilder',
    'ConceptGraph',
    # Transformers
    'PresentationTransformer',
    # Main extractor
    'SemanticStructureExtractor',
    'ChapterStructure',
    'SectionStructure',
    'ExtractedProcedure',
    'ExtractedExample',
    'ReviewQuestion',
    'extract_textbook_structure',
]
