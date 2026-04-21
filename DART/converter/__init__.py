"""DART ontology-aware HTML converter package (Wave 12 foundation).

Four-phase pipeline that replaces the monolithic regex-driven converter:

    1. segment  - split raw text into RawBlocks (block_segmenter)
    2. classify - assign a BlockRole to each block (heuristic_classifier)
    3. template - render each classified block via role template
    4. assemble - stitch rendered blocks into a full HTML document

Wave 12 ships the foundation: enum, dataclasses, segmenter, heuristic
classifier (ports existing regex logic from pipeline_tools), minimal
template registry, and minimal assembler. Subsequent waves:

    Wave 13 - expand templates with DPUB-ARIA + schema.org + rich HTML
    Wave 14 - add Claude-backed classifier behind LLMBackend
    Wave 15 - document-level decoration (Dublin Core, JSON-LD, WCAG CSS)
    Wave 16 - dual-extraction + MathML + figure pipeline

The existing ``_raw_text_to_accessible_html`` in ``MCP/tools/pipeline_tools.py``
is left untouched. Wave 15 will flip the production path to this package.
"""

from DART.converter.block_roles import (
    BlockRole,
    ClassifiedBlock,
    RawBlock,
)
from DART.converter.block_segmenter import segment_pdftotext_output
from DART.converter.block_templates import TEMPLATE_REGISTRY, render_block
from DART.converter.document_assembler import assemble_html
from DART.converter.heuristic_classifier import HeuristicClassifier


def convert_pdftotext_to_html(
    raw_text: str,
    title: str,
    metadata: dict | None = None,
) -> str:
    """Full 4-phase pipeline: segment -> classify -> template -> assemble.

    Wave 12 scope: heuristic classifier only, minimal templates, minimal
    assembler. Subsequent waves add LLM classifier + ontology expansion +
    document-level decoration.
    """
    blocks = segment_pdftotext_output(raw_text)
    classifier = HeuristicClassifier()
    classified = classifier.classify(blocks)
    return assemble_html(classified, title, metadata or {})


__all__ = [
    "BlockRole",
    "ClassifiedBlock",
    "HeuristicClassifier",
    "RawBlock",
    "TEMPLATE_REGISTRY",
    "assemble_html",
    "convert_pdftotext_to_html",
    "render_block",
    "segment_pdftotext_output",
]
