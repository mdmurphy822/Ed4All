"""Block role enum + dataclasses for the ontology-aware DART converter.

Wave 12 foundation. The closed-set taxonomy derives from the ontology-survey
investigation (see ``plans/we-have-several-branches-gentle-melody.md``).

Every block emitted by the converter carries exactly one ``BlockRole``.
The template registry (``block_templates.py``) maps each role to a
template function that produces ontology-aware HTML. Subsequent waves
(13+) enrich the template outputs with DPUB-ARIA / schema.org /
Dublin Core layers without changing the role set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class BlockRole(Enum):
    """Closed-set taxonomy of semantic roles a text block may carry.

    Grouped by pedagogical / structural function. Every raw block
    produced by the segmenter receives exactly one role. When no role
    applies, the classifier falls back to ``PARAGRAPH``.
    """

    # ---- Structural -------------------------------------------------
    CHAPTER_OPENER = "chapter_opener"
    SECTION_HEADING = "section_heading"
    SUBSECTION_HEADING = "subsection_heading"
    PARAGRAPH = "paragraph"
    TOC_NAV = "toc_nav"
    PAGE_BREAK = "page_break"
    # ---- Lists (Wave 21) --------------------------------------------
    # LIST_ITEM is an intermediate role produced by the classifier for a
    # single marker-prefixed block. The document assembler groups
    # consecutive LIST_ITEM blocks of the same marker_type into a
    # synthesized LIST_UNORDERED / LIST_ORDERED block with
    # ``attributes.items = [{text, marker, sub_items}, ...]``. A stray
    # LIST_ITEM that survives grouping still renders as a single-item
    # ``<ul>`` so the output remains valid.
    LIST_UNORDERED = "list_unordered"
    LIST_ORDERED = "list_ordered"
    LIST_ITEM = "list_item"

    # ---- Educational ------------------------------------------------
    LEARNING_OBJECTIVES = "learning_objectives"
    KEY_TAKEAWAYS = "key_takeaways"
    ACTIVITY = "activity"
    SELF_CHECK = "self_check"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    GLOSSARY_ENTRY = "glossary_entry"

    # ---- Reference --------------------------------------------------
    ABSTRACT = "abstract"
    BIBLIOGRAPHY_ENTRY = "bibliography_entry"
    FOOTNOTE = "footnote"
    CITATION = "citation"
    CROSS_REFERENCE = "cross_reference"

    # ---- Content-rich -----------------------------------------------
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    FORMULA_MATH = "formula_math"
    BLOCKQUOTE = "blockquote"
    EPIGRAPH = "epigraph"
    PULLQUOTE = "pullquote"

    # ---- Notice -----------------------------------------------------
    CALLOUT_INFO = "callout_info"
    CALLOUT_WARNING = "callout_warning"
    CALLOUT_TIP = "callout_tip"
    CALLOUT_DANGER = "callout_danger"

    # ---- Metadata ---------------------------------------------------
    TITLE = "title"
    AUTHOR_AFFILIATION = "author_affiliation"
    COPYRIGHT_LICENSE = "copyright_license"
    KEYWORDS = "keywords"
    BIBLIOGRAPHIC_METADATA = "bibliographic_metadata"


@dataclass
class RawBlock:
    """A contiguous candidate block produced by the segmenter.

    ``text`` is the raw block content. ``block_id`` is a stable identifier
    (content-hash + positional) used to key classifier decisions and
    downstream provenance (``data-dart-block-id``). ``page`` and ``bbox``
    are populated when layout extraction is available. ``extractor``
    names which upstream extractor produced the text. ``neighbors`` is
    a convenience map carrying the ``prev`` and ``next`` sibling text,
    which the classifier uses for context disambiguation.

    Wave 16 additions:

    * ``extractor_hint`` — when the segmenter knows a block's role at
      extraction time (pdfplumber table, PyMuPDF figure), it stamps
      the hinted :class:`BlockRole` here so classifiers skip text
      classification and emit the hinted role at confidence 1.0.
    * ``extra`` — structured attributes from the extractor that the
      classifier forwards into :attr:`ClassifiedBlock.attributes` (e.g.
      ``header_rows`` / ``body_rows`` for a table; ``image_path`` /
      ``alt`` / ``caption`` for a figure).

    Both fields default to empty so pre-Wave-16 callers stay compatible.
    """

    text: str
    block_id: str
    page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    extractor: str = "pdftotext"
    neighbors: dict = field(default_factory=dict)
    extractor_hint: Optional["BlockRole"] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ClassifiedBlock:
    """A ``RawBlock`` plus its classifier decision.

    ``role`` is the assigned ``BlockRole``. ``confidence`` is a float in
    ``[0.0, 1.0]`` expressing classifier certainty. ``attributes`` is a
    role-specific dict (e.g. a ``FIGURE`` block may carry ``number``
    and ``caption_text`` keys; a ``BIBLIOGRAPHY_ENTRY`` may carry
    ``ref_id``). ``classifier_source`` records which classifier made the
    decision so downstream consumers can distinguish heuristic vs
    LLM-backed classifications.
    """

    raw: RawBlock
    role: BlockRole
    confidence: float
    attributes: dict = field(default_factory=dict)
    classifier_source: str = "heuristic"


__all__ = ["BlockRole", "ClassifiedBlock", "RawBlock"]
