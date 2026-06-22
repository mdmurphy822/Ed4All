"""Common dataclasses for the 8-stage pipeline.

Types are named for the stage that produces them:
    stage 1 extract     -> RawBlock
    stage 2 features    -> FeatureBlock (wraps RawBlock + feature flags)
    stage 3 classify    -> ClassifiedBlock (wraps FeatureBlock + Role + confidence)
    stage 4 hierarchy   -> ResolvedBlock (wraps ClassifiedBlock + depth + position)
    stage 5 ontology    -> str (HTML output)
    stage 6 enrich      -> ResolvedBlock[] with enrichments attached
    stage 7 validate    -> ValidationResult (from validate.py)
    stage 8 escalate    -> EscalationDecision (from escalate.py)

Each wrapper keeps a reference to the prior stage's block so provenance
is preserved — we can always trace an HTML element back to the raw bbox
on the source PDF page. That's the audit trail procurement reviewers want.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawBlock:
    """Stage 1: a text block straight from the PDF.

    Geometry is in page-pixel coordinates. Font info is exact when sourced
    from a text layer; estimated (font_size only) when sourced from OCR.
    """
    text: str
    page: int                                   # 1-indexed
    bbox: tuple[float, float, float, float]     # (x0, y0, x1, y1)
    page_width: float
    page_height: float
    font_size: float | None = None
    font_name: str | None = None
    is_bold: bool | None = None
    is_italic: bool | None = None
    confidence: float = 1.0
    source: str = "pymupdf"                     # "pymupdf" | "tesseract"


@dataclass
class FeatureBlock:
    """Stage 2: a RawBlock plus derived layout features in a stable vocabulary.

    The feature flags are the same vocabulary the classifier consumes and
    the rules reason over. Keep this dict-like for easy serialization into
    training data and debug dumps.
    """
    raw: RawBlock
    size_bucket: str                            # "xl" | "lg" | "md" | "sm"
    gap_above: str | None                       # "lg" | None
    is_top_of_page: bool                        # top 15% of page
    is_centered: bool                           # short + midpoint near page center
    caps: str | None                            # "all" | "title" | None
    indent_bucket: int                          # left-edge bucket 0..9
    relative_font_ratio: float                  # font_size / page median
    # Context carried from neighbors — useful for rules and classifier.
    prev_size_bucket: str | None = None
    next_size_bucket: str | None = None
    # Multi-extractor layout signals (populated when featurize_from_shared is
    # used; default to None/False when featurizing from bare RawBlocks).
    in_table: bool = False                      # pdfplumber detected a table containing this bbox
    in_header_row: bool = False                 # block is in the top 25% of its table (likely header)
    in_widget: bool = False                     # pikepdf AcroForm widget overlaps this bbox
    widget_kind: str | None = None              # "text" | "button" | "select" | "signature"
    provenance: str | None = None               # "pypdfium2" | "pdfplumber+pypdfium2" | "tesseract" | ...


@dataclass
class ClassifiedBlock:
    """Stage 3: a FeatureBlock plus its structural role and confidence."""
    features: FeatureBlock
    role: str                                   # classify.Role value
    confidence: float                           # [0, 1]
    source: str                                 # "rule:<name>" | "model" | "default"


@dataclass
class ResolvedBlock:
    """Stage 4: a ClassifiedBlock with its resolved hierarchical position.

    `depth` is stage-4-assigned:
      - Role.HEADING: 1..6 (h1..h6)
      - Role.LIST_ITEM: nesting depth 0+
      - Role.TABLE_ROW: row index within its table group
      - everything else: 0 (unused)
    """
    classified: ClassifiedBlock
    depth: int
    # For tables + lists, points at the owning group's index so stage 5 can
    # assemble them. -1 when not part of a group.
    group_id: int = -1
    enrichment: "Enrichment | None" = None


@dataclass
class Enrichment:
    """Stage 6: metadata attached to a ResolvedBlock that needs enrichment.

    Re-exported from enrich.py here so pipeline types live in one place.
    """
    alt_text: str | None = None
    alt_text_source: str | None = None
    language: str | None = None                 # BCP-47
    table_summary: str | None = None
    extended_description: str | None = None
    mathml: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1: Stage 2 union artifact
# ---------------------------------------------------------------------------
# Re-export the typed candidate classes so consumers can do
#   from dart_semantic.types import FeatureSet, TableCandidate, MathCandidate
# without reaching into region_detection.py.
from .region_detection import (  # noqa: E402  (re-export at module bottom)
    MathCandidate,
    RegionCandidate,
    TableCandidate,
)


@dataclass
class FeatureSet:
    """Stage 2 union artifact: FeatureBlock stream + typed region candidates.

    Returned by `dart_semantic.features.featurize_with_regions`. Phase 1
    keeps the legacy `featurize_from_shared` (which returns just the
    `FeatureBlock` list) callable; this class is the new aggregator
    that downstream Phase-2/3 detector BERTs consume.

    All three lists are independent — `feature_blocks` is the flat block
    stream, `table_candidates` and `math_candidates` are typed regions
    that may overlap zero or more feature blocks (see
    `RegionCandidate.member_block_indices`).
    """
    feature_blocks: list[FeatureBlock] = field(default_factory=list)
    table_candidates: list[TableCandidate] = field(default_factory=list)
    math_candidates: list[MathCandidate] = field(default_factory=list)


__all__ = [
    "RawBlock",
    "FeatureBlock",
    "ClassifiedBlock",
    "ResolvedBlock",
    "Enrichment",
    "FeatureSet",
    "RegionCandidate",
    "TableCandidate",
    "MathCandidate",
]
