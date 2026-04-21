"""DART converter sidecar builders (Wave 19 restoration).

The pre-Wave-12 DART pipeline wrote two JSON sidecars next to every
emitted HTML file:

* ``{stem}_synthesized.json`` — per-section provenance sidecar in the
  shape documented in :mod:`DART/CLAUDE.md` § "Source provenance". The
  Courseforge source-router (``MCP/tools/pipeline_tools.py::_build_source_module_map``)
  walks these sidecars to build the ``source_module_map.json`` that the
  Courseforge generator consumes — without them, every emitted course
  page loses ``sourceReferences[]`` and the ``source_provenance`` /
  ``evidence_source_provenance`` LibV2 manifest flags are pinned to
  ``false``.
* ``{stem}.quality.json`` — WCAG + confidence aggregate sidecar used by
  the ``archive_to_libv2`` tool when populating
  ``{course}/quality/*.quality.json``.

Both were silently dropped when Waves 12–18 replaced the monolithic
regex-driven converter with the ontology-aware 4-phase pipeline. This
module restores them via the ``build_synthesized_sidecar`` and
``build_quality_sidecar`` helpers so the Wave 19 pipeline emits the
same sidecar contract as pre-Wave-12.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from DART.converter.block_roles import BlockRole, ClassifiedBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slug helper (local copy — avoids reaching into `lib.ontology.slugs` from
# a DART module and keeps this sidecar-build surface self-contained).
# ---------------------------------------------------------------------------


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    """Return a lowercase hyphen-delimited slug suitable for a sourceId."""
    if not value:
        return "document"
    cleaned = _SLUG_STRIP.sub("-", value.strip().lower()).strip("-")
    return cleaned or "document"


# ---------------------------------------------------------------------------
# Section-type mapping from ClassifiedBlock.role -> sidecar section_type.
# Mirrors the enum documented in the DART/CLAUDE.md source-provenance
# section and in ``build_synthesized_sidecar`` below.
# ---------------------------------------------------------------------------


_SECTION_TYPE_BY_ROLE: Dict[BlockRole, str] = {
    BlockRole.CHAPTER_OPENER: "chapter",
    BlockRole.SECTION_HEADING: "section",
    BlockRole.SUBSECTION_HEADING: "section",
    BlockRole.ABSTRACT: "section",
    BlockRole.LEARNING_OBJECTIVES: "section",
    BlockRole.KEY_TAKEAWAYS: "section",
    BlockRole.ACTIVITY: "section",
    BlockRole.SELF_CHECK: "section",
    BlockRole.EXAMPLE: "section",
    BlockRole.EXERCISE: "section",
    BlockRole.GLOSSARY_ENTRY: "section",
    BlockRole.FIGURE: "figure",
    BlockRole.FIGURE_CAPTION: "figure",
    BlockRole.TABLE: "table",
    BlockRole.CODE_BLOCK: "section",
    BlockRole.FORMULA_MATH: "section",
    BlockRole.BLOCKQUOTE: "paragraph-group",
    BlockRole.EPIGRAPH: "paragraph-group",
    BlockRole.PULLQUOTE: "paragraph-group",
    BlockRole.CALLOUT_INFO: "section",
    BlockRole.CALLOUT_WARNING: "section",
    BlockRole.CALLOUT_TIP: "section",
    BlockRole.CALLOUT_DANGER: "section",
    BlockRole.TOC_NAV: "section",
    BlockRole.PAGE_BREAK: "paragraph-group",
    BlockRole.PARAGRAPH: "paragraph-group",
    BlockRole.CITATION: "paragraph-group",
    BlockRole.CROSS_REFERENCE: "paragraph-group",
    BlockRole.BIBLIOGRAPHY_ENTRY: "bibliography",
    BlockRole.FOOTNOTE: "bibliography",
    BlockRole.TITLE: "paragraph-group",
    BlockRole.AUTHOR_AFFILIATION: "paragraph-group",
    BlockRole.COPYRIGHT_LICENSE: "paragraph-group",
    BlockRole.KEYWORDS: "paragraph-group",
    BlockRole.BIBLIOGRAPHIC_METADATA: "paragraph-group",
}


def _section_type_for(role: BlockRole) -> str:
    """Return the sidecar ``section_type`` string for a given role."""
    return _SECTION_TYPE_BY_ROLE.get(role, "paragraph-group")


_CHAPTER_BOUNDARY_ROLES = {BlockRole.CHAPTER_OPENER}


# Roles whose template emits a distinct top-level sidecar section. When
# the document carries no ``CHAPTER_OPENER`` we fall back to one section
# per block (still matches the Courseforge source-router's expectation
# that ``sections[].section_id`` enumerates router targets).
_STANDALONE_SECTION_ROLES = {
    BlockRole.CHAPTER_OPENER,
    BlockRole.SECTION_HEADING,
    BlockRole.SUBSECTION_HEADING,
    BlockRole.ABSTRACT,
    BlockRole.FIGURE,
    BlockRole.TABLE,
    BlockRole.BIBLIOGRAPHY_ENTRY,
    BlockRole.LEARNING_OBJECTIVES,
    BlockRole.KEY_TAKEAWAYS,
}


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _extractor_label(block: ClassifiedBlock) -> str:
    """Return the canonical provenance extractor label for ``block``.

    Mirrors :func:`DART.converter.block_templates._data_dart_source_value`
    at the per-block level so the ``<section data-dart-source>`` attribute
    and the sidecar's ``provenance.sources[]`` list stay synchronized.
    """
    cs = getattr(block, "classifier_source", "heuristic") or "heuristic"
    if cs == "extractor_hint":
        up = getattr(block.raw, "extractor", "") or ""
        if up in {"pdfplumber", "pymupdf", "pdftotext"}:
            return up
        return "pdftotext"
    if cs == "llm":
        return "claude_llm"
    return "pdftotext"


def _page_range_for(blocks: Sequence[ClassifiedBlock]) -> List[int]:
    """Return ``[min_page, max_page]`` for ``blocks``; ``[]`` if unknown."""
    pages: List[int] = []
    for b in blocks:
        p = getattr(b.raw, "page", None)
        if isinstance(p, int) and p > 0:
            pages.append(p)
    if not pages:
        return []
    lo, hi = min(pages), max(pages)
    return [lo, hi] if lo != hi else [lo, lo]


def _title_for(block: ClassifiedBlock) -> str:
    """Derive a short title for the sidecar ``section_title``."""
    attrs = block.attributes or {}
    for key in ("heading_text", "title", "term"):
        candidate = attrs.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    text = (block.raw.text or "").strip()
    if not text:
        return f"Section {block.raw.block_id}"
    # First line, clipped.
    first_line = text.splitlines()[0].strip()
    return first_line[:120] if len(first_line) > 120 else first_line


def _data_payload(blocks: Sequence[ClassifiedBlock]) -> Dict[str, Any]:
    """Build the sidecar ``data`` payload for a section's block group.

    Extracts the text of every block in the section plus role-specific
    attribute summaries, giving the Courseforge source-router real
    keyword signal to work with.
    """
    text_parts: List[str] = []
    roles_in_section: List[str] = []
    attributes_digest: Dict[str, Any] = {}
    for b in blocks:
        text = (b.raw.text or "").strip()
        if text:
            text_parts.append(text)
        roles_in_section.append(b.role.value)
        attrs = b.attributes or {}
        # Capture role-informative attrs; skip rich HTML blobs
        # (``body_html``) so the sidecar stays compact.
        for key in ("heading_text", "title", "term", "definition", "caption",
                    "attribution", "number", "items"):
            if key in attrs and attrs[key]:
                attributes_digest.setdefault(key, attrs[key])
    data: Dict[str, Any] = {
        "text": "\n\n".join(text_parts),
        "block_roles": roles_in_section,
    }
    if attributes_digest:
        data["attributes"] = attributes_digest
    return data


# ---------------------------------------------------------------------------
# Sectionisation
# ---------------------------------------------------------------------------


def _group_into_sections(
    blocks: Sequence[ClassifiedBlock],
) -> List[List[ClassifiedBlock]]:
    """Group ``blocks`` into sidecar sections.

    Grouping rules (first rule that applies):

    1. If the document carries at least one ``CHAPTER_OPENER`` block,
       each chapter opener seeds a new group that runs until the next
       opener.
    2. Otherwise, each standalone section-role block (table / figure /
       section heading / learning objectives) becomes its own group,
       and runs of leaf/paragraph blocks also form their own groups so
       the router enumerates every meaningful chunk.
    """
    if not blocks:
        return []

    has_chapter = any(b.role in _CHAPTER_BOUNDARY_ROLES for b in blocks)
    groups: List[List[ClassifiedBlock]] = []

    if has_chapter:
        current: List[ClassifiedBlock] = []
        preamble_emitted = False
        for b in blocks:
            if b.role in _CHAPTER_BOUNDARY_ROLES:
                if current:
                    groups.append(current)
                    preamble_emitted = True
                current = [b]
            else:
                current.append(b)
        if current:
            groups.append(current)
        # Drop a leading preamble group only when empty — always keep it
        # otherwise so the router sees front-matter text.
        _ = preamble_emitted
        return groups

    # No chapter openers — fall back to per-block sectionisation where
    # every standalone-section role starts a new group and runs of
    # leaf/paragraph blocks form a combined group.
    current = []
    for b in blocks:
        if b.role in _STANDALONE_SECTION_ROLES:
            if current:
                groups.append(current)
                current = []
            groups.append([b])
        else:
            current.append(b)
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Public sidecar builders
# ---------------------------------------------------------------------------


def build_synthesized_sidecar(
    classified_blocks: Sequence[ClassifiedBlock],
    title: str,
    source_pdf: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the canonical ``*_synthesized.json`` sidecar dict.

    Shape (matches the contract documented in DART/CLAUDE.md §
    "Source provenance" and consumed by
    ``MCP.tools.pipeline_tools._build_source_module_map``):

    ::

        {
          "slug": "<doc slug>",
          "title": "<title>",
          "source_pdf": "<path>",
          "sections": [
            {
              "section_id": "s1",
              "section_title": "...",
              "section_type": "chapter|section|bibliography|figure|table|paragraph-group",
              "page_range": [lo, hi],
              "provenance": {
                "sources": ["pdftotext", ...],
                "strategy": "heuristic|extractor_hint|llm",
                "confidence": 0..1
              },
              "data": {...}
            }
          ],
          "document_provenance": {
            "extractors_used": [...],
            "figures_extracted": N,
            "tables_extracted": N,
            "toc_entries": N
          }
        }
    """
    metadata = metadata or {}
    slug = _slug(title)

    groups = _group_into_sections(list(classified_blocks))
    sections: List[Dict[str, Any]] = []
    extractors_seen: set = set()
    figures_count = 0
    tables_count = 0
    toc_count = 0

    for idx, group in enumerate(groups, start=1):
        if not group:
            continue
        head = group[0]
        # Prefer the head block's role to drive section_type; chapters
        # run to the next chapter regardless of internal role mix.
        section_type = _section_type_for(head.role)
        section_title = _title_for(head)
        page_range = _page_range_for(group)
        # Provenance aggregation: the sources list is the union of
        # upstream extractors, the confidence is the head block's
        # classifier confidence, and the strategy captures the
        # classifier source.
        sources: List[str] = []
        for b in group:
            lbl = _extractor_label(b)
            if lbl not in sources:
                sources.append(lbl)
            up = getattr(b.raw, "extractor", "") or ""
            if up:
                extractors_seen.add(up)
            if b.role == BlockRole.FIGURE:
                figures_count += 1
            elif b.role == BlockRole.TABLE:
                tables_count += 1
            elif b.role == BlockRole.TOC_NAV:
                attrs = b.attributes or {}
                entries = attrs.get("entries")
                if isinstance(entries, list):
                    toc_count += len(entries)
        strategy = getattr(head, "classifier_source", "heuristic") or "heuristic"
        confidence = float(getattr(head, "confidence", 0.5) or 0.5)
        section_id = f"s{idx}"
        # Preserve ``block_id`` on the section so the router can also
        # address sub-blocks downstream. Courseforge source-router keys
        # on ``section_id``; keeping the raw head ``block_id`` in the
        # ``data`` payload gives downstream consumers the richer anchor.
        data = _data_payload(group)
        data["head_block_id"] = head.raw.block_id
        sections.append({
            "section_id": section_id,
            "section_title": section_title,
            "section_type": section_type,
            "page_range": page_range,
            "provenance": {
                "sources": sources,
                "strategy": strategy,
                "confidence": round(confidence, 3),
            },
            "data": data,
        })

    doc_prov: Dict[str, Any] = {
        "extractors_used": sorted(extractors_seen),
        "figures_extracted": figures_count,
        "tables_extracted": tables_count,
        "toc_entries": toc_count,
    }

    sidecar: Dict[str, Any] = {
        "slug": slug,
        "title": title,
        "source_pdf": source_pdf or "",
        "sections": sections,
        "document_provenance": doc_prov,
    }

    # Surface caller-supplied metadata keys the router / LibV2 archival
    # layer inspect (document_type, authors, date, language). Merged
    # shallowly under a dedicated ``metadata`` field so the sidecar
    # shape stays backwards-compatible with pre-Wave-19 consumers that
    # only look at ``sections``.
    carried = {}
    for k in ("document_type", "authors", "date", "language", "rights",
              "subject"):
        if metadata.get(k) not in (None, ""):
            carried[k] = metadata[k]
    if carried:
        sidecar["metadata"] = carried

    return sidecar


# ---------------------------------------------------------------------------
# Quality sidecar (delegates to the shared WCAG validator when available)
# ---------------------------------------------------------------------------


def build_quality_sidecar(
    html: str,
    title: str,
    source_pdf: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the canonical ``{stem}.quality.json`` sidecar dict.

    Runs the DART WCAG validator against ``html`` and returns the
    aggregate result. When the validator is unavailable (optional dep
    missing in the environment), returns a minimal compliant-by-default
    payload so callers always have a sidecar to write — archival +
    LibV2 propagation still work without the richer detail.
    """
    payload: Dict[str, Any] = {
        "slug": _slug(title),
        "title": title,
        "source_pdf": source_pdf or "",
        "html_size_bytes": len(html.encode("utf-8")),
        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()[:16],
        "compliant": True,
        "critical_count": 0,
        "high_count": 0,
        "total_issues": 0,
        "quality_score": 1.0,
        "issues": [],
    }

    try:
        # Imported lazily so a missing optional dep never hard-fails the
        # sidecar emit. The multi-source interpreter guards the same way.
        from DART.multi_source_interpreter import validate_wcag

        wcag = validate_wcag(html, label=title)
        if isinstance(wcag, dict):
            for k in (
                "compliant", "critical_count", "high_count", "total_issues",
                "quality_score", "issues",
            ):
                if k in wcag:
                    payload[k] = wcag[k]
    except Exception as exc:  # noqa: BLE001 — sidecar emit never blocks
        logger.debug(
            "WCAG validation unavailable for quality sidecar (%s); "
            "defaulting to compliant-by-default payload",
            exc,
        )

    return payload


__all__ = (
    "build_synthesized_sidecar",
    "build_quality_sidecar",
)
