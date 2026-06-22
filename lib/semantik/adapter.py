"""SemantiK v2 → Ed4All DART output-contract adapter (Phase P2a).

This is the *keystone* of the SemantiK migration
(``plans/finegrain/semantic-v2-dart-migration-2026-06-21.md`` §3): it
normalizes a Semantic v2 cascade RESULT into the HTML + sidecar shape
Ed4All's downstream contract consumers require.

Without this adapter the critical ``dart_markers`` gate blocks every run
(Semantic v2 emits NO ``role="main"`` / ``aria-labelledby`` sections /
``dart-*`` classes / ``data-dart-*`` attrs / page provenance / sidecar) and
the chunker / ``SemanticStructureExtractor`` / ``source_refs`` gate silently
lose all structure.

§3 subsection → code-section map
--------------------------------
* §3.0  single ``sid`` INVARIANT ......... :func:`_mint_sid` (the ONE shared
  minting fn) + :func:`_AdapterSection.sid` — the same value feeds
  ``aria-labelledby`` / heading ``id`` / ``data-dart-block-id`` / sidecar
  ``section_id`` / ``#fragment``.
* §3.1  four critical markers ............ :func:`_render_html`
  (``role="main"`` + ``class="dart-document"`` on ``<main>``;
  ``<section aria-labelledby … class="dart-section">`` per block; skip-link
  preserved).
* §3.2  source-provenance ................ :data:`_DATA_DART_SOURCE_VALUE`
  (the M6 ``synthesized`` decision) + placement-on-wrapper-only in
  :func:`_render_section`.
* §3.3  sourceId scheme .................. :func:`_slug` (reuses
  ``dart_slug_from_filename``) + parity in :func:`build_synthesized_sidecar`.
* §3.3a region-id determinism ............ :func:`_mint_sid` mints from the
  RAW extracted block document-order (``raw_block_index``), anchored to a
  region's FIRST raw block — never post-model region order.
* §3.4  structure ........................ :func:`_render_chapters`
  (``<article role="doc-chapter">`` + inner ``<h2>`` + ``<section>`` blocks)
  + non-content-heading filtering (:func:`_is_noncontent_heading`) so a
  single chapter never trips the >40-section collapse.
* §3.5  page-provenance ................... :func:`_format_pages` +
  ``data-dart-page-kind="physical"`` (honest; never fabricated ``printed``);
  ``#fragment == #{sid}``.
* §3.5b sidecars ......................... :func:`build_synthesized_sidecar`
  + :func:`build_quality_sidecar` (canonical shapes, ported here so this
  surface is self-contained and model-free).
* §3.6  confidence ....................... :func:`_band_confidence` (pinned
  5-point ``1.0/0.8/0.6/0.4/0.2`` band; omitted at 1.0).
* §3.7  success mapping ................... :func:`_resolve_success` +
  :func:`normalize_cascade_to_ed4all` return dict.

P2a constraint: this module is ADDITIVE — it imports Ed4All contract helpers
and (optionally) the SemantiK cascade result types, but modifies no existing
Ed4All file. The seam wiring into ``pipeline_tools.py`` is P3.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Ed4All contract helpers (the targets). These are the single sources of
# truth — REUSE, never re-implement.
from gui.services.source_page import heading_slug
from lib.validators.source_refs import dart_slug_from_filename

# Non-content-heading filter (§3.4) — reused so chapter/section emission
# drops answer-key / numeric / front-matter noise exactly like the
# extractor does, keeping a single chapter under the >40-section collapse.
from lib.semantic_structure_extractor.semantic_structure_extractor import (
    _is_noncontent_heading,
)


# ---------------------------------------------------------------------------
# §3.2 — the M6 data-dart-source decision.
#
# Grep evidence (settled this session): NO consumer of the
# ``data-dart-source`` HTML attribute VALUE branches on a specific enum
# value. The chunker (``Trainforge/chunker/helpers.py``) harvests only
# block-id / pages / page-kind — there is NO ``data-dart-source`` value
# regex. ``lib/validators/dart_markers.py`` checks presence + non-empty
# only (``_DATA_DART_SOURCE_RE`` is a presence regex; the empty check is
# ``_EMPTY_DATA_DART_SOURCE_RE``). ``_build_source_module_map`` /
# ``_dart_block_source_references`` set ``"extractor": "synthesized"`` as a
# HARDCODED python string, never read from the HTML attribute. There is no
# code-enforced allowlist (only a suggestion-text list in the validator).
#
# => Tagging every SemantiK block ``synthesized`` mis-routes NOTHING. We do
# NOT add a new ``semantik`` enum member. ``synthesized`` is the most honest
# label: SemantiK's output IS the council-synthesized combination of
# pypdfium2/pdfplumber/OCR extraction, exactly what ``synthesized`` has
# always denoted. NEVER empty (``EMPTY_DATA_DART_SOURCE`` is critical).
# ---------------------------------------------------------------------------
_DATA_DART_SOURCE_VALUE = "synthesized"

# §3.5 — page-kind is honest physical PDF pages. Semantic v2 resolves only
# physical PDF pages today; never upgrade physical→printed (RISK-A
# anti-fabrication). Absent normalizes to physical anyway (back-compat).
_DATA_DART_PAGE_KIND = "physical"

# §3.6 — pinned 5-point confidence band. Map any per-region confidence into
# exactly these points; never invent scale points.
_CONFIDENCE_BANDS = (1.0, 0.8, 0.6, 0.4, 0.2)

# §3.7 — exit_action → (success, certification_status). A bare
# ``wcag_status=="passed"`` AND wrongly hard-fails a deliberately-flagged
# ship. Honor Semantic's three exit states.
_EXIT_ACTION_MAP: Dict[str, tuple[bool, str]] = {
    "ship_with_confidence": (True, "certified"),
    "ship_with_flag": (True, "flagged"),
    "non_certified_stamp": (True, "non_certified"),
}

# RegionKind → sidecar section_type (mirrors the DART converter's
# role→section_type mapping; SemantiK region kinds are the source vocab).
_SECTION_TYPE_BY_KIND: Dict[str, str] = {
    "heading": "section",
    "paragraph": "paragraph-group",
    "list": "paragraph-group",
    "definition_list": "paragraph-group",
    "table": "table",
    "math": "section",
    "code_block": "section",
    "blockquote": "paragraph-group",
    "figure": "figure",
    "form": "section",
    "metadata_drop": "paragraph-group",
}


def _resolve_content_hash_ids() -> bool:
    """Whether ``TRAINFORGE_CONTENT_HASH_IDS=1`` is set (§3.3 hash mode)."""
    import os

    val = (os.environ.get("TRAINFORGE_CONTENT_HASH_IDS") or "").strip().lower()
    return val in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Normalized intermediate representation.
#
# The real ``PipelineV2Result`` does NOT serialize ``regions`` /
# ``feature_blocks`` into its result dict (verified: ``run_full_cascade``
# emits only ``heading_tree`` + ``html`` + telemetry — see P4 gap note in
# the report). So the adapter consumes a normalized section IR that the P3
# seam derives from the cascade's regions + feature_blocks. The synthetic
# test fixture builds this IR directly; the real seam (P3) populates it from
# ``regions[i].feature_block_indices`` → ``feature_blocks[i].raw``.
# ---------------------------------------------------------------------------


@dataclass
class _AdapterBlock:
    """One emitted block within a section (a SemantiK Region's product).

    ``raw_block_index`` is the §3.3a determinism anchor: the document-order
    index of the region's FIRST raw extracted block (pre-merge), so a model
    re-classification that re-groups regions does NOT renumber blocks.
    """

    html: str  # the inner content HTML (no wrapper) for this block
    region_kind: str  # SemantiK RegionKind value (heading/paragraph/...)
    raw_block_index: int  # §3.3a: first raw-block doc-order index
    raw_text: str = ""  # deterministic extracted text (hash input, §3.3)
    heading_text: Optional[str] = None  # drives sid via heading_slug (§3.0)
    pages: Sequence[int] = field(default_factory=tuple)  # 1-indexed physical
    confidence: Optional[float] = None  # per-region cascade confidence
    block_role: Optional[str] = None  # council/Qwen role label (§4)
    wcag_status: Optional[str] = None  # per-region gate (passed/flagged/...)
    figure_alt: Optional[str] = None  # SmolVLM2 caption (figure blocks)


@dataclass
class _AdapterChapter:
    """One chapter = one ``<article role="doc-chapter">`` (§3.4)."""

    title: str
    blocks: List[_AdapterBlock] = field(default_factory=list)


# ---------------------------------------------------------------------------
# §3.0 / §3.3 / §3.3a — the ONE shared sid minting function.
# ---------------------------------------------------------------------------


def _mint_sid(block: _AdapterBlock) -> str:
    """Derive the single deterministic ``sid`` for a block (§3.0 INVARIANT).

    Positional mode (default): heading text via the FROZEN ``heading_slug``
    (§3.5), with a positional fallback ``s{raw_block_index}`` for headingless
    blocks. ``raw_block_index`` is the RAW extracted-block document-order
    position (§3.3a) so model re-grouping never renumbers.

    Content-hash mode (``TRAINFORGE_CONTENT_HASH_IDS=1``, §3.3): a 16-hex
    hash of the DETERMINISTIC extracted text (``raw_text``) — NEVER
    post-model-rewrite HTML (which reintroduces M2 nondeterminism). The HTML
    stamp and the sidecar both call THIS fn, so they hash identically.
    """
    if _resolve_content_hash_ids():
        basis = (block.raw_text or block.heading_text or block.html or "")
        digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
        return digest

    if block.heading_text:
        slug = heading_slug(block.heading_text)
        if slug:
            return slug
    # Positional fallback — anchored to the raw-block doc-order index.
    return f"s{int(block.raw_block_index)}"


# ---------------------------------------------------------------------------
# §3.6 — confidence band mapping.
# ---------------------------------------------------------------------------


def _band_confidence(value: Optional[float]) -> Optional[float]:
    """Map an arbitrary confidence onto the pinned 5-point band (§3.6).

    Returns ``None`` (=> omit the attribute) when the value bands to ``1.0``
    or when no confidence is supplied. Snaps to the nearest band point.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    band = min(_CONFIDENCE_BANDS, key=lambda b: abs(b - v))
    if band >= 1.0:
        return None  # §3.6 — omitted when 1.0 (implicit default)
    return band


# ---------------------------------------------------------------------------
# §3.5 — page formatting.
# ---------------------------------------------------------------------------


def _format_pages(pages: Sequence[int]) -> Optional[str]:
    """Format a 1-indexed physical page set as ``"3"`` / ``"3-5"`` / ``"3,5,7"``.

    Contiguous runs collapse to a range; gaps comma-join. Returns ``None``
    when unknown (omit the attribute — never fabricate).
    """
    nums = sorted({int(p) for p in pages if isinstance(p, int) and p > 0})
    if not nums:
        return None
    if len(nums) == 1:
        return str(nums[0])
    # Contiguous run?
    if nums == list(range(nums[0], nums[-1] + 1)):
        return f"{nums[0]}-{nums[-1]}"
    return ",".join(str(n) for n in nums)


def _page_range(pages: Sequence[int]) -> List[int]:
    """Return ``[lo, hi]`` for the sidecar ``page_range``; ``[]`` if unknown."""
    nums = sorted({int(p) for p in pages if isinstance(p, int) and p > 0})
    if not nums:
        return []
    return [nums[0], nums[-1]]


def _esc_attr(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _esc_text(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# §3.1 / §3.2 / §3.4 / §3.5 — HTML rendering.
# ---------------------------------------------------------------------------


def _render_section(block: _AdapterBlock, sid: str) -> str:
    """Render one ``<section>`` wrapper for a block (§3.1/§3.2/§3.5).

    All ``data-dart-*`` attributes land on the SAME opening tag as
    ``data-dart-block-id`` (placement rule — never on leaf nodes; chunker
    same-element pairing requires it). The inner heading carries
    ``id={sid}`` so ``aria-labelledby`` resolves and ``#{sid}`` deep-links.
    """
    attrs: List[str] = [
        'class="dart-section"',
        f'aria-labelledby="{_esc_attr(sid)}"',
        f'data-dart-block-id="{_esc_attr(sid)}"',
        f'data-dart-source="{_DATA_DART_SOURCE_VALUE}"',
    ]
    pages = _format_pages(block.pages)
    if pages:
        attrs.append(f'data-dart-pages="{_esc_attr(pages)}"')
        attrs.append(f'data-dart-page-kind="{_DATA_DART_PAGE_KIND}"')
    conf = _band_confidence(block.confidence)
    if conf is not None:
        attrs.append(f'data-dart-confidence="{conf:.2f}"')
    if block.block_role:
        attrs.append(f'data-dart-block-role="{_esc_attr(block.block_role)}"')
    if block.wcag_status:
        attrs.append(f'data-dart-wcag="{_esc_attr(block.wcag_status)}"')

    # The heading (id={sid}) is the aria-labelledby target. Headingless
    # blocks still need a labelledby target → emit a visually-hidden
    # heading carrying the sid so the contract holds.
    heading_text = block.heading_text or _first_text_line(block)
    heading_html = (
        f'<h3 id="{_esc_attr(sid)}">{_esc_text(heading_text)}</h3>'
    )
    inner = block.html or ""
    return (
        f"<section {' '.join(attrs)}>\n"
        f"{heading_html}\n"
        f"{inner}\n"
        f"</section>"
    )


def _first_text_line(block: _AdapterBlock) -> str:
    """Best-effort short label for a headingless block's hidden heading."""
    text = (block.raw_text or "").strip()
    if not text:
        return f"Block {block.raw_block_index}"
    first = text.splitlines()[0].strip()
    return (first[:80] if len(first) > 80 else first) or f"Block {block.raw_block_index}"


def _render_chapters(chapters: Sequence[_AdapterChapter]) -> str:
    """Render every chapter as an ``<article role="doc-chapter">`` (§3.4)."""
    parts: List[str] = []
    for ch_idx, chapter in enumerate(chapters, start=1):
        ch_title = chapter.title or f"Chapter {ch_idx}"
        sections_html: List[str] = []
        for block in chapter.blocks:
            # §3.4 non-content-heading filtering: drop answer-key / numeric
            # / front-matter blocks so a chapter never balloons past the
            # >40-section collapse threshold.
            if block.heading_text and _is_noncontent_heading(block.heading_text):
                continue
            sid = _mint_sid(block)
            sections_html.append(_render_section(block, sid))
        article_id = f"chap-{ch_idx}"
        parts.append(
            f'<article role="doc-chapter" id="{article_id}">\n'
            f"<h2>{_esc_text(ch_title)}</h2>\n"
            + "\n".join(sections_html)
            + "\n</article>"
        )
    return "\n".join(parts)


def _render_html(chapters: Sequence[_AdapterChapter], *, title: str, lang: str) -> str:
    """Assemble the full normalized document (§3.1 four critical markers).

    Keeps the skip-link; adds ``role="main"`` + ``class="dart-document"`` to
    ``<main>``; wraps every block in an aria-labelled ``dart-section``.
    """
    body = _render_chapters(chapters)
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{_esc_attr(lang)}">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_esc_text(title)}</title>\n"
        "</head>\n"
        "<body>\n"
        '<a class="skip-link" href="#main-content">Skip to main content</a>\n'
        '<main id="main-content" role="main" class="dart-document">\n'
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


# ---------------------------------------------------------------------------
# §3.5b — sidecar builders (ported; model-free; canonical shapes).
# ---------------------------------------------------------------------------


def build_synthesized_sidecar(
    chapters: Sequence[_AdapterChapter],
    *,
    title: str,
    source_pdf: Optional[str] = None,
    slug: str,
) -> Dict[str, Any]:
    """Return the canonical ``{stem}_synthesized.json`` sidecar (§3.5b).

    PARITY INVARIANT (§3.3): every ``sections[].section_id`` here EQUALS the
    ``data-dart-block-id`` stamped in the HTML — both call :func:`_mint_sid`.
    The source_refs gate harvests its valid-ID universe from this sidecar, so
    any divergence trips ``UNRESOLVED_SOURCE_ID`` on every run.
    """
    sections: List[Dict[str, Any]] = []
    extractors_seen = {_DATA_DART_SOURCE_VALUE}
    figures_count = 0
    tables_count = 0
    for chapter in chapters:
        for block in chapter.blocks:
            if block.heading_text and _is_noncontent_heading(block.heading_text):
                continue
            sid = _mint_sid(block)
            if block.region_kind == "figure":
                figures_count += 1
            elif block.region_kind == "table":
                tables_count += 1
            data: Dict[str, Any] = {
                "text": block.raw_text or "",
                "block_roles": [block.block_role or block.region_kind],
                "head_block_id": sid,
            }
            if block.figure_alt:
                data["figure_alt"] = block.figure_alt
            conf = block.confidence if block.confidence is not None else 1.0
            sections.append(
                {
                    "section_id": sid,
                    "section_title": (
                        block.heading_text or _first_text_line(block)
                    ),
                    "section_type": _SECTION_TYPE_BY_KIND.get(
                        block.region_kind, "paragraph-group"
                    ),
                    "page_range": _page_range(block.pages),
                    "provenance": {
                        "sources": [_DATA_DART_SOURCE_VALUE],
                        "strategy": "semantik_v2",
                        "confidence": round(float(conf), 3),
                    },
                    "data": data,
                }
            )
    return {
        "slug": slug,
        "title": title,
        "source_pdf": source_pdf or "",
        "sections": sections,
        "document_provenance": {
            "extractors_used": sorted(extractors_seen),
            "figures_extracted": figures_count,
            "tables_extracted": tables_count,
            "toc_entries": 0,
        },
    }


def build_quality_sidecar(
    html: str,
    *,
    title: str,
    slug: str,
    source_pdf: Optional[str] = None,
    wcag_status: str = "passed",
    theta_score: Optional[float] = None,
    exit_action: Optional[str] = None,
    certification_status: Optional[str] = None,
    flags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return the canonical ``{stem}.quality.json`` sidecar (§3.5b).

    Self-contained (no DART WCAG validator load): the WCAG verdict + theta
    + exit decision come from the cascade result, not a re-run.
    """
    compliant = wcag_status == "passed"
    payload: Dict[str, Any] = {
        "slug": slug,
        "title": title,
        "source_pdf": source_pdf or "",
        "html_size_bytes": len(html.encode("utf-8")),
        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()[:16],
        "compliant": compliant,
        "critical_count": 0 if compliant else 1,
        "high_count": 0,
        "total_issues": 0 if compliant else 1,
        "quality_score": 1.0 if compliant else 0.0,
        "issues": [],
        # SemantiK provenance enrichment (§3.7 / §4 doc-level signals).
        "wcag_status": wcag_status,
        "theta_score": theta_score,
        "exit_action": exit_action,
        "certification_status": certification_status,
        "flags": list(flags or []),
    }
    return payload


# ---------------------------------------------------------------------------
# §3.7 — exit_action → success mapping.
# ---------------------------------------------------------------------------


def _resolve_success(exit_action: Optional[str]) -> tuple[bool, str]:
    """Map ``exit_action`` to ``(success, certification_status)`` (§3.7).

    Unknown / hard-error / missing exit actions fail closed (a mock-runtime
    or cascade error never silently ships). The ``runtime_mode=='real'``
    assertion is a SEPARATE precondition handled by the P3 seam.
    """
    if not exit_action:
        return (False, "error")
    return _EXIT_ACTION_MAP.get(exit_action, (False, "error"))


# ---------------------------------------------------------------------------
# Cascade-result → normalized-IR extraction.
# ---------------------------------------------------------------------------


def _extract_chapters_from_result(cascade_result: Any) -> List[_AdapterChapter]:
    """Pull the normalized chapter IR off a cascade result.

    Accepts (in priority order):

    1. A ``.sections`` / ``.chapters`` attribute already shaped as the
       adapter IR (the P3 seam threads this in from regions+feature_blocks;
       the synthetic test fixture builds it directly).
    2. A raw mapping/dict carrying ``"chapters"`` of dicts.

    The real ``PipelineV2Result`` does NOT carry regions/feature_blocks in
    its serialized dict (P4 gap), so the P3 seam is responsible for
    constructing chapters from ``regions[i].feature_block_indices`` →
    ``feature_blocks[i].raw`` BEFORE calling the adapter. This fn is the
    single normalization choke point.
    """
    raw_chapters = getattr(cascade_result, "chapters", None)
    if raw_chapters is None and isinstance(cascade_result, dict):
        raw_chapters = cascade_result.get("chapters")
    if raw_chapters is None:
        return []

    chapters: List[_AdapterChapter] = []
    for raw_ch in raw_chapters:
        if isinstance(raw_ch, _AdapterChapter):
            chapters.append(raw_ch)
            continue
        if not isinstance(raw_ch, dict):
            continue
        blocks: List[_AdapterBlock] = []
        for raw_b in raw_ch.get("blocks", []):
            if isinstance(raw_b, _AdapterBlock):
                blocks.append(raw_b)
            elif isinstance(raw_b, dict):
                blocks.append(_AdapterBlock(**raw_b))
        chapters.append(
            _AdapterChapter(title=raw_ch.get("title", ""), blocks=blocks)
        )
    return chapters


def _word_count(html: str) -> int:
    """Count words in the rendered HTML text (tags stripped)."""
    text = re.sub(r"<[^>]+>", " ", html)
    return len(text.split())


# ---------------------------------------------------------------------------
# §3.7 — public adapter entry point.
# ---------------------------------------------------------------------------


def normalize_cascade_to_ed4all(
    cascade_result: Any,
    *,
    pdf_stem: str,
    figures_dir: Optional[str] = None,
    canonical_course_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a Semantic v2 cascade RESULT into Ed4All's DART contract.

    Parameters
    ----------
    cascade_result
        A ``PipelineV2Result`` (or duck-typed object) carrying
        ``html`` / ``wcag_status`` / ``exit_action`` / ``theta_score`` /
        ``flags`` / ``lane_used`` plus a normalized ``chapters`` IR (the P3
        seam attaches it; the synthetic fixture builds it directly — see
        :func:`_extract_chapters_from_result`).
    pdf_stem
        The staged-HTML file stem; the ``dart:{slug}`` slug derives from it
        via ``dart_slug_from_filename`` (§3.3, gentle slug — NOT
        ``canonical_slug``).
    figures_dir
        Optional figures directory (recorded; figure copy is the P3 seam's
        job).
    canonical_course_code
        Optional course code, recorded for provenance.

    Returns
    -------
    dict
        The Ed4All tool JSON contract (§3.7): ``html`` + the required
        ``config/workflows.yaml`` keys (``output_path``/``html_path``/
        ``success``/``html_length`` are populated by the P3 seam once it
        knows the write path; here we surface ``html`` / ``html_length`` /
        ``success`` / sidecars / provenance) + ``synthesized_sidecar`` +
        ``quality_sidecar`` + ``method`` + ``word_count`` + ``wcag_status`` +
        ``exit_action`` + ``theta_score`` + ``flags`` + ``certification_status``.
    """
    slug = dart_slug_from_filename(pdf_stem)

    exit_action = getattr(cascade_result, "exit_action", None)
    if exit_action is None and isinstance(cascade_result, dict):
        exit_action = cascade_result.get("exit_action")
    wcag_status = getattr(cascade_result, "wcag_status", None)
    if wcag_status is None and isinstance(cascade_result, dict):
        wcag_status = cascade_result.get("wcag_status")
    wcag_status = wcag_status or "failed"
    theta_score = getattr(cascade_result, "theta_score", None)
    if theta_score is None and isinstance(cascade_result, dict):
        theta_score = cascade_result.get("theta_score")
    flags = getattr(cascade_result, "flags", None)
    if flags is None and isinstance(cascade_result, dict):
        flags = cascade_result.get("flags")
    flags = list(flags or [])
    lane_used = getattr(cascade_result, "lane_used", None)
    if lane_used is None and isinstance(cascade_result, dict):
        lane_used = cascade_result.get("lane_used")

    # Title: prefer the first chapter title, else the stem.
    chapters = _extract_chapters_from_result(cascade_result)
    title = chapters[0].title if chapters else pdf_stem
    lang = getattr(cascade_result, "lang", None) or "en"

    html = _render_html(chapters, title=title, lang=lang)
    success, certification_status = _resolve_success(exit_action)

    synthesized_sidecar = build_synthesized_sidecar(
        chapters,
        title=title,
        source_pdf=pdf_stem,
        slug=slug,
    )
    quality_sidecar = build_quality_sidecar(
        html,
        title=title,
        slug=slug,
        source_pdf=pdf_stem,
        wcag_status=wcag_status,
        theta_score=theta_score,
        exit_action=exit_action,
        certification_status=certification_status,
        flags=flags,
    )

    return {
        "html": html,
        "html_length": len(html),
        "word_count": _word_count(html),
        "success": success,
        "method": "semantik_v2",
        "wcag_status": wcag_status,
        "exit_action": exit_action,
        "theta_score": theta_score,
        "flags": flags,
        "lane_used": lane_used,
        "certification_status": certification_status,
        "slug": slug,
        "canonical_course_code": canonical_course_code,
        "figures_dir": figures_dir,
        "synthesized_sidecar": synthesized_sidecar,
        "quality_sidecar": quality_sidecar,
    }


__all__ = [
    "normalize_cascade_to_ed4all",
    "build_synthesized_sidecar",
    "build_quality_sidecar",
    "_AdapterBlock",
    "_AdapterChapter",
    "_mint_sid",
    "_DATA_DART_SOURCE_VALUE",
]
