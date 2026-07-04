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
import html as _html_module
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

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

# Deterministic OCR heading-candidate classifier (2026-07-03 scan audit). Both
# the live-conversion render and the ``scripts/semantik_rerender.py`` re-render
# path funnel through :func:`normalize_cascade_to_ed4all`, so demoting OCR
# furniture / garbage headings HERE fixes both without a cascade re-run.
# LaTeX control-sequence stripper (2026-07-04 scan audit — FIX 2). Reused so
# heading text minted for HTML emit is prose, not raw markup (see
# ``_sanitize_heading_text``); the SAME normalized IR re-renders identically.
from lib.semantik.math_fold import (
    escape_currency_dollars,
    escape_math_angle_brackets,
    linkify_urls,
    sanitize_body_latex,
    sanitize_math_spans,
    strip_latex_commands,
    strip_tikz_figures,
    wrap_bare_math,
)
from lib.semantik.structure_emit import (
    STRUCTURAL_ROLES,
    emit_structure,
    parse_table,
)
from lib.semantik import composite_units as _cu
from lib.ontology.taxonomy import (
    get_lexicon_confusables as _lexicon_confusables,
)
from lib.semantik.opener_classifier import (
    OPENER_ASSOCIATION_ROLE,
    OPENER_ROLES,
    ROLE_OBJECTIVES,
    classify_opener_label,
    split_interior_openers,
    split_label_only_openers,
    split_leading_opener,
)
from lib.semantik.heading_classifier import (
    SOLUTION_CSS_CLASS,
    is_apparatus_heading,
    is_chapter_running_header,
    is_decorated_solution_label,
    is_emphasis_label_heading,
    is_fused_heading,
    is_running_header,
    is_standalone_apparatus_heading,
    is_standalone_folio,
    is_watermark_garbage_heading,
    repeated_running_header_indices,
    split_interior_apparatus_heading,
    split_leading_apparatus_heading,
    strip_chapter_title_prefix,
    strip_folio_prefix,
    strip_trailing_running_header,
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
#
# Vendor-ingest discriminator (this session): the publisher-supplied
# accessible-HTML path (``lib/semantik/vendor_ingest.py``) threads
# ``source="vendor"`` through ``normalize_cascade_to_ed4all`` so its blocks
# carry ``data-dart-source="vendor"`` — the AUTHORITATIVE provenance label
# for already-accessible HTML we did NOT synthesize. Same M6 finding holds:
# no consumer branches on the VALUE, so adding the ``vendor`` value
# mis-routes nothing; ``synthesized`` stays the SemantiK default.
# ---------------------------------------------------------------------------
_DATA_DART_SOURCE_VALUE = "synthesized"

# Honest provenance labels accepted for the per-document source override.
# Suggestion-only (the dart_markers validator's allowlist is suggestion text,
# not a code-enforced enum — see lib/validators/dart_markers.py): an unknown
# value still passes the markers gate so long as it is non-empty, but we pin
# the two we emit so a typo (e.g. "vender") is caught at the adapter boundary.
_KNOWN_DATA_DART_SOURCE_VALUES = frozenset({"synthesized", "vendor"})


def _resolve_source_value(source: Optional[str]) -> str:
    """Resolve the per-document ``data-dart-source`` value (§3.2 + vendor).

    ``None``/empty → the SemantiK default ``synthesized``. A known value
    (``synthesized``/``vendor``) is honored verbatim. NEVER empty (the
    ``EMPTY_DATA_DART_SOURCE`` gate is critical), so a blank override falls
    back to the default rather than stamping ``data-dart-source=""``.
    """
    val = (source or "").strip()
    if not val:
        return _DATA_DART_SOURCE_VALUE
    return val

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
    # A7 — the visible heading level for a genuine heading block. Section
    # headings render at <h3> (default); pedagogical OPENERS promoted by the
    # opener pass render one level deeper (<h4>) so they nest correctly under
    # their section without a level skip. Additive (default 3) so every existing
    # IR builder / re-render path is unaffected.
    heading_level: int = 3
    raw_text: str = ""  # deterministic extracted text (hash input, §3.3)
    heading_text: Optional[str] = None  # drives sid via heading_slug (§3.0)
    pages: Sequence[int] = field(default_factory=tuple)  # 1-indexed physical
    confidence: Optional[float] = None  # per-region cascade confidence
    block_role: Optional[str] = None  # council/Qwen role label (§4)
    wcag_status: Optional[str] = None  # per-region gate (passed/flagged/...)
    figure_alt: Optional[str] = None  # SmolVLM2 caption (figure blocks)
    image_src: Optional[str] = None  # Part F — relative sidecar PNG path (figures)
    # SEMANTIK_OCR_CONFUSABLE_REPAIR — the render-time repaired text (verbatim
    # ``raw_text`` PLUS the gated OCR-confusable micro-edits) and the accepted-
    # edit count. ``repaired_text`` drives the emitted body + the sidecar
    # ``text`` (downstream chunker/retrieval fidelity); ``raw_text`` STAYS
    # verbatim (the content-hash sid basis). Absent (None/0) → byte-identical to
    # a no-repair block (no ``data-dart-repair`` attribute).
    repaired_text: Optional[str] = None
    ocr_repair_count: int = 0
    # Scorecard blind-spot guard (wave-19 table-regression, 2026-07-04): when
    # ``_emit_structured_bodies`` reconciles a block that DECLARED list/table/
    # definition_list but delivered no high-confidence shape down to
    # ``paragraph``, the original declared role is recorded here and stamped as
    # ``data-dart-demoted-role`` so the structure-scorecard can COUNT the honest
    # demotions (otherwise invisible post-reconciliation — a mass demotion, e.g.
    # a scrub eating every pipe row, would silently show a perfect deliver rate).
    demoted_role: Optional[str] = None
    # Wave #22 Tier-1 reading-order grammar — the sequence-derived FLOW phase of
    # a content block WITHIN a pedagogical unit: ``statement`` (an example's
    # problem, between the example opener and its solution), ``solution-steps``
    # (the worked steps after the solution opener), or ``procedure-steps`` (the
    # steps after a How-To opener). Annotation-only (stamped as ``data-dart-flow``
    # on the section wrapper); never changes text or structure. ``None`` -> no
    # attribute (byte-stable to a pre-annotation block).
    flow: Optional[str] = None


@dataclass
class _AdapterChapter:
    """One chapter = one ``<article role="doc-chapter">`` (§3.4).

    ``continuation`` marks a chapter minted by the §3.4 overflow guard
    (``cascade_ir.build_chapters_ir``): a page-spanning section that overflowed
    the per-chapter block budget spills into a continuation carrying the SAME
    title as the chapter it continues. A continuation MUST NOT re-emit a visible
    ``<h2>`` heading — downstream ``SemanticStructureExtractor`` reads every
    heading element as a distinct section, so a repeated "(cont.)" heading
    inflates the chapter/section hierarchy with phantom pseudo-sections. The
    renderer therefore demotes a continuation's title to an aria-hidden
    presentation ``<div>`` (visual continuity cue, NOT in the heading stream) so
    each real section title appears in the HTML heading stream exactly once.
    """

    title: str
    blocks: List[_AdapterBlock] = field(default_factory=list)
    continuation: bool = False


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


def _emitted_blocks(
    chapters: Sequence["_AdapterChapter"],
) -> "Iterator[_AdapterBlock]":
    """Yield blocks in canonical EMIT order, applying the shared drop filters.

    The single source of truth for which blocks reach the HTML / sidecar (and
    in what order), so the ``data-dart-block-id`` uniquifier assigns identical
    ids to both renders (§3.3 parity). Mirrors the furniture / non-content
    filters in ``_render_chapters`` + ``build_synthesized_sidecar`` exactly.
    """
    for chapter in chapters:
        for block in chapter.blocks:
            if _is_furniture_block(block):
                continue
            if (
                block.heading_text
                and not _is_apparatus_promoted(block)
                and not _is_opener_promoted(block)
                and _is_noncontent_heading(block.heading_text)
            ):
                continue
            yield block


def _mint_unique_sids(
    chapters: Sequence["_AdapterChapter"],
) -> Dict[int, str]:
    """Minor 5 — per-block UNIQUE sid map (keyed by ``id(block)``).

    ``_mint_sid`` is deterministic per block, so two headings sharing a slug
    (the SAME section title on two pages) or two blocks with identical
    content-hash text collide on ``data-dart-block-id`` — breaking
    ``dart:{slug}#{block_id}`` anchor uniqueness (9 collisions on the audited
    chapter). This walks the emit-order block stream ONCE and appends a stable
    ``-2`` / ``-3`` … suffix to each repeat of a base sid (the first occurrence
    keeps the bare base), guarding against a pre-existing literal ``base-2``.
    Both the HTML render and the sidecar consult this map, so the ids stay in
    lockstep. Deterministic per (PDF, flags): same input → same id set.
    """
    used: set[str] = set()
    out: Dict[int, str] = {}
    for block in _emitted_blocks(chapters):
        base = _mint_sid(block)
        candidate = base
        n = 1
        while candidate in used:
            n += 1
            candidate = f"{base}-{n}"
        used.add(candidate)
        out[id(block)] = candidate
    return out


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


# Region kinds that carry a GENUINE document heading (the SemantiK structure
# graph's only heading kind is ``"heading"`` — see
# ``SemantiK/dart_semantic/structure_graph.py::REGION_KINDS``). Only a block
# of one of these kinds, AND carrying real ``heading_text``, gets a VISIBLE
# ``<h3>``; every other block satisfies the aria-labelledby contract with a
# visually-hidden label so we never fabricate a visible heading (the
# duplication / spurious-h3 defect).
_HEADING_REGION_KINDS = frozenset({"heading"})


def _is_heading_block(block: _AdapterBlock) -> bool:
    """Whether a block is a GENUINE heading (visible ``<h3>``) vs a content
    block that gets only a visually-hidden aria-labelledby label.

    True iff the block's region is a heading kind AND it carries non-empty
    ``heading_text``. A content/list/table/figure block — even one the cascade
    happened to stamp ``heading_text`` on — is NOT a visible heading, so its
    body is never duplicated into a fabricated ``<h3>``.
    """
    return bool(block.heading_text) and block.region_kind in _HEADING_REGION_KINDS


# ---------------------------------------------------------------------------
# LaTeX-markup heading sanitation (2026-07-04 scan audit — FIX 2).
# ---------------------------------------------------------------------------
# Scan-OCR / VLM-fused extraction sometimes types a heading string as raw LaTeX
# markup (``\textbf{EXAMPLE 1.58}``, ``$\checkmark \text{ Solution }$``) or even
# a stray table row typed as a heading (``14 \text{ ft} &amp; \\ \hline``).
# Because the chunker derives a chunk's ``section_heading`` from the emitted
# VISIBLE heading TEXT, that markup poisons downstream section attribution.
# Every heading-text -> HTML seam in this module therefore routes the string
# through ``_sanitize_heading_text``, and table-row-shaped / empty-after-strip
# headings are DEMOTED (rendered as body, never a visible ``<h*>``).
# Deterministic; stdlib + ``math_fold`` only, so the SAME normalized IR
# re-renders to byte-identical sanitized HTML.

# A heading that, in its ORIGINAL (pre-strip) form, carries a table-cell
# separator (``&`` / ``&amp;``), a rule command (``\hline``), or a column pipe
# is a mis-typed table row, not a section title.
_TABLE_ROW_HEADING_RE = re.compile(r"&amp;|&|\\hline|\|")


def _sanitize_heading_text(raw: Optional[str]) -> str:
    r"""Strip LaTeX markup from a heading string minted for HTML emit.

    Reuses ``math_fold.strip_latex_commands`` (the prose-safe control-sequence
    stripper): ``\text{X}`` / ``\textbf{X}`` -> ``X``, ``\checkmark`` dropped,
    ``$...$`` unwrapped, residual bare commands (``\hline``, ``\\``) dropped,
    then collapses whitespace. A strict no-op on markup-free prose (no
    backslash / brace / ``$`` -> unchanged modulo collapsed whitespace).
    """
    if not raw:
        return ""
    return " ".join(strip_latex_commands(raw).split())


def _is_table_row_heading(raw: Optional[str]) -> bool:
    r"""Whether a heading string is a mis-typed table row / empty-after-strip.

    True when the ORIGINAL string carries a table-cell / rule marker
    (``&`` / ``&amp;`` / ``\hline`` / ``|``) OR sanitizing it yields empty text.
    Such a block is DEMOTED from a visible heading to a paragraph body so its
    markup never becomes a chunk ``section_heading``.
    """
    if not raw:
        return True
    if _TABLE_ROW_HEADING_RE.search(raw):
        return True
    return not _sanitize_heading_text(raw)


def _is_furniture_block(block: _AdapterBlock) -> bool:
    """Whether a block is ``metadata_drop`` page furniture suppressed from emit.

    A ``metadata_drop``-role block is page furniture — a running header/footer
    ("Chapter 9 Roots and Radicals 1035"), a copyright line ("This OpenStax
    book is available for free at …"), an OCR title-page fragment — that the
    upstream deterministic passes (``structure_graph._detect_running_headers``
    +  ``deterministic_structure.clean_structure``) re-tagged out of the content
    stream. It carries NO teaching content and MUST NOT emit body HTML.

    Root-cause fix (SemantiK furniture-emission defect): the assembler drop path
    (``assembler/pass_9a.py`` — "a metadata_drop region is ALWAYS dropped from
    the render") was not mirrored on THIS adapter path, so on OCR-sourced scans
    the running header / footer leaked into 198 ``<p>`` bodies (the footer 61×,
    the running header 136×). Filtered from BOTH the rendered HTML and the
    ``build_synthesized_sidecar`` so ``data-dart-block-id`` parity (the
    source_refs gate's valid-ID universe) is preserved.
    """
    return (
        block.region_kind == "metadata_drop"
        or (block.block_role or "") == "metadata_drop"
    )


def _render_section(
    block: _AdapterBlock, sid: str, *, source_value: str = _DATA_DART_SOURCE_VALUE
) -> str:
    """Render one ``<section>`` wrapper for a block (§3.1/§3.2/§3.5).

    All ``data-dart-*`` attributes land on the SAME opening tag as
    ``data-dart-block-id`` (placement rule — never on leaf nodes; chunker
    same-element pairing requires it).

    B4 (2026-07-04 end-user-HTML audit) — a genuine heading block gets a
    VISIBLE ``<hN>`` (``<h3>`` for a section heading, ``<h4>`` for an A7
    pedagogical opener) that is the ``aria-labelledby`` target: the section is
    a region landmark with a DESCRIPTIVE name. A NON-heading content block
    (paragraph/list/figure OR a demoted table-row heading) carries the ``sid``
    as a bare ``id`` on the ``<section>`` (so ``#{sid}`` still deep-links) but
    NO ``aria-labelledby`` and NO ``sr-only`` label — so it is NOT a landmark
    named "Paragraph block" (the ~5,000 generic-region-landmark defect). This
    does not affect the chunker (it keys on the ``data-dart-block-id``
    attribute, and already skips ``sr-only`` subtrees — removing a label it was
    already skipping changes no chunk text). ``source_value`` is the resolved
    provenance discriminator (``synthesized`` for SemantiK; ``vendor`` for
    publisher-supplied HTML).
    """
    attrs: List[str] = [
        'class="dart-section"',
        # index 1 reserved: aria-labelledby (heading blocks) OR id (content).
        f'data-dart-block-id="{_esc_attr(sid)}"',
        f'data-dart-source="{_esc_attr(source_value)}"',
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
    # Scorecard blind-spot guard — a block whose declared list/table/
    # definition_list role was reconciled to paragraph (no delivered shape)
    # carries the original declared role so the structure-scorecard can COUNT
    # the honest demotion (post-reconciliation it is otherwise invisible).
    if block.demoted_role:
        attrs.append(
            f'data-dart-demoted-role="{_esc_attr(block.demoted_role)}"'
        )
    # SEMANTIK_OCR_CONFUSABLE_REPAIR — a block whose body carries gated OCR
    # micro-repairs is stamped on the SAME opening tag as data-dart-block-id
    # (placement rule — never a leaf node) so a downstream text-vs-source
    # auditor can EXEMPT the (data-dart-repair-annotated) diff. Present ONLY
    # when >=1 edit landed → byte-stable to a no-repair block otherwise.
    if block.repaired_text is not None and block.ocr_repair_count > 0:
        attrs.append('data-dart-repair="ocr-confusable"')
        attrs.append(f'data-dart-repair-count="{int(block.ocr_repair_count)}"')
    # A7 — a promoted pedagogical opener carries a machine-readable role so a
    # downstream consumer (chunker boundary / retrieval) can key on the
    # structure without string-matching the visible label.
    if _is_opener_promoted(block):
        attrs.append(f'data-dart-opener="{_esc_attr(block.block_role or "")}"')
    # Wave #22 Tier-1 — the reading-order FLOW phase (statement / solution-steps
    # / procedure-steps) on a content block inside a pedagogical unit. Present
    # only when the sequence grammar assigned one (byte-stable otherwise).
    if block.flow:
        attrs.append(f'data-dart-flow="{_esc_attr(block.flow)}"')

    raw_heading = block.heading_text or ""
    heading_block = _is_heading_block(block)
    # FIX 2 — a genuine heading whose (pre-strip) text is a mis-typed table row
    # or is empty once LaTeX markup is stripped is DEMOTED: no visible <hN>,
    # so its markup never poisons the chunker's derived section_heading.
    demote_heading = heading_block and _is_table_row_heading(raw_heading)
    inner = block.html or ""

    if heading_block and not demote_heading:
        # Genuine heading region → a VISIBLE <hN> is the aria-labelledby target
        # (a descriptively-named region landmark). A7 openers render at <h4>
        # (one level under their <h3> section); section headings stay <h3>. Its
        # LaTeX markup is stripped so the chunker's derived section_heading is
        # clean prose. Its body html (if any) follows.
        attrs.insert(1, f'aria-labelledby="{_esc_attr(sid)}"')
        level = int(getattr(block, "heading_level", 3) or 3)
        level = level if 2 <= level <= 6 else 3
        label_html = (
            f'<h{level} id="{_esc_attr(sid)}">'
            f"{_esc_text(_sanitize_heading_text(raw_heading))}</h{level}>"
        )
    else:
        # B4 — content / list / figure block OR a DEMOTED table-row / empty
        # heading: carry the sid as a bare id on the <section> so ``#{sid}``
        # deep-links, but emit NO aria-labelledby and NO sr-only label, so the
        # section is not a generic "Paragraph block" region landmark.
        attrs.insert(1, f'id="{_esc_attr(sid)}"')
        label_html = ""
        if demote_heading and not inner:
            # Preserve the demoted heading's sanitized text as a paragraph body
            # so content isn't dropped (a genuinely-empty demote emits nothing).
            sanitized = _sanitize_heading_text(raw_heading)
            if sanitized:
                inner = f"<p>{_esc_text(sanitized)}</p>"
    body = "\n".join(part for part in (label_html, inner) if part)
    return f"<section {' '.join(attrs)}>\n{body}\n</section>"


def _first_text_line(block: _AdapterBlock) -> str:
    """Best-effort short label for a headingless block's hidden heading."""
    text = (block.raw_text or "").strip()
    if not text:
        return f"Block {block.raw_block_index}"
    first = text.splitlines()[0].strip()
    return (first[:80] if len(first) > 80 else first) or f"Block {block.raw_block_index}"


def _wrap_callout_group(parts: Sequence[str], role: str) -> str:
    """Wrap an opener + its following content sections in ONE boxed container.

    The ``data-dart-opener-group`` attribute mirrors the opener's
    ``data-dart-opener`` value so ``dart_content.css`` can draw a SINGLE box that
    encloses the whole pedagogical unit (opener label + steps/solution/prose),
    instead of the opener heading floating in its own box while the content spills
    outside it. Standalone (no-CSS) re-render output is unaffected — the wrapper
    is an inert ``<div>``.
    """
    inner = "\n".join(parts)
    return (
        f'<div class="dart-callout-group" '
        f'data-dart-opener-group="{_esc_attr(role)}">\n{inner}\n</div>'
    )


# Wave #22 Tier-2 — human-readable name for each composite-unit type, used as
# the ``aria-label`` fallback when a unit's lead item has no heading id (a
# ``<dl>``-led definition_group / figure-led figure_group) so the ``role="group"``
# still carries an accessible name (WCAG 4.1.2).
_UNIT_LABELS: Dict[str, str] = {
    "worked_example": "Worked example",
    "section_opener": "Section opener",
    "procedure": "Procedure",
    "exercise_set": "Exercise set",
    "definition_group": "Definition group",
    "figure_group": "Figure group",
}

#: A short standalone prose block (<= this many words) is treated as a candidate
#: exercise-set DIRECTIVE lead — but only grouped when immediately followed by a
#: ``dart-exercise-list`` (the planner's guard), so the cap keeps it conservative.
_DIRECTIVE_MAX_WORDS = 45


def _standalone_unit_role(block: _AdapterBlock, html: str) -> Optional[str]:
    """Abstract composite-unit role for a NON-heading standalone content section.

    Shape-driven (never subject vocabulary): a delivered ``<dl>`` -> definition,
    a ``figure`` region -> figure, a ``dart-exercise-list`` -> exercise_list, a
    short prose -> directive (exercise-set lead candidate). Everything else is
    inert prose (``None``) — never grouped.
    """
    if block.region_kind == "figure":
        return _cu.ROLE_FIGURE
    if "<dl" in html:
        return _cu.ROLE_DEFINITION
    if 'class="dart-exercise-list"' in html:
        return _cu.ROLE_EXERCISE_LIST
    text = re.sub(r"<[^>]+>", " ", html)
    n = len(text.split())
    if 0 < n <= _DIRECTIVE_MAX_WORDS:
        return _cu.ROLE_DIRECTIVE
    return None


def _wrap_composite_unit(
    parts: Sequence[str], unit_type: str, lead_sid: Optional[str]
) -> str:
    """Wrap the constituent rendered items of a composite unit (Tier-2).

    Emits ``<section class="dart-unit dart-unit-<type>" data-dart-unit="<type>"
    role="group" …>`` with an accessible name: ``aria-labelledby`` -> the lead
    item's heading id when present, else ``aria-label`` -> the unit-type name.
    """
    inner = "\n".join(parts)
    if lead_sid:
        name = f'aria-labelledby="{_esc_attr(lead_sid)}"'
    else:
        name = f'aria-label="{_esc_attr(_UNIT_LABELS.get(unit_type, unit_type))}"'
    return (
        f'<section class="dart-unit dart-unit-{unit_type}" '
        f'data-dart-unit="{_esc_attr(unit_type)}" role="group" {name}>\n'
        f"{inner}\n</section>"
    )


def _apply_composite_units(
    item_html: List[str],
    item_meta: List["_cu.UnitItem"],
    item_lead: List[Optional[str]],
) -> List[str]:
    """Plan + wrap composite units over one section's top-level item stream.

    Runs the pure :func:`composite_units.plan_units` planner over the item
    metadata and rebuilds the HTML list: a matched span is wrapped in a
    ``dart-unit`` ``<section>``; a pass-through span emits its item verbatim.
    """
    out: List[str] = []
    for span in _cu.plan_units(item_meta):
        if span.unit_type is None:
            out.append(item_html[span.start])
            continue
        lead_sid = (
            item_lead[span.lead_index]
            if span.lead_index is not None
            else None
        )
        out.append(
            _wrap_composite_unit(
                item_html[span.start : span.end], span.unit_type, lead_sid
            )
        )
    return out


def _render_chapters(
    chapters: Sequence[_AdapterChapter],
    *,
    source_value: str = _DATA_DART_SOURCE_VALUE,
) -> str:
    """Render every chapter as an ``<article role="doc-chapter">`` (§3.4)."""
    parts: List[str] = []
    sid_map = _mint_unique_sids(chapters)  # Minor 5 — collision-free block ids
    for ch_idx, chapter in enumerate(chapters, start=1):
        ch_title = chapter.title or f"Chapter {ch_idx}"
        # Callout grouping (end-user-HTML audit): a promoted opener heading
        # (How To / Example / Solution / Try It / Objectives / Be Prepared) plus
        # the content blocks that FOLLOW it are collected into one boxed
        # container div until the next boundary — the next opener, a genuine
        # section/chapter heading, or the chapter end. Conservative: a real
        # <h3>/<h2> heading always closes the group (so a group never crosses a
        # section boundary) and the per-chapter loop never crosses a chapter.
        #
        # Wave #22 Tier-2 — the top-level items (callout-groups + standalone
        # sections) are also collected with their abstract composite-unit role so
        # a run of adjacent siblings that form one pedagogical whole (a worked
        # example + its solution + practice; an objectives + readiness opener) is
        # wrapped in a ``dart-unit`` group. Units are planned per genuine-heading
        # SEGMENT so a unit never crosses a section boundary.
        item_html: List[str] = []
        item_meta: List["_cu.UnitItem"] = []
        item_lead: List[Optional[str]] = []
        sections_html: List[str] = []
        group_parts: List[str] = []
        group_role: str = ""
        group_lead_sid: Optional[str] = None
        group_members: int = 0

        def _flush_group() -> None:
            nonlocal group_parts, group_role, group_lead_sid, group_members
            if not group_parts:
                return
            item_html.append(_wrap_callout_group(group_parts, group_role))
            item_meta.append(
                _cu.UnitItem(
                    role=OPENER_ASSOCIATION_ROLE.get(group_role, group_role),
                    boundary=False,
                    members=group_members,
                    has_heading=True,
                )
            )
            item_lead.append(group_lead_sid)
            group_parts = []
            group_role = ""
            group_lead_sid = None
            group_members = 0

        def _flush_segment() -> None:
            # Plan units over the accumulated items, then reset the item buffer.
            nonlocal item_html, item_meta, item_lead
            _flush_group()
            if item_meta:
                sections_html.extend(
                    _apply_composite_units(item_html, item_meta, item_lead)
                )
            item_html, item_meta, item_lead = [], [], []

        for block in chapter.blocks:
            # Furniture drop: a metadata_drop-role block (running header /
            # footer / copyright line) is page furniture — never emit its body
            # (mirrors assembler pass_9a). Filtered here AND in the sidecar so
            # block-id parity holds.
            if _is_furniture_block(block):
                continue
            # §3.4 non-content-heading filtering: drop answer-key / numeric
            # / front-matter blocks so a chapter never balloons past the
            # >40-section collapse threshold.
            if (
                block.heading_text
                and not _is_apparatus_promoted(block)
                and not _is_opener_promoted(block)
                and _is_noncontent_heading(block.heading_text)
            ):
                continue
            sid = sid_map[id(block)]
            section_html = _render_section(
                block, sid, source_value=source_value
            )
            is_opener = _is_opener_promoted(block)
            is_genuine_heading = _is_heading_block(block) and not is_opener
            if is_genuine_heading:
                # A genuine section heading closes any open unit segment, is
                # emitted as its own boundary item, and starts a fresh segment.
                _flush_segment()
                sections_html.append(section_html)
            elif is_opener:
                # Opener → flush any open callout-group, then open a new one.
                _flush_group()
                group_parts = [section_html]
                group_role = block.block_role or ""
                group_lead_sid = sid
                group_members = 1
            elif group_parts:
                # Content block under an open opener → fold into its box.
                group_parts.append(section_html)
                group_members += 1
            else:
                # Standalone content section → a top-level item with a derived
                # composite-unit role (dl / figure / exercise-list / directive).
                _flush_group()
                item_html.append(section_html)
                item_meta.append(
                    _cu.UnitItem(
                        role=_standalone_unit_role(block, section_html),
                        boundary=False,
                        members=1,
                        has_heading=False,
                    )
                )
                item_lead.append(None)
        _flush_segment()  # trailing group + items at chapter end
        article_id = f"chap-{ch_idx}"
        # A continuation chapter (§3.4 overflow spill) carries the SAME title as
        # the chapter it continues. Emitting it as a visible <h2> mints a
        # duplicate heading that the downstream SemanticStructureExtractor reads
        # as a NEW section (chapter/section hierarchy pollution — the 8× "Chapter
        # Outline (cont.)" pseudo-section defect). Demote it to an aria-hidden
        # presentation <div> (NOT a heading element) so the visual "… (cont.)"
        # continuity cue survives while the HTML heading stream carries each real
        # section title exactly once.
        # FIX 2 — strip LaTeX markup from the chapter title too (it renders as a
        # visible <h2> the chunker reads for section attribution). Fall back to
        # the raw title only if sanitizing empties it (pure-symbol edge).
        ch_title_display = _sanitize_heading_text(ch_title) or ch_title
        if getattr(chapter, "continuation", False):
            # Round-6: the ``:: MEDIA :: …`` colon-run residue (ch04 / ch07
            # continuation banners) escapes the marker scrub — that fold runs on
            # block bodies, never on the chapter-title / continuation-banner path.
            # These banners are aria-hidden/presentational (user-invisible) but the
            # source stays clean, so fold the ``::`` marker runs + stray gutter
            # glyphs here with the SAME conservative fold as the block path.
            banner_text = (
                _scrub_marker_artifacts(ch_title_display, html=False)
                or ch_title_display
            )
            heading_html = (
                f'<div class="dart-continuation" role="presentation" '
                f'aria-hidden="true">{_esc_text(banner_text)}</div>'
            )
        else:
            heading_html = f"<h2>{_esc_text(ch_title_display)}</h2>"
        parts.append(
            f'<article role="doc-chapter" id="{article_id}">\n'
            f"{heading_html}\n"
            + "\n".join(sections_html)
            + "\n</article>"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Table-of-contents nav (exemplar-parity wave — §4 / A-nav).
# ---------------------------------------------------------------------------
def _resolve_emit_toc() -> bool:
    """Whether the ``SEMANTIK_EMIT_TOC`` nav emit is on (default ON)."""
    import os

    val = (os.environ.get("SEMANTIK_EMIT_TOC") or "").strip().lower()
    return val not in {"0", "false", "no", "off"}


def _norm_toc_key(text: Optional[str]) -> str:
    """Case/whitespace-insensitive comparison key for a TOC entry title.

    Collapses internal whitespace runs, strips, and case-folds so two entries
    that differ only in casing / spacing compare equal (the TOC-quality dedup
    contract). Empty / whitespace-only titles fold to ``""`` (never deduped —
    they carry no comparable text).
    """
    return " ".join((text or "").split()).casefold()


def _build_toc_html(
    chapters: Sequence[_AdapterChapter], sid_map: Dict[int, str]
) -> str:
    """Build a ``<nav class="toc">`` anchor list from the h2/h3 heading tree.

    Each non-continuation chapter (``<h2 id="chap-N">``) becomes a top-level
    ``<li>`` linking ``#chap-N``, nesting an ``<ol>`` of its genuine ``<h3>``
    section headings (linking each section's ``#{sid}``). Openers (``<h4>``) are
    excluded — the TOC mirrors the h2/h3 tree only. Uses the SAME
    :func:`_mint_unique_sids` map as the render so every anchor resolves. Returns
    "" when there are no chapters (a fragment mints no TOC).
    """
    entries: List[tuple[str, str, List[tuple[str, str]]]] = []
    for ch_idx, chapter in enumerate(chapters, start=1):
        if getattr(chapter, "continuation", False):
            continue
        raw_title = chapter.title or f"Chapter {ch_idx}"
        ch_title = _sanitize_heading_text(raw_title) or raw_title
        sections: List[tuple[str, str]] = []
        for block in chapter.blocks:
            if _is_furniture_block(block) or not _is_heading_block(block):
                continue
            if (
                block.heading_text
                and not _is_apparatus_promoted(block)
                and not _is_opener_promoted(block)
                and _is_noncontent_heading(block.heading_text)
            ):
                continue
            raw_heading = block.heading_text or ""
            if _is_table_row_heading(raw_heading):
                continue
            level = int(getattr(block, "heading_level", 3) or 3)
            if level != 3 or _is_opener_promoted(block):
                continue  # only <h3> sections enter the TOC
            sections.append(
                (sid_map[id(block)], _sanitize_heading_text(raw_heading))
            )
        entries.append((f"chap-{ch_idx}", ch_title, sections))
    if not entries:
        return ""
    # TOC-quality dedup (end-user-HTML audit, ch09 shots) — a SINGLE global
    # seen-set walked in TOC document order (chapter, then its sections, then the
    # next chapter …) subsumes all three rules with "first occurrence wins":
    #  (a)/(c) ancestor — the chapter key is added BEFORE its sections, so a
    #          section identical to its parent chapter is dropped;
    #  (b)     same-level — an identical sibling chapter / section dedupes
    #          against the earlier one;
    #  (ITEM 3) cross-parent — a section that already appeared under ANY earlier
    #          chapter (e.g. "9.1 Simplify and Use Square Roots" listed under both
    #          its own chapter AND "Chapter 9 Review") is dropped globally.
    # Anchors keep pointing at the FIRST occurrence (the kept one).
    items: List[str] = []
    global_seen: set[str] = set()
    for href, title, sections in entries:
        ch_key = _norm_toc_key(title)
        if ch_key and ch_key in global_seen:
            continue  # (b) duplicate chapter title anywhere earlier
        if ch_key:
            global_seen.add(ch_key)
        kept: List[tuple[str, str]] = []
        for sid, t in sections:
            key = _norm_toc_key(t)
            if key and key in global_seen:
                continue  # ancestor / same-level / cross-parent duplicate
            if key:
                global_seen.add(key)
            kept.append((sid, t))
        inner = ""
        if kept:
            inner = (
                "<ol>"
                + "".join(
                    f'<li><a href="#{_esc_attr(sid)}">{_esc_text(t)}</a></li>'
                    for sid, t in kept
                )
                + "</ol>"
            )
        items.append(
            f'<li><a href="#{_esc_attr(href)}">{_esc_text(title)}</a>{inner}</li>'
        )
    return (
        '<nav class="toc" aria-label="Contents">\n<ol>'
        + "".join(items)
        + "</ol>\n</nav>"
    )


def _render_html(
    chapters: Sequence[_AdapterChapter],
    *,
    title: str,
    lang: str,
    source_value: str = _DATA_DART_SOURCE_VALUE,
) -> str:
    """Assemble the full normalized document (§3.1 four critical markers).

    Keeps the skip-link; adds ``role="main"`` + ``class="dart-document"`` to
    ``<main>``; wraps every block in an aria-labelled ``dart-section``.
    """
    body = _render_chapters(chapters, source_value=source_value)
    # §3.1 WCAG 2.4.6 — a single document-title <h1> at the top of <main>,
    # from the SAME source as <title>. Chapters stay at <h2> and sections at
    # <h3>, giving proper h1 > h2 > h3 nesting. ADDITIVE: it does NOT renumber
    # existing headings, so downstream Courseforge-staging / chunker that read
    # the h2/h3 section structure are unaffected (they key off <article
    # role="doc-chapter"> <h2> and <section> blocks, not the new <h1>).
    # FIX 2 — the document title feeds both the visible <h1> and <head><title>;
    # strip any LaTeX markup so neither carries raw markup (kept in lockstep so
    # the two stay same-source). No-op on a clean title.
    display_title = _sanitize_heading_text(title) or title
    h1_html = f"<h1>{_esc_text(display_title)}</h1>"
    # §4 — a Contents nav after the <h1> (SEMANTIK_EMIT_TOC, default ON). Pure
    # addition: anchors reference EXISTING chapter/section ids (mints no new id,
    # adds no heading), so the heading hierarchy + landmark contract are intact.
    toc_html = (
        _build_toc_html(chapters, _mint_unique_sids(chapters))
        if _resolve_emit_toc()
        else ""
    )
    head_block = f"{h1_html}\n{toc_html}\n" if toc_html else f"{h1_html}\n"
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{_esc_attr(lang)}">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_esc_text(display_title)}</title>\n"
        "</head>\n"
        "<body>\n"
        '<a class="skip-link" href="#main-content">Skip to main content</a>\n'
        '<main id="main-content" role="main" class="dart-document">\n'
        f"{head_block}"
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
    source_value: str = _DATA_DART_SOURCE_VALUE,
) -> Dict[str, Any]:
    """Return the canonical ``{stem}_synthesized.json`` sidecar (§3.5b).

    PARITY INVARIANT (§3.3): every ``sections[].section_id`` here EQUALS the
    ``data-dart-block-id`` stamped in the HTML — both call :func:`_mint_sid`.
    The source_refs gate harvests its valid-ID universe from this sidecar, so
    any divergence trips ``UNRESOLVED_SOURCE_ID`` on every run.
    ``source_value`` matches the HTML ``data-dart-source`` discriminator so
    the sidecar provenance agrees with the markup (``vendor`` vs the SemantiK
    ``synthesized`` default).
    """
    sections: List[Dict[str, Any]] = []
    extractors_seen = {source_value}
    figures_count = 0
    tables_count = 0
    sid_map = _mint_unique_sids(chapters)  # Minor 5 — same map the HTML uses
    for chapter in chapters:
        for block in chapter.blocks:
            # Mirror the HTML furniture drop so the sidecar's valid-ID universe
            # (source_refs gate) matches the emitted block-id set exactly.
            if _is_furniture_block(block):
                continue
            if (
                block.heading_text
                and not _is_apparatus_promoted(block)
                and not _is_opener_promoted(block)
                and _is_noncontent_heading(block.heading_text)
            ):
                continue
            sid = sid_map[id(block)]
            if block.region_kind == "figure":
                figures_count += 1
            elif block.region_kind == "table":
                tables_count += 1
            # SEMANTIK_OCR_CONFUSABLE_REPAIR — the chunker/retrieval sidecar text
            # is the REPAIRED string (the point of the pass: retrieval fidelity),
            # with a mirrored ``repair`` object so the ORIGINAL is reconstructible
            # (provenance amended, not faked). ``raw_text`` (the sid basis) is
            # UNTOUCHED. Absent → byte-identical to a no-repair sidecar.
            repaired = block.repaired_text
            has_repair = repaired is not None and block.ocr_repair_count > 0
            data: Dict[str, Any] = {
                "text": (repaired if has_repair else block.raw_text) or "",
                "block_roles": [block.block_role or block.region_kind],
                "head_block_id": sid,
            }
            if has_repair:
                data["repair"] = {
                    "type": "ocr-confusable",
                    "n_edits": int(block.ocr_repair_count),
                    "original_text": block.raw_text or "",
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
                        "sources": [source_value],
                        "strategy": (
                            "vendor_ingest"
                            if source_value == "vendor"
                            else "semantik_v2"
                        ),
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


# ---------------------------------------------------------------------------
# OCR heading-furniture normalization (2026-07-03 scan audit).
#
# Runs BEFORE title selection + render so BOTH the live-conversion path and the
# re-render path (--from-html / --ir) shed the OCR defects. Mutates the adapter
# IR in place (dataclasses are mutable; the IR is single-use per call):
#   * decorated run-in "Solution" heading block  → paragraph (pedagogy-solution)
#   * scanner-watermark heading block             → paragraph (text preserved)
#   * page-numbered running-header heading block  → metadata_drop (furniture)
#   * running-header / repeat / watermark chapter TITLE → demoted out of the
#     heading stream via the existing continuation presentation-<div> path (the
#     blocks under the phantom chapter are untouched).
# ---------------------------------------------------------------------------


def _demote_heading_block_to_paragraph(
    block: _AdapterBlock, *, css_class: Optional[str] = None
) -> None:
    r"""Demote a heading ``block`` to a paragraph carrying its text verbatim.

    The heading text becomes a ``<p>`` body (optionally class-hinted) and
    ``heading_text`` is cleared so the block renders as body prose (no visible
    ``<h3>``, no duplicate ``id``) and re-mints a positional sid.

    B3+ — the demoted body is built AFTER the ``_sanitize_block_body_latex`` pass
    (this runs in the heading-furniture loop), so it sanitizes + wraps its own
    text here (mirroring that pass) rather than shipping a raw ``\textbf`` /
    bare-``\sqrt`` / stray-``$`` heading blob into a ``<p>`` body.
    """
    text = block.heading_text or ""
    cls = f' class="{_esc_attr(css_class)}"' if css_class else ""
    block.html = wrap_bare_math(
        sanitize_body_latex(f"<p{cls}>{_esc_text(text)}</p>", html=True),
        html=True,
    )
    block.region_kind = "paragraph"
    block.heading_text = None


def _drop_heading_block_as_furniture(block: _AdapterBlock) -> None:
    """Re-tag a running-header heading ``block`` as suppressed page furniture."""
    block.region_kind = "metadata_drop"
    block.block_role = "metadata_drop"


# Block-role marker for a Defect-4 apparatus paragraph promoted to a heading.
# Such a heading BYPASSES the ``_is_noncontent_heading`` emit filter (which
# otherwise drops "Practice Test" / "Review Exercises" as EOC noise).
_APPARATUS_HEADING_ROLE = "apparatus_heading"


def _is_apparatus_promoted(block: _AdapterBlock) -> bool:
    """Whether ``block`` is a Defect-4 promoted apparatus heading."""
    return (block.block_role or "") == _APPARATUS_HEADING_ROLE


def _is_opener_promoted(block: _AdapterBlock) -> bool:
    """Whether ``block`` is an A7 promoted pedagogical-opener heading.

    An opener heading (``Learning Objectives`` / ``Try It 9.1`` / ``Example
    9.1`` / ``Be Prepared 9.1`` / ``How To`` / ``Solution``) carries an
    :data:`~lib.semantik.opener_classifier.OPENER_ROLES` block_role. Such a
    heading BYPASSES the ``_is_noncontent_heading`` emit filter (like the
    apparatus-promoted case) and drives the ``data-dart-opener`` attribute +
    the ``<h4>`` level in :func:`_render_section`.
    """
    return (block.block_role or "") in OPENER_ROLES


def _promote_paragraph_block_to_heading(block: _AdapterBlock) -> None:
    """Promote a mis-typed apparatus paragraph ``block`` to a heading (Defect 4).

    The block's verbatim text becomes the visible ``<h3>`` heading; the body
    html is cleared so the label is not ALSO duplicated into the section body
    (the §1.1 duplication contract). Idempotent on an already-heading block.
    """
    text = (block.heading_text or block.raw_text or "").strip()
    if not text:
        return
    block.region_kind = "heading"
    # Marker role so the emit filters keep this heading even for apparatus names
    # (`Practice Test` / `Review Exercises`) that `_is_noncontent_heading`
    # otherwise drops as end-of-chapter noise — the WHOLE point of Defect 4 is
    # to recover them as visible sections.
    block.block_role = _APPARATUS_HEADING_ROLE
    block.heading_text = text
    block.html = ""


# ---------------------------------------------------------------------------
# Literal HTML-entity artifact scrub (ch02 audit, 2026-07-04).
#
# The VLM transcribes blank table spacing as LITERAL ``&nbsp;`` entity text in
# its markdown; fusion carries it into block text verbatim, and the adapter's
# HTML-escaping then renders it as visible ``&amp;nbsp;`` (520 occurrences in
# the audited ch02 — 509 of its 685 repeated-12gram excess in gold_compare).
# Scrubbed HERE (the adapter seam) so the planned ``--ir`` re-render corrects
# the CURRENT corpus without a re-conversion; the fusion-side scrub in
# ``SemantiK/dart_semantic/vlm_fusion.py::_strip_markdown_structure`` keeps
# FUTURE extractions clean at source (salted ``|vlmfuse5``).
#
# Conservative by design: ONLY the nbsp entity-artifact shapes are folded —
# ``&nbsp;`` (incl. multiply-escaped ``&amp;nbsp;`` / ``&amp;amp;nbsp;``) and
# the bare ``nbsp;`` fragment (semicolon REQUIRED; not preceded by a letter,
# so a prose word "nbsp" without entity context is never touched). A run of
# artifacts collapses to ONE space; residual multi-space runs are trimmed.
# ---------------------------------------------------------------------------
_NBSP_ARTIFACT_TOKEN = r"(?:&(?:amp;)*nbsp;|(?<![A-Za-z&])nbsp;)"
_NBSP_ARTIFACT_RUN_RE = re.compile(rf"(?:\s*{_NBSP_ARTIFACT_TOKEN})+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _scrub_entity_artifacts(text: Optional[str]) -> Optional[str]:
    """Collapse literal nbsp-entity artifact runs in ``text`` to single spaces.

    ``None``-safe pass-through; a fast substring guard keeps the common
    no-artifact path allocation-free. Only the nbsp artifact shapes above are
    folded — never general text.
    """
    if not text or "nbsp;" not in text:
        return text
    s = _NBSP_ARTIFACT_RUN_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", s)


def _scrub_block_entity_artifacts(chapters: Sequence[_AdapterChapter]) -> None:
    """Scrub nbsp-entity artifacts from every block's emitted text fields.

    Mutates the IR in place (the ``_normalize_ocr_headings`` convention) BEFORE
    render + sidecar build, so the HTML render and the sidecar ``text`` stay in
    parity (same seam as the apparatus split). Covers ``html`` (the rendered
    body), ``raw_text`` (the sidecar text + predicate input), ``repaired_text``
    (the OCR-repair sidecar override), and ``heading_text``.
    """
    for ch in chapters:
        for block in ch.blocks:
            block.html = _scrub_entity_artifacts(block.html) or ""
            block.raw_text = _scrub_entity_artifacts(block.raw_text) or ""
            block.repaired_text = _scrub_entity_artifacts(block.repaired_text)
            block.heading_text = _scrub_entity_artifacts(block.heading_text)


# ---------------------------------------------------------------------------
# Pedagogical-marker label-residue scrub (end-user-HTML audit, ch02, 2026-07-04)
#
# The OpenStax scan transcribes the exercise/opener MARKERS with a doubled-colon
# separator + stray OCR gutter glyphs — "TRY IT :: 2.1", "TRY IT : : 9.129 >",
# ":: GENERAL STRATEGY FOR SOLVING LINEAR EQUATIONS." — so the visible label
# reads "TRY IT :: 2.1" and, worse, the doubled colon severs the number from the
# opener name so ``opener_classifier`` refuses the promotion (its number regex
# needs "Try It 2.1", not "Try It :: 2.1"). This scrub folds the ":: " / ": : "
# colon-runs to a single space and drops the isolated ">" / "|" gutter glyphs, so
# the marker reads "TRY IT 2.1" (chosen convention: drop the separator) and the
# opener promotion fires downstream. Conservative: math is masked (so "$a:b$" and
# display math are untouched), a SINGLE colon is never collapsed (label colons
# like "Solving Applications with Formulas:" survive), and ">"/"|" are dropped
# only as whitespace-bounded standalone runs (an inline math inequality lives
# inside a masked "$…$" run and is never seen here).
# ---------------------------------------------------------------------------
_MARKER_MATH_MASK_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$[^$]*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
_MARKER_TAG_MASK_RE = re.compile(r"<[^>]+>")
_MARKER_COLON_RUN_RE = re.compile(r":(?:[ \t]*:)+")
# A whitespace-bounded run of stray gutter glyphs — bare ``|`` / ``>`` (plain
# text) or the escaped ``&gt;`` (already-rendered html body). Bounded so a ``|``
# inside a word or a ``>`` glued to a token is left alone.
#
# ROUND-6 masked-tag boundary fix: the left boundary now also fires after a mask
# sentinel (``(?<=\x00)`` — the placeholder a masked ``<p>`` tag / math span
# collapses to) and the right boundary before one (``\x00`` in the lookahead), so
# a leading ``&gt;| `` run glued to a masked ``<p>`` tag (``\x00N\x00&gt;| TRYIT``,
# ch09 [355]) — and a trailing ``&gt;`` glued to a masked ``</p>`` — fold. The
# old ``(?<=\s)|^`` / ``(?=\s|$)`` whitespace-only boundary could not reach a run
# sitting BETWEEN the tag sentinel and the text, so it shipped as visible debris.
_MARKER_GUTTER_RE = re.compile(
    r"(?:(?<=\s)|^|(?<=\x00))(?:&gt;|[>|])+(?=\s|$|\x00)"
)
# TABLE-DELIVERY-AWARE variant (wave-19 table-regression fix, 2026-07-04): the
# arm above eats EVERY whitespace-bounded ``|`` — including the pipe rows of a
# markdown ``| a | b |`` table — so it would scrub the pipes out before
# ``_emit_structured_bodies`` / ``parse_table`` can reconstruct the ``<table>``
# (ch02 went 47 tables -> 0, shipping the rows as prose debris). When — and ONLY
# when — :func:`~lib.semantik.structure_emit.parse_table` confirms the text
# actually DELIVERS a ``<table>`` this ``>``-only variant is used instead, so a
# stray ``>`` gutter glyph still scrubs while the delivered table's pipes
# survive. Keying on the parse OUTCOME (not a pipe-density heuristic) is what
# keeps prose/list debris honest: a paragraph like ``&gt;| TRYIT &gt;| TRYIT``
# or a list whose items carry a trailing ``|`` do NOT parse to a table, so their
# incidental pipes are still scrubbed (a density heuristic over-protected them,
# shipping visible ``<p>|</p>`` / trailing-pipe ``<li>`` debris).
_MARKER_GUTTER_NOPIPE_RE = re.compile(
    r"(?:(?<=\s)|^|(?<=\x00))(?:&gt;|>)+(?=\s|$|\x00)"
)
# ROUND-6 comparison guard: the broad gutter fold above would also eat a GENUINE
# ``>`` / ``&gt;`` comparison operator flanked by value operands (``x &gt; y``,
# ``6 > 4``, ``0 > 0``). Protect ONLY that shape — a LONE single-character operand
# (a variable / single digit, whitespace-or-edge bounded on its outer side, so
# NOT part of a longer token) on EACH side of the operator — into the mask stash
# before the gutter fold, so a real inequality survives while OCR gutter/marker
# debris still folds. The lone-operand guard is what folds ``TRY IT: 9.133 >
# Simplify`` (a decimal exercise number ``>`` a capitalized instruction word) and
# ``system. &gt; ![]`` (edge/punctuation adjacent) — neither is a value operand.
# In-math inequalities live inside a masked ``$…$`` run and never reach here.
_MARKER_COMPARISON_RE = re.compile(
    r"(?<![\w.])[\w][ \t](?:&gt;|>)[ \t][\w](?![\w.])"
)
_MARKER_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:)])")
# ITEM 2 (round-5 audit) — a ``:: `` colon-run fused into a math span escapes the
# post-mask colon fold because it sits INSIDE the masked ``$…$``. The OCR fuses a
# pedagogical marker into the following display math, so the ``::`` lands right
# after (or before) a math structural boundary — a ``$`` delimiter or a ``{``/
# ``}`` group brace: ``$\begin{array}{ll}\text { TRY IT }:: 9.19 & … $`` (``::``
# after the ``\text{…}`` close brace) and ``$\textbf{TRY IT ::} 2.24$`` (``::``
# before the ``\textbf{…}`` close brace). These two arms run BEFORE the math mask
# and fold a colon-run that abuts such a boundary (``$`` / ``{`` / ``}``, modulo
# whitespace, either side), preserving the boundary char AND the math itself. A
# ``::`` is NEVER a valid math operator in this corpus (verified: 0 brace/``$``-
# adjacent colon-runs across every DART exemplar, incl. the linear-algebra math
# textbook), so real math is untouched; a legitimate SINGLE label colon and an
# interior ``::`` not adjacent to a boundary both survive. ``(?<!\\)`` guards a
# currency ``\$``.
_MARKER_COLON_AFTER_MATH_RE = re.compile(
    r"((?<!\\)\$|[{}])[ \t]*:(?:[ \t]*:)+"
)
_MARKER_COLON_BEFORE_MATH_RE = re.compile(
    r":(?:[ \t]*:)+[ \t]*((?<!\\)\$|[{}])"
)
# Third arm — a colon-run BETWEEN a marker word and its exercise number
# (``TRY IT :: 9.19``, ``\textbf{TRY IT :: 1.173}``): the ``::`` is interior to a
# fused ``$…$`` span but adjacent to NEITHER delimiter nor brace, so the two arms
# above miss it. A colon-run PRECEDED BY A LETTER and FOLLOWED BY A DIGIT is the
# pedagogical marker-separator shape and NEVER valid content: a Haskell/type-sig
# ``f :: A`` (letter after ``::``) and a C++ ``std::vector`` (letter after) are
# untouched (digit-follow guard), and a ratio ``1::2`` (digit before) is untouched
# (letter-precede guard). Verified 0 ``LETTER::DIGIT`` runs across every DART
# exemplar, so real content on the wide-net corpus is safe.
_MARKER_COLON_MARKER_NUM_RE = re.compile(
    r"(?<=[A-Za-z])[ \t]*:(?:[ \t]*:)+[ \t]*(?=\d)"
)


def _scrub_marker_artifacts(text: Optional[str], *, html: bool) -> Optional[str]:
    r"""Fold ``:: `` / ``: : `` colon-runs + stray ``>``/``|`` gutter glyphs.

    ``None``-safe pass-through; a fast substring guard keeps the common
    no-artifact path allocation-free. Math runs (``$…$`` / ``\(…\)`` / display)
    are masked before the fold; for ``html`` bodies the tags are masked too so a
    ``<td>`` / attribute ``>`` is never touched.
    """
    if not text or (
        "::" not in text
        and ": :" not in text
        and "|" not in text
        and ">" not in text
        and "&gt;" not in text
    ):
        return text
    # ITEM 2 — fold a ``::`` colon-run GLUED to a ``$`` math delimiter BEFORE the
    # math mask (else the run is trapped inside a spurious ``$ :: N & $`` span and
    # never reaches the colon fold below). The ``$`` is preserved; genuine math is
    # untouched (a real span never abuts ``::`` to its delimiter).
    if "::" in text or ": :" in text:
        text = _MARKER_COLON_AFTER_MATH_RE.sub(r"\1 ", text)
        text = _MARKER_COLON_BEFORE_MATH_RE.sub(r" \1", text)
        text = _MARKER_COLON_MARKER_NUM_RE.sub(" ", text)
    stash: List[str] = []

    def _prot(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    masked = _MARKER_MATH_MASK_RE.sub(_prot, text)
    if html:
        masked = _MARKER_TAG_MASK_RE.sub(_prot, masked)
    masked = _MARKER_COLON_RUN_RE.sub(" ", masked)
    # ROUND-6 comparison guard: stash operand-flanked ``>``/``&gt;`` comparison
    # operators (``x &gt; y``) into the mask so the gutter fold below never eats
    # them; OCR gutter debris (a ``>`` at a node edge / after punctuation / before
    # an image) is NOT operand-flanked, so it still folds.
    masked = _MARKER_COMPARISON_RE.sub(_prot, masked)
    # TABLE-DELIVERY-AWARE gutter fold: keep the ``|`` rows ONLY when
    # ``parse_table`` confirms the text delivers a real ``<table>`` (they feed
    # ``_emit_structured_bodies``); otherwise scrub isolated ``>``/``|`` prose
    # gutter runs as before (so prose/list debris pipes stay scrubbed).
    # The check previews the EMIT-TIME text — ``sanitize_body_latex`` runs
    # between this scrub and structure emission, and its ``| --- |``
    # separator-row strip can drop a (sep + single data row) block below
    # ``parse_table``'s 2-row floor; gating on the sanitized preview keeps this
    # verdict in lock-step with what ``_emit_structured_bodies`` will actually
    # see (else pipes survive on a block that then falls to ``parse_list`` and
    # ships ``<p>|</p>`` / trailing-pipe ``<li>`` debris).
    # ``parse_table`` masks math itself, so an inline ``$a|b$`` never trips it.
    keep_pipes = (
        "|" in text
        and parse_table(sanitize_body_latex(text, html=html)) is not None
    )
    gutter_re = _MARKER_GUTTER_NOPIPE_RE if keep_pipes else _MARKER_GUTTER_RE
    masked = gutter_re.sub(" ", masked)
    masked = _MULTI_SPACE_RE.sub(" ", masked)
    masked = _MARKER_SPACE_BEFORE_PUNCT_RE.sub(r"\1", masked)
    restored = re.sub(
        r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], masked
    )
    return restored if html else restored.strip()


# ITEM 3 (round-5 audit) — the OCR-confusable ``trvit`` (a ``v``-for-``y``
# misread of "TRY IT", ch02 s519 "… trvit:: 26: see 8 th +12") reaches no opener
# path: neither the standalone/leading/interior opener matchers (they key on
# ``try\s*it``) nor the marker scrub recognise it, so it ships as visible garble.
# This deterministic single-token normalization rewrites ``trvit`` → ``TRY IT``
# so the DOWNSTREAM marker scrub + fused-heading demotion + interior-opener split
# machinery handle the block. Conservative: fires ONLY when the confusable is a
# whole word (``(?<![A-Za-z]) … (?![A-Za-z])``) that is FOLLOWED BY A NUMBER
# (modulo ``::`` / gutter / period) — the "TRY IT 26" exercise-marker shape — so
# a hypothetical prose word is never touched. Case-insensitive; the replacement
# is canonical ALL-CAPS so the marker scrub / opener casing guards accept it.
# Wave #22 — the OCR-confusable marker vocabulary is sourced from the lexicon
# (``schemas/taxonomies/semantik_lexicon.json``, ``confusables`` per profile) so
# a new corpus's confusable adds a lexicon row, not code. Behavior-preserving:
# the default profile yields exactly the historical ``trvit``->``TRY IT`` rule.
# The STRUCTURAL scaffolding (word-boundary guards + the number lookahead that
# scopes the rewrite to the "TRY IT 26" exercise-marker shape) stays in code —
# only the confusable BODY pattern + its canonical replacement are vocab.
_CONFUSABLES: tuple[tuple["re.Pattern[str]", str], ...] = tuple(
    (
        re.compile(
            rf"(?<![A-Za-z])(?:{c['pattern']})(?![A-Za-z])(?=[\s:.>|&;]*\d)",
            re.IGNORECASE,
        ),
        c["canonical"],
    )
    for c in _lexicon_confusables()
    if c.get("pattern") and c.get("canonical")
)


# Back-compat alias — the first (historically the only, ``trvit``) confusable
# regex; a ledger test asserts this symbol still exists.
_CONFUSABLE_TRYIT_RE = _CONFUSABLES[0][0] if _CONFUSABLES else re.compile(r"(?!x)x")


def _normalize_confusable_marker_text(text: Optional[str]) -> Optional[str]:
    """Rewrite OCR-confusable marker tokens → canonical (number-guarded).

    Iterates the active lexicon's confusables (default: the single
    ``trvit``->``TRY IT`` rule). ``None``-safe pass-through; a no-op when no
    confusable pattern fires.
    """
    if not text:
        return text
    for rx, canonical in _CONFUSABLES:
        text = rx.sub(canonical, text)
    return text


def _normalize_confusable_markers(chapters: Sequence[_AdapterChapter]) -> None:
    """Normalize the ``trvit`` OCR-confusable across every block (ITEM 3).

    Runs FIRST in :func:`_normalize_ocr_headings` (before the entity / marker
    scrubs) on ALL text fields so the rendered ``html`` and the sidecar ``text``
    stay in parity and the de-garbled ``TRY IT 26`` flows through the existing
    marker scrub + heading/opener machinery.
    """
    for ch in chapters:
        for block in ch.blocks:
            block.html = _normalize_confusable_marker_text(block.html) or ""
            block.raw_text = _normalize_confusable_marker_text(block.raw_text) or ""
            block.repaired_text = _normalize_confusable_marker_text(
                block.repaired_text
            )
            block.heading_text = _normalize_confusable_marker_text(
                block.heading_text
            )


# ITEM 1 (round-5 audit) — pedagogical-marker DEBRIS fragments. The scan emits
# tiny standalone blocks that are ONLY OCR gutter glyphs + a bare garbled marker
# token with no number and no content: ``>| TRYIT::`` (ch02 ×5 + a ``>| TRYIT::
# >| TRYIT::`` double, ch09 ×2). After the marker scrub these ship as visible
# ``<p>&gt;| TRYIT </p>`` debris (the leading gutter run sits after the masked
# ``<p>`` tag, so the whitespace-bounded gutter fold cannot reach it). Such a
# block carries no learner value and is re-tagged ``metadata_drop`` (suppressed
# from render + sidecar, like the folio-furniture path). STRICT anti-fabrication
# guards — a fragment is debris ONLY when it (1) contains a gutter glyph AND a
# recognised marker token, and (2) contains NO digit, NO math ($ / \( / \[ / a
# ``\command``), and FEWER THAN 3 word tokens, and (3) is nothing but gutter +
# marker tokens once the markers are removed. So a numbered ``> TRY IT :: 9.174``
# (promotes), a math/content-bearing fragment, and any real prose are all left
# untouched.
_DEBRIS_MARKER_RE = re.compile(
    r"try\s*it|tr[vy]it|example|be\s*prepared|how\s*to", re.IGNORECASE
)
_DEBRIS_GUTTER_RE = re.compile(r"[>|]|&gt;|::")
_DEBRIS_LATEX_CMD_RE = re.compile(r"\\[A-Za-z(\[]")
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _is_marker_debris(text: Optional[str]) -> bool:
    """Whether ``text`` is a pure gutter-glyph + garbled-marker debris fragment."""
    if not text:
        return False
    visible = _html_module.unescape(_TAG_STRIP_RE.sub(" ", text))
    if not _DEBRIS_GUTTER_RE.search(text):
        return False  # a clean (gutter-free) marker is opener-promotion territory
    if not _DEBRIS_MARKER_RE.search(visible):
        return False  # no recognised marker token → not marker debris
    if any(c.isdigit() for c in visible):
        return False  # a number → a promotable "TRY IT 9.1", never dropped
    if "$" in visible or _DEBRIS_LATEX_CMD_RE.search(visible):
        return False  # math-bearing → content, never dropped
    if len(_WORD_TOKEN_RE.findall(visible)) >= 3:
        return False  # >= 3 word tokens → real content, never dropped
    # Nothing but gutter + marker tokens may remain.
    residue = _DEBRIS_MARKER_RE.sub(" ", visible)
    residue = _DEBRIS_GUTTER_RE.sub(" ", residue)
    return not any(c.isalpha() for c in residue)


def _collapse_marker_debris_blocks(chapters: Sequence[_AdapterChapter]) -> None:
    """Re-tag pure gutter + garbled-marker debris blocks as furniture (ITEM 1).

    Runs BEFORE the marker scrub (so the gutter glyphs the predicate keys on are
    still present) and BEFORE opener promotion / structure emit (so a debris
    ``code_block`` / ``table`` mis-typing never tries to become a ``<table>``).
    A non-debris block passes through untouched.
    """
    for ch in chapters:
        for block in ch.blocks:
            if _is_heading_block(block) or _is_furniture_block(block):
                continue
            if _is_marker_debris(block.raw_text) or (
                not (block.raw_text or "").strip()
                and _is_marker_debris(block.html)
            ):
                block.region_kind = "metadata_drop"
                block.block_role = "metadata_drop"


def _scrub_block_marker_artifacts(chapters: Sequence[_AdapterChapter]) -> None:
    """Scrub pedagogical-marker label residue from every block's text fields.

    Runs BEFORE the opener split (so a de-doubled "TRY IT 2.1" promotes) and in
    the same in-place seam as ``_scrub_block_entity_artifacts`` so the rendered
    ``html`` and sidecar ``text`` stay in parity.
    """
    for ch in chapters:
        for block in ch.blocks:
            block.html = _scrub_marker_artifacts(block.html, html=True) or ""
            block.raw_text = (
                _scrub_marker_artifacts(block.raw_text, html=False) or ""
            )
            block.repaired_text = _scrub_marker_artifacts(
                block.repaired_text, html=False
            )
            block.heading_text = _scrub_marker_artifacts(
                block.heading_text, html=False
            )


# Content-block kinds the apparatus split may own when their body is FLAT <p>
# text: genuine paragraphs, plus council-typed "list"/"definition_list" blobs
# whose html fell back to a flat <p> (the cascade_ir `_render_list_html` → None
# path). The rerender_v5 interior residuals (ch01/ch04 KEY CONCEPTS) are such
# flat "list" glossary blobs.
_FLAT_TEXT_SPLIT_KINDS = frozenset({"paragraph", "list", "definition_list"})


def _split_leading_apparatus_blocks(
    blocks: Sequence[_AdapterBlock],
) -> List[_AdapterBlock]:
    """Split any block that fused an apparatus heading (defect f + Rule A).

    LEADING splits (predicate :func:`split_leading_apparatus_heading`) emit
    heading + remainder; INTERIOR splits (Rule A, rerender_v5 —
    :func:`split_interior_apparatus_heading`, exact ALL-CAPS token mid-block)
    emit preceding-text + heading + remainder. Two source arms feed both:

    * PARAGRAPH arm — a paragraph whose text starts with an apparatus name
      (the section heading "Key Terms" fused into the front of the glossary
      paragraph).
    * HEADING arm (ch09 live re-render validation) — the cascade sometimes
      types the WHOLE fused blob as a ``heading`` region ("Key Terms index
      $\\sqrt[n]{a}$ … is called the index of the radical. …" as
      ``heading_text``). Without this arm the block skipped the split and fell
      through to the ``is_fused_heading`` DEMOTION in ``_normalize_ocr_headings``
      (which buries the apparatus name in a demoted paragraph — section still
      reported missing). This split runs BEFORE that demotion loop, so for an
      apparatus-LED fused heading the split WINS: the demotion loop then sees a
      clean apparatus heading (kept via ``is_apparatus_heading``) + a remainder
      paragraph. A fused NON-apparatus heading still takes the demotion path.

    Both arms emit a promoted apparatus HEADING block (same
    ``_APPARATUS_HEADING_ROLE`` path as the standalone promotion) followed by
    the REMAINDER paragraph. Both halves inherit the source block's provenance
    (``pages`` / ``confidence`` / ``wcag_status``) so their ``data-dart-*``
    attributes agree — mirroring the in-place demote/promote paths, which
    preserve provenance by mutating the same block. Returns a NEW ordered block
    list (a non-matching block passes through untouched — including the
    STANDALONE apparatus case, where the predicate's remainder floor returns
    ``None`` and the existing promotion/keep paths handle the whole block).
    """
    out: List[_AdapterBlock] = []
    for block in blocks:
        if block.region_kind == "heading":
            # HEADING arm — the fused blob lives in heading_text (raw_text as
            # the fallback the promotion paths also use).
            source_text = (block.heading_text or block.raw_text or "").strip()
        elif (
            block.region_kind in _FLAT_TEXT_SPLIT_KINDS
            and not block.heading_text
            # Flat-text gate: the split rebuilds halves as <p> bodies, so it
            # only owns blocks whose html IS flat <p> text (or empty). The
            # rerender_v5 residual fusions are council-typed "list" glossary
            # blobs whose html fell back to a flat <p> (no bullet markers); a
            # REAL <ul>/<dl>/<table> body must never be flattened.
            and (not block.html or block.html.lstrip().startswith("<p"))
        ):
            # PARAGRAPH arm — fused blob within a flat text body.
            source_text = (block.raw_text or "").strip()
        else:
            out.append(block)
            continue
        pre_text: Optional[str] = None
        split = split_leading_apparatus_heading(source_text)
        if split is not None:
            heading_text, remainder = split
        else:
            # RULE A — interior ALL-CAPS apparatus banner fused MID-block
            # ("…numbers 0, 1, 2, 3, … KEY CONCEPTS 1.1 Introduction to Whole
            # Numbers"). The preceding text stays in the ORIGINAL block slot;
            # the apparatus heading + trailing remainder follow it.
            interior = split_interior_apparatus_heading(source_text)
            if interior is None:
                out.append(block)
                continue
            pre_text, heading_text, remainder = interior
        heading_block = _AdapterBlock(
            html="",
            region_kind="heading",
            raw_block_index=block.raw_block_index,
            raw_text=heading_text,
            heading_text=heading_text,
            pages=tuple(block.pages),
            confidence=block.confidence,
            # Apparatus role marker so the `_is_noncontent_heading` emit filter
            # keeps it (parity with `_promote_paragraph_block_to_heading`).
            block_role=_APPARATUS_HEADING_ROLE,
            wcag_status=block.wcag_status,
        )
        if pre_text is None:
            # LEADING split: mutate the ORIGINAL block into the remainder
            # paragraph in place, so it keeps its provenance (pages/confidence/
            # wcag) exactly like the demote paths do. The apparatus prefix is
            # stripped from both text and body.
            block.raw_text = remainder
            block.html = f"<p>{_esc_text(remainder)}</p>"
            block.heading_text = None
            block.region_kind = "paragraph"
            out.append(heading_block)
            out.append(block)
        else:
            # INTERIOR split (Rule A): the original block keeps the PRECEDING
            # text (provenance preserved in place); a NEW remainder paragraph
            # inherits the same provenance fields as the heading block.
            block.raw_text = pre_text
            block.html = f"<p>{_esc_text(pre_text)}</p>"
            block.heading_text = None
            block.region_kind = "paragraph"
            remainder_block = _AdapterBlock(
                html=f"<p>{_esc_text(remainder)}</p>",
                region_kind="paragraph",
                raw_block_index=block.raw_block_index,
                raw_text=remainder,
                pages=tuple(block.pages),
                confidence=block.confidence,
                wcag_status=block.wcag_status,
            )
            out.append(block)
            out.append(heading_block)
            out.append(remainder_block)
    return out


# ---------------------------------------------------------------------------
# A7 — pedagogical-opener promotion (end-user-HTML audit, 2026-07-04).
# ---------------------------------------------------------------------------
# OpenStax scan chapters emit the pedagogical openers (Learning Objectives / Be
# Prepared / Try It / Example / How To / Solution) as flat <p> prose, so the
# chunker never breaks a section on them and a single <p> fuses a worked
# example, its solution, and the next example. This pass PROMOTES a standalone
# opener label to a real <h4> heading, and SPLITS a paragraph that FUSED a
# leading opener into an <h4> heading + a remainder paragraph — both carrying a
# machine-readable ``block_role`` (an OPENER_ROLES member) so ``_render_section``
# stamps ``data-dart-opener`` and the emit-filter bypass (``_is_opener_promoted``)
# keeps the heading. Deterministic; a strict no-op on prose with no opener label.


def _objectives_body_html(text: str) -> str:
    r"""Render a Learning-Objectives remainder as ``<ul>`` when list-shaped.

    The OpenStax objectives block reads "By the end of this section, you will
    be able to: <item> <item> …". When the post-colon items are newline- or
    bullet-delimited they are emitted as ``<ul><li>…</li></ul>`` (the chunker
    harvests ``<li>`` bullets). When the items are FLAT run-on prose (no
    reliable delimiter) the whole remainder stays a single ``<p>`` — the pass
    NEVER fabricates bullet boundaries out of run-on prose.
    """
    body = text.strip()
    lead = ""
    # Keep the "By the end … able to:" lead-in as a <p> above the list.
    m = re.match(r"^(.*?\bable to\s*:?)\s*(.*)$", body, re.IGNORECASE | re.DOTALL)
    if m and m.group(2).strip():
        lead, body = m.group(1).strip(), m.group(2).strip()
    # Split only on UNAMBIGUOUS delimiters: newlines or bullet glyphs.
    items = [
        seg.strip(" \t•‣◦-*")
        for seg in re.split(r"[\n\r]+|[•‣◦]", body)
    ]
    items = [seg for seg in items if seg]
    lead_html = f"<p>{_esc_text(lead)}</p>" if lead else ""
    if len(items) >= 2:
        lis = "".join(f"<li>{_esc_text(it)}</li>" for it in items)
        return f"{lead_html}<ul>{lis}</ul>"
    # Single run-on item → keep as prose (with the optional lead-in).
    return f"{lead_html}<p>{_esc_text(body)}</p>" if body else lead_html


def _make_opener_heading(
    block: _AdapterBlock, display: str, role: str, level: int = 4
) -> None:
    """Mutate ``block`` in place into a promoted opener heading at ``level``."""
    block.region_kind = "heading"
    block.heading_text = display
    block.block_role = role
    block.heading_level = level
    block.html = ""


def _split_interior_openers_in_blocks(
    blocks: Sequence[_AdapterBlock],
) -> List[_AdapterBlock]:
    """Split flat-text blocks that FUSED interior numbered openers (ITEM 4).

    A run-on ``<p>`` that fuses "… TRY IT 9.201 Simplify: … TRY IT 9.202 … EXAMPLE
    9.102 …" is split at each interior numbered-opener marker into: the preceding
    prose (mutated in place, keeping provenance), then per marker an opener
    HEADING block (``block_role`` in OPENER_ROLES, so the emit filter keeps it and
    ``_render_section`` stamps ``data-dart-opener``) + its remainder paragraph.
    Both new blocks inherit the source block's provenance. Runs BEFORE
    :func:`_promote_openers_in_blocks` (which then promotes a LEADING opener + the
    interior heading levels) and only owns flat ``<p>`` bodies (never a delivered
    ``<ul>/<dl>/<table>``). A non-matching block passes through untouched.

    Owned kinds: the flat-text split kinds PLUS flat-``<p>`` ``code_block`` /
    ``math`` / ``table`` blocks — on this scan corpus those kinds are cascade
    MIS-TYPINGS / debris (their body html is a flat ``<p>``, never
    ``<pre>``/MathML/``<table>``), and the residual audit showed TRY-IT / EXAMPLE
    fusions concentrating there. Round-3 (Defect 3) adds ``table``: the ch02
    pipe-table debris blocks (s320/s331/s399) spilled trailing ``TRY IT`` /
    ``EXAMPLE`` markers PAST the last pipe; :func:`split_interior_openers` masks
    the pipe-cell runs so only those OUTSIDE-pipe markers split off (a marker
    still inside an intact cell stays put — a re-cascade item). The flat-``<p>``
    gate still refuses any genuinely-structured body; ``heading`` stays excluded
    (a fused heading is a re-cascade item).
    """
    out: List[_AdapterBlock] = []
    split_kinds = _FLAT_TEXT_SPLIT_KINDS | {"code_block", "math", "table"}
    for block in blocks:
        flat_text = not block.html or block.html.lstrip().startswith("<p")
        if (
            block.region_kind not in split_kinds
            or block.heading_text
            or not flat_text
            or _is_furniture_block(block)
            or _is_apparatus_promoted(block)
        ):
            out.append(block)
            continue
        parts = split_interior_openers(block.raw_text or "")
        if parts is None:
            out.append(block)
            continue
        # Mutate the ORIGINAL block into the first part (provenance preserved) —
        # a leading prose span OR (when the block starts with an opener) the first
        # opener heading — then emit the remaining heading / remainder blocks.
        first = parts[0]
        if first[0] == "text":
            block.raw_text = first[1]
            block.html = f"<p>{_esc_text(first[1])}</p>"
            block.heading_text = None
            block.region_kind = "paragraph"
        else:  # ("opener", display, role)
            block.raw_text = first[1]
            block.heading_text = first[1]
            block.html = ""
            block.region_kind = "heading"
            block.block_role = first[2]
        out.append(block)
        for part in parts[1:]:
            if part[0] == "opener":
                _display, role = part[1], part[2]
                out.append(
                    _AdapterBlock(
                        html="",
                        region_kind="heading",
                        raw_block_index=block.raw_block_index,
                        raw_text=part[1],
                        heading_text=part[1],
                        pages=tuple(block.pages),
                        confidence=block.confidence,
                        block_role=role,
                        wcag_status=block.wcag_status,
                    )
                )
            else:  # ("text", content)
                content = part[1]
                out.append(
                    _AdapterBlock(
                        html=f"<p>{_esc_text(content)}</p>",
                        region_kind="paragraph",
                        raw_block_index=block.raw_block_index,
                        raw_text=content,
                        pages=tuple(block.pages),
                        confidence=block.confidence,
                        wcag_status=block.wcag_status,
                    )
                )
    return out


def _promote_openers_in_blocks(
    blocks: Sequence[_AdapterBlock],
) -> List[_AdapterBlock]:
    """Promote / split pedagogical openers in one chapter's block list.

    Two arms mirror the apparatus split: (1) a STANDALONE opener label
    (paragraph- or heading-typed, flat-text body) is promoted whole to an opener
    heading; (2) a flat-text paragraph that FUSED a leading opener is split into
    the opener heading + a remainder paragraph (objectives remainders render as
    <ul> when list-shaped). A non-matching block passes through untouched; a
    block whose html is a REAL <ul>/<dl>/<table> is never flattened (the
    flat-text gate).

    A5 heading-level fix — a promoted opener nests one level UNDER its nearest
    ancestor SECTION heading (never a fixed <h4>): tracking the last genuine
    heading block's level as we walk the chapter, an opener directly under the
    chapter <h2> (no <h3> section extracted yet) renders at <h3>, and an opener
    under an <h3> section renders at <h4>. This closes the h2->h4 level skip that
    a fixed-<h4> opener produced. Openers do NOT update the tracked section level
    (consecutive openers are siblings, not nested).
    """
    out: List[_AdapterBlock] = []
    # Nearest ancestor section-heading level; the chapter is <h2>, real section
    # headings are <h3>. An opener nests at section_level + 1.
    section_level = 2
    for block in blocks:
        text = (block.heading_text or block.raw_text or "").strip()
        # A genuine (non-opener) heading block re-anchors the section level so a
        # following opener nests correctly under it.
        if (
            block.region_kind == "heading"
            and block.heading_text
            and classify_opener_label(text) is None
        ):
            section_level = int(getattr(block, "heading_level", 3) or 3)
        if _is_furniture_block(block) or _is_apparatus_promoted(block):
            out.append(block)
            continue
        flat_text = not block.html or block.html.lstrip().startswith("<p")
        opener_level = min(section_level + 1, 6)

        # Arm 0 (ITEM 4) — a LABEL-ONLY stacked-opener block ("EXAMPLE 2.3
        # Solution", body dropped upstream) becomes the stacked opener headings
        # (empty groups) — an honest marker of where the example sits. Runs before
        # the standalone arm (which handles a SINGLE label); the strict
        # decompose-to->=2-labels-and-nothing-else guard leaves real prose alone.
        if flat_text or block.region_kind == "heading":
            label_only = split_label_only_openers(text)
            if label_only is not None:
                first_display, first_role = label_only[0]
                _make_opener_heading(
                    block, first_display, first_role, level=opener_level
                )
                out.append(block)
                for disp, role in label_only[1:]:
                    out.append(
                        _AdapterBlock(
                            html="",
                            region_kind="heading",
                            raw_block_index=block.raw_block_index,
                            raw_text=disp,
                            heading_text=disp,
                            pages=tuple(block.pages),
                            confidence=block.confidence,
                            block_role=role,
                            wcag_status=block.wcag_status,
                            heading_level=opener_level,
                        )
                    )
                continue

        # Arm 1 — standalone opener label.
        standalone = classify_opener_label(text)
        if standalone is not None and (
            flat_text or block.region_kind == "heading"
        ):
            display, role = standalone
            _make_opener_heading(block, display, role, level=opener_level)
            out.append(block)
            continue

        # Arm 2 — leading opener fused into a flat-text paragraph body.
        if (
            block.region_kind in _FLAT_TEXT_SPLIT_KINDS
            and not block.heading_text
            and flat_text
        ):
            split = split_leading_opener(block.raw_text or "")
            if split is not None:
                display, role, remainder = split
                heading_block = _AdapterBlock(
                    html="",
                    region_kind="heading",
                    raw_block_index=block.raw_block_index,
                    raw_text=display,
                    heading_text=display,
                    pages=tuple(block.pages),
                    confidence=block.confidence,
                    block_role=role,
                    wcag_status=block.wcag_status,
                    heading_level=opener_level,
                )
                block.raw_text = remainder
                block.heading_text = None
                block.region_kind = "paragraph"
                if role == ROLE_OBJECTIVES:
                    block.html = _objectives_body_html(remainder)
                else:
                    block.html = f"<p>{_esc_text(remainder)}</p>"
                out.append(heading_block)
                out.append(block)
                continue

        out.append(block)
    return out


# ---------------------------------------------------------------------------
# Tier-1 reading-order grammar + unit-skeleton heading re-derivation (Wave #22).
# ---------------------------------------------------------------------------
# Owner directives 2 (logical reading order as a semantic signal) + the heading
# re-derivation half of directive 1. Both are deterministic, block-level, and
# run AFTER opener promotion (so example/solution/how_to opener roles are final)
# and BEFORE render. The flow pass is ANNOTATION-ONLY (stamps ``data-dart-flow``,
# never touches text/structure); the heading re-derivation demotes ONLY the
# unambiguous mis-typed heading that severs a worked example's problem statement
# from its solution, and reports the count.


def _is_genuine_section_heading(block: _AdapterBlock) -> bool:
    """A visible section heading that is NOT a promoted opener (a unit boundary)."""
    return _is_heading_block(block) and not _is_opener_promoted(block)


def _segment_blocks(
    blocks: Sequence[_AdapterBlock],
) -> "Iterator[List[_AdapterBlock]]":
    """Yield content+opener runs split at genuine (non-opener) section headings.

    A genuine section/chapter heading (or apparatus heading) is a hard boundary
    — it delimits segments and is itself excluded (the reading-order grammar
    never spans a real section boundary). Furniture is skipped.
    """
    seg: List[_AdapterBlock] = []
    for b in blocks:
        if _is_furniture_block(b):
            continue
        if _is_genuine_section_heading(b):
            if seg:
                yield seg
                seg = []
            continue
        seg.append(b)
    if seg:
        yield seg


def _annotate_reading_order_flow(chapters: Sequence[_AdapterChapter]) -> None:
    """Stamp the sequence-derived FLOW phase on content blocks (Tier-1, directive 2).

    Within each section segment: the content between an ``worked_example`` opener
    and its following ``solution`` opener is the example STATEMENT
    (``data-dart-flow="statement"``); the content after the ``solution`` opener
    (until the next opener / segment end) is the WORKED STEPS
    (``solution-steps``); the content after a ``how_to`` opener is the PROCEDURE
    STEPS (``procedure-steps``). Annotation-only; a lone example (no following
    solution) gets NO statement flow (conservative — "between" needs both ends).
    """
    for ch in chapters:
        for seg in _segment_blocks(ch.blocks):
            openers = [
                (i, b.block_role)
                for i, b in enumerate(seg)
                if _is_opener_promoted(b)
            ]
            for k, (i, role) in enumerate(openers):
                nxt = openers[k + 1][0] if k + 1 < len(openers) else len(seg)
                content = [
                    b
                    for b in seg[i + 1 : nxt]
                    if not _is_opener_promoted(b) and not _is_heading_block(b)
                ]
                if role == "worked_example":
                    if k + 1 < len(openers) and openers[k + 1][1] == "solution":
                        for b in content:
                            b.flow = "statement"
                elif role == "solution":
                    for b in content:
                        b.flow = "solution-steps"
                elif role == "how_to":
                    for b in content:
                        b.flow = "procedure-steps"


# A numbered section title ("9.2 Add Square Roots") is a REAL section even when
# oddly placed — never demoted by the unit-skeleton re-derivation.
_NUMBERED_SECTION_TITLE_RE = re.compile(r"^\s*\d+\.\d+")


def _nearest_opener_role(
    blocks: Sequence[_AdapterBlock], idx: int, step: int
) -> Optional[str]:
    """Role of the nearest opener in ``step`` direction, over CONTENT only.

    Scans away from ``idx``; returns the first opener's role. Stops (returns
    ``None``) at any other genuine/apparatus heading or the block-list edge —
    so an opener is "nearest" only when nothing but content lies between.
    """
    j = idx + step
    n = len(blocks)
    while 0 <= j < n:
        b = blocks[j]
        if _is_furniture_block(b):
            j += step
            continue
        if _is_opener_promoted(b):
            return b.block_role
        if _is_heading_block(b):
            return None  # another heading between → ambiguous
        j += step
    return None


def _rederive_unit_headings(chapters: Sequence[_AdapterChapter]) -> int:
    """Demote the mis-typed heading that severs a worked example (Tier-1, directive 1).

    UNAMBIGUOUS case only: a genuine (non-opener, non-apparatus, non-numbered-
    section-title) heading strictly sandwiched between a ``worked_example``
    opener and its ``solution`` opener — with only content on each side — is a
    mis-typed heading interrupting the example's problem statement (a real
    section heading never appears between "Example N" and its "Solution"). It is
    demoted to prose so the worked-example unit re-forms and the TOC (rebuilt
    from the corrected heading tree) sheds the stray entry. Returns the count of
    demotions. Every other heading is untouched (conservative + test-locked).
    """
    count = 0
    for ch in chapters:
        for idx, b in enumerate(ch.blocks):
            if not (
                _is_heading_block(b)
                and not _is_opener_promoted(b)
                and not _is_apparatus_promoted(b)
            ):
                continue
            # Arm (k) — a heading that is WHOLLY a ``\textbf{…}`` / ``\textit{…}``
            # inline-emphasis label is a mis-typed bold callout, not a section:
            # demote it to a bold paragraph (sanitize_body_latex converts the
            # wrapper to <strong>) so it leaves the heading stream + TOC. Runs
            # before the unit-severing check (an emphasis label is never a real
            # section boundary). Numbered-section titles are exempt below.
            if is_emphasis_label_heading(b.heading_text or ""):
                _demote_heading_block_to_paragraph(b)
                count += 1
                continue
            if _NUMBERED_SECTION_TITLE_RE.match(b.heading_text or ""):
                continue
            prev_role = _nearest_opener_role(ch.blocks, idx, -1)
            next_role = _nearest_opener_role(ch.blocks, idx, +1)
            if prev_role == "worked_example" and next_role == "solution":
                _demote_heading_block_to_paragraph(b)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Shape-driven structural-body emission (exemplar-parity wave — A1/A4/A5).
# ---------------------------------------------------------------------------
# The cascade ships list/table/definition_list-declared blocks with a FLAT <p>
# body, so the emitted HTML never delivers the structure it declares (the
# structure-scorecard "deliver" dimension reads ~0). This pass reconstructs the
# real <ul>/<ol>/<table>/<dl> from the block's deterministic ``raw_text`` when —
# and ONLY when — the SHAPE is high-confidence (see lib.semantik.structure_emit),
# and honours the "declare -> deliver" contract in the OTHER direction too: a
# block that DECLARED a structural role but carries no high-confidence shape is
# reconciled to ``paragraph`` (a mis-typed "9.2 Simplify Square Roots" list, a
# "ROOTS AND RADICALS" definition_list) rather than shipping a false declaration.
_STRUCTURAL_HTML_TAGS = ("<ul", "<ol", "<table", "<dl")


def _emit_structured_bodies(chapters: Sequence[_AdapterChapter]) -> None:
    """Deliver <ul>/<ol>/<table>/<dl> bodies for high-confidence shapes (A1/A4/A5).

    Runs AFTER the opener / apparatus split (so a split-out glossary remainder
    can become a <dl>) and reads the SANITIZED ``raw_text`` (math already wrapped
    in ``$…$``). Skips headings, furniture, and blocks whose html is already a
    real structural element (never flattens a delivered <ul>/<table>/<dl>).
    """
    for ch in chapters:
        for block in ch.blocks:
            if _is_heading_block(block) or _is_furniture_block(block):
                continue
            html = block.html or ""
            stripped = html.lstrip()
            if stripped and not stripped.startswith("<p"):
                continue  # a non-<p> body is already structured
            if any(tag in html for tag in _STRUCTURAL_HTML_TAGS):
                continue  # a <p>-led body that already embeds a structure
            result = emit_structure(block.raw_text or "")
            if result is not None:
                kind, structured_html = result
                # Re-balance the emitted structure: splitting a block into
                # <p>/<table>/<p> (or <li> items) can cut a $$…$$ display span
                # across the new element boundary, so wrap_bare_math re-runs on
                # the structured html to keep every fragment self-balanced.
                block.html = wrap_bare_math(structured_html, html=True)
                block.block_role = kind
                block.region_kind = kind
            elif (
                (block.block_role or "") in STRUCTURAL_ROLES
                or block.region_kind in STRUCTURAL_ROLES
            ):
                # Declared a structural role but no high-confidence shape ->
                # reconcile the declaration to the (prose) delivery. Record the
                # ORIGINAL declared role (block_role, else region_kind) so the
                # scorecard can count the demotion (blind-spot guard) — the
                # reconciliation itself is otherwise invisible downstream.
                declared = (
                    block.block_role
                    if (block.block_role or "") in STRUCTURAL_ROLES
                    else block.region_kind
                )
                block.demoted_role = declared
                block.block_role = "paragraph"
                block.region_kind = "paragraph"


def _sanitize_block_body_latex(chapters: Sequence[_AdapterChapter]) -> None:
    r"""Fold visible text-mode LaTeX/markdown garbage out of body text (B3).

    Mutates the IR in place BEFORE render + sidecar build (the
    ``_scrub_block_entity_artifacts`` convention), so the rendered ``html`` and
    the sidecar ``text`` stay in parity. ``html`` gets ``<strong>``/``<em>``
    for ``\textbf``/``\textit``; ``raw_text``/``repaired_text`` get the bare
    word (chunk text is plain). Math runs (``$…$``) are protected. Heading text
    is untouched — it already routes through ``_sanitize_heading_text``.

    Exemplar-parity wave (B3+) — after the whole-fragment fold, bare
    (un-delimited) LaTeX math runs (``\sqrt{5} \approx 2.236`` with no ``$``) are
    wrapped back into ``$…$`` via :func:`~lib.semantik.math_fold.wrap_bare_math``
    (so MathJax renders them and they stop reading as visible ``\command``
    garbage) and orphaned tabular/array scaffolding (``\hline`` / ``\begin{array}``
    left by a page-split table) is dropped — on the SAME fields so render and
    sidecar stay in parity.
    """
    for ch in chapters:
        for block in ch.blocks:
            if block.html:
                block.html = wrap_bare_math(
                    sanitize_body_latex(block.html, html=True), html=True
                )
            if block.raw_text:
                block.raw_text = wrap_bare_math(
                    sanitize_body_latex(block.raw_text, html=False), html=False
                )
            if block.repaired_text:
                block.repaired_text = wrap_bare_math(
                    sanitize_body_latex(block.repaired_text, html=False),
                    html=False,
                )


def _linkify_block_urls(chapters: Sequence[_AdapterChapter]) -> None:
    """Linkify bare / angle-wrapped vendor URLs across the IR in place (ITEM 1).

    Runs LAST (after the self-balance ``wrap_bare_math`` sweep) so the emitted
    ``<a>`` anchors are never re-mangled by a later math pass. ``html`` gets a
    ``mathjax_ignore`` anchor (so MathJax's global ``$`` scan cannot swallow the
    link into italic soup); ``raw_text`` / ``repaired_text`` get the bare
    normalized URL (plain chunk / sidecar text). HTML + sidecar stay in parity.
    """
    for ch in chapters:
        for block in ch.blocks:
            if block.html:
                block.html = linkify_urls(block.html, html=True)
            if block.raw_text:
                block.raw_text = linkify_urls(block.raw_text, html=False)
            if block.repaired_text:
                block.repaired_text = linkify_urls(
                    block.repaired_text, html=False
                )


def _escape_currency_dollars(chapters: Sequence[_AdapterChapter]) -> None:
    r"""Escape preserved currency ``$`` → ``\$`` in block HTML ONLY (round-7b).

    The assembled end-user page enables MathJax v3 with ``inlineMath [['$','$']]``,
    so two currency amounts in one paragraph ("costs $5 … and $3") FALSE-PAIR
    into an italic math span at render. This HTML-only pass rewrites each
    preserved lone currency ``$`` (a ``$`` immediately before a digit, currency
    by construction after the ``_pair_dollars`` sweep) to ``\$`` — a literal
    dollar under the assembler's ``processEscapes: true`` config — masking genuine
    ``$…$`` / ``$$…$$`` / ``\(…\)`` / ``\[…\]`` math so real inline math is never
    touched.

    HTML-ONLY: ``raw_text`` / ``repaired_text`` (the sidecar + chunker/retrieval
    text) keep plain ``$5`` untouched, so the escape never reaches the index.
    Runs LAST (after ``_linkify_block_urls``) so no later pass re-mangles the
    ``\$``; idempotent, so re-rendering an emitted page is a fixed point.
    """
    for ch in chapters:
        for block in ch.blocks:
            if block.html:
                block.html = escape_currency_dollars(block.html)


def _escape_math_angle_brackets(chapters: Sequence[_AdapterChapter]) -> None:
    r"""Escape raw ``<`` / ``>`` INSIDE math spans in block HTML ONLY (round-8).

    An OCR inequality glued to a letter (``\( a<b \)``) reaches the learner page
    as a literal ``<`` that the browser tokenizer reads as a phantom start tag —
    swallowing the rest of the ``\(…\)`` span, so MathJax leaks the ``\(`` as a
    visible backslash-paren and reds the orphan ``\)``. This HTML-only pass
    rewrites every raw ``<`` / ``>`` between math delimiters to ``&lt;`` /
    ``&gt;``; the browser decodes the entity so MathJax reads identical math and
    renders byte-identically, but no phantom tag ever opens.

    HTML-ONLY: ``raw_text`` / ``repaired_text`` (sidecar + chunker/retrieval
    text) keep plain ``x < 5`` untouched. Runs LAST (after
    ``_escape_currency_dollars``) so the ``\$`` currency escape is already
    settled and never mistaken for a span delimiter; idempotent.
    """
    for ch in chapters:
        for block in ch.blocks:
            if block.html:
                block.html = escape_math_angle_brackets(block.html)


def _sanitize_math_spans(chapters: Sequence[_AdapterChapter]) -> None:
    r"""Fold misplaced ``&`` + dangling ``\sqrt`` out of math spans (round-9).

    The headless render audit (``scripts/render_audit.py``) surfaced MathJax
    ``mjx-merror`` typeset failures that no text audit caught: a tabular ``&``
    the VLM pulled into a non-alignment ``$…$`` run ("Misplaced &") and an
    OCR-truncated ``\sqrt`` / ``\frac`` / ``\stackrel`` / trailing ``^``/``_``
    ("Missing argument"). :func:`~lib.semantik.math_fold.sanitize_math_spans`
    repairs BOTH inside the delimited span content, conservatively (alignment
    ``&`` inside ``\begin{array}…`` and a valid ``\sqrt{x}`` are untouched;
    ``&lt;``/``&gt;`` entities stay real operators).

    HTML-ONLY (mirrors ``_escape_math_angle_brackets``): ``raw_text`` /
    ``repaired_text`` keep the plain fused text for the chunker/retrieval; only
    the rendered learner page is repaired. Runs LAST (after the angle-bracket
    escape) so the ``&lt;``/``&gt;`` entities are already settled and the
    misplaced-``&`` fold never bites a real inequality; idempotent.
    """
    for ch in chapters:
        for block in ch.blocks:
            if block.html:
                block.html = sanitize_math_spans(block.html)


def _strip_tikz_figures(chapters: Sequence[_AdapterChapter]) -> None:
    r"""Replace TikZ/pgfplots figure code in math spans with an a11y placeholder (round-10).

    The final visual-convergence pass. The headless render audit surfaced a
    ``mjx-merror`` family the round-9 span sanitizer does not touch: the VLM
    transcribed coordinate-plane FIGURES as raw TikZ picture code inside math
    delimiters (``$$\begin{tikzpicture}…\end{tikzpicture}$$`` — MathJax reds it
    "Undefined environment tikzpicture"; also the pgfplots ``\begin{axis}…``
    sibling). This is figure content, not math — the corpus-wide figure story is
    the accepted scan-corpus limitation (``SEMANTIK_DETECT_FIGURES`` off) — so
    :func:`~lib.semantik.math_fold.strip_tikz_figures` replaces a pure-figure
    span with the accessible ``.dart-figure-notation`` placeholder (the raw TikZ
    source is noise to every reader and never ships visibly) and keeps the math
    of a mixed span (real math + embedded figure), dropping only the figure env.

    HTML-ONLY (mirrors ``_sanitize_math_spans``): ``raw_text`` / ``repaired_text``
    keep the plain fused text (TikZ and all) for the chunker/retrieval; only the
    rendered learner page is repaired. Runs LAST (after the round-9 span
    sanitizer) so the figure env is intact when this pass looks for it;
    idempotent (the emitted placeholder carries no ``\begin{tikz…}`` marker).
    """
    for ch in chapters:
        for block in ch.blocks:
            if block.html:
                block.html = strip_tikz_figures(block.html)


def _strip_body_folios(chapters: Sequence[_AdapterChapter]) -> None:
    """Drop / strip leaked printed folios (page numbers) from BODY blocks (Defect 2).

    Two arms, both mutate the IR in place (HTML + sidecar parity by
    construction):

    * ARM A — a non-heading content block whose ENTIRE text is a bare 2-4 digit
      number (:func:`is_standalone_folio`) is a page number that leaked into a
      standalone block; it is re-tagged ``metadata_drop`` so it is suppressed
      from both the rendered HTML and the sidecar (matching the existing folio
      furniture path).
    * ARM B — a Title-Case ``"Chapter N <Title …> <folio>"`` running header fused
      onto the TAIL of a body block (:func:`strip_trailing_running_header`) is
      stripped from ``html`` / ``raw_text`` / ``repaired_text``.

    A bare TRAILING folio with no running-header anchor is deliberately left
    alone (the physical ``pages`` metadata cannot be mapped to the printed folio,
    and a naive strip corrupts real answers) — see the heading_classifier
    module banner (j).
    """
    for ch in chapters:
        for block in ch.blocks:
            if _is_heading_block(block) or _is_furniture_block(block):
                continue
            if is_standalone_folio(block.raw_text or ""):
                block.region_kind = "metadata_drop"
                block.block_role = "metadata_drop"
                continue
            if block.html:
                stripped = strip_trailing_running_header(block.html)
                if stripped is not None:
                    block.html = stripped
            if block.raw_text:
                stripped = strip_trailing_running_header(block.raw_text)
                if stripped is not None:
                    block.raw_text = stripped
            if block.repaired_text:
                stripped = strip_trailing_running_header(block.repaired_text)
                if stripped is not None:
                    block.repaired_text = stripped


def _normalize_ocr_headings(
    chapters: Sequence[_AdapterChapter],
    doc_title: Optional[str] = None,
) -> int:
    """Demote OCR heading furniture / garbage across the adapter IR in place.

    Returns the count of Wave #22 unit-skeleton heading re-derivations (the
    mis-typed headings demoted because they severed a worked example) so the
    caller can report it.

    See the module-section banner above for the per-case mapping. Chapter TITLE
    furniture is neutralized via the continuation presentation-<div> (the first
    occurrence of a repeated running header stays a real <h2>).

    ``doc_title`` is the render-time chapter-title knowledge (the ``--title`` /
    ``--title-map`` override) consumed by the Rule C running-header arms: the
    bare "Chapter N <title>" demotion and the "Chapter N <title> <label>"
    prefix strip fire only when the tail matches this known title, so a
    legitimate "Chapter N …" heading is never demoted by shape alone.
    """
    # ITEM 3 (round-5 audit) — normalize the ``trvit`` OCR-confusable → ``TRY IT``
    # FIRST (before every scrub / split), so the de-garbled marker flows through
    # the marker scrub + heading/opener machinery instead of shipping as garble.
    _normalize_confusable_markers(chapters)

    # ch02 audit — scrub literal nbsp-entity artifact runs from block text
    # FIRST, so every predicate / split below (and the render + sidecar) sees
    # clean text. In-place, HTML/sidecar parity by construction.
    _scrub_block_entity_artifacts(chapters)

    # ITEM 1 (round-5 audit) — collapse pure gutter + garbled-marker DEBRIS blocks
    # (``>| TRYIT::``) to suppressed furniture. Runs BEFORE the marker scrub (the
    # predicate keys on the gutter glyphs the scrub would remove) and before
    # opener promotion / structure emit.
    _collapse_marker_debris_blocks(chapters)

    # ch02 audit — fold pedagogical-marker label residue (":: " / ": : " colon
    # runs + stray ">"/"|" gutter glyphs) so "TRY IT :: 2.1" reads "TRY IT 2.1"
    # BEFORE the opener split below — a de-doubled numbered marker now promotes.
    _scrub_block_marker_artifacts(chapters)

    # B3 (end-user-HTML audit) — fold visible text-mode LaTeX/markdown garbage
    # (\textbf / \textit / \begin{tabular} / | --- | / \checkmark) out of body
    # text, protecting $…$ math runs. Runs BEFORE the opener split so a
    # \textbf-decorated opener label ("\textbf{Learning Objectives} …") is
    # already clean when the split predicate sees it.
    _sanitize_block_body_latex(chapters)

    # Defect 2 (round-3 audit) — drop standalone leaked folios + strip a trailing
    # "Chapter N <Title> <folio>" running header fused into a body block. Runs
    # AFTER the body-latex sanitize (clean text) and BEFORE the opener / apparatus
    # splits (so a de-fused block is folio-free when the split predicates see it).
    _strip_body_folios(chapters)

    # Chapter-title furniture: page-numbered running headers, watermark garbage,
    # and repeats of a bare "Chapter N Title" (all but the first).
    titles = [ch.title or "" for ch in chapters]
    repeat_furniture = repeated_running_header_indices(titles)
    for i, ch in enumerate(chapters):
        if getattr(ch, "continuation", False):
            continue
        title = ch.title or ""
        # RULE C strip arm — a folio-prefixed REAL section title ("130 The
        # Real Numbers") keeps its heading; only the folio is furniture. Runs
        # BEFORE the demote predicates so the stripped title is what they see.
        stripped = strip_folio_prefix(title)
        if stripped is not None:
            ch.title = stripped
            title = stripped
        if (
            i in repeat_furniture
            or is_running_header(title)
            # RULE C demote arm — folio-prefixed "D Chapter N …" (any folio
            # length) or a bare "Chapter N <title>" matching the KNOWN chapter
            # title: a running header, not a section (the ch04 "Chapter 4
            # Graphs" ×2 phantom — below the ×3 repeat threshold).
            or is_chapter_running_header(title, chapter_title=doc_title)
            or is_watermark_garbage_heading(title)
            # Defect 3 — a chapter title that is an obviously-FUSED blob (footer
            # + exercises + a word problem swallowed into one heading) is not a
            # real chapter title; demote it out of the visible heading stream
            # (aria-hidden continuation div) so it never mints a spurious <h2>.
            or is_fused_heading(title)
        ):
            # Out of the heading stream (chunker no longer sees a phantom
            # boundary); blocks under it still render.
            ch.continuation = True

    # Defect f — a block that fused a leading apparatus heading ("Key Terms
    # <glossary prose>", paragraph- OR heading-typed) is split into a promoted
    # heading + remainder paragraph BEFORE the promotion/furniture loop below.
    # Ordering is load-bearing for the heading-typed arm: the split must WIN
    # over the `is_fused_heading` demotion in that loop (else the apparatus
    # name stays buried in a demoted paragraph — the ch09 re-render defect);
    # the loop then keeps the clean apparatus heading via `is_apparatus_heading`
    # and leaves the remainder paragraph alone. Rebuilds block lists in place.
    for ch in chapters:
        ch.blocks = _split_leading_apparatus_blocks(ch.blocks)

    # ITEM 4 (round-2 audit) — split a flat <p> that FUSED interior numbered
    # openers ("… TRY IT 9.201 … EXAMPLE 9.102 …") into one unit per opener.
    # Runs AFTER the apparatus split and BEFORE opener promotion, so the interior
    # opener headings it emits get their nesting level set by the promotion pass.
    for ch in chapters:
        ch.blocks = _split_interior_openers_in_blocks(ch.blocks)

    # A7 (end-user-HTML audit) — promote / split pedagogical openers (Learning
    # Objectives / Be Prepared / Try It / Example / How To / Solution) to real
    # <h4> headings carrying a data-dart-opener role. Runs AFTER the apparatus
    # split (so "Key Terms …" stays apparatus-owned) and BEFORE the heading
    # furniture loop (the promoted opener heading is then kept, not demoted).
    for ch in chapters:
        ch.blocks = _promote_openers_in_blocks(ch.blocks)

    # Heading-block furniture / labels / promotions.
    for ch in chapters:
        for block in ch.blocks:
            # Defect 4 — an apparatus heading ("PRACTICE TEST", "Review
            # Exercises") the pipeline demoted to a paragraph is promoted back
            # to a heading. Runs on NON-heading blocks (a paragraph whose whole
            # text is a standalone apparatus name); a real heading is untouched.
            if block.region_kind != "heading":
                para_text = (block.raw_text or "").strip()
                # RULE B — the standalone-exact-match predicate additionally
                # accepts a bare "Introduction" paragraph (ch04's page-1
                # opener), promoted via the same apparatus path. The
                # leading/interior SPLIT arms deliberately do NOT know
                # "Introduction" ("Introduction to Whole Numbers …" leading
                # prose must never split).
                if para_text and is_standalone_apparatus_heading(para_text):
                    _promote_paragraph_block_to_heading(block)
                continue
            if not block.heading_text:
                continue
            text = block.heading_text
            # RULE C prefix arm — a running header fused ONTO a real label
            # ("Chapter 6 Polynomials Solution"): strip the "Chapter N <known
            # title>" prefix and let the remainder re-enter the predicates
            # below (a bare "Solution" then takes the decorated-solution
            # demotion as-is).
            prefix_rem = strip_chapter_title_prefix(text, doc_title)
            if prefix_rem is not None:
                block.heading_text = prefix_rem
                text = prefix_rem
            else:
                # RULE C strip arm on heading blocks — folio-prefixed REAL
                # section heading keeps its title, sheds the folio.
                stripped = strip_folio_prefix(text)
                if stripped is not None:
                    block.heading_text = stripped
                    text = stripped
            # Defect B (coordinator follow-up, ch09 live validation) — a
            # heading-TYPED apparatus name ("PRACTICE TEST", "REVIEW
            # EXERCISES") is a REAL section heading: exempt it from every
            # furniture/garbage predicate below AND stamp the apparatus role so
            # the `_is_noncontent_heading` emit filter (which classifies these
            # EOC names as answer-key noise and was EATING the block entirely —
            # my paragraph-arm fix only covered paragraph-typed apparatus)
            # keeps it. A running header repeats across many pages; an
            # apparatus heading appears once per chapter, so repetition-based
            # furniture rules never own it.
            if is_apparatus_heading(text):
                block.block_role = _APPARATUS_HEADING_ROLE
                continue
            if is_running_header(text) or is_chapter_running_header(
                text, chapter_title=doc_title
            ):
                _drop_heading_block_as_furniture(block)
            elif is_decorated_solution_label(text):
                _demote_heading_block_to_paragraph(
                    block, css_class=SOLUTION_CSS_CLASS
                )
            elif is_watermark_garbage_heading(text):
                _demote_heading_block_to_paragraph(block)
            elif is_fused_heading(text):
                # Defect 3 — a section heading whose text is an obviously-fused
                # blob (sentence-length prose or multiple $…$ runs) is demoted
                # to prose (refuse to keep obviously-fused text as a heading).
                _demote_heading_block_to_paragraph(block)

    # Wave #22 Tier-1 — unit-skeleton heading re-derivation (directive 1): demote
    # the unambiguous mis-typed heading severing a worked example's statement from
    # its solution, so the composite unit re-forms and the TOC sheds the stray
    # entry. Runs AFTER opener promotion / the furniture loop (opener roles final)
    # and BEFORE the flow pass (a demoted heading no longer splits the segment).
    releveled = _rederive_unit_headings(chapters)

    # Wave #22 Tier-1 — reading-order FLOW annotation (directive 2): stamp
    # statement / solution-steps / procedure-steps on content blocks within a
    # pedagogical unit. Annotation-only; runs after re-derivation.
    _annotate_reading_order_flow(chapters)

    # A1/A4/A5 (exemplar-parity wave) — deliver <ul>/<ol>/<table>/<dl> bodies for
    # high-confidence shapes and reconcile mis-declared structural roles to
    # paragraph. Runs LAST so it sees the post-split / post-demotion block set
    # (a glossary remainder split out of a "Key Terms" heading can become a <dl>)
    # and the SANITIZED body text (bare math already wrapped in $…$).
    _emit_structured_bodies(chapters)

    # B3+ final self-balance sweep — the apparatus / opener splits above rebuild
    # <p> bodies from cut text, which can slice a $$…$$ / \[…\] display span
    # across the new block boundary (leaving an orphan opener/closer). Re-running
    # the sanitizer (idempotent on already-clean bodies) guarantees EVERY emitted
    # block is delimiter-balanced, so the whole-document math strip never desyncs.
    _sanitize_block_body_latex(chapters)

    # ITEM 1 (round-2 audit) — linkify bare / angle-wrapped vendor URLs LAST, so
    # the emitted mathjax_ignore <a> anchors are past the final wrap_bare_math
    # sweep and never re-mangled. Kills the MathJax-italic URL soup.
    _linkify_block_urls(chapters)

    # Round-7b — escape PRESERVED currency ``$`` (``$5``) → ``\$`` in block HTML
    # ONLY, so two currency amounts in one paragraph never FALSE-PAIR into a
    # MathJax italic span at render. HTML-only (raw_text/sidecar keep plain
    # ``$5``); runs LAST so the ``\$`` is past every other html transform.
    _escape_currency_dollars(chapters)

    # Round-8 — escape raw ``<`` / ``>`` INSIDE math spans (``\( a<b \)``) → HTML
    # entities in block HTML ONLY, so an OCR inequality glued to a letter can
    # never open a phantom browser tag that slices the span and leaks the ``\(``
    # / reds the ``\)``. HTML-only; runs LAST so the currency ``\$`` is settled.
    _escape_math_angle_brackets(chapters)

    # Round-9 — fold misplaced tabular ``&`` (non-alignment math) + dangling
    # ``\sqrt`` / ``\frac`` / ``\stackrel`` / ``^``/``_`` out of math-span
    # CONTENT so MathJax stops emitting ``mjx-merror`` typeset errors the
    # headless render audit catches. HTML-only; runs LAST so the ``&lt;``/``&gt;``
    # entities the angle-bracket pass just wrote are settled first.
    _sanitize_math_spans(chapters)

    # Round-10 (final) — replace VLM-emitted TikZ / pgfplots FIGURE code inside
    # math spans (``$$\begin{tikzpicture}…\end{tikzpicture}$$`` coordinate-plane
    # graphs) with an accessible ``.dart-figure-notation`` placeholder so MathJax
    # stops emitting "Undefined environment tikzpicture" ``mjx-merror`` nodes and
    # the raw TikZ source never ships visibly. HTML-only; runs LAST so the round-9
    # sanitizer's edits are settled and the figure env is intact when found.
    _strip_tikz_figures(chapters)

    return releveled


# A real chapter opener: "Chapter N[: Title]" (requires the ordinal digit, so
# a front-matter "Chapter Outline" — no digit — never matches).
_CHAPTER_N_TITLE_RE = re.compile(r"^\s*chapter\s+\d+\b", re.IGNORECASE)

# End-of-chapter apparatus qualifiers that make a "Chapter N …" heading an
# exercise/review/outline SECTION, not the chapter title itself. "CHAPTER 9
# REVIEW" matches ``_CHAPTER_N_TITLE_RE`` (it IS "Chapter 9 …") but is the
# end-of-chapter review banner, so it must NOT be picked as the document title.
# Matched case-insensitively anywhere in the heading (OpenStax emits "Chapter N
# Review", "Chapter N Exercises", "Chapter N Key Terms", "Chapter N Practice
# Test", and a front-matter "Chapter N Outline").
_CHAPTER_APPARATUS_RE = re.compile(
    r"\b(review|outline|exercises?|key\s*terms?|practice\s*test)\b",
    re.IGNORECASE,
)


def _select_document_title(
    chapters: Sequence[_AdapterChapter], fallback: str
) -> str:
    """Pick the document ``<h1>`` / ``<title>`` string (§3.1, defect 3d).

    Prefers the EARLIEST chapter whose title is a genuine ``"Chapter N <Title>"``
    opener over a front-matter / end-of-chapter artifact ("Chapter Outline",
    "CHAPTER 9 REVIEW", a derived "Document" placeholder). The h1 was previously
    ``chapters[0].title`` verbatim, so a leading front-matter "Chapter Outline"
    heading minted a misleading document title; and the naive "Chapter N"-prefix
    check wrongly accepted "CHAPTER 9 REVIEW" (the end-of-chapter review banner)
    as the document title. A genuine "Chapter 9 Roots and Radicals" opener —
    carrying NO review/outline/exercise/key-terms/practice-test qualifier — is
    the honest title. Falls back to the first chapter title, then the stem —
    never fabricates.
    """
    for ch in chapters:
        title = ch.title or ""
        if _CHAPTER_N_TITLE_RE.match(title) and not _CHAPTER_APPARATUS_RE.search(
            title
        ):
            return ch.title
    return chapters[0].title if chapters else fallback


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
    source: Optional[str] = None,
    title_override: Optional[str] = None,
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
    title_override
        Optional explicit document ``<h1>`` / ``<title>`` string. When
        non-empty it bypasses :func:`_select_document_title` (used by
        ``scripts/semantik_rerender.py`` ``--title`` / ``--title-map`` so a
        batch re-render can pin honest titles over OCR running-header noise).

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
    source_value = _resolve_source_value(source)

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

    # Title: prefer a real "Chapter N <Title>" chapter, else the first chapter
    # title, else the stem. An explicit ``title_override`` (rerender --title /
    # --title-map) wins outright, bypassing the heuristic.
    chapters = _extract_chapters_from_result(cascade_result)
    # OCR heading-furniture normalization (2026-07-03 scan audit) — demote
    # running-header / watermark / decorated-Solution headings BEFORE title
    # selection + render so both the conversion and re-render paths are clean.
    # The title override (--title / --title-map) doubles as the Rule C
    # chapter-title knowledge for the conservative running-header arms.
    override = (title_override or "").strip()
    heading_releveling_count = _normalize_ocr_headings(
        chapters, doc_title=override or None
    )
    title = override or _select_document_title(chapters, pdf_stem)
    lang = getattr(cascade_result, "lang", None) or "en"

    html = _render_html(chapters, title=title, lang=lang, source_value=source_value)
    success, certification_status = _resolve_success(exit_action)

    synthesized_sidecar = build_synthesized_sidecar(
        chapters,
        title=title,
        source_pdf=pdf_stem,
        slug=slug,
        source_value=source_value,
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
        "data_dart_source": source_value,
        "slug": slug,
        "canonical_course_code": canonical_course_code,
        "figures_dir": figures_dir,
        "synthesized_sidecar": synthesized_sidecar,
        "quality_sidecar": quality_sidecar,
        # Wave #22 Tier-1 — count of unit-skeleton heading re-derivations.
        "heading_releveling_count": heading_releveling_count,
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
