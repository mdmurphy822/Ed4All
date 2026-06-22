"""Stage-5d-det — the DETERMINISTIC structure-correction pass.

The 70B Stage-5d reviewer's prompt (``reviewer_prompt.py``) explicitly defers
front-matter / TOC handling to "a separate DETERMINISTIC pass". This module IS
that pass. It runs BEFORE the 70B reviewer (and before the per-kind cap +
assembly) so BOTH the assembled HTML and the ``region_provenance`` IR pick up
the corrected region list — they derive from the same list.

Three CPU-only, model-free sub-passes, all reusing the reviewer's invariant
harness (admission + token-conservation + fail-closed; re-tag via
``dataclasses.replace``; the FB partition is immutable):

  (A) FRONT-MATTER DROP — re-tag leading title-page / preface / TOC heading
      regions to ``metadata_drop`` using FB-page position + the ported
      ``toc_frontmatter_detector`` predicates + the reviewer's cluster-signal
      phantom-TOC discriminator. Kills the phantom "Chapter 5/7/8/9/10" run and
      OCR title-page noise ("O PEN S TAX" / "R ICE U NIVERSITY"). Anti-FP: only
      the early-page front-matter zone; never a real body chapter / section.

  (B) PEDAGOGICAL DEMOTION — regex-match heading regions whose text is an
      OpenStax pedagogical label ("EXAMPLE N", "Solution", "Try It", "How To",
      "Step N", "Learning Objectives", "Be Prepared", "Practice Makes
      Perfect") and demote the region kind heading -> paragraph (no pedagogical
      Region.kind exists yet; demotion removes the h4-h6 over-nesting while
      preserving the text verbatim). The demoted block ALSO carries a semantic
      class hint on ``payload['css_class']`` mapped from the matched label
      (EXAMPLE -> "pedagogy-example", Solution -> "pedagogy-solution", …) so
      the worked-example / solution semantics survive the demotion to <p>
      (styleable + chunker-readable + a training signal) pending the eventual
      pedagogical-Region retrain. Real section headings ("1.1 Introduction to
      Whole Numbers", "Chapter 1") are NOT demoted.

  (C) HEADING NORMALIZATION — left to the assembler's existing
      ``normalize_heading_levels`` (``heading_tree.py``); this pass only needs
      to leave a clean surviving heading set. No re-leveling here.

Gated by ``SEMANTIK_STRUCTURE_CLEAN`` (default ON — deterministic + fixes real
defects). A falsey value (``0`` / ``false`` / ``no`` / ``off``) makes the pass
byte-identical pass-through (returns the input region list unchanged).

Invariant contract (shared with the 70B reviewer):
  * the FB partition is immutable (``feature_block_indices`` /
    ``source_region_id`` never change — every re-tag is ``dataclasses.replace``
    of ``kind`` / ``payload`` only);
  * text is preserved VERBATIM (a metadata_drop / demotion keeps
    ``payload['text']`` byte-identical);
  * document-level token-conservation holds (``kept ⊎ dropped == source``),
    asserted via the reviewer's ``assert_token_conservation``;
  * fail-closed — a conservation failure reverts the WHOLE pass to the input
    region list (never ships a content-losing correction set).
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any

from ..structure_graph import Region
from ..types import FeatureBlock
from .reviewer import (
    TokenConservationError,
    _region_payload_text,
    assert_token_conservation,
    compute_cluster_signals,
)

# ---------------------------------------------------------------------------
# Env gate.
# ---------------------------------------------------------------------------

_FLAG_ENV = "SEMANTIK_STRUCTURE_CLEAN"
_FALSEY = {"0", "false", "no", "off"}


def resolve_structure_clean_mode() -> bool:
    """Whether the deterministic structure-correction pass runs.

    Reads ``SEMANTIK_STRUCTURE_CLEAN``, DEFAULT ON (parse-with-fallback). Only
    an explicit falsey value (``0`` / ``false`` / ``no`` / ``off``,
    case-insensitive) disables it; unset / truthy / garbage keeps it on
    (deterministic + fixes real defects). Mirrors
    ``toc_frontmatter_detector._flag_enabled``.
    """
    raw = os.environ.get(_FLAG_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


# ---------------------------------------------------------------------------
# Region text + page helpers (Region-native; the ported toc_frontmatter_detector
# predicates key on text + the front-matter page zone).
# ---------------------------------------------------------------------------


def _region_text(region: Region) -> str:
    """Best text signal for a region — payload['text'], stripped."""
    return _region_payload_text(region).strip()


def _region_first_page(
    region: Region, feature_blocks: list[FeatureBlock]
) -> int | None:
    """The smallest positive physical page any owned FeatureBlock appears on.

    A region whose FBs carry no positive page is page-unknown and never enters
    the early-page front-matter zone (so it is never dropped by the positional
    rule). Mirrors ``toc_frontmatter_detector._first_page`` but resolves the
    page from the owned FeatureBlocks rather than a ``pages`` provenance list.
    """
    pages: list[int] = []
    for idx in region.feature_block_indices or ():
        if not (0 <= idx < len(feature_blocks)):
            continue
        raw = getattr(feature_blocks[idx], "raw", None)
        page = getattr(raw, "page", None)
        try:
            p = int(page)
        except (TypeError, ValueError):
            continue
        if p > 0:
            pages.append(p)
    return min(pages) if pages else None


def _is_heading(region: Region) -> bool:
    return str(region.kind) == "heading"


# ---------------------------------------------------------------------------
# Ported toc_frontmatter_detector predicates (Region-native copies).
#
# These mirror ``lib/semantik/toc_frontmatter_detector.py`` predicate-for-
# predicate (the task forbids editing that module, so its logic is copied here
# and re-pointed at Region objects). The text-shape regexes + the OpenStax
# front-matter zone constants are byte-identical to the source.
# ---------------------------------------------------------------------------

# A real chapter heading shape: "Chapter N[: Title]".
_CHAPTER_HEADING_RE = re.compile(r"^\s*chapter\s+(\d+)\b", re.IGNORECASE)
# A bare-ordinal chapter-index shape: "N Title" (no "Chapter", no trailing page).
_BARE_ORDINAL_HEADING_RE = re.compile(r"^\s*(\d{1,3})\s+(?=[A-Za-z])\S")
# A real section heading shape: "N.M[.K] Title".
_SECTION_HEADING_RE = re.compile(r"^\s*\d+\.\d+(?:\.\d+)*\s+\S")
# A TOC-entry shape: title (>=1 letter) + leader + trailing page number.
_TOC_ENTRY_RE = re.compile(
    r"^\s*(?P<title>.*?[A-Za-z].*?)"
    r"[\s\.…]+"
    r"(?P<page>\d{1,5})\s*$"
)

# Front-matter zone + page-density constants (byte-identical to the source).
_ZONE_MIN_CHAPTER_CLUSTER = 4
_ZONE_MAX_PAGE_WINDOW = 3
_ZONE_MAX_FRONTMATTER_PAGE = 16
_MIN_CHAPTER_INDEX_RUN = 3

# OCR title-page noise — letter-spaced caps the OCR mangles a title page into
# ("O PEN S TAX" -> ['O','PEN','S','TAX'], "R ICE U NIVERSITY" ->
# ['R','ICE','U','NIVERSITY']). The signature is letter-spacing: a stylized
# title-page word gets a SPACE injected after its (usually drop-cap) first
# letter, leaving a single-letter caps token followed by an all-caps remainder
# ("O"+"PEN STAX", "U"+"NIVERSITY"). A heading whose tokens are ALL uppercase
# alphabetic AND that carries >=2 single-letter caps tokens is this artifact —
# a real all-caps title ("LEARNING OBJECTIVES") has at most one stray initial.
# Conservative: only fires inside the early-page front-matter zone.
_OCR_ALLCAPS_TOKEN_RE = re.compile(r"^[A-Z]+$")
_OCR_SINGLE_CAP_RE = re.compile(r"^[A-Z]$")


def _toc_entry_page(text: str) -> int | None:
    """The trailing page number if ``text`` is a TOC-entry shape, else None."""
    if not text:
        return None
    m = _TOC_ENTRY_RE.match(text)
    if not m:
        return None
    title = m.group("title").strip()
    if not re.search(r"[A-Za-z]", title):
        return None
    try:
        return int(m.group("page"))
    except (TypeError, ValueError):
        return None


def _chapter_index_ordinal(region: Region) -> int | None:
    """The chapter ordinal if a heading region is a chapter-INDEX-pattern entry.

    Matches "Chapter N[: Title]" or a bare-ordinal "N Title" carrying NO
    trailing page number (a trailing page number -> a real TOC line). A
    "N.M Title" section heading is NOT a chapter-index entry. Mirrors
    ``toc_frontmatter_detector._chapter_index_ordinal``.
    """
    if not _is_heading(region):
        return None
    text = _region_text(region)
    if not text:
        return None
    if _toc_entry_page(text) is not None:
        return None
    m = _CHAPTER_HEADING_RE.match(text)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None
    if _SECTION_HEADING_RE.match(text):
        return None
    m2 = _BARE_ORDINAL_HEADING_RE.match(text)
    if m2:
        try:
            return int(m2.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _is_real_chapter_or_section(region: Region) -> bool:
    """Whether a heading region is a genuine numbered chapter/section opener.

    Anchors ONLY on an explicit "Chapter N[: Title]" / "N.M Title" numbered
    heading that is NOT itself a TOC line (no trailing page number). Mirrors
    ``toc_frontmatter_detector._is_real_chapter_or_section``.
    """
    if not _is_heading(region):
        return False
    text = _region_text(region)
    if not text:
        return False
    if _toc_entry_page(text) is not None:
        return False
    return bool(
        _CHAPTER_HEADING_RE.match(text) or _SECTION_HEADING_RE.match(text)
    )


def _is_ocr_titlepage_noise(text: str) -> bool:
    """Whether ``text`` is OCR-mangled title-page noise ("O PEN S TAX").

    Fires when a heading is >=3 whitespace tokens, EVERY token is uppercase
    alphabetic, AND >=2 tokens are single-letter caps — the letter-spacing
    artifact of OCR over a stylized title page ("O PEN S TAX", "R ICE U
    NIVERSITY"). A real all-caps title ("LEARNING OBJECTIVES") carries at most
    one stray single-letter token, and a real heading ("1.1 Introduction…") has
    digits / mixed-case words, so neither matches.
    """
    tokens = text.split()
    if len(tokens) < 3:
        return False
    if not all(_OCR_ALLCAPS_TOKEN_RE.match(t) for t in tokens):
        return False
    single_caps = sum(1 for t in tokens if _OCR_SINGLE_CAP_RE.match(t))
    return single_caps >= 2


# ---------------------------------------------------------------------------
# (A) Front-matter / phantom-TOC / OCR-noise DROP detection.
# ---------------------------------------------------------------------------


def _front_matter_zone_end(
    regions: list[Region],
) -> int:
    """Index of the first REAL numbered chapter/section heading (zone anchor).

    Everything strictly before it is the eligible front-matter zone; at/after
    is PROTECTED content. A TOC line never anchors. Returns ``len(regions)``
    when no real chapter is found (fail-safe: keep everything). Mirrors
    ``toc_frontmatter_detector._front_matter_zone_end``.
    """
    for idx, region in enumerate(regions):
        text = _region_text(region)
        if _toc_entry_page(text) is not None:
            continue
        if _is_real_chapter_or_section(region):
            return idx
    return len(regions)


def _detect_front_matter_drops(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    cluster_signals: list[Any],
) -> set[int]:
    """Indices of leading front-matter / phantom-TOC / OCR-noise HEADING regions
    to re-tag metadata_drop.

    Three deterministic signals, all confined to the early-page front-matter
    zone (never a real body chapter/section):

      1. PHANTOM chapter-index headings — a heading whose cluster signal is a
         same-level run with NO content following (``content_blocks_following
         == 0``) AND that is a chapter-pattern entry, OR a contiguous dense
         page-window chapter cluster (the EA2e phantom "Chapter 5/7/8/9/10").
      2. TOC lines — a heading whose text is a "<title> <trailing-pagenum>"
         shape (a printed table-of-contents line).
      3. OCR title-page noise — a heading that is letter-spaced caps fragments.

    Anti-FP: only headings strictly before the first real numbered chapter
    anchor (the front-matter zone) are eligible; a real "Chapter 1" / "1.1 …"
    opener at/after the anchor is NEVER dropped.
    """
    drops: set[int] = set()
    zone_end = _front_matter_zone_end(regions)

    # The page-density chapter cluster + early-page guard need a notion of the
    # early front-matter page zone. Use the FB-resolved first page.
    first_pages = [_region_first_page(r, feature_blocks) for r in regions]

    for idx, region in enumerate(regions):
        if not _is_heading(region):
            continue
        text = _region_text(region)
        if not text:
            continue

        in_zone = idx < zone_end
        page = first_pages[idx]
        early = page is not None and page <= _ZONE_MAX_FRONTMATTER_PAGE

        # (2) TOC line — anywhere in the zone (a printed TOC line carries a
        # trailing page number; real body headings do not).
        if in_zone and _toc_entry_page(text) is not None:
            drops.add(idx)
            continue

        # (3) OCR title-page noise — only in the early front-matter zone.
        if in_zone and early and _is_ocr_titlepage_noise(text):
            drops.add(idx)
            continue

        # (1) Phantom chapter-index entry — a chapter-pattern heading with no
        # content following it (cluster signal), inside the front-matter zone.
        ordinal = _chapter_index_ordinal(region)
        if ordinal is not None and in_zone:
            sig = cluster_signals[idx] if idx < len(cluster_signals) else None
            cbf = getattr(sig, "content_blocks_following", 0) if sig else 0
            run_len = getattr(sig, "same_level_run_len", 0) if sig else 0
            # A phantom entry: part of a same-level run AND no content beneath
            # it (the TOC/index shape), OR an early-page dense cluster member.
            if (run_len >= 2 and cbf == 0) or early:
                drops.add(idx)
                continue

    # Page-density cluster pass — a contiguous run of >= K chapter-pattern
    # headings within a <= P page window on an early front-matter page is a
    # preface chapter-summary / chapter index; drop the phantom HEADINGS whole.
    drops |= _detect_dense_chapter_cluster(regions, first_pages)
    return drops


def _detect_dense_chapter_cluster(
    regions: list[Region],
    first_pages: list[int | None],
) -> set[int]:
    """Page-density front-matter chapter clusters (the robust positional rule).

    >= ``_ZONE_MIN_CHAPTER_CLUSTER`` chapter-pattern headings inside a
    ``<= _ZONE_MAX_PAGE_WINDOW`` page span whose first heading sits on a page
    ``<= _ZONE_MAX_FRONTMATTER_PAGE`` is a front-matter cluster (dropped whole).
    Mirrors ``toc_frontmatter_detector._find_frontmatter_zone_clusters``.
    """
    drops: set[int] = set()
    n = len(regions)
    i = 0
    while i < n:
        ordinal = _chapter_index_ordinal(regions[i])
        first_pg = first_pages[i] if ordinal is not None else None
        if ordinal is None or first_pg is None:
            i += 1
            continue
        run = [i]
        run_first_page = first_pg
        run_last_page = first_pg
        last_chapter_pos = i
        j = i + 1
        while j < n:
            o2 = _chapter_index_ordinal(regions[j])
            if o2 is None:
                j += 1
                continue
            pg2 = first_pages[j]
            if pg2 is None:
                j += 1
                continue
            if pg2 - run_first_page > _ZONE_MAX_PAGE_WINDOW:
                break
            run.append(j)
            run_last_page = max(run_last_page, pg2)
            last_chapter_pos = j
            j += 1
        span = run_last_page - run_first_page
        if (
            len(run) >= _ZONE_MIN_CHAPTER_CLUSTER
            and span <= _ZONE_MAX_PAGE_WINDOW
            and run_first_page <= _ZONE_MAX_FRONTMATTER_PAGE
        ):
            drops.update(run)
        i = max(last_chapter_pos + 1, i + 1)
    return drops


# ---------------------------------------------------------------------------
# (B) Pedagogical-label demotion detection.
# ---------------------------------------------------------------------------

# OpenStax pedagogical labels that the council mis-fires as headings (deep
# h4-h6 over-nesting). Matched at the START of the heading text. A real
# numbered section heading ("1.1 Introduction…", "Chapter 1") never matches.
#
# SINGLE SOURCE OF TRUTH: an ordered (per-label anchored-pattern, semantic
# CSS class hint) list. When a pedagogical heading is demoted to a paragraph
# (sub-pass B) the demoted block loses its heading kind, so the matched
# label's CLASS HINT is stamped on ``payload['css_class']`` to preserve the
# worked-example / solution / step / objective semantics (styleable +
# chunker-readable + a training signal) until the pedagogical-Region retrain
# lands. The combined demotion regex (`_PEDAGOGICAL_LABEL_RE`) is DERIVED from
# this list so the match predicate and the class resolver never drift.
#
# Order matters: the first matching pattern wins (the patterns are mutually
# exclusive on real OpenStax labels, but a defined order makes the resolution
# deterministic and auditable).
_PEDAGOGICAL_LABEL_CLASSES: tuple[tuple[re.Pattern[str], str], ...] = (
    # EXAMPLE / EXAMPLE 1 / EXAMPLE 1.2
    (re.compile(r"^\s*EXAMPLE(?:\s+\d+(?:\.\d+)*)?", re.IGNORECASE),
     "pedagogy-example"),
    (re.compile(r"^\s*Solution\b", re.IGNORECASE), "pedagogy-solution"),
    (re.compile(r"^\s*Try\s+It\b", re.IGNORECASE), "pedagogy-try-it"),
    (re.compile(r"^\s*How\s+To\b", re.IGNORECASE), "pedagogy-how-to"),
    (re.compile(r"^\s*Step\s+\d+\b", re.IGNORECASE), "pedagogy-step"),
    (re.compile(r"^\s*Learning\s+Objectives\b", re.IGNORECASE),
     "pedagogy-objectives"),
    (re.compile(r"^\s*Be\s+Prepared\b", re.IGNORECASE), "pedagogy-be-prepared"),
    (re.compile(r"^\s*Practice\s+Makes\s+Perfect\b", re.IGNORECASE),
     "pedagogy-practice"),
)


def _pedagogical_class_for(text: str) -> str | None:
    """The semantic CSS class hint for a pedagogical-label ``text``, or None.

    Walks the single-source-of-truth ``_PEDAGOGICAL_LABEL_CLASSES`` list in
    order; the first anchored pattern that matches wins. Returns None when no
    pedagogical label matches (not a demotable label).
    """
    if not text:
        return None
    for pattern, css_class in _PEDAGOGICAL_LABEL_CLASSES:
        if pattern.match(text):
            return css_class
    return None


def _is_pedagogical_label(region: Region) -> bool:
    """Whether a heading region's text is an OpenStax pedagogical label.

    A real numbered section/chapter heading ("Chapter 1", "1.1 Title") is
    explicitly NOT a pedagogical label (the patterns anchor on the label
    words, never on a numeric section prefix).
    """
    if not _is_heading(region):
        return False
    text = _region_text(region)
    if not text:
        return False
    # Never demote a real numbered chapter/section heading.
    if _CHAPTER_HEADING_RE.match(text) or _SECTION_HEADING_RE.match(text):
        return False
    return _pedagogical_class_for(text) is not None


# ---------------------------------------------------------------------------
# Re-tag application (FB-partition-immutable, text-verbatim).
# ---------------------------------------------------------------------------


def _retag(
    region: Region, new_kind: str, *, css_class: str | None = None
) -> Region:
    """Return a copy of ``region`` re-tagged to ``new_kind``.

    Only ``kind`` (and a ``payload`` provenance breadcrumb, plus an OPTIONAL
    semantic ``css_class`` hint) changes — the FB partition
    (``feature_block_indices`` / ``source_region_id``) and the verbatim
    ``payload['text']`` are preserved (``dataclasses.replace`` of a frozen
    Region). Mirrors the reviewer's admission contract.

    When ``css_class`` is given (a pedagogical-demotion class hint), it is
    stamped on ``payload['css_class']`` so the downstream assembler emits
    ``<p class="…">`` and the bridge provenance can surface the pedagogical
    semantics lost when the heading kind is demoted to paragraph.
    """
    payload = dict(region.payload or {})
    prov = dict(payload.get("structure_clean") or {})
    prov["from_kind"] = str(region.kind)
    payload["structure_clean"] = prov
    if css_class:
        payload["css_class"] = css_class
    return dataclasses.replace(region, kind=new_kind, payload=payload)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def clean_structure(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
) -> tuple[list[Region], dict[str, Any]]:
    """Deterministic Stage-5d structure correction.

    Runs sub-passes (A) front-matter/phantom-TOC/OCR drop + (B) pedagogical
    demotion over ``regions`` (sub-pass C heading normalization is the
    assembler's job). Returns ``(corrected_regions, diagnostics)``.

    The pass is GATED by ``SEMANTIK_STRUCTURE_CLEAN`` (default ON); when off it
    is byte-identical pass-through. Token-conservation is asserted before
    return; on failure the WHOLE pass is reverted to the input list (fail-closed
    — never ships content-losing corrections). Text is preserved verbatim and
    the FB partition is immutable.
    """
    n = len(regions)
    headings_before = _heading_level_histogram(regions)
    diagnostics: dict[str, Any] = {
        "enabled": resolve_structure_clean_mode(),
        "front_matter_dropped": 0,
        "pedagogical_demoted": 0,
        "pedagogical_classed": 0,
        "pedagogical_class_counts": {},
        "headings_before": sum(headings_before.values()),
        "headings_after": sum(headings_before.values()),
        "headings_before_by_level": headings_before,
        "headings_after_by_level": dict(headings_before),
        "text_changes": 0,
        "reverted_for_invariant": False,
    }
    if not resolve_structure_clean_mode() or n == 0:
        return list(regions), diagnostics

    cluster_signals = compute_cluster_signals(regions)

    # (A) Front-matter / phantom-TOC / OCR-noise -> metadata_drop.
    fm_drops = _detect_front_matter_drops(regions, feature_blocks, cluster_signals)

    # (B) Pedagogical-label headings -> paragraph (only those NOT already
    # being dropped in (A) — a dropped phantom never needs demoting).
    peda_demote = {
        i
        for i, region in enumerate(regions)
        if i not in fm_drops and _is_pedagogical_label(region)
    }

    corrected: list[Region] = []
    pedagogical_class_counts: dict[str, int] = {}
    for i, region in enumerate(regions):
        if i in fm_drops:
            corrected.append(_retag(region, "metadata_drop"))
        elif i in peda_demote:
            # Stamp the matched pedagogical label's semantic class hint so the
            # demoted <p> keeps its worked-example / solution / step / objective
            # semantics (styleable + chunker-readable training signal).
            css_class = _pedagogical_class_for(_region_text(region))
            corrected.append(_retag(region, "paragraph", css_class=css_class))
            if css_class:
                pedagogical_class_counts[css_class] = (
                    pedagogical_class_counts.get(css_class, 0) + 1
                )
        else:
            corrected.append(region)

    # Fail-closed token-conservation: re-tags must preserve every token (only
    # metadata_drop removes tokens, and that is the intentional-drop accounting
    # the reviewer's check verifies). On failure, revert the whole pass.
    try:
        assert_token_conservation(regions, corrected, feature_blocks)
    except TokenConservationError as exc:  # pragma: no cover - safety net
        diagnostics["reverted_for_invariant"] = True
        diagnostics["revert_reason"] = str(exc)
        return list(regions), diagnostics

    headings_after = _heading_level_histogram(corrected)
    diagnostics["front_matter_dropped"] = len(fm_drops)
    diagnostics["pedagogical_demoted"] = len(peda_demote)
    diagnostics["pedagogical_classed"] = sum(pedagogical_class_counts.values())
    diagnostics["pedagogical_class_counts"] = pedagogical_class_counts
    diagnostics["headings_after"] = sum(headings_after.values())
    diagnostics["headings_after_by_level"] = headings_after
    diagnostics["text_changes"] = _count_text_changes(regions, corrected)
    return corrected, diagnostics


def _heading_level_histogram(regions: list[Region]) -> dict[str, int]:
    """Count heading regions by h1-h6 level (level_hint), for the audit.

    A heading with a missing / out-of-range level lands in ``"unknown"``.
    """
    hist: dict[str, int] = {f"h{lvl}": 0 for lvl in range(1, 7)}
    hist["unknown"] = 0
    for region in regions:
        if not _is_heading(region):
            continue
        level = (region.payload or {}).get("level_hint")
        try:
            lvl = int(level)
        except (TypeError, ValueError):
            hist["unknown"] += 1
            continue
        if 1 <= lvl <= 6:
            hist[f"h{lvl}"] += 1
        else:
            hist["unknown"] += 1
    # Drop all-zero buckets for a compact report.
    return {k: v for k, v in hist.items() if v}


def _count_text_changes(
    before: list[Region], after: list[Region]
) -> int:
    """Number of regions whose payload text changed (must be 0 — verbatim)."""
    changes = 0
    for b, a in zip(before, after):
        if _region_payload_text(b) != _region_payload_text(a):
            changes += 1
    return changes


__all__ = [
    "clean_structure",
    "resolve_structure_clean_mode",
    "_pedagogical_class_for",
]
