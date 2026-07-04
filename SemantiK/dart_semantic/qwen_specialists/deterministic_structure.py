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

from ..structure_graph import (
    Region,
    _vlm_struct_hints_enabled,
    resolve_bert_authoritative,
)
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
_TRUTHY = {"1", "true", "yes", "on"}

# Defect 3(c) — CONSERVATIVE inline-section-heading PROMOTION gate. Default OFF:
# a paragraph→heading promotion is behaviour-changing (it re-shapes the chapter
# structure), so it follows the house flag pattern rather than the always-on
# default path used by the outright-bug fixes (furniture / OCR-label demotion).
_PROMOTE_FLAG_ENV = "SEMANTIK_PROMOTE_SECTION_HEADINGS"


def resolve_promote_section_headings_mode() -> bool:
    """Whether the conservative N.M section-heading PROMOTION pass runs.

    Reads ``SEMANTIK_PROMOTE_SECTION_HEADINGS``, DEFAULT OFF (parse-with-
    fallback). Only an explicit truthy value (``1`` / ``true`` / ``yes`` /
    ``on``, case-insensitive) enables it; unset / falsey / garbage keeps it off
    (byte-identical pass-through). Promotes a STANDALONE paragraph region whose
    whole text is a clean ``"N.M Title-Case Words"`` section heading the council
    BERT mis-typed as body prose, so a real section heading that was demoted to
    a ``<p>`` becomes an ``<h2>`` again (and re-enables the cascade_ir
    section-number chapter derivation).
    """
    raw = os.environ.get(_PROMOTE_FLAG_ENV)
    if raw is None:
        return False
    return str(raw).strip().lower() in _TRUTHY


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
# Defect 3(a) — a RUNNING-HEADER heading carrying a trailing page number:
# "Chapter 9 Roots and Radicals 1039". This is page furniture the
# FB-position-based ``structure_graph._detect_running_headers`` MISSES when OCR
# bbox noise pushes the recurring header out of the top/bottom margin band, so
# it survives as a heading and is promoted to a bogus <h2> (39× on the EA2e
# scan). Text-based backstop: a "Chapter N <words> <3-4 digit page>" heading is
# furniture ANYWHERE in the document (not zone-confined). Conservative — the
# "Chapter N" prefix AND a standalone trailing 3-4-digit page number are BOTH
# required, so a real chapter title "Chapter 9 Roots and Radicals" (no trailing
# number) never matches.
_RUNNING_HEADER_TEXT_RE = re.compile(
    r"^\s*chapter\s+\d+\b.*\s\d{2,4}\s*$", re.IGNORECASE
)
# Defect 3(b) — the LEADING-page-number running-header form the audit surfaced
# ("188 Chapter 1 Foundations", "686 Chapter 6 Polynomials"): a 2-4-digit page
# number, then "Chapter N …". Same furniture as the trailing form; caught
# anywhere in the doc.
_RUNNING_HEADER_LEADING_PAGE_RE = re.compile(
    r"^\s*\d{2,4}\s+chapter\s+\d+\b", re.IGNORECASE
)
# Defect 3(c) — a STANDALONE "N.M Title-Case Words" section heading a paragraph
# region carries verbatim (the council mis-typed the section title as prose).
# Captures N (parent chapter ordinal) + M (section index) so the promotion pass
# can require N == the document's dominant chapter number and M a small int.
_STANDALONE_SECTION_RE = re.compile(r"^\s*(\d+)\.(\d+)\s+([A-Z][A-Za-z].*)$")
# Any "N.M Title" section-shaped text (heading OR paragraph), for computing the
# document's dominant chapter ordinal N used as the promotion anchor.
_ANY_SECTION_ORDINAL_RE = re.compile(r"^\s*(\d+)\.(\d+)\s+[A-Za-z]")
# A bare-ordinal chapter-index shape: "N Title" (no "Chapter", no trailing page).
_BARE_ORDINAL_HEADING_RE = re.compile(r"^\s*(\d{1,3})\s+(?=[A-Za-z])\S")
# A real section heading shape: "N.M[.K] Title".
_SECTION_HEADING_RE = re.compile(r"^\s*\d+\.\d+(?:\.\d+)*\s+\S")
# A FUSED-TOC heading shape: a run carrying >=2 "N.M" section-number tokens
# (e.g. "1.1 Introduction to Whole Numbers 1.2 Use the Language of Algebra
# 1.3 …"). A real section heading is a SINGLE numbered title (exactly one
# "N.M"); >=2 "N.M" tokens means several TOC entries were concatenated into one
# region. Anti-FP: ONE "N.M" never matches (the second match is required).
_FUSED_TOC_HEADING_RE = re.compile(r"\b\d+\.\d+\b.*\b\d+\.\d+\b")
# Bullet / list-marker glyphs that prefix a list ITEM the council mis-typed as a
# heading ("◦ Yes–add 1 …", "• …", "‣ …", "▪ …", "– …"/"- …"). A real heading
# never opens with one of these markers.
_LIST_MARKER_CHARS = "◦•‣▪·"
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*[◦•‣▪·]\s+\S")
# A leading dash/hyphen BULLET ("– text" / "- text") — only when followed by a
# space + word (an em/en/hyphen used as a list marker, NOT a hyphenated word
# like "-40" or a minus sign hugging its operand).
_LEADING_DASH_BULLET_RE = re.compile(r"^\s*[–-]\s+\S")
# The repeated-bullet shape ("◦ … ◦ …"): >=2 of the same bullet glyph inside the
# text means the region is a flattened bullet LIST, not a heading.
_REPEATED_BULLET_RE = re.compile(r"[◦•‣▪]\s+\S.*[◦•‣▪]\s+\S")
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

# ---------------------------------------------------------------------------
# Fused-section-title recovery SoT (lane B — SEMANTIK_SPLIT_FUSED_SECTION_TITLES).
#
# When a scan fuses a section's APPARATUS opener into its first body paragraph
# ("[y| Introduction Suppose a stone falls ...") no standalone heading candidate
# survives, so the section vanishes from the heading tree. The Stage-5e
# fused-title SPLIT detector (``block_resegment._detect_fused_title_splits``) and
# the post-5e promotion hook (:func:`promote_fused_title_children`) BOTH key off
# this single anchored-pattern tuple + gutter primitive so the split-boundary
# check and the promotion re-verify can never drift.
# ---------------------------------------------------------------------------

# The canonical apparatus / front-of-section openers (an anchored-pattern SoT
# tuple). Each pattern matches the opener at text START.
_APPARATUS_OPENER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Introduction\b", re.IGNORECASE),
    re.compile(r"Key\s+Terms\b", re.IGNORECASE),
    re.compile(r"Key\s+Concepts\b", re.IGNORECASE),
    re.compile(r"Chapter\s+Review\b", re.IGNORECASE),
    re.compile(r"Review\s+Exercises\b", re.IGNORECASE),
    re.compile(r"Practice\s+Test\b", re.IGNORECASE),
    re.compile(r"Summary\b", re.IGNORECASE),
)

# Word ceiling for a WHOLE-prefix apparatus-opener title (gutter + opener).
_APPARATUS_OPENER_MAX_WORDS = 5

# Gutter-glyph run reusing the defect-3(b) precedent shape (see
# ``_PEDAGOGICAL_LABEL_CLASSES`` EXAMPLE gutter): up to two stray alphanumerics
# around >= 1 non-alphanumeric symbol — OR empty (a bare exact opener). The whole
# run is optional so a clean "Introduction" (no gutter) still matches.
_APPARATUS_GUTTER_RE = re.compile(
    r"^\s*(?:[A-Za-z0-9]{0,2}[^A-Za-z0-9]+[A-Za-z0-9]{0,2}[^A-Za-z0-9]*)?"
)


def _apparatus_opener_match_len(text: str) -> int | None:
    """Char length of a leading ``(optional gutter)(apparatus opener)`` run.

    A PREFIX match — there may be prose AFTER the opener (the intra-FB fusion
    case). Returns the number of characters the gutter + opener span consumes at
    the START of ``text``, or ``None`` when no opener is present. The shared
    prefix primitive behind the whole-match :func:`_matches_apparatus_opener` and
    the fused-title intra-FB diagnostic.
    """
    if not text:
        return None
    gm = _APPARATUS_GUTTER_RE.match(text)
    start = gm.end() if gm else 0
    rest = text[start:]
    for pattern in _APPARATUS_OPENER_PATTERNS:
        m = pattern.match(rest)
        if m:
            return start + m.end()
    return None


def _matches_apparatus_opener(text: str) -> bool:
    """Whether ``text`` is WHOLLY a ``(gutter)(apparatus opener)`` prefix.

    The whole-prefix-match variant (<= ``_APPARATUS_OPENER_MAX_WORDS`` words)
    used by the fused-title SPLIT title-shape guard and the promotion re-verify.
    Any trailing remainder after the opener (prose) makes it NOT a whole match.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    n = _apparatus_opener_match_len(stripped)
    if n is None:
        return False
    if stripped[n:].strip():
        return False  # content after the opener -> not a whole-match title
    return len(stripped.split()) <= _APPARATUS_OPENER_MAX_WORDS


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


def _is_fused_toc_heading(text: str) -> bool:
    """Whether ``text`` is a FUSED run of >=2 "N.M Title" TOC entries.

    A real section heading is a SINGLE numbered title (exactly one "N.M"
    section-number token). When the council fuses several contiguous
    table-of-contents entries into one region ("1.1 Introduction to Whole
    Numbers 1.2 Use the Language of Algebra 1.3 Add and Subtract Integers …"),
    the text carries >=2 "N.M" tokens. Anti-FP: a legitimate "Section 1.1
    Title" single heading has exactly ONE "N.M" and never matches (the regex
    requires a second "N.M" token).
    """
    if not text:
        return False
    return bool(_FUSED_TOC_HEADING_RE.search(text))


def _is_list_item_heading(text: str) -> bool:
    """Whether ``text`` is a bullet / list ITEM the council mis-typed as a heading.

    Fires when the text (1) STARTS with a list-marker glyph (``◦ • ‣ ▪ ·`` or a
    dash/hyphen bullet "– "/"- " followed by a space + word), or (2) carries the
    repeated-bullet "◦ … ◦ …" shape (a flattened bullet LIST). A real heading
    never opens with a bullet marker. Anti-FP: a hyphen hugging its operand
    ("-40", "minus-sign") never matches the dash-bullet rule (it requires a
    SPACE after the dash).
    """
    if not text:
        return False
    if _LEADING_LIST_MARKER_RE.match(text):
        return True
    if _LEADING_DASH_BULLET_RE.match(text):
        return True
    if _REPEATED_BULLET_RE.search(text):
        return True
    return False


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

    # A pedagogical label (EXAMPLE N.M / TRY IT / Solution / HOW TO / Step N …)
    # is NEVER front-matter — it anchors a worked-example unit and must survive
    # to be tagged ``pedagogy-*`` in sub-pass (B) (→ worked_example box). Without
    # this guard a TOC-SHAPED label like "EXAMPLE 1.16" (title + a "1.16" that
    # looks like a trailing page number) is dropped — especially on a mid-chapter
    # SLICE with no chapter anchor, where the front-matter zone spans the whole
    # doc and every EXAMPLE label matches the TOC-drop. A real printed TOC line
    # never matches ``_is_pedagogical_label``.
    drops = {i for i in drops if not _is_pedagogical_label(regions[i])}
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
    # Defect 3(b) — OCR-garbled EXAMPLE label a scan mangles with leading gutter
    # glyphs ("| EXAMPLE9.9 | PLE 9.9", "[y] EXAMPLE 3"). A short leading run of
    # OCR gutter artefacts — REQUIRING at least one non-alphanumeric symbol
    # (pipe / bracket / rule), with up to two stray alphanumeric chars mixed in —
    # then EXAMPLE. Requiring a symbol is the anti-FP guard: a clean multi-word
    # heading ("The Example Below", all letters) does NOT match, but a mangled
    # label the council mis-promoted to <h2> (20× on the EA2e scan) does — and is
    # routed to the paragraph/pedagogy-example path.
    (re.compile(
        r"^\s*[A-Za-z0-9]{0,2}[^A-Za-z0-9]+[A-Za-z0-9]{0,2}[^A-Za-z0-9]*EXAMPLE",
        re.IGNORECASE),
     "pedagogy-example"),
    # Defect 3(a) — a run-in "Solution" label, optionally OCR-decorated with up
    # to 3 leading gutter glyphs (") Solution", "™ Solution", "“ Solution"; 90×
    # across the EA2e scan) so the mangled label demotes to pedagogy-solution
    # instead of surviving as a spurious <h3> section boundary. Anchored so a
    # real "Solution set of …" heading (trailing words) still matches the label
    # prefix — intentional: a "Solution" prefix IS the apparatus label.
    (re.compile(r"^\s*[^A-Za-z0-9]{0,3}\s*Solution\b", re.IGNORECASE),
     "pedagogy-solution"),
    (re.compile(r"^\s*Try\s+It\b", re.IGNORECASE), "pedagogy-try-it"),
    # Defect 3(b) — OCR-garbled TRY IT ("TRYIT::", "| TRY IT"): optional leading
    # gutter glyphs, TRY optionally fused to IT (lost inter-word space).
    (re.compile(r"^\s*[^A-Za-z0-9]{0,4}\s*TRY\s*IT\b", re.IGNORECASE),
     "pedagogy-try-it"),
    (re.compile(r"^\s*How\s+To\b", re.IGNORECASE), "pedagogy-how-to"),
    (re.compile(r"^\s*Step\s+\d+\b", re.IGNORECASE), "pedagogy-step"),
    (re.compile(r"^\s*Learning\s+Objectives\b", re.IGNORECASE),
     "pedagogy-objectives"),
    (re.compile(r"^\s*Be\s+Prepared\b", re.IGNORECASE), "pedagogy-be-prepared"),
    (re.compile(r"^\s*Practice\s+Makes\s+Perfect\b", re.IGNORECASE),
     "pedagogy-practice"),
)


def _is_fused_toc_region(region: Region) -> bool:
    """Whether a HEADING region's text is a fused run of >=2 "N.M" TOC entries.

    A real single numbered section heading ("1.1 Introduction…", with exactly
    one "N.M") is explicitly NOT a fused-TOC region — only >=2 "N.M" tokens
    trip it. Non-heading regions are never fused-TOC.
    """
    if not _is_heading(region):
        return False
    return _is_fused_toc_heading(_region_text(region))


def _is_list_item_region(region: Region) -> bool:
    """Whether a HEADING region's text is a bullet / list ITEM mis-typed as a heading.

    A real numbered chapter/section heading ("Chapter 1", "1.1 Title") never
    opens with a bullet marker, so this can never demote one. Non-heading
    regions are never list-item headings.
    """
    if not _is_heading(region):
        return False
    text = _region_text(region)
    if not text:
        return False
    # Never demote a real numbered chapter/section heading.
    if _CHAPTER_HEADING_RE.match(text) or _SECTION_HEADING_RE.match(text):
        return False
    return _is_list_item_heading(text)


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
# (A'') Running-header text backstop (defect 3a) + (E) section promotion (3c).
# ---------------------------------------------------------------------------


def _is_running_header_region(region: Region) -> bool:
    """Whether a HEADING region's text is a running header with a page number.

    Text-based backstop for the FB-position-based
    ``structure_graph._detect_running_headers`` (which misses headers OCR bbox
    noise pushed out of the margin band). A "Chapter N <words> <3-4 digit page>"
    heading — e.g. "Chapter 9 Roots and Radicals 1039" — is page furniture; a
    real chapter title ("Chapter 9 Roots and Radicals", no trailing number) is
    NOT. Non-heading regions are never running headers here.
    """
    if not _is_heading(region):
        return False
    text = _region_text(region)
    return bool(
        _RUNNING_HEADER_TEXT_RE.match(text)
        or _RUNNING_HEADER_LEADING_PAGE_RE.match(text)
    )


# Defect 3(b) — bare "Chapter N Title" shape (no page number) participating in
# the repeat-count furniture rule. A SINGLE such heading is a real chapter
# opener; a running header recurs on every page and repeats many times.
_BARE_CHAPTER_SHAPE_RE = re.compile(r"^\s*chapter\s+\d+\b", re.IGNORECASE)
_REPEAT_FURNITURE_THRESHOLD = 3


def _detect_repeated_chapter_furniture(regions: list[Region]) -> set[int]:
    """Indices of REPEATED bare "Chapter N Title" heading regions (furniture).

    A normalized bare chapter heading appearing ``> _REPEAT_FURNITURE_THRESHOLD``
    times across the heading candidates is a running header the FB-position
    detector missed (OCR bbox noise). Every occurrence EXCEPT the first
    (the real chapter opener) is dropped. Anti-FP: a real book never repeats an
    identical "Chapter N Title" opener 4+ times; the audit's recurred 34×.
    """
    order: dict[str, list[int]] = {}
    for idx, region in enumerate(regions):
        if not _is_heading(region):
            continue
        text = _region_text(region)
        if not _BARE_CHAPTER_SHAPE_RE.match(text):
            continue
        key = re.sub(r"\s+", " ", text).strip().lower()
        order.setdefault(key, []).append(idx)
    furniture: set[int] = set()
    for idxs in order.values():
        if len(idxs) > _REPEAT_FURNITURE_THRESHOLD:
            furniture.update(idxs[1:])
    return furniture


def _dominant_chapter_ordinal(regions: list[Region]) -> int | None:
    """The most common parent chapter ordinal ``N`` across "N.M Title" texts.

    Scans EVERY region's text (heading or paragraph) for the section-shape
    ``N.M Title``, tallies the ``N`` values, and returns the mode — the
    document's dominant chapter number (9 for an EA2e ch9 scan). Returns None
    when no section-shaped text exists or the winner is ambiguous (a tie), so
    the promotion pass anchors ONLY on an unambiguous single chapter.
    """
    counts: dict[int, int] = {}
    for region in regions:
        m = _ANY_SECTION_ORDINAL_RE.match(_region_text(region))
        if not m:
            continue
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            continue
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return None
    top = max(counts.values())
    winners = [n for n, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else None


def _region_seed_vlm_hint(
    region: Region, feature_blocks: list[FeatureBlock]
) -> dict | None:
    """The VLM hint on a region's SEED (min-index) FeatureBlock, if any.

    P2 audit-only accessor for sub-pass E's corroboration breadcrumb. Returns
    ``None`` when the region owns no in-range FB or the seed carries no hint
    (every region until the VLM source lands, and all regions when the flags
    are off)."""
    idxs = [i for i in (region.feature_block_indices or ()) if 0 <= i < len(feature_blocks)]
    if not idxs:
        return None
    seed = feature_blocks[min(idxs)]
    hint = getattr(seed, "vlm_hint", None)
    return hint if isinstance(hint, dict) else None


def _detect_section_promotions(
    regions: list[Region], excluded: set[int]
) -> set[int]:
    """Indices of PARAGRAPH regions to promote to section headings (defect 3c/4).

    A conservative promotion. TWO standalone-title shapes qualify:

      * a clean ``"N.M Title-Case Words"`` (<=10 words) where ``N`` equals the
        document's dominant chapter ordinal and ``M`` is a small int (1..40) —
        a real section heading the council mis-typed as prose; and
      * a whole-match end-of-chapter apparatus opener (``Key Terms`` /
        ``Chapter Review`` / ``Review Exercises`` / ``Practice Test`` /
        ``Key Concepts`` / ``Introduction`` / ``Summary`` via
        :func:`_matches_apparatus_opener`) — the Defect-4 recovery for an
        apparatus heading mis-typed as a ``<p>`` (e.g. ``PRACTICE TEST``).

    Only STANDALONE regions are promoted (the title is the entire region) — a
    title FUSED into a body paragraph is left to the Stage-5e split. Excludes
    anything already dropped/demoted.
    """
    dominant = _dominant_chapter_ordinal(regions)
    out: set[int] = set()
    for i, region in enumerate(regions):
        if i in excluded or _is_heading(region):
            continue
        text = _region_text(region)
        # Defect 4 — a standalone apparatus opener promotes regardless of the
        # dominant-ordinal anchor (it carries no N.M ordinal).
        if _matches_apparatus_opener(text):
            out.add(i)
            continue
        if dominant is None:
            continue
        m = _STANDALONE_SECTION_RE.match(text)
        if not m:
            continue
        # The match must span the WHOLE region text (standalone title only).
        if m.group(0).strip() != text:
            continue
        try:
            n, mm = int(m.group(1)), int(m.group(2))
        except (TypeError, ValueError):
            continue
        if n != dominant or not (1 <= mm <= 40):
            continue
        if len(text.split()) > 10:
            continue
        out.add(i)
    return out


# ---------------------------------------------------------------------------
# Defect 3 — obviously-FUSED heading demotion (refuse to keep as a heading).
# ---------------------------------------------------------------------------
# The page-top mega-heading the audit found (a running footer + two exercise
# items + a word problem fused into one heading) is NOT a real heading — it
# swallowed body content. Detect it and demote heading -> paragraph, so the
# fused prose renders as prose. Conservative thresholds so a genuine short
# numbered / title-case heading is untouched. Mirrors the
# ``heading_classifier.is_fused_heading`` adapter-seam twin.
_FUSED_HEADING_MIN_PROSE_WORDS = 12
_INLINE_MATH_RUN_RE = re.compile(r"\$[^$]+\$")
_SENTENCE_TERMINAL_RE = re.compile(r"[.!?]")


def _is_fused_prose_heading(region: Region) -> bool:
    """Whether a HEADING region's text is an obviously-fused blob (defect 3).

    Fires on a heading whose text carries EITHER >= 2 inline ``$…$`` math runs
    OR a sentence-length prose run (>= ``_FUSED_HEADING_MIN_PROSE_WORDS`` tokens
    AND a sentence-terminal). A genuine short heading has neither.
    """
    if not _is_heading(region):
        return False
    text = _region_text(region)
    if not text:
        return False
    if len(_INLINE_MATH_RUN_RE.findall(text)) >= 2:
        return True
    return (
        len(text.split()) >= _FUSED_HEADING_MIN_PROSE_WORDS
        and bool(_SENTENCE_TERMINAL_RE.search(text))
    )


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


def _promote_to_heading(region: Region, *, level_hint: int = 2) -> Region:
    """Return a copy of ``region`` promoted paragraph→heading (defect 3c).

    Stamps ``level_hint`` (the assembler's ``normalize_heading_levels`` owns the
    final h-level) and a ``structure_clean`` breadcrumb. Text + the FB partition
    are preserved verbatim (``dataclasses.replace`` of kind + payload only).
    """
    payload = dict(region.payload or {})
    prov = dict(payload.get("structure_clean") or {})
    prov["from_kind"] = str(region.kind)
    prov["promoted"] = "section_heading"
    payload["structure_clean"] = prov
    payload["level_hint"] = int(level_hint)
    return dataclasses.replace(region, kind="heading", payload=payload)


# ---------------------------------------------------------------------------
# Post-Stage-5e fused-title child PROMOTION (lane B).
#
# The clean_structure sub-pass E promotion (``_detect_section_promotions``) runs
# at Stage-5d-det — BEFORE Stage-5e — so it can never see a title child the
# Stage-5e fused-title SPLIT only just carved off. This narrow hook closes that
# gap: it promotes the split-off title child (``child_index==0`` of a
# ``subtype='fused_title'`` split) to a section heading, re-verifying the title
# shape via the SAME SoT (``_STANDALONE_SECTION_RE`` + dominant ordinal, or the
# apparatus-opener tuple) and reusing ``_promote_to_heading`` verbatim (the
# assembler owns the final h-level). Fail-closed: any token-conservation failure
# reverts the WHOLE hook to the input list (never ships a content-losing pass),
# mirroring ``clean_structure``.
# ---------------------------------------------------------------------------


def _is_promotable_fused_title(text: str, dominant: int | None) -> bool:
    """Whether ``text`` is a promotable fused-title child (defense in depth).

    Re-verifies the SAME title shape the Stage-5e detector matched: a
    chapter-consistent standalone ``"N.M Title-Case"`` (N == the document's
    dominant chapter ordinal, M in 1..40, <= 10 words) OR a whole-match
    apparatus opener (``_matches_apparatus_opener``). Anything else is left as a
    paragraph.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    m = _STANDALONE_SECTION_RE.match(stripped)
    if m and m.group(0).strip() == stripped and dominant is not None:
        try:
            n, mm = int(m.group(1)), int(m.group(2))
        except (TypeError, ValueError):
            return False
        if n == dominant and 1 <= mm <= 40 and len(stripped.split()) <= 10:
            return True
    return _matches_apparatus_opener(stripped)


def promote_fused_title_children(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
) -> tuple[list[Region], dict[str, Any]]:
    """Promote Stage-5e fused-title SPLIT title children (``child_index==0``).

    Selects ONLY regions carrying a ``payload['resegment']`` breadcrumb of the
    fused-title split's title child (``op=='split'``, ``subtype=='fused_title'``,
    ``child_index==0``) that is not already a heading and whose text re-verifies
    as a title shape (:func:`_is_promotable_fused_title`), and promotes each via
    :func:`_promote_to_heading`. Returns ``(corrected_regions, diagnostics)``.
    Fail-closed on token-conservation (whole-revert to the input list). Byte-
    identical pass-through when no such child exists (the caller gates on BOTH
    ``SEMANTIK_SPLIT_FUSED_SECTION_TITLES`` and ``SEMANTIK_PROMOTE_SECTION_HEADINGS``).
    """
    diagnostics: dict[str, Any] = {"enabled": True, "promoted": 0}
    if not regions:
        return list(regions), diagnostics

    dominant = _dominant_chapter_ordinal(regions)
    promote_idx: set[int] = set()
    for i, region in enumerate(regions):
        breadcrumb = (region.payload or {}).get("resegment")
        if not isinstance(breadcrumb, dict):
            continue
        if (
            breadcrumb.get("op") != "split"
            or breadcrumb.get("subtype") != "fused_title"
            or breadcrumb.get("child_index") != 0
        ):
            continue
        if _is_heading(region):
            continue
        if _is_promotable_fused_title(_region_text(region), dominant):
            promote_idx.add(i)

    if not promote_idx:
        return list(regions), diagnostics

    corrected = [
        _promote_to_heading(region) if i in promote_idx else region
        for i, region in enumerate(regions)
    ]
    try:
        assert_token_conservation(regions, corrected, feature_blocks)
    except TokenConservationError as exc:  # pragma: no cover - safety net
        diagnostics["reverted_for_invariant"] = True
        diagnostics["revert_reason"] = str(exc)
        return list(regions), diagnostics

    diagnostics["promoted"] = len(promote_idx)
    return corrected, diagnostics


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
        "running_header_dropped": 0,
        "repeated_furniture_dropped": 0,
        "fused_toc_dropped": 0,
        "list_item_demoted": 0,
        "fused_heading_demoted": 0,
        "section_promoted": 0,
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

    # (A'') Running-header text backstop (defect 3a): a "Chapter N <words>
    # <page>" heading that escaped the FB-position running-header detector is
    # page furniture ANYWHERE in the doc -> metadata_drop (never zone-confined;
    # a running header recurs on every page, deep in the body).
    running_header_drops = {
        i
        for i, region in enumerate(regions)
        if i not in fm_drops and _is_running_header_region(region)
    }
    fm_drops |= running_header_drops

    # (A''') Repeat-count running-header furniture (defect 3b): a bare
    # "Chapter N Title" recurring >3× is per-page furniture — drop all but the
    # first (the real chapter opener). Prevents the phantom-chapter run at
    # build_chapters_ir grouping time.
    repeated_furniture = {
        i
        for i in _detect_repeated_chapter_furniture(regions)
        if i not in fm_drops
    }
    fm_drops |= repeated_furniture

    # (A') Fused-TOC headings (>=2 "N.M" entries concatenated into one region)
    # -> metadata_drop. A run of TOC entries is front-matter, not a real
    # heading; mirror the (A) TOC-run drop convention. Anywhere in the doc (a
    # fused TOC run is unambiguous regardless of page zone). A single-"N.M"
    # real section heading is never matched.
    fused_toc_drops = {
        i
        for i, region in enumerate(regions)
        if i not in fm_drops and _is_fused_toc_region(region)
    }
    fm_drops |= fused_toc_drops

    # (B) Pedagogical-label headings -> paragraph (only those NOT already
    # being dropped in (A) — a dropped phantom never needs demoting).
    #
    # RETIRED under SEMANTIK_BERT_AUTHORITATIVE: this sub-pass is a band-aid for
    # the old under-scoped council head — it demotes pedagogical-label headings
    # to <p class="pedagogy-…">. With the authoritative head the
    # pedagogical/role kind is emitted directly (and the AXIS-2 pedagogical_role
    # head stamps payload['semantic_class'] in structure_graph), so the demotion
    # is no longer needed. Flag OFF -> byte-identical (sub-pass B runs).
    if resolve_bert_authoritative():
        peda_demote: set[int] = set()
    else:
        peda_demote = {
            i
            for i, region in enumerate(regions)
            if i not in fm_drops and _is_pedagogical_label(region)
        }

    # (B') Bullet / list-item headings -> paragraph. A list item the council
    # mis-typed as a heading ("◦ Yes–add 1 …") is demoted to a non-heading
    # kind (paragraph). Not front-matter (real body content), so a plain
    # demote rather than a drop; excludes anything already dropped/demoted.
    list_item_demote = {
        i
        for i, region in enumerate(regions)
        if i not in fm_drops
        and i not in peda_demote
        and _is_list_item_region(region)
    }

    # (B'') Fused-prose heading -> paragraph (defect 3). A heading region whose
    # text swallowed body content (sentence-length prose or multiple $…$ runs)
    # is refused as a heading and demoted to prose. Excludes anything already
    # dropped/demoted.
    fused_heading_demote = {
        i
        for i, region in enumerate(regions)
        if i not in fm_drops
        and i not in peda_demote
        and i not in list_item_demote
        and _is_fused_prose_heading(region)
    }

    # (E) Section-heading PROMOTION (defect 3c) — GATED off by default. A
    # standalone "N.M Title" paragraph the council mis-typed as prose is
    # promoted back to a section heading. Excludes anything already
    # dropped/demoted (a dropped region is furniture, not a section title).
    if resolve_promote_section_headings_mode():
        section_promote = _detect_section_promotions(
            regions,
            fm_drops | peda_demote | list_item_demote | fused_heading_demote,
        )
    else:
        section_promote: set[int] = set()

    corrected: list[Region] = []
    pedagogical_class_counts: dict[str, int] = {}
    for i, region in enumerate(regions):
        if i in fm_drops:
            corrected.append(_retag(region, "metadata_drop"))
        elif i in section_promote:
            promoted = _promote_to_heading(region)
            # P2 — audit-only VLM corroboration breadcrumb (NON-AUTHORITATIVE).
            # The promotable SET is the deterministic _STANDALONE_SECTION_RE +
            # dominant-ordinal trigger ALONE (set-equality with hints on/off);
            # a matching whole-block VLM heading hint on the region's seed FB
            # only ADDS a `structure_clean` breadcrumb. It NEVER widens the set
            # (widening = new authority = explicitly deferred).
            if _vlm_struct_hints_enabled():
                hint = _region_seed_vlm_hint(region, feature_blocks)
                if (
                    isinstance(hint, dict)
                    and hint.get("kind") == "heading"
                    and hint.get("coverage") == "whole_block"
                ):
                    prov = dict(promoted.payload.get("structure_clean") or {})
                    prov["vlm_corroborated"] = True
                    promoted.payload["structure_clean"] = prov
            corrected.append(promoted)
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
        elif i in list_item_demote:
            # A bullet/list-item heading -> paragraph (a list/non-heading kind);
            # text preserved verbatim, no pedagogical class hint.
            corrected.append(_retag(region, "paragraph"))
        elif i in fused_heading_demote:
            # Defect 3 — an obviously-fused heading -> paragraph (prose);
            # text preserved verbatim.
            corrected.append(_retag(region, "paragraph"))
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
    diagnostics["running_header_dropped"] = len(running_header_drops)
    diagnostics["repeated_furniture_dropped"] = len(repeated_furniture)
    diagnostics["fused_toc_dropped"] = len(fused_toc_drops)
    diagnostics["list_item_demoted"] = len(list_item_demote)
    diagnostics["fused_heading_demoted"] = len(fused_heading_demote)
    diagnostics["section_promoted"] = len(section_promote)
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
    "promote_fused_title_children",
    "resolve_structure_clean_mode",
    "resolve_promote_section_headings_mode",
    "_pedagogical_class_for",
    "_is_fused_toc_heading",
    "_is_list_item_heading",
    "_matches_apparatus_opener",
]
