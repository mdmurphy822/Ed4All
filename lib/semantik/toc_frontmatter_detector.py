"""Conservative, deterministic phantom-TOC + front-matter detector.

Root-cause fix for the **phantom-TOC defect**: a PDF's full-book Table of
Contents, living in the front matter, gets classified by the SemantiK
conversion as a run of real chapter headings — e.g. an extract covering only
chapters 1-3 sprouts phantom "Chapter 5: Systems of Linear Equations" /
"Chapter 6: Polynomials" headings because the book's whole TOC was printed in
the front matter. Those phantoms become fabricated IR chapters with no
content behind them.

This module supplies a single entry point, :func:`drop_toc_and_frontmatter`,
that takes the ordered ``region_provenance`` list (the per-region dicts the
cascade surfaces; see ``lib/semantik/cascade_ir.py`` for the exact shape) and
returns a filtered list with the phantom-TOC run + leading front-matter
boilerplate removed. It is applied INSIDE ``build_chapters_ir`` /
``from_bridge_json`` BEFORE chapters are assembled, so it operates on the
adapter-wrapped output the Ed4All pipeline consumes WITHOUT touching any
SemantiK-internal code (no Semantic-repo divergence).

Design — three deterministic, CPU-only signals (no models, no GPU):

1. **front-matter zone anchor** — the first region that is a *real* chapter or
   section heading (``Chapter N: Title`` / ``N.M Title`` / a content-bearing
   plain title) AND is NOT itself part of a TOC run. Everything strictly
   before that anchor is the eligible front-matter zone; everything at/after
   it is PROTECTED content (never dropped).

2. **TOC run** — a CONTIGUOUS run (``>= _MIN_TOC_RUN``) of regions inside the
   front-matter zone whose text matches a TOC-entry shape (``<title>
   <trailing page-number>``, including ``Chapter N Title 301``, ``N.M Title
   301``, and dot-leader ``Title ...... 301``) whose trailing page numbers are
   GENERALLY MONOTONICALLY INCREASING across the run. The
   increasing-page-number + contiguous-run requirement is THE discriminator:
   real numbered headings ("Chapter 1: Foundations", "1.1 Introduction to
   Whole Numbers") do not form a run of title+increasing-pagenum lines.

3. **front-matter boilerplate** — regions in the front-matter zone matching
   known front-matter (Preface / Copyright / "SENIOR CONTRIBUTING AUTHORS" /
   donor-foundation lists / a bare "Table of Contents" header). REUSES the
   tested predicates from ``lib/semantic_structure_extractor`` rather than
   reinventing them: :func:`_is_noncontent_heading`, :func:`_is_toc_heading`,
   and the ``_FRONT_MATTER_HEADING_TEXTS`` set.

Conservative by construction — when in doubt, KEEP. A single real "Chapter 5"
with content (not in a run), a list of plain section titles WITHOUT page
numbers, and anything after the first real chapter are NEVER dropped.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Mapping, Tuple

# Reuse the tested front-matter / non-content predicates rather than
# reinventing them (the task's explicit "do NOT reinvent if a tested
# predicate exists" requirement).
from lib.semantic_structure_extractor.semantic_structure_extractor import (
    _FRONT_MATTER_HEADING_TEXTS,
    _is_noncontent_heading,
    _is_toc_heading,
)

# Env gate. Default ON inside the IR builder (the IR builder only runs in the
# SemantiK conversion path); set falsey to disable for debugging.
_FLAG_ENV = "SEMANTIK_DROP_FRONTMATTER_TOC"

# Env gate for the PAGE-DENSITY front-matter-zone pass (the robust positional
# detector below). Default ON, parse-with-fallback, mirroring
# ``SEMANTIK_DROP_FRONTMATTER_TOC``. Set falsey (``0`` / ``false`` / ``no`` /
# ``off``) for byte-identical pass-through of THIS pass only (the TOC-run +
# chapter-index-cluster + boilerplate passes still run under their own gate).
_FLAG_ZONE_ENV = "SEMANTIK_DROP_FRONTMATTER_ZONE"

# A TOC run must be at least this many contiguous entries — one or two
# title+pagenum lines could be a coincidental real heading; four in a row with
# increasing page numbers is unambiguously a printed table of contents.
_MIN_TOC_RUN = 4

# A chapter-INDEX cluster must be at least this many back-to-back
# chapter-pattern headings (with only headings/metadata between consecutive
# entries) to be dropped. Real chapter openers are separated by substantial
# content (paragraphs / sections), so they never form such a back-to-back run;
# a rendered chapter index (a page-number-less TOC variant) does. Three in a
# row with no content between them is unambiguously an index cluster.
_MIN_CHAPTER_INDEX_RUN = 3

# --- PAGE-DENSITY front-matter-zone discriminator (the robust positional fix) -
#
# The chapter-index-cluster heuristic above keys on "NO content-bearing region
# between consecutive entries". A PREFACE chapter-by-chapter SUMMARY defeats it:
# each "Chapter N: Title" is followed by a real summary paragraph, so content
# DOES sit between consecutive entries and the run never forms. The reliable
# signal that survives content-between is PAGE DENSITY: a preface summary /
# chapter index packs MANY chapter-level headings into a TINY page span (the
# real EA2e defect — 8 "Chapter N" headings on pages 9-10), whereas the REAL
# body is page-SPARSE (Chapter 1's body spans many pages before Chapter 2
# appears). So a run of >= _ZONE_MIN_CHAPTER_CLUSTER chapter-pattern headings
# whose page span is <= _ZONE_MAX_PAGE_WINDOW pages, located in the early-page
# front-matter zone (before the body-start page), is a front-matter cluster and
# is dropped WHOLE — regardless of content between entries or trailing page
# numbers.
#
# K — minimum chapter-pattern headings in the dense cluster.
_ZONE_MIN_CHAPTER_CLUSTER = 4
# P — maximum page span (last_page - first_page) the cluster may straddle. A
# cluster of K chapter headings within P pages is impossibly dense for real
# body chapters (a real chapter spans many pages of content before the next
# opener).
_ZONE_MAX_PAGE_WINDOW = 3
# The front-matter zone is the maximal early-page prefix BEFORE the body-start
# page. A dense chapter cluster is only dropped when it lies within this many
# pages of the document start — guarding the real body (whose first chapter
# opener sits on a much later page) from ever being mistaken for a front-matter
# cluster. Generous (front matter — copyright / authors / preface / TOC — can
# run a dozen+ pages) but bounded so a dense run deep in the body is never
# swept up.
_ZONE_MAX_FRONTMATTER_PAGE = 16

# A real chapter heading shape: "Chapter N[: Title]" or "Chapter N - Title"
# or "Chapter N Title". The number is the canonical chapter ordinal.
_CHAPTER_HEADING_RE = re.compile(
    r"^\s*chapter\s+(\d+)\b", re.IGNORECASE
)

# A bare-ordinal chapter-index shape: "N Title" (e.g. "3 Math Models") — a
# rendered chapter-index entry that names the chapter by leading ordinal +
# title WITHOUT the word "Chapter" and WITHOUT a trailing page number. Requires
# a leading integer then an alphabetic title word. Deliberately does NOT match
# a "N.M Title" section heading (the dot disqualifies it) nor a bare numeric
# row (the title must carry a letter).
_BARE_ORDINAL_HEADING_RE = re.compile(r"^\s*(\d{1,3})\s+(?=[A-Za-z])\S")

# A real section heading shape: "N.M[.K] Title" (e.g. "1.1 Introduction to
# Whole Numbers"). Must have a leading decimal-numbered prefix.
_SECTION_HEADING_RE = re.compile(r"^\s*\d+\.\d+(?:\.\d+)*\s+\S")

# Supplementary front-matter phrases NOT covered by the reused extractor
# predicate ``_is_noncontent_heading`` (whose ``_FRONT_MATTER_HEADING_TEXTS``
# set omits these). Conservative: applied ONLY to regions in the front-matter
# zone (before the first real chapter), so a real chapter/section can never be
# caught even if its title coincidentally contains one of these phrases.
# Substring (lowercased) match — a "SENIOR CONTRIBUTING AUTHORS" / "Contributing
# Authors" header reliably names this front-matter contributor block.
_FRONT_MATTER_PHRASES = (
    "contributing authors",
    "contributing author",
    "list of contributors",
)

# A TOC-entry shape: a title followed by a TRAILING page number, optionally
# separated by dot leaders or whitespace. The trailing integer (>= 1) is the
# page number we test for monotonicity. Examples:
#   "Chapter 3 Math Models 301"
#   "1.1 Introduction to Whole Numbers .......... 5"
#   "Systems of Linear Equations    577"
# We require >= 1 alphabetic title word before the trailing number so a bare
# numeric answer-key row ("78 41. 900 42.") does NOT match as a TOC entry.
_TOC_ENTRY_RE = re.compile(
    r"^\s*(?P<title>.*?[A-Za-z].*?)"      # title with >=1 letter
    r"[\s\.…]+"                       # leader: spaces / dots / ellipsis
    r"(?P<page>\d{1,5})\s*$"               # trailing page number
)


def _truthy(val: str) -> bool:
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _flag_enabled() -> bool:
    """Parse-with-fallback, DEFAULT ON. Only an explicit falsey value
    (``0`` / ``false`` / ``no`` / ``off``) disables the detector; anything
    else (unset, truthy, garbage) keeps it on."""
    raw = os.environ.get(_FLAG_ENV)
    if raw is None:
        return True
    stripped = raw.strip().lower()
    if stripped in {"0", "false", "no", "off"}:
        return False
    return True


def _heading_text_of(prov: Mapping[str, Any]) -> str:
    """The best text signal for a region — heading_text else raw_text."""
    text = prov.get("heading_text")
    if not text:
        text = prov.get("raw_text")
    return str(text or "").strip()


def _is_real_chapter_or_section(prov: Mapping[str, Any]) -> bool:
    """Whether a region is a genuine content-bearing chapter/section heading
    that can anchor "first real chapter".

    Conservative: anchors ONLY on an explicit NUMBERED heading
    (``Chapter N[: Title]`` or ``N.M Title``) that is NOT non-content noise
    and NOT itself a TOC entry (a trailing page number → a printed TOC line,
    never a real anchor). A plain (un-numbered) title is DELIBERATELY not an
    anchor — front-matter headers ("SENIOR CONTRIBUTING AUTHORS", a section
    title printed in the preface) are indistinguishable from a real plain
    section opener, and anchoring on one would shrink the front-matter zone to
    nothing and let the phantom-TOC run through. The numbered-heading anchor
    is reliable: the first real chapter of a textbook IS numbered.
    """
    if str(prov.get("region_kind")) != "heading":
        return False
    text = _heading_text_of(prov)
    if not text or _is_noncontent_heading(text):
        return False
    # A TOC line ("Chapter 5 Systems 577") carries a trailing page number and
    # is NEVER an anchor, even though its prefix matches the chapter regex.
    if _toc_entry_page(text) is not None:
        return False
    # An explicit "Chapter N" / "N.M" numbered heading is the real-content
    # anchor (the only reliable signal in the front-matter zone).
    return bool(_CHAPTER_HEADING_RE.match(text) or _SECTION_HEADING_RE.match(text))


def _toc_entry_page(text: str) -> int | None:
    """The trailing page number if the text is a TOC-entry shape, else None.

    Conservative: requires a title with at least one alphabetic character
    BEFORE the trailing number, so a bare numeric answer row does not match.
    A region with NO trailing number is never a TOC entry (returns None) —
    this is the "absent increasing-page-number signal" guard.
    """
    if not text:
        return None
    m = _TOC_ENTRY_RE.match(text)
    if not m:
        return None
    title = m.group("title").strip()
    # The title must carry at least one alphabetic word (not just punctuation
    # or stray digits) so "78 41. 900 42." does not register as a TOC entry.
    if not re.search(r"[A-Za-z]", title):
        return None
    try:
        return int(m.group("page"))
    except (TypeError, ValueError):
        return None


def _front_matter_zone_end(provenance: List[Mapping[str, Any]]) -> int:
    """Index of the first REAL chapter/section heading (the front-matter zone
    anchor). Everything strictly before it is eligible; at/after is protected.

    A region that is a TOC entry (title + trailing page number) does NOT
    anchor even if it superficially looks like "Chapter N Title", because a
    printed TOC line carries a trailing page number. The anchor must be a
    real heading with NO trailing page number (i.e. followed by content, not
    by more pagenum entries). Returns ``len(provenance)`` when no real chapter
    is found (degenerate: whole doc is front matter — we then keep all of it
    rather than dropping everything, the fail-safe).
    """
    for idx, prov in enumerate(provenance):
        text = _heading_text_of(prov)
        # A TOC line ("Chapter 5 Systems 577") is NOT a real anchor.
        if _toc_entry_page(text) is not None:
            continue
        if _is_real_chapter_or_section(prov):
            return idx
    return len(provenance)


def _find_toc_run(
    provenance: List[Mapping[str, Any]], zone_end: int
) -> set[int]:
    """Indices belonging to a contiguous TOC run inside the front-matter zone.

    A TOC run is a maximal contiguous block (within ``[0, zone_end)``) of
    regions that are TOC entries (title + trailing page number) whose page
    numbers are GENERALLY MONOTONICALLY INCREASING. "Generally" tolerates a
    small number of non-decreasing steps (a multi-line title wraps, or a
    page repeats) but requires the run to TREND upward and span the minimum
    length. Only runs of ``>= _MIN_TOC_RUN`` qualify.
    """
    dropped: set[int] = set()
    i = 0
    while i < zone_end:
        page = _toc_entry_page(_heading_text_of(provenance[i]))
        if page is None:
            i += 1
            continue
        # Extend a contiguous run of TOC-entry regions.
        run: List[Tuple[int, int]] = [(i, page)]
        j = i + 1
        while j < zone_end:
            pg = _toc_entry_page(_heading_text_of(provenance[j]))
            if pg is None:
                break
            run.append((j, pg))
            j += 1
        if len(run) >= _MIN_TOC_RUN and _is_increasing(
            [pg for _, pg in run]
        ):
            dropped.update(idx for idx, _ in run)
        i = j if j > i else i + 1
    return dropped


def _is_increasing(pages: List[int]) -> bool:
    """Whether a page-number sequence is GENERALLY monotonically increasing.

    Strict end-to-end monotonicity is too brittle (a wrapped title line or a
    repeated page number breaks it), so we require: (a) the last page strictly
    exceeds the first, AND (b) at least ``ceil(0.75 * (n-1))`` of the adjacent
    steps are non-decreasing. This trends-upward test is the key discriminator
    from a coincidental run of plain headings (which carry no trailing pages
    at all, so they never reach here).
    """
    if len(pages) < 2:
        return False
    if pages[-1] <= pages[0]:
        return False
    steps = len(pages) - 1
    non_decreasing = sum(
        1 for a, b in zip(pages, pages[1:]) if b >= a
    )
    needed = -(-3 * steps // 4)  # ceil(0.75 * steps)
    return non_decreasing >= needed


def _chapter_index_ordinal(prov: Mapping[str, Any]) -> int | None:
    """The chapter ordinal if a region is a chapter-INDEX-pattern heading.

    Matches a heading whose text is ``Chapter N[: Title]`` or a bare-ordinal
    ``N Title`` (rendered chapter-index entry) and carries NO trailing page
    number (a trailing page number → a real TOC line, handled by the
    increasing-page-number TOC-run path, not this one). Returns the leading
    ordinal ``N`` for cluster grouping, else ``None``.

    Conservative: a ``N.M Title`` section heading is NOT a chapter-index entry
    (the ``_BARE_ORDINAL_HEADING_RE`` lookahead requires a non-dot first token
    boundary; a section's dot keeps it out). A plain un-numbered title is also
    not matched (no leading ordinal).
    """
    if str(prov.get("region_kind")) != "heading":
        return None
    text = _heading_text_of(prov)
    if not text:
        return None
    # A trailing-page-number TOC line is handled by the TOC-run path, not here.
    if _toc_entry_page(text) is not None:
        return None
    m = _CHAPTER_HEADING_RE.match(text)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None
    # Bare-ordinal "N Title" — but never a "N.M Title" section heading.
    if _SECTION_HEADING_RE.match(text):
        return None
    m2 = _BARE_ORDINAL_HEADING_RE.match(text)
    if m2:
        try:
            return int(m2.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _is_content_bearing(prov: Mapping[str, Any]) -> bool:
    """Whether a region carries real teaching content (a paragraph / list /
    table / code / blockquote — anything that is NOT a heading or a dropped
    metadata region).

    This is the discriminator between a rendered chapter INDEX (chapter-pattern
    headings back-to-back, separated only by other headings / metadata) and
    real chapter OPENERS (each followed by substantial content). A chapter
    index has NO content-bearing region between consecutive entries; real
    openers do.
    """
    kind = str(prov.get("region_kind") or "")
    if kind in {"heading", "metadata_drop", ""}:
        return False
    return True


def _find_chapter_index_clusters(
    provenance: List[Mapping[str, Any]],
) -> set[int]:
    """Indices belonging to a chapter-INDEX cluster (a page-number-less TOC
    variant) ANYWHERE in the document.

    A chapter-index cluster is a run of ``>= _MIN_CHAPTER_INDEX_RUN``
    chapter-pattern headings (``Chapter N[: Title]`` / bare-ordinal ``N
    Title``) with only OTHER headings / metadata between consecutive entries —
    i.e. NO content-bearing region (paragraph / list / table) separates two
    consecutive chapter-pattern headings in the run. Real chapter openers are
    always separated by substantial content, so they never form such a run;
    the rendered chapter index does.

    Unlike the front-matter TOC-run path, this is NOT zoned to the front matter
    and NOT gated on trailing page numbers — the real EA2e defect is a cluster
    of ``Chapter 1: …``, ``Chapter 2: …`` … back-to-back AFTER the first real
    chapter anchor (so the front-matter zone never reaches it).

    Conservative: only chapter-pattern headings entered the run; any
    content-bearing region between two candidates BREAKS the run (so a real
    opener followed by its content is never swept up). Page-number-bearing TOC
    lines never enter (``_chapter_index_ordinal`` returns ``None`` for them).
    """
    dropped: set[int] = set()
    n = len(provenance)
    i = 0
    while i < n:
        if _chapter_index_ordinal(provenance[i]) is None:
            i += 1
            continue
        # Begin a candidate cluster at i. Walk forward collecting
        # chapter-pattern headings; skip intervening NON-content headings /
        # metadata; STOP at the first content-bearing region.
        run = [i]
        j = i + 1
        while j < n:
            if _is_content_bearing(provenance[j]):
                break  # real content → not an index; close the run here.
            if _chapter_index_ordinal(provenance[j]) is not None:
                run.append(j)
                j += 1
                continue
            # A non-content, non-chapter heading (a stray running header like
            # "2 Preface") inside the cluster — skip it, keep the run open.
            j += 1
        if len(run) >= _MIN_CHAPTER_INDEX_RUN:
            # Conservative final-entry guard: the last chapter-pattern heading
            # of the run is a REAL opener (kept) ONLY when content sits
            # DIRECTLY behind it (the very next region is content-bearing) — a
            # real "Chapter 1: Foundations\n<intro paragraph>". When the next
            # region is ANOTHER heading (a section like "1.1 …" or a preface
            # header), the entry is an index line whose "content" actually
            # belongs to a following section, so it is dropped with the rest.
            last = run[-1]
            keep_last = (
                last + 1 < n and _is_content_bearing(provenance[last + 1])
            )
            to_drop = run[:-1] if keep_last else list(run)
            dropped.update(to_drop)
        # Resume scanning after the consumed span (j is the first
        # content-bearing region or end-of-run).
        i = max(j, i + 1)
    return dropped


def _zone_flag_enabled() -> bool:
    """Parse-with-fallback, DEFAULT ON, for the page-density front-matter-zone
    pass. Mirrors :func:`_flag_enabled` but keyed on ``_FLAG_ZONE_ENV`` so the
    positional pass can be disabled independently of the TOC-run pass."""
    raw = os.environ.get(_FLAG_ZONE_ENV)
    if raw is None:
        return True
    stripped = raw.strip().lower()
    if stripped in {"0", "false", "no", "off"}:
        return False
    return True


def _first_page(prov: Mapping[str, Any]) -> int | None:
    """The smallest positive page number a region appears on, else ``None``.

    ``pages`` is the per-region page list the cascade surfaces (see
    ``cascade_ir._block_from_provenance``). A region with no positive page is
    page-unknown and never participates in the page-density rule.
    """
    pages = prov.get("pages")
    if not isinstance(pages, (list, tuple)):
        return None
    candidates = [int(p) for p in pages if isinstance(p, int) and p > 0]
    return min(candidates) if candidates else None


def _find_frontmatter_zone_clusters(
    provenance: List[Mapping[str, Any]],
) -> set[int]:
    """Indices of dense chapter-heading clusters in the early-page front-matter
    zone (the robust page-density discriminator).

    A cluster is a maximal contiguous run (in document order, skipping
    intervening non-chapter regions) of ``>= _ZONE_MIN_CHAPTER_CLUSTER``
    chapter-pattern headings (``Chapter N[: Title]`` / bare-ordinal ``N Title``)
    whose page span ``last_page - first_page <= _ZONE_MAX_PAGE_WINDOW`` AND
    whose first heading sits on a page ``<= _ZONE_MAX_FRONTMATTER_PAGE`` (the
    early front-matter zone). Such a run is a preface chapter-summary / chapter
    index — many chapter-level headings packed into a tiny page span in the
    front matter — and is dropped WHOLE.

    Crucially this fires WHETHER OR NOT content sits between consecutive
    entries (a preface summary has a paragraph after each "Chapter N") and
    WHETHER OR NOT they carry trailing page numbers — the two cases the
    existing TOC-run / index-cluster passes miss.

    Anti-false-positive (the core contract):
      * page-span guard — a real body chapter span is MANY pages
        (``Chapter 1`` body runs long before ``Chapter 2`` appears), so K real
        openers never fit inside ``_ZONE_MAX_PAGE_WINDOW`` pages;
      * front-matter-zone guard — the first chapter of the run must sit on an
        early page (``<= _ZONE_MAX_FRONTMATTER_PAGE``), so a dense run DEEP in
        the body is never swept up;
      * page-unknown regions never enter a cluster (no page → can't be dense).

    Only the chapter-pattern HEADINGS of the cluster are returned for drop;
    intervening content (a preface summary paragraph) is NOT dropped here — the
    zone-boilerplate pass and the re-anchored zone handle residue. Removing the
    phantom HEADINGS is what stops them opening chapters, which is the defect.
    """
    dropped: set[int] = set()
    n = len(provenance)
    i = 0
    while i < n:
        ordinal = _chapter_index_ordinal(provenance[i])
        first_pg = _first_page(provenance[i]) if ordinal is not None else None
        if ordinal is None or first_pg is None:
            i += 1
            continue
        # Begin a candidate cluster: collect contiguous chapter-pattern headings
        # (skipping intervening non-chapter regions — paragraphs / other
        # headings / metadata) as long as each chapter heading stays inside the
        # page window measured from the cluster's first page.
        run: List[int] = [i]
        run_first_page = first_pg
        run_last_page = first_pg
        j = i + 1
        last_chapter_pos = i
        while j < n:
            o2 = _chapter_index_ordinal(provenance[j])
            if o2 is None:
                # A non-chapter region (content / other heading) — keep scanning;
                # the page-density rule tolerates content between entries.
                j += 1
                continue
            pg2 = _first_page(provenance[j])
            if pg2 is None:
                j += 1
                continue
            # A chapter heading that breaks out of the page window CLOSES the
            # cluster — this is the body-start: a chapter opener many pages
            # after the cluster is the real body, not part of the dense run.
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
            dropped.update(run)
        # Resume after the last chapter heading we considered for this cluster
        # (do not re-scan its members; the body-start chapter that broke the
        # window is reconsidered as a fresh cluster start, and being page-sparse
        # it will never form a dense run).
        i = max(last_chapter_pos + 1, i + 1)
    return dropped


def _is_front_matter_boilerplate(prov: Mapping[str, Any]) -> bool:
    """Whether a region in the front-matter zone is droppable boilerplate.

    REUSES the extractor's tested predicates: a non-content heading
    (``_is_noncontent_heading`` — Preface / Copyright / donor-foundation
    lists / numeric answer rows) OR a bare TOC header (``_is_toc_heading`` —
    "Table of Contents" / "Contents" / "Index"). A front-matter exact match
    is already covered by ``_is_noncontent_heading``'s
    ``_FRONT_MATTER_HEADING_TEXTS`` arm; we reference the set name in the
    docstring for traceability.
    """
    text = _heading_text_of(prov)
    if not text:
        return False
    if _is_toc_heading(text):
        return True
    if _is_noncontent_heading(text):
        return True
    lowered = text.lower()
    if any(phrase in lowered for phrase in _FRONT_MATTER_PHRASES):
        return True
    return False


def drop_toc_and_frontmatter(
    provenance: List[Mapping[str, Any]],
    diagnostics: dict | None = None,
) -> Tuple[List[Mapping[str, Any]], int]:
    """Drop the phantom-TOC run + leading front-matter boilerplate.

    Parameters
    ----------
    provenance
        The ordered ``region_provenance`` list (document order).
    diagnostics
        Optional out-param dict. When provided it is populated with per-pass
        drop counts for audit (mirroring the resegment / structure
        diagnostics): ``frontmatter_zone_dropped`` (the page-density cluster
        count) and ``total_dropped``. Left untouched when the detector is
        disabled or no drops occur (the keys are always set to ``0`` when the
        function runs, so a caller can stamp the diagnostic unconditionally).

    Returns
    -------
    (filtered, dropped_count)
        ``filtered`` is the provenance list with the phantom-TOC run + the
        leading front-matter boilerplate removed; ``dropped_count`` is the
        number of regions removed (surfaced for observability). When the env
        flag is disabled, returns the input UNCHANGED with ``dropped_count=0``
        (byte-identical, no drops).

    Conservative contract — NEVER drops:
      * anything at/after the first real chapter/section heading,
      * a single real chapter heading that is not part of a TOC run,
      * a run of plain titles WITHOUT trailing increasing page numbers,
      * a page-SPARSE real body chapter cluster (the page-density guard).
    """
    if diagnostics is not None:
        diagnostics.setdefault("frontmatter_zone_dropped", 0)
        diagnostics.setdefault("total_dropped", 0)
    if not _flag_enabled():
        return list(provenance), 0
    if not provenance:
        return list(provenance), 0

    zone_end = _front_matter_zone_end(provenance)

    # (2) the contiguous, page-increasing TOC run inside the front-matter zone.
    drop_idx = _find_toc_run(provenance, zone_end)

    # (2b) chapter-INDEX clusters ANYWHERE in the document — a page-number-less
    # TOC variant (a back-to-back run of "Chapter N[: Title]" / bare-ordinal
    # "N Title" headings with NO content-bearing region between consecutive
    # entries). NOT zoned to the front matter and NOT gated on trailing page
    # numbers: the real EA2e defect is a cluster of "Chapter 1: …" … "Chapter
    # 10: …" rendered AFTER the first real chapter anchor, so the front-matter
    # zone never reaches it. Real chapter openers (each followed by content)
    # never form such a run.
    drop_idx.update(_find_chapter_index_clusters(provenance))

    # (2c) PAGE-DENSITY front-matter-zone clusters — the robust positional fix.
    # A preface chapter-by-chapter SUMMARY (each "Chapter N: Title" followed by
    # a real summary paragraph) packs MANY chapter-level headings into a TINY
    # page span in the front matter; the index-cluster pass (2b) misses it
    # because content sits BETWEEN consecutive entries. The page-density rule
    # catches it: >= _ZONE_MIN_CHAPTER_CLUSTER chapter headings inside a
    # <= _ZONE_MAX_PAGE_WINDOW page span on an early front-matter page is a
    # cluster → drop the phantom HEADINGS (so they never open chapters). Real
    # body chapters are page-SPARSE (a chapter spans many pages before the next
    # opener), so they can never satisfy the density window. Independently gated
    # by SEMANTIK_DROP_FRONTMATTER_ZONE (default ON).
    if _zone_flag_enabled():
        zone_dropped = _find_frontmatter_zone_clusters(provenance)
    else:
        zone_dropped = set()
    drop_idx.update(zone_dropped)
    if diagnostics is not None:
        diagnostics["frontmatter_zone_dropped"] = len(zone_dropped)

    # (3) front-matter boilerplate inside the zone (Preface / Copyright /
    # authors / bare TOC header / donor lists). Reuses tested predicates.
    for idx in range(zone_end):
        if _is_front_matter_boilerplate(provenance[idx]):
            drop_idx.add(idx)

    if not drop_idx:
        return list(provenance), 0

    if diagnostics is not None:
        diagnostics["total_dropped"] = len(drop_idx)

    filtered = [
        prov for idx, prov in enumerate(provenance) if idx not in drop_idx
    ]
    return filtered, len(drop_idx)


__all__ = ["drop_toc_and_frontmatter"]
