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

# A TOC run must be at least this many contiguous entries — one or two
# title+pagenum lines could be a coincidental real heading; four in a row with
# increasing page numbers is unambiguously a printed table of contents.
_MIN_TOC_RUN = 4

# A real chapter heading shape: "Chapter N[: Title]" or "Chapter N - Title"
# or "Chapter N Title". The number is the canonical chapter ordinal.
_CHAPTER_HEADING_RE = re.compile(
    r"^\s*chapter\s+(\d+)\b", re.IGNORECASE
)

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
) -> Tuple[List[Mapping[str, Any]], int]:
    """Drop the phantom-TOC run + leading front-matter boilerplate.

    Parameters
    ----------
    provenance
        The ordered ``region_provenance`` list (document order).

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
      * a run of plain titles WITHOUT trailing increasing page numbers.
    """
    if not _flag_enabled():
        return list(provenance), 0
    if not provenance:
        return list(provenance), 0

    zone_end = _front_matter_zone_end(provenance)

    # (2) the contiguous, page-increasing TOC run inside the front-matter zone.
    drop_idx = _find_toc_run(provenance, zone_end)

    # (3) front-matter boilerplate inside the zone (Preface / Copyright /
    # authors / bare TOC header / donor lists). Reuses tested predicates.
    for idx in range(zone_end):
        if _is_front_matter_boilerplate(provenance[idx]):
            drop_idx.add(idx)

    if not drop_idx:
        return list(provenance), 0

    filtered = [
        prov for idx, prov in enumerate(provenance) if idx not in drop_idx
    ]
    return filtered, len(drop_idx)


__all__ = ["drop_toc_and_frontmatter"]
