"""Defensive heading-sanity filter for the Ed4All chunk path.

A chunk's ``source.section_heading`` is relayed faithfully by the chunker
from whatever ``<h*>`` the upstream SemantiK heading classifier
emitted. When the upstream classifier mis-tags answer-key rows, exercise
banners, or full-sentence prose as headings, that noise becomes the chunk's
display heading — polluting retrieval display + starving definition queries.

This module is a CONSERVATIVE, defensive hardening pass (SemantiK retraining
is the upstream root fix). It is gated behind ``TRAINFORGE_HEADING_SANITY_FILTER``
(default OFF for byte-stable back-compat) and wired into both chunk-emit sites
(``Trainforge/process_course.py::CourseProcessor._create_chunk`` and the inline
``_create_chunk`` in ``MCP/tools/pipeline_tools.py::_run_semantik_chunking``).

Design contract (NEVER demote a real heading):

* **Reuse, don't reinvent.** Noise detection is built on the canonical
  ``lib/semantic_structure_extractor/semantic_structure_extractor.py::
  _is_noncontent_heading`` predicate — the single noise-heading detector with
  the strongest precedent in the repo (consumed by the SemantiK adapter +
  cascade_ir + the extractor's own heading-hierarchy fallback, with a
  comprehensive positive/negative regression suite at
  ``lib/tests/test_semantic_structure_extractor_noncontent_filter.py``). It
  rejects answer-key numeric runs, circled-answer markers (ⓐⓑⓒⓓⓔ),
  front-matter, and donor/foundation lists.

* **Plus two NARROW additive clauses** for noise families that detector does
  NOT catch but the task names explicitly (calibrated against a real full-book
  corpus): (1) a publisher exercise-section banner
  / ≥2 embedded ``\\d{2,3}.`` exercise-answer markers, and (2)
  full-sentence imperative/declarative prose (>15 words ending in a period).
  Both have generous floors so a real title containing a number or a short
  phrase survives.

* **Repair, don't drop.** When a heading is suspect, REPLACE it with the
  nearest CLEAN ancestor heading derived from the chunk's heading hierarchy
  (``merged_headings`` breadcrumb first, then the page ``title`` /
  ``module_title``) — each candidate itself re-checked so we never repair
  noise → noise. The original noise text stays in the chunk body (no content
  loss). If no clean ancestor exists, the heading is left AS-IS (never blanked)
  and a ``heading_suspect`` provenance flag is stamped.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

# Reuse the canonical noise-heading detector (strongest precedent + tests).
from lib.semantic_structure_extractor.semantic_structure_extractor import (
    _is_noncontent_heading,
)

__all__ = [
    "FLAG_ENV",
    "is_heading_sanity_filter_enabled",
    "is_suspect_section_heading",
    "repair_section_heading",
]

FLAG_ENV = "TRAINFORGE_HEADING_SANITY_FILTER"

# ``(part N)`` suffix the chunker appends when a long section splits across
# chunks. Stripped before noise-checking so the suffix never perturbs the
# word-count / trailing-period heuristics; the surviving stem is what we test.
_PART_SUFFIX_RE = re.compile(r"\s*\(part\s+\d+\)\s*$", re.IGNORECASE)

# Unambiguous publisher exercise-section banner. These are exercise prose the
# font-size promoter lifts into a heading; never a real section title.
# Consolidated into lib/objectives/apparatus_lexicon (single source of truth);
# re-exported here as ``_EXERCISE_BANNER_RE`` — byte-identical, no behavior change.
from lib.objectives.apparatus_lexicon import (  # noqa: E402
    EXERCISE_BANNER_RE as _EXERCISE_BANNER_RE,
)

# Embedded answer-key / exercise NUMBER markers ("313. 9a", "126. 5"). The
# canonical detector's ``_ANSWER_SEQUENCE_RE`` needs THREE; a math-prose run
# that mixes only two ("... 313. ... 314. ...") slips through. This is the
# 2-or-3-digit variant the task names; we require >=2 so a single "N." in a
# real "Section 2. Foo" title never trips.
_EXERCISE_NUMBER_RE = re.compile(r"(?<!\d)\d{2,3}\.\s")
_EXERCISE_NUMBER_MIN = 2

# Full-sentence prose floor: a heading longer than this many words AND ending
# in a sentence-terminal period is prose, not a title. Real section titles are
# all well under this (a typical numbered title like "1.1 Introduction ...
# (part 1)" is a handful of words and carries no terminal period).
_PROSE_WORD_FLOOR = 15

# Internal sentence-terminal period: a "." followed by whitespace + more text,
# where the char before the "." is NOT a digit (so decimals "3.5" / section
# numbers "1.1" and the answer-key "N." markers handled above don't trip). A
# real title never embeds a mid-string sentence period; combined with a word
# floor this catches prose like "Line up the values one under the other, keeping
# the decimal points aligned. 0.6" while sparing short abbreviation titles
# ("U.S. Customary Units", "Mr. Smith") which sit well under the floor.
_INTERNAL_SENTENCE_RE = re.compile(r"(?<![0-9])\.\s+\S")
_INTERNAL_SENTENCE_WORD_FLOOR = 8


def is_heading_sanity_filter_enabled() -> bool:
    """Whether the ``TRAINFORGE_HEADING_SANITY_FILTER`` gate is on.

    Parse-with-fallback (mirrors the other Trainforge opt-in flags): truthy
    values ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive) enable;
    anything else (including unset) keeps the default-OFF byte-stable path.
    """
    raw = os.environ.get(FLAG_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _strip_part_suffix(text: str) -> str:
    return _PART_SUFFIX_RE.sub("", text).strip()


def is_suspect_section_heading(text: Optional[str]) -> bool:
    """Whether ``text`` is clearly NOT a real heading (defensive).

    Conservative by design — returns ``True`` only on unambiguous noise so a
    legitimate title is NEVER demoted. Three layers, any one of which fires:

    1. The canonical ``_is_noncontent_heading`` (answer-key numeric runs,
       circled-answer markers, front-matter, donor lists).
    2. A publisher exercise-section banner / >=2 embedded ``\\d{2,3}.``
       exercise-answer markers.
    3. Full-sentence prose: >15 words AND ending in a sentence-terminal period.
    """
    if not text or not text.strip():
        return False
    stem = _strip_part_suffix(text)
    if not stem:
        return False

    # Layer 1 — canonical detector (the reuse).
    if _is_noncontent_heading(stem):
        return True

    # Layer 2a — exercise-section banner.
    if _EXERCISE_BANNER_RE.search(stem):
        return True

    # Layer 2b — >=2 embedded exercise-answer number markers.
    if len(_EXERCISE_NUMBER_RE.findall(stem)) >= _EXERCISE_NUMBER_MIN:
        return True

    # Layer 3a — full-sentence prose ending in terminal punctuation. Require
    # BOTH the long-word floor AND a terminal period so a short title ending in
    # a period (rare) survives and a long-but-period-less display heading
    # survives.
    words = stem.split()
    if len(words) > _PROSE_WORD_FLOOR and stem.rstrip().endswith(
        (".", "!", "?")
    ):
        return True

    # Layer 3b — prose carrying an INTERNAL sentence-terminal period (a
    # complete sentence followed by more content), gated on a word floor so a
    # short abbreviation title ("U.S. Customary Units") is spared. This catches
    # prose whose sentence period sits mid-string with a numeric/measurement
    # tail ("... decimal points. 0.6").
    if (
        len(words) > _INTERNAL_SENTENCE_WORD_FLOOR
        and _INTERNAL_SENTENCE_RE.search(stem)
    ):
        return True

    return False


def _clean_ancestor_candidates(
    item: Optional[Dict[str, Any]],
    merged_headings: Optional[List[str]],
) -> List[str]:
    """Ordered nearest-real-ancestor candidates, noise pre-filtered.

    Preference order (nearest → coarsest): the ``merged_headings`` breadcrumb
    (the ordered section headings the chunk was built from), then the page-level
    ``title`` / ``module_title``. Each candidate is itself re-checked through
    ``is_suspect_section_heading`` so a noisy ancestor is never used to repair
    a noisy heading (no noise → noise repair).
    """
    candidates: List[str] = []
    for h in merged_headings or []:
        if isinstance(h, str) and h.strip():
            candidates.append(h.strip())
    if item:
        for key in ("title", "module_title"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

    clean: List[str] = []
    seen = set()
    for cand in candidates:
        stem = _strip_part_suffix(cand)
        if not stem or stem in seen:
            continue
        seen.add(stem)
        if not is_suspect_section_heading(cand):
            clean.append(stem)
    return clean


def repair_section_heading(
    section_heading: str,
    *,
    item: Optional[Dict[str, Any]] = None,
    merged_headings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a repair decision for a chunk's ``section_heading``.

    Result dict:

    * ``heading`` — the heading to stamp into ``source.section_heading``. Equal
      to the input when the input is not suspect, OR when it is suspect but no
      clean ancestor exists (never blanked).
    * ``suspect`` — ``True`` iff the input heading was detected as noise.
    * ``repaired`` — ``True`` iff ``heading`` differs from the input (a clean
      ancestor was substituted).
    * ``original_heading`` — the input, preserved for provenance/audit.

    Pure + side-effect free. The caller decides whether to stamp a
    ``heading_suspect`` provenance flag (``suspect and not repaired``).
    """
    original = section_heading if isinstance(section_heading, str) else ""
    if not is_suspect_section_heading(original):
        return {
            "heading": original,
            "suspect": False,
            "repaired": False,
            "original_heading": original,
        }

    ancestors = _clean_ancestor_candidates(item, merged_headings)
    if ancestors:
        return {
            "heading": ancestors[0],
            "suspect": True,
            "repaired": True,
            "original_heading": original,
        }

    # Suspect but no clean ancestor — never blank, leave as-is + flag.
    return {
        "heading": original,
        "suspect": True,
        "repaired": False,
        "original_heading": original,
    }
