"""Canonical chunker logic lifted out of ``Trainforge/process_course.py``.

Historical context: Phase 7a Subtask 4 originally lifted the chunker
proper into a standalone ``ed4all-chunker`` workspace package so DART,
Courseforge, and Trainforge could share one chunker implementation;
the post-Phase-8 review re-merged that package back into
``Trainforge.chunker`` (this module's current home) to remove the
workspace-member friction. Public names and the ``ChunkerContext``
callback contract are unchanged. The functions in this module
orchestrate:

    1. ``chunk_content`` — top-level loop over parsed IMSCC items;
       handles boilerplate stripping, assessment-feedback stripping,
       per-item section iteration, and follows-chunk linkage.
    2. ``chunk_text_block`` — split a single section's text/html into
       one or more chunk dicts; resolves xpath / char-span provenance
       and dispatches to the caller-provided ``create_chunk`` callback.
    3. ``merge_small_sections`` — merge adjacent ``<MIN_CHUNK_SIZE``
       sections into combined buffers up to ``MAX_CHUNK_SIZE``.
    4. ``merge_section_source_ids`` — union two sourceId-string lists,
       dedupe, preserve insertion order.
    5. ``type_from_heading`` / ``split_by_sentences`` — pure helper
       static functions used internally by the orchestrator.

Constants ``MIN_CHUNK_SIZE = 100`` and ``MAX_CHUNK_SIZE = 800`` are
re-exported as module-level names; callers can override per-call via
the ``min_chunk_size`` / ``max_chunk_size`` keyword arguments on
``chunk_content``.

Architectural note — the ``ChunkerContext`` dependency boundary
=============================================================

``CourseProcessor._create_chunk`` (the function that materialises one
chunk dict from text + html + item-state) is deeply coupled to the
``CourseProcessor`` instance: it calls ``self._extract_concept_tags``,
``self._extract_section_metadata``, ``self._extract_objective_refs``,
``self._resolve_chunk_source_references``,
``self._fill_or_drop_empty_key_term_definitions``,
``self._determine_difficulty``, and reads ``self._lo_parent_map``,
``self.OBJECTIVE_CODE_RE``, ``self.WEEK_PREFIX_RE``,
``self.NON_CONCEPT_TAGS``, ``self.capture``, ``self.stats``,
``self._all_concept_tags``, ``self.course_code``. Lifting that body
into the package would require lifting the whole ontology / metadata
/ provenance surface of Trainforge — outside the scope of Phase 7a
(Subtask 4 estimate of ~450 LOC).

Pragmatic resolution: this module owns the orchestration flow
(looping, merging, splitting, boilerplate, feedback, xpath/char-span
resolution, follows-chunk linkage, position-in-module tracking) but
delegates per-chunk materialisation back to the caller via a
``ChunkerContext`` callable. ``CourseProcessor._create_chunk`` becomes
the callback in Subtask 6's wrapper:

    def _chunk_content(self, parsed_items):
        chunks, pages_with_misconceptions = chunk_content(
            parsed_items,
            self.course_code,
            self._boilerplate_spans,
            min_chunk_size=self.MIN_CHUNK_SIZE,
            max_chunk_size=self.MAX_CHUNK_SIZE,
            ctx=ChunkerContext(create_chunk=self._create_chunk),
        )
        self._pages_with_misconceptions = pages_with_misconceptions
        self.stats["total_chunks"] = len(chunks)
        return chunks

The empty-input contract from the plan's verification command —
``chunk_content([], 'TEST_101') == []`` — works with ``ctx=None``
because the loop never reaches the ``create_chunk`` call site. A
non-empty input requires a non-None ``ctx``; mismatch raises
``ChunkerContextRequired`` so silent no-op output can't mask a
mis-wiring.

Imports
=======

``Trainforge.parsers.xpath_walker`` is imported at module top now that
the chunker lives inside ``Trainforge``. Pre-Phase-7a-revert this was
a lazy import (the chunker was a sibling package and the lazy form
dodged a hypothetical module-load cycle); the import-cycle risk no
longer applies because ``xpath_walker`` is stdlib-only and never
reaches ``Trainforge.process_course``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from Trainforge.chunker.boilerplate import strip_boilerplate
from Trainforge.chunker.cross_course_dedup import (
    exact_content_hash,
    exact_token_count,
    resolve_chunk_dedup_enabled,
    resolve_chunk_dedup_min_tokens,
)
from Trainforge.chunker.helpers import (
    _block_element_spans,
    build_semantik_block_offset_index,
    extract_plain_text_with_curies,
    extract_section_html,
    harvest_semantik_source_refs,
    resolve_dart_refs_for_chunk,
    strip_assessment_feedback,
    strip_feedback_from_text,
    type_from_resource,
)
from Trainforge.parsers.xpath_walker import (
    find_body_xpath,
    find_section_container_xpath,
    resolve_xpath,
)

__all__ = [
    "MIN_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "TARGET_CHUNK_SIZE",
    "CANONICAL_CHUNK_TYPES",
    "ChunkerContext",
    "ChunkerContextRequired",
    "MergedSectionResult",
    "chunk_content",
    "chunk_text_block",
    "merge_small_sections",
    "merge_section_source_ids",
    "type_from_heading",
    "retype_from_leading_text",
    "leads_with_exercise_banner",
    "leads_with_lo_boilerplate",
    "split_by_sentences",
    "apply_chunk_overlap",
    "resolve_chunk_overlap_words",
    "CHUNK_OVERLAP_WORDS_ENV",
    "resolve_chunk_code_split",
    "looks_like_code_listing",
    "split_code_by_lines",
    "CHUNK_CODE_SPLIT_ENV",
    "resolve_merge_fragment_floor",
    "MERGE_FRAGMENT_FLOOR_ENV",
    "fold_fragment_results",
    "fold_fragment_texts",
    "resolve_chunk_section_hard_break",
    "CHUNK_SECTION_HARD_BREAK_ENV",
    "resolve_chunk_subsection_break",
    "resolve_chunk_subsection_min_words",
    "CHUNK_SUBSECTION_BREAK_ENV",
    "CHUNK_SUBSECTION_MIN_WORDS_ENV",
    "SUBSECTION_MIN_WORDS_DEFAULT",
    "is_section_boundary_heading",
    "aggregate_composite_unit",
    "aggregate_unit_subclass",
    "section_unit_signals",
    "section_subclass_signal",
    "DEDUP_UNIT_KEY_SEPARATOR",
]


# ---------------------------------------------------------------------------
# Module-level constants (mirror ``CourseProcessor.MIN_CHUNK_SIZE`` /
# ``MAX_CHUNK_SIZE`` / ``TARGET_CHUNK_SIZE`` at ``process_course.py:985-987``).
# ---------------------------------------------------------------------------

#: Minimum chunk size in words. Courseforge pages can be short
#: (overviews, summaries) so we don't drop under-min sections — we
#: merge them into the next adjacent section instead.
MIN_CHUNK_SIZE: int = 100

#: Maximum chunk size in words. ``chunk_text_block`` sentence-splits
#: above this floor.
MAX_CHUNK_SIZE: int = 800

#: Target chunk size in words for sentence splitting (``chunk_text_block``).
TARGET_CHUNK_SIZE: int = 500

#: Canonical chunk-type enum used by ``merge_small_sections`` to gate
#: ``data-cf-template-type`` propagation. Source of truth:
#: ``schemas/taxonomies/content_type.json::ChunkType``. Mirrors the
#: ``CANONICAL_CHUNK_TYPES`` frozenset at ``process_course.py:103-114``.
CANONICAL_CHUNK_TYPES: frozenset = frozenset({
    "assessment_item",
    "overview",
    "summary",
    "exercise",
    "explanation",
    "example",
    "procedure",
    "real_world_scenario",
    "common_pitfall",
    "problem_solution",
})


# ---------------------------------------------------------------------------
# Chunk overlap — verbatim trailing-word recovery (Track K, default off)
# ---------------------------------------------------------------------------

#: Env var gating the optional verbatim chunk-overlap pass. Default off
#: (unset / non-positive / garbage → 0 → byte-identical legacy emit).
CHUNK_OVERLAP_WORDS_ENV: str = "TRAINFORGE_CHUNK_OVERLAP_WORDS"

#: Sentinel distinguishing an absent ``follows_chunk`` key (generic dicts
#: in tests → pure sequential overlap) from an explicit ``None`` linkage
#: (a real chunk at a lesson boundary → no cross-lesson bleed).
_OVERLAP_FOLLOWS_MISSING = object()


def resolve_chunk_overlap_words(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve ``TRAINFORGE_CHUNK_OVERLAP_WORDS`` (parse-with-fallback).

    Returns the number of verbatim trailing words to prepend from each
    chunk's prior emitted chunk onto the next. Default ``0`` (feature
    off → byte-identical legacy emit). Garbage / non-integer /
    non-positive values fall back to ``0`` (mirrors the other
    ``TRAINFORGE_*`` numeric knobs).
    """
    import os

    src = env if env is not None else os.environ
    raw = src.get(CHUNK_OVERLAP_WORDS_ENV)
    if raw is None:
        return 0
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return val if val > 0 else 0


def apply_chunk_overlap(
    chunks: List[Dict[str, Any]], overlap_words: int
) -> List[Dict[str, Any]]:
    """Prepend the verbatim trailing ``overlap_words`` of each chunk's
    prior emitted chunk onto the next chunk's ``text`` (in place).

    Anti-fabrication: the prepended prefix is composed solely from the
    PRIOR chunk's existing words — never synthesized, never invented.
    The trailing words are read from a pre-mutation snapshot of every
    chunk's text so the overlap never compounds across the sequence (the
    prefix is always the prior chunk's OWN tail, not a tail that already
    carries an earlier chunk's bleed).

    Continuity guard: when a chunk carries the canonical ``follows_chunk``
    linkage, the prefix is bled only when this chunk genuinely follows
    the prior one (``follows_chunk == prev["id"]``) — so overlap never
    crosses a lesson / module boundary where ``follows_chunk`` resets to
    ``None``. Generic chunk dicts that carry no ``follows_chunk`` key
    fall back to pure sequential overlap.

    ``overlap_words <= 0`` (or fewer than two chunks) → no-op; the list
    is returned unchanged (byte-identical legacy emit). ``word_count`` /
    ``tokens_estimate`` are recomputed for any mutated chunk so the
    derived counts stay consistent with the new text.
    """
    if overlap_words <= 0 or len(chunks) < 2:
        return chunks

    original_texts = [str(c.get("text", "")) for c in chunks]
    for i in range(1, len(chunks)):
        cur = chunks[i]
        prev = chunks[i - 1]
        follows = cur.get("follows_chunk", _OVERLAP_FOLLOWS_MISSING)
        if follows is not _OVERLAP_FOLLOWS_MISSING:
            if follows is None or follows != prev.get("id"):
                continue
        prev_words = original_texts[i - 1].split()
        if not prev_words:
            continue
        tail = prev_words[-overlap_words:]
        if not tail:
            continue
        prefix = " ".join(tail)
        cur_text = str(cur.get("text", ""))
        cur["text"] = (prefix + " " + cur_text) if cur_text else prefix
        new_word_count = len(cur["text"].split())
        if "word_count" in cur:
            cur["word_count"] = new_word_count
        if "tokens_estimate" in cur:
            cur["tokens_estimate"] = int(new_word_count * 1.3)
    return chunks


# ---------------------------------------------------------------------------
# W1b.2 — code-listing split (opt-in, default off)
#
# A large fenced code listing (a notebook cell, a full class, a config file)
# lands as a single ``explanation`` chunk. When it exceeds ``max_chunk_size``
# the legacy path sentence-splits it — but code has almost no sentence
# punctuation, so ``split_by_sentences`` returns ONE giant sub-chunk that
# still overflows. When ``ED4ALL_CHUNK_CODE_SPLIT`` is on, an oversized chunk
# whose body reads as a code listing is split on LINE boundaries instead, so
# each piece fits the window. Default off → byte-identical legacy emit.
# ---------------------------------------------------------------------------

#: Env var gating the code-listing line-split path.
CHUNK_CODE_SPLIT_ENV: str = "ED4ALL_CHUNK_CODE_SPLIT"

#: Minimum non-blank line count before a block is even a code candidate.
_CODE_MIN_LINES: int = 6

#: Fraction of non-blank lines that must read as code for the block to split
#: on line boundaries rather than sentences.
_CODE_LINE_FRACTION: float = 0.55

#: Per-line code signatures (any hit → the line reads as code).
_CODE_LINE_RE = re.compile(
    r"(?:^\s{2,}\S)"                       # indented continuation
    r"|(?:[;{}]\s*$)"                      # trailing brace / semicolon
    r"|(?:\)\s*:?\s*$)"                    # trailing paren / def colon
    r"|(?:^\s*(?:def|class|import|from|return|for|while|if|elif|else|"
    r"try|except|with|async|await|print|package|public|private|const|"
    r"let|var|function|#include|@)\b)"
    r"|(?:=>|::|->|>>>|\$\s)"              # arrows / REPL / shell prompt
    r"|(?:\w+\s*\([^)]*\)\s*[:{;]?\s*$)"  # call/def-ish line
)


def resolve_chunk_code_split(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CHUNK_CODE_SPLIT`` (parse-with-fallback, default OFF)."""
    import os

    src = env if env is not None else os.environ
    return src.get(CHUNK_CODE_SPLIT_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def looks_like_code_listing(text: str) -> bool:
    """Heuristically decide whether ``text`` is a code listing (not prose).

    Conservative: requires at least :data:`_CODE_MIN_LINES` non-blank lines
    AND that :data:`_CODE_LINE_FRACTION` of them carry a code signature. Pure /
    deterministic; no dependency. Real running prose (few indented lines, no
    braces/def keywords) returns False, so the code-split path never fires on
    ordinary explanation chunks.
    """
    if not text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < _CODE_MIN_LINES:
        return False
    code_lines = sum(1 for ln in lines if _CODE_LINE_RE.search(ln))
    return (code_lines / len(lines)) >= _CODE_LINE_FRACTION


def split_code_by_lines(text: str, target_words: int) -> List[str]:
    """Split a code listing into line-grouped pieces of up to ``target_words``.

    Breaks on LINE boundaries (preferring blank-line separators between
    logical blocks) so no piece exceeds the word budget and no line is cut
    mid-token. Anti-fabrication: every emitted piece is a contiguous run of
    the original lines — nothing synthesized. A single line wider than the
    budget is emitted alone. Mirrors :func:`split_by_sentences`'s
    accumulate-then-flush shape but over lines.
    """
    lines = text.splitlines()
    pieces: List[str] = []
    current: List[str] = []
    current_wc = 0
    for line in lines:
        lwc = len(line.split())
        if current and current_wc + lwc > target_words:
            pieces.append("\n".join(current))
            current = [line]
            current_wc = lwc
        else:
            current.append(line)
            current_wc += lwc
    if current:
        pieces.append("\n".join(current))
    # Drop empty trailing pieces (all-blank runs) but keep ≥1.
    non_empty = [p for p in pieces if p.strip()]
    return non_empty or ([text] if text.strip() else [])


# ---------------------------------------------------------------------------
# W1b.3 — sub-N-word fragment floor for the small-section merger (opt-in)
#
# ``merge_small_sections`` can still emit a runt result: a tiny trailing
# section (measured as low as 2 words) that overflowed the buffer becomes its
# own ``MergedSectionResult``, and that fragment survives into the chunkset as
# a near-empty chunk. When ``ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR`` is a positive
# int, a post-merge pass FOLDS any result whose combined body is below the
# floor into the PREVIOUS result (unioning its provenance), or drops it when
# it is the only/leading result with nowhere to fold. Normal chunks (bodies
# above the floor) are untouched, so default off (0) is byte-identical and a
# small positive floor only rescues genuine fragments.
# ---------------------------------------------------------------------------

#: Env var gating the fragment-floor fold. Default 0 → OFF → byte-identical.
MERGE_FRAGMENT_FLOOR_ENV: str = "ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR"


def resolve_merge_fragment_floor(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve ``ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR`` (parse-with-fallback → 0).

    Returns the minimum word count a merged result must carry to stand alone.
    ``0`` (default / garbage / non-positive) disables the fold entirely
    (byte-identical legacy merge). A recommended operator value is ``5``.
    """
    import os

    src = env if env is not None else os.environ
    raw = src.get(MERGE_FRAGMENT_FLOOR_ENV)
    if raw is None:
        return 0
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return val if val > 0 else 0


def fold_fragment_results(
    merged: List["MergedSectionResult"], fragment_floor: int
) -> List["MergedSectionResult"]:
    """Fold sub-``fragment_floor``-word merged results into the prior result.

    A result whose ``combined_text`` word count is below ``fragment_floor`` is
    appended (text + provenance) onto the PREVIOUS surviving result. A leading
    fragment with no prior result to fold into is dropped (it is by definition
    below the useful-chunk floor). ``fragment_floor <= 0`` → no-op (returns the
    input list unchanged). Provenance union mirrors the merger's own buffer
    accumulation (``merge_section_source_ids`` +
    ``_merge_objective_alignment_dedup`` + heading/claim concat).
    """
    if fragment_floor <= 0 or not merged:
        return merged
    out: List[MergedSectionResult] = []
    for result in merged:
        wc = len(str(result.combined_text or "").split())
        # A section-boundary result (Fix 1) is NEVER folded backward: folding
        # it into the prior result would re-fuse across the exact section
        # boundary the hard break enforced. It stands alone even when sub-floor
        # (the documented irreducible case) — including a leading one, which is
        # kept rather than dropped so a section opener is not lost.
        if wc >= fragment_floor or not out or result.section_boundary:
            if wc >= fragment_floor or result.section_boundary:
                out.append(result)
            elif not out:
                # Leading non-boundary fragment, nowhere to fold — drop it.
                continue
            else:  # pragma: no cover - unreachable (guarded above)
                out.append(result)
            continue
        prev = out[-1]
        prev.combined_text = (
            f"{prev.combined_text}\n\n{result.combined_text}"
            if prev.combined_text
            else result.combined_text
        )
        merge_section_source_ids(prev.merged_source_ids, result.merged_source_ids)
        for h in result.merged_headings:
            if h not in prev.merged_headings:
                prev.merged_headings.append(h)
        prev.merged_key_claims.extend(result.merged_key_claims)
        _merge_objective_alignment_dedup(
            prev.merged_objective_alignment, result.merged_objective_alignment
        )
        # Wave #22 quick-wins: union the folded fragment's pedagogical roles into
        # the prior result (sorted, deduped); keep the prior result's unit unless
        # it had none (the surviving result is the dominant one — no fabrication).
        if result.merged_unit_roles:
            prev.merged_unit_roles = sorted(
                set(prev.merged_unit_roles)
                | set(result.merged_unit_roles)
            )
        if not prev.merged_composite_unit and result.merged_composite_unit:
            prev.merged_composite_unit = result.merged_composite_unit
    return out


# ---------------------------------------------------------------------------
# W1b.3b — sub-``fragment_floor``-word runt fold for the SENTENCE/CODE-split
# path (opt-in, shares the ``ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR`` floor).
#
# ``fold_fragment_results`` only runs at the ``merge_small_sections`` seam —
# i.e. BEFORE ``chunk_text_block`` sentence-splits an over-``max_chunk_size``
# merged result. A sentence split (or the code-listing line split) accumulates
# then flushes, so a tiny TRAILING sentence ("Chapter 1 Foundations 7.096.",
# an orphaned question stem) becomes its own sub-``fragment_floor`` piece that
# the section-level fold never sees. This helper folds those runt pieces back
# into an adjacent piece at the TEXT level, before ``ctx.create_chunk`` mints a
# chunk — so the emitted chunk list carries no split-produced runt.
# ---------------------------------------------------------------------------


def fold_fragment_texts(pieces: List[str], fragment_floor: int) -> List[str]:
    """Fold sub-``fragment_floor``-word split pieces into an adjacent piece.

    Operates on the ``sub_texts`` list produced by :func:`split_by_sentences`
    (or :func:`split_code_by_lines`). Any piece whose word count is below
    ``fragment_floor`` is concatenated onto the PREVIOUS surviving piece; a
    leading runt with no prior piece is prepended onto the NEXT piece instead
    (so no content is lost — unlike the section-level fold, which can drop a
    leading section-fragment because that fragment's provenance is already
    carried by the merged result). ``fragment_floor <= 0`` or fewer than two
    pieces → no-op (returns the input unchanged → byte-identical legacy emit).

    The result may still contain a single sub-floor piece when the WHOLE input
    is below the floor (nothing to fold into) — that is the documented
    irreducible case (a genuinely tiny section/item is one chunk).
    """
    if fragment_floor <= 0 or len(pieces) < 2:
        return pieces
    out: List[str] = []
    pending_prefix = ""
    for piece in pieces:
        text = piece if not pending_prefix else f"{pending_prefix}\n\n{piece}"
        pending_prefix = ""
        wc = len(text.split())
        if wc >= fragment_floor:
            out.append(text)
        elif out:
            # Sub-floor piece with a prior surviving piece → fold backward.
            out[-1] = f"{out[-1]}\n\n{text}"
        else:
            # Leading sub-floor piece, nothing before it → carry it forward
            # onto the next piece so content is never dropped.
            pending_prefix = text
    if pending_prefix:
        if out:
            out[-1] = f"{out[-1]}\n\n{pending_prefix}"
        else:
            out.append(pending_prefix)
    return out or list(pieces)


# ---------------------------------------------------------------------------
# Fix 1 — hard chunk break at SECTION-heading boundaries (opt-in, default off)
#
# ``merge_small_sections`` soft-packs an incoming section into the accumulating
# buffer whenever ``buffer_wc + section.word_count <= max_chunk_size``. On a
# textbook that leaves a section's closing apparatus (a "Self Check", an
# exercise tail) FUSED with the next section's opener ("Use the Language of
# Algebra / Learning Objectives", a "BE PREPARED" readiness quiz) whenever the
# buffer is still under target. When ``ED4ALL_CHUNK_SECTION_HARD_BREAK`` is on,
# crossing a SECTION-heading boundary forces a flush before the new section is
# accumulated, so a chunk never straddles two textbook sections. Default off →
# byte-identical legacy merge (the RDF/SHACL calibration corpus, whose
# Courseforge pages carry "Learning Objectives" headings, stays byte-stable).
# ---------------------------------------------------------------------------

#: Env var gating the section-boundary hard break. Default OFF → byte-identical.
CHUNK_SECTION_HARD_BREAK_ENV: str = "ED4ALL_CHUNK_SECTION_HARD_BREAK"

#: A numbered top-level section opener (``"1.2 Use the Language of Algebra"``,
#: ``"4.5"``). Requires TWO dotted numbers so a single-number exercise label
#: (``"1."``, ``"27."``) never trips it.
_SECTION_BREAK_NUM_RE = re.compile(r"^\s*\d+\.\d+")

#: Fallback textbook section-opener phrases (matched case-insensitively anywhere
#: in the heading) used when the canonical lexicon loader is unavailable.
#: ``learning objective`` covers textbook "Learning Objectives" openers;
#: ``be prepared`` covers the readiness-quiz opener. The live values are
#: normally derived from ``lib.ontology.taxonomy.get_lexicon_openers`` — see
#: :func:`_section_opener_substrings` — with this tuple as the no-taxonomy
#: fallback so the chunker never loses opener detection if the lexicon can't
#: load.
_SECTION_OPENER_SUBSTRINGS_FALLBACK = ("learning objective", "be prepared")

#: Opener identities (``role`` / ``association_role``) that mark a new-SECTION
#: boundary: the objectives header and the readiness ("Be Prepared") quiz. Other
#: openers (worked_example, try_it, solution, how_to) are interior apparatus, NOT
#: section boundaries, so they are deliberately excluded here.
_SECTION_OPENER_ROLES = frozenset({"objectives", "readiness_check"})
_SECTION_OPENER_ASSOCIATION_ROLES = frozenset({"objectives", "readiness"})

#: Regex metacharacters that make a lexicon ``pattern`` infeasible to reduce to a
#: plain case-insensitive substring (after whitespace-class normalization).
_REGEX_META_CHARS = frozenset(r"\^$.|?*+()[]{}")


def _pattern_to_substring(pattern: str) -> Optional[str]:
    """Reduce a lexicon opener ``pattern`` to a plain lowercase substring.

    Normalizes the common whitespace regex classes (``\\s+`` / ``\\s*`` /
    ``\\s``) to a single space, lowercases, and collapses runs of whitespace.
    Returns ``None`` when the residue still carries regex metacharacters (so the
    caller falls back rather than matching a broken literal).
    """
    if not pattern:
        return None
    sub = re.sub(r"\\s[+*]?", " ", pattern)
    sub = re.sub(r"\s+", " ", sub).strip().lower()
    if not sub or any(ch in _REGEX_META_CHARS for ch in sub):
        return None
    return sub


@lru_cache(maxsize=1)
def _section_opener_substrings() -> Tuple[str, ...]:
    """Resolve the section-opener substrings from the canonical lexicon.

    Derives the objectives + readiness opener label substrings from
    ``lib.ontology.taxonomy.get_lexicon_openers`` (regex patterns reduced to
    plain lowercase substrings). Falls back to
    :data:`_SECTION_OPENER_SUBSTRINGS_FALLBACK` when the taxonomy can't be
    imported/loaded or yields no feasible substrings, so opener detection is
    never silently lost. Cached — the lexicon is static per process.
    """
    try:
        from lib.ontology.taxonomy import get_lexicon_openers

        derived: List[str] = []
        for opener in get_lexicon_openers():
            role = str(opener.get("role", "")).strip()
            assoc = str(opener.get("association_role", "")).strip()
            if (
                role not in _SECTION_OPENER_ROLES
                and assoc not in _SECTION_OPENER_ASSOCIATION_ROLES
            ):
                continue
            sub = _pattern_to_substring(str(opener.get("pattern", "")))
            if sub and sub not in derived:
                derived.append(sub)
        if derived:
            return tuple(derived)
    except Exception:  # pragma: no cover - defensive: never lose detection
        pass
    return _SECTION_OPENER_SUBSTRINGS_FALLBACK


def resolve_chunk_section_hard_break(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CHUNK_SECTION_HARD_BREAK`` (parse-with-fallback, OFF)."""
    import os

    src = env if env is not None else os.environ
    return src.get(CHUNK_SECTION_HARD_BREAK_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Fix 1b — SIZE-GUARDED sub-section break (opt-in, default off)
#
# The size-guarded successor to ``ED4ALL_CHUNK_SECTION_HARD_BREAK``. The hard
# break flushes at EVERY matching boundary with no size guard — which on a
# heading-dense corpus (many small heading-led sub-sections) produced a runt
# cascade (measured: 498 chunks / p50 42 words / 208 runts). This flag instead
# treats a genuine content SUB-SECTION heading — one that
# ``is_section_boundary_heading`` misses because it is NOT ``N.M``-numbered and
# NOT a lexicon opener (e.g. a level-3/4 "Use Place Value with Whole Numbers")
# — as a break CANDIDATE, but only FLUSHES the accumulating buffer once it has
# reached ``subsection_min_words`` (default 250). So a break lands on a real
# pedagogical boundary once enough content has accumulated, never on the first
# heading after a flush → no runt cascade. Apparatus/pedagogy-label headings
# (exercise / review reprints, LO-intro boilerplate, promoted openers, bare
# pedagogy labels) are excluded so a worked example is never split from its
# solution and an "N.M Exercises" reprint never becomes a topic break. Default
# off → byte-identical legacy merge.
# ---------------------------------------------------------------------------

#: Env var gating the size-guarded sub-section break. Default OFF → byte-identical.
CHUNK_SUBSECTION_BREAK_ENV: str = "ED4ALL_CHUNK_SUBSECTION_BREAK"

#: Env var setting the accumulation floor (words) a buffer must reach before a
#: sub-section heading is allowed to flush it. Default 250 — well above the
#: MIN_CHUNK_SIZE (100) merge floor so the break lands on real content, not a runt.
CHUNK_SUBSECTION_MIN_WORDS_ENV: str = "ED4ALL_CHUNK_SUBSECTION_MIN_WORDS"

#: Default sub-section accumulation floor in words.
SUBSECTION_MIN_WORDS_DEFAULT: int = 250

#: Bare pedagogy-label headings ("Try It", "Solution", "Example") are at most
#: this many words; a longer heading carrying an opener token (e.g. "Solutions
#: of Linear Equations") is real content and stays a break candidate.
_PEDAGOGY_LABEL_MAX_WORDS: int = 3


def resolve_chunk_subsection_break(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CHUNK_SUBSECTION_BREAK`` (parse-with-fallback, OFF)."""
    import os

    src = env if env is not None else os.environ
    return src.get(CHUNK_SUBSECTION_BREAK_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def resolve_chunk_subsection_min_words(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve ``ED4ALL_CHUNK_SUBSECTION_MIN_WORDS`` (parse-with-fallback).

    Positive int wins; garbage / non-positive / unset falls back to
    :data:`SUBSECTION_MIN_WORDS_DEFAULT` (250).
    """
    import os

    src = env if env is not None else os.environ
    raw = src.get(CHUNK_SUBSECTION_MIN_WORDS_ENV, "")
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return SUBSECTION_MIN_WORDS_DEFAULT
    return val if val > 0 else SUBSECTION_MIN_WORDS_DEFAULT


@lru_cache(maxsize=1)
def _all_opener_substrings() -> Tuple[str, ...]:
    """Resolve EVERY lexicon opener label reduced to a plain lowercase substring.

    Unlike :func:`_section_opener_substrings` (objectives + readiness only, for
    the SECTION-boundary text arm), this harvests all opener roles — try_it,
    worked_example, how_to, solution — so a bare pedagogy-label heading is
    recognised and excluded from the sub-section break candidate set. Falls back
    to a static tuple when the taxonomy can't be loaded, so exclusion is never
    silently lost. Cached — the lexicon is static per process.
    """
    fallback = (
        "learning objective", "be prepared", "try it", "example",
        "how to", "solution",
    )
    try:
        from lib.ontology.taxonomy import get_lexicon_openers

        derived: List[str] = []
        for opener in get_lexicon_openers():
            sub = _pattern_to_substring(str(opener.get("pattern", "")))
            if sub and sub not in derived:
                derived.append(sub)
        if derived:
            return tuple(derived)
    except Exception:  # pragma: no cover - defensive: never lose exclusion
        pass
    return fallback


def _heading_is_bare_pedagogy_label(heading: str) -> bool:
    """Whether ``heading`` is a SHORT bare pedagogy-opener label.

    A short heading (``<= _PEDAGOGY_LABEL_MAX_WORDS`` words) carrying a lexicon
    opener token — "Try It", "Solution", "Example", "Learning Objectives" — is
    an interior apparatus label, not a content sub-section. A longer heading that
    merely CONTAINS an opener token ("Solutions of Linear Equations") is real
    content and is NOT excluded.
    """
    h = heading.strip().lower()
    if not h or len(h.split()) > _PEDAGOGY_LABEL_MAX_WORDS:
        return False
    return any(sub in h for sub in _all_opener_substrings())


def _is_subsection_break_candidate(section: Any) -> bool:
    """Whether ``section`` starts a genuine content SUB-SECTION (a size-guarded
    break point under ``ED4ALL_CHUNK_SUBSECTION_BREAK``).

    A candidate is a HEADING-LED section (non-empty heading) whose heading reads
    as genuine content. Excluded (reusing the existing apparatus / lexicon
    classification): an exercise / practice / review banner incl. the numbered
    "N.M Exercises" reprint (:func:`leads_with_exercise_banner`), LO-intro
    boilerplate (:func:`leads_with_lo_boilerplate`), a promoted pedagogical
    opener (``data_dart_opener`` attribute), and a bare pedagogy label
    (:func:`_heading_is_bare_pedagogy_label`). The non-numbered content
    sub-section headings (h3/h4, e.g. "Use Place Value with Whole Numbers") that
    :func:`is_section_boundary_heading` misses are the intended target; numbered
    ``N.M`` section headings also qualify (they are content, not apparatus).
    """
    heading = (getattr(section, "heading", "") or "").strip()
    if not heading:
        return False  # continuation body, not a heading-led boundary
    if leads_with_exercise_banner(heading):
        return False
    if leads_with_lo_boilerplate(heading):
        return False
    if getattr(section, "data_dart_opener", None):
        return False
    if _heading_is_bare_pedagogy_label(heading):
        return False
    return True


def is_section_boundary_heading(heading: str) -> bool:
    """True when ``heading`` reads as a new-SECTION opener (textual signals).

    Fires on a numbered ``N.N`` section title or on a "Learning Objectives" /
    "BE PREPARED" opener phrase — the unambiguous new-section signals that
    should never be mid-chunk continuations. Level-based detection (an ``h1``/
    ``h2`` heading) is handled separately in :func:`_section_is_boundary` where
    the section object's ``level`` is available; this helper is the text-only
    arm (also used for openers that render at ``h3``).
    """
    if not heading:
        return False
    h = heading.strip().lower()
    if any(sig in h for sig in _section_opener_substrings()):
        return True
    return bool(_SECTION_BREAK_NUM_RE.match(heading))


def _section_is_boundary(section: Any) -> bool:
    """Whether ``section`` starts a new textbook SECTION (a hard-break point).

    Two arms: (1) a genuine top-level heading — ``level`` in ``{1, 2}`` (h1
    chapter / h2 section) — is always a boundary; (2) a heading matching a
    numbered ``N.N`` title or a "Learning Objectives" / "BE PREPARED" opener
    phrase (:func:`is_section_boundary_heading`) at any level. ``h3+``
    subsections that are NOT openers are deliberately NOT boundaries so normal
    sub-section merging is preserved (breaking every subsection would
    over-fragment retrieval chunks). Duck-typed: ``level`` is read via
    ``getattr`` so a section-like without the attribute still resolves on the
    textual arm.
    """
    lvl = getattr(section, "level", None)
    try:
        if lvl is not None and 1 <= int(lvl) <= 2:
            return True
    except (TypeError, ValueError):
        pass
    # A7 (end-user-HTML audit): a section carrying a ``data-dart-opener`` role
    # (a promoted pedagogical opener — objectives / try_it / worked_example / …
    # from the SemantiK adapter) is a soft sub-boundary, so a chunk never fuses
    # a worked example, its solution, and the next example. Attribute-driven —
    # supersedes literal-string matching for DART openers. Only consulted under
    # ED4ALL_CHUNK_SECTION_HARD_BREAK (this fn's sole call site is gated on it).
    if getattr(section, "data_dart_opener", None):
        return True
    # Wave #22 Tier-2 (composite units): a section at a composite-unit EDGE
    # (its ``data-dart-unit`` type harvested from the ``<section class="dart-unit">``
    # wrapper) is a preferred boundary, so a chunk never straddles two units.
    # Attribute-driven; only consulted under ED4ALL_CHUNK_SECTION_HARD_BREAK
    # (this fn's sole call site is gated on it).
    if getattr(section, "data_dart_unit", None):
        return True
    return is_section_boundary_heading(getattr(section, "heading", "") or "")


# ---------------------------------------------------------------------------
# Wave #22 quick-wins — chunk pedagogical-role metadata (additive, no gate)
#
# The SemantiK-converted HTML carries ``data-dart-unit`` (composite-unit type)
# on unit ``<section>``s, ``data-dart-flow`` (statement / solution-steps /
# procedure-steps) on body blocks, and ``data-dart-opener`` roles on promoted
# opener headings. The parser harvests these onto ``ContentSection``
# (``data_dart_unit`` / ``data_dart_flows`` / ``data_dart_opener``). These pure
# helpers aggregate the section-level signals into the chunk-level
# ``composite_unit`` (type string or null) + ``unit_roles`` (list) that
# the create_chunk callbacks stamp additively. Legacy / non-DART sections carry
# none of the attributes → unit None, roles [] → the chunk fields stay absent
# (byte-identical). Values are the lexicon role vocabulary (openers) + the flow
# vocabulary; profile-extensible, so no fixed enum is imposed.
# ---------------------------------------------------------------------------


def section_unit_signals(section: Any) -> Tuple[Optional[str], List[str]]:
    """Extract ``(composite_unit, roles)`` from one section-like object.

    ``composite_unit`` is the section's ``data_dart_unit`` (or ``None``).
    ``roles`` is the union of the section's ``data_dart_opener`` role (when
    present) and its ``data_dart_flows`` list. Duck-typed via ``getattr`` so a
    legacy ContentSection-like without the attributes yields ``(None, [])``.
    """

    unit = getattr(section, "data_dart_unit", None) or None
    roles: List[str] = []
    opener = getattr(section, "data_dart_opener", None)
    if opener:
        roles.append(str(opener).strip())
    for flow in getattr(section, "data_dart_flows", None) or []:
        if flow and str(flow).strip():
            roles.append(str(flow).strip())
    return unit, [r for r in roles if r]


def aggregate_composite_unit(units: List[Optional[str]]) -> Optional[str]:
    """Resolve a chunk's ``composite_unit`` from its constituent units.

    A single distinct non-null value → that value. Multiple distinct values →
    the plurality winner, or ``None`` on a tie (honest: no fabricated unit when
    the merged sections genuinely disagree). All-null / empty → ``None``.
    """

    present = [u for u in units if u]
    if not present:
        return None
    ranked = Counter(present).most_common()
    top_val, top_n = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_n:
        # Tie for the top count → no clear majority unit.
        return None
    return top_val


def section_subclass_signal(section: Any) -> Optional[str]:
    """Extract the section's ``data_dart_subclass`` Tier-3 label (or ``None``).

    Duck-typed via ``getattr`` so a legacy ContentSection-like without the
    attribute yields ``None``.
    """

    return getattr(section, "data_dart_subclass", None) or None


def aggregate_unit_subclass(
    units: List[Optional[str]], subclasses: List[Optional[str]]
) -> Optional[str]:
    """Resolve a chunk's ``unit_subclass`` from its constituent sections.

    Build #23 — the subclass RIDES the composite unit (both attributes co-locate
    on the SemantiK unit wrapper), so no independent plurality vote is needed: a
    chunk's ``unit_subclass`` is the subclass paired with its RESOLVED
    ``composite_unit`` (:func:`aggregate_composite_unit`). Returns ``None`` when
    the unit doesn't resolve (a chunk straddling two units → no confident unit →
    no subclass) or when the winning unit carried no subclass. ``units`` and
    ``subclasses`` are parallel per-section lists.
    """

    resolved = aggregate_composite_unit(units)
    if resolved is None:
        return None
    for unit, subclass in zip(units, subclasses):
        if unit == resolved and subclass:
            return subclass
    return None


# ---------------------------------------------------------------------------
# ChunkerContext — caller-provided callbacks for chunk materialisation
# ---------------------------------------------------------------------------


class ChunkerContextRequired(RuntimeError):
    """Raised when ``chunk_content`` receives parsed items but no ``ctx``.

    The empty-input case (``chunk_content([], ...)``) is intentionally
    permitted with ``ctx=None`` so the package's verification contract
    works without wiring a full Trainforge state. Any non-empty input
    needs the caller-side ``create_chunk`` callback to materialise the
    per-chunk dict (concept tags, objective refs, bloom level, source
    references, etc.) — see the module docstring for the architectural
    rationale.
    """


@dataclass
class ChunkerContext:
    """Caller-provided callbacks the chunker dispatches to per-chunk.

    ``create_chunk`` receives the chunker's resolved arguments and
    returns the materialised chunk dict. The signature mirrors
    ``CourseProcessor._create_chunk`` at
    ``Trainforge/process_course.py:1823`` so the Subtask 6 wrapper
    can pass the bound method straight through.

    ``type_from_heading_fn`` overrides the package's default
    heading-keyword heuristic (``type_from_heading`` below). Defaults
    to the package implementation; override only when a downstream
    consumer wants to swap in a different heading-classifier (e.g. a
    DART-side classifier that knows about ``data-dart-*`` attributes).
    """

    create_chunk: Callable[..., Dict[str, Any]]
    type_from_heading_fn: Optional[Callable[[str], str]] = None

    def heading_type(self, heading: str) -> str:
        """Resolve the chunk type from a section heading."""

        fn = self.type_from_heading_fn or type_from_heading
        return fn(heading)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def type_from_heading(heading: str) -> str:
    """Classify a chunk type from its section heading.

    Mirrors ``CourseProcessor._type_from_heading`` at
    ``process_course.py:2591`` — keyword-based classifier with
    ``"explanation"`` as the default. Pure / static; no class state.

    Default behaviour is the legacy pure-substring matcher. Setting
    ``TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE=true`` swaps in the
    whole-word / course-title-aware classifier
    (:func:`_type_from_heading_content_aware`) which fixes the ~7%
    chunk-type misclassification audited on the NVIDIA corpus
    (substring hits on class names like "DocumentSummaryBase",
    bare "NVIDIA Course NN:" page titles mis-typed off an incidental
    word, and "[Exercise] ... Example" headings losing to ``example``).
    Default off so the RDF/SHACL calibration corpus stays
    byte-identical on rebuild.
    """

    import os

    if os.getenv("TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE", "").lower() == "true":
        return _type_from_heading_content_aware(heading)

    h = heading.lower()
    if any(kw in h for kw in ("example", "case study", "scenario")):
        return "example"
    if any(kw in h for kw in ("exercise", "activity", "practice", "application")):
        return "exercise"
    if any(kw in h for kw in ("summary", "recap", "key takeaway", "conclusion")):
        return "summary"
    if any(kw in h for kw in ("overview", "introduction", "welcome")):
        return "overview"
    if any(kw in h for kw in (
        "self-check", "self check", "knowledge check", "quiz", "check your"
    )):
        return "assessment_item"
    if any(kw in h for kw in ("discussion", "reflection")):
        return "exercise"
    return "explanation"


#: A bare course/notebook page-title heading carries no pedagogical
#: signal — its chunk type should fall through to the neutral default
#: rather than key off an incidental word. Matches forms like
#: ``"NVIDIA Course 07: Vector Stores"``, ``"Course 03: Guardrails"``,
#: and ``"Course Solutions 05: ..."``.
_COURSE_TITLE_RE = re.compile(
    r"^(?:nvidia\s+)?course\s+(?:solutions\s+)?\d+\s*:",
    re.IGNORECASE,
)


def _whole_word(keyword: str, text: str) -> bool:
    """Whole-word containment — ``\\bkeyword\\b`` on word boundaries.

    Prevents substring false-positives such as "DocumentSummaryBase"
    triggering ``summary`` or "classifier" triggering a keyword match.
    ``keyword`` is matched case-insensitively against ``text``.
    """

    return re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE) is not None


#: Whole-word pedagogical keywords that, if present, keep a
#: course-title heading from being demoted to the neutral default.
_PEDAGOGICAL_WORDS = (
    "exercise", "activity", "practice", "example", "scenario",
    "summary", "recap", "conclusion", "overview", "introduction",
    "welcome", "quiz", "discussion", "reflection",
)
#: Multi-word pedagogical phrases (matched as plain substrings).
_PEDAGOGICAL_PHRASES = (
    "case study", "key takeaway", "self-check", "self check",
    "knowledge check", "check your",
)


def _has_pedagogical_keyword(text: str) -> bool:
    """True when ``text`` carries any pedagogical keyword / phrase.

    Used to decide whether a course-title heading's post-colon
    remainder still warrants a non-default classification.
    """

    if any(_whole_word(kw, text) for kw in _PEDAGOGICAL_WORDS):
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _PEDAGOGICAL_PHRASES)


def _type_from_heading_content_aware(heading: str) -> str:
    """Whole-word / course-title-aware heading classifier.

    Behavioural deltas vs the legacy substring matcher:

    1. **Whole-word matching.** ``summary`` / ``exercise`` / ``example``
       / ``overview`` (and the other keywords) match only on
       ``\\bword\\b`` boundaries, so "DocumentSummaryBase" no longer
       trips ``summary`` and "Guardrails" no longer trips ``exercise``.
    2. **Bare course-title demotion.** A heading matching
       ``^(NVIDIA )?Course (Solutions )?NN:`` (or otherwise carrying no
       pedagogical keyword) returns the neutral ``"explanation"``
       default instead of a type keyed off an incidental word.
    3. **Precedence.** ``exercise`` BEFORE ``example`` BEFORE
       ``summary`` — an "[Exercise] ... Realistic Example" heading
       classifies as ``exercise``, not ``example``.

    Worked-example pages titled "Course Solutions NN" carry no
    ``exercise``/``example``/``summary`` whole word, so the
    course-title demotion routes them to ``explanation`` rather than
    the legacy substring-driven ``exercise``.
    """

    h = heading.lower()

    # A bare course/notebook page title ("NVIDIA Course 07: Vector
    # Stores", "Course Solutions 05: ...") is demoted to the neutral
    # default UNLESS it carries a genuine pedagogical keyword after the
    # colon (e.g. "Course 03: Summary"). Strip the title prefix and
    # only keep classifying the post-colon remainder.
    title_match = _COURSE_TITLE_RE.match(heading)
    if title_match:
        remainder = heading[title_match.end():]
        if not _has_pedagogical_keyword(remainder):
            return "explanation"

    # Precedence: exercise before example before summary. A heading
    # carrying both "[Exercise]" and "Example" is an exercise.
    if (
        any(_whole_word(kw, h) for kw in ("exercise", "activity", "practice"))
        or _whole_word("application", h)
    ):
        return "exercise"
    if (
        any(_whole_word(kw, h) for kw in ("example", "scenario"))
        or "case study" in h
    ):
        return "example"
    if (
        any(_whole_word(kw, h) for kw in ("summary", "recap", "conclusion"))
        or "key takeaway" in h
    ):
        return "summary"
    if any(_whole_word(kw, h) for kw in ("overview", "introduction", "welcome")):
        return "overview"
    if (
        any(_whole_word(kw, h) for kw in ("quiz",))
        or "self-check" in h
        or "self check" in h
        or "knowledge check" in h
        or "check your" in h
    ):
        return "assessment_item"
    if any(_whole_word(kw, h) for kw in ("discussion", "reflection")):
        return "exercise"

    # No pedagogical keyword survived whole-word matching — a bare
    # course/notebook page title (matched or not) falls through to the
    # neutral default rather than keying off an incidental word.
    return "explanation"


# ---------------------------------------------------------------------------
# Leading-text apparatus retype (FIX 2 + FIX 3) — apparatus-typing fidelity
#
# The apparatus-heading demotion (lib/semantik/heading_classifier +
# lib/semantik/adapter) moved per-section exercise BANNERS ("N.M Exercises",
# "Practice Makes Perfect") out of the heading stream and into leading prose.
# The chunker's chunk_type is heading-driven, so a demoted-banner exercise run
# now inherits the preceding pedagogical section's type (example/explanation)
# and real problem sets stop typing "exercise" (a 137->79 regression in the
# post-demotion census). These two deterministic, corpus-agnostic passes restore
# fidelity by inspecting a chunk's OWN leading window:
#
#   FIX 2 — banner-leading -> "exercise": a chunk whose leading window begins
#     with an ordinal-prefixed exercise/practice banner types as exercise.
#   FIX 3 — LO-boilerplate-leading vetoes "exercise": a chunk whose leading
#     window is section-intro boilerplate ("By the end of this section, ...") is
#     an INTRO, never an exercise, so an exercise type is demoted to "overview".
#
# Composition: the banner rule wins over the veto (an "In the following
# exercises" line is a banner even though the shared lexicon also lists it as
# LO-boilerplate). All VOCABULARY is data-driven from the shared lexicons
# (schemas/taxonomies/{semantik_lexicon,exercise_apparatus_lexicon}.json) via
# the canonical loaders — only the leading-window COMPOSITION is code here.
# Deterministic, no LLM, no new behavior flag (a correctness restore of the
# always-on demotion class, byte-identical for corpora without these banners —
# e.g. the RDF/SHACL calibration corpus). CHUNKER_SCHEMA_VERSION stays "v4":
# the chunk SHAPE is unchanged; only a per-chunk chunk_type VALUE is corrected.
# ---------------------------------------------------------------------------

#: Leading-window size for the apparatus retype: first 2 lines, or >= this many
#: chars, whichever covers more (the demoted banner lands at the head of the
#: chunk's prose, possibly a short pedagogical lead-in ahead of it).
_LEADING_WINDOW_CHARS: int = 200


def _leading_window(text: str) -> str:
    """Return the chunk's leading window (first ~2 lines / 200 chars)."""

    two_lines = "\n".join((text or "").splitlines()[:2])
    if len(two_lines) >= _LEADING_WINDOW_CHARS:
        return two_lines
    return (text or "")[:_LEADING_WINDOW_CHARS]


@lru_cache(maxsize=1)
def _leading_apparatus_predicates() -> Tuple[Tuple[Any, ...], Callable[[str], bool]]:
    """Compile the leading-text apparatus predicates from the shared lexicons.

    Returns ``(banner_regexes, leads_with_lo_boilerplate_fn)``:

    * ``banner_regexes`` — exercise/practice BANNER matchers applied to the
      leading window. Built from the numbered-apparatus names
      (``lib.ontology.taxonomy.get_lexicon_numbered_apparatus_names`` — the
      ordinal-prefixed "N.M Exercises" / "N.M Section Exercises" form, with a
      lowercase-continuation negative lookahead so a real "N.M Exercises in
      <topic>" title never matches) PLUS the exercise-banner phrase regexes
      already canonicalised in ``lib.objectives.apparatus_lexicon``
      (``EXERCISE_BANNER_RE`` / ``EXTRA_EXERCISE_BANNER_RE`` — "Practice Makes
      Perfect", "In the following exercises", "Section/Review Exercises").
    * ``leads_with_lo_boilerplate_fn`` — the shared
      ``compile_profile().leads_with_lo_boilerplate`` predicate (LO section-intro
      boilerplate), used for the FIX 3 veto.

    Vocabulary stays in the lexicon JSON; only the leading composition is code.
    Cached — the lexicons are static per process. Lazy imports mirror the
    existing ``_section_opener_substrings`` lib-import discipline (lib.* is
    already a chunker dependency); a load failure degrades to no-op predicates so
    the chunker never loses typing.
    """

    banner_res: List[Any] = []
    lo_fn: Callable[[str], bool] = lambda _t: False  # noqa: E731

    try:
        from lib.ontology.taxonomy import get_lexicon_numbered_apparatus_names

        names = [
            re.escape(n.strip()).replace(r"\ ", r"\s+")
            for n in get_lexicon_numbered_apparatus_names()
            if n and n.strip()
        ]
        if names:
            # A numbered banner at the start of ANY line in the window, NOT
            # followed by a lowercase continuation word (that would be a real
            # "N.M Exercises in Measure Theory" title, not a drill banner).
            banner_res.append(
                re.compile(
                    r"(?m)^\s*\d+(?:\.\d+)*\s+(?:%s)\b(?!\s+[a-z])"
                    % "|".join(names),
                    re.IGNORECASE,
                )
            )
    except Exception:  # pragma: no cover - defensive: never lose typing
        pass

    try:
        from lib.objectives.apparatus_lexicon import (
            EXERCISE_BANNER_RE,
            EXTRA_EXERCISE_BANNER_RE,
            compile_profile,
        )

        banner_res.append(EXERCISE_BANNER_RE)
        banner_res.append(EXTRA_EXERCISE_BANNER_RE)
        lo_fn = compile_profile().leads_with_lo_boilerplate
    except Exception:  # pragma: no cover - defensive: never lose typing
        pass

    return tuple(banner_res), lo_fn


def leads_with_exercise_banner(text: str) -> bool:
    """Whether ``text``'s leading window begins with an exercise/practice banner."""

    if not text:
        return False
    head = _leading_window(text)
    banner_res, _ = _leading_apparatus_predicates()
    return any(rgx.search(head) for rgx in banner_res)


def leads_with_lo_boilerplate(text: str) -> bool:
    """Whether ``text``'s leading window is LO section-intro boilerplate."""

    if not text:
        return False
    _, lo_fn = _leading_apparatus_predicates()
    return bool(lo_fn(_leading_window(text)))


def retype_from_leading_text(text: str, chunk_type: str) -> str:
    """Re-type a chunk from its LEADING text (deterministic, corpus-agnostic).

    * FIX 2 — a leading exercise/practice banner types the chunk ``"exercise"``.
    * FIX 3 — leading LO-intro boilerplate vetoes an ``"exercise"`` type down to
      ``"overview"`` (a section intro is never a problem set).

    Composition: the banner rule is evaluated FIRST, so an "In the following
    exercises" line (a banner that the shared lexicon ALSO lists as
    LO-boilerplate) types ``"exercise"`` rather than tripping the veto. Any other
    chunk_type is returned unchanged — the passes only fire on the two
    unambiguous leading shapes, so a corpus without these banners is
    byte-identical.
    """

    if not text:
        return chunk_type
    if leads_with_exercise_banner(text):
        return "exercise"
    if chunk_type == "exercise" and leads_with_lo_boilerplate(text):
        return "overview"
    return chunk_type


def split_by_sentences(text: str, target_words: int) -> List[str]:
    """Split ``text`` into sentence-grouped chunks of up to ``target_words``.

    Mirrors ``CourseProcessor._split_by_sentences`` at
    ``process_course.py:2984``. Pure / static.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_wc = 0

    for sentence in sentences:
        swc = len(sentence.split())
        if current_wc + swc > target_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_wc = swc
        else:
            current.append(sentence)
            current_wc += swc

    if current:
        chunks.append(" ".join(current))
    return chunks


def merge_section_source_ids(
    accumulated: List[str], section_source_ids: List[str]
) -> List[str]:
    """Union two sourceId-string lists, dedupe, preserve insertion order.

    Mirrors ``CourseProcessor._merge_section_source_ids`` at
    ``process_course.py:1572``. Mutates ``accumulated`` in place AND
    returns it (parity with the original — the original returns the
    mutated list so callers can chain).
    """

    seen = {sid for sid in accumulated}
    for sid in section_source_ids:
        if sid and sid not in seen:
            seen.add(sid)
            accumulated.append(sid)
    return accumulated


def _merge_objective_alignment_dedup(
    accumulated: List[Dict[str, Any]],
    section_objective_alignment: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union two ``objective_alignment`` lists, dedupe by ``objective_id``.

    Wave 5 (W5.F — chunker merge-path provenance preservation). Mirrors
    :func:`merge_section_source_ids` (`:260-276`) — first-seen-wins
    dedupe; mutates ``accumulated`` in place AND returns it for chaining.

    Per-objective entries carry tri-axis status (`status`,
    `declared_bloom`, `observed_bloom`, `statement_entailment_score`,
    `action_verb_present`) per W1.7 emit. Two adjacent merged sections
    that both teach the same objective_id can each declare an entry —
    we keep the FIRST declaration so the merged chunk's reading of "did
    this section deliver objective TO-05?" is anchored to the first
    block in document order, mirroring the chunk-buffer's anchor-heading
    semantics. Non-dict entries are silently dropped (defensive against
    malformed payloads that slipped past schema validation).
    """

    seen = {
        entry.get("objective_id")
        for entry in accumulated
        if isinstance(entry, dict) and entry.get("objective_id")
    }
    for entry in section_objective_alignment:
        if not isinstance(entry, dict):
            continue
        oid = entry.get("objective_id")
        if not oid or oid in seen:
            continue
        seen.add(oid)
        accumulated.append(entry)
    return accumulated


# ---------------------------------------------------------------------------
# MergedSectionResult — return shape from ``merge_small_sections``
# ---------------------------------------------------------------------------


@dataclass
class MergedSectionResult:
    """One merged-section result emitted by :func:`merge_small_sections`.

    Wave 5 (W5.F) bumped the legacy 5-tuple return to a dataclass so the
    new W1.5 ``key_claims`` and W1.7 ``objective_alignment`` audit fields
    can ride alongside the legacy ``merged_source_ids`` /
    ``merged_headings`` without forcing every caller to switch to a
    7-element tuple destructure.

    Back-compat contract: ``__iter__`` yields exactly the legacy
    5-element shape ``(heading, combined_text, chunk_type,
    merged_source_ids, merged_headings)``. Every existing destructure
    site (``chunk_content`` internal loop, ``test_merge_small_sections_*``
    test sites) keeps working byte-identical to pre-W5.F.

    New consumers access ``merged_key_claims`` /
    ``merged_objective_alignment`` via attribute. The two new fields
    default to ``[]`` (NOT ``None``) so a legacy ContentSection without
    the W1.5 / W1.7 fields produces a result whose new positions are
    safely empty lists — downstream stamp sites can no-op on
    ``if merged.merged_key_claims:`` without a ``None``-check guard.

    Indexing also routes through ``__iter__`` for back-compat —
    ``result[3]`` yields ``merged_source_ids`` (legacy index 3) so
    existing tests at ``test_merge_small_sections_source_refs.py:104``
    keep working.
    """

    heading: str
    combined_text: str
    chunk_type: str
    merged_source_ids: List[str] = field(default_factory=list)
    merged_headings: List[str] = field(default_factory=list)
    merged_key_claims: List[Dict[str, Any]] = field(default_factory=list)
    merged_objective_alignment: List[Dict[str, Any]] = field(default_factory=list)
    #: Fix 1 — True when this result's buffer opened on a SECTION-boundary
    #: section (a new textbook section). Consulted by
    #: :func:`fold_fragment_results` so a sub-floor boundary result is never
    #: folded BACKWARD into the prior section (which would re-fuse across the
    #: exact boundary the hard break just enforced). Default False → legacy
    #: fold behaviour is unchanged when the hard-break flag is off.
    section_boundary: bool = False
    #: Wave #22 quick-wins — the chunk-level pedagogical-unit type aggregated
    #: from every constituent section's ``data_dart_unit`` (plurality winner /
    #: ``None`` on tie via :func:`aggregate_composite_unit`). ``None`` when no
    #: constituent section carried a unit (legacy / non-DART). Attribute-access
    #: only (not in the legacy 5-tuple ``__iter__``).
    merged_composite_unit: Optional[str] = None
    #: Wave #22 quick-wins — distinct pedagogical flow/opener role slugs present
    #: across the constituent sections (union of ``data_dart_opener`` +
    #: ``data_dart_flows``), sorted for deterministic diffs. Empty on legacy /
    #: non-DART sections. Attribute-access only.
    merged_unit_roles: List[str] = field(default_factory=list)
    #: Build #23 Tier-3 — the chunk-level composite-unit SUBCLASS, resolved from
    #: the constituent sections' ``data_dart_subclass`` PAIRED with the resolved
    #: ``merged_composite_unit`` (:func:`aggregate_unit_subclass`). ``None`` when
    #: the unit doesn't resolve or carried no subclass. Attribute-access only.
    merged_unit_subclass: Optional[str] = None

    def __iter__(self):
        # Permit legacy 5-tuple destructure ergonomics:
        #   ``heading, text, chunk_type, source_ids, merged_headings = result``
        # New consumers reach the W5.F audit fields via attribute access
        # (``.merged_key_claims`` / ``.merged_objective_alignment``)
        # rather than positional unpacking, so the destructure arity
        # stays at 5 forever even if more audit fields land later.
        yield self.heading
        yield self.combined_text
        yield self.chunk_type
        yield self.merged_source_ids
        yield self.merged_headings

    def __len__(self) -> int:
        # Legacy ``len(result) == 5`` invariant — see ``__iter__`` docstring.
        return 5

    def __getitem__(self, index):
        # Back-compat positional access for tests that index by integer
        # (``result[3]`` for ``merged_source_ids``). Mirrors ``__iter__``
        # so the legacy 5-element shape is the canonical positional view.
        return tuple(self)[index]


# ---------------------------------------------------------------------------
# merge_small_sections — adjacent-section merger up to MAX_CHUNK_SIZE
# ---------------------------------------------------------------------------


def merge_small_sections(
    sections: List[Any],
    *,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    type_from_heading_fn: Optional[Callable[[str], str]] = None,
    fragment_floor: int = 0,
    section_hard_break: bool = False,
    subsection_break: bool = False,
    subsection_min_words: int = SUBSECTION_MIN_WORDS_DEFAULT,
) -> List[MergedSectionResult]:
    """Merge adjacent sections under MIN_CHUNK_SIZE into combined blocks.

    Mirrors ``CourseProcessor._merge_small_sections`` at
    ``process_course.py:1590`` byte-for-byte (the only refactor is
    parameterising ``self.MAX_CHUNK_SIZE`` → ``max_chunk_size`` and
    ``self._type_from_heading`` → ``type_from_heading_fn or type_from_heading``).

    Each ``section`` is expected to be a ``ContentSection``-like object
    with ``.heading``, ``.content``, ``.word_count``, and optionally
    ``.source_references``, ``.template_type``, ``.key_claims``, and
    ``.objective_alignment`` attributes. The chunker doesn't import
    ``ContentSection`` directly — it duck-types the attributes so the
    same orchestrator works against any section-like model (Trainforge
    / DART / future Courseforge).

    Returns a list of :class:`MergedSectionResult` instances. Each
    result tuple-unpacks to the legacy 5-element shape
    ``(heading, combined_text, chunk_type, merged_source_ids,
    merged_headings)`` for back-compat, AND exposes the new W5.F audit
    fields via attribute access:

    * ``merged_source_ids`` — union of every section's
      ``data-cf-source-ids`` attribute (Wave 10), deduped first-seen.
    * ``merged_headings`` — ordered list of every heading that
      collapsed into the buffer (Wave 84 Bug 1 fix).
    * ``merged_key_claims`` — concatenation of every section's
      ``key_claims`` array (W1.5 per-claim attribution; per-claim
      entries are independent, so concat is safe — see W5.F brief).
    * ``merged_objective_alignment`` — union of every section's
      ``objective_alignment`` array, deduped by ``objective_id`` with
      first-seen-wins (W1.7 per-objective tri-axis audit; mirrors the
      ``merge_section_source_ids`` first-seen policy).

    Without the W5.F merge-path carry, ~30-50% of chunks (every chunk
    built from a merge boundary on small-section corpora like
    the RDF/SHACL calibration corpus) would silently drop the W1.5 / W1.7 audit signals
    that the W5.A parser-side carry just landed on the section objects.
    """

    fn = type_from_heading_fn or type_from_heading

    merged: List[MergedSectionResult] = []
    buffer_heading = ""
    buffer_text = ""
    buffer_wc = 0
    buffer_type = "explanation"
    buffer_template_type: Optional[str] = None
    buffer_source_ids: List[str] = []
    buffer_headings: List[str] = []
    buffer_key_claims: List[Dict[str, Any]] = []
    buffer_objective_alignment: List[Dict[str, Any]] = []
    buffer_units: List[Optional[str]] = []
    buffer_subclasses: List[Optional[str]] = []
    buffer_roles: List[str] = []
    buffer_started = False
    buffer_boundary = False

    def _resolve_buffer_type() -> str:
        # Wave 81: prefer template_type when present and canonical.
        if buffer_template_type and buffer_template_type in CANONICAL_CHUNK_TYPES:
            return buffer_template_type
        return buffer_type

    def _flush() -> MergedSectionResult:
        return MergedSectionResult(
            heading=buffer_heading,
            combined_text=buffer_text,
            chunk_type=_resolve_buffer_type(),
            merged_source_ids=buffer_source_ids,
            merged_headings=list(buffer_headings),
            merged_key_claims=list(buffer_key_claims),
            merged_objective_alignment=list(buffer_objective_alignment),
            section_boundary=buffer_boundary,
            merged_composite_unit=aggregate_composite_unit(buffer_units),
            merged_unit_roles=sorted(set(buffer_roles)),
            merged_unit_subclass=aggregate_unit_subclass(
                buffer_units, buffer_subclasses
            ),
        )

    for section in sections:
        section_type = fn(section.heading)
        # Fix 1: when the hard-break flag is on, crossing into a new textbook
        # SECTION forces a flush before this section is accumulated, so a chunk
        # never straddles a section's closing apparatus and the next section's
        # opener. No-op when the flag is off → byte-identical legacy merge.
        force_break = (
            section_hard_break
            and buffer_started
            and _section_is_boundary(section)
        )
        # Fix 1b: the SIZE-GUARDED sub-section break. A genuine content
        # sub-section heading forces a flush ONLY once the buffer has reached
        # the ``subsection_min_words`` floor — so a break lands on a real
        # pedagogical boundary after enough content accumulated, never a runt
        # cascade. No-op when the flag is off → byte-identical legacy merge.
        force_break = force_break or (
            subsection_break
            and buffer_started
            and buffer_wc >= subsection_min_words
            and _is_subsection_break_candidate(section)
        )
        section_src = list(getattr(section, "source_references", []) or [])
        section_template = getattr(section, "template_type", None)
        # W5.F: harvest the W5.A per-section audit fields. ``getattr``
        # with default ``[]`` so a duck-typed section without the new
        # attrs (legacy DART-side ContentSection-likes, pre-W5.A
        # parsers) doesn't trip an AttributeError.
        section_key_claims = list(
            getattr(section, "key_claims", []) or []
        )
        section_objective_alignment = list(
            getattr(section, "objective_alignment", []) or []
        )
        # Wave #22 quick-wins: the section's pedagogical unit + flow/opener roles.
        section_unit, section_roles = section_unit_signals(section)
        # Build #23 Tier-3: the section's unit subclass (rides the unit).
        section_subclass = section_subclass_signal(section)

        if not buffer_started:
            buffer_heading = section.heading
            buffer_text = section.content
            buffer_wc = section.word_count
            buffer_type = section_type
            buffer_template_type = section_template
            buffer_source_ids = list(section_src)
            buffer_headings = [section.heading]
            buffer_key_claims = list(section_key_claims)
            buffer_objective_alignment = []
            _merge_objective_alignment_dedup(
                buffer_objective_alignment, section_objective_alignment
            )
            buffer_boundary = section_hard_break and _section_is_boundary(section)
            buffer_units = [section_unit]
            buffer_subclasses = [section_subclass]
            buffer_roles = list(section_roles)
            buffer_started = True
        elif not force_break and buffer_wc + section.word_count <= max_chunk_size:
            buffer_text += "\n\n" + section.content
            buffer_wc += section.word_count
            if buffer_type == "explanation" and section_type != "explanation":
                buffer_type = section_type
            if not buffer_template_type and section_template:
                buffer_template_type = section_template
            merge_section_source_ids(buffer_source_ids, section_src)
            buffer_headings.append(section.heading)
            # W5.F: per-claim entries are independent, so concat is
            # safe (no semantic collision between two adjacent blocks
            # that each declared their own claims). Per-objective
            # entries dedupe by ``objective_id`` with first-seen-wins
            # so the merged chunk's tri-axis status anchors to the
            # earliest declaration in document order.
            buffer_key_claims.extend(section_key_claims)
            _merge_objective_alignment_dedup(
                buffer_objective_alignment, section_objective_alignment
            )
            buffer_units.append(section_unit)
            buffer_subclasses.append(section_subclass)
            buffer_roles.extend(section_roles)
        else:
            merged.append(_flush())
            buffer_heading = section.heading
            buffer_text = section.content
            buffer_wc = section.word_count
            buffer_type = section_type
            buffer_template_type = section_template
            buffer_source_ids = list(section_src)
            buffer_headings = [section.heading]
            buffer_key_claims = list(section_key_claims)
            buffer_objective_alignment = []
            _merge_objective_alignment_dedup(
                buffer_objective_alignment, section_objective_alignment
            )
            buffer_units = [section_unit]
            buffer_subclasses = [section_subclass]
            buffer_roles = list(section_roles)
            buffer_boundary = section_hard_break and _section_is_boundary(section)

    if buffer_text.strip():
        merged.append(_flush())

    # W1b.3: fold sub-``fragment_floor``-word runt results into the prior
    # result so a 2-word fragment can't survive into the chunkset. No-op when
    # ``fragment_floor <= 0`` (default) → byte-identical.
    if fragment_floor > 0:
        merged = fold_fragment_results(merged, fragment_floor)

    return merged


# ---------------------------------------------------------------------------
# chunk_text_block — split one text block into one or more chunk dicts
# ---------------------------------------------------------------------------


def chunk_text_block(
    text: str,
    html: str,
    item: Dict[str, Any],
    heading: str,
    chunk_type: str,
    prefix: str,
    start_id: int,
    *,
    ctx: ChunkerContext,
    follows_chunk_id: Optional[str] = None,
    position_in_module: int = 0,
    section_source_ids: Optional[List[str]] = None,
    merged_headings: Optional[List[str]] = None,
    merged_key_claims: Optional[List[Dict[str, Any]]] = None,
    merged_objective_alignment: Optional[List[Dict[str, Any]]] = None,
    curie_anchors: Optional[List[str]] = None,
    forced_curie_anchors: Optional[List[str]] = None,
    dart_source_refs: Optional[List[Dict[str, Any]]] = None,
    dart_block_scope_html: Optional[str] = None,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    target_chunk_size: int = TARGET_CHUNK_SIZE,
    fragment_floor: int = 0,
    composite_unit: Optional[str] = None,
    unit_roles: Optional[List[str]] = None,
    unit_subclass: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Split a text block into one or more chunks.

    Mirrors ``CourseProcessor._chunk_text_block`` at
    ``process_course.py:1699``. Resolves the container xpath +
    plain-text once per call, then either emits a single chunk (when
    ``word_count <= max_chunk_size``) or splits by sentences via
    :func:`split_by_sentences` and emits one chunk per sub-text.

    Each chunk's audit-trail provenance (``html_xpath``, ``char_span``)
    is computed here and passed into the caller's
    ``ctx.create_chunk`` callback as keyword arguments. The callback
    is responsible for the rest of the chunk dict (concept tags,
    objective refs, bloom level, etc.) — see the module docstring for
    the architectural rationale.

    Wave 5 (W5.F): accepts ``merged_key_claims`` /
    ``merged_objective_alignment`` kwargs threaded from
    :func:`merge_small_sections` and forwards them to the
    ``ctx.create_chunk`` callback. The callback (W5.B's
    ``_create_chunk`` stamp site) reads these to materialize the
    ``chunk["key_claims"]`` / ``chunk["objective_alignment"]`` audit
    fields. When the chunk gets sentence-split into multiple sub-chunks
    (the ``word_count > max_chunk_size`` branch), every sub-chunk
    receives the same merged audit arrays — there's no per-sub-chunk
    attribution because the W1.5 / W1.7 emit is per-section, not
    per-sentence. Downstream consumers that want per-sentence
    attribution use the existing per-claim NLI fan-out path
    (W4.A / W5.D).

    CURIE-anchor threading: accepts ``curie_anchors`` /
    ``forced_curie_anchors`` — the ``data-cf-curie`` tokens harvested
    by :func:`Trainforge.chunker.helpers.extract_plain_text_with_curies`
    from the section / item HTML. They are forwarded to the
    ``ctx.create_chunk`` callback as the optional ``curie_anchors`` /
    ``forced_curie_anchors`` kwargs (additive, ``TypeError``-fallback —
    same contract as the W5.F merged-audit kwargs). The callback
    (``_create_chunk``) folds them with regex-extracted prose CURIEs
    into the chunk's ``curies`` / ``forced_curies`` fields so the
    ``curie_anchoring`` gate has an authoritative source-CURIE set.
    When a chunk is sentence-split, every sub-chunk receives the same
    section-level anchor lists (force-injection is per-section, not
    per-sentence).

    DART source-provenance threading: accepts ``dart_source_refs`` — the
    ``{block_id, pages}`` pairs harvested by
    :func:`Trainforge.chunker.helpers.harvest_semantik_source_refs` from the
    ``data-dart-block-id`` / ``data-dart-pages`` attributes on the
    section / item HTML. Forwarded to the ``ctx.create_chunk`` callback as
    the optional ``dart_source_refs`` kwarg (additive, ``TypeError``-fallback
    — same contract as the CURIE / merged-audit kwargs). The DART-side
    callback mints ``dart:{slug}#{block_id}`` sourceIds and folds them into
    the chunk's ``source.source_references[]``. HTML without ``data-dart-*``
    attributes yields an empty list and the kwarg is never passed, so
    non-DART corpora stay byte-identical. When a chunk is sentence-split,
    every sub-chunk receives the same section-level DART refs (block-level
    provenance is per-section, not per-sentence).

    ``dart_block_scope_html`` (optional): the DART-block-aligned HTML slice for
    THIS chunk's own synthesized section (see
    :func:`_section_block_scope_html`). When supplied, the per-chunk block-probe
    index is built over it instead of the whole item HTML, so a chunk can only
    be attributed blocks belonging to its own section — cross-section /
    cross-chapter provenance smear is eliminated by construction (no locality
    threshold). ``None`` (the unsectioned item path, or a section whose heading
    couldn't be located) falls back to the whole-item scope, byte-identical to
    the pre-scope behavior.
    """

    word_count = len(text.split())
    chunks: List[Dict[str, Any]] = []

    raw_html_for_xpath = item.get("raw_html", "") or html
    container_xpath: Optional[str] = None
    if heading and heading != item.get("title"):
        container_xpath = find_section_container_xpath(raw_html_for_xpath, heading)
    if not container_xpath:
        container_xpath = find_body_xpath(raw_html_for_xpath)

    container_text = resolve_xpath(raw_html_for_xpath, container_xpath) or ""

    def _locate(needle: str, search_from: int = 0) -> List[int]:
        """Return [start, end] of ``needle`` in the container text."""

        if container_text and needle:
            idx = container_text.find(needle, search_from)
            if idx >= 0:
                return [idx, idx + len(needle)]
            collapsed_container = " ".join(container_text.split())
            collapsed_needle = " ".join(needle.split())
            prefix_str = " ".join(collapsed_needle.split()[:8])
            if prefix_str:
                idx = collapsed_container.find(prefix_str, search_from)
                if idx >= 0:
                    return [idx, idx + len(collapsed_needle)]
        return [search_from, search_from + len(needle)]

    # Worker N (REC-ID-01): stable per-source locator for content-hash IDs.
    source_locator = item.get("item_path") or f"{item['module_id']}/{item['item_id']}"

    # W5.F: build the create_chunk kwargs once. The callback contract
    # adds two new kwargs (``merged_key_claims`` /
    # ``merged_objective_alignment``); but a legacy callback (e.g. a
    # downstream consumer that wraps ``CourseProcessor._create_chunk``
    # before the W5.B stamp site lands) won't accept them. We pass the
    # extras only when at least one is non-empty, AND fall back to the
    # legacy kwarg set on TypeError so the callback contract stays
    # additive — pre-W5.B callbacks keep working byte-identical.
    base_kwargs: Dict[str, Any] = {
        "section_source_ids": section_source_ids,
        "merged_headings": merged_headings,
    }
    extra_kwargs: Dict[str, Any] = {}
    if merged_key_claims:
        extra_kwargs["merged_key_claims"] = merged_key_claims
    if merged_objective_alignment:
        extra_kwargs["merged_objective_alignment"] = merged_objective_alignment
    if curie_anchors:
        extra_kwargs["curie_anchors"] = curie_anchors
    if forced_curie_anchors:
        extra_kwargs["forced_curie_anchors"] = forced_curie_anchors
    # Wave #22 quick-wins: additive pedagogical metadata. When a chunk is
    # sentence-split into sub-chunks, every sub-chunk inherits the same
    # section-level unit/roles (per-section, not per-sentence — mirrors the
    # merged key_claims / objective_alignment carry).
    if composite_unit:
        extra_kwargs["composite_unit"] = composite_unit
    if unit_roles:
        extra_kwargs["unit_roles"] = list(unit_roles)
    # Build #23 Tier-3: the unit subclass rides the unit → carry it alongside.
    if unit_subclass:
        extra_kwargs["unit_subclass"] = unit_subclass

    # DART source-provenance: resolve the block refs that genuinely overlap
    # THIS chunk's char span, in document order. Historically the chunker
    # passed the whole section's (and, via a no-match fallback, the whole
    # DOCUMENT's) block list to every chunk verbatim, so the first ref — the
    # one the "View original source" deep link + cited page derive from — was
    # identical across every chunk of a page. We now build a char-offset index
    # of every DART block against this call's ``container_text`` once, then
    # per-chunk select only the blocks inside the chunk's char span.
    #
    # ``dart_source_refs`` (the caller's section-scoped harvest) is retained
    # ONLY as the bounded no-match fallback: when text-containment resolution
    # finds no block inside a chunk's text, the chunk gets AT MOST ONE
    # enclosing-section reference (the first section block) — never the
    # whole-document list.
    #
    # The probe index is built over the SECTION's own DART block range
    # (``dart_block_scope_html``, computed by ``_section_block_scope_html`` at
    # the caller from the section's heading→next-heading boundary) rather than
    # the whole document, so a chunk can only ever be attributed blocks that
    # belong to its own synthesized section — cross-section / cross-chapter
    # smear (a Section-1.1 chunk citing a distant chapter's "Solution" block) is
    # eliminated BY CONSTRUCTION, with no locality distance threshold. When the
    # caller couldn't cleanly bound the section (``dart_block_scope_html`` is
    # None — the unsectioned item path, or a heading we couldn't locate) we fall
    # back to the whole-item HTML so provenance is never LOST, only narrowed;
    # matching stays chunk-text containment so a merged multi-section chunk is
    # attributed every in-scope block whose prose it contains, in doc order.
    _probe_scope_html = dart_block_scope_html or raw_html_for_xpath
    block_probe_index = (
        build_semantik_block_offset_index(_probe_scope_html)
        if dart_source_refs
        else []
    )

    def _refs_for_chunk(chunk_text: str) -> List[Dict[str, Any]]:
        """Per-chunk DART refs: text-containment match, ≤1 section fallback."""

        if not dart_source_refs:
            return []
        precise = resolve_dart_refs_for_chunk(block_probe_index, chunk_text)
        if precise:
            return precise
        # No block's prose appears in the chunk text. Emit at most ONE
        # enclosing-section reference (the first section-scoped block in
        # document order) so the chunk still carries a coarse, correct anchor —
        # never the whole document's list.
        return [dart_source_refs[0]]

    def _dispatch_create_chunk(**call_kwargs: Any) -> Dict[str, Any]:
        """Invoke the create_chunk callback with W5.F extras when accepted.

        Tries the full kwarg set first (post-W5.B callback). Falls back
        to the legacy kwarg set on ``TypeError: unexpected keyword
        argument`` so a pre-W5.B caller (e.g. a test that bound a
        custom callback before this brief landed) keeps working without
        modification. The fallback is silent — the audit fields are
        simply not stamped on the chunk in that path, which matches
        the pre-W5.F baseline behavior.
        """

        # Per-chunk DART refs ride alongside the section-level W5.F extras as
        # additive (strippable-on-TypeError) kwargs. ``call_kwargs`` may carry
        # a per-chunk ``dart_source_refs`` resolved by ``_refs_for_chunk``.
        per_chunk_extra: Dict[str, Any] = dict(extra_kwargs)
        chunk_refs = call_kwargs.pop("dart_source_refs", None)
        if chunk_refs:
            per_chunk_extra["dart_source_refs"] = chunk_refs

        merged_kwargs = {**call_kwargs, **base_kwargs, **per_chunk_extra}
        try:
            return ctx.create_chunk(**merged_kwargs)
        except TypeError as exc:
            # Defensive back-compat: only swallow the specific
            # "unexpected keyword argument" error that maps to a legacy
            # callback signature. Re-raise on other TypeErrors so a
            # genuine callback bug isn't masked.
            msg = str(exc)
            if not per_chunk_extra or "unexpected keyword argument" not in msg:
                raise
            legacy_kwargs = {**call_kwargs, **base_kwargs}
            return ctx.create_chunk(**legacy_kwargs)

    if word_count <= max_chunk_size:
        char_span = _locate(text, search_from=0)
        chunks.append(_dispatch_create_chunk(
            chunk_id=_generate_chunk_id(prefix, start_id, text, source_locator),
            text=text,
            html=html,
            item=item,
            section_heading=heading,
            # FIX 2/3: correct the heading-driven type from this chunk's own
            # leading text (banner-leading -> exercise; LO-boilerplate-leading
            # vetoes exercise). No-op for prose without those leading shapes.
            chunk_type=retype_from_leading_text(text, chunk_type),
            follows_chunk_id=follows_chunk_id,
            position_in_module=position_in_module,
            html_xpath=container_xpath,
            char_span=char_span,
            dart_source_refs=_refs_for_chunk(text),
        ))
    else:
        # W1b.2: an oversized code listing has no sentence punctuation, so
        # sentence-splitting leaves ONE over-window piece. When the flag is on
        # and the body reads as code, split on line boundaries instead.
        if resolve_chunk_code_split() and looks_like_code_listing(text):
            sub_texts = split_code_by_lines(text, target_chunk_size)
        else:
            sub_texts = split_by_sentences(text, target_chunk_size)
        # W1b.3b (Fix 3): the section-level fragment fold ran BEFORE this split.
        # A sentence/code split can flush a tiny trailing piece (an orphaned
        # question stem, a running-header residue line) that the section-level
        # fold never saw. Fold those sub-``fragment_floor``-word pieces into an
        # adjacent piece at the TEXT level so no split-produced runt is emitted.
        # No-op when ``fragment_floor <= 0`` → byte-identical legacy emit.
        sub_texts = fold_fragment_texts(sub_texts, fragment_floor)
        prev_end = 0
        last_chunk_id = follows_chunk_id
        for i, sub_text in enumerate(sub_texts):
            part_heading = (
                f"{heading} (part {i + 1})" if len(sub_texts) > 1 else heading
            )
            prev_id = last_chunk_id
            this_chunk_id = _generate_chunk_id(
                prefix, start_id + i, sub_text, source_locator
            )
            char_span = _locate(sub_text, search_from=prev_end)
            if char_span[0] < prev_end:
                char_span = [prev_end, prev_end + (char_span[1] - char_span[0])]
            prev_end = char_span[1]
            chunks.append(_dispatch_create_chunk(
                chunk_id=this_chunk_id,
                text=sub_text,
                html="" if i > 0 else html,
                item=item,
                section_heading=part_heading,
                # FIX 2/3: per-part leading-text retype (each split piece is
                # judged on its own leading window — the banner leads part 1).
                chunk_type=retype_from_leading_text(sub_text, chunk_type),
                follows_chunk_id=prev_id,
                position_in_module=position_in_module + i,
                html_xpath=container_xpath,
                char_span=char_span,
                dart_source_refs=_refs_for_chunk(sub_text),
            ))
            last_chunk_id = this_chunk_id

    return chunks


def _enclosing_section_dart_ref(
    raw_html: str, heading: str
) -> Optional[Dict[str, Any]]:
    """Return the single DART block whose span encloses ``heading``, or None.

    Locates the ``heading`` element in ``raw_html`` and returns the
    ``{block_id, pages}`` of the nearest DART block whose opening tag precedes
    the heading and whose span (open tag → next block's open tag, blocks being
    flat document-order siblings) covers the heading position. Used as the
    bounded ≤1-ref no-match fallback so a chunk built from a section whose own
    HTML slice carried no ``data-dart-block-id`` still anchors to its enclosing
    section block — NOT the whole document's block list.

    Returns ``None`` when ``heading`` isn't found, the document carries no DART
    blocks, or no block precedes the heading.
    """
    # DART->semantik purge Stage 3 (dual-READ): accept either attr spelling.
    if not raw_html or not heading or (
        "data-dart-block-id" not in raw_html
        and "data-semantik-block-id" not in raw_html
    ):
        return None
    h_re = re.compile(
        r"<h([1-6])\b[^>]*>\s*" + re.escape(heading) + r"\s*</h\1>",
        re.DOTALL | re.IGNORECASE,
    )
    h_match = h_re.search(raw_html)
    if not h_match:
        return None
    h_pos = h_match.start()
    spans = _block_element_spans(raw_html)
    enclosing: Optional[Dict[str, Any]] = None
    for html_start, html_next, ref in spans:
        if html_start <= h_pos < html_next:
            enclosing = ref
            break
        if html_start > h_pos:
            break
        # Track the nearest preceding block in case the heading sits between
        # block boundaries (whitespace gap) rather than strictly inside one.
        enclosing = ref
    if enclosing is None:
        return None
    return {"block_id": enclosing.get("block_id"), "pages": list(enclosing.get("pages") or [])}


def _heading_char_pos(raw_html: str, heading: str) -> Optional[int]:
    """Return the char offset of ``heading``'s ``<hN>`` element in ``raw_html``.

    Uses the same tolerant heading regex as :func:`_enclosing_section_dart_ref`
    / :func:`extract_section_html`. Returns ``None`` when the heading isn't
    found (so the caller can degrade to whole-document scope rather than
    fabricate a boundary).
    """
    if not raw_html or not heading:
        return None
    h_re = re.compile(
        r"<h([1-6])\b[^>]*>\s*" + re.escape(heading) + r"\s*</h\1>",
        re.DOTALL | re.IGNORECASE,
    )
    m = h_re.search(raw_html)
    return m.start() if m else None


def _section_block_scope_html(
    raw_html: str, heading: str, next_heading_pos: Optional[int]
) -> Optional[str]:
    """Return the DART-block-aligned HTML slice for ONE synthesized section.

    THRESHOLD-FREE per-section provenance scope. DART emits its blocks as flat
    document-order ``<section data-dart-block-id=...>`` siblings; a synthesized
    section owns every block from its own heading up to (but excluding) the
    NEXT section's heading. This returns ``raw_html`` sliced to exactly that
    block range so the caller's block-probe index carries ONLY the section's
    own blocks — cross-section (and cross-chapter) attribution is eliminated by
    construction, NOT by any locality distance heuristic.

    Boundaries are snapped to DART block openings so the slice always begins at
    a complete ``<section>`` opening tag (a mid-block slice would drop the block
    from the probe index): ``scope_start`` is the opening of the block enclosing
    /nearest-preceding ``heading``; ``scope_end`` is the opening of the block
    enclosing ``next_heading_pos`` (that block belongs to the NEXT section, so it
    is excluded). When the next heading is not wrapped in a block, the cut falls
    at the heading position itself. The last section (``next_heading_pos`` None)
    runs to the end of the document — it owns its own trailing blocks, never
    another section's.

    Returns ``None`` (=> caller keeps the whole-document scope, byte-identical
    legacy behavior) when the heading can't be located, the document carries no
    DART blocks, or the computed range is degenerate — so a section we can't
    cleanly bound never LOSES provenance, it just isn't narrowed.
    """
    h_pos = _heading_char_pos(raw_html, heading)
    if h_pos is None:
        return None
    spans = _block_element_spans(raw_html)
    if not spans:
        return None

    # scope_start: opening of the block enclosing / nearest-preceding the
    # heading (the latest block whose open tag is at-or-before the heading).
    scope_start = 0
    for html_start, _html_next, _ref in spans:
        if html_start <= h_pos:
            scope_start = html_start
        else:
            break

    # scope_end: default to end-of-doc (last section owns its tail). When there
    # IS a next section, cut at the opening of the block that wraps its heading
    # so that block stays with the NEXT section; if the next heading isn't
    # block-wrapped, cut at the heading position itself.
    scope_end = len(raw_html)
    if next_heading_pos is not None:
        scope_end = next_heading_pos
        for html_start, html_next, _ref in spans:
            if html_start <= next_heading_pos < html_next:
                scope_end = html_start
                break

    if scope_end <= scope_start:
        return None
    return raw_html[scope_start:scope_end]


def _generate_chunk_id(
    prefix: str, start_id: int, text: str, source_locator: str
) -> str:
    """Generate a chunk ID — package-local mirror of the Trainforge helper.

    Mirrors ``Trainforge/process_course.py::_generate_chunk_id`` at
    ``:156``. Default position-based; opt-in content-hash mode via
    ``TRAINFORGE_CONTENT_HASH_IDS=true``. The env-var name is preserved
    for backward compatibility with already-ingested LibV2 corpora.
    """

    import hashlib
    import os

    if os.getenv("TRAINFORGE_CONTENT_HASH_IDS", "").lower() == "true":
        # Schema version is fixed to "v4" here — matches the only
        # value Trainforge has ever shipped (CHUNK_SCHEMA_VERSION at
        # process_course.py:92). When the schema bumps, both this
        # helper and the Trainforge-side constant move in lockstep.
        payload = f"{text}|{source_locator}|v4"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}{digest}"
    return f"{prefix}{start_id:05d}"


# ---------------------------------------------------------------------------
# Within-package exact-normalised dedup (ED4ALL_CHUNK_DEDUP, default off)
# ---------------------------------------------------------------------------

#: Separator joining a unit's heading to its text before hashing. Chosen
#: because ``normalize_exact`` collapses only ``\s`` runs and NUL is not
#: whitespace, so the boundary survives normalisation — ``("A", "B C")`` and
#: ``("A B", "C")`` therefore hash differently instead of colliding.
DEDUP_UNIT_KEY_SEPARATOR: str = "\x00"


def _dedup_unit_key(heading: str, text: str, min_tokens: int) -> Optional[str]:
    """Exact-normalised dedup key for one emit unit, or ``None`` if ineligible.

    A "unit" is one ``chunk_text_block`` call site — an unsectioned item's whole
    text, or one merged section group. Hashing the unit rather than the emitted
    chunk is what lets the dedup run BEFORE ``_generate_chunk_id`` mints
    anything, which is a hard correctness constraint: chunk IDs are the join key
    for the coverage map, the KG-quality coverage floor, assessment
    ``source_chunk_ids``, training pairs, and the concept graph, and a post-mint
    drop would leave a non-contiguous positional sequence and orphan
    ``follows_chunk`` (silently disabling :func:`apply_chunk_overlap`, which only
    bleeds when ``cur["follows_chunk"] == prev["id"]``).

    The key covers the HEADING as well as the text — deliberately MORE
    conservative than text alone. Two sections whose bodies happen to match
    verbatim but whose headings differ carry different section provenance, and
    collapsing them would silently discard one heading. The target of this pass
    is verbatim repeated material (shared chrome, a licence banner, a
    prerequisites block reproduced on every page), which repeats heading and
    body together.

    ``None`` (never a duplicate) is returned when the unit falls under
    ``min_tokens`` exact-normalised tokens — counted on the TEXT alone, so a
    long heading can never push a two-word body over the floor — or when the
    unit normalises to nothing at all.
    """
    if exact_token_count(text) < min_tokens:
        return None
    digest = exact_content_hash(
        DEDUP_UNIT_KEY_SEPARATOR.join((heading or "", text or ""))
    )
    return digest or None


def _dedup_drop_record(
    *,
    dropped_index: int,
    kept: Tuple[int, str],
    normalized_hash: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one reversible ledger row for a skipped duplicate unit.

    ``dropped_index`` is the 0-based ordinal of the DROPPED unit within
    ``chunk_content``'s source-ordered unit sequence (unique per ledger).
    ``kept_chunk_index`` is the 0-based index into the emitted chunk list of the
    FIRST chunk of the surviving first occurrence, and ``kept_chunk_id`` is that
    chunk's id — so a consumer can resolve every drop back to a row that is
    genuinely present in ``chunks.jsonl``. ``source_item_path`` is the dropped
    unit's source locator, minted by the same expression ``chunk_text_block``
    uses for ``source_locator``.
    """
    kept_index, kept_id = kept
    return {
        "dropped_index": dropped_index,
        "kept_chunk_index": kept_index,
        "kept_chunk_id": kept_id,
        "normalized_hash": normalized_hash,
        "source_item_path": (
            item.get("item_path") or f"{item['module_id']}/{item['item_id']}"
        ),
    }


# ---------------------------------------------------------------------------
# chunk_content — top-level loop over parsed IMSCC items
# ---------------------------------------------------------------------------


@dataclass
class ChunkContentResult:
    """Return container for ``chunk_content``.

    Carries the chunk list AND the side-channel ``pages_with_misconceptions``
    set (was ``CourseProcessor._pages_with_misconceptions`` at
    ``process_course.py:1476``). The Subtask 6 wrapper writes the set
    back to ``self._pages_with_misconceptions`` so downstream
    quality-report metrics (``misconceptions_present_rate``) keep their
    correct denominator.

    ``dedup_drops`` is the within-package dedup ledger (see
    :func:`_dedup_drop_record`) — one row per unit skipped as an
    exact-normalised repeat. It is ALWAYS empty unless ``ED4ALL_CHUNK_DEDUP``
    is on. The chunker never writes files; persisting the ledger (and
    registering its count as a named ``source_coverage`` drop reason) is the
    caller's job.
    """

    chunks: List[Dict[str, Any]] = field(default_factory=list)
    pages_with_misconceptions: set = field(default_factory=set)
    dedup_drops: List[Dict[str, Any]] = field(default_factory=list)

    def __iter__(self):
        # Permit tuple-unpacking ergonomics:
        # ``chunks, pages_with_misconceptions = chunk_content(...)``
        # DELIBERATELY still two yields — ``dedup_drops`` is read by attribute
        # so every existing two-name unpack keeps working unchanged.
        yield self.chunks
        yield self.pages_with_misconceptions


def chunk_content(
    parsed_items: List[Dict[str, Any]],
    course_code: str,
    boilerplate_spans: Optional[List[str]] = None,
    *,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    target_chunk_size: int = TARGET_CHUNK_SIZE,
    ctx: Optional[ChunkerContext] = None,
) -> ChunkContentResult:
    """Chunk parsed IMSCC items into a list of chunk dicts.

    Mirrors ``CourseProcessor._chunk_content`` at
    ``process_course.py:1462``. Top-level loop over ``parsed_items``;
    handles boilerplate stripping (via
    :func:`Trainforge.chunker.boilerplate.strip_boilerplate`),
    assessment-feedback stripping (via
    :func:`Trainforge.chunker.helpers.strip_assessment_feedback` /
    :func:`strip_feedback_from_text`), per-item section iteration via
    :func:`merge_small_sections`, and follows-chunk linkage at
    lesson/module boundaries.

    Per-chunk materialisation is delegated to ``ctx.create_chunk`` —
    see the module docstring for the architectural rationale. The
    ``min_chunk_size`` parameter is currently unused inside this
    function (the merger uses the package-level ``MIN_CHUNK_SIZE`` via
    section ``word_count`` thresholds inside ``ContentSection`` — the
    parameter is in the signature so a future cleanup can plumb it
    through to a parameterised section parser).

    Returns a ``ChunkContentResult`` carrying the chunk list and the
    ``pages_with_misconceptions`` side channel. Tuple-unpacks for
    ergonomic call sites:

        chunks, pages_with_misconceptions = chunk_content(...)

    Empty-input contract: ``chunk_content([], 'TEST_101')`` returns an
    empty result without requiring ``ctx`` (the loop never reaches the
    ``create_chunk`` call site). Non-empty input requires
    ``ctx is not None`` — mismatch raises ``ChunkerContextRequired``.

    Within-package dedup (``ED4ALL_CHUNK_DEDUP``, default OFF): when on, an
    emit unit whose heading-scoped text exact-normalises to one already emitted
    is SKIPPED, and a reversible row lands on
    ``ChunkContentResult.dedup_drops``. The first source-ordered occurrence is
    always kept. Because the skip happens before ``chunk_text_block`` is called,
    ``chunk_counter`` / ``prev_chunk_id`` / ``position_in_module`` never advance
    for a dropped unit, so chunk IDs stay a dense 1-based sequence and every
    surviving chunk's ``follows_chunk`` still resolves. OFF is byte-identical to
    the pre-dedup emit.
    """

    if parsed_items and ctx is None:
        raise ChunkerContextRequired(
            "chunk_content received non-empty parsed_items but no "
            "ChunkerContext; the chunker delegates per-chunk "
            "materialisation back to the caller (see "
            "Trainforge.chunker.chunker module docstring)."
        )

    boilerplate = boilerplate_spans or []

    # W1b.3: resolve the sub-N-word fragment floor once per invocation and
    # thread it into every merge_small_sections call. 0 (default) → no-op.
    _fragment_floor = resolve_merge_fragment_floor()
    # Fix 1: resolve the section-boundary hard-break gate once. Off (default) →
    # byte-identical legacy merge.
    _section_hard_break = resolve_chunk_section_hard_break()
    # Fix 1b: resolve the size-guarded sub-section break + its floor once. Off
    # (default) → byte-identical legacy merge.
    _subsection_break = resolve_chunk_subsection_break()
    _subsection_min_words = resolve_chunk_subsection_min_words()
    # W6: resolve the within-package exact-normalised dedup gate + its token
    # floor once. Off (default) → no hashing, empty ledger, byte-identical emit.
    _dedup_enabled = resolve_chunk_dedup_enabled()
    _dedup_min_tokens = resolve_chunk_dedup_min_tokens() if _dedup_enabled else 0
    # Exact-normalised unit key → (index of the kept unit's first chunk in
    # ``chunks``, that chunk's id). Registered only AFTER a unit actually
    # emitted at least one chunk, so a ledger row can never point at a first
    # occurrence that produced nothing.
    _dedup_seen: Dict[str, Tuple[int, str]] = {}
    _dedup_unit_ordinal = 0
    dedup_drops: List[Dict[str, Any]] = []

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1
    prefix = f"{course_code.lower()}_chunk_"
    prev_chunk_id: Optional[str] = None
    current_module_id: Optional[str] = None
    current_lesson_id: Optional[str] = None
    position_in_module = 0

    # Wave-era denominator for misconceptions_present_rate: pages whose
    # parsed JSON-LD declared at least one misconception.
    pages_with_misconceptions = {
        item["item_id"]
        for item in parsed_items
        if item.get("misconceptions")
    }

    for item in parsed_items:
        if item["module_id"] != current_module_id:
            current_module_id = item["module_id"]
            position_in_module = 0

        if item["item_id"] != current_lesson_id:
            current_lesson_id = item["item_id"]
            prev_chunk_id = None

        raw_html = item["raw_html"]
        if item["resource_type"] == "quiz":
            raw_html = strip_assessment_feedback(raw_html)

        if boilerplate:
            raw_html, _ = strip_boilerplate(raw_html, boilerplate)

        if not item["sections"]:
            # Harvest data-cf-curie tokens alongside the plain-text
            # projection so force-injected CURIE anchors survive into
            # the emitted chunk's ``curies`` / ``forced_curies`` fields.
            text, item_curies, item_forced_curies = (
                extract_plain_text_with_curies(raw_html)
            )
            # Harvest DART source-provenance pairs from the whole item HTML
            # (the unsectioned path has no per-section block to scope to).
            item_dart_refs = harvest_semantik_source_refs(raw_html)
            if item["resource_type"] == "quiz":
                text = strip_feedback_from_text(text)
            if text.strip():
                _unit_ordinal = _dedup_unit_ordinal
                _dedup_unit_ordinal += 1
                _unit_key = (
                    _dedup_unit_key(item["title"], text, _dedup_min_tokens)
                    if _dedup_enabled
                    else None
                )
                _prior = _dedup_seen.get(_unit_key) if _unit_key else None
                if _prior is not None:
                    dedup_drops.append(
                        _dedup_drop_record(
                            dropped_index=_unit_ordinal,
                            kept=_prior,
                            normalized_hash=_unit_key,
                            item=item,
                        )
                    )
                    continue
                _unit_first_index = len(chunks)
                item_chunks = chunk_text_block(
                    text=text,
                    html=raw_html,
                    item=item,
                    heading=item["title"],
                    chunk_type=type_from_resource(item["resource_type"]),
                    prefix=prefix,
                    start_id=chunk_counter,
                    follows_chunk_id=prev_chunk_id,
                    position_in_module=position_in_module,
                    curie_anchors=item_curies,
                    forced_curie_anchors=item_forced_curies,
                    dart_source_refs=item_dart_refs,
                    ctx=ctx,
                    max_chunk_size=max_chunk_size,
                    target_chunk_size=target_chunk_size,
                    fragment_floor=_fragment_floor,
                )
                chunks.extend(item_chunks)
                chunk_counter += len(item_chunks)
                if item_chunks:
                    prev_chunk_id = item_chunks[-1]["id"]
                    position_in_module += len(item_chunks)
                    if _unit_key:
                        _dedup_seen[_unit_key] = (
                            _unit_first_index,
                            str(item_chunks[0].get("id") or ""),
                        )
            continue

        merged = merge_small_sections(
            item["sections"],
            max_chunk_size=max_chunk_size,
            type_from_heading_fn=ctx.type_from_heading_fn if ctx else None,
            fragment_floor=_fragment_floor,
            section_hard_break=_section_hard_break,
            subsection_break=_subsection_break,
            subsection_min_words=_subsection_min_words,
        )

        # THRESHOLD-FREE per-section DART-provenance scope: each merged group's
        # primary heading marks a section boundary in ``raw_html``. Precompute
        # every group's primary-heading char position (in document order) so a
        # section's block-probe scope can be bounded by the NEXT group's heading
        # — a structural section boundary, never a distance heuristic. A heading
        # we can't locate contributes ``None`` and is skipped as a boundary
        # (that section degrades to whole-document scope, never LOSING refs).
        _group_head_positions = [
            _heading_char_pos(raw_html, mr[0]) for mr in merged
        ]

        for _group_idx, merged_result in enumerate(merged):
            # W5.F: legacy 5-element destructure preserved via
            # ``MergedSectionResult.__iter__``; new W1.5 / W1.7 audit
            # fields read off the dataclass via attribute access. Keeps
            # this internal loop byte-compatible with pre-W5.F while
            # threading the new fields through to ``chunk_text_block``.
            heading, text, chunk_type, section_source_ids, merged_headings = (
                merged_result
            )
            merged_key_claims = merged_result.merged_key_claims
            merged_objective_alignment = merged_result.merged_objective_alignment
            merged_composite_unit = merged_result.merged_composite_unit
            merged_unit_roles = merged_result.merged_unit_roles
            merged_unit_subclass = merged_result.merged_unit_subclass
            if not text.strip():
                continue
            if item["resource_type"] == "quiz":
                text = strip_feedback_from_text(text)
            if boilerplate:
                text, _ = strip_boilerplate(text, boilerplate)
            if not text.strip():
                continue
            # W6: dedup BEFORE the per-section provenance harvest below — the
            # text is final at this point, and a skipped unit should not pay
            # for CURIE / source-ref extraction it will never emit.
            _unit_ordinal = _dedup_unit_ordinal
            _dedup_unit_ordinal += 1
            _unit_key = (
                _dedup_unit_key(heading, text, _dedup_min_tokens)
                if _dedup_enabled
                else None
            )
            _prior = _dedup_seen.get(_unit_key) if _unit_key else None
            if _prior is not None:
                dedup_drops.append(
                    _dedup_drop_record(
                        dropped_index=_unit_ordinal,
                        kept=_prior,
                        normalized_hash=_unit_key,
                        item=item,
                    )
                )
                continue
            html_block = extract_section_html(raw_html, heading)
            # Harvest data-cf-curie tokens from this section's HTML.
            # When merge_small_sections collapsed several sections into
            # this chunk, each merged heading's section HTML is its own
            # extracted block — union the harvested anchor lists across
            # all of them so the chunk's curies set is complete.
            _, sec_curies, sec_forced_curies = (
                extract_plain_text_with_curies(html_block)
            )
            section_curies: List[str] = list(sec_curies)
            section_forced_curies: List[str] = list(sec_forced_curies)
            # Harvest DART {block_id, pages} from this section's HTML block.
            section_dart_refs: List[Dict[str, Any]] = list(
                harvest_semantik_source_refs(html_block)
            )
            for extra_heading in merged_headings or []:
                if extra_heading == heading:
                    continue
                extra_html = extract_section_html(raw_html, extra_heading)
                if not extra_html:
                    continue
                _, extra_curies, extra_forced = (
                    extract_plain_text_with_curies(extra_html)
                )
                section_curies.extend(extra_curies)
                section_forced_curies.extend(extra_forced)
                section_dart_refs.extend(harvest_semantik_source_refs(extra_html))
            # DART stamps data-dart-* on the block wrapper, not on leaf
            # headings, so a heading-derived ``html_block`` slice often carries
            # no attribute. We pass ``section_dart_refs`` to ``chunk_text_block``
            # as (a) the DART-provenance ENABLING signal and (b) the bounded
            # ≤1-ref no-match fallback; the precise per-chunk attribution is
            # resolved INSIDE ``chunk_text_block`` by char-span containment over
            # a document-wide block offset index, so the whole-document block
            # list never reaches a chunk verbatim.
            #
            # When the section slice carried no block, fall back to the SINGLE
            # block enclosing this section's heading in document order (the
            # nearest preceding block whose span covers the heading position) —
            # NOT the whole document's list. ``harvest_semantik_source_refs(raw_html)``
            # below is the document-presence probe; we then narrow it to the one
            # enclosing block so ``dart_source_refs[0]`` is a correct coarse
            # anchor for this section, not the file's first (colophon) block.
            if not section_dart_refs:
                enclosing = _enclosing_section_dart_ref(raw_html, heading)
                if enclosing is not None:
                    section_dart_refs = [enclosing]
                else:
                    # No heading-anchored block found; if the document carries
                    # DART blocks at all, keep DART provenance ENABLED but defer
                    # the actual per-chunk selection to char-span containment
                    # (chunk_text_block sees the full offset index). The first
                    # document block is used only as the last-resort ≤1 fallback.
                    doc_refs = harvest_semantik_source_refs(raw_html)
                    section_dart_refs = doc_refs[:1] if doc_refs else []
            # Bound this section's block-probe scope by the NEXT group's heading
            # (structural section boundary). None → whole-item scope fallback in
            # chunk_text_block (heading unlocatable / no DART blocks).
            _next_head_pos = next(
                (
                    _group_head_positions[j]
                    for j in range(_group_idx + 1, len(merged))
                    if _group_head_positions[j] is not None
                ),
                None,
            )
            _section_scope_html = _section_block_scope_html(
                raw_html, heading, _next_head_pos
            )
            _unit_first_index = len(chunks)
            item_chunks = chunk_text_block(
                text=text,
                html=html_block,
                item=item,
                heading=heading,
                chunk_type=chunk_type,
                prefix=prefix,
                start_id=chunk_counter,
                follows_chunk_id=prev_chunk_id,
                position_in_module=position_in_module,
                section_source_ids=section_source_ids,
                merged_headings=merged_headings,
                merged_key_claims=merged_key_claims,
                merged_objective_alignment=merged_objective_alignment,
                curie_anchors=section_curies,
                forced_curie_anchors=section_forced_curies,
                dart_source_refs=section_dart_refs,
                dart_block_scope_html=_section_scope_html,
                ctx=ctx,
                max_chunk_size=max_chunk_size,
                target_chunk_size=target_chunk_size,
                fragment_floor=_fragment_floor,
                composite_unit=merged_composite_unit,
                unit_roles=merged_unit_roles,
                unit_subclass=merged_unit_subclass,
            )
            chunks.extend(item_chunks)
            chunk_counter += len(item_chunks)
            if item_chunks:
                prev_chunk_id = item_chunks[-1]["id"]
                position_in_module += len(item_chunks)
                if _unit_key:
                    _dedup_seen[_unit_key] = (
                        _unit_first_index,
                        str(item_chunks[0].get("id") or ""),
                    )

    return ChunkContentResult(
        chunks=chunks,
        pages_with_misconceptions=pages_with_misconceptions,
        dedup_drops=dedup_drops,
    )
