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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from Trainforge.chunker.boilerplate import strip_boilerplate
from Trainforge.chunker.helpers import (
    _block_element_spans,
    build_dart_block_offset_index,
    extract_plain_text_with_curies,
    extract_section_html,
    harvest_dart_source_refs,
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
        if wc >= fragment_floor or not out:
            if wc >= fragment_floor:
                out.append(result)
            elif not out:
                # Leading fragment, nowhere to fold — drop it.
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
    return out


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
    buffer_started = False

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
        )

    for section in sections:
        section_type = fn(section.heading)
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
            buffer_started = True
        elif buffer_wc + section.word_count <= max_chunk_size:
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
    max_chunk_size: int = MAX_CHUNK_SIZE,
    target_chunk_size: int = TARGET_CHUNK_SIZE,
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
    :func:`Trainforge.chunker.helpers.harvest_dart_source_refs` from the
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
    # whole-document list. The probe index is built against the item's full
    # raw HTML so every block has a probe regardless of per-section container
    # scoping; matching is by chunk-text containment so a merged multi-section
    # chunk is attributed every block whose prose it contains, in doc order.
    block_probe_index = (
        build_dart_block_offset_index(raw_html_for_xpath)
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
            chunk_type=chunk_type,
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
                chunk_type=chunk_type,
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
    if not raw_html or not heading or "data-dart-block-id" not in raw_html:
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
    """

    chunks: List[Dict[str, Any]] = field(default_factory=list)
    pages_with_misconceptions: set = field(default_factory=set)

    def __iter__(self):
        # Permit tuple-unpacking ergonomics:
        # ``chunks, pages_with_misconceptions = chunk_content(...)``
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
            item_dart_refs = harvest_dart_source_refs(raw_html)
            if item["resource_type"] == "quiz":
                text = strip_feedback_from_text(text)
            if text.strip():
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
                )
                chunks.extend(item_chunks)
                chunk_counter += len(item_chunks)
                if item_chunks:
                    prev_chunk_id = item_chunks[-1]["id"]
                    position_in_module += len(item_chunks)
            continue

        merged = merge_small_sections(
            item["sections"],
            max_chunk_size=max_chunk_size,
            type_from_heading_fn=ctx.type_from_heading_fn if ctx else None,
            fragment_floor=_fragment_floor,
        )

        for merged_result in merged:
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
            if not text.strip():
                continue
            if item["resource_type"] == "quiz":
                text = strip_feedback_from_text(text)
            if boilerplate:
                text, _ = strip_boilerplate(text, boilerplate)
            if not text.strip():
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
                harvest_dart_source_refs(html_block)
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
                section_dart_refs.extend(harvest_dart_source_refs(extra_html))
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
            # NOT the whole document's list. ``harvest_dart_source_refs(raw_html)``
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
                    doc_refs = harvest_dart_source_refs(raw_html)
                    section_dart_refs = doc_refs[:1] if doc_refs else []
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
                ctx=ctx,
                max_chunk_size=max_chunk_size,
                target_chunk_size=target_chunk_size,
            )
            chunks.extend(item_chunks)
            chunk_counter += len(item_chunks)
            if item_chunks:
                prev_chunk_id = item_chunks[-1]["id"]
                position_in_module += len(item_chunks)

    return ChunkContentResult(
        chunks=chunks,
        pages_with_misconceptions=pages_with_misconceptions,
    )
