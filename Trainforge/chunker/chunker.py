"""Canonical chunker logic lifted out of ``Trainforge/process_course.py``.

Phase 7a Subtask 4 lifts the chunker proper into the ed4all-chunker
package so DART, Courseforge, and Trainforge can share one chunker
implementation. The functions in this module orchestrate:

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
(see ``plans/phase7_chunker_dual_chunkset.md`` Subtask 4 estimate of
~450 LOC).

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
    extract_plain_text,
    extract_section_html,
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
    "chunk_content",
    "chunk_text_block",
    "merge_small_sections",
    "merge_section_source_ids",
    "type_from_heading",
    "split_by_sentences",
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
    """

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


# ---------------------------------------------------------------------------
# merge_small_sections — adjacent-section merger up to MAX_CHUNK_SIZE
# ---------------------------------------------------------------------------


def merge_small_sections(
    sections: List[Any],
    *,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    type_from_heading_fn: Optional[Callable[[str], str]] = None,
) -> List[Tuple[str, str, str, List[str], List[str]]]:
    """Merge adjacent sections under MIN_CHUNK_SIZE into combined blocks.

    Mirrors ``CourseProcessor._merge_small_sections`` at
    ``process_course.py:1590`` byte-for-byte (the only refactor is
    parameterising ``self.MAX_CHUNK_SIZE`` → ``max_chunk_size`` and
    ``self._type_from_heading`` → ``type_from_heading_fn or type_from_heading``).

    Each ``section`` is expected to be a ``ContentSection``-like object
    with ``.heading``, ``.content``, ``.word_count``, and optionally
    ``.source_references`` and ``.template_type`` attributes. The
    chunker doesn't import ``ContentSection`` directly — it duck-types
    the attributes so the same orchestrator works against any
    section-like model (Trainforge / DART / future Courseforge).

    Returns a list of
    ``(heading, combined_text, chunk_type, merged_source_ids, merged_headings)``
    tuples, where ``merged_source_ids`` is the union of every section's
    ``data-cf-source-ids`` attribute (Wave 10) and ``merged_headings``
    is the ordered list of every heading that collapsed into the
    buffer (Wave 84 Bug 1 fix).
    """

    fn = type_from_heading_fn or type_from_heading

    merged: List[Tuple[str, str, str, List[str], List[str]]] = []
    buffer_heading = ""
    buffer_text = ""
    buffer_wc = 0
    buffer_type = "explanation"
    buffer_template_type: Optional[str] = None
    buffer_source_ids: List[str] = []
    buffer_headings: List[str] = []
    buffer_started = False

    def _resolve_buffer_type() -> str:
        # Wave 81: prefer template_type when present and canonical.
        if buffer_template_type and buffer_template_type in CANONICAL_CHUNK_TYPES:
            return buffer_template_type
        return buffer_type

    for section in sections:
        section_type = fn(section.heading)
        section_src = list(getattr(section, "source_references", []) or [])
        section_template = getattr(section, "template_type", None)

        if not buffer_started:
            buffer_heading = section.heading
            buffer_text = section.content
            buffer_wc = section.word_count
            buffer_type = section_type
            buffer_template_type = section_template
            buffer_source_ids = list(section_src)
            buffer_headings = [section.heading]
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
        else:
            merged.append((
                buffer_heading,
                buffer_text,
                _resolve_buffer_type(),
                buffer_source_ids,
                list(buffer_headings),
            ))
            buffer_heading = section.heading
            buffer_text = section.content
            buffer_wc = section.word_count
            buffer_type = section_type
            buffer_template_type = section_template
            buffer_source_ids = list(section_src)
            buffer_headings = [section.heading]

    if buffer_text.strip():
        merged.append((
            buffer_heading,
            buffer_text,
            _resolve_buffer_type(),
            buffer_source_ids,
            list(buffer_headings),
        ))

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

    if word_count <= max_chunk_size:
        char_span = _locate(text, search_from=0)
        chunks.append(ctx.create_chunk(
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
            section_source_ids=section_source_ids,
            merged_headings=merged_headings,
        ))
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
            chunks.append(ctx.create_chunk(
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
                section_source_ids=section_source_ids,
                merged_headings=merged_headings,
            ))
            last_chunk_id = this_chunk_id

    return chunks


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
    :func:`ed4all_chunker.boilerplate.strip_boilerplate`),
    assessment-feedback stripping (via
    :func:`ed4all_chunker.helpers.strip_assessment_feedback` /
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
            "ed4all_chunker.chunker module docstring)."
        )

    boilerplate = boilerplate_spans or []

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
            text = extract_plain_text(raw_html)
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
        )

        for heading, text, chunk_type, section_source_ids, merged_headings in merged:
            if not text.strip():
                continue
            if item["resource_type"] == "quiz":
                text = strip_feedback_from_text(text)
            if boilerplate:
                text, _ = strip_boilerplate(text, boilerplate)
            if not text.strip():
                continue
            html_block = extract_section_html(raw_html, heading)
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
