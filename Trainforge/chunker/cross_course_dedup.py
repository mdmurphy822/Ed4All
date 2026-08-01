"""W1b.4 — reusable cross-course boilerplate dedup for chunk ingestion.

A multi-course import batch (e.g. a multi-course mkdocs documentation corpus) ships
the SAME site chrome on every course: a shared footer, a licence banner, an
"Edit this page" nav strip, a boilerplate "About" blurb. Ingested verbatim,
each of those becomes an identical chunk in EVERY course's chunkset — retrieval
noise that also inflates cross-course near-duplicate rates.

This module is a TRACKED, importer-agnostic helper. It hashes the NORMALISED
text of every chunk, finds the hashes that recur across ``min_courses`` or more
DISTINCT courses, and lets an importer drop those cross-course duplicates. It
is deliberately conservative:

- Normalisation is lossy ONLY for whitespace / case / punctuation, so genuinely
  identical boilerplate collides while real prose (which differs course to
  course) does not.
- A minimum token floor (``min_tokens``) protects short-but-legitimate chunks
  (a shared one-line definition that happens to recur) from being flagged — the
  target is repeated CHROME, not repeated facts.
- Anti-fabrication: dedup only ever DROPS a chunk; it never rewrites text or
  invents ids. The kept set is always a strict subset of the input.

Wiring is opt-in. An importer reads :func:`resolve_cross_course_dedup_enabled`
(env ``ED4ALL_CROSS_COURSE_DEDUP``, default OFF → no dedup, byte-identical) and,
when on, builds a :class:`CrossCourseDedupIndex` over the whole batch before
writing any course's chunks.

Two distinct dedup policies live here
=====================================

The module hosts a SECOND, independent policy — the **within-package**
exact-normalised primitives (:func:`normalize_exact`,
:func:`exact_content_hash`, :func:`exact_token_count`, and the
``ED4ALL_CHUNK_DEDUP`` / ``ED4ALL_CHUNK_DEDUP_MIN_TOKENS`` resolvers). They are
deliberately NOT built on the cross-course helpers above, for two reasons that
are correctness issues rather than preferences:

1. :func:`normalize_for_dedup` strips **all** punctuation, so ``sh:minCount``
   and ``sh minCount``, or ``f(x) = 1`` and ``f x 1``, hash identically. On a
   formal-notation corpus that is a false-positive DELETION.
   :func:`normalize_exact` collapses whitespace and casefolds only.
2. :class:`CrossCourseDedupIndex` requires a hash to recur across
   ``min_courses`` DISTINCT courses, so it structurally flags nothing inside a
   single package, and :func:`drop_boilerplate_chunks` drops **every**
   occurrence including the first. The within-package pass keeps the first
   source-ordered occurrence.

The within-package pass itself lives in
``Trainforge/chunker/chunker.py::chunk_content`` — it must run BEFORE
``_generate_chunk_id`` mints anything, so it cannot be a post-hoc filter over
an already-emitted chunk list.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

__all__ = [
    "ENV_CROSS_COURSE_DEDUP",
    "resolve_cross_course_dedup_enabled",
    "normalize_for_dedup",
    "chunk_content_hash",
    "CrossCourseDedupIndex",
    "drop_boilerplate_chunks",
    "DEFAULT_MIN_COURSES",
    "DEFAULT_MIN_TOKENS",
    # Within-package exact-normalised dedup (second, independent policy).
    "ENV_CHUNK_DEDUP",
    "ENV_CHUNK_DEDUP_MIN_TOKENS",
    "DEFAULT_CHUNK_DEDUP_MIN_TOKENS",
    "resolve_chunk_dedup_enabled",
    "resolve_chunk_dedup_min_tokens",
    "normalize_exact",
    "exact_content_hash",
    "exact_token_count",
]

ENV_CROSS_COURSE_DEDUP = "ED4ALL_CROSS_COURSE_DEDUP"

#: Env var gating the WITHIN-package exact-normalised dedup pass run by
#: ``Trainforge.chunker.chunker.chunk_content``. Default OFF — the pass deletes
#: content, so flipping it on is a separate owner decision.
ENV_CHUNK_DEDUP = "ED4ALL_CHUNK_DEDUP"

#: Env var setting the within-package dedup token floor.
ENV_CHUNK_DEDUP_MIN_TOKENS = "ED4ALL_CHUNK_DEDUP_MIN_TOKENS"

#: A hash must recur across at least this many DISTINCT courses to be treated
#: as cross-course boilerplate.
DEFAULT_MIN_COURSES = 2

#: Chunks with fewer normalised tokens than this are never flagged (protects
#: short legitimate content from the dedup).
DEFAULT_MIN_TOKENS = 8

#: Within-package dedup token floor: a repeat carrying fewer exact-normalised
#: tokens than this is NEVER dropped. Guards short-but-legitimate content
#: (a one-line definition that genuinely recurs) against deletion.
DEFAULT_CHUNK_DEDUP_MIN_TOKENS = 8

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def resolve_cross_course_dedup_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CROSS_COURSE_DEDUP`` (parse-with-fallback, default OFF)."""
    src = env if env is not None else os.environ
    return src.get(ENV_CROSS_COURSE_DEDUP, "").strip().lower() in _TRUTHY


def resolve_chunk_dedup_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CHUNK_DEDUP`` (parse-with-fallback, default OFF).

    OFF is the shipped default and means ``chunk_content`` emits a
    byte-identical chunkset — no hashing, no ledger, no skipped unit.
    """
    src = env if env is not None else os.environ
    return src.get(ENV_CHUNK_DEDUP, "").strip().lower() in _TRUTHY


def resolve_chunk_dedup_min_tokens(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve ``ED4ALL_CHUNK_DEDUP_MIN_TOKENS`` (parse-with-fallback).

    Returns the exact-normalised token floor below which a repeated unit is
    never dropped. Default :data:`DEFAULT_CHUNK_DEDUP_MIN_TOKENS`. ``0`` is a
    legitimate operator choice (no floor — every exact repeat is eligible) and
    is honoured; garbage, non-integer, and NEGATIVE values fall back to the
    default rather than silently disabling the guard.
    """
    src = env if env is not None else os.environ
    raw = src.get(ENV_CHUNK_DEDUP_MIN_TOKENS)
    if raw is None:
        return DEFAULT_CHUNK_DEDUP_MIN_TOKENS
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_DEDUP_MIN_TOKENS
    return val if val >= 0 else DEFAULT_CHUNK_DEDUP_MIN_TOKENS


def normalize_exact(text: str) -> str:
    """EXACT normalisation for within-package dedup: collapse whitespace, casefold.

    Deliberately narrower than :func:`normalize_for_dedup`: punctuation is
    PRESERVED, so ``sh:minCount`` and ``sh minCount`` — or ``f(x) = 1`` and
    ``f x 1`` — do NOT collide. The only differences erased are the ones that
    cannot carry meaning in extracted prose: leading / trailing / runs of
    whitespace, and case.

    ``casefold`` rather than ``lower`` so full case-folding (e.g. ``ß`` →
    ``ss``) is applied consistently for non-ASCII corpora.
    """
    return _WS_RE.sub(" ", text or "").strip().casefold()


def exact_content_hash(text: str) -> str:
    """SHA-256 (hex) of :func:`normalize_exact` output. Empty text → ``""``.

    An empty digest is the caller's signal that the text carries no
    dedup-eligible content; callers must treat it as "never a duplicate"
    rather than as a hash that collides with every other empty string.
    """
    norm = normalize_exact(text)
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def exact_token_count(text: str) -> int:
    """Whitespace-token count of :func:`normalize_exact` output."""
    norm = normalize_exact(text)
    return len(norm.split()) if norm else 0


def normalize_for_dedup(text: str) -> str:
    """Lossy normalisation for dedup hashing: lowercase, strip punctuation,
    collapse whitespace.

    Two chunks whose only difference is casing / punctuation / whitespace
    normalise to the same string (so shared chrome collides). Real prose that
    differs by even one content word does not.
    """
    low = (text or "").lower()
    low = _PUNCT_RE.sub(" ", low)
    return _WS_RE.sub(" ", low).strip()


def chunk_content_hash(text: str) -> str:
    """SHA-256 (hex) of the normalised chunk text. Empty text → ``""``."""
    norm = normalize_for_dedup(text)
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _token_count(text: str) -> int:
    norm = normalize_for_dedup(text)
    return len(norm.split()) if norm else 0


@dataclass
class CrossCourseDedupIndex:
    """Accumulates normalised chunk hashes across courses to find boilerplate.

    Usage::

        idx = CrossCourseDedupIndex()
        for course_id, chunks in batch.items():
            idx.add_course(course_id, chunks)
        boilerplate = idx.boilerplate_hashes()
        for course_id, chunks in batch.items():
            kept, dropped = drop_boilerplate_chunks(chunks, boilerplate)
    """

    min_courses: int = DEFAULT_MIN_COURSES
    min_tokens: int = DEFAULT_MIN_TOKENS
    #: hash → set of course ids it appeared in (only hashes meeting the token
    #: floor are tracked).
    _hash_courses: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_course(
        self, course_id: str, chunks: Iterable[Dict[str, Any]], *, text_key: str = "text"
    ) -> None:
        """Register every eligible chunk's hash under ``course_id``."""
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get(text_key, "") or "")
            if _token_count(text) < self.min_tokens:
                continue
            h = chunk_content_hash(text)
            if h:
                self._hash_courses[h].add(str(course_id))

    def boilerplate_hashes(self) -> Set[str]:
        """Hashes that appear in ``>= min_courses`` DISTINCT courses."""
        return {
            h
            for h, courses in self._hash_courses.items()
            if len(courses) >= self.min_courses
        }

    def is_boilerplate(self, text: str) -> bool:
        """True when ``text`` normalises to a known cross-course-boilerplate hash."""
        h = chunk_content_hash(text)
        return bool(h) and len(self._hash_courses.get(h, ())) >= self.min_courses


def drop_boilerplate_chunks(
    chunks: List[Dict[str, Any]],
    boilerplate_hashes: Set[str],
    *,
    text_key: str = "text",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition ``chunks`` into ``(kept, dropped)`` by boilerplate hash.

    Anti-fabrication: kept is a strict subset of the input (same objects); no
    chunk is rewritten. A chunk whose normalised text hashes into
    ``boilerplate_hashes`` lands in ``dropped``.
    """
    if not boilerplate_hashes:
        return list(chunks), []
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            kept.append(chunk)
            continue
        h = chunk_content_hash(str(chunk.get(text_key, "") or ""))
        if h and h in boilerplate_hashes:
            dropped.append(chunk)
        else:
            kept.append(chunk)
    return kept, dropped
