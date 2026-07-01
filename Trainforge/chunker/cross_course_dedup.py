"""W1b.4 — reusable cross-course boilerplate dedup for chunk ingestion.

A multi-course import batch (e.g. the multi-course 10-course mkdocs corpus) ships
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
]

ENV_CROSS_COURSE_DEDUP = "ED4ALL_CROSS_COURSE_DEDUP"

#: A hash must recur across at least this many DISTINCT courses to be treated
#: as cross-course boilerplate.
DEFAULT_MIN_COURSES = 2

#: Chunks with fewer normalised tokens than this are never flagged (protects
#: short legitimate content from the dedup).
DEFAULT_MIN_TOKENS = 8

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def resolve_cross_course_dedup_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CROSS_COURSE_DEDUP`` (parse-with-fallback, default OFF)."""
    src = env if env is not None else os.environ
    return src.get(ENV_CROSS_COURSE_DEDUP, "").strip().lower() in _TRUTHY


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
