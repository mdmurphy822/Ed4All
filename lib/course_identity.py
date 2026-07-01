"""Course-identity resolution — one canonical slug + course_id per run.

W0.5: the LibV2 archive can SPLIT-BRAIN a single course across two directories:

* The chunk-write path (``MCP/tools/pipeline_tools.py::_run_dart_chunking``)
  names the course dir from the VERBATIM lowercased course name
  (``"Ed4All"`` → ``courses/ed4all/``) and populates it with real chunks.
* The decision-capture path (``lib/decision_capture.py``) can be handed the
  ``normalize_course_code`` form, which mints a ``sha256(name) % 1000`` numeric
  suffix (``"Ed4All"`` → ``"ED_472"`` → ``courses/ed-472/``) and creates an
  EMPTY skeleton there via ``LibV2Storage(auto_create=True)``.

Both land in the master catalog, so an operator sees one populated course and
one empty hashed twin for the same content.

This module is the single, idempotent resolver: from the verbatim course name
(plus any alternate course-code variants a caller knows about) it produces ONE
canonical slug + course_id, detects whether an empty-skeleton twin exists
alongside the populated course, and offers a one-shot cleanup that is gated
STRICTLY on "a populated twin already exists" so an in-progress, legitimately
empty course is never deleted.

All behaviour here is opt-in: callers gate on
:func:`course_identity_dedup_enabled` (env ``ED4ALL_COURSE_IDENTITY_DEDUP``,
default OFF, parse-with-fallback). With the flag unset the caller takes its
legacy path and this module is never consulted.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from lib.ontology.slugs import libv2_course_slug

logger = logging.getLogger(__name__)

__all__ = [
    "course_identity_dedup_enabled",
    "CourseIdentity",
    "resolve_course_identity",
    "course_is_populated",
    "course_is_empty_skeleton",
    "cleanup_empty_skeletons",
]

_TRUTHY = {"1", "true", "yes", "on"}


def course_identity_dedup_enabled(env: Optional[dict] = None) -> bool:
    """Return whether ``ED4ALL_COURSE_IDENTITY_DEDUP`` is truthy.

    Parse-with-fallback: unset / empty / any non-truthy token → ``False``
    (the byte-identical legacy path). Only ``1`` / ``true`` / ``yes`` / ``on``
    (case-insensitive) enable the dedup behaviour.
    """
    source = env if env is not None else os.environ
    raw = str(source.get("ED4ALL_COURSE_IDENTITY_DEDUP", "")).strip().lower()
    return raw in _TRUTHY


@dataclass
class CourseIdentity:
    """The resolved canonical identity for a course + any detected twins."""

    course_id: str
    slug: str
    populated_twin: Optional[str] = None
    empty_skeleton_twins: List[str] = field(default_factory=list)

    @property
    def split_brain_detected(self) -> bool:
        """True when an empty-skeleton twin exists alongside this identity."""
        return bool(self.empty_skeleton_twins)


def _resolve_root(libv2_root: Optional[os.PathLike]) -> Path:
    if libv2_root is not None:
        return Path(libv2_root)
    # Lazy import keeps this module importable in CLI-only environments.
    from lib.paths import libv2_path

    return libv2_path()


def course_is_populated(course_dir: Path) -> bool:
    """Return True when ``course_dir`` holds real archived content.

    Populated = a course ``manifest.json`` exists, OR a non-empty
    ``chunks.jsonl`` exists in any known chunkset dir (``dart_chunks`` /
    ``imscc_chunks`` / legacy ``corpus``).
    """
    course_dir = Path(course_dir)
    if not course_dir.is_dir():
        return False
    if (course_dir / "manifest.json").is_file():
        return True
    for sub in ("dart_chunks", "imscc_chunks", "corpus"):
        chunks = course_dir / sub / "chunks.jsonl"
        try:
            if chunks.is_file() and chunks.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def course_is_empty_skeleton(course_dir: Path) -> bool:
    """Return True when ``course_dir`` exists but holds ZERO regular files.

    A fresh ``LibV2Storage(auto_create=True)`` creates empty subdirectories
    (``imscc_chunks/``, ``sources/``, ``concept_graph/`` …) but writes no
    files. Any regular file anywhere under the dir (a manifest, a chunks file,
    an objectives sidecar) disqualifies it — anti-fabrication: we never delete
    a dir that carries any content.
    """
    course_dir = Path(course_dir)
    if not course_dir.is_dir():
        return False
    if course_is_populated(course_dir):
        return False
    try:
        for entry in course_dir.rglob("*"):
            if entry.is_file():
                return False
    except OSError:
        # If we can't fully walk it, treat it as non-empty (never delete).
        return False
    return True


def resolve_course_identity(
    course_name: str,
    *,
    alt_course_codes: Iterable[str] = (),
    libv2_root: Optional[os.PathLike] = None,
) -> CourseIdentity:
    """Resolve ONE canonical slug + course_id for ``course_name``.

    The canonical slug is the verbatim-name archive slug
    (:func:`lib.ontology.slugs.libv2_course_slug`), which matches the
    chunk-write path's directory name for the common cases — i.e. the populated
    course dir. ``alt_course_codes`` lets a caller pass known alternate
    spellings (e.g. the ``normalize_course_code`` hashed form) whose slugs are
    checked for empty-skeleton twins.

    The resolution is PURE (no filesystem writes): it inspects which candidate
    dirs exist and classifies them, returning a :class:`CourseIdentity`. The
    caller decides whether to act (warn / cleanup) — see
    :func:`cleanup_empty_skeletons`.

    Anti-fabrication: the canonical identity is always derived from the real
    course name; a hashed alternate is only ever a CLEANUP candidate, never the
    canonical answer.
    """
    root = _resolve_root(libv2_root)
    courses_dir = root / "courses"

    canonical_slug = libv2_course_slug(course_name)
    course_id = (course_name or "").strip().replace(" ", "_").upper()

    # Build the candidate slug set: canonical + every alternate code's slug,
    # de-duplicated, excluding the canonical itself for twin classification.
    candidate_slugs: List[str] = []
    seen = {canonical_slug}
    for alt in alt_course_codes:
        alt_slug = libv2_course_slug(str(alt))
        if alt_slug and alt_slug not in seen:
            seen.add(alt_slug)
            candidate_slugs.append(alt_slug)

    canonical_dir = courses_dir / canonical_slug
    canonical_populated = course_is_populated(canonical_dir)

    empty_twins: List[str] = []
    for slug in candidate_slugs:
        twin_dir = courses_dir / slug
        if course_is_empty_skeleton(twin_dir):
            empty_twins.append(slug)

    return CourseIdentity(
        course_id=course_id,
        slug=canonical_slug,
        populated_twin=canonical_slug if canonical_populated else None,
        empty_skeleton_twins=empty_twins,
    )


def cleanup_empty_skeletons(
    identity: CourseIdentity,
    *,
    libv2_root: Optional[os.PathLike] = None,
) -> List[str]:
    """Remove the empty-skeleton twins for ``identity`` (one-shot, strict).

    STRICT GATE (anti-fabrication): cleanup only runs when the canonical course
    is itself POPULATED on disk — i.e. a real populated twin exists. A
    legitimately-empty in-progress course (no populated canonical) is left
    untouched. Each twin is re-verified as an empty skeleton immediately before
    removal, so a twin that gained content between resolution and cleanup is
    spared.

    Returns the list of slugs actually removed.
    """
    root = _resolve_root(libv2_root)
    courses_dir = root / "courses"

    # Strict gate: do nothing unless the canonical course is populated.
    if not course_is_populated(courses_dir / identity.slug):
        return []

    removed: List[str] = []
    for slug in identity.empty_skeleton_twins:
        if slug == identity.slug:
            continue
        twin_dir = courses_dir / slug
        # Re-verify right before deleting (TOCTOU-safe against late writes).
        if not course_is_empty_skeleton(twin_dir):
            continue
        try:
            shutil.rmtree(twin_dir)
            removed.append(slug)
            logger.warning(
                "COURSE_IDENTITY_SPLIT_BRAIN: removed empty skeleton twin "
                "%r (populated canonical=%r).",
                slug,
                identity.slug,
            )
        except OSError as exc:
            logger.warning(
                "cleanup_empty_skeletons: failed to remove %s (%s); leaving "
                "it in place.",
                twin_dir,
                exc,
            )
    return removed
