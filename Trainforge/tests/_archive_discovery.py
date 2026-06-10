"""Dynamic discovery of a real LibV2 course archive for regression tests.

Real-archive regression tests assert the pedagogy/concept-graph builder
re-builds a previously-archived corpus within a measured edge-count
envelope. The corpus lives under the (gitignored) ``LibV2/courses/``
tree, so the tests must NOT hardcode a course slug or a concrete path
into that directory. This helper discovers the first course archive
that carries the artifacts a real-archive regression needs and derives
the ``course_id`` from the discovered directory name. Tests skip
cleanly when no such archive is present (the default on a clean
checkout).

Honors ``ED4ALL_LIBV2_ROOT`` (same contract as
``test_emit_pipeline_full_archive.py``); falls back to the in-repo
``LibV2/`` root otherwise.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.libv2_storage import resolve_imscc_chunks_path  # noqa: E402


def _libv2_courses_root() -> Path:
    root = os.environ.get("ED4ALL_LIBV2_ROOT")
    base = Path(root) if root else _PROJECT_ROOT / "LibV2"
    return base / "courses"


def _find_synthesized_objectives(course_dir: Path) -> Optional[Path]:
    """First ``synthesized_objectives.json`` anywhere under the course."""
    matches = sorted(course_dir.rglob("synthesized_objectives.json"))
    return matches[0] if matches else None


def _course_id_from_dir(course_dir: Path) -> str:
    """Derive an upper-snake course_id from a course-slug directory name.

    Mirrors the historical ``<slug>`` → ``<COURSE_ID>`` convention: the
    slug's first two hyphen-tokens, upper-cased and underscore-joined
    (e.g. ``chemistry-201-organic`` → ``CHEMISTRY_201``). Falls back to
    the whole de-hyphenated name when the slug carries fewer tokens.
    """
    tokens = [t for t in course_dir.name.split("-") if t]
    head = tokens[:2] if len(tokens) >= 2 else tokens
    return "_".join(head).upper() if head else course_dir.name.upper()


def _archive_tuple(
    course_dir: Path,
) -> Optional[Tuple[Path, Path, Optional[Path], str]]:
    """Build the discovery tuple for one course dir, or None if unusable."""
    chunks = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    if not chunks.exists():
        return None
    objectives = _find_synthesized_objectives(course_dir)
    if objectives is None:
        return None
    concept_graph = course_dir / "graph" / "concept_graph.json"
    return (
        chunks,
        objectives,
        concept_graph if concept_graph.exists() else None,
        _course_id_from_dir(course_dir),
    )


def discover_real_archive() -> Tuple[
    Optional[Path], Optional[Path], Optional[Path], Optional[str]
]:
    """Discover a real course archive for a regression test.

    Returns ``(corpus_chunks, synthesized_objectives, concept_graph,
    course_id)``. Any element is ``None`` when the corresponding
    artifact is absent; callers skip when the artifacts they need are
    missing.

    Selection: a course carrying a resolvable ``chunks.jsonl``
    (``imscc_chunks/`` → ``dart_chunks/`` → legacy ``corpus/``) plus a
    discoverable ``synthesized_objectives.json`` wins. The real-archive
    regression envelopes that consume this helper are calibrated against
    the original (pre-rename, legacy ``corpus/``-layout) calibration
    archive, so a legacy-``corpus/``-layout course is PREFERRED when one
    is present; resolver-found ``imscc_chunks/`` / ``dart_chunks/``
    archives serve as the fallback so the helper still discovers
    post-rename corpora when no legacy archive remains. An operator can
    pin a specific course directory name via ``ED4ALL_ARCHIVE_FIXTURE_SLUG``
    (the sanctioned slug-free escape hatch — the slug never lands in
    tracked code).
    """
    courses_root = _libv2_courses_root()
    if not courses_root.is_dir():
        return None, None, None, None

    pinned = os.environ.get("ED4ALL_ARCHIVE_FIXTURE_SLUG")
    if pinned:
        course_dir = courses_root / pinned
        if course_dir.is_dir():
            tup = _archive_tuple(course_dir)
            if tup is not None:
                return tup
        return None, None, None, None

    dirs = [d for d in sorted(courses_root.iterdir()) if d.is_dir()]
    # The real-archive envelope assertions are calibrated against the
    # original (legacy ``corpus/``-layout) calibration archive. When ANY
    # legacy-``corpus/``-layout course is present we restrict selection to
    # those — a post-rename corpus with different edge counts would fail
    # the calibrated envelope, so the regression must target the
    # calibration archive (or skip cleanly when it lacks the artifacts a
    # given consumer needs, e.g. synthesized_objectives.json).
    legacy_dirs = [
        d for d in dirs if (d / "corpus" / "chunks.jsonl").exists()
    ]
    if legacy_dirs:
        for course_dir in legacy_dirs:
            tup = _archive_tuple(course_dir)
            if tup is not None:
                return tup
        return None, None, None, None
    # No legacy archive remains (fully migrated checkout): fall back to
    # any resolver-found layout (``imscc_chunks/`` / ``dart_chunks/``).
    for course_dir in dirs:
        tup = _archive_tuple(course_dir)
        if tup is not None:
            return tup
    return None, None, None, None
