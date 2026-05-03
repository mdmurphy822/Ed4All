"""Phase 7c Subtask 15 — back-compat shim for ``imscc_chunks/`` rename.

Verifies the read-fallback contract documented in
``lib/libv2_storage.py::resolve_imscc_chunks_dir``:

1. New archive (only ``imscc_chunks/``) → returns ``imscc_chunks/``
   silently (no DeprecationWarning).
2. Legacy archive (only ``corpus/``) → returns ``corpus/`` with a
   DeprecationWarning naming the migration script.
3. Both present → prefers the new ``imscc_chunks/`` path silently.
4. Neither present → returns the canonical ``imscc_chunks/`` path so
   downstream readers see a clean ``FileNotFoundError`` rather than
   a misleading legacy path.

The shim is dropped in Phase 8.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from lib.libv2_storage import (
    IMSCC_CHUNKS_DIRNAME,
    LEGACY_CORPUS_DIRNAME,
    resolve_imscc_chunks_dir,
    resolve_imscc_chunks_path,
)


def test_new_archive_returns_imscc_chunks_silently(tmp_path: Path) -> None:
    """Phase 7c: archive with only ``imscc_chunks/`` → no warning."""
    course_dir = tmp_path / "course-foo"
    (course_dir / IMSCC_CHUNKS_DIRNAME).mkdir(parents=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_imscc_chunks_dir(course_dir)

    assert result == course_dir / IMSCC_CHUNKS_DIRNAME
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation == [], (
        "New archives must NOT emit a deprecation warning"
    )


def test_legacy_archive_returns_corpus_with_deprecation(tmp_path: Path) -> None:
    """Phase 7c: archive with only ``corpus/`` → returns legacy path + warns."""
    course_dir = tmp_path / "course-bar"
    (course_dir / LEGACY_CORPUS_DIRNAME).mkdir(parents=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_imscc_chunks_dir(course_dir)

    assert result == course_dir / LEGACY_CORPUS_DIRNAME
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation) == 1, (
        "Legacy archive must emit exactly one DeprecationWarning"
    )
    msg = str(deprecation[0].message)
    assert "Phase 7c" in msg
    assert "course-bar" in msg
    assert "backfill_dart_chunks.py" in msg, (
        "Deprecation must name the migration script"
    )


def test_both_present_prefers_imscc_chunks_silently(tmp_path: Path) -> None:
    """Phase 7c: both dirs present → prefer new, no warning."""
    course_dir = tmp_path / "course-baz"
    (course_dir / IMSCC_CHUNKS_DIRNAME).mkdir(parents=True)
    (course_dir / LEGACY_CORPUS_DIRNAME).mkdir(parents=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_imscc_chunks_dir(course_dir)

    assert result == course_dir / IMSCC_CHUNKS_DIRNAME
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation == [], (
        "Both-present must prefer new path silently"
    )


def test_neither_present_returns_canonical_path(tmp_path: Path) -> None:
    """Phase 7c: neither dir present → return canonical (new) path.

    Callers will get a clean ``FileNotFoundError`` on the subsequent
    open, not a misleading legacy path.
    """
    course_dir = tmp_path / "course-qux"
    course_dir.mkdir()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_imscc_chunks_dir(course_dir)

    assert result == course_dir / IMSCC_CHUNKS_DIRNAME
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation == [], (
        "Empty-archive case must not emit a deprecation warning"
    )


def test_resolve_imscc_chunks_path_chunks_jsonl(tmp_path: Path) -> None:
    """Convenience helper returns ``<chunkset_dir>/chunks.jsonl``."""
    course_dir = tmp_path / "course-quux"
    (course_dir / IMSCC_CHUNKS_DIRNAME).mkdir(parents=True)
    (course_dir / IMSCC_CHUNKS_DIRNAME / "chunks.jsonl").write_text("{}\n")

    result = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
    assert result == course_dir / IMSCC_CHUNKS_DIRNAME / "chunks.jsonl"
    assert result.is_file()


def test_resolve_imscc_chunks_path_legacy_chunks_jsonl(tmp_path: Path) -> None:
    """Convenience helper falls back to legacy ``corpus/chunks.jsonl``."""
    course_dir = tmp_path / "course-corge"
    (course_dir / LEGACY_CORPUS_DIRNAME).mkdir(parents=True)
    (course_dir / LEGACY_CORPUS_DIRNAME / "chunks.jsonl").write_text("{}\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")

    assert result == course_dir / LEGACY_CORPUS_DIRNAME / "chunks.jsonl"
    assert result.is_file()
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation) == 1


def test_resolve_accepts_string_path(tmp_path: Path) -> None:
    """Both helpers accept string course_dir, not just Path."""
    course_dir = tmp_path / "course-grault"
    (course_dir / IMSCC_CHUNKS_DIRNAME).mkdir(parents=True)

    # Pass as string
    result = resolve_imscc_chunks_dir(str(course_dir))
    assert result == course_dir / IMSCC_CHUNKS_DIRNAME

    result_path = resolve_imscc_chunks_path(str(course_dir), "chunks.jsonl")
    assert result_path == course_dir / IMSCC_CHUNKS_DIRNAME / "chunks.jsonl"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
