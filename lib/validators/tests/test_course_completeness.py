"""Regression net for CourseCompletenessValidator ("true full course" gate).

Warning-day-1 posture: incomplete archives surface a code but PASS (never
block) by default; the opt-in ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE strict
mode flips them to critical/blocking. Uses synthetic tmp course dirs only
(no torch / embedding / model load).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.libv2.course_completeness import (  # noqa: E402
    CourseCompletenessValidator,
    resolve_archive_min_chunks,
    resolve_require_full_course,
)

_STRICT = "ED4ALL_ARCHIVE_REQUIRE_FULL_COURSE"
_ALLOW_FAKE = "ED4ALL_EMBEDDING_ALLOW_FAKE"
_MIN = "ED4ALL_MIN_CHUNKS"


def _codes(result):
    return {i.code for i in result.issues}


def _mkcourse(base, name, *, imscc=0, dart=0, index=None, provider="st",
              vec_count=None, chunkset_kind="imscc", full=False):
    d = base / name
    d.mkdir(parents=True)
    if imscc:
        cd = d / "imscc_chunks"; cd.mkdir()
        (cd / "chunks.jsonl").write_text(
            "".join(json.dumps({"id": f"c{i}"}) + "\n" for i in range(imscc)))
    if dart:
        cd = d / "dart_chunks"; cd.mkdir()
        (cd / "chunks.jsonl").write_text(
            "".join(json.dumps({"id": f"d{i}"}) + "\n" for i in range(dart)))
    if full:
        (d / "course.json").write_text("{}")
        (d / "concept_graph").mkdir()
    if index is not None:
        vi = d / "vector_index"; vi.mkdir()
        n = vec_count if vec_count is not None else index
        (vi / "embeddings.npy").write_bytes(b"\x93NUMPY-fake")
        (vi / "id_map.json").write_text(
            json.dumps({"chunk_ids": [f"c{i}" for i in range(n)]}))
        (vi / "manifest.json").write_text(json.dumps(
            {"embedding_provider": provider, "chunkset_kind": chunkset_kind,
             "chunks_count": n}))
    return d


def _validate(monkeypatch, d, *, strict=False, allow_fake=False):
    if strict:
        monkeypatch.setenv(_STRICT, "1")
    else:
        monkeypatch.delenv(_STRICT, raising=False)
    if allow_fake:
        monkeypatch.setenv(_ALLOW_FAKE, "1")
    else:
        monkeypatch.delenv(_ALLOW_FAKE, raising=False)
    monkeypatch.delenv(_MIN, raising=False)
    return CourseCompletenessValidator().validate({"course_dir": str(d)})


# --- resolvers -------------------------------------------------------- #

def test_resolvers_default_off_and_floor(monkeypatch):
    monkeypatch.delenv(_STRICT, raising=False)
    monkeypatch.delenv(_MIN, raising=False)
    assert resolve_require_full_course() is False
    assert resolve_archive_min_chunks() == 20


@pytest.mark.parametrize("val", ["1", "true", "on", "YES"])
def test_strict_truthy(monkeypatch, val):
    monkeypatch.setenv(_STRICT, val)
    assert resolve_require_full_course() is True


@pytest.mark.parametrize("val", ["", "0", "no", "garbage"])
def test_strict_falsey_fallback(monkeypatch, val):
    monkeypatch.setenv(_STRICT, val)
    assert resolve_require_full_course() is False


def test_min_chunks_env_override(monkeypatch):
    monkeypatch.setenv(_MIN, "50")
    assert resolve_archive_min_chunks() == 50
    monkeypatch.setenv(_MIN, "garbage")
    assert resolve_archive_min_chunks() == 20  # parse-with-fallback


# --- PASSING shapes (both valid archive kinds) ------------------------ #

def test_full_course_passes_clean(monkeypatch, tmp_path):
    d = _mkcourse(tmp_path, "full", imscc=120, index=120, full=True)
    r = _validate(monkeypatch, d)
    assert r.passed is True
    assert _codes(r) == set()


def test_chunk_only_import_passes_clean(monkeypatch, tmp_path):
    d = _mkcourse(tmp_path, "chunkonly", imscc=199, index=199)
    r = _validate(monkeypatch, d)
    assert r.passed is True
    assert _codes(r) == set()
    # Still passes even in strict mode (a complete chunk-only import).
    r2 = _validate(monkeypatch, d, strict=True)
    assert r2.passed is True and _codes(r2) == set()


# --- FLAGGED shapes --------------------------------------------------- #

@pytest.mark.parametrize("kw,code", [
    (dict(),                                   "ARCHIVE_NO_CHUNKS"),
    (dict(imscc=50),                           "ARCHIVE_NO_INDEX"),
    (dict(imscc=5, index=5),                   "ARCHIVE_TOO_THIN"),
    (dict(imscc=100, index=100, vec_count=40), "ARCHIVE_INDEX_MISMATCH"),
    (dict(imscc=80, index=80, provider="fake"), "ARCHIVE_FAKE_INDEX"),
])
def test_flagged_default_warns_strict_blocks(monkeypatch, tmp_path, kw, code):
    d = _mkcourse(tmp_path, "c", **kw)
    # Default (warning): code surfaced, but never blocks.
    r = _validate(monkeypatch, d)
    assert r.passed is True
    assert code in _codes(r)
    assert all(i.severity != "critical" for i in r.issues)
    # Strict: same code, now critical/blocking.
    r2 = _validate(monkeypatch, d, strict=True)
    assert r2.passed is False
    assert code in _codes(r2)


def test_fake_index_allowed_when_opted_in(monkeypatch, tmp_path):
    d = _mkcourse(tmp_path, "fake", imscc=80, index=80, provider="fake")
    r = _validate(monkeypatch, d, strict=True, allow_fake=True)
    assert r.passed is True
    assert "ARCHIVE_FAKE_INDEX" not in _codes(r)


def test_unresolved_course_dir_is_info_noop(monkeypatch):
    monkeypatch.delenv(_STRICT, raising=False)
    r = CourseCompletenessValidator().validate({})
    assert r.passed is True
    assert "COURSE_DIR_UNRESOLVED" in _codes(r)


def test_decision_capture_fires(monkeypatch, tmp_path):
    d = _mkcourse(tmp_path, "empty")
    captured = []

    class _Cap:
        def log_decision(self, **kw):
            captured.append(kw)

    monkeypatch.delenv(_STRICT, raising=False)
    CourseCompletenessValidator().validate(
        {"course_dir": str(d), "decision_capture": _Cap()})
    assert len(captured) == 1
    assert captured[0]["decision_type"] == "validation_result"
    assert len(captured[0]["rationale"]) >= 20
