"""Regression tests for the W0.5 course-identity split-brain resolver."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib import course_identity as ci


def _populate(libv2_root: Path, slug: str) -> Path:
    d = libv2_root / "courses" / slug / "dart_chunks"
    d.mkdir(parents=True)
    (d / "chunks.jsonl").write_text('{"id": "x_chunk_00000"}\n')
    return libv2_root / "courses" / slug


def _empty_skeleton(libv2_root: Path, slug: str) -> Path:
    d = libv2_root / "courses" / slug
    (d / "imscc_chunks").mkdir(parents=True)
    (d / "sources").mkdir()
    return d


# --- flag parsing -----------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True),
     ("0", False), ("false", False), ("", False), ("garbage", False)],
)
def test_dedup_flag_parse_with_fallback(val, expected):
    assert ci.course_identity_dedup_enabled({"ED4ALL_COURSE_IDENTITY_DEDUP": val}) is expected


@pytest.mark.unit
def test_dedup_flag_unset_is_false():
    assert ci.course_identity_dedup_enabled({}) is False


# --- populated / empty classification ---------------------------------------

@pytest.mark.unit
def test_course_is_populated_and_empty(tmp_path):
    _populate(tmp_path, "ed4all")
    _empty_skeleton(tmp_path, "ed-472")
    assert ci.course_is_populated(tmp_path / "courses" / "ed4all")
    assert not ci.course_is_populated(tmp_path / "courses" / "ed-472")
    assert ci.course_is_empty_skeleton(tmp_path / "courses" / "ed-472")
    assert not ci.course_is_empty_skeleton(tmp_path / "courses" / "ed4all")


@pytest.mark.unit
def test_manifest_only_counts_as_populated(tmp_path):
    d = tmp_path / "courses" / "with-manifest"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    assert ci.course_is_populated(d)
    assert not ci.course_is_empty_skeleton(d)


# --- resolution -------------------------------------------------------------

@pytest.mark.unit
def test_resolve_detects_split_brain(tmp_path):
    _populate(tmp_path, "ed4all")
    _empty_skeleton(tmp_path, "ed-472")
    identity = ci.resolve_course_identity(
        "Ed4All", alt_course_codes=["ED_472"], libv2_root=tmp_path
    )
    assert identity.slug == "ed4all"
    assert identity.course_id == "ED4ALL"
    assert identity.populated_twin == "ed4all"
    assert identity.empty_skeleton_twins == ["ed-472"]
    assert identity.split_brain_detected is True


@pytest.mark.unit
def test_resolve_no_twin_no_split_brain(tmp_path):
    _populate(tmp_path, "phys-101")
    identity = ci.resolve_course_identity("PHYS_101", libv2_root=tmp_path)
    assert identity.slug == "phys-101"
    assert identity.split_brain_detected is False
    assert identity.empty_skeleton_twins == []


# --- cleanup (strict gate) --------------------------------------------------

@pytest.mark.unit
def test_cleanup_removes_empty_twin_when_canonical_populated(tmp_path):
    _populate(tmp_path, "ed4all")
    _empty_skeleton(tmp_path, "ed-472")
    identity = ci.resolve_course_identity(
        "Ed4All", alt_course_codes=["ED_472"], libv2_root=tmp_path
    )
    removed = ci.cleanup_empty_skeletons(identity, libv2_root=tmp_path)
    assert removed == ["ed-472"]
    assert not (tmp_path / "courses" / "ed-472").exists()
    assert (tmp_path / "courses" / "ed4all").exists()  # canonical untouched


@pytest.mark.unit
def test_cleanup_no_op_when_canonical_not_populated(tmp_path):
    # No populated canonical → an in-progress empty course is NEVER deleted.
    _empty_skeleton(tmp_path, "ed4all")
    _empty_skeleton(tmp_path, "ed-472")
    identity = ci.resolve_course_identity(
        "Ed4All", alt_course_codes=["ED_472"], libv2_root=tmp_path
    )
    removed = ci.cleanup_empty_skeletons(identity, libv2_root=tmp_path)
    assert removed == []
    assert (tmp_path / "courses" / "ed-472").exists()


@pytest.mark.unit
def test_cleanup_spares_twin_that_gained_content(tmp_path):
    _populate(tmp_path, "ed4all")
    identity = ci.resolve_course_identity(
        "Ed4All", alt_course_codes=["ED_472"], libv2_root=tmp_path
    )
    # Resolution saw no twin; now create a populated 'twin' and assert cleanup
    # never touches a content-bearing dir even if asked.
    _populate(tmp_path, "ed-472")
    identity.empty_skeleton_twins = ["ed-472"]
    removed = ci.cleanup_empty_skeletons(identity, libv2_root=tmp_path)
    assert removed == []
    assert (tmp_path / "courses" / "ed-472").exists()
