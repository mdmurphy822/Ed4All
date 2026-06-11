"""Tests for the shared course-removal service (Marketable-v1 D5).

All deletions run against SYNTHETIC ``tmp_path`` course dirs — never a real
LibV2 course. Covers: removes only the target dir, containment rejection
(``..`` / absolute / escaping slug), missing-course + empty-slug errors,
derived-catalog entry cleanup, and the disk-usage helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LibV2.tools.libv2.remove import (
    CourseRemovalError,
    dir_size_bytes,
    human_size,
    prune_catalog_entries,
    remove_course,
    resolve_course_dir,
)


def _make_course(root: Path, slug: str, *, files: int = 3, size: int = 100) -> Path:
    cdir = root / "courses" / slug
    (cdir / "imscc_chunks").mkdir(parents=True)
    (cdir / "source").mkdir(parents=True)
    for i in range(files):
        (cdir / "imscc_chunks" / f"chunk_{i}.jsonl").write_bytes(b"x" * size)
    (cdir / "manifest.json").write_text(json.dumps({"slug": slug}), encoding="utf-8")
    return cdir


# --------------------------------------------------------------------------- #
# Happy path — removes only the target
# --------------------------------------------------------------------------- #


def test_remove_deletes_only_target(tmp_path):
    root = tmp_path / "libv2"
    target = _make_course(root, "demo-101")
    sibling = _make_course(root, "keep-202")

    result = remove_course(root, "demo-101")

    assert not target.exists(), "target course dir must be deleted"
    assert sibling.exists(), "sibling course must be untouched"
    assert result.slug == "demo-101"
    assert result.disk_bytes > 0
    assert "imscc_chunks" in result.top_level
    assert "manifest.json" in result.top_level


def test_remove_result_reports_disk_and_contents(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101", files=2, size=500)
    result = remove_course(root, "demo-101")
    # 2 chunk files * 500 + manifest bytes > 1000.
    assert result.disk_bytes >= 1000
    assert sorted(result.top_level) == ["imscc_chunks", "manifest.json", "source"]


# --------------------------------------------------------------------------- #
# Containment rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_slug", ["..", "../etc", "../../secret", "/etc/passwd"])
def test_remove_rejects_escaping_slug(tmp_path, bad_slug):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    with pytest.raises(CourseRemovalError) as exc:
        remove_course(root, bad_slug)
    assert exc.value.status in (422,)
    assert exc.value.code in ("invalid_slug", "slug_escapes_root")
    # Nothing was deleted.
    assert (root / "courses" / "demo-101").exists()


@pytest.mark.parametrize("empty", ["", "   "])
def test_remove_rejects_empty_slug(tmp_path, empty):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    with pytest.raises(CourseRemovalError) as exc:
        remove_course(root, empty)
    assert exc.value.status == 422
    assert exc.value.code == "invalid_slug"
    assert (root / "courses" / "demo-101").exists()


def test_remove_missing_course_404(tmp_path):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    with pytest.raises(CourseRemovalError) as exc:
        remove_course(root, "nope-999")
    assert exc.value.status == 404
    assert exc.value.code == "course_not_found"


def test_resolve_course_dir_returns_resolved_path(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    resolved = resolve_course_dir(root, "demo-101")
    assert resolved == cdir.resolve()


# --------------------------------------------------------------------------- #
# Derived-catalog entry cleanup
# --------------------------------------------------------------------------- #


def _write_catalog(root: Path) -> None:
    catalog = root / "catalog"
    (catalog / "by_domain").mkdir(parents=True)
    (catalog / "master_catalog.json").write_text(
        json.dumps({
            "total_courses": 2,
            "courses": [{"slug": "demo-101", "title": "A"}, {"slug": "keep-202", "title": "B"}],
        }),
        encoding="utf-8",
    )
    (catalog / "course_index.json").write_text(
        json.dumps({"demo-101": {"path": "courses/demo-101"}, "keep-202": {"path": "courses/keep-202"}}),
        encoding="utf-8",
    )
    (catalog / "by_domain" / "physics.json").write_text(
        json.dumps({"domain": "physics", "count": 2, "courses": [{"slug": "demo-101"}, {"slug": "keep-202"}]}),
        encoding="utf-8",
    )


def test_remove_prunes_catalog_entries(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    _make_course(root, "keep-202")
    _write_catalog(root)

    result = remove_course(root, "demo-101")

    assert result.catalog_files_pruned, "catalog files referencing the slug must be pruned"

    master = json.loads((root / "catalog" / "master_catalog.json").read_text())
    assert master["total_courses"] == 1
    assert [c["slug"] for c in master["courses"]] == ["keep-202"]

    index = json.loads((root / "catalog" / "course_index.json").read_text())
    assert "demo-101" not in index
    assert "keep-202" in index

    domain = json.loads((root / "catalog" / "by_domain" / "physics.json").read_text())
    assert domain["count"] == 1
    assert [c["slug"] for c in domain["courses"]] == ["keep-202"]


def test_prune_catalog_no_catalog_dir_is_noop(tmp_path):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    assert prune_catalog_entries(root, "demo-101") == []


def test_remove_without_prune_leaves_catalog(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    _write_catalog(root)
    result = remove_course(root, "demo-101", prune_catalog=False)
    assert result.catalog_files_pruned == []
    master = json.loads((root / "catalog" / "master_catalog.json").read_text())
    assert any(c["slug"] == "demo-101" for c in master["courses"])


# --------------------------------------------------------------------------- #
# Disk-usage helpers
# --------------------------------------------------------------------------- #


def test_dir_size_bytes_sums_recursively(tmp_path):
    d = tmp_path / "c"
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"x" * 100)
    (d / "sub" / "b.bin").write_bytes(b"y" * 250)
    assert dir_size_bytes(d) == 350


def test_dir_size_bytes_missing_path_zero(tmp_path):
    assert dir_size_bytes(tmp_path / "nope") == 0


def test_dir_size_does_not_follow_dir_symlinks(tmp_path):
    # A symlinked dir must not be recursed into (avoid double counting / escape).
    real = tmp_path / "real"
    real.mkdir()
    (real / "big.bin").write_bytes(b"z" * 1000)
    course = tmp_path / "course"
    course.mkdir()
    (course / "small.bin").write_bytes(b"q" * 10)
    try:
        (course / "link").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    # Only small.bin (10) + the link's own tiny size — never the 1000-byte target.
    assert dir_size_bytes(course) < 1000


@pytest.mark.parametrize(
    "n,expected",
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB"),
     (3 * 1024 * 1024 * 1024, "3.00 GB")],
)
def test_human_size(n, expected):
    assert human_size(n) == expected
