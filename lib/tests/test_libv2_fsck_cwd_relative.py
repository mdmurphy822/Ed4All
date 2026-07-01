"""Regression tests for the LibV2 fsck cwd-relative destructive-delete bug (W0.1).

``course_index.json`` stores each course's ``path`` RELATIVE to the LibV2 root
(``courses/<slug>``). The pre-fix ``_check_catalog`` resolved that path against
the process *current working directory*, so ``fsck --fix`` invoked from any cwd
other than the LibV2 root saw every (real) course as a dangling reference and
deleted the whole index. These tests pin the cwd-independent behavior.
"""

import json
from pathlib import Path

from lib.libv2_fsck import LibV2Fsck
from lib.ontology.slugs import canonical_slug


def _build_fixture_library(root: Path, slugs):
    """Materialise a minimal populated LibV2 fixture under ``root``.

    Writes a ``catalog/course_index.json`` whose entries reference courses by a
    path RELATIVE to ``root`` (mirroring the real catalog writer) and creates a
    populated course directory for each slug.
    """
    catalog_dir = root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for raw in slugs:
        slug = canonical_slug(raw)
        course_dir = root / "courses" / slug
        course_dir.mkdir(parents=True, exist_ok=True)
        (course_dir / "course_manifest.json").write_text("{}", encoding="utf-8")
        index[slug] = {
            "path": f"courses/{slug}",
            "title": raw,
            "division": "TEST",
        }
    index_path = catalog_dir / "course_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def test_fix_from_different_cwd_does_not_wipe_populated_index(tmp_path, monkeypatch):
    """`--fix` run from a foreign cwd must NOT delete real course entries."""
    libv2_root = tmp_path / "lib_root" / "LibV2"
    index_path = _build_fixture_library(libv2_root, ["Physics 101", "Bio 201"])

    # Run from a completely unrelated working directory — the bug's trigger.
    foreign_cwd = tmp_path / "elsewhere"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    fsck = LibV2Fsck(libv2_root)
    result = fsck.check_all(fix=True)

    surviving = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(surviving) == {canonical_slug("Physics 101"), canonical_slug("Bio 201")}
    assert result.fixed_count == 0
    # No dangling-reference issue should have been raised for real courses.
    assert not any(i.category == "dangling_reference" for i in result.issues)


def test_fix_still_removes_genuinely_dangling_entry(tmp_path, monkeypatch):
    """A real dangling entry (missing dir under the root) is still repaired."""
    libv2_root = tmp_path / "LibV2"
    index_path = _build_fixture_library(libv2_root, ["Real Course"])

    # Inject an entry whose course directory does NOT exist under the root.
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["ghost-course"] = {
        "path": "courses/ghost-course",
        "title": "Ghost",
        "division": "TEST",
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    foreign_cwd = tmp_path / "somewhere_else"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    fsck = LibV2Fsck(libv2_root)
    result = fsck.check_all(fix=True)

    surviving = json.loads(index_path.read_text(encoding="utf-8"))
    # Only the genuinely-missing entry is removed; the real course is kept.
    assert canonical_slug("Real Course") in surviving
    assert "ghost-course" not in surviving
    assert result.fixed_count == 1


def test_libv2_root_is_resolved_absolute(tmp_path, monkeypatch):
    """A relative root arg is anchored absolutely, immune to cwd changes."""
    libv2_root = tmp_path / "LibV2"
    _build_fixture_library(libv2_root, ["Course A"])

    monkeypatch.chdir(tmp_path)
    fsck = LibV2Fsck(Path("LibV2"))

    assert fsck.libv2_root.is_absolute()
    assert fsck.libv2_root == libv2_root.resolve()
