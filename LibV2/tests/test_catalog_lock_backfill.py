"""Regression tests for the W0.2 catalog lock + W0.8 backfill/discovery."""

import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LibV2.tools.libv2.catalog import (
    _register_course_in_catalog,
    backfill_master_catalog,
    load_master_catalog,
)


def _write_course_manifest(libv2_root: Path, slug: str, title: str) -> dict:
    course_dir = libv2_root / "courses" / slug
    course_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"slug": slug, "title": title, "classification": {"division": "STEM"}}
    (course_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


# --- W0.2: concurrent registration must not lost-update ---------------------

def _register_worker(args):
    libv2_root, slug = args
    manifest = {"slug": slug, "title": slug, "classification": {}}
    _register_course_in_catalog(slug, manifest, Path(libv2_root))


@pytest.mark.integration
def test_concurrent_registration_no_lost_update(tmp_path):
    libv2_root = tmp_path / "libv2"
    (libv2_root / "catalog").mkdir(parents=True)
    slugs = [f"course-{i:03d}" for i in range(12)]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=6) as pool:
        pool.map(_register_worker, [(str(libv2_root), s) for s in slugs])

    catalog = load_master_catalog(libv2_root)
    assert catalog is not None
    got = {c.slug for c in catalog.courses}
    # Every concurrently-registered course must survive (no race drop).
    assert got == set(slugs)
    assert catalog.total_courses == len(slugs)


@pytest.mark.unit
def test_register_is_idempotent(tmp_path):
    libv2_root = tmp_path / "libv2"
    manifest = {"slug": "course-kappa", "title": "Physics", "classification": {}}
    _register_course_in_catalog("course-kappa", manifest, libv2_root)
    _register_course_in_catalog("course-kappa", manifest, libv2_root)
    catalog = load_master_catalog(libv2_root)
    assert [c.slug for c in catalog.courses] == ["course-kappa"]


# --- W0.8: backfill enumerates ALL archived course dirs ---------------------

@pytest.mark.unit
def test_backfill_enumerates_all_courses(tmp_path):
    libv2_root = tmp_path / "libv2"
    # 5 courses on disk; catalog initially registered with only 1.
    for i in range(5):
        _write_course_manifest(libv2_root, f"course-{i}", f"Course {i}")
    _register_course_in_catalog(
        "course-0",
        {"slug": "course-0", "title": "Course 0", "classification": {}},
        libv2_root,
    )

    catalog_before = load_master_catalog(libv2_root)
    assert len(catalog_before.courses) == 1  # the live path saw only 1

    summary = backfill_master_catalog(libv2_root)
    assert summary["discovered"] == 5
    assert summary["total"] == 5

    catalog_after = load_master_catalog(libv2_root)
    assert {c.slug for c in catalog_after.courses} == {
        f"course-{i}" for i in range(5)
    }


@pytest.mark.unit
def test_backfill_is_idempotent(tmp_path):
    libv2_root = tmp_path / "libv2"
    for i in range(3):
        _write_course_manifest(libv2_root, f"c-{i}", f"C{i}")
    first = backfill_master_catalog(libv2_root)
    second = backfill_master_catalog(libv2_root)
    assert first["total"] == second["total"] == 3
    assert second["added"] == 0  # nothing new the second time
    catalog = load_master_catalog(libv2_root)
    assert len(catalog.courses) == 3


@pytest.mark.unit
def test_backfill_skips_dirs_without_manifest(tmp_path):
    libv2_root = tmp_path / "libv2"
    _write_course_manifest(libv2_root, "real-course", "Real")
    # A bare skeleton dir with no manifest must NOT be enrolled.
    (libv2_root / "courses" / "empty-skeleton" / "imscc_chunks").mkdir(parents=True)
    summary = backfill_master_catalog(libv2_root)
    assert summary["discovered"] == 1
    catalog = load_master_catalog(libv2_root)
    assert {c.slug for c in catalog.courses} == {"real-course"}


@pytest.mark.unit
def test_backfill_merges_without_truncating(tmp_path):
    libv2_root = tmp_path / "libv2"
    # Pre-existing catalog entry whose dir is NOT on disk anymore.
    _register_course_in_catalog(
        "legacy-course",
        {"slug": "legacy-course", "title": "Legacy", "classification": {}},
        libv2_root,
    )
    _write_course_manifest(libv2_root, "new-course", "New")
    backfill_master_catalog(libv2_root)
    catalog = load_master_catalog(libv2_root)
    slugs = {c.slug for c in catalog.courses}
    # Merge: keep the pre-existing entry AND add the discovered one.
    assert slugs == {"legacy-course", "new-course"}
