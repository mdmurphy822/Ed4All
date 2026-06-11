"""Disk-usage cache behavior for the Studio Library cards (D5).

The per-course byte total is TTL-cached keyed by the resolved course path so an
``os.scandir`` walk doesn't re-run on every Library load. These tests pin the
cache semantics (hit within TTL, recompute after expiry, invalidation) against a
synthetic ``tmp_path`` course — no real LibV2 dir is touched.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("bs4")

from gui.services import imscc_service  # noqa: E402

_MANIFEST = """<?xml version='1.0'?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1" identifier="M">
  <organizations><organization identifier="O"><item identifier="I" identifierref="R">
    <title>Page</title></item></organization></organizations>
  <resources><resource identifier="R" type="webcontent" href="p.html">
    <file href="p.html"/></resource></resources>
</manifest>"""


def _course(root: Path, slug: str, *, payload: bytes = b"x" * 500) -> Path:
    cdir = root / "courses" / slug / "source" / "imscc"
    cdir.mkdir(parents=True)
    with zipfile.ZipFile(cdir / "C.imscc", "w") as zf:
        zf.writestr("imsmanifest.xml", _MANIFEST)
        zf.writestr("p.html", "<html><body><h1>P</h1></body></html>")
    # An extra payload file so the dir size is meaningful + mutable.
    (root / "courses" / slug / "blob.bin").write_bytes(payload)
    return root / "courses" / slug


@pytest.fixture(autouse=True)
def _clear_cache():
    imscc_service._invalidate_disk_cache()
    yield
    imscc_service._invalidate_disk_cache()


def test_cached_disk_bytes_hits_within_ttl(tmp_path, monkeypatch):
    course = _course(tmp_path / "libv2", "demo-101", payload=b"a" * 1000)
    first = imscc_service._cached_disk_bytes(course)
    assert first > 1000

    # Grow the dir; within the TTL the cached value is returned (stale-by-design).
    (course / "blob.bin").write_bytes(b"a" * 5000)
    second = imscc_service._cached_disk_bytes(course)
    assert second == first, "within TTL the cached size is reused (no rescan)"


def test_cached_disk_bytes_recomputes_after_expiry(tmp_path, monkeypatch):
    course = _course(tmp_path / "libv2", "demo-101", payload=b"a" * 1000)

    fake_now = [1000.0]
    monkeypatch.setattr(imscc_service.time, "monotonic", lambda: fake_now[0])

    first = imscc_service._cached_disk_bytes(course)
    (course / "blob.bin").write_bytes(b"a" * 9000)
    # Advance past the TTL → recompute picks up the new size.
    fake_now[0] += imscc_service._DISK_CACHE_TTL_S + 1
    second = imscc_service._cached_disk_bytes(course)
    assert second > first


def test_invalidate_drops_entry(tmp_path):
    course = _course(tmp_path / "libv2", "demo-101", payload=b"a" * 1000)
    first = imscc_service._cached_disk_bytes(course)
    (course / "blob.bin").write_bytes(b"a" * 7000)
    imscc_service._invalidate_disk_cache(course)
    second = imscc_service._cached_disk_bytes(course)
    assert second > first, "after invalidation the next read rescans"
