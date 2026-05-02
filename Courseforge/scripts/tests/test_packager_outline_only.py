"""Phase 2 Subtask 29: --outline-only CLI flag on package_multifile_imscc.

Builds a minimal content-dir layout in `tmp_path` (overview / content /
application / self_check / summary / discussion HTML files for one
week), invokes `build_manifest(..., outline_only=True)` and
`package_imscc(..., outline_only=True)`, and asserts:

  * The manifest `<lom> <general> <description>` text carries the
    `"[OUTLINE] "` prefix.
  * The manifest organization tree excludes content / application /
    self_check / discussion entries (only overview + summary survive).
  * The IMSCC zip payload mirrors the manifest filter — only overview /
    summary HTML appear in the zip's namelist.
  * Backward-compat: with `outline_only=False` (default), all six pages
    survive in both manifest and zip, and the description has NO
    `[OUTLINE] ` prefix.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


_PAGE_NAMES = [
    "week_01_overview.html",
    "week_01_content_01_intro.html",
    "week_01_application.html",
    "week_01_self_check.html",
    "week_01_summary.html",
    "week_01_discussion.html",
]


@pytest.fixture
def content_dir(tmp_path: Path) -> Path:
    """Minimal content-dir layout exercising every page type."""
    week = tmp_path / "week_01"
    week.mkdir()
    for name in _PAGE_NAMES:
        # Page bodies are intentionally minimal — the packager only
        # cares about filenames + an <h1> for title extraction. The
        # overview <h1> follows the `Week N Overview: Title` shape so
        # `_extract_week_title` produces a sensible label.
        if name.endswith("overview.html"):
            (week / name).write_text(
                "<html><head></head><body>"
                "<h1>Week 1 Overview: Foundations</h1>"
                "</body></html>",
                encoding="utf-8",
            )
        else:
            (week / name).write_text(
                f"<html><head></head><body><h1>{name}</h1></body></html>",
                encoding="utf-8",
            )
    return tmp_path


class TestBuildManifestOutlineOnly:
    def test_outline_prefix_on_description(self, content_dir):
        from package_multifile_imscc import build_manifest  # noqa: E402

        xml = build_manifest(
            content_dir, "OUTLINE_101", "Outline Test", outline_only=True,
        )
        assert "[OUTLINE] " in xml, (
            "outline_only=True must prefix the LOM general description with "
            "`[OUTLINE] ` so LMS-side viewers can detect outline-tier shape"
        )

    def test_no_outline_prefix_when_default(self, content_dir):
        """Backward compat: default (outline_only=False) leaves description as-is."""
        from package_multifile_imscc import build_manifest  # noqa: E402

        xml = build_manifest(content_dir, "OUTLINE_101", "Outline Test")
        assert "[OUTLINE] " not in xml

    def test_outline_only_excludes_content_pages(self, content_dir):
        from package_multifile_imscc import build_manifest  # noqa: E402

        xml = build_manifest(
            content_dir, "OUTLINE_101", "Outline Test", outline_only=True,
        )
        # The dropped pages must NOT appear as href / resource entries.
        assert "week_01_content_01_intro.html" not in xml
        assert "week_01_application.html" not in xml
        assert "week_01_self_check.html" not in xml
        assert "week_01_discussion.html" not in xml
        # Outline pages survive.
        assert "week_01_overview.html" in xml
        assert "week_01_summary.html" in xml

    def test_full_mode_includes_all_pages(self, content_dir):
        """Without the flag, all six pages should be present in manifest."""
        from package_multifile_imscc import build_manifest  # noqa: E402

        xml = build_manifest(content_dir, "OUTLINE_101", "Outline Test")
        for name in _PAGE_NAMES:
            assert name in xml, f"page {name} missing from default-mode manifest"


class TestPackageImsccOutlineOnly:
    def test_outline_only_zip_payload_excludes_dropped_pages(
        self, content_dir, tmp_path
    ):
        from package_multifile_imscc import package_imscc  # noqa: E402

        out = tmp_path / "outline.imscc"
        package_imscc(
            content_dir, out, "OUTLINE_101", "Outline Test",
            skip_validation=True,
            outline_only=True,
        )
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        # Dropped pages absent from zip.
        for dropped in (
            "week_01/week_01_content_01_intro.html",
            "week_01/week_01_application.html",
            "week_01/week_01_self_check.html",
            "week_01/week_01_discussion.html",
        ):
            assert dropped not in names, (
                f"outline_only zip payload must exclude {dropped}; got {names}"
            )
        # Outline surfaces survive.
        assert "week_01/week_01_overview.html" in names
        assert "week_01/week_01_summary.html" in names
        # Manifest still present.
        assert "imsmanifest.xml" in names

    def test_default_mode_zip_payload_includes_all_pages(
        self, content_dir, tmp_path
    ):
        from package_multifile_imscc import package_imscc  # noqa: E402

        out = tmp_path / "full.imscc"
        package_imscc(
            content_dir, out, "OUTLINE_101", "Outline Test",
            skip_validation=True,
        )
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        for name in _PAGE_NAMES:
            assert f"week_01/{name}" in names, (
                f"default-mode zip payload must include {name}; got {names}"
            )

    def test_outline_only_manifest_carries_outline_prefix_in_zip(
        self, content_dir, tmp_path
    ):
        """End-to-end: the manifest stored in the zip carries `[OUTLINE] `."""
        from package_multifile_imscc import package_imscc  # noqa: E402

        out = tmp_path / "outline.imscc"
        package_imscc(
            content_dir, out, "OUTLINE_101", "Outline Test",
            skip_validation=True,
            outline_only=True,
        )
        with zipfile.ZipFile(out) as zf:
            manifest = zf.read("imsmanifest.xml").decode("utf-8")
        assert "[OUTLINE] " in manifest
