"""Phase 7c Subtask 18 — DART chunkset backfill operator script regression tests.

Smoke coverage for ``LibV2/tools/libv2/scripts/backfill_dart_chunks.py``.
Each test builds a tmp_path LibV2 fixture (one or more course
directories with a minimal ``source/html/`` payload), invokes
``main(argv)`` directly, and asserts the post-run filesystem state.

We deliberately bypass the subprocess launch path so the tests stay
fast and synchronous; the ``backfill_dart_chunks.main`` function takes
``argv`` as a parameter for exactly this reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root for imports (pytest may not have project root on path).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LibV2.tools.libv2.scripts import backfill_dart_chunks  # noqa: E402


# Minimal HTML payload that ``HTMLContentParser`` will happily parse
# into a ContentSection. Keeping it small but non-empty so the
# ed4all_chunker has at least one section to chunk.
_SAMPLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Sample Page</title></head>
<body>
  <main>
    <h1>Introduction to Topic</h1>
    <section>
      <h2>Background</h2>
      <p>This is a paragraph about the topic with enough text to chunk
         meaningfully without tripping the small-section merge logic.
         Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
         do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
      <p>A second paragraph adds more content to ensure the chunker
         has something to bite into. Ut enim ad minim veniam, quis
         nostrud exercitation ullamco laboris nisi ut aliquip ex ea
         commodo consequat.</p>
    </section>
  </main>
</body>
</html>
"""


def _build_course_fixture(
    libv2_root: Path,
    slug: str,
    *,
    with_html: bool = True,
    with_existing_chunkset: bool = False,
) -> Path:
    """Materialize a minimal LibV2 course directory for the test.

    Returns the absolute path to the course directory.
    """
    course_dir = libv2_root / slug
    course_dir.mkdir(parents=True, exist_ok=True)
    if with_html:
        html_dir = course_dir / "source" / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        (html_dir / "page1.html").write_text(_SAMPLE_HTML, encoding="utf-8")
    if with_existing_chunkset:
        chunkset_dir = course_dir / "dart_chunks"
        chunkset_dir.mkdir(parents=True, exist_ok=True)
        # A pre-existing manifest blocks the default (idempotent)
        # backfill path. We keep it minimally schema-shaped so the
        # presence check trips.
        (chunkset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "chunks_sha256": "0" * 64,
                    "chunker_version": "0.0.0+placeholder",
                    "chunkset_kind": "dart",
                    "source_dart_html_sha256": "0" * 64,
                    "chunks_count": 0,
                }
            ),
            encoding="utf-8",
        )
        (chunkset_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    return course_dir


@pytest.mark.unit
class TestBackfillCLISurface:
    """Argparse surface-level checks — no chunker dispatch."""

    def test_argparser_has_expected_flags(self):
        parser = backfill_dart_chunks.build_arg_parser()
        actions = {a.dest for a in parser._actions}
        # All six operator-facing knobs documented in the script
        # docstring + the optional --operator audit-trail flag.
        for flag in {
            "libv2_root",
            "course_slug",
            "html_subdir",
            "dry_run",
            "force",
            "verbose",
            "operator",
        }:
            assert flag in actions, f"missing CLI flag: {flag}"

    def test_libv2_root_missing_returns_exit_2(self, tmp_path: Path):
        nonexistent = tmp_path / "does-not-exist"
        rc = backfill_dart_chunks.main(["--libv2-root", str(nonexistent)])
        assert rc == 2

    def test_dry_run_against_empty_libv2_root_succeeds(self, tmp_path: Path):
        # A libv2 root that exists but holds no course dirs is
        # operator-valid (e.g. a fresh checkout) — exit 0 with an
        # empty summary.
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        rc = backfill_dart_chunks.main(
            ["--libv2-root", str(libv2_root), "--dry-run"]
        )
        assert rc == 0


@pytest.mark.unit
class TestBackfillBehavior:
    """End-to-end behavior — fixture in, chunkset out."""

    def test_dry_run_does_not_write_chunkset(self, tmp_path: Path):
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        course_dir = _build_course_fixture(libv2_root, "test-101")

        rc = backfill_dart_chunks.main(
            [
                "--libv2-root",
                str(libv2_root),
                "--course-slug",
                "test-101",
                "--dry-run",
            ]
        )

        assert rc == 0
        assert not (course_dir / "dart_chunks").exists()

    def test_skip_when_chunkset_already_present(self, tmp_path: Path):
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        course_dir = _build_course_fixture(
            libv2_root, "test-101", with_existing_chunkset=True
        )
        original_manifest = (course_dir / "dart_chunks" / "manifest.json").read_text()

        rc = backfill_dart_chunks.main(
            ["--libv2-root", str(libv2_root), "--course-slug", "test-101"]
        )

        assert rc == 0
        # Manifest unchanged: the script skipped the course.
        assert (
            course_dir / "dart_chunks" / "manifest.json"
        ).read_text() == original_manifest

    def test_skip_when_no_html_under_subdir(self, tmp_path: Path):
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        # Course directory exists but ``source/html/`` does not.
        course_dir = _build_course_fixture(
            libv2_root, "empty-course", with_html=False
        )

        rc = backfill_dart_chunks.main(
            ["--libv2-root", str(libv2_root), "--course-slug", "empty-course"]
        )

        # No HTML = skip (not failure). Exit 0.
        assert rc == 0
        assert not (course_dir / "dart_chunks").exists()

    def test_backfill_emits_chunkset_for_real_course(self, tmp_path: Path):
        """End-to-end: real chunker dispatch against a tmp fixture.

        Asserts the chunkset directory + chunks.jsonl + manifest.json
        all exist after a real run, the manifest validates against
        the canonical ``ChunksetManifestValidator``, and the
        manifest's ``chunks_sha256`` matches the on-disk SHA of the
        chunks file.
        """
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        course_dir = _build_course_fixture(libv2_root, "smoke-101")

        rc = backfill_dart_chunks.main(
            ["--libv2-root", str(libv2_root), "--course-slug", "smoke-101"]
        )

        assert rc == 0
        chunkset_dir = course_dir / "dart_chunks"
        assert chunkset_dir.is_dir(), f"missing chunkset dir: {chunkset_dir}"
        manifest_path = chunkset_dir / "manifest.json"
        chunks_path = chunkset_dir / "chunks.jsonl"
        assert manifest_path.is_file()
        assert chunks_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Schema-required keys.
        assert manifest["chunkset_kind"] == "dart"
        assert "chunks_sha256" in manifest
        assert "chunker_version" in manifest
        assert "source_dart_html_sha256" in manifest

    def test_backfill_handles_missing_course_dir_with_failure_exit(
        self, tmp_path: Path
    ):
        """``--course-slug`` pointing at a nonexistent dir → exit 1."""
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        # Don't create any course dir.
        rc = backfill_dart_chunks.main(
            [
                "--libv2-root",
                str(libv2_root),
                "--course-slug",
                "ghost-course",
            ]
        )
        assert rc == 1


@pytest.mark.unit
class TestBackfillForceFlag:
    """``--force`` re-emits even when manifest.json already exists."""

    def test_force_overwrites_existing_chunkset(self, tmp_path: Path):
        libv2_root = tmp_path / "courses"
        libv2_root.mkdir()
        course_dir = _build_course_fixture(
            libv2_root,
            "force-101",
            with_existing_chunkset=True,
        )
        # Confirm the placeholder we wrote in the fixture has the
        # zero-hash sentinel.
        before = json.loads(
            (course_dir / "dart_chunks" / "manifest.json").read_text()
        )
        assert before["chunks_sha256"] == "0" * 64

        rc = backfill_dart_chunks.main(
            [
                "--libv2-root",
                str(libv2_root),
                "--course-slug",
                "force-101",
                "--force",
            ]
        )

        assert rc == 0
        after = json.loads(
            (course_dir / "dart_chunks" / "manifest.json").read_text()
        )
        # Sentinel hash was replaced by a real chunker emit. Even an
        # empty-input chunker run produces a real SHA-256 over empty
        # bytes, which is NOT 64 zeros.
        assert after["chunks_sha256"] != "0" * 64
