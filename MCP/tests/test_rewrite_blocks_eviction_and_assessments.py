"""Pipeline lane Q1 + Q4 — rewrite ``--blocks`` type-eviction + assessments archival.

Two independent runtime seams landed in ``MCP/tools/pipeline_tools.py``:

* Q1: ``_evict_rewrite_cache_by_block_type`` — the ADDITIVE type-eviction the
  ``content_generation_rewrite`` phase applies when ``--blocks`` names one or
  more block TYPES. Cached (previously-successful) blocks of a named type are
  dropped so they re-author; every other type keeps byte-identical reuse. The
  unset-flag path (empty target set) is a byte-identical no-op.

* Q4: ``_copy_export_assessments`` — copies a Courseforge export's
  ``06_assessments/`` (QTI/imsdt/assignment XML + manifest) into the LibV2
  course dir so ``gui/services/quiz_service.py`` (which scans
  ``LibV2/courses/<slug>/06_assessments/*.xml``) resolves them. Absent dir →
  no-op, no failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _copy_export_assessments,
    _evict_rewrite_cache_by_block_type,
)


def _blk(block_id: str, block_type: str) -> SimpleNamespace:
    """Minimal duck-typed stand-in for a Courseforge Block."""
    return SimpleNamespace(block_id=block_id, block_type=block_type)


# --------------------------------------------------------------------------- #
# Q1 — _evict_rewrite_cache_by_block_type
# --------------------------------------------------------------------------- #


class TestRewriteCacheEviction:
    def _fixture(self):
        blocks = [
            _blk("b1", "objective"),
            _blk("b2", "concept"),
            _blk("b3", "objective"),
            _blk("b4", "example"),
        ]
        cache = {
            "b1": {"block_id": "b1", "content": "<p>obj1</p>"},
            "b2": {"block_id": "b2", "content": "<p>concept</p>"},
            "b3": {"block_id": "b3", "content": "<p>obj2</p>"},
            "b4": {"block_id": "b4", "content": "<p>example</p>"},
        }
        return blocks, cache

    def test_unset_is_byte_identical_noop(self):
        """Empty target set → nothing evicted; cache is untouched."""
        blocks, cache = self._fixture()
        before = dict(cache)
        removed = _evict_rewrite_cache_by_block_type(cache, blocks, set())
        assert removed == 0
        assert cache == before

    def test_evicts_only_named_type(self):
        """Only ``objective`` blocks are dropped; the rest stay cached."""
        blocks, cache = self._fixture()
        removed = _evict_rewrite_cache_by_block_type(
            cache, blocks, {"objective"}
        )
        assert removed == 2
        assert "b1" not in cache and "b3" not in cache
        # Non-targeted types keep byte-identical reuse.
        assert cache["b2"] == {"block_id": "b2", "content": "<p>concept</p>"}
        assert cache["b4"] == {"block_id": "b4", "content": "<p>example</p>"}

    def test_multiple_types_evicted(self):
        blocks, cache = self._fixture()
        removed = _evict_rewrite_cache_by_block_type(
            cache, blocks, {"objective", "example"}
        )
        assert removed == 3
        assert set(cache) == {"b2"}

    def test_type_not_in_cache_is_noop(self):
        """A named type with no cached block evicts nothing (additive-only)."""
        blocks, cache = self._fixture()
        removed = _evict_rewrite_cache_by_block_type(
            cache, blocks, {"assessment_item"}
        )
        assert removed == 0
        assert len(cache) == 4

    def test_empty_cache_is_noop(self):
        blocks, _ = self._fixture()
        cache: dict = {}
        removed = _evict_rewrite_cache_by_block_type(
            cache, blocks, {"objective"}
        )
        assert removed == 0
        assert cache == {}

    def test_never_adds_entries(self):
        """Eviction is subtractive only — never introduces a new key."""
        blocks, cache = self._fixture()
        # A block whose id is NOT in the cache is targeted; nothing added.
        blocks.append(_blk("b5", "objective"))
        _evict_rewrite_cache_by_block_type(cache, blocks, {"objective"})
        assert "b5" not in cache


# --------------------------------------------------------------------------- #
# Q4 — _copy_export_assessments
# --------------------------------------------------------------------------- #


class TestCopyExportAssessments:
    def _make_export_assessments(self, export_dir: Path) -> None:
        adir = export_dir / "06_assessments"
        adir.mkdir(parents=True)
        (adir / "quiz_week1.xml").write_text(
            "<questestinterop></questestinterop>", encoding="utf-8"
        )
        (adir / "discussion_week1.xml").write_text(
            "<topic></topic>", encoding="utf-8"
        )
        (adir / "manifest.json").write_text(
            json.dumps({"schema_version": "v1", "assessments": []}),
            encoding="utf-8",
        )
        # A non-assessment sibling that MUST NOT be copied.
        (adir / "notes.txt").write_text("scratch", encoding="utf-8")

    def test_copies_xml_and_manifest(self, tmp_path):
        export_dir = tmp_path / "export"
        self._make_export_assessments(export_dir)
        course_dir = tmp_path / "courses" / "slug"
        course_dir.mkdir(parents=True)

        copied = _copy_export_assessments([export_dir], course_dir)

        assert copied == 3  # two xml + manifest.json (not notes.txt)
        dest = course_dir / "06_assessments"
        assert (dest / "quiz_week1.xml").exists()
        assert (dest / "discussion_week1.xml").exists()
        assert (dest / "manifest.json").exists()
        assert not (dest / "notes.txt").exists()

    def test_first_candidate_with_dir_wins(self, tmp_path):
        empty = tmp_path / "empty_export"
        empty.mkdir()
        real = tmp_path / "real_export"
        self._make_export_assessments(real)
        course_dir = tmp_path / "courses" / "slug"
        course_dir.mkdir(parents=True)

        # ``empty`` has no 06_assessments → the resolver skips to ``real``.
        copied = _copy_export_assessments([empty, real], course_dir)
        assert copied == 3
        assert (course_dir / "06_assessments" / "quiz_week1.xml").exists()

    def test_absent_dir_is_noop(self, tmp_path):
        """No candidate carries 06_assessments → 0 copied, nothing written."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()  # no 06_assessments/ inside
        course_dir = tmp_path / "courses" / "slug"
        course_dir.mkdir(parents=True)

        copied = _copy_export_assessments([export_dir], course_dir)
        assert copied == 0
        assert not (course_dir / "06_assessments").exists()

    def test_empty_candidate_list_is_noop(self, tmp_path):
        course_dir = tmp_path / "courses" / "slug"
        course_dir.mkdir(parents=True)
        assert _copy_export_assessments([], course_dir) == 0
        assert not (course_dir / "06_assessments").exists()


# --------------------------------------------------------------------------- #
# Q4 — end-to-end: the live registry archive tool resolves the export from
#      project_id and copies 06_assessments into the LibV2 course dir.
# --------------------------------------------------------------------------- #


class TestArchiveToLibv2AssessmentsWiring:
    def test_registry_archive_copies_export_assessments(
        self, tmp_path, monkeypatch
    ):
        import asyncio

        from lib.ontology.slugs import libv2_course_slug

        import MCP.tools.pipeline_tools as pt

        # Redirect BOTH project-root seams so LibV2 + Courseforge exports
        # resolve under tmp_path (courseforge_exports_dir honours whichever
        # is patched away from the import-time default).
        monkeypatch.setattr(pt, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(pt, "_PROJECT_ROOT", tmp_path)

        project_id = "PROJ-QUIZWIRE"
        adir = (
            tmp_path / "Courseforge" / "exports" / project_id / "06_assessments"
        )
        adir.mkdir(parents=True)
        (adir / "quiz.xml").write_text(
            "<questestinterop></questestinterop>", encoding="utf-8"
        )
        (adir / "manifest.json").write_text("{}", encoding="utf-8")

        registry = pt._build_tool_registry()
        archive = registry["archive_to_libv2"]
        result = json.loads(
            asyncio.run(
                archive(
                    course_name="FXQUIZ_101",
                    project_id=project_id,
                    domain="general",
                )
            )
        )
        assert result.get("success") is True, result

        slug = libv2_course_slug("FXQUIZ_101")
        dest = tmp_path / "LibV2" / "courses" / slug / "06_assessments"
        assert dest.is_dir()
        assert (dest / "quiz.xml").exists()
        assert (dest / "manifest.json").exists()

    def test_registry_archive_noop_without_assessments(
        self, tmp_path, monkeypatch
    ):
        import asyncio

        from lib.ontology.slugs import libv2_course_slug

        import MCP.tools.pipeline_tools as pt

        monkeypatch.setattr(pt, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(pt, "_PROJECT_ROOT", tmp_path)

        project_id = "PROJ-NOQUIZ"
        # Export dir exists but has NO 06_assessments/.
        (tmp_path / "Courseforge" / "exports" / project_id).mkdir(parents=True)

        registry = pt._build_tool_registry()
        archive = registry["archive_to_libv2"]
        result = json.loads(
            asyncio.run(
                archive(
                    course_name="FXNOQUIZ_101",
                    project_id=project_id,
                    domain="general",
                )
            )
        )
        assert result.get("success") is True, result

        slug = libv2_course_slug("FXNOQUIZ_101")
        dest = tmp_path / "LibV2" / "courses" / slug / "06_assessments"
        assert not dest.exists()


# --------------------------------------------------------------------------- #
# Q1 (GUI seam) — gui/services/run_service._normalize_blocks_param
#
# Lives here (not gui/tests) because this lane owns MCP/tests; the helper is a
# backend run-service seam that turns the studio failure panel's ``blocks`` CSV
# option into the same ``target_block_ids`` param the pipeline routing consumes.
# --------------------------------------------------------------------------- #


class TestGuiBlocksNormalization:
    def test_csv_string_becomes_target_block_ids(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"course_name": "X", "blocks": "objective,concept"}
        _normalize_blocks_param(wp)
        assert wp["target_block_ids"] == ["objective", "concept"]
        # The dead ``blocks`` key is consumed.
        assert "blocks" not in wp

    def test_dedupes_and_strips(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"blocks": " objective , concept , objective ,"}
        _normalize_blocks_param(wp)
        assert wp["target_block_ids"] == ["objective", "concept"]

    def test_list_form_accepted(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"blocks": ["objective", "example"]}
        _normalize_blocks_param(wp)
        assert wp["target_block_ids"] == ["objective", "example"]

    def test_absent_blocks_is_noop(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"course_name": "X"}
        _normalize_blocks_param(wp)
        assert "target_block_ids" not in wp

    def test_empty_string_is_noop(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"blocks": "   "}
        _normalize_blocks_param(wp)
        assert "target_block_ids" not in wp
        assert "blocks" not in wp

    def test_explicit_target_block_ids_wins(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"blocks": "objective", "target_block_ids": ["concept"]}
        _normalize_blocks_param(wp)
        # Existing explicit param is not overwritten.
        assert wp["target_block_ids"] == ["concept"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
