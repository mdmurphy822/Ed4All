"""Source-router heuristic tests.

Validates ``_build_source_module_map`` produces non-empty, well-shaped
``source_module_map.json`` output (investigation Issue 7). Previous
behavior: unconditional empty-dict emit, which pinned Wave 10/11
provenance flags to false and dropped every ``sourceReferences[]``
entry emitted by Courseforge.

This test builds a minimal fixture:

  * A staging dir with two ``*_synthesized.json`` sidecars that carry
    realistic ``sections[]`` entries (the Wave 8 shape documented in
    ``DART/CLAUDE.md`` § Source provenance).
  * A Courseforge project dir containing a ``project_config.json`` so
    the router can discover ``duration_weeks`` and ``course_name``.
  * No textbook_structure / objectives file — forces the fallback to
    DART-driven topic bags, which is the worst-case path in the
    heuristic. If even this path produces populated refs, better inputs
    will too.

Assertions cover:
  * ``source_module_map.json`` is written and non-empty.
  * Output dict keys are ``week_NN`` strings.
  * Each week has at least one page entry with a ``primary`` ref list.
  * Refs conform to the ``semantik:{slug}#{block_id}`` shape (dual-read still
    accepts the legacy ``dart:`` prefix until the Stage-4 tighten).
  * ``routing_mode`` reports ``keyword_overlap_heuristic`` when DART
    blocks were indexed.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

# SemantiK source-reference canonical shape: semantik:{slug}#{block_id}
# (dual-read: the legacy dart: prefix is still accepted until the Stage-4 tighten).
_SOURCE_ID_RE = re.compile(r"^(?:dart|semantik):[a-z0-9_\-]+#[A-Za-z0-9_]+$")


def _write_synthesized(path: Path, slug: str, sections: list) -> None:
    doc = {
        "campus_code": slug,
        "campus_name": slug.replace("_", " ").title(),
        "sections": sections,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_project_config(project_dir: Path, course_name: str,
                          duration_weeks: int = 4) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "project_id": project_dir.name,
        "course_name": course_name,
        "duration_weeks": duration_weeks,
        "objectives_path": None,
        "credit_hours": 3,
        "status": "initialized",
    }
    (project_dir / "project_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )


@pytest.fixture
def source_router_fixture(tmp_path, monkeypatch):
    """Build a minimal DART staging + Courseforge project dir."""
    # Redirect PROJECT_ROOT targets so we don't pollute the real repo.
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    exports = fake_root / "Courseforge" / "exports"
    exports.mkdir(parents=True)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS",
                        fake_root / "Courseforge" / "inputs" / "textbooks")
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    project_id = "PROJ-TEST-001"
    project_dir = exports / project_id
    _write_project_config(project_dir, course_name="DEMO_101",
                          duration_weeks=4)

    # Staging: two textbooks' worth of synthesized sidecars.
    staging = tmp_path / "staging"
    staging.mkdir()

    _write_synthesized(staging / "textbook_a_synthesized.json", "textbook_a", [
        {
            "section_id": "s1",
            "section_type": "overview",
            "section_title": "Introduction to Core Concepts",
            "page_range": [1, 3],
            "provenance": {"sources": ["pdftotext"], "strategy": "text_only"},
            "data": {
                "paragraphs": [
                    "This chapter introduces the foundational concepts, "
                    "including key terminology, relevant frameworks, and the "
                    "methodology that will be applied throughout the text."
                ]
            },
        },
        {
            "section_id": "s2",
            "section_type": "content",
            "section_title": "Curriculum Reform Strategies",
            "page_range": [4, 9],
            "data": {
                "paragraphs": [
                    "Curriculum reform strategies target pedagogy, assessment, "
                    "and teacher preparation simultaneously."
                ]
            },
        },
        {
            "section_id": "s3",
            "section_type": "content",
            "section_title": "Assessment Transformation in Schools",
            "page_range": [10, 15],
            "data": {
                "paragraphs": [
                    "Assessment transformation replaces summative testing "
                    "with formative, competency-based evaluation methods."
                ]
            },
        },
    ])

    _write_synthesized(staging / "textbook_design_synthesized.json", "textbook_design", [
        {
            "section_id": "s1",
            "section_type": "content",
            "section_title": "Online Teaching Foundations",
            "page_range": [1, 5],
            "data": {
                "paragraphs": [
                    "Online teaching foundations demand new pedagogy, new "
                    "assessment models, and new teacher competencies."
                ]
            },
        },
        {
            "section_id": "s4",
            "section_type": "content",
            "section_title": "Online Learning Design Principles",
            "page_range": [6, 12],
            "data": {
                "paragraphs": [
                    "Online learning design principles include cognitive load "
                    "management, interaction patterns, and feedback cycles."
                ]
            },
        },
    ])

    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "staging_dir": staging,
    }


def _invoke_router(project_id: str, staging_dir: Path,
                   textbook_structure_path: str = "") -> dict:
    registry = _build_tool_registry()
    tool = registry["build_source_module_map"]
    result = asyncio.run(tool(
        project_id=project_id,
        staging_dir=str(staging_dir),
        textbook_structure_path=textbook_structure_path,
    ))
    return json.loads(result)


class TestMapIsPopulated:
    def test_source_module_map_file_written(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        map_path = Path(payload["source_module_map_path"])
        assert map_path.exists(), "source_module_map.json not written"
        doc = json.loads(map_path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict)

    def test_source_module_map_is_non_empty(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        map_path = Path(payload["source_module_map_path"])
        doc = json.loads(map_path.read_text(encoding="utf-8"))
        assert doc, (
            "source_module_map.json is empty — heuristic failed to "
            "route any DART blocks to Courseforge pages."
        )

    def test_heuristic_routing_mode_reported(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        assert payload["routing_mode"] == "keyword_overlap_heuristic"
        assert payload["dart_blocks_indexed"] >= 5, (
            "Expected at least 5 DART blocks indexed from the two "
            "synthesized sidecars."
        )
        assert payload["weeks_routed"] >= 1


class TestMapShape:
    def test_keys_are_week_nn_strings(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        doc = json.loads(Path(payload["source_module_map_path"]).read_text())
        for week_key in doc.keys():
            assert re.match(r"^week_\d{2}$", week_key), (
                f"Invalid week key: {week_key!r}. Expected 'week_NN'."
            )

    def test_pages_have_primary_refs(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        doc = json.loads(Path(payload["source_module_map_path"]).read_text())
        pages_with_primary = 0
        for week_entries in doc.values():
            for page_id, entry in week_entries.items():
                assert "primary" in entry
                assert isinstance(entry["primary"], list)
                if entry["primary"]:
                    pages_with_primary += 1
        assert pages_with_primary > 0, (
            "No page had any primary refs — provenance chain broken."
        )

    def test_refs_match_dart_source_id_shape(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        doc = json.loads(Path(payload["source_module_map_path"]).read_text())
        all_ids: list = []
        for week_entries in doc.values():
            for entry in week_entries.values():
                all_ids.extend(entry.get("primary") or [])
                all_ids.extend(entry.get("contributing") or [])
        assert all_ids, "No source IDs produced."
        for sid in all_ids:
            assert _SOURCE_ID_RE.match(sid), (
                f"Source id {sid!r} does not match "
                f"'dart:{{slug}}#{{block_id}}' shape."
            )

    def test_confidence_is_a_float_in_unit_interval(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        doc = json.loads(Path(payload["source_module_map_path"]).read_text())
        for week_entries in doc.values():
            for entry in week_entries.values():
                conf = entry.get("confidence")
                assert isinstance(conf, (int, float))
                assert 0.0 <= float(conf) <= 1.0


class TestChunkIdsExposed:
    def test_source_chunk_ids_deduplicated(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        ids = payload["source_chunk_ids"]
        assert isinstance(ids, list)
        assert len(set(ids)) == len(ids), "source_chunk_ids has duplicates"

    def test_source_chunk_ids_subset_of_map(self, source_router_fixture):
        fx = source_router_fixture
        payload = _invoke_router(fx["project_id"], fx["staging_dir"])
        doc = json.loads(Path(payload["source_module_map_path"]).read_text())
        seen_in_map: set = set()
        for week_entries in doc.values():
            for entry in week_entries.values():
                seen_in_map.update(entry.get("primary") or [])
                seen_in_map.update(entry.get("contributing") or [])
        declared_in_payload = set(payload["source_chunk_ids"])
        assert declared_in_payload == seen_in_map, (
            "source_chunk_ids must exactly enumerate the source IDs "
            "emitted in the map."
        )


class TestDegradedPath:
    """When staging is empty, router must not crash and must report the
    empty-map routing_mode so callers know provenance is unavailable."""

    def test_empty_staging_emits_empty_map_without_error(self,
                                                        source_router_fixture,
                                                        tmp_path):
        fx = source_router_fixture
        empty_staging = tmp_path / "empty_staging"
        empty_staging.mkdir()
        payload = _invoke_router(fx["project_id"], empty_staging)
        assert "error" not in payload
        assert payload["routing_mode"] == "stub_empty_map"
        doc = json.loads(Path(payload["source_module_map_path"]).read_text())
        assert doc == {}


class TestHeadingTextStructureRouting:
    """Regression: SemanticStructureExtractor emits chapter/section headings
    under ``headingText`` (+ ``chapter_text``), not ``title``. The router used
    to read ``title``, so every topic bag tokenized to empty and the whole
    map collapsed to one alphabetically-first source at 0.3 confidence."""

    def _write_structure(self, path: Path) -> None:
        # Chapters keyed by ``headingText`` matching the two fixture
        # textbooks' distinct topics (Core Concepts / Curriculum / Assessment
        # vs Online Teaching). NO ``title`` key — exactly what the extractor
        # emits.
        doc = {
            "chapters": [
                {
                    "id": "ch1",
                    "headingText": "Curriculum Reform Strategies and Core Concepts",
                    "chapter_text": (
                        "Curriculum reform strategies introduce foundational "
                        "concepts, terminology, frameworks and methodology."
                    ),
                    "sections": [
                        {"id": "s1a", "headingText": "Assessment Transformation"},
                    ],
                    "source_file": "textbook_a.html",
                },
                {
                    "id": "ch2",
                    "headingText": "Online Teaching Foundations and Learning Design",
                    "chapter_text": (
                        "Online teaching foundations cover learning design "
                        "principles, interaction patterns and feedback cycles."
                    ),
                    "sections": [],
                    "source_file": "textbook_b.html",
                },
            ]
        }
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_headingtext_structure_does_not_collapse(self, source_router_fixture):
        fx = source_router_fixture
        struct = fx["staging_dir"].parent / "textbook_structure.json"
        self._write_structure(struct)
        payload = _invoke_router(
            fx["project_id"], fx["staging_dir"], textbook_structure_path=str(struct)
        )
        # The headline guarantee: NOT collapsed.
        assert payload["routing_collapsed"] is False
        assert payload["pages_with_primary"] > 0
        # Distinct sources must exceed 1 — the collapse pinned everything to
        # a single alphabetically-first file.
        assert payload["distinct_sources_routed"] >= 2, payload
        assert payload["routing_primary_rate"] > 0.0

    def test_collapse_is_detected_and_flagged(self, source_router_fixture):
        # A structure whose headings tokenize to nothing (empty title key,
        # no headingText) must NOT silently collapse — but if it does fall
        # all the way through with no signal the guard reports it. Here we
        # give chapters with ONLY a legacy empty ``title`` and no sections,
        # forcing the all-empty topic_pool fall-through to DART blocks
        # (which still yields primaries — i.e. the fix prevents collapse).
        fx = source_router_fixture
        struct = fx["staging_dir"].parent / "empty_structure.json"
        struct.write_text(json.dumps({"chapters": [
            {"id": "c1", "title": "", "sections": []},
            {"id": "c2", "title": "", "sections": []},
        ]}), encoding="utf-8")
        payload = _invoke_router(
            fx["project_id"], fx["staging_dir"], textbook_structure_path=str(struct)
        )
        # Fall-through to DART-block keyword bags keeps the router alive.
        assert payload["routing_collapsed"] is False
        assert payload["pages_with_primary"] > 0
