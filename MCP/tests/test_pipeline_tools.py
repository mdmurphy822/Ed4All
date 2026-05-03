"""Smoke tests for ``MCP.tools.pipeline_tools`` registry helpers.

Phase 6 ST 12 — exercises the new ``_run_concept_extraction`` helper
registered by ``_build_tool_registry`` to confirm it:

  * Reads DART staging output (``*_synthesized.json`` sidecars).
  * Persists ``concept_graph_semantic.json`` + ``manifest.json`` under
    ``LibV2/courses/<slug>/concept_graph/``.
  * Emits a SHA-256 hex digest of the graph bytes.
  * Returns the canonical ``concept_graph_path`` /
    ``concept_graph_sha256`` keys the workflow runner threads through
    ``phase_outputs.concept_extraction``.

The helper is a pure file-IO + ``Trainforge.pedagogy_graph_builder``
dispatch path (no LLM, no network), so the test is fast (~50 ms) and
fully hermetic via ``tmp_path`` + monkeypatched ``_PROJECT_ROOT``.
"""

from __future__ import annotations

import asyncio
import hashlib
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


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _write_synthesized(path: Path, slug: str) -> None:
    """Emit a minimal DART ``*_synthesized.json`` sidecar at ``path``."""
    doc = {
        "campus_code": slug,
        "campus_name": slug.replace("_", " ").title(),
        "sections": [
            {
                "section_id": "intro",
                "section_type": "overview",
                "section_title": "Introduction to Pedagogical Concepts",
                "data": {
                    "paragraphs": [
                        "Pedagogical concepts include alignment, "
                        "assessment, scaffolding, learning outcomes, "
                        "and curriculum design."
                    ]
                },
            },
            {
                "section_id": "scaffold",
                "section_type": "content",
                "section_title": "Scaffolding Strategies",
                "data": {
                    "paragraphs": [
                        "Scaffolding strategies provide structured "
                        "support during initial learning, gradually "
                        "fading as competence develops."
                    ]
                },
            },
            {
                "section_id": "assess",
                "section_type": "self_check",
                "section_title": "Assessment Check",
                "data": {
                    "paragraphs": [
                        "Formative assessment validates learner "
                        "understanding before summative evaluation."
                    ]
                },
            },
        ],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


@pytest.fixture
def concept_extraction_fixture(tmp_path, monkeypatch):
    """Build a minimal DART staging dir + fake LibV2 root."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    # Redirect _PROJECT_ROOT + COURSEFORGE_INPUTS so the helper's
    # ``LibV2/courses/<slug>/concept_graph/...`` write lands in tmp_path
    # instead of the real repo.
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    # Staging dir with one synthesized sidecar.
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_synthesized(staging / "demo_textbook_synthesized.json", "demo_textbook")

    return {
        "fake_root": fake_root,
        "staging_dir": staging,
        "course_name": "DEMO_303",
        "course_slug": "demo-303",
    }


def _invoke(course_name: str, staging_dir: Path) -> dict:
    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name=course_name,
            staging_dir=str(staging_dir),
        )
    )
    return json.loads(result)


class TestRunConceptExtractionEmitsGraph:
    def test_run_concept_extraction_emits_graph(self, concept_extraction_fixture):
        """ST 12 plan-cited verification — helper writes a graph file."""
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        assert payload["success"] is True
        assert "concept_graph_path" in payload
        assert "concept_graph_sha256" in payload

        graph_path = Path(payload["concept_graph_path"])
        assert graph_path.exists(), (
            f"concept_graph_semantic.json not written at {graph_path}"
        )
        assert graph_path.name == "concept_graph_semantic.json"

        # Path lands under LibV2/courses/<slug>/concept_graph/.
        rel = graph_path.relative_to(fx["fake_root"])
        parts = rel.parts
        assert parts[0] == "LibV2"
        assert parts[1] == "courses"
        assert parts[2] == fx["course_slug"]
        assert parts[3] == "concept_graph"

    def test_sha256_matches_file_bytes(self, concept_extraction_fixture):
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        assert _SHA256_RE.match(payload["concept_graph_sha256"]), (
            f"sha256 not in canonical hex shape: {payload['concept_graph_sha256']!r}"
        )

        graph_path = Path(payload["concept_graph_path"])
        on_disk_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        assert on_disk_hash == payload["concept_graph_sha256"], (
            "Returned sha256 must match on-disk graph bytes."
        )

    def test_manifest_emitted(self, concept_extraction_fixture):
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        manifest_path = Path(payload["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["course_slug"] == fx["course_slug"]
        assert manifest["concept_graph_sha256"] == payload["concept_graph_sha256"]
        assert manifest["phase"] == "concept_extraction"
        assert manifest["source_chunks"] == payload["chunk_count"]

    def test_graph_has_expected_typed_nodes(self, concept_extraction_fixture):
        """build_pedagogy_graph always emits BloomLevel + DifficultyLevel
        typed nodes regardless of input — verify the dispatch landed."""
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        graph = json.loads(
            Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
        )
        node_classes = {n.get("class") for n in graph.get("nodes", [])}
        assert "BloomLevel" in node_classes, (
            "build_pedagogy_graph should emit BloomLevel typed nodes."
        )
        assert "DifficultyLevel" in node_classes, (
            "build_pedagogy_graph should emit DifficultyLevel typed nodes."
        )

    def test_chunks_derived_from_staging(self, concept_extraction_fixture):
        """Three sections in the fixture sidecar -> 3 chunks projected."""
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        # Three sections in the fixture (intro / scaffold / assess).
        assert payload["chunk_count"] == 3

    def test_empty_staging_emits_shell_graph(self, tmp_path, monkeypatch):
        """When no sidecars exist, helper still emits a graph shell so
        downstream gates have something to validate against."""
        fake_root = tmp_path / "root"
        fake_root.mkdir()
        monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
        monkeypatch.setattr(
            pipeline_tools,
            "COURSEFORGE_INPUTS",
            fake_root / "Courseforge" / "inputs" / "textbooks",
        )
        (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

        empty_staging = tmp_path / "empty_staging"
        empty_staging.mkdir()

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name="EMPTY_001",
                staging_dir=str(empty_staging),
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["chunk_count"] == 0
        # BloomLevel + DifficultyLevel typed nodes always emit.
        assert payload["node_count"] >= 6
