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


# ---------------------------------------------------------------------------
# Phase 7b Subtask 11 — _run_dart_chunking smoke tests
# ---------------------------------------------------------------------------


def _write_dart_html(path: Path, title: str) -> None:
    """Emit a minimal DART-shaped HTML file at ``path``."""
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>This DART HTML file is a fixture for the Phase 7b chunking smoke test. {' '.join(['Chunk content padding sentence.'] * 60)}</p>
      <h2>Sub-section about pedagogy</h2>
      <p>Pedagogy describes the methods and practice of teaching. {' '.join(['Additional padding text to clear the chunker minimum-size threshold.'] * 60)}</p>
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


@pytest.fixture
def dart_chunking_fixture(tmp_path, monkeypatch):
    """Build a minimal DART staging dir + fake LibV2 root."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_dart_html(staging / "chapter_01.html", "Chapter One")
    _write_dart_html(staging / "chapter_02.html", "Chapter Two")

    return {
        "fake_root": fake_root,
        "staging_dir": staging,
        "course_name": "DEMO_777",
        "course_slug": "demo-777",
    }


def _invoke_dart_chunking(course_name: str, staging_dir: Path) -> dict:
    registry = _build_tool_registry()
    tool = registry["run_dart_chunking"]
    result = asyncio.run(
        tool(
            course_name=course_name,
            staging_dir=str(staging_dir),
        )
    )
    return json.loads(result)


class TestRunDartChunkingEmitsChunksJsonl:
    def test_run_dart_chunking_emits_chunks_jsonl(self, dart_chunking_fixture):
        """ST 11 plan-cited verification — helper writes chunks.jsonl
        and a sibling manifest.json under
        ``LibV2/courses/<slug>/dart_chunks/``."""
        fx = dart_chunking_fixture
        payload = _invoke_dart_chunking(fx["course_name"], fx["staging_dir"])

        assert payload["success"] is True
        assert "dart_chunks_path" in payload
        assert "dart_chunks_sha256" in payload

        chunks_path = Path(payload["dart_chunks_path"])
        assert chunks_path.exists(), (
            f"chunks.jsonl not written at {chunks_path}"
        )
        assert chunks_path.name == "chunks.jsonl"

        # Path lands under LibV2/courses/<slug>/dart_chunks/.
        rel = chunks_path.relative_to(fx["fake_root"])
        parts = rel.parts
        assert parts[0] == "LibV2"
        assert parts[1] == "courses"
        assert parts[2] == fx["course_slug"]
        assert parts[3] == "dart_chunks"
        assert parts[4] == "chunks.jsonl"

    def test_dart_chunks_sha256_matches_file_bytes(self, dart_chunking_fixture):
        fx = dart_chunking_fixture
        payload = _invoke_dart_chunking(fx["course_name"], fx["staging_dir"])

        assert _SHA256_RE.match(payload["dart_chunks_sha256"]), (
            f"sha256 not in canonical hex shape: {payload['dart_chunks_sha256']!r}"
        )

        chunks_path = Path(payload["dart_chunks_path"])
        on_disk_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        assert on_disk_hash == payload["dart_chunks_sha256"], (
            "Returned sha256 must match on-disk chunks.jsonl bytes."
        )

    def test_manifest_emitted_and_validates(self, dart_chunking_fixture):
        """Manifest.json is emitted with the canonical chunkset shape
        and passes the ChunksetManifestValidator gate."""
        fx = dart_chunking_fixture
        payload = _invoke_dart_chunking(fx["course_name"], fx["staging_dir"])

        manifest_path = Path(payload["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Required schema fields.
        assert manifest["chunks_sha256"] == payload["dart_chunks_sha256"]
        assert manifest["chunkset_kind"] == "dart"
        assert _SHA256_RE.match(manifest["source_dart_html_sha256"])
        assert isinstance(manifest["chunker_version"], str)
        assert manifest["chunks_count"] == payload["chunks_count"]
        # additionalProperties: false — only the canonical keys.
        assert set(manifest.keys()).issubset({
            "chunks_sha256",
            "chunker_version",
            "chunkset_kind",
            "source_dart_html_sha256",
            "source_imscc_sha256",
            "chunks_count",
            "generated_at",
        })

        # Validator round-trip: the emitted manifest must pass the
        # ChunksetManifestValidator gate (Phase 7b ST 13).
        from lib.validators.chunkset_manifest import ChunksetManifestValidator

        validator = ChunksetManifestValidator()
        result = validator.validate({"chunkset_manifest_path": str(manifest_path)})
        assert result.passed is True, (
            f"Validator failed on emitted manifest: "
            f"{[i.code for i in result.issues]}"
        )

    def test_chunks_jsonl_lines_match_count(self, dart_chunking_fixture):
        """``chunks_count`` in manifest matches actual JSONL line count."""
        fx = dart_chunking_fixture
        payload = _invoke_dart_chunking(fx["course_name"], fx["staging_dir"])

        chunks_path = Path(payload["dart_chunks_path"])
        actual_lines = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
        assert actual_lines == payload["chunks_count"]

    def test_empty_staging_emits_chunks_shell(self, tmp_path, monkeypatch):
        """Empty staging dir -> empty chunks.jsonl shell + valid manifest."""
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
        tool = registry["run_dart_chunking"]
        result = asyncio.run(
            tool(
                course_name="EMPTY_777",
                staging_dir=str(empty_staging),
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["chunks_count"] == 0
        # Empty file but still emitted, valid SHA-256 shape.
        assert _SHA256_RE.match(payload["dart_chunks_sha256"])
        chunks_path = Path(payload["dart_chunks_path"])
        assert chunks_path.exists()
        assert chunks_path.read_bytes() == b""

    def test_run_dart_chunking_registered_in_registry(self):
        """Forward-reference closure from Phase 7b ST 9's
        AGENT_TOOL_MAPPING entry: the tool must be registered."""
        registry = _build_tool_registry()
        assert "run_dart_chunking" in registry
        assert callable(registry["run_dart_chunking"])
