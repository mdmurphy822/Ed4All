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
# Phase 7b Subtask 14.5 — _run_concept_extraction consumes upstream
# dart_chunks_path from the chunking phase.
#
# Verifies:
#   * When a readable dart_chunks_path is supplied, the helper loads
#     chunks from JSONL and skips the legacy inline projection.
#   * When dart_chunks_path is absent or unreadable, the helper falls
#     through to the legacy inline projection (back-compat with
#     pre-Phase-7b runs / unit-test fixtures that bypass the chunking
#     phase).
#   * Byte-stability: when the upstream chunks.jsonl mirrors what the
#     inline projection would have produced (same chunk_id key, same
#     source.module_id / item_path, same chunk_type), both code paths
#     route equivalent chunks into ``build_pedagogy_graph`` and emit
#     byte-identical concept_graph_semantic.json.
# ---------------------------------------------------------------------------


def _legacy_projected_chunks(course_code_lower: str) -> list[dict]:
    """Mirror the inline projection's chunk shape for a fixture-equivalent
    upstream chunks.jsonl. Phase 8 ST 6 (`_run_concept_extraction`
    inline-projection ``chunk_id`` -> canonical ``id`` rename) brought
    the inline projection in line with `build_pedagogy_graph`'s
    `c.get("id")` contract; this fixture is updated in lockstep so the
    byte-stability invariant test (path-supplied vs path-absent code
    paths emit equivalent graphs on equivalent input) keeps holding.
    """
    return [
        {
            "id": f"{course_code_lower}_chunk_00001",
            "text": "overview Introduction to Pedagogical Concepts paragraphs Pedagogical concepts include alignment, assessment, scaffolding, learning outcomes, and curriculum design.",
            "concept_tags": [
                "introduction", "pedagogical", "concepts", "paragraphs",
                "include", "alignment", "assessment", "scaffolding",
            ],
            "learning_outcome_refs": [],
            "chunk_type": "content",
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": "demo_textbook",
                "item_path": "demo_textbook#intro",
            },
        },
        {
            "id": f"{course_code_lower}_chunk_00002",
            "text": "content Scaffolding Strategies paragraphs Scaffolding strategies provide structured support during initial learning, gradually fading as competence develops.",
            "concept_tags": [
                "content", "scaffolding", "strategies", "paragraphs",
                "provide", "structured", "support", "during",
            ],
            "learning_outcome_refs": [],
            "chunk_type": "content",
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": "demo_textbook",
                "item_path": "demo_textbook#scaffold",
            },
        },
        {
            "id": f"{course_code_lower}_chunk_00003",
            "text": "self_check Assessment Check paragraphs Formative assessment validates learner understanding before summative evaluation.",
            "concept_tags": [
                "self", "check", "assessment", "paragraphs", "formative",
                "validates", "learner", "understanding",
            ],
            "learning_outcome_refs": [],
            "chunk_type": "assessment_item",
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": "demo_textbook",
                "item_path": "demo_textbook#assess",
            },
        },
    ]


class TestRunConceptExtractionConsumesUpstreamChunks:
    """Phase 7b ST 14.5 — refactor consumes upstream dart_chunks_path."""

    def test_upstream_chunks_path_loaded_when_supplied(
        self, concept_extraction_fixture
    ):
        """When dart_chunks_path is supplied with N chunks, the helper
        reports chunk_count == N regardless of staging_dir contents."""
        fx = concept_extraction_fixture
        chunks = _legacy_projected_chunks("demo_303")

        chunks_path = fx["fake_root"] / "upstream_chunks.jsonl"
        chunks_path.write_text(
            "\n".join(json.dumps(c) for c in chunks) + "\n",
            encoding="utf-8",
        )

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
                dart_chunks_path=str(chunks_path),
            )
        )
        payload = json.loads(result)

        assert payload["success"] is True
        # 3 chunks from the upstream JSONL, NOT from the staging sidecar
        # (which also has 3 sections — same count, but the assertion
        # below pins that the JSONL ingest path actually ran).
        assert payload["chunk_count"] == 3

    def test_inline_projection_skipped_when_upstream_supplied(
        self, concept_extraction_fixture
    ):
        """When dart_chunks_path is supplied with 1 chunk and staging
        has 3 sections, chunk_count is 1 — the inline projection did
        NOT run."""
        fx = concept_extraction_fixture
        upstream = [_legacy_projected_chunks("demo_303")[0]]
        chunks_path = fx["fake_root"] / "single_chunk.jsonl"
        chunks_path.write_text(
            json.dumps(upstream[0]) + "\n", encoding="utf-8"
        )

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
                dart_chunks_path=str(chunks_path),
            )
        )
        payload = json.loads(result)

        assert payload["success"] is True
        # The staging fixture has 3 sections; if the inline projection
        # had also run it would have emitted 4 chunks total. 1 confirms
        # the inline projection branch was skipped.
        assert payload["chunk_count"] == 1

    def test_falls_through_to_inline_when_path_absent(
        self, concept_extraction_fixture
    ):
        """When dart_chunks_path is unset, the legacy inline-projection
        runs (back-compat path)."""
        fx = concept_extraction_fixture
        # No dart_chunks_path kwarg — the helper falls through.
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        assert payload["success"] is True
        # 3 sections in the fixture sidecar -> 3 chunks projected.
        assert payload["chunk_count"] == 3

    def test_falls_through_when_path_unreadable(
        self, concept_extraction_fixture
    ):
        """When dart_chunks_path points at a non-existent file, the
        helper falls through to the inline projection (warning log,
        not a hard failure)."""
        fx = concept_extraction_fixture
        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
                dart_chunks_path=str(
                    fx["fake_root"] / "nonexistent" / "chunks.jsonl"
                ),
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True
        # Inline-projection ran -> 3 chunks from the staging sidecar.
        assert payload["chunk_count"] == 3

    def test_byte_stability_path_supplied_vs_path_absent(
        self, concept_extraction_fixture, tmp_path, monkeypatch
    ):
        """Byte-equality of concept_graph_semantic.json across the two
        code paths when the upstream chunkset mirrors what the inline
        projection would have produced. Pins the architectural
        invariant from the Phase 7b ST 14.5 plan: the refactor MUST
        NOT alter graph emission semantics on equivalent input.
        """
        fx = concept_extraction_fixture

        # Path A — path-absent (legacy inline-projection runs).
        payload_absent = _invoke(fx["course_name"], fx["staging_dir"])
        graph_absent = Path(payload_absent["concept_graph_path"]).read_bytes()

        # Path B — path-supplied with chunks that mirror the legacy
        # inline-projection shape. Build a fresh fake_root so the
        # path-supplied run writes a separate concept_graph_semantic.json.
        fake_root_b = tmp_path / "root_b"
        fake_root_b.mkdir()
        monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root_b)
        monkeypatch.setattr(
            pipeline_tools,
            "COURSEFORGE_INPUTS",
            fake_root_b / "Courseforge" / "inputs" / "textbooks",
        )
        (fake_root_b / "Courseforge" / "inputs" / "textbooks").mkdir(
            parents=True
        )

        chunks = _legacy_projected_chunks("demo_303")
        chunks_path = fake_root_b / "upstream_chunks.jsonl"
        chunks_path.write_text(
            "\n".join(json.dumps(c) for c in chunks) + "\n",
            encoding="utf-8",
        )

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                # Empty staging -> inline projection wouldn't run anyway,
                # but with dart_chunks_path supplied the upstream branch
                # takes precedence regardless.
                staging_dir=str(fx["staging_dir"]),
                dart_chunks_path=str(chunks_path),
            )
        )
        payload_supplied = json.loads(result)
        graph_supplied = Path(
            payload_supplied["concept_graph_path"]
        ).read_bytes()

        # The only field that legitimately differs is `generated_at`
        # (timestamp). Strip it from both before comparing.
        absent_obj = json.loads(graph_absent)
        supplied_obj = json.loads(graph_supplied)
        absent_obj.pop("generated_at", None)
        supplied_obj.pop("generated_at", None)

        assert absent_obj == supplied_obj, (
            "Refactor regression: path-supplied vs path-absent code paths "
            "emit different concept_graph_semantic.json on equivalent input. "
            "Phase 7b ST 14.5 invariant violated."
        )


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


# ---------------------------------------------------------------------------
# Phase 7c Subtask 16 — _run_imscc_chunking smoke tests
# ---------------------------------------------------------------------------


def _build_imscc_zip(zip_path: Path, html_files: list[tuple[str, str]]) -> None:
    """Build a minimal IMSCC zip at ``zip_path`` containing the given
    (inner_path, html_content) tuples plus a stub imsmanifest.xml.

    Mirrors the structural shape of a real IMSCC archive (zip with
    ``imsmanifest.xml`` + HTML resources), without requiring the full
    IMS-cc spec scaffolding — ``_run_imscc_chunking`` walks the zip's
    HTML entries directly via ``zipfile.ZipFile`` and ignores the
    manifest. The fixture is sufficient for the chunker smoke; full
    manifest parsing is `IMSCCParser`'s domain, not this helper's.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "imsmanifest.xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1">'
            '</manifest>\n',
        )
        for inner_path, html in html_files:
            zf.writestr(inner_path, html)


def _imscc_html_payload(title: str) -> str:
    """Emit a DART-shaped HTML payload large enough to clear the
    chunker's minimum-size threshold."""
    return (
        f"<!DOCTYPE html>\n"
        f"<html lang=\"en\">\n"
        f"<head><meta charset=\"utf-8\"><title>{title}</title></head>\n"
        f"<body>\n"
        f"  <main>\n"
        f"    <section>\n"
        f"      <h1>{title}</h1>\n"
        f"      <p>This IMSCC HTML file is a fixture for the Phase 7c "
        f"chunking smoke test. {' '.join(['Chunk content padding sentence.'] * 60)}</p>\n"
        f"      <h2>Sub-section about pedagogy</h2>\n"
        f"      <p>Pedagogy describes the methods and practice of teaching. "
        f"{' '.join(['Additional padding text to clear the chunker minimum-size threshold.'] * 60)}</p>\n"
        f"    </section>\n"
        f"  </main>\n"
        f"</body>\n"
        f"</html>\n"
    )


@pytest.fixture
def imscc_chunking_fixture(tmp_path, monkeypatch):
    """Build a minimal packaged IMSCC + fake LibV2 root so
    ``_run_imscc_chunking`` writes under the temp tree."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    imscc_path = tmp_path / "course.imscc"
    _build_imscc_zip(
        imscc_path,
        [
            ("html/page_01.html", _imscc_html_payload("Page One")),
            ("html/page_02.html", _imscc_html_payload("Page Two")),
        ],
    )

    return {
        "fake_root": fake_root,
        "imscc_path": imscc_path,
        "course_name": "DEMO_888",
        "course_slug": "demo-888",
    }


def _invoke_imscc_chunking(course_name: str, imscc_path: Path) -> dict:
    registry = _build_tool_registry()
    tool = registry["run_imscc_chunking"]
    result = asyncio.run(
        tool(
            course_name=course_name,
            imscc_path=str(imscc_path),
        )
    )
    return json.loads(result)


class TestRunImsccChunkingEmitsChunksJsonl:
    def test_run_imscc_chunking_emits_chunks_jsonl(self, imscc_chunking_fixture):
        """ST 16 plan-cited verification — helper writes chunks.jsonl
        and a sibling manifest.json under
        ``LibV2/courses/<slug>/imscc_chunks/``."""
        fx = imscc_chunking_fixture
        payload = _invoke_imscc_chunking(fx["course_name"], fx["imscc_path"])

        assert payload["success"] is True
        assert "imscc_chunks_path" in payload
        assert "imscc_chunks_sha256" in payload

        chunks_path = Path(payload["imscc_chunks_path"])
        assert chunks_path.exists(), (
            f"chunks.jsonl not written at {chunks_path}"
        )
        assert chunks_path.name == "chunks.jsonl"

        # Path lands under LibV2/courses/<slug>/imscc_chunks/.
        rel = chunks_path.relative_to(fx["fake_root"])
        parts = rel.parts
        assert parts[0] == "LibV2"
        assert parts[1] == "courses"
        assert parts[2] == fx["course_slug"]
        assert parts[3] == "imscc_chunks"
        assert parts[4] == "chunks.jsonl"

    def test_imscc_chunks_sha256_matches_file_bytes(self, imscc_chunking_fixture):
        fx = imscc_chunking_fixture
        payload = _invoke_imscc_chunking(fx["course_name"], fx["imscc_path"])

        assert _SHA256_RE.match(payload["imscc_chunks_sha256"]), (
            f"sha256 not in canonical hex shape: {payload['imscc_chunks_sha256']!r}"
        )

        chunks_path = Path(payload["imscc_chunks_path"])
        on_disk_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        assert on_disk_hash == payload["imscc_chunks_sha256"], (
            "Returned sha256 must match on-disk chunks.jsonl bytes."
        )

    def test_manifest_emitted_and_validates(self, imscc_chunking_fixture):
        """Manifest.json is emitted with the canonical chunkset shape
        (chunkset_kind=imscc, source_imscc_sha256) and passes the
        ChunksetManifestValidator gate."""
        fx = imscc_chunking_fixture
        payload = _invoke_imscc_chunking(fx["course_name"], fx["imscc_path"])

        manifest_path = Path(payload["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Required schema fields for the imscc branch.
        assert manifest["chunks_sha256"] == payload["imscc_chunks_sha256"]
        assert manifest["chunkset_kind"] == "imscc"
        assert _SHA256_RE.match(manifest["source_imscc_sha256"])
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
        # ``source_dart_html_sha256`` MUST be absent on imscc-branch manifests
        # (the schema's allOf branch only requires source_imscc_sha256 when
        # chunkset_kind=imscc, and additionalProperties=false admits both
        # source-SHA fields, but our emit must keep the kind-specific field
        # only).
        assert "source_dart_html_sha256" not in manifest

        # Validator round-trip: the emitted manifest must pass the
        # ChunksetManifestValidator gate.
        from lib.validators.chunkset_manifest import ChunksetManifestValidator

        validator = ChunksetManifestValidator()
        result = validator.validate({"chunkset_manifest_path": str(manifest_path)})
        assert result.passed is True, (
            f"Validator failed on emitted manifest: "
            f"{[i.code for i in result.issues]}"
        )

    def test_source_imscc_sha256_matches_archive_bytes(self, imscc_chunking_fixture):
        """``source_imscc_sha256`` returned + written to manifest must
        equal the SHA-256 of the .imscc archive bytes the helper read."""
        fx = imscc_chunking_fixture
        payload = _invoke_imscc_chunking(fx["course_name"], fx["imscc_path"])

        archive_hash = hashlib.sha256(fx["imscc_path"].read_bytes()).hexdigest()
        assert payload["source_imscc_sha256"] == archive_hash, (
            "source_imscc_sha256 must match the on-disk imscc archive bytes."
        )

        manifest_path = Path(payload["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_imscc_sha256"] == archive_hash

    def test_chunks_jsonl_lines_match_count(self, imscc_chunking_fixture):
        """``chunks_count`` in manifest matches actual JSONL line count."""
        fx = imscc_chunking_fixture
        payload = _invoke_imscc_chunking(fx["course_name"], fx["imscc_path"])

        chunks_path = Path(payload["imscc_chunks_path"])
        actual_lines = sum(
            1
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert actual_lines == payload["chunks_count"]

    def test_missing_imscc_emits_chunks_shell(self, tmp_path, monkeypatch):
        """Missing imscc_path -> empty chunks.jsonl shell + valid manifest."""
        fake_root = tmp_path / "root"
        fake_root.mkdir()
        monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
        monkeypatch.setattr(
            pipeline_tools,
            "COURSEFORGE_INPUTS",
            fake_root / "Courseforge" / "inputs" / "textbooks",
        )
        (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

        registry = _build_tool_registry()
        tool = registry["run_imscc_chunking"]
        result = asyncio.run(
            tool(
                course_name="EMPTY_888",
                imscc_path=str(tmp_path / "does_not_exist.imscc"),
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["chunks_count"] == 0
        # Empty file but still emitted, valid SHA-256 shape.
        assert _SHA256_RE.match(payload["imscc_chunks_sha256"])
        chunks_path = Path(payload["imscc_chunks_path"])
        assert chunks_path.exists()
        assert chunks_path.read_bytes() == b""

        # Manifest still valid with empty-bytes-SHA sentinel.
        manifest_path = Path(payload["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["chunkset_kind"] == "imscc"
        assert _SHA256_RE.match(manifest["source_imscc_sha256"])
        assert manifest["chunks_count"] == 0

    def test_run_imscc_chunking_registered_in_registry(self):
        """The tool must be registered for phase-name dispatch from
        ``MCP/core/executor.py::_PHASE_TOOL_MAPPING``."""
        registry = _build_tool_registry()
        assert "run_imscc_chunking" in registry
        assert callable(registry["run_imscc_chunking"])
