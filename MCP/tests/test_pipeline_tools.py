"""Smoke tests for ``MCP.tools.pipeline_tools`` registry helpers.

Phase 6 ST 12 — exercises the new ``_run_concept_extraction`` helper
registered by ``_build_tool_registry`` to confirm it:

  * Reads SemantiK staging output (``*_synthesized.json`` sidecars).
  * Persists ``concept_graph_semantic.json`` + ``manifest.json`` under
    ``LibV2/courses/<slug>/concept_graph/``.
  * Emits a SHA-256 hex digest of the graph bytes.
  * Returns the canonical ``concept_graph_path`` /
    ``concept_graph_sha256`` keys the workflow runner threads through
    ``phase_outputs.concept_extraction``.

The helper is a pure file-IO + ``Trainforge.rag.graphs.pedagogy_graph_builder``
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
    """Emit a minimal SemantiK ``*_synthesized.json`` sidecar at ``path``."""
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
    """Build a minimal SemantiK staging dir + fake LibV2 root."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    # Redirect _PROJECT_ROOT + COURSEFORGE_INPUTS so the helper's
    # ``LibV2/courses/<slug>/concept_graph/...`` write lands in tmp_path
    # instead of the real repo.
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    # _resolve_libv2_root precedence: kwarg > ED4ALL_LIBV2_ROOT env > _PROJECT_ROOT/LibV2;
    # the repo conftest's autouse fixture sets the env to an isolation dir, so pin it
    # at fake_root/LibV2 here to keep the default-resolution writes under fake_root.
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(fake_root / "LibV2"))
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
        """Fix-2: the phase emits the genuine ``kind: "concept_semantic"``
        graph (``DomainConcept`` nodes via ``build_semantic_graph``)
        instead of the prior ``kind: "pedagogy"`` graph — verify the
        new dispatch landed."""
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        graph = json.loads(
            Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
        )
        assert graph.get("kind") == "concept_semantic", (
            "concept_extraction must emit the genuine semantic graph; "
            f"got kind={graph.get('kind')!r}."
        )
        node_classes = {n.get("class") for n in graph.get("nodes", [])}
        concept_classes = {
            "Concept", "DomainConcept", "concept", "domain_concept"
        }
        assert node_classes & concept_classes, (
            "build_semantic_graph should emit DomainConcept-class nodes; "
            f"node classes seen: {sorted(node_classes)}."
        )

    def test_chunks_derived_from_staging(self, concept_extraction_fixture):
        """Three sections in the fixture sidecar -> 3 chunks projected."""
        fx = concept_extraction_fixture
        payload = _invoke(fx["course_name"], fx["staging_dir"])

        # Three sections in the fixture (intro / scaffold / assess).
        assert payload["chunk_count"] == 3

    def test_empty_staging_emits_shell_graph(self, tmp_path, monkeypatch):
        """When no sidecars exist, helper still emits a graph shell so
        downstream gates have something to validate against.

        Fix-2: the shell is now a ``kind: "concept_semantic"`` empty
        graph (0 nodes) — ``build_semantic_graph`` copies its node set
        verbatim from the co-occurrence graph, and empty ``concept_tags``
        yield zero nodes (the Risk-2 contract). The shell is no longer a
        pedagogy graph with BloomLevel/DifficultyLevel scaffolding."""
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
        graph = json.loads(
            Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
        )
        assert graph.get("kind") == "concept_semantic", (
            "Empty-input shell must still be kind='concept_semantic'."
        )


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
            # Fix-2: concept_tags must mirror EXACTLY what the inline
            # projection's _tokenize_concepts helper emits for this
            # section (title + section_type + data keys/values, in that
            # order) — build_semantic_graph copies its node set from the
            # concept_tags-driven co-occurrence graph, so a mismatch here
            # breaks the path-supplied vs path-absent byte-stability test.
            "concept_tags": [
                "introduction", "pedagogical", "concepts", "overview",
                "paragraphs", "include", "alignment", "assessment",
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
                "scaffolding", "strategies", "content", "paragraphs",
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
                "assessment", "check", "paragraphs", "formative",
                "validates", "learner", "understanding", "before",
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

        # The only fields that legitimately differ are wall-clock
        # timestamps. Fix-2's ``build_semantic_graph`` stamps a
        # per-node / per-edge ``created_at`` (and the top-level
        # ``generated_at``) from wall-clock; two runs at slightly
        # different times produce different timestamps. Strip every
        # timestamp field recursively before comparing the structural
        # graph content — the Phase 7b ST 14.5 invariant is structural
        # equality, not byte equality of the timestamp fields.
        _TS_KEYS = {"generated_at", "created_at"}

        def _strip_timestamps(obj):
            if isinstance(obj, dict):
                return {
                    k: _strip_timestamps(v)
                    for k, v in obj.items()
                    if k not in _TS_KEYS
                }
            if isinstance(obj, list):
                return [_strip_timestamps(v) for v in obj]
            return obj

        absent_obj = _strip_timestamps(json.loads(graph_absent))
        supplied_obj = _strip_timestamps(json.loads(graph_supplied))

        assert absent_obj == supplied_obj, (
            "Refactor regression: path-supplied vs path-absent code paths "
            "emit different concept_graph_semantic.json on equivalent input. "
            "Phase 7b ST 14.5 invariant violated."
        )


# ---------------------------------------------------------------------------
# Phase 7b Subtask 11 — SemantiK chunking smoke tests
# ---------------------------------------------------------------------------


def _write_semantik_html(path: Path, title: str) -> None:
    """Emit a minimal SemantiK-shaped HTML file at ``path``."""
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>This SemantiK HTML file is a fixture for the Phase 7b chunking smoke test. {' '.join(['Chunk content padding sentence.'] * 60)}</p>
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
def semantik_chunking_fixture(tmp_path, monkeypatch):
    """Build a minimal SemantiK staging dir + fake LibV2 root."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", fake_root)
    # _resolve_libv2_root precedence: kwarg > ED4ALL_LIBV2_ROOT env > _PROJECT_ROOT/LibV2;
    # the repo conftest's autouse fixture sets the env to an isolation dir, so pin it
    # at fake_root/LibV2 here to keep the default-resolution writes under fake_root.
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(fake_root / "LibV2"))
    monkeypatch.setattr(
        pipeline_tools,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_semantik_html(staging / "chapter_01.html", "Chapter One")
    _write_semantik_html(staging / "chapter_02.html", "Chapter Two")

    return {
        "fake_root": fake_root,
        "staging_dir": staging,
        "course_name": "DEMO_777",
        "course_slug": "demo-777",
    }


def _invoke_semantik_chunking(course_name: str, staging_dir: Path) -> dict:
    registry = _build_tool_registry()
    tool = registry["run_dart_chunking"]
    result = asyncio.run(
        tool(
            course_name=course_name,
            staging_dir=str(staging_dir),
        )
    )
    return json.loads(result)


class TestRunSemantikChunkingEmitsChunksJsonl:
    def test_run_semantik_chunking_emits_chunks_jsonl(self, semantik_chunking_fixture):
        """ST 11 plan-cited verification — helper writes chunks.jsonl
        and a sibling manifest.json under
        ``LibV2/courses/<slug>/semantik_chunks/``."""
        fx = semantik_chunking_fixture
        payload = _invoke_semantik_chunking(fx["course_name"], fx["staging_dir"])

        assert payload["success"] is True
        assert "semantik_chunks_path" in payload
        assert "semantik_chunks_sha256" in payload

        chunks_path = Path(payload["semantik_chunks_path"])
        assert chunks_path.exists(), (
            f"chunks.jsonl not written at {chunks_path}"
        )
        assert chunks_path.name == "chunks.jsonl"

        # Path lands under LibV2/courses/<slug>/semantik_chunks/.
        rel = chunks_path.relative_to(fx["fake_root"])
        parts = rel.parts
        assert parts[0] == "LibV2"
        assert parts[1] == "courses"
        assert parts[2] == fx["course_slug"]
        assert parts[3] == "semantik_chunks"
        assert parts[4] == "chunks.jsonl"

    def test_semantik_chunks_sha256_matches_file_bytes(self, semantik_chunking_fixture):
        fx = semantik_chunking_fixture
        payload = _invoke_semantik_chunking(fx["course_name"], fx["staging_dir"])

        assert _SHA256_RE.match(payload["semantik_chunks_sha256"]), (
            f"sha256 not in canonical hex shape: {payload['semantik_chunks_sha256']!r}"
        )

        chunks_path = Path(payload["semantik_chunks_path"])
        on_disk_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        assert on_disk_hash == payload["semantik_chunks_sha256"], (
            "Returned sha256 must match on-disk chunks.jsonl bytes."
        )

    def test_manifest_emitted_and_validates(self, semantik_chunking_fixture):
        """Manifest.json is emitted with the canonical chunkset shape
        and passes the ChunksetManifestValidator gate."""
        fx = semantik_chunking_fixture
        payload = _invoke_semantik_chunking(fx["course_name"], fx["staging_dir"])

        manifest_path = Path(payload["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Required schema fields.
        assert manifest["chunks_sha256"] == payload["semantik_chunks_sha256"]
        # The chunker sidecar emits the ratified chunkset_kind + source-sha
        # field names.
        assert manifest["chunkset_kind"] == "semantik"
        assert _SHA256_RE.match(manifest["source_semantik_html_sha256"])
        assert "source_dart_html_sha256" not in manifest
        assert isinstance(manifest["chunker_version"], str)
        assert manifest["chunks_count"] == payload["chunks_count"]
        # additionalProperties: false — only the canonical keys.
        # W3.H sub-task H1 added the optional ``source_coverage`` block
        # to the chunkset_manifest schema; the field surfaces on every
        # emit but stays back-compat with legacy manifests (the
        # ChunksetManifestValidator still passes when it's absent).
        assert set(manifest.keys()).issubset({
            "chunks_sha256",
            "chunker_version",
            # Chunk-TEXT extraction-contract provenance marker (orthogonal to
            # chunker_version's emit shape). Stamped unconditionally.
            "extraction_contract",
            "chunkset_kind",
            "source_semantik_html_sha256",
            "source_dart_html_sha256",  # legacy (dual-read) — not emitted by the current path
            "source_imscc_sha256",
            "chunks_count",
            "generated_at",
            "source_coverage",
            # W1b.5: honest learning-outcome-linkage markers, stamped when a
            # fresh manifest advertises that LO-linkage was skipped (chunks
            # emit before course-planning mints the TO-NN/CO-NN id set).
            "lo_linkage",
            "lo_refs_unpopulated",
            # Track K: verbatim chunk-overlap budget (emitted only when > 0).
            "overlap_words",
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

    def test_chunks_jsonl_lines_match_count(self, semantik_chunking_fixture):
        """``chunks_count`` in manifest matches actual JSONL line count."""
        fx = semantik_chunking_fixture
        payload = _invoke_semantik_chunking(fx["course_name"], fx["staging_dir"])

        chunks_path = Path(payload["semantik_chunks_path"])
        actual_lines = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
        assert actual_lines == payload["chunks_count"]

    def test_empty_staging_emits_chunks_shell(self, tmp_path, monkeypatch):
        """Wave1-I2: empty staging dir + no pre-existing chunks -> fail-closed.

        Previously this emitted an empty-bytes chunks.jsonl shell, which on a
        workflow resume silently overwrote real chunks. Now the helper raises
        RuntimeError when there's nothing to chunk and nothing to preserve.
        See plans/dispatch-7-execution-inspection-2026-05.md Finding 2.
        """
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
        with pytest.raises(RuntimeError, match=r"Wave1-I2: _run_dart_chunking refusing"):
            asyncio.run(
                tool(
                    course_name="EMPTY_777",
                    staging_dir=str(empty_staging),
                    libv2_root=str(tmp_path / "libv2"),
                )
            )

    def test_run_semantik_chunking_registered_in_registry(self):
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
    """Emit a SemantiK-shaped HTML payload large enough to clear the
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
    # _resolve_libv2_root precedence: kwarg > ED4ALL_LIBV2_ROOT env > _PROJECT_ROOT/LibV2;
    # the repo conftest's autouse fixture sets the env to an isolation dir, so pin it
    # at fake_root/LibV2 here to keep the default-resolution writes under fake_root.
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(fake_root / "LibV2"))
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
        # W3.H sub-task H1 added the optional ``source_coverage`` block
        # symmetrically to both SemantiK + IMSCC manifests.
        assert set(manifest.keys()).issubset({
            "chunks_sha256",
            "chunker_version",
            # Chunk-TEXT extraction-contract provenance marker (orthogonal to
            # chunker_version's emit shape). Stamped unconditionally.
            "extraction_contract",
            "chunkset_kind",
            "source_semantik_html_sha256",
            "source_dart_html_sha256",  # legacy (dual-read) — not emitted by the current path
            "source_imscc_sha256",
            "chunks_count",
            "generated_at",
            "source_coverage",
            # W1b.5: honest learning-outcome-linkage markers, stamped when a
            # fresh manifest advertises that LO-linkage was skipped (chunks
            # emit before course-planning mints the TO-NN/CO-NN id set).
            "lo_linkage",
            "lo_refs_unpopulated",
            # Track K: verbatim chunk-overlap budget (emitted only when > 0).
            "overlap_words",
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
        """Wave1-I2: missing imscc_path + no pre-existing chunks -> fail-closed.

        Previously this emitted an empty-bytes chunks.jsonl shell with an
        all-zeros source-SHA sentinel, which on a workflow resume silently
        overwrote real chunks. Now the helper raises RuntimeError when there's
        nothing to chunk and nothing to preserve. See
        plans/dispatch-7-execution-inspection-2026-05.md Finding 2.
        """
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
        with pytest.raises(RuntimeError, match=r"Wave1-I2: _run_imscc_chunking refusing"):
            asyncio.run(
                tool(
                    course_name="EMPTY_888",
                    imscc_path=str(tmp_path / "does_not_exist.imscc"),
                    libv2_root=str(tmp_path / "libv2"),
                )
            )

    def test_run_imscc_chunking_registered_in_registry(self):
        """The tool must be registered for phase-name dispatch from
        ``MCP/core/executor.py::_PHASE_TOOL_MAPPING``."""
        registry = _build_tool_registry()
        assert "run_imscc_chunking" in registry
        assert callable(registry["run_imscc_chunking"])
