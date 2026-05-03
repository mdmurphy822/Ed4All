"""Phase 8 ST 6 regression test — `_run_concept_extraction` inline-
projection fallback emits canonical ``id`` keys (not legacy
``chunk_id``) so ``build_pedagogy_graph`` actually consumes them.

Pre-Phase-8: the inline projection emitted ``chunk_id``-keyed dicts;
``Trainforge/pedagogy_graph_builder.py:593`` reads ``cid = c.get("id")``
strictly (no ``chunk_id`` fallback). Result: every chunk produced by
the fallback path was silently dropped, and any workflow run where
``dart_chunks_path`` was missing / unreadable / where the upstream
``chunking`` phase was skipped fed ``build_pedagogy_graph`` an
effectively empty chunk set. Phase 7b ST 14.5 reconciled the FORWARD
path (upstream JSONL load now provides canonical ``id``-keyed chunks
via the ``Trainforge.chunker`` package) but did NOT migrate the inline-
projection's emit shape. Phase 8 ST 6 closes that residual gap.

Test contract (mirrors `TestRunConceptExtractionConsumesUpstreamChunks`
fixture pattern from Phase 7b ST 14.5):

  1. Construct a fixture staging dir with one ``*_synthesized.json``.
  2. Invoke ``_run_concept_extraction`` with ``dart_chunks_path=None``
     (forces the inline-projection fallback path).
  3. Assert the emitted chunks (intercepted via a build-side spy)
     have ``"id"`` keys and ZERO ``"chunk_id"`` keys.
  4. Pass the chunks to ``build_pedagogy_graph(chunks=...,
     objectives=[], course_id="TEST")`` and assert the returned graph
     contains at least one ``Chunk``-class node — pre-Phase-8 the
     count would have been zero due to the silent-drop bug.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


def _write_synthesized(path: Path) -> None:
    """Emit a minimal DART ``*_synthesized.json`` fixture sidecar.

    Three sections so the inline projection emits three chunks — same
    shape as the Phase 7b ST 14.5 precedent fixture in
    ``test_pipeline_tools.py::concept_extraction_fixture``.
    """
    doc = {
        "campus_code": "phase8st6",
        "campus_name": "Phase 8 ST 6 Regression",
        "sections": [
            {
                "section_id": "intro",
                "section_type": "overview",
                "section_title": "Pedagogical Concept Anchors",
                "data": {
                    "paragraphs": [
                        "Pedagogical concept anchors include "
                        "alignment, scaffolding, and assessment "
                        "interlinks across the curriculum design."
                    ]
                },
            },
            {
                "section_id": "scaffold",
                "section_type": "content",
                "section_title": "Scaffolding Patterns",
                "data": {
                    "paragraphs": [
                        "Scaffolding patterns gradually fade support "
                        "as learner competence develops over time."
                    ]
                },
            },
            {
                "section_id": "selfcheck",
                "section_type": "self_check",
                "section_title": "Self-Check Probe",
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
def fallback_fixture(tmp_path, monkeypatch):
    """Hermetic staging dir + redirected LibV2 root so the helper
    writes ``concept_graph_semantic.json`` under ``tmp_path`` rather
    than the real repo tree."""
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
    _write_synthesized(staging / "demo_textbook_synthesized.json")

    return {
        "fake_root": fake_root,
        "staging_dir": staging,
        "course_name": "PHASE8_ST6",
        "course_slug": "phase8-st6",
    }


def _intercept_build_pedagogy_graph(monkeypatch) -> List[List[Dict[str, Any]]]:
    """Spy on ``Trainforge.pedagogy_graph_builder.build_pedagogy_graph``
    so we can assert the chunks the inline-projection passed in carry
    canonical ``id`` keys.

    Returns a list that the spy appends each ``chunks=`` arg into.
    """
    from Trainforge import pedagogy_graph_builder as _builder_mod

    original = _builder_mod.build_pedagogy_graph
    captures: List[List[Dict[str, Any]]] = []

    def _spy(chunks, **kwargs):
        # Snapshot a shallow copy so later mutations by the helper
        # can't retroactively affect our assertions.
        captures.append(list(chunks))
        return original(chunks=chunks, **kwargs)

    monkeypatch.setattr(_builder_mod, "build_pedagogy_graph", _spy)
    # The helper imports the symbol lazily inside the function body
    # (``from Trainforge.pedagogy_graph_builder import
    # build_pedagogy_graph``) so a single module-level patch on
    # ``_builder_mod`` is sufficient — the lazy import resolves
    # against our spy at call time.
    return captures


class TestInlineProjectionEmitsCanonicalIdKey:
    """Phase 8 ST 6 — the inline-projection fallback at
    ``MCP/tools/pipeline_tools.py:6327-6343`` emits canonical ``id``
    keys instead of the legacy ``chunk_id`` key the builder silently
    dropped."""

    def test_emitted_chunks_use_id_key_not_chunk_id(
        self, fallback_fixture, monkeypatch
    ):
        """Direct shape assertion: every chunk dict the inline
        projection feeds into ``build_pedagogy_graph`` carries an
        ``id`` field; zero carry the legacy ``chunk_id`` field.

        Pre-Phase-8 this assertion would fail — every chunk would
        carry ``chunk_id`` and zero would carry ``id``."""
        fx = fallback_fixture
        captures = _intercept_build_pedagogy_graph(monkeypatch)

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
                # dart_chunks_path intentionally omitted so the
                # fallback path runs.
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True

        # Spy must have been called exactly once.
        assert len(captures) == 1, (
            f"Expected exactly one build_pedagogy_graph dispatch; "
            f"got {len(captures)}."
        )
        chunks_passed = captures[0]
        # Three sections in the fixture -> three projected chunks.
        assert len(chunks_passed) == 3, (
            f"Inline projection should emit 3 chunks for the 3-section "
            f"fixture; emitted {len(chunks_passed)}."
        )

        ids = [c.get("id") for c in chunks_passed]
        legacy_ids = [c.get("chunk_id") for c in chunks_passed]

        assert all(isinstance(i, str) and i for i in ids), (
            f"Every projected chunk MUST carry a non-empty 'id' "
            f"string field; got: {ids!r}"
        )
        assert all(c.get("chunk_id") is None for c in chunks_passed), (
            f"No projected chunk should carry the legacy 'chunk_id' "
            f"field; got: {legacy_ids!r}. Phase 8 ST 6 rename "
            f"regression."
        )

    def test_inline_projection_produces_chunk_nodes_in_graph(
        self, fallback_fixture, monkeypatch
    ):
        """End-to-end behavioral assertion: with the canonical ``id``
        rename in place, ``build_pedagogy_graph`` actually consumes
        the projected chunks and emits ``Chunk``-class nodes in the
        resulting graph.

        Pre-Phase-8 this assertion would fail because the
        ``c.get("id")`` lookup at builder line 593 returned ``None``
        for every ``chunk_id``-keyed dict and the loop short-circuited
        at line 594 (``if not cid: continue``). Result: zero ``Chunk``
        nodes despite three input chunks. The bare typed-node count
        (BloomLevel + DifficultyLevel) emitted unconditionally would
        still pass the existing
        ``test_empty_staging_emits_shell_graph`` test, masking the
        regression class this test pins."""
        fx = fallback_fixture
        captures = _intercept_build_pedagogy_graph(monkeypatch)

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["chunk_count"] == 3, (
            f"helper-reported chunk_count should reflect the 3 "
            f"projected chunks; got {payload['chunk_count']}."
        )

        # Graph round-trip: load the on-disk graph and assert at
        # least one Chunk-class node landed.
        graph = json.loads(
            Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
        )
        chunk_nodes = [
            n for n in graph.get("nodes", []) if n.get("class") == "Chunk"
        ]
        assert len(chunk_nodes) >= 1, (
            f"Phase 8 ST 6 regression: build_pedagogy_graph emitted "
            f"zero Chunk-class nodes despite 3 inline-projected chunks. "
            f"Likely cause: the inline projection regressed to "
            f"emitting 'chunk_id' keys that the builder silently "
            f"drops at pedagogy_graph_builder.py:593-595."
        )
        # The intercepted chunk list and the on-disk graph chunk
        # count should agree (plus-or-minus chunks the builder
        # legitimately filters for orthogonal reasons — the basic
        # presence floor above is the load-bearing assertion).
        assert len(chunk_nodes) == len(captures[0]), (
            f"Builder consumed {len(chunk_nodes)} chunks but the "
            f"helper passed in {len(captures[0])}. A drift here "
            f"likely re-opens the silent-drop class even with the "
            f"'id' rename in place."
        )

    def test_chunks_pass_through_build_pedagogy_graph_directly(
        self, fallback_fixture, monkeypatch
    ):
        """Plan-cited assertion: pass the inline-projection chunks
        directly into ``build_pedagogy_graph(chunks=...,
        objectives={}, course_id="TEST")`` and confirm the returned
        graph carries at least one Chunk node.

        This bypasses the helper's on-disk write entirely so a
        regression that re-introduced ``chunk_id`` while preserving
        graph file output via some unrelated mechanism couldn't slip
        past."""
        fx = fallback_fixture
        captures = _intercept_build_pedagogy_graph(monkeypatch)

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
            )
        )
        assert captures, "spy never fired — helper dispatch path broke"
        chunks = captures[0]

        from Trainforge.pedagogy_graph_builder import (
            build_pedagogy_graph as _real_builder,
        )

        # Restore the real builder for this direct-invocation leg
        # (the spy chained through to it, but invoking it via the
        # spy here would double-record the captures list).
        graph = _real_builder(
            chunks=chunks,
            objectives={},
            course_id="TEST",
        )
        chunk_nodes = [
            n for n in graph.get("nodes", []) if n.get("class") == "Chunk"
        ]
        assert len(chunk_nodes) >= 1, (
            "Direct build_pedagogy_graph dispatch over the "
            "inline-projection chunks emitted zero Chunk-class "
            "nodes. Likely a 'chunk_id' regression."
        )
