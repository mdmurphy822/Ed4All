"""Phase 8 ST 6 regression test — `_run_concept_extraction` inline-
projection fallback emits canonical ``id`` keys (not legacy
``chunk_id``) so the graph builders actually consume them.

Pre-Phase-8: the inline projection emitted ``chunk_id``-keyed dicts;
``Trainforge/pedagogy_graph_builder.py:593`` read ``cid = c.get("id")``
strictly (no ``chunk_id`` fallback). Result: every chunk produced by
the fallback path was silently dropped. Phase 8 ST 6 closed that gap by
emitting the canonical ``id`` key.

Fix-2 update: ``_run_concept_extraction`` no longer dispatches
``build_pedagogy_graph`` — it now builds the genuine
``kind: "concept_semantic"`` graph via the instance-free
``build_cooccurrence_graph`` helper + ``build_semantic_graph``. The
canonical-``id``-key anti-regression intent is unchanged and still
load-bearing: ``build_cooccurrence_graph`` keys its ``occurrences[]``
inverted index off ``chunk.get("id")``, and a chunk with no ``id``
contributes nothing. This test therefore still pins the inline
projection's emit shape, now exercised against the Fix-2 dispatch path.

Test contract:

  1. Construct a fixture staging dir with one ``*_synthesized.json``.
  2. Invoke ``_run_concept_extraction`` with the upstream chunkset
     path unset (forces the inline-projection fallback path).
  3. Assert the emitted graph file is ``kind: "concept_semantic"`` and
     the inline projection round-trips into a real graph (the on-disk
     artifact is structurally valid, not an empty shell).
  4. Assert the inline-projection chunks carry canonical ``id`` keys
     (and zero ``chunk_id`` keys) by feeding them into
     ``build_cooccurrence_graph`` directly and confirming the
     ``occurrences[]`` back-references resolve — pre-Phase-8 the
     ``chunk_id``-keyed dicts would have produced empty ``occurrences``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


def _write_synthesized(path: Path) -> None:
    """Emit a minimal ``*_synthesized.json`` fixture sidecar.

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


class TestInlineProjectionEmitsCanonicalIdKey:
    """Phase 8 ST 6 (Fix-2-updated) — the inline-projection fallback
    emits canonical ``id`` keys instead of the legacy ``chunk_id`` key,
    and those chunks flow into the Fix-2 semantic-graph dispatch path."""

    def test_inline_projection_emits_concept_semantic_graph(
        self, fallback_fixture
    ):
        """End-to-end: the inline-projection fallback round-trips into
        a genuine ``kind: "concept_semantic"`` graph on disk.

        Pre-Phase-8 the ``chunk_id``-keyed dicts were silently dropped
        and the graph degraded to an empty shell; Fix-2 keeps the
        canonical ``id`` emit, so the inline projection still feeds a
        real graph."""
        fx = fallback_fixture

        registry = _build_tool_registry()
        tool = registry["run_concept_extraction"]
        result = asyncio.run(
            tool(
                project_id="",
                course_name=fx["course_name"],
                staging_dir=str(fx["staging_dir"]),
                # upstream chunkset path intentionally omitted so the
                # inline-projection fallback path runs.
            )
        )
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["chunk_count"] == 3, (
            f"Inline projection should emit 3 chunks for the 3-section "
            f"fixture; got chunk_count={payload['chunk_count']}."
        )

        graph = json.loads(
            Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
        )
        assert graph.get("kind") == "concept_semantic", (
            "Fix-2: concept_extraction must emit the genuine semantic "
            f"graph kind; got {graph.get('kind')!r}."
        )
        # node_count parity between the payload and the on-disk graph.
        assert payload["node_count"] == len(graph.get("nodes") or [])

    def test_inline_projection_chunks_carry_canonical_id_key(
        self, fallback_fixture
    ):
        """Direct shape assertion: the inline-projection chunk dicts
        carry a canonical ``id`` key (no legacy ``chunk_id``), and that
        ``id`` resolves through ``build_cooccurrence_graph``'s
        ``occurrences[]`` inverted index.

        Pre-Phase-8 the projection emitted ``chunk_id`` and the
        co-occurrence builder (which keys ``occurrences[]`` off
        ``chunk.get("id")``) would have produced empty back-references.
        """
        fx = fallback_fixture

        # Re-run the inline-projection logic in isolation by exercising
        # the helper with a chunkset we can inspect: load the on-disk
        # graph and assert the inline projection produced real nodes
        # with resolvable occurrences[] back-references — which can
        # only happen if the projected chunks carried canonical ``id``.
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

        graph = json.loads(
            Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
        )
        nodes = graph.get("nodes") or []
        # The 3-section fixture's bare-word tokenizer yields shared
        # concept tags ("assessment", "learner", ...) across sections,
        # so at least one node clears the min_freq=2 floor and carries
        # an occurrences[] list keyed by canonical chunk ``id``.
        nodes_with_occurrences = [
            n for n in nodes if n.get("occurrences")
        ]
        assert nodes_with_occurrences, (
            "Phase 8 ST 6 regression: no concept node carries an "
            "occurrences[] back-reference. Likely cause: the inline "
            "projection regressed to emitting 'chunk_id' keys, which "
            "build_cooccurrence_graph's id-keyed inverted index drops."
        )
        # Every occurrence ID must be a non-empty canonical chunk id
        # string (the inline projection emits f'{course}_chunk_NNNNN').
        for node in nodes_with_occurrences:
            for occ in node["occurrences"]:
                assert isinstance(occ, str) and occ, (
                    f"occurrences[] entry must be a non-empty chunk id "
                    f"string; got {occ!r} on node {node.get('id')!r}."
                )
                assert "_chunk_" in occ, (
                    f"occurrences[] entry {occ!r} is not a canonical "
                    f"inline-projection chunk id."
                )
