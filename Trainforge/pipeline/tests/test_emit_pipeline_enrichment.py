"""Emit-pipeline enrichment is baked into ``CourseProcessor.process``.

Concept classification, concept-noise filtering, chunk retagging, and
the complete pedagogy-graph builder must all run inside a normal
``process()`` call. These tests pin the contract at the integration boundary:

* Every concept-graph node carries ``class``.
* The pedagogy graph from ``process()`` has > 1 node and > 0 edges
  across multiple relation types.
* Chunks whose body text matches a retag vocabulary entry carry the
  matching component-objective reference plus its terminal parent.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.pipeline.process_course import CourseProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic IMSCC fixture builder
# ---------------------------------------------------------------------------


def _imscc_manifest() -> str:
    """Minimal IMS CC v1.1 manifest pointing at our two HTML resources."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.1.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title>
          <lomimscc:string language="en">Synthetic Systems Lab</lomimscc:string>
        </lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  <resources>
    <resource identifier="r1" type="webcontent" href="week_01/content.html">
      <file href="week_01/content.html"/>
    </resource>
    <resource identifier="r2" type="webcontent" href="week_02/content.html">
      <file href="week_02/content.html"/>
    </resource>
  </resources>
</manifest>
"""


def _page_one() -> str:
    """First page with neutral system-modeling terminology.

    Repeated CamelCase and prefixed identifiers give concept classification and
    runtime-derived objective matching stable, synthetic signals.
    """
    return """<!DOCTYPE html>
<html>
<head><title>Signal Network Foundations</title></head>
<body>
<h1>Signal Network Foundations</h1>
<p>Learning goal: identify SYS:ALPHA and its NodeType.</p>

<h2>Signal Nodes</h2>
<p>SYS:ALPHA is a NodeType in the synthetic signal network.
   A SignalRouter connects a SourceNode to a TargetNode and records a
   stable ChannelCode for each connection.</p>
<p>The network groups SignalRouter nodes into ordered paths.</p>

<h2>Working with Signals</h2>
<p>Operators inspect a SourceNode, TargetNode, and ChannelCode together.
   The NodeType determines which transformations are permitted.</p>
<p>A SignalRouter preserves the order of the synthetic path.</p>
</body>
</html>
"""


def _page_two() -> str:
    """Second page with the runtime-derived retagging signal.

    ``SYS:BETA`` and ``ThermalNode`` appear in both the supplied objective and
    page body, so no built-in vocabulary is needed for the match.
    """
    return """<!DOCTYPE html>
<html>
<head><title>Thermal Signal Controls</title></head>
<body>
<h1>Thermal Signal Controls</h1>
<p>Learning goal: apply SYS:BETA with a ThermalNode.</p>

<h2>Thermal Nodes</h2>
<p>SYS:BETA routes each ThermalNode through a HeatLimiter.
   The HeatLimiter records a TemperatureBand and a SafetyState before
   forwarding the signal.</p>
<p>A ThermalNode enters HoldMode when its TemperatureBand is exceeded.</p>

<h2>Practice</h2>
<p>Trace SYS:BETA through the ThermalNode and explain when the
   HeatLimiter selects HoldMode.</p>
</body>
</html>
"""


def _objectives_payload() -> Dict[str, Any]:
    """Objective payload with runtime-derived terms and parent links.

    ``OBJ-43`` rolls up to ``GOAL-72``. Its prefixed and
    CamelCase terms also appear on page two, which exercises generic vocabulary
    extraction rather than a built-in course table.
    """
    # ``bloom_level`` must stay snake_case: that is the canonical key in
    # ``schemas/knowledge/objectives_v1.schema.json`` and the key
    # ``Trainforge.rag.graphs.pedagogy_graph_builder`` reads. Courseforge's
    # synthesized objectives use camelCase ``bloomLevel``, which the
    # builder does NOT read — using it here would silently skip the
    # ``at_bloom_level`` rule.
    return {
        "schema_version": "v1",
        "course_code": "SYNTHETIC_SYSTEMS",
        "duration_weeks": 2,
        "domain": "systems_modeling",
        "terminal_objectives": [
            {
                "id": "GOAL-71",
                "statement": "Analyze SignalNetwork behavior.",
                "bloom_level": "analyze",
            },
            {
                "id": "GOAL-72",
                "statement": "Apply ThermalControl safeguards.",
                "bloom_level": "apply",
            },
        ],
        "chapter_objectives": [
            {
                "id": "OBJ-41",
                "statement": "Identify SYS:ALPHA NodeType behavior.",
                "parent_to": "GOAL-71",
                "bloom_level": "remember",
                "week": 1,
            },
            {
                "id": "OBJ-42",
                "statement": "Distinguish SourceNode from TargetNode.",
                "parent_to": "GOAL-71",
                "bloom_level": "understand",
                "week": 2,
            },
            {
                "id": "OBJ-43",
                "statement": "Apply SYS:BETA ThermalNode controls.",
                "parent_to": "GOAL-72",
                "bloom_level": "apply",
                "week": 2,
            },
        ],
    }


def _build_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Write a minimal IMSCC + objectives sidecar.

    Returns ``(imscc_path, objectives_path)``.
    """
    imscc = tmp_path / "synthetic_systems.imscc"
    with zipfile.ZipFile(imscc, "w") as zf:
        zf.writestr("imsmanifest.xml", _imscc_manifest())
        zf.writestr("week_01/content.html", _page_one())
        zf.writestr("week_02/content.html", _page_two())

    obj_path = tmp_path / "objectives.json"
    obj_path.write_text(json.dumps(_objectives_payload()), encoding="utf-8")

    return imscc, obj_path


def _run_processor(tmp_path: Path) -> Path:
    """Run the full ``CourseProcessor.process`` pipeline; return ``output_dir``."""
    imscc, obj_path = _build_fixture(tmp_path)
    out = tmp_path / "trainforge_out"
    out.mkdir()

    proc = CourseProcessor(
        imscc_path=str(imscc),
        output_dir=str(out),
        course_code="SYNTHETIC_SYSTEMS",
        domain="systems_modeling",
        objectives_path=str(obj_path),
    )
    proc.process()
    return out


# ---------------------------------------------------------------------------
# Test 1: every concept-graph node carries ``class``
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concept_graph_nodes_carry_class_field(tmp_path):
    """``concept_graph.json`` must carry ``class`` on every node.

    The pedagogy graph consumes this field: if the stamping regressed,
    the concept-supports-outcome filter would silently default every
    node to DomainConcept and emit no edges of that type. Pinned at the
    integration boundary, not just at the classifier.
    """
    out = _run_processor(tmp_path)
    cg_path = out / "graph" / "concept_graph.json"
    assert cg_path.exists(), "concept_graph.json must be written"

    cg = json.loads(cg_path.read_text(encoding="utf-8"))
    nodes = cg.get("nodes") or []
    # Node count depends on chunker behavior, so assert the property
    # that must hold for however many nodes are emitted.
    assert all(
        isinstance(n.get("class"), str) and n.get("class")
        for n in nodes
    ), (
        "every concept_graph node must carry a non-empty 'class' "
        "field. Offending nodes: "
        + repr([n for n in nodes if not n.get("class")])[:500]
    )


# ---------------------------------------------------------------------------
# Test 2: pedagogy graph contains meaningful structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pedagogy_graph_has_multiple_nodes_and_edge_types(tmp_path):
    """The emitted pedagogy graph must contain nodes and typed edges.

    The builder emits at minimum:

    * BloomLevel typed nodes (6) — unconditional.
    * DifficultyLevel typed nodes (3) — unconditional.
    * Outcome / ComponentObjective nodes from objectives.
    * Chunk + Module nodes from the corpus.
    * Multiple typed edges (``teaches`` / ``supports_outcome`` /
      ``at_bloom_level`` / ``follows`` / ``belongs_to_module`` /
      ``derived_from_objective`` / ``chunk_at_difficulty`` / …).

    The fixture supplies multiple objectives, pages, and modules so the graph
    must be nontrivial.
    """
    out = _run_processor(tmp_path)
    pg_path = out / "graph" / "pedagogy_graph.json"
    assert pg_path.exists(), "pedagogy_graph.json must be written"

    pg = json.loads(pg_path.read_text(encoding="utf-8"))
    nodes = pg.get("nodes") or []
    edges = pg.get("edges") or []

    assert len(nodes) > 1, (
        f"pedagogy_graph is not structurally useful ({len(nodes)} nodes)."
    )
    assert len(edges) > 0, (
        "pedagogy_graph emitted no relationships."
    )

    relation_types = {e.get("relation_type") for e in edges}
    assert len(relation_types) >= 2, (
        f"pedagogy_graph emitted {len(relation_types)} distinct "
        f"relation_types — must be at least 2. "
        f"Got: {sorted(relation_types)}"
    )

    # Named-type spot checks lock the WIRING rather than a count: a
    # builder that emits plenty of edges but never routes objectives
    # would still clear the thresholds above.
    assert "supports_outcome" in relation_types, (
        "pedagogy graph missing 'supports_outcome' edges — objectives "
        "weren't routed to the builder."
    )
    assert "at_bloom_level" in relation_types, (
        "pedagogy graph missing 'at_bloom_level' edges — BloomLevel "
        "typed nodes weren't connected to objectives."
    )

    # Same ``class`` contract as the concept graph, locked here at the
    # integration boundary as well as in the builder's own unit tests.
    assert all(
        isinstance(n.get("class"), str) and n.get("class")
        for n in nodes
    ), "pedagogy_graph node missing 'class' field"


@pytest.mark.unit
def test_pedagogy_graph_includes_chunk_anchored_relations(tmp_path):
    """Chunk-anchored relation types ship from ``process()`` itself.

    The fixture's chunks carry runtime-derived ``learning_outcome_refs``, so the
    builder must fire ``derived_from_objective`` on every chunk with
    ≥ 1 ref. Every chunk lands a ``difficulty`` value (``foundational``
    by default for explanation chunks), so ``chunk_at_difficulty``
    must always emit at least one edge.
    """
    out = _run_processor(tmp_path)
    pg = json.loads((out / "graph" / "pedagogy_graph.json").read_text("utf-8"))
    relation_types = {e.get("relation_type") for e in pg.get("edges") or []}

    assert "derived_from_objective" in relation_types, (
        "'derived_from_objective' edges missing — chunk LO refs aren't "
        f"reaching the pedagogy builder. Got: {sorted(relation_types)}"
    )
    assert "chunk_at_difficulty" in relation_types, (
        "'chunk_at_difficulty' edges missing — chunks aren't carrying "
        "a 'difficulty' attribute, or DifficultyLevel nodes weren't "
        f"seeded. Got: {sorted(relation_types)}"
    )


# ---------------------------------------------------------------------------
# Test 3: retag fired on chunks that mention vocabulary terms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunks_pick_up_retag_vocabulary_refs(tmp_path):
    """``retag_chunk_outcomes`` fires at chunk-emit time.

    A chunk whose ``text`` matches the retag vocabulary must pick up
    the matching objective reference and its parent via parent rollup. The fixture's
    page-two chunk carries ``SYS:BETA`` and ``ThermalNode``, and
    ``OBJ-43`` declares ``parent_to: GOAL-72``, so the chunk
    must carry both references.

    The retag is wired inside ``_create_chunk``; this pins that
    integration contract rather than the retag helper in isolation.
    """
    out = _run_processor(tmp_path)
    # process_course.py writes its chunkset to imscc_chunks/.
    chunks_path = out / "imscc_chunks" / "chunks.jsonl"
    assert chunks_path.exists(), "imscc_chunks/chunks.jsonl must be written"

    chunks = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    assert len(chunks) > 0, "fixture should produce at least one chunk"

    # Find chunks containing the supplied objective's retained identifiers.
    thermal_chunks = [
        c for c in chunks
        if isinstance(c.get("text"), str)
        and "SYS:BETA" in c["text"]
        and "ThermalNode" in c["text"]
    ]
    assert thermal_chunks, (
        "fixture regression: at least one chunk must carry "
        "'SYS:BETA' and 'ThermalNode' in its text; without that signal "
        "the retag assertions below would pass vacuously."
    )

    # Compare case-insensitively: TRAINFORGE_PRESERVE_LO_CASE controls
    # emit casing, so pinning a literal case would make this test
    # flag-dependent. The canonical id is what matters here.
    objective_hits = [
        c for c in thermal_chunks
        if any(
            isinstance(r, str) and r.lower() == "obj-43"
            for r in c.get("learning_outcome_refs") or []
        )
    ]
    assert objective_hits, (
        "no chunk picked up OBJ-43 from runtime vocabulary. "
        "Matching chunks:\n"
        + "\n".join(
            f"  text='...{c['text'][:100]}...' refs={c.get('learning_outcome_refs')}"
            for c in thermal_chunks
        )
    )

    parent_hits = [
        c for c in thermal_chunks
        if any(
            isinstance(r, str) and r.lower() == "goal-72"
            for r in c.get("learning_outcome_refs") or []
        )
    ]
    assert parent_hits, (
        "parent rollup did not add GOAL-72 to a chunk tagged with "
        "OBJ-43. Check that build_parent_map runs at CourseProcessor "
        "construction time and that _create_chunk calls the retag."
    )
