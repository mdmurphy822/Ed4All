"""Emit-pipeline enrichment is baked into ``CourseProcessor.process``.

Concept classification, concept-noise filtering, chunk retagging, and
the real pedagogy-graph builder must all run inside a normal
``process()`` call. When they only existed as retroactive scripts a
fresh archive emitted a stub pedagogy graph (1 node / 0 edges) and
unclassified concept-graph nodes, and had to be repaired by hand
afterwards. These tests pin the contract at the integration boundary:

* Every concept-graph node carries ``class``.
* The pedagogy graph from ``process()`` has > 1 node and > 0 edges
  across multiple relation types — i.e. the real builder, not the stub.
* Chunks whose body text matches a retag vocabulary entry carry the
  matching CO ref plus its parent TO.

A companion test (``test_emit_pipeline_full_archive.py``) exercises a
full IMSCC end-to-end behind an env gate so this suite stays fast.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402


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
          <lomimscc:string language="en">Enrichment Fixture</lomimscc:string>
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
    """Week 1 page — RDF triples + intro vocabulary.

    Names ``RDF Graph``, ``RDF Triple``, ``Subject``, ``Predicate``,
    ``Blank Node`` so the concept-graph extractor lands real domain
    terms. Carries ``Component Objective: CO-01`` so the per-chunk LO
    ref extractor binds chunks to ``co-01``.
    """
    return """<!DOCTYPE html>
<html>
<head><title>Introduction to RDF</title></head>
<body>
<h1>Introduction to RDF</h1>
<p>Component Objective: CO-01 — Identify subjects, predicates, and objects in RDF triples.</p>

<h2>RDF Triples</h2>
<p>An RDF triple consists of a subject, a predicate, and an object.
   Each subject is an IRI or a blank node. Predicates are always IRIs.
   The object can be an IRI, a literal, or a blank node.</p>
<p>An RDF graph is a set of RDF triples.</p>

<h2>Working with Triples</h2>
<p>Triples can be serialized in Turtle, RDF/XML, or JSON-LD.
   The Turtle serialization is the most human-readable.</p>
<p>A blank node is an existential variable in an RDF graph.</p>
</body>
</html>
"""


def _page_two() -> str:
    """Week 2 page — IRIs + SPARQL vocabulary.

    Includes the SHACL Core constraint vocabulary the retag rule looks
    for (``sh:minCount`` / ``sh:maxCount`` / ``sh:datatype``) so chunks
    emitted from this page should pick up ``co-18`` plus its parent
    ``to-04``.
    """
    return """<!DOCTYPE html>
<html>
<head><title>SHACL Constraint Components</title></head>
<body>
<h1>SHACL Constraint Components</h1>
<p>Component Objective: CO-02 — Distinguish IRIs from literals.</p>

<h2>SHACL Core Constraints</h2>
<p>SHACL Core defines the foundational constraint components.
   sh:minCount sets a lower bound on the number of values for a property.
   sh:maxCount sets an upper bound.
   sh:datatype constrains the datatype of literal values.</p>
<p>An IRI is an Internationalized Resource Identifier; a literal is
   a typed value (xsd:string, xsd:integer, etc.).</p>

<h2>Practice</h2>
<p>Write a SPARQL SELECT query that retrieves all triples where the
   predicate is rdfs:label.</p>
</body>
</html>
"""


def _objectives_payload() -> Dict[str, Any]:
    """Canonical objectives.json shape with the COs the retag rule
    rolls up to.

    ``co-18`` rolls up to ``to-04`` via ``parent_to``; the chunk text
    on page two cites ``sh:minCount`` / ``sh:maxCount`` / ``sh:datatype``
    so the vocabulary retag should add ``co-18`` AND ``to-04`` to the
    chunk's ``learning_outcome_refs``.
    """
    # ``bloom_level`` must stay snake_case: that is the canonical key in
    # ``schemas/knowledge/objectives_v1.schema.json`` and the key
    # ``Trainforge.pedagogy_graph_builder`` reads. Courseforge's
    # synthesized objectives use camelCase ``bloomLevel``, which the
    # builder does NOT read — using it here would silently skip the
    # ``at_bloom_level`` rule.
    return {
        "schema_version": "v1",
        "course_code": "ENRICHMENT_FIXTURE",
        "duration_weeks": 2,
        "domain": "knowledge_graphs",
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Analyze RDF graphs.",
                "bloom_level": "analyze",
            },
            {
                "id": "TO-04",
                "statement": "Apply SHACL constraints.",
                "bloom_level": "apply",
            },
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "statement": "Identify subject, predicate, and object.",
                "parent_to": "TO-01",
                "bloom_level": "remember",
                "week": 1,
            },
            {
                "id": "CO-02",
                "statement": "Distinguish IRIs from literals.",
                "parent_to": "TO-01",
                "bloom_level": "understand",
                "week": 2,
            },
            {
                "id": "CO-18",
                "statement": "Apply SHACL Core constraint components.",
                "parent_to": "TO-04",
                "bloom_level": "apply",
                "week": 2,
            },
        ],
    }


def _build_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Write a minimal IMSCC + objectives sidecar.

    Returns ``(imscc_path, objectives_path)``.
    """
    imscc = tmp_path / "enrichment_fixture.imscc"
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
        course_code="ENRICHMENT_FIXTURE",
        domain="knowledge_graphs",
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
# Test 2: pedagogy graph is the real builder, not the stub
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pedagogy_graph_has_multiple_nodes_and_edge_types(tmp_path):
    """pedagogy_graph.json must be real builder output, not the stub.

    The legacy stub (tag co-occurrence over PEDAGOGY/LOGISTICS tags)
    emitted ~1 node and 0 edges on real corpora, which is why the
    thresholds below are the discriminator. The real builder emits at
    minimum:

    * BloomLevel typed nodes (6) — unconditional.
    * DifficultyLevel typed nodes (3) — unconditional.
    * Outcome / ComponentObjective nodes from objectives.
    * Chunk + Module nodes from the corpus.
    * Multiple typed edges (``teaches`` / ``supports_outcome`` /
      ``at_bloom_level`` / ``follows`` / ``belongs_to_module`` /
      ``derived_from_objective`` / ``chunk_at_difficulty`` / …).

    On this fixture (2 chunks, 3 COs, 2 TOs, 2 modules) that is at
    least 14 nodes (6 bloom + 3 difficulty + 2 TO + 3 CO).
    """
    out = _run_processor(tmp_path)
    pg_path = out / "graph" / "pedagogy_graph.json"
    assert pg_path.exists(), "pedagogy_graph.json must be written"

    pg = json.loads(pg_path.read_text(encoding="utf-8"))
    nodes = pg.get("nodes") or []
    edges = pg.get("edges") or []

    assert len(nodes) > 1, (
        f"stub-shaped pedagogy_graph ({len(nodes)} nodes). The real "
        "builder emits at least 14 nodes (6 bloom + 3 difficulty + "
        "objectives); the legacy stub emitted 1."
    )
    assert len(edges) > 0, (
        "pedagogy_graph emitted 0 edges — the legacy stub's signature."
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
def test_pedagogy_graph_includes_wave78_relations(tmp_path):
    """Chunk-anchored relation types ship from ``process()`` itself.

    The fixture's chunks carry ``learning_outcome_refs`` (the in-page
    ``Component Objective: CO-01`` token plus retag pickups), so the
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
    the right CO ref AND its parent TO via parent-rollup. The fixture's
    page-two chunk carries ``sh:minCount`` / ``sh:maxCount`` /
    ``sh:datatype`` (vocabulary entries for ``co-18``), and ``CO-18``
    declares ``parent_to: TO-04``, so the chunk must also carry
    ``to-04``.

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

    # Find the chunk(s) whose text mentions the SHACL vocabulary —
    # those are the ones the retag rule targets.
    shacl_chunks = [
        c for c in chunks
        if isinstance(c.get("text"), str)
        and ("sh:minCount" in c["text"] or "sh:maxCount" in c["text"])
    ]
    assert len(shacl_chunks) > 0, (
        "fixture regression: at least one chunk must carry "
        "'sh:minCount'/'sh:maxCount' in its text — without that signal "
        "the retag assertions below would pass vacuously."
    )

    # Compare case-insensitively: TRAINFORGE_PRESERVE_LO_CASE controls
    # emit casing, so pinning a literal case would make this test
    # flag-dependent. The canonical id is what matters here.
    co18_hits = [
        c for c in shacl_chunks
        if any(
            isinstance(r, str) and r.lower() == "co-18"
            for r in c.get("learning_outcome_refs") or []
        )
    ]
    assert co18_hits, (
        "no chunk picked up co-18 via the vocabulary retag. Chunks "
        "with SHACL vocab in text:\n"
        + "\n".join(
            f"  text='...{c['text'][:100]}...' refs={c.get('learning_outcome_refs')}"
            for c in shacl_chunks
        )
    )

    to04_hits = [
        c for c in shacl_chunks
        if any(
            isinstance(r, str) and r.lower() == "to-04"
            for r in c.get("learning_outcome_refs") or []
        )
    ]
    assert to04_hits, (
        "parent-rollup didn't add to-04 to a chunk that already cites "
        "co-18. Check that build_parent_map runs at CourseProcessor "
        "construction time and that _create_chunk calls the retag."
    )
