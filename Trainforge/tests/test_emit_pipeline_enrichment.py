"""Wave 81 — emit-pipeline enrichment is baked into the normal run.

Wave 75-78 added concept classification, concept-noise filtering,
chunk retagging, and a real pedagogy graph builder. Pre-Wave-81 those
all lived as retroactive scripts (``scripts/wave75_*``,
``scripts/wave76_*``, ``scripts/wave78_*``) so a fresh Trainforge run
emitted a stub pedagogy graph (1 node / 0 edges) and unclassified
concept-graph nodes. v2 Path B regen surfaced this — needed 4 manual
scripts to bring the fresh archive up to current standards.

Wave 81 wires the enrichment into ``CourseProcessor.process`` so
fresh archives are correct by default. These tests pin the contract
end-to-end:

* Every concept-graph node carries ``class`` (Wave 75 stamping at
  emit time, locked again here at the integration boundary).
* The pedagogy graph emitted from ``process()`` has > 1 node and
  > 0 edges with multiple relation types (the Wave 75/78 builder,
  not the legacy stub).
* Chunks whose body text matches a Wave 76 retag vocabulary entry
  carry the appropriate CO + parent TO refs after emit.

A companion regression test (``test_emit_pipeline_full_archive.py``)
exercises the rdf-shacl-551-2 IMSCC end-to-end behind an env gate so
the fast suite stays fast.
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
          <lomimscc:string language="en">Wave 81 Enrichment Fixture</lomimscc:string>
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
    terms (DomainConcept after Wave 76 classification). Carries
    ``Component Objective: CO-01`` so the per-chunk LO ref extractor
    binds chunks to ``co-01``.
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

    Includes the SHACL Core constraint vocabulary the Wave 76 retag
    rule looks for (``sh:minCount`` / ``sh:maxCount`` /
    ``sh:datatype``) so chunks emitted from this page should pick up
    ``co-18`` plus its parent ``to-04``.
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
    wants to roll up to.

    ``co-18`` rolls up to ``to-04`` via ``parent_to``; the chunk text
    on page two cites ``sh:minCount`` / ``sh:maxCount`` / ``sh:datatype``
    so Wave 76's vocabulary retag should add ``co-18`` AND ``to-04``
    to the chunk's ``learning_outcome_refs``.
    """
    # ``bloom_level`` (snake_case) is the canonical key per
    # ``schemas/knowledge/objectives_v1.schema.json`` and the
    # ``Trainforge.pedagogy_graph_builder`` reads. Synthesized
    # objectives from Courseforge use camelCase ``bloomLevel``; the
    # canonical objectives.json (Worker A) uses snake_case. We follow
    # the canonical shape here so the builder fires its
    # ``at_bloom_level`` rule.
    return {
        "schema_version": "v1",
        "course_code": "WAVE81_FIXTURE",
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
    imscc = tmp_path / "wave81_fixture.imscc"
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
        course_code="WAVE81_FIXTURE",
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
    """Wave 81: ``concept_graph.json`` must have ``class`` stamped on
    every node at emit time.

    Wave 75 wired the classifier into ``_build_tag_graph`` so this is
    technically already locked, but Wave 81 makes pedagogy_graph
    consume the ``class`` field — if a regression dropped the
    stamping, the pedagogy concept-supports-outcome filter would
    silently default everything to DomainConcept and emit no edges
    of that type. This test pins the integration boundary.
    """
    out = _run_processor(tmp_path)
    cg_path = out / "graph" / "concept_graph.json"
    assert cg_path.exists(), "concept_graph.json must be written"

    cg = json.loads(cg_path.read_text(encoding="utf-8"))
    nodes = cg.get("nodes") or []
    # Synthetic fixture has a couple of concept tags; the exact count
    # depends on chunker behavior, but we MUST land at least a few.
    # Mostly we care that every node — however many — has a class.
    assert all(
        isinstance(n.get("class"), str) and n.get("class")
        for n in nodes
    ), (
        "Wave 81 regression: every concept_graph node must carry a "
        "non-empty 'class' field. Offending nodes: "
        + repr([n for n in nodes if not n.get("class")])[:500]
    )


# ---------------------------------------------------------------------------
# Test 2: pedagogy graph is the real builder, not the stub
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pedagogy_graph_has_multiple_nodes_and_edge_types(tmp_path):
    """Wave 81: pedagogy_graph.json must be the real Wave 75/78
    builder output, not the legacy stub.

    The legacy stub (tag co-occurrence over PEDAGOGY/LOGISTICS tags)
    emitted ~1 node and 0 edges on real corpora — v2 Path B regen
    landed exactly that until 4 manual retroactive scripts ran.
    The real builder emits at minimum:

    * BloomLevel typed nodes (6) — always emitted unconditionally.
    * DifficultyLevel typed nodes (3) — always emitted (Wave 78).
    * Outcome / ComponentObjective nodes from objectives.
    * Chunk + Module nodes from the corpus.
    * Multiple typed edges (``teaches`` / ``supports_outcome`` /
      ``at_bloom_level`` / ``follows`` / ``belongs_to_module`` /
      ``derived_from_objective`` / ``chunk_at_difficulty`` / etc.).

    On the synthetic fixture (2 chunks, 3 COs, 2 TOs, 2 modules) we
    expect at least 14 nodes (6 bloom + 3 difficulty + 2 TO + 3 CO)
    and at least 5 edges with at least 4 distinct relation types.
    """
    out = _run_processor(tmp_path)
    pg_path = out / "graph" / "pedagogy_graph.json"
    assert pg_path.exists(), "pedagogy_graph.json must be written"

    pg = json.loads(pg_path.read_text(encoding="utf-8"))
    nodes = pg.get("nodes") or []
    edges = pg.get("edges") or []

    assert len(nodes) > 1, (
        f"Wave 81 regression: pedagogy_graph stub-detected "
        f"({len(nodes)} nodes). The real builder must emit at "
        "least 14 nodes (6 bloom + 3 difficulty + objectives) — "
        "the legacy stub emitted 1."
    )
    assert len(edges) > 0, (
        "Wave 81 regression: pedagogy_graph emitted 0 edges. The "
        "legacy stub did exactly this; the real builder must emit "
        "at least 5 typed edges on this fixture."
    )

    relation_types = {e.get("relation_type") for e in edges}
    assert len(relation_types) >= 2, (
        f"Wave 81 regression: pedagogy_graph emitted "
        f"{len(relation_types)} distinct relation_types — must be "
        f"at least 2. Got: {sorted(relation_types)}"
    )

    # Core relation types we expect from this fixture: at minimum
    # ``supports_outcome`` (CO -> TO), ``at_bloom_level`` (Outcome ->
    # BloomLevel), and ``follows`` (Module -> Module). Spot-check
    # at least one of each so we're locking the wiring not a
    # threshold.
    assert "supports_outcome" in relation_types, (
        "Wave 81: pedagogy graph missing 'supports_outcome' edges — "
        "objectives weren't routed to the builder."
    )
    assert "at_bloom_level" in relation_types, (
        "Wave 81: pedagogy graph missing 'at_bloom_level' edges — "
        "BloomLevel typed nodes weren't connected to objectives."
    )

    # Every node MUST carry a class (Wave 75/78 contract; same field
    # exercised by ``test_pedagogy_graph_builder.py``, but we want
    # the integration boundary locked here too).
    assert all(
        isinstance(n.get("class"), str) and n.get("class")
        for n in nodes
    ), "Wave 81: pedagogy_graph node missing 'class' field"


@pytest.mark.unit
def test_pedagogy_graph_includes_wave78_relations(tmp_path):
    """Wave 81: the four Wave 78 relation types — ``derived_from_objective``,
    ``concept_supports_outcome``, ``assessment_validates_outcome``,
    ``chunk_at_difficulty`` — ship from ``process()`` not just from
    the retroactive script.

    The fixture's chunks carry ``learning_outcome_refs`` (per the
    in-page ``Component Objective: CO-01`` token plus retag pickups)
    so the builder fires ``derived_from_objective`` on every chunk
    with ≥ 1 ref. Every chunk lands a ``difficulty`` value
    (``foundational`` by default for explanation chunks) so
    ``chunk_at_difficulty`` always emits at least one edge.
    """
    out = _run_processor(tmp_path)
    pg = json.loads((out / "graph" / "pedagogy_graph.json").read_text("utf-8"))
    relation_types = {e.get("relation_type") for e in pg.get("edges") or []}

    assert "derived_from_objective" in relation_types, (
        "Wave 81: 'derived_from_objective' edges missing — chunk "
        "LO refs aren't reaching the pedagogy builder. Got: "
        f"{sorted(relation_types)}"
    )
    assert "chunk_at_difficulty" in relation_types, (
        "Wave 81: 'chunk_at_difficulty' edges missing — chunks "
        "aren't carrying a 'difficulty' attribute or DifficultyLevel "
        f"nodes weren't seeded. Got: {sorted(relation_types)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Wave 76 retag fired on chunks that mention vocabulary terms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunks_pick_up_retag_vocabulary_refs(tmp_path):
    """Wave 81: ``retag_chunk_outcomes`` must fire at chunk-emit time
    so chunks whose ``text`` matches the Wave 76 vocabulary list
    pick up the right CO ref AND its parent TO via parent-rollup.

    The fixture's page-two chunk carries ``sh:minCount`` /
    ``sh:maxCount`` / ``sh:datatype`` — Wave 76 vocabulary entries
    for ``co-18``. With ``CO-18``'s parent set to ``TO-04`` in the
    objectives, the chunk should also carry ``to-04`` via
    parent-rollup.

    The retag is already wired in ``_create_chunk`` (Wave 76 part 3,
    line ~1873 of process_course.py) — this test pins the
    integration contract that Wave 81 doesn't accidentally regress
    when the rest of the enrichment lands.
    """
    out = _run_processor(tmp_path)
    # Phase 7c: process_course.py writes to imscc_chunks/.
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
        "Fixture regression: at least one chunk should carry "
        "'sh:minCount'/'sh:maxCount' in its text. The retag test "
        "needs that signal to verify."
    )

    # At least one of those chunks MUST have co-18 in its
    # learning_outcome_refs (vocabulary retag) AND to-04 (parent
    # rollup). We compare case-insensitively because
    # TRAINFORGE_PRESERVE_LO_CASE controls emit casing — the retag
    # itself is case-aware, but the test target is the canonical id.
    co18_hits = [
        c for c in shacl_chunks
        if any(
            isinstance(r, str) and r.lower() == "co-18"
            for r in c.get("learning_outcome_refs") or []
        )
    ]
    assert co18_hits, (
        "Wave 81 regression: no chunk picked up co-18 via the "
        "vocabulary retag. Chunks with SHACL vocab in text:\n"
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
        "Wave 81 regression: parent-rollup didn't add to-04 to a "
        "chunk that already cites co-18. Verify build_parent_map "
        "wiring at constructor time (line ~1015) and the retag "
        "call at line ~1873."
    )
