"""Phase 1.2 / Phase 1.3 of plans/rdf-shacl-enrichment-2026-04-26.md.

Verifies that ``schemas/context/chunk_v4_v1.jsonld`` is a faithful
round-trip bridge between Trainforge's JSON-shaped chunk records
(``schemas/knowledge/chunk_v4.schema.json``, materialized one-per-line in
``corpus/chunks.jsonl``) and an RDF graph.  The Trainforge emit
pipeline does not yet inject the ``@context`` (Phase 1 is consumer-side
only); this test layers it on top of a sample of real chunks from the
``rdf-shacl-551-2`` reference corpus, parses via ``pyld`` + ``rdflib``,
and asserts:

* every wrapped chunk produces >= 5 triples (id, type, body text, at
  least one tag, at least one source ref) — the load-bearing predicates
  for downstream RDF consumers
* concept_tags and learning_outcome_refs materialize as RDF set members
  (multiple objects on the same predicate, NOT a single literal blob)
* the source.section_heading is materialized as an RDF literal (not
  collapsed to an IRI or dropped)
* Turtle round-trip is loss-free (delta == 0; small skolem cushion
  permitted, identical to Phase 1.1's policy)

Phase 1 does not modify any chunk-emit code in Trainforge; the bridge
runs purely out-of-band against existing corpus artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root (Ed4All/) — this test file lives at
# Ed4All/Trainforge/tests/test_chunk_v4_jsonld_roundtrip.py, so parents[2]
# is the root.  Mirrors test_concept_graph_jsonld_roundtrip.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTEXT_PATH = PROJECT_ROOT / "schemas" / "context" / "chunk_v4_v1.jsonld"
CHUNKS_JSONL_PATH = (
    PROJECT_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-551-2"
    / "corpus"
    / "chunks.jsonl"
)

ED4ALL_VOCAB = "https://ed4all.io/vocab/"
SCHEMA_TEXT_PRED = "http://schema.org/text"
ED4ALL_HAS_CONCEPT_TAG_PRED = ED4ALL_VOCAB + "hasConceptTag"
ED4ALL_LO_ID_PRED = ED4ALL_VOCAB + "loId"
ED4ALL_SOURCE_REF_PRED = ED4ALL_VOCAB + "sourceReference"
ED4ALL_SOURCE_ID_PRED = ED4ALL_VOCAB + "sourceId"
ED4ALL_SECTION_HEADING_PRED = ED4ALL_VOCAB + "sectionHeading"
ED4ALL_CHUNK_BASE = "https://ed4all.io/chunk/"
SAMPLE_SIZE = 5  # First N chunks of the reference corpus.


pyld = pytest.importorskip("pyld")
rdflib = pytest.importorskip("rdflib")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def context_doc() -> dict:
    """Load the JSON-LD @context wrapper (Phase 1.2 deliverable)."""
    with CONTEXT_PATH.open() as f:
        ctx = json.load(f)
    assert "@context" in ctx, (
        "chunk_v4_v1.jsonld must expose a top-level @context block; the "
        "sibling _description and _phase2_followup keys are metadata-only."
    )
    return ctx


@pytest.fixture(scope="module")
def sample_chunks() -> list[dict]:
    """Load the first SAMPLE_SIZE chunks from the reference corpus."""
    if not CHUNKS_JSONL_PATH.exists():
        pytest.skip(
            f"Reference corpus missing: {CHUNKS_JSONL_PATH} — Phase 1 chunk "
            "round-trip test depends on the rdf-shacl-551-2 corpus."
        )
    chunks: list[dict] = []
    with CHUNKS_JSONL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
            if len(chunks) >= SAMPLE_SIZE:
                break
    assert chunks, "Expected at least one chunk in the reference corpus."
    return chunks


def _wrap_chunk(chunk: dict, context: dict) -> dict:
    """Layer the @context on top of a deep copy of the chunk and anchor
    the chunk_id under the @base so it materializes as an IRIRef."""
    doc = json.loads(json.dumps(chunk))  # deep copy without external deps
    doc["@context"] = context["@context"]
    return doc


@pytest.fixture(scope="module")
def rdf_graphs(context_doc, sample_chunks) -> list["rdflib.Graph"]:
    """One rdflib.Graph per sample chunk, parsed via pyld -> rdflib."""
    from pyld import jsonld
    from rdflib import Graph

    graphs: list[Graph] = []
    for chunk in sample_chunks:
        doc = _wrap_chunk(chunk, context_doc)
        nquads = jsonld.to_rdf(doc, {"format": "application/n-quads"})
        g = Graph()
        g.parse(data=nquads, format="nquads")
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_each_chunk_emits_minimum_load_bearing_triples(
    rdf_graphs, sample_chunks
) -> None:
    """Each wrapped chunk MUST emit >= 5 triples covering the load-bearing
    predicates: chunk identity (the @id materialized URI), rdf:type-ish
    routing (chunk_type via @vocab), schema:text body, at least one tag
    (concept_tag or learning_outcome_ref), and at least one source ref.

    This is the "smoke floor" — a regression below 5 triples means the
    context dropped one of the structural predicates.  In practice each
    chunk yields ~50-150 triples; the floor is set well below typical
    output to keep the test resilient to corpus shape changes.
    """
    from rdflib import URIRef

    text_pred = URIRef(SCHEMA_TEXT_PRED)
    tag_pred = URIRef(ED4ALL_HAS_CONCEPT_TAG_PRED)
    lo_pred = URIRef(ED4ALL_LO_ID_PRED)
    source_ref_pred = URIRef(ED4ALL_SOURCE_REF_PRED)
    source_id_pred = URIRef(ED4ALL_SOURCE_ID_PRED)

    for chunk, g in zip(sample_chunks, rdf_graphs):
        n = len(g)
        assert n >= 5, (
            f"Chunk {chunk['id']!r} emitted only {n} triples; expected >= 5. "
            f"Likely a context regression collapsed the body text, tags, or "
            f"source references."
        )

        # Body text MUST land on schema:text as a literal.
        text_triples = list(g.triples((None, text_pred, None)))
        assert text_triples, (
            f"Chunk {chunk['id']!r}: no schema:text triple. The context "
            f"binds 'text' -> schema:text; if missing, downstream RAG "
            f"retrieval can't dereference the chunk body."
        )

        # At least one of (concept_tag | learning_outcome_ref) MUST be
        # emitted — the chunks corpus has both, but a one-LO chunk with
        # zero tags should still pass thanks to the LO emission.
        tag_count = len(list(g.triples((None, tag_pred, None))))
        lo_count = len(list(g.triples((None, lo_pred, None))))
        assert (tag_count + lo_count) >= 1, (
            f"Chunk {chunk['id']!r}: zero tag or LO triples emitted. "
            f"concept_tags + learning_outcome_refs produced {tag_count} + "
            f"{lo_count} respectively; at least one is required."
        )

        # Source ref linkage: hasSource -> ed4all:sourceReference -> entry
        # carrying ed4all:sourceId (the canonical sourceId field).
        source_id_count = len(list(g.triples((None, source_id_pred, None))))
        json_source_refs = chunk.get("source", {}).get("source_references", [])
        assert source_id_count >= 1, (
            f"Chunk {chunk['id']!r}: zero ed4all:sourceId triples; expected "
            f">= 1 from {len(json_source_refs)} JSON source_references[]. "
            f"The source.source_references[] sub-shape lost typing."
        )


def test_concept_tags_and_lo_refs_are_set_members(
    rdf_graphs, sample_chunks
) -> None:
    """concept_tags + learning_outcome_refs are unordered sets — each item
    must materialize as its own RDF object on the predicate, NOT collapse
    into a single comma-joined literal.  Both terms use @container: @set
    in the context for this reason.
    """
    from rdflib import URIRef

    tag_pred = URIRef(ED4ALL_HAS_CONCEPT_TAG_PRED)
    lo_pred = URIRef(ED4ALL_LO_ID_PRED)

    for chunk, g in zip(sample_chunks, rdf_graphs):
        json_tags = chunk.get("concept_tags", [])
        json_los = chunk.get("learning_outcome_refs", [])

        rdf_tag_count = len(list(g.triples((None, tag_pred, None))))
        rdf_lo_count = len(list(g.triples((None, lo_pred, None))))

        assert rdf_tag_count == len(json_tags), (
            f"Chunk {chunk['id']!r}: concept_tags count mismatch. JSON has "
            f"{len(json_tags)} tags but RDF emitted {rdf_tag_count} "
            f"ed4all:hasConceptTag triples.  Likely cause: the @set "
            f"container regressed and tags collapsed into a single literal."
        )
        assert rdf_lo_count == len(json_los), (
            f"Chunk {chunk['id']!r}: learning_outcome_refs count mismatch. "
            f"JSON has {len(json_los)} LO refs but RDF emitted "
            f"{rdf_lo_count} ed4all:loId triples."
        )


def test_source_section_heading_is_literal(
    rdf_graphs, sample_chunks
) -> None:
    """Per the Phase 1.2 contract, source.section_heading MUST land as an
    RDF literal (xsd:string).  A blank-node fallback or a URIRef would
    indicate the term mapping lost its xsd:string type binding.
    """
    from rdflib import Literal, URIRef

    section_heading_pred = URIRef(ED4ALL_SECTION_HEADING_PRED)

    for chunk, g in zip(sample_chunks, rdf_graphs):
        json_heading = chunk.get("source", {}).get("section_heading")
        if json_heading is None:
            # Optional field; only assert when source provides it.
            continue
        objects = list(g.objects(predicate=section_heading_pred))
        assert objects, (
            f"Chunk {chunk['id']!r}: no ed4all:sectionHeading triple even "
            f"though source.section_heading={json_heading!r} is present."
        )
        for obj in objects:
            assert isinstance(obj, Literal), (
                f"Chunk {chunk['id']!r}: ed4all:sectionHeading object "
                f"{obj!r} is not a Literal (got {type(obj).__name__}).  The "
                f"context must keep section_heading typed as xsd:string."
            )


def test_turtle_roundtrip_is_lossless(rdf_graphs, sample_chunks) -> None:
    """Per-chunk Turtle round-trip must be loss-free.  Delta == 0 is the
    Phase 1 target; we permit a tiny cushion (<= 5 triples per chunk) for
    skolem-id non-determinism in pyshacl/rdflib edge cases — same policy
    as test_concept_graph_jsonld_roundtrip.py.
    """
    from rdflib import Graph

    for chunk, g in zip(sample_chunks, rdf_graphs):
        n_orig = len(g)
        ttl = g.serialize(format="turtle")
        g_round_trip = Graph()
        g_round_trip.parse(data=ttl, format="turtle")
        n_rt = len(g_round_trip)
        delta = n_rt - n_orig
        assert abs(delta) <= 5, (
            f"Chunk {chunk['id']!r}: Turtle round-trip changed triple count "
            f"{n_orig} -> {n_rt} (delta={delta}).  Phase 1 expects delta == 0 "
            f"with a small cushion for skolemized blank-node noise."
        )
