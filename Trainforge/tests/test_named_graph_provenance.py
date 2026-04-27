"""Phase 3 of plans/rdf-shacl-enrichment-2026-04-26.md — named-graph
provenance tests for the typed-edge inference orchestrator.

Asserts:

* The default-off behaviour of ``TRAINFORGE_EMIT_TRIG`` keeps JSON output
  byte-identical with the legacy emit (Worker F's golden fixture).
* When the flag is on, ``build_semantic_graph_with_dataset`` returns an
  ``rdflib.Dataset`` containing exactly nine named graphs — one per
  inference rule — including any rule that produced zero edges. This is
  the Wave 82 self-detection mechanism: a rule that silently regresses
  to zero output now leaves an audit trail in TriG that SPARQL can
  query.
* Per-graph metadata (rule, rule_version, generated_at, run_id,
  edge_count, input_chunk_count) lands in the dataset's *default*
  graph with the named-graph IRI as subject. Reuses Worker A's
  predicate vocabulary (``ed4all:rule``, ``ed4all:ruleVersionApplied``,
  ``dcterms:created``, ``prov:wasGeneratedBy``) per the JSON-LD
  context already in tree.
* IRI scheme: ``https://ed4all.io/run/<run_id>/rule/<rule_name>`` is
  re-run-stable; distinct ``run_id`` values produce disjoint graph
  IRIs, supporting the cross-run diff use case.
* Edge predicates inside named graphs round-trip through
  ``lib.ontology.edge_predicates.SLUG_TO_IRI`` (registry reuse —
  no fork).
* TriG serialization is lossless under round-trip parse.

Sub-plan: ``plans/phase-3-named-graph-provenance.md``.

Citations: Q3 (q_20260426_205702_83cd5b5d), Q49
(q_20260426_205724_4b21cb83), Q5 (q_20260426_205702_6d4302e5),
fresh-retrieve q_20260426_230212_b9be9116.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Project root resolution (mirrors test_concept_graph_jsonld_roundtrip.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = (
    PROJECT_ROOT / "Trainforge" / "tests" / "fixtures" / "mini_course_typed_graph"
)
RDF_SHACL_FIXTURE = (
    PROJECT_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-551-2"
    / "graph"
    / "concept_graph_semantic.json"
)

# Optional dep: rdflib. Skip the whole module if unavailable so the
# legacy default-off path is still exercised by adjacent suites.
rdflib = pytest.importorskip("rdflib")

from Trainforge.rag import named_graph_writer  # noqa: E402
from Trainforge.rag.typed_edge_inference import (  # noqa: E402
    build_semantic_graph,
    build_semantic_graph_with_dataset,
)

FIXED_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
FIXED_RUN = "test"

# The 9 inference rules we expect named graphs for, regardless of fixture.
EXPECTED_RULES = {
    "is_a_from_key_terms",
    "prerequisite_from_lo_order",
    "related_from_cooccurrence",
    "assesses_from_question_lo",
    "defined_by_from_first_mention",
    "derived_from_lo_ref",
    "exemplifies_from_example_chunks",
    "misconception_of_from_misconception_ref",
    "targets_concept_from_lo",
}

ED4ALL_NS = named_graph_writer.ED4ALL_NS
PROV_NS = named_graph_writer.PROV_NS
DCTERMS_NS = named_graph_writer.DCTERMS_NS


def _load_mini_fixture():
    with open(FIXTURE_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    with open(FIXTURE_DIR / "course.json", encoding="utf-8") as f:
        course = json.load(f)
    with open(FIXTURE_DIR / "concept_graph.json", encoding="utf-8") as f:
        concept_graph = json.load(f)
    with open(FIXTURE_DIR / "expected_semantic_graph.json", encoding="utf-8") as f:
        expected = json.load(f)
    return chunks, course, concept_graph, expected


# ---------------------------------------------------------------------------
# 1. Flag off → JSON identical to legacy emit, dataset is None.
# ---------------------------------------------------------------------------

def test_flag_off_emits_no_trig(monkeypatch):
    """With ``TRAINFORGE_EMIT_TRIG`` off the dataset must be ``None`` and
    the JSON dict must match what ``build_semantic_graph`` returns for
    the same inputs (byte-equal under ``json.dumps`` with sorted keys).
    """
    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", False)
    chunks, course, concept_graph, _ = _load_mini_fixture()

    legacy = build_semantic_graph(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )
    new_json, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )

    assert dataset is None, "EMIT_TRIG=False must skip dataset construction"
    assert json.dumps(legacy, sort_keys=True) == json.dumps(new_json, sort_keys=True)


def test_flag_off_existing_emit_pipeline_unchanged(monkeypatch):
    """Smoke: flag explicitly false. Worker F's golden tuples still match.

    Pins the regression that the Phase 3 refactor did not perturb the
    JSON precedence/ordering for any consumer.
    """
    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", False)
    chunks, course, concept_graph, expected = _load_mini_fixture()

    artifact = build_semantic_graph(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )
    actual_tuples = [[e["type"], e["source"], e["target"]] for e in artifact["edges"]]
    expected_tuples = [list(t) for t in expected["expected_edge_tuples"]]
    assert actual_tuples == expected_tuples


# ---------------------------------------------------------------------------
# 2. Flag on → 9 named graphs with metadata.
# ---------------------------------------------------------------------------

def test_flag_on_emits_nine_named_graphs(monkeypatch):
    """Each inference rule must register its own named graph in the
    dataset, even when it produced zero edges. Wave 82 self-detection
    requires the graph to exist as a queryable entity.
    """
    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph, _ = _load_mini_fixture()

    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )
    assert dataset is not None

    # rdflib's Dataset.contexts() yields all named graphs *plus* the
    # default context. Filter to the rule-graph IRI prefix.
    rule_iris = [
        str(ctx.identifier)
        for ctx in dataset.contexts()
        if str(ctx.identifier).startswith(named_graph_writer.RULE_GRAPH_BASE)
    ]
    assert len(rule_iris) == len(EXPECTED_RULES), (
        f"Expected {len(EXPECTED_RULES)} named graphs (one per rule); "
        f"got {len(rule_iris)}: {rule_iris}"
    )

    rule_names_seen = {iri.rsplit("/rule/", 1)[1] for iri in rule_iris}
    assert rule_names_seen == EXPECTED_RULES


def test_flag_on_metadata_predicates_attached(monkeypatch):
    """Each named-graph IRI must carry the 7 metadata predicates:
    ``rdf:type ed4all:RuleProvenanceGraph``, ``rdf:type prov:Bundle``,
    ``ed4all:rule``, ``ed4all:ruleVersionApplied``, ``dcterms:created``,
    ``prov:wasGeneratedBy``, ``ed4all:edgeCount``,
    ``ed4all:inputChunkCount``. Metadata lives in the default graph.
    """
    from rdflib import URIRef

    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph, _ = _load_mini_fixture()

    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )

    default_graph = dataset.default_context
    rule_iris = [
        ctx.identifier
        for ctx in dataset.contexts()
        if str(ctx.identifier).startswith(named_graph_writer.RULE_GRAPH_BASE)
    ]
    assert rule_iris, "no rule named graphs registered"

    required = [
        URIRef(ED4ALL_NS + "rule"),
        URIRef(ED4ALL_NS + "ruleVersionApplied"),
        URIRef(DCTERMS_NS + "created"),
        URIRef(PROV_NS + "wasGeneratedBy"),
        URIRef(ED4ALL_NS + "edgeCount"),
        URIRef(ED4ALL_NS + "inputChunkCount"),
    ]
    for graph_iri in rule_iris:
        for pred in required:
            objs = list(default_graph.objects(graph_iri, pred))
            assert objs, (
                f"Graph {graph_iri} missing metadata predicate {pred}; "
                f"every named-graph IRI must carry the full provenance set."
            )


# ---------------------------------------------------------------------------
# 3. IRI scheme stability across runs.
# ---------------------------------------------------------------------------

def test_flag_on_iri_scheme_stable(monkeypatch):
    """Two runs with the same ``run_id`` produce identical graph IRIs;
    distinct ``run_id`` produces disjoint sets — the diff surface for
    cross-run regression detection.
    """
    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph, _ = _load_mini_fixture()

    _, ds_a1 = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id="run-A"
    )
    _, ds_a2 = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id="run-A"
    )
    _, ds_b = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id="run-B"
    )

    def rule_iris(ds):
        return {
            str(ctx.identifier)
            for ctx in ds.contexts()
            if str(ctx.identifier).startswith(named_graph_writer.RULE_GRAPH_BASE)
        }

    iris_a1 = rule_iris(ds_a1)
    iris_a2 = rule_iris(ds_a2)
    iris_b = rule_iris(ds_b)

    assert iris_a1 == iris_a2, "same run_id must produce identical graph IRI set"
    assert iris_a1.isdisjoint(iris_b), (
        "distinct run_id must produce disjoint graph IRI sets so cross-run "
        "SPARQL diffs are non-empty"
    )


# ---------------------------------------------------------------------------
# 4. Wave 82 zero-edge regression detection.
# ---------------------------------------------------------------------------

def _zero_edge_inputs():
    """Synthetic chunks/course/concept_graph where every rule produces
    zero edges. The base co-occurrence concept graph carries one node
    pair so the orchestrator runs cleanly, but no chunk has the
    metadata required for any rule to fire (no key_terms, no LO
    pointers, no examples, no misconceptions, no questions).

    Used to assert that *every* rule contributes a named graph even
    when nothing was inferred — the Wave 82 audit trail.
    """
    chunks = [
        {
            "id": "chunk-x",
            "concept_tags": ["alpha"],
            "learning_outcome_refs": [],
            "text": "Plain text content with no key-term definition.",
        }
    ]
    course = {"learning_outcomes": []}
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {"id": "alpha", "label": "alpha", "frequency": 1},
            {"id": "beta", "label": "beta", "frequency": 1},
        ],
        "edges": [],
    }
    return chunks, course, concept_graph


def test_flag_on_zero_edge_rule_emits_empty_named_graph(monkeypatch):
    """Even when a rule produces zero edges, its named graph must be
    registered in the dataset with ``ed4all:edgeCount 0``.

    This is the structural payoff of the Phase 3 refactor: the Wave 82
    bug (silent zero-edge regression with unchanged rule_version)
    becomes a queryable diff in TriG.
    """
    from rdflib import Literal, URIRef, XSD

    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph = _zero_edge_inputs()

    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )
    assert dataset is not None

    default_graph = dataset.default_context
    edge_count_pred = URIRef(ED4ALL_NS + "edgeCount")

    rule_iris = [
        ctx.identifier
        for ctx in dataset.contexts()
        if str(ctx.identifier).startswith(named_graph_writer.RULE_GRAPH_BASE)
    ]
    assert len(rule_iris) == len(EXPECTED_RULES)

    zero = Literal(0, datatype=XSD.integer)
    zero_count_iris = [
        iri for iri in rule_iris
        if (iri, edge_count_pred, zero) in default_graph
    ]
    # On the synthetic zero-edge fixture, *every* rule should have
    # edgeCount 0 — that's by construction.
    assert len(zero_count_iris) == len(EXPECTED_RULES), (
        f"Expected all {len(EXPECTED_RULES)} rules to register edgeCount=0; "
        f"got {len(zero_count_iris)} matches: {zero_count_iris}"
    )


def test_flag_on_sparql_zero_edge_query_finds_regressions(monkeypatch):
    """A SPARQL SELECT query against the dataset must return the
    zero-edge rule graphs. This is the consumer-side surface that
    replaces the per-validator Python check in Wave 82's Phase A3.
    """
    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph = _zero_edge_inputs()

    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )

    query = """
    PREFIX ed4all: <https://ed4all.io/vocab/>
    SELECT ?g ?rule ?ver WHERE {
      ?g ed4all:edgeCount 0 ;
         ed4all:rule ?rule ;
         ed4all:ruleVersionApplied ?ver .
    }
    """
    # rdflib SPARQL over Dataset queries the default + named graphs;
    # metadata lives in the default graph so the query resolves there.
    results = list(dataset.query(query))
    rule_names = {str(row[1]) for row in results}
    assert rule_names == EXPECTED_RULES, (
        f"SPARQL zero-edge query missed rules; got {rule_names}, "
        f"expected {EXPECTED_RULES}"
    )


# ---------------------------------------------------------------------------
# 5. TriG serialization round-trip.
# ---------------------------------------------------------------------------

def test_flag_on_trig_serialization_roundtrip(monkeypatch):
    """Serialize the dataset to TriG and re-parse; the quad count
    delta must stay within blank-node skolem cushion.

    Mirrors the cushion convention from
    ``test_concept_graph_jsonld_roundtrip.py::test_turtle_roundtrip_is_lossless``.
    """
    from rdflib import Dataset

    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph, _ = _load_mini_fixture()
    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )

    trig = named_graph_writer.serialize_trig(dataset)
    assert trig.strip(), "TriG serialization produced empty output"

    n_orig = sum(1 for _ in dataset.quads())
    rt = Dataset(default_union=False)
    rt.parse(data=trig, format="trig")
    n_rt = sum(1 for _ in rt.quads())

    delta = n_rt - n_orig
    assert abs(delta) <= 5, (
        f"TriG round-trip changed quad count: {n_orig} -> {n_rt} "
        f"(delta={delta})"
    )


# ---------------------------------------------------------------------------
# 6. Edge predicate IRIs come from SLUG_TO_IRI registry (no fork).
# ---------------------------------------------------------------------------

def test_flag_on_edge_predicates_resolve_via_slug_registry(monkeypatch):
    """Every bare-asserted-triple predicate inside a named graph must be
    one of the IRIs registered in
    ``lib.ontology.edge_predicates.SLUG_TO_IRI``. Pins reuse of the
    canonical registry (no fork).
    """
    from lib.ontology.edge_predicates import SLUG_TO_IRI

    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)
    chunks, course, concept_graph, _ = _load_mini_fixture()
    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id=FIXED_RUN
    )

    registered_iris = set(SLUG_TO_IRI.values())
    # Predicates we use for *reified* edge structure (not slug-derived).
    reified_struct_iris = {
        ED4ALL_NS + "edgeSource",
        ED4ALL_NS + "edgeTarget",
        ED4ALL_NS + "edgeType",
        ED4ALL_NS + "confidence",
        ED4ALL_NS + "hasProvenance",
        ED4ALL_NS + "rule",
        ED4ALL_NS + "ruleVersionApplied",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    }

    for ctx in dataset.contexts():
        if not str(ctx.identifier).startswith(named_graph_writer.RULE_GRAPH_BASE):
            continue
        for s, p, o in ctx:
            p_str = str(p)
            if p_str in reified_struct_iris:
                continue
            # Anything else must be a registered slug IRI.
            assert p_str in registered_iris, (
                f"Predicate {p_str} in named graph {ctx.identifier} is "
                f"neither a reified-structure predicate nor a registered "
                f"SLUG_TO_IRI value; ad-hoc predicates are forbidden."
            )


# ---------------------------------------------------------------------------
# 7. mint_rule_graph_iri determinism — pure-function unit test.
# ---------------------------------------------------------------------------

def test_mint_rule_graph_iri_is_deterministic():
    """The IRI minter must be a pure function of (run_id, rule_name).

    Tests the ``RULE_GRAPH_BASE/<run_id>/rule/<rule_name>`` shape
    documented in the sub-plan § 2.
    """
    iri = named_graph_writer.mint_rule_graph_iri(
        "run-X", "is_a_from_key_terms", "2026-01-01T00:00:00+00:00"
    )
    assert iri == "https://ed4all.io/run/run-X/rule/is_a_from_key_terms"

    # Special chars in run_id are slugified deterministically.
    iri2 = named_graph_writer.mint_rule_graph_iri(
        "WF/2026.04.26@abc", "rule_x", "2026-01-01T00:00:00+00:00"
    )
    assert "/run/" in iri2 and "/rule/rule_x" in iri2
    assert " " not in iri2

    # No run_id → deterministic local- prefix derived from timestamp.
    iri3a = named_graph_writer.mint_rule_graph_iri(
        None, "rule_x", "2026-01-01T00:00:00+00:00"
    )
    iri3b = named_graph_writer.mint_rule_graph_iri(
        None, "rule_x", "2026-01-01T00:00:00+00:00"
    )
    assert iri3a == iri3b
    assert "/run/local-" in iri3a


# ---------------------------------------------------------------------------
# 8. RDF-SHACL-551-2 corpus smoke (skipped if fixture missing).
# ---------------------------------------------------------------------------

def test_flag_on_rdf_shacl_551_2_corpus_smoke(monkeypatch):
    """When the rdf-shacl-551-2 graph fixture is present, derive a
    minimal chunks/concept_graph stand-in from its node set and verify
    the dataset still composes with 9 named graphs.

    Acts as the parent-plan-mandated metric on the production fixture.
    The rule emits will largely be zero on this synthetic-input form
    (we don't replay the full chunks pipeline) — but the named-graph
    structural contract still holds.
    """
    if not RDF_SHACL_FIXTURE.exists():
        pytest.skip(f"rdf-shacl-551-2 fixture not present at {RDF_SHACL_FIXTURE}")

    monkeypatch.setattr(named_graph_writer, "EMIT_TRIG", True)

    with RDF_SHACL_FIXTURE.open() as f:
        artifact = json.load(f)
    # Re-shape the artifact's nodes into a minimal co-occurrence graph
    # suitable for re-running the rule pipeline. Edges are dropped —
    # the rule pipeline rebuilds typed edges from chunks, which we
    # don't have in this fixture form.
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {"id": n["id"], "label": n.get("label", n["id"]),
             "frequency": n.get("frequency", 0)}
            for n in artifact.get("nodes", [])
        ],
        "edges": [],
    }
    chunks = []  # No chunks → most rules produce zero edges; that's fine.
    course = {"learning_outcomes": []}

    _, dataset = build_semantic_graph_with_dataset(
        chunks, course, concept_graph, now=FIXED_NOW, run_id="rdf-shacl-551-2"
    )
    assert dataset is not None

    rule_iris = [
        ctx.identifier
        for ctx in dataset.contexts()
        if str(ctx.identifier).startswith(named_graph_writer.RULE_GRAPH_BASE)
    ]
    assert len(rule_iris) == len(EXPECTED_RULES), (
        f"On rdf-shacl-551-2 fixture, expected {len(EXPECTED_RULES)} "
        f"named graphs; got {len(rule_iris)}"
    )

    # Exercise the TriG serializer end-to-end on a corpus-scale node set
    # (the fixture has ~672 nodes per Phase 1 evidence).
    trig = named_graph_writer.serialize_trig(dataset)
    assert "@prefix ed4all:" in trig
    # All 9 rule graph IRIs must appear in the TriG output as named-graph
    # block headers.
    for iri in rule_iris:
        assert str(iri) in trig
