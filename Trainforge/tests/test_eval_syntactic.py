"""Wave 92 — Tier 1 (syntactic) evaluator tests.

Synthetic Turtle / SPARQL / SHACL inputs at edge cases. rdflib +
pyshacl are project dependencies (see pyproject.toml core deps), so
these tests run without skip on the default install.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.syntactic import (  # noqa: E402
    evaluate_owl_entailment,
    evaluate_shacl_shape,
    evaluate_shacl_validation,
    evaluate_sparql,
    evaluate_turtle,
)


# ---------------------------------------------------------------------- #
# Turtle                                                                  #
# ---------------------------------------------------------------------- #


def test_turtle_valid_parses():
    src = """
    @prefix : <http://example.com/> .
    :alice :knows :bob .
    :bob :knows :charlie .
    """
    out = evaluate_turtle(src)
    assert out["parses"] is True
    assert out["triple_count"] == 2
    assert out["errors"] == []


def test_turtle_invalid_fails():
    src = "this is not turtle at all {{{"
    out = evaluate_turtle(src)
    assert out["parses"] is False
    assert out["triple_count"] == 0
    assert out["errors"]


def test_turtle_empty_parses_zero_triples():
    src = "@prefix : <http://example.com/> .\n"
    out = evaluate_turtle(src)
    assert out["parses"] is True
    assert out["triple_count"] == 0


# ---------------------------------------------------------------------- #
# SPARQL                                                                  #
# ---------------------------------------------------------------------- #


def test_sparql_select_query_parses():
    q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
    out = evaluate_sparql(q)
    assert out["parses"] is True
    assert out["kind"] == "query"


def test_sparql_construct_query_parses():
    q = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
    out = evaluate_sparql(q)
    assert out["parses"] is True


def test_sparql_invalid_fails():
    q = "SELECT WHERE { ??? }"
    out = evaluate_sparql(q)
    assert out["parses"] is False
    assert out["syntax_errors"]


def test_sparql_update_parses():
    """SPARQL Update operations parse on the parseUpdate fallback path."""
    q = "INSERT DATA { <http://example.com/a> <http://example.com/b> <http://example.com/c> }"
    out = evaluate_sparql(q)
    # Either "query" or "update" is acceptable; both indicate it parses.
    assert out["parses"] is True


# ---------------------------------------------------------------------- #
# SHACL shape                                                             #
# ---------------------------------------------------------------------- #


def test_shacl_shape_with_node_shape_recognised():
    src = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix : <http://example.com/> .
    :PersonShape a sh:NodeShape ;
        sh:targetClass :Person ;
        sh:property [ sh:path :name ; sh:datatype <http://www.w3.org/2001/XMLSchema#string> ] .
    """
    out = evaluate_shacl_shape(src)
    assert out["parses"] is True
    assert out["is_shacl"] is True
    assert out["shape_count"] >= 1


def test_shacl_shape_turtle_without_shape_fails():
    src = """
    @prefix : <http://example.com/> .
    :alice :knows :bob .
    """
    out = evaluate_shacl_shape(src)
    assert out["parses"] is True
    assert out["is_shacl"] is False
    assert out["shape_count"] == 0


def test_shacl_shape_invalid_turtle_fails():
    out = evaluate_shacl_shape("garbled )))")
    assert out["parses"] is False
    assert out["is_shacl"] is False


# ---------------------------------------------------------------------- #
# SHACL validation                                                        #
# ---------------------------------------------------------------------- #


def _data_graph_violating() -> str:
    return """
    @prefix : <http://example.com/> .
    :alice a :Person .
    """


def _data_graph_conforming() -> str:
    return """
    @prefix : <http://example.com/> .
    :alice a :Person ;
        :name "Alice" .
    """


def _shapes_graph() -> str:
    return """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix : <http://example.com/> .
    :PersonShape a sh:NodeShape ;
        sh:targetClass :Person ;
        sh:property [
            sh:path :name ;
            sh:minCount 1 ;
            sh:datatype xsd:string ;
        ] .
    """


def test_shacl_validation_conforms():
    out = evaluate_shacl_validation(_data_graph_conforming(), _shapes_graph())
    if out["conforms"] is None:
        pytest.skip("pyshacl not available in this env")
    assert out["conforms"] is True
    assert out["violation_count"] == 0


def test_shacl_validation_detects_violation():
    out = evaluate_shacl_validation(_data_graph_violating(), _shapes_graph())
    if out["conforms"] is None:
        pytest.skip("pyshacl not available in this env")
    assert out["conforms"] is False
    assert out["violation_count"] >= 1


def test_shacl_validation_claim_precision_recall():
    """When claimed_violations match actual, precision+recall = 1.0."""
    out = evaluate_shacl_validation(
        _data_graph_violating(),
        _shapes_graph(),
        claimed_violations=[],
    )
    if out["conforms"] is None:
        pytest.skip("pyshacl not available in this env")
    # No claims, but >0 actual → recall = 0.0, precision = 1.0 (no FPs).
    assert out["claim_precision"] == 1.0
    assert out["claim_recall"] == 0.0


# ---------------------------------------------------------------------- #
# OWL entailment                                                          #
# ---------------------------------------------------------------------- #


def test_owl_entailment_explicit_triple_present():
    src = """
    @prefix : <http://example.com/> .
    :alice :knows :bob .
    """
    expected = ["@prefix : <http://example.com/> . :alice :knows :bob ."]
    out = evaluate_owl_entailment(src, expected)
    assert out["entailed"] is True
    assert out["matches"][0]["entailed"] is True


def test_owl_entailment_missing_triple_fails():
    src = """
    @prefix : <http://example.com/> .
    :alice :knows :bob .
    """
    expected = ["@prefix : <http://example.com/> . :alice :knows :charlie ."]
    out = evaluate_owl_entailment(src, expected)
    assert out["entailed"] is False
    assert out["matches"][0]["entailed"] is False


def test_owl_entailment_invalid_source_returns_error():
    out = evaluate_owl_entailment("garbled", ["@prefix : <a:> . :a :b :c ."])
    assert out["entailed"] is False
    assert out["errors"]


def test_evaluate_predicate_usage_accepts_curie_and_uri() -> None:
    """A graph that uses sh:datatype should match required={'sh:datatype'}
    AND required={'<http://www.w3.org/ns/shacl#datatype>'}."""
    pytest.importorskip("rdflib")
    from Trainforge.eval.syntactic import evaluate_predicate_usage
    g = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    ex:NodeShape1 sh:datatype <http://www.w3.org/2001/XMLSchema#string> .
    """
    r1 = evaluate_predicate_usage(g, required_predicates=["sh:datatype"])
    assert r1["uses_all"] is True
    assert "sh:datatype" in r1["used"]
    assert not r1["missing"]

    r2 = evaluate_predicate_usage(
        g, required_predicates=["<http://www.w3.org/ns/shacl#datatype>"]
    )
    assert r2["uses_all"] is True


def test_evaluate_predicate_usage_flags_missing_predicate() -> None:
    """The strict-mode regression class: model used sh:type instead of
    sh:datatype. Must surface as missing."""
    pytest.importorskip("rdflib")
    from Trainforge.eval.syntactic import evaluate_predicate_usage
    g = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    ex:NodeShape1 sh:class <http://example.org/Person> .
    """
    r = evaluate_predicate_usage(g, required_predicates=["sh:datatype"])
    assert r["uses_all"] is False
    assert r["missing"] == ["sh:datatype"]
