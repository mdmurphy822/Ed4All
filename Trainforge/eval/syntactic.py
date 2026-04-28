"""Wave 92 — Tier 1 machine-verifiable correctness for RDF/SHACL outputs.

Drives binary ground-truth checks via rdflib + pyshacl + the rdflib
SPARQL parser. Every helper returns a structured ``dict`` (not a bool)
so the harness can show *why* a generation failed, not just *that* it
did.

The module imports rdflib + pyshacl lazily so a CPU-only dev box
without those installed can still ``import Trainforge.eval`` (the
generic-corpus profile in ``configs/generic.yaml`` skips Tier 1
entirely).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Lazy rdflib import                                                      #
# ---------------------------------------------------------------------- #


def _try_import_rdflib():
    try:
        import rdflib  # type: ignore
        return rdflib
    except ImportError:
        return None


def _try_import_pyshacl():
    try:
        import pyshacl  # type: ignore
        return pyshacl
    except ImportError:
        return None


# ---------------------------------------------------------------------- #
# Turtle parsing                                                          #
# ---------------------------------------------------------------------- #


def evaluate_turtle(generated: str) -> Dict[str, Any]:
    """Does ``generated`` parse as Turtle? How many triples?

    Returns:
        ``{"parses": bool, "triple_count": int, "errors": List[str]}``
    """
    rdflib = _try_import_rdflib()
    if rdflib is None:
        return {
            "parses": False,
            "triple_count": 0,
            "errors": ["rdflib not installed; cannot evaluate Turtle"],
        }

    g = rdflib.Graph()
    try:
        g.parse(data=generated, format="turtle")
    except Exception as exc:  # noqa: BLE001 — rdflib raises many subtypes
        return {
            "parses": False,
            "triple_count": 0,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    return {
        "parses": True,
        "triple_count": len(g),
        "errors": [],
    }


# ---------------------------------------------------------------------- #
# SPARQL syntax check                                                     #
# ---------------------------------------------------------------------- #


def evaluate_sparql(generated: str) -> Dict[str, Any]:
    """Does ``generated`` parse as a SPARQL query?

    Uses ``rdflib.plugins.sparql.parser.parseQuery`` for query syntax,
    falling back to ``parseUpdate`` for SPARQL Update operations.

    Returns:
        ``{"parses": bool, "syntax_errors": List[str], "kind": Optional[str]}``
    """
    rdflib = _try_import_rdflib()
    if rdflib is None:
        return {
            "parses": False,
            "syntax_errors": ["rdflib not installed; cannot evaluate SPARQL"],
            "kind": None,
        }

    try:
        from rdflib.plugins.sparql.parser import parseQuery, parseUpdate  # type: ignore
    except ImportError as exc:
        return {
            "parses": False,
            "syntax_errors": [f"rdflib SPARQL parser not available: {exc}"],
            "kind": None,
        }

    query_err: Optional[str] = None
    try:
        parseQuery(generated)
        return {"parses": True, "syntax_errors": [], "kind": "query"}
    except Exception as exc:  # noqa: BLE001
        query_err = f"{type(exc).__name__}: {exc}"

    try:
        parseUpdate(generated)
        return {"parses": True, "syntax_errors": [], "kind": "update"}
    except Exception as exc:  # noqa: BLE001
        return {
            "parses": False,
            "syntax_errors": [
                f"as query: {query_err}",
                f"as update: {type(exc).__name__}: {exc}",
            ],
            "kind": None,
        }


# ---------------------------------------------------------------------- #
# SHACL shape syntax                                                      #
# ---------------------------------------------------------------------- #


def evaluate_shacl_shape(generated: str) -> Dict[str, Any]:
    """Does ``generated`` parse as a SHACL shapes graph?

    Currently we only check that the Turtle parses AND that at least
    one ``sh:NodeShape`` or ``sh:PropertyShape`` declaration is
    present. Stronger semantic checks (every shape has a target,
    constraints are well-formed) require pyshacl which is exercised
    by ``evaluate_shacl_validation``.

    Returns:
        ``{"parses": bool, "is_shacl": bool, "shape_count": int,
        "errors": List[str]}``
    """
    parse_result = evaluate_turtle(generated)
    if not parse_result["parses"]:
        return {
            "parses": False,
            "is_shacl": False,
            "shape_count": 0,
            "errors": parse_result["errors"],
        }

    rdflib = _try_import_rdflib()
    if rdflib is None:
        # parses=True is impossible without rdflib but we guard
        # defensively.
        return {
            "parses": False,
            "is_shacl": False,
            "shape_count": 0,
            "errors": ["rdflib not installed; cannot inspect SHACL shapes"],
        }

    g = rdflib.Graph()
    try:
        g.parse(data=generated, format="turtle")
    except Exception as exc:  # noqa: BLE001
        return {
            "parses": False,
            "is_shacl": False,
            "shape_count": 0,
            "errors": [f"reparse failed: {type(exc).__name__}: {exc}"],
        }

    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    RDF = rdflib.RDF
    shape_count = 0
    for shape_class in (SH.NodeShape, SH.PropertyShape):
        shape_count += sum(1 for _ in g.subjects(RDF.type, shape_class))
    return {
        "parses": True,
        "is_shacl": shape_count > 0,
        "shape_count": shape_count,
        "errors": [] if shape_count > 0 else [
            "no sh:NodeShape or sh:PropertyShape declarations found"
        ],
    }


# ---------------------------------------------------------------------- #
# SHACL validation                                                        #
# ---------------------------------------------------------------------- #


def evaluate_shacl_validation(
    data_graph: str,
    shapes_graph: str,
    claimed_violations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run pyshacl over ``data_graph`` against ``shapes_graph``.

    When ``claimed_violations`` is provided, the eval compares the
    actual pyshacl violations to the model's claims by counting
    matches on focus node + source shape. The returned dict carries
    both the raw pyshacl outcome (``conforms``, ``violation_count``)
    and the comparison summary (``claim_precision``, ``claim_recall``).

    Args:
        data_graph: Turtle text for the instance graph.
        shapes_graph: Turtle text for the shapes graph.
        claimed_violations: Optional list of
            ``{"focus_node": "...", "source_shape": "..."}`` dicts.
    """
    pyshacl = _try_import_pyshacl()
    if pyshacl is None:
        return {
            "conforms": None,
            "violation_count": 0,
            "errors": ["pyshacl not installed; cannot validate"],
        }

    try:
        conforms, results_graph, results_text = pyshacl.validate(
            data_graph,
            shacl_graph=shapes_graph,
            data_graph_format="turtle",
            shacl_graph_format="turtle",
            inference="none",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "conforms": None,
            "violation_count": 0,
            "errors": [f"pyshacl exception: {type(exc).__name__}: {exc}"],
        }

    rdflib = _try_import_rdflib()
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#") if rdflib else None
    actual_violations: List[Dict[str, str]] = []
    if rdflib is not None and SH is not None:
        for vr in results_graph.subjects(rdflib.RDF.type, SH.ValidationResult):
            focus = next(results_graph.objects(vr, SH.focusNode), None)
            source = next(results_graph.objects(vr, SH.sourceShape), None)
            actual_violations.append({
                "focus_node": str(focus) if focus is not None else "",
                "source_shape": str(source) if source is not None else "",
            })

    summary: Dict[str, Any] = {
        "conforms": bool(conforms),
        "violation_count": len(actual_violations),
        "actual_violations": actual_violations,
        "results_text": results_text,
        "errors": [],
    }

    if claimed_violations is not None:
        actual_keys = {
            (v["focus_node"], v["source_shape"]) for v in actual_violations
        }
        claimed_keys = {
            (c.get("focus_node", ""), c.get("source_shape", ""))
            for c in claimed_violations
        }
        tp = len(actual_keys & claimed_keys)
        fp = len(claimed_keys - actual_keys)
        fn = len(actual_keys - claimed_keys)
        summary["claim_precision"] = (
            tp / (tp + fp) if (tp + fp) > 0 else 1.0
        )
        summary["claim_recall"] = (
            tp / (tp + fn) if (tp + fn) > 0 else 1.0
        )
        summary["claim_true_positives"] = tp
        summary["claim_false_positives"] = fp
        summary["claim_false_negatives"] = fn
    return summary


# ---------------------------------------------------------------------- #
# RDFS / OWL entailment                                                   #
# ---------------------------------------------------------------------- #


def evaluate_owl_entailment(
    triples: str,
    expected_entailments: List[str],
) -> Dict[str, Any]:
    """Check whether a ground-truth set of entailments holds.

    Uses rdflib's RDFS reasoner via the ``RDFS`` semantics (or owlrl
    if installed for richer OWL2 RL coverage). Each expected entailment
    is itself a Turtle snippet of one or more triples; we check that
    each triple in the snippet is present in the entailed graph.

    Args:
        triples: Turtle for the source graph.
        expected_entailments: List of Turtle snippets that must be
            entailed by the source.
    """
    rdflib = _try_import_rdflib()
    if rdflib is None:
        return {
            "entailed": False,
            "matches": [],
            "errors": ["rdflib not installed; cannot run entailment"],
        }

    try:
        g = rdflib.Graph()
        g.parse(data=triples, format="turtle")
    except Exception as exc:  # noqa: BLE001
        return {
            "entailed": False,
            "matches": [],
            "errors": [f"source graph parse failed: {type(exc).__name__}: {exc}"],
        }

    used_owlrl = False
    try:
        import owlrl  # type: ignore
        owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(g)
        used_owlrl = True
    except ImportError:
        # Fall back to rdflib's built-in RDFS support: replay each
        # expected entailment as a SPARQL ASK over the source graph.
        # Without a full reasoner we can't materialize the closure,
        # so we restrict this code path to entailments that are
        # already explicitly present.
        pass

    matches: List[Dict[str, Any]] = []
    all_entailed = True
    for snippet in expected_entailments:
        check = rdflib.Graph()
        try:
            check.parse(data=snippet, format="turtle")
        except Exception as exc:  # noqa: BLE001
            matches.append({"snippet": snippet, "entailed": False, "error": str(exc)})
            all_entailed = False
            continue

        snippet_entailed = all(t in g for t in check)
        matches.append({"snippet": snippet, "entailed": snippet_entailed})
        if not snippet_entailed:
            all_entailed = False

    return {
        "entailed": all_entailed,
        "matches": matches,
        "used_reasoner": "owlrl" if used_owlrl else "none",
        "errors": [],
    }


# ---------------------------------------------------------------------- #
# Strict-mode predicate-usage check (Wave 108 / Phase B)                  #
# ---------------------------------------------------------------------- #


_DEFAULT_PREDICATE_NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "sh": "http://www.w3.org/ns/shacl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dct": "http://purl.org/dc/terms/",
}


def _resolve_predicate(token: str, namespaces: Dict[str, str]) -> Optional[str]:
    """Resolve a predicate token (CURIE ``prefix:local`` or full
    ``<http://...>`` URI) to its full URI string. Returns None on
    unparsable input or unknown prefix."""
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if ":" in token:
        prefix, local = token.split(":", 1)
        ns = namespaces.get(prefix)
        if ns is None:
            return None
        return f"{ns}{local}"
    return None


def evaluate_predicate_usage(
    generated: str,
    *,
    required_predicates: List[str],
    extra_namespaces: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Strict-mode predicate-usage check (Wave 108 / Phase B).

    Parses ``generated`` as Turtle, then asserts that EACH entry in
    ``required_predicates`` appears as a predicate in the graph. Each
    entry may be either a CURIE (``sh:datatype``) or a full URI in
    angle brackets (``<http://www.w3.org/ns/shacl#datatype>``).

    The match is by full URI only — a synonym predicate (e.g. the
    model emitted ``sh:class`` instead of ``sh:datatype``) is NOT
    credited.

    Returns:
        ``{"uses_all": bool, "used": List[str], "missing": List[str],
        "errors": List[str]}``
    """
    rdflib = _try_import_rdflib()
    if rdflib is None:
        return {
            "uses_all": False,
            "used": [],
            "missing": list(required_predicates),
            "errors": ["rdflib not installed; cannot evaluate predicate usage"],
        }

    namespaces = dict(_DEFAULT_PREDICATE_NAMESPACES)
    if extra_namespaces:
        namespaces.update(extra_namespaces)

    g = rdflib.Graph()
    try:
        g.parse(data=generated, format="turtle")
    except Exception as exc:  # noqa: BLE001
        return {
            "uses_all": False,
            "used": [],
            "missing": list(required_predicates),
            "errors": [f"parse failed: {type(exc).__name__}: {exc}"],
        }

    used_uris = {str(p) for _s, p, _o in g}
    # Merge graph-declared prefixes (override defaults).
    for prefix, ns in g.namespaces():
        namespaces[prefix] = str(ns)

    used: List[str] = []
    missing: List[str] = []
    for token in required_predicates:
        resolved = _resolve_predicate(token, namespaces)
        if resolved is None:
            missing.append(token)
            continue
        if resolved in used_uris:
            used.append(token)
        else:
            missing.append(token)

    return {
        "uses_all": not missing,
        "used": used,
        "missing": missing,
        "errors": [],
    }


__all__ = [
    "evaluate_turtle",
    "evaluate_sparql",
    "evaluate_shacl_shape",
    "evaluate_shacl_validation",
    "evaluate_owl_entailment",
    "evaluate_predicate_usage",
]
