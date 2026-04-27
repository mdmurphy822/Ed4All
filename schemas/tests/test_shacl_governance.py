"""Phase 7.2 + 7.4 of plans/rdf-shacl-enrichment-2026-04-26.md.

Two governance gates over every SHACL shape file in the repo:

- **7.2 — Meta-validation:** every shape file parses cleanly; every
  ``sh:NodeShape`` has a target (one of ``sh:targetClass``,
  ``sh:targetNode``, ``sh:targetSubjectsOf``, ``sh:targetObjectsOf``)
  OR an implicit class target; every ``sh:property`` block has an
  ``sh:path``.
- **7.4 — Severity matrix:** every shape that emits ``sh:Violation``
  (the default severity when unset) declares an authored
  ``sh:message`` somewhere in its constraint graph — either at the
  shape level or on every property block. Per corpus Q43, generic
  default messages are not actionable; we want author-supplied
  diagnostics on every Violation surface.

These run as a single test module (paired by domain). Auto-discovers
shape files from ``schemas/context/**.shacl*.ttl`` and
``lib/validators/shacl/**.ttl`` so the gate scales with the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple

import pytest

rdflib = pytest.importorskip("rdflib")
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

SH = Namespace("http://www.w3.org/ns/shacl#")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _discover_shape_files() -> List[Path]:
    """All SHACL shape files we expect to govern."""
    files: List[Path] = []
    files += list((PROJECT_ROOT / "schemas" / "context").glob("*.shacl*.ttl"))
    files += list((PROJECT_ROOT / "lib" / "validators" / "shacl").rglob("*.ttl"))
    # Filter out the rules file from the validation-meta gate (rules
    # are derivation-only — sh:targetClass is optional on a rule
    # shape because it can be triggered via sh:condition + entailment).
    return sorted(p for p in files if p.is_file())


SHAPE_FILES = _discover_shape_files()


def pytest_generate_tests(metafunc):
    """Parametrize across discovered shape files for clear failure attribution."""
    if "shape_file" in metafunc.fixturenames:
        if SHAPE_FILES:
            metafunc.parametrize(
                "shape_file",
                SHAPE_FILES,
                ids=[p.relative_to(PROJECT_ROOT).as_posix() for p in SHAPE_FILES],
            )
        else:
            metafunc.parametrize("shape_file", [], ids=[])


# ---------------------------------------------------------------------------
# Discovery sanity
# ---------------------------------------------------------------------------


def test_at_least_one_shape_file_discovered():
    """If discovery returns zero, the rest of this module is silent —
    fail loudly instead so a reorg that drops every shape file gets caught."""
    assert SHAPE_FILES, (
        "No SHACL shape files discovered under schemas/context/ or "
        "lib/validators/shacl/. The governance gates only cover what "
        "they can see; verify the discovery globs."
    )


# ---------------------------------------------------------------------------
# Phase 7.2 — Meta-validation
# ---------------------------------------------------------------------------


def _is_rule_file(shape_file: Path) -> bool:
    """SHACL rules are derivation-only and don't need targets the same way."""
    return ".shacl-rules" in shape_file.name


def test_shape_file_parses_as_turtle(shape_file: Path):
    g = Graph()
    g.parse(shape_file, format="turtle")
    assert len(g) > 0, f"Empty graph parsed from {shape_file}"


def test_every_node_shape_has_a_target(shape_file: Path):
    """Per corpus Q17, every NodeShape needs a target predicate (or an
    implicit class target). Untargeted shapes are silently inert and
    waste authoring effort."""
    if _is_rule_file(shape_file):
        pytest.skip("rule files use sh:condition + entailment, not targets")

    g = Graph()
    g.parse(shape_file, format="turtle")

    target_predicates: Set[URIRef] = {
        SH.targetClass, SH.targetNode, SH.targetSubjectsOf,
        SH.targetObjectsOf, SH.target,
    }

    untargeted: List[str] = []
    for shape in g.subjects(RDF.type, SH.NodeShape):
        has_target = any(
            (shape, pred, None) in g for pred in target_predicates
        )
        # Implicit class target: shape is itself an rdfs:Class.
        is_class = (shape, RDF.type, RDFS.Class) in g
        if has_target or is_class:
            continue
        # Sub-shapes referenced by parent shapes (sh:property blocks
        # via blank nodes) are not top-level targets — only flag
        # shapes with a non-blank IRI.
        if isinstance(shape, URIRef):
            untargeted.append(str(shape))

    assert not untargeted, (
        f"NodeShapes in {shape_file.name} lack a target declaration: "
        f"{untargeted}. Add sh:targetClass / sh:targetNode / "
        "sh:targetSubjectsOf / sh:targetObjectsOf or declare the "
        "shape as rdfs:Class for an implicit class target (Q21)."
    )


def test_every_property_shape_has_a_path(shape_file: Path):
    """Every PropertyShape must declare sh:path — without one, the
    shape evaluates against nothing. This is also the behaviour the
    SHACL spec mandates."""
    g = Graph()
    g.parse(shape_file, format="turtle")

    pathless: List[str] = []
    for prop in g.subjects(RDF.type, SH.PropertyShape):
        has_path = (prop, SH.path, None) in g
        if not has_path:
            pathless.append(str(prop))

    # Inline property shapes referenced via sh:property: the parent
    # uses a blank node + sh:property, and the blank node carries
    # sh:path. Walk those too.
    for parent, _, prop in g.triples((None, SH.property, None)):
        has_path = (prop, SH.path, None) in g
        if not has_path:
            pathless.append(f"sh:property of {parent}")

    assert not pathless, (
        f"PropertyShapes in {shape_file.name} lack sh:path: {pathless}"
    )


# ---------------------------------------------------------------------------
# Phase 7.4 — Severity matrix
# ---------------------------------------------------------------------------


def _violation_emitting_shapes(g: Graph) -> List[Tuple[URIRef, bool]]:
    """Return (shape, is_violation_severity) pairs for shapes whose
    severity is or defaults to sh:Violation. SHACL default is Violation
    when sh:severity is not declared."""
    out: List[Tuple[URIRef, bool]] = []
    for shape in g.subjects(RDF.type, SH.NodeShape):
        severities = list(g.objects(shape, SH.severity))
        if not severities or SH.Violation in severities:
            out.append((shape, True))
    return out


def _shape_has_authored_message(g: Graph, shape: URIRef) -> bool:
    """A shape's Violation results carry an authored message when the
    shape itself declares sh:message OR every sh:property under it
    declares sh:message. This prevents silently-default 'Violates
    constraint sh:MinCountConstraintComponent' diagnostics (Q43)."""
    if (shape, SH.message, None) in g:
        return True
    # All property blocks must individually carry sh:message.
    properties = list(g.objects(shape, SH.property))
    if not properties:
        # Pure NodeShape with constraints attached directly to shape level —
        # the shape-level sh:message check above covers it.
        return False
    return all(
        (prop, SH.message, None) in g for prop in properties
    )


def test_every_violation_shape_has_authored_message(shape_file: Path):
    """Per Q43, sh:Violation defaults are not actionable — every
    Violation-emitting shape needs an author-supplied sh:message
    (either at shape level or on every sh:property block)."""
    if _is_rule_file(shape_file):
        pytest.skip("rules don't emit ValidationResults — Phase 7.4 doesn't apply")

    g = Graph()
    g.parse(shape_file, format="turtle")

    missing: List[str] = []
    for shape, is_violation in _violation_emitting_shapes(g):
        if not is_violation:
            continue
        # Skip blank-node shapes (sub-shapes); they inherit messaging
        # from their parent's sh:property blocks.
        if not isinstance(shape, URIRef):
            continue
        if not _shape_has_authored_message(g, shape):
            missing.append(str(shape))

    assert not missing, (
        f"Violation-emitting shapes in {shape_file.name} lack authored "
        f"sh:message (shape-level or on every sh:property): {missing}. "
        "Add an actionable message that names what was expected vs found "
        "(Q43)."
    )
