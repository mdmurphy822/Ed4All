"""SHACL-violation SFT pair generator (Wave 125 expansion).

The cc07cc76 adapter scored zero on negative_grounding because the
training corpus had no pairs that taught it to refuse a graph that
violates a shape. Pyshacl is an oracle that gives us ground-truth
"this graph is invalid because <reason>" labels for free; this
generator runs pyshacl over a programmatically-expanded catalog of
shape + graph fixtures and emits one SFT pair per (shape, graph,
valid?, reason) tuple.

Pair shape (Wave 125 prompt restructure — TTL moves into the
completion so prompts stay well below the 400-char schema cap):

    prompt:
        "Does this RDF graph satisfy the SHACL shape `<name>`
         (constraint: <curie>)?\n\nGraph:\n```turtle\n<graph>\n```"

    completion (valid):
        "Yes. The graph satisfies the shape; the <curie> constraint
         is met by every focus node.\n\nShape:\n```turtle\n<shape>\n```"

    completion (invalid):
        "No.\n\nReason: <pyshacl violation reason>.\n\nShape:\n
         ```turtle\n<shape>\n```"

`bloom_level=evaluate` for invalid cases ("evaluate the shape against
the graph"); `apply` for valid cases. `template_id` carries shape
kind + validity tag.

Decision capture: one `violation_generation` event per fixture with
rationale referencing shape kind, pyshacl version, oracle agreement
counts, seed, and the fixture's chunk anchor.

Pyshacl is an OPTIONAL dependency — see `pyproject.toml::dependencies`.
The generator raises `RuntimeError` if pyshacl isn't installed when a
caller tries to use it; tests `pytest.skip` the import-error path.

Wave 125 expansion (audit fix, 2026-04-30): catalog grew from 6
hand-authored fixtures (12 pairs) to ~430 fixtures (>= 800 pairs)
covering every surface form in `property_manifest.rdf_shacl.yaml`
with deterministic programmatic factories. SHACL families:

* `_datatype_fixtures()` — sh:datatype across 8 xsd types × predicates
* `_class_fixtures()` — sh:class across 5+ hierarchies × variants
* `_cardinality_fixtures()` — sh:minCount / sh:maxCount sweeps
* `_nodeshape_fixtures()` — sh:NodeShape declarations with diverse
  constraint bodies
* `_propertyshape_fixtures()` — sh:PropertyShape (top-level)
* `_subclass_fixtures()` — rdfs:subClassOf chain depths × variants
* `_sameas_fixtures()` — owl:sameAs nodeKind / cardinality variants
* `_pattern_length_fixtures()` — sh:pattern, sh:minLength, sh:maxLength
* `_enumeration_fixtures()` — sh:in / sh:hasValue
* `_compound_fixtures()` — multi-constraint shapes (datatype+minCount,
  class+maxCount, ...)

Per-family fixture authoring uses small Python factories rather than
hand-writing each — keeping fixture counts honest and the diff size
auditable. Names are deterministic (`<family>_<param1>_<param2>_...`)
so they don't collide across re-runs.

Pinned fixture names (kept stable so the existing test suite stays
green): `datatype_int_age`, `class_constraint_owns`,
`nodeshape_min_count`, `propertyshape_max_count`,
`subclass_of_class_constraint`, `sameas_iri_kind`. All other names
derive programmatically from sweep parameters.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# Default Bloom levels per validity outcome.
_INVALID_BLOOM = "evaluate"
_VALID_BLOOM = "apply"


@dataclass
class ShapeFixture:
    """One catalog entry: a shape + a valid graph + 1+ invalid graphs.

    `kind` is the high-level shape category (datatype, class,
    NodeShape, ...); used to tag `template_id` so downstream diversity
    scorers see the per-kind distribution.

    Each `(graph, expected_valid)` tuple in `graphs` is run through
    pyshacl; the oracle's verdict must match `expected_valid` or the
    fixture's graph is dropped (no wrong-labeled pairs in the corpus).
    """

    name: str
    kind: str
    curie: str
    shape_ttl: str
    graphs: List[Tuple[str, bool]] = field(default_factory=list)
    surface_form: Optional[str] = None


@dataclass
class ViolationStats:
    fixtures_used: int = 0
    pairs_emitted: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    oracle_disagreements: int = 0
    per_kind: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TTL prefix block (shared across every fixture for pyshacl parsing).
# Prompts intentionally drop these prefix lines — a 6-line prefix block
# is the same chars overhead for every pair, and the SLM sees the
# canonical sh: / xsd: / rdfs: / owl: / ex: tokens often enough during
# training to reason about abbreviated CURIEs without the headers.
# ---------------------------------------------------------------------------
_FULL_PREFIXES = (
    "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
    "@prefix ex: <http://example.org/> .\n"
)


def _render_ttl_for_pyshacl(body: str) -> str:
    """Add the full prefix block so pyshacl can parse every fixture
    consistently."""
    return _FULL_PREFIXES + body


def _ttl(body: str) -> str:
    """Identity helper kept for back-compat with fixture construction
    sites; prefix injection happens lazily at pyshacl + prompt-render
    time."""
    return body


# ---------------------------------------------------------------------------
# Programmatic fixture factories
# ---------------------------------------------------------------------------


def _datatype_fixtures() -> List[ShapeFixture]:
    """Sweep sh:datatype across 8 xsd datatypes × 4 predicates.

    The first emitted fixture is pinned to ``datatype_int_age`` so the
    existing test suite continues to find the canonical example by
    name. Each fixture has 1 valid + 1 invalid graph.
    """

    # (datatype CURIE, valid literal, invalid literal)
    datatypes = [
        ("xsd:integer", '"42"^^xsd:integer', '"forty-two"'),
        ("xsd:string", '"hello"', "42"),
        ("xsd:decimal", '"3.14"^^xsd:decimal', '"three"'),
        ("xsd:boolean", '"true"^^xsd:boolean', '"yes"'),
        ("xsd:dateTime", '"2024-01-01T00:00:00Z"^^xsd:dateTime', '"yesterday"'),
        ("xsd:date", '"2024-01-01"^^xsd:date', '"yesterday"'),
        ("xsd:anyURI", '"http://x.org"^^xsd:anyURI', "42"),
        ("xsd:double", '"1.5"^^xsd:double', '"three"'),
    ]
    # Predicate-name variants drive uniqueness across the sweep so
    # names + prompt graphs don't collide.
    predicates = ["age", "score", "value", "amount"]
    fixtures: List[ShapeFixture] = []
    for dtype, valid_lit, invalid_lit in datatypes:
        # Short slug like "int", "string", "dec", "bool", "dt", "date",
        # "uri", "double" — keeps fixture names compact.
        slug = dtype.split(":", 1)[1].replace("xsd_", "")
        slug_short = {
            "integer": "int",
            "string": "string",
            "decimal": "dec",
            "boolean": "bool",
            "dateTime": "dt",
            "date": "date",
            "anyURI": "uri",
            "double": "double",
        }.get(slug, slug)
        for pred in predicates:
            # Pin the canonical name for the integer/age combo so the
            # existing tests can still find it.
            if dtype == "xsd:integer" and pred == "age":
                name = "datatype_int_age"
            else:
                name = f"datatype_{slug_short}_{pred}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:T_{slug_short}_{pred} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; sh:datatype {dtype} ] .\n"
            )
            valid_graph = (
                f"ex:a_{slug_short}_{pred} a ex:T_{slug_short}_{pred} ; "
                f"ex:{pred} {valid_lit} .\n"
            )
            invalid_graph = (
                f"ex:b_{slug_short}_{pred} a ex:T_{slug_short}_{pred} ; "
                f"ex:{pred} {invalid_lit} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="datatype",
                    curie="sh:datatype",
                    surface_form="sh:datatype",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _class_fixtures() -> List[ShapeFixture]:
    """Sweep sh:class across 7 class hierarchies × 4 predicates.

    Pins ``class_constraint_owns`` for back-compat. Each fixture: 1
    valid (target instance is a member) + 1 invalid (target is some
    other class).
    """

    # (expected-class slug, expected-class IRI, wrong-class slug,
    #  wrong-class IRI)
    hierarchies = [
        ("vehicle", "Veh", "foo", "Foo"),
        ("animal", "Animal", "plant", "Plant"),
        ("person", "Person", "robot", "Robot"),
        ("book", "Book", "movie", "Movie"),
        ("invoice", "Invoice", "receipt", "Receipt"),
        ("course", "Course", "lesson", "Lesson"),
        ("device", "Device", "service", "Service"),
    ]
    predicates = ["has", "owns", "uses", "links"]
    fixtures: List[ShapeFixture] = []
    for cls_slug, cls_iri, wrong_slug, wrong_iri in hierarchies:
        for pred in predicates:
            # Pin the canonical (Veh / owns) combo for back-compat.
            if cls_iri == "Veh" and pred == "owns":
                name = "class_constraint_owns"
                target_class = "Own"
            else:
                name = f"class_{cls_slug}_{pred}"
                target_class = f"T_{cls_slug}_{pred}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; sh:class ex:{cls_iri} ] .\n"
            )
            valid_graph = (
                f"ex:a_{cls_slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} ex:c_{cls_slug}_{pred} .\n"
                f"ex:c_{cls_slug}_{pred} a ex:{cls_iri} .\n"
            )
            invalid_graph = (
                f"ex:b_{cls_slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} ex:x_{cls_slug}_{pred} .\n"
                f"ex:x_{cls_slug}_{pred} a ex:{wrong_iri} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="class",
                    curie="sh:class",
                    surface_form="sh:class",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _cardinality_fixtures() -> List[ShapeFixture]:
    """Sweep sh:minCount and sh:maxCount in isolation.

    Two streams: pure minCount sweeps (``cardinality_min_*``) and pure
    maxCount sweeps (``cardinality_max_*``). The shape is declared as
    a NodeShape (the standard shape kind that hosts cardinality
    constraints in real-world SHACL), tagged kind=NodeShape.
    """

    fixtures: List[ShapeFixture] = []
    # minCount sweep: 1, 2, 3 across 4 predicates -> 12 fixtures.
    predicates = ["item", "tag", "label", "ref"]
    for n in (1, 2, 3):
        for pred in predicates:
            name = f"cardinality_min_{n}_{pred}"
            target_class = f"TMin_{n}_{pred}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; sh:minCount {n} ] .\n"
            )
            # Valid graph: exactly n distinct values.
            literals = ", ".join(f'"v{i}_{pred}"' for i in range(n))
            valid_graph = (
                f"ex:a_min_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {literals} .\n"
            )
            # Invalid graph: n-1 values when n > 1, or zero when n == 1.
            if n == 1:
                invalid_graph = (
                    f"ex:b_min_{n}_{pred} a ex:{target_class} .\n"
                )
            else:
                short_lits = ", ".join(
                    f'"v{i}_{pred}"' for i in range(n - 1)
                )
                invalid_graph = (
                    f"ex:b_min_{n}_{pred} a ex:{target_class} ; "
                    f"ex:{pred} {short_lits} .\n"
                )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="NodeShape",
                    curie="sh:NodeShape",
                    surface_form="sh:NodeShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    # maxCount sweep: 1, 2, 3 across 4 predicates -> 12 fixtures.
    for n in (1, 2, 3):
        for pred in predicates:
            name = f"cardinality_max_{n}_{pred}"
            target_class = f"TMax_{n}_{pred}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; sh:maxCount {n} ] .\n"
            )
            valid_lits = ", ".join(
                f'"v{i}_{pred}"' for i in range(n)
            )
            valid_graph = (
                f"ex:a_max_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_lits} .\n"
            )
            invalid_lits = ", ".join(
                f'"v{i}_{pred}"' for i in range(n + 1)
            )
            invalid_graph = (
                f"ex:b_max_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_lits} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="NodeShape",
                    curie="sh:NodeShape",
                    surface_form="sh:NodeShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _nodeshape_fixtures() -> List[ShapeFixture]:
    """NodeShape declarations with diverse single-constraint bodies.

    Pins ``nodeshape_min_count`` for back-compat. Provides per-shape
    NodeShape variants tagged kind=NodeShape so the catalog can carry
    50+ pairs surface-form-tagged ``sh:NodeShape``.
    """

    fixtures: List[ShapeFixture] = []
    # Pinned canonical fixture.
    fixtures.append(
        ShapeFixture(
            name="nodeshape_min_count",
            kind="NodeShape",
            curie="sh:NodeShape",
            surface_form="sh:NodeShape",
            shape_ttl=_ttl(
                "ex:S a sh:NodeShape ; sh:targetClass ex:C ;\n"
                "  sh:property [ sh:path ex:name ; sh:minCount 1 ] .\n"
            ),
            graphs=[
                (_ttl('ex:a a ex:C ; ex:name "Eve" .\n'), True),
                (_ttl("ex:b a ex:C .\n"), False),
            ],
        )
    )
    # NodeShape with sh:targetClass + nodeKind sweep.
    nodekind_combos = [
        ("IRI", "ex:o", '"x"', "iri"),
        ("Literal", '"x"', "ex:o", "lit"),
        ("BlankNodeOrIRI", "ex:o", '"x"', "boi"),
    ]
    for kind, valid_obj, invalid_obj, slug in nodekind_combos:
        for pred in ("ref", "tag", "link"):
            name = f"nodeshape_nodekind_{slug}_{pred}"
            target_class = f"TN_{slug}_{pred}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; sh:nodeKind sh:{kind} ] .\n"
            )
            valid_graph = (
                f"ex:a_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_obj} .\n"
            )
            invalid_graph = (
                f"ex:b_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_obj} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="NodeShape",
                    curie="sh:NodeShape",
                    surface_form="sh:NodeShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    # NodeShape with sh:closed (closed-world) sweep.
    closed_combos = [
        ("status", "name", "extra"),
        ("title", "code", "ghost"),
        ("kind", "label", "junk"),
    ]
    for c1, c2, intruder in closed_combos:
        name = f"nodeshape_closed_{c1}_{c2}"
        target_class = f"TC_{c1}_{c2}"
        shape_ttl = (
            f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
            f"  sh:closed true ; sh:ignoredProperties (rdf:type) ;\n"
            f"  sh:property [ sh:path ex:{c1} ] ;\n"
            f"  sh:property [ sh:path ex:{c2} ] .\n"
        )
        valid_graph = (
            f"ex:a_{c1}_{c2} a ex:{target_class} ; "
            f'ex:{c1} "x" ; ex:{c2} "y" .\n'
        )
        invalid_graph = (
            f"ex:b_{c1}_{c2} a ex:{target_class} ; "
            f'ex:{c1} "x" ; ex:{intruder} "junk" .\n'
        )
        fixtures.append(
            ShapeFixture(
                name=name,
                kind="NodeShape",
                curie="sh:NodeShape",
                surface_form="sh:NodeShape",
                shape_ttl=_ttl(shape_ttl),
                graphs=[
                    (_ttl(valid_graph), True),
                    (_ttl(invalid_graph), False),
                ],
            )
        )
    # NodeShape with sh:targetNode (instance-targeted) sweep.
    target_node_combos = [("foo", "ok", "bad"), ("bar", "yes", "no"), ("baz", "go", "stop")]
    for tn, val, inv in target_node_combos:
        name = f"nodeshape_targetnode_{tn}"
        shape_ttl = (
            f"ex:S a sh:NodeShape ; sh:targetNode ex:tn_{tn} ;\n"
            f"  sh:property [ sh:path ex:state ; sh:hasValue \"{val}\" ] .\n"
        )
        valid_graph = f'ex:tn_{tn} ex:state "{val}" .\n'
        invalid_graph = f'ex:tn_{tn} ex:state "{inv}" .\n'
        fixtures.append(
            ShapeFixture(
                name=name,
                kind="NodeShape",
                curie="sh:NodeShape",
                surface_form="sh:NodeShape",
                shape_ttl=_ttl(shape_ttl),
                graphs=[
                    (_ttl(valid_graph), True),
                    (_ttl(invalid_graph), False),
                ],
            )
        )
    # NodeShape with sh:targetSubjectsOf sweep.
    tso_combos = [("emits", "src"), ("contains", "box"), ("knows", "node")]
    for pred, slug in tso_combos:
        name = f"nodeshape_subjectsof_{pred}"
        target_class = f"TSO_{slug}"
        shape_ttl = (
            f"ex:S a sh:NodeShape ; sh:targetSubjectsOf ex:{pred} ;\n"
            f"  sh:property [ sh:path ex:label ; sh:minCount 1 ] .\n"
        )
        valid_graph = (
            f'ex:a_{slug} ex:{pred} ex:other_{slug} ; '
            f'ex:label "ok" .\n'
        )
        invalid_graph = f"ex:b_{slug} ex:{pred} ex:other2_{slug} .\n"
        fixtures.append(
            ShapeFixture(
                name=name,
                kind="NodeShape",
                curie="sh:NodeShape",
                surface_form="sh:NodeShape",
                shape_ttl=_ttl(shape_ttl),
                graphs=[
                    (_ttl(valid_graph), True),
                    (_ttl(invalid_graph), False),
                ],
            )
        )
    return fixtures


def _propertyshape_fixtures() -> List[ShapeFixture]:
    """Top-level PropertyShape fixtures (PropertyShape declared
    directly, not embedded inside a NodeShape).

    Pins ``propertyshape_max_count`` for back-compat. Each fixture
    declares ``ex:S a sh:PropertyShape ; sh:path ... ; <constraint>``,
    surface form sh:PropertyShape.
    """

    fixtures: List[ShapeFixture] = []
    # Pinned canonical fixture.
    fixtures.append(
        ShapeFixture(
            name="propertyshape_max_count",
            kind="PropertyShape",
            curie="sh:PropertyShape",
            surface_form="sh:PropertyShape",
            shape_ttl=_ttl(
                "ex:S a sh:PropertyShape ; sh:targetClass ex:Acc ;\n"
                "  sh:path ex:email ; sh:maxCount 1 .\n"
            ),
            graphs=[
                (_ttl('ex:a a ex:Acc ; ex:email "a@x.org" .\n'), True),
                (
                    _ttl(
                        'ex:b a ex:Acc ; ex:email "a@x.org" , "b@x.org" .\n'
                    ),
                    False,
                ),
            ],
        )
    )
    # PropertyShape sweep: each shape has one constraint kind.
    # (constraint TTL fragment, valid value, invalid value, slug)
    constraints = [
        ("sh:minCount 1", '"v"', None, "minc1"),
        ("sh:maxCount 2", '"a"', '"a" , "b" , "c"', "maxc2"),
        ("sh:datatype xsd:integer", '"42"^^xsd:integer', '"abc"', "dt_int"),
        ("sh:datatype xsd:string", '"hi"', "42", "dt_str"),
        ("sh:datatype xsd:boolean", '"true"^^xsd:boolean', '"yes"', "dt_bool"),
        ("sh:nodeKind sh:IRI", "ex:thing", '"x"', "nk_iri"),
        ("sh:nodeKind sh:Literal", '"x"', "ex:thing", "nk_lit"),
        ("sh:minLength 3", '"hello"', '"hi"', "mlen3"),
        ("sh:maxLength 5", '"hi"', '"way too long"', "xlen5"),
        ('sh:pattern "^[a-z]+$"', '"abc"', '"ABC"', "pat_az"),
        ('sh:hasValue "ok"', '"ok"', '"nope"', "hv_ok"),
    ]
    predicates = ["field", "attr", "prop", "slot"]
    for ctt, valid_v, invalid_v, slug in constraints:
        for pred in predicates:
            name = f"propertyshape_{slug}_{pred}"
            target_class = f"TPS_{slug}_{pred}"
            # For minCount we have no invalid value (need empty graph).
            shape_ttl = (
                f"ex:S a sh:PropertyShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:path ex:{pred} ; {ctt} .\n"
            )
            valid_graph = (
                f"ex:a_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_v} .\n"
            )
            if invalid_v is None:
                # min-count style: invalid graph has no value.
                invalid_graph = (
                    f"ex:b_{slug}_{pred} a ex:{target_class} .\n"
                )
            else:
                invalid_graph = (
                    f"ex:b_{slug}_{pred} a ex:{target_class} ; "
                    f"ex:{pred} {invalid_v} .\n"
                )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="PropertyShape",
                    curie="sh:PropertyShape",
                    surface_form="sh:PropertyShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _subclass_fixtures() -> List[ShapeFixture]:
    """rdfs:subClassOf chain depths × variants × predicates.

    Pins ``subclass_of_class_constraint`` for back-compat. The
    rdfs:subClassOf triple lives in the data graph; pyshacl's
    ``inference="rdfs"`` pass propagates membership up the chain. A
    valid graph has the focus's instance reachable through the chain;
    an invalid graph wires the instance to a parallel-but-not-related
    class.
    """

    fixtures: List[ShapeFixture] = []
    # Pinned canonical fixture.
    fixtures.append(
        ShapeFixture(
            name="subclass_of_class_constraint",
            kind="subClassOf",
            curie="rdfs:subClassOf",
            surface_form="rdfs:subClassOf",
            shape_ttl=_ttl(
                "ex:S a sh:NodeShape ; sh:targetClass ex:AO ;\n"
                "  sh:property [ sh:path ex:keeps ; sh:class ex:Animal ] .\n"
            ),
            graphs=[
                (
                    _ttl(
                        "ex:Dog rdfs:subClassOf ex:Animal .\n"
                        "ex:a a ex:AO ; ex:keeps ex:rex .\nex:rex a ex:Dog .\n"
                    ),
                    True,
                ),
                (
                    _ttl(
                        "ex:Dog rdfs:subClassOf ex:Animal .\n"
                        "ex:b a ex:AO ; ex:keeps ex:p .\nex:p a ex:Plant .\n"
                    ),
                    False,
                ),
            ],
        )
    )
    # Chain-depth sweep: depth 1, 2, 3 (instance -> child -> parent /
    # grandparent / great-grandparent).
    chain_specs = [
        # (parent slug, parent IRI, intermediates ordered child->root, slug)
        ("animal", "Animal", ["Mammal", "Dog"], "dog_mammal"),
        ("animal", "Animal", ["Bird", "Sparrow"], "sparrow_bird"),
        ("animal", "Animal", ["Reptile", "Lizard", "Gecko"], "gecko_lizard"),
        ("vehicle", "Vehicle", ["Car"], "car"),
        ("vehicle", "Vehicle", ["Boat", "Sailboat"], "sail_boat"),
        ("vehicle", "Vehicle", ["Aircraft", "Jet", "AirJet"], "airjet"),
        ("food", "Food", ["Fruit", "Apple"], "apple"),
        ("food", "Food", ["Veggie", "Carrot"], "carrot"),
        ("device", "Device", ["Phone"], "phone"),
        ("device", "Device", ["Laptop", "Notebook"], "laptop_nb"),
    ]
    predicates = ["keeps", "owns", "has"]
    for parent_slug, parent_iri, chain, fix_slug in chain_specs:
        for pred in predicates:
            name = f"subclass_{fix_slug}_{pred}"
            target_class = f"SCT_{fix_slug}_{pred}"
            # Build the chain TTL: chain[0] subClassOf chain[1] ...
            # subClassOf parent.
            chain_lines = []
            full_chain = list(chain) + [parent_iri]
            for child, par in zip(full_chain[:-1], full_chain[1:]):
                chain_lines.append(
                    f"ex:{child} rdfs:subClassOf ex:{par} ."
                )
            chain_ttl = "\n".join(chain_lines) + "\n"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; "
                f"sh:class ex:{parent_iri} ] .\n"
            )
            # Valid: focus -> instance (chain[0])
            leaf = chain[0]
            valid_graph = (
                f"{chain_ttl}"
                f"ex:a_{fix_slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} ex:i_{fix_slug}_{pred} .\n"
                f"ex:i_{fix_slug}_{pred} a ex:{leaf} .\n"
            )
            # Invalid: focus -> instance of an unrelated class.
            invalid_graph = (
                f"{chain_ttl}"
                f"ex:b_{fix_slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} ex:j_{fix_slug}_{pred} .\n"
                f"ex:j_{fix_slug}_{pred} a ex:Other_{parent_slug} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="subClassOf",
                    curie="rdfs:subClassOf",
                    surface_form="rdfs:subClassOf",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _sameas_fixtures() -> List[ShapeFixture]:
    """owl:sameAs constraint sweep.

    Pins ``sameas_iri_kind`` for back-compat. Surface form
    ``owl:sameAs``. Variants drawn from the SHACL constraints
    naturally applied to identity links: ``sh:nodeKind sh:IRI``,
    ``sh:minCount 1`` (must declare at least one twin), ``sh:maxCount
    1`` (only one twin), and ``sh:minCount 2`` (transitive chain
    requires multiple aliases).
    """

    fixtures: List[ShapeFixture] = []
    # Pinned canonical fixture.
    fixtures.append(
        ShapeFixture(
            name="sameas_iri_kind",
            kind="sameAs",
            curie="owl:sameAs",
            surface_form="owl:sameAs",
            shape_ttl=_ttl(
                "ex:S a sh:NodeShape ; sh:targetClass ex:L ;\n"
                "  sh:property [ sh:path owl:sameAs ; sh:nodeKind sh:IRI ; "
                "sh:minCount 1 ] .\n"
            ),
            graphs=[
                (_ttl("ex:a a ex:L ; owl:sameAs ex:o .\n"), True),
                (_ttl('ex:b a ex:L ; owl:sameAs "x" .\n'), False),
            ],
        )
    )
    # sameAs constraint sweep.
    # (constraint TTL, valid value, invalid value, slug)
    sameas_constraints = [
        ("sh:nodeKind sh:IRI ; sh:minCount 1", "ex:other_$$",
         '"literal_$$"', "iri_min1"),
        ("sh:minCount 1", "ex:peer_$$", None, "min1"),
        ("sh:maxCount 1", "ex:t1_$$",
         "ex:t1_$$ , ex:t2_$$ , ex:t3_$$", "max1"),
        ("sh:minCount 2 ; sh:nodeKind sh:IRI", "ex:p1_$$ , ex:p2_$$",
         "ex:p1_$$", "min2"),
        ("sh:nodeKind sh:Literal", '"twin_$$"', "ex:peer_$$", "lit"),
    ]
    targets = [
        ("Person", "person", "alice"),
        ("Account", "account", "acct"),
        ("Doc", "doc", "papyrus"),
        ("Org", "org", "ent"),
        ("Item", "item", "thing"),
        ("Asset", "asset", "rig"),
    ]
    for ctt, valid_v_template, invalid_v_template, c_slug in sameas_constraints:
        for cls_iri, cls_slug, focus_slug in targets:
            name = f"sameas_{c_slug}_{cls_slug}_{focus_slug}"
            target_class = f"SA_{c_slug}_{cls_slug}_{focus_slug}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path owl:sameAs ; {ctt} ] .\n"
            )
            unique = f"{c_slug}_{cls_slug}_{focus_slug}"
            valid_v = valid_v_template.replace("$$", unique)
            valid_graph = (
                f"ex:{focus_slug}_{c_slug} a ex:{target_class} ; "
                f"owl:sameAs {valid_v} .\n"
            )
            if invalid_v_template is None:
                invalid_graph = (
                    f"ex:{focus_slug}_{c_slug}_inv a ex:{target_class} .\n"
                )
            else:
                invalid_v = invalid_v_template.replace("$$", unique)
                invalid_graph = (
                    f"ex:{focus_slug}_{c_slug}_inv a ex:{target_class} ; "
                    f"owl:sameAs {invalid_v} .\n"
                )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="sameAs",
                    curie="owl:sameAs",
                    surface_form="owl:sameAs",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _pattern_length_fixtures() -> List[ShapeFixture]:
    """sh:pattern, sh:minLength, sh:maxLength sweep.

    These are tagged kind=PropertyShape since they appear inside
    PropertyShape bodies in real SHACL — kept under sh:PropertyShape
    surface form.
    """

    fixtures: List[ShapeFixture] = []
    # sh:pattern sweep — 4 patterns × 4 predicates = 16 fixtures.
    patterns = [
        # (regex, valid value, invalid value, slug)
        ("^[a-z]+$", '"abc"', '"ABC"', "lower"),
        ("^[A-Z]+$", '"HELLO"', '"hello"', "upper"),
        (r"^\\d+$", '"42"', '"forty"', "digits"),
        ("^[a-zA-Z0-9_]+$", '"id_42"', '"id 42"', "ident"),
    ]
    pred_set = ["code", "tag", "id", "ref"]
    for regex, valid_v, invalid_v, slug in patterns:
        for pred in pred_set:
            name = f"pattern_{slug}_{pred}"
            target_class = f"TPL_{slug}_{pred}"
            shape_ttl = (
                f"ex:S a sh:PropertyShape ; sh:targetClass ex:{target_class} ;\n"
                f'  sh:path ex:{pred} ; sh:pattern "{regex}" .\n'
            )
            valid_graph = (
                f"ex:a_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_v} .\n"
            )
            invalid_graph = (
                f"ex:b_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_v} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="PropertyShape",
                    curie="sh:PropertyShape",
                    surface_form="sh:PropertyShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    # sh:minLength sweep — 3 values × 3 predicates.
    for n in (3, 5, 8):
        for pred in ("name", "title", "phrase"):
            valid_v = '"' + "x" * (n + 2) + '"'
            invalid_v = '"' + "y" * max(n - 2, 1) + '"'
            name = f"minlength_{n}_{pred}"
            target_class = f"TML_{n}_{pred}"
            shape_ttl = (
                f"ex:S a sh:PropertyShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:path ex:{pred} ; sh:minLength {n} .\n"
            )
            valid_graph = (
                f"ex:a_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_v} .\n"
            )
            invalid_graph = (
                f"ex:b_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_v} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="PropertyShape",
                    curie="sh:PropertyShape",
                    surface_form="sh:PropertyShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    # sh:maxLength sweep.
    for n in (3, 5, 8):
        for pred in ("nick", "abbr", "key"):
            valid_v = '"' + "x" * max(n - 1, 1) + '"'
            invalid_v = '"' + "y" * (n + 4) + '"'
            name = f"maxlength_{n}_{pred}"
            target_class = f"TMX_{n}_{pred}"
            shape_ttl = (
                f"ex:S a sh:PropertyShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:path ex:{pred} ; sh:maxLength {n} .\n"
            )
            valid_graph = (
                f"ex:a_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_v} .\n"
            )
            invalid_graph = (
                f"ex:b_{n}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_v} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="PropertyShape",
                    curie="sh:PropertyShape",
                    surface_form="sh:PropertyShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _enumeration_fixtures() -> List[ShapeFixture]:
    """sh:in / sh:hasValue enumeration sweep — tagged
    kind=PropertyShape (these appear in PropertyShape bodies)."""

    fixtures: List[ShapeFixture] = []
    # sh:in sweep.
    in_combos = [
        ('"active" "closed" "pending"', '"active"', '"deleted"', "status"),
        ('"red" "green" "blue"', '"green"', '"purple"', "color"),
        ('"low" "med" "high"', '"med"', '"extreme"', "level"),
        ('"draft" "published"', '"draft"', '"archived"', "state"),
    ]
    in_predicates = ["state", "tier", "tag"]
    for items, valid_v, invalid_v, slug in in_combos:
        for pred in in_predicates:
            name = f"enum_in_{slug}_{pred}"
            target_class = f"TEN_{slug}_{pred}"
            shape_ttl = (
                f"ex:S a sh:PropertyShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:path ex:{pred} ; sh:in ({items}) .\n"
            )
            valid_graph = (
                f"ex:a_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_v} .\n"
            )
            invalid_graph = (
                f"ex:b_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_v} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="PropertyShape",
                    curie="sh:PropertyShape",
                    surface_form="sh:PropertyShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    # sh:hasValue sweep.
    has_combos = [
        ('"admin"', '"admin"', '"user"', "admin"),
        ('"yes"', '"yes"', '"no"', "yes"),
        ('"english"', '"english"', '"klingon"', "english"),
        ('"true"^^xsd:boolean', '"true"^^xsd:boolean',
         '"false"^^xsd:boolean', "tbool"),
    ]
    has_predicates = ["role", "flag", "lang"]
    for value, valid_v, invalid_v, slug in has_combos:
        for pred in has_predicates:
            name = f"enum_hasval_{slug}_{pred}"
            target_class = f"TEH_{slug}_{pred}"
            shape_ttl = (
                f"ex:S a sh:PropertyShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:path ex:{pred} ; sh:hasValue {value} .\n"
            )
            valid_graph = (
                f"ex:a_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {valid_v} .\n"
            )
            invalid_graph = (
                f"ex:b_{slug}_{pred} a ex:{target_class} ; "
                f"ex:{pred} {invalid_v} .\n"
            )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="PropertyShape",
                    curie="sh:PropertyShape",
                    surface_form="sh:PropertyShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def _compound_fixtures() -> List[ShapeFixture]:
    """Multi-constraint shapes — every fixture has 2+ distinct SHACL
    constraint predicates inside the same property body. Tagged
    kind=NodeShape so these contribute to that surface form's count.
    """

    fixtures: List[ShapeFixture] = []
    # (constraint TTL fragment with 2+ constraints, valid graph body,
    #  invalid graph body, slug describing the combo)
    combos = [
        (
            "sh:datatype xsd:integer ; sh:minCount 1",
            'ex:age "30"^^xsd:integer .',
            ".",  # missing -> minCount fails
            "dt_int_min1",
            "age",
        ),
        (
            "sh:datatype xsd:string ; sh:maxCount 1",
            'ex:label "x" .',
            'ex:label "x" , "y" .',
            "dt_str_max1",
            "label",
        ),
        # Note: cls_max1 / cls_min1 inline both the focus type and the
        # typed object so pyshacl's sh:class check resolves. The "@@TS@@"
        # placeholder is rewritten per target slug below to keep IRIs
        # unique across the 7-target sweep.
        (
            "sh:class ex:Veh ; sh:maxCount 1",
            "ex:has ex:cveh_@@TS@@ .\nex:cveh_@@TS@@ a ex:Veh .",
            "ex:has ex:cveh_a_@@TS@@ , ex:cveh_b_@@TS@@ .\n"
            "ex:cveh_a_@@TS@@ a ex:Veh .\nex:cveh_b_@@TS@@ a ex:Veh .",
            "cls_max1",
            "has",
        ),
        (
            "sh:class ex:Animal ; sh:minCount 1",
            "ex:keeps ex:rex_@@TS@@ .\nex:rex_@@TS@@ a ex:Animal .",
            ".",
            "cls_min1",
            "keeps",
        ),
        (
            "sh:nodeKind sh:IRI ; sh:minCount 1",
            "ex:link ex:o .",
            ".",
            "nk_iri_min",
            "link",
        ),
        (
            "sh:nodeKind sh:Literal ; sh:maxCount 2",
            'ex:title "x" .',
            'ex:title "x" , "y" , "z" .',
            "nk_lit_max",
            "title",
        ),
        (
            "sh:datatype xsd:string ; sh:minLength 3",
            'ex:tag "abc" .',
            'ex:tag "x" .',
            "str_minlen",
            "tag",
        ),
        (
            "sh:datatype xsd:string ; sh:maxLength 5",
            'ex:abbr "abc" .',
            'ex:abbr "way too long" .',
            "str_maxlen",
            "abbr",
        ),
        (
            'sh:pattern "^[a-z]+$" ; sh:minLength 2',
            'ex:slug "ok" .',
            'ex:slug "X" .',
            "pat_minlen",
            "slug",
        ),
        (
            'sh:in ("a" "b") ; sh:minCount 1',
            'ex:opt "a" .',
            ".",
            "in_min1",
            "opt",
        ),
        (
            'sh:hasValue "ok" ; sh:datatype xsd:string',
            'ex:flag "ok" .',
            "ex:flag 42 .",
            "hv_str",
            "flag",
        ),
        (
            "sh:minCount 1 ; sh:maxCount 3",
            'ex:item "a" .',
            'ex:item "a" , "b" , "c" , "d" .',
            "min1max3",
            "item",
        ),
        (
            "sh:datatype xsd:integer ; sh:minCount 2 ; sh:maxCount 3",
            'ex:n "1"^^xsd:integer , "2"^^xsd:integer .',
            'ex:n "1"^^xsd:integer .',
            "int_2to3",
            "n",
        ),
        (
            "sh:class ex:Person ; sh:nodeKind sh:IRI",
            "ex:auth ex:p_@@TS@@ .\nex:p_@@TS@@ a ex:Person .",
            'ex:auth "alice" .',
            "cls_iri",
            "auth",
        ),
        (
            'sh:datatype xsd:integer ; sh:minInclusive 0',
            'ex:qty "5"^^xsd:integer .',
            'ex:qty "-3"^^xsd:integer .',
            "int_minincl",
            "qty",
        ),
        (
            'sh:datatype xsd:integer ; sh:maxInclusive 100',
            'ex:pct "50"^^xsd:integer .',
            'ex:pct "150"^^xsd:integer .',
            "int_maxincl",
            "pct",
        ),
        (
            'sh:nodeKind sh:IRI ; sh:maxCount 1',
            "ex:owner ex:o_@@TS@@ .",
            "ex:owner ex:o1_@@TS@@ , ex:o2_@@TS@@ .",
            "iri_max1",
            "owner",
        ),
        (
            'sh:datatype xsd:string ; sh:pattern "^[A-Z]"',
            'ex:code "Hello" .',
            'ex:code "lower" .',
            "str_pat_upper",
            "code",
        ),
        (
            'sh:minLength 2 ; sh:maxLength 6',
            'ex:label "abcd" .',
            'ex:label "x" .',
            "len_2to6",
            "label",
        ),
        (
            'sh:minCount 1 ; sh:nodeKind sh:Literal',
            'ex:note "v" .',
            "ex:note ex:o_@@TS@@ .",
            "min1_lit",
            "note",
        ),
    ]
    targets = [
        "Acc",
        "Rec",
        "Doc",
        "Lib",
        "Reg",
        "Plan",
        "Bin",
    ]
    for combo in combos:
        ctt, valid_body, invalid_body, slug, pred = combo
        for tslug in targets:
            name = f"compound_{slug}_{tslug.lower()}"
            target_class = f"CT_{slug}_{tslug}"
            shape_ttl = (
                f"ex:S a sh:NodeShape ; sh:targetClass ex:{target_class} ;\n"
                f"  sh:property [ sh:path ex:{pred} ; {ctt} ] .\n"
            )
            # Per-target placeholder rewrite — fixtures that need to
            # name a typed individual (sh:class branch, cls_iri) carry
            # an "@@TS@@" token; substitute the lower-cased target slug
            # so each variant ends up with a unique IRI.
            ts_token = tslug.lower()
            v_body = valid_body.replace("@@TS@@", ts_token)
            iv_body = invalid_body.replace("@@TS@@", ts_token)
            # Build valid/invalid graphs by replacing the leading "."
            # placeholder when the body is just a sentinel for an empty
            # graph. Multi-line bodies (with "\n") declare extra
            # triples (e.g. typed individuals); inline them after the
            # focus-node spine.
            if v_body == ".":
                valid_graph = (
                    f"ex:a_{slug}_{tslug} a ex:{target_class} .\n"
                )
            elif "\n" in v_body:
                first, rest = v_body.split("\n", 1)
                valid_graph = (
                    f"ex:a_{slug}_{tslug} a ex:{target_class} ; "
                    f"{first}\n{rest}\n"
                )
            else:
                valid_graph = (
                    f"ex:a_{slug}_{tslug} a ex:{target_class} ; "
                    f"{v_body}\n"
                )
            if iv_body == ".":
                invalid_graph = (
                    f"ex:b_{slug}_{tslug} a ex:{target_class} .\n"
                )
            elif "\n" in iv_body:
                first, rest = iv_body.split("\n", 1)
                invalid_graph = (
                    f"ex:b_{slug}_{tslug} a ex:{target_class} ; "
                    f"{first}\n{rest}\n"
                )
            else:
                invalid_graph = (
                    f"ex:b_{slug}_{tslug} a ex:{target_class} ; "
                    f"{iv_body}\n"
                )
            fixtures.append(
                ShapeFixture(
                    name=name,
                    kind="NodeShape",
                    curie="sh:NodeShape",
                    surface_form="sh:NodeShape",
                    shape_ttl=_ttl(shape_ttl),
                    graphs=[
                        (_ttl(valid_graph), True),
                        (_ttl(invalid_graph), False),
                    ],
                )
            )
    return fixtures


def built_in_shape_catalog() -> List[ShapeFixture]:
    """Return the union of all family fixtures.

    Wave 125 expansion: ~430 fixtures × 2 graphs/fixture = ~860 pairs
    deterministically generated and pyshacl-verified. Per-family
    breakdown is computed at call time so the count tracks with any
    factory edits.

    The returned list is a fresh copy each call so caller mutations
    don't leak back into the module-level singleton.
    """

    families: List[ShapeFixture] = []
    families.extend(_datatype_fixtures())
    families.extend(_class_fixtures())
    families.extend(_cardinality_fixtures())
    families.extend(_nodeshape_fixtures())
    families.extend(_propertyshape_fixtures())
    families.extend(_subclass_fixtures())
    families.extend(_sameas_fixtures())
    families.extend(_pattern_length_fixtures())
    families.extend(_enumeration_fixtures())
    families.extend(_compound_fixtures())
    # Defensive copy of each fixture.
    return [
        ShapeFixture(
            name=f.name,
            kind=f.kind,
            curie=f.curie,
            surface_form=f.surface_form,
            shape_ttl=f.shape_ttl,
            graphs=list(f.graphs),
        )
        for f in families
    ]


# ---------------------------------------------------------------------------
# pyshacl wrapper
# ---------------------------------------------------------------------------


def _validate_with_pyshacl(
    shape_ttl: str, graph_ttl: str,
) -> Tuple[bool, str]:
    """Run pyshacl on (shape, graph). Returns (conforms, message).

    Caller catches `RuntimeError` for the missing-pyshacl case so
    tests can `pytest.skip`.
    """
    try:
        import pyshacl  # noqa: PLC0415 — lazy by design
        from rdflib import Graph  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "violation_generator requires pyshacl + rdflib. "
            "Install with: pip install -e .[training] or "
            "pip install pyshacl rdflib"
        ) from exc

    sg = Graph()
    sg.parse(data=_render_ttl_for_pyshacl(shape_ttl), format="turtle")
    dg = Graph()
    dg.parse(data=_render_ttl_for_pyshacl(graph_ttl), format="turtle")
    conforms, _results, msg = pyshacl.validate(
        dg, shacl_graph=sg, inference="rdfs",
    )
    return bool(conforms), str(msg)


def _extract_first_violation_reason(msg: str) -> str:
    """Pluck the first violation block from pyshacl's report message.

    pyshacl's textual report is multi-line; the first
    "Constraint Violation" block carries the most actionable signal.
    Truncated to keep the SFT completion under the 600-char schema cap
    once shape TTL is appended.
    """
    if not msg:
        return "Graph fails the shape."
    lines = msg.splitlines()
    keep: List[str] = []
    capture = False
    for line in lines:
        if "Constraint Violation" in line:
            capture = True
        if capture:
            if not line.strip():
                if keep:
                    break
                continue
            keep.append(line.strip())
            if len(keep) >= 4:
                break
    if not keep:
        return msg.strip().splitlines()[0][:160]
    # Trim hard so the shape TTL still fits in the completion.
    return " | ".join(keep)[:240]


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _last_event_id(capture: Any) -> str:
    decisions = getattr(capture, "decisions", None) or []
    if not decisions:
        raise RuntimeError(
            "violation_generator: capture has no logged decisions; "
            "log a stage-start decision before generating pairs."
        )
    last = decisions[-1]
    return str(last.get("event_id", "")) if isinstance(last, dict) else ""


def _resolve_chunk_id_for_property(
    surface_form: Optional[str],
    chunks_by_form: Optional[Dict[str, List[str]]],
) -> Optional[str]:
    """Pick a chunk that teaches a property's surface form, when one
    exists. Returns None when no chunk teaches it (caller writes the
    CURIE into `concept_tags` instead)."""
    if not surface_form or not chunks_by_form:
        return None
    candidates = chunks_by_form.get(surface_form) or []
    if not candidates:
        return None
    return candidates[0]


def _build_pair(
    *,
    fixture: ShapeFixture,
    graph_ttl: str,
    expected_valid: bool,
    pyshacl_msg: str,
    decision_capture_id: str,
    seed: int,
    chunk_id: Optional[str],
) -> Dict[str, Any]:
    """Render one instruction-pair record from a (shape, graph,
    validity) tuple.

    Wave 125 prompt restructure: prompt carries the question + graph
    TTL only (every pair is unique because the graph is unique per
    fixture-graph tuple). Shape TTL moves into the completion alongside
    the verdict + (for invalid pairs) pyshacl's violation reason. This
    keeps prompt < 400 even when fixtures programmatically blow up
    shape size, while completion < 600 still leaves room for the shape.
    """
    shape_render = fixture.shape_ttl.strip()
    graph_render = graph_ttl.strip()
    prompt = (
        f"Does this RDF graph satisfy the SHACL shape `{fixture.name}` "
        f"(constraint: {fixture.curie})?\n\n"
        f"Graph:\n```turtle\n{graph_render}\n```"
    )
    if expected_valid:
        # Anchor "Yes." at the head so a yes/no classifier scores
        # affirm; the explanation grounds the answer in the surface
        # form.
        completion = (
            f"Yes. The graph satisfies the shape; the {fixture.curie} "
            f"constraint is met by every focus node.\n\n"
            f"Shape:\n```turtle\n{shape_render}\n```"
        )
        bloom = _VALID_BLOOM
        validity = "valid"
    else:
        reason = _extract_first_violation_reason(pyshacl_msg)
        completion = (
            f"No.\n\nReason: {reason}\n\n"
            f"Shape:\n```turtle\n{shape_render}\n```"
        )
        bloom = _INVALID_BLOOM
        validity = "invalid"
    template_id = f"violation_detection.{fixture.kind}.{validity}"

    pair: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "chunk_id": chunk_id or f"violation_fixture:{fixture.name}",
        "lo_refs": ["violation-detection"],
        "bloom_level": bloom,
        "content_type": "violation_detection",
        "seed": seed,
        "decision_capture_id": decision_capture_id,
        "template_id": template_id,
        "provider": "mock",
        "schema_version": "v1",
        "requires_source_citation": False,
        "concept_tags": [fixture.curie],
        "shape_kind": fixture.kind,
        "shape_curie": fixture.curie,
        "expected_validity": validity,
    }
    return pair


def generate_violation_pairs(
    *,
    capture: Any,
    fixtures: Optional[List[ShapeFixture]] = None,
    chunks_by_surface_form: Optional[Dict[str, List[str]]] = None,
    seed: int = 17,
    max_pairs: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], ViolationStats]:
    """Run pyshacl over each fixture and emit one pair per graph.

    Args:
        capture: DecisionCapture-shaped object. Receives one
            `violation_generation` event per FIXTURE (not per pair).
        fixtures: Override list. Defaults to `built_in_shape_catalog()`.
        chunks_by_surface_form: Map property surface form (e.g.
            ``"sh:datatype"``) -> list of chunk ids that teach that
            form. The first chunk is used as the pair's anchor when
            the fixture's `surface_form` matches a key here. Falls
            back to a synthetic `violation_fixture:<name>` id when
            unset / no match — the `concept_tags` CURIE preserves the
            linkage either way.
        seed: Carried into the emitted pair's `seed` field.
        max_pairs: Optional cap on emitted pairs (Wave 125a). When
            set, the post-validation pair list is trimmed via a
            family-balanced round-robin so every surface form keeps
            representation up to the cap. ``None`` (default) =
            unlimited; the entire validated catalog is returned.
            Truncation is deterministic — same fixtures + same cap
            always select the same pairs.

    Returns:
        `(pairs, stats)`. Stats reports oracle disagreements (where
        pyshacl said something different than the fixture claimed) —
        a non-zero value means the fixture catalog is broken and we
        skipped those pairs rather than emit wrong-labeled training
        data. Tests should assert `oracle_disagreements == 0`.
    """
    if capture is None:
        raise ValueError(
            "violation_generator requires a DecisionCapture (got None); "
            "the generator emits a violation_generation event per "
            "shape fixture and anchors decision_capture_id from it."
        )
    fixtures = fixtures if fixtures is not None else built_in_shape_catalog()
    stats = ViolationStats()
    pairs: List[Dict[str, Any]] = []

    # Resolve pyshacl version once (for the audit trail in the
    # decision-capture rationale). Defer the import error so callers
    # who only want to exercise the fixture catalog can still load
    # this module.
    try:
        import pyshacl  # noqa: PLC0415
        pyshacl_version = getattr(pyshacl, "__version__", "unknown")
    except ImportError:
        pyshacl_version = "missing"

    for fixture in fixtures:
        valid_count = 0
        invalid_count = 0
        oracle_disagree = 0
        fixture_pairs: List[Dict[str, Any]] = []

        # Run pyshacl over each (graph, expected_valid) tuple BEFORE
        # logging the per-fixture decision so the rationale can carry
        # accurate counts. Disagreements (pyshacl says one thing, the
        # fixture claims another) drop the fixture's pair entirely
        # rather than poison the corpus with wrong-labeled training
        # data.
        validated_graphs: List[Tuple[str, bool, str]] = []
        for graph_ttl, expected_valid in fixture.graphs:
            try:
                actual_valid, msg = _validate_with_pyshacl(
                    fixture.shape_ttl, graph_ttl,
                )
            except RuntimeError:
                # Pyshacl missing -> propagate so the caller (or test)
                # sees the actionable error instead of a silent zero
                # emission.
                raise
            if actual_valid != expected_valid:
                oracle_disagree += 1
                stats.oracle_disagreements += 1
                logger.warning(
                    "violation_generator: pyshacl disagrees with fixture "
                    "%r expected_valid=%s; skipping this graph "
                    "rather than emit a wrong-labeled pair.",
                    fixture.name, expected_valid,
                )
                continue
            validated_graphs.append((graph_ttl, actual_valid, msg))

        if not validated_graphs:
            continue

        chunk_id = _resolve_chunk_id_for_property(
            fixture.surface_form, chunks_by_surface_form,
        )

        capture.log_decision(
            decision_type="violation_generation",
            decision=(
                f"Emitting violation-detection pairs for shape "
                f"{fixture.name!r} (kind={fixture.kind}, "
                f"curie={fixture.curie}). Validated "
                f"{len(validated_graphs)}/{len(fixture.graphs)} graphs "
                f"against pyshacl=={pyshacl_version}."
            ),
            rationale=(
                f"Pyshacl is the ground-truth oracle for SHACL "
                f"conformance; running it offline lets us emit "
                f"(graph, valid?, reason) SFT pairs whose labels are "
                f"verified by the same engine the eval harness uses. "
                f"Shape kind={fixture.kind} mirrors property manifest "
                f"surface form {fixture.curie!r}; chunk anchor="
                f"{chunk_id or 'synthetic'}; oracle_disagreements="
                f"{oracle_disagree}; pyshacl_version={pyshacl_version}; "
                f"seed={seed}."
            ),
            alternatives_considered=[
                {
                    "option": "regex-based validity check",
                    "reason_rejected": (
                        "regex can't handle datatype / class / "
                        "node-kind constraints; pyshacl is the only "
                        "ToS-clean oracle that catches every shape "
                        "violation class."
                    ),
                },
                {
                    "option": "skip oracle (trust the fixture)",
                    "reason_rejected": (
                        "fixture authors make mistakes; an oracle "
                        "catches wrong-labeled pairs before they "
                        "poison the corpus."
                    ),
                },
            ],
        )
        decision_id = _last_event_id(capture)

        for graph_ttl, actual_valid, msg in validated_graphs:
            pair = _build_pair(
                fixture=fixture,
                graph_ttl=graph_ttl,
                expected_valid=actual_valid,
                pyshacl_msg=msg,
                decision_capture_id=decision_id,
                seed=seed,
                chunk_id=chunk_id,
            )
            fixture_pairs.append(pair)
            if actual_valid:
                valid_count += 1
                stats.valid_pairs += 1
            else:
                invalid_count += 1
                stats.invalid_pairs += 1

        pairs.extend(fixture_pairs)
        stats.fixtures_used += 1
        stats.pairs_emitted += len(fixture_pairs)
        stats.per_kind[fixture.kind] = (
            stats.per_kind.get(fixture.kind, 0) + len(fixture_pairs)
        )

    # Wave 125a: optional cap with family-balanced round-robin so a
    # cap below the catalog size keeps every surface form represented
    # rather than dropping a whole family. Deterministic — same input
    # always produces the same selection.
    if max_pairs is not None and 0 <= max_pairs < len(pairs):
        pairs, stats = _apply_max_pairs_cap(pairs, stats, int(max_pairs))

    return pairs, stats


def _apply_max_pairs_cap(
    pairs: List[Dict[str, Any]],
    stats: ViolationStats,
    max_pairs: int,
) -> Tuple[List[Dict[str, Any]], ViolationStats]:
    """Trim ``pairs`` to ``max_pairs`` via family-balanced round-robin.

    Pairs are bucketed by ``shape_curie`` (= surface form). Each pass
    of the round-robin picks one pair per non-empty bucket in
    deterministic CURIE-sorted order; the loop stops when ``max_pairs``
    is reached. Within a bucket the original emission order is
    preserved (so valid + invalid pairs interleave the way pyshacl
    yielded them).

    Stats are recomputed from the trimmed pair list so
    ``valid_pairs`` / ``invalid_pairs`` / ``per_kind`` /
    ``pairs_emitted`` reflect what the caller actually receives.
    ``fixtures_used`` and ``oracle_disagreements`` are preserved (they
    describe what pyshacl saw, not what was emitted).
    """
    by_curie: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_curie.setdefault(str(p.get("shape_curie", "")), []).append(p)
    keys = sorted(by_curie.keys())
    queues = {k: list(by_curie[k]) for k in keys}
    out: List[Dict[str, Any]] = []
    while len(out) < max_pairs:
        progressed = False
        for k in keys:
            if not queues[k]:
                continue
            out.append(queues[k].pop(0))
            progressed = True
            if len(out) >= max_pairs:
                break
        if not progressed:
            break

    trimmed_stats = ViolationStats(
        fixtures_used=stats.fixtures_used,
        pairs_emitted=len(out),
        valid_pairs=sum(
            1 for p in out if p.get("expected_validity") == "valid"
        ),
        invalid_pairs=sum(
            1 for p in out if p.get("expected_validity") == "invalid"
        ),
        oracle_disagreements=stats.oracle_disagreements,
        per_kind={},
    )
    for p in out:
        k = str(p.get("shape_kind", ""))
        trimmed_stats.per_kind[k] = trimmed_stats.per_kind.get(k, 0) + 1
    return out, trimmed_stats


__all__ = [
    "ShapeFixture",
    "ViolationStats",
    "built_in_shape_catalog",
    "generate_violation_pairs",
]
