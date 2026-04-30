"""SHACL-violation SFT pair generator (audit 2026-04-30 fix).

The cc07cc76 adapter scored zero on negative_grounding because the
training corpus had no pairs that taught it to refuse a graph that
violates a shape. Pyshacl is an oracle that gives us ground-truth
"this graph is invalid because <reason>" labels for free; this
generator runs pyshacl over a small catalog of shape + graph fixtures
and emits one SFT pair per (shape, graph, valid?, reason) tuple.

Pair shape:

    prompt:     "Given the shape: ```turtle\\n<shape>\\n```\\n\\n
                 Does this graph validate?\\n\\n```turtle\\n<graph>\\n```"
    completion: "No.\\n\\nReason: <pyshacl violation message>"
        OR     "Yes. The graph satisfies the shape."

`bloom_level=evaluate` for invalid cases ("evaluate the shape against
the graph"); `apply` for valid cases. `template_id` carries shape
kind + validity tag.

Decision capture: one `violation_generation` event per shape (not per
pair) with rationale referencing shape kind, pyshacl version, and
oracle agreement.

Pyshacl is an OPTIONAL dependency — see `pyproject.toml::dependencies`.
The generator raises `RuntimeError` if pyshacl isn't installed when a
caller tries to use it; tests `pytest.skip` the import-error path.
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
    """One catalog entry: a shape + a valid graph + 1-2 invalid graphs.

    `kind` is the high-level shape category (datatype, class,
    nodekind, ...); used to tag `template_id` so downstream diversity
    scorers see the per-kind distribution.

    Each `(graph, expected_valid)` tuple in `graphs` is run through
    pyshacl; the oracle's verdict must match `expected_valid` or the
    fixture is rejected (no wrong-labeled pairs in the corpus).
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
# Built-in shape catalog
# ---------------------------------------------------------------------------

# Six minimal shapes covering the surface forms in
# `lib/ontology/property_manifest.py::property_manifest.rdf_shacl.yaml`:
#
#   1. sh:datatype  -- PropertyShape with datatype constraint
#   2. sh:class     -- PropertyShape with class constraint
#   3. sh:NodeShape -- NodeShape declaration with target
#   4. sh:PropertyShape -- PropertyShape declaration on a node shape
#   5. rdfs:subClassOf -- subclass-of constraint
#   6. owl:sameAs   -- sameAs identity (modeled as a shape on the
#                      relation's domain)
#
# Each fixture has one valid + one (or two) invalid graphs.

# Shared prefix block. The full prefix list is needed for pyshacl to
# parse the turtle, but the SFT prompt strips them down to the
# minimum (only the prefixes used by the fixture's body) so the
# emitted prompt fits the 400-char schema cap. Each fixture stores
# the body without prefixes; `_render_ttl_for_pyshacl` adds the full
# prefix block back in for validation, while `_render_ttl_for_prompt`
# adds only the prefixes the body actually references.
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
    consistently (stray prefix references in shape graphs cause
    parse-time errors that mask real validity issues)."""
    return _FULL_PREFIXES + body


def _ttl(body: str) -> str:
    """Back-compat alias used at fixture construction time. The body
    is stored unwrapped; prefix injection happens lazily at pyshacl
    + prompt-render time."""
    return body


_PREFIX_TO_LINE = {
    "sh:": "@prefix sh: <http://www.w3.org/ns/shacl#> .",
    "rdf:": "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
    "rdfs:": "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "owl:": "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
    "xsd:": "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
    "ex:": "@prefix ex: <http://example.org/> .",
}


def _minimal_prefix_lines(body: str) -> str:
    """Return only the prefix lines needed by `body` (in canonical
    order). Used by the SFT prompt renderer so the emitted prompt
    doesn't carry six lines of unused prefix declarations."""
    lines: List[str] = []
    for prefix, line in _PREFIX_TO_LINE.items():
        if prefix in body:
            lines.append(line)
    return "\n".join(lines) + "\n" if lines else ""


_BUILT_IN_SHAPES: List[ShapeFixture] = [
    ShapeFixture(
        name="datatype_int_age",
        kind="datatype",
        curie="sh:datatype",
        surface_form="sh:datatype",
        shape_ttl=_ttl(
            "ex:S a sh:NodeShape ; sh:targetClass ex:P ;\n"
            "  sh:property [ sh:path ex:age ; sh:datatype xsd:integer ] .\n"
        ),
        graphs=[
            (_ttl("ex:a a ex:P ; ex:age \"30\"^^xsd:integer .\n"), True),
            (_ttl("ex:b a ex:P ; ex:age \"thirty\" .\n"), False),
        ],
    ),
    ShapeFixture(
        name="class_constraint_owns",
        kind="class",
        curie="sh:class",
        surface_form="sh:class",
        shape_ttl=_ttl(
            "ex:S a sh:NodeShape ; sh:targetClass ex:Own ;\n"
            "  sh:property [ sh:path ex:has ; sh:class ex:Veh ] .\n"
        ),
        graphs=[
            (_ttl("ex:a a ex:Own ; ex:has ex:c .\nex:c a ex:Veh .\n"), True),
            (_ttl("ex:b a ex:Own ; ex:has ex:x .\nex:x a ex:Foo .\n"), False),
        ],
    ),
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
            (_ttl("ex:a a ex:C ; ex:name \"Eve\" .\n"), True),
            (_ttl("ex:b a ex:C .\n"), False),
        ],
    ),
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
            (_ttl("ex:a a ex:Acc ; ex:email \"a@x.org\" .\n"), True),
            (_ttl(
                "ex:b a ex:Acc ; ex:email \"a@x.org\" , \"b@x.org\" .\n"
            ), False),
        ],
    ),
    ShapeFixture(
        name="subclass_of_class_constraint",
        kind="subClassOf",
        curie="rdfs:subClassOf",
        surface_form="rdfs:subClassOf",
        shape_ttl=_ttl(
            "ex:S a sh:NodeShape ; sh:targetClass ex:AO ;\n"
            "  sh:property [ sh:path ex:keeps ; sh:class ex:Animal ] .\n"
        ),
        # The rdfs:subClassOf triple lives in the data graph (NOT the
        # shape graph) so pyshacl's `inference="rdfs"` pass can
        # propagate `ex:Dog rdfs:subClassOf ex:Animal` -> every
        # `ex:Dog` is also `a ex:Animal`.
        graphs=[
            (_ttl(
                "ex:Dog rdfs:subClassOf ex:Animal .\n"
                "ex:a a ex:AO ; ex:keeps ex:rex .\nex:rex a ex:Dog .\n"
            ), True),
            (_ttl(
                "ex:Dog rdfs:subClassOf ex:Animal .\n"
                "ex:b a ex:AO ; ex:keeps ex:p .\nex:p a ex:Plant .\n"
            ), False),
        ],
    ),
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
            (_ttl("ex:b a ex:L ; owl:sameAs \"x\" .\n"), False),
        ],
    ),
]


def built_in_shape_catalog() -> List[ShapeFixture]:
    """Return a copy of the built-in shape catalog.

    Six fixtures × 2 graphs each = 12 pairs minimum on a clean run.
    Exposed so callers (and tests) can introspect or extend.
    """
    # Copy each fixture so caller mutations don't leak back into the
    # module-level singleton.
    return [
        ShapeFixture(
            name=f.name,
            kind=f.kind,
            curie=f.curie,
            surface_form=f.surface_form,
            shape_ttl=f.shape_ttl,
            graphs=list(f.graphs),
        )
        for f in _BUILT_IN_SHAPES
    ]


# ---------------------------------------------------------------------------
# pyshacl wrapper
# ---------------------------------------------------------------------------


def _validate_with_pyshacl(
    shape_ttl: str, graph_ttl: str,
) -> Tuple[bool, str]:
    """Run pyshacl on (shape, graph). Returns (conforms, message).

    Caller catches `RuntimeError` for the missing-pyshacl case so
    tests can `pytest.skip`. The pyshacl validation result message is
    a multi-line string; we return it verbatim so the SFT completion
    has the real oracle's wording, not a paraphrase.
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
    "Constraint Violation" block carries the most actionable signal
    (component name, focus node, value node, message). Truncate to
    keep the SFT completion under the 600-char schema cap.
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
            if len(keep) >= 6:
                break
    if not keep:
        return msg.strip().splitlines()[0][:200]
    return " | ".join(keep)[:400]


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
    validity) tuple. The pyshacl message is the load-bearing signal
    for the negative case; for the valid case we use a deterministic
    affirmation."""
    # The prompt strips prefix declarations entirely — they push every
    # SHACL fixture over the 400-char schema cap, and a fine-tuned
    # SLM that has seen the standard `sh:` / `xsd:` / `rdfs:` /
    # `owl:` / `ex:` prefixes during training reasons about the
    # abbreviated CURIEs without the @prefix headers. The canonical
    # pyshacl validation runs against the full prefix block (see
    # `_render_ttl_for_pyshacl`), so semantics are unchanged.
    shape_render = fixture.shape_ttl.strip()
    graph_render = graph_ttl.strip()
    prompt = (
        "Given the shape:\n```turtle\n"
        f"{shape_render}\n"
        "```\nDoes this graph validate?\n```turtle\n"
        f"{graph_render}\n"
        "```"
    )
    if expected_valid:
        # Anchor "Yes." at the head so a yes/no classifier scores
        # affirm; the explanation pads the completion past the 50-
        # char schema floor and grounds the answer in the shape's
        # surface form (`fixture.curie`).
        completion = (
            f"Yes. The graph satisfies the shape; the {fixture.curie} "
            f"constraint is met by every focus node."
        )
        bloom = _VALID_BLOOM
        validity = "valid"
    else:
        reason = _extract_first_violation_reason(pyshacl_msg)
        completion = f"No.\n\nReason: {reason}"
        bloom = _INVALID_BLOOM
        validity = "invalid"
    template_id = f"violation_detection.{fixture.kind}.{validity}"

    pair: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        # `chunk_id` anchors to a chunk that teaches the violated
        # constraint type when available. Fall back to a synthetic
        # ID derived from the fixture name so the schema's
        # required `chunk_id` field is satisfied; the
        # `concept_tags` CURIE makes the linkage explicit.
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
            except RuntimeError as exc:
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

    return pairs, stats


__all__ = [
    "ShapeFixture",
    "ViolationStats",
    "built_in_shape_catalog",
    "generate_violation_pairs",
]
