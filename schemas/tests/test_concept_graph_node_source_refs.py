"""Wave 10 — concept_graph_semantic node source_refs[] schema extension.

Wave 10 adds an optional ``source_refs[]`` array on concept graph nodes,
populated by copying ``source.source_references[]`` from each concept's
first chunk (by sorted occurrences[0]). Entries conform to the
canonical ``schemas/knowledge/source_reference.schema.json`` shape.

Contract locked by this suite:

- node with ``source_refs[]`` populated validates
- node without ``source_refs`` still validates (legacy corpora, or
  concepts whose first-occurrence chunk carries no refs)
- multiple refs per node (carried from a chunk that merged 2+ sections)
- malformed refs (missing required fields, bad enums) rejected via the
  ``$ref`` to source_reference.schema.json
- ``source_refs`` field declaration doesn't tighten the otherwise-lenient
  ``additionalProperties: true`` on nodes
- graph still validates under the evidence-arm strict-mode stripper
  (Wave 11 evidence changes are NOT landed here; strict mode must still
  accept all 8 known rules unchanged)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
GRAPH_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "concept_graph_semantic.schema.json"


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _build_validator():
    _require_jsonschema()
    from jsonschema import Draft7Validator, RefResolver

    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    store: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    return Draft7Validator(schema, resolver=resolver)


def _base_graph(nodes: List[Dict[str, Any]] = None, edges: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-04-20T00:00:00Z",
        "nodes": nodes or [],
        "edges": edges or [],
    }


def _base_node(**overrides: Any) -> Dict[str, Any]:
    base = {
        "id": "cognitive-load",
        "label": "Cognitive Load",
        "frequency": 5,
    }
    base.update(overrides)
    return base


def _valid_ref(**overrides: Any) -> Dict[str, Any]:
    base = {
        "sourceId": "dart:science_of_learning#s3_c0",
        "role": "primary",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- #
# Meta: schema remains valid
# --------------------------------------------------------------------- #


def test_schema_remains_valid_draft_07():
    _require_jsonschema()
    from jsonschema import Draft7Validator

    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)


def test_node_source_refs_property_declared():
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    node_props = schema["properties"]["nodes"]["items"]["properties"]
    assert "source_refs" in node_props
    prop = node_props["source_refs"]
    assert prop["type"] == "array"
    assert "$ref" in prop["items"]
    assert prop["items"]["$ref"].endswith("source_reference.schema.json")


def test_evidence_arms_unchanged_no_wave11_drift():
    """Wave 10 must NOT touch the 8 evidence arm sub-schemas (that's Wave
    11). Lock down the shapes so any accidental edit is caught."""
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    defs = schema["$defs"]
    # The 8 arms' required fields must match the Wave 6 discriminator.
    expected_required = {
        "IsAEvidence": {"chunk_id", "term", "definition_excerpt", "pattern"},
        "PrerequisiteEvidence": {
            "target_first_lo", "target_first_lo_position",
            "source_first_lo", "source_first_lo_position",
        },
        "RelatedEvidence": {"cooccurrence_weight", "threshold"},
        "AssessesEvidence": {"question_id", "objective_id"},
        "ExemplifiesEvidence": {"chunk_id", "concept_slug", "content_type"},
        "MisconceptionOfEvidence": {"misconception_id", "concept_id"},
        "DerivedFromObjectiveEvidence": {"chunk_id", "objective_id"},
        "DefinedByEvidence": {
            "chunk_id", "concept_slug", "first_mention_position",
        },
    }
    for arm, required in expected_required.items():
        assert set(defs[arm]["required"]) == required, (
            f"{arm} required fields drifted — Wave 11 change leaked into Wave 10?"
        )
        assert defs[arm]["additionalProperties"] is False, (
            f"{arm} lost strictness"
        )


# --------------------------------------------------------------------- #
# Positive: nodes with/without source_refs validate
# --------------------------------------------------------------------- #


def test_legacy_node_without_source_refs_validates():
    """Pre-Wave-9 corpora → nodes lack source_refs → still valid."""
    validator = _build_validator()
    graph = _base_graph(nodes=[_base_node()])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]


def test_node_with_single_source_ref_validates():
    validator = _build_validator()
    node = _base_node(source_refs=[_valid_ref()])
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]


def test_node_with_multiple_source_refs_validates():
    """Multiple refs per node (from a chunk that merged 2+ sections)."""
    validator = _build_validator()
    refs = [
        _valid_ref(sourceId="dart:slug#s1_c0", role="primary"),
        _valid_ref(sourceId="dart:slug#s2_c0", role="contributing"),
    ]
    node = _base_node(source_refs=refs)
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]


def test_node_with_empty_source_refs_validates():
    validator = _build_validator()
    node = _base_node(source_refs=[])
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]


def test_node_with_source_refs_and_occurrences_validates():
    """source_refs[] coexists with occurrences[] (Wave 5)."""
    validator = _build_validator()
    node = _base_node(
        occurrences=["course_chunk_00001", "course_chunk_00002"],
        source_refs=[_valid_ref()],
    )
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]


def test_node_with_source_refs_and_course_id_validates():
    """Scoped concept IDs (Worker O) stay compatible with source_refs."""
    validator = _build_validator()
    node = _base_node(
        id="INT_101:cognitive-load",
        course_id="INT_101",
        source_refs=[_valid_ref()],
    )
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]


# --------------------------------------------------------------------- #
# Negative: malformed refs rejected
# --------------------------------------------------------------------- #


def test_node_with_ref_missing_sourceId_rejected():
    validator = _build_validator()
    node = _base_node(source_refs=[{"role": "primary"}])
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors, "Missing sourceId should be rejected"


def test_node_with_ref_missing_role_rejected():
    validator = _build_validator()
    node = _base_node(source_refs=[{"sourceId": "dart:slug#s0_c0"}])
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors, "Missing role should be rejected"


@pytest.mark.parametrize(
    "bad_source_id",
    [
        "",
        "no_dart_prefix",
        "dart:BAD_UPPER#s0_c0",
        "dart:slug#",
        "dart:slug#SHOUTY",
    ],
)
def test_node_with_ref_malformed_source_id_rejected(bad_source_id):
    validator = _build_validator()
    node = _base_node(
        source_refs=[{"sourceId": bad_source_id, "role": "primary"}]
    )
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors, f"Malformed sourceId {bad_source_id!r} should fail"


@pytest.mark.parametrize("bad_role", ["", "SUPPORTING", "main", "Primary"])
def test_node_with_ref_bad_role_rejected(bad_role):
    validator = _build_validator()
    node = _base_node(
        source_refs=[{"sourceId": "dart:slug#s0_c0", "role": bad_role}]
    )
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors, f"Bad role {bad_role!r} should fail"


def test_node_with_ref_bad_extractor_rejected():
    validator = _build_validator()
    node = _base_node(
        source_refs=[_valid_ref(extractor="tesseract")]
    )
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors, "Unknown extractor should be rejected"


def test_node_source_refs_always_array():
    """Scalar (non-array) source_refs rejected."""
    validator = _build_validator()
    node = _base_node(source_refs=_valid_ref())
    graph = _base_graph(nodes=[node])
    errors = list(validator.iter_errors(graph))
    assert errors, "Scalar source_refs must be rejected"


# --------------------------------------------------------------------- #
# Strict-mode evidence validator compatibility
# --------------------------------------------------------------------- #


def test_strict_evidence_mode_still_accepts_wave6_shapes():
    """Wave 10 node extension must NOT break Wave 6 strict-evidence mode.

    The lib.validators.evidence.get_schema(strict=True) path strips the
    FallbackProvenance arm. Validating a graph with only wave-6 evidence
    shapes and a wave-10 node source_refs must still pass — but only when
    the node.source_refs ``$ref`` is resolvable. Because the strict-schema
    mixes remote ($ref to source_reference.schema.json) and local
    (#/$defs/*Provenance) references under a single base URI, the simplest
    route is to validate against the strict schema with ``jsonschema.validate``
    (the pattern already used by lib/tests/test_evidence_discriminator.py).
    That path uses a default resolver which can't fetch the remote ref,
    so we validate the edge-level strictness using a graph whose node has
    no source_refs — purely exercising the evidence strict path — and
    separately validate the source_refs shape through the main builder.
    """
    jsonschema = pytest.importorskip("jsonschema")
    import importlib
    ev_mod = importlib.import_module("lib.validators.evidence")
    strict_schema = ev_mod.get_schema(strict=True)

    # (a) Graph with a node carrying source_refs but no edges — exercises
    # the main-schema validator (with resolver) that knows about the
    # source_reference.schema.json remote ref.
    validator = _build_validator()
    graph_no_edges = _base_graph(
        nodes=[_base_node(source_refs=[_valid_ref()])]
    )
    errors_a = list(validator.iter_errors(graph_no_edges))
    assert errors_a == [], [e.message for e in errors_a]

    # (b) Graph with a wave-6 edge + a node without source_refs —
    # exercises strict-evidence mode via jsonschema.validate (the same
    # path lib/tests/test_evidence_discriminator.py uses).
    graph_no_refs = _base_graph(
        nodes=[_base_node()],
        edges=[
            {
                "source": "a",
                "target": "b",
                "type": "is-a",
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "rule_version": 1,
                    "evidence": {
                        "chunk_id": "c_00001",
                        "term": "bar",
                        "definition_excerpt": "is a type of",
                        "pattern": r"is a",
                    },
                },
            }
        ],
    )
    # Should not raise.
    jsonschema.validate(instance=graph_no_refs, schema=strict_schema)
