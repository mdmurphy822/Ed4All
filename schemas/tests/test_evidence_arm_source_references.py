"""Wave 11 — evidence-arm source_references[] schema extension.

Wave 11 extends five chunk-anchored evidence arms with an optional
``source_references[]`` array (items ``$ref``
``source_reference.schema.json``):

- ``IsAEvidence``
- ``ExemplifiesEvidence``
- ``DerivedFromObjectiveEvidence``
- ``DefinedByEvidence``
- ``AssessesEvidence`` (complements existing optional ``source_chunk_id``)

Three abstract arms are deliberately NOT extended (P4 deferral):

- ``PrerequisiteEvidence``
- ``RelatedEvidence``
- ``MisconceptionOfEvidence``

And ``FallbackProvenance`` is untouched by design.

Contract locked by this suite:

- Each of the 5 arms accepts evidence with + without ``source_references``.
- Malformed refs (bad sourceId, bad role, non-array, etc.) rejected.
- The 3 abstract arms still reject the new field under ``additionalProperties: false``.
- Evidence-arm strict-mode (``lib.validators.evidence.get_schema(strict=True)``)
  still passes for Wave 6 shapes (no regression from the Wave 11 property add).
- Arm ``required`` / ``additionalProperties:false`` discipline unchanged for
  all 8 modeled arms.
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
    from jsonschema import Draft7Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7

    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)

    resources = []
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            # Force draft-07 specification so both the main graph schema
            # (draft-07) and the remote source_reference.schema.json
            # (draft-2020-12) interop under a single validator — the
            # oneOf descent into evidence arms needs a consistent resolver
            # scope or it stacks the remote URI and fails to resolve sibling
            # local #/$defs/... refs on subsequent arm checks.
            resources.append((sid, Resource(contents=s, specification=DRAFT7)))
    registry = Registry().with_resources(resources)
    return Draft7Validator(schema, registry=registry)


def _base_graph(edges: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-04-20T00:00:00Z",
        "nodes": [],
        "edges": edges or [],
    }


def _valid_ref(**overrides: Any) -> Dict[str, Any]:
    base = {
        "sourceId": "dart:science_of_learning#s3_c0",
        "role": "primary",
    }
    base.update(overrides)
    return base


def _is_a_edge(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "a",
        "target": "b",
        "type": "is-a",
        "provenance": {
            "rule": "is_a_from_key_terms",
            "rule_version": 2,
            "evidence": evidence,
        },
    }


def _exemplifies_edge(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "chunk_0001",
        "target": "concept-x",
        "type": "exemplifies",
        "provenance": {
            "rule": "exemplifies_from_example_chunks",
            "rule_version": 2,
            "evidence": evidence,
        },
    }


def _derived_edge(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "chunk_0001",
        "target": "to-01",
        "type": "derived-from-objective",
        "provenance": {
            "rule": "derived_from_lo_ref",
            "rule_version": 2,
            "evidence": evidence,
        },
    }


def _defined_edge(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "concept-x",
        "target": "chunk_0001",
        "type": "defined-by",
        "provenance": {
            "rule": "defined_by_from_first_mention",
            "rule_version": 2,
            "evidence": evidence,
        },
    }


def _assesses_edge(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "q-001",
        "target": "to-01",
        "type": "assesses",
        "provenance": {
            "rule": "assesses_from_question_lo",
            "rule_version": 2,
            "evidence": evidence,
        },
    }


# --------------------------------------------------------------------- #
# Meta: schema remains valid
# --------------------------------------------------------------------- #


def test_schema_remains_valid_draft_07():
    _require_jsonschema()
    from jsonschema import Draft7Validator

    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)


def test_five_chunk_anchored_arms_declare_source_references():
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    defs = schema["$defs"]
    for arm in (
        "IsAEvidence",
        "ExemplifiesEvidence",
        "DerivedFromObjectiveEvidence",
        "DefinedByEvidence",
        "AssessesEvidence",
    ):
        props = defs[arm]["properties"]
        assert "source_references" in props, f"{arm} missing source_references"
        srefs = props["source_references"]
        assert srefs["type"] == "array"
        assert "$ref" in srefs["items"]
        assert srefs["items"]["$ref"].endswith("source_reference.schema.json")


def test_three_abstract_arms_do_not_declare_source_references():
    """P4 deferral — abstract arms must NOT carry source_references yet."""
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    defs = schema["$defs"]
    for arm in ("PrerequisiteEvidence", "RelatedEvidence", "MisconceptionOfEvidence"):
        props = defs[arm]["properties"]
        assert "source_references" not in props, (
            f"{arm} must not carry source_references yet (P4 deferred)"
        )


def test_fallback_provenance_unchanged():
    """FallbackProvenance is lenient — no schema change in Wave 11."""
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    fallback = schema["$defs"]["FallbackProvenance"]
    # Evidence remains the catch-all object (unchanged from Wave 6).
    assert fallback["properties"]["evidence"]["type"] == "object"
    assert fallback["properties"]["evidence"].get("additionalProperties") is True


def test_all_arms_retain_strict_additional_properties():
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    defs = schema["$defs"]
    for arm in (
        "IsAEvidence",
        "PrerequisiteEvidence",
        "RelatedEvidence",
        "AssessesEvidence",
        "ExemplifiesEvidence",
        "MisconceptionOfEvidence",
        "DerivedFromObjectiveEvidence",
        "DefinedByEvidence",
    ):
        assert defs[arm]["additionalProperties"] is False, (
            f"{arm} must keep additionalProperties: false"
        )


def test_arm_required_fields_unchanged():
    """Wave 11 adds optional source_references — required sets must be untouched."""
    with open(GRAPH_SCHEMA_PATH) as f:
        schema = json.load(f)
    defs = schema["$defs"]
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
            f"{arm} required drifted"
        )


# --------------------------------------------------------------------- #
# Positive: each chunk-anchored arm accepts with + without refs
# --------------------------------------------------------------------- #


def test_is_a_without_source_references_validates():
    validator = _build_validator()
    edge = _is_a_edge({
        "chunk_id": "c1",
        "term": "entry-point",
        "definition_excerpt": "is a type of door",
        "pattern": "is a",
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_is_a_with_source_references_validates():
    validator = _build_validator()
    edge = _is_a_edge({
        "chunk_id": "c1",
        "term": "entry-point",
        "definition_excerpt": "is a type of door",
        "pattern": "is a",
        "source_references": [_valid_ref()],
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_exemplifies_without_source_references_validates():
    validator = _build_validator()
    edge = _exemplifies_edge({
        "chunk_id": "c1",
        "concept_slug": "x",
        "content_type": "chunk_type",
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_exemplifies_with_source_references_validates():
    validator = _build_validator()
    edge = _exemplifies_edge({
        "chunk_id": "c1",
        "concept_slug": "x",
        "content_type": "chunk_type",
        "source_references": [_valid_ref(role="contributing")],
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_derived_without_source_references_validates():
    validator = _build_validator()
    edge = _derived_edge({"chunk_id": "c1", "objective_id": "to-01"})
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_derived_with_source_references_validates():
    validator = _build_validator()
    edge = _derived_edge({
        "chunk_id": "c1",
        "objective_id": "to-01",
        "source_references": [_valid_ref()],
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_defined_without_source_references_validates():
    validator = _build_validator()
    edge = _defined_edge({
        "chunk_id": "c1",
        "concept_slug": "x",
        "first_mention_position": 0,
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_defined_with_source_references_validates():
    validator = _build_validator()
    edge = _defined_edge({
        "chunk_id": "c1",
        "concept_slug": "x",
        "first_mention_position": 0,
        "source_references": [_valid_ref()],
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_assesses_without_source_references_validates():
    validator = _build_validator()
    edge = _assesses_edge({"question_id": "q-001", "objective_id": "to-01"})
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_assesses_with_source_references_validates():
    validator = _build_validator()
    edge = _assesses_edge({
        "question_id": "q-001",
        "objective_id": "to-01",
        "source_chunk_id": "c1",
        "source_references": [_valid_ref()],
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_multiple_refs_per_arm_validates():
    """A chunk that merged multiple sections carries multiple refs."""
    validator = _build_validator()
    refs = [
        _valid_ref(sourceId="dart:slug#s1_c0", role="primary"),
        _valid_ref(sourceId="dart:slug#s2_c0", role="contributing"),
        _valid_ref(sourceId="dart:slug#s3_c0", role="corroborating"),
    ]
    edge = _is_a_edge({
        "chunk_id": "c1",
        "term": "x",
        "definition_excerpt": "y",
        "pattern": "p",
        "source_references": refs,
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


def test_empty_source_references_array_validates():
    """Empty array is still a valid array (absence is also valid)."""
    validator = _build_validator()
    edge = _is_a_edge({
        "chunk_id": "c1",
        "term": "x",
        "definition_excerpt": "y",
        "pattern": "p",
        "source_references": [],
    })
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors == [], [e.message for e in errors]


# --------------------------------------------------------------------- #
# Negative: malformed refs rejected across all 5 arms
# --------------------------------------------------------------------- #


_CHUNK_ANCHORED_BUILDERS = [
    ("is_a", _is_a_edge, {
        "chunk_id": "c1",
        "term": "x",
        "definition_excerpt": "y",
        "pattern": "p",
    }),
    ("exemplifies", _exemplifies_edge, {
        "chunk_id": "c1",
        "concept_slug": "x",
        "content_type": "chunk_type",
    }),
    ("derived", _derived_edge, {"chunk_id": "c1", "objective_id": "to-01"}),
    ("defined", _defined_edge, {
        "chunk_id": "c1",
        "concept_slug": "x",
        "first_mention_position": 0,
    }),
    ("assesses", _assesses_edge, {
        "question_id": "q-001",
        "objective_id": "to-01",
    }),
]


@pytest.mark.parametrize("name,builder,base", _CHUNK_ANCHORED_BUILDERS)
def test_each_arm_rejects_ref_missing_source_id(name, builder, base):
    validator = _build_validator()
    ev = dict(base)
    ev["source_references"] = [{"role": "primary"}]
    errors = list(validator.iter_errors(_base_graph([builder(ev)])))
    assert errors, f"{name} should reject ref missing sourceId"


@pytest.mark.parametrize("name,builder,base", _CHUNK_ANCHORED_BUILDERS)
def test_each_arm_rejects_ref_missing_role(name, builder, base):
    validator = _build_validator()
    ev = dict(base)
    ev["source_references"] = [{"sourceId": "dart:slug#s0_c0"}]
    errors = list(validator.iter_errors(_base_graph([builder(ev)])))
    assert errors, f"{name} should reject ref missing role"


@pytest.mark.parametrize("name,builder,base", _CHUNK_ANCHORED_BUILDERS)
def test_each_arm_rejects_scalar_source_references(name, builder, base):
    validator = _build_validator()
    ev = dict(base)
    ev["source_references"] = _valid_ref()
    errors = list(validator.iter_errors(_base_graph([builder(ev)])))
    assert errors, f"{name} should reject scalar source_references"


@pytest.mark.parametrize("name,builder,base", _CHUNK_ANCHORED_BUILDERS)
def test_each_arm_rejects_bad_role(name, builder, base):
    validator = _build_validator()
    ev = dict(base)
    ev["source_references"] = [{"sourceId": "dart:slug#s0_c0", "role": "SUPPORTING"}]
    errors = list(validator.iter_errors(_base_graph([builder(ev)])))
    assert errors, f"{name} should reject bad role"


@pytest.mark.parametrize("name,builder,base", _CHUNK_ANCHORED_BUILDERS)
def test_each_arm_rejects_malformed_source_id(name, builder, base):
    validator = _build_validator()
    ev = dict(base)
    ev["source_references"] = [{"sourceId": "no-dart-prefix", "role": "primary"}]
    errors = list(validator.iter_errors(_base_graph([builder(ev)])))
    assert errors, f"{name} should reject malformed sourceId"


# --------------------------------------------------------------------- #
# Negative: abstract arms still reject the new field
# --------------------------------------------------------------------- #


def test_prerequisite_evidence_rejects_source_references():
    validator = _build_validator()
    edge = {
        "source": "a",
        "target": "b",
        "type": "prerequisite",
        "provenance": {
            "rule": "prerequisite_from_lo_order",
            "rule_version": 1,
            "evidence": {
                "target_first_lo": "to-02",
                "target_first_lo_position": 2,
                "source_first_lo": "to-01",
                "source_first_lo_position": 1,
                "source_references": [_valid_ref()],
            },
        },
    }
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors, "PrerequisiteEvidence must reject source_references (P4 deferral)"


def test_related_evidence_rejects_source_references():
    validator = _build_validator()
    edge = {
        "source": "a",
        "target": "b",
        "type": "related-to",
        "provenance": {
            "rule": "related_from_cooccurrence",
            "rule_version": 1,
            "evidence": {
                "cooccurrence_weight": 5,
                "threshold": 3,
                "source_references": [_valid_ref()],
            },
        },
    }
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors, "RelatedEvidence must reject source_references (P4 deferral)"


def test_misconception_of_evidence_rejects_source_references():
    validator = _build_validator()
    edge = {
        "source": "mc_abcd1234abcd1234",
        "target": "concept-x",
        "type": "misconception-of",
        "provenance": {
            "rule": "misconception_of_from_misconception_ref",
            "rule_version": 1,
            "evidence": {
                "misconception_id": "mc_abcd1234abcd1234",
                "concept_id": "concept-x",
                "source_references": [_valid_ref()],
            },
        },
    }
    errors = list(validator.iter_errors(_base_graph([edge])))
    assert errors, "MisconceptionOfEvidence must reject source_references (P4 deferral)"


# --------------------------------------------------------------------- #
# Evidence strict-mode compatibility
# --------------------------------------------------------------------- #


def test_strict_evidence_mode_exposes_wave11_property_decls():
    """Strict schema still carries source_references on the 5 chunk-anchored
    arms (the strict pass only strips FallbackProvenance from the oneOf).
    """
    _require_jsonschema()
    import importlib

    ev_mod = importlib.import_module("lib.validators.evidence")
    strict_schema = ev_mod.get_schema(strict=True)

    defs = strict_schema["$defs"]
    for arm in (
        "IsAEvidence",
        "ExemplifiesEvidence",
        "DerivedFromObjectiveEvidence",
        "DefinedByEvidence",
        "AssessesEvidence",
    ):
        assert "source_references" in defs[arm]["properties"], (
            f"{arm} lost source_references property after strict-mode stripping"
        )


def test_strict_evidence_mode_still_accepts_wave6_shapes():
    """Regression: no source_references in evidence still passes under strict mode."""
    _require_jsonschema()
    import importlib
    import jsonschema

    ev_mod = importlib.import_module("lib.validators.evidence")
    strict_schema = ev_mod.get_schema(strict=True)

    graph = _base_graph([
        _is_a_edge({
            "chunk_id": "c1",
            "term": "bar",
            "definition_excerpt": "is a type of",
            "pattern": "is a",
        }),
    ])
    jsonschema.validate(instance=graph, schema=strict_schema)


def test_lenient_evidence_mode_accepts_wave11_shape_via_main_validator():
    """Wave 11 shape (evidence with source_references) validates through the
    main validator that knows about the remote source_reference.schema.json
    $ref. This exercises the full validation path including the remote ref."""
    validator = _build_validator()
    graph = _base_graph([
        _is_a_edge({
            "chunk_id": "c1",
            "term": "x",
            "definition_excerpt": "y",
            "pattern": "p",
            "source_references": [_valid_ref()],
        }),
    ])
    errors = list(validator.iter_errors(graph))
    assert errors == [], [e.message for e in errors]
