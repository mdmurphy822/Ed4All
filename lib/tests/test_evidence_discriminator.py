"""Tests for the REC-PRV-02 per-rule evidence discriminator.

Exercises the ``oneOf`` discriminator on ``edges[].provenance`` in
``schemas/knowledge/concept_graph_semantic.schema.json``. Each of the 8
known rules has a matching ``{Rule}Provenance`` arm that binds
``rule = {name}`` to a specific evidence ``$def``; a 9th ``FallbackProvenance``
arm matches any edge whose rule isn't one of the 8 (lenient backward-compat).

Two modes are tested:

1. **Lenient (default).** Ships with FallbackProvenance — unknown rules pass
   with arbitrary evidence. Known rules still enforce evidence shape.
2. **Strict** (``TRAINFORGE_STRICT_EVIDENCE=true`` or
   ``get_schema(strict=True)``). FallbackProvenance is stripped — unknown
   rules fail.

Consumed directly by developers during KG-publish validation; not (yet)
wired into any existing callsite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.validators.evidence import SCHEMA_PATH, STRICT_ENV_VAR, get_schema

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

FIXED_NOW = "2026-04-19T00:00:00+00:00"


def _artifact(edges: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap a list of edges into a minimal valid concept-graph artifact."""
    return {
        "kind": "concept_semantic",
        "generated_at": FIXED_NOW,
        "nodes": [],
        "edges": edges,
    }


def _edge(edge_type: str, rule: str, evidence: Dict[str, Any] | None) -> Dict[str, Any]:
    """Build a single edge dict with optional evidence."""
    provenance: Dict[str, Any] = {"rule": rule, "rule_version": 1}
    if evidence is not None:
        provenance["evidence"] = evidence
    return {
        "source": "src",
        "target": "tgt",
        "type": edge_type,
        "confidence": 0.5,
        "provenance": provenance,
    }


# Per-rule (edge_type, rule_name, correct_evidence, wrong_evidence) quads.
# wrong_evidence violates the shape (missing required field, wrong type, or
# stray field under additionalProperties:false).
RULE_SPECS = [
    (
        "is-a",
        "is_a_from_key_terms",
        {
            "chunk_id": "chunk_1",
            "term": "widget",
            "definition_excerpt": "A widget is a type of gizmo.",
            "pattern": r"\bis\s+(?:a|an)\s+(?:type)\s+of\s+([^.,;:\n]+)",
        },
        # Missing required 'pattern'
        {
            "chunk_id": "chunk_1",
            "term": "widget",
            "definition_excerpt": "A widget is a type of gizmo.",
        },
    ),
    (
        "prerequisite",
        "prerequisite_from_lo_order",
        {
            "target_first_lo": "to-01",
            "target_first_lo_position": 0,
            "source_first_lo": "to-02",
            "source_first_lo_position": 1,
        },
        # Wrong type: position as string
        {
            "target_first_lo": "to-01",
            "target_first_lo_position": "zero",
            "source_first_lo": "to-02",
            "source_first_lo_position": 1,
        },
    ),
    (
        "related-to",
        "related_from_cooccurrence",
        {"cooccurrence_weight": 5, "threshold": 3},
        # Stray field not allowed under additionalProperties:false
        {"cooccurrence_weight": 5, "threshold": 3, "bogus": "field"},
    ),
    (
        "assesses",
        "assesses_from_question_lo",
        {
            "question_id": "q1",
            "objective_id": "to-01",
            "source_chunk_id": "chunk_1",
        },
        # Missing required 'objective_id'
        {"question_id": "q1"},
    ),
    (
        "exemplifies",
        "exemplifies_from_example_chunks",
        {
            "chunk_id": "chunk_1",
            "concept_slug": "widget",
            "content_type": "chunk_type",
        },
        # Invalid enum value for content_type
        {
            "chunk_id": "chunk_1",
            "concept_slug": "widget",
            "content_type": "something_else",
        },
    ),
    (
        "misconception-of",
        "misconception_of_from_misconception_ref",
        {"misconception_id": "mc_" + "a" * 16, "concept_id": "widget"},
        # Missing required 'concept_id'
        {"misconception_id": "mc_" + "a" * 16},
    ),
    (
        "derived-from-objective",
        "derived_from_lo_ref",
        {"chunk_id": "chunk_1", "objective_id": "to-01"},
        # Wrong type: chunk_id as int
        {"chunk_id": 123, "objective_id": "to-01"},
    ),
    (
        "defined-by",
        "defined_by_from_first_mention",
        {"chunk_id": "chunk_1", "concept_slug": "widget", "first_mention_position": 0},
        # Missing required 'first_mention_position'
        {"chunk_id": "chunk_1", "concept_slug": "widget"},
    ),
]


# ---------------------------------------------------------------------------
# Sanity — loader + schema shape
# ---------------------------------------------------------------------------


def test_schema_path_exists():
    assert SCHEMA_PATH.exists()


def test_get_schema_returns_oneOf_with_nine_arms_default():
    schema = get_schema(strict=False)
    oneof = (
        schema["properties"]["edges"]["items"]["properties"]["provenance"]["oneOf"]
    )
    assert len(oneof) == 9
    refs = [arm["$ref"] for arm in oneof]
    # Last arm is the fallback; first 8 are specific.
    assert refs[-1].endswith("/FallbackProvenance")
    specific = {
        "#/$defs/IsAProvenance",
        "#/$defs/PrerequisiteProvenance",
        "#/$defs/RelatedProvenance",
        "#/$defs/AssessesProvenance",
        "#/$defs/ExemplifiesProvenance",
        "#/$defs/MisconceptionOfProvenance",
        "#/$defs/DerivedFromObjectiveProvenance",
        "#/$defs/DefinedByProvenance",
    }
    assert set(refs[:-1]) == specific


def test_get_schema_strict_drops_fallback_arm():
    schema = get_schema(strict=True)
    oneof = (
        schema["properties"]["edges"]["items"]["properties"]["provenance"]["oneOf"]
    )
    refs = [arm["$ref"] for arm in oneof]
    assert not any(r.endswith("/FallbackProvenance") for r in refs)
    assert len(oneof) == 8


def test_get_schema_env_var_toggles_strict(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, "true")
    # Clear the lru_cache so the env var is re-read -- actually the cache is
    # on _load_schema_raw, not get_schema; get_schema itself re-reads the env
    # on every call when strict=None.
    schema = get_schema()  # strict=None → reads env
    oneof = (
        schema["properties"]["edges"]["items"]["properties"]["provenance"]["oneOf"]
    )
    assert len(oneof) == 8
    monkeypatch.setenv(STRICT_ENV_VAR, "false")
    schema2 = get_schema()
    oneof2 = (
        schema2["properties"]["edges"]["items"]["properties"]["provenance"]["oneOf"]
    )
    assert len(oneof2) == 9


# ---------------------------------------------------------------------------
# Per-rule happy-path (8 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_type,rule,correct,_wrong", RULE_SPECS)
def test_per_rule_correct_evidence_validates_lenient(edge_type, rule, correct, _wrong):
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=False)
    artifact = _artifact([_edge(edge_type, rule, correct)])
    jsonschema.validate(instance=artifact, schema=schema)


@pytest.mark.parametrize("edge_type,rule,correct,_wrong", RULE_SPECS)
def test_per_rule_correct_evidence_validates_strict(edge_type, rule, correct, _wrong):
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=True)
    artifact = _artifact([_edge(edge_type, rule, correct)])
    jsonschema.validate(instance=artifact, schema=schema)


# ---------------------------------------------------------------------------
# Per-rule wrong-shape evidence rejection (8 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_type,rule,_correct,wrong", RULE_SPECS)
def test_per_rule_wrong_evidence_rejected_lenient(edge_type, rule, _correct, wrong):
    """Even in lenient mode, a KNOWN rule with wrong evidence shape fails.

    The specific arm rejects the wrong shape; the FallbackProvenance arm
    rejects because the rule is in its ``not: enum`` list. Zero arms match →
    oneOf fails. This is intentionally stricter than a pure "ignore unknown
    shapes" mode — we still catch typos and drift in known rules.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=False)
    artifact = _artifact([_edge(edge_type, rule, wrong)])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=artifact, schema=schema)


@pytest.mark.parametrize("edge_type,rule,_correct,wrong", RULE_SPECS)
def test_per_rule_wrong_evidence_rejected_strict(edge_type, rule, _correct, wrong):
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=True)
    artifact = _artifact([_edge(edge_type, rule, wrong)])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=artifact, schema=schema)


# ---------------------------------------------------------------------------
# Fallback behavior for unknown rules
# ---------------------------------------------------------------------------


def test_unknown_rule_with_any_evidence_passes_lenient():
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=False)
    edge = _edge(
        "related-to",
        "llm_typed_edge",  # not one of the 8 modeled rules
        {"some_future_field": 42, "nested": {"ok": True}},
    )
    artifact = _artifact([edge])
    jsonschema.validate(instance=artifact, schema=schema)


def test_unknown_rule_fails_strict():
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=True)
    edge = _edge(
        "related-to",
        "llm_typed_edge",
        {"some_future_field": 42},
    )
    artifact = _artifact([edge])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=artifact, schema=schema)


def test_unknown_rule_without_evidence_passes_lenient():
    """Legacy graphs may carry provenance without an evidence field at all."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=False)
    edge = _edge("related-to", "future_rule_v2", None)
    artifact = _artifact([edge])
    jsonschema.validate(instance=artifact, schema=schema)


# ---------------------------------------------------------------------------
# Known rule without evidence is OK — evidence is optional on specific arms
# ---------------------------------------------------------------------------


def test_known_rule_without_evidence_validates_lenient():
    """Matches the existing ``test_edge_enum_includes_new_types`` fixture
    pattern in ``Trainforge/tests/test_pedagogical_edges.py`` where edges
    carry provenance with no evidence field at all — a valid legacy shape."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=False)
    edge = _edge("derived-from-objective", "derived_from_lo_ref", None)
    artifact = _artifact([edge])
    jsonschema.validate(instance=artifact, schema=schema)


def test_known_rule_without_evidence_validates_strict():
    jsonschema = pytest.importorskip("jsonschema")
    schema = get_schema(strict=True)
    edge = _edge("derived-from-objective", "derived_from_lo_ref", None)
    artifact = _artifact([edge])
    jsonschema.validate(instance=artifact, schema=schema)


# ---------------------------------------------------------------------------
# Smoke: validate any existing LibV2 concept_graph_semantic.json files
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_libv2_semantic_graphs_validate_if_present():
    """Smoke-level check: any existing concept_graph_semantic.json in
    ``LibV2/courses/`` validates under the lenient discriminator.

    If no LibV2 corpus files exist in this worktree (e.g. fresh clone), the
    test skips gracefully — it's a regression fence, not a gating check.
    """
    jsonschema = pytest.importorskip("jsonschema")
    root = _repo_root()
    candidates = list(root.glob("LibV2/courses/*/graph/concept_graph_semantic.json"))
    if not candidates:
        pytest.skip("no LibV2 concept_graph_semantic.json found in worktree")
    schema = get_schema(strict=False)
    for path in candidates:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        jsonschema.validate(instance=data, schema=schema)
