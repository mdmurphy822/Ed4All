"""Tests for the asserted/inferred edge_kind classification.

GPT-feedback (12 May 2026) item 1. Pins:

1. Every inference rule exported by ``Trainforge/rag/inference_rules/``
   (plus the ``llm_typed_edge`` pseudo-rule) is classified in the
   canonical ``lib.ontology.edge_kind`` registry. A new rule that
   lands without a registry entry fails this test loudly.
2. Materializer rules (rules that promote an explicit upstream
   pointer) classify as ``"asserted"``.
3. Heuristic / statistical / co-occurrence / LLM rules classify as
   ``"inferred"``.
4. Unknown rule names return ``None`` (back-compat for legacy graphs
   and not-yet-registered future rules).
5. ``build_semantic_graph`` stamps ``edge_kind`` on every emitted
   edge (both a materializer rule path and an inference rule path).
6. Legacy edges without ``edge_kind`` still validate against the
   schema (additive contract).
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.ontology.edge_kind import (
    EDGE_KIND_ASSERTED,
    EDGE_KIND_INFERRED,
    edge_kind_for_rule,
    known_rule_names,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)
SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


# ---------------------------------------------------------------------- #
# 1. Coverage: every Trainforge inference rule appears in the registry
# ---------------------------------------------------------------------- #


def _discover_inference_rule_names() -> List[str]:
    """Discover every ``RULE_NAME`` constant exported by submodules of
    ``Trainforge.rag.inference_rules``. Excludes the ``__init__`` re-export
    surface; only walks per-rule modules.
    """
    import Trainforge.rag.inference_rules as inference_rules

    names: List[str] = []
    for mod_info in pkgutil.iter_modules(inference_rules.__path__):
        if mod_info.ispkg:
            continue
        full = f"{inference_rules.__name__}.{mod_info.name}"
        mod = importlib.import_module(full)
        rule_name = getattr(mod, "RULE_NAME", None)
        if isinstance(rule_name, str) and rule_name:
            names.append(rule_name)
    return sorted(names)


def test_every_rule_module_is_classified():
    """Each ``RULE_NAME`` exported by ``Trainforge/rag/inference_rules/``
    must have an entry in the canonical edge_kind registry. The
    ``llm_typed_edge`` pseudo-rule (synthesized by
    ``typed_edge_inference._llm_escalate`` for LLM-proposed edges) is
    not a module — it's asserted separately below.
    """
    discovered = set(_discover_inference_rule_names())
    classified = known_rule_names()
    missing = discovered - classified
    assert not missing, (
        f"Inference rules without an edge_kind registry entry: "
        f"{sorted(missing)}. Add them to lib/ontology/edge_kind.py."
    )


def test_llm_typed_edge_pseudo_rule_is_classified():
    """``llm_typed_edge`` is synthesized by the orchestrator (not a
    module under inference_rules/); pin it as inferred so LLM-
    escalated edges aren't silently unclassified.
    """
    assert edge_kind_for_rule("llm_typed_edge") == EDGE_KIND_INFERRED


# ---------------------------------------------------------------------- #
# 2 & 3. Asserted / inferred classification per rule
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rule_name",
    [
        "assesses_from_question_lo",
        "derived_from_lo_ref",
        "misconception_of_from_misconception_ref",
        "targets_concept_from_lo",
        "defined_by_from_first_mention",
    ],
)
def test_materializer_rules_classify_as_asserted(rule_name: str):
    assert edge_kind_for_rule(rule_name) == EDGE_KIND_ASSERTED


@pytest.mark.parametrize(
    "rule_name",
    [
        "is_a_from_key_terms",
        "exemplifies_from_example_chunks",
        "prerequisite_from_lo_order",
        "related_from_cooccurrence",
        "llm_typed_edge",
    ],
)
def test_inference_rules_classify_as_inferred(rule_name: str):
    assert edge_kind_for_rule(rule_name) == EDGE_KIND_INFERRED


# ---------------------------------------------------------------------- #
# 4. Unknown rule name -> None (legacy / future-rule back-compat)
# ---------------------------------------------------------------------- #


def test_unknown_rule_returns_none():
    assert edge_kind_for_rule("future_rule_that_does_not_exist_yet") is None


def test_non_string_input_returns_none():
    """Defensive: graph-build can hand a None / int / dict if the
    upstream provenance is malformed; the helper should not crash.
    """
    assert edge_kind_for_rule(None) is None  # type: ignore[arg-type]
    assert edge_kind_for_rule(123) is None  # type: ignore[arg-type]
    assert edge_kind_for_rule({}) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# 5. End-to-end: build_semantic_graph stamps edge_kind on every edge
# ---------------------------------------------------------------------- #


def test_build_semantic_graph_stamps_edge_kind_on_emitted_edges():
    """Drive both a materializer rule (derived_from_lo_ref via the
    chunk's learning_outcome_refs) and an inference rule
    (related_from_cooccurrence via concept_graph weight) through the
    real graph-build, then assert every emitted edge carries
    edge_kind matching its provenance.rule classification.
    """
    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    chunks = [
        {
            "id": "chunk_001",
            "concept_tags": ["alpha", "beta"],
            "learning_outcome_refs": ["to-01"],
            "key_terms": [],
        },
        {
            "id": "chunk_002",
            "concept_tags": ["alpha", "beta"],
            "learning_outcome_refs": ["to-01"],
            "key_terms": [],
        },
        {
            "id": "chunk_003",
            "concept_tags": ["alpha", "beta"],
            "learning_outcome_refs": ["to-01"],
            "key_terms": [],
        },
    ]
    course = {"learning_outcomes": [{"id": "to-01", "statement": "Alpha"}]}
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {"id": "alpha", "label": "alpha", "frequency": 3,
             "occurrences": ["chunk_001", "chunk_002", "chunk_003"]},
            {"id": "beta", "label": "beta", "frequency": 3,
             "occurrences": ["chunk_001", "chunk_002", "chunk_003"]},
        ],
        # Co-occurrence weight = 3 hits the related-to threshold.
        "edges": [{"source": "alpha", "target": "beta", "weight": 3}],
    }

    fixed_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    graph = build_semantic_graph(
        chunks=chunks,
        course=course,
        concept_graph=concept_graph,
        now=fixed_now,
    )

    edges = graph.get("edges") or []
    assert edges, "expected at least one edge from the fixture"

    seen_kinds: Dict[str, set] = {EDGE_KIND_ASSERTED: set(), EDGE_KIND_INFERRED: set()}
    for edge in edges:
        rule = edge["provenance"]["rule"]
        expected = edge_kind_for_rule(rule)
        # Every rule in this fixture is registered, so every edge MUST
        # carry edge_kind. The schema-additive contract preserves None
        # for unregistered rules; that branch is covered by the legacy-
        # back-compat test below.
        assert "edge_kind" in edge, (
            f"edge from rule={rule!r} missing edge_kind stamp; "
            f"build-time stamp regressed."
        )
        assert edge["edge_kind"] == expected, (
            f"edge_kind={edge['edge_kind']!r} disagrees with registry "
            f"classification {expected!r} for rule={rule!r}."
        )
        seen_kinds[expected].add(rule)

    # Fixture is designed to exercise BOTH classifications so a
    # regression that stamps only one kind fails here.
    assert seen_kinds[EDGE_KIND_ASSERTED], (
        "fixture should drive at least one asserted edge "
        "(derived_from_lo_ref via learning_outcome_refs)."
    )
    assert seen_kinds[EDGE_KIND_INFERRED], (
        "fixture should drive at least one inferred edge "
        "(related_from_cooccurrence above threshold)."
    )


# ---------------------------------------------------------------------- #
# 6. Legacy back-compat: edge without edge_kind still validates
# ---------------------------------------------------------------------- #


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _build_schema_validator():
    _require_jsonschema()
    from jsonschema import Draft7Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7

    with open(SCHEMA_PATH) as f:
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
            resources.append((sid, Resource(contents=s, specification=DRAFT7)))
    registry = Registry().with_resources(resources)
    return Draft7Validator(schema, registry=registry)


def _base_graph(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-05-12T00:00:00Z",
        "nodes": [],
        "edges": edges,
    }


def test_legacy_edge_without_edge_kind_still_validates():
    """An edge emitted before this wave (no edge_kind field) must
    validate against the post-wave schema — additive-contract pin.
    """
    validator = _build_schema_validator()

    legacy_edge = {
        "source": "alpha",
        "target": "beta",
        "type": "related-to",
        "provenance": {
            "rule": "related_from_cooccurrence",
            "rule_version": 1,
            "evidence": {"cooccurrence_weight": 5, "threshold": 3},
        },
    }
    graph = _base_graph([legacy_edge])
    errors = list(validator.iter_errors(graph))
    assert not errors, (
        "Legacy edge without edge_kind should validate; got: "
        f"{[e.message for e in errors]}"
    )


def test_new_edge_with_edge_kind_validates():
    """An edge emitted post-wave with edge_kind populated must validate."""
    validator = _build_schema_validator()

    new_edge = {
        "source": "alpha",
        "target": "beta",
        "type": "related-to",
        "edge_kind": "inferred",
        "provenance": {
            "rule": "related_from_cooccurrence",
            "rule_version": 1,
            "evidence": {"cooccurrence_weight": 5, "threshold": 3},
        },
    }
    graph = _base_graph([new_edge])
    errors = list(validator.iter_errors(graph))
    assert not errors, (
        "New edge with edge_kind should validate; got: "
        f"{[e.message for e in errors]}"
    )


def test_invalid_edge_kind_value_rejected():
    """edge_kind is enum-constrained — only 'asserted' / 'inferred'.
    Pins that a typo / out-of-band value fails closed.
    """
    validator = _build_schema_validator()

    bad_edge = {
        "source": "alpha",
        "target": "beta",
        "type": "related-to",
        "edge_kind": "materialized",  # not in enum
        "provenance": {
            "rule": "related_from_cooccurrence",
            "rule_version": 1,
            "evidence": {"cooccurrence_weight": 5, "threshold": 3},
        },
    }
    graph = _base_graph([bad_edge])
    errors = list(validator.iter_errors(graph))
    assert errors, "edge_kind='materialized' should be rejected (not in enum)"
