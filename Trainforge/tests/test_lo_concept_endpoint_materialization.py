"""LO-authored ``targets-concept`` endpoints are materialized as nodes.

With ``TRAINFORGE_MERGE_DUPLICATE_CONCEPTS`` on, the merge pass dropped
``targets-concept`` edges whose target was an objectives ``key_concepts`` slug
that the corpus chunks never tagged (a phantom endpoint with no co-occurrence
node). Those same phantom targets also fired ``CONCEPT_GRAPH_ORPHAN_NODE``
warnings in the flagless build.

Fix (Step 1): ``typed_edge_inference._materialize_endpoint_nodes`` synthesizes a
provenance-flagged ``DomainConcept`` node (``node_provenance="lo_key_concept"``,
``frequency=0``) for every unresolved ``targets-concept`` target, so the edge
resolves and the merge / orphan stages can fold or segment the node instead of
dropping the edge.

Covers (mirrors the fixture pattern of ``test_targets_concept_edges.py``):

 1. key_concept slug absent from co-occurrence nodes → materialized node +
    zero dangling targets-concept edges.
 2. key_concept slug equal to an existing node id → no duplicate node, no
    node_provenance stamped on the existing node.
 3. Determinism: two pinned-``now`` builds are byte-identical.
 4. Non-targets-concept junk-slug endpoint stays non-materialized (the
    dangling-surfacing contract is preserved for other namespaces).
 5. The emitted graph validates against ``concept_graph_semantic.schema.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.rag.inference_rules.targets_concept_from_lo import (  # noqa: E402
    EDGE_TYPE,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures (same shape as test_targets_concept_edges.py)
# ---------------------------------------------------------------------- #


def _lo(lo_id: str, targets: list) -> dict:
    return {
        "id": lo_id,
        "statement": f"{lo_id} statement",
        "bloomLevel": "apply",
        "targetedConcepts": targets,
    }


def _target(concept: str, bloom: str = "apply") -> dict:
    return {"concept": concept, "bloomLevel": bloom}


_PINNED_NOW = datetime(2026, 4, 24, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------- #
# 1. Phantom key_concept target → materialized DomainConcept node
# ---------------------------------------------------------------------- #


def test_phantom_key_concept_target_materialized_as_domain_concept():
    """A key_concept slug with no co-occurrence node is materialized rather
    than left as a dangling targets-concept endpoint."""
    los = [_lo("TO-01", [_target("vector-stores")])]
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        objectives_metadata=los,
        now=_PINNED_NOW,
    )

    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    assert "vector-stores" in nodes_by_id
    node = nodes_by_id["vector-stores"]
    assert node["class"] == "DomainConcept"
    assert node["label"] == "vector-stores"
    assert node["frequency"] == 0
    assert node["node_provenance"] == "lo_key_concept"

    # Every targets-concept edge resolves to a node on both endpoints.
    node_ids = set(nodes_by_id)
    target_edges = [e for e in graph["edges"] if e["type"] == EDGE_TYPE]
    assert target_edges, "expected at least one targets-concept edge"
    for e in target_edges:
        assert e["source"] in node_ids
        assert e["target"] in node_ids


# ---------------------------------------------------------------------- #
# 2. key_concept slug equal to an existing node id → no duplicate
# ---------------------------------------------------------------------- #


def test_existing_node_target_not_duplicated_or_reflagged():
    """When the key_concept slug already names a co-occurrence node, no second
    node is minted and the existing node is NOT stamped with node_provenance."""
    los = [_lo("TO-01", [_target("framework")])]
    concept_graph = {
        "nodes": [
            {"id": "framework", "label": "framework", "frequency": 7,
             "class": "DomainConcept"},
        ],
        "edges": [],
    }
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=concept_graph,
        objectives_metadata=los,
        now=_PINNED_NOW,
    )
    framework_nodes = [n for n in graph["nodes"] if n["id"] == "framework"]
    assert len(framework_nodes) == 1
    assert "node_provenance" not in framework_nodes[0]
    assert framework_nodes[0]["frequency"] == 7


# ---------------------------------------------------------------------- #
# 3. Determinism
# ---------------------------------------------------------------------- #


def test_materialization_is_deterministic_byte_identical():
    los = [
        _lo("TO-01", [_target("vector-stores"), _target("embeddings")]),
        _lo("CO-01", [_target("retrieval")]),
    ]
    kwargs = dict(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        objectives_metadata=los,
        now=_PINNED_NOW,
    )
    a = build_semantic_graph(**kwargs)
    b = build_semantic_graph(**kwargs)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------- #
# 4. Non-targets-concept junk endpoint stays non-materialized
# ---------------------------------------------------------------------- #


def test_non_targets_concept_junk_endpoint_not_materialized():
    """A dangling endpoint from a NON-targets-concept edge keeps surfacing as a
    dangling edge — the materializer only covers targets-concept targets and
    the pedagogical namespaces, never arbitrary slugs."""
    los = [_lo("TO-01", [_target("vector-stores")])]
    # Inject a stray related-to edge onto a junk slug that classifies as
    # neither a pedagogical endpoint nor a targets-concept target.
    concept_graph = {
        "nodes": [],
        "edges": [
            {
                "source": "vector-stores",
                "target": "some-unconnected-junk-slug",
                "type": "related-to",
                "confidence": 0.5,
            }
        ],
    }
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=concept_graph,
        objectives_metadata=los,
        now=_PINNED_NOW,
    )
    node_ids = {n["id"] for n in graph["nodes"]}
    # The targets-concept target IS materialized...
    assert "vector-stores" in node_ids
    # ...but the junk related-to endpoint is NOT.
    assert "some-unconnected-junk-slug" not in node_ids


# ---------------------------------------------------------------------- #
# 5. Schema round trip
# ---------------------------------------------------------------------- #


def test_materialized_graph_validates_against_semantic_graph_schema():
    from jsonschema import Draft7Validator

    schema_path = (
        _PROJECT_ROOT / "schemas" / "knowledge" / "concept_graph_semantic.schema.json"
    )
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    los = [_lo("TO-01", [_target("vector-stores"), _target("embeddings")])]
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        objectives_metadata=los,
        now=_PINNED_NOW,
    )
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(graph), key=lambda e: list(e.absolute_path))
    assert not errors, (
        f"Schema violations: {[e.message for e in errors]}\n"
        f"Failing payload: {json.dumps(graph, indent=2)}"
    )
