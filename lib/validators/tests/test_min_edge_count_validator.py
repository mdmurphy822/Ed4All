"""Wave 91 Action C: tests for MinEdgeCountValidator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.min_edge_count import (  # noqa: E402
    DEFAULT_MIN_CONCEPT_NODES,
    DEFAULT_MIN_EDGE_TYPES,
    DEFAULT_MIN_EDGES,
    MinEdgeCountValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_graph(
    path: Path,
    *,
    edges: List[Dict[str, Any]] | None = None,
    nodes: List[Dict[str, Any]] | None = None,
) -> Path:
    payload: Dict[str, Any] = {}
    if edges is not None:
        payload["edges"] = edges
    if nodes is not None:
        payload["nodes"] = nodes
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_edges(
    n: int, *, relation_types: List[str] | None = None
) -> List[Dict[str, Any]]:
    types = relation_types or ["prerequisite_of"]
    return [
        {
            "source": f"c{i}",
            "target": f"c{i+1}",
            "relation_type": types[i % len(types)],
        }
        for i in range(n)
    ]


def _make_nodes(n: int) -> List[Dict[str, Any]]:
    return [{"id": f"c{i}", "label": f"concept-{i}"} for i in range(n)]


def _inputs(
    tmp_path: Path,
    *,
    pedagogy_edges: int,
    relation_types: List[str],
    concept_nodes: int,
) -> Dict[str, Any]:
    pedagogy = _write_graph(
        tmp_path / "pedagogy_graph.json",
        edges=_make_edges(pedagogy_edges, relation_types=relation_types),
    )
    concept = _write_graph(
        tmp_path / "concept_graph.json",
        nodes=_make_nodes(concept_nodes),
    )
    return {
        "pedagogy_graph_path": str(pedagogy),
        "concept_graph_path": str(concept),
    }


# ---------------------------------------------------------------------------
# Pass paths
# ---------------------------------------------------------------------------


def test_passes_at_default_thresholds(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=DEFAULT_MIN_EDGES,
        relation_types=[
            "prerequisite_of", "teaches", "exemplifies", "assesses",
        ],
        concept_nodes=DEFAULT_MIN_CONCEPT_NODES,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is True
    assert result.score == 1.0
    assert not [i for i in result.issues if i.severity == "critical"]


def test_passes_well_above_thresholds(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=8000,
        relation_types=[
            "prerequisite_of", "teaches", "exemplifies", "assesses",
            "concept_supports_outcome", "follows",
        ],
        concept_nodes=600,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is True
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Critical-fail paths
# ---------------------------------------------------------------------------


def test_fails_when_edge_count_below_floor(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=DEFAULT_MIN_EDGES - 1,
        relation_types=[
            "prerequisite_of", "teaches", "exemplifies", "assesses",
        ],
        concept_nodes=DEFAULT_MIN_CONCEPT_NODES,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PEDAGOGY_EDGES_BELOW_FLOOR" in codes


def test_fails_when_edge_types_below_floor(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=DEFAULT_MIN_EDGES,
        relation_types=["prerequisite_of"],  # only 1 distinct type
        concept_nodes=DEFAULT_MIN_CONCEPT_NODES,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PEDAGOGY_EDGE_TYPES_BELOW_FLOOR" in codes


def test_fails_when_concept_nodes_below_floor(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=DEFAULT_MIN_EDGES,
        relation_types=[
            "prerequisite_of", "teaches", "exemplifies", "assesses",
        ],
        concept_nodes=DEFAULT_MIN_CONCEPT_NODES - 1,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "CONCEPT_NODES_BELOW_FLOOR" in codes


def test_fails_with_all_three_below_floors(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=10,
        relation_types=["prerequisite_of"],
        concept_nodes=5,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PEDAGOGY_EDGES_BELOW_FLOOR" in codes
    assert "PEDAGOGY_EDGE_TYPES_BELOW_FLOOR" in codes
    assert "CONCEPT_NODES_BELOW_FLOOR" in codes
    assert result.score < 0.5  # all three signals well below floors


# ---------------------------------------------------------------------------
# Threshold overrides
# ---------------------------------------------------------------------------


def test_threshold_override_relaxes_gate(tmp_path):
    inputs = _inputs(
        tmp_path,
        pedagogy_edges=10,
        relation_types=["prerequisite_of", "teaches"],
        concept_nodes=5,
    )
    inputs.update(
        min_edges=10,
        min_edge_types=2,
        min_concept_nodes=5,
    )
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is True


def test_relation_type_alternative_keys(tmp_path):
    """Validator accepts ``type`` / ``edge_type`` / ``predicate`` as
    fallbacks for ``relation_type`` so it works against multiple emits."""
    edges = []
    for i in range(DEFAULT_MIN_EDGES):
        edge: Dict[str, Any] = {"source": f"a{i}", "target": f"b{i}"}
        # Cycle through 4 different keys to assert all are honored.
        if i % 4 == 0:
            edge["relation_type"] = "rel-a"
        elif i % 4 == 1:
            edge["type"] = "rel-b"
        elif i % 4 == 2:
            edge["edge_type"] = "rel-c"
        else:
            edge["predicate"] = "rel-d"
        edges.append(edge)
    pedagogy = _write_graph(tmp_path / "pedagogy.json", edges=edges)
    concept = _write_graph(
        tmp_path / "concept.json",
        nodes=_make_nodes(DEFAULT_MIN_CONCEPT_NODES),
    )
    result = MinEdgeCountValidator().validate({
        "pedagogy_graph_path": str(pedagogy),
        "concept_graph_path": str(concept),
    })
    assert result.passed is True


# ---------------------------------------------------------------------------
# Missing / malformed input handling
# ---------------------------------------------------------------------------


def test_missing_inputs_fails_critical():
    result = MinEdgeCountValidator().validate({})
    assert result.passed is False
    assert {"MISSING_INPUTS"} == {i.code for i in result.issues}


def test_missing_pedagogy_file_fails_critical(tmp_path):
    concept = _write_graph(
        tmp_path / "concept.json",
        nodes=_make_nodes(DEFAULT_MIN_CONCEPT_NODES),
    )
    result = MinEdgeCountValidator().validate({
        "pedagogy_graph_path": str(tmp_path / "missing-pedagogy.json"),
        "concept_graph_path": str(concept),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PEDAGOGY_GRAPH_NOT_FOUND" in codes


def test_invalid_json_fails_critical(tmp_path):
    bad = tmp_path / "pedagogy.json"
    bad.write_text("{not json", encoding="utf-8")
    concept = _write_graph(
        tmp_path / "concept.json",
        nodes=_make_nodes(DEFAULT_MIN_CONCEPT_NODES),
    )
    result = MinEdgeCountValidator().validate({
        "pedagogy_graph_path": str(bad),
        "concept_graph_path": str(concept),
    })
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "PEDAGOGY_GRAPH_INVALID_JSON" in codes
