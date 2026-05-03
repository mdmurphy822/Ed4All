"""Phase 6 Subtask 14 — tests for ConceptGraphValidator.

Mirrors the test surface of ``test_min_edge_count_validator.py`` (the
closest sibling validator). Covers:

* happy path with default thresholds (≥10 nodes, ≥5 edge types).
* sparsity issues (too-few-nodes, too-few-edge-types).
* integrity issues (missing relation_type, orphan endpoints,
  self-edges, nodes missing class).
* file / JSON / shape critical-error paths (block action).
* threshold-knob respected via constructor + per-call inputs.
* ``TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true`` opt-in flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.concept_graph import (  # noqa: E402
    DEFAULT_MIN_EDGE_TYPES,
    DEFAULT_MIN_NODES,
    ConceptGraphValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(node_id: str, *, cls: str = "DomainConcept", **extra: Any) -> Dict[str, Any]:
    n: Dict[str, Any] = {"id": node_id, "class": cls, "label": node_id}
    n.update(extra)
    return n


def _edge(
    source: str,
    target: str,
    relation_type: str,
    *,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    e: Dict[str, Any] = {
        "source": source,
        "target": target,
        "relation_type": relation_type,
    }
    if provenance is not None:
        e["provenance"] = provenance
    return e


def _make_healthy_graph(
    n_nodes: int = DEFAULT_MIN_NODES,
    *,
    edge_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a structurally clean graph that passes default thresholds."""
    types = edge_types or [
        "is-a",
        "prerequisite",
        "related-to",
        "teaches",
        "exemplifies",
    ]
    nodes = [_node(f"c{i}") for i in range(n_nodes)]
    # Chain edges across the node set so every endpoint resolves.
    edges: List[Dict[str, Any]] = []
    for i in range(n_nodes - 1):
        rt = types[i % len(types)]
        edges.append(
            _edge(
                f"c{i}",
                f"c{i + 1}",
                rt,
                provenance={"rule": "fixture", "evidence": []},
            )
        )
    return {
        "kind": "concept_semantic",
        "generated_at": "2026-05-03T00:00:00Z",
        "nodes": nodes,
        "edges": edges,
    }


def _write_graph(tmp_path: Path, payload: Any) -> Path:
    p = tmp_path / "concept_graph_semantic.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_passes_at_default_thresholds(tmp_path):
    p = _write_graph(tmp_path, _make_healthy_graph())
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    assert result.passed is True
    assert result.action is None
    # No warnings should fire on the healthy fixture.
    assert [i.code for i in result.issues] == []
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Sparsity floors
# ---------------------------------------------------------------------------


def test_too_few_nodes_emits_warning(tmp_path):
    payload = _make_healthy_graph(n_nodes=5)
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    # Warning-only, gate still passes (no critical issues).
    assert result.passed is True
    assert result.action is None
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_TOO_FEW_NODES" in codes
    # All issues should be warning-severity at this code.
    too_few = [
        i for i in result.issues if i.code == "CONCEPT_GRAPH_TOO_FEW_NODES"
    ]
    assert len(too_few) == 1
    assert too_few[0].severity == "warning"


def test_too_few_edge_types_emits_warning(tmp_path):
    payload = _make_healthy_graph(
        n_nodes=DEFAULT_MIN_NODES,
        edge_types=["is-a", "prerequisite"],
    )
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_TOO_FEW_EDGE_TYPES" in codes


def test_threshold_knob_lowered_via_inputs_admits_small_graph(tmp_path):
    payload = _make_healthy_graph(
        n_nodes=5, edge_types=["is-a", "prerequisite", "teaches"]
    )
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate(
        {
            "concept_graph_path": str(p),
            "min_nodes": 5,
            "min_edge_types": 3,
        }
    )
    assert result.passed is True
    assert [i.code for i in result.issues] == []


def test_threshold_knob_lowered_via_constructor(tmp_path):
    payload = _make_healthy_graph(
        n_nodes=5, edge_types=["is-a", "prerequisite", "teaches"]
    )
    p = _write_graph(tmp_path, payload)
    validator = ConceptGraphValidator(min_nodes=5, min_edge_types=3)
    result = validator.validate({"concept_graph_path": str(p)})
    assert result.passed is True
    assert [i.code for i in result.issues] == []


# ---------------------------------------------------------------------------
# Integrity issues
# ---------------------------------------------------------------------------


def test_orphan_node_emits_warning(tmp_path):
    payload = _make_healthy_graph()
    payload["edges"].append(
        _edge("c0", "c-does-not-exist", "is-a")
    )
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_ORPHAN_NODE" in codes
    assert result.passed is True


def test_self_edge_emits_warning(tmp_path):
    payload = _make_healthy_graph()
    payload["edges"].append(_edge("c0", "c0", "related-to"))
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_SELF_EDGE" in codes
    assert result.passed is True


def test_node_missing_class_emits_warning(tmp_path):
    payload = _make_healthy_graph()
    # Strip 'class' from the first node.
    payload["nodes"][0].pop("class", None)
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_NODE_MISSING_CLASS" in codes
    assert result.passed is True


def test_edge_missing_relation_type_emits_warning(tmp_path):
    payload = _make_healthy_graph()
    payload["edges"].append({"source": "c0", "target": "c1"})
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_EDGE_MISSING_RELATION_TYPE" in codes
    assert result.passed is True


def test_canonical_type_field_accepted_as_relation_type(tmp_path):
    """The canonical schema uses 'type' rather than 'relation_type' —
    both should be accepted."""
    payload = _make_healthy_graph()
    payload["edges"].append({"source": "c0", "target": "c1", "type": "broader-than"})
    p = _write_graph(tmp_path, payload)
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    # No missing-relation-type warning should fire.
    assert "CONCEPT_GRAPH_EDGE_MISSING_RELATION_TYPE" not in [
        i.code for i in result.issues
    ]


# ---------------------------------------------------------------------------
# Critical errors
# ---------------------------------------------------------------------------


def test_missing_input_returns_critical_block(tmp_path):
    result = ConceptGraphValidator().validate({})
    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CONCEPT_GRAPH_MISSING_INPUT" and i.severity == "critical"
        for i in result.issues
    )


def test_missing_file_returns_critical_block(tmp_path):
    p = tmp_path / "does_not_exist.json"
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CONCEPT_GRAPH_NOT_FOUND" and i.severity == "critical"
        for i in result.issues
    )


def test_malformed_json_returns_critical_block(tmp_path):
    p = tmp_path / "concept_graph_semantic.json"
    p.write_text("{ not json", encoding="utf-8")
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CONCEPT_GRAPH_INVALID_JSON" and i.severity == "critical"
        for i in result.issues
    )


def test_wrong_root_shape_returns_critical_block(tmp_path):
    p = _write_graph(tmp_path, ["not", "a", "dict"])
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CONCEPT_GRAPH_BAD_SHAPE" and i.severity == "critical"
        for i in result.issues
    )


def test_missing_nodes_or_edges_keys_returns_critical_block(tmp_path):
    p = _write_graph(tmp_path, {"kind": "concept_semantic", "edges": []})
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CONCEPT_GRAPH_BAD_SHAPE" and i.severity == "critical"
        for i in result.issues
    )


# ---------------------------------------------------------------------------
# TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE opt-in
# ---------------------------------------------------------------------------


def test_provenance_required_when_env_on_emits_warning(
    tmp_path, monkeypatch
):
    payload = _make_healthy_graph()
    # Strip provenance from one edge so the gate fires.
    payload["edges"][0].pop("provenance", None)
    p = _write_graph(tmp_path, payload)
    monkeypatch.setenv("TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE", "true")
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_EDGE_MISSING_PROVENANCE" in codes
    # Warning only.
    assert result.passed is True


def test_provenance_default_off_does_not_flag_legacy_edges(
    tmp_path, monkeypatch
):
    payload = _make_healthy_graph()
    for e in payload["edges"]:
        e.pop("provenance", None)
    p = _write_graph(tmp_path, payload)
    monkeypatch.delenv(
        "TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE", raising=False
    )
    result = ConceptGraphValidator().validate({"concept_graph_path": str(p)})
    codes = [i.code for i in result.issues]
    assert "CONCEPT_GRAPH_EDGE_MISSING_PROVENANCE" not in codes


# ---------------------------------------------------------------------------
# Validator metadata
# ---------------------------------------------------------------------------


def test_validator_metadata():
    v = ConceptGraphValidator()
    assert v.name == "concept_graph"
    assert v.version == "0.1.0"
    assert DEFAULT_MIN_NODES == 10
    assert DEFAULT_MIN_EDGE_TYPES == 5
