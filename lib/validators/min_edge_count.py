"""Wave 91 Action C: MinEdgeCountValidator.

Pre-synthesis gate that fails closed when the upstream pedagogy +
concept graphs are too sparse to produce useful training pairs. Wired
into ``config/workflows.yaml`` at
``textbook_to_course::training_synthesis`` ahead of the
``assessment_quality`` gate so a thin graph never silently generates
sparse synthesis output.

Critical issues:
    - ``pedagogy_graph.json`` has fewer than ``min_edges`` (default
      100) total edges.
    - ``pedagogy_graph.json`` carries fewer than ``min_edge_types``
      distinct edge ``relation_type`` values (default 4).
    - ``concept_graph.json`` has fewer than ``min_concept_nodes``
      (default 50) nodes.

Inputs:
    pedagogy_graph_path: Path to ``pedagogy_graph.json``. Required.
    concept_graph_path: Path to ``concept_graph.json``. Required.
    min_edges: Optional override for the edge-count floor.
    min_edge_types: Optional override for the distinct-edge-type floor.
    min_concept_nodes: Optional override for the concept-node floor.
    gate_id: Optional override for the gate id (otherwise
      ``min_edge_count``).

Mirrors ``lib/validators/libv2_manifest.py`` shape so workflow gate
wiring is uniform.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Default thresholds. Values chosen to exclude trivially-thin graphs
# (a near-empty corpus produces ~0 edges, ~0 nodes; a real chapter
# extraction routinely emits 1000+ edges across 8+ relation types).
DEFAULT_MIN_EDGES = 100
DEFAULT_MIN_EDGE_TYPES = 4
DEFAULT_MIN_CONCEPT_NODES = 50


class MinEdgeCountValidator:
    """Pre-synthesis sparsity gate over the pedagogy + concept graphs."""

    name = "min_edge_count"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "min_edge_count")
        issues: List[GateIssue] = []

        pedagogy_path_raw = inputs.get("pedagogy_graph_path")
        concept_path_raw = inputs.get("concept_graph_path")
        if not pedagogy_path_raw or not concept_path_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_INPUTS",
                    message=(
                        "MinEdgeCountValidator requires "
                        "pedagogy_graph_path and concept_graph_path."
                    ),
                )],
            )

        min_edges = int(
            inputs.get("min_edges", DEFAULT_MIN_EDGES) or DEFAULT_MIN_EDGES
        )
        min_edge_types = int(
            inputs.get("min_edge_types", DEFAULT_MIN_EDGE_TYPES)
            or DEFAULT_MIN_EDGE_TYPES
        )
        min_concept_nodes = int(
            inputs.get("min_concept_nodes", DEFAULT_MIN_CONCEPT_NODES)
            or DEFAULT_MIN_CONCEPT_NODES
        )

        pedagogy_path = Path(pedagogy_path_raw)
        concept_path = Path(concept_path_raw)

        # ---------- Pedagogy graph ----------
        pedagogy_data, pedagogy_issue = self._load_graph(
            pedagogy_path, "PEDAGOGY"
        )
        if pedagogy_issue is not None:
            issues.append(pedagogy_issue)
            pedagogy_data = None

        edge_count = 0
        edge_type_count = 0
        if pedagogy_data is not None:
            edges = pedagogy_data.get("edges") or []
            edge_count = len(edges)
            relation_types = set()
            for e in edges:
                if not isinstance(e, dict):
                    continue
                rt = (
                    e.get("relation_type")
                    or e.get("type")
                    or e.get("edge_type")
                    or e.get("predicate")
                )
                if rt:
                    relation_types.add(rt)
            edge_type_count = len(relation_types)

            if edge_count < min_edges:
                issues.append(GateIssue(
                    severity="critical",
                    code="PEDAGOGY_EDGES_BELOW_FLOOR",
                    message=(
                        f"pedagogy_graph.json has {edge_count} edges "
                        f"(< min_edges={min_edges})."
                    ),
                    location=str(pedagogy_path),
                    suggestion=(
                        "A thin pedagogy graph yields sparse training "
                        "pairs. Verify upstream chunk extraction + "
                        "pedagogy emit before regenerating training pairs."
                    ),
                ))
            if edge_type_count < min_edge_types:
                issues.append(GateIssue(
                    severity="critical",
                    code="PEDAGOGY_EDGE_TYPES_BELOW_FLOOR",
                    message=(
                        f"pedagogy_graph.json has {edge_type_count} "
                        f"distinct edge types (< min_edge_types="
                        f"{min_edge_types})."
                    ),
                    location=str(pedagogy_path),
                    suggestion=(
                        "Few edge types signal a single-relation graph; "
                        "downstream curriculum + prereq-windowing assume "
                        "multiple typed relations."
                    ),
                ))

        # ---------- Concept graph ----------
        concept_data, concept_issue = self._load_graph(
            concept_path, "CONCEPT"
        )
        if concept_issue is not None:
            issues.append(concept_issue)
            concept_data = None

        concept_node_count = 0
        if concept_data is not None:
            nodes = concept_data.get("nodes") or []
            concept_node_count = len(nodes)
            if concept_node_count < min_concept_nodes:
                issues.append(GateIssue(
                    severity="critical",
                    code="CONCEPT_NODES_BELOW_FLOOR",
                    message=(
                        f"concept_graph.json has {concept_node_count} "
                        f"nodes (< min_concept_nodes={min_concept_nodes})."
                    ),
                    location=str(concept_path),
                    suggestion=(
                        "A thin concept graph means most chunks share "
                        "almost no concepts; preference pairs degrade "
                        "to template-collapse."
                    ),
                ))

        critical = sum(1 for i in issues if i.severity == "critical")
        passed = critical == 0
        # Score: simple linear blend of the three signals against their
        # respective floors (capped at 1.0).
        score = self._compose_score(
            edge_count=edge_count,
            min_edges=min_edges,
            edge_type_count=edge_type_count,
            min_edge_types=min_edge_types,
            concept_node_count=concept_node_count,
            min_concept_nodes=min_concept_nodes,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _load_graph(path: Path, label: str):
        if not path.exists():
            return None, GateIssue(
                severity="critical",
                code=f"{label}_GRAPH_NOT_FOUND",
                message=f"{label.lower()} graph not found at {path}",
                location=str(path),
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, GateIssue(
                severity="critical",
                code=f"{label}_GRAPH_INVALID_JSON",
                message=f"{label.lower()} graph JSON failed to parse: {exc}",
                location=str(path),
            )
        if not isinstance(data, dict):
            return None, GateIssue(
                severity="critical",
                code=f"{label}_GRAPH_BAD_SHAPE",
                message=(
                    f"{label.lower()} graph root is not a JSON object "
                    f"(got {type(data).__name__})"
                ),
                location=str(path),
            )
        return data, None

    @staticmethod
    def _compose_score(
        *,
        edge_count: int,
        min_edges: int,
        edge_type_count: int,
        min_edge_types: int,
        concept_node_count: int,
        min_concept_nodes: int,
    ) -> float:
        """Blend the three signals into a single 0..1 score for the gate."""
        def _ratio(actual: int, floor: int) -> float:
            if floor <= 0:
                return 1.0
            return min(1.0, max(0.0, actual / float(floor)))

        return round(
            (
                _ratio(edge_count, min_edges)
                + _ratio(edge_type_count, min_edge_types)
                + _ratio(concept_node_count, min_concept_nodes)
            )
            / 3.0,
            4,
        )


__all__ = [
    "MinEdgeCountValidator",
    "DEFAULT_MIN_EDGES",
    "DEFAULT_MIN_EDGE_TYPES",
    "DEFAULT_MIN_CONCEPT_NODES",
]
