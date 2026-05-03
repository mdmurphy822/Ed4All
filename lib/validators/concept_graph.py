"""Phase 6 Subtask 14 — ConceptGraphValidator.

Gates the structure + minimal quality of ``concept_graph_semantic.json``
emitted by the new ``concept_extraction`` workflow phase (Subtask 11)
via ``MCP/tools/pipeline_tools.py::_run_concept_extraction`` (Subtask
12), which delegates to
``Trainforge.pedagogy_graph_builder.build_pedagogy_graph``.

Wired as the ``concept_graph`` validation gate on
``textbook_to_course::concept_extraction``. Phase 6 lands as
**warning-severity** so a thin first run surfaces drift without
blocking the pipeline; Phase 7+ promotes the relevant issue codes to
critical once corpus calibration confirms safe.

Per-graph contract (mirrors `build_pedagogy_graph` output shape +
the canonical schema at
``schemas/knowledge/concept_graph_semantic.schema.json``):

1. **File / shape errors** are CRITICAL (the graph either doesn't
   exist, doesn't parse, or has the wrong root shape — there's no
   warning-actionable signal in those cases).
2. **Sparsity floors** are WARNING:
   - Fewer than ``min_nodes`` (default 10) nodes -> ``CONCEPT_GRAPH_TOO_FEW_NODES``.
   - Fewer than ``min_edge_types`` (default 5) distinct ``relation_type``
     values -> ``CONCEPT_GRAPH_TOO_FEW_EDGE_TYPES``. The plan calls
     for "taxonomic + pedagogical" diversity; 5 captures both
     `is-a`/`prerequisite_of`/`related-to`-class taxonomic ties plus
     at least 2 pedagogical relations like `teaches` / `exemplifies`.
3. **Per-node integrity** is WARNING:
   - Any node missing ``class`` -> ``CONCEPT_GRAPH_NODE_MISSING_CLASS``.
4. **Per-edge integrity** is WARNING:
   - Any edge missing ``relation_type`` -> ``CONCEPT_GRAPH_EDGE_MISSING_RELATION_TYPE``.
   - Any edge whose ``source`` / ``target`` does not resolve to a
     declared node ID -> ``CONCEPT_GRAPH_ORPHAN_NODE``.
   - Any edge with ``source == target`` -> ``CONCEPT_GRAPH_SELF_EDGE``.
5. **Edge provenance** is OPT-IN via
   ``TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true``: when on, every
   edge MUST carry a non-empty ``provenance`` field, else
   ``CONCEPT_GRAPH_EDGE_MISSING_PROVENANCE``. Default off so legacy /
   pedagogy-graph-derived emit (which carries provenance only for
   semantic edges) doesn't false-flag.

Action signal: this validator never sets ``action="regenerate"``
because the gate is wired warning-only initially and the upstream
``build_pedagogy_graph`` is deterministic — re-rolling won't change
the emit. Critical-severity file/shape errors set ``action="block"``
so the runner halts on a structurally broken graph.

Inputs contract:

* ``inputs["concept_graph_path"]`` — path to the concept graph JSON.
  Required.
* ``inputs["min_nodes"]`` — optional override for the node-count floor
  (default 10).
* ``inputs["min_edge_types"]`` — optional override for the
  distinct-edge-type floor (default 5).
* ``inputs["gate_id"]`` — optional override for the gate ID stamped
  on the returned ``GateResult``. Defaults to the validator name.

Cross-references:

* ``schemas/knowledge/concept_graph_semantic.schema.json`` — canonical
  schema shape. The validator does NOT JSON-Schema-validate the
  graph; it operates on the in-memory dict to keep wall-time low and
  emit per-issue codes.
* ``Trainforge/pedagogy_graph_builder.py::build_pedagogy_graph`` —
  the upstream emitter Subtask 12 dispatches.
* ``lib/validators/min_edge_count.py`` — Wave 91 sibling validator
  that gates pre-synthesis sparsity at higher floors (≥100 edges,
  ≥50 nodes). This validator complements but does not duplicate it:
  ``min_edge_count`` runs at ``training_synthesis`` against an
  already-imported corpus; ``concept_graph`` runs at
  ``concept_extraction`` ahead of objective synthesis.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Cap the per-issue list so a uniformly-broken graph (e.g. every edge
# orphaned) doesn't drown the gate report. Mirrors the cap convention
# used by ``lib/validators/abcd_objective.py``.
_ISSUE_LIST_CAP: int = 50


# Defaults are calibrated to the smallest plausible non-degenerate
# concept graph. ``build_pedagogy_graph`` emits 6 BloomLevel +
# 3 DifficultyLevel nodes unconditionally (= 9 nodes baseline), so
# requiring ≥10 means at least one substantive concept / objective /
# misconception node landed. ``min_edge_types=5`` is the plan's
# "≥5 edge types present (taxonomic + pedagogical)" floor — any
# corpus that produced both kinds of edges easily clears it; a corpus
# with only taxonomic edges (``is-a``, ``prerequisite``, ``related-to``)
# is exactly 3, surfacing the gap.
DEFAULT_MIN_NODES: int = 10
DEFAULT_MIN_EDGE_TYPES: int = 5


#: Env-var that flips the per-edge ``provenance`` integrity check on.
#: Off by default so legacy pedagogy-graph emit (which only carries
#: ``provenance`` on the typed semantic edges) doesn't false-flag.
_PROVENANCE_ENV: str = "TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE"


def _is_truthy(value: Optional[str]) -> bool:
    """Match the canonical truthy set used elsewhere in the project."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _edge_relation_type(edge: Dict[str, Any]) -> Optional[str]:
    """Pull the canonical relation-type field, tolerating legacy aliases.

    ``build_pedagogy_graph`` emits ``relation_type``; the typed-edge
    concept-graph schema uses ``type``. Both are accepted so the
    validator works whether Subtask 12's helper writes the pedagogy-
    graph form (current Phase 6 plan) or a future emit using the
    canonical concept-graph schema.
    """
    rt = edge.get("relation_type")
    if isinstance(rt, str) and rt.strip():
        return rt.strip()
    rt = edge.get("type")
    if isinstance(rt, str) and rt.strip():
        return rt.strip()
    return None


def _edge_endpoint(edge: Dict[str, Any], key: str) -> Optional[str]:
    """Return a non-empty string endpoint or None."""
    raw = edge.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _has_provenance(edge: Dict[str, Any]) -> bool:
    """Return True iff the edge carries a non-empty ``provenance``."""
    prov = edge.get("provenance")
    if prov is None:
        return False
    if isinstance(prov, (str, list, dict)) and not prov:
        return False
    return True


class ConceptGraphValidator:
    """Phase 6 concept-graph structural + minimal-quality gate.

    Validator-protocol-compatible class wired as the
    ``concept_graph`` gate on
    ``textbook_to_course::concept_extraction``. Severity warning
    initially (Phase 6 ST 11 / 14); Phase 7+ promotes select codes
    to critical once corpus calibration confirms.
    """

    name = "concept_graph"
    version = "0.1.0"  # Phase 6 ST 14 PoC

    def __init__(
        self,
        *,
        min_nodes: int = DEFAULT_MIN_NODES,
        min_edge_types: int = DEFAULT_MIN_EDGE_TYPES,
    ) -> None:
        self._default_min_nodes = int(min_nodes)
        self._default_min_edge_types = int(min_edge_types)

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        issues: List[GateIssue] = []

        path_raw = inputs.get("concept_graph_path")
        if not path_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CONCEPT_GRAPH_MISSING_INPUT",
                        message=(
                            "ConceptGraphValidator requires "
                            "inputs['concept_graph_path']."
                        ),
                    )
                ],
                action="block",
            )

        path = Path(path_raw)
        if not path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CONCEPT_GRAPH_NOT_FOUND",
                        message=(
                            f"concept_graph_semantic.json not found at "
                            f"{path}"
                        ),
                        location=str(path),
                    )
                ],
                action="block",
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CONCEPT_GRAPH_INVALID_JSON",
                        message=(
                            f"concept_graph_semantic.json failed to parse: "
                            f"{exc.__class__.__name__}: {exc}"
                        ),
                        location=str(path),
                    )
                ],
                action="block",
            )

        if not isinstance(data, dict):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="CONCEPT_GRAPH_BAD_SHAPE",
                        message=(
                            f"concept_graph root is not a JSON object "
                            f"(got {type(data).__name__})."
                        ),
                        location=str(path),
                    )
                ],
                action="block",
            )

        nodes_raw = data.get("nodes")
        edges_raw = data.get("edges")
        shape_errors: List[GateIssue] = []
        if not isinstance(nodes_raw, list):
            shape_errors.append(
                GateIssue(
                    severity="critical",
                    code="CONCEPT_GRAPH_BAD_SHAPE",
                    message=(
                        f"concept_graph['nodes'] is missing or not a list "
                        f"(got {type(nodes_raw).__name__})."
                    ),
                    location=str(path),
                )
            )
        if not isinstance(edges_raw, list):
            shape_errors.append(
                GateIssue(
                    severity="critical",
                    code="CONCEPT_GRAPH_BAD_SHAPE",
                    message=(
                        f"concept_graph['edges'] is missing or not a list "
                        f"(got {type(edges_raw).__name__})."
                    ),
                    location=str(path),
                )
            )
        if shape_errors:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=shape_errors,
                action="block",
            )

        # Resolve thresholds: per-call inputs win; constructor defaults
        # are the next layer.
        min_nodes = int(
            inputs.get("min_nodes", self._default_min_nodes)
            or self._default_min_nodes
        )
        min_edge_types = int(
            inputs.get("min_edge_types", self._default_min_edge_types)
            or self._default_min_edge_types
        )
        require_provenance = _is_truthy(os.environ.get(_PROVENANCE_ENV))

        nodes: List[Dict[str, Any]] = [
            n for n in nodes_raw if isinstance(n, dict)
        ]
        edges: List[Dict[str, Any]] = [
            e for e in edges_raw if isinstance(e, dict)
        ]

        # ----------- Sparsity floors -----------
        node_count = len(nodes)
        if node_count < min_nodes:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CONCEPT_GRAPH_TOO_FEW_NODES",
                    message=(
                        f"concept_graph has {node_count} nodes "
                        f"(< min_nodes={min_nodes})."
                    ),
                    location=str(path),
                    suggestion=(
                        "Verify upstream chunk extraction + concept "
                        "tagging emit. A thin concept graph means "
                        "objectives + content authored downstream will "
                        "have nothing to anchor on."
                    ),
                )
            )

        relation_types: Set[str] = set()
        edges_missing_relation = 0
        edges_with_self = 0
        edges_orphaned = 0
        edges_missing_provenance = 0

        node_ids: Set[str] = set()
        nodes_missing_class = 0
        for n in nodes:
            nid = n.get("id")
            if isinstance(nid, str) and nid.strip():
                node_ids.add(nid.strip())
            cls = n.get("class")
            if not (isinstance(cls, str) and cls.strip()):
                nodes_missing_class += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CONCEPT_GRAPH_NODE_MISSING_CLASS",
                            message=(
                                f"Node {nid!r} is missing the required "
                                f"'class' field. The concept-graph "
                                f"contract requires every node to "
                                f"carry a class label (e.g. "
                                f"'DomainConcept', 'Outcome', "
                                f"'BloomLevel')."
                            ),
                            location=str(path),
                        )
                    )

        # ----------- Per-edge integrity -----------
        for idx, e in enumerate(edges):
            rt = _edge_relation_type(e)
            if rt is None:
                edges_missing_relation += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CONCEPT_GRAPH_EDGE_MISSING_RELATION_TYPE",
                            message=(
                                f"Edge[{idx}] is missing 'relation_type' "
                                f"(or canonical 'type') field. Every "
                                f"concept-graph edge must declare its "
                                f"relation."
                            ),
                            location=str(path),
                        )
                    )
            else:
                relation_types.add(rt)

            src = _edge_endpoint(e, "source")
            tgt = _edge_endpoint(e, "target")

            if src is not None and tgt is not None and src == tgt:
                edges_with_self += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CONCEPT_GRAPH_SELF_EDGE",
                            message=(
                                f"Edge[{idx}] has source == target "
                                f"({src!r}). Self-edges aren't useful "
                                f"in a concept graph and usually "
                                f"signal an upstream emission bug."
                            ),
                            location=str(path),
                        )
                    )

            for end_label, end_id in (("source", src), ("target", tgt)):
                if end_id is None:
                    continue
                if end_id not in node_ids:
                    edges_orphaned += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="CONCEPT_GRAPH_ORPHAN_NODE",
                                message=(
                                    f"Edge[{idx}] {end_label}={end_id!r} "
                                    f"does not resolve to a declared "
                                    f"node ID. Edge endpoints must "
                                    f"reference nodes in the same "
                                    f"graph."
                                ),
                                location=str(path),
                            )
                        )

            if require_provenance and not _has_provenance(e):
                edges_missing_provenance += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CONCEPT_GRAPH_EDGE_MISSING_PROVENANCE",
                            message=(
                                f"Edge[{idx}] is missing 'provenance' "
                                f"but {_PROVENANCE_ENV}=true requires "
                                f"every edge to carry a non-empty "
                                f"provenance object."
                            ),
                            location=str(path),
                        )
                    )

        edge_type_count = len(relation_types)
        if edge_type_count < min_edge_types:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CONCEPT_GRAPH_TOO_FEW_EDGE_TYPES",
                    message=(
                        f"concept_graph has {edge_type_count} distinct "
                        f"edge types (< min_edge_types={min_edge_types}). "
                        f"Saw: {sorted(relation_types) if relation_types else '<none>'}."
                    ),
                    location=str(path),
                    suggestion=(
                        "The plan's 'taxonomic + pedagogical' contract "
                        "expects both is-a / prerequisite / related-to "
                        "ties AND pedagogical edges (teaches, "
                        "exemplifies, assesses). Few edge types signal "
                        "a single-relation graph."
                    ),
                )
            )

        # Score: linear blend of sparsity + integrity signals, clamped
        # at [0, 1]. Mirrors the convention used by sibling Phase 4 /
        # Wave 91 validators (see ``min_edge_count.py::_compose_score``).
        score = self._compose_score(
            node_count=node_count,
            min_nodes=min_nodes,
            edge_type_count=edge_type_count,
            min_edge_types=min_edge_types,
            edges_total=len(edges),
            edges_clean=len(edges)
            - edges_missing_relation
            - edges_with_self
            - edges_orphaned
            - edges_missing_provenance,
            nodes_total=len(nodes),
            nodes_clean=len(nodes) - nodes_missing_class,
        )

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0

        # Action: warning-only by design. Critical-severity file/shape
        # errors took the early-return paths above with action="block";
        # by the time we reach here the only issues are warnings.
        action: Optional[str] = None

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _compose_score(
        *,
        node_count: int,
        min_nodes: int,
        edge_type_count: int,
        min_edge_types: int,
        edges_total: int,
        edges_clean: int,
        nodes_total: int,
        nodes_clean: int,
    ) -> float:
        """Blend sparsity + integrity into a single 0..1 score."""

        def _ratio(actual: int, floor: int) -> float:
            if floor <= 0:
                return 1.0
            return min(1.0, max(0.0, actual / float(floor)))

        node_floor_ratio = _ratio(node_count, min_nodes)
        edge_type_floor_ratio = _ratio(edge_type_count, min_edge_types)
        edge_integrity_ratio = (
            1.0 if edges_total == 0
            else max(0.0, min(1.0, edges_clean / float(edges_total)))
        )
        node_integrity_ratio = (
            1.0 if nodes_total == 0
            else max(0.0, min(1.0, nodes_clean / float(nodes_total)))
        )

        return round(
            (
                node_floor_ratio
                + edge_type_floor_ratio
                + edge_integrity_ratio
                + node_integrity_ratio
            )
            / 4.0,
            4,
        )


__all__ = [
    "ConceptGraphValidator",
    "DEFAULT_MIN_NODES",
    "DEFAULT_MIN_EDGE_TYPES",
]
