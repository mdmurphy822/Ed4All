"""Concept-coverage top-level aggregator (Worker W4.1).

Concept-graph analogue of ``coverage_map.py``. Where the coverage-map
keys on OBJECTIVES, this aggregator keys on CONCEPT NODES: for every
``DomainConcept`` node in ``concept_graph_semantic.json`` it tallies
which pedagogical SURFACES touch the concept —

* ``explained`` — the concept is touched by a ``defined-by`` edge
  (``defined_by_from_first_mention``) OR carries any occurrence chunk
  (content prose mentions it at least once).
* ``defined_in_glossary`` — the concept is touched by an ``is-a`` edge
  (``is_a_from_key_terms`` — the key-terms / glossary definition rule).
* ``assessed`` — the concept is touched by a ``targets-concept`` edge
  (a learning objective explicitly targets the concept at a Bloom
  level).
* ``demonstrated`` — the concept is touched by an ``exemplifies`` edge
  (an example / worked-example chunk exemplifies the concept).
* ``prereq_scaffolded`` — the concept is an endpoint of a
  ``prerequisite`` edge in the concept graph (concept-level
  prerequisite scaffolding). Course-level content-derived prerequisite
  edges (W3.1 ``prerequisite_from_definition_mention``) are additionally
  surfaced in ``summary.content_derived_prereq_edges``.

Output: ``<libv2_course>/concept_coverage.json``.

Opt-in posture (W4.1): the aggregator only runs when
``ED4ALL_CONCEPT_COVERAGE`` is truthy — the ``WorkflowRunner`` post-loop
helper short-circuits on the OFF (default) resolver, so a default run is
byte-identical (no file emitted). :func:`resolve_concept_coverage`
mirrors the parse-with-fallback contract of the other opt-in flags.

Best-effort posture (matches ``CoverageMapAggregator``): missing inputs
degrade gracefully — the aggregator emits a partial report with empty
arrays rather than crashing, and ``WorkflowRunner`` wraps the call in
try/except so an aggregator failure does NOT change ``final_status``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

ENV_CONCEPT_COVERAGE = "ED4ALL_CONCEPT_COVERAGE"
_TRUTHY = {"1", "true", "yes", "on"}

#: Concept-node classes treated as concept vocabulary (mirrors the
#: kg_quality coverage denominator: DomainConcept OR classless).
_CONCEPT_CLASSES = {"DomainConcept", ""}

#: Edge-type -> surface-axis mapping for concept-endpoint edges.
_EDGE_TYPE_AXIS = {
    "defined-by": "explained",
    "is-a": "defined_in_glossary",
    "targets-concept": "assessed",
    "exemplifies": "demonstrated",
    "prerequisite": "prereq_scaffolded",
}

#: W3.1 content-derived prerequisite rule name (surfaced in summary).
_CONTENT_PREREQ_RULE = "prerequisite_from_definition_mention"


def resolve_concept_coverage(value: object = None) -> bool:
    """Return True iff concept-coverage aggregation is enabled.

    ``value`` overrides the env when not ``None``. Falsey / garbage /
    unset -> False (parse-with-fallback, mirroring the other opt-in
    ``ED4ALL_*`` toggles).
    """
    if value is None:
        raw = os.environ.get(ENV_CONCEPT_COVERAGE, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


class ConceptCoverageAggregator:
    """Aggregate concept -> pedagogical-surface coverage.

    Parameters
    ----------
    course_code:
        Operator-facing course code. Surfaces as ``course_code``.
    run_id:
        Workflow ID.
    libv2_course_path:
        Optional override for the LibV2 course directory
        (``LibV2/courses/<slug>``). Used to resolve the concept graph
        (both ``graph/`` and ``concept_graph/`` layouts).
    concept_graph_path:
        Optional explicit path to a ``concept_graph_semantic.json``.
        Wins over the ``libv2_course_path`` probe.
    decision_capture:
        Optional ``DecisionCapture`` instance. When wired the aggregator
        emits one ``concept_coverage_aggregated`` event per ``build()``
        call so the audit trail records the resolved surface counts.
    """

    def __init__(
        self,
        *,
        course_code: str = "",
        run_id: str = "",
        libv2_course_path: Optional[Path] = None,
        concept_graph_path: Optional[Path] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.libv2_course_path = (
            Path(libv2_course_path) if libv2_course_path else None
        )
        self.concept_graph_path = (
            Path(concept_graph_path) if concept_graph_path else None
        )
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Walk the concept graph and build the concept-coverage dict."""
        graph = self._load_concept_graph()
        nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
        edges = graph.get("edges") if isinstance(graph, Mapping) else None
        nodes = nodes if isinstance(nodes, list) else []
        edges = edges if isinstance(edges, list) else []

        # 1. Build the canonical concept table (DomainConcept / classless
        # nodes only — endpoint nodes like Chunk / Outcome are excluded).
        concepts_by_id: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            node_class = str(node.get("class") or "").strip()
            if node_class not in _CONCEPT_CLASSES:
                continue
            occ = node.get("occurrences")
            touching = sorted(
                {
                    str(c).strip()
                    for c in occ
                    if isinstance(c, str) and str(c).strip()
                }
                if isinstance(occ, list)
                else set()
            )
            concepts_by_id[node_id] = {
                "concept_id": node_id,
                "label": str(node.get("label") or node_id),
                "surfaces": {
                    "explained": bool(touching),
                    "defined_in_glossary": False,
                    "assessed": False,
                    "demonstrated": False,
                    "prereq_scaffolded": False,
                },
                "touching_chunks": touching,
            }

        concept_ids: Set[str] = set(concepts_by_id.keys())

        # 2. Walk edges. For each edge whose type maps to a surface axis,
        # mark every concept-node endpoint touched by that edge.
        content_derived_prereq = 0
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            etype = str(edge.get("type") or "").strip()
            axis = _EDGE_TYPE_AXIS.get(etype)
            if axis is None:
                continue
            if etype == "prerequisite":
                if self._edge_rule(edge) == _CONTENT_PREREQ_RULE:
                    content_derived_prereq += 1
            for endpoint in (edge.get("source"), edge.get("target")):
                eid = str(endpoint or "").strip()
                if eid and eid in concept_ids:
                    concepts_by_id[eid]["surfaces"][axis] = True

        # 3. Finalize per-concept rows (surface_count) + summary.
        concepts_list: List[Dict[str, Any]] = []
        for cid in sorted(concepts_by_id.keys()):
            row = concepts_by_id[cid]
            row["surface_count"] = sum(
                1 for v in row["surfaces"].values() if v
            )
            concepts_list.append(row)

        summary = self._build_summary(
            concepts_list,
            content_derived_prereq_edges=content_derived_prereq,
        )

        report = {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "concepts": concepts_list,
            "summary": summary,
        }

        self._emit_aggregator_decision(report)
        return report

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved path on success. Raises ``OSError`` on
        filesystem failure — the workflow-runner caller wraps the call in
        try/except (best-effort; the per-phase reports remain the source
        of truth).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build()
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # Helpers — input resolution
    # ------------------------------------------------------------------
    def _load_concept_graph(self) -> Dict[str, Any]:
        """Best-effort load of ``concept_graph_semantic.json``.

        Priority chain:

        1. Explicit ``self.concept_graph_path`` override.
        2. ``<libv2_course>/graph/concept_graph_semantic.json``
           (process_course route).
        3. ``<libv2_course>/concept_graph/concept_graph_semantic.json``
           (``_run_concept_extraction`` route).

        Returns ``{}`` when no source resolvable (every downstream walk
        no-ops on an empty graph).
        """
        if self.concept_graph_path is not None:
            payload = self._read_json(self.concept_graph_path)
            if isinstance(payload, dict):
                return payload

        if self.libv2_course_path is not None:
            for subdir in ("graph", "concept_graph"):
                cand = (
                    self.libv2_course_path
                    / subdir
                    / "concept_graph_semantic.json"
                )
                if cand.exists():
                    payload = self._read_json(cand)
                    if isinstance(payload, dict):
                        return payload
        return {}

    @staticmethod
    def _edge_rule(edge: Mapping[str, Any]) -> str:
        """Return the provenance rule name for an edge (empty if absent)."""
        prov = edge.get("provenance")
        if isinstance(prov, Mapping):
            raw = prov.get("rule")
            if isinstance(raw, str):
                return raw.strip()
        return ""

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        """Best-effort JSON read; returns ``None`` on any failure."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "concept_coverage: cannot read %s: %s", path, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(
        concepts_list: Sequence[Mapping[str, Any]],
        *,
        content_derived_prereq_edges: int,
    ) -> Dict[str, Any]:
        """Compute the summary block."""
        total = len(concepts_list)

        def _count(axis: str) -> int:
            return sum(
                1 for c in concepts_list if c["surfaces"].get(axis)
            )

        histogram = {str(i): 0 for i in range(6)}
        fully_covered: List[str] = []
        uncovered: List[str] = []
        for c in concepts_list:
            n = int(c.get("surface_count", 0))
            histogram[str(n)] = histogram.get(str(n), 0) + 1
            if n == 5:
                fully_covered.append(c["concept_id"])
            elif n == 0:
                uncovered.append(c["concept_id"])

        return {
            "total_concepts": total,
            "concepts_explained": _count("explained"),
            "concepts_defined_in_glossary": _count("defined_in_glossary"),
            "concepts_assessed": _count("assessed"),
            "concepts_demonstrated": _count("demonstrated"),
            "concepts_prereq_scaffolded": _count("prereq_scaffolded"),
            "fully_covered_concepts": sorted(set(fully_covered)),
            "uncovered_concepts": sorted(set(uncovered)),
            "content_derived_prereq_edges": int(content_derived_prereq_edges),
            "surface_count_histogram": histogram,
        }

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------
    def _emit_aggregator_decision(self, report: Mapping[str, Any]) -> None:
        """Emit one ``concept_coverage_aggregated`` decision per build.

        Best-effort: any failure in the capture path is logged and
        swallowed. Rationale interpolates the resolved surface counts so
        it clears the 20-char floor and stays replayable post-hoc.
        """
        capture = self.decision_capture
        if capture is None:
            return
        try:
            summary = report.get("summary") or {}
            total = summary.get("total_concepts", 0)
            explained = summary.get("concepts_explained", 0)
            glossary = summary.get("concepts_defined_in_glossary", 0)
            assessed = summary.get("concepts_assessed", 0)
            demonstrated = summary.get("concepts_demonstrated", 0)
            prereq = summary.get("concepts_prereq_scaffolded", 0)
            uncovered = summary.get("uncovered_concepts") or []
            fully = summary.get("fully_covered_concepts") or []

            rationale = (
                f"Concept-coverage aggregated: total_concepts={total}, "
                f"explained={explained}, defined_in_glossary={glossary}, "
                f"assessed={assessed}, demonstrated={demonstrated}, "
                f"prereq_scaffolded={prereq}, "
                f"fully_covered={len(fully)}, uncovered={len(uncovered)}"
            )
            capture.log_decision(
                decision_type="concept_coverage_aggregated",
                decision=f"uncovered_concepts={len(uncovered)}/{total}",
                rationale=rationale,
                context=str({
                    "course_code": report.get("course_code"),
                    "run_id": report.get("run_id"),
                    "schema_version": report.get("schema_version"),
                }),
                metrics={
                    "total_concepts": total,
                    "concepts_explained": explained,
                    "concepts_defined_in_glossary": glossary,
                    "concepts_assessed": assessed,
                    "concepts_demonstrated": demonstrated,
                    "concepts_prereq_scaffolded": prereq,
                    "fully_covered_count": len(fully),
                    "uncovered_count": len(uncovered),
                    "content_derived_prereq_edges": summary.get(
                        "content_derived_prereq_edges", 0
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "concept_coverage_aggregated: %s",
                exc,
            )


__all__ = [
    "ConceptCoverageAggregator",
    "SCHEMA_VERSION",
    "ENV_CONCEPT_COVERAGE",
    "resolve_concept_coverage",
]
