"""Prerequisite-sequencing validator (warning-day-1).

Companion gate to :mod:`lib.generation.prereq_sequencer`. Reads the emitted
terminal-objective ordering + the archived ``concept_graph_semantic.json``,
projects the concept ``prerequisite`` edges (unioned with Lane P's
content-derived *federation* ``TO -> TO`` edges from
``prerequisite_from_definition_mention``) onto the same TO->TO dependency graph
the sequencer uses, and flags ``PREREQ_ORDER_VIOLATION`` (warning) when a
DEPENDENT TO precedes ANY of its PREREQUISITE TOs in the emitted order.

W3.5 — **transitive reachability**. The gate no longer checks only DIRECT
(adjacent) TO edges: it computes, for every TO, the full set of its
*transitively* required prerequisite TOs (all ancestors reachable by following
``prereq -> dependent`` edges) and flags a violation whenever a dependent TO
precedes ANY transitive prerequisite. This catches multi-hop ordering defects
(``A -> B -> C`` emitted as ``[C, A, B]`` violates the ``A`` and ``B``
constraints on ``C`` even though no single adjacent pair is inverted end-to-end)
that the direct-edge check missed. Direct violations are a strict subset of the
transitive violations reported here. Cycles are handled by DFS visited-tracking
(a node is never its own transitive prerequisite).

Default OFF / no graph / no edges → ``PREREQ_SEQUENCING_DISABLED`` info issue +
``passed=True`` (byte-stable no-op). Anti-fabrication: only audits real TO
edges projected from real ``key_concepts`` + real prereq edges; an edge whose
endpoints map to no TO is ignored (no TO dependency is invented). Rides the
existing ``ED4ALL_PREREQ_SEQUENCING`` flag (no new flag; identity when off).
``# TODO(calibration)`` deferred critical-flip.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.generation.prereq_sequencer import (
    _build_to_edges,
    _to_id,
    resolve_prereq_sequencing,
)

logger = logging.getLogger(__name__)

_ISSUE_LIST_CAP = 50


def _load_json(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("prereq_sequencing: failed to load %s: %s", path, exc)
        return None


def _resolve_terminal_objectives(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve the emitted TO list (in emitted order) from inputs."""
    tos = inputs.get("terminal_objectives")
    if isinstance(tos, list) and tos:
        return [t for t in tos if isinstance(t, dict)]
    doc = _load_json(inputs.get("synthesized_objectives_path"))
    if isinstance(doc, dict):
        cand = doc.get("terminal_objectives") or doc.get("terminal_outcomes") or []
        if isinstance(cand, list):
            return [t for t in cand if isinstance(t, dict)]
    return []


def _resolve_concept_graph(inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    graph = inputs.get("concept_graph")
    if isinstance(graph, dict):
        return graph
    doc = _load_json(inputs.get("concept_graph_path"))
    return doc if isinstance(doc, dict) else None


def _build_dependency_maps(
    to_edges: List[Tuple[str, str, float]],
) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """Return ``(succ, pred)`` adjacency for the TO->TO prereq graph.

    ``to_edges`` are ``(prereq, dependent, confidence)`` meaning
    ``prereq -> dependent`` (the prereq must precede the dependent). ``succ``
    maps prereq -> [dependents]; ``pred`` maps dependent -> {direct prereqs}.
    Self-edges are dropped (a TO is never its own prerequisite).
    """
    succ: Dict[str, List[str]] = defaultdict(list)
    pred: Dict[str, Set[str]] = defaultdict(set)
    for pre, dep, _conf in to_edges:
        if pre == dep:
            continue
        succ[pre].append(dep)
        pred[dep].add(pre)
    return succ, pred


def _transitive_ancestors(pred: Dict[str, Set[str]], node: str) -> Set[str]:
    """Return every TO that is a (transitive) prerequisite of ``node``.

    Follows the direct-predecessor map upward with visited-tracking so a cycle
    can never spin or make a node its own transitive prerequisite (``node`` is
    excluded from the returned set even when it participates in a cycle).
    """
    seen: Set[str] = set()
    stack = list(pred.get(node, ()))
    while stack:
        anc = stack.pop()
        if anc == node or anc in seen:
            continue
        seen.add(anc)
        stack.extend(pred.get(anc, ()))
    return seen


def _shortest_path(succ: Dict[str, List[str]], src: str, dst: str) -> List[str]:
    """Deterministic BFS shortest path ``src -> ... -> dst`` (or [] if none).

    Neighbors are visited in sorted order so the reported path is stable. A
    length-2 path is a direct edge; longer paths explain a multi-hop
    (transitive) dependency in the violation message.
    """
    if src == dst:
        return [src]
    prev: Dict[str, Optional[str]] = {src: None}
    queue: deque = deque([src])
    while queue:
        node = queue.popleft()
        if node == dst:
            break
        for nxt in sorted(succ.get(node, ())):
            if nxt not in prev:
                prev[nxt] = node
                queue.append(nxt)
    if dst not in prev:
        return []
    path: List[str] = []
    cur: Optional[str] = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


class PrereqSequencingValidator:
    """Warning-day-1 prerequisite-order violation gate."""

    name = "prerequisite_sequencing"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def _skip(self, gate_id: str, code: str, message: str) -> GateResult:
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            issues=[GateIssue(severity="info", code=code, message=message)],
            action=None,
            metadata={"prerequisite_sequencing": {"enabled": True, "skipped": code}},
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)

        enabled = inputs.get("prereq_sequencing_enabled")
        if enabled is None:
            enabled = resolve_prereq_sequencing()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="PREREQ_SEQUENCING_DISABLED",
                        message=(
                            "ED4ALL_PREREQ_SEQUENCING unset — prerequisite "
                            "sequencing gate skipped (byte-stable no-op)."
                        ),
                    )
                ],
                action=None,
                metadata={"prerequisite_sequencing": {"enabled": False}},
            )

        tos = _resolve_terminal_objectives(inputs)
        if not tos:
            return self._skip(
                gate_id, "PREREQ_SEQUENCING_NO_OBJECTIVES",
                "no terminal objectives resolved; cannot audit ordering.",
            )
        graph = _resolve_concept_graph(inputs)
        if graph is None:
            return self._skip(
                gate_id, "PREREQ_SEQUENCING_NO_GRAPH",
                "no concept_graph_semantic.json resolved; ordering not auditable.",
            )

        to_edges, _edge_stats = _build_to_edges(tos, graph)
        n_concept_edges = _edge_stats["prereq_concept_edges"]
        ignored = _edge_stats["ignored_unmapped_edges"]
        if not to_edges:
            return self._skip(
                gate_id, "PREREQ_SEQUENCING_NO_EDGES",
                "no TO->TO prerequisite edges projected; nothing to audit.",
            )

        position = {_to_id(t): i for i, t in enumerate(tos)}
        succ, pred = _build_dependency_maps(to_edges)

        # W3.5 — TRANSITIVE reachability. For every dependent TO, audit ALL of
        # its transitive prerequisites (not just direct edges): a dependent must
        # not precede ANY TO from which it is reachable. Iterate the dependents
        # in EMITTED order + the prereqs sorted for deterministic issue output.
        issues: List[GateIssue] = []
        n_violations = 0
        n_direct_violations = 0
        n_transitive_pairs = 0
        for t in tos:
            dep_to = _to_id(t)
            p_dep = position.get(dep_to)
            if p_dep is None:
                continue
            ancestors = _transitive_ancestors(pred, dep_to)
            n_transitive_pairs += len(ancestors)
            direct_preds = pred.get(dep_to, set())
            for pre_to in sorted(ancestors):
                p_pre = position.get(pre_to)
                if p_pre is None:
                    continue
                # Prerequisite must precede dependent. A dependent BEFORE a
                # (transitive) prerequisite is a residual (cycle-broken /
                # unsatisfied) ordering violation.
                if p_dep < p_pre:
                    n_violations += 1
                    is_direct = pre_to in direct_preds
                    if is_direct:
                        n_direct_violations += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        path = _shortest_path(succ, pre_to, dep_to)
                        if is_direct or len(path) <= 2:
                            hop = "directly requires"
                            via = ""
                        else:
                            hop = "transitively requires"
                            via = f" (via {' -> '.join(path)})"
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="PREREQ_ORDER_VIOLATION",
                                message=(
                                    f"dependent objective {dep_to!r} (pos "
                                    f"{p_dep}) {hop} prerequisite {pre_to!r} "
                                    f"(pos {p_pre}){via}, but is taught BEFORE "
                                    f"it in the emitted order; the prerequisite "
                                    f"topic is sequenced AFTER the topic that "
                                    f"depends on it."
                                ),
                                location=f"{dep_to} before {pre_to}",
                                suggestion=(
                                    "Enable ED4ALL_PREREQ_SEQUENCING, or resolve "
                                    "the prerequisite cycle so the prerequisite "
                                    "topic precedes its dependent."
                                ),
                            )
                        )

        self._emit_decision(
            inputs.get("decision_capture") or self._decision_capture,
            n_tos=len(tos),
            to_edges=len(to_edges),
            n_concept_edges=n_concept_edges,
            n_violations=n_violations,
            n_transitive_pairs=n_transitive_pairs,
            n_direct_violations=n_direct_violations,
        )

        # Warning-day-1: passed stays True.
        #
        # TODO(calibration): flip PREREQ_ORDER_VIOLATION to critical (severity:
        # critical + on_fail: block, set passed=n_violations==0) AFTER
        # scripts/harness/calibration_harness.py confirms the FP rate on >=2 corpora.
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=round(1.0 - n_violations / max(1, n_transitive_pairs), 4),
            issues=issues,
            action=None,
            metadata={
                "prerequisite_sequencing": {
                    "enabled": True,
                    "n_terminal_objectives": len(tos),
                    "to_edges": len(to_edges),
                    "prereq_concept_edges": n_concept_edges,
                    "ignored_unmapped_edges": ignored,
                    "n_violations": n_violations,
                    "n_direct_violations": n_direct_violations,
                    "n_transitive_prereq_pairs": n_transitive_pairs,
                }
            },
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        n_tos: int,
        to_edges: int,
        n_concept_edges: int,
        n_violations: int,
        n_transitive_pairs: int,
        n_direct_violations: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="validation_result",
                decision=(
                    f"prerequisite_sequencing violations={n_violations}/"
                    f"{n_transitive_pairs} transitive prereq pairs "
                    f"({n_direct_violations} direct)"
                ),
                rationale=(
                    "Audited the emitted terminal-objective order against the "
                    "TRANSITIVE closure of the projected TO->TO prerequisite "
                    f"graph: {n_concept_edges} concept prereq edge(s) -> "
                    f"{to_edges} direct TO edge(s) -> {n_transitive_pairs} "
                    f"transitive (dependent, prerequisite) pair(s) over {n_tos} "
                    f"TOs; {n_violations} dependent-before-prerequisite "
                    f"violation(s) ({n_direct_violations} on a direct edge, "
                    f"{n_violations - n_direct_violations} multi-hop only). "
                    "Warning-day-1: enumerated but does not yet block "
                    "(calibration-deferred critical-flip)."
                ),
                ml_features={
                    "n_terminal_objectives": int(n_tos),
                    "to_edges": int(to_edges),
                    "n_violations": int(n_violations),
                    "n_transitive_prereq_pairs": int(n_transitive_pairs),
                    "n_direct_violations": int(n_direct_violations),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("prereq_sequencing: decision capture raised: %s", exc)


__all__ = ["PrereqSequencingValidator"]
