"""Prerequisite-sequencing validator (warning-day-1).

Companion gate to :mod:`lib.generation.prereq_sequencer`. Reads the emitted
terminal-objective ordering + the archived ``concept_graph_semantic.json``,
projects the concept ``prerequisite`` edges onto the same TO->TO dependency
graph the sequencer uses, and flags ``PREREQ_ORDER_VIOLATION`` (warning) when a
DEPENDENT TO precedes its PREREQUISITE TO in the emitted order — i.e. a
prerequisite the sequencer could not honor (a cycle-broken / unsatisfiable
edge, or a run where the sequencer was off).

Default OFF / no graph / no edges → ``PREREQ_SEQUENCING_DISABLED`` info issue +
``passed=True`` (byte-stable no-op). Anti-fabrication: only audits real TO
edges projected from real ``key_concepts`` + real prereq edges; an edge whose
endpoints map to no TO is ignored. ``# TODO(calibration)`` deferred
critical-flip.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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

        to_edges, n_concept_edges, ignored = _build_to_edges(tos, graph)
        if not to_edges:
            return self._skip(
                gate_id, "PREREQ_SEQUENCING_NO_EDGES",
                "no TO->TO prerequisite edges projected; nothing to audit.",
            )

        position = {_to_id(t): i for i, t in enumerate(tos)}
        issues: List[GateIssue] = []
        n_violations = 0
        for pre_to, dep_to, conf in to_edges:
            p_pre = position.get(pre_to)
            p_dep = position.get(dep_to)
            if p_pre is None or p_dep is None:
                continue
            # Prerequisite must precede dependent. A dependent BEFORE its
            # prerequisite is a residual (cycle-broken / unsatisfied) violation.
            if p_dep < p_pre:
                n_violations += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="PREREQ_ORDER_VIOLATION",
                            message=(
                                f"dependent objective {dep_to!r} (pos {p_dep}) "
                                f"precedes its prerequisite {pre_to!r} (pos "
                                f"{p_pre}) in the emitted order (edge confidence "
                                f"{conf:.2f}); the prerequisite topic is taught "
                                f"AFTER the topic that depends on it."
                            ),
                            location=f"{dep_to} before {pre_to}",
                            suggestion=(
                                "Enable ED4ALL_PREREQ_SEQUENCING, or resolve the "
                                "prerequisite cycle so the prerequisite topic "
                                "precedes its dependent."
                            ),
                        )
                    )

        self._emit_decision(
            inputs.get("decision_capture") or self._decision_capture,
            n_tos=len(tos),
            to_edges=len(to_edges),
            n_concept_edges=n_concept_edges,
            n_violations=n_violations,
        )

        # Warning-day-1: passed stays True.
        #
        # TODO(calibration): flip PREREQ_ORDER_VIOLATION to critical (severity:
        # critical + on_fail: block, set passed=n_violations==0) AFTER
        # scripts/calibration_harness.py confirms the FP rate on >=2 corpora.
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=round(1.0 - n_violations / max(1, len(to_edges)), 4),
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
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="validation_result",
                decision=(
                    f"prerequisite_sequencing violations={n_violations}/"
                    f"{to_edges} TO edges"
                ),
                rationale=(
                    "Audited the emitted terminal-objective order against the "
                    f"projected TO->TO prerequisite graph: {n_concept_edges} "
                    f"concept prereq edge(s) -> {to_edges} TO edge(s) over "
                    f"{n_tos} TOs; {n_violations} dependent-before-prerequisite "
                    "violation(s). Warning-day-1: enumerated but does not yet "
                    "block (calibration-deferred critical-flip)."
                ),
                ml_features={
                    "n_terminal_objectives": int(n_tos),
                    "to_edges": int(to_edges),
                    "n_violations": int(n_violations),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("prereq_sequencing: decision capture raised: %s", exc)


__all__ = ["PrereqSequencingValidator"]
