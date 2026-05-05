"""
Courseforge top-level validation-report aggregator (Worker W5).

Walks every per-phase ``report.json`` (currently emitted by
``inter_tier_validation`` + ``post_rewrite_validation``) plus the
in-memory gate-result chain accumulated by the workflow runner across
phases that don't write their own JSON report (``packaging``,
``libv2_archival``, etc.), normalises each gate result to a unified
shape, and writes a single top-level
``<project_path>/courseforge_validation_report.json``.

Without the aggregator an operator has to manually open four-plus
per-phase report files to assemble a per-block / per-phase pass/fail
picture. The aggregator does this once, deterministically, post-loop
in :meth:`MCP.core.workflow_runner.WorkflowRunner.run_workflow`.

Schema (mirrors the Worker W5 spec — no separate JSON schema lands for
the first commit; the Python dataclasses are the source of truth):

::

    {
      "schema_version": "1.0",
      "course_code": "<course_name>",
      "run_id": "<workflow_id>",
      "generated_at": "<iso8601>",
      "status": "pass" | "fail",
      "summary": {
        "total_gates": <int>,
        "passed_count": <int>,
        "failed_count": <int>,
        "warning_count": <int>,
        "skipped_count": <int>,
      },
      "blocking_failures": [
        {phase, gate_id, severity, code, message},
        ...
      ],
      "warnings": [
        {phase, gate_id, severity, code, message},
        ...
      ],
      "per_phase": [
        {
          "phase": "<phase_name>",
          "report_path": "<absolute_path | null>",
          "gates": [
            {gate_id, severity, passed, action, issue_count, top_issues[]},
            ...
          ]
        },
        ...
      ]
    }

Top-level ``status`` rules:
* ``"fail"`` when *any* normalised gate has ``severity == "critical"``
  AND ``passed is False``.
* ``"pass"`` otherwise (warnings + skipped gates do not block).

The aggregator never raises on a missing per-phase report — phases
that legitimately don't emit a JSON report (e.g. ``dart_conversion``)
still surface in ``per_phase`` via their in-memory ``gate_results`` if
provided, otherwise are skipped silently and contribute to
``summary.skipped_count`` only when both signals are absent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"
TOP_ISSUES_LIMIT = 5

# Phases that own a known per-phase ``report.json`` writer in
# ``MCP.core.workflow_runner.WorkflowRunner._write_validation_report``.
# Maps phase_name -> relative path (under project_path) of report.json.
_KNOWN_REPORT_PATHS: Tuple[Tuple[str, str], ...] = (
    ("inter_tier_validation", "02_validation_report/report.json"),
    (
        "post_rewrite_validation",
        "04_rewrite/02_validation_report/report.json",
    ),
)


@dataclass(frozen=True)
class _NormalisedGate:
    """Unified per-gate row before it lands in the aggregator output."""

    phase: str
    gate_id: str
    severity: str
    passed: bool
    action: Optional[str]
    issue_count: int
    top_issues: List[Dict[str, Any]] = field(default_factory=list)
    code: Optional[str] = None
    message: Optional[str] = None

    def to_per_phase_entry(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "severity": self.severity,
            "passed": self.passed,
            "action": self.action,
            "issue_count": self.issue_count,
            "top_issues": list(self.top_issues),
        }

    def to_summary_entry(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "gate_id": self.gate_id,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


class CourseforgeValidationReport:
    """Aggregate per-phase validation reports + gate results into one JSON.

    Parameters
    ----------
    project_path:
        Absolute path of the Courseforge project export root
        (``Courseforge/exports/PROJ-<course>-<timestamp>``). Per-phase
        ``report.json`` files are resolved relative to this root.
    phase_outputs:
        ``WorkflowRunner.run_workflow``'s accumulated
        ``phase_outputs`` map. Each entry may carry a private
        ``_gate_results`` list (a sequence of dict-shaped
        ``GateResult`` payloads) so phases that don't emit a JSON
        report still contribute to the aggregator. The aggregator
        treats this map as read-only.
    course_code:
        Operator-facing course code (e.g. ``PHYS_101``). Surfaces as
        the top-level ``course_code`` field for at-a-glance reading.
    run_id:
        Workflow ID (e.g. ``WF-20260505-abc12345``). Surfaces as the
        top-level ``run_id`` field so cross-run diffs key cleanly.
    """

    def __init__(
        self,
        project_path: Path,
        phase_outputs: Mapping[str, Mapping[str, Any]],
        *,
        course_code: str = "",
        run_id: str = "",
    ) -> None:
        self.project_path = Path(project_path)
        self.phase_outputs = phase_outputs or {}
        self.course_code = course_code or ""
        self.run_id = run_id or ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Walk all signals and build the unified report dict."""
        per_phase: List[Dict[str, Any]] = []
        blocking: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0
        warning_count = 0
        skipped_count = 0

        seen_phases: set = set()

        # 1. Walk known per-phase JSON report files.
        for phase, rel_path in _KNOWN_REPORT_PATHS:
            seen_phases.add(phase)
            report_path = self.project_path / rel_path
            phase_entry = self._build_phase_entry_from_report(
                phase, report_path
            )
            if phase_entry is None:
                # Report file missing on disk — fall through to in-
                # memory ``_gate_results`` if present, else mark as
                # skipped.
                in_mem = self._build_phase_entry_from_memory(phase)
                if in_mem is None:
                    skipped_count += 1
                    per_phase.append({
                        "phase": phase,
                        "report_path": None,
                        "gates": [],
                        "skipped": True,
                        "skip_reason": (
                            "missing_report_file_and_in_memory_gates"
                        ),
                    })
                    continue
                per_phase.append(in_mem)
                gates_payload = in_mem["gates"]
            else:
                per_phase.append(phase_entry)
                gates_payload = phase_entry["gates"]

            (
                passed_count,
                failed_count,
                warning_count,
                blocking,
                warnings,
            ) = self._tally(
                gates_payload,
                phase,
                passed_count,
                failed_count,
                warning_count,
                blocking,
                warnings,
            )

        # 2. Walk in-memory gate_results for phases that don't own a
        # known report file (packaging, libv2_archival, etc.).
        for phase_name, phase_payload in self.phase_outputs.items():
            if phase_name in seen_phases:
                continue
            entry = self._build_phase_entry_from_memory(phase_name)
            if entry is None:
                # Phase ran but emitted no gates — not interesting for
                # the aggregator; skip silently (don't bump
                # skipped_count, since "no gates declared" isn't the
                # same as "couldn't read a known report").
                continue
            per_phase.append(entry)
            (
                passed_count,
                failed_count,
                warning_count,
                blocking,
                warnings,
            ) = self._tally(
                entry["gates"],
                phase_name,
                passed_count,
                failed_count,
                warning_count,
                blocking,
                warnings,
            )

        total_gates = passed_count + failed_count + warning_count
        status = "fail" if blocking else "pass"

        return {
            "schema_version": SCHEMA_VERSION,
            "course_code": self.course_code,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": status,
            "summary": {
                "total_gates": total_gates,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "warning_count": warning_count,
                "skipped_count": skipped_count,
            },
            "blocking_failures": blocking,
            "warnings": warnings,
            "per_phase": per_phase,
        }

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved absolute path on success. Raises ``OSError``
        on filesystem failure — the caller wraps the call in
        try/except so an aggregator failure does not abort the
        workflow (aggregation is best-effort; the per-phase reports
        remain the source of truth).
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
    # Helpers
    # ------------------------------------------------------------------
    def _build_phase_entry_from_report(
        self, phase: str, report_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Read a per-phase ``report.json`` and project to per_phase shape.

        Returns ``None`` when the report is missing / unreadable so the
        caller can fall back to in-memory gate_results.
        """
        if not report_path.exists():
            return None
        try:
            raw = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "courseforge_validation_report: cannot read %s: %s",
                report_path, exc,
            )
            return None

        # Phase 5 ``report.json`` shape from
        # ``WorkflowRunner._write_validation_report`` is per-block, but
        # every per_block entry carries the same ``gate_results`` chain
        # summary (gate_id / action / passed / issue_count). Pull the
        # chain off the first per_block entry; if there are no blocks
        # the chain is empty and the gates list collapses to [].
        per_block = raw.get("per_block") or []
        chain: Sequence[Mapping[str, Any]] = []
        if per_block:
            first = per_block[0]
            if isinstance(first, Mapping):
                chain = first.get("gate_results") or []

        # Two-pass router gates default to critical severity (the seam
        # is the architecture's safety net — every CURIE / source-ref /
        # objective gate that fires on the inter-tier or post-rewrite
        # seam blocks the workflow on failure). Per-block report.json
        # doesn't carry severity per gate, so we derive it from the
        # gate_id naming convention: warning-style gates are explicitly
        # tagged in workflows.yaml; everything else is critical.
        gates: List[Dict[str, Any]] = []
        for gr in chain:
            if not isinstance(gr, Mapping):
                continue
            normalised = self._normalise_two_pass_gate(phase, gr, raw)
            gates.append(normalised.to_per_phase_entry())

        return {
            "phase": phase,
            "report_path": str(report_path),
            "gates": gates,
            "report_summary": {
                "total_blocks": raw.get("total_blocks"),
                "passed": raw.get("passed"),
                "failed": raw.get("failed"),
                "escalated": raw.get("escalated"),
            },
        }

    def _build_phase_entry_from_memory(
        self, phase: str
    ) -> Optional[Dict[str, Any]]:
        """Build per_phase entry from a phase's in-memory ``_gate_results``."""
        phase_payload = self.phase_outputs.get(phase) or {}
        gate_results = phase_payload.get("_gate_results") or []
        if not gate_results:
            return None

        gates: List[Dict[str, Any]] = []
        for gr in gate_results:
            if not isinstance(gr, Mapping):
                continue
            normalised = self._normalise_full_gate(phase, gr)
            gates.append(normalised.to_per_phase_entry())

        return {
            "phase": phase,
            "report_path": None,
            "gates": gates,
        }

    def _normalise_full_gate(
        self, phase: str, gate_result: Mapping[str, Any]
    ) -> _NormalisedGate:
        """Normalise a full ``GateResult.to_dict()`` payload."""
        issues = list(gate_result.get("issues") or [])
        top_issues = [
            self._summarise_issue(i)
            for i in issues[:TOP_ISSUES_LIMIT]
            if isinstance(i, Mapping)
        ]
        # Severity falls back to "critical" so unknown gates fail closed
        # in the blocking_failures list. Validators that want to be
        # surfaced as warnings set severity explicitly.
        severity = str(gate_result.get("severity") or "critical").lower()
        first = issues[0] if issues and isinstance(issues[0], Mapping) else None

        return _NormalisedGate(
            phase=phase,
            gate_id=str(gate_result.get("gate_id") or "unknown"),
            severity=severity,
            passed=bool(gate_result.get("passed", True)),
            action=gate_result.get("action"),
            issue_count=len(issues),
            top_issues=top_issues,
            code=(first or {}).get("code") if first else None,
            message=(first or {}).get("message") if first else None,
        )

    def _normalise_two_pass_gate(
        self,
        phase: str,
        gate_result: Mapping[str, Any],
        raw_report: Mapping[str, Any],
    ) -> _NormalisedGate:
        """Normalise a two-pass ``report.json`` per-block gate summary.

        The Phase 5 ``report.json`` chain summary only carries
        ``{gate_id, action, passed, issue_count}``. Severity is derived
        from the gate_id naming convention (the four shape-discriminating
        ``Block*`` gates are critical by construction); top_issues is
        empty because the per-block report doesn't surface raw issue
        bodies.
        """
        gate_id = str(gate_result.get("gate_id") or "unknown")
        # Every two-pass router gate is critical by architecture (the
        # seam blocks on failure). When new warning-style gates land
        # they should be explicitly listed here.
        severity = "critical"
        passed = bool(gate_result.get("passed", True))
        # Folded report.json chain only stores issue_count. Synthesise
        # a code/message from the report's failed/escalated counts when
        # the chain says passed=False, so blocking_failures rows still
        # carry actionable text.
        message = None
        if not passed:
            failed = raw_report.get("failed", 0)
            escalated = raw_report.get("escalated", 0)
            message = (
                f"Two-pass gate '{gate_id}' rejected blocks "
                f"(failed={failed}, escalated={escalated})."
            )
        return _NormalisedGate(
            phase=phase,
            gate_id=gate_id,
            severity=severity,
            passed=passed,
            action=gate_result.get("action"),
            issue_count=int(gate_result.get("issue_count") or 0),
            top_issues=[],
            code=(f"TWO_PASS_{gate_id.upper()}_FAIL" if not passed else None),
            message=message,
        )

    @staticmethod
    def _summarise_issue(issue: Mapping[str, Any]) -> Dict[str, Any]:
        """Project a GateIssue dict to the top_issues row shape."""
        return {
            "severity": issue.get("severity"),
            "code": issue.get("code"),
            "message": issue.get("message"),
            "location": issue.get("location"),
        }

    @staticmethod
    def _tally(
        gates: Iterable[Mapping[str, Any]],
        phase: str,
        passed_count: int,
        failed_count: int,
        warning_count: int,
        blocking: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]],
    ) -> Tuple[int, int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Update counters + blocking/warnings buckets for a phase's gates."""
        for gate in gates:
            severity = (gate.get("severity") or "critical").lower()
            passed = bool(gate.get("passed", True))
            if passed:
                passed_count += 1
                continue
            # Failed gate: bucket by severity.
            top_issue = (gate.get("top_issues") or [{}])[0] if gate.get(
                "top_issues"
            ) else {}
            row = {
                "phase": phase,
                "gate_id": gate.get("gate_id"),
                "severity": severity,
                "code": top_issue.get("code") if top_issue else None,
                "message": top_issue.get("message") if top_issue else None,
            }
            if severity == "warning":
                warning_count += 1
                warnings.append(row)
            else:
                # critical / unknown / info(failed) all fail closed.
                failed_count += 1
                blocking.append(row)
        return passed_count, failed_count, warning_count, blocking, warnings
