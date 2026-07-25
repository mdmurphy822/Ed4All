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

Schema bump 1.0 -> 1.1 (GPT Feedback v2 Wave 2 W2.A — additive only):

* ``per_block_results[]`` rolls up the per-block JSONL files
  (``02_validation_report/blocks_validated.jsonl`` /
  ``blocks_failed.jsonl`` and the post-rewrite siblings under
  ``04_rewrite/``) so an operator can see one row per block / per
  phase without grepping four JSONL files.
* ``source_grounding_results`` projects gates that scored
  source-claim grounding (``*SourceGroundingValidator`` /
  ``*RetrievalGroundingValidator``).
* ``accessibility_results`` projects ``WCAGValidator`` rows with a
  rolled-up per-page issue count when present.
* ``statistical_semantic_results`` projects ``*SimilarityValidator``
  rows + the ``EMBEDDING_DEPS_MISSING`` warning rate (the informational
  signal that the ``[embedding]`` extras were absent at run-time).
* ``manifest_hash_results`` projects
  ``LibV2ManifestValidator`` / ``LibV2ModelValidator`` rows + a
  passthrough of the seven ``provenance.*_hash`` fields read from
  ``LibV2/courses/<slug>/models/<id>/model_card.json`` when present.
* ``final_promotion_decision`` is the new top-level enum from
  ``schemas/governance/course_status.schema.json`` — one of
  ``failed | non_certified_archive | certified_accessible |
  certified_instructional | certified_trainable``. Carries
  ``rationale`` + ``contributing_gate_ids[]`` so the operator sees
  *why* a course landed in each bucket. Wave 1's
  ``lib.governance.course_status.compose_course_status`` only
  implements the ``failed`` and ``non_certified_archive`` branches;
  the aggregator falls back to a heuristic for the three
  ``certified_*`` branches and stamps a TODO pointing at Wave 3 G1
  (master promotion-chain aggregator) which will drive the canonical
  composition.

Existing 1.0 readers parse 1.1 output unchanged: every new key is
additive and no 1.0 key has been renamed.

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
that legitimately don't emit a JSON report (e.g. ``semantik_conversion``)
still surface in ``per_phase`` via their in-memory ``gate_results`` if
provided, otherwise are skipped silently and contribute to
``summary.skipped_count`` only when both signals are absent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.1"
TOP_ISSUES_LIMIT = 5

# Per-block JSONL writer locations (relative to project_path). Inter-tier
# files live alongside the inter-tier ``report.json``; the post-rewrite
# siblings live under ``04_rewrite/02_validation_report/``. Sourced from
# ``MCP/tools/pipeline_tools.py::_run_inter_tier_validation`` and
# ``_run_post_rewrite_validation`` respectively (file names + parent
# directories).
_PER_BLOCK_JSONL_PATHS: Tuple[Tuple[str, str, str], ...] = (
    (
        "inter_tier_validation",
        "02_validation_report/blocks_validated.jsonl",
        "passed",
    ),
    (
        "inter_tier_validation",
        "02_validation_report/blocks_failed.jsonl",
        "failed",
    ),
    (
        "post_rewrite_validation",
        "04_rewrite/02_validation_report/blocks_validated.jsonl",
        "passed",
    ),
    (
        "post_rewrite_validation",
        "04_rewrite/02_validation_report/blocks_failed.jsonl",
        "failed",
    ),
)

# Semantic-bucket dispatch tables (gate_id substrings + validator-class
# suffixes). The aggregator projects matching rows into the new top-level
# bucket sub-objects so an operator can read one bucket at a time without
# scanning every per_phase entry. Membership is union-style: a gate row
# appears in a bucket if EITHER its gate_id matches one of the substrings
# OR its validator_name (when carried on the in-memory ``_gate_results``
# row) matches one of the class suffixes (case-insensitive).
_SOURCE_GROUNDING_GATE_ID_SUBSTRINGS: Tuple[str, ...] = (
    "_source_",
    "_grounding",
    "source_refs",
    "content_grounding",
)
_SOURCE_GROUNDING_VALIDATOR_SUFFIXES: Tuple[str, ...] = (
    "sourcegroundingvalidator",
    "retrievalgroundingvalidator",
    "contentgroundingvalidator",
    "sourcerefsvalidator",
    "pagesourcerefvalidator",
)

_ACCESSIBILITY_GATE_IDS: Tuple[str, ...] = (
    "wcag_compliance",
    "wcag_aa_compliance",
)
_ACCESSIBILITY_VALIDATOR_SUFFIXES: Tuple[str, ...] = (
    "wcagvalidator",
)

_STATISTICAL_SEMANTIC_VALIDATOR_SUFFIXES: Tuple[str, ...] = (
    "similarityvalidator",
)
_STATISTICAL_SEMANTIC_GATE_ID_SUBSTRINGS: Tuple[str, ...] = (
    "_similarity",
    "objective_assessment_similarity",
    "concept_example_similarity",
    "objective_roundtrip_similarity",
)
# Informational "embedding deps were absent at run-time" sentinel emitted
# by the four similarity validators when the ``[embedding]`` extras are
# missing. Aggregator surfaces the rate so an operator sees how much of
# the statistical-tier signal was degraded for the run.
_EMBEDDING_DEPS_MISSING_CODE = "EMBEDDING_DEPS_MISSING"

_MANIFEST_HASH_GATE_IDS: Tuple[str, ...] = (
    "libv2_manifest",
    "libv2_model",
    "kg_quality_report",
)
_MANIFEST_HASH_VALIDATOR_SUFFIXES: Tuple[str, ...] = (
    "libv2manifestvalidator",
    "libv2modelvalidator",
    "kgqualityvalidator",
)

# The seven ``provenance.*_hash`` fields the aggregator passes through from
# ``model_card.json``. Source: ``schemas/models/model_card.schema.json``
# under ``properties.provenance.required``.
_MODEL_CARD_PROVENANCE_HASH_FIELDS: Tuple[str, ...] = (
    "chunks_hash",
    "pedagogy_graph_hash",
    "instruction_pairs_hash",
    "preference_pairs_hash",
    "concept_graph_hash",
    "vocabulary_ttl_hash",
    "holdout_graph_hash",
)

# Final-promotion-decision enum (canonical: schemas/governance/course_status.schema.json).
_FINAL_PROMOTION_DECISIONS: Tuple[str, ...] = (
    "failed",
    "non_certified_archive",
    "certified_accessible",
    "certified_instructional",
    "certified_trainable",
)

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
        entry: Dict[str, Any] = {
            "gate_id": self.gate_id,
            "severity": self.severity,
            "passed": self.passed,
            "action": self.action,
            "issue_count": self.issue_count,
            "top_issues": list(self.top_issues),
        }
        # ADDITIVE, and only when populated: `_tally` falls back to these when
        # a gate carries no per-issue rows (the two-pass chain summary never
        # does), so a blocking row keeps actionable text instead of nulls. A
        # gate with neither emits the exact pre-existing key set.
        if self.code is not None:
            entry["code"] = self.code
        if self.message is not None:
            entry["message"] = self.message
        return entry

    def to_summary_entry(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "gate_id": self.gate_id,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


@lru_cache(maxsize=1)
def _configured_gate_severities() -> Dict[Tuple[str, str], str]:
    """``(phase, gate_id) -> severity`` read from ``config/workflows.yaml``.

    The gate's CONFIGURED severity is the authority on whether a failure
    blocks. Deriving it here instead of defaulting to ``"critical"`` is what
    keeps a warning-configured gate out of ``blocking_failures`` (and so out
    of ``final_promotion_decision`` / ``course_status``).

    Keyed by ``(phase, gate_id)`` because the same gate legitimately carries
    different severities in different phases (``assessment_quality`` is a
    warning at ``assessment_synthesis`` and critical at
    ``trainforge_assessment``). When two workflows disagree on the SAME
    ``(phase, gate_id)`` the entry resolves to ``"critical"`` — an ambiguous
    gate fails closed rather than silently downgrading.

    Best-effort: an unreadable/absent config yields an empty map and every
    caller falls back to its previous default.
    """
    severities: Dict[Tuple[str, str], str] = {}
    try:
        import yaml  # noqa: PLC0415 — optional at import time

        from lib.paths import PROJECT_ROOT  # noqa: PLC0415

        raw = yaml.safe_load(
            (Path(PROJECT_ROOT) / "config" / "workflows.yaml").read_text(
                encoding="utf-8"
            )
        )
    except Exception:  # noqa: BLE001 — config read is best-effort
        return severities
    if not isinstance(raw, dict):
        return severities
    for wf in (raw.get("workflows") or {}).values():
        if not isinstance(wf, dict):
            continue
        for phase in wf.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            phase_name = str(phase.get("name") or "")
            for gate in phase.get("validation_gates") or []:
                if not isinstance(gate, dict):
                    continue
                gate_id = str(gate.get("gate_id") or "")
                if not phase_name or not gate_id:
                    continue
                sev = str(gate.get("severity") or "critical").strip().lower()
                key = (phase_name, gate_id)
                prior = severities.get(key)
                # Disagreement across workflows -> fail closed.
                severities[key] = "critical" if (prior and prior != sev) else sev
    return severities


def _configured_severity(phase: str, gate_id: str) -> Optional[str]:
    """Configured severity for one gate, or ``None`` when not in config."""
    return _configured_gate_severities().get((str(phase), str(gate_id)))


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
        libv2_root: Optional[Path] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.project_path = Path(project_path)
        self.phase_outputs = phase_outputs or {}
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        # Optional LibV2 root override. When unset the aggregator falls
        # back to ``<repo_root>/LibV2/courses/<course_slug>``; the
        # override is mostly useful in tests + ops topologies that mount
        # LibV2 elsewhere.
        self.libv2_root = Path(libv2_root) if libv2_root else None
        # Optional ``DecisionCapture`` instance. When wired the
        # aggregator emits one ``courseforge_validation_aggregated``
        # event per ``build()`` call so the audit trail records the
        # resolved final-promotion-decision + the contributing gate IDs.
        self.decision_capture = decision_capture

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

        # GPT Feedback v2 Wave 2 W2.A — additive sub-objects + the
        # final-promotion-decision rollup. Each helper is read-only over
        # ``per_phase`` / ``self.phase_outputs`` / ``self.project_path``
        # and never raises (any IO error degrades to an empty bucket
        # rather than failing the aggregator build).
        per_block_results = self._build_per_block_results()
        source_grounding_results = self._build_bucket_results(
            per_phase,
            gate_id_substrings=_SOURCE_GROUNDING_GATE_ID_SUBSTRINGS,
            validator_suffixes=_SOURCE_GROUNDING_VALIDATOR_SUFFIXES,
        )
        accessibility_results = self._build_accessibility_results(per_phase)
        statistical_semantic_results = (
            self._build_statistical_semantic_results(per_phase)
        )
        manifest_hash_results = self._build_manifest_hash_results(per_phase)

        final_promotion_decision = self._compose_final_promotion_decision(
            status=status,
            blocking_failures=blocking,
            warnings=warnings,
            accessibility_results=accessibility_results,
            manifest_hash_results=manifest_hash_results,
            source_grounding_results=source_grounding_results,
            statistical_semantic_results=statistical_semantic_results,
        )

        report = {
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
            # --- Wave 2 W2.A additive sub-objects -----------------------
            "per_block_results": per_block_results,
            "source_grounding_results": source_grounding_results,
            "accessibility_results": accessibility_results,
            "statistical_semantic_results": statistical_semantic_results,
            "manifest_hash_results": manifest_hash_results,
            "final_promotion_decision": final_promotion_decision,
        }

        # Best-effort decision-capture emit; never raises into the
        # aggregator build path.
        self._emit_aggregator_decision(report)

        return report

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
        # Severity precedence: the gate payload's own value > the CONFIGURED
        # severity in config/workflows.yaml > "critical" (unknown gates fail
        # closed in blocking_failures). Consulting the config is what stops a
        # payload that simply omits the key from being reported as critical
        # when the workflow configured it as a warning.
        severity = str(
            gate_result.get("severity")
            or _configured_severity(phase, gate_result.get("gate_id") or "")
            or "critical"
        ).lower()
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
        # The per-block report.json chain summary carries no severity, so it
        # is read from the gate's CONFIGURED severity in config/workflows.yaml
        # (falling back to "critical" for a gate absent from config, which
        # keeps an unknown gate failing closed).
        #
        # This previously hardcoded "critical" with a note that warning-style
        # gates "should be explicitly listed here" — a list that was never
        # added. The result: five gates configured `severity: warning,
        # on_fail: warn` landed in blocking_failures and drove
        # final_promotion_decision -> "failed" on a course whose own
        # report.json showed 3037/3037 blocks passed, 0 failed, 0 escalated.
        severity = _configured_severity(phase, gate_id) or "critical"
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
            # Fall back to the gate's OWN synthesised code/message when it has
            # no per-issue rows. The two-pass chain summary carries only an
            # issue COUNT (top_issues is always empty there), so reading
            # top_issues alone produced blocking_failures rows with null code
            # AND null message — a "critical failure" with no reason attached,
            # which is close to untriageable.
            row = {
                "phase": phase,
                "gate_id": gate.get("gate_id"),
                "severity": severity,
                "code": (top_issue.get("code") if top_issue else None)
                or gate.get("code"),
                "message": (top_issue.get("message") if top_issue else None)
                or gate.get("message"),
            }
            if severity == "warning":
                warning_count += 1
                warnings.append(row)
            else:
                # critical / unknown / info(failed) all fail closed.
                failed_count += 1
                blocking.append(row)
        return passed_count, failed_count, warning_count, blocking, warnings

    # ------------------------------------------------------------------
    # Wave 2 W2.A — additive sub-objects
    # ------------------------------------------------------------------

    def _build_per_block_results(self) -> List[Dict[str, Any]]:
        """Roll up the per-block ``blocks_validated.jsonl`` / ``blocks_failed.jsonl``.

        Walks the four canonical files emitted by the inter-tier and
        post-rewrite pipeline-tools handlers and projects each entry to
        the flat row shape::

            {block_id, block_type, page_id, phase, status,
             gate_chain[], escalation_marker?}

        Missing files degrade silently (the pipeline can legitimately
        skip a phase). Malformed JSONL rows are skipped (best-effort —
        the per-phase ``report.json`` aggregator above has the same
        posture).
        """
        rows: List[Dict[str, Any]] = []
        # Build the inter-tier / post-rewrite gate-chain summary keyed
        # by phase so we can carry the chain through to every per-block
        # row that came from that phase.
        gate_chain_by_phase: Dict[str, List[Dict[str, Any]]] = {}
        for phase, rel_path in _KNOWN_REPORT_PATHS:
            report_path = self.project_path / rel_path
            if not report_path.exists():
                continue
            try:
                raw = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            per_block_entries = raw.get("per_block") or []
            if not per_block_entries:
                continue
            first = per_block_entries[0]
            if isinstance(first, Mapping):
                chain = first.get("gate_results") or []
                gate_chain_by_phase[phase] = [
                    dict(g) for g in chain if isinstance(g, Mapping)
                ]

        for phase, rel_path, default_status in _PER_BLOCK_JSONL_PATHS:
            jsonl_path = self.project_path / rel_path
            if not jsonl_path.exists():
                continue
            try:
                lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.warning(
                    "courseforge_validation_report: cannot read %s: %s",
                    jsonl_path, exc,
                )
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, Mapping):
                    continue
                row: Dict[str, Any] = {
                    "block_id": entry.get("block_id"),
                    "block_type": entry.get("block_type"),
                    "page_id": entry.get("page_id"),
                    "phase": phase,
                    "status": default_status,
                    "gate_chain": gate_chain_by_phase.get(phase, []),
                }
                escalation_marker = entry.get("escalation_marker")
                if escalation_marker:
                    row["escalation_marker"] = escalation_marker
                rows.append(row)
        return rows

    def _build_bucket_results(
        self,
        per_phase: Sequence[Mapping[str, Any]],
        *,
        gate_id_substrings: Sequence[str] = (),
        validator_suffixes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Project per_phase gates into a semantic bucket sub-object.

        Returns a dict with ``gates[]`` (matching projected gate rows)
        and a small summary block. Used by source_grounding_results,
        accessibility_results, and statistical_semantic_results.
        Membership is union-style: a gate row appears in the bucket if
        EITHER its ``gate_id`` matches one of ``gate_id_substrings`` OR
        its ``validator_name`` (carried on the in-memory _gate_results
        rows) matches one of ``validator_suffixes`` (case-insensitive).
        """
        gates: List[Dict[str, Any]] = []
        # Walk per_phase first (the projected shape carries gate_id +
        # severity + passed but not validator_name; we cross-ref the
        # in-memory _gate_results to recover the class name).
        validator_lookup = self._build_validator_lookup()
        for phase_entry in per_phase:
            phase_name = phase_entry.get("phase")
            for gate in phase_entry.get("gates") or []:
                gate_id = str(gate.get("gate_id") or "")
                validator_name = validator_lookup.get(
                    (phase_name, gate_id), ""
                )
                if self._matches_bucket(
                    gate_id=gate_id,
                    validator_name=validator_name,
                    gate_id_substrings=gate_id_substrings,
                    validator_suffixes=validator_suffixes,
                ):
                    gates.append({
                        "phase": phase_name,
                        "gate_id": gate_id,
                        "severity": gate.get("severity"),
                        "passed": gate.get("passed"),
                        "action": gate.get("action"),
                        "issue_count": gate.get("issue_count"),
                        "validator_name": validator_name or None,
                    })
        passed = sum(1 for g in gates if g.get("passed"))
        failed = sum(1 for g in gates if g.get("passed") is False)
        return {
            "gates": gates,
            "summary": {
                "total": len(gates),
                "passed": passed,
                "failed": failed,
            },
        }

    def _build_validator_lookup(
        self,
    ) -> Dict[Tuple[Optional[str], str], str]:
        """Map ``(phase, gate_id)`` -> ``validator_name`` for in-memory gates.

        Per-phase JSON reports project to gate_ids without carrying the
        validator class name; we read the live ``_gate_results`` payload
        to recover that class so the bucket dispatch can fall back on
        validator-class membership when gate_id naming is ambiguous.
        """
        lookup: Dict[Tuple[Optional[str], str], str] = {}
        for phase_name, phase_payload in self.phase_outputs.items():
            for gr in phase_payload.get("_gate_results") or []:
                if not isinstance(gr, Mapping):
                    continue
                gid = str(gr.get("gate_id") or "")
                vname = str(gr.get("validator_name") or "")
                if gid and vname:
                    lookup[(phase_name, gid)] = vname
        return lookup

    @staticmethod
    def _matches_bucket(
        *,
        gate_id: str,
        validator_name: str,
        gate_id_substrings: Sequence[str],
        validator_suffixes: Sequence[str],
    ) -> bool:
        """Union-style membership test for the semantic-bucket dispatch."""
        gid_lower = gate_id.lower()
        for sub in gate_id_substrings:
            if sub and sub.lower() in gid_lower:
                return True
        if validator_name:
            vname_lower = validator_name.lower()
            for suffix in validator_suffixes:
                if suffix and vname_lower.endswith(suffix):
                    return True
        return False

    def _build_accessibility_results(
        self, per_phase: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """WCAGValidator projection + per-page issue rollup.

        Membership: gate_id in {``wcag_compliance``, ``wcag_aa_compliance``}
        OR validator class endswith ``WCAGValidator`` (case-insensitive).
        Also carries a ``per_page_issue_count`` rollup when the live
        ``_gate_results`` payload surfaces per-page issue locations.
        """
        bucket = self._build_bucket_results(
            per_phase,
            gate_id_substrings=_ACCESSIBILITY_GATE_IDS,
            validator_suffixes=_ACCESSIBILITY_VALIDATOR_SUFFIXES,
        )
        per_page_issues: Dict[str, int] = {}
        for phase_name, phase_payload in self.phase_outputs.items():
            for gr in phase_payload.get("_gate_results") or []:
                if not isinstance(gr, Mapping):
                    continue
                gid = str(gr.get("gate_id") or "")
                vname = str(gr.get("validator_name") or "")
                if not self._matches_bucket(
                    gate_id=gid,
                    validator_name=vname,
                    gate_id_substrings=_ACCESSIBILITY_GATE_IDS,
                    validator_suffixes=_ACCESSIBILITY_VALIDATOR_SUFFIXES,
                ):
                    continue
                for issue in gr.get("issues") or []:
                    if not isinstance(issue, Mapping):
                        continue
                    location = issue.get("location")
                    if isinstance(location, str) and location:
                        per_page_issues[location] = (
                            per_page_issues.get(location, 0) + 1
                        )
        bucket["per_page_issue_count"] = per_page_issues
        return bucket

    def _build_statistical_semantic_results(
        self, per_phase: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """``*SimilarityValidator`` projection + EMBEDDING_DEPS_MISSING rate.

        The four statistical-tier validators (``ObjectiveAssessmentSimilarity``,
        ``ConceptExampleSimilarity``, ``ObjectiveRoundtripSimilarity``,
        plus the ``CourseforgeOutlineShacl`` semantic check) all degrade to
        a warning-severity ``EMBEDDING_DEPS_MISSING`` GateIssue when the
        ``[embedding]`` extras are absent. We surface the rate so an
        operator sees how much of the statistical-tier signal was
        degraded for the run.
        """
        bucket = self._build_bucket_results(
            per_phase,
            gate_id_substrings=_STATISTICAL_SEMANTIC_GATE_ID_SUBSTRINGS,
            validator_suffixes=_STATISTICAL_SEMANTIC_VALIDATOR_SUFFIXES,
        )
        # Walk in-memory _gate_results to count EMBEDDING_DEPS_MISSING
        # warnings even when the gate itself technically passed.
        deps_missing_count = 0
        bucket_gate_count = bucket["summary"]["total"]
        for phase_name, phase_payload in self.phase_outputs.items():
            for gr in phase_payload.get("_gate_results") or []:
                if not isinstance(gr, Mapping):
                    continue
                gid = str(gr.get("gate_id") or "")
                vname = str(gr.get("validator_name") or "")
                if not self._matches_bucket(
                    gate_id=gid,
                    validator_name=vname,
                    gate_id_substrings=(
                        _STATISTICAL_SEMANTIC_GATE_ID_SUBSTRINGS
                    ),
                    validator_suffixes=(
                        _STATISTICAL_SEMANTIC_VALIDATOR_SUFFIXES
                    ),
                ):
                    continue
                for issue in gr.get("issues") or []:
                    if not isinstance(issue, Mapping):
                        continue
                    if issue.get("code") == _EMBEDDING_DEPS_MISSING_CODE:
                        deps_missing_count += 1
        bucket["embedding_deps_missing_count"] = deps_missing_count
        bucket["embedding_deps_missing_rate"] = (
            (deps_missing_count / bucket_gate_count)
            if bucket_gate_count > 0
            else 0.0
        )
        return bucket

    def _build_manifest_hash_results(
        self, per_phase: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """LibV2 manifest / model gates + passthrough of provenance hashes.

        Reads ``LibV2/courses/<slug>/models/<id>/model_card.json`` when
        present (the W1.C / Wave-1 model_card schema declares the seven
        ``provenance.*_hash`` fields as required) and surfaces those
        hashes verbatim so the audit trail can correlate the final
        decision with the seven-hash provenance fingerprint without
        re-reading the model card.
        """
        bucket = self._build_bucket_results(
            per_phase,
            gate_id_substrings=_MANIFEST_HASH_GATE_IDS,
            validator_suffixes=_MANIFEST_HASH_VALIDATOR_SUFFIXES,
        )
        bucket["provenance_hashes"] = self._read_provenance_hashes()
        return bucket

    def _read_provenance_hashes(self) -> Optional[Dict[str, Any]]:
        """Best-effort read of ``model_card.json::provenance.*_hash`` fields.

        Returns ``None`` when no model card can be located. Returns a
        dict mapping each of the seven canonical hash field names to
        the read value (or ``None`` if absent on the model card).
        """
        course_slug = self._resolve_course_slug()
        if not course_slug:
            return None
        # Resolve LibV2 root: explicit override > repo-root sibling.
        if self.libv2_root is not None:
            libv2_root = self.libv2_root
        else:
            # ``project_path`` is the Courseforge export root; LibV2
            # typically lives at the repo root next to Courseforge.
            try:
                libv2_root = (
                    self.project_path.parent.parent.parent / "LibV2"
                )
            except Exception:
                return None
        models_dir = libv2_root / "courses" / course_slug / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            return None
        # Pick the lexicographically last subdir (deterministic in the
        # absence of a "current" pointer; Wave 3 G1 tightens this to
        # the published pointer in ``model_pointers.json``).
        try:
            candidates = sorted(
                p for p in models_dir.iterdir() if p.is_dir()
            )
        except OSError:
            return None
        if not candidates:
            return None
        model_card_path = candidates[-1] / "model_card.json"
        if not model_card_path.exists():
            return None
        try:
            payload = json.loads(model_card_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        provenance = payload.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            return None
        return {
            field_name: provenance.get(field_name)
            for field_name in _MODEL_CARD_PROVENANCE_HASH_FIELDS
        }

    def _resolve_course_slug(self) -> Optional[str]:
        """Best-effort course slug for LibV2 model_card.json lookup.

        Returns the lowercased ``course_code`` when set; falls back to
        the ``project_path`` directory name's slug suffix when the
        course code is empty.
        """
        if self.course_code:
            return self.course_code.lower()
        # ``Courseforge/exports/PROJ-PHYS_101-20260505`` -> "phys_101"
        name = self.project_path.name
        # Strip ``PROJ-`` prefix + trailing timestamp if present.
        parts = name.split("-")
        if len(parts) >= 3 and parts[0].upper() == "PROJ":
            return parts[1].lower()
        return None

    # ------------------------------------------------------------------
    # final_promotion_decision composition
    # ------------------------------------------------------------------

    def _compose_final_promotion_decision(
        self,
        *,
        status: str,
        blocking_failures: Sequence[Mapping[str, Any]],
        warnings: Sequence[Mapping[str, Any]],
        accessibility_results: Mapping[str, Any],
        manifest_hash_results: Mapping[str, Any],
        source_grounding_results: Mapping[str, Any],
        statistical_semantic_results: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compute the canonical 5-value final-promotion-decision.

        Tries the canonical Wave 1 helper
        :func:`lib.governance.course_status.compose_course_status` first
        — that helper ships the ``failed`` and ``non_certified_archive``
        branches; the three ``certified_*`` branches raise
        :class:`NotImplementedError` until Wave 3 G1 lands the master
        promotion-chain aggregator that drives the canonical
        composition. The aggregator only honours the canonical helper's
        return for the ``failed`` branch; ``non_certified_archive`` is
        the helper's "arrows 1-5 pass, 6+ missing" stub which would
        short-circuit the heuristic before it can route the run to
        ``certified_accessible`` / ``certified_instructional`` /
        ``certified_trainable``. When the helper raises (the three
        ``certified_*`` branches always do in Wave 1) OR when it returns
        ``non_certified_archive``, the aggregator falls back to the
        heuristic decision-table below using the live aggregator output:

        * ANY critical-fail (blocking_failures non-empty)
          -> ``failed``.
        * Warnings only (no blocking, warnings non-empty)
          -> ``certified_accessible``.
        * Clean run with no detected training-data signal
          (``manifest_hash_results.provenance_hashes`` absent OR has no
          training-related hashes; no ``training_synthesis`` /
          ``trainforge_assessment`` phase ran)
          -> ``certified_instructional``.
        * Clean run with training-data signal present
          -> ``certified_trainable``.

        TODO(Wave 3 G1): replace the heuristic with the master
        promotion-chain aggregator output once
        ``compose_course_status`` ships its three ``certified_*``
        branches. The heuristic is intentionally conservative: a clean
        run without a model card lands at ``certified_instructional``,
        which is strictly less generous than ``certified_trainable``.
        """
        # Build per-arrow decisions skeleton from the aggregator output
        # so the canonical helper has something to consume. Wave 1's
        # helper only routes the ``failed`` / ``non_certified_archive``
        # branches; the three ``certified_*`` branches raise.
        contributing_gate_ids: List[str] = []
        rationale_parts: List[str] = []

        if blocking_failures:
            contributing_gate_ids = sorted({
                str(bf.get("gate_id"))
                for bf in blocking_failures
                if bf.get("gate_id")
            })

        per_arrow_decisions = self._derive_per_arrow_decisions(
            status=status,
            blocking_failures=blocking_failures,
            warnings=warnings,
        )

        decision: Optional[str] = None
        try:
            from lib.governance.course_status import compose_course_status
            canonical = compose_course_status(per_arrow_decisions)
            # Wave 1's helper only routes the ``failed`` /
            # ``non_certified_archive`` branches; the three
            # ``certified_*`` branches raise. The aggregator only honours
            # the canonical helper for ``failed`` because that branch is
            # the only one Wave 1 fully implements against live signals;
            # ``non_certified_archive`` is the helper's "arrows 1-5 pass,
            # 6+ missing" stub which would short-circuit before the
            # heuristic could route the run to ``certified_accessible`` /
            # ``certified_instructional`` / ``certified_trainable`` based
            # on actual training-data signal. TODO(Wave 3 G1): swap to
            # the master promotion-chain aggregator's full enum routing
            # once it lands.
            if canonical == "failed":
                decision = canonical
                rationale_parts.append(
                    "compose_course_status routed to 'failed' "
                    "via canonical helper"
                )
            else:
                decision = self._heuristic_final_promotion_decision(
                    blocking_failures=blocking_failures,
                    warnings=warnings,
                    manifest_hash_results=manifest_hash_results,
                )
                rationale_parts.append(
                    "compose_course_status returned "
                    f"{canonical!r} (Wave 1 stub for non-failed branches); "
                    f"heuristic fallback -> {decision!r}"
                )
        except NotImplementedError:
            # Heuristic fallback (Wave 2 W2.A — TODO Wave 3 G1).
            decision = self._heuristic_final_promotion_decision(
                blocking_failures=blocking_failures,
                warnings=warnings,
                manifest_hash_results=manifest_hash_results,
            )
            rationale_parts.append(
                "compose_course_status NotImplementedError on "
                f"certified_* branch; heuristic fallback -> {decision!r}"
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "compose_course_status raised non-NotImplementedError "
                "(falling back to heuristic): %s", exc,
            )
            decision = self._heuristic_final_promotion_decision(
                blocking_failures=blocking_failures,
                warnings=warnings,
                manifest_hash_results=manifest_hash_results,
            )
            rationale_parts.append(
                f"compose_course_status raised ({type(exc).__name__}); "
                f"heuristic fallback -> {decision!r}"
            )

        # Defensive enum guard.
        if decision not in _FINAL_PROMOTION_DECISIONS:
            decision = "failed"
            rationale_parts.append(
                "decision fell outside enum; coerced to 'failed'"
            )

        # Stamp dynamic signals in the rationale so it interpolates
        # the live counts and never falls below the 20-char floor.
        rationale_parts.append(
            f"blocking_count={len(blocking_failures)}, "
            f"warning_count={len(warnings)}, "
            f"accessibility_passed="
            f"{accessibility_results.get('summary', {}).get('passed', 0)}, "
            f"source_grounding_passed="
            f"{source_grounding_results.get('summary', {}).get('passed', 0)}, "
            f"statistical_passed="
            f"{statistical_semantic_results.get('summary', {}).get('passed', 0)}"
        )

        return {
            "value": decision,
            "rationale": "; ".join(rationale_parts),
            "contributing_gate_ids": contributing_gate_ids,
        }

    @staticmethod
    def _derive_per_arrow_decisions(
        *,
        status: str,
        blocking_failures: Sequence[Mapping[str, Any]],
        warnings: Sequence[Mapping[str, Any]],
    ) -> Dict[str, str]:
        """Coarse per-arrow mapping for the canonical Wave-1 compose helper.

        Wave 1's helper expects ``{arrow_name: promotion_decision}``;
        the canonical 9-arrow chain isn't yet wired (Wave 3 G1).
        For Wave 2 we synthesise a 5-arrow slice from the aggregator's
        bucket counts so the helper's ``failed`` /
        ``non_certified_archive`` branches still fire. The canonical
        decisions enum: ``"pass" | "warn" | "fail" | "missing"``.
        """
        # Single-arrow rollup for the accessible+instructional slice
        # (arrows 1-5 in the helper's vocabulary).
        if blocking_failures:
            verdict = "fail"
        elif warnings:
            verdict = "warn"
        else:
            verdict = "pass"
        return {
            "arrow1": verdict,
            "arrow2": verdict,
            "arrow3": verdict,
            "arrow4": verdict,
            "arrow5": verdict,
            "arrow6": "missing",
            "arrow7": "missing",
            "arrow8": "missing",
            "arrow9": "missing",
        }

    def _heuristic_final_promotion_decision(
        self,
        *,
        blocking_failures: Sequence[Mapping[str, Any]],
        warnings: Sequence[Mapping[str, Any]],
        manifest_hash_results: Mapping[str, Any],
    ) -> str:
        """Heuristic decision table (Wave 2 fallback for ``certified_*``).

        Mirrors the docstring on :meth:`_compose_final_promotion_decision`.
        Replaced by the canonical Wave 3 G1 master aggregator once that
        lands.
        """
        if blocking_failures:
            return "failed"
        if warnings:
            return "certified_accessible"
        # Clean run — bucket on training-data signal.
        if self._has_training_data_signal(manifest_hash_results):
            return "certified_trainable"
        return "certified_instructional"

    def _has_training_data_signal(
        self, manifest_hash_results: Mapping[str, Any]
    ) -> bool:
        """Return True when the run produced a training-data signal.

        Two routes:

        1. ``manifest_hash_results.provenance_hashes`` carries any of
           the three training-related hashes
           (``instruction_pairs_hash``, ``preference_pairs_hash``,
           ``holdout_graph_hash``).
        2. The run executed a training-related phase (``training_synthesis``
           or ``trainforge_assessment``) per ``self.phase_outputs``.
        """
        hashes = (manifest_hash_results or {}).get("provenance_hashes") or {}
        training_hash_keys = (
            "instruction_pairs_hash",
            "preference_pairs_hash",
            "holdout_graph_hash",
        )
        if isinstance(hashes, Mapping):
            for key in training_hash_keys:
                if hashes.get(key):
                    return True
        for phase_name in ("training_synthesis", "trainforge_assessment"):
            payload = self.phase_outputs.get(phase_name) or {}
            if payload.get("_completed"):
                return True
        return False

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------

    def _emit_aggregator_decision(self, report: Mapping[str, Any]) -> None:
        """Emit one ``courseforge_validation_aggregated`` decision per build.

        Best-effort: any failure in the capture path is logged and
        swallowed. Rationale interpolates dynamic signals (total gate
        count, per-bucket gate counts, the resolved
        ``final_promotion_decision`` value, contributing_gate_ids
        length) so it interpolates at least the 20-char floor.
        """
        capture = self.decision_capture
        if capture is None:
            return
        try:
            summary = report.get("summary") or {}
            fpd = report.get("final_promotion_decision") or {}
            decision_value = fpd.get("value", "unknown")
            contributing = fpd.get("contributing_gate_ids") or []
            source_grounding = report.get("source_grounding_results") or {}
            accessibility = report.get("accessibility_results") or {}
            statistical = report.get("statistical_semantic_results") or {}
            manifest_hash = report.get("manifest_hash_results") or {}
            per_block = report.get("per_block_results") or []

            rationale = (
                f"Courseforge aggregated verdict={decision_value}, "
                f"total_gates={summary.get('total_gates', 0)}, "
                f"blocking={len(report.get('blocking_failures') or [])}, "
                f"warnings={len(report.get('warnings') or [])}, "
                f"per_block_rows={len(per_block)}, "
                f"source_grounding_total="
                f"{source_grounding.get('summary', {}).get('total', 0)}, "
                f"accessibility_total="
                f"{accessibility.get('summary', {}).get('total', 0)}, "
                f"statistical_total="
                f"{statistical.get('summary', {}).get('total', 0)}, "
                f"manifest_hash_total="
                f"{manifest_hash.get('summary', {}).get('total', 0)}, "
                f"contributing_gate_ids_count={len(contributing)}"
            )
            capture.log_decision(
                decision_type="courseforge_validation_aggregated",
                decision=decision_value,
                rationale=rationale,
                context=str({
                    "course_code": report.get("course_code"),
                    "run_id": report.get("run_id"),
                    "status": report.get("status"),
                    "contributing_gate_ids": contributing,
                }),
                metrics={
                    "total_gates": summary.get("total_gates", 0),
                    "passed_count": summary.get("passed_count", 0),
                    "failed_count": summary.get("failed_count", 0),
                    "warning_count": summary.get("warning_count", 0),
                    "skipped_count": summary.get("skipped_count", 0),
                    "per_block_count": len(per_block),
                    "final_promotion_decision": decision_value,
                    "contributing_gate_ids_count": len(contributing),
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "courseforge_validation_aggregated: %s",
                exc,
            )
