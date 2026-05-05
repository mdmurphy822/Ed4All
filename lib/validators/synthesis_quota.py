"""Wave 110 / Phase D — SynthesisQuotaValidator.

Pre-synthesis advisory gate that estimates how many session dispatches
a run will need and warns when the estimate exceeds the configured
ceiling. Default severity is 'warning' so operators get awareness
without blocking; flip to 'critical' via inputs.severity to fail
closed for batch / unattended runs.

Estimate formula:
    estimated_dispatches = eligible_chunks * (instruction_variants + 1)

Where the "+ 1" accounts for the per-chunk preference pair.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


_DEFAULT_THRESHOLDS = {
    "max_estimated_dispatches": 1500,
}


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    eligible_chunks: int,
    instruction_variants: int,
    estimated_dispatches: int,
    ceiling: int,
    skip_reason: Optional[str] = None,
) -> None:
    """Emit one ``synthesis_quota_check`` decision per validate() call.

    H3 Wave W4: every over-ceiling / pass / soft-skip path emits one
    event so post-hoc replay can distinguish "course_dir missing /
    chunks.jsonl absent" (legitimate skip → passed=True) from
    "estimate exceeds ceiling" (warning or critical depending on
    configured severity).
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rationale = (
        f"synthesis_quota gate verdict: eligible_chunks="
        f"{eligible_chunks}, instruction_variants="
        f"{instruction_variants}, "
        f"estimated_dispatches={estimated_dispatches} "
        f"(ceiling={ceiling}); skip_reason={skip_reason or 'none'}; "
        f"failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "eligible_chunks": int(eligible_chunks),
        "instruction_variants": int(instruction_variants),
        "estimated_dispatches": int(estimated_dispatches),
        "ceiling": int(ceiling),
        "skip_reason": skip_reason,
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="synthesis_quota_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "synthesis_quota_check: %s",
            exc,
        )


class SynthesisQuotaValidator:
    name = "synthesis_quota"
    version = "1.0.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "synthesis_quota")
        capture = inputs.get("decision_capture") or self._decision_capture
        thresholds = dict(_DEFAULT_THRESHOLDS)
        for k, v in (inputs.get("thresholds") or {}).items():
            if k in thresholds:
                thresholds[k] = int(v)
        configured_severity = inputs.get("severity", "warning")
        if configured_severity not in ("warning", "critical"):
            configured_severity = "warning"
        ceiling = thresholds["max_estimated_dispatches"]
        variants = max(1, int(inputs.get("instruction_variants_per_chunk") or 1))

        course_dir = inputs.get("course_dir")
        if not course_dir:
            _emit_decision(
                capture, passed=True, code=None,
                eligible_chunks=0, instruction_variants=variants,
                estimated_dispatches=0, ceiling=ceiling,
                skip_reason="course_dir_missing",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )
        course_dir = Path(course_dir)
        # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
        from lib.libv2_storage import resolve_imscc_chunks_path
        chunks_path = resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
        if not chunks_path.exists():
            _emit_decision(
                capture, passed=True, code=None,
                eligible_chunks=0, instruction_variants=variants,
                estimated_dispatches=0, ceiling=ceiling,
                skip_reason="chunks_jsonl_absent",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        eligible = 0
        try:
            with chunks_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("learning_outcome_refs"):
                        eligible += 1
        except OSError:
            _emit_decision(
                capture, passed=True, code=None,
                eligible_chunks=0, instruction_variants=variants,
                estimated_dispatches=0, ceiling=ceiling,
                skip_reason="chunks_read_error",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        estimated = eligible * (variants + 1)

        issues: List[GateIssue] = []
        if estimated > ceiling:
            issues.append(GateIssue(
                severity=configured_severity,
                code="SYNTHESIS_QUOTA_OVER_CEILING",
                message=(
                    f"Estimated {estimated} session dispatches "
                    f"({eligible} eligible chunks × {variants + 1} "
                    f"calls/chunk) exceeds ceiling {ceiling}. "
                    f"Consider running --max-dispatches in stages, or "
                    f"raise the ceiling via inputs.thresholds."
                ),
                location=str(chunks_path),
            ))

        critical_issues = [i for i in issues if i.severity == "critical"]
        passed = not critical_issues
        # H3 W4: emit even on the over-ceiling-warning path so replay can
        # see the estimate even when the run wasn't blocked.
        failure_code = None
        if issues and not passed:
            failure_code = issues[0].code
        elif issues:
            # Warning-severity over-ceiling: passed stays True but we
            # still record the failure code in the metric payload so
            # replay can spot near-ceiling runs.
            failure_code = issues[0].code
        _emit_decision(
            capture, passed=passed, code=failure_code,
            eligible_chunks=eligible, instruction_variants=variants,
            estimated_dispatches=estimated, ceiling=ceiling,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - 0.1 * len(issues)),
            issues=issues,
        )
