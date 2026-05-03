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
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


_DEFAULT_THRESHOLDS = {
    "max_estimated_dispatches": 1500,
}


class SynthesisQuotaValidator:
    name = "synthesis_quota"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "synthesis_quota")
        thresholds = dict(_DEFAULT_THRESHOLDS)
        for k, v in (inputs.get("thresholds") or {}).items():
            if k in thresholds:
                thresholds[k] = int(v)
        configured_severity = inputs.get("severity", "warning")
        if configured_severity not in ("warning", "critical"):
            configured_severity = "warning"

        course_dir = inputs.get("course_dir")
        if not course_dir:
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
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        variants = max(1, int(inputs.get("instruction_variants_per_chunk") or 1))
        estimated = eligible * (variants + 1)
        ceiling = thresholds["max_estimated_dispatches"]

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
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - 0.1 * len(issues)),
            issues=issues,
        )
