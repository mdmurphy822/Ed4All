"""Wave 109 / Phase C — PropertyCoverageValidator.

Reads ``<course_dir>/training_specs/instruction_pairs.jsonl`` and
counts how many rows reference each property declared in the
course's property manifest (via surface-form substring match against
``prompt + completion``). Fails closed when ANY property has fewer
than ``min_pairs`` matching rows — the exact regression class where
a paraphrased corpus drops a hard surface form (e.g. owl:sameAs)
because the LLM rewriter found a more natural English form, leaving
the SLM with no training signal for it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    properties_declared: int,
    properties_covered: int,
    properties_below_floor: int,
    coverage_rate: float,
    per_property_counts: Dict[str, int],
    skip_reason: Optional[str] = None,
) -> None:
    """Emit one ``property_coverage_check`` decision per validate() call.

    H3 Wave W4: every below-floor / pass / no-manifest-skip path emits
    one event. The per-property counts map carries the actual coverage
    distribution so post-hoc replay can identify which property
    regressed without re-loading the manifest.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rationale = (
        f"property_coverage gate verdict: properties_declared="
        f"{properties_declared}, properties_covered="
        f"{properties_covered}, properties_below_floor="
        f"{properties_below_floor}, coverage_rate="
        f"{coverage_rate:.4f}; skip_reason={skip_reason or 'none'}; "
        f"failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "properties_declared": int(properties_declared),
        "properties_covered": int(properties_covered),
        "properties_below_floor": int(properties_below_floor),
        "coverage_rate": float(coverage_rate),
        "per_property_counts": dict(per_property_counts),
        "skip_reason": skip_reason,
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="property_coverage_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "property_coverage_check: %s",
            exc,
        )


class PropertyCoverageValidator:
    name = "property_coverage"
    version = "1.0.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "property_coverage")
        capture = inputs.get("decision_capture") or self._decision_capture
        course_dir_raw = inputs.get("course_dir")
        course_slug = inputs.get("course_slug")
        if not course_dir_raw or not course_slug:
            _emit_decision(
                capture, passed=False, code="MISSING_INPUTS",
                properties_declared=0, properties_covered=0,
                properties_below_floor=0, coverage_rate=0.0,
                per_property_counts={},
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_INPUTS",
                    message=(
                        "PropertyCoverageValidator requires course_dir + "
                        "course_slug inputs."
                    ),
                )],
            )
        course_dir = Path(course_dir_raw)
        inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
        if not inst_path.exists():
            _emit_decision(
                capture, passed=False, code="INSTRUCTION_PAIRS_NOT_FOUND",
                properties_declared=0, properties_covered=0,
                properties_below_floor=0, coverage_rate=0.0,
                per_property_counts={},
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INSTRUCTION_PAIRS_NOT_FOUND",
                    message=(
                        f"instruction_pairs.jsonl not found at {inst_path}; "
                        "run the synthesis phase before the coverage gate."
                    ),
                    location=str(inst_path),
                )],
            )

        from lib.ontology.property_manifest import load_property_manifest
        try:
            manifest = load_property_manifest(course_slug)
        except FileNotFoundError as exc:
            # Courses outside the rdf-shacl family don't need property
            # gating; we don't want this gate to break the
            # textbook_to_course workflow on courses that haven't been
            # calibrated yet.
            logger.info(
                "PropertyCoverageValidator: no manifest for course '%s' "
                "(%s); skipping gate.", course_slug, exc,
            )
            _emit_decision(
                capture, passed=True, code=None,
                properties_declared=0, properties_covered=0,
                properties_below_floor=0, coverage_rate=1.0,
                per_property_counts={},
                skip_reason="no_property_manifest",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        counts: Dict[str, int] = {p.id: 0 for p in manifest.properties}
        try:
            with inst_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = f"{row.get('prompt', '')} {row.get('completion', '')}"
                    for prop in manifest.properties:
                        if prop.matches(text):
                            counts[prop.id] += 1
        except OSError as exc:
            _emit_decision(
                capture, passed=False, code="INSTRUCTION_PAIRS_READ_FAILED",
                properties_declared=len(manifest.properties),
                properties_covered=0, properties_below_floor=0,
                coverage_rate=0.0, per_property_counts=dict(counts),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="INSTRUCTION_PAIRS_READ_FAILED",
                    message=f"Could not read {inst_path}: {exc}",
                    location=str(inst_path),
                )],
            )

        issues: List[GateIssue] = []
        below = []
        for prop in manifest.properties:
            seen = counts[prop.id]
            if seen < prop.min_pairs:
                below.append((prop.id, seen, prop.min_pairs))
        if below:
            details = "; ".join(
                f"{pid}: {seen}/{floor}" for pid, seen, floor in below
            )
            issues.append(GateIssue(
                severity="critical",
                code="PROPERTY_COVERAGE_BELOW_FLOOR",
                message=(
                    f"Synthesis output is missing minimum coverage for "
                    f"{len(below)} of {len(manifest.properties)} declared "
                    f"properties ({details}). Re-run synthesis with a "
                    f"property-aware provider, or revise the manifest "
                    f"floors if intentional."
                ),
                location=str(inst_path),
            ))

        passed = not [i for i in issues if i.severity == "critical"]

        # H3 W4: emit terminal capture. coverage_rate = fraction of
        # declared properties whose count met or exceeded their floor.
        n_declared = len(manifest.properties)
        n_covered = sum(
            1 for p in manifest.properties if counts[p.id] >= p.min_pairs
        )
        coverage_rate = (n_covered / n_declared) if n_declared else 1.0
        failure_code = None
        if not passed:
            for i in issues:
                if i.severity == "critical":
                    failure_code = i.code
                    break
        _emit_decision(
            capture,
            passed=passed,
            code=failure_code,
            properties_declared=n_declared,
            properties_covered=n_covered,
            properties_below_floor=len(below),
            coverage_rate=coverage_rate,
            per_property_counts=dict(counts),
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - 0.1 * len(issues)),
            issues=issues,
        )
