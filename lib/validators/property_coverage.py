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
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


class PropertyCoverageValidator:
    name = "property_coverage"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "property_coverage")
        course_dir_raw = inputs.get("course_dir")
        course_slug = inputs.get("course_slug")
        if not course_dir_raw or not course_slug:
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
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - 0.1 * len(issues)),
            issues=issues,
        )
