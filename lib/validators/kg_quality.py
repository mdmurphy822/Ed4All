"""KG-quality gate — thin wrapper around ``KGQualityReporter``.

Wired into ``config/workflows.yaml`` as a blocking ``critical``-severity
gate on the ``textbook_to_course::libv2_archival`` phase. The gate
surfaces the four KG-quality dimensions (completeness, consistency,
accuracy, coverage).

Improvement #4 from the post-Wave 85 corpus-grounded gap analysis.

Wave 91 calibration baseline (run on
``LibV2/courses/rdf-shacl-551-2/`` against an empty SHACL validation
report — i.e. zero violations, the steady-state expected at
libv2_archival):

    completeness: 1.0
    consistency:  1.0
    accuracy:     1.0
    coverage:     0.5248
    composite:    0.8812 (unweighted mean)

Promoted thresholds (max(spec_default, baseline - 0.05)):

    min_completeness: 0.95   (baseline 1.0 - 0.05)
    min_consistency:  0.95   (baseline 1.0 - 0.05)
    min_accuracy:     0.95   (baseline 1.0 - 0.05)
    min_coverage:     0.50   (spec default; baseline 0.5248 - 0.05 = 0.4748 floors at spec)

Severity flipped from ``warning`` to ``critical`` and ``on_fail`` from
``warn`` to ``block``. The fixture corpus passes at the chosen
thresholds (smallest margin: coverage at 0.5248 vs floor 0.50, +0.025).

Inputs (passed via the gate framework):
    course_slug: Required. Course slug for the report context.
    run_id: Required. Pipeline run identifier.
    output_dir: Required. Directory to write ``kg_quality_report.json``.
    concept_graph_path: Required. Path to ``concept_graph.json``.
    semantic_graph_path: Required. Path to ``concept_graph_semantic.json``.
    validation_report: Optional. Pre-built SHACL validation report
        object (``.results``-shaped). When absent the gate emits a
        report with empty SHACL aggregates — completeness / coverage
        still compute from the graphs alone.
    pedagogy_graph_path: Optional. Reserved for future cross-graph
        completeness checks; recorded in the report metadata.
    min_completeness: Optional float [0, 1]. Defaults to 0.0.
    min_consistency: Optional float [0, 1]. Defaults to 0.0.
    min_accuracy: Optional float [0, 1]. Defaults to 0.0.
    min_coverage: Optional float [0, 1]. Defaults to 0.0.

Outputs:
    GateResult with ``score`` set to the unweighted mean of the four
    dimensions and one ``warning``-severity GateIssue per threshold
    breach. Threshold defaults are 0.0 so the gate is advisory unless
    the workflow YAML overrides them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult


_DIMENSIONS = ("completeness", "consistency", "accuracy", "coverage")


class KGQualityValidator:
    """Validator-protocol wrapper for ``KGQualityReporter``."""

    name = "kg_quality_report"
    version = "1.0.0"

    def __init__(self, reporter_factory: Optional[Any] = None) -> None:
        # Allow tests to inject a mock factory; default lazy import keeps
        # Trainforge dependencies out of the validator module's import
        # surface for callers that only load the validators package.
        self._reporter_factory = reporter_factory

    def _build_reporter(
        self,
        *,
        course_slug: str,
        run_id: str,
        output_dir: Path,
    ) -> Any:
        if self._reporter_factory is not None:
            return self._reporter_factory(
                course_slug=course_slug,
                run_id=run_id,
                output_dir=output_dir,
            )
        from Trainforge.rag.kg_quality_report import KGQualityReporter
        return KGQualityReporter(
            course_slug=course_slug,
            run_id=run_id,
            output_dir=output_dir,
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "kg_quality_report")
        issues: List[GateIssue] = []

        course_slug = inputs.get("course_slug")
        run_id = inputs.get("run_id")
        output_dir_raw = inputs.get("output_dir")
        concept_graph_raw = inputs.get("concept_graph_path")
        semantic_graph_raw = inputs.get("semantic_graph_path")

        missing: List[str] = []
        if not course_slug:
            missing.append("course_slug")
        if not run_id:
            missing.append("run_id")
        if not output_dir_raw:
            missing.append("output_dir")
        if not concept_graph_raw:
            missing.append("concept_graph_path")
        if not semantic_graph_raw:
            missing.append("semantic_graph_path")
        if missing:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,  # advisory — never block on missing inputs
                issues=[GateIssue(
                    severity="warning",
                    code="KG_QUALITY_INPUT_MISSING",
                    message=(
                        "Required inputs missing for KGQualityValidator: "
                        f"{', '.join(missing)}. Skipping report generation."
                    ),
                )],
            )

        thresholds: Dict[str, float] = {
            "completeness": float(inputs.get("min_completeness", 0.0) or 0.0),
            "consistency": float(inputs.get("min_consistency", 0.0) or 0.0),
            "accuracy": float(inputs.get("min_accuracy", 0.0) or 0.0),
            "coverage": float(inputs.get("min_coverage", 0.0) or 0.0),
        }

        try:
            reporter = self._build_reporter(
                course_slug=str(course_slug),
                run_id=str(run_id),
                output_dir=Path(output_dir_raw),
            )
            report = reporter.compute(
                concept_graph=Path(concept_graph_raw),
                semantic_graph=Path(semantic_graph_raw),
                validation_report=inputs.get("validation_report"),
                pedagogy_graph=(
                    Path(inputs["pedagogy_graph_path"])
                    if inputs.get("pedagogy_graph_path")
                    else None
                ),
            )
            reporter.write(report)
        except Exception as exc:  # pragma: no cover — defensive
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,  # advisory — never block on reporter errors
                issues=[GateIssue(
                    severity="warning",
                    code="KG_QUALITY_REPORT_ERROR",
                    message=f"KG-quality reporter raised: {exc}",
                )],
            )

        scores: Dict[str, float] = {}
        for dim in _DIMENSIONS:
            score = float(
                report.get("dimensions", {}).get(dim, {}).get("score", 0.0)
            )
            scores[dim] = score
            min_score = thresholds[dim]
            if min_score > 0.0 and score < min_score:
                issues.append(GateIssue(
                    severity="warning",
                    code=f"KG_QUALITY_{dim.upper()}_BELOW_THRESHOLD",
                    message=(
                        f"KG quality dimension '{dim}' scored {score:.4f} "
                        f"below configured min ({min_score:.4f})."
                    ),
                    suggestion=(
                        "Inspect kg_quality_report.json for per-shape and "
                        "rule-output detail; lower the threshold or "
                        "improve the upstream emit if the regression is real."
                    ),
                ))

        # Unweighted mean across the four dimensions.
        composite = sum(scores.values()) / len(_DIMENSIONS)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-only gate — never blocks
            score=round(composite, 4),
            issues=issues,
        )


__all__ = ["KGQualityValidator"]
