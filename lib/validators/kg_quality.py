"""KG-quality gate — thin wrapper around ``KGQualityReporter``.

Wired into ``config/workflows.yaml`` as a blocking ``critical``-severity
gate on the ``textbook_to_course::libv2_archival`` phase. The gate
surfaces the four KG-quality dimensions (completeness, consistency,
accuracy, coverage).

Improvement #4 from the post-Wave 85 corpus-grounded gap analysis.

Wave 91 calibration baseline (run on the RDF/SHACL calibration
corpus against an empty SHACL validation
report — i.e. zero violations, the steady-state expected at
libv2_archival):

    completeness: 1.0
    consistency:  1.0
    accuracy:     1.0
    coverage:     0.5248   (under the OLD edge-share coverage metric)
    composite:    0.8812 (unweighted mean)

Promoted thresholds (max(spec_default, baseline - 0.05)):

    min_completeness: 0.95   (baseline 1.0 - 0.05)
    min_consistency:  0.95   (baseline 1.0 - 0.05)
    min_accuracy:     0.95   (baseline 1.0 - 0.05)
    min_coverage:     0.50   (spec default; baseline 0.5248 - 0.05 = 0.4748 floors at spec)

Severity flipped from ``warning`` to ``critical`` and ``on_fail`` from
``warn`` to ``block``. The fixture corpus passes at the chosen
thresholds (smallest margin: coverage at 0.5248 vs floor 0.50, +0.025).

Coverage-semantics redesign — gate path aligned with the production
reporter. The ``min_coverage: 0.50`` floor thresholds *chunk-anchored
DomainConcept node-grounding* — the share of DomainConcept-or-classless
concept nodes incident to ≥1 chunk-evidenced edge — NOT the old
asserted/(asserted+derived) edge share. The redesign first landed in
``KGQualityReporter.compute_metrics_only`` (the authoring-time entry
point used by ``_run_concept_extraction``); ``KGQualityReporter.compute``
— the method THIS gate calls — was subsequently brought into line so the
gate verdict matches the production reporter. Before that alignment the
gate computed the legacy edge-share over an asserted ``concept_graph.json``
the textbook_to_course path never writes, so coverage scored 0.0 and the
critical gate FAILED a graph the authoring reporter scored ~0.69. The old
edge metric grew a quadratic LO-order-derived denominator with LO count
(the rdf-shacl calibration corpus fell to 0.047 under it) and
anti-correlated with quality (every quality-improving graph flag lowered
it). The new node-grounding metric reads on the RDF/SHACL calibration
corpus graph at 1.0 and on a RAG-course graph (76 frequency-0
lo_key_concept nodes) at ~0.69. Recalibration basis: the 0.50 floor is
RETAINED as-is (re-derived against the node-grounding metric, both
calibration corpora clear it). The coverage figure surfaces
``grounded_node_count`` / ``concept_node_count`` plus the legacy ratio as
an informational ``asserted_edge_share`` (see
``Trainforge/rag/kg_quality_report.py``). NB: this gate is now wired
end-to-end — an input builder is registered and ``run_gate`` forwards
``threshold:`` keys (``min_completeness`` etc.) into validator inputs,
so metric-floor breaches fail closed instead of passing with warnings.

Silent-degradation finding C3 (post-Wave 91 audit) — fail-closed
inversions:

    * Missing the required ``semantic_graph_path`` (or ``course_slug`` /
      ``run_id`` / ``output_dir``) → ``passed=False`` with
      ``KG_QUALITY_PEDAGOGY_GRAPH_MISSING`` (critical). A
      libv2_archival run with NO semantic graph is a critical fail, not a
      pass — thresholds are meaningless when no graph exists, and
      downstream property_coverage / min_edge_count would also pass
      vacuously and ship an empty knowledge graph to LibV2. The asserted
      ``concept_graph_path`` is OPTIONAL (completeness falls back to the
      semantic graph's nodes when it is absent), so its omission alone
      does NOT trip this arm.
    * Reporter exception → ``passed=False`` with
      ``KG_QUALITY_REPORTER_ERROR`` (critical). The previous
      ``passed=True`` swallowed silent reporter regressions.
    * Threshold-breach path is INTENTIONALLY untouched: it emits
      warning-severity issues with ``passed=True`` and the workflow
      YAML configures critical-severity at the gate level — see
      ``config/workflows.yaml::textbook_to_course::libv2_archival``.
    * Decision-capture wiring (audit H3): emits one
      ``kg_quality_report_check`` event per ``validate()`` call with
      the four computed dimension scores and the gate verdict, so
      post-hoc replay can distinguish "graph missing → fail-closed"
      from "graph present and below threshold → warning" from "graph
      present and above threshold → pass".

Inputs (passed via the gate framework):
    course_slug: Required. Course slug for the report context.
    run_id: Required. Pipeline run identifier.
    output_dir: Required. Directory to write ``kg_quality_report.json``.
    concept_graph_path: Optional. Path to the asserted
        ``concept_graph.json``. Authoritative node set for completeness
        WHEN PRESENT (the rdf_shacl path); when absent — the
        textbook_to_course path emits only the semantic graph — the
        reporter falls back to the semantic graph's nodes, so a missing
        asserted graph is NOT a hard fail.
    semantic_graph_path: Required. Path to ``concept_graph_semantic.json``.
        The load-bearing input: coverage (node-grounding) and the
        completeness fallback both compute from it. A missing semantic
        graph fails the gate closed.
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
    decision_capture: Optional. ``DecisionCapture`` instance; when
        wired, one ``kg_quality_report_check`` decision event fires per
        ``validate()`` call (see audit H3 closure).

Outputs:
    GateResult with ``score`` set to the unweighted mean of the four
    dimensions and one ``warning``-severity GateIssue per threshold
    breach. Threshold defaults are 0.0 so the gate is advisory at
    threshold-breach severity unless the workflow YAML overrides them
    — but missing-graph and reporter-exception now fail closed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


_DIMENSIONS = ("completeness", "consistency", "accuracy", "coverage")


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    scores: Dict[str, float],
    thresholds: Dict[str, float],
    composite: Optional[float],
    course_slug: Optional[str],
    run_id: Optional[str],
) -> None:
    """Emit one ``kg_quality_report_check`` decision per validate() call.

    Closes audit H3 partially for this validator: every pass /
    threshold-fail / missing-graph / reporter-exception path emits one
    event so post-hoc replay can distinguish the four outcomes.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    composite_str = (
        f"{composite:.4f}" if composite is not None else "n/a"
    )
    score_strs = ", ".join(
        f"{dim}={scores.get(dim, 0.0):.4f}" for dim in _DIMENSIONS
    )
    threshold_strs = ", ".join(
        f"min_{dim}={thresholds.get(dim, 0.0):.4f}" for dim in _DIMENSIONS
    )
    rationale = (
        f"KG-quality gate verdict for course={course_slug or 'n/a'} "
        f"run_id={run_id or 'n/a'}: "
        f"composite={composite_str}, "
        f"scores=({score_strs}), "
        f"thresholds=({threshold_strs}), "
        f"failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        **{dim: float(scores.get(dim, 0.0)) for dim in _DIMENSIONS},
        "composite": float(composite) if composite is not None else None,
        "passed": bool(passed),
        "failure_code": code,
    }
    # Threaded via context (canonical home for free-form structured
    # metric blobs in the DecisionCapture record) AND mirrored as a
    # ``metrics`` kwarg so test mocks can assert on it directly. The
    # rigid ``MLFeatures`` dataclass has no metric fields, so we don't
    # route through that surface.
    try:
        capture.log_decision(
            decision_type="kg_quality_report_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "kg_quality_report_check: %s",
            exc,
        )


class KGQualityValidator:
    """Validator-protocol wrapper for ``KGQualityReporter``."""

    name = "kg_quality_report"
    version = "1.0.0"

    def __init__(
        self,
        reporter_factory: Optional[Any] = None,
        *,
        decision_capture: Optional[Any] = None,
    ) -> None:
        # Allow tests to inject a mock factory; default lazy import keeps
        # Trainforge dependencies out of the validator module's import
        # surface for callers that only load the validators package.
        self._reporter_factory = reporter_factory
        # Optional capture instance; the workflow runner may also thread
        # one in via ``inputs["decision_capture"]`` per call. Pattern
        # matches ``lib/validators/rewrite_source_grounding.py``.
        self._decision_capture = decision_capture

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

        # Per-call capture override (workflow runner threads it in)
        # falls back to the constructor-injected one.
        capture = inputs.get("decision_capture") or self._decision_capture

        course_slug = inputs.get("course_slug")
        run_id = inputs.get("run_id")
        output_dir_raw = inputs.get("output_dir")
        concept_graph_raw = inputs.get("concept_graph_path")
        semantic_graph_raw = inputs.get("semantic_graph_path")

        thresholds: Dict[str, float] = {
            "completeness": float(inputs.get("min_completeness", 0.0) or 0.0),
            "consistency": float(inputs.get("min_consistency", 0.0) or 0.0),
            "accuracy": float(inputs.get("min_accuracy", 0.0) or 0.0),
            "coverage": float(inputs.get("min_coverage", 0.0) or 0.0),
        }
        zero_scores: Dict[str, float] = {dim: 0.0 for dim in _DIMENSIONS}

        # The SEMANTIC graph is the load-bearing input — coverage
        # (node-grounding) and the completeness fallback both compute from
        # it. The asserted ``concept_graph.json`` is OPTIONAL: the
        # textbook_to_course path emits only the semantic graph (no separate
        # asserted graph), and the reporter degrades a missing asserted graph
        # to ``{}`` internally, falling back to the semantic graph's nodes
        # for completeness. So a present semantic_graph + absent
        # concept_graph is NOT a hard fail; only a missing semantic graph
        # (a libv2_archival run with NO knowledge graph at all) fails closed.
        missing: List[str] = []
        if not course_slug:
            missing.append("course_slug")
        if not run_id:
            missing.append("run_id")
        if not output_dir_raw:
            missing.append("output_dir")
        if not semantic_graph_raw:
            missing.append("semantic_graph_path")
        if missing:
            # Audit C3 fail-closed inversion: a libv2_archival run with
            # NO graph is a critical fail, not a pass. Thresholds are
            # meaningless when no graph exists; downstream
            # property_coverage / min_edge_count would also pass
            # vacuously and ship an empty KG to LibV2.
            _emit_decision(
                capture,
                passed=False,
                code="KG_QUALITY_PEDAGOGY_GRAPH_MISSING",
                scores=zero_scores,
                thresholds=thresholds,
                composite=None,
                course_slug=str(course_slug) if course_slug else None,
                run_id=str(run_id) if run_id else None,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="KG_QUALITY_PEDAGOGY_GRAPH_MISSING",
                    message=(
                        "Required inputs missing for "
                        "KGQualityValidator: "
                        f"{', '.join(missing)}. A libv2_archival run "
                        "with no semantic knowledge graph is a critical "
                        "fail — refusing to ship an empty knowledge "
                        "graph to LibV2. (The asserted concept_graph.json "
                        "is OPTIONAL — completeness falls back to the "
                        "semantic graph's nodes when it is absent.)"
                    ),
                    suggestion=(
                        "Verify the upstream concept_extraction phase "
                        "emitted concept_graph_semantic.json; inspect that "
                        "phase's logs for silent failures."
                    ),
                )],
                action="block",
            )

        try:
            reporter = self._build_reporter(
                course_slug=str(course_slug),
                run_id=str(run_id),
                output_dir=Path(output_dir_raw),
            )
            report = reporter.compute(
                concept_graph=(
                    Path(concept_graph_raw) if concept_graph_raw else None
                ),
                semantic_graph=Path(semantic_graph_raw),
                validation_report=inputs.get("validation_report"),
                pedagogy_graph=(
                    Path(inputs["pedagogy_graph_path"])
                    if inputs.get("pedagogy_graph_path")
                    else None
                ),
            )
            reporter.write(report)
        except Exception as exc:
            # Audit C3 fail-closed inversion: a reporter raise was
            # previously swallowed as passed=True (silent regression
            # vector). Now critical-fail with the exception class +
            # message threaded through the issue.
            exc_msg = (
                f"KG-quality reporter raised "
                f"{type(exc).__name__}: {exc}"
            )
            _emit_decision(
                capture,
                passed=False,
                code="KG_QUALITY_REPORTER_ERROR",
                scores=zero_scores,
                thresholds=thresholds,
                composite=None,
                course_slug=str(course_slug),
                run_id=str(run_id),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="KG_QUALITY_REPORTER_ERROR",
                    message=exc_msg,
                    suggestion=(
                        "Inspect Trainforge.rag.kg_quality_report logs "
                        "and the inputs to KGQualityReporter.compute; "
                        "rerun with --debug to capture the traceback."
                    ),
                )],
                action="block",
            )

        scores: Dict[str, float] = {}
        for dim in _DIMENSIONS:
            score = float(
                report.get("dimensions", {}).get(dim, {}).get("score", 0.0)
            )
            scores[dim] = score

        # GPT-feedback-12-may item 2: cross-rule edge consensus. Walk
        # the just-emitted concept_graph_semantic.json and stamp every
        # edge with edge_status + consensus_signals[]. Multiply the
        # consistency axis by (1 - contradiction_rate) BEFORE the
        # threshold check so contradicted-edge clusters take a
        # proportional hit on consistency. Best-effort: any aggregator
        # failure logs and leaves scores unchanged.
        consensus_rate: Optional[float] = None
        contradiction_rate: Optional[float] = None
        try:
            from lib.aggregators.edge_consensus import EdgeConsensusAggregator
            aggregator = EdgeConsensusAggregator(
                semantic_graph_path=Path(semantic_graph_raw),
                course_slug=str(course_slug),
                run_id=str(run_id),
                decision_capture=capture,
            )
            consensus_report = aggregator.build()
            # Emit alongside the existing kg_quality_report.json.
            try:
                aggregator.write(
                    Path(output_dir_raw) / "edge_consensus_report.json"
                )
            except Exception as write_exc:  # noqa: BLE001
                logger.warning(
                    "EdgeConsensusAggregator.write raised %s; consensus "
                    "signal still applied to the in-memory score.",
                    write_exc,
                )
            summary = consensus_report.get("summary", {})
            contradiction_rate = float(summary.get("contradiction_rate", 0.0))
            consensus_rate = float(summary.get("consensus_rate", 0.0))
            scores["consistency"] = max(
                0.0,
                scores["consistency"] * (1.0 - contradiction_rate),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "EdgeConsensusAggregator failed (%s); consistency "
                "axis left at its base value.",
                exc,
            )

        breached = False
        for dim in _DIMENSIONS:
            score = scores[dim]
            min_score = thresholds[dim]
            if min_score > 0.0 and score < min_score:
                breached = True
                issues.append(GateIssue(
                    severity="critical",
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

        # A breached per-dimension floor fails the gate closed. The
        # configured floors (``min_completeness`` / ``min_consistency`` /
        # ``min_accuracy`` / ``min_coverage``) reach this validator via the
        # gate manager's ``threshold:`` -> inputs forward; a dimension below
        # its floor emits a critical GateIssue and inverts ``passed`` so the
        # critical/block gate (textbook_to_course::libv2_archival) actually
        # refuses to ship a degraded knowledge graph to LibV2. When no floor
        # is configured (all default to 0.0) or every dimension clears its
        # floor, the gate passes as before. The missing-graph + reporter-
        # exception arms above remain the other two fail-closed vectors.

        _emit_decision(
            capture,
            passed=not breached,
            code=(
                "KG_QUALITY_DIMENSION_BELOW_THRESHOLD" if breached else None
            ),
            scores=scores,
            thresholds=thresholds,
            composite=composite,
            course_slug=str(course_slug),
            run_id=str(run_id),
        )

        result_metadata: Dict[str, Any] = {}
        if consensus_rate is not None:
            result_metadata["edge_consensus_rate"] = round(consensus_rate, 4)
        if contradiction_rate is not None:
            result_metadata["edge_contradiction_rate"] = round(
                contradiction_rate, 4
            )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=not breached,  # a breached floor fails the critical gate
            score=round(composite, 4),
            issues=issues,
            action=("block" if breached else None),
            metadata=result_metadata or None,
        )


__all__ = ["KGQualityValidator"]
