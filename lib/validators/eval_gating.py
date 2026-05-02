"""Wave 108 / Phase B — EvalGatingValidator.

Reads ``<model_dir>/eval/eval_report.json`` (the artifact emitted by
:class:`Trainforge.eval.slm_eval_harness.SLMEvalHarness`) and decides
whether the run is allowed to promote into ``models/_pointers.json``.

Critical-severity gates (any one fails the result):

* ``faithfulness < min_faithfulness``                             -> EVAL_FAITHFULNESS_BELOW_THRESHOLD
* ``source_match < min_source_match`` (when present)              -> EVAL_SOURCE_MATCH_BELOW_THRESHOLD
* ``baseline_delta < 0.0`` (regression vs base model)             -> EVAL_BASELINE_REGRESSION
* ``negative_grounding_accuracy < min_negative_grounding``        -> EVAL_NEGATIVE_GROUNDING_BELOW_THRESHOLD
* ``yes_rate > max_yes_rate`` (yes-bias)                          -> EVAL_YES_BIAS_DETECTED
* eval_report.json missing or unparseable                         -> EVAL_REPORT_NOT_FOUND / EVAL_REPORT_INVALID_JSON

Warning-severity advisories (logged, never block):

* ``metrics.hallucination_rate > max_hallucination_rate``         -> EVAL_HALLUCINATION_HIGH
* ``calibration_ece > max_calibration_ece`` (when present)        -> EVAL_CALIBRATION_HIGH

Per the root CLAUDE.md mandate, every validator that participates in a
load-bearing decision MUST log to a ``DecisionCapture`` when one is
provided. The rationale interpolates the actual metric values so a
post-hoc audit can see why the gate fired.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


_DEFAULT_THRESHOLDS = {
    "min_faithfulness": 0.50,
    "min_source_match": 0.30,
    "min_negative_grounding": 0.50,
    "max_yes_rate": 0.85,
    "max_hallucination_rate": 0.50,
    "max_calibration_ece": 0.30,
    # Wave 109 / Phase C: per-property accuracy floor.
    "min_per_property_accuracy": 0.40,
    # Wave 138a / W3: content_type_role_alignment_rate floor. Tier-2
    # corpus-derived check (no LLM dispatch). Warning-severity for v1
    # per operator; promote to critical after two clean rebuilds.
    "min_content_type_role_alignment_rate": 0.70,
}


class EvalGatingValidator:
    name = "eval_gating"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "eval_gating")
        thresholds = dict(_DEFAULT_THRESHOLDS)
        for k, v in (inputs.get("thresholds") or {}).items():
            if k in thresholds:
                thresholds[k] = float(v)

        model_dir_raw = inputs.get("model_dir")
        if not model_dir_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_MODEL_DIR",
                    message="model_dir is required for EvalGatingValidator",
                )],
            )
        model_dir = Path(model_dir_raw)
        report_path = model_dir / "eval" / "eval_report.json"
        if not report_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="EVAL_REPORT_NOT_FOUND",
                    message=(
                        f"eval_report.json not found at {report_path}; the "
                        "training phase must run the eval harness before "
                        "the gating validator runs."
                    ),
                    location=str(report_path),
                )],
            )

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="EVAL_REPORT_INVALID_JSON",
                    message=f"eval_report.json failed to parse: {exc}",
                    location=str(report_path),
                )],
            )

        # 2026-04-30 smoke mode: refuse to gate on a smoke report.
        # Smoke runs cap each evaluator at N=3 so the metrics are
        # statistically meaningless for promotion gating. The harness
        # writes smoke runs to smoke_eval_report.json, but a defensive
        # check on the in-report `smoke_mode` field protects against
        # operators copying / renaming a smoke report into eval_report.json.
        if bool(report.get("smoke_mode")):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="EVAL_REPORT_IS_SMOKE",
                    message=(
                        "eval_report.json carries smoke_mode=true; refusing "
                        "to gate. Smoke runs cap evaluators at N=3 prompts "
                        "and are intended for plumbing verification only. "
                        "Re-run the harness without --smoke to produce a "
                        "promotion-gated eval_report.json."
                    ),
                    location=str(report_path),
                )],
            )

        issues: List[GateIssue] = []
        # --- Critical thresholds -----------------------------------------
        faithfulness = _as_float(report.get("faithfulness"))
        if faithfulness is None or faithfulness < thresholds["min_faithfulness"]:
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_FAITHFULNESS_BELOW_THRESHOLD",
                message=(
                    f"faithfulness={faithfulness} below threshold "
                    f"{thresholds['min_faithfulness']}"
                ),
                location=str(report_path),
            ))
        source_match = _as_float(report.get("source_match"))
        if source_match is not None and source_match < thresholds["min_source_match"]:
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_SOURCE_MATCH_BELOW_THRESHOLD",
                message=(
                    f"source_match={source_match} below threshold "
                    f"{thresholds['min_source_match']}"
                ),
                location=str(report_path),
            ))
        baseline_delta = _as_float(report.get("baseline_delta"))
        if baseline_delta is not None and baseline_delta < 0.0:
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_BASELINE_REGRESSION",
                message=(
                    f"baseline_delta={baseline_delta} indicates regression "
                    "against base model; refusing to promote."
                ),
                location=str(report_path),
            ))
        neg = _as_float(report.get("negative_grounding_accuracy"))
        if neg is not None and neg < thresholds["min_negative_grounding"]:
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_NEGATIVE_GROUNDING_BELOW_THRESHOLD",
                message=(
                    f"negative_grounding_accuracy={neg} below threshold "
                    f"{thresholds['min_negative_grounding']} -- model is "
                    "yes-biased (template-recognizer regression class)."
                ),
                location=str(report_path),
            ))
        yes_rate = _as_float(report.get("yes_rate"))
        if yes_rate is not None and yes_rate > thresholds["max_yes_rate"]:
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_YES_BIAS_DETECTED",
                message=(
                    f"yes_rate={yes_rate} above threshold "
                    f"{thresholds['max_yes_rate']} -- model is "
                    "over-affirming on positive probes."
                ),
                location=str(report_path),
            ))

        # Wave 109 / Phase C: per-property accuracy gate. Skip
        # properties with None accuracy (unscored — no probes matched
        # the surface forms).
        per_property = report.get("per_property_accuracy") or {}
        min_pp = thresholds["min_per_property_accuracy"]
        below_pp = []
        for prop_id, score in per_property.items():
            if score is None:
                continue
            score_f = _as_float(score)
            if score_f is None:
                continue
            if score_f < min_pp:
                below_pp.append((prop_id, score_f))
        if below_pp:
            details = "; ".join(f"{pid}={s:.3f}" for pid, s in below_pp)
            issues.append(GateIssue(
                severity="critical",
                code="EVAL_PER_PROPERTY_BELOW_THRESHOLD",
                message=(
                    f"Per-property accuracy below {min_pp} threshold for: "
                    f"{details}. Adapter has not learned at least one of "
                    f"the declared properties; refusing to promote."
                ),
                location=str(report_path),
            ))

        # --- Warning advisories ------------------------------------------
        hallucination = _as_float((report.get("metrics") or {}).get("hallucination_rate"))
        if hallucination is not None and hallucination > thresholds["max_hallucination_rate"]:
            issues.append(GateIssue(
                severity="warning",
                code="EVAL_HALLUCINATION_HIGH",
                message=(
                    f"hallucination_rate={hallucination} above advisory "
                    f"threshold {thresholds['max_hallucination_rate']}."
                ),
                location=str(report_path),
            ))
        ece = _as_float(report.get("calibration_ece"))
        if ece is not None and ece > thresholds["max_calibration_ece"]:
            issues.append(GateIssue(
                severity="warning",
                code="EVAL_CALIBRATION_HIGH",
                message=(
                    f"calibration_ece={ece} above advisory threshold "
                    f"{thresholds['max_calibration_ece']}."
                ),
                location=str(report_path),
            ))

        # Wave 138a / W3: teaching-role alignment advisory. Tier-2
        # corpus-derived signal — the harness emits a per-content-type
        # `content_type_role_alignment` block plus a sibling
        # `content_type_role_alignment_summary` carrying the aggregate
        # `alignment_rate` and the list of mismatched content types.
        # Skip silently when the report does not carry the field
        # (legacy / pre-Wave-138 reports).
        ctra_summary = report.get("content_type_role_alignment_summary")
        ctra_alignment_rate: Optional[float] = None
        ctra_mismatched: List[str] = []
        if isinstance(ctra_summary, dict) and report.get("content_type_role_alignment") is not None:
            ctra_alignment_rate = _as_float(ctra_summary.get("alignment_rate"))
            mismatched_raw = ctra_summary.get("mismatched_content_types") or []
            if isinstance(mismatched_raw, list):
                ctra_mismatched = [str(m) for m in mismatched_raw]
            min_ctra = thresholds["min_content_type_role_alignment_rate"]
            if ctra_alignment_rate is not None and ctra_alignment_rate < min_ctra:
                issues.append(GateIssue(
                    severity="warning",
                    code="EVAL_CONTENT_TYPE_ROLE_ALIGNMENT_LOW",
                    message=(
                        f"alignment_rate {ctra_alignment_rate:.3f} below "
                        f"{min_ctra}; mismatched: {ctra_mismatched}"
                    ),
                    location=str(report_path),
                ))

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        # CLAUDE.md mandate: emit a decision capture so the gating
        # decision is replayable post-hoc.
        capture = inputs.get("capture")
        if capture is not None:
            try:
                rationale = (
                    f"EvalGatingValidator {('PASSED' if passed else 'BLOCKED')}: "
                    f"faithfulness={faithfulness} "
                    f"source_match={source_match} "
                    f"baseline_delta={baseline_delta} "
                    f"yes_rate={yes_rate} "
                    f"negative_grounding_accuracy={neg} "
                    f"per_property={per_property} "
                    f"alignment_rate={ctra_alignment_rate} "
                    f"mismatched={ctra_mismatched}. "
                    f"Critical issues: {critical_count}; "
                    f"thresholds={thresholds}."
                )
                capture.log_decision(
                    decision_type="eval_gating_decision",
                    decision=("eval_gating::passed" if passed else "eval_gating::blocked"),
                    rationale=rationale,
                )
            except Exception as exc:  # noqa: BLE001 - capture is advisory
                logger.warning("eval_gating_decision capture failed: %s", exc)

        # Wave 137d-2: emit a form_data coverage checkpoint row to
        # ``LibV2/courses/<slug>/eval/form_data_coverage_checkpoint.jsonl``.
        # Advisory metric — the checkpoint is wrapped in try/except so
        # any I/O / loader failure NEVER breaks the gate result. This
        # gives operators a per-promotion-attempt audit trail of
        # backfill coverage progress alongside the eval gating decision.
        try:
            self._emit_coverage_checkpoint(
                model_dir=model_dir,
                passed=passed,
                issues=issues,
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint is advisory
            logger.warning(
                "form_data_coverage checkpoint emission failed: %s", exc
            )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # Wave 137d-2: helper that locates the LibV2 course root from the
    # model_dir, resolves the family via property_manifest, computes
    # coverage metrics, and atomic-appends a JSONL row.
    def _emit_coverage_checkpoint(
        self,
        *,
        model_dir: Path,
        passed: bool,
        issues: List[GateIssue],
    ) -> None:
        """Write one ``form_data_coverage_checkpoint.jsonl`` row.

        Path resolution: ``model_dir`` follows the canonical
        ``LibV2/courses/<slug>/models/<model_id>/`` layout. The
        course root is ``model_dir.parent.parent``; the checkpoint
        file lives at ``<course_root>/eval/form_data_coverage_checkpoint.jsonl``.

        Schema (each row is one JSON object):
          * ``timestamp`` — ISO-8601 UTC.
          * ``model_id`` — model_dir basename.
          * ``course_slug`` — course root basename.
          * ``family`` — resolved via property manifest.
          * ``manifest_coverage_pct`` / ``complete_count`` /
            ``degraded_count`` / ``family_coverage_map`` — from
            :func:`lib.validators.form_data_coverage.compute_coverage_metrics`.
          * ``promotion_decision`` — ``"passed"`` or ``"blocked"``.
          * ``promotion_block_reasons`` — list of issue codes (critical
            severity only) when blocked; empty list when passed.
        """
        from lib.validators.form_data_coverage import compute_coverage_metrics
        from lib.ontology.property_manifest import load_property_manifest

        model_id = model_dir.name
        course_root = model_dir.parent.parent
        course_slug = course_root.name

        # Resolve family via the property manifest (which itself
        # derives family from the course slug). When no manifest
        # exists we still emit a row with ``family=None`` so the
        # operator sees the gate ran.
        family: Optional[str]
        try:
            manifest = load_property_manifest(course_slug)
            family = manifest.family
        except (FileNotFoundError, AttributeError):
            family = None

        if family is not None:
            metrics = compute_coverage_metrics(family)
        else:
            metrics = {
                "manifest_coverage_pct": None,
                "complete_count": None,
                "degraded_count": None,
                "family_coverage_map": {},
            }

        block_reasons = (
            [i.code for i in issues if i.severity == "critical"]
            if not passed
            else []
        )

        row = {
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "model_id": model_id,
            "course_slug": course_slug,
            "family": family,
            "manifest_coverage_pct": metrics["manifest_coverage_pct"],
            "complete_count": metrics["complete_count"],
            "degraded_count": metrics["degraded_count"],
            "family_coverage_map": metrics["family_coverage_map"],
            "promotion_decision": "passed" if passed else "blocked",
            "promotion_block_reasons": block_reasons,
        }

        eval_dir = course_root / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = eval_dir / "form_data_coverage_checkpoint.jsonl"
        # Append-only: every validate() call adds one row.
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
