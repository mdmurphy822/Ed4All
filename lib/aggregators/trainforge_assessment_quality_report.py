"""
TrainForge assessment-quality top-level aggregator (Worker W2.B).

Aggregates four signals that today live in disjoint surfaces — synthesis-
pair quality (``training_synthesis`` gate results), assessment-corpus
quality (``trainforge_assessment`` gate results +
``quality_report.json::assessments`` dimension), KG quality
(``libv2_archival`` gate results), and post-training eval gating
(``LibV2/courses/<slug>/models/<current>/eval/eval_report.json``) — into
a single canonical
``<libv2_course>/quality/trainforge_assessment_quality_report.json``
that operators read to decide whether the assessment corpus is
shippable as training data.

Schema sketch (mirrors the P0a "Fix 2" canonical sketch + W2.B plan)::

    {
      "schema_version": "1.0",
      "generated_at": "<iso8601>",
      "course_code": "<slug>",
      "run_id": "<workflow_id>",
      "status": "pass" | "fail" | "warn",
      "summary": {
        "total_questions": <int | null>,
        "answerable_rate": <0..1 | null>,
        "single_correct_rate": <0..1 | null>,
        "bloom_alignment_rate": <0..1 | null>,
        "placeholder_rate": <0..1 | null>,
        "source_support_rate": <0..1 | null>
      },
      "synthesis_quality": {
        "synthesis_diversity": {<gate-projection> | null},
        "synthesis_leakage": {<gate-projection> | null},
        "curie_anchoring": {<gate-projection> | null},
        "property_coverage": {<gate-projection> | null},
        "min_edge_count": {<gate-projection> | null}
      },
      "kg_quality": {<gate-projection> | null},
      "eval_summary": {<eval_report.json passthrough> | null},
      "per_question_issues": [...],
      "promotion_decision": {
        "value": "failed | non_certified_archive | trainable",
        "rationale": "...",
        "contributing_gate_ids": [...]
      }
    }

Promotion-decision heuristic (W2.B-internal — replaced by the master
promotion-chain compose in Wave 3 G1):

* ``failed`` — any of the following hard signals fired:
  * ``synthesis_diversity`` blocking-fail (critical + ``passed=False``);
  * ``min_edge_count`` blocking-fail;
  * ``synthesis_leakage`` blocking-fail;
  * ``placeholder_rate > 0.0`` (zero-tolerance — placeholder residues
    poison both training AND retrieval, see Trainforge invariants);
  * ``summary.answerable_rate < 0.50`` (a corpus where half the
    questions can't be answered from the cited sources is unshippable);
  * KG-quality blocking-fail (``kg_quality_report`` /
    ``libv2_manifest`` critical fail);
  * eval-gating blocking-fail (when ``eval_report.json`` exists AND
    the gating validator's blocking signals fired).
* ``non_certified_archive`` — completed run with warnings only (no
  blocking failures from the list above, but at least one warning row
  in ``synthesis_quality`` / ``kg_quality`` / ``trainforge_assessment``
  buckets). Course is archivable, not trainable.
* ``trainable`` — clean run, no warnings, all hard signals pass.

Best-effort posture: every contributing input is read with
graceful-degradation defaults, so a missing input resolves to
``null`` for the corresponding output field rather than raising.
``WorkflowRunner`` wraps the aggregator call in try/except so a build
failure does NOT change ``final_status``; the per-phase reports remain
the authoritative quality signal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Synthesis-quality gate IDs that surface in the top-level
# ``synthesis_quality`` bucket. Source: ``config/workflows.yaml`` ::
# ``textbook_to_course.training_synthesis.validation_gates``.
_SYNTHESIS_QUALITY_GATE_IDS = (
    "synthesis_diversity",
    "synthesis_leakage",
    "curie_anchoring",
    "property_coverage",
    "min_edge_count",
)

# Trainforge-assessment gate IDs that contribute to ``summary``.
# Source: ``config/workflows.yaml`` ::
# ``textbook_to_course.trainforge_assessment.validation_gates``.
_TRAINFORGE_ASSESSMENT_GATE_IDS = (
    "imscc_input_valid",
    "assessment_quality",
    "assessment_objective_alignment",
)

# KG-quality gate IDs that surface in the top-level ``kg_quality``
# bucket. Source: ``config/workflows.yaml`` ::
# ``textbook_to_course.libv2_archival.validation_gates``.
_KG_QUALITY_GATE_IDS = (
    "kg_quality_report",
    "libv2_manifest",
)

# Retrieval-grounding gate IDs that contribute the ``answerable_rate``
# / ``source_support_rate`` summary fields. The validator
# ``AssessmentRetrievalGroundingValidator`` emits ``score = grounded /
# audited`` per its docstring at L561; the aggregator reads that value
# verbatim.
_RETRIEVAL_GROUNDING_GATE_IDS = (
    "outline_assessment_retrieval_grounding",
    "rewrite_assessment_retrieval_grounding",
    "assessment_retrieval_grounding",
)

# Promotion-decision enum. W2.B-internal; Wave 3 G1's
# ``compose_course_status`` will replace the heuristic with the canonical
# master-promotion-chain compose.
_PROMOTION_DECISIONS = ("failed", "non_certified_archive", "trainable")

# Fields the aggregator passes through verbatim from
# ``LibV2/courses/<slug>/models/<id>/eval/eval_report.json`` when an
# adapter has been imported. Source: ``Trainforge/CLAUDE.md`` §
# "Eval — 5 generic layers × 3 corpus-aware tiers".
_EVAL_REPORT_PASSTHROUGH_FIELDS = (
    "faithfulness",
    "yes_rate",
    "negative_grounding_accuracy",
    "source_match",
    "baseline_delta",
    "per_property_accuracy",
    "alignment_rate",
    "calibration_ece",
    "metrics",
    "smoke_mode",
    "content_type_role_alignment_summary",
)


@dataclass(frozen=True)
class _GateProjection:
    """Compact projection of a single gate-result row for bucket emit."""

    gate_id: str
    severity: str
    passed: bool
    action: Optional[str]
    score: Optional[float]
    issue_count: int
    code: Optional[str]
    message: Optional[str]
    is_blocking_fail: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "severity": self.severity,
            "passed": self.passed,
            "action": self.action,
            "score": self.score,
            "issue_count": self.issue_count,
            "code": self.code,
            "message": self.message,
        }


class TrainforgeAssessmentQualityReport:
    """Aggregate Trainforge assessment-quality signals into one JSON.

    Parameters
    ----------
    phase_outputs:
        ``WorkflowRunner.run_workflow``'s accumulated ``phase_outputs``
        map. Each entry may carry a private ``_gate_results`` list (a
        sequence of dict-shaped ``GateResult`` payloads). The aggregator
        treats this map as read-only.
    course_code:
        Operator-facing course code. Surfaces as ``course_code`` so a
        diff across runs keys cleanly.
    run_id:
        Workflow ID.
    libv2_course_path:
        Optional override for the LibV2 course directory
        (``LibV2/courses/<slug>``). When unset the aggregator
        resolves this lazily from
        ``phase_outputs.libv2_archival.course_dir``.
    trainforge_dir:
        Optional override for the Trainforge workspace directory
        (used for the secondary ``quality_report.json::assessments``
        + fallback emit path). When unset the aggregator resolves
        from ``phase_outputs.trainforge_assessment.trainforge_dir``
        / ``phase_outputs.training_synthesis.corpus_dir``.
    decision_capture:
        Optional ``DecisionCapture`` instance. When wired the
        aggregator emits one ``trainforge_quality_aggregated`` event
        per ``build()`` call.
    """

    def __init__(
        self,
        phase_outputs: Mapping[str, Mapping[str, Any]],
        *,
        course_code: str = "",
        run_id: str = "",
        libv2_course_path: Optional[Path] = None,
        trainforge_dir: Optional[Path] = None,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.phase_outputs = phase_outputs or {}
        self.course_code = course_code or ""
        self.run_id = run_id or ""
        self.libv2_course_path = (
            Path(libv2_course_path) if libv2_course_path else None
        )
        self.trainforge_dir = (
            Path(trainforge_dir) if trainforge_dir else None
        )
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        """Walk all signals and build the unified report dict."""
        # 1. Synthesis-quality bucket — projection of the five
        # training_synthesis gates. Each entry resolves to None when
        # the gate didn't run (graceful degradation).
        synthesis_gates = self._extract_phase_gates("training_synthesis")
        synthesis_quality: Dict[str, Optional[Dict[str, Any]]] = {}
        for gate_id in _SYNTHESIS_QUALITY_GATE_IDS:
            projection = synthesis_gates.get(gate_id)
            synthesis_quality[gate_id] = (
                projection.to_dict() if projection else None
            )

        # 2. Trainforge-assessment summary fields (single_correct_rate,
        # bloom_alignment_rate, placeholder_rate). Score is the per-gate
        # ``score`` field on the GateResult payload.
        assessment_gates = self._extract_phase_gates(
            "trainforge_assessment"
        )

        # 3. Retrieval-grounding signal: walks both two-pass seam phases
        # so post-rewrite evidence is preferred over inter-tier when
        # both fired.
        retrieval_score = self._extract_retrieval_grounding_score()

        # 4. Pull total_questions / per_question_issues / placeholder_rate
        # / single_correct_rate / bloom_alignment_rate from
        # quality_report.json::assessments dimension when present.
        assessment_dim = self._read_assessment_dimension()

        summary = self._build_summary(
            assessment_dim=assessment_dim,
            assessment_gates=assessment_gates,
            retrieval_score=retrieval_score,
        )

        # 5. KG-quality bucket — projection of libv2_archival gates that
        # land at the manifest / KG-quality seam.
        archival_gates = self._extract_phase_gates("libv2_archival")
        kg_quality = self._build_kg_quality_bucket(archival_gates)

        # 6. Eval-summary passthrough — optional, only when the LibV2
        # course has an imported adapter with eval_report.json.
        eval_summary = self._read_eval_report()

        # 6b. Worker W3.B: phase_results.synthesis.promotion_ladder
        # passthrough from dataset_config.json::statistics.promotion_ladder.
        # Empty-shape placeholder when the dataset_config is missing the
        # block entirely (legacy run / pre-W3.B) so a downstream
        # consumer always sees the same key shape.
        promotion_ladder = self._read_promotion_ladder()

        # 7. per_question_issues passthrough.
        per_question_issues: List[Any] = []
        if assessment_dim:
            issues = assessment_dim.get("per_question_issues") or []
            if isinstance(issues, list):
                per_question_issues = list(issues)

        # 8. Determine top-level status before computing the promotion
        # decision so the rationale interpolates the same signals.
        # ``warn`` when degraded inputs (missing _gate_results) or
        # warning-only failures dominate; ``fail`` on hard blocking
        # signals; ``pass`` on a clean run.
        blocking_failures = self._collect_blocking_failures(
            synthesis_gates=synthesis_gates,
            synthesis_quality=synthesis_quality,
            kg_quality=kg_quality,
            assessment_gates=assessment_gates,
            archival_gates=archival_gates,
        )
        warnings = self._collect_warnings(
            synthesis_gates=synthesis_gates,
            synthesis_quality=synthesis_quality,
            kg_quality=kg_quality,
            assessment_gates=assessment_gates,
            archival_gates=archival_gates,
        )

        # 9. Promotion decision — heuristic; replaced by Wave 3 G1.
        promotion_decision = self._compute_promotion_decision(
            synthesis_quality=synthesis_quality,
            kg_quality=kg_quality,
            summary=summary,
            blocking_failures=blocking_failures,
            warnings=warnings,
            eval_summary=eval_summary,
        )

        if blocking_failures:
            status = "fail"
        elif (
            not synthesis_gates
            and not assessment_gates
            and not archival_gates
        ):
            status = "warn"
        elif warnings:
            status = "warn"
        else:
            status = "pass"

        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "course_code": self.course_code,
            "run_id": self.run_id,
            "status": status,
            "summary": summary,
            "synthesis_quality": synthesis_quality,
            "kg_quality": kg_quality,
            "eval_summary": eval_summary,
            "per_question_issues": per_question_issues,
            "blocking_failures": blocking_failures,
            "warnings": warnings,
            "promotion_decision": promotion_decision,
            # Worker W3.B: phase_results carries per-phase mirrored
            # blocks for downstream consumers (eval_report.json
            # ``training_corpus_promotion`` pass-through, future
            # per-phase widening). The synthesis sub-dict mirrors the
            # promotion-ladder telemetry from
            # dataset_config.json::statistics.promotion_ladder so the
            # eval-side pass-through can copy through unchanged.
            "phase_results": {
                "synthesis": {
                    "promotion_ladder": promotion_ladder,
                },
            },
        }

        # Best-effort decision-capture emit; never raises into the
        # aggregator build path.
        self._emit_aggregator_decision(report)

        return report

    def write(self, output_path: Path) -> Path:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved absolute path on success. Raises ``OSError``
        on filesystem failure — the workflow-runner caller wraps this
        in try/except so an aggregator failure does not abort the
        workflow.
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
    # Helpers — gate-result extraction
    # ------------------------------------------------------------------
    def _extract_phase_gates(
        self, phase_name: str
    ) -> Dict[str, _GateProjection]:
        """Project ``phase_outputs[phase].``_gate_results`` to a gate-id map."""
        phase_payload = self.phase_outputs.get(phase_name) or {}
        gate_results = phase_payload.get("_gate_results") or []
        out: Dict[str, _GateProjection] = {}
        for gr in gate_results:
            if not isinstance(gr, Mapping):
                continue
            gate_id = str(gr.get("gate_id") or "")
            if not gate_id:
                continue
            out[gate_id] = self._project_gate(gr)
        return out

    @staticmethod
    def _project_gate(gate_result: Mapping[str, Any]) -> _GateProjection:
        """Project a ``GateResult.to_dict()`` payload to ``_GateProjection``."""
        issues = list(gate_result.get("issues") or [])
        first = issues[0] if issues and isinstance(issues[0], Mapping) else None
        severity = str(gate_result.get("severity") or "critical").lower()
        passed = bool(gate_result.get("passed", True))
        score_raw = gate_result.get("score")
        score: Optional[float]
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        is_blocking_fail = (not passed) and severity == "critical"
        return _GateProjection(
            gate_id=str(gate_result.get("gate_id") or ""),
            severity=severity,
            passed=passed,
            action=gate_result.get("action"),
            score=score,
            issue_count=len(issues),
            code=(first or {}).get("code") if first else None,
            message=(first or {}).get("message") if first else None,
            is_blocking_fail=is_blocking_fail,
        )

    def _extract_retrieval_grounding_score(self) -> Optional[float]:
        """Pull ``score`` from the most recently-fired retrieval-grounding gate.

        Walks every phase in ``phase_outputs`` (not just the two two-pass
        seams) so a future wiring under a different phase still surfaces
        the score. Prefers post-rewrite over outline when both fired.
        """
        # Prefer rewrite over outline when both ran.
        priority_order = (
            "rewrite_assessment_retrieval_grounding",
            "outline_assessment_retrieval_grounding",
            "assessment_retrieval_grounding",
        )
        scores: Dict[str, float] = {}
        for phase_payload in self.phase_outputs.values():
            for gr in phase_payload.get("_gate_results") or []:
                if not isinstance(gr, Mapping):
                    continue
                gate_id = str(gr.get("gate_id") or "")
                if gate_id not in _RETRIEVAL_GROUNDING_GATE_IDS:
                    continue
                score_raw = gr.get("score")
                try:
                    if score_raw is not None:
                        scores[gate_id] = float(score_raw)
                except (TypeError, ValueError):
                    continue
        for gid in priority_order:
            if gid in scores:
                return scores[gid]
        return None

    # ------------------------------------------------------------------
    # Helpers — input artefact reads (graceful-degradation)
    # ------------------------------------------------------------------

    def _resolve_trainforge_dir(self) -> Optional[Path]:
        """Resolve the Trainforge workspace dir from phase outputs.

        Priority:
        1. Explicit ``self.trainforge_dir`` override (used by tests).
        2. ``phase_outputs.trainforge_assessment.trainforge_dir`` (the
           ``_generate_assessments`` output key).
        3. ``phase_outputs.training_synthesis.corpus_dir`` (the
           ``_synthesize_training`` output key).
        """
        if self.trainforge_dir is not None:
            return self.trainforge_dir
        ta = self.phase_outputs.get("trainforge_assessment") or {}
        tdir = ta.get("trainforge_dir")
        if tdir:
            return Path(tdir)
        ts = self.phase_outputs.get("training_synthesis") or {}
        cdir = ts.get("corpus_dir") or ts.get("trainforge_dir")
        if cdir:
            return Path(cdir)
        return None

    def _resolve_libv2_course_path(self) -> Optional[Path]:
        """Resolve the LibV2 course dir from ``phase_outputs.libv2_archival``.

        Priority:
        1. Explicit ``self.libv2_course_path`` override (used by tests).
        2. ``phase_outputs.libv2_archival.course_dir`` (the
           ``_archive_to_libv2`` output key).
        """
        if self.libv2_course_path is not None:
            return self.libv2_course_path
        archival = self.phase_outputs.get("libv2_archival") or {}
        course_dir = archival.get("course_dir")
        if course_dir:
            return Path(course_dir)
        return None

    def _read_promotion_ladder(self) -> Dict[str, Any]:
        """Read ``<trainforge_dir>/training_specs/dataset_config.json::statistics.promotion_ladder``.

        Worker W3.B mirror surface. Returns the empty-shape placeholder
        (zeros + empty histogram) when the dataset_config is missing
        the block entirely (legacy / pre-W3.B run) so the wire-shape
        downstream consumers see is invariant. Reason histogram keys
        are alphabetised on emit so a run-vs-run diff is byte-stable.
        """
        empty_ladder: Dict[str, Any] = {
            "candidate_pairs_total": 0,
            "validated_pairs_total": 0,
            "trainable_pairs_total": 0,
            "rejected_promotion_pairs": 0,
            "promotion_rejection_reasons": {},
        }
        tdir = self._resolve_trainforge_dir()
        if tdir is None:
            return empty_ladder
        config_path = tdir / "training_specs" / "dataset_config.json"
        if not config_path.exists():
            return empty_ladder
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "trainforge_assessment_quality_report: cannot read "
                "promotion_ladder from %s: %s",
                config_path, exc,
            )
            return empty_ladder
        statistics = payload.get("statistics") if isinstance(payload, Mapping) else None
        if not isinstance(statistics, Mapping):
            return empty_ladder
        ladder = statistics.get("promotion_ladder")
        if not isinstance(ladder, Mapping):
            return empty_ladder
        # Project canonical fields with defaults so a partially-emitted
        # block (e.g. mid-migration corpus) still produces a uniform
        # wire shape.
        reasons = ladder.get("promotion_rejection_reasons") or {}
        if not isinstance(reasons, Mapping):
            reasons = {}
        return {
            "candidate_pairs_total": int(
                ladder.get("candidate_pairs_total") or 0
            ),
            "validated_pairs_total": int(
                ladder.get("validated_pairs_total") or 0
            ),
            "trainable_pairs_total": int(
                ladder.get("trainable_pairs_total") or 0
            ),
            "rejected_promotion_pairs": int(
                ladder.get("rejected_promotion_pairs") or 0
            ),
            "promotion_rejection_reasons": {
                k: int(reasons[k] or 0) for k in sorted(reasons)
            },
        }

    def _read_assessment_dimension(self) -> Optional[Dict[str, Any]]:
        """Read ``<trainforge_dir>/quality/quality_report.json::assessments``."""
        tdir = self._resolve_trainforge_dir()
        if tdir is None:
            return None
        qr_path = tdir / "quality" / "quality_report.json"
        if not qr_path.exists():
            return None
        try:
            payload = json.loads(qr_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "trainforge_assessment_quality_report: cannot read %s: %s",
                qr_path, exc,
            )
            return None
        dim = payload.get("assessments")
        if isinstance(dim, Mapping):
            return dict(dim)
        return None

    def _read_eval_report(self) -> Optional[Dict[str, Any]]:
        """Read the most-recently-imported adapter's eval_report.json.

        Returns the projected passthrough dict (subset of canonical
        eval_report fields) when present; ``None`` otherwise.
        """
        course_path = self._resolve_libv2_course_path()
        if course_path is None:
            return None
        models_dir = course_path / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            return None
        try:
            candidates = sorted(
                p for p in models_dir.iterdir() if p.is_dir()
            )
        except OSError:
            return None
        if not candidates:
            return None
        # Lexicographically last subdir matches the
        # CourseforgeValidationReport's posture (Wave 3 G1 tightens this
        # to the published pointer in model_pointers.json).
        eval_path = candidates[-1] / "eval" / "eval_report.json"
        if not eval_path.exists():
            return None
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        return {
            field: payload.get(field)
            for field in _EVAL_REPORT_PASSTHROUGH_FIELDS
            if field in payload
        }

    # ------------------------------------------------------------------
    # Helpers — summary + bucket builders
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        *,
        assessment_dim: Optional[Mapping[str, Any]],
        assessment_gates: Mapping[str, _GateProjection],
        retrieval_score: Optional[float],
    ) -> Dict[str, Any]:
        """Build the ``summary`` block (six canonical fields)."""
        total_questions: Optional[int] = None
        single_correct_rate: Optional[float] = None
        bloom_alignment_rate: Optional[float] = None
        placeholder_rate: Optional[float] = None

        if assessment_dim:
            tq = assessment_dim.get("total_questions")
            if isinstance(tq, int):
                total_questions = tq
            # ``distinct_correct_answer_ratio`` is the closest analog to
            # GPT's "single_correct_rate" — the ratio of unique correct
            # answers to total. Different surface, identical semantics
            # (a question whose correct answer collides with another
            # signals authoring leakage).
            scr = assessment_dim.get("distinct_correct_answer_ratio")
            if isinstance(scr, (int, float)):
                single_correct_rate = float(scr)

            # Bloom-alignment rate is the share of questions whose
            # observed Bloom level was the targeted Bloom level.
            # Computed from ``bloom_distribution_observed`` when an
            # ``objectives_targeted`` Bloom is present; otherwise None.
            bloom_alignment_rate = self._compute_bloom_alignment_rate(
                assessment_dim
            )

            # Placeholder rate: count rows in per_question_issues that
            # carry a placeholder-class issue divided by total_questions.
            placeholder_rate = self._compute_placeholder_rate(
                assessment_dim
            )

        # Pull single_correct_rate from the assessment_quality gate
        # ``score`` when the assessment dimension lacked it. The
        # ``AssessmentQualityValidator`` writes that score per its
        # contract.
        if single_correct_rate is None:
            aq = assessment_gates.get("assessment_quality")
            if aq is not None and aq.score is not None:
                single_correct_rate = aq.score

        if bloom_alignment_rate is None:
            ba = assessment_gates.get("bloom_alignment")
            if ba is not None and ba.score is not None:
                bloom_alignment_rate = ba.score

        # answerable_rate / source_support_rate share the same retrieval-
        # grounding score. When the validator emits two gates (outline +
        # rewrite) the rewrite score wins (post-validation evidence).
        answerable_rate = retrieval_score
        source_support_rate = retrieval_score

        return {
            "total_questions": total_questions,
            "answerable_rate": answerable_rate,
            "single_correct_rate": single_correct_rate,
            "bloom_alignment_rate": bloom_alignment_rate,
            "placeholder_rate": placeholder_rate,
            "source_support_rate": source_support_rate,
        }

    @staticmethod
    def _compute_bloom_alignment_rate(
        assessment_dim: Mapping[str, Any],
    ) -> Optional[float]:
        """Best-effort derivation from per_question_issues.

        When per_question_issues carries ``BLOOM_MISALIGNED`` /
        ``BLOOM_OBSERVED_TARGET_MISMATCH`` rows the rate is
        ``(total - misaligned) / total``; otherwise ``None``.
        """
        total = assessment_dim.get("total_questions")
        if not isinstance(total, int) or total <= 0:
            return None
        issues = assessment_dim.get("per_question_issues") or []
        if not isinstance(issues, list):
            return None
        misaligned = 0
        for row in issues:
            if not isinstance(row, Mapping):
                continue
            row_issues = row.get("issues") or []
            if not isinstance(row_issues, list):
                continue
            for code in row_issues:
                if not isinstance(code, str):
                    continue
                upper = code.upper()
                if (
                    "BLOOM" in upper
                    and ("MISALIGN" in upper or "MISMATCH" in upper)
                ):
                    misaligned += 1
                    break
        return round(max(total - misaligned, 0) / total, 4)

    @staticmethod
    def _compute_placeholder_rate(
        assessment_dim: Mapping[str, Any],
    ) -> Optional[float]:
        """Count rows with a placeholder-class issue / total_questions."""
        total = assessment_dim.get("total_questions")
        if not isinstance(total, int) or total <= 0:
            return None
        issues = assessment_dim.get("per_question_issues") or []
        if not isinstance(issues, list):
            return None
        placeholder_count = 0
        for row in issues:
            if not isinstance(row, Mapping):
                continue
            row_issues = row.get("issues") or []
            if not isinstance(row_issues, list):
                continue
            for code in row_issues:
                if not isinstance(code, str):
                    continue
                upper = code.upper()
                if "PLACEHOLDER" in upper or "TEMPLATE" in upper:
                    placeholder_count += 1
                    break
        return round(placeholder_count / total, 4)

    @staticmethod
    def _build_kg_quality_bucket(
        archival_gates: Mapping[str, _GateProjection],
    ) -> Optional[Dict[str, Any]]:
        """Project ``libv2_archival`` KG-quality gates to a single bucket.

        Returns ``None`` when no KG gate ran (graceful degradation).
        """
        rows: List[Dict[str, Any]] = []
        any_blocking = False
        any_warning = False
        for gate_id in _KG_QUALITY_GATE_IDS:
            projection = archival_gates.get(gate_id)
            if projection is None:
                continue
            rows.append(projection.to_dict())
            if projection.is_blocking_fail:
                any_blocking = True
            elif not projection.passed and projection.severity == "warning":
                any_warning = True
        if not rows:
            return None
        return {
            "gates": rows,
            "summary": {
                "total": len(rows),
                "passed": sum(1 for r in rows if r.get("passed")),
                "failed": sum(
                    1 for r in rows if r.get("passed") is False
                ),
                "any_blocking_fail": any_blocking,
                "any_warning_only": any_warning,
            },
        }

    # ------------------------------------------------------------------
    # Helpers — failure / warning collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_blocking_failures(
        *,
        synthesis_gates: Mapping[str, _GateProjection],
        synthesis_quality: Mapping[str, Optional[Mapping[str, Any]]],
        kg_quality: Optional[Mapping[str, Any]],
        assessment_gates: Mapping[str, _GateProjection],
        archival_gates: Mapping[str, _GateProjection],
    ) -> List[Dict[str, Any]]:
        """Aggregate every critical-severity failed gate into one list.

        Walks both the canonical ``synthesis_quality`` bucket AND the
        full ``synthesis_gates`` map so non-canonical gates wired on
        ``training_synthesis`` (e.g. ``synthesis_quota``) still surface.
        """
        out: List[Dict[str, Any]] = []
        # Synthesis bucket — walk every gate on the phase, not just
        # the canonical 5, so wired gates outside the canonical list
        # (e.g. ``synthesis_quota``) still surface.
        for gate_id, projection in synthesis_gates.items():
            if projection.is_blocking_fail:
                out.append({
                    "phase": "training_synthesis",
                    "gate_id": gate_id,
                    "severity": projection.severity,
                    "code": projection.code,
                    "message": projection.message,
                })
        # Trainforge-assessment bucket — walk all gates seen on the
        # phase, not just the canonical id list, so any new gate auto-
        # surfaces.
        for gate_id, projection in assessment_gates.items():
            if projection.is_blocking_fail:
                out.append({
                    "phase": "trainforge_assessment",
                    "gate_id": gate_id,
                    "severity": projection.severity,
                    "code": projection.code,
                    "message": projection.message,
                })
        # KG-quality bucket.
        if kg_quality and kg_quality.get("summary", {}).get("any_blocking_fail"):
            for row in kg_quality.get("gates") or []:
                if (
                    row.get("severity") == "critical"
                    and row.get("passed") is False
                ):
                    out.append({
                        "phase": "libv2_archival",
                        "gate_id": row.get("gate_id"),
                        "severity": "critical",
                        "code": row.get("code"),
                        "message": row.get("message"),
                    })
        # Surface any other libv2_archival blocking gate the canonical
        # KG-quality list didn't catch (e.g. ``libv2_model``).
        for gate_id, projection in archival_gates.items():
            if gate_id in _KG_QUALITY_GATE_IDS:
                continue
            if projection.is_blocking_fail:
                out.append({
                    "phase": "libv2_archival",
                    "gate_id": gate_id,
                    "severity": projection.severity,
                    "code": projection.code,
                    "message": projection.message,
                })
        return out

    @staticmethod
    def _collect_warnings(
        *,
        synthesis_gates: Mapping[str, _GateProjection],
        synthesis_quality: Mapping[str, Optional[Mapping[str, Any]]],
        kg_quality: Optional[Mapping[str, Any]],
        assessment_gates: Mapping[str, _GateProjection],
        archival_gates: Mapping[str, _GateProjection],
    ) -> List[Dict[str, Any]]:
        """Aggregate warning-severity failed gates into one list.

        Walks the full ``synthesis_gates`` map so non-canonical
        warning-severity gates (e.g. ``synthesis_quota``) surface in
        the warnings bucket.
        """
        out: List[Dict[str, Any]] = []
        for gate_id, projection in synthesis_gates.items():
            if (
                projection.severity == "warning"
                and not projection.passed
            ):
                out.append({
                    "phase": "training_synthesis",
                    "gate_id": gate_id,
                    "severity": projection.severity,
                    "code": projection.code,
                    "message": projection.message,
                })
        for gate_id, projection in assessment_gates.items():
            if (
                projection.severity == "warning"
                and not projection.passed
            ):
                out.append({
                    "phase": "trainforge_assessment",
                    "gate_id": gate_id,
                    "severity": projection.severity,
                    "code": projection.code,
                    "message": projection.message,
                })
        for gate_id, projection in archival_gates.items():
            if (
                projection.severity == "warning"
                and not projection.passed
            ):
                out.append({
                    "phase": "libv2_archival",
                    "gate_id": gate_id,
                    "severity": projection.severity,
                    "code": projection.code,
                    "message": projection.message,
                })
        return out

    # ------------------------------------------------------------------
    # Promotion-decision heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_promotion_decision(
        *,
        synthesis_quality: Mapping[str, Optional[Mapping[str, Any]]],
        kg_quality: Optional[Mapping[str, Any]],
        summary: Mapping[str, Any],
        blocking_failures: Sequence[Mapping[str, Any]],
        warnings: Sequence[Mapping[str, Any]],
        eval_summary: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Three-way heuristic — ``failed`` / ``non_certified_archive`` / ``trainable``.

        See module docstring for the full table. Rationale string
        interpolates the contributing dynamic signals so it remains
        replayable post-hoc.
        """
        contributing_gate_ids: List[str] = []
        rationale_parts: List[str] = []

        # Hard signal #1: any blocking gate failure.
        if blocking_failures:
            for bf in blocking_failures:
                gid = bf.get("gate_id")
                if gid:
                    contributing_gate_ids.append(str(gid))
            rationale_parts.append(
                f"blocking_failures={len(blocking_failures)}"
            )

        # Hard signal #2: placeholder_rate > 0.0 (zero-tolerance).
        placeholder_rate = summary.get("placeholder_rate")
        placeholder_blocking = (
            isinstance(placeholder_rate, (int, float))
            and placeholder_rate > 0.0
        )
        if placeholder_blocking:
            rationale_parts.append(
                f"placeholder_rate={placeholder_rate:.4f} > 0.0"
            )

        # Hard signal #3: answerable_rate < 0.50.
        answerable_rate = summary.get("answerable_rate")
        answerable_blocking = (
            isinstance(answerable_rate, (int, float))
            and answerable_rate < 0.50
        )
        if answerable_blocking:
            rationale_parts.append(
                f"answerable_rate={answerable_rate:.4f} < 0.50"
            )

        # Hard signal #4: eval-gating blocking (when present).
        eval_blocking = False
        if isinstance(eval_summary, Mapping):
            # When the eval report is a smoke-mode sidecar, the gating
            # validator refuses to run on it (see EvalGatingValidator's
            # EVAL_REPORT_IS_SMOKE critical). Aggregator mirrors that
            # posture: a smoke-mode signal alone never trips ``failed``.
            if eval_summary.get("smoke_mode") is not True:
                # Pull conservative thresholds matching the eval-gating
                # validator defaults at Trainforge/CLAUDE.md ::
                # "Eval-gating thresholds".
                faithfulness = eval_summary.get("faithfulness")
                if (
                    isinstance(faithfulness, (int, float))
                    and faithfulness < 0.50
                ):
                    eval_blocking = True
                    rationale_parts.append(
                        f"eval_faithfulness={faithfulness:.4f} < 0.50"
                    )
                yes_rate = eval_summary.get("yes_rate")
                if (
                    isinstance(yes_rate, (int, float))
                    and yes_rate > 0.85
                ):
                    eval_blocking = True
                    rationale_parts.append(
                        f"eval_yes_rate={yes_rate:.4f} > 0.85"
                    )
                neg_grounding = eval_summary.get("negative_grounding_accuracy")
                if (
                    isinstance(neg_grounding, (int, float))
                    and neg_grounding < 0.50
                ):
                    eval_blocking = True
                    rationale_parts.append(
                        f"eval_negative_grounding="
                        f"{neg_grounding:.4f} < 0.50"
                    )

        is_failed = bool(
            blocking_failures
            or placeholder_blocking
            or answerable_blocking
            or eval_blocking
        )

        if is_failed:
            decision = "failed"
        elif warnings:
            decision = "non_certified_archive"
            rationale_parts.append(f"warnings={len(warnings)}")
        else:
            decision = "trainable"
            rationale_parts.append("clean run, no warnings")

        # Defensive enum guard.
        if decision not in _PROMOTION_DECISIONS:
            decision = "failed"
            rationale_parts.append(
                "decision fell outside enum; coerced to 'failed'"
            )

        # Stamp the dynamic signals so the rationale is always replayable.
        rationale_parts.append(
            f"total_questions={summary.get('total_questions')}, "
            f"answerable_rate={answerable_rate}, "
            f"single_correct_rate={summary.get('single_correct_rate')}, "
            f"placeholder_rate={placeholder_rate}, "
            f"resolved_decision={decision!r}"
        )

        return {
            "value": decision,
            "rationale": "; ".join(rationale_parts),
            "contributing_gate_ids": sorted(set(contributing_gate_ids)),
        }

    # ------------------------------------------------------------------
    # Decision capture
    # ------------------------------------------------------------------

    def _emit_aggregator_decision(
        self, report: Mapping[str, Any]
    ) -> None:
        """Emit one ``trainforge_quality_aggregated`` decision per build.

        Best-effort: any failure in the capture path is logged and
        swallowed. Rationale interpolates dynamic signals (total_questions,
        answerable_rate, single_correct_rate, placeholder_rate, resolved
        promotion_decision value) so it interpolates well above the
        20-char floor.
        """
        capture = self.decision_capture
        if capture is None:
            return
        try:
            summary = report.get("summary") or {}
            promo = report.get("promotion_decision") or {}
            decision_value = promo.get("value", "unknown")
            contributing = promo.get("contributing_gate_ids") or []
            blocking = report.get("blocking_failures") or []
            warnings = report.get("warnings") or []
            synthesis_quality = report.get("synthesis_quality") or {}
            kg_quality = report.get("kg_quality") or {}
            eval_summary = report.get("eval_summary")

            rationale = (
                f"Trainforge quality aggregated verdict={decision_value}, "
                f"total_questions={summary.get('total_questions')}, "
                f"answerable_rate={summary.get('answerable_rate')}, "
                f"single_correct_rate={summary.get('single_correct_rate')}, "
                f"placeholder_rate={summary.get('placeholder_rate')}, "
                f"bloom_alignment_rate="
                f"{summary.get('bloom_alignment_rate')}, "
                f"blocking={len(blocking)}, warnings={len(warnings)}, "
                f"synthesis_gate_count="
                f"{sum(1 for v in synthesis_quality.values() if v)}, "
                f"kg_quality_present={bool(kg_quality)}, "
                f"eval_summary_present={bool(eval_summary)}, "
                f"contributing_gate_ids_count={len(contributing)}"
            )
            capture.log_decision(
                decision_type="trainforge_quality_aggregated",
                decision=decision_value,
                rationale=rationale,
                context=str({
                    "course_code": report.get("course_code"),
                    "run_id": report.get("run_id"),
                    "status": report.get("status"),
                    "contributing_gate_ids": contributing,
                }),
                metrics={
                    "total_questions": summary.get("total_questions"),
                    "answerable_rate": summary.get("answerable_rate"),
                    "single_correct_rate": summary.get(
                        "single_correct_rate"
                    ),
                    "bloom_alignment_rate": summary.get(
                        "bloom_alignment_rate"
                    ),
                    "placeholder_rate": summary.get("placeholder_rate"),
                    "source_support_rate": summary.get(
                        "source_support_rate"
                    ),
                    "blocking_count": len(blocking),
                    "warning_count": len(warnings),
                    "promotion_decision": decision_value,
                    "contributing_gate_ids_count": len(contributing),
                },
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "trainforge_quality_aggregated: %s",
                exc,
            )


__all__ = [
    "TrainforgeAssessmentQualityReport",
    "SCHEMA_VERSION",
]
