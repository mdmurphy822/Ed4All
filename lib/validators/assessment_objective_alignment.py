"""Assessment-Objective Alignment Validator (Wave 24 scope 7).

Fails closed when any assessment question's ``objective_id`` is a phantom
— i.e. does not appear in any chunk's ``learning_outcome_refs[]`` list.

Before Wave 24 landed, two disjoint LO naming schemes flowed through the
pipeline:

  * ``TO-NN`` / ``CO-NN`` — minted by Courseforge, emitted to HTML,
    harvested by Trainforge into ``chunks[*].learning_outcome_refs``.
  * ``{COURSE}_OBJ_N`` — minted by ``create_course_project``, routed to
    assessment generation. Every resulting
    ``assessments.json.questions[].objective_id`` was a phantom never
    referenced by any HTML page → 896 broken refs downstream.

Wave 24 unified the mint to the ``TO-NN`` / ``CO-NN`` scheme (see
``lib/ontology/learning_objectives.py``). This validator exists to
prevent the failure mode from resurfacing — if a future change reintroduces
a disjoint scheme, the ``trainforge_assessment`` phase fails closed here
rather than silently emitting 896 phantom refs into the training corpus.

Referenced by: ``config/workflows.yaml`` →
``textbook_to_course.trainforge_assessment.validation_gates[assessment_objective_alignment]``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_alignment_decision(
    capture: Any,
    *,
    question_id: str,
    declared_objective_ids: List[str],
    resolved_in_chunk: bool,
    n_chunks_searched: int,
    passed: bool,
    code: Optional[str],
    resolved_via_synthesized_objectives: bool = False,
) -> None:
    """Emit one ``assessment_objective_alignment_check`` decision per question.

    Per H3 W5 contract: per-question cardinality. Dynamic signals:
    question_id, declared_objective_id(s), resolved_in_chunk,
    n_chunks_searched.

    Wave 5 W5.E extension: ``resolved_via_synthesized_objectives``
    surfaces whether the union resolution surface was active for this
    audit (i.e. ``synthesized_objectives_path`` was provided + extant
    and contributed objective ids to the resolution set). Per-question
    rationale + metrics carry the flag so post-hoc replay can attribute
    a passing question to the chunks-side refs OR to the synthesized
    objectives' ``id`` set.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rationale = (
        f"AssessmentObjectiveAlignmentValidator audited question "
        f"{question_id!r}: declared_objective_ids={declared_objective_ids!r}, "
        f"resolved_in_chunk={resolved_in_chunk}, "
        f"n_chunks_searched={n_chunks_searched}, "
        f"resolved_via_synthesized_objectives={resolved_via_synthesized_objectives}, "
        f"failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "question_id": question_id,
        "declared_objective_ids": list(declared_objective_ids),
        "resolved_in_chunk": bool(resolved_in_chunk),
        "n_chunks_searched": int(n_chunks_searched),
        "resolved_via_synthesized_objectives": bool(
            resolved_via_synthesized_objectives
        ),
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="assessment_objective_alignment_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "assessment_objective_alignment_check: %s",
            exc,
        )


class AssessmentObjectiveAlignmentValidator:
    """Validator: every assessment.questions[].objective_id is chunk-resolvable."""

    name = "assessment_objective_alignment"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate alignment between assessment question objective_ids
        and chunk learning_outcome_refs.

        Expected inputs:
            assessments_path: Path to an assessments.json (or assessment.json).
                              Required.
            chunks_path: Path to chunks.jsonl (Trainforge corpus output).
                         Required.
            synthesized_objectives_path: Optional path to the
                ``synthesized_objectives.json`` emitted by the
                Wave 1.6 / Wave 24 ``course_planning`` phase.
                **Wave 5 W5.E resolution-surface union** — when this
                kwarg is provided AND the file exists, the resolution
                set against which ``question.objective_id`` is matched
                becomes ``chunk.learning_outcome_refs[]`` ∪ ``{lo["id"]
                for lo in flatten(synthesized_objectives)}``. This
                closes Finding F: a chunks-side empty
                ``learning_outcome_refs`` triggers phantom-ref
                criticality even when the assessment objective IS in
                the synthesized objectives. Loaded via
                :func:`lib.validators.abcd_objective._flatten_objectives`,
                which is the canonical loader handling both the
                Courseforge synthesized form
                (``terminal_objectives`` / ``chapter_objectives``) and
                the LibV2 archive form (``terminal_outcomes`` /
                ``component_objectives``). Back-compat: when unset OR
                the file does not exist (legacy corpora /
                ``rag_training`` workflow), behaviour is byte-identical
                to the pre-W5.E chunks-only resolution surface.
        """
        gate_id = inputs.get("gate_id", "assessment_objective_alignment")
        issues: List[GateIssue] = []
        capture = inputs.get("decision_capture")

        assessments_raw = inputs.get("assessments_path") or inputs.get("assessment_path")
        chunks_raw = inputs.get("chunks_path")
        synthesized_objectives_raw = inputs.get("synthesized_objectives_path")

        # Missing inputs → critical fail (the gate skips entirely when
        # the builder couldn't resolve them, so reaching here means the
        # builder believed it could but the files vanished).
        if not assessments_raw:
            return self._fail(
                gate_id,
                "MISSING_ASSESSMENTS_PATH",
                "assessments_path is required for AssessmentObjectiveAlignmentValidator",
            )
        if not chunks_raw:
            return self._fail(
                gate_id,
                "MISSING_CHUNKS_PATH",
                "chunks_path is required for AssessmentObjectiveAlignmentValidator",
            )

        assessments_path = Path(assessments_raw)
        chunks_path = Path(chunks_raw)

        if not assessments_path.exists():
            return self._fail(
                gate_id,
                "ASSESSMENTS_NOT_FOUND",
                f"Assessments file does not exist: {assessments_path}",
            )
        if not chunks_path.exists():
            return self._fail(
                gate_id,
                "CHUNKS_NOT_FOUND",
                f"Chunks file does not exist: {chunks_path}",
            )

        # Parse both inputs. Parse errors are critical.
        try:
            assessments_data = json.loads(
                assessments_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            return self._fail(
                gate_id,
                "INVALID_ASSESSMENTS_JSON",
                f"Failed to parse assessments JSON: {exc}",
                location=str(assessments_path),
            )

        # Resolve the synthesized-objectives path (W5.E union arm).
        # When provided AND the file exists, _collect_chunk_refs loads
        # the LO ids and unions them into the resolution surface. When
        # provided but missing, emit a warning-severity GateIssue and
        # fall back to the chunks-only surface (graceful-degrade — the
        # validator stays usable on legacy corpora that don't co-locate
        # the synthesized objectives next to the chunks).
        synthesized_path: Optional[Path] = None
        synthesized_objectives_present = False
        if synthesized_objectives_raw:
            candidate = Path(synthesized_objectives_raw)
            if candidate.exists():
                synthesized_path = candidate
                synthesized_objectives_present = True
            else:
                issues.append(GateIssue(
                    severity="warning",
                    code="SYNTHESIZED_OBJECTIVES_NOT_FOUND",
                    message=(
                        f"synthesized_objectives_path {candidate!s} does not "
                        "exist; falling back to chunks-only resolution surface."
                    ),
                    location=str(candidate),
                ))

        chunk_refs = self._collect_chunk_refs(
            chunks_path,
            synthesized_objectives_path=synthesized_path,
        )
        if chunk_refs is None:
            return self._fail(
                gate_id,
                "INVALID_CHUNKS_JSONL",
                f"Failed to parse chunks JSONL: {chunks_path}",
                location=str(chunks_path),
            )

        # Silent-degradation guard (C4 audit fix): a chunks file that
        # parses cleanly but yields zero learning_outcome_refs means the
        # upstream Trainforge chunking phase produced no chunks (or every
        # chunk dropped its refs). The gate would otherwise vacuously
        # accept any objective_id since the "covered by at least one
        # chunk's learning_outcome_refs" predicate is trivially false
        # for every question — but the failure mode would be reported
        # downstream as 100% phantom refs rather than as the upstream
        # chunking failure it actually is. Fail closed here with a
        # named code so operators know to inspect imscc_chunking.
        #
        # W5.E exception: when synthesized_objectives_path was provided
        # AND parsed successfully AND contributed refs to the union, an
        # empty chunks-side surface is no longer a silent-degradation
        # signal — the synthesized objectives are the structural source
        # of truth here. The check is whether the UNION resolution set
        # is empty.
        if not chunk_refs:
            return self._fail(
                gate_id,
                "ASSESSMENT_ALIGNMENT_NO_CHUNKS",
                (
                    f"Chunks file at {chunks_path} contained zero "
                    "learning_outcome_refs across all chunks "
                    f"(synthesized_objectives_path={synthesized_objectives_raw!r}). "
                    "Did the imscc_chunking / trainforge_assessment "
                    "phase run successfully? Without chunk refs every "
                    "assessment objective_id is vacuously phantom; "
                    "fail closed rather than ship 100% phantom refs."
                ),
                location=str(chunks_path),
            )

        # Normalize refs to lowercase for case-insensitive comparison —
        # Trainforge emits lowercase on chunk refs, Courseforge emits
        # mixed-case IDs. Opt-in preservation via TRAINFORGE_PRESERVE_LO_CASE
        # doesn't affect the gate: we compare by normalized form.
        normalized_refs: Set[str] = {r.lower() for r in chunk_refs if r}

        questions = self._extract_questions(assessments_data)
        if not questions:
            # Empty assessment file → skip with warning, not critical.
            issues.append(GateIssue(
                severity="warning",
                code="NO_QUESTIONS",
                message=(
                    "Assessment payload contains no questions to validate. "
                    "This may be a legitimate skip (optional phase) or an "
                    "upstream generation failure."
                ),
                location=str(assessments_path),
            ))
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=issues,
            )

        # Core alignment check: every question.objective_id (and optional
        # question.objective_ids list) must appear in normalized_refs.
        # Wave H3-W5 wiring: emit one
        # ``assessment_objective_alignment_check`` decision per
        # question audited so post-hoc replay can reconstruct which
        # specific objective_id resolved (or didn't) in chunk refs.
        mismatches: List[Dict[str, Any]] = []
        n_chunks_searched = len(normalized_refs)
        for idx, q in enumerate(questions):
            q_id = str(q.get("question_id") or q.get("id") or f"idx:{idx}")
            q_objectives = self._question_objectives(q)
            if not q_objectives:
                # Missing objective_id on a question → critical: the
                # question can't be evaluated for alignment.
                issues.append(GateIssue(
                    severity="critical",
                    code="QUESTION_MISSING_OBJECTIVE",
                    message=(
                        f"Question at index {idx} has no objective_id "
                        f"(question_id={q.get('question_id') or q.get('id') or '?'})"
                    ),
                    location=str(assessments_path),
                ))
                _emit_alignment_decision(
                    capture,
                    question_id=q_id,
                    declared_objective_ids=[],
                    resolved_in_chunk=False,
                    n_chunks_searched=n_chunks_searched,
                    passed=False,
                    code="QUESTION_MISSING_OBJECTIVE",
                    resolved_via_synthesized_objectives=(
                        synthesized_objectives_present
                    ),
                )
                continue
            q_unresolved: List[str] = []
            for obj_id in q_objectives:
                if obj_id.lower() not in normalized_refs:
                    q_unresolved.append(obj_id)
                    mismatches.append({
                        "question_index": idx,
                        "question_id": q.get("question_id") or q.get("id") or "?",
                        "objective_id": obj_id,
                    })
            q_resolved = not q_unresolved
            _emit_alignment_decision(
                capture,
                question_id=q_id,
                declared_objective_ids=list(q_objectives),
                resolved_in_chunk=q_resolved,
                n_chunks_searched=n_chunks_searched,
                passed=q_resolved,
                code=None if q_resolved else "PHANTOM_OBJECTIVE_REFS",
                resolved_via_synthesized_objectives=(
                    synthesized_objectives_present
                ),
            )

        if mismatches:
            # Roll up into a single critical issue with up to 10 samples
            # in the message body so the log is informative without spamming.
            samples = mismatches[:10]
            sample_str = ", ".join(
                f"{m['question_id']}→{m['objective_id']}" for m in samples
            )
            issues.append(GateIssue(
                severity="critical",
                code="PHANTOM_OBJECTIVE_REFS",
                message=(
                    f"{len(mismatches)} question(s) reference objective_ids "
                    f"not present in any chunk's learning_outcome_refs. "
                    f"Sample (first 10): {sample_str}. "
                    f"This indicates the disjoint-LO-scheme failure mode "
                    f"that Wave 24 closed; investigate whether "
                    f"synthesized_objectives.json drifted from "
                    f"phase_outputs.course_planning.objective_ids."
                ),
                location=str(assessments_path),
                suggestion=(
                    "Ensure phase_outputs.course_planning.objective_ids is "
                    "populated by plan_course_structure (Wave 24) and "
                    "forwarded into the trainforge_assessment phase via "
                    "workflows.yaml.inputs_from. Regenerate the IMSCC if "
                    "needed so chunks pick up the real TO-NN / CO-NN refs."
                ),
            ))

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0

        # Score: ratio of aligned questions.
        total = max(1, len(questions))
        aligned = total - len(mismatches)
        score = aligned / total

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _fail(
        gate_id: str,
        code: str,
        message: str,
        *,
        location: Optional[str] = None,
    ) -> GateResult:
        """Return a critical-failure GateResult with a single issue."""
        return GateResult(
            gate_id=gate_id,
            validator_name=AssessmentObjectiveAlignmentValidator.name,
            validator_version=AssessmentObjectiveAlignmentValidator.version,
            passed=False,
            issues=[GateIssue(
                severity="critical",
                code=code,
                message=message,
                location=location,
            )],
        )

    @staticmethod
    def _collect_chunk_refs(
        chunks_path: Path,
        *,
        synthesized_objectives_path: Optional[Path] = None,
    ) -> Optional[Set[str]]:
        """Collect every learning_outcome_ref across chunks.jsonl.

        When ``synthesized_objectives_path`` is provided AND the file
        exists, additionally load the canonical synthesized objectives
        via :func:`lib.validators.abcd_objective._flatten_objectives`
        (handles both the Courseforge synthesized form
        ``terminal_objectives`` / ``chapter_objectives`` AND the LibV2
        archive form ``terminal_outcomes`` / ``component_objectives``)
        and union every ``obj["id"]`` (lower-cased) into the resolution
        set. This is the W5.E union surface — closes Finding F where
        an empty chunks-side ``learning_outcome_refs`` triggered
        phantom-ref criticality even when the assessment objective IS
        in the synthesized objectives.

        When ``synthesized_objectives_path`` is unset OR the file does
        not exist, behaviour is byte-identical to the pre-W5.E
        chunks-only resolution surface (caller is responsible for
        emitting the warning when the path was provided but the file
        was missing).

        Returns ``None`` on chunks-side parse error. An empty set means
        no refs across either input — the caller's downstream check
        treats that as silent-degradation (per ``ASSESSMENT_ALIGNMENT_NO_CHUNKS``).
        """
        refs: Set[str] = set()
        try:
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Soft-skip malformed lines; Trainforge should
                        # have already filtered these.
                        continue
                    los = chunk.get("learning_outcome_refs") or []
                    if isinstance(los, list):
                        for lo in los:
                            if isinstance(lo, str):
                                refs.add(lo)
        except OSError:
            return None

        # W5.E: union the synthesized objectives' id set into the
        # resolution surface when the path is provided + extant.
        if (
            synthesized_objectives_path is not None
            and synthesized_objectives_path.exists()
        ):
            try:
                # Local import to avoid a hard module-load coupling
                # between this validator and abcd_objective; the
                # canonical loader stays in one place.
                from lib.validators.abcd_objective import _flatten_objectives

                payload = json.loads(
                    synthesized_objectives_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Failed to load synthesized_objectives_path %r for "
                    "AssessmentObjectiveAlignmentValidator W5.E union: %s",
                    str(synthesized_objectives_path),
                    exc,
                )
                return refs

            for lo in _flatten_objectives(payload):
                lo_id_raw = lo.get("id") or lo.get("objective_id")
                if isinstance(lo_id_raw, str):
                    lo_id = lo_id_raw.strip()
                    if lo_id:
                        refs.add(lo_id)

        return refs

    @staticmethod
    def _extract_questions(data: Any) -> List[Dict[str, Any]]:
        """Pull the flat list of questions from an assessments payload.

        Supports both:
          * ``{"questions": [...]}`` (assessment.json shape)
          * ``{"assessments": [{"questions": [...]}, ...]}`` (batch shape)
          * plain list ``[{...}, ...]``
        """
        if isinstance(data, list):
            return [q for q in data if isinstance(q, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("questions"), list):
                return [q for q in data["questions"] if isinstance(q, dict)]
            if isinstance(data.get("assessments"), list):
                out: List[Dict[str, Any]] = []
                for a in data["assessments"]:
                    if isinstance(a, dict) and isinstance(a.get("questions"), list):
                        out.extend(q for q in a["questions"] if isinstance(q, dict))
                return out
        return []

    @staticmethod
    def _question_objectives(q: Dict[str, Any]) -> List[str]:
        """Extract objective_id(s) from a single question payload.

        Supports single ``objective_id`` and plural ``objective_ids``.
        Empty lists + None values are filtered.
        """
        out: List[str] = []
        single = q.get("objective_id")
        if isinstance(single, str) and single:
            out.append(single)
        plural = q.get("objective_ids")
        if isinstance(plural, list):
            for o in plural:
                if isinstance(o, str) and o:
                    out.append(o)
        # Dedupe while preserving order.
        seen: Set[str] = set()
        deduped: List[str] = []
        for o in out:
            if o not in seen:
                seen.add(o)
                deduped.append(o)
        return deduped


__all__ = ["AssessmentObjectiveAlignmentValidator"]
