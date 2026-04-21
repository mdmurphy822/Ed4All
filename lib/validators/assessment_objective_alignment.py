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
from typing import Any, Dict, Iterable, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


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
        """
        gate_id = inputs.get("gate_id", "assessment_objective_alignment")
        issues: List[GateIssue] = []

        assessments_raw = inputs.get("assessments_path") or inputs.get("assessment_path")
        chunks_raw = inputs.get("chunks_path")

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

        chunk_refs = self._collect_chunk_refs(chunks_path)
        if chunk_refs is None:
            return self._fail(
                gate_id,
                "INVALID_CHUNKS_JSONL",
                f"Failed to parse chunks JSONL: {chunks_path}",
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
        mismatches: List[Dict[str, Any]] = []
        for idx, q in enumerate(questions):
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
                continue
            for obj_id in q_objectives:
                if obj_id.lower() not in normalized_refs:
                    mismatches.append({
                        "question_index": idx,
                        "question_id": q.get("question_id") or q.get("id") or "?",
                        "objective_id": obj_id,
                    })

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
    def _collect_chunk_refs(chunks_path: Path) -> Optional[Set[str]]:
        """Collect every learning_outcome_ref across chunks.jsonl.

        Returns None on parse error. Empty set means no refs — which
        means all assessment objective_ids will be flagged as phantoms
        (correct failure mode).
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
