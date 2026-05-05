"""
Bloom's Taxonomy Alignment Validator

Validates assessment alignment with Bloom's taxonomy levels:
- Remember, Understand, Apply, Analyze, Evaluate, Create
- Verifies question stems match targeted cognitive level
- Checks distribution across taxonomy levels
- Validates alignment between objectives and assessment items

Referenced by: config/workflows.yaml (rag_training assessment_generation phase)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from lib.ontology.bloom import detect_bloom_level as _canonical_detect_bloom_level
from lib.ontology.bloom import get_verbs as _get_canonical_verbs
from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_bloom_alignment_decision(
    capture: Any,
    *,
    question_id: str,
    declared_level: str,
    detected_level: Optional[str],
    match: bool,
    permissive_mode: bool,
    aligned: bool,
) -> None:
    """Emit one ``bloom_alignment_check`` per question.

    Per H3 W5 contract: per-question cardinality. Dynamic signals:
    question_id, declared_level, detected_level, match.
    """
    if capture is None:
        return
    decision = "aligned" if aligned else "unaligned"
    rationale = (
        f"BloomAlignmentValidator audited question {question_id!r}: "
        f"declared_level={declared_level or 'n/a'}, "
        f"detected_level={detected_level or 'none'}, "
        f"verb_match={match}, permissive_mode={permissive_mode}, "
        f"aligned={aligned}."
    )
    metrics: Dict[str, Any] = {
        "question_id": question_id,
        "declared_level": declared_level or "",
        "detected_level": detected_level or "",
        "match": bool(match),
        "permissive_mode": bool(permissive_mode),
        "aligned": bool(aligned),
    }
    try:
        capture.log_decision(
            decision_type="bloom_alignment_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on bloom_alignment_check: %s",
            exc,
        )

# Bloom's taxonomy verb indicators per level.
# Source of truth: schemas/taxonomies/bloom_verbs.json (loaded via
# lib.ontology.bloom). Migrated from a hand-maintained dict in Wave 1.2 /
# Worker H (REC-BL-01). Behavior-preserving: the canonical set is a
# superset of the previous hand-maintained list, so every pre-migration
# detection still fires.
BLOOM_VERBS: Dict[str, Set[str]] = _get_canonical_verbs()


def detect_bloom_level(stem: str) -> Optional[str]:
    """Detect the Bloom's taxonomy level from a question stem.

    Wave 55: delegates to ``lib.ontology.bloom.detect_bloom_level`` (the
    canonical matcher) and discards the verb. The pre-Wave-55 local loop
    iterated ``create → remember`` and used ``re.search(\\b{verb}\\b, ...)``
    — behavior-preserving for this wrapper's ``Optional[str]`` signature
    but duplicated the detection logic. Delegation removes the duplicate
    and automatically picks up any future additions to the canonical
    matcher (e.g., longest-multi-word-verb ties).
    """
    level, _verb = _canonical_detect_bloom_level(stem)
    return level


class BloomAlignmentValidator:
    """Validates assessment alignment with Bloom's taxonomy."""

    name = "bloom_alignment"
    version = "1.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate Bloom's taxonomy alignment.

        Expected inputs:
            assessment_path: Path to assessment JSON
            assessment_data: Assessment dict (alternative to path)
            target_levels: List of targeted Bloom's levels (optional)
            min_alignment_score: Minimum alignment score (default 0.7)
            permissive_mode: Back-compat flag (default False, Wave 26). When
                True, verb-less stems count as aligned (pre-Wave-26
                behavior). When False (default), verb-less stems count
                as UNALIGNED and emit per-question VERB_LESS_STEM
                diagnostics.
        """
        gate_id = inputs.get("gate_id", "bloom_alignment")
        issues: List[GateIssue] = []
        min_score = inputs.get("min_alignment_score", 0.7)
        permissive_mode = bool(inputs.get("permissive_mode", False))
        capture = inputs.get("decision_capture")

        # Load assessment data
        data = inputs.get("assessment_data")
        if not data and inputs.get("assessment_path"):
            path = Path(inputs["assessment_path"])
            if not path.exists():
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[
                        GateIssue(
                            severity="error",
                            code="FILE_NOT_FOUND",
                            message=f"Assessment file not found: {path}",
                        )
                    ],
                )
            data = json.loads(path.read_text(encoding="utf-8"))

        if not data:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="NO_DATA",
                        message="No assessment data provided",
                    )
                ],
            )

        questions = data.get("questions", [])
        if not questions:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="NO_QUESTIONS",
                        message="Assessment contains no questions",
                    )
                ],
            )

        target_levels = set(inputs.get("target_levels", []))

        # Check each question's Bloom alignment.
        # Wave 26 fix: verb-less stems (detect_bloom_level == None) are
        # treated as UNALIGNED by default. The legacy "None counts as
        # aligned" behavior is preserved behind permissive_mode=True for
        # back-compat with fixtures that rely on the old scoring.
        # Wave H3-W5 wiring: emit one ``bloom_alignment_check`` per
        # question audited so post-hoc replay can reconstruct
        # declared-vs-detected per-question alignment trail.
        aligned = 0
        for q in questions:
            stem = q.get("stem", "")
            # Strip HTML so we don't try to detect verbs inside tags.
            stem_text = re.sub(r"<[^>]+>", " ", stem).strip()
            declared = q.get("bloom_level", "")
            detected = detect_bloom_level(stem_text)
            q_id = q.get("question_id", "unknown")
            q_aligned = False
            q_match = False

            if detected is None:
                # No Bloom verb found in the stem.
                if permissive_mode:
                    # Legacy behavior: count as aligned.
                    aligned += 1
                    q_aligned = True
                else:
                    # Wave 26 strict: count as unaligned and emit diagnostic.
                    excerpt = stem_text[:80]
                    if len(stem_text) > 80:
                        excerpt += "..."
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="VERB_LESS_STEM",
                            message=(
                                f"Question {q_id}: stem has no detectable "
                                f"Bloom verb: '{excerpt}'"
                            ),
                        )
                    )
            elif declared and detected != declared:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="BLOOM_MISMATCH",
                        message=(
                            f"Question {q_id}: declared '{declared}' "
                            f"but stem suggests '{detected}'"
                        ),
                    )
                )
            else:
                # detected is not None and either matches declared OR
                # declared is empty — treat as aligned.
                aligned += 1
                q_aligned = True
                # match is True only if declared was non-empty AND
                # equals detected (not the empty-declared positive
                # path).
                q_match = bool(declared) and detected == declared

            _emit_bloom_alignment_decision(
                capture,
                question_id=str(q_id),
                declared_level=str(declared or ""),
                detected_level=detected,
                match=q_match,
                permissive_mode=permissive_mode,
                aligned=q_aligned,
            )

            # Check target level coverage
            if target_levels and declared not in target_levels:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="OFF_TARGET_LEVEL",
                        message=(
                            f"Question {q_id}: level '{declared}' "
                            f"not in target levels {sorted(target_levels)}"
                        ),
                    )
                )

        alignment_score = aligned / len(questions) if questions else 0.0
        passed = alignment_score >= min_score

        if not passed:
            issues.append(
                GateIssue(
                    severity="error",
                    code="LOW_ALIGNMENT",
                    message=(
                        f"Bloom alignment score {alignment_score:.2f} "
                        f"below minimum {min_score}"
                    ),
                )
            )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=alignment_score,
            issues=issues,
        )
