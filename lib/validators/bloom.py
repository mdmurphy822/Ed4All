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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.bloom import get_verbs as _get_canonical_verbs

# Bloom's taxonomy verb indicators per level.
# Source of truth: schemas/taxonomies/bloom_verbs.json (loaded via
# lib.ontology.bloom). Migrated from a hand-maintained dict in Wave 1.2 /
# Worker H (REC-BL-01). Behavior-preserving: the canonical set is a
# superset of the previous hand-maintained list, so every pre-migration
# detection still fires.
BLOOM_VERBS: Dict[str, Set[str]] = _get_canonical_verbs()


def detect_bloom_level(stem: str) -> Optional[str]:
    """Detect the Bloom's taxonomy level from a question stem."""
    stem_lower = stem.lower()
    # Check for verb matches, prioritizing higher-order levels
    levels = ["create", "evaluate", "analyze", "apply", "understand", "remember"]
    for level in levels:
        for verb in BLOOM_VERBS[level]:
            if re.search(rf"\b{verb}\b", stem_lower):
                return level
    return None


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
        aligned = 0
        for q in questions:
            stem = q.get("stem", "")
            # Strip HTML so we don't try to detect verbs inside tags.
            stem_text = re.sub(r"<[^>]+>", " ", stem).strip()
            declared = q.get("bloom_level", "")
            detected = detect_bloom_level(stem_text)
            q_id = q.get("question_id", "unknown")

            if detected is None:
                # No Bloom verb found in the stem.
                if permissive_mode:
                    # Legacy behavior: count as aligned.
                    aligned += 1
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
