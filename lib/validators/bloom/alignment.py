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


#: A cloze (fill-in-the-blank) gap marker. Four shapes cover every emitter in
#: the codebase and every external QTI import seen so far: a rule of underscores
#: (``_______``), a bracketed or braced ``blank`` token, a run of ellipses, and
#: a rule of em/horizontal/full-width dashes some publishers use as the gap.
#: Each requires a RUN (3+ characters, 2+ ellipses) so ordinary punctuation and
#: code identifiers never register. Deliberately structural —
#: no subject vocabulary, no publisher phrase list — so a different book, a
#: different discipline and a different converter all exercise the same rule.
_CLOZE_GAP_RE = re.compile(
    r"_{3,}"
    r"|\[\s*blank\s*\]"
    r"|\{\s*blank\s*\}"
    r"|…{2,}"
    r"|[—―＿]{3,}",
    re.IGNORECASE,
)


def stem_is_cloze(stem_text: str) -> bool:
    """True when the stem is a cloze / fill-in-the-blank sentence.

    A cloze stem is *a sentence with a gap in it*, not an instruction: its
    cognitive task is carried by the item TYPE and by the gap itself, and the
    only imperative it can carry is the item-type-appropriate one the emitter
    prepends ("Complete the following: …"). Demanding a Bloom cognitive verb
    inside such a stem is category-inappropriate — ``_______ is a multiple of
    4.`` is a correctly-shaped item, and flagging it as ``VERB_LESS_STEM``
    was a linter false positive that (at scale) escalated to a blocking
    ``PERVASIVE_VERBLESS_STEMS`` critical.

    Detection is on the STEM SHAPE rather than on ``question_type`` on
    purpose: an imported cartridge routinely mislabels a cloze item as
    ``short_answer`` / ``essay``, and a genuinely verb-less non-cloze stem
    carries no gap marker and so is still reported.

    Callers must pass TAG-STRIPPED text; the gap markers survive stripping.
    """
    return bool(_CLOZE_GAP_RE.search(stem_text or ""))


#: Finite copula / auxiliary verbs — the hinge of a declarative proposition.
#: Same closed function-word class the content extractor screens declaratives
#: with, kept in sync deliberately: a sentence the miner accepted as a factual
#: claim is the same sentence the linter has to recognise as a proposition.
_FINITE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|will|would|shall|should|"
    r"may|might|must|does|do|did|means|equals)\b",
    re.IGNORECASE,
)

#: Function words that can OPEN a subject noun phrase (determiners,
#: quantifiers, demonstratives, possessives, subject pronouns). An English
#: IMPERATIVE opens with a bare verb instead — "Find three consecutive
#: integers whose sum is -36." — so requiring one of these is the structural
#: discriminator between a proposition to judge and a directive to follow.
#: Closed function-word list: no subject vocabulary, no domain terms.
_SUBJECT_OPENERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "all", "any", "both", "each", "either", "every", "few", "many",
    "most", "much", "neither", "no", "none", "one", "several", "some",
    "his", "her", "hers", "its", "their", "theirs", "our", "ours",
    "your", "yours", "my", "mine",
    "he", "she", "it", "they", "we", "you", "i", "there",
})

#: A subject that is a lone pronoun / demonstrative has no antecedent once the
#: sentence is lifted out of its paragraph, so the item is unanswerable on its
#: own. Those are a genuine MINING defect (fixed at the harvest layer) and must
#: keep being reported here rather than exempted.
_BARE_ANAPHORIC_SUBJECTS = frozenset({
    "this", "that", "these", "those", "it", "they", "them",
    "he", "she", "there", "here", "one", "both", "each", "such",
})

_WORD_RE = re.compile(r"[A-Za-z0-9'’\-]+")


def stem_is_declarative_proposition(stem_text: str) -> bool:
    """True when the stem is a self-contained declarative PROPOSITION.

    A true/false stem is a claim the learner judges, not an instruction the
    learner follows: its cognitive demand rides on the item type, so it can
    never carry a Bloom imperative and ``VERB_LESS_STEM`` is category-
    inappropriate for it — the same argument that justifies the cloze
    exemption in :func:`stem_is_cloze`.

    The exemption is deliberately NOT "a true/false item may have any stem".
    All four structural conditions must hold:

    1. **Not interrogative** — no ``?``.
    2. **Terminally punctuated** — ends ``.`` / ``!``, so a bare fragment
       ("Step 2", a stray operand) is never exempted.
    3. **Has a finite verb** — a copula/auxiliary hinge, so the stem asserts
       something rather than naming it.
    4. **Has a real subject before that verb**, whose first token is a
       subject-opening function word. This is what keeps an IMPERATIVE out:
       "Find three consecutive integers whose sum is -36." has a finite verb
       ("is") but opens on a bare verb, not a determiner, so it stays
       reported. A BARE anaphoric subject ("This is determined by its
       position…") is likewise refused — that is a real mining defect, not a
       linter artifact.

    Condition 4 is conservative by construction: a legitimate proposition that
    opens on a bare common noun ("Absolute value is the distance from zero.")
    is NOT exempted and keeps warning. That direction is deliberate — a missed
    exemption costs a warning, a wrong exemption silently retires the rule.

    Callers must pass TAG-STRIPPED text.
    """
    text = re.sub(r"\s+", " ", stem_text or "").strip()
    if not text or "?" in text:
        return False
    if not text.endswith((".", "!")):
        return False
    match = _FINITE_VERB_RE.search(text)
    if match is None:
        return False
    subject_tokens = _WORD_RE.findall(text[: match.start()])
    if not subject_tokens:
        return False
    first = subject_tokens[0].lower()
    if len(subject_tokens) == 1 and first in _BARE_ANAPHORIC_SUBJECTS:
        return False
    return first in _SUBJECT_OPENERS


def _is_declarative_tf(question: Any, stem_text: str) -> bool:
    """True when ``question`` is a true/false item with a proposition stem.

    BOTH conditions are required. The item-type half keeps the exemption from
    leaking onto multiple-choice / short-answer stems (which SHOULD carry a
    task verb); the stem-shape half keeps it from covering the malformed
    true/false items — apparatus fragments, dangling anaphors, exercise
    directives — that are real defects.
    """
    from lib.validators._assessment_helpers.placeholders import (
        _normalize_question_type,
    )

    if _normalize_question_type(question) != "true_false":
        return False
    return stem_is_declarative_proposition(stem_text)


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
                elif stem_is_cloze(stem_text) or _is_declarative_tf(
                    q, stem_text
                ):
                    # Two item types carry their cognitive demand in the TYPE
                    # rather than in a stem imperative: a cloze (a sentence
                    # with a gap) and a true/false (a proposition to judge).
                    # For both, "no Bloom verb" is the CORRECT shape, not a
                    # defect. Count aligned (scoring them unaligned would
                    # depress the score for structurally valid item types) and
                    # emit no diagnostic.
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
