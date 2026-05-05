"""Worker W6 — ``BloomStructuralEnforcementValidator``.

Closes the GPT-feedback §1.5 surface gap (Bloom enforcement is
classification-only). Whereas
:class:`lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator`
*classifies* the surface text and complains when the BERT ensemble
disagrees with the declared ``bloom_level``, this validator *enforces*
deterministic structural rules per Bloom level — clause-count floor +
verb-set membership + answer-length floor — so a rewrite tier that
emits a ``"What is TCP?"`` stem on a block declared
``bloom_level="analyze"`` fails closed regardless of what the
classifier says about the verb.

The two validators are complementary:

- ``BloomClassifierDisagreement*`` — soft signal, ML-based. Catches
  semantic drift (verb is "remember"-flavoured, ensemble says
  "remember", declared said "analyze").
- ``BloomStructuralEnforcement*`` — hard rule, regex-based. Catches
  structural mismatch (declared "analyze" but stem has zero
  comparison markers, fewer than 2 clauses, or fails the verb-set
  check). Pure CPU; no LLM, no embeddings, no transformer extras.

Per the W6 plan, this validator is a forcing-function on the upstream
rewrite prompt: a "remember"-shaped stem on an "analyze"-declared
block fails closed, kicks ``action="regenerate"``, and the router
re-rolls with the structural-failure remediation suffix appended.

Per-level rule table (from `plans/gpt-feedback-w2-w7-execution-2026-05.md`
§3.D, verb sets sourced from ``lib/ontology/bloom.py``):

+-----------+----------+--------------------------------------+--------------+
| Level     | Clauses  | Stem markers + verb constraint       | Answer floor |
+-----------+----------+--------------------------------------+--------------+
| remember  | >=1      | verb ∈ remember-set                  | any          |
| understand| >=1      | verb ∈ understand-set                | any          |
| apply     | >=2      | verb ∈ apply-set                     | >=4 tokens   |
| analyze   | >=2      | verb ∈ analyze-set + comparison      | >=6 tokens   |
|           |          | marker (compare/contrast/...)        |              |
| evaluate  | >=2      | verb ∈ evaluate-set + judgment       | >=8 tokens   |
|           |          | marker (justify/assess/...)          |              |
| create    | >=2      | verb ∈ create-set + design marker    | open-ended   |
|           |          | (design/construct/...)               | answer       |
+-----------+----------+--------------------------------------+--------------+

Failure GateIssue codes (all critical, action="regenerate"):

- ``BLOOM_STRUCTURE_INSUFFICIENT_CLAUSES`` — fewer clauses than the
  level's floor.
- ``BLOOM_VERB_MISMATCH`` — stem doesn't carry a verb from the
  declared level's verb set.
- ``BLOOM_ANSWER_TOO_SHORT`` — answer text shorter than the level's
  token floor.
- ``BLOOM_MISSING_COMPARISON_MARKER`` — analyze level missing
  comparison/contrast marker.
- ``BLOOM_MISSING_JUDGMENT_MARKER`` — evaluate level missing
  justify/assess marker.
- ``BLOOM_MISSING_DESIGN_MARKER`` — create level missing
  design/construct marker.
- ``CREATE_LEVEL_MUST_BE_OPEN_ENDED`` — create level emitted
  multiple-choice / true-false (must be essay / short-answer).

Per-validate() decision capture: one
``decision_type="bloom_structural_enforcement_check"`` event with
rationale interpolating the audited / passing / failing block counts
+ the per-level distribution. The validator does NOT instantiate a
:class:`DecisionCapture` itself — it accepts an optional ``capture``
arg from the caller (router / runner) so the test surface can pass
an in-memory stub without touching disk. Mirrors the wiring pattern
in ``lib/validators/courseforge_outline_shacl.py``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.bloom import BLOOM_LEVELS, get_verbs

logger = logging.getLogger(__name__)


#: Block types whose declared ``bloom_level`` we structurally enforce.
#: Per the plan, the W6 surface targets assessment items specifically
#: — that's where the "remember-shaped stem on analyze block" failure
#: mode bites. Objective blocks have their own ABCD + Bloom-alignment
#: gates; structural enforcement on objective stems is a separate
#: surface (would re-fire on every "Define X" objective and false-
#: positive at scale).
_AUDITED_BLOCK_TYPES: frozenset = frozenset({"assessment_item"})


#: Cap on per-block GateIssue emit so a uniformly-broken batch can't
#: drown the gate report. Mirrors the convention in
#: ``inter_tier_gates.py`` and ``bloom_classifier_disagreement.py``.
_ISSUE_LIST_CAP: int = 80


#: Comparison/contrast markers required by ``analyze``-level stems.
_COMPARISON_MARKERS: frozenset = frozenset({
    "compare",
    "contrast",
    "differentiate",
    "distinguish",
    "versus",
    "vs",
    "differences",
    "similarities",
})


#: Judgment / reasoned-position markers required by ``evaluate``-level
#: stems.
_JUDGMENT_MARKERS: frozenset = frozenset({
    "justify",
    "assess",
    "evaluate",
    "critique",
    "defend",
    "judge",
    "appraise",
})


#: Design / generation markers required by ``create``-level stems.
_DESIGN_MARKERS: frozenset = frozenset({
    "design",
    "construct",
    "develop",
    "formulate",
    "compose",
    "generate",
    "produce",
    "build",
    "create",
})


#: Per-level required clause floor. ``remember`` and ``understand`` are
#: single-clause-friendly ("What is TCP?"); higher cognitive levels
#: need at least two clauses to carry the cognitive demand.
_CLAUSE_FLOORS: Dict[str, int] = {
    "remember": 1,
    "understand": 1,
    "apply": 2,
    "analyze": 2,
    "evaluate": 2,
    "create": 2,
}


#: Per-level required answer-token floor. Token = whitespace-separated
#: word. ``create`` is special-cased in the validator: rather than a
#: token floor, the answer must be open-ended (essay / short-answer
#: question_type), so the value here is unused for that level.
_ANSWER_TOKEN_FLOORS: Dict[str, int] = {
    "remember": 0,
    "understand": 0,
    "apply": 4,
    "analyze": 6,
    "evaluate": 8,
    "create": 0,  # ignored; create uses open-ended check
}


#: Question types accepted for ``create``-level assessments. Anything
#: with a fixed answer set (multiple_choice / true_false / matching)
#: structurally cannot capture ``create``-level cognitive demand.
_OPEN_ENDED_QUESTION_TYPES: frozenset = frozenset({
    "essay",
    "short_answer",
    "short-answer",
    "shortanswer",
    "open_ended",
    "open-ended",
    "free_response",
    "free-response",
})


#: Coordinating / subordinating conjunctions counted toward clause
#: separation. Combined with sentence-final ``?`` / ``.`` and inline
#: ``,`` / ``;`` separators in :func:`_count_clauses`.
_CLAUSE_CONJUNCTIONS: frozenset = frozenset({
    "and",
    "but",
    "or",
    "while",
    "although",
    "because",
    "since",
    "whereas",
    "if",
    "then",
})


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b[\w'-]+\b")
_SENTENCE_END_RE = re.compile(r"[.?!]+(?=\s|$)")


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Cheap surface extractor matching the helper in
    ``inter_tier_gates.py`` / ``bloom_classifier_disagreement.py``.
    """
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _count_clauses(stem: str) -> int:
    """Approximate clause count for ``stem``.

    Heuristic: count terminal sentence punctuation (``.`` / ``?`` /
    ``!``) AT LEAST ONCE plus inline clause separators (``,`` / ``;``)
    plus coordinating / subordinating conjunctions. Floors at 1 for
    any non-empty stem because every stem is at least one clause.
    """
    if not stem.strip():
        return 0
    # Start at 1: any non-empty stem is at least one clause.
    count = 1
    # Each inline ``,`` / ``;`` separator adds a clause.
    count += stem.count(",")
    count += stem.count(";")
    # Each coordinating / subordinating conjunction (whole-word match)
    # adds a clause.
    lowered = stem.lower()
    for conj in _CLAUSE_CONJUNCTIONS:
        # ``\b`` so "andrew" doesn't trigger "and".
        count += len(re.findall(rf"\b{re.escape(conj)}\b", lowered))
    # Multiple sentences also add clauses; count terminal punctuation
    # past the first.
    sentence_ends = len(_SENTENCE_END_RE.findall(stem))
    if sentence_ends > 1:
        count += sentence_ends - 1
    return count


def _count_tokens(text: str) -> int:
    """Whitespace-tolerant word count via ``\\b\\w+\\b`` regex."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def _verb_in_level_set(stem: str, level: str, verb_sets: Dict[str, set]) -> bool:
    """True iff ``stem`` (case-insensitive) carries a whole-word match
    for any verb in the canonical ``level`` verb set.

    Source of truth is ``lib/ontology/bloom.py::get_verbs()`` — no
    duplicate registry per the W6 acceptance criteria.
    """
    if not stem:
        return False
    lowered = stem.lower()
    verbs = verb_sets.get(level, set())
    for verb in verbs:
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            return True
    return False


def _stem_has_marker(stem: str, markers: frozenset) -> bool:
    """Whole-word match for any marker in ``markers``."""
    if not stem:
        return False
    lowered = stem.lower()
    for marker in markers:
        if re.search(rf"\b{re.escape(marker)}\b", lowered):
            return True
    return False


def _block_attr(block: Any, key: str) -> Any:
    """Get ``block.<key>`` for dataclass blocks OR ``block[<key>]`` for dicts."""
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, dict):
        return block.get(key)
    return None


def _extract_stem(block: Any) -> Optional[str]:
    """Pull the stem text from a Block (or block-shaped dict).

    Dict path priority: ``content["stem"]`` (assessment_item canonical
    field) > ``content["statement"]`` > ``content["text"]``.
    Str path: strips HTML, returns the text body (rewrite-tier shape).
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        for key in ("stem", "statement", "text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(content, str):
        stripped = _strip_html(content)
        return stripped if stripped else None
    return None


def _extract_answer_text(block: Any) -> str:
    """Pull a representative answer surface from an assessment_item Block.

    For multiple-choice blocks: concatenate the correct answer's text.
    For short-answer / essay: use ``content["correct_answer"]`` /
    ``content["sample_answer"]``. Falls back to empty string when no
    answer surface is present (the answer-floor check then trips).
    """
    content = getattr(block, "content", None)
    if not isinstance(content, dict):
        return ""
    # Direct answer fields (short-answer / essay).
    for key in ("correct_answer", "sample_answer", "answer"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Multiple-choice: pull the option(s) tagged correct.
    options = content.get("options") or content.get("choices")
    if isinstance(options, list):
        correct_texts: List[str] = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            is_correct = opt.get("correct") or opt.get("is_correct")
            text = opt.get("text") or opt.get("answer") or opt.get("value")
            if is_correct and isinstance(text, str):
                correct_texts.append(text.strip())
        if correct_texts:
            return " ".join(correct_texts)
    return ""


def _extract_question_type(block: Any) -> Optional[str]:
    """Pull the ``question_type`` discriminator (mc / essay / etc.).

    Looks at ``content["question_type"]`` first, then ``template_type``
    on the Block dataclass, then ``content["type"]``.
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        for key in ("question_type", "type"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    template_type = _block_attr(block, "template_type")
    if isinstance(template_type, str) and template_type.strip():
        return template_type.strip().lower()
    return None


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` from ``inputs``.

    Mirrors the ``inputs["blocks"]`` contract used by the inter-tier
    gates. Returns ``(blocks, error_issue)``.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances or block-shaped dicts."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


class BloomStructuralEnforcementValidator:
    """Worker W6 — Bloom structural enforcement.

    Per-block contract:

    1. Skip blocks whose ``block_type`` is not in
       :data:`_AUDITED_BLOCK_TYPES` (currently only ``assessment_item``).
    2. Skip blocks whose declared ``bloom_level`` is empty / unknown
       — the gate can't enforce a level that wasn't claimed (the
       ``page_objectives`` gate covers field-presence requirements).
    3. Run the per-level structural check chain on ``content["stem"]``
       (or ``statement``/``text`` fallback) + the answer surface +
       the question-type discriminator.
    4. Emit one GateIssue per failed check (capped at
       :data:`_ISSUE_LIST_CAP`); set ``action="regenerate"`` when any
       block fails any check. ``action=None`` on a clean pass.

    Decision capture: one ``bloom_structural_enforcement_check`` event
    per validate() call when a non-None ``capture`` is wired in.
    """

    name = "bloom_structural_enforcement"
    version = "0.1.0"  # Worker W6 first cut

    def __init__(self, capture: Optional[Any] = None) -> None:
        self._capture = capture
        # Cache the canonical verb sets at construction time so the
        # per-block loop doesn't re-read schema on every call.
        self._verb_sets = get_verbs()

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Empty input is a no-op pass (mirrors sibling validators).
        if not blocks:
            self._emit_decision(audited=0, passed_count=0, failed_count=0,
                                level_distribution={})
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0
        failed_count = 0
        level_distribution: Dict[str, int] = {}

        for block in blocks:
            block_type = _block_attr(block, "block_type")
            if block_type not in _AUDITED_BLOCK_TYPES:
                continue

            declared = _block_attr(block, "bloom_level")
            if not isinstance(declared, str) or not declared:
                continue
            declared = declared.strip().lower()
            if declared not in BLOOM_LEVELS:
                # Unknown level — silently skip. The page_objectives /
                # content_type validators cover enum-membership.
                continue

            stem = _extract_stem(block)
            if not stem:
                continue

            audited += 1
            level_distribution[declared] = level_distribution.get(declared, 0) + 1
            block_id = _block_attr(block, "block_id") or "<unknown>"

            block_issues = self._check_block(
                stem=stem,
                answer_text=_extract_answer_text(block),
                question_type=_extract_question_type(block),
                declared=declared,
                block_id=block_id,
            )

            if block_issues:
                failed_count += 1
                for issue in block_issues:
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(issue)
            else:
                passed_count += 1

        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        # Every issue this validator emits is critical-severity (the
        # forcing-function contract); ``action="regenerate"`` when any
        # block fails so the rewrite tier re-rolls.
        action: Optional[str] = "regenerate" if issues else None
        passed = not issues

        self._emit_decision(
            audited=audited,
            passed_count=passed_count,
            failed_count=failed_count,
            level_distribution=level_distribution,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    def _check_block(
        self,
        *,
        stem: str,
        answer_text: str,
        question_type: Optional[str],
        declared: str,
        block_id: str,
    ) -> List[GateIssue]:
        """Run the per-level structural rule chain. Returns a list of
        GateIssues — one per failed sub-rule. Empty list on a clean
        pass.
        """
        issues: List[GateIssue] = []

        # Rule 1: clause-count floor.
        clauses = _count_clauses(stem)
        clause_floor = _CLAUSE_FLOORS[declared]
        if clauses < clause_floor:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="BLOOM_STRUCTURE_INSUFFICIENT_CLAUSES",
                    message=(
                        f"Block {block_id!r} declares "
                        f"bloom_level={declared!r} but stem has "
                        f"{clauses} clause(s) (floor={clause_floor}). "
                        f"Stem={stem!r}"
                    ),
                    location=block_id,
                    suggestion=(
                        f"Re-roll the block with a multi-clause stem "
                        f"(at least {clause_floor} clauses) — use "
                        f"comma / semicolon / coordinating conjunction "
                        f"to add structure for {declared!r}-level "
                        f"cognitive demand."
                    ),
                )
            )

        # Rule 2: verb-set membership.
        if not _verb_in_level_set(stem, declared, self._verb_sets):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="BLOOM_VERB_MISMATCH",
                    message=(
                        f"Block {block_id!r} declares "
                        f"bloom_level={declared!r} but stem carries no "
                        f"canonical {declared!r}-set verb. "
                        f"Stem={stem!r}"
                    ),
                    location=block_id,
                    suggestion=(
                        f"Re-roll the block using a verb from the "
                        f"canonical {declared!r}-set (see "
                        f"schemas/taxonomies/bloom_verbs.json)."
                    ),
                )
            )

        # Rule 3: per-level marker (analyze / evaluate / create).
        if declared == "analyze" and not _stem_has_marker(stem, _COMPARISON_MARKERS):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="BLOOM_MISSING_COMPARISON_MARKER",
                    message=(
                        f"Block {block_id!r} declares "
                        f"bloom_level='analyze' but stem has no "
                        f"comparison marker (compare/contrast/"
                        f"differentiate/distinguish/versus). "
                        f"Stem={stem!r}"
                    ),
                    location=block_id,
                    suggestion=(
                        "Re-roll with explicit comparison/contrast "
                        "language so the cognitive demand matches the "
                        "declared level."
                    ),
                )
            )
        elif declared == "evaluate" and not _stem_has_marker(stem, _JUDGMENT_MARKERS):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="BLOOM_MISSING_JUDGMENT_MARKER",
                    message=(
                        f"Block {block_id!r} declares "
                        f"bloom_level='evaluate' but stem has no "
                        f"judgment marker (justify/assess/critique/"
                        f"defend). Stem={stem!r}"
                    ),
                    location=block_id,
                    suggestion=(
                        "Re-roll with explicit reasoned-position "
                        "language (justify / assess / critique / defend) "
                        "so the cognitive demand matches the declared "
                        "level."
                    ),
                )
            )
        elif declared == "create" and not _stem_has_marker(stem, _DESIGN_MARKERS):
            issues.append(
                GateIssue(
                    severity="critical",
                    code="BLOOM_MISSING_DESIGN_MARKER",
                    message=(
                        f"Block {block_id!r} declares "
                        f"bloom_level='create' but stem has no "
                        f"design / generation marker (design/construct/"
                        f"develop/formulate). Stem={stem!r}"
                    ),
                    location=block_id,
                    suggestion=(
                        "Re-roll with explicit design / construction "
                        "language so the cognitive demand matches the "
                        "declared level."
                    ),
                )
            )

        # Rule 4: ``create`` MUST be open-ended (no fixed-answer types).
        if declared == "create":
            if question_type and question_type not in _OPEN_ENDED_QUESTION_TYPES:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CREATE_LEVEL_MUST_BE_OPEN_ENDED",
                        message=(
                            f"Block {block_id!r} declares "
                            f"bloom_level='create' but question_type="
                            f"{question_type!r} is fixed-answer. "
                            f"create-level cognitive demand requires "
                            f"essay / short-answer."
                        ),
                        location=block_id,
                        suggestion=(
                            "Switch the assessment to essay / "
                            "short_answer; create-level can't be "
                            "captured by fixed-answer types."
                        ),
                    )
                )
        else:
            # Rule 5: per-level answer-token floor (skipped for create
            # because the open-ended check supersedes it).
            answer_floor = _ANSWER_TOKEN_FLOORS[declared]
            answer_tokens = _count_tokens(answer_text)
            if answer_floor > 0 and answer_tokens < answer_floor:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="BLOOM_ANSWER_TOO_SHORT",
                        message=(
                            f"Block {block_id!r} declares "
                            f"bloom_level={declared!r} but answer "
                            f"surface has {answer_tokens} token(s) "
                            f"(floor={answer_floor}). "
                            f"Answer={answer_text!r}"
                        ),
                        location=block_id,
                        suggestion=(
                            f"Re-roll the block with a richer answer "
                            f"surface (at least {answer_floor} tokens) "
                            f"— short answers structurally cannot "
                            f"capture {declared!r}-level cognitive "
                            f"demand."
                        ),
                    )
                )

        return issues

    def _emit_decision(
        self,
        *,
        audited: int,
        passed_count: int,
        failed_count: int,
        level_distribution: Dict[str, int],
    ) -> None:
        """Emit one ``bloom_structural_enforcement_check`` event per
        validate() call. No-op when no capture is wired.
        """
        if self._capture is None:
            return
        try:
            distribution_summary = ", ".join(
                f"{lvl}={n}" for lvl, n in sorted(level_distribution.items())
            ) or "none"
            self._capture.log_decision(
                decision_type="bloom_structural_enforcement_check",
                decision=(
                    f"audited={audited} passed={passed_count} "
                    f"failed={failed_count}"
                ),
                rationale=(
                    f"Worker W6 — Bloom structural enforcement. "
                    f"Audited {audited} assessment_item block(s); "
                    f"{passed_count} passed all rules, "
                    f"{failed_count} failed at least one structural "
                    f"check. Per-level distribution: "
                    f"{distribution_summary}."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — capture failures must not abort the gate
            logger.warning(
                "BloomStructuralEnforcementValidator decision-capture "
                "emit failed: %s",
                exc,
            )


__all__ = [
    "BloomStructuralEnforcementValidator",
    "_AUDITED_BLOCK_TYPES",
    "_CLAUSE_FLOORS",
    "_ANSWER_TOKEN_FLOORS",
    "_COMPARISON_MARKERS",
    "_JUDGMENT_MARKERS",
    "_DESIGN_MARKERS",
    "_OPEN_ENDED_QUESTION_TYPES",
]
