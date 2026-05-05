"""Worker W3b — DistractorPlausibilityValidator.

Closes the rewrite[2] regression class: pre-W3b, distractors that were
purely syntactic permutations of the correct answer (e.g. an
``assessment_item`` Block emitting ``answer_key="subject, predicate,
object"`` alongside distractors ``["object, predicate, subject",
"predicate, subject, object", "subject, object, predicate"]`` — token-
identical, just reordered) sailed through every existing gate. The
four ``Block*Validator``s in
``Courseforge/router/inter_tier_gates.py`` validate CURIE / content_type
/ objective_ref / source_id shape only; the W7 payload-shape gate
asserts presence + count + correct-answer-index range; nothing
inspected the SEMANTIC distinctness of distractor strings against
their answer key. A learner taking the resulting MCQ would see four
indistinguishable choices.

This validator gates that semantic axis with a CPU-only Jaccard
heuristic — no embedding dependency, runs at gate-evaluation cost
on the laptop. Two failure modes:

- ``DISTRACTOR_NEAR_DUPLICATE_ANSWER`` (severity ``critical``,
  ``action="regenerate"``): a distractor's tokenised Jaccard overlap
  with the answer_key is above ``max_overlap_with_answer`` (default
  0.7). Catches the rewrite[2] permutation class — three distractors
  built from the same word-bag as the answer all clear this floor.
- ``DISTRACTORS_NEAR_DUPLICATE_PAIR`` (severity ``critical``,
  ``action="regenerate"``): two distractors' pairwise Jaccard overlap
  is above ``max_pairwise_overlap`` (default 0.85). Catches the
  parallel "distractors are paraphrases of each other" class so a
  4-option MCQ doesn't collapse to a 2-option choice.

Failure ``action`` is always ``"regenerate"``: a rewrite-tier re-roll
with the offending pair surfaced in the prompt remediation suffix
typically diversifies the distractor pool.

The validator filters on ``block.block_type == "assessment_item"``;
non-assessment Blocks are silently skipped. Both outline-tier (dict
content with ``distractors[]`` + ``answer_key`` keys) and rewrite-tier
(HTML string content with ``<li data-cf-distractor-index="N">``
siblings) shapes are dispatched, mirroring the W7 payload-shape gate's
two-mode dispatch.

Decision capture: emits exactly one
``decision_type="distractor_plausibility_check"`` event per
``validate()`` call, with rationale interpolating block count,
near-duplicate-answer count, near-duplicate-pair count, and the
configured thresholds — replayable from the audit log post-hoc.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

# ``Block`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# import bridge ``assessment_item_payload.py`` uses so this module
# loads regardless of how it's invoked.
_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — import-bridge tested via the test suite
    from blocks import Block  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    Block = None  # type: ignore[assignment,misc]


# Cap per-block issues so a uniformly broken assessment batch doesn't
# drown the gate report (mirrors ``_ISSUE_LIST_CAP`` in
# ``assessment_item_payload.py`` + ``inter_tier_gates.py``).
_ISSUE_LIST_CAP: int = 50

#: Default Jaccard overlap floor between a distractor and the answer
#: key. A distractor whose token-set overlaps the answer's token-set
#: above this floor is essentially the answer with the words shuffled
#: — the rewrite[2] regression case landed at 1.0 (identical token
#: bags), so 0.7 leaves headroom for legitimate near-misses while
#: catching the obvious permutation class.
DEFAULT_MAX_OVERLAP_WITH_ANSWER: float = 0.7

#: Default Jaccard overlap floor between two distractors. Tighter than
#: the answer-overlap floor because two distractors SHOULD be
#: independent misconceptions; ≥ 0.85 means they're paraphrases.
DEFAULT_MAX_PAIRWISE_OVERLAP: float = 0.85

# Lightweight tokeniser: lowercase, split on non-word characters, drop
# empties. No stopword removal — for short MCQ-distractor strings (≤ 20
# tokens typically), preserving function words keeps the Jaccard signal
# stable. Mirrors the regex shape used by
# ``lib/validators/assessment_objective_alignment.py``.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Rewrite-mode HTML scraper. Captures the body text of every
# ``<li data-cf-distractor-index="N">`` sibling. Mirrors the regex in
# ``lib/validators/assessment_item_payload.py``.
_DATA_CF_DISTRACTOR_INDEX_LI_RE = re.compile(
    r'<li[^>]*data-cf-distractor-index=["\'](\d+)["\'][^>]*>(.*?)</li>',
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Strip tags + collapse whitespace."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _tokenise(text: str) -> Set[str]:
    """Lowercase + word-token Jaccard token set.

    Returns the empty set on falsy / whitespace-only input. The
    Jaccard helper below treats empty-set comparisons as overlap=0 so
    no division-by-zero leaks out.
    """
    if not text:
        return set()
    return {tok.lower() for tok in _WORD_RE.findall(text)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Set Jaccard: ``|a ∩ b| / |a ∪ b|``. Returns 0.0 when the union
    is empty (both inputs empty) so callers get a stable below-floor
    signal instead of a NaN."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs['blocks']``.

    Mirrors ``assessment_item_payload._coerce_blocks`` so the same
    shape contract applies.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got "
                f"{type(raw).__name__}."
            ),
        )
    return list(raw), None


def _extract_outline_distractors(
    block: Any,
) -> Tuple[Optional[str], List[str]]:
    """Return ``(answer_key, [distractor_text, ...])`` for an outline
    Block, or ``(None, [])`` when the block can't be audited.

    Outline-tier ``assessment_item`` Blocks carry ``answer_key`` (str)
    and ``distractors[]`` (list of dicts with a ``text`` key) per the
    per-block JSON schema in
    ``Courseforge/generators/_outline_provider.py:431-438``.
    """
    content = getattr(block, "content", None)
    if not isinstance(content, dict):
        return None, []
    answer_key = content.get("answer_key")
    if not isinstance(answer_key, str) or not answer_key.strip():
        answer_key = None
    raw = content.get("distractors")
    if not isinstance(raw, list):
        return answer_key, []
    texts: List[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return answer_key, texts


def _extract_rewrite_distractors(
    block: Any,
) -> Tuple[Optional[str], List[str]]:
    """Return ``(answer_key, [distractor_text, ...])`` for a rewrite
    Block.

    Rewrite-tier HTML doesn't carry the answer_key as a separable
    field — the correct answer is one of the ``<li>`` siblings, and
    the outline-tier ``correct_answer_index`` was the cross-walk. The
    rewrite path therefore dispatches answer-overlap audits with
    ``answer_key=None`` (skipped) and only audits pairwise overlap.
    """
    content = getattr(block, "content", None)
    if not isinstance(content, str):
        return None, []
    matches = list(_DATA_CF_DISTRACTOR_INDEX_LI_RE.finditer(content))
    texts: List[str] = []
    for match in matches:
        body = _strip_html(match.group(2) or "")
        if body:
            texts.append(body)
    return None, texts


def _extract_distractors(
    block: Any,
) -> Tuple[Optional[str], List[str]]:
    """Dispatch to the correct extractor based on content shape."""
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        return _extract_outline_distractors(block)
    if isinstance(content, str):
        return _extract_rewrite_distractors(block)
    return None, []


def _audit_block(
    block: Any,
    *,
    max_overlap_with_answer: float,
    max_pairwise_overlap: float,
    issues: List[GateIssue],
) -> Tuple[int, int]:
    """Audit one assessment_item Block.

    Returns ``(near_duplicate_answer_count, near_duplicate_pair_count)``
    so the caller can roll up totals into the per-validate decision-
    capture rationale.
    """
    answer_key, distractors = _extract_distractors(block)
    if not distractors:
        return 0, 0

    block_id = getattr(block, "block_id", "<unknown>")
    answer_tokens = _tokenise(answer_key) if answer_key else None

    # Tokenise every distractor once so the O(n^2) pairwise pass below
    # is cheap (set ops only).
    distractor_tokens: List[Set[str]] = [_tokenise(d) for d in distractors]

    near_dup_answer = 0
    if answer_tokens is not None and answer_tokens:
        for idx, dtokens in enumerate(distractor_tokens):
            if not dtokens:
                continue
            overlap = _jaccard(dtokens, answer_tokens)
            if overlap > max_overlap_with_answer:
                near_dup_answer += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="DISTRACTOR_NEAR_DUPLICATE_ANSWER",
                        message=(
                            f"assessment_item Block {block_id!r} "
                            f"distractor[{idx}] tokenised Jaccard "
                            f"overlap with answer_key={overlap:.4f} "
                            f"exceeds floor {max_overlap_with_answer:.4f}; "
                            f"distractor reads as a permutation of the "
                            f"correct answer (closes the rewrite[2] "
                            f"regression class)."
                        ),
                        location=block_id,
                        suggestion=(
                            "Re-roll the rewrite tier with a remediation "
                            "suffix surfacing the offending distractor "
                            "and answer_key — emit distractors anchored "
                            "to distinct misconceptions, not word-bag "
                            "shuffles of the correct answer."
                        ),
                    ))

    near_dup_pair = 0
    n = len(distractor_tokens)
    for i in range(n):
        for j in range(i + 1, n):
            a = distractor_tokens[i]
            b = distractor_tokens[j]
            if not a or not b:
                continue
            overlap = _jaccard(a, b)
            if overlap > max_pairwise_overlap:
                near_dup_pair += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="DISTRACTORS_NEAR_DUPLICATE_PAIR",
                        message=(
                            f"assessment_item Block {block_id!r} "
                            f"distractor[{i}] / distractor[{j}] "
                            f"tokenised Jaccard overlap={overlap:.4f} "
                            f"exceeds pairwise floor "
                            f"{max_pairwise_overlap:.4f}; the two "
                            f"distractors are paraphrases of each other, "
                            f"collapsing the MCQ option set."
                        ),
                        location=block_id,
                        suggestion=(
                            "Re-roll the rewrite tier — emit distractors "
                            "that target independent misconceptions so "
                            "the option pool stays diverse."
                        ),
                    ))
    return near_dup_answer, near_dup_pair


def _emit_decision(
    capture: Any,
    *,
    audited_blocks: int,
    near_dup_answer_total: int,
    near_dup_pair_total: int,
    max_overlap_with_answer: float,
    max_pairwise_overlap: float,
    passed: bool,
) -> None:
    """Emit one ``distractor_plausibility_check`` decision per
    ``validate()`` invocation.

    Rationale interpolates audited block count, both near-duplicate
    counts, and both thresholds so the audit log lets an operator
    replay the pass/fail decision without re-running the gate.
    """
    if capture is None:
        return
    decision = "passed" if passed else "failed"
    rationale = (
        f"Distractor plausibility check across "
        f"{audited_blocks} assessment_item block(s): "
        f"near_duplicate_answer={near_dup_answer_total}, "
        f"near_duplicate_pair={near_dup_pair_total}, "
        f"max_overlap_with_answer={max_overlap_with_answer:.4f}, "
        f"max_pairwise_overlap={max_pairwise_overlap:.4f}, "
        f"outcome={decision}."
    )
    try:
        capture.log_decision(
            decision_type="distractor_plausibility_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "distractor_plausibility_check: %s",
            exc,
        )


class DistractorPlausibilityValidator:
    """Worker W3b — distractor distinctness gate.

    CPU-only Jaccard heuristic over ``assessment_item`` Block
    distractors. Two GateIssue codes (both ``critical``,
    ``action="regenerate"``):

    - ``DISTRACTOR_NEAR_DUPLICATE_ANSWER`` — distractor token-set
      overlaps the answer_key above ``max_overlap_with_answer``
      (default 0.7).
    - ``DISTRACTORS_NEAR_DUPLICATE_PAIR`` — two distractors' token-set
      overlap above ``max_pairwise_overlap`` (default 0.85).

    Filters to ``block.block_type == "assessment_item"``; other types
    silently skip. Dispatches outline-tier (dict content) and rewrite-
    tier (HTML string content) shapes via ``_extract_distractors``.

    Per-call kwargs (all in ``inputs``):

    - ``max_overlap_with_answer``: float in [0, 1], default 0.7.
    - ``max_pairwise_overlap``: float in [0, 1], default 0.85.
    - ``decision_capture``: optional ``DecisionCapture`` for the
      per-validate audit event.
    """

    name = "distractor_plausibility"
    version = "1.0.0"

    def __init__(
        self,
        *,
        max_overlap_with_answer: float = DEFAULT_MAX_OVERLAP_WITH_ANSWER,
        max_pairwise_overlap: float = DEFAULT_MAX_PAIRWISE_OVERLAP,
    ) -> None:
        self._max_overlap_with_answer = max_overlap_with_answer
        self._max_pairwise_overlap = max_pairwise_overlap

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        max_overlap_with_answer = float(
            inputs.get(
                "max_overlap_with_answer", self._max_overlap_with_answer
            )
        )
        max_pairwise_overlap = float(
            inputs.get(
                "max_pairwise_overlap", self._max_pairwise_overlap
            )
        )

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            _emit_decision(
                capture,
                audited_blocks=0,
                near_dup_answer_total=0,
                near_dup_pair_total=0,
                max_overlap_with_answer=max_overlap_with_answer,
                max_pairwise_overlap=max_pairwise_overlap,
                passed=False,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        issues: List[GateIssue] = []
        audited = 0
        near_dup_answer_total = 0
        near_dup_pair_total = 0

        for block in blocks:
            if getattr(block, "block_type", None) != "assessment_item":
                continue
            audited += 1
            n_ans, n_pair = _audit_block(
                block,
                max_overlap_with_answer=max_overlap_with_answer,
                max_pairwise_overlap=max_pairwise_overlap,
                issues=issues,
            )
            near_dup_answer_total += n_ans
            near_dup_pair_total += n_pair

        passed = len(issues) == 0
        score = 1.0
        if audited > 0 and not passed:
            # Rough quality signal: fraction of clean axes per block,
            # bounded to [0, 1]. Two axes per block (answer-overlap +
            # pairwise) so the denominator is 2 × audited.
            denom = max(1, 2 * audited)
            score = max(
                0.0,
                round(
                    (denom - near_dup_answer_total - near_dup_pair_total)
                    / denom,
                    4,
                ),
            )

        _emit_decision(
            capture,
            audited_blocks=audited,
            near_dup_answer_total=near_dup_answer_total,
            near_dup_pair_total=near_dup_pair_total,
            max_overlap_with_answer=max_overlap_with_answer,
            max_pairwise_overlap=max_pairwise_overlap,
            passed=passed,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


__all__ = ["DistractorPlausibilityValidator"]
