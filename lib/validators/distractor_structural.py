"""Worker W2.C — DistractorStructuralValidator.

GPT Feedback v2 Wave 2 net-new validator covering two structural
sanity sub-checks that the existing ``DistractorPlausibilityValidator``
(W3b — token-distinctness Jaccard) and
``DistractorMisconceptionAlignmentValidator`` (W3a — embedding-gated
distractor↔misconception alignment) do not reach. Both gaps are
cited in ``plans/gpt feedback 2.txt`` lines 220-228 and the canonical
sub-check spec in
``plans/gpt-feedback-2-p0b-distractor-bloom-padding-2026-05.md``
§ "Fix 4".

Two sub-checks per ``assessment_item`` Block:

1. **Exactly one correct answer** on the Trainforge ``choices[]`` shape.
   The assessment-item content carries ``choices: List[{text: str,
   is_correct: bool}]``; the validator counts how many entries have
   ``is_correct=true``. Anything other than exactly one fires a
   structural gate. Mirrors the existing
   ``correct_answer_index`` range check in
   ``lib/validators/assessment_item_payload.py`` but on the Trainforge
   ``choices[]`` shape (which ``BlockAssessmentItemPayloadValidator``
   doesn't reach — that validator gates the Block-shape's
   ``distractors[]`` + ``correct_answer_index`` cross-walk).

2. **Distractor not entailed by source** — symmetric to
   ``AssessmentRetrievalGroundingValidator``'s answer-side gate that
   asserts the CORRECT answer IS grounded in cited source chunks. This
   validator asserts that distractors are NOT grounded: a distractor
   that's literally stated in the source isn't a misconception, it's a
   true alternative — a structural bug. CPU-only Jaccard overlap
   (deliberately no embedding deps, per the W2.C plan); the threshold
   default ``max_source_overlap=0.70`` (any source chunk) is calibrated
   higher than the answer-grounding 0.30 floor because legitimately-
   mentioned terms sharing one or two tokens shouldn't trip the gate;
   only structural overlap above ~0.70 indicates the distractor IS the
   source claim.

GateIssue codes (six total):

- ``DISTRACTOR_STRUCTURAL_MULTI_CORRECT`` (critical, action=regenerate)
  — ``choices[]`` carries >1 entry with ``is_correct=true``.
- ``DISTRACTOR_STRUCTURAL_NO_CORRECT`` (critical, action=regenerate) —
  ``choices[]`` carries zero ``is_correct=true`` entries.
- ``DISTRACTOR_STRUCTURAL_DISTRACTOR_ENTAILED`` (critical,
  action=regenerate) — at least one distractor's Jaccard overlap
  against any cited source chunk exceeds ``max_source_overlap``
  (default 0.70).
- ``DISTRACTOR_STRUCTURAL_NO_CHOICES`` (warning) — block content
  declares neither ``choices[]`` nor a legacy ``distractors[]`` +
  ``answer_key`` pair; validator skips this block (the validator's job
  is choices, not whether they exist; W7's
  ``BlockAssessmentItemPayloadValidator`` owns the existence axis).
- ``DISTRACTOR_STRUCTURAL_NO_SOURCES`` (warning) —
  ``inputs["source_chunks"]`` / ``inputs["chunks_lookup"]`` is absent
  for the validate() call; the entailment sub-check skips. The
  exactly-one-correct sub-check still runs.
- ``MISSING_BLOCKS_INPUT`` / ``INVALID_BLOCKS_INPUT`` (critical,
  action=regenerate) — symmetric to the sibling validators' input-shape
  guards.

Filters to ``block.block_type == "assessment_item"``; non-assessment
blocks (``concept`` / ``example`` / chrome / recap / etc.) are silently
skipped.

Decision capture: emits exactly one
``decision_type="distractor_structural_check"`` event per audited
``assessment_item`` block per ``validate()`` call, mirroring the
per-block-aggregated rationale style established by
``DistractorMisconceptionAlignmentValidator``. Rationale interpolates
the block_id, choice count, correct count, max source-overlap signal,
threshold, and the resolved per-block decision so audit captures
replay post-hoc. Decision-type enum entry was added in Wave 1
(commit ``d6e19a1`` per the plan handoff).

References:
    - ``lib/validators/distractor_plausibility.py`` — sibling W3b
      validator, ``_emit_decision()`` pattern this validator mirrors.
    - ``lib/validators/distractor_misconception_alignment.py`` — sibling
      W3a validator; per-block decision-emit pattern.
    - ``lib/validators/assessment_retrieval_grounding.py`` — answer-side
      grounding gate; this module reuses its ``_content_tokens`` +
      ``_jaccard`` helpers via private import (single source of truth
      for the Jaccard overlap signal).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Reuse the Jaccard + token helpers from the answer-side grounding
# validator so the structural-distractor gate and the answer-grounding
# gate share one tokenisation contract — the W2.C plan called out
# "DON'T duplicate" the helper. Both helpers are module-private by
# convention, but the alternative (forking the regex / stop-word set
# into a third copy) drifts the two grounding axes apart over time.
from lib.validators.assessment_retrieval_grounding import (
    _content_tokens,
    _jaccard,
)

logger = logging.getLogger(__name__)

# ``Block`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# import bridge the sibling validators use so ``Block`` resolves
# regardless of how this module is loaded.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — import-bridge tested via the test suite
    from blocks import Block  # type: ignore[import-not-found]  # noqa: E402
except Exception:  # noqa: BLE001
    Block = None  # type: ignore[assignment,misc]


#: Default Jaccard floor at which a distractor is treated as entailed
#: by a cited source chunk. Calibrated higher than the 0.30 answer-side
#: grounding floor because distractors LEGITIMATELY mention vocabulary
#: from the source (the scenario being assessed); only when the
#: distractor is essentially a quoted-from-source claim does the gate
#: fire. Empirical confirmation deferred to the wave's reference
#: corpus pilot per the W2.C plan §Risks. Strict ``>=`` comparison so
#: a distractor that exactly hits the floor is treated as entailed.
DEFAULT_MAX_SOURCE_OVERLAP: float = 0.70

#: Cap per-validate() issue list so a uniformly broken batch doesn't
#: drown the gate report. Mirrors ``_ISSUE_LIST_CAP`` in the sibling
#: validators.
_ISSUE_LIST_CAP: int = 50


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs['blocks']``.

    Mirrors the sibling-validator helper. Returns ``(blocks,
    error_issue)``; non-None ``error_issue`` is wrapped into a
    ``passed=False, action="regenerate"`` GateResult by the caller.
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
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _resolve_source_chunks(inputs: Dict[str, Any]) -> Dict[str, str]:
    """Pull the source-chunks mapping from ``inputs``.

    Accepts either ``source_chunks`` (canonical for this validator —
    matches the W2.C plan's wording) or ``chunks_lookup`` (sibling-
    validator alias used by ``AssessmentRetrievalGroundingValidator``).
    Both map sourceId → chunk text. Non-string values are dropped
    silently. Returns ``{}`` when neither input is present (caller
    short-circuits the entailment sub-check via the
    ``DISTRACTOR_STRUCTURAL_NO_SOURCES`` warning path).
    """
    raw = inputs.get("source_chunks")
    if raw is None:
        raw = inputs.get("chunks_lookup")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, str)}


def _block_referenced_chunk_ids(block: Any) -> List[str]:
    """Return the union of source IDs declared by the block.

    Mirrors ``assessment_retrieval_grounding._block_referenced_chunk_ids``;
    the two grounding gates resolve cited chunks identically. Order
    preserved, dedup against a seen-set.
    """
    seen: Set[str] = set()
    ids: List[str] = []
    for ref in getattr(block, "source_references", ()) or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    for sid in getattr(block, "source_ids", ()) or ():
        if isinstance(sid, str) and sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def _extract_choices(
    block: Any,
) -> Tuple[Optional[List[Tuple[str, bool]]], str]:
    """Resolve a ``[(text, is_correct)]`` list from the block content.

    Resolution order:

    1. Trainforge canonical ``content["choices"]`` — list of dicts each
       with a ``text`` (str) and ``is_correct`` (bool) key. Returned as
       ``(choices_list, "choices")``.
    2. Legacy split shape ``content["distractors"]`` (list of
       ``{text}`` dicts) + ``content["answer_key"]`` (str). Reconstructs
       the choices list by emitting each distractor as
       ``(text, False)`` plus a synthesised ``(answer_key, True)`` row.
       Returned as ``(choices_list, "legacy_split")``.
    3. Neither shape resolves → ``(None, "absent")``. Caller emits the
       ``DISTRACTOR_STRUCTURAL_NO_CHOICES`` warning and skips.
    """
    content = getattr(block, "content", None)
    if not isinstance(content, dict):
        return None, "absent"

    raw = content.get("choices")
    if isinstance(raw, list) and raw:
        out: List[Tuple[str, bool]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                # Reject any non-dict choice entry by treating the
                # whole choices[] as malformed; the W7 payload-shape
                # gate owns the granular per-entry shape error.
                return None, "absent"
            text = entry.get("text")
            if not isinstance(text, str):
                return None, "absent"
            is_correct = bool(entry.get("is_correct", False))
            out.append((text, is_correct))
        return out, "choices"

    distractors = content.get("distractors")
    answer_key = content.get("answer_key")
    if (
        isinstance(distractors, list)
        and isinstance(answer_key, str)
        and answer_key.strip()
    ):
        out_legacy: List[Tuple[str, bool]] = []
        for entry in distractors:
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str):
                    out_legacy.append((text, False))
            elif isinstance(entry, str):
                out_legacy.append((entry, False))
        out_legacy.append((answer_key, True))
        return out_legacy, "legacy_split"

    return None, "absent"


def _max_source_overlap(
    distractor_text: str,
    source_chunk_texts: List[str],
) -> Tuple[float, int]:
    """Return ``(max_jaccard_overlap, matching_chunk_index)``.

    Jaccard overlap of content tokens between the distractor text and
    each source chunk; returns the maximum across chunks. Empty source
    list → ``(0.0, -1)`` (caller checks the no-sources path before
    reaching here).
    """
    if not source_chunk_texts:
        return 0.0, -1
    distractor_tokens = _content_tokens(distractor_text)
    if not distractor_tokens:
        return 0.0, -1
    best = 0.0
    best_idx = -1
    for idx, chunk_text in enumerate(source_chunk_texts):
        chunk_tokens = _content_tokens(chunk_text)
        if not chunk_tokens:
            continue
        overlap = _jaccard(distractor_tokens, chunk_tokens)
        if overlap > best:
            best = overlap
            best_idx = idx
    return best, best_idx


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    decision: str,
    choice_count: int,
    correct_count: int,
    max_source_overlap_observed: float,
    threshold: float,
    sub_check_run_entailment: bool,
) -> None:
    """Emit one ``distractor_structural_check`` decision per audited
    assessment_item block.

    Per the CLAUDE.md call-site instrumentation contract, rationale
    must interpolate dynamic per-block signals (not boilerplate) so the
    audit log replays post-hoc. ≥20 chars enforced by the schema-side
    decision_validation gate; the multi-field interpolation here lands
    well above that floor.
    """
    if capture is None:
        return
    rationale = (
        f"distractor_structural on Block {block_id!r}: "
        f"choice_count={choice_count}, correct_count={correct_count}, "
        f"max_source_overlap={max_source_overlap_observed:.4f}, "
        f"threshold={threshold:.4f}, "
        f"entailment_check_run={sub_check_run_entailment}, "
        f"outcome={decision}."
    )
    try:
        capture.log_decision(
            decision_type="distractor_structural_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — degrade quietly on emit.
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "distractor_structural_check: %s",
            exc,
        )


class DistractorStructuralValidator:
    """Worker W2.C — structural distractor sanity gate.

    Two sub-checks per ``assessment_item`` Block:

    1. **Exactly-one-correct** on the Trainforge ``choices[]`` shape
       (with a fallback that reconstructs the implicit ``choices[]``
       from the legacy ``distractors[] + answer_key`` shape some emit
       paths use).
    2. **Distractor-not-entailed-by-source** — Jaccard overlap floor
       (default 0.70) between each distractor and the union of the
       block's cited source chunks; above the floor the distractor is
       structurally a quoted-source claim, not a misconception, and
       fires ``action="regenerate"``.

    Per-call kwargs (all in ``inputs``):

    - ``max_source_overlap``: float in [0, 1], default 0.70.
    - ``source_chunks`` / ``chunks_lookup``: ``Dict[sourceId, str]``;
      missing → entailment sub-check skips with a warning code.
    - ``decision_capture``: optional ``DecisionCapture`` for the
      per-block audit event.

    Filters to ``block.block_type == "assessment_item"``; non-assessment
    blocks are silently skipped.
    """

    name = "distractor_structural"
    version = "1.0.0"

    DEFAULT_MAX_SOURCE_OVERLAP = DEFAULT_MAX_SOURCE_OVERLAP

    def __init__(
        self,
        *,
        max_source_overlap: float = DEFAULT_MAX_SOURCE_OVERLAP,
    ) -> None:
        self._max_source_overlap = float(max_source_overlap)

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        threshold = float(
            inputs.get("max_source_overlap", self._max_source_overlap)
        )

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        source_chunks = _resolve_source_chunks(inputs)
        sources_present = bool(source_chunks)

        issues: List[GateIssue] = []
        audited = 0
        any_critical = False

        for block in blocks:
            if getattr(block, "block_type", None) != "assessment_item":
                continue
            audited += 1
            block_id = getattr(block, "block_id", "<unknown>")
            choices, shape = _extract_choices(block)

            # Sub-check 1 prerequisites: choices must resolve.
            if choices is None:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="warning",
                        code="DISTRACTOR_STRUCTURAL_NO_CHOICES",
                        message=(
                            f"assessment_item Block {block_id!r} content "
                            f"declares neither a Trainforge ``choices[]`` "
                            f"shape nor a legacy ``distractors[]`` + "
                            f"``answer_key`` pair; the structural-distractor "
                            f"gate cannot audit this block. (W7's "
                            f"BlockAssessmentItemPayloadValidator owns the "
                            f"existence axis.)"
                        ),
                        location=block_id,
                    ))
                _emit_decision(
                    capture,
                    block_id=block_id,
                    decision="no_choices",
                    choice_count=0,
                    correct_count=0,
                    max_source_overlap_observed=0.0,
                    threshold=threshold,
                    sub_check_run_entailment=False,
                )
                continue

            choice_count = len(choices)
            correct_count = sum(1 for _, is_correct in choices if is_correct)

            # Sub-check 1: exactly one correct.
            block_decision = "passed"
            if correct_count > 1:
                any_critical = True
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="DISTRACTOR_STRUCTURAL_MULTI_CORRECT",
                        message=(
                            f"assessment_item Block {block_id!r} ``choices[]`` "
                            f"declares {correct_count} entries with "
                            f"is_correct=true (expected exactly 1); the MCQ "
                            f"is structurally ambiguous."
                        ),
                        location=block_id,
                        suggestion=(
                            "Re-roll the assessment item so exactly one "
                            "choice is marked is_correct=true; the others "
                            "must be misconception-targeted distractors."
                        ),
                    ))
                block_decision = "multi_correct"
            elif correct_count == 0:
                any_critical = True
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="DISTRACTOR_STRUCTURAL_NO_CORRECT",
                        message=(
                            f"assessment_item Block {block_id!r} ``choices[]`` "
                            f"declares zero entries with is_correct=true "
                            f"(expected exactly 1); the question is "
                            f"unanswerable as authored."
                        ),
                        location=block_id,
                        suggestion=(
                            "Re-roll the assessment item so exactly one "
                            "choice carries is_correct=true."
                        ),
                    ))
                block_decision = "no_correct"

            # Sub-check 2 prerequisites: a chunks_lookup must be present
            # AND the block must declare at least one source reference
            # OR the all-chunks union must be defined. Per the W2.C plan,
            # absent source_chunks input → skip entailment sub-check
            # entirely (warning-coded, not a failure).
            entailment_run = False
            max_overlap_observed = 0.0
            if sources_present:
                chunk_ids = _block_referenced_chunk_ids(block)
                # Resolve cited chunk texts; fall back to the union of
                # all chunks when the block declares no source IDs (a
                # distractor entailed by ANY chunk in the corpus is a
                # signal worth surfacing, even without per-block
                # attribution).
                if chunk_ids:
                    cited_texts = [
                        source_chunks[sid]
                        for sid in chunk_ids
                        if sid in source_chunks
                    ]
                else:
                    cited_texts = list(source_chunks.values())

                if cited_texts:
                    entailment_run = True
                    # Audit only the wrong-options (is_correct=False).
                    # The correct-answer side is gated by the symmetric
                    # ``AssessmentRetrievalGroundingValidator`` which
                    # asserts the answer IS grounded; here we want to
                    # ensure DISTRACTORS are NOT grounded.
                    for d_idx, (d_text, is_correct) in enumerate(choices):
                        if is_correct:
                            continue
                        overlap, chunk_idx = _max_source_overlap(
                            d_text, cited_texts,
                        )
                        if overlap > max_overlap_observed:
                            max_overlap_observed = overlap
                        if overlap >= threshold:
                            any_critical = True
                            if len(issues) < _ISSUE_LIST_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code=(
                                        "DISTRACTOR_STRUCTURAL_DISTRACTOR_"
                                        "ENTAILED"
                                    ),
                                    message=(
                                        f"assessment_item Block {block_id!r} "
                                        f"distractor[{d_idx}] tokenised "
                                        f"Jaccard overlap "
                                        f"{overlap:.4f} >= threshold "
                                        f"{threshold:.4f} against cited "
                                        f"source chunk #{chunk_idx}; the "
                                        f"distractor is structurally "
                                        f"entailed by the source — it "
                                        f"reads as a true alternative, "
                                        f"not a misconception."
                                    ),
                                    location=block_id,
                                    suggestion=(
                                        "Re-roll the distractor against a "
                                        "misconception-targeted prompt; a "
                                        "distractor that's literally stated "
                                        "in the cited source can't function "
                                        "as a wrong option."
                                    ),
                                ))
                            if block_decision == "passed":
                                block_decision = "distractor_entailed"
            else:
                # Source-chunks input absent — surface once per validate()
                # call (not per block) and keep running the
                # exactly-one-correct sub-check. Emit once outside the
                # per-block loop... handled below the loop.
                pass

            _emit_decision(
                capture,
                block_id=block_id,
                decision=block_decision,
                choice_count=choice_count,
                correct_count=correct_count,
                max_source_overlap_observed=max_overlap_observed,
                threshold=threshold,
                sub_check_run_entailment=entailment_run,
            )

        # Single-emit warning for the no-sources path. Surfaced once per
        # validate() call (not per block) so a 247-block batch missing
        # source_chunks doesn't drown the gate report. Emitted only when
        # there was at least one assessment_item block to audit; an
        # empty batch returns vacuously-passed with no issues.
        if not sources_present and audited > 0:
            issues.insert(0, GateIssue(
                severity="warning",
                code="DISTRACTOR_STRUCTURAL_NO_SOURCES",
                message=(
                    "inputs['source_chunks'] / inputs['chunks_lookup'] is "
                    "absent or empty; the structural-distractor gate "
                    "ran the exactly-one-correct sub-check but skipped "
                    "the distractor-not-entailed-by-source sub-check."
                ),
            ))

        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        # Score: 1.0 when all audited blocks pass; rough quality signal
        # when not, computed as (audited - critical_count) / audited.
        if audited == 0:
            score: Optional[float] = 1.0
        elif passed:
            score = 1.0
        else:
            score = round(max(0.0, audited - len(critical)) / audited, 4)

        action: Optional[str] = None
        if not passed:
            action = "regenerate"
        # any_critical is a defensive belt-and-braces against a future
        # edit that adds a new critical path without bumping issues; the
        # GateResult check above against ``critical`` is the canonical
        # source-of-truth signal.
        _ = any_critical

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )


__all__ = [
    "DistractorStructuralValidator",
    "DEFAULT_MAX_SOURCE_OVERLAP",
]
