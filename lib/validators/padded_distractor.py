"""Worker W2.D — PaddedDistractorValidator.

GPT Feedback v2 Wave 2 net-new validator that closes the regression
loop on the padded-distractor bug killed at the source in
``Trainforge/generators/assessment_generator.py``. The pre-W2.D path
emitted a literal ``f"A concept unrelated to {target.term} in this
context"`` template string into the choices[] when the
misconception/distractor synthesis exhausted plausible options. The
pad text trivially passed structural validators (it's a string, it's
not a duplicate of the correct answer) and got baked into training
pairs. W2.D Part A killed that fallback at the source by emitting a
``SkippedItem(reason="padded_distractor_fallback_eliminated")`` early
return; W2.D Part B (this module) is the regression net so the bug
never silently returns.

Five canonical padding patterns are detected via regex on every
emitted distractor (across instruction pairs, preference pairs, and
assessment items in IMSCC). Any match fails closed with an
``action="regenerate"`` directive at outline tier (where the gate is
expected to drive the rewrite tier to re-roll), and ``action="block"``
at rewrite tier / packaging tier (where rewrite-tier emit is the
canonical published HTML and a padded distractor in published HTML is
a structural unfixable). The ``phase`` input toggles the action.

GateIssue codes:

- ``PADDED_DISTRACTOR_PATTERN_1`` (critical) — matches
  ``r"^\\s*(?:[Aa])\\s+(?:concept|topic|item|element|idea|notion)\\s+unrelated\\s+to\\s+"``;
  the canonical pre-W2.D ``f"A concept unrelated to {term}..."`` pad.
- ``PADDED_DISTRACTOR_PATTERN_2`` (critical) — matches
  ``r"^\\s*[Nn]ot\\s+(?:related|associated|connected)\\s+to\\s+"``;
  paraphrased "not related to" pad variants.
- ``PADDED_DISTRACTOR_PATTERN_3`` (critical) — matches
  ``r"^\\s*(?:Distractor|Wrong\\s+answer|Incorrect)\\s*\\d*\\s*$"``;
  literal placeholder strings ("Distractor", "Wrong answer 2",
  "Incorrect").
- ``PADDED_DISTRACTOR_PATTERN_4`` (critical) — matches
  ``r"^\\s*[Nn]one\\s+of\\s+the\\s+above\\s*$"``; lazy
  "None of the above" non-distractor pad.
- ``PADDED_DISTRACTOR_PATTERN_5`` (critical) — matches
  ``r"^\\s*(?:[Tt]his\\s+is\\s+(?:a\\s+)?(?:wrong|incorrect|distractor)\\s*answer)"``;
  meta-description-of-distractor pad.

- ``MISSING_BLOCKS_INPUT`` / ``INVALID_BLOCKS_INPUT`` (critical,
  ``action="regenerate"``) — symmetric to the sibling validators'
  input-shape guards.

Filters to ``block.block_type == "assessment_item"``; non-assessment
blocks (``concept`` / ``example`` / chrome / recap / etc.) are silently
skipped. Pattern matching is performed on the wrong-options
(``is_correct=False``) only — a pattern match in the stem or the
correct answer is NOT a distractor padding bug; the validator's
domain is wrong-options exclusively.

Decision capture: emits exactly one
``decision_type="padded_distractor_check"`` event per ``validate()``
call (per-call, not per-block — the regression is rare enough that
per-call rationale carrying counts is more useful than per-block
rationale). Decision-type enum entry was added in Wave 1 (commit
``d6e19a1`` per the plan handoff).

References:
    - ``Trainforge/generators/assessment_generator.py`` — Part A site
      where the padded-fallback was killed (``_generate_multiple_choice``).
    - ``lib/validators/distractor_structural.py`` — sibling W2.C
      validator; this module mirrors its Block import bridge,
      ``_coerce_blocks`` helper shape, ``_extract_choices`` helper,
      and ``_emit_decision`` capture pattern.
    - ``lib/validators/assessment.py::ASSESSMENT_PLACEHOLDER_PATTERNS``
      — pre-existing 13-regex placeholder catalog applied to the
      ``assessment_quality`` gate; this module is narrower (5 patterns,
      distractor-side only) but uses the same fail-closed posture.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

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


#: Five canonical padding patterns. Each entry is
#: ``(code, compiled_regex, description)``. Order is load-bearing —
#: the first matching pattern wins (pattern-1 wins over pattern-2 on
#: a string that hits both, so the GateIssue code is deterministic
#: across runs). The corresponding GateIssue code is
#: ``f"PADDED_DISTRACTOR_PATTERN_{N}"`` per the plan.
_PADDING_PATTERNS: Tuple[Tuple[str, Pattern[str], str], ...] = (
    (
        "PADDED_DISTRACTOR_PATTERN_1",
        re.compile(
            r"^\s*(?:[Aa])\s+(?:concept|topic|item|element|idea|notion)"
            r"\s+unrelated\s+to\s+"
        ),
        (
            "Canonical pre-W2.D pad: 'A concept unrelated to ... in this "
            "context' (and lexical variants)."
        ),
    ),
    (
        "PADDED_DISTRACTOR_PATTERN_2",
        re.compile(
            r"^\s*[Nn]ot\s+(?:related|associated|connected)\s+to\s+"
        ),
        (
            "Paraphrased pad: 'Not related/associated/connected to ...'."
        ),
    ),
    (
        "PADDED_DISTRACTOR_PATTERN_3",
        re.compile(
            r"^\s*(?:Distractor|Wrong\s+answer|Incorrect)\s*\d*\s*$"
        ),
        (
            "Literal placeholder string: 'Distractor', 'Wrong answer 2', "
            "'Incorrect'."
        ),
    ),
    (
        "PADDED_DISTRACTOR_PATTERN_4",
        re.compile(r"^\s*[Nn]one\s+of\s+the\s+above\s*$"),
        (
            "Lazy non-distractor pad: 'None of the above'."
        ),
    ),
    (
        "PADDED_DISTRACTOR_PATTERN_5",
        re.compile(
            r"^\s*(?:[Tt]his\s+is\s+(?:a\s+)?"
            r"(?:wrong|incorrect|distractor)\s*answer)"
        ),
        (
            "Meta-description-of-distractor pad: 'This is a wrong/incorrect/"
            "distractor answer'."
        ),
    ),
)


#: Cap per-validate() issue list so a uniformly broken batch doesn't
#: drown the gate report. Mirrors ``_ISSUE_LIST_CAP`` in the sibling
#: validators.
_ISSUE_LIST_CAP: int = 50

#: Phase values that route to ``action="block"`` instead of the
#: outline-tier default ``action="regenerate"``. Rewrite-tier emit is
#: the canonical published HTML, and a padded distractor in published
#: HTML is a structural unfixable — re-rolling at that seam is too late
#: in the pipeline. Packaging / trainforge_assessment surfaces are
#: similarly post-publication so they get the same posture.
_BLOCK_TIER_PHASES: Tuple[str, ...] = (
    "post_rewrite_validation",
    "packaging",
    "trainforge_assessment",
)


def _strip_html(value: str) -> str:
    """Strip a minimal HTML wrapper (``<p>...</p>``) for pattern matching.

    The Trainforge generator wraps emitted choice text in ``<p>...</p>``;
    matching against the wrapped form would miss the canonical pad
    string. Use a simple non-greedy strip: any leading ``<...>`` and
    any trailing ``</...>``. Heavier HTML (nested spans / strong) is
    deliberately not handled — the padding patterns are short enough
    that nested HTML in the pad is not a realistic generator output.
    """
    if not isinstance(value, str):
        return ""
    # Strip leading single tags + trailing closing tags iteratively.
    stripped = value.strip()
    # Drop a single leading ``<...>`` (non-greedy) and trailing
    # ``</...>``. Iteration handles ``<p><span>...`` flattening.
    for _ in range(4):  # cap iterations to avoid pathological inputs
        if stripped.startswith("<") and ">" in stripped:
            close = stripped.find(">")
            stripped = stripped[close + 1:].lstrip()
            continue
        break
    for _ in range(4):
        if stripped.endswith(">") and "</" in stripped[-12:]:
            open_close = stripped.rfind("</")
            stripped = stripped[:open_close].rstrip()
            continue
        break
    return stripped


def _match_padding_pattern(text: str) -> Optional[Tuple[str, Pattern[str]]]:
    """Return the first matching ``(code, pattern)`` for ``text``.

    Returns ``None`` when no pattern matches. Empty / non-string input
    silently returns ``None`` — non-distractor input is the validator
    skip path, not a fail.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = _strip_html(text)
    if not stripped:
        return None
    for code, pattern, _desc in _PADDING_PATTERNS:
        if pattern.search(stripped):
            return code, pattern
    return None


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


def _extract_distractor_texts(block: Any) -> List[str]:
    """Return the wrong-option (``is_correct=False``) texts.

    Walks the same shape resolution as
    ``DistractorStructuralValidator._extract_choices``:

    1. Trainforge canonical ``content["choices"]`` — list of
       ``{text: str, is_correct: bool}`` dicts. Returns the texts
       where ``is_correct`` is falsy.
    2. Legacy split shape ``content["distractors"]`` (list of
       ``{text}`` dicts or bare strings). All entries are wrong-options
       by construction (the answer_key is the correct side).
    3. Neither shape resolves → empty list.
    """
    content = getattr(block, "content", None)
    if not isinstance(content, dict):
        return []

    out: List[str] = []
    raw = content.get("choices")
    if isinstance(raw, list) and raw:
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if not isinstance(text, str):
                continue
            if bool(entry.get("is_correct", False)):
                continue
            out.append(text)
        return out

    distractors = content.get("distractors")
    if isinstance(distractors, list):
        for entry in distractors:
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str):
                    out.append(text)
            elif isinstance(entry, str):
                out.append(entry)
    return out


def _emit_decision(
    capture: Any,
    *,
    audited_blocks: int,
    matched_count: int,
    matched_codes: List[str],
    action: Optional[str],
    threshold_phase: str,
) -> None:
    """Emit one ``padded_distractor_check`` decision per ``validate()`` call.

    Per-call (not per-block) emit policy: the regression is rare enough
    that per-call rationale carrying counts + matched codes is more
    useful than per-block rationale; mirrors the W2.D plan call-out.

    Rationale interpolates ``audited_blocks``, ``matched_count``,
    ``matched_codes``, the resolved ``action``, and the
    ``threshold_phase`` (outline / rewrite / packaging) so the audit
    log replays post-hoc.
    """
    if capture is None:
        return
    if matched_codes:
        codes_summary = ",".join(sorted(set(matched_codes)))
    else:
        codes_summary = "none"
    rationale = (
        f"padded_distractor on {audited_blocks} assessment_item blocks: "
        f"matched_count={matched_count}, "
        f"matched_codes=[{codes_summary}], "
        f"phase={threshold_phase!r}, action={action!r}, "
        f"posture={'fail_closed' if matched_count > 0 else 'pass'}."
    )
    decision = (
        f"padded_distractor_pass({audited_blocks}_blocks)"
        if matched_count == 0
        else (
            f"padded_distractor_fail({matched_count}_distractor_matches/"
            f"{audited_blocks}_blocks)"
        )
    )
    try:
        capture.log_decision(
            decision_type="padded_distractor_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — degrade quietly on emit.
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "padded_distractor_check: %s",
            exc,
        )


class PaddedDistractorValidator:
    """Worker W2.D — regression-net regex validator for padded distractors.

    Matches the 5 canonical padding patterns against every distractor
    (``is_correct=False`` choice text) on every ``assessment_item``
    Block. Any match fails closed.

    Per-call kwargs (all in ``inputs``):

    - ``blocks`` (required): list of Courseforge Block instances.
    - ``phase`` (optional, default ``"inter_tier_validation"``):
      determines whether the action is ``"regenerate"`` (outline /
      inter-tier seam) or ``"block"`` (post-rewrite / packaging /
      trainforge_assessment seam).
    - ``decision_capture`` (optional): a DecisionCapture for the
      per-call audit event.

    Filters to ``block.block_type == "assessment_item"``; non-assessment
    blocks are silently skipped.
    """

    name = "padded_distractor"
    version = "1.0.0"

    PADDING_PATTERNS = _PADDING_PATTERNS

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        phase = str(inputs.get("phase", "inter_tier_validation"))

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

        # Resolve per-tier action up front so the GateResult posture
        # matches the seam the validator is wired on.
        if phase in _BLOCK_TIER_PHASES:
            action_on_fail: str = "block"
        else:
            action_on_fail = "regenerate"

        issues: List[GateIssue] = []
        audited = 0
        matched_count = 0
        matched_codes: List[str] = []

        for block in blocks:
            if getattr(block, "block_type", None) != "assessment_item":
                continue
            audited += 1
            block_id = getattr(block, "block_id", "<unknown>")
            distractors = _extract_distractor_texts(block)
            for d_idx, d_text in enumerate(distractors):
                hit = _match_padding_pattern(d_text)
                if hit is None:
                    continue
                code, pattern = hit
                matched_count += 1
                matched_codes.append(code)
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code=code,
                        message=(
                            f"assessment_item Block {block_id!r} "
                            f"distractor[{d_idx}] matches padded-distractor "
                            f"pattern {code!r} (regex={pattern.pattern!r}); "
                            f"the W2.D regression net flagged a literal "
                            f"padding template. Distractor text: "
                            f"{d_text[:120]!r}"
                        ),
                        location=block_id,
                        suggestion=(
                            "Re-author the distractor against a "
                            "misconception-targeted prompt; padded-"
                            "fallback templates ('A concept unrelated "
                            "to ...', 'None of the above', 'Distractor 2', "
                            "etc.) silently corrupt the training corpus "
                            "when used as wrong-options."
                        ),
                    ))

        passed = matched_count == 0
        if audited == 0:
            score: Optional[float] = 1.0
        elif passed:
            score = 1.0
        else:
            # Score: fraction of audited blocks WITHOUT a padding hit.
            # We report block-level rather than distractor-level because
            # one bad distractor on a block is enough to fail that block.
            blocks_with_hits = len({
                # location is the block_id stamped on each issue
                getattr(i, "location", None)
                for i in issues
                if i.severity == "critical"
                and getattr(i, "location", None) is not None
            })
            score = round(
                max(0.0, audited - blocks_with_hits) / audited, 4
            )

        action: Optional[str] = None if passed else action_on_fail

        _emit_decision(
            capture,
            audited_blocks=audited,
            matched_count=matched_count,
            matched_codes=matched_codes,
            action=action,
            threshold_phase=phase,
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


__all__ = [
    "PaddedDistractorValidator",
    "_PADDING_PATTERNS",
]
