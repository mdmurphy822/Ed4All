"""AssessmentNumericEquivalenceValidator — numeric-value distractor gate.

Closes the ``4/6 ≡ 2/3`` blind spot that investigation a6d6291b
surfaced: even the well-formed distractor validators only do token-
Jaccard distinctness. ``DistractorPlausibilityValidator`` tokenises
``"4/6"`` to ``{"4", "6"}`` and ``"2/3"`` to ``{"2", "3"}`` — Jaccard
overlap 0.0, so the pair reads as "distinct" and ships. But ``4/6``
EQUALS ``2/3`` in VALUE, so an "equivalent form" distractor is a
SECOND CORRECT ANSWER, not a distractor. No existing gate inspected
the mathematical equivalence axis.

This validator gates that axis with a domain-aware numeric parse — no
embedding dependency, runs at gate-evaluation cost on the laptop. It
parses each option string to a ``fractions.Fraction`` (handling
integers, decimals, and ``a/b`` fractions via Python stdlib so
``Fraction(4, 6) == Fraction(2, 3)`` resolves true) and compares every
distractor's parsed value against the correct answer's parsed value.

GateIssue code:

- ``DISTRACTOR_EQUALS_ANSWER_VALUE`` (severity ``warning``,
  ``action="regenerate"``): a distractor parses to the SAME numeric
  value as the correct answer. Names the offending distractor + its
  parsed value so the rewrite remediation suffix can re-roll with a
  value-distinct option pool.

**Graceful scope-limit:** if the correct answer or the options DON'T
parse as numbers / fractions (a non-math MCQ — text options like
``"A class hierarchy declaration."``), the block is SKIPPED (pass) —
the validator never false-positives on a non-numeric stem. Empty /
no-assessment input → pass.

Two content shapes are dispatched, mirroring the W3b / W7 distractor
gates:

- **Outline mode** (``isinstance(block.content, dict)``): the correct
  answer is ``content["answer_key"]``; the wrong options are the
  ``content["distractors"]`` ``text`` fields. The audited option pool
  is ``[answer_key] + distractor_texts``.
- **Rewrite mode** (``isinstance(block.content, str)``): the options
  are the ``<li data-cf-distractor-index="N">`` siblings; the correct
  option is the one carrying ``data-cf-correct="true"`` (the canonical
  rewrite-tier marker emitted per ``Courseforge/generators/
  _rewrite_provider.py``). When no option carries the correct marker
  the rewrite-tier block can't resolve a correct value, so the block
  is skipped (pass) — the W7 payload gate owns the missing-marker
  axis.

Decision capture: emits exactly one
``decision_type="assessment_numeric_equivalence_check"`` event per
AUDITED assessment_item Block, with signals ``block_id``,
``n_options``, ``n_numeric_parsed``, and ``equivalent_found`` — so the
audit log lets an operator replay the per-block verdict post-hoc.
"""
from __future__ import annotations

import logging
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

# ``Block`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# import bridge the sibling distractor validators use so this module
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
# drown the gate report (mirrors ``_ISSUE_LIST_CAP`` in the sibling
# distractor validators).
_ISSUE_LIST_CAP: int = 50

# Rewrite-mode HTML scraper. Captures the FULL ``<li>`` element (group 1)
# plus the body text (group 2) of every ``<li data-cf-distractor-
# index="N">`` sibling — the full element is needed to detect the
# ``data-cf-correct="true"`` correct-answer marker before tags are
# stripped. Mirrors the regex in ``lib/validators/
# distractor_plausibility.py`` / ``assessment_item_payload.py``.
_DATA_CF_DISTRACTOR_INDEX_LI_RE = re.compile(
    r'(<li[^>]*data-cf-distractor-index=["\']\d+["\'][^>]*>(.*?)</li>)',
    re.IGNORECASE | re.DOTALL,
)
# Correct-answer marker on a rewrite-tier ``<li>`` (per the rewrite
# provider's canonical contract). Matched against the FULL element
# (attributes included) before HTML strip.
_DATA_CF_CORRECT_RE = re.compile(
    r'data-cf-correct=["\']true["\']',
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# A token that looks like a parseable number / fraction. Strips
# surrounding markup-ish noise (whitespace, a single leading sign,
# enclosing punctuation) before the Fraction parse. Mixed numbers like
# ``1 1/2`` are handled by ``_parse_numeric`` directly.
_FRACTION_RE = re.compile(r"^[+-]?\d+\s*/\s*\d+$")
_MIXED_RE = re.compile(r"^([+-]?)(\d+)\s+(\d+)\s*/\s*(\d+)$")
_DECIMAL_INT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")
# Strip a leading/trailing run of non-numeric, non-operator chars so
# ``"(2/3)"`` / ``"2/3."`` / ``"x = 2/3"`` reduce to the numeric core.
# Conservative: only peels chars that can't be part of a number.
_NUMERIC_CORE_RE = re.compile(
    r"[+-]?(?:\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d+\.?\d*|\.\d+)"
)


def _strip_html(html: str) -> str:
    """Strip tags + collapse whitespace."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _parse_numeric(text: str) -> Optional[Fraction]:
    """Parse ``text`` into a ``Fraction`` value, or ``None`` if it's not
    a number / fraction.

    Handles integers (``"5"``), decimals (``"0.5"``), fractions
    (``"4/6"``), and simple mixed numbers (``"1 1/2"``). Surrounding
    markup / whitespace / trivial punctuation is stripped to the
    numeric core. Returns ``None`` for any string with no parseable
    numeric core (text options, multi-number expressions, etc.) so the
    caller can scope-limit cleanly.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    # Canonical MCQ options are authored as "VALUE — rationale" (em/en-dash
    # rationale separator, per the rewrite-tier assessment_item contract).
    # Parse ONLY the leading VALUE so the trailing rationale's numbers (e.g.
    # "2/3 — divide 20 and 30 by 10") don't make this look like a multi-value
    # expression and get declined.
    for _sep in ("—", "–"):
        if _sep in raw:
            raw = raw.split(_sep, 1)[0].strip()
            break

    # Peel a numeric core out of light surrounding markup. If the whole
    # string already is a clean numeric token the regex match equals the
    # input; otherwise we extract the first numeric run ONLY when it is
    # the sole numeric content (a string with two distinct numbers — an
    # expression, not an answer value — must NOT silently collapse to
    # its first number).
    candidate = raw
    if not (
        _FRACTION_RE.match(raw)
        or _MIXED_RE.match(raw)
        or _DECIMAL_INT_RE.match(raw)
    ):
        cores = _NUMERIC_CORE_RE.findall(raw)
        if len(cores) != 1:
            # Zero numeric cores → non-numeric text option.
            # Two+ numeric cores → an expression / multi-value string;
            # ambiguous, so decline to parse (scope-limit to single-
            # value answers).
            return None
        candidate = cores[0].strip()

    # Mixed number, e.g. "1 1/2" → 3/2.
    m = _MIXED_RE.match(candidate)
    if m:
        sign, whole, num, den = m.groups()
        try:
            if int(den) == 0:
                return None
            value = Fraction(int(whole)) + Fraction(int(num), int(den))
        except (ValueError, ZeroDivisionError):
            return None
        return -value if sign == "-" else value

    # Fraction, e.g. "4/6" → 2/3.
    if _FRACTION_RE.match(candidate):
        try:
            return Fraction(candidate.replace(" ", ""))
        except (ValueError, ZeroDivisionError):
            return None

    # Integer / decimal, e.g. "5" / "0.5".
    if _DECIMAL_INT_RE.match(candidate):
        try:
            # Fraction(str) handles "0.5" exactly via Decimal-style parse
            # ("0.5" → Fraction(1, 2)).
            return Fraction(candidate)
        except (ValueError, ZeroDivisionError):
            return None

    return None


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs['blocks']``.

    Mirrors the sibling distractor validators' ``_coerce_blocks`` so the
    same shape contract applies.
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


def _extract_outline(
    block: Any,
) -> Tuple[Optional[str], List[str]]:
    """Return ``(correct_answer_text, [option_text, ...])`` for an
    outline-tier (dict-content) Block, or ``(None, [])`` when it can't
    be audited.

    The correct answer is ``content["answer_key"]``; the wrong options
    are the per-distractor ``text`` fields. The returned option pool is
    ``[answer_key] + distractor_texts`` so the caller compares every
    distractor against the answer value.
    """
    content = getattr(block, "content", None)
    if not isinstance(content, dict):
        return None, []
    answer_key = content.get("answer_key")
    if not isinstance(answer_key, str) or not answer_key.strip():
        return None, []
    raw = content.get("distractors")
    options: List[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if isinstance(text, str) and text.strip():
                options.append(text)
    return answer_key, options


def _extract_rewrite(
    block: Any,
) -> Tuple[Optional[str], List[str]]:
    """Return ``(correct_answer_text, [distractor_text, ...])`` for a
    rewrite-tier (str-content) Block, or ``(None, [])`` when it can't be
    audited.

    Scrapes the ``<li data-cf-distractor-index="N">`` siblings; the
    correct option is the one carrying ``data-cf-correct="true"``. When
    no option carries the correct marker the block can't resolve a
    correct value, so ``(None, [])`` is returned (skip — the W7 payload
    gate owns the missing-marker axis).
    """
    content = getattr(block, "content", None)
    if not isinstance(content, str):
        return None, []
    correct: Optional[str] = None
    distractors: List[str] = []
    for match in _DATA_CF_DISTRACTOR_INDEX_LI_RE.finditer(content):
        full_el = match.group(1) or ""
        body = _strip_html(match.group(2) or "")
        if not body:
            continue
        if _DATA_CF_CORRECT_RE.search(full_el):
            # First marked-correct option wins (the W7 gate handles the
            # multiple-correct anomaly; we just need one correct value).
            if correct is None:
                correct = body
        else:
            distractors.append(body)
    if correct is None:
        return None, []
    return correct, distractors


def _extract(block: Any) -> Tuple[Optional[str], List[str]]:
    """Dispatch to the correct extractor based on content shape.

    Returns ``(correct_answer_text, [distractor_text, ...])``. The
    outline extractor returns ALL wrong options; the rewrite extractor
    returns the wrong options (correct option pulled out as the answer).
    """
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        return _extract_outline(block)
    if isinstance(content, str):
        return _extract_rewrite(block)
    return None, []


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    n_options: int,
    n_numeric_parsed: int,
    equivalent_found: bool,
) -> None:
    """Emit one ``assessment_numeric_equivalence_check`` decision per
    AUDITED assessment_item Block.

    Rationale interpolates block_id + option count + numeric-parsed
    count + the equivalent-found verdict so the audit log lets an
    operator replay the per-block decision without re-running the gate.
    """
    if capture is None:
        return
    decision = "failed" if equivalent_found else "passed"
    rationale = (
        f"Numeric-equivalence audit of assessment_item block "
        f"{block_id!r}: n_options={n_options}, "
        f"n_numeric_parsed={n_numeric_parsed}, "
        f"equivalent_found={equivalent_found}; a distractor equal in "
        f"VALUE to the correct answer (e.g. 4/6 == 2/3) is a second "
        f"correct answer, not a distractor — outcome={decision}."
    )
    metrics: Dict[str, Any] = {
        "block_id": block_id,
        "n_options": int(n_options),
        "n_numeric_parsed": int(n_numeric_parsed),
        "equivalent_found": bool(equivalent_found),
    }
    try:
        capture.log_decision(
            decision_type="assessment_numeric_equivalence_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "assessment_numeric_equivalence_check: %s",
            exc,
        )


def _audit_block(
    block: Any,
    *,
    capture: Any,
    issues: List[GateIssue],
) -> bool:
    """Audit one assessment_item Block for numeric-value equivalence.

    Returns ``True`` iff the block was AUDITED (a numeric MCQ with a
    parseable correct answer); returns ``False`` when the block was
    scope-skipped (non-numeric / unresolvable correct answer). Side-
    effect: appends ``DISTRACTOR_EQUALS_ANSWER_VALUE`` GateIssue entries
    on each value-equivalent distractor and emits the per-block decision
    capture when audited.
    """
    correct_text, distractor_texts = _extract(block)
    if correct_text is None:
        # Couldn't resolve a correct answer (non-dict/str content, no
        # answer_key, no correct marker) — scope-skip, not audited.
        return False

    correct_value = _parse_numeric(correct_text)
    if correct_value is None:
        # Correct answer isn't numeric — non-math MCQ. Scope-skip
        # (pass), never false-positive on text options.
        return False

    block_id = str(getattr(block, "block_id", "<unknown>"))

    # Parse every distractor; count numeric parses for the capture
    # signal. n_options = correct + distractors.
    n_options = 1 + len(distractor_texts)
    n_numeric_parsed = 1  # the correct answer parsed
    equivalent_found = False

    for idx, dtext in enumerate(distractor_texts):
        dvalue = _parse_numeric(dtext)
        if dvalue is None:
            continue
        n_numeric_parsed += 1
        if dvalue == correct_value:
            equivalent_found = True
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(GateIssue(
                    severity="warning",
                    code="DISTRACTOR_EQUALS_ANSWER_VALUE",
                    message=(
                        f"assessment_item Block {block_id!r} "
                        f"distractor[{idx}]={dtext!r} parses to the "
                        f"SAME numeric value ({dvalue}) as the correct "
                        f"answer {correct_text!r}; an equivalent form "
                        f"(e.g. 4/6 == 2/3) is a SECOND correct answer, "
                        f"not a distractor. Token-Jaccard distinctness "
                        f"can't catch this (4/6 vs 2/3 → Jaccard 0.0)."
                    ),
                    location=block_id,
                    suggestion=(
                        "Re-roll the rewrite tier — emit distractors "
                        "that differ in actual numeric VALUE from the "
                        "correct answer, not merely in surface form. "
                        "Never use a fraction/decimal/integer that "
                        "reduces to the same value as the key."
                    ),
                ))

    _emit_decision(
        capture,
        block_id=block_id,
        n_options=n_options,
        n_numeric_parsed=n_numeric_parsed,
        equivalent_found=equivalent_found,
    )
    return True


class AssessmentNumericEquivalenceValidator:
    """Numeric-value distractor-equivalence gate.

    CPU-only ``fractions.Fraction`` parse over ``assessment_item`` Block
    options. One GateIssue code (``warning``, ``action="regenerate"``):

    - ``DISTRACTOR_EQUALS_ANSWER_VALUE`` — a distractor parses to the
      same numeric value as the correct answer (the ``4/6 ≡ 2/3`` blind
      spot token-Jaccard can't detect).

    Filters to ``block.block_type == "assessment_item"``; other types
    silently skip. Dispatches outline-tier (dict content, correct =
    ``answer_key``) and rewrite-tier (HTML string content, correct =
    the ``data-cf-correct="true"`` ``<li>``) shapes. Scope-limited:
    non-numeric / unresolvable-correct blocks are skipped (pass), never
    false-positived.

    Per-call kwargs (in ``inputs``):

    - ``decision_capture``: optional ``DecisionCapture`` for the
      per-audited-block ``assessment_numeric_equivalence_check`` event.
    """

    name = "assessment_numeric_equivalence"
    version = "0.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

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

        issues: List[GateIssue] = []
        audited = 0

        for block in blocks:
            if getattr(block, "block_type", None) != "assessment_item":
                continue
            if _audit_block(block, capture=capture, issues=issues):
                audited += 1

        passed = len(issues) == 0
        score = 1.0
        if audited > 0 and not passed:
            # Rough quality signal: fraction of audited blocks with no
            # value-equivalent distractor. Each issue maps to one
            # offending distractor; bound the divisor to audited so the
            # score stays in [0, 1].
            denom = max(1, audited)
            clean = max(0, audited - len(issues))
            score = round(clean / denom, 4)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


__all__ = ["AssessmentNumericEquivalenceValidator"]
