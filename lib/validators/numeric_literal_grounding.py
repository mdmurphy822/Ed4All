"""Numeric-literal grounding verifier — the fabrication control for NUMERIC /
math content that the number-blind NLI gate (``block_prose_entailment``) cannot
provide.

Established this session (``plans/finegrain/content-block-quality-2026-06.md``
iters 5/5b): **DeBERTa-v3-mnli is NUMBER-BLIND.** With the premise fixed it
scores ``-55/84``(correct) 0.502 vs ``-1/2``(wrong) 0.487 — a 0.015 gap — and
the fabrication ``-40/88 = -5/11`` (0.542, present in ZERO of the 72 real
source chunks) OUTSCORES every grounded math claim. So ``block_prose_entailment``
gives no fabrication protection on numbers, and ``groundedness._is_computational``
currently EXEMPTS such claims (zero protection). This validator runs a
numeric-literal cross-check ALONGSIDE NLI (not a text mutation) — the principled
fix recommended at iter-5b.

Precedent reused
----------------
The numeric-literal extraction behind ``ED4ALL_ANSWER_NLI_ADD``
(``lib.retrieval.citation_attribution.claim_numeric_literals`` /
``claim_numerics_present``) — "every numeric literal in the claim present in the
chunk (NLI is number-blind)". That guard handles plain decimals / currency /
percent but NOT signed fractions ``a/b`` or the OCR space-separated digit groups
the real DART chunkset stores (``-55/84`` lives as ``− 55 84``). This module
EXTENDS the precedent's normalization (unicode minus / slash) and adds a
**paired-fraction** extractor + OCR-tolerant containment so a fraction's
``(numerator, denominator)`` pair is matched as adjacent digit groups in source.

The measured rule (real-data, see the module-level investigation)
-----------------------------------------------------------------
On the 8 real 7B blocks vs the 72 real chunks of a course archive
(``LibV2/courses/<course-slug>/dart_chunks/chunks.jsonl``):

* The fabrication ``-40/88`` (a worked-example INPUT that appears NOWHERE as an
  adjacent digit-group pair in the 72 chunks) is FLAGGED — present in BOTH the
  ``explanation`` and ``misconception`` blocks.
* Every grounded fraction (``-55/84``, ``-11/12``, ``-5/7``, ``ex1.75`` etc.) —
  including OCR-flattened forms (``− 55 84``) and legitimately COMPUTED RESULTS
  that textbook worked examples print (``-55/84`` is the result of
  ``-11/12 · 5/7``) — is matched and NOT flagged.

The grounded blocks have source-absent-fraction ratio **0.00**; the fabrication
blocks **0.20 / 0.50** with ≥1 source-absent fraction. The separation is clean,
so the rule is: **flag a block when it carries ≥ 1 source-absent numeric
literal** (a fraction whose ``num/denom`` pair, or a standalone decimal, appears
nowhere in the block's cited source as adjacent digit groups). The block's
``source_absent_ratio`` is additionally surfaced as a calibration signal so a
future flip can move to a ratio threshold if a different corpus shows legitimate
computed-result false-positives.

Nuance: fabricated INPUTS vs computed RESULTS
---------------------------------------------
A naive "every literal must be present" would risk false-positiving a correct
reduction's RESULT that the source happens not to print. On this corpus that
never bites — textbooks print worked-example results — so the strict "≥1 absent"
rule is the strictest defensible rule with ZERO false positives here. To stay
robust on corpora that DON'T print every result, the validator (a) restricts the
flag to fractions (paired ``a/b`` — a fabricated worked-example INPUT is almost
always a fraction, while a stray bare integer is rarely a fabrication) and (b)
surfaces ``source_absent_ratio`` so a calibration flip can switch to a ratio
floor instead of the count floor. The ``# TODO(calibration)`` marker at the gate
in ``config/workflows.yaml`` tracks that flip.

Graceful-degrade
----------------
This is a PURE text/regex validator (no embeddings, no NLI) so there is no
dependency to degrade on — it always runs. It still honors the
statistical-tier shape: a block with no cited source chunks emits a warning
``NUMERIC_NO_GROUNDING_SOURCE`` (``passed=True``) and the resolution gate
(``rewrite_source_refs``) owns the structural missing-source failure.

Shadow knob (mirrors ``block_prose_entailment`` §4.3)
-----------------------------------------------------
``inputs["shadow"]`` (merged from the gate ``config.shadow: true`` by the
executor). When truthy every emitted ``NUMERIC_*`` GateIssue is forced to
``severity="warning"`` and ``action`` is never ``"regenerate"`` — the gate
COMPUTES + CAPTURES but does NOT gate. The critical flip is
``config.shadow: false`` + YAML ``severity: critical`` (DEFERRED — see the
``# TODO(calibration)`` marker at the gate in ``config/workflows.yaml``).

Decision-capture
----------------
One ``numeric_literal_grounding_check`` event per SCORED block; rationale
interpolates dynamic per-call signals (≥20 chars): block_id, literals found,
the source-absent set, ratio. Skip-silent blocks emit nothing (matches
``block_prose_entailment`` / ``claim_support``).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.utils import strip_html_to_text as _strip_html_to_text  # W-D6 canonical
from lib.validators.claim_support import _SKIPPED_BLOCK_TYPES

# Reuse the precedent's normalized decimal / currency / percent extractor so the
# two numeric guards never diverge on what a "numeric literal" is.
from lib.retrieval.citation_attribution import (
    _normalize_numeric,
    claim_numeric_literals,
)

# Block lives at Courseforge/scripts/blocks.py — same import bridge as the
# sibling block validators (block_prose_entailment / claim_support).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# OCR-tolerant normalization + paired-fraction extraction
# --------------------------------------------------------------------------- #

#: Unicode minus variants the DART OCR emits in place of ASCII ``-``.
_UNICODE_MINUS = ("−", "–", "—", "‐", "‑")
#: Unicode slash / division-slash variants.
_UNICODE_SLASH = ("⁄", "∕")


def normalize_math_text(text: str) -> str:
    """OCR-tolerant projection used for the membership test ONLY.

    Unifies unicode minus / slash to ASCII so ``− 55 84`` (OCR) and ``-55/84``
    (model LaTeX) compare on a common surface. NEVER mutates stored content —
    callers pass a copy. Whitespace is preserved (the fraction matcher needs the
    digit-group gaps).
    """
    for m in _UNICODE_MINUS:
        text = text.replace(m, "-")
    for s in _UNICODE_SLASH:
        text = text.replace(s, "/")
    return text


#: LaTeX ``\frac{a}{b}`` — the model's clean rendering. Captures the signed
#: numerator/denominator integers.
_FRAC_LATEX_RE = re.compile(r"\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}")
#: Inline ``a/b`` fraction (NOT preceded/followed by a digit or dot so we don't
#: split a decimal or a date). Operates on the unicode-normalized text.
_FRAC_INLINE_RE = re.compile(r"(?<![\d.])(-?\d+)\s*/\s*(-?\d+)(?![\d.])")


def extract_fractions(text: Optional[str]) -> List[Tuple[str, str]]:
    """Normalized ``(numerator, denominator)`` pairs in ``text`` (order-preserving).

    Both LaTeX ``\\frac{a}{b}`` and inline ``a/b`` forms. Signs are stripped (the
    pair is matched on magnitude — OCR rarely preserves the sign adjacent to the
    digit group, and the sign is not the fabrication signal). Returns ``[]`` when
    no fraction is present.
    """
    norm = normalize_math_text(text or "")
    out: List[Tuple[str, str]] = []
    for a, b in _FRAC_LATEX_RE.findall(norm):
        out.append((a.lstrip("-"), b.lstrip("-")))
    for a, b in _FRAC_INLINE_RE.findall(norm):
        out.append((a.lstrip("-"), b.lstrip("-")))
    return out


def fraction_in_source(frac: Tuple[str, str], source_text: str) -> bool:
    """OCR-tolerant containment of a fraction pair in ``source_text``.

    The real chunks store ``-55/84`` as ``− 55 84`` (space-separated digit
    groups), so the membership test allows whitespace / a unicode-or-ASCII
    minus / slash / division-dot BETWEEN the numerator and denominator groups.
    Digit-boundary guards (``(?<!\\d)`` / ``(?!\\d)``) stop ``5/8`` matching a
    longer ``55/84`` run. Both sides are unicode-normalized first.
    """
    a, b = frac
    if not a or not b:
        return False
    src = normalize_math_text(source_text or "")
    pattern = (
        r"(?<!\d)" + re.escape(a) + r"\s*[-/.·]*\s*" + re.escape(b) + r"(?!\d)"
    )
    return bool(re.search(pattern, src))


def decimal_in_source(literal: str, source_text: str) -> bool:
    """OCR-tolerant containment of a normalized decimal/integer literal.

    Reuses the precedent's :func:`claim_numeric_literals` to enumerate the
    source's normalized literals and tests membership on the normalized form,
    so ``1.75`` matches a chunk's ``1.75`` (and the precedent's rule that
    ``27.78`` does NOT match ``27.8`` carries over — decimals are literal).
    Also tolerates the OCR-flattened ``1 . 75`` form by collapsing
    ``digit . digit`` spacing before extraction.
    """
    if not literal:
        return False
    norm_src = normalize_math_text(source_text or "")
    # Collapse OCR-spaced decimals: ``1 . 75`` -> ``1.75``.
    norm_src = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", norm_src)
    source_literals = set(claim_numeric_literals(norm_src))
    return literal in source_literals


# --------------------------------------------------------------------------- #
# Canonical GateIssue codes + decision type
# --------------------------------------------------------------------------- #

_CODE_NUMERIC_FABRICATION: str = "NUMERIC_LITERAL_UNGROUNDED"
_CODE_NUMERIC_NO_GROUNDING_SOURCE: str = "NUMERIC_NO_GROUNDING_SOURCE"

_DECISION_TYPE: str = "numeric_literal_grounding_check"

#: Cap per-block issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Default flag rule: a block fires when it carries at least this many
#: source-absent FRACTION literals. Day-1 1 (the strictest defensible rule —
#: zero false-positives on the measured corpus). Calibratable to a ratio floor
#: (``source_absent_ratio``) before the critical flip.
_DEFAULT_MIN_ABSENT_FRACTIONS: int = 1


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]`` (mirrors siblings)."""
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


def _block_should_skip(block: Block) -> bool:
    """True when the block's block_type / content_type is in the skip set."""
    if block.block_type in _SKIPPED_BLOCK_TYPES:
        return True
    if (
        block.content_type_label
        and block.content_type_label in _SKIPPED_BLOCK_TYPES
    ):
        return True
    return False


def _resolve_block_source_text(
    block: Block,
    source_chunks: Dict[str, str],
) -> str:
    """Concatenate the block's cited chunk texts (mirrors block_prose_entailment).

    Walks ``block.source_references[].sourceId`` + ``block.source_ids`` and
    joins each resolved chunk body. Empty string when nothing resolves.
    """
    seen: set = set()
    texts: List[str] = []
    candidate_ids: List[str] = []
    for ref in block.source_references or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid:
                candidate_ids.append(sid)
    for sid in block.source_ids or ():
        if isinstance(sid, str) and sid:
            candidate_ids.append(sid)
    for sid in candidate_ids:
        if sid in seen:
            continue
        seen.add(sid)
        text = source_chunks.get(sid)
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts)


def _emit_decision(
    capture: Any,
    block: Block,
    *,
    passed: bool,
    fractions: Sequence[Tuple[str, str]],
    absent_fractions: Sequence[Tuple[str, str]],
    source_absent_ratio: float,
    min_absent_fractions: int,
    shadow: bool,
) -> None:
    """Emit one ``numeric_literal_grounding_check`` decision per scored block.

    Rationale interpolates dynamic per-call signals (≥20 chars) — block_id, the
    literal set, the source-absent set, the ratio — so the capture is replayable
    post-hoc (root ``CLAUDE.md`` § "LLM call-site instrumentation"). Swallows
    ``log_decision`` exceptions (capture must not break the gate).
    """
    if capture is None:
        return
    frac_str = ", ".join(f"{a}/{b}" for a, b in fractions) or "none"
    absent_str = ", ".join(f"{a}/{b}" for a, b in absent_fractions) or "none"
    decision = "passed" if passed else f"failed:{_CODE_NUMERIC_FABRICATION}"
    rationale = (
        f"Numeric-literal grounding on Block {block.block_id!r}: "
        f"block_type={block.block_type!r}, "
        f"fractions_found=[{frac_str}], "
        f"source_absent_fractions=[{absent_str}], "
        f"absent_count={len(absent_fractions)}, "
        f"fraction_count={len(fractions)}, "
        f"source_absent_ratio={source_absent_ratio:.4f}, "
        f"min_absent_fractions={min_absent_fractions}, "
        f"shadow={shadow}. "
        f"NLI is number-blind so fabricated worked-example inputs "
        f"(e.g. -40/88) pass block_prose_entailment; this cross-check "
        f"catches a fraction pair absent from the cited source."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s: %s",
            _DECISION_TYPE, exc,
        )


class NumericLiteralGroundingValidator:
    """Per-block numeric-literal grounding gate (fabrication control for math).

    For each rewrite-tier (string-content) block, extracts numeric FRACTION
    literals from the prose, then asserts each fraction's ``num/denom`` pair
    appears in the block's cited source chunk(s) under OCR-tolerant containment
    (the real chunks store ``-55/84`` as ``− 55 84``). Fires
    ``NUMERIC_LITERAL_UNGROUNDED`` when a block carries ≥
    ``min_absent_fractions`` source-absent fractions.

    Inputs:
        blocks: List[Block]
        source_chunks: Dict[str, str]
            Mapping of canonical sourceId (e.g. ``dart:slug#blk_0``) to the
            chunk's plain-text body. The gate-input builder
            (``_build_rewrite_block_input``) populates this from the staging
            manifest + DART chunks.jsonl body fallback; tests inject it directly.
        shadow: bool
            Forces emitted issues to warning-severity (compute + capture only).
        min_absent_fractions: Optional[int]
            Flag threshold (merged from gate ``config.thresholds``).
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per scored block.
    """

    name = "numeric_literal_grounding"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_absent_fractions: int = _DEFAULT_MIN_ABSENT_FRACTIONS,
    ) -> None:
        self._min_absent = int(min_absent_fractions)

    def _resolve_thresholds(self, inputs: Dict[str, Any]) -> int:
        raw = inputs.get("min_absent_fractions", self._min_absent)
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = self._min_absent
        return max(1, val)

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        shadow = bool(inputs.get("shadow", False))

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

        min_absent = self._resolve_thresholds(inputs)

        source_chunks_raw = inputs.get("source_chunks", {}) or {}
        if not isinstance(source_chunks_raw, dict):
            source_chunks_raw = {}
        source_chunks: Dict[str, str] = {
            str(k): v for k, v in source_chunks_raw.items() if isinstance(v, str)
        }

        # Severity honors the shadow knob (warning during calibration → critical
        # after the flip), mirroring block_prose_entailment.
        fail_severity = "warning" if shadow else "critical"

        issues: List[GateIssue] = []
        any_regenerate = False
        audited_blocks = 0
        passed_blocks = 0

        for block in blocks:
            content = block.content
            if not isinstance(content, str):
                # Outline-tier dict content — skip silently (rewrite-tier seam).
                continue
            if _block_should_skip(block):
                continue

            prose_text = _strip_html_to_text(content)
            if not prose_text.strip():
                continue

            fractions = list(dict.fromkeys(extract_fractions(prose_text)))
            if not fractions:
                # No fraction literals — nothing to ground. Not audited (no
                # numeric content), no event (matches skip-silent discipline).
                continue

            source_text = _resolve_block_source_text(block, source_chunks)

            if not source_text.strip():
                # No grounding surface. The resolution gate (rewrite_source_refs)
                # owns the structural missing-source failure; this gate emits a
                # warning so the operator knows the numeric signal is unavailable
                # but passes the block.
                audited_blocks += 1
                passed_blocks += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=_CODE_NUMERIC_NO_GROUNDING_SOURCE,
                            message=(
                                f"Rewrite-tier Block {block.block_id!r} "
                                f"(block_type={block.block_type!r}) carries "
                                f"{len(fractions)} fraction literal(s) but "
                                f"declares no resolvable cited source chunks; "
                                f"numeric-literal grounding cannot be evaluated."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block,
                    passed=True,
                    fractions=fractions,
                    absent_fractions=[],
                    source_absent_ratio=0.0,
                    min_absent_fractions=min_absent,
                    shadow=shadow,
                )
                continue

            audited_blocks += 1
            absent_fractions = [
                f for f in fractions if not fraction_in_source(f, source_text)
            ]
            source_absent_ratio = (
                len(absent_fractions) / len(fractions) if fractions else 0.0
            )

            block_passed = len(absent_fractions) < min_absent
            if block_passed:
                passed_blocks += 1
            else:
                if not shadow:
                    any_regenerate = True
                if len(issues) < _ISSUE_LIST_CAP:
                    absent_str = ", ".join(
                        f"{a}/{b}" for a, b in absent_fractions
                    )
                    issues.append(
                        GateIssue(
                            severity=fail_severity,
                            code=_CODE_NUMERIC_FABRICATION,
                            message=(
                                f"Rewrite-tier Block {block.block_id!r} "
                                f"(block_type={block.block_type!r}): "
                                f"{len(absent_fractions)}/{len(fractions)} "
                                f"({source_absent_ratio:.1%}) fraction "
                                f"literal(s) [{absent_str}] do NOT appear in "
                                f"the block's cited source chunk(s) under "
                                f"OCR-tolerant containment. NLI is "
                                f"number-blind (it scores a fabricated "
                                f"worked-example input ABOVE grounded math), "
                                f"so block_prose_entailment cannot catch this; "
                                f"a source-absent fraction pair is a fabricated "
                                f"worked-example input."
                            ),
                            location=block.block_id,
                            suggestion=(
                                "The prose introduces a fraction whose "
                                "numerator/denominator pair appears nowhere in "
                                "the cited source. Re-prompt with the source "
                                "chunks pinned; worked-example numbers must be "
                                "copied from the cited source's worked examples."
                            ),
                        )
                    )

            _emit_decision(
                capture, block,
                passed=block_passed,
                fractions=fractions,
                absent_fractions=absent_fractions,
                source_absent_ratio=source_absent_ratio,
                min_absent_fractions=min_absent,
                shadow=shadow,
            )

        # ``passed`` honors only critical-severity issues — warning during
        # shadow so ``passed`` stays True; after the flip the NUMERIC_*
        # failures are critical and ``passed`` flips on a real fabrication
        # (mirrors block_prose_entailment / rewrite_source_grounding).
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = (
            1.0
            if audited_blocks == 0
            else round(passed_blocks / audited_blocks, 4)
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action="regenerate" if any_regenerate else None,
        )


__all__ = [
    "NumericLiteralGroundingValidator",
    "extract_fractions",
    "fraction_in_source",
    "decimal_in_source",
    "normalize_math_text",
    "_DEFAULT_MIN_ABSENT_FRACTIONS",
    "_CODE_NUMERIC_FABRICATION",
    "_CODE_NUMERIC_NO_GROUNDING_SOURCE",
    "_DECISION_TYPE",
]
