"""W1.5 — KeyTermsDefinitionQualityValidator (glossary definition quality).

The deterministic Key-Terms page builder (``lib/generation/key_terms.py``) lifts
each term's definition verbatim from a source chunk or the domain-concept
``definition_hint``. Three glossary-quality defects survive that anti-fabrication
contract because they are about the SHAPE of an otherwise-real definition, not
its provenance:

* ``KEYTERM_DEF_CIRCULAR`` — the term's own surface form appears (as a whole
  word) inside its definition (a circular / self-referential gloss:
  "A prime number is a number that is prime.").
* ``KEYTERM_DEF_TOO_LONG`` — the definition exceeds the ~200-char atomic
  single-idea ceiling (reuses ``resolve_body_char_ceiling`` for ``vocab_card``);
  a glossary entry is a one-line definition, not a paragraph.
* ``KEYTERM_DEF_NOT_DISTINCT`` — two (or more) terms on the page resolved to the
  SAME normalized definition (a copy-paste / wrong-chunk gloss).
* ``KEYTERM_DEF_MISSING_CONDITION`` — a definition that states a ratio / division
  / quotient (``\\frac``, ``/``, "divide" / "divided" / "ratio" / "quotient" /
  "denominator") but carries NO nonzero-denominator side-condition, OR a
  slope / difference-quotient definition that references two points without a
  distinct-points condition (``x₂ ≠ x₁``). The math definition is undefined at
  the omitted boundary; the message names WHICH condition is missing.

All four checks are PURE-DETERMINISTIC (no model, no embedding). The validator
audits two surfaces:

* key-terms vocab cards (``template_type == "key_terms"`` or a
  ``block_type == "vocab_card"`` that carries no other deterministic-card
  marker), reading the term + definition off the block's explicit fields when
  present, else parsing them out of the pre-rendered card HTML via the sibling
  ``key_terms`` helpers. The deterministic FAQ page builder
  (``lib/generation/faq_page.py``) reuses the ``vocab_card`` wrapper but stamps
  ``template_type == "faq"``; such a card is a grounded Q/A entry, NOT a
  term/definition glossary card, so it is explicitly EXCLUDED here (auditing a
  Q/A answer as a "definition" would be a false positive — the block-identity
  aliasing this exclusion guards against); and
* inline ``<div class="definition-box">`` blocks parsed out of the HTML of
  concept / explanation blocks. Each definition-box becomes a synthetic
  checkable unit carrying its PARENT block id (so a failure is actionable via
  ``--block-ids``) and runs through the same four checks. This widening closes
  the biggest blind spot — a formal definition buried inside an explanation
  block never reached the glossary-quality gate before.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + a single info
issue) unless ``ED4ALL_KEYTERM_DEF_QUALITY`` is truthy. When on, fires
WARNING-severity issues (``action="regenerate"``) — warning-day-1 with a
deferred critical-flip (``# TODO(calibration)`` after a ≥2-corpus FP
measurement, mirroring the ``callout_structure`` / ``mayer_ctml`` posture).

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks. Only key-terms vocab cards audited.
    keyterm_def_quality_enabled: Optional[bool]
        Override the ``ED4ALL_KEYTERM_DEF_QUALITY`` resolution (testing seam).
    body_char_ceiling: Optional[int]
        Override the ~200-char definition ceiling (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one
        ``content_structure_check`` event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"key_terms_definition_quality"``).

References:
    - ``lib/generation/key_terms.py`` (card shape + extractors).
    - ``lib/validators/callout_structure.py`` (sibling flag-gated block-type
      gate — structure mirrored).
    - ``lib/validators/_block_rubric_helpers.py::resolve_body_char_ceiling``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on term-dense corpora.
_ISSUE_LIST_CAP: int = 50

_KEYTERM_DEF_QUALITY_ENV = "ED4ALL_KEYTERM_DEF_QUALITY"
_TRUTHY: frozenset = frozenset({"1", "true", "yes", "on"})

#: The deterministic-block marker the key-terms builder stamps on
#: ``Block.template_type``. Kept as a local literal (not imported) to keep the
#: validator dependency-thin.
_KEY_TERMS_TEMPLATE_TYPE = "key_terms"

#: The deterministic-block marker the FAQ page builder
#: (``lib/generation/faq_page.py::FAQ_TEMPLATE_TYPE``) stamps on a ``vocab_card``
#: it reuses. A FAQ card is a grounded Q/A entry, not a glossary term, so it is
#: excluded from the glossary definition-quality audit. Local literal to stay
#: dependency-thin (kept in lock-step with the builder's ``FAQ_TEMPLATE_TYPE``).
_FAQ_TEMPLATE_TYPE = "faq"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

#: Block types whose HTML is scanned for inline ``<div class="definition-box">``
#: units (the gate-widening surface — a formal definition inside an explanation).
_DEFINITION_BOX_BLOCK_TYPES: frozenset = frozenset({"concept", "explanation"})

_DEFINITION_BOX_RE = re.compile(
    r'<div[^>]*class=["\'][^"\']*\bdefinition-box\b[^"\']*["\'][^>]*>(.*?)</div>',
    re.IGNORECASE | re.S,
)
_STRONG_RE = re.compile(r"<strong[^>]*>(.*?)</strong>", re.IGNORECASE | re.S)

#: Ratio / division / quotient STRUCTURE signal (word-boundary on the words so
#: "divides" / "divisible" do NOT match "divide"; literal ``/`` + ``\frac``).
_RATIO_WORD_RE = re.compile(
    r"\b(divide|divided|ratio|quotient|denominator)\b", re.IGNORECASE
)
#: Nonzero-denominator side-condition tokens (substring match on normalized text;
#: the operator forms ``!= 0`` / ``≠ 0`` are matched space-insensitively).
_NONZERO_CONDITION_TOKENS: tuple = (
    "nonzero",
    "non-zero",
    "not equal to zero",
    "not both zero",
    "cannot be zero",
    "excluding zero",
)
#: Distinct-points condition tokens for the slope / difference-quotient shape.
_DISTINCTNESS_TOKENS: tuple = (
    "distinct",
    "different",
    "unequal",
    "not equal",
    "not the same",
)


def _keyterm_def_quality_enabled() -> bool:
    """True iff ``ED4ALL_KEYTERM_DEF_QUALITY`` is truthy (read each call)."""
    return os.environ.get(_KEYTERM_DEF_QUALITY_ENV, "").strip().lower() in _TRUTHY


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    if hasattr(block, key):
        return getattr(block, key)
    return None


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _template_type_of(block: Any) -> str:
    tt = _block_attr(block, "template_type")
    return tt.strip().lower() if isinstance(tt, str) else ""


def _is_key_terms_card(block: Any) -> bool:
    """True iff the block is a key-terms vocab card (either marker).

    A ``vocab_card`` carrying the FAQ marker (``template_type == "faq"``) is a
    grounded Q/A card, not a glossary term, so it is excluded — otherwise the
    two deterministic card families alias on the shared ``vocab_card``
    ``block_type`` and FAQ answers get audited as term "definitions".
    """
    template_type = _template_type_of(block)
    if template_type == _KEY_TERMS_TEMPLATE_TYPE:
        return True
    if template_type == _FAQ_TEMPLATE_TYPE:
        return False
    return _block_type_of(block) == "vocab_card"


def _strip_text(s: Any) -> str:
    if not isinstance(s, str) or not s:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace for comparison keys."""
    return _WS_RE.sub(" ", str(text).lower()).strip()


def _term_and_definition(block: Any) -> Optional[Dict[str, str]]:
    """Resolve ``{term, definition}`` for a key-terms card (both shapes).

    Prefers explicit ``display``/``term`` + ``definition`` fields; falls back to
    parsing the pre-rendered vocab-card HTML in ``content`` via the sibling
    ``key_terms`` extractors. Returns ``None`` when neither resolves.
    """
    display = _block_attr(block, "display") or _block_attr(block, "term")
    definition = _block_attr(block, "definition")
    term = display.strip() if isinstance(display, str) else ""
    defn = definition.strip() if isinstance(definition, str) else ""

    content = _block_attr(block, "content")
    if isinstance(content, dict):
        # Outline-tier structured card ({term, definition}).
        if not term and isinstance(content.get("term"), str):
            term = content["term"].strip()
        if not defn and isinstance(content.get("definition"), str):
            defn = content["definition"].strip()
    elif isinstance(content, str) and content:
        try:
            from lib.generation.key_terms import (
                extract_card_definition_html,
                extract_card_term_html,
            )
        except Exception:  # noqa: BLE001 — degrade to str-strip
            extract_card_term_html = None  # type: ignore[assignment]
            extract_card_definition_html = None  # type: ignore[assignment]
        if not term and extract_card_term_html is not None:
            term = extract_card_term_html(content)
        if not defn and extract_card_definition_html is not None:
            defn = extract_card_definition_html(content)
        if not defn:
            # Last resort: whole stripped body (no shape) — still lets the
            # too-long / not-distinct checks run.
            defn = _strip_text(content)

    if not defn:
        return None
    return {"term": term or "", "definition": defn}


def _block_html(block: Any) -> str:
    """Best-effort HTML body of a concept/explanation block."""
    content = _block_attr(block, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("html", "body", "text"):
            v = content.get(key)
            if isinstance(v, str) and v:
                return v
    html = _block_attr(block, "html")
    return html if isinstance(html, str) else ""


def _definition_box_units(block: Any, parent_id: str) -> List[Dict[str, str]]:
    """Parse ``<div class="definition-box">`` units out of a block's HTML.

    Each definition-box (shape ``<div class="definition-box"><strong>Term</strong>
    …</div>``) becomes ``{term, definition, unit_id, location}``. The term is the
    first ``<strong>`` (removed from the definition body so it does not spuriously
    trip the circular check); ``location`` is the PARENT block id so a failure is
    actionable via ``--block-ids``; ``unit_id`` disambiguates multiple boxes in
    one parent. Pure string ops, deterministic, no parser dependency.
    """
    html = _block_html(block)
    if not html:
        return []
    units: List[Dict[str, str]] = []
    for idx, inner in enumerate(_DEFINITION_BOX_RE.findall(html)):
        strong = _STRONG_RE.search(inner)
        if strong is not None:
            term = _strip_text(strong.group(1))
            def_html = inner[: strong.start()] + inner[strong.end():]
        else:
            term = ""
            def_html = inner
        definition = _strip_text(def_html)
        if not definition:
            continue
        units.append(
            {
                "term": term,
                "definition": definition,
                "unit_id": f"{parent_id}#definition_box_{idx:02d}",
                "location": parent_id,
            }
        )
    return units


class KeyTermsDefinitionQualityValidator:
    """W1.5 — glossary definition-quality gate (circular / too-long / dupe)."""

    name = "key_terms_definition_quality"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import resolve_body_char_ceiling

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("keyterm_def_quality_enabled")
        if enabled is None:
            enabled = _keyterm_def_quality_enabled()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="KEYTERM_DEF_QUALITY_DISABLED",
                        message=(
                            "ED4ALL_KEYTERM_DEF_QUALITY unset — key-terms "
                            "definition-quality check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"key_terms_definition_quality": {"enabled": False}},
            )

        ceiling = resolve_body_char_ceiling(
            inputs.get("body_char_ceiling"), block_type="vocab_card"
        )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        cards_audited = 0
        flagged = 0
        per_block: Dict[str, Dict[str, Any]] = {}

        # First pass — resolve each checkable UNIT (key-terms cards + inline
        # definition-box units parsed out of concept/explanation blocks) +
        # accumulate normalized-definition buckets so the not-distinct
        # (shared-definition) check can fire across BOTH surfaces.
        resolved: List[Dict[str, Any]] = []
        def_buckets: Dict[str, List[str]] = {}
        for idx, block in enumerate(blocks):
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            units: List[Dict[str, str]] = []
            if _is_key_terms_card(block):
                td = _term_and_definition(block)
                if td is not None:
                    units.append(
                        {
                            "term": td["term"],
                            "definition": td["definition"],
                            "unit_id": block_id,
                            "location": block_id,
                        }
                    )
            elif _block_type_of(block) in _DEFINITION_BOX_BLOCK_TYPES:
                # Gate-widening surface: inline <div class="definition-box">
                # units inside a concept/explanation block.
                units.extend(_definition_box_units(block, block_id))

            for unit in units:
                cards_audited += 1
                resolved.append(unit)
                norm_def = _norm(unit["definition"])
                if norm_def:
                    def_buckets.setdefault(norm_def, []).append(unit["location"])

        # Normalized definitions shared by ≥2 distinct units → not-distinct.
        shared_defs = {d for d, ids in def_buckets.items() if len(set(ids)) > 1}

        for entry in resolved:
            unit_id = entry["unit_id"]
            location = entry["location"]
            term = entry["term"]
            definition = entry["definition"]
            codes: List[str] = []
            missing_cond: Optional[str] = None

            # (a) CIRCULAR — the term surface appears as a whole word in its
            #     own definition.
            norm_term = _norm(term)
            if norm_term and self._term_in_definition(norm_term, _norm(definition)):
                codes.append("KEYTERM_DEF_CIRCULAR")

            # (b) TOO_LONG — a glossary entry is a one-line definition.
            if len(definition) > ceiling:
                codes.append("KEYTERM_DEF_TOO_LONG")

            # (c) NOT_DISTINCT — two terms share a normalized definition.
            if _norm(definition) in shared_defs:
                codes.append("KEYTERM_DEF_NOT_DISTINCT")

            # (d) MISSING_CONDITION — a ratio/division/quotient (or slope /
            #     difference-quotient) definition that omits its side-condition.
            missing_cond = self._missing_math_condition(definition)
            if missing_cond is not None:
                codes.append("KEYTERM_DEF_MISSING_CONDITION")

            for code in codes:
                flagged += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        self._issue_for(
                            code,
                            location,
                            term,
                            len(definition),
                            ceiling,
                            missing_cond=missing_cond,
                        )
                    )
            per_block[unit_id] = {
                "term": term,
                "location": location,
                "def_chars": len(definition),
                "issues": codes,
            }

        self._emit_decision(
            capture,
            cards_audited=cards_audited,
            flagged=flagged,
            ceiling=ceiling,
            shared_defs=len(shared_defs),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1  # TODO(calibration): deferred critical-flip
            issues=issues,
            action="regenerate" if flagged else None,
            metadata={
                "key_terms_definition_quality": {
                    "enabled": True,
                    "cards_audited": cards_audited,
                    "flagged": flagged,
                    "def_char_ceiling": ceiling,
                    "shared_definition_groups": len(shared_defs),
                    "per_block": per_block,
                }
            },
        )

    @staticmethod
    def _missing_math_condition(definition: str) -> Optional[str]:
        """Return the missing side-condition kind, or ``None`` when well-formed.

        ``"distinct-points"`` — a slope / difference-quotient definition that
        references a subtraction over a quotient WITHOUT a distinctness token
        (``x₂ ≠ x₁``). ``"nonzero-denominator"`` — any other ratio / division /
        quotient definition WITHOUT a nonzero-denominator side-condition. The
        two branches are mutually exclusive: a slope shape is judged only on its
        distinct-points condition (its denominator ``x₂ - x₁`` IS that
        condition), never double-flagged on the generic nonzero rule.
        """
        norm = _norm(definition)
        spaceless = norm.replace(" ", "")

        # Slope / difference-quotient shape: a subtraction inside a quotient.
        has_sub = any(
            t in norm for t in ("-", "−", "minus", "difference", "change in")
        )
        has_quot = (
            "/" in definition
            or "\\frac" in definition
            or _RATIO_WORD_RE.search(definition) is not None
        )
        if "slope" in norm and has_sub and has_quot:
            has_distinctness = (
                "≠" in norm
                or "!=" in norm
                or any(t in norm for t in _DISTINCTNESS_TOKENS)
            )
            return None if has_distinctness else "distinct-points"

        # Generic ratio / division / quotient structure.
        if has_quot:
            has_nonzero = (
                any(t in norm for t in _NONZERO_CONDITION_TOKENS)
                or "!=0" in spaceless
                or "≠0" in spaceless
            )
            return None if has_nonzero else "nonzero-denominator"

        return None

    @staticmethod
    def _term_in_definition(norm_term: str, norm_def: str) -> bool:
        """Whole-word containment of ``norm_term`` in ``norm_def``."""
        if not norm_term or not norm_def:
            return False
        return re.search(
            r"\b" + re.escape(norm_term) + r"\b", norm_def
        ) is not None

    @staticmethod
    def _issue_for(
        code: str,
        block_id: str,
        term: str,
        def_chars: int,
        ceiling: int,
        *,
        missing_cond: Optional[str] = None,
    ) -> GateIssue:
        if missing_cond == "distinct-points":
            condition_msg = (
                f"Key term {term!r} (block {block_id!r}) states a "
                f"slope / difference-quotient without a distinct-points "
                f"condition (e.g. x₂ ≠ x₁ / nonzero run) — the "
                f"definition is undefined when the two points share an "
                f"x-coordinate."
            )
            condition_hint = "State the distinct-points condition (e.g. x₂ ≠ x₁)."
        else:  # nonzero-denominator (default for MISSING_CONDITION)
            condition_msg = (
                f"Key term {term!r} (block {block_id!r}) states a "
                f"ratio / division / quotient without a nonzero-denominator "
                f"condition (e.g. 'denominator ≠ 0') — the definition is "
                f"undefined when the divisor is zero."
            )
            condition_hint = "State the nonzero-denominator condition (e.g. b ≠ 0)."
        messages = {
            "KEYTERM_DEF_CIRCULAR": (
                f"Key term {term!r} (block {block_id!r}) has a circular "
                f"definition — the term's own surface form appears inside its "
                f"definition."
            ),
            "KEYTERM_DEF_TOO_LONG": (
                f"Key term {term!r} (block {block_id!r}) definition is "
                f"{def_chars} chars (> {ceiling}) — a glossary entry is a "
                f"one-line definition, not a paragraph."
            ),
            "KEYTERM_DEF_NOT_DISTINCT": (
                f"Key term {term!r} (block {block_id!r}) shares its definition "
                f"verbatim with another term on the page — each term needs a "
                f"distinct gloss."
            ),
            "KEYTERM_DEF_MISSING_CONDITION": condition_msg,
        }
        suggestions = {
            "KEYTERM_DEF_CIRCULAR": "Re-author the definition without restating the term.",
            "KEYTERM_DEF_TOO_LONG": "Trim the definition to a single line.",
            "KEYTERM_DEF_NOT_DISTINCT": "Give each term its own distinct definition.",
            "KEYTERM_DEF_MISSING_CONDITION": condition_hint,
        }
        return GateIssue(
            severity="warning",
            code=code,
            message=messages.get(code, code),
            location=block_id,
            suggestion=suggestions.get(code),
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        cards_audited: int,
        flagged: int,
        ceiling: int,
        shared_defs: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"key_terms_definition_quality_flagged={flagged}",
                rationale=(
                    f"W1.5 glossary definition-quality audit over {cards_audited} "
                    f"key-terms card(s) (def ceiling {ceiling}): {flagged} "
                    f"circular / too-long / not-distinct issue(s); "
                    f"{shared_defs} shared-definition group(s)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "key_terms_definition_quality: %s",
                exc,
            )


__all__ = ["KeyTermsDefinitionQualityValidator"]
