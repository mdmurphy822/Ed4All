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

All three checks are PURE-DETERMINISTIC (no model, no embedding). The validator
audits key-terms vocab cards only (``template_type == "key_terms"`` or
``block_type == "vocab_card"``), reading the term + definition off the block's
explicit fields when present, else parsing them out of the pre-rendered card
HTML via the sibling ``key_terms`` helpers.

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
#: ``Block.template_type``. Imported lazily to keep the validator dependency-thin.
_KEY_TERMS_TEMPLATE_TYPE = "key_terms"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


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
    """True iff the block is a key-terms vocab card (either marker)."""
    if _template_type_of(block) == _KEY_TERMS_TEMPLATE_TYPE:
        return True
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

        # First pass — resolve each card + accumulate normalized-definition
        # buckets so the not-distinct (shared-definition) check can fire.
        resolved: List[Dict[str, Any]] = []
        def_buckets: Dict[str, List[str]] = {}
        for idx, block in enumerate(blocks):
            if not _is_key_terms_card(block):
                continue
            td = _term_and_definition(block)
            if td is None:
                continue
            cards_audited += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            entry = {
                "block_id": block_id,
                "term": td["term"],
                "definition": td["definition"],
            }
            resolved.append(entry)
            norm_def = _norm(td["definition"])
            if norm_def:
                def_buckets.setdefault(norm_def, []).append(block_id)

        # Normalized definitions shared by ≥2 distinct cards → not-distinct.
        shared_defs = {d for d, ids in def_buckets.items() if len(ids) > 1}

        for entry in resolved:
            block_id = entry["block_id"]
            term = entry["term"]
            definition = entry["definition"]
            codes: List[str] = []

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

            for code in codes:
                flagged += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        self._issue_for(
                            code, block_id, term, len(definition), ceiling
                        )
                    )
            per_block[block_id] = {
                "term": term,
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
    def _term_in_definition(norm_term: str, norm_def: str) -> bool:
        """Whole-word containment of ``norm_term`` in ``norm_def``."""
        if not norm_term or not norm_def:
            return False
        return re.search(
            r"\b" + re.escape(norm_term) + r"\b", norm_def
        ) is not None

    @staticmethod
    def _issue_for(
        code: str, block_id: str, term: str, def_chars: int, ceiling: int
    ) -> GateIssue:
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
        }
        suggestions = {
            "KEYTERM_DEF_CIRCULAR": "Re-author the definition without restating the term.",
            "KEYTERM_DEF_TOO_LONG": "Trim the definition to a single line.",
            "KEYTERM_DEF_NOT_DISTINCT": "Give each term its own distinct definition.",
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
