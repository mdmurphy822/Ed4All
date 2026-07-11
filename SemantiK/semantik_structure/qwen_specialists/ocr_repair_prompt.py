"""Stage-repair prompt builder + parser (SEMANTIK_OCR_CONFUSABLE_REPAIR).

A THIRD reviewer ROLE (plan Phase 5 channel 2): the 7B is a bounded OCR-
confusable micro-edit PROPOSER. It sees a window of FLAGGED region texts and
returns a JSON array of ``{region_index, before, after}`` micro-edits — it is
the CANDIDATE GENERATOR only. The deterministic accept gate in
:mod:`semantik_structure.qwen_specialists.ocr_repair` owns the decision (a known-
confusable map, edit-distance ≤ 2, char-class conservation, never inside a
numeric context), so a hallucinated / off-target proposal is dropped by
auditable code — never trusted on the model's say-so.

House style beside :mod:`reviewer_prompt` / :mod:`block_resegment_prompt`: a
self-contained ``SYSTEM:`` line (NEVER the ``_ENVELOPE_DIRECTIVE`` content-
authoring framing) + a one-line USER JSON. The builder returns ``""`` when the
flag is off so the flag-off path never dispatches (mirrors
``build_second_pass_verify_request``). The tolerant parser is the safety net,
not the JSON-only mandate.
"""

from __future__ import annotations

import json
from typing import Any


def _repair_enabled() -> bool:
    """Return ``resolve_ocr_confusable_repair_mode()`` (lazy import to break
    the ``ocr_repair`` <-> ``ocr_repair_prompt`` cycle — mirrors
    ``reviewer_prompt._second_pass_enabled``)."""
    from .ocr_repair import resolve_ocr_confusable_repair_mode

    return resolve_ocr_confusable_repair_mode()


_SYSTEM_OCR_REPAIR = (
    "You are an OCR-CONFUSABLE MICRO-REPAIR proposer for scanned-textbook "
    "text. Optical character recognition mangles a few characters around inline "
    "math and decorated glyphs: a radical sign becomes a capital V (\"Vx\" for "
    "\"√x\"), the letter l and the digit 1 swap, O and 0 swap, S and 5, B and "
    "8, the two-letter \"rn\" reads as \"m\", \"vv\" reads as \"w\", and stray "
    "decoration marks (™, curly quotes, a pipe |, a cent sign) are scanned "
    "into the text.\n"
    "\n"
    "Your ONE job: propose the SMALLEST possible character fixes that undo an "
    "obvious OCR confusion. You are a PROPOSER — a deterministic gate re-checks "
    "every proposal against a fixed confusable map (edit-distance ≤ 2, "
    "character-class safe, never inside a number), so an unsafe or wrong "
    "proposal is simply dropped. Propose ONLY within the region texts you are "
    "shown, keyed by their region_index.\n"
    "\n"
    "HARD RULES:\n"
    "  1. NEVER rewrite, paraphrase, translate, add, or delete WORDS. You only "
    "fix garbled CHARACTERS inside an existing token.\n"
    "  2. NEVER touch a genuine number, an answer key, a page number, or a "
    "numbered exercise item. If a token is mostly digits, leave it alone.\n"
    "  3. Each edit changes at most a couple of characters WITHIN one token; the "
    "token count and word spacing stay identical.\n"
    "  4. When in doubt, propose NOTHING — an empty array [] is the correct "
    "answer for clean text.\n"
    "\n"
    "OUTPUT — JSON ARRAY ONLY. Return EXACTLY ONE JSON array and nothing else "
    "(NO Markdown, NO code fences, NO commentary, NO surrounding prose). Emit "
    "ONE object per micro-edit. Each object's keys are: region_index (int — "
    "ECHO the region's region_index; an index NOT in this window is DROPPED), "
    "before (the exact garbled substring as it appears in the text), and after "
    "(the corrected substring). An EMPTY array [] means no repair is needed."
)


def build_ocr_repair_request(window: dict[str, Any]) -> str:
    """Emit the OCR-confusable repair prompt for ONE window (flagged texts in,
    micro-edit array out).

    Builds NOTHING (returns ``""``) when :func:`_repair_enabled` is False — the
    flag-off byte-stable contract. The SYSTEM turn carries the proposer mandate;
    the USER turn is the JSON object ``{"regions": [{region_index, text}],
    "allowed_region_indices": [...]}`` — ``allowed_region_indices`` PINS which
    ids this window may edit (it may only edit what it can see). Pure formatter —
    no LLM, no mutation, no env side effect beyond the gate read.
    """
    if not _repair_enabled():
        return ""
    regions = [
        {
            "region_index": e.get("region_index"),
            "text": str(e.get("text") or ""),
        }
        for e in window.get("regions", [])
        if isinstance(e, dict) and e.get("region_index") is not None
    ]
    allowed = window.get("allowed_region_indices")
    if allowed is None:
        allowed = [r["region_index"] for r in regions]
    user_payload = {
        "regions": regions,
        "allowed_region_indices": sorted(
            {i for i in allowed if isinstance(i, int)}
        ),
    }
    user_json = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    return f"SYSTEM: {_SYSTEM_OCR_REPAIR}\nUSER: {user_json}"


def _extract_json_array(raw: str | None) -> list[Any] | None:
    """Tolerant fail-soft extraction of a JSON ARRAY (fences/commentary stripped).

    ``None`` on empty / whitespace / garbage (no raise), so a broken proposer
    never crashes the loop. Mirrors ``reviewer_prompt._extract_json_array``.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text[3:]
        nl = text.find("\n")
        if nl != -1 and text[:nl].strip().isalpha():
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except (ValueError, TypeError):
        pass
    i = text.find("[")
    j = text.rfind("]")
    if i != -1 and j != -1 and j > i:
        try:
            val = json.loads(text[i : j + 1])
            if isinstance(val, list):
                return val
        except (ValueError, TypeError):
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    """Tolerant int coercion (rejects bool, an ``int`` subclass)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def parse_ocr_repair_response(raw: str | None) -> list[dict[str, Any]]:
    """Parse a proposer raw response into a list of candidate edit dicts.

    Returns ``[]`` (never raises) on unparseable / empty / non-array input
    (fail-soft — a broken proposer contributes no edits). Each surviving element
    is ``{"region_index": int, "before": str, "after": str}``; elements missing
    any of the three (or with a non-int region_index / empty before) are dropped.
    """
    data = _extract_json_array(raw)
    if data is None:
        return []
    edits: list[dict[str, Any]] = []
    for el in data:
        if not isinstance(el, dict):
            continue
        ri = _coerce_int(el.get("region_index"))
        if ri is None:
            continue
        before = el.get("before")
        after = el.get("after")
        if not isinstance(before, str) or not isinstance(after, str):
            continue
        if not before:
            continue
        edits.append({"region_index": ri, "before": before, "after": after})
    return edits


__all__ = [
    "build_ocr_repair_request",
    "parse_ocr_repair_response",
]
