"""Deterministic math / text normalizer for the GLM-OCR lane (D2).

NO LLM, pure string transforms. The GLM-OCR SDK emits per-glyph OCR tokens, so
math inside ``$...$`` arrives digit-split (``\\sqrt {5 0}``, ``4 2 m n``,
``\\frac{1 1}{2 8}``), ``\\mathrm{}``/``\\text{}`` arguments arrive
letter-spaced (``\\mathrm {S i m p l i f y}``), and a prose section ordinal
splits around its dot (``1. 2 Use the Language of Algebra``). This module
repairs those deterministically as a FINAL pass over the transform's emitted
``region_provenance`` (see ``transform.transform_document``).

Cross-venv-clean posture (mirrors ``region_map`` / ``stop_seam``): NO Ed4All
``lib/`` imports — the ``glmocr`` package must run inside the bare SemantiK
venv behind the JSON bridge.

The gate ``SEMANTIK_GLMOCR_MATH_NORMALIZE`` is DEFAULT-ON *within the lane*
(``resolve_math_normalize_mode``): the falsey-set parse mirrors
``page_arranger.resolve_arranger_heading_sanity`` — only the explicit falsey
tokens (``0``/``false``/``no``/``off``) disable it; unset / blank / truthy /
garbage → on. Because the lane itself (``SEMANTIK_GLMOCR_LANE``) is opt-in and
default OFF, default-on HERE changes NO global default behaviour (the
``SEMANTIK_ARRANGER_HEADING_SANITY`` precedent — a correctness pass that only
runs inside an already-opt-in path). The explicit ``=0`` is the byte-identical
revert lever.

All transforms are conservative and idempotent:
  * digits are joined only across a SINGLE space and only digit-to-digit
    (never a digit-to-letter join, never across an operator);
  * letter-spacing is collapsed only inside ``\\mathrm{}`` / ``\\text{}`` /
    ``\\operatorname{}`` arguments AND only for a run of LONE single letters
    (so ``\\text{the answer}`` is never mangled to ``theanswer``);
  * an ordinal is joined only on the exact ``digit(s) . SPACE digit(s)
    WHITESPACE CAPITAL`` section-number shape.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "resolve_math_normalize_mode",
    "normalize_math_text",
    "fix_spaced_ordinals",
]

_FALSEY = {"0", "false", "no", "off"}


def resolve_math_normalize_mode() -> bool:
    """Is the deterministic math/text normalizer active? Default ON in-lane.

    Falsey-set parse (mirrors ``resolve_arranger_heading_sanity``): only the
    explicit falsey tokens disable it; unset / blank / truthy / garbage → on.
    Read at CALL time so tests can flip it without a re-import.
    """
    return (os.environ.get("SEMANTIK_GLMOCR_MATH_NORMALIZE") or "").strip().lower() not in _FALSEY


# ── Digit / letter collapse primitives. ─────────────────────────────────────
_DIGIT_PAIR_RE = re.compile(r"(\d) (\d)")
# A maximal run of LONE single letters each separated by a single space, bounded
# by non-letters ("S i m p l i f y" — but NOT "the answer" / "a b" inside a real
# phrase, since those tokens are multi-letter or the run is bounded by letters).
_LETTER_SPACED_RE = re.compile(r"(?<![A-Za-z])([A-Za-z](?: [A-Za-z])+)(?![A-Za-z])")
# The three text-mode LaTeX commands whose brace argument may hold OCR
# letter-spacing (``\mathrm {S i m p l i f y}`` — note the optional space before
# the brace). ``[^{}]*`` keeps this to a single, non-nested brace group.
_TEXT_CMD_RE = re.compile(r"(\\(?:mathrm|text|operatorname)\s*)\{([^{}]*)\}")
# Math delimiters, longest-first so ``$$`` wins over ``$``.
_MATH_SPAN_RE = re.compile(
    r"\$\$(.+?)\$\$"       # $$ ... $$
    r"|\$(.+?)\$"          # $ ... $
    r"|\\\((.+?)\\\)"      # \( ... \)
    r"|\\\[(.+?)\\\]",     # \[ ... \]
    re.DOTALL,
)
_SPAN_DELIMS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))


def _collapse_digits(s: str) -> str:
    """Join single-space-separated lone digits to a fixpoint ("1 2 3" → "123").

    Only a SINGLE space between two digits collapses, so an operator gap
    ("1 + 0") or a double space ("5  0") is never joined, and a digit is never
    joined to a letter ("42 m n").
    """
    prev = None
    while prev != s:
        prev = s
        s = _DIGIT_PAIR_RE.sub(r"\1\2", s)
    return s


def _collapse_letter_spacing(s: str) -> str:
    """Collapse a run of lone single letters ("S i m p l i f y" → "Simplify")."""
    return _LETTER_SPACED_RE.sub(lambda m: m.group(1).replace(" ", ""), s)


def _text_cmd_repl(m: "re.Match[str]") -> str:
    inner = _collapse_letter_spacing(m.group(2))
    inner = _collapse_digits(inner)
    return f"{m.group(1)}{{{inner}}}"


def _math_span_repl(m: "re.Match[str]") -> str:
    for gi, (open_d, close_d) in enumerate(_SPAN_DELIMS, start=1):
        inner = m.group(gi)
        if inner is not None:
            return f"{open_d}{_collapse_digits(inner)}{close_d}"
    return m.group(0)


def normalize_math_text(text: str) -> str:
    """Repair per-glyph OCR spacing inside math delimiters + LaTeX text-command
    arguments. Text OUTSIDE math delimiters / those command args is untouched.

    (a) inside ``$...$`` / ``$$...$$`` / ``\\(...\\)`` / ``\\[...\\]`` and inside
        ``\\mathrm{}`` / ``\\text{}`` / ``\\operatorname{}`` arguments, join
        single-space-separated lone digits ("5 0" → "50", to a fixpoint);
    (b) inside ``\\mathrm{}`` / ``\\text{}`` / ``\\operatorname{}`` arguments,
        collapse lone-letter spacing ("S i m p l i f y" → "Simplify").

    Idempotent; a document with no math is returned unchanged.
    """
    if not text:
        return text
    # Text-command args first (covers digits + letters, in OR out of a math
    # span); then a math-span digit collapse over the whole span content (which
    # re-covers command args inside a span — harmless, idempotent).
    text = _TEXT_CMD_RE.sub(_text_cmd_repl, text)
    text = _MATH_SPAN_RE.sub(_math_span_repl, text)
    return text


# ── Spaced-ordinal repair. ──────────────────────────────────────────────────
# "1. 2 Use the Language" → "1.2 Use the Language": digit(s) . SPACE digit(s)
# then whitespace then a Capital. The trailing whitespace+Capital is kept in
# group 3 so the boundary is preserved. Requiring the SECOND bare digit run is
# what leaves a numbered list item ("1. Simplify the expression") untouched, and
# requiring a SPACE after the dot leaves an already-joined decimal ("1.2 X")
# untouched.
_SPACED_ORDINAL_RE = re.compile(r"(\d+)\. +(\d+)(\s+[A-Z])")


def fix_spaced_ordinals(text: str) -> str:
    """Rejoin a section-number ordinal split around its dot. Conservative — see
    ``_SPACED_ORDINAL_RE``. Applies to prose paragraph AND heading text."""
    if not text:
        return text
    return _SPACED_ORDINAL_RE.sub(r"\1.\2\3", text)
