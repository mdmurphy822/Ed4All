"""Wave 16: formula detection + minimal-accessible MathML emission.

Raw pdftotext produces no MathML. The Wave 12-15 pipeline mapped every
formula-ish block to the ``FORMULA_MATH`` role and rendered it via the
``<math><annotation>`` placeholder template. Wave 16 finishes the
minimum-viable path: detect plausible formulas in prose and emit
proper ``<math>`` / ``<semantics>`` / ``<annotation>`` so screen
readers narrate the original source while HTML remains valid.

Scope
-----

* LaTeX delimiter patterns: ``$...$``, ``\\(...\\)``, ``\\[...\\]``
* Equation-on-a-line pattern: a standalone line with ``=`` and
  reasonable alphanumeric / operator mix (e.g. ``E = mc^2``)
* Empty / whitespace input returns an empty detection list

Non-goals
---------

* Full LaTeX-to-MathML compilation (would require latexml / mathjax
  / pandoc and a heavy dependency). The ``<annotation
  encoding="text/plain">`` fallback preserves assistive-tech access
  to the raw source — that is the accessibility floor we commit to.
* Inline math detection inside prose paragraphs. This wave keeps
  detection block-level; inline math stays as prose until a future
  wave can wire per-token rewriting.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# ``$...$`` inline LaTeX (non-greedy, disallows a stray ``$`` next to digits
# so ``$5.00`` currency strings don't match). The pattern requires at least
# one non-$ character on each side so we never match ``$$`` alone.
_DOLLAR_LATEX = re.compile(r"\$(?!\$)([^$\n]+?)\$(?!\d)")

# ``\(...\)`` inline LaTeX.
_PAREN_LATEX = re.compile(r"\\\((.+?)\\\)", re.DOTALL)

# ``\[...\]`` display LaTeX.
_BRACKET_LATEX = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

# Plain-text equation candidate: standalone-line ``LHS = RHS`` where both
# sides have a reasonable alphanumeric / operator mix. The guardrail on
# acceptable chars keeps ordinary prose (``We agreed = good``) from matching.
_PLAIN_EQUATION = re.compile(
    r"^\s*"
    r"([A-Za-z0-9_^(){}\[\]\s+\-*/.,:]+?)"  # LHS
    r"\s*=\s*"
    r"([A-Za-z0-9_^(){}\[\]\s+\-*/.,:]+?)"  # RHS
    r"\s*$"
)


@dataclass
class DetectedFormula:
    """A span of detected formula source + the surrounding raw text.

    ``raw`` is the original text snippet before stripping delimiters
    (used for provenance / debugging). ``body`` is the delimiter-free
    formula source the template emits as the plain-text annotation.
    ``delimiter`` names which pattern matched: ``dollar``, ``paren``,
    ``bracket``, or ``plain``.
    """

    raw: str
    body: str
    delimiter: str


def detect_formulas(text: str) -> List[DetectedFormula]:
    """Return every detected formula span in ``text``.

    Returns an empty list when ``text`` is empty, whitespace-only, or
    contains no detectable formulas. Callers treat the empty list as
    "this block is not a formula" and emit the prose template instead.
    """
    if not text or not text.strip():
        return []

    found: List[DetectedFormula] = []

    for match in _BRACKET_LATEX.finditer(text):
        body = match.group(1).strip()
        if body:
            found.append(DetectedFormula(raw=match.group(0), body=body, delimiter="bracket"))

    for match in _PAREN_LATEX.finditer(text):
        body = match.group(1).strip()
        if body:
            found.append(DetectedFormula(raw=match.group(0), body=body, delimiter="paren"))

    for match in _DOLLAR_LATEX.finditer(text):
        body = match.group(1).strip()
        if body:
            found.append(DetectedFormula(raw=match.group(0), body=body, delimiter="dollar"))

    # Only try the plain-equation fallback when no LaTeX-style delimiter
    # hit. An ``E = mc^2`` line inside a doc otherwise full of ``$...$``
    # should already be caught upstream; we don't want to double-detect.
    if not found:
        stripped = text.strip()
        # Single-line equations only — multi-line prose is never a formula.
        if "\n" not in stripped:
            m = _PLAIN_EQUATION.match(stripped)
            if m and _looks_equation_like(stripped):
                found.append(
                    DetectedFormula(raw=stripped, body=stripped, delimiter="plain")
                )

    return found


def _looks_equation_like(text: str) -> bool:
    """Guardrail against false positives for the plain-equation pattern.

    A plausible equation has at least one non-word symbol on each side
    of ``=`` and is not just an English sentence with an ``=``. We
    approximate "looks symbolic" by requiring the text to contain at
    least one operator character beyond the ``=`` itself.
    """
    # Strip the equal sign, then check if the remaining text contains
    # operator characters or superscript / subscript markers.
    operators = set("+-*/^_(){}[]")
    remaining = text.replace("=", "", 1)
    has_operator = any(ch in operators for ch in remaining)
    if has_operator:
        return True
    # Allow single-letter identifiers on both sides (e.g. ``y = x``)
    # when the tokens are short and look symbolic (<=3 chars each).
    lhs, _, rhs = text.partition("=")
    return 0 < len(lhs.strip()) <= 3 and 0 < len(rhs.strip()) <= 3


# ---------------------------------------------------------------------------
# MathML rendering
# ---------------------------------------------------------------------------


def render_mathml(
    body: str,
    *,
    fallback: Optional[str] = None,
    display: str = "block",
) -> str:
    """Render a minimal-accessible MathML element.

    Shape::

        <math xmlns="..." display="{display}">
          <semantics>
            <mtext>{body}</mtext>
            <annotation encoding="text/plain">{fallback}</annotation>
          </semantics>
        </math>

    ``<mtext>`` carries the raw source in a way that passes MathML
    validation (we don't claim to have real expression structure),
    and ``<annotation>`` duplicates the source so assistive tech with
    a plain-text MathML reader still narrates the formula verbatim.
    """
    escaped_body = html.escape(body or "", quote=False)
    fallback_text = fallback if fallback is not None else (body or "")
    escaped_fallback = html.escape(fallback_text, quote=False)

    return (
        f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{html.escape(display, quote=True)}">'
        f"<semantics>"
        f"<mtext>{escaped_body}</mtext>"
        f'<annotation encoding="text/plain">{escaped_fallback}</annotation>'
        f"</semantics>"
        f"</math>"
    )


def render_block_mathml(text: str) -> Optional[str]:
    """Render the first detected formula in ``text`` as MathML, or ``None``.

    Used by the ``FORMULA_MATH`` template when ``attributes`` does not
    already carry a pre-extracted formula. Returns ``None`` when no
    formula could be detected so the caller can fall back to prose.
    """
    detected = detect_formulas(text)
    if not detected:
        return None
    first = detected[0]
    return render_mathml(first.body, fallback=first.body)


__all__ = [
    "DetectedFormula",
    "detect_formulas",
    "render_block_mathml",
    "render_mathml",
]
