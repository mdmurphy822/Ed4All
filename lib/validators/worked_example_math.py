"""Symbolic worked-example verification gate — the correctness control for the
MATH in worked examples that provenance / NLI / numeric-grounding gates cannot
provide.

Motivation (real defects, hand-found in a 7B-authored vendor-HTML build)
------------------------------------------------------------------------
The number-blind NLI gate (``block_prose_entailment``) and the fabrication
control (``numeric_literal_grounding``) both check whether a block's *numbers*
appear in the cited source — they say NOTHING about whether the block's *math*
is internally correct. Two classes of defect slipped through both:

* A substitution worked example that isolates ``x - 2y = -2`` as ``x = 2y + 2``
  (a SIGN error — it should be ``x = 2y - 2``), derives ``y = 3.5``, then asserts
  the solution ``(8, 5)``. The asserted pair does not satisfy the second system
  equation.
* A sibling block concluding ``(8, 3.5)`` — a pair that satisfies NEITHER
  equation of the system.

Neither the numeric-grounding gate (every literal ``8``/``5``/``3.5`` DOES appear
somewhere in source) nor NLI (number-blind) can catch these: the numbers are
individually grounded, but the *derivation* is wrong. Only a symbolic re-check
under a computer algebra system separates provably-wrong math from
novel-but-correct math.

This gate is also the named RE-PROMOTION condition for the
``numeric_literal_grounding`` gate — see commit ``329aa9d3`` and the
``# TODO(calibration)`` marker at that gate in ``config/workflows.yaml``:
``numeric_literal_grounding`` was recalibrated critical->warning because novel
worked examples inherently mint practice numbers absent from source, and it can
be re-promoted once a symbolic step-verification gate (this one) can separate
wrong math from novel-but-correct math.

The high-precision detectable patterns (precision over recall)
--------------------------------------------------------------
For each ``example`` / ``worked_example`` block (and ``concept`` / ``explanation``
blocks that embed a worked-example structure), the validator harvests
well-DELIMITED math fragments — LaTeX inline ``\\(...\\)`` / display ``\\[...\\]``,
``<code>...</code>`` inner text, ``class="solution-line"`` content, and
``\\boxed{...}`` — and applies three checks:

* (a) **single-variable solve**: a stated final solution ``v = c`` (from a
  solution context) is substituted into every non-trivial single-variable
  equation of the same variable in the block; a definite numeric inequality is a
  ``WORKED_EXAMPLE_MATH_WRONG`` flag.
* (b) **two-equation systems**: a stated ordered-pair solution ``(a, b)`` (from a
  solution context) is substituted into BOTH equations of a genuine two-variable
  system (>=2 distinct, non-equivalent equations in the same variable pair); a
  pair that fails either equation is flagged.
* (c) **numeric simplify/evaluate chains**: an ``expr1 = expr2 = expr3`` chain
  whose sides are ALL variable-free is verified for pairwise equality; a definite
  (beyond rounding tolerance) mismatch is flagged.
* (b') **plus-minus (±) solution sets**: a claimed ``x = a ± b`` (or the full
  quadratic-formula shape ``x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}``) is expanded
  into its two roots ``a+b`` / ``a-b``; it verifies iff BOTH roots satisfy the
  same block equation for ``x``. Radicals (``\\sqrt{...}`` / ``\\sqrt[n]{...}`` /
  unicode ``√``) are translated balanced-brace-aware by ``_translate_radicals``
  (or failed closed), so radical equations — previously skipped wholesale — now
  flow to the checker.
* (d) **quadratic features (vertex / discriminant)**: for a block carrying a
  single-variable quadratic ``a·v² + b·v + c``, a claimed ``vertex (h, k)`` is
  re-derived as ``(-b/2a, c - b²/4a)`` and a claimed ``discriminant = N`` as
  ``b² - 4ac``; a mismatch is flagged (deterministic sympy re-derivation).

Anything unparseable is SKIPPED SILENTLY (counted in ``skipped_unparseable``,
never an issue). An issue MUST mean the math is provably wrong under sympy — the
message carries the parsed equation + the claimed-vs-computed result. Ordered
pairs / single-variable solutions are only read from a SOLUTION CONTEXT so the
gate never fires on the ubiquitous points-on-a-line tables that graphing blocks
emit (a point on line 1 does not satisfy line 2 — but it is not a system
solution, so it must not be flagged).

Graceful degrade
----------------
sympy is the only dependency. When it is absent the validator emits a single
warning-severity ``SYMPY_MISSING`` GateIssue with ``passed=True`` (mirrors the
``EMBEDDING_DEPS_MISSING`` convention of the statistical-tier validators). sympy
is declared in the ``[embedding]`` optional extra (NOT the default install) —
see ``pyproject.toml`` and ``docs/validation/validators.md``.

Shadow knob (mirrors ``numeric_literal_grounding`` / ``block_prose_entailment``)
-------------------------------------------------------------------------------
``inputs["shadow"]`` (merged from the gate ``config.shadow``) forces every
emitted ``WORKED_EXAMPLE_MATH_WRONG`` issue to warning-severity. Day-1 the gate
is warning-severity in ``config/workflows.yaml`` regardless; the shadow knob is
kept for parity with the sibling numeric gates and the deferred critical flip.

Decision-capture
----------------
One ``worked_example_math_check`` event per SCORED block (a block that yielded
>=1 parseable equation/chain); rationale interpolates dynamic per-call signals
(>=20 chars): block_id, block_type, the checks that ran, equations parsed,
fragments skipped, and the flagged claim. Skip-silent blocks (no parseable math)
emit nothing (matches ``numeric_literal_grounding`` / ``block_prose_entailment``
skip discipline).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Block lives at Courseforge/scripts/blocks.py — same import bridge as the
# sibling block validators (numeric_literal_grounding / block_prose_entailment).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — Block always present in this repo
    from blocks import Block  # type: ignore[import-not-found]  # noqa: E402
except Exception:  # noqa: BLE001
    Block = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# sympy graceful-degrade guard (mirrors EMBEDDING_DEPS_MISSING convention)
# --------------------------------------------------------------------------- #
try:
    import sympy  # type: ignore
    from sympy.parsing.sympy_parser import (  # type: ignore
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _SYMPY_TRANSFORMS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import failure degrades gracefully
    sympy = None  # type: ignore
    parse_expr = None  # type: ignore
    _SYMPY_TRANSFORMS = ()  # type: ignore
    _SYMPY_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Canonical GateIssue codes + decision type
# --------------------------------------------------------------------------- #

_CODE_MATH_WRONG: str = "WORKED_EXAMPLE_MATH_WRONG"
_CODE_SYMPY_MISSING: str = "SYMPY_MISSING"
_DECISION_TYPE: str = "worked_example_math_check"

#: Cap per-block issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Block types that carry a worked-example structure worth symbol-checking.
#: ``example`` / ``worked_example`` are the primary targets; ``concept`` /
#: ``explanation`` frequently embed a "Example: Solve ..." worked block.
_TARGET_BLOCK_TYPES: frozenset = frozenset(
    {"example", "worked_example", "concept", "explanation"}
)

#: Relative + absolute tolerance for the numeric simplify-chain check (pattern
#: c) so a legitimately ROUNDED decimal ("0.166 = 0.17") is not flagged. The
#: real defects are exact/large errors (5 vs 6), so a 1% relative floor keeps
#: precision while tolerating display rounding.
_NUMERIC_REL_TOL: float = 0.01
_NUMERIC_ABS_TOL: float = 1e-9

#: Inequality / relation markers — a fragment carrying one of these is NOT a
#: plain equality and is skipped for the ``=`` split (never verifiable as an
#: equation).
_INEQUALITY_MARKERS: Tuple[str, ...] = (
    r"\le", r"\ge", r"\ne", r"\neq", r"\leq", r"\geq", r"\approx",
    "≈", "≤", "≥", "≠", "<", ">",
)


# --------------------------------------------------------------------------- #
# Math-fragment extraction + LaTeX -> sympy translation
# --------------------------------------------------------------------------- #

#: Well-delimited math fragments. LaTeX inline / display are the primary
#: carriers; <code> inner text + solution-line content catch the plain
#: caret/ascii style the corpus also emits.
_RE_LATEX_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_RE_LATEX_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_RE_CODE = re.compile(r"<code[^>]*>(.+?)</code>", re.DOTALL | re.IGNORECASE)
#: A ``class="solution-line"`` element's inner content (approximate: to the next
#: closing div/span). Used both as a fragment source AND to mark its char span a
#: solution context.
_RE_SOLUTION_LINE = re.compile(
    r'class="[^"]*solution-line[^"]*"[^>]*>(.*?)</(?:div|span|p)>',
    re.DOTALL | re.IGNORECASE,
)
_RE_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_RE_TAG = re.compile(r"<[^>]+>")

#: Solution-context keywords — a fragment whose preceding window contains one of
#: these is treated as a stated-solution context (pattern a/b harvest).
#: ``intersect at`` (fix e) makes the stated intersection point of a linear
#: system ("the lines intersect at (3, 2)") a checkable claimed solution.
_SOLUTION_KEYWORDS: Tuple[str, ...] = (
    "solution", "therefore", "the answer", "answer is", "answer:", "thus",
    "we get", "we find", "yields", "so ", "hence", "intersect at",
)
_CTX_WINDOW: int = 120

#: Premise markers (fix b) — a ``var = value`` equation whose TIGHT preceding
#: window carries one of these is a PREMISE (an assumption / substitution input),
#: NOT a claimed solution to verify. Demotes a keyword-derived solution context
#: (``let x = 5`` under a nearby "answer"/"so" keyword is still a premise). A
#: tight window keeps this from over-suppressing a genuine solution keyword far
#: upstream. A fragment INSIDE a solution-line / \boxed span is NOT demoted (that
#: is an unambiguous stated-solution carrier).
#: Word-boundaried so a genuine word never triggers on a coincidental substring
#: (``offset`` must not read as ``set``, ``wallet`` must not read as ``let``).
#: Covers the assumption / substitution-input phrasings (``let`` / ``when`` /
#: ``suppose`` / ``substitute``) AND the graphing-premise phrasings (family 4 FP
#: fix): ``set x = 0`` (the SET-x-to-0 y-intercept premise) and ``choose`` /
#: ``pick`` an arbitrary graphing point (``choose x = 6``). A value introduced by
#: one of these is a substitution INPUT / chosen coordinate, not a claimed
#: solution to verify.
_RE_PREMISE = re.compile(
    r"\b(?:let|when|suppose|set|choose|choosing|chosen|pick|picking"
    r"|substitut\w*)\b",
    re.IGNORECASE,
)
_PREMISE_WINDOW: int = 40


def _premise_before(html: str, pos: int) -> bool:
    """True when a premise marker sits in the tight window before ``pos``."""
    window = _RE_TAG.sub(" ", html[max(0, pos - _PREMISE_WINDOW):pos])
    return bool(_RE_PREMISE.search(window))


def _find_brace_group(text: str, open_idx: int) -> Optional[Tuple[str, int]]:
    """``text[open_idx]`` is ``{``; return ``(inner, idx_after_close)`` or None.

    Balanced-brace aware (nested ``{}`` handled). None when the group never
    closes (unbalanced) — the caller fails that fragment closed.
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    j = open_idx
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j], j + 1
        j += 1
    return None


def _find_paren_group(text: str, open_idx: int) -> Optional[Tuple[str, int]]:
    """``text[open_idx]`` is ``(``; return ``(inner, idx_after_close)`` or None."""
    if open_idx >= len(text) or text[open_idx] != "(":
        return None
    depth = 0
    j = open_idx
    while j < len(text):
        ch = text[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j], j + 1
        j += 1
    return None


_RE_SQRT_HEAD = re.compile(r"\\sqrt\s*(?:\[([^\]]*)\])?\s*")


def _translate_radicals(s: str) -> str:
    """Translate ``\\sqrt`` / unicode ``√`` radicals to a sympy-parseable form.

    * ``\\sqrt{body}`` -> ``sqrt((body))`` (balanced-brace aware, so a ``\\frac``
      or nested ``\\sqrt`` inside the radicand keeps its grouping).
    * ``\\sqrt[n]{body}`` -> ``((body)**(1/(n)))`` (principal nth root; sympy
      auto-simplifies e.g. ``27**(1/3) -> 3``).
    * ``√{body}`` / ``√(body)`` / ``√<number-or-token>`` -> ``sqrt(...)``.

    FAIL-CLOSED: a radical that cannot be confidently resolved (a ``\\sqrt`` not
    followed by a brace group, or an unbalanced radicand) is replaced with an
    illegal ``@`` sentinel so the downstream ``parse_expr`` raises and the
    fragment is SKIPPED (never mis-parsed into a false finding). Without this a
    bare ``\\sqrt 27`` would lose its ``\\sqrt`` to the generic command-strip and
    silently mis-verify as ``27``.
    """
    # \sqrt / \sqrt[n] forms (leftmost-outermost; nested radicals resolve on
    # subsequent iterations because the balanced group captures them whole).
    for _ in range(12):
        m = _RE_SQRT_HEAD.search(s)
        if not m:
            break
        start, idx, nth = m.start(), m.end(), m.group(1)
        grp = _find_brace_group(s, idx) if idx < len(s) and s[idx] == "{" else None
        if grp is None:
            # No resolvable radicand -> poison (parse fails -> skip); len('\\sqrt')==5.
            s = s[:start] + "@" + s[start + 5:]
            continue
        body, end = grp
        if nth is not None and nth.strip():
            repl = f"(({body})**(1/({nth})))"
        else:
            repl = f"sqrt(({body}))"
        s = s[:start] + repl + s[end:]
    # Unicode √ forms.
    for _ in range(12):
        k = s.find("√")
        if k == -1:
            break
        after = k + 1
        while after < len(s) and s[after] == " ":
            after += 1
        if after < len(s) and s[after] == "{":
            grp = _find_brace_group(s, after)
            if grp is None:
                s = s[:k] + "@" + s[k + 1:]
                continue
            body, end = grp
            s = s[:k] + f"sqrt(({body}))" + s[end:]
        elif after < len(s) and s[after] == "(":
            grp = _find_paren_group(s, after)
            if grp is None:
                s = s[:k] + "@" + s[k + 1:]
                continue
            body, end = grp
            s = s[:k] + f"sqrt(({body}))" + s[end:]
        else:
            tok = re.match(r"\d+(?:\.\d+)?|[A-Za-z]\w*", s[after:])
            if tok:
                s = s[:k] + f"sqrt({tok.group(0)})" + s[after + tok.end():]
            else:
                s = s[:k] + "@" + s[k + 1:]  # nothing to root -> poison
    return s


def _clean_math(s: str) -> str:
    """Translate a restricted-LaTeX / ascii math string into a parseable form.

    Deterministic (no antlr): unwraps ``\\boxed{}``, translates ``\\sqrt`` /
    unicode ``√`` radicals (``_translate_radicals``), rewrites ``\\frac{a}{b}`` ->
    ``((a)/(b))`` (recursively for nested fracs), maps ``\\cdot`` / ``·`` /
    ``\\times`` / ``×`` -> ``*`` and ``\\div`` / ``÷`` -> ``/``, folds unicode
    minus, drops any remaining ``\\command`` tokens and ``\\,`` spacing, strips
    ``$`` currency + thousands-separator commas, and normalises braces to parens.
    The cleaned string is fed to sympy ``parse_expr`` with
    implicit-multiplication so ``3x`` / ``2(y-1)`` / ``5 · (1/3) · 3`` parse.
    """
    s = s.strip()
    # Unwrap \boxed{...} (possibly nested) -> its content.
    for _ in range(4):
        new = _RE_BOXED.sub(r"\1", s)
        if new == s:
            break
        s = new
    # Radicals BEFORE fracs so a \frac{\sqrt{a}}{b} radicand keeps its grouping
    # (once \sqrt -> sqrt((a)) the frac numerator is brace-free and expands).
    s = _translate_radicals(s)
    # \frac{a}{b} -> ((a)/(b)), innermost-first for nesting.
    for _ in range(6):
        new = re.sub(
            r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s
        )
        if new == s:
            break
        s = new
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*").replace("·", "*")
    s = s.replace(r"\times", "*").replace("×", "*")
    s = s.replace(r"\div", "/").replace("÷", "/")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace("‐", "-")
    s = s.replace(r"\%", "").replace("%", "")
    # Drop remaining LaTeX \command tokens (\text, \mathrm, \displaystyle, ...).
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = _RE_TAG.sub(" ", s)  # any stray inner tags (from <code> content)
    s = s.replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators 17,852
    s = s.replace("{", "(").replace("}", ")")
    return s.strip()


def _parse_side(side: str) -> Optional[Any]:
    """Parse one side of an equation to a sympy expression, or None.

    Returns None on any parse failure (unparseable -> skip silently) or when the
    cleaned string is empty / contains no math token.
    """
    if not _SYMPY_AVAILABLE:
        return None
    cleaned = _clean_math(side)
    if not cleaned or not re.search(r"[0-9a-zA-Z]", cleaned):
        return None
    # Guard obvious non-expressions: trailing operators, bare punctuation.
    if not re.search(r"[0-9a-zA-Z)]", cleaned):
        return None
    try:
        expr = parse_expr(  # type: ignore[misc]
            cleaned, transformations=_SYMPY_TRANSFORMS, evaluate=True
        )
    except Exception:  # noqa: BLE001 — unparseable -> skip
        return None
    # Only accept genuine scalar sympy expressions. A fragment like ``(8, 5)``
    # parses to a Python tuple / sympy Tuple, a relational parses to a
    # Boolean — neither is a verifiable equation side, so skip.
    if not isinstance(expr, sympy.Expr):  # type: ignore[union-attr]
        return None
    if not hasattr(expr, "free_symbols"):
        return None
    # Reject non-real / matrix / tuple-bearing heads we can't compare.
    try:
        if expr.is_number and expr.is_real is False:
            return None
    except Exception:  # noqa: BLE001
        pass
    return expr


#: Constructs the deterministic LaTeX->sympy translator handles UNRELIABLY, so a
#: fragment carrying one is skipped (precision over recall). Radicals (``\sqrt``
#: / ``√``) are NO LONGER here — ``_translate_radicals`` resolves them
#: balanced-brace-aware (or fails closed), so a radical equation flows to the
#: checker. ``\pm`` / ``±`` (and ``\mp`` / ``∓``) STAY skipped for the single-
#: equation path — a ``\pm`` fragment is TWO-valued (``a+b`` and ``a-b``), not one
#: equation; it is harvested + verified separately by ``_extract_pm_solutions`` /
#: ``_check_pm_solutions``. Matrices, sums/integrals/products, limits/derivatives
#: and other genuinely-unreliable constructs remain skipped.
_UNRELIABLE_RE = re.compile(
    r"\\pm(?![a-zA-Z])|±|\\mp(?![a-zA-Z])|∓|"
    r"\\sum|\\int|\\iint|\\oint|\\prod|\\lim|\\partial|∂|\\infty|∞|\\nabla|"
    r"\\begin\b|\\matrix|\\pmatrix|\\bmatrix|\\vmatrix|\\det\b"
)


def _split_equation_sides(fragment: str) -> Optional[List[str]]:
    """Split a fragment on top-level ``=`` into >=2 sides, or None.

    Skips fragments carrying an inequality / relation marker (not a plain
    equality), an unreliable-to-translate construct (radicals / ``\\pm`` /
    unicode ``√`` — see ``_UNRELIABLE_RE``), or a ``\\frac`` that survives full
    expansion (a nested-brace fraction the translator cannot group). Splits on a
    bare ``=`` not part of ``==`` / ``<=`` / ``>=`` / ``!=``. Returns None when
    there is no ``=``.
    """
    low = fragment
    for marker in _INEQUALITY_MARKERS:
        if marker in low:
            return None
    if _UNRELIABLE_RE.search(fragment):
        return None
    if "=" not in fragment:
        return None
    # A \frac that does not fully expand (nested-brace numerator/denominator) is
    # mis-grouped by the brace->paren fallback → skip. Translate radicals first so
    # a \frac{\sqrt{a}}{b} (radical numerator) still expands (matches _clean_math).
    expanded = _translate_radicals(fragment)
    for _ in range(6):
        nxt = re.sub(
            r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1/\2)", expanded
        )
        if nxt == expanded:
            break
        expanded = nxt
    if "frac" in expanded:
        return None
    sides = re.split(r"(?<![<>=!])=(?![=])", fragment)
    sides = [s for s in sides if s.strip()]
    if len(sides) < 2:
        return None
    return sides


#: Implication markers (fix c). A fragment carrying a derivation-chain arrow
#: (``2x + 3 = 11 \implies x = 4``) is TWO equations, not one; splitting on the
#: arrow parses each independently (without the split the arrow is dropped as a
#: stray ``\command`` and the two equalities collapse into a garbage chain that
#: mis-evaluates). Covers LaTeX ``\implies`` / ``\Rightarrow`` / ``\Longrightarrow``
#: and their unicode glyphs.
_RE_IMPLICATION = re.compile(
    r"\\implies|\\Longrightarrow|\\Rightarrow|⟹|⇒"
)


def _split_on_implication(fragment: str) -> List[str]:
    """Split a fragment on implication arrows into independent sub-fragments."""
    parts = _RE_IMPLICATION.split(fragment)
    return [p for p in parts if p.strip()]


#: Problem-restart markers (fix a). A block that presents several problems back-
#: to-back restarts at an ``Example`` / ``Self-Check`` / ``Try It`` heading or a
#: fresh ``Step 1``. Pooling candidate equations block-wide across those restarts
#: lets a solution pair from problem 1 be cross-checked against a system equation
#: from problem 2 (a false positive), and mis-attributes a genuine defect to the
#: wrong problem. Scoping the system check to one segment fixes both.
#:
#: Mechanism A (second precision pass): the original marker set only caught the
#: colon+digit forms (``Example N:`` / ``Step 1:``), so a block using a BARE
#: ``Example`` / ``Example 1`` heading (no colon), the ``Step 1.`` period form,
#: or a ``Self-Check`` / ``Try It`` heading collapsed into ONE segment and
#: cross-pooled. The bare-heading forms are anchored to a ``>`` tag-close so a
#: mid-prose "for example" never triggers a split (and the ``data-cf-content-type
#: ="example"`` ATTRIBUTE — whose ``example`` is not preceded by ``>`` — is never
#: matched); the colon/period ``Example N`` / ``Step 1`` forms keep the looser
#: ``\b`` word-boundary anchor (they are already unambiguous).
_RE_PROBLEM_RESTART = re.compile(
    r"(?i)(?:"
    r"\bexample\s+\d+\s*[:.]"           # "Example 1:" / "Example 2."
    r"|\bstep\s+1\s*[:.]"              # "Step 1:" / "Step 1."
    r"|>\s*example(?:\s+\d+)?\b"       # heading "<h3>Example</h3>" / "Example 2"
    r"|>\s*self[\s-]*check\b"          # heading "Self-Check" / "Self Check"
    r"|>\s*try\s+it\b"                 # heading "Try It"
    r")"
)


def _split_problem_segments(html: str) -> List[str]:
    """Split block HTML into problem segments at the ``_RE_PROBLEM_RESTART`` marks.

    Each restart marker (``Example`` / ``Self-Check`` / ``Try It`` heading, or a
    fresh ``Step 1:`` / ``Step 1.``) begins a new segment; any text before the
    first marker is its own (usually equation-free) leading segment. A block with
    no restart marker is one segment (byte-identical to the block-wide behavior).
    """
    starts = sorted({m.start() for m in _RE_PROBLEM_RESTART.finditer(html)})
    if not starts:
        return [html]
    segments: List[str] = []
    if starts[0] > 0:
        segments.append(html[:starts[0]])
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(html)
        segments.append(html[s:e])
    return segments


#: A ``\begin{cases}...\end{cases}`` system environment (family 3 FP fix). The
#: real worked system frequently lives in a ``cases`` block, which the standalone
#: fragment harvest SKIPS wholesale (``\begin`` is in ``_UNRELIABLE_RE``). That
#: left a claimed solution pool-able against an UNRELATED same-page equation (a
#: self-check / given line). ``_cases_equations`` parses the cases rows so the
#: pair is checked against ITS OWN system, and a cases block present in a segment
#: means the segment's standalone equations are treated as foreign to the system.
_RE_CASES = re.compile(
    r"\\begin\s*\{\s*cases\s*\}(.+?)\\end\s*\{\s*cases\s*\}",
    re.DOTALL | re.IGNORECASE,
)


def _cases_equations(html: str) -> List[Tuple[Any, Any, str, bool]]:
    """Parse every ``\\begin{cases}`` row into a ``(lhs, rhs, raw, ctx)`` equation.

    Rows split on the LaTeX ``\\\\`` line break; a leading/trailing ``&`` alignment
    marker is folded to whitespace. Rows that are not a plain two-side equality
    (or fail to parse) are skipped silently — precision over recall.
    """
    if not _SYMPY_AVAILABLE:
        return []
    out: List[Tuple[Any, Any, str, bool]] = []
    for cm in _RE_CASES.finditer(html):
        body = cm.group(1)
        for row in re.split(r"\\\\|\\newline\b", body):
            row = row.replace("&", " ").strip()
            if not row:
                continue
            sides = _split_equation_sides(row)
            if sides is None or len(sides) != 2:
                continue
            lhs = _parse_side(sides[0])
            rhs = _parse_side(sides[1])
            if lhs is None or rhs is None:
                continue
            out.append((lhs, rhs, row, False))
    return out


def _segment_system_findings(
    segment: str, block_text: str
) -> List[Tuple[str, str, str]]:
    """System-check one problem segment against ITS OWN equation set (family 3).

    When the segment carries one or more ``\\begin{cases}`` environments the real
    system lives THERE; the segment's standalone equations (self-checks,
    given/derived lines) are foreign to the system. Mechanism A: each solution
    pair is checked ONLY against the cases block that PRECEDES it (the nearest
    enclosing/preceding one), so an intro single-equation verification pair
    stated BEFORE any cases block — or a pair belonging to a different cases
    block earlier in the same segment — never pools with a foreign system. A pair
    with no preceding cases block is not a cases-system solution and is skipped;
    a cases block whose rows did not parse is skipped (never falls back to
    foreign standalone equations). Segments with no cases block use the standalone
    equations as before.
    """
    if _RE_CASES.search(segment):
        cases_blocks: List[Tuple[int, List[Tuple[Any, Any, str, bool]]]] = []
        for cm in _RE_CASES.finditer(segment):
            cases_blocks.append((cm.start(), _cases_equations(cm.group(0))))
        if not cases_blocks:
            return []
        findings: List[Tuple[str, str, str]] = []
        for pair, pos in _extract_solution_pairs_pos(segment):
            preceding = [(s, eqs) for s, eqs in cases_blocks if s <= pos]
            if not preceding:
                continue
            _start, eqs = max(preceding, key=lambda t: t[0])
            if not eqs:
                continue
            findings.extend(_check_system(eqs, [pair], block_text))
        return findings
    seg_pairs = _extract_solution_pairs(segment)
    seg_eqs, _seg_chains = _collect_equations(_iter_fragments(segment))
    return _check_system(seg_eqs, seg_pairs, block_text)


#: Negation-suppression markers (fix d). When the window just AFTER a flagged
#: equation says the candidate FAILS ("(False)" / "not a solution"), the block is
#: DELIBERATELY teaching a non-solution — suppress the finding (it is intentional,
#: not a defect).
#: (family 2 FP fix) also covers ``Which is false`` / ``is false`` and the
#: ``extraneous solution`` phrasing a taught non-solution demo uses to REJECT the
#: candidate it just substituted; the window is widened (60 -> 120) because the
#: rejection sentence often trails the flagged equation by a clause.
_RE_NEGATION_SUPPRESS = re.compile(
    r"\(\s*false\s*\)|not\s+a\s+solution|is\s+false|extraneous",
    re.IGNORECASE,
)
_SUPPRESS_WINDOW: int = 120


def _is_taught_failure(content: str, raw: str) -> bool:
    """True when a negation marker follows the flagged equation ``raw`` in ``content``.

    Scans every occurrence of the flagged fragment text and inspects the
    ``_SUPPRESS_WINDOW`` chars that follow it. Best-effort: if the fragment text
    cannot be located, no suppression (the finding stands — conservative).
    """
    if not raw:
        return False
    idx = content.find(raw)
    while idx != -1:
        tail = content[idx + len(raw): idx + len(raw) + _SUPPRESS_WINDOW]
        if _RE_NEGATION_SUPPRESS.search(tail):
            return True
        idx = content.find(raw, idx + 1)
    return False


def _iter_fragments(html: str) -> List[Tuple[str, bool]]:
    """Return ``[(fragment, in_solution_context), ...]`` for a block's HTML.

    Harvests only well-delimited math (LaTeX inline/display, <code>, solution-
    line content, \\boxed). ``in_solution_context`` is True when the fragment is
    inside a solution-line / \\boxed span OR its preceding ``_CTX_WINDOW`` chars
    (lower-cased, tag-stripped) contain a solution keyword.
    """
    # Solution-region char spans (solution-line elements + \boxed{...}).
    solution_spans: List[Tuple[int, int]] = []
    for m in _RE_SOLUTION_LINE.finditer(html):
        solution_spans.append((m.start(), m.end()))
    for m in _RE_BOXED.finditer(html):
        solution_spans.append((m.start(), m.end()))

    def _in_solution_span(pos: int) -> bool:
        return any(a <= pos <= b for a, b in solution_spans)

    def _ctx_keyword_before(pos: int) -> bool:
        window = html[max(0, pos - _CTX_WINDOW):pos]
        window = _RE_TAG.sub(" ", window).lower()
        return any(k in window for k in _SOLUTION_KEYWORDS)

    fragments: List[Tuple[str, bool]] = []
    seen_spans: set = set()
    for regex in (_RE_LATEX_INLINE, _RE_LATEX_DISPLAY, _RE_CODE, _RE_BOXED):
        for m in regex.finditer(html):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            frag = m.group(1)
            # A solution-line / \boxed span is an unambiguous stated-solution
            # carrier; otherwise a keyword-derived context is demoted to a
            # premise when a let/when/substituting marker sits immediately before
            # (fix b — a premise is a substitution input, not a claim to verify).
            in_span = _in_solution_span(m.start())
            ctx = in_span or (
                _ctx_keyword_before(m.start())
                and not _premise_before(html, m.start())
            )
            fragments.append((frag, ctx))
    return fragments


# --------------------------------------------------------------------------- #
# Numeric ordered-pair extraction (pattern b solution pairs)
# --------------------------------------------------------------------------- #

#: A numeric or simple-fraction coordinate.
_NUM = r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?"
_RE_PAIR = re.compile(
    r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)"
)


def _extract_solution_pairs_pos(
    html: str,
) -> List[Tuple[Tuple[Any, Any], int]]:
    """Return ``[((a, b), start_pos), ...]`` for solution-context ordered pairs.

    Only pairs whose preceding ``_CTX_WINDOW`` window carries a solution keyword
    (or that sit inside a solution-line / boxed span) are returned — this keeps
    the ubiquitous points-on-a-line graphing tables (``(0, -2)`` / ``(3, 0)``)
    out of the system check. Each coordinate is parsed to a sympy number. The
    match start position (Mechanism A) lets the caller scope a pair to the cases
    system that PRECEDES it (an intro verification pair stated before any cases
    block never pools with a later foreign system).
    """
    if not _SYMPY_AVAILABLE:
        return []
    solution_spans: List[Tuple[int, int]] = []
    for m in _RE_SOLUTION_LINE.finditer(html):
        solution_spans.append((m.start(), m.end()))
    for m in _RE_BOXED.finditer(html):
        solution_spans.append((m.start(), m.end()))

    def _ctx(pos: int) -> bool:
        if any(a <= pos <= b for a, b in solution_spans):
            return True
        window = _RE_TAG.sub(" ", html[max(0, pos - _CTX_WINDOW):pos]).lower()
        if not any(k in window for k in _SOLUTION_KEYWORDS):
            return False
        # Demote a keyword-derived context to a premise (fix b).
        return not _premise_before(html, pos)

    pairs: List[Tuple[Tuple[Any, Any], int]] = []
    for m in _RE_PAIR.finditer(html):
        if not _ctx(m.start()):
            continue
        av = _parse_side(m.group(1))
        bv = _parse_side(m.group(2))
        if av is None or bv is None:
            continue
        if av.free_symbols or bv.free_symbols:
            continue
        pairs.append(((av, bv), m.start()))
    return pairs


def _extract_solution_pairs(html: str) -> List[Tuple[Any, Any]]:
    """Numeric solution-context ordered pairs (positions dropped).

    Thin wrapper over ``_extract_solution_pairs_pos`` for callers that do not
    need the positional scoping (the standalone-equation system path).
    """
    return [pair for pair, _pos in _extract_solution_pairs_pos(html)]


# --------------------------------------------------------------------------- #
# sympy comparison helpers
# --------------------------------------------------------------------------- #


def _definitely_nonzero(diff: Any) -> bool:
    """True iff ``diff`` is a concrete number provably != 0 beyond tolerance.

    Symbolic residue (free symbols remain) -> False (can't tell -> skip).
    A tiny non-zero (rounding artifact within relative tolerance) -> False.
    """
    if diff is None:
        return False
    try:
        simplified = sympy.simplify(diff)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return False
    if getattr(simplified, "free_symbols", set()):
        return False
    if simplified == 0:
        return False
    try:
        val = float(simplified)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        # Non-real / non-evaluable difference — treat as unknown, skip.
        return False
    return abs(val) > _NUMERIC_ABS_TOL


def _relative_mismatch(a: Any, b: Any) -> bool:
    """True iff ``a`` and ``b`` are numeric and differ beyond rel+abs tolerance.

    Both must be variable-free numbers. Tolerates display rounding (0.166 vs
    0.17 is below the 1% floor only for near-equal values; large errors fire).
    """
    try:
        diff = sympy.simplify(a - b)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return False
    if getattr(diff, "free_symbols", set()) or diff == 0:
        return False
    try:
        dval = abs(float(diff))
        mag = max(abs(float(a)), abs(float(b)), 1.0)
    except Exception:  # noqa: BLE001
        return False
    if dval <= _NUMERIC_ABS_TOL:
        return False
    return (dval / mag) > _NUMERIC_REL_TOL


def _are_equivalent(e1: Any, e2: Any) -> bool:
    """True iff residuals ``e1``/``e2`` are scalar multiples (same line/eqn)."""
    try:
        if sympy.simplify(e1 - e2) == 0:  # type: ignore[union-attr]
            return True
        ratio = sympy.simplify(e1 / e2)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return False
    return (not getattr(ratio, "free_symbols", set())) and ratio != 0


# --------------------------------------------------------------------------- #
# Per-block checks (a) solve  (b) system  (c) numeric chain
# --------------------------------------------------------------------------- #


def _collect_equations(
    fragments: Sequence[Tuple[str, bool]]
) -> Tuple[List[Tuple[Any, Any, str, bool]], List[Tuple[List[Any], str, bool]]]:
    """Parse fragments into (single-``=`` equations, numeric chains).

    Returns ``(equations, chains)`` where an equation is
    ``(lhs_expr, rhs_expr, raw_fragment, in_ctx)`` and a chain is
    ``(side_exprs, raw_fragment, in_ctx)`` for fragments with >=3 sides. Fully
    unparseable fragments are dropped by the caller's skip counter.
    """
    equations: List[Tuple[Any, Any, str, bool]] = []
    chains: List[Tuple[List[Any], str, bool]] = []
    # Split each fragment on implication arrows first (fix c) so a derivation
    # chain like ``2x + 3 = 11 \implies x = 4`` is parsed as TWO equations.
    exploded = [
        (sub, ctx) for frag, ctx in fragments for sub in _split_on_implication(frag)
    ]
    for frag, ctx in exploded:
        sides = _split_equation_sides(frag)
        if sides is None:
            continue
        exprs = [_parse_side(s) for s in sides]
        if any(e is None for e in exprs):
            continue
        if len(exprs) == 2:
            equations.append((exprs[0], exprs[1], frag.strip(), ctx))
        else:
            chains.append((exprs, frag.strip(), ctx))
            # Also register adjacent pairs as equations for the solve/system
            # checks (a chain of transformations is a set of equations).
            for i in range(len(exprs) - 1):
                equations.append(
                    (exprs[i], exprs[i + 1], frag.strip(), ctx)
                )
    return equations, chains


def _is_trivial_solution_eq(lhs: Any, rhs: Any, var: str) -> bool:
    """True when the equation is just the solution restatement ``var = number``."""
    for a, b in ((lhs, rhs), (rhs, lhs)):
        if getattr(a, "is_Symbol", False) and str(a) == var:
            if not getattr(b, "free_symbols", set()):
                return True
    return False


def _stated_solutions(
    equations: Sequence[Tuple[Any, Any, str, bool]]
) -> Dict[str, set]:
    """Map ``var -> {numeric values}`` for solution-context ``var = number`` eqns."""
    out: Dict[str, set] = {}
    for lhs, rhs, _raw, ctx in equations:
        if not ctx:
            continue
        for a, b in ((lhs, rhs), (rhs, lhs)):
            if getattr(a, "is_Symbol", False) and not getattr(
                b, "free_symbols", set()
            ):
                out.setdefault(str(a), set()).add(b)
    return out


def _check_solve(
    equations: Sequence[Tuple[Any, Any, str, bool]]
) -> List[Tuple[str, str, str, str]]:
    """Pattern (a) — single-variable solve verification.

    Returns ``[(var, claimed, detail, raw), ...]`` for each provable violation
    (``raw`` is the failing equation fragment, used for negation-suppression).
    """
    findings: List[Tuple[str, str, str, str]] = []
    solutions = _stated_solutions(equations)
    for var, vals in solutions.items():
        # Non-trivial single-variable equations of this variable in the block.
        var_eqs: List[Tuple[Any, Any, str]] = []
        for lhs, rhs, raw, _ctx in equations:
            free = {str(s) for s in (lhs.free_symbols | rhs.free_symbols)}
            if free != {var}:
                continue
            if _is_trivial_solution_eq(lhs, rhs, var):
                continue
            var_eqs.append((lhs, rhs, raw))
        if not var_eqs:
            continue
        for c in vals:
            # PRECISION RULE: flag a claimed ``var = c`` ONLY when ``c`` solves
            # NONE of the block's non-trivial equations for ``var``. This is the
            # multi-problem guard — a block that presents several distinct
            # problems in the same variable (each with its own correct answer)
            # never fires, because each stated answer solves ITS OWN equation
            # (so ``c`` is a solution of >=1 equation). A genuinely WRONG answer
            # solves none of the presented equations.
            solves_some = False
            failing: Optional[Tuple[Any, Any, str]] = None
            for lhs, rhs, raw in var_eqs:
                try:
                    diff = sympy.simplify((lhs - rhs).subs(sympy.Symbol(var), c))  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    continue
                if getattr(diff, "free_symbols", set()):
                    continue
                if diff == 0:
                    solves_some = True
                    break
                if failing is None and _definitely_nonzero(diff):
                    failing = (lhs, rhs, raw)
            if solves_some or failing is None:
                continue
            lhs, rhs, raw = failing
            resid = sympy.simplify((lhs - rhs).subs(sympy.Symbol(var), c))  # type: ignore[union-attr]
            findings.append((
                var,
                f"{var} = {c}",
                f"the claimed solution {var}={c} solves NONE of the block's "
                f"equations for {var}; e.g. substituting into '{raw}' gives "
                f"lhs-rhs = {resid} (must be 0)",
                raw,
            ))
    return findings


#: System-solve markers — pattern (b) fires ONLY when the block is explicitly
#: solving a SYSTEM (not graphing a line by plotting points, which also yields
#: two-variable equations + ordered pairs but where each point lies on exactly
#: one line). Without this gate a point on line 1 would be flagged against
#: line 2 — a false positive on the ubiquitous graphing-by-points blocks.
_SYSTEM_MARKERS: Tuple[str, ...] = (
    "system of equation", "the system", "solve the system", "linear system",
    "substitution method", "elimination method", "point of intersection",
    "satisfies both", "solution to the system",
)
#: Graphing-by-points blocks emit a TABLE of pairs (points on a line). A genuine
#: system solve produces ONE solution pair; cap distinct solution-context pairs
#: so a point table can never be mistaken for a system solution.
_MAX_SYSTEM_PAIRS: int = 2


def _check_system(
    equations: Sequence[Tuple[Any, Any, str, bool]],
    pairs: Sequence[Tuple[Any, Any]],
    block_text: str,
) -> List[Tuple[str, str, str]]:
    """Pattern (b) — two-equation system + ordered-pair solution verification.

    Returns ``[(pair_str, detail), ...]`` per provable violation. Fires only when
    (1) the block carries an explicit system-solve marker (``_SYSTEM_MARKERS``)
    so graphing-by-points blocks are excluded, (2) a genuine system (>=2
    distinct, non-equivalent equations in the SAME two variables) is present,
    (3) the pair sits in a solution context, and (4) there are at most
    ``_MAX_SYSTEM_PAIRS`` distinct solution pairs (a point TABLE is not a system
    solution).
    """
    if not pairs:
        return []
    plain = _RE_TAG.sub(" ", block_text).lower()
    if not any(marker in plain for marker in _SYSTEM_MARKERS):
        return []
    # De-duplicate pairs by value; a point table (many distinct pairs) is not a
    # single system solution.
    uniq_pairs: List[Tuple[Any, Any]] = []
    for p in pairs:
        if all(not (p[0] == q[0] and p[1] == q[1]) for q in uniq_pairs):
            uniq_pairs.append(p)
    if len(uniq_pairs) > _MAX_SYSTEM_PAIRS:
        return []
    pairs = uniq_pairs
    # Group the residual (lhs - rhs) of every two-variable equation by its
    # variable pair.
    groups: Dict[frozenset, List[Tuple[Any, Any, str]]] = {}
    for lhs, rhs, raw, _ctx in equations:
        vars_ = {str(s) for s in (lhs.free_symbols | rhs.free_symbols)}
        if len(vars_) != 2:
            continue
        groups.setdefault(frozenset(vars_), []).append(
            (sympy.simplify(lhs - rhs), raw, (lhs, rhs))  # type: ignore[union-attr]
        )

    findings: List[Tuple[str, str, str]] = []
    for varset, residuals in groups.items():
        # Keep distinct (non-equivalent) residuals.
        distinct: List[Tuple[Any, str]] = []
        for res, raw, _lr in residuals:
            if all(not _are_equivalent(res, d_res) for d_res, _ in distinct):
                distinct.append((res, raw))
        if len(distinct) < 2:
            continue
        # Coordinate -> variable mapping. For the {x, y} convention the reading
        # order (x, y) is canonical, so a SINGLE mapping is used (unchanged
        # behavior). For any OTHER variable pair the coordinate order is
        # ambiguous — a pair stated for a system in variables a, s must not be
        # remapped onto the same system read as s, a (Mechanism A3 remap false
        # positive). So BOTH orderings are tried and the pair is flagged ONLY
        # when it fails the system under EVERY ordering (precision over recall: a
        # genuinely wrong pair fails both, while a correctly-stated pair whose
        # coordinate order merely differs from sorted-name order satisfies one).
        names = sorted(varset)
        if set(names) == {"x", "y"}:
            orderings = [["x", "y"]]
        elif names[0] == names[1]:  # degenerate single-name (defensive)
            orderings = [names]
        else:
            orderings = [names, [names[1], names[0]]]
        for (av, bv) in pairs:
            ordering_fails: List[Tuple[str, List[str], Any]] = []
            satisfied_some = False
            for order in orderings:
                subs = {  # type: ignore[union-attr]
                    sympy.Symbol(order[0]): av, sympy.Symbol(order[1]): bv
                }
                this_fail: Optional[Tuple[str, List[str], Any]] = None
                for res, raw in distinct:
                    try:
                        val = res.subs(subs)
                    except Exception:  # noqa: BLE001
                        continue
                    if _definitely_nonzero(val):
                        this_fail = (raw, order, res)
                        break
                if this_fail is None:
                    satisfied_some = True
                    break
                ordering_fails.append(this_fail)
            if satisfied_some or not ordering_fails:
                continue
            raw, order, res = ordering_fails[0]
            subs = {  # type: ignore[union-attr]
                sympy.Symbol(order[0]): av, sympy.Symbol(order[1]): bv
            }
            resid = sympy.simplify(res.subs(subs))  # type: ignore[union-attr]
            findings.append((
                f"({av}, {bv})",
                f"the pair ({order[0]}={av}, {order[1]}={bv}) does not "
                f"satisfy the system equation '{raw}' "
                f"(residual = {resid}, must be 0)",
                raw,
            ))
    return findings


def _check_numeric_chains(
    chains: Sequence[Tuple[List[Any], str, bool]],
    equations: Sequence[Tuple[Any, Any, str, bool]],
) -> List[Tuple[str, str]]:
    """Pattern (c) — numeric simplify/evaluate chain pairwise equivalence.

    Verifies both multi-side chains AND single-``=`` equations whose sides are
    ALL variable-free. Returns ``[(raw, detail), ...]`` per provable mismatch.
    """
    findings: List[Tuple[str, str]] = []

    def _all_numeric(exprs: Sequence[Any]) -> bool:
        return all(not getattr(e, "free_symbols", set()) for e in exprs)

    # Multi-side chains.
    for sides, raw, _ctx in chains:
        if not _all_numeric(sides):
            continue
        for i in range(len(sides) - 1):
            if _relative_mismatch(sides[i], sides[i + 1]):
                findings.append((
                    raw,
                    f"chain '{raw}' asserts {sides[i]} = {sides[i + 1]} but "
                    f"they are not equal (difference "
                    f"{sympy.simplify(sides[i] - sides[i + 1])})"  # type: ignore[union-attr]
                ))
                break
    # Single-'=' numeric equalities. Exclude raws already covered as multi-side
    # chains above (a chain registers its adjacent pairs as equations too, so
    # without this its numeric identity would be double-flagged).
    chain_raws = {raw for _sides, raw, _ctx in chains}
    seen: set = set()
    for lhs, rhs, raw, _ctx in equations:
        if raw in seen or raw in chain_raws:
            continue
        if getattr(lhs, "free_symbols", set()) or getattr(
            rhs, "free_symbols", set()
        ):
            continue
        seen.add(raw)
        if _relative_mismatch(lhs, rhs):
            findings.append((
                raw,
                f"numeric identity '{raw}' asserts {lhs} = {rhs} but they are "
                f"not equal (difference {sympy.simplify(lhs - rhs)})"  # type: ignore[union-attr]
            ))
    return findings


# --------------------------------------------------------------------------- #
# (b) plus-minus (±) solution-set verification
# --------------------------------------------------------------------------- #

#: A ``\pm`` / ``±`` operator (the two-root operator of the quadratic formula).
_RE_PM = re.compile(r"\\pm(?![a-zA-Z])|±")


def _num_equal(a: Any, b: Any) -> bool:
    """True iff ``a`` and ``b`` are variable-free numbers equal within tolerance."""
    try:
        diff = sympy.simplify(a - b)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return False
    if getattr(diff, "free_symbols", set()):
        return False
    if diff == 0:
        return True
    try:
        return abs(float(diff)) <= _NUMERIC_ABS_TOL
    except Exception:  # noqa: BLE001
        return False


def _extract_pm_solutions(html: str) -> List[Tuple[str, List[Any]]]:
    """Return ``[(var, [v_plus, v_minus]), ...]`` for solution-context ``v = a±b``.

    A claimed ``x = a ± b`` (or the full quadratic-formula shape
    ``x = \\frac{-b \\pm \\sqrt{...}}{2a}``) is EXPANDED into its two roots by
    substituting ``+`` then ``-`` for the ``±``. Only solution-context fragments
    of the shape ``<single variable> = <expr containing ±>`` whose two roots both
    parse to variable-free numbers are returned. Fail-closed: any parse failure
    drops the fragment.
    """
    if not _SYMPY_AVAILABLE:
        return []
    out: List[Tuple[str, List[Any]]] = []
    for frag, ctx in _iter_fragments(html):
        if not ctx:
            continue
        for sub in _split_on_implication(frag):
            if not _RE_PM.search(sub):
                continue
            sides = re.split(r"(?<![<>=!])=(?![=])", sub)
            sides = [s for s in sides if s.strip()]
            if len(sides) != 2:
                continue
            lhs = _parse_side(sides[0])
            if lhs is None or not getattr(lhs, "is_Symbol", False):
                continue
            vname = str(lhs)
            plus = _parse_side(_RE_PM.sub("+", sides[1]))
            minus = _parse_side(_RE_PM.sub("-", sides[1]))
            if plus is None or minus is None:
                continue
            if getattr(plus, "free_symbols", set()) or getattr(
                minus, "free_symbols", set()
            ):
                continue
            out.append((vname, [plus, minus]))
    return out


def _check_pm_solutions(
    pm_solutions: Sequence[Tuple[str, List[Any]]],
    equations: Sequence[Tuple[Any, Any, str, bool]],
) -> List[Tuple[str, str, str, str]]:
    """Pattern (b') — ``±`` solution-set verification.

    A claimed ``x = a ± b`` verifies iff BOTH roots ``a+b`` and ``a-b`` satisfy
    the SAME block equation for ``x`` (a quadratic's two-element solution set).
    Returns ``[(var, claimed, detail, raw), ...]`` per provable violation. Mirrors
    the ``_check_solve`` precision rule: flags only when NO single block equation
    for ``x`` is satisfied by BOTH roots AND at least one root definitely fails an
    equation (so a multi-problem block never cross-fires).
    """
    findings: List[Tuple[str, str, str, str]] = []
    for var, roots in pm_solutions:
        sym = sympy.Symbol(var)  # type: ignore[union-attr]
        var_eqs: List[Tuple[Any, Any, str]] = []
        for lhs, rhs, raw, _ctx in equations:
            free = {str(s) for s in (lhs.free_symbols | rhs.free_symbols)}
            if free != {var}:
                continue
            if _is_trivial_solution_eq(lhs, rhs, var):
                continue
            var_eqs.append((lhs, rhs, raw))
        if not var_eqs:
            continue
        # Is there an equation BOTH roots satisfy? (the correct solution set)
        satisfied_together = False
        for lhs, rhs, _raw in var_eqs:
            both = True
            for r in roots:
                try:
                    d = sympy.simplify((lhs - rhs).subs(sym, r))  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    both = False
                    break
                if getattr(d, "free_symbols", set()) or d != 0:
                    both = False
                    break
            if both:
                satisfied_together = True
                break
        if satisfied_together:
            continue
        # Flag: locate an equation + root that DEFINITELY fails.
        for lhs, rhs, raw in var_eqs:
            failing_root = None
            resid = None
            for r in roots:
                try:
                    d = sympy.simplify((lhs - rhs).subs(sym, r))  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    continue
                if _definitely_nonzero(d):
                    failing_root, resid = r, d
                    break
            if failing_root is None:
                continue
            root_str = ", ".join(str(r) for r in roots)
            findings.append((
                var,
                f"{var} = {{{root_str}}}",
                f"the claimed ± solution set {var} ∈ {{{root_str}}} is wrong: "
                f"root {failing_root} does not satisfy '{raw}' "
                f"(lhs-rhs = {resid}, must be 0 for BOTH roots)",
                raw,
            ))
            break
    return findings


# --------------------------------------------------------------------------- #
# Quadratic features — vertex + discriminant re-derivation (gap #9)
# --------------------------------------------------------------------------- #

#: A claimed discriminant value: ``discriminant ... = / is / : <tail>``. The tail
#: is captured broadly (family 1 FP fix): a claimed discriminant is often written
#: out as the arithmetic chain ``b^2 - 4ac`` and only THEN equated to its value —
#: ``the discriminant is 9^2 - 4(2)(-5) = 121``. The old capture grabbed the
#: leading ``9`` (the b^2 first term) and "verified" it against b^2-4ac, a
#: false-positive family. ``_claimed_discriminant`` resolves the tail to the value
#: AFTER the FINAL ``=`` (121), never the intermediate first term.
_RE_DISCRIMINANT = re.compile(
    r"discriminant\b.{0,40}?(?:=|is|equals|:)\s*([^<\n]{1,80})",
    re.IGNORECASE | re.DOTALL,
)

#: Operators that mark a leading number as a SUB-TERM of an unfinished arithmetic
#: expression (``9^2 - ...`` / ``9 * ...``) rather than a finished claim — used to
#: reject the b^2 first term when the discriminant tail has no terminating ``=``.
_ARITH_CONTINUATION: frozenset = frozenset(
    {"+", "-", "*", "/", "^", "(", "·", "×", "÷", "−", "–", "—"}
)

#: Sentence / clause boundary in a discriminant tail (Mechanism B). The DOTALL
#: ``_RE_DISCRIMINANT`` tail can run PAST the discriminant value into the next
#: sentence — often a standard-form recap ``ax^2 + bx + c = 0`` — and the old
#: ``split('=')[-1]`` then read that trailing ``= 0`` as the claimed value. The
#: claim is bounded at the first sentence/clause terminator (period or semicolon
#: followed by whitespace, or a newline). A decimal point (``12.5``) is NOT a
#: boundary because it is not followed by whitespace.
_RE_DISC_CLAUSE_END = re.compile(r"[.;]\s|\n")
#: A VARIABLE (letter) raised to the 2nd power — the signature of a full
#: one-variable quadratic equation that crossed into the tail (``x^2 ... = 0``).
#: A NUMERIC power like ``9^2`` (the discriminant's own b^2 first term) has a
#: digit, not a letter, before the exponent and is deliberately NOT matched.
_RE_VAR_SQUARED = re.compile(r"[A-Za-z]\s*(?:\^\s*2|\*\*\s*2|²)")


def _claimed_discriminant(tail: str) -> Optional[Any]:
    """Resolve the claimed discriminant value from a ``_RE_DISCRIMINANT`` tail.

    The tail is a bare number (``121``), a chain (``9^2 - 4(2)(-5) = 121``) whose
    CLAIM is the value after the final ``=``, or a symbolic / unfinished
    expression (``b^2 - 4ac``) that carries no numeric claim. Returns the
    variable-free numeric claim, or None (skip — precision over recall) when the
    final segment is not a clean, finished number.

    Mechanism B — the tail is first bounded at the first sentence/clause boundary
    so a following standard-form equation (``ax^2 + bx + c = 0`` in the next
    sentence) is not read as the claim. As a belt-and-suspenders for a
    comma-joined crossing (no sentence boundary), a final ``= 0`` whose immediate
    left side is a full one-variable quadratic (a variable squared) is rejected
    (that ``= 0`` is the quadratic's own RHS, not the discriminant).
    """
    if not _SYMPY_AVAILABLE:
        return None
    # Bound the claim at the first sentence / clause boundary (Mechanism B).
    tail = _RE_DISC_CLAUSE_END.split(tail, 1)[0]
    parts = tail.split("=")
    if len(parts) >= 2:
        rhs_final = parts[-1].strip()
        lhs_before_final = parts[-2]
        if re.match(r"0(?![.\d])", rhs_final) and _RE_VAR_SQUARED.search(
            lhs_before_final
        ):
            return None
    seg = parts[-1].strip()
    m = re.match(r"[+-]?\d+(?:\.\d+)?", seg)
    if not m:
        return None
    after = seg[m.end():].lstrip()
    # A leading number followed by more arithmetic (an unfinished expression with
    # NO terminating '=') is a sub-term (e.g. the b^2 in ``9^2 - 40``), not the
    # claim -> skip. Trailing prose ("121 for the quadratic") is fine.
    if after[:1] in _ARITH_CONTINUATION or after[:1].isdigit():
        return None
    val = _parse_side(m.group(0))
    if val is None or getattr(val, "free_symbols", set()):
        return None
    return val
#: A ``vertex`` keyword (the ordered pair is harvested from the following window).
_RE_VERTEX_KW = re.compile(r"vertex", re.IGNORECASE)
_VERTEX_WINDOW: int = 80

#: A ``<span data-cf-term="...">`` element's inner text (Mechanism C). An equation
#: that lives as the plain-text CONTENT of a term span (``<span data-cf-term=
#: "quadratic">x^2 + 5x + 6</span>``) carries no LaTeX delimiter, so the ordinary
#: ``_iter_fragments`` harvest (delimited math only) never sees it — leaving the
#: quadratic invisible to ``_quadratic_coeffs`` and a CORRECT discriminant/vertex
#: claim mis-flagged against some OTHER (delimited) quadratic. Adding the span
#: text to the quadratic CANDIDATE pool can only SUPPRESS a claim (more candidates
#: for a claim to match), never mint a finding, so it is scoped to the quadratic-
#: feature candidate harvest and never widens the solve/system/chain surfaces.
_RE_CF_TERM = re.compile(
    r"<span\b[^>]*\bdata-cf-term\b[^>]*>(.*?)</span>",
    re.DOTALL | re.IGNORECASE,
)


def _quadratic_coeffs(content: str) -> List[Tuple[Any, Any, Any, Any]]:
    """Return ``[(a, b, c, var), ...]`` for every single-variable quadratic found.

    Scans each math fragment side (split on ``=`` / implication) for a degree-2
    polynomial ``a·v² + b·v + c`` in exactly one variable. Deduplicated by
    coefficient tuple. Non-quadratics / multi-variable / unparseable are skipped.

    Mechanism C: in addition to the well-delimited ``_iter_fragments`` math, the
    plain-text content of every ``<span data-cf-term="...">`` is harvested as a
    candidate (a quadratic that lives as term-span text is otherwise invisible).
    """
    if not _SYMPY_AVAILABLE:
        return []
    frags: List[Tuple[str, bool]] = list(_iter_fragments(content))
    for cm in _RE_CF_TERM.finditer(content):
        inner = _RE_TAG.sub(" ", cm.group(1)).strip()
        if inner:
            frags.append((inner, False))
    out: List[Tuple[Any, Any, Any, Any]] = []
    seen: set = set()
    for frag, _ctx in frags:
        for sub in _split_on_implication(frag):
            for piece in re.split(r"(?<![<>=!])=(?![=])", sub):
                expr = _parse_side(piece)
                if expr is None:
                    continue
                syms = list(getattr(expr, "free_symbols", set()))
                if len(syms) != 1:
                    continue
                v = syms[0]
                try:
                    poly = sympy.Poly(expr, v)  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    continue
                if poly.degree() != 2:
                    continue
                coeffs = poly.all_coeffs()
                if len(coeffs) != 3:
                    continue
                a, b, c = coeffs
                if a == 0:
                    continue
                key = (a, b, c, str(v))
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, b, c, v))
    return out


def _check_quadratic_features(content: str) -> List[Tuple[str, str, str, str]]:
    """Gap #9 — re-derive a claimed VERTEX / DISCRIMINANT and flag a mismatch.

    For a block carrying a single-variable quadratic ``a·v² + b·v + c``:

    * a claimed ``discriminant = N`` is compared against ``b² - 4ac``;
    * a claimed ``vertex (h, k)`` is compared against ``(-b/2a, c - b²/4a)``.

    PRECISION RULE (matches the solve/system checks): a claim is flagged ONLY when
    it matches NONE of the block's quadratic candidates (a block with two
    quadratics never mis-attributes). Returns ``[(kind, claimed, detail, raw), ...]``.
    """
    candidates = _quadratic_coeffs(content)
    if not candidates:
        return []
    findings: List[Tuple[str, str, str, str]] = []

    plain = content
    for delim in (r"\(", r"\)", r"\[", r"\]"):
        plain = plain.replace(delim, " ")
    plain = _RE_TAG.sub(" ", plain)

    def _disc(a: Any, b: Any, c: Any) -> Any:
        return sympy.simplify(b ** 2 - 4 * a * c)  # type: ignore[union-attr]

    def _vertex(a: Any, b: Any, c: Any) -> Tuple[Any, Any]:
        h = sympy.simplify(-b / (2 * a))  # type: ignore[union-attr]
        k = sympy.simplify(c - b ** 2 / (4 * a))  # type: ignore[union-attr]
        return h, k

    # Discriminant claims.
    for m in _RE_DISCRIMINANT.finditer(plain):
        claimed = _claimed_discriminant(m.group(1))
        if claimed is None:
            continue
        computed = [_disc(a, b, c) for (a, b, c, _v) in candidates]
        if any(_num_equal(claimed, cv) for cv in computed):
            continue
        a, b, c, v = candidates[0]
        cv = _disc(a, b, c)
        findings.append((
            "discriminant",
            f"discriminant = {claimed}",
            f"claimed discriminant {claimed} != b^2-4ac = {cv} for the quadratic "
            f"{a}*{v}^2 + {b}*{v} + {c}",
            m.group(0).strip(),
        ))

    # Vertex claims.
    for vm in _RE_VERTEX_KW.finditer(plain):
        window = plain[vm.end(): vm.end() + _VERTEX_WINDOW]
        pm = _RE_PAIR.search(window)
        if not pm:
            continue
        h_claim = _parse_side(pm.group(1))
        k_claim = _parse_side(pm.group(2))
        if h_claim is None or k_claim is None:
            continue
        if getattr(h_claim, "free_symbols", set()) or getattr(
            k_claim, "free_symbols", set()
        ):
            continue
        matched = False
        for (a, b, c, _v) in candidates:
            vh, vk = _vertex(a, b, c)
            if _num_equal(h_claim, vh) and _num_equal(k_claim, vk):
                matched = True
                break
        if matched:
            continue
        a, b, c, v = candidates[0]
        vh, vk = _vertex(a, b, c)
        findings.append((
            "vertex",
            f"vertex ({h_claim}, {k_claim})",
            f"claimed vertex ({h_claim}, {k_claim}) != computed vertex "
            f"({vh}, {vk}) for the quadratic {a}*{v}^2 + {b}*{v} + {c}",
            pm.group(0).strip(),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Coercion + decision capture
# --------------------------------------------------------------------------- #


def _coerce_blocks(
    inputs: Dict[str, Any]
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Resolve ``inputs['blocks']`` or hydrate from ``inputs['blocks_final_path']``.

    The gate-input router surfaces ``blocks`` (rewrite-tier hydrated Block list);
    tests / standalone callers may pass ``blocks_final_path`` (a JSONL of block
    dicts) instead.
    """
    raw = inputs.get("blocks")
    if raw is not None:
        if not isinstance(raw, list):
            return [], GateIssue(
                severity="critical",
                code="INVALID_BLOCKS_INPUT",
                message=(
                    f"inputs['blocks'] must be a list; got {type(raw).__name__}."
                ),
            )
        return list(raw), None

    path_raw = inputs.get("blocks_final_path")
    if not path_raw:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs requires 'blocks' (List[Block]) or 'blocks_final_path' "
                "(a JSONL path of rewrite-tier block dicts)."
            ),
        )
    try:
        path = Path(path_raw)
    except (TypeError, ValueError):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=f"blocks_final_path is not a path: {path_raw!r}.",
        )
    if not path.exists():
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=f"blocks_final_path does not exist: {path}.",
        )
    blocks: List[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return blocks, None


def _block_field(block: Any, field: str) -> Any:
    """Read ``field`` from a Block instance OR a plain dict (JSONL row)."""
    if isinstance(block, dict):
        return block.get(field)
    return getattr(block, field, None)


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    block_type: str,
    equations: int,
    chains: int,
    skipped: int,
    passed: bool,
    flagged_detail: Optional[str],
    shadow: bool,
) -> None:
    """Emit one ``worked_example_math_check`` decision per SCORED block.

    Rationale interpolates dynamic per-call signals (>=20 chars) so the capture
    is replayable post-hoc (root ``CLAUDE.md`` § "LLM call-site instrumentation").
    Swallows ``log_decision`` exceptions (capture must not break the gate).
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{_CODE_MATH_WRONG}"
    rationale = (
        f"Symbolic worked-example verification on Block {block_id!r}: "
        f"block_type={block_type!r}, equations_parsed={equations}, "
        f"chains_parsed={chains}, fragments_skipped_unparseable={skipped}, "
        f"shadow={shadow}, flagged={flagged_detail or 'none'}. "
        f"sympy re-checks the derivation (single-var solve / two-eq system / "
        f"numeric simplify chain) — a wrong worked example (e.g. a sign-error "
        f"substitution asserting an ordered pair that satisfies neither system "
        f"equation) is caught here where provenance / number-blind NLI / "
        f"numeric-grounding gates cannot."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s: %s", _DECISION_TYPE, exc
        )


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class WorkedExampleMathValidator:
    """Symbolic worked-example verification gate (math-correctness control).

    For each ``example`` / ``worked_example`` (and worked-example-bearing
    ``concept`` / ``explanation``) block, harvests well-delimited math fragments
    and re-checks the derivation under sympy across three high-precision
    patterns (single-variable solve / two-equation system / numeric simplify
    chain). Fires ``WORKED_EXAMPLE_MATH_WRONG`` when the math is PROVABLY wrong;
    anything unparseable is skipped silently.

    Inputs:
        blocks: List[Block] | List[dict]
            Rewrite-tier blocks (hydrated Block instances from the router, or
            plain dict rows from a JSONL).
        blocks_final_path: str | Path
            Alternative to ``blocks`` — a rewrite-tier blocks JSONL.
        shadow: bool
            Forces emitted issues to warning-severity (parity with the numeric
            gates; the gate is warning day-1 regardless).
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per scored block.
    """

    name = "worked_example_math"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        shadow = bool(inputs.get("shadow", False))

        # Graceful degrade — sympy is the only dependency (mirrors the
        # EMBEDDING_DEPS_MISSING convention: a single warning issue, passed=True).
        if not _SYMPY_AVAILABLE:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="warning",
                        code=_CODE_SYMPY_MISSING,
                        message=(
                            "sympy is not installed; symbolic worked-example "
                            "verification is skipped (install the [embedding] "
                            "optional extra). Gate passes vacuously."
                        ),
                        suggestion="pip install -e '.[embedding]'  # provides sympy",
                    )
                ],
                action=None,
            )

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

        # Warning day-1 regardless of the shadow knob (the gate is warning-
        # severity in config/workflows.yaml; the shadow knob is kept for parity
        # with the sibling numeric gates and the deferred critical flip, at
        # which point a truthy shadow would keep issues at warning while a
        # critical YAML severity gates). Until the flip, both are warning.
        fail_severity = "warning"
        _ = shadow  # reserved for the deferred critical-flip severity split
        issues: List[GateIssue] = []
        scored_blocks = 0
        passed_blocks = 0
        total_skipped = 0

        for block in blocks:
            block_type = _block_field(block, "block_type")
            if block_type not in _TARGET_BLOCK_TYPES:
                continue
            content = _block_field(block, "content")
            if not isinstance(content, str) or not content.strip():
                continue
            block_id = _block_field(block, "block_id") or "<unknown>"

            fragments = _iter_fragments(content)
            if not fragments:
                continue

            equations, chains = _collect_equations(fragments)
            # A fragment is "skipped_unparseable" when it split into sides but a
            # side failed to parse — approximate as (fragments with '=') minus
            # (equations + chains registered). Precise per-fragment tracking is
            # not needed; report the aggregate signal.
            frags_with_eq = sum(
                1
                for f, _ in fragments
                for sub in _split_on_implication(f)
                if _split_equation_sides(sub) is not None
            )
            parsed_units = len(
                {raw for _, _, raw, _ in equations}
                | {raw for _, raw, _ in chains}
            )
            skipped = max(0, frags_with_eq - parsed_units)
            total_skipped += skipped

            # (b') ± solution sets + (gap #9) quadratic vertex/discriminant
            # claims are their own checkable math even when the ordinary single-
            # ``=`` equation harvest is thin, so they contribute to the scored
            # condition below. Family 3 — a ``\begin{cases}`` system is checkable
            # math too even when the block has no standalone equations (a pure-
            # cases worked system), so it likewise counts toward "scored".
            pm_solutions = _extract_pm_solutions(content)
            quad_candidates = _quadratic_coeffs(content)
            cases_equations = _cases_equations(content)

            if not equations and not chains and not pm_solutions and (
                not quad_candidates
            ) and not cases_equations:
                # No parseable math — skip silently (no event).
                continue

            scored_blocks += 1

            solve_findings = _check_solve(equations)
            pm_findings = _check_pm_solutions(pm_solutions, equations)
            quad_findings = (
                _check_quadratic_features(content) if quad_candidates else []
            )
            # Fix a — scope the system cross-check to one PROBLEM SEGMENT so a
            # solution pair from one problem is never checked against another
            # problem's equation. Marker detection still sees the whole block.
            # Family 3 — each segment is system-checked against ITS OWN equation
            # set, with a \begin{cases} environment (the real system) parsed and
            # foreign standalone equations excluded (_segment_system_findings).
            segments = _split_problem_segments(content)
            if len(segments) > 1:
                system_findings: List[Tuple[str, str, str]] = []
                for seg in segments:
                    system_findings.extend(
                        _segment_system_findings(seg, content)
                    )
            else:
                system_findings = _segment_system_findings(content, content)
            chain_findings = _check_numeric_chains(chains, equations)

            # Fix d — suppress a finding when the block is DELIBERATELY teaching a
            # non-solution ("(False)" / "not a solution" within ~60 chars after
            # the flagged equation).
            block_details: List[str] = []
            _seen_details: set = set()

            def _add(detail: str) -> None:
                if detail not in _seen_details:
                    _seen_details.add(detail)
                    block_details.append(detail)

            for _var, claimed, detail, raw in solve_findings:
                if _is_taught_failure(content, raw):
                    continue
                _add(f"solve: claimed {claimed}; {detail}")
            for _var, claimed, detail, raw in pm_findings:
                if _is_taught_failure(content, raw):
                    continue
                _add(f"pm: claimed {claimed}; {detail}")
            for _pair, detail, raw in system_findings:
                if _is_taught_failure(content, raw):
                    continue
                _add(f"system: {detail}")
            for raw, detail in chain_findings:
                if _is_taught_failure(content, raw):
                    continue
                _add(f"chain: {detail}")
            for _kind, _claimed, detail, raw in quad_findings:
                if _is_taught_failure(content, raw):
                    continue
                _add(f"quadratic: {detail}")

            if not block_details:
                passed_blocks += 1
                _emit_decision(
                    capture, block_id=block_id, block_type=block_type,
                    equations=len(equations), chains=len(chains),
                    skipped=skipped, passed=True, flagged_detail=None,
                    shadow=shadow,
                )
                continue

            for detail in block_details:
                if len(issues) >= _ISSUE_LIST_CAP:
                    break
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_MATH_WRONG,
                        message=(
                            f"Worked-example block {block_id!r} "
                            f"(block_type={block_type!r}) contains provably "
                            f"WRONG math under sympy — {detail}. Provenance / "
                            f"number-blind NLI / numeric-grounding gates cannot "
                            f"catch a wrong DERIVATION (the individual numbers "
                            f"may be grounded)."
                        ),
                        location=str(block_id),
                        suggestion=(
                            "Re-derive the worked example. A sign error in an "
                            "isolation step, or an asserted solution that does "
                            "not satisfy the original equation(s), fails the "
                            "symbolic re-check. Copy the worked example verbatim "
                            "from the cited source or fix the algebra."
                        ),
                    )
                )
            _emit_decision(
                capture, block_id=block_id, block_type=block_type,
                equations=len(equations), chains=len(chains),
                skipped=skipped, passed=False,
                flagged_detail="; ".join(block_details)[:400], shadow=shadow,
            )

        # Warning-severity day-1: ``passed`` honors only critical issues (there
        # are none day-1), so the gate never blocks — it COMPUTES + CAPTURES.
        # The deferred critical flip (# TODO(calibration) in workflows.yaml)
        # promotes WORKED_EXAMPLE_MATH_WRONG to critical.
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = (
            1.0 if scored_blocks == 0 else round(passed_blocks / scored_blocks, 4)
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
            metadata={
                "scored_blocks": scored_blocks,
                "passed_blocks": passed_blocks,
                "flagged_blocks": scored_blocks - passed_blocks,
                "skipped_unparseable": total_skipped,
            },
        )


__all__ = [
    "WorkedExampleMathValidator",
    "_CODE_MATH_WRONG",
    "_CODE_SYMPY_MISSING",
    "_DECISION_TYPE",
    "_TARGET_BLOCK_TYPES",
]
