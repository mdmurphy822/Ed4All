"""Math-representation folding — reconcile LaTeX, unicode-math, and plain-text.

A single deterministic, stdlib-only normalizer that folds three divergent
math *representations* onto one canonical token vocabulary so that text QA
metrics (recall shingles / junk-OOV / garbage-word / repetition) compare the
*content* of a math span rather than its *notation*.

Why this exists
---------------
SemantiK's VLM fused-extraction arm emits math as inline LaTeX
(``$\\sqrt{h}$``, ``\\frac{a}{b}``, ``a \\cdot b``) while a clean vendor gold
chunkset writes the same math as unicode / plain text (``√h``, ``a/b``,
``a · b``). Without folding, every LaTeX control word (``sqrt`` 2664×, ``frac``
1027×, ``cdot`` 244× in one measured chapter) reads as OCR garbage / OOV junk,
and math-dense exercise chunks read as recall failures — all *representation
artifacts*, not real conversion defects.

Folding rules (deterministic; both candidate AND gold pass through the same
transform so they meet in a shared vocabulary)
------------------------------------------------------------------------------
1. **Known LaTeX symbol-commands → a canonical bare word.** ``\\sqrt`` →
   ``sqrt``, ``\\cdot`` → ``cdot``, ``\\neq`` → ``neq``, ``\\geq`` → ``geq`` …
   (:data:`_LATEX_KEEP`). These are the commands whose gold counterpart is a
   *unicode symbol*, so we fold BOTH to the same word (rule 3) — they meet in
   the middle at the word.
2. **All other LaTeX control sequences → dropped to a space**, so their
   ARGUMENT survives: ``\\frac{a}{b}`` → `` a  b `` matches gold ``a/b`` →
   ``a b``; ``\\text{hi}`` → ``hi``. This mirrors
   ``SemantiK/semantik_structure/vlm_fusion.py::_strip_latex`` (the fusion-local
   scoring precedent) — the same "strip control sequences / braces / ``$`` to
   their plain content" transform. The vlm_fusion sibling is *scoring-only and
   fusion-local* (it additionally applies the aggressive ``√``-OCR ``V``-strip
   in ``_fold_confusables`` that would destroy English prose); this module is
   the general, prose-safe home. Kept behaviourally consistent — not a third
   divergent fork.
3. **Unicode math symbols → the SAME canonical word.** ``√`` → ``sqrt``,
   ``≠`` → ``neq``, ``≥`` → ``geq``, ``·`` → ``cdot``, ``×`` → ``times`` …
   plus super/subscript digits → their ASCII digit (:data:`_UNICODE_MAP`).
4. **Digit↔letter fusions are split** in both directions: ``9c`` → ``9 c`` (a
   fraction/coefficient the gold spaces), ``x2`` (from a folded superscript) →
   ``x 2``. Applied to both sides so they normalize consistently.
5. **Exercise part markers → the bare letter token.** A gold chunkset renders
   exercise part markers as UNICODE circled letters (``ⓐ`` U+24D0.., ``Ⓐ``
   U+24B6..) while the VLM-fused candidate emits parenthesized ASCII (``(a)``).
   BOTH fold to the same plain-letter token (``ⓐ`` == ``(a)`` == ``a``) so an
   exercise list scored purely on marker notation is not a recall miss. The
   parenthesized arm is anchored to a SINGLE letter in parens so it never
   touches arbitrary ``(see fig)`` paren content.

Everything is a **no-op on plain English**: no backslashes, no math unicode,
and no digit↔letter adjacency means the input is returned unchanged (modulo
whitespace the downstream tokenizer already collapses). That invariance is
what lets a math-aware metric stay stable on non-math corpora.
"""
from __future__ import annotations

import re
from html import escape as _html_escape
from html import unescape as _html_unescape

__all__ = [
    "fold_math",
    "count_math_folds",
    "MATH_WORDS",
    "strip_latex_commands",
    "sanitize_body_latex",
    "wrap_bare_math",
    "escape_currency_dollars",
    "escape_math_angle_brackets",
    "sanitize_math_spans",
    "strip_tikz_figures",
    "strip_markdown_images",
    "strip_literal_img_tags",
    "separate_adjacent_math_spans",
    "linkify_urls",
]

# ---------------------------------------------------------------------------
# LaTeX command folding.
# ---------------------------------------------------------------------------
# Symbol-commands whose gold counterpart is a UNICODE symbol → keep as a bare
# canonical word so ``\sqrt`` (candidate LaTeX) and ``√`` (gold unicode) both
# resolve to ``sqrt``. The VALUE is the canonical token; several aliases can
# map to one canonical (``\ne``/``\neq`` → ``neq``).
_LATEX_KEEP: dict[str, str] = {
    "sqrt": "sqrt",
    "cbrt": "sqrt",
    "cdot": "cdot",
    "times": "times",
    "div": "div",
    "pm": "pm",
    "mp": "mp",
    "neq": "neq",
    "ne": "neq",
    "geq": "geq",
    "ge": "geq",
    "leq": "leq",
    "le": "leq",
    "approx": "approx",
    "equiv": "equiv",
    "infty": "infty",
    "sum": "sum",
    "prod": "prod",
    "int": "int",
    "pi": "pi",
    "theta": "theta",
    "alpha": "alpha",
    "beta": "beta",
    "degree": "deg",
}

# The canonical folded words (values above) PLUS the structural command names
# that survive as bare tokens in adversarial input — the garbage-word detector
# exempts every one of these so a legitimately-folded math token is never
# counted as OCR garble (``sqrt`` has no vowel; ``frac`` reads consonant-heavy).
MATH_WORDS: frozenset[str] = frozenset(
    set(_LATEX_KEEP.values())
    | {
        "frac", "dfrac", "tfrac", "cfrac",
        "aligned", "align", "array", "matrix", "pmatrix", "bmatrix",
        "quad", "qquad", "cases", "cdots", "ldots", "vdots", "ddots",
        "checkmark", "overline", "underline", "boldsymbol", "mathrm",
        "mathbf", "operatorname", "displaystyle", "textstyle",
        "gamma", "delta", "sigma", "omega", "lambda", "mu", "nu",
    }
)

# One control sequence: ``\`` followed by letters (a named command) OR a single
# non-letter (``\,`` ``\;`` ``\\`` spacing / escapes).
_LATEX_CMD_RE = re.compile(r"\\([a-zA-Z]+|.)")

# Residual LaTeX punctuation / delimiters once commands are consumed.
_LATEX_PUNCT_RE = re.compile(r"[{}$^_&~]")

# Digit↔letter boundary (zero-width) — split in BOTH directions.
_DIGIT_LETTER_RE = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")

# ---------------------------------------------------------------------------
# Unicode math folding.
# ---------------------------------------------------------------------------
_UNICODE_MAP: dict[str, str] = {
    # roots / operators
    "√": " sqrt ",
    "∛": " sqrt ",
    "∜": " sqrt ",
    "·": " cdot ",
    "⋅": " cdot ",
    "×": " times ",
    "÷": " div ",
    "±": " pm ",
    "∓": " mp ",
    # relations
    "≠": " neq ",
    "≥": " geq ",
    "⩾": " geq ",
    "≤": " leq ",
    "⩽": " leq ",
    "≈": " approx ",
    "≅": " approx ",
    "≡": " equiv ",
    # big operators / constants
    "∞": " infty ",
    "∑": " sum ",
    "∏": " prod ",
    "∫": " int ",
    "π": " pi ",
    "θ": " theta ",
    "α": " alpha ",
    "β": " beta ",
    # normalisation of look-alikes to ASCII
    "−": "-",   # U+2212 minus sign → hyphen-minus
    "⁄": "/",   # fraction slash → solidus
    # superscript digits → ASCII digit (so ``x²`` → ``x2`` → split → ``x 2``)
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "ⁿ": "n",
    # subscript digits → ASCII digit
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}
_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_MAP))

# ---------------------------------------------------------------------------
# Exercise part-marker folding (rule 5).
# ---------------------------------------------------------------------------
# Circled letters — lowercase ``ⓐ``..``ⓩ`` (U+24D0..U+24E9) and uppercase
# ``Ⓐ``..``Ⓩ`` (U+24B6..U+24CF) — fold to the SAME plain-letter token (spaced,
# so an adjacent run tokenizes cleanly). Kept SEPARATE from ``_UNICODE_MAP`` so
# ``count_math_folds`` can report the ``part_markers`` bucket distinctly from the
# math-symbol bucket.
_CIRCLED_LETTER_MAP: dict[str, str] = {}
for _i in range(26):
    _CIRCLED_LETTER_MAP[chr(0x24D0 + _i)] = f" {chr(ord('a') + _i)} "  # ⓐ..ⓩ
    _CIRCLED_LETTER_MAP[chr(0x24B6 + _i)] = f" {chr(ord('A') + _i)} "  # Ⓐ..Ⓩ
_CIRCLED_LETTER_RE = re.compile("|".join(re.escape(k) for k in _CIRCLED_LETTER_MAP))

# Parenthesized single-letter part marker ``(a)`` / ``(B)`` → the bare letter.
# Anchored to EXACTLY one ASCII letter in parens (nothing else inside), so a
# multi-char group (``(6-11)``, ``(see fig)``) never matches — only enumeration
# markers fold.
_PAREN_LETTER_RE = re.compile(r"\(([A-Za-z])\)")


def _sub_latex_cmd(m: "re.Match[str]") -> str:
    body = m.group(1)
    if len(body) > 1:  # a named command (letters)
        canon = _LATEX_KEEP.get(body.lower())
        return f" {canon} " if canon is not None else " "
    return " "  # single-char escape / spacing command → drop


def strip_latex_commands(s: str) -> str:
    """Drop every LaTeX control sequence + delimiter to a space.

    The general, prose-safe mirror of
    ``SemantiK/semantik_structure/vlm_fusion.py::_strip_latex`` (kept behaviourally
    consistent so this is not a divergent third implementation). Unlike
    :func:`fold_math` it does NOT keep symbol-commands as words — it is the raw
    "everything to its plain argument" strip. Harmless on text with no
    backslashes / braces.
    """
    s = re.sub(r"\\([a-zA-Z]+|.)", " ", s or "")
    return _LATEX_PUNCT_RE.sub(" ", s)


# ---------------------------------------------------------------------------
# Body-text LaTeX/markdown sanitation (2026-07-04 end-user-HTML audit — B3).
# ---------------------------------------------------------------------------
# MathJax only processes ``$…$`` / ``$$…$$`` / ``\(…\)`` / ``\[…\]``. Text-mode
# LaTeX OUTSIDE those delimiters renders literally to the learner:
# ``\textbf{Square Root of a Number}`` / ``\textit{square root}`` /
# ``\checkmark`` / ``\begin{tabular}…\end{tabular}`` / markdown ``| --- | --- |``
# separator rows (110 ``\textbf`` + 22 ``\begin{tabular}`` corpus-wide). Unlike
# :func:`strip_latex_commands` (the heading stripper that DROPS every command),
# body sanitation CONVERTS the two visible emphasis commands to real HTML
# (``<strong>`` / ``<em>``) and drops the whole-fragment garbage shapes.
#
# CONSERVATIVE by design — the report's contract:
#   * NEVER touch a ``$…$`` / ``\(…\)`` / ``\[…\]`` / ``$$…$$`` math run —
#     MathJax owns those. They are stashed out before any transform and
#     restored verbatim after.
#   * Only WHOLE-FRAGMENT shapes are folded: ``\textbf{X}`` / ``\textit{X}``
#     (single-brace body), a whole ``\begin{tabular}…\end{tabular}`` block, a
#     markdown separator row (pipes + dashes only), and the bare ``\checkmark``
#     control word. Prose with none of these round-trips unchanged.
_BODY_MATH_RUN_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL
)
_TABULAR_RE = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)
# A markdown table separator row: ``| --- | --- |`` (each cell dashes-only,
# optional leading/trailing alignment colons). Never a real data row.
_MD_SEP_ROW_RE = re.compile(r"\|(?:\s*:?-{2,}:?\s*\|)+")
_TEXTBF_RE = re.compile(r"\\textbf\s*\{([^{}]*)\}")
_TEXTIT_RE = re.compile(r"\\textit\s*\{([^{}]*)\}")
_CHECKMARK_RE = re.compile(r"\\checkmark\b")
_STASH_RE = re.compile("\x00(\\d+)\x00")


def sanitize_body_latex(text: str, *, html: bool = True) -> str:
    r"""Fold visible text-mode LaTeX / markdown garbage out of BODY text (B3).

    ``html=True`` (rendered body): ``\textbf{X}`` -> ``<strong>X</strong>``,
    ``\textit{X}`` -> ``<em>X</em>``. ``html=False`` (plain chunk/sidecar text):
    both -> the bare ``X``. In BOTH modes a whole ``\begin{tabular}…\end{tabular}``
    block, a markdown ``| --- |`` separator row, and ``\checkmark`` are dropped.
    Math runs (``$…$`` etc.) are protected verbatim. A fast guard keeps the
    common no-markup path allocation-free (a string with no ``\`` and no ``|``
    is returned unchanged).
    """
    if not text or ("\\" not in text and "|" not in text):
        return text or ""
    stash: list[str] = []

    def _protect(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    s = _BODY_MATH_RUN_RE.sub(_protect, text)
    s = _TABULAR_RE.sub(" ", s)
    s = _MD_SEP_ROW_RE.sub(" ", s)
    if html:
        s = _TEXTBF_RE.sub(r"<strong>\1</strong>", s)
        s = _TEXTIT_RE.sub(r"<em>\1</em>", s)
    else:
        s = _TEXTBF_RE.sub(r"\1", s)
        s = _TEXTIT_RE.sub(r"\1", s)
    s = _CHECKMARK_RE.sub("", s)
    s = _STASH_RE.sub(lambda m: stash[int(m.group(1))], s)
    return re.sub(r"[ \t]{2,}", " ", s)


# ---------------------------------------------------------------------------
# Bare (un-delimited) LaTeX math wrapping (2026-07-04 exemplar-parity wave — B3+).
# ---------------------------------------------------------------------------
# The VLM fused-extraction arm sometimes emits math with NO ``$``/``\(``/``\[``
# delimiter at all: ``\sqrt{5} \approx 2.236 \sqrt{6} \approx 2.45`` or a whole
# answer-key crammed as ``\sqrt{16n^2} = 4n \sqrt{64x^2} \sqrt{169y^2}``. Because
# MathJax only processes DELIMITED math, these render as literal backslash
# garbage to the learner AND read as OCR junk to the chunker; the structure
# scorecard's cleanliness dimension counts every such ``\command`` as a leak.
#
# :func:`wrap_bare_math` folds those bare runs back into ``$…$`` so MathJax
# renders them (and the scorecard counts them as math, not leakage), and strips
# pure-layout LaTeX scaffolding (``\begin{tabular}…`` / ``\hline`` / ``\\`` /
# ``\begin{array}{r}`` / ``\caption{}`` / ``\label{}`` …) that a page-spanning
# table left orphaned in a block. Deterministic; a strict no-op on prose with no
# backslash.
#
# CONSERVATIVE by design (the anti-fabrication contract):
#   * Already-delimited math (``$…$`` / ``$$…$$`` / ``\(…\)`` / ``\[…\]``) is
#     stashed verbatim and never re-wrapped; escaped ``\$`` currency is NOT a
#     delimiter (the ``(?<!\\)`` guard), so ``\$24,493`` never opens a span.
#   * A run is wrapped ONLY when it carries a real math control word
#     (:data:`_BARE_MATH_CMDS`) — a run of bare numbers/letters/operators with no
#     command (ordinary prose arithmetic "3 + 5 = 8") is left untouched.
#   * The identifier atom matches a SINGLE isolated letter (``x`` in ``x^3``),
#     never a multi-letter prose word ("and" / "the"), so prose adjacent to a
#     bare command is not swallowed into the ``$…$`` span.

# Delimited-math spans stashed out before any transform. ``(?<!\\)`` so an
# escaped ``\$`` (literal currency dollar) never opens/closes a span.
# The single-``$`` arm requires NON-EMPTY content (``[^$]+?``) so an orphan
# ``$$`` (an open display delimiter split from its close) is NOT mis-stashed as
# an empty inline span — it falls through to the orphan-``$`` escape below.
_BARE_STASH_MATH_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$[^$]+?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
# An HTML tag (html=True mode operates on rendered ``<p>…</p>`` bodies) — stashed
# so a tag is never mistaken for math content.
_BARE_TAG_RE = re.compile(r"<[^>]+>")
# Pure-layout LaTeX scaffolding dropped outright (never meaningful prose): float
# / tabular / array environments, rules, row breaks, list-item markers, and the
# caption/label/ref/phantom machinery a page-split table leaves orphaned.
_BARE_SCAFFOLD_RE = re.compile(
    r"\\begin\{[^}]*\}(?:\[[^\]]*\])?(?:\{[^}]*\})*"
    r"|\\end\{[^}]*\}"
    r"|\\(?:hline|item|centering|noindent|newline|qquad|quad)\b"
    r"|\\cline\{[^}]*\}"
    r"|\\(?:caption|label|ref|eqref|cite|phantom|vspace|hspace|multicolumn)\{[^}]*\}"
    r"|\\\\"
)
# One level of brace nesting (``\sqrt{x^{6}}`` → ``{x^{6}}``); deeper nesting is
# rare and simply ends the atom early (still safe).
_BARE_BRACE = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
# A math atom: a command, a braced arg, a ``[index]``, an ISOLATED single letter
# (never part of a prose word), a number, or a math operator. ``<`` / ``>`` are
# deliberately excluded so HTML tags / inequality prose never join a run.
_BARE_ATOM = (
    r"(?:\\[a-zA-Z]+"
    rf"|{_BARE_BRACE}"
    r"|\[[^\]]*\]"
    r"|(?<![A-Za-z])[A-Za-z]'?(?![A-Za-z])"
    r"|\d+(?:\.\d+)?"
    r"|[=+\-*/^_(),.])"
)
_BARE_RUN_RE = re.compile(rf"{_BARE_ATOM}(?:[ \t\n]*{_BARE_ATOM})*")
_BARE_CMD_RE = re.compile(r"\\([a-zA-Z]+)")
# The bracket/paren passes forbid a NESTED same-opener inside the span
# (``(?:(?!\\\[).)*?``) so ``\[ … \[ … \]`` does NOT match as one span hiding
# the first (orphan) ``\[`` — the first ``\[`` finds no clean close, falls through
# to the orphan drop, and the well-formed ``\[ … \]`` matches on its own. BOTH the
# display ``$$…$$`` AND the inline ``$…$`` arms are handled by the SAME
# prose/opener-guarded manual scan (:func:`_pair_dollars`), NOT a regex sub — so
# an orphan-derived ``$``/``$$`` opener can never glue onto the next real opener
# and swallow the intervening prose, and an orphan display ``$$`` is dropped in
# the same pass (never reinterpreted as two inline ``$`` downstream).
_BALANCE_PAIR_PASSES = (
    re.compile(r"\\\[(?:(?!\\\[).)*?\\\]", re.DOTALL),           # \[ … \]
    re.compile(r"\\\((?:(?!\\\().)*?\\\)", re.DOTALL),           # \( … \)
)
# Orphan CLOSER / OPENER bracket-paren delimiters left after the pair passes
# (``\)`` / ``\]`` closers split from their open, ``\(`` / ``\[`` openers split
# from their close) — dropped to a space so no literal delimiter ships to the
# learner (Round-7 closer-leak fix; round-5 only covered block-START openers).
_BALANCE_ORPHAN_DELIM_RE = re.compile(r"\\\[|\\\]|\\\(|\\\)")
# Leftover ``$$`` / lone ``$`` after :func:`_pair_dollars` are genuine orphans.
# A lone ``$`` IMMEDIATELY before a digit is CURRENCY (``$5``) and is PRESERVED;
# every other orphan dollar (a ``$$`` remnant, a lone ``$`` not before a digit) is
# DROPPED to a space — never escaped to a visible ``\$`` literal (Round-7). The
# ``(?<!\\)`` guard leaves an already-escaped ``\$`` untouched.
_BALANCE_ORPHAN_DOLLAR_RE = re.compile(r"(?<!\\)\$\$|(?<!\\)\$(?!\d)")
# A STRAY single-letter escape ``\y`` / ``\q`` (an OCR artifact — NOT a real LaTeX
# command; standard LaTeX has no bare-in-prose single-letter LETTER command) folds
# to the bare letter. The ``(?<!\\)`` guard never bites the second ``\`` of a
# ``\\`` row separator, and the ``(?![a-zA-Z])`` tail never truncates a multi-letter
# command (``\sqrt`` / ``\text`` stay whole). Real one-letter commands, if any ever
# matter, would be whitelisted here; there are none today.
_STRAY_ESCAPE_RE = re.compile(r"(?<!\\)\\([a-zA-Z])(?![a-zA-Z])")
_BALANCE_STASH_RE = re.compile("\x01(\\d+)\x01")


def _fold_stray_escapes(s: str) -> str:
    r"""Fold stray single-letter escapes ``\y`` → ``y`` (OCR artifacts).

    Applied only to text where real math spans are already stashed as
    placeholders (so paired-span content — incl. ``\\`` array row separators — is
    never touched) and where any surviving bare command is multi-letter (so a
    genuine ``\sqrt`` bare run is left intact for :func:`wrap_bare_math`). A strict
    no-op on text with no backslash.
    """
    if "\\" not in s:
        return s
    return _STRAY_ESCAPE_RE.sub(r"\1", s)

# Pedagogical opener markers that betray a would-be inline ``$…$`` span's content
# as swallowed PROSE, not math — ``TRY IT`` / ``EXAMPLE`` / ``BE PREPARED`` /
# ``Solution`` / ``Solve`` / ``Simplify`` never occur inside a genuine inline math
# run, so their presence marks the span as an orphan-``$``-glued prose blob.
_MATH_SPAN_OPENER_RE = re.compile(
    r"(?i)\b(?:try\s*it|example|be\s+prepared|how\s+to|learning\s+objectives"
    r"|solution|solve|simplif(?:y|ies))\b"
)
_MATH_SPAN_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")


def _is_prose_math_span(content: str) -> bool:
    """Whether a candidate inline ``$…$`` span's CONTENT reads as prose, not math.

    True when the content carries a pedagogical opener marker
    (:data:`_MATH_SPAN_OPENER_RE`) OR is a MULTI-token run that is majority prose
    words — a token that is PURELY alphabetic (bar trailing punctuation), carries
    a vowel, is >= 2 chars and is not a folded math command-word
    (:data:`MATH_WORDS`). A single-token span is treated as math (a lone
    variable like ``$x$`` / ``$abc$``), and a LaTeX-command / brace / digit token
    (``\\text{to``, ``5p``) never counts as prose, so real inline math
    (``3p - 14 = 5p``, ``\\text{if } x > 0``) stays math. Used to REFUSE pairing
    an orphan-derived ``$`` with the next real math opener (Defect 1).
    """
    if not content or not content.strip():
        return False
    if _MATH_SPAN_OPENER_RE.search(content):
        return True
    tokens = content.split()
    if len(tokens) < 2:  # a lone token is a variable, not swallowed prose
        return False
    prose = 0
    for tok in tokens:
        core = tok.strip(":.;,()[]{}\"'`")
        if (
            core.isalpha()
            and len(core) >= 2
            and _MATH_SPAN_VOWEL_RE.search(core)
            and core.lower() not in MATH_WORDS
        ):
            prose += 1
    return prose * 2 >= len(tokens)


def _find_display_close(text: str, start: int) -> int:
    r"""Index of the next unescaped ``$$`` at/after ``start``, or -1."""
    n = len(text)
    k = start
    while k < n - 1:
        if text[k] == "$" and text[k + 1] == "$" and (k == 0 or text[k - 1] != "\\"):
            return k
        k += 1
    return -1


def _pair_dollars(text: str, stash: list[str]) -> str:
    r"""Pair ``$$…$$`` display AND ``$…$`` inline spans, refusing prose/openers.

    A single manual cursor scan (NOT a regex sub, so the cursor can REWIND) that
    recognizes ``$$`` display delimiters BEFORE their constituent single ``$`` —
    so an orphan display opener can never be reinterpreted as two inline ``$``
    downstream:

    * A ``$$…$$`` / ``$…$`` span whose content is real math is stashed verbatim.
    * A span whose content is prose / an opener marker (:func:`_is_prose_math_span`)
      is REFUSED — its opener is dropped to space(s) and the scan rewinds past the
      opener so the closing delimiter (the real math opener the orphan glued onto)
      is free to start the next span. This is the orphan-``$$`` phantom-math
      prose-swallow fix, now covering the case where the orphan ``$$`` opener
      GLUES onto a later well-formed ``$$…$$`` display pair.
    * An UNMATCHED display opener (a ``$$…$$`` span the cascade SPLIT across a
      block boundary) is dropped to spaces in-pass. An unmatched inline ``$`` is
      left for the currency-aware orphan-dollar drop.

    ``\\$`` (escaped currency) is never treated as a delimiter.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            if i + 1 < n and text[i + 1] == "$":  # display ``$$`` delimiter
                close = _find_display_close(text, i + 2)
                if close != -1:
                    content = text[i + 2 : close]
                    if "\x01" in content or _is_prose_math_span(content):
                        # A ``\x01`` placeholder inside the content means this
                        # ``$$…$$`` is spuriously gluing across an ALREADY-paired
                        # nested ``\(..\)`` / ``\[..\]`` span (currency ``$`` on
                        # either side of a real math cell in an OCR table). Pairing
                        # it would (a) swallow real math and (b) leak the nested
                        # placeholder (single-pass restore can't reach it). Refuse:
                        # drop the opener, rewind — same as a prose refusal.
                        out.append("  ")  # drop orphan/prose opener, rewind
                        i += 2
                        continue
                    stash.append(text[i : close + 2])
                    out.append(f"\x01{len(stash) - 1}\x01")
                    i = close + 2
                    continue
                # Unmatched display opener (split across block boundary) → drop.
                out.append("  ")
                i += 2
                continue
            # inline ``$…$`` delimiter
            j = i + 1
            while j < n and text[j] != "$":
                j += 1
            if j < n and j > i + 1 and text[j - 1] != "\\":
                content = text[i + 1 : j]
                if "\x01" in content or _is_prose_math_span(content):
                    # Orphan-derived opener, prose, OR a span swallowing an
                    # already-paired nested ``\(..\)`` / ``\[..\]`` placeholder
                    # (``\x01`` — currency ``$`` glued across a real math cell in
                    # an OCR table) → drop it; rewind to keep the closing ``$``
                    # available as the next span's opener.
                    out.append(" ")
                    i += 1
                    continue
                stash.append(text[i : j + 1])
                out.append(f"\x01{len(stash) - 1}\x01")
                i = j + 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _balance_math_delimiters(text: str) -> str:
    r"""Neutralize UNMATCHED math delimiters so every block is self-balanced.

    Balanced spans are stashed in SEQUENTIAL passes — ``\[…\]`` / ``\(…\)``
    first (regex), then ``$$…$$`` display AND ``$…$`` inline together via the
    prose/opener-guarded manual pairing (:func:`_pair_dollars`) — so an unmatched
    OPENER (a display span the cascade SPLIT across a block boundary, or an orphan
    ``$$`` that would GLUE onto a later well-formed ``$$…$$`` pair) never poisons
    the pairing of the well-formed spans that follow it, and never drags the
    intervening pedagogical prose into a MathJax-italic span (Defect 1 + Round-7
    closer/mid-block leak). With real math stashed, a STRAY single-letter escape
    (``\y`` — an OCR artifact, NOT valid LaTeX) folds to its bare letter. Whatever
    delimiter is left after all pairs is a genuine orphan: an orphan
    ``\[ \] \( \)`` is DROPPED, an orphan ``$$`` / lone ``$`` is DROPPED too
    (never shipped as a visible literal, never escaped to ``\$``) — EXCEPT a lone
    ``$`` before a digit, which is CURRENCY (``$5``) and is PRESERVED. This keeps
    MathJax from swallowing a whole block on an unclosed delimiter and keeps a
    whole-document ``$``-pairing pass (the cleanliness scan) from desyncing. An
    already-escaped ``\$`` (currency) is untouched.
    """
    has_delim = bool(text) and any(
        d in text for d in ("$", "\\[", "\\]", "\\(", "\\)")
    )
    if not has_delim:
        # No delimiters to balance — but a stray ``\y`` escape may still leak.
        return _fold_stray_escapes(text) if text else (text or "")
    stash: list[str] = []

    def _protect(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return f"\x01{len(stash) - 1}\x01"

    s = text
    for pat in _BALANCE_PAIR_PASSES:
        s = pat.sub(_protect, s)
    # Display ``$$…$$`` + inline ``$…$`` pairing, prose/opener-guarded (rewinds on
    # a refused span; drops orphan display openers in-pass).
    s = _pair_dollars(s, stash)
    # Stray single-letter escapes (real math is now stashed → protected).
    s = _fold_stray_escapes(s)
    # Leftover delimiters are orphans: drop bracket/paren + non-currency dollars.
    s = _BALANCE_ORPHAN_DELIM_RE.sub(" ", s)
    s = _BALANCE_ORPHAN_DOLLAR_RE.sub(" ", s)
    # Iterative restore to a fixpoint: a stashed span's text can itself carry an
    # EARLIER placeholder (a ``$…$`` that legitimately wrapped a nested ``\(..\)``),
    # and a single-pass ``re.sub`` never re-scans its own substitution — so the
    # inner placeholder (a raw ``\x01`` control char) would leak to the browser
    # ("Math input error"). Loop until no placeholder remains; every stash index
    # references an EARLIER entry, so expansion strictly terminates.
    for _ in range(len(stash) + 1):
        if "\x01" not in s:
            break
        s = _BALANCE_STASH_RE.sub(lambda m: stash[int(m.group(1))], s)
    return s


# ---------------------------------------------------------------------------
# HTML-only currency-``$`` escape (2026-07-04 round-7b — MathJax false-pairing).
# ---------------------------------------------------------------------------
# The assembled end-user page enables MathJax v3 with ``inlineMath [['$','$']]``.
# After :func:`_pair_dollars` genuine inline math is delimiter-paired and every
# REMAINING lone ``$`` immediately before a digit is CURRENCY (``$5``) by
# construction. Two such currency amounts in one paragraph ("costs $5 … and $3")
# would FALSE-PAIR into an italic MathJax span at render. :func:`escape_currency_dollars`
# rewrites the currency ``$`` to ``\$`` — a literal dollar under MathJax
# ``processEscapes: true`` — so the pairer never bites.
#
# HTML-ONLY by contract: ``raw_text`` / the sidecar keep plain ``$5`` (that is
# what the chunker + retrieval must index). Genuine ``$…$`` / ``$$…$$`` /
# ``\(…\)`` / ``\[…\]`` math is stashed verbatim first and never escaped, so real
# inline math is untouched. A prose-content ``$…$`` candidate ("$5 … and $3" —
# two currency amounts a naive pairer would glue) is NOT stashed, so BOTH dollars
# are escaped. Idempotent: the ``(?<!\\)`` guard skips an already-escaped ``\$``,
# so re-running on emitted HTML is a fixed point.
_CURRENCY_DOLLAR_RE = re.compile(r"(?<!\\)\$(?=\d)")
_ESCAPE_STASH_RE = re.compile("\x02(\\d+)\x02")


def _is_math_content(content: str) -> bool:
    r"""Whether a balanced ``$…$`` span's CONTENT is genuine math, at escape time.

    At the currency-escape stage the text is ALREADY finalized: :func:`_pair_dollars`
    has decided every span during sanitation, so a balanced ``$…$`` pair in the
    emitted HTML is math MathJax will render. The one shape we still want to
    escape is a pure-currency-prose "span" a naive pairer would glue —
    ``$5 to enter and $3`` (no LaTeX, prose words between two currency amounts).

    So a span is MATH (protect, never escape its delimiters) when it carries a
    backslash LaTeX command (``\text``, ``\frac`` — a decisive math signal, even
    when a prose word like "Solution" also appears inside an answer-key run) OR
    is not prose per :func:`_is_prose_math_span`. It is currency-prose (leave the
    dollars exposed for escape) only when it has NO backslash AND reads as prose.
    """
    return ("\\" in content) or (not _is_prose_math_span(content))


def _stash_genuine_math(text: str, stash: list[str]) -> str:
    r"""Stash genuine ``$$…$$`` / ``$…$`` math as placeholders, NON-destructively.

    Mirrors :func:`_pair_dollars`' ``$$``-display-before-single-``$`` scan but —
    unlike :func:`_pair_dollars`, which DROPS a refused or unmatched ``$`` to a
    space — emits every non-math ``$`` (currency, currency-prose, orphan) VERBATIM
    so the currency escape below can still see it. A balanced span is protected
    per :func:`_is_math_content` (LaTeX-command spans stay math even when a prose
    word appears inside them). Placeholder sentinel ``\x02`` is distinct from the
    ``\x00`` / ``\x01`` sentinels the other passes use.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            if i + 1 < n and text[i + 1] == "$":  # display ``$$`` delimiter
                close = _find_display_close(text, i + 2)
                if close != -1 and _is_math_content(text[i + 2 : close]):
                    stash.append(text[i : close + 2])
                    out.append(f"\x02{len(stash) - 1}\x02")
                    i = close + 2
                    continue
                # orphan / currency-prose display opener → emit the ``$`` literally.
                out.append(ch)
                i += 1
                continue
            # inline ``$…$`` delimiter
            j = i + 1
            while j < n and text[j] != "$":
                j += 1
            if (
                j < n
                and j > i + 1
                and text[j - 1] != "\\"
                and _is_math_content(text[i + 1 : j])
            ):
                stash.append(text[i : j + 1])
                out.append(f"\x02{len(stash) - 1}\x02")
                i = j + 1
                continue
            # currency-prose span / unmatched opener → emit the ``$`` literally.
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def escape_currency_dollars(text: str) -> str:
    r"""Escape a lone currency ``$`` before a digit as ``\$`` (HTML render only).

    Delimited math (``\[…\]`` / ``\(…\)`` regex passes, then genuine ``$$…$$`` /
    ``$…$`` via :func:`_stash_genuine_math`) is stashed verbatim, so ONLY a lone
    currency ``$`` (a ``$`` immediately before a digit that is NOT part of a
    paired math span) is escaped. A strict no-op on text with no ``$``; idempotent
    (``(?<!\\)`` never re-escapes an existing ``\$``).
    """
    if not text or "$" not in text:
        return text or ""
    stash: list[str] = []

    def _protect(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return f"\x02{len(stash) - 1}\x02"

    s = text
    for pat in _BALANCE_PAIR_PASSES:  # \[…\] then \(…\) — never touch their body
        s = pat.sub(_protect, s)
    s = _stash_genuine_math(s, stash)  # genuine $$…$$ / $…$ — currency ``$`` survives
    s = _CURRENCY_DOLLAR_RE.sub(r"\\$", s)
    return _ESCAPE_STASH_RE.sub(lambda m: stash[int(m.group(1))], s)


# ---------------------------------------------------------------------------
# HTML-only raw ``<`` / ``>`` escape INSIDE math spans (2026-07-04 round-8 —
# phantom-tag span break).
# ---------------------------------------------------------------------------
# A math span whose OCR content carries a raw inequality — ``\( x < 5 \)`` /
# ``$ a < b $`` — reaches the assembled learner page with a LITERAL ``<``. The
# browser's HTML tokenizer treats a ``<`` IMMEDIATELY followed by an ASCII
# letter (``\( a<b \)``) as the start of a phantom tag and SWALLOWS everything
# up to the next ``>`` — slicing the ``\(…\)`` span in half. MathJax then sees a
# bare ``\(`` (no close → the delimiter leaks to the reader as a literal
# backslash-paren) and a lone ``\)`` (no open → a RED merror). This is the exact
# "leaked ``\(`` + red ``\)`` + swallowed text" signature; ``<`` before a space
# or digit (``x < 5`` / ``x<5``) is HTML-safe by the same tokenizer rule, which
# is why the artifact is latent until an OCR run emits ``<`` glued to a letter.
#
# :func:`escape_math_angle_brackets` escapes every raw ``<`` / ``>`` that sits
# INSIDE a delimited math span to ``&lt;`` / ``&gt;``. The browser decodes the
# entity back to ``<`` / ``>`` in the text node, so MathJax reads identical math
# and renders byte-identically — but the tokenizer never opens a phantom tag.
#
# HTML-ONLY by contract (the sidecar / chunker text keeps plain ``x < 5`` — that
# is what retrieval must index). Idempotent: an already-escaped ``&lt;`` carries
# no raw ``<``, so re-running is a fixed point. A strict no-op on text with no
# ``<`` and no ``>``. Real HTML tags (``<p>`` / ``<strong>``) live OUTSIDE math
# delimiters and are never matched.
_MATH_SPAN_ANGLE_RE = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$"        # $$…$$ display
    r"|(?<!\\)\$[^$]+?(?<!\\)\$"        # $…$ inline (non-empty)
    r"|\\\(.*?\\\)"                       # \(…\) inline
    r"|\\\[.*?\\\]",                     # \[…\] display
    re.DOTALL,
)


def escape_math_angle_brackets(text: str) -> str:
    r"""Escape raw ``<`` / ``>`` INSIDE delimited math spans (HTML render only).

    Rewrites a literal ``<`` / ``>`` that occurs between math delimiters
    (``$…$`` / ``$$…$$`` / ``\(…\)`` / ``\[…\]``) to ``&lt;`` / ``&gt;`` so the
    browser tokenizer never mistakes an inequality (``\( a<b \)``) for a phantom
    start tag that would swallow the rest of the span. MathJax decodes the entity
    and renders identical math. Prose OUTSIDE math (already ``_esc_text``-escaped)
    and real HTML tags are untouched. A strict no-op on text with no ``<``/``>``;
    idempotent (an existing ``&lt;`` has no raw ``<`` to re-escape).
    """
    if not text or ("<" not in text and ">" not in text):
        return text or ""

    def _esc(m: "re.Match[str]") -> str:
        return m.group(0).replace("<", "&lt;").replace(">", "&gt;")

    return _MATH_SPAN_ANGLE_RE.sub(_esc, text)


# ---------------------------------------------------------------------------
# Math-span CONTENT sanitizer (2026-07-04 round-9 — MathJax typeset errors).
# ---------------------------------------------------------------------------
# The round-9 headless render audit (``scripts/render_audit.py``) surfaced
# ``mjx-merror`` nodes that NO text audit caught — genuine MathJax *typeset*
# failures that only appear after the browser runs. Two OCR/VLM-emission
# families dominate, both INSIDE already-delimited math spans (so the bare-math
# wrap / delimiter-balance passes above never see them):
#
#  (1) "Misplaced &" — the VLM pulled a textbook TABLE into a single ``$…$`` /
#      ``\(…\)`` run, so a tabular column separator ``&`` (or its ``&amp;``
#      entity, decoded by the browser to a literal ``&`` before MathJax reads
#      it) sits in math with NO alignment environment. Table-cell debris also
#      shows up as a lone ``$ & $`` span. MathJax errors because ``&`` is only
#      legal as an alignment tab inside ``\begin{aligned|array|cases|…}``.
#  (2) "Missing argument for \sqrt" (and the identical-family ``\frac`` /
#      ``\stackrel`` / trailing ``^``/``_``) — OCR truncated the radicand /
#      operand, leaving the command dangling at the span's end.
#
# :func:`sanitize_math_spans` operates ONLY on the CONTENT between math
# delimiters (reusing :data:`_MATH_SPAN_ANGLE_RE`) and is CONSERVATIVE by
# contract (anti-fabrication):
#   * A ``&`` is dropped to a space ONLY when the span carries no alignment
#     environment (:data:`_MATH_ALIGN_ENV_RE`). A well-formed
#     ``\begin{array}…&…\end{array}`` keeps every column ``&`` verbatim.
#   * ``&lt;`` / ``&gt;`` / ``&#NN;`` / other named entities are NEVER touched —
#     they decode to real operators/glyphs MathJax renders (an inequality
#     ``a &lt; b`` is valid math, not a misplaced tab). Only a raw ``&`` and the
#     ``&amp;`` entity (both → a literal alignment ``&``) are folded.
#   * A dangling ``\sqrt`` / ``\frac`` / ``\stackrel`` (optionally with a
#     ``[index]`` or empty ``{}``) or a trailing ``^`` / ``_`` at the span END
#     is dropped. A VALID ``\sqrt{x}`` / ``\sqrt[3]{x}`` / single-token
#     ``\sqrt 2`` (unbraced single char is legal LaTeX) is untouched because the
#     command is NOT at the span end. If a span sanitizes to empty, its now-bare
#     delimiters are dropped too (no empty ``$$ $$`` shipped).
# HTML-ONLY by contract (mirrors the currency / angle-bracket passes): the
# chunker/retrieval ``raw_text`` keeps the plain fused text; only the rendered
# page is repaired.

# Alignment / tabular environments where ``&`` is a legitimate column tab.
_MATH_ALIGN_ENV_RE = re.compile(
    r"\\begin\s*\{\s*(?:aligned|align|alignat|array|cases|matrix"
    r"|[bBpvV]matrix|smallmatrix|split|gathered|gather|multline"
    r"|eqnarray|subarray|CD)\*?\s*\}"
)
# A misplaced alignment ``&``: a raw ``&`` or the ``&amp;`` entity (both decode
# to a literal ``&``). The negative lookahead SPARES ``&lt;`` / ``&gt;`` /
# ``&#NN;`` / ``&#xNN;`` / any other ``&name;`` entity — those are real glyphs.
_MATH_STRAY_AMP_RE = re.compile(r"&amp;|&(?![A-Za-z]+;|#\d+;|#[xX][0-9A-Fa-f]+;)")
# A command that REQUIRES a following argument, left dangling at the span END
# (OCR truncated the operand). ``\sqrt`` / ``\cbrt`` optionally carry a
# ``[index]``; an empty ``{}`` counts as dangling too. ``\Z`` anchors the very
# end of the (delimiter-stripped) span content.
_DANGLING_ARGCMD_RE = re.compile(
    r"\\(?:sqrt|cbrt|frac|dfrac|tfrac|cfrac|stackrel|overset|underset)"
    r"(?:\s*\[[^\]]*\])?\s*(?:\{\s*\})?\s*\Z"
)
# A ``\frac`` / ``\dfrac`` (…) left with only its FIRST braced argument at the
# span end — the second operand was OCR-truncated (``\frac{1}`` → "Missing
# argument for \frac"). Dropped whole; a well-formed ``\frac{1}{2}`` is NOT at
# the span end (``{2}`` follows) so it never matches.
_DANGLING_FRAC1_RE = re.compile(r"\\(?:frac|dfrac|tfrac|cfrac)\s*\{[^{}]*\}\s*\Z")
# A ``\frac``-family command that heads the WHOLE span and carries only ONE
# balanced brace group (with arbitrary NESTING) and nothing after it — the OCR
# double-wrapped a complete fraction (``\dfrac{\dfrac{a}{b}}`` → "Missing
# argument for \dfrac"). Unlike ``_DANGLING_FRAC1_RE`` (no-nesting ``[^{}]*``),
# this is brace-depth-aware (:func:`_unwrap_single_arg_frac`) so it fires on a
# nested single arg; a VALID two-arg ``\dfrac{a}{b}`` has a second group after
# the first, so it is never unwrapped.
_SINGLE_ARG_FRAC_HEAD_RE = re.compile(r"^\s*\\(?:d|t|c)?frac\s*")
# A trailing superscript / subscript operator with no operand (``x^`` / ``a_``
# at the span end). The ``(?<!\\)`` lookbehind SPARES an ESCAPED ``\_`` / ``\^``
# — a LITERAL underscore / caret (an OCR fill-in-the-blank ``$$x^2 + \_\_ $$``),
# NOT a dangling operator. Without it the trailing ``_`` of ``\_`` was stripped,
# leaving a dangling ``\`` that ESCAPED the closing ``$$`` / ``$`` delimiter and
# either leaked a literal ``$$`` (display close broken) or orphaned the inline
# ``$`` open (desync that swallowed the following bare table ``&`` → a
# "Misplaced &" ``mjx-merror``). A BARE ``x^`` / ``a_`` (no preceding ``\``) is
# still a real dangling operator and is dropped.
_DANGLING_SUPSUB_RE = re.compile(r"(?<!\\)[\^_]\s*\Z")
# An environment opener (``\begin{array}{|c|c|}`` — with its optional column /
# option spec groups) and closer, for the unbalanced-env drop.
_MATH_BEGIN_RE = re.compile(
    r"\\begin\s*\{\s*([A-Za-z*]+)\s*\}(?:\s*\[[^\]]*\]|\s*\{[^{}]*\})*"
)
_MATH_END_RE = re.compile(r"\\end\s*\{\s*([A-Za-z*]+)\s*\}")
# A ``\left`` / ``\right`` sizing command (never part of a longer control word —
# the ``(?![A-Za-z])`` guard spares ``\lefteqn`` / ``\rightarrow``).
_MATH_LEFTRIGHT_RE = re.compile(r"\\(?:left|right)(?![A-Za-z])")
# A ``\hline`` / ``\cline`` rule left orphaned once its array env is gone
# (a bare ``\\`` row break is NOT dropped — it is a legal inline/display line
# break outside an environment and dropping it would reflow valid math).
_MATH_ROW_SCAFFOLD_RE = re.compile(r"\\hline\b|\\cline\s*\{[^}]*\}")


def _split_math_delims(span: str) -> "tuple[str, str, str]":
    r"""Split a delimited math ``span`` into ``(open, inner, close)``.

    ``span`` is one whole match of :data:`_MATH_SPAN_ANGLE_RE` — a ``$$…$$`` /
    ``$…$`` / ``\(…\)`` / ``\[…\]`` run — so exactly one of the four delimiter
    pairs brackets it.
    """
    if span.startswith("$$"):
        return "$$", span[2:-2], "$$"
    if span.startswith("$"):
        return "$", span[1:-1], "$"
    if span.startswith("\\("):
        return "\\(", span[2:-2], "\\)"
    return "\\[", span[2:-2], "\\]"


def _match_balanced_brace(s: str, start: int) -> int:
    r"""Index just past the ``}`` matching the ``{`` at ``s[start]``, or -1.

    Depth-aware so a nested ``\dfrac{a}{b}`` inside the group is spanned whole.
    """
    if start >= len(s) or s[start] != "{":
        return -1
    depth = 0
    for k in range(start, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


def _unwrap_single_arg_frac(inner: str) -> str:
    r"""Unwrap an OCR-double-wrapped single-arg ``\dfrac{…}`` to its brace body.

    ``\dfrac{\dfrac{a}{b}}`` (a ``\frac``-family command heading the whole span
    with ONE balanced brace group and nothing after) is missing its second
    argument → MathJax errors "Missing argument for \dfrac". The bogus outer
    command keyword is dropped, keeping its (already-valid) brace content. A
    genuine two-arg ``\dfrac{a}{b}`` has a second group after the first and is
    left untouched. Iterates so a triple-wrap collapses fully.

    Unwrap fires ONLY when the single brace body is ITSELF a ``\frac``-family
    expression (the OCR double-wrapped a complete fraction). A truncated
    ``\frac{1}`` (single arg is a bare atom, not a nested fraction) is left for
    :func:`_drop_dangling_commands` to DROP — unwrapping it to a bare ``1`` would
    fabricate content the truncation destroyed.
    """
    while True:
        m = _SINGLE_ARG_FRAC_HEAD_RE.match(inner)
        if not m:
            return inner
        p = m.end()
        end = _match_balanced_brace(inner, p)
        if end == -1:
            return inner
        if inner[end:].strip() != "":  # a second arg (or trailing math) follows
            return inner
        body = inner[p + 1 : end - 1].strip()
        if not _SINGLE_ARG_FRAC_HEAD_RE.match(body):
            # Single arg is not itself a fraction → a genuine truncation, not a
            # double-wrap. Leave it for the dangling-command drop.
            return inner
        inner = body  # drop the bogus outer keyword, unwrap the nested fraction


def _drop_dangling_commands(inner: str) -> str:
    r"""Drop argument-requiring commands / sup-sub left dangling at span end.

    Iterates to a fixpoint: dropping the trailing ``\sqrt`` in ``\frac \sqrt``
    exposes the now-trailing ``\frac``, which is dropped on the next turn.
    """
    while True:
        new = _DANGLING_ARGCMD_RE.sub("", inner)
        new = _DANGLING_FRAC1_RE.sub("", new)
        new = _DANGLING_SUPSUB_RE.sub("", new)
        if new == inner:
            # No drop this turn — preserve the span's original whitespace.
            return inner
        # A command was dropped; trim the space that preceded it so a later
        # turn can see a newly-exposed trailing command.
        inner = new.rstrip()


def _drop_unbalanced_envs(inner: str) -> str:
    r"""Drop orphan ``\begin{env}`` / ``\end{env}`` when their counts disagree.

    An OCR-truncated ``$$\begin{aligned} $$`` (opener with no ``\end``) errors
    "Missing \end{aligned}". When a span's ``\begin{X}`` and ``\end{X}`` counts
    differ, that environment is broken debris: every ``\begin{X}`` (with its
    column/option spec) AND ``\end{X}`` for the mismatched name ``X`` is dropped,
    leaving any real content. A WELL-FORMED environment (matched counts) is
    untouched, so a valid ``\begin{array}…\end{array}`` — including its ``&``
    column tabs — survives intact.
    """
    if "\\begin" not in inner and "\\end" not in inner:
        return inner
    from collections import Counter

    begins = Counter(m.group(1) for m in _MATH_BEGIN_RE.finditer(inner))
    ends = Counter(m.group(1) for m in _MATH_END_RE.finditer(inner))
    bad = {n for n in set(begins) | set(ends) if begins[n] != ends[n]}
    if not bad:
        return inner
    inner = _MATH_BEGIN_RE.sub(
        lambda m: "" if m.group(1) in bad else m.group(0), inner
    )
    inner = _MATH_END_RE.sub(
        lambda m: "" if m.group(1) in bad else m.group(0), inner
    )
    return inner


def _strip_unbalanced_leftright(inner: str) -> str:
    r"""Strip ``\left`` / ``\right`` sizing commands when they don't balance.

    ``(a) \left(9p`` (an OCR-truncated span with a ``\left(`` and no ``\right``)
    errors "Extra \left or missing \right". When the ``\left`` / ``\right``
    counts disagree the span can't render as sized-delimiter math, so both
    commands are stripped — the bare ``(`` / ``)`` glyphs still render. A
    balanced ``\left(…\right)`` (equal counts) is untouched.
    """
    if inner.count("\\left") == inner.count("\\right"):
        return inner
    return _MATH_LEFTRIGHT_RE.sub("", inner)


def sanitize_math_spans(text: str) -> str:
    r"""Fold misplaced ``&`` + dangling ``\sqrt`` family out of math spans (round-9).

    Repairs the two dominant MathJax ``mjx-merror`` families the headless render
    audit surfaced — a tabular ``&`` inside non-alignment math ("Misplaced &")
    and an OCR-truncated ``\sqrt`` / ``\frac`` / ``\stackrel`` / trailing
    ``^``/``_`` ("Missing argument"). Operates ONLY on the content between math
    delimiters; conservative per the module docstring (alignment ``&`` and valid
    ``\sqrt{x}`` are untouched). A span that sanitizes to empty loses its bare
    delimiters. A fast guard returns text with no ``&`` / backslash / ``^`` /
    ``_`` unchanged.
    """
    if not text or not any(t in text for t in ("&", "\\", "^", "_")):
        return text or ""

    def _fix(m: "re.Match[str]") -> str:
        o, inner, c = _split_math_delims(m.group(0))
        # Drop orphan environments FIRST so a broken ``\begin{array}`` no longer
        # shields its ``&`` tabs from the misplaced-``&`` fold below.
        inner = _drop_unbalanced_envs(inner)
        inner = _unwrap_single_arg_frac(inner)
        inner = _drop_dangling_commands(inner)
        inner = _strip_unbalanced_leftright(inner)
        # Misplaced ``&`` (+ orphan row scaffolding) ONLY when the span has no
        # alignment environment left.
        if not _MATH_ALIGN_ENV_RE.search(inner):
            inner = _MATH_STRAY_AMP_RE.sub(" ", inner)
            inner = _MATH_ROW_SCAFFOLD_RE.sub(" ", inner)
            inner = re.sub(r"[ \t]{2,}", " ", inner)
        if not inner.strip():
            return ""  # nothing left → drop the now-empty delimiters
        return f"{o}{inner}{c}"

    return _MATH_SPAN_ANGLE_RE.sub(_fix, text)


# ---------------------------------------------------------------------------
# TikZ / pgfplots FIGURE-code graceful fallback (2026-07-04 round-10 — final).
# ---------------------------------------------------------------------------
# The round-10 headless render audit surfaced a NEW ``mjx-merror`` family the
# round-9 span sanitizer does not touch: the VLM transcribed coordinate-plane
# FIGURES (the "Identify the x- and y-intercepts on a graph" exercises) as raw
# TikZ picture code INSIDE math delimiters —
# ``$$\begin{tikzpicture} \draw[...] ... \end{tikzpicture}$$`` (and the pgfplots
# ``\begin{axis}…\end{axis}`` sibling). MathJax has no TikZ support, so it emits
# a red "Undefined environment tikzpicture" error for every such block (12 on
# ch04). This is FIGURE content, not math: the corpus-wide figure story is the
# accepted scan-corpus limitation (``SEMANTIK_DETECT_FIGURES`` off), so the
# honest treatment is an accessible PLACEHOLDER, not a render attempt — and the
# raw TikZ source (``\draw`` / ``grid`` / ``coordinates`` noise) must NOT ship
# visibly to any reader.
#
# :func:`strip_tikz_figures` operates ONLY on the CONTENT between math delimiters
# (reusing :data:`_MATH_SPAN_ANGLE_RE`, mirroring the round-8/9 passes) and is
# CONSERVATIVE by contract:
#   * A span is only touched when it actually carries a TikZ / pgfplots
#     environment (:data:`_TIKZ_PRESENT_RE`) — ``\begin{aligned}`` /
#     ``\begin{array}`` and every other math env are left completely alone (no
#     false fire).
#   * A PURE-figure span (the whole delimited run is one tikzpicture/axis block,
#     with no surviving real math after the env is stripped) is replaced by the
#     accessible placeholder :data:`_TIKZ_FIGURE_PLACEHOLDER` — the delimiters go
#     with it (no empty ``$$ $$`` left behind, no visible TikZ source).
#   * A MIXED span (real math + an embedded TikZ block) keeps its math: only the
#     tikzpicture/axis environment is stripped, the delimiters + surviving math
#     are preserved.
#   * An OCR-TRUNCATED orphan opener (``\begin{tikzpicture}`` with no matching
#     ``\end``) is figure debris to the span end and is dropped too.
# HTML-ONLY by contract (mirrors the currency / angle-bracket / round-9 passes):
# the chunker/retrieval ``raw_text`` / sidecar keep the plain fused text (TikZ
# and all) for retrieval fidelity; only the rendered learner page is repaired.
# Idempotent: the emitted placeholder carries no ``\begin{tikz…}`` marker, so a
# second pass is a strict no-op.

# TikZ / pgfplots environment names the VLM emits as figure code. The
# backreference in :data:`_TIKZ_ENV_RE` lets an OUTER ``\begin{tikzpicture}``
# wrapping a nested ``\begin{axis}…\end{axis}`` match to its OWN ``\end`` (the
# nested ``\end{axis}`` is skipped because the backref pins the env name).
_TIKZ_ENV_NAMES = r"tikzpicture|axis|pgfplots|pgfpicture|tikzcd"
_TIKZ_PRESENT_RE = re.compile(r"\\begin\s*\{\s*(?:" + _TIKZ_ENV_NAMES + r")\s*\}")
_TIKZ_ENV_RE = re.compile(
    r"\\begin\s*\{\s*(" + _TIKZ_ENV_NAMES + r")\s*\}.*?\\end\s*\{\s*\1\s*\}",
    re.DOTALL,
)
# An OCR-truncated orphan opener (a ``\begin{tikzpicture}`` whose ``\end`` was
# cut off) — figure debris from the opener to the span's end.
_TIKZ_ORPHAN_RE = re.compile(
    r"\\begin\s*\{\s*(?:" + _TIKZ_ENV_NAMES + r")\s*\}.*", re.DOTALL
)
_TIKZ_FIGURE_PLACEHOLDER = (
    '<span class="dart-figure-notation" role="img" '
    'aria-label="Coordinate-plane figure (notation could not be rendered)">'
    "[coordinate-plane figure]</span>"
)


def strip_tikz_figures(text: str) -> str:
    r"""Replace TikZ/pgfplots figure code in math spans with an a11y placeholder.

    The final round-10 visual-convergence pass. A delimited math span carrying a
    ``\begin{tikzpicture}…\end{tikzpicture}`` (or pgfplots ``\begin{axis}…``)
    block is a coordinate-plane FIGURE the VLM mis-transcribed as raw TikZ code —
    MathJax reds it ("Undefined environment tikzpicture"). A PURE-figure span is
    replaced whole by the accessible :data:`_TIKZ_FIGURE_PLACEHOLDER` (delimiters
    and TikZ source both gone — the source is noise to every reader). A MIXED
    span (real math + an embedded figure) keeps its math and drops only the
    figure env. Conservative per the module docstring: a span with no TikZ /
    pgfplots environment is untouched (``\begin{aligned}`` never fires). HTML-only
    (``raw_text``/sidecar keep the plain fused text). Idempotent; a fast guard
    returns text with no TikZ begin unchanged.
    """
    if not text or not _TIKZ_PRESENT_RE.search(text):
        return text or ""

    def _fix(m: "re.Match[str]") -> str:
        o, inner, c = _split_math_delims(m.group(0))
        if not _TIKZ_PRESENT_RE.search(inner):
            return m.group(0)  # no figure code in THIS span — leave it alone
        stripped = _TIKZ_ENV_RE.sub(" ", inner)      # balanced env(s)
        stripped = _TIKZ_ORPHAN_RE.sub(" ", stripped)  # OCR-truncated opener
        stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
        if not stripped:
            return _TIKZ_FIGURE_PLACEHOLDER  # pure figure → placeholder, no delims
        return f"{o}{stripped}{c}"  # mixed → keep the surviving real math

    return _MATH_SPAN_ANGLE_RE.sub(_fix, text)


# ---------------------------------------------------------------------------
# Markdown-image sanitizer (VLM fabricated-image-URL boundary — scan lane).
# ---------------------------------------------------------------------------
# The Qwen2.5-VL scan-lane transcriber sometimes INVENTS a Markdown image
# ``![alt](url)`` for a figure it cannot transcribe — the ``url`` is a
# hallucinated external host (``i.imgur.com`` / ``example.com`` / a bare
# relative ``image.png``) that resolves to nothing, poisons the assembled page
# with a broken external ``<img>`` (WCAG 1.1.1 content loss + a CSP-violating
# off-origin fetch), and — if it reached ``_linkify_block_urls`` — would be
# turned into a live ``<a href>`` anchor to the fabricated host. This is figure
# content the cascade cannot recover, so — mirroring :func:`strip_tikz_figures`
# — EVERY image (any host, any relative filename, URL-agnostic) is replaced by
# the accessible ``.dart-figure-notation`` placeholder. Runs BEFORE the
# linkifier so a fabricated URL never becomes a clickable anchor.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\(\s*(?P<target>[^)]*)\)")


def _figure_placeholder(alt: str, *, html: bool) -> str:
    r"""The shared accessible not-recoverable-figure placeholder.

    HTML mode → the ``.dart-figure-notation`` span (``alt`` HTML-escaped into the
    ``aria-label``); plain mode → a bare ``[figure: {alt}]``. The single source
    of truth for both :func:`strip_markdown_images` and
    :func:`strip_literal_img_tags` so the two fabricated-image sanitizers emit an
    identical placeholder.
    """
    alt = (alt or "").strip() or "Figure"
    if not html:
        return f"[figure: {alt}]"
    label = _html_escape(f"{alt} (image not recoverable)", quote=True)
    return (
        '<span class="dart-figure-notation" role="img" '
        f'aria-label="{label}">[figure]</span>'
    )


def strip_markdown_images(text: str, *, html: bool = True) -> str:
    r"""Replace EVERY Markdown image ``![alt](target)`` with an a11y placeholder.

    The VLM scan lane invents Markdown image syntax pointing at hallucinated
    image URLs (external host or bare relative filename); the target resolves to
    nothing and must never ship as a broken ``<img>`` / live ``<a href>`` to the
    fabricated host. URL-agnostic: any ``target`` (``https://i.imgur.com/…``,
    ``image.png``, ``example.com/x``) is stripped. In HTML mode the match is
    replaced by the accessible ``.dart-figure-notation`` placeholder span
    (mirroring :data:`_TIKZ_FIGURE_PLACEHOLDER`), the ``alt`` HTML-escaped into
    the ``aria-label``; in plain mode (``raw_text`` / sidecar / chunker text) by
    a bare ``[figure]`` placeholder carrying the alt — so the fabricated URL
    reaches neither the learner page nor the retrieval index. A fast guard
    returns text with no ``![`` unchanged; idempotent (the emitted placeholder
    carries no ``![`` marker, so re-application is a no-op).
    """
    if not text or "![" not in text:
        return text or ""

    def _fix(m: "re.Match[str]") -> str:
        return _figure_placeholder(m.group("alt") or "", html=html)

    return _MARKDOWN_IMAGE_RE.sub(_fix, text)


# ---------------------------------------------------------------------------
# Literal / escaped ``<img>`` tag-TEXT sanitizer (VLM fabricated-image boundary,
# 2026-07-06 census — RAW <img> in table cells the markdown strip never saw).
# ---------------------------------------------------------------------------
# The Qwen2.5-VL scan lane also emits a RAW HTML ``<img src="https://i.imgur.com/
# 1.png">`` tag as LITERAL TEXT inside a table cell / prose (not markdown
# ``![]()``). It arrives at the adapter either ESCAPED (``&lt;img src=&quot;…
# &quot;&gt;`` — the HTML-emit path escapes ``<``/``"``) or as a plain literal
# ``<img …>`` string in ``raw_text``, so the markdown-image regex never matches
# and :func:`linkify_urls` then turns the fabricated URL into a live ``<a href>``.
# :func:`strip_literal_img_tags` removes such tag TEXT — mirroring the markdown
# strip — replacing it with the SAME accessible placeholder, but ONLY when the
# ``src`` is external (http/https/protocol-relative) OR a bare fabricated image
# filename (no directory + an image extension). A REAL, DOM-level figure ``<img>``
# the cascade emits carries a LOCAL relative path WITH a directory
# (``{stem}_figures/fig-N.png`` → has a ``/`` and no scheme), so it is never
# matched — this pass is a text sanitizer over fabricated tag text, never a
# rewrite of legitimate image elements. Runs at the SAME adapter seam as the
# markdown strip (before the linkifier). Idempotent (the placeholder carries no
# ``img`` token) + fast-guarded.
_LITERAL_IMG_TAG_RE = re.compile(
    r"(?:&lt;|<)\s*img\b(?P<attrs>(?:(?!&gt;|>).)*)(?:&gt;|>)",
    re.IGNORECASE | re.DOTALL,
)
# ``src`` / ``alt`` value, tolerant of double / single / entity-escaped
# (``&quot;`` / ``&#39;``) / unquoted quoting.
_IMG_ATTR_SRC_RE = re.compile(
    r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|&quot;(.*?)&quot;|&#0*39;(.*?)&#0*39;"""
    r"""|([^\s>&]+))""",
    re.IGNORECASE,
)
_IMG_ATTR_ALT_RE = re.compile(
    r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)'|&quot;(.*?)&quot;|&#0*39;(.*?)&#0*39;"""
    r"""|([^\s>&]+))""",
    re.IGNORECASE,
)
# External: an explicit http/https scheme or a protocol-relative ``//host``.
_IMG_EXTERNAL_SRC_RE = re.compile(r"^\s*(?:https?:)?//", re.IGNORECASE)
# Bare fabricated filename: NO directory component + a recognizable image
# extension (``image.png`` / ``chart.jpg``) — a real local figure src always
# carries a ``{stem}_figures/`` directory, so a bare filename is VLM debris.
_IMG_BARE_FABRICATED_RE = re.compile(
    r"^[^/\s]+\.(?:png|jpe?g|gif|svg|webp|bmp|tiff?)\s*$", re.IGNORECASE
)


def _img_attr_value(rx: "re.Pattern[str]", attrs: str) -> str | None:
    m = rx.search(attrs or "")
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return g
    return None


def _is_fabricated_img_src(src: str | None) -> bool:
    """Whether an ``<img>`` ``src`` is an external ref or a bare fabricated file.

    True for an ``http(s)://`` / protocol-relative ``//`` src, or a bare
    image filename with no directory. False for an empty src or a real local
    relative path with a directory (``{stem}_figures/fig-N.png``) — those are
    legitimate DOM image elements and must never be stripped.
    """
    if not src:
        return False
    s = src.strip()
    if not s:
        return False
    if _IMG_EXTERNAL_SRC_RE.match(s):
        return True
    return "/" not in s and bool(_IMG_BARE_FABRICATED_RE.match(s))


def strip_literal_img_tags(text: str, *, html: bool = True) -> str:
    r"""Replace literal / escaped ``<img>`` tag TEXT with an a11y placeholder.

    Handles the RAW / escaped ``<img …>`` tag the VLM emits as literal text (a
    fabricated external / bare-filename image the markdown strip never sees):
    ESCAPED ``&lt;img src=&quot;https://…&quot; /&gt;``, unescaped literal
    ``<img src=https://… >`` in ``raw_text``, self-closing and not, single /
    double / no quotes. The ``src`` is external (http/https/protocol-relative) or
    a bare fabricated filename → the tag is replaced by :func:`_figure_placeholder`
    (preserving any ``alt``); otherwise (a real local ``{stem}_figures/…`` src,
    or no/empty src) the tag TEXT is left byte-identical. A fast guard returns
    text with no ``img`` token unchanged; idempotent (the placeholder carries no
    ``img``).
    """
    if not text:
        return text or ""
    low = text.lower()
    if "<img" not in low and "&lt;img" not in low:
        return text

    def _fix(m: "re.Match[str]") -> str:
        attrs = m.group("attrs") or ""
        src = _img_attr_value(_IMG_ATTR_SRC_RE, attrs)
        if not _is_fabricated_img_src(src):
            return m.group(0)  # real/legit DOM img (or no src) → untouched
        alt_raw = _img_attr_value(_IMG_ATTR_ALT_RE, attrs)
        alt = _html_unescape(alt_raw).strip() if alt_raw else ""
        return _figure_placeholder(alt, html=html)

    return _LITERAL_IMG_TAG_RE.sub(_fix, text)


# ---------------------------------------------------------------------------
# Adjacent inline-math separation (2026-07-04 round-11 — scorecard strip-desync).
# ---------------------------------------------------------------------------
# An exercise list emits several math atoms back-to-back with no separator —
# ``Simplify $4^3$ $7^1$ …`` fuses to ``$4^3$$7^1$…``. The ``$$`` at each
# close-then-open junction is a FALSE display delimiter: MathJax's left-to-right
# open-time scan still pairs them as two adjacent INLINE spans (so the headless
# render is clean), but a downstream ``$$``-display-FIRST strip (the structure
# scorecard's cleanliness math-strip) mis-pairs across the whole block — it
# consumes the odd/even junction ``$$`` pairs and STRANDS the intervening span
# CONTENT (``\left \frac \right``…) as counted "LaTeX leakage" (ch06: 224 phantom
# leaks → cleanliness 0.0 despite a clean browser render).
#
# :func:`separate_adjacent_math_spans` inserts a single space at every inline
# close-``$`` immediately followed by an opening ``$`` (or ``$$``), mirroring
# MathJax's own open-time decision: a genuine ``$$…$$`` DISPLAY span (``$$`` with
# a matching display close) is recognized and passed through verbatim, so only
# the false close+open junctions gain a separator. The rendered math is
# byte-identical (two adjacent inline spans with vs. without an inter-span space
# typeset the same), so this never changes an output glyph — it only removes the
# lexical ambiguity a naive display-first pairer trips on. HTML-only by contract
# (mirrors the currency / angle / round-9/10 passes); idempotent (a separated
# ``$a$ $b$`` has no ``$``-adjacent-``$`` junction left); fast-guarded on ``$``.
def separate_adjacent_math_spans(text: str) -> str:
    r"""Insert a space between adjacent inline math spans (``$a$$b$`` → ``$a$ $b$``).

    Walks ``text`` mirroring MathJax's open-time delimiter decision: a genuine
    ``$$…$$`` display span (opening ``$$`` with a matching display close) is
    emitted verbatim; an inline ``$…$`` span whose closing ``$`` is immediately
    followed by another opening ``$`` / ``$$`` gets a single separating space, so
    the false close+open ``$$`` junction never reads as a display delimiter to a
    downstream ``$$``-first strip. Renders byte-identically under MathJax. A
    strict no-op on text with fewer than two ``$``.
    """
    if not text or text.count("$") < 2:
        return text or ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            if i + 1 < n and text[i + 1] == "$":  # potential display ``$$``
                close = _find_display_close(text, i + 2)
                if close != -1:  # genuine display span — pass through verbatim
                    out.append(text[i : close + 2])
                    i = close + 2
                    continue
                out.append("$$")  # unmatched (post-balance this is rare) — keep
                i += 2
                continue
            # inline ``$…$``
            j = i + 1
            while j < n and text[j] != "$":
                j += 1
            if j < n and j > i + 1 and text[j - 1] != "\\":
                out.append(text[i : j + 1])
                i = j + 1
                # A close ``$`` immediately followed by an opening ``$`` (the
                # next span / a display open) → separate so no false ``$$`` forms.
                if i < n and text[i] == "$" and text[i - 1] != "\\":
                    out.append(" ")
                continue
        out.append(ch)
        i += 1
    return "".join(out)


# Control words whose presence marks a run as genuine math worth wrapping.
_BARE_MATH_CMDS: frozenset[str] = frozenset(
    {
        "sqrt", "cbrt", "frac", "dfrac", "tfrac", "cfrac", "cdot", "times",
        "div", "pm", "mp", "neq", "ne", "geq", "ge", "leq", "le", "approx",
        "equiv", "infty", "sum", "prod", "int", "pi", "theta", "alpha", "beta",
        "gamma", "delta", "lambda", "sigma", "omega", "mu", "nu", "left",
        "right", "cdots", "ldots", "vdots", "ddots", "overline", "underline",
        "vec", "hat", "bar", "mathbf", "mathrm", "mathcal", "boldsymbol",
        "partial", "nabla", "stackrel", "text", "phantom",
    }
)


def wrap_bare_math(text: str, *, html: bool = True) -> str:
    r"""Wrap bare (un-delimited) LaTeX math runs in ``$…$`` and drop scaffolding.

    ``html=True`` protects HTML tags (operates on a rendered ``<p>…</p>`` body);
    ``html=False`` operates on plain chunk/sidecar text. In BOTH modes: existing
    delimited math is preserved verbatim, layout scaffolding
    (:data:`_BARE_SCAFFOLD_RE`) is dropped, and a maximal run of math atoms
    carrying a real math command (:data:`_BARE_MATH_CMDS`) is wrapped in ``$…$``.
    A strict no-op on text with no backslash (fast guard).
    """
    if not text:
        return text or ""
    # Neutralize orphan / cross-block-split delimiters FIRST so the block is
    # self-balanced before anything is stashed or wrapped.
    text = _balance_math_delimiters(text)
    if "\\" not in text:
        return re.sub(r"[ \t]{2,}", " ", text)
    stash: list[str] = []

    def _protect(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    s = _BARE_STASH_MATH_RE.sub(_protect, text)
    if html:
        s = _BARE_TAG_RE.sub(_protect, s)
    s = _BARE_SCAFFOLD_RE.sub(" ", s)

    def _wrap(m: "re.Match[str]") -> str:
        run = m.group(0)
        cmds = _BARE_CMD_RE.findall(run)
        if not any(c.lower() in _BARE_MATH_CMDS for c in cmds):
            return run
        stripped = run.strip()
        return f" ${stripped}$ " if stripped else run

    s = _BARE_RUN_RE.sub(_wrap, s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)
    return re.sub(r"[ \t]{2,}", " ", s)


# ---------------------------------------------------------------------------
# Bare / angle-wrapped URL linkification (2026-07-04 round-2 audit — ITEM 1).
# ---------------------------------------------------------------------------
# Scanned textbooks carry vendor MEDIA links ("Access these online resources
# … Product Property (https://vendor.example/l/25ProductProp)") as bare or
# angle-bracket-wrapped URLs in block prose. In the assembled learner page these
# render as MathJax-italic soup — MathJax's global ``$``-delimiter scan (an
# orphan display ``$$`` split across a block boundary) swallows the surrounding
# text, so ``<https://vendor.example/l/25AddSubtrHR>`` reads as spaced italic
# "< https : //…>". :func:`linkify_urls` turns every bare URL into a real
# ``<a href>`` anchor stamped ``mathjax_ignore`` so MathJax NEVER typesets the
# link subtree (the assembler's ``ignoreHtmlClass`` catches it), and DROPS the
# ``<…>`` / ``&lt;…&gt;`` angle wrapper and any ``$`` the fusion glued directly
# onto the URL. Deterministic; a strict no-op on text with no ``http``.
#
# CONSERVATIVE by design:
#   * The scheme tolerates OCR spacing (``https : //`` / ``http : / /``) and is
#     normalized to a canonical ``https://`` in the emitted href + text.
#   * The URL body is a conservative RFC-ish char class that STOPS before a
#     trailing sentence period, an ``&gt;`` entity, a ``)`` / whitespace — so a
#     URL at a sentence end ("… /25SquareRoots. Next") never eats the period.
#   * ``html=True`` emits the ``<a>`` anchor (rendered body); ``html=False``
#     emits the bare normalized URL (plain chunk / sidecar text — never a tag).
#   * Existing ``<a>…</a>`` anchors are stashed first so the pass is idempotent
#     (a URL inside an already-emitted ``href`` is never re-linkified).
_URL_RE = re.compile(
    r"(?P<open>&lt;|<)?"          # optional opening angle wrapper
    r"(?P<mopen>\$)?"           # optional math-open glued to the URL (fusion)
    r"(?P<scheme>https?)\s*:\s*/\s*/"   # scheme, OCR-spaced tolerant; NO space
    #   after the second ``/`` — a host char must immediately follow ``//`` so a
    #   scheme-only ``https://`` (or ``https:// Note`` column-bleed) can never
    #   cross whitespace into the next token and mint a bare-scheme anchor.
    r"(?P<body>[A-Za-z0-9][A-Za-z0-9.\-_%#?=+~:/]*[A-Za-z0-9/])"  # url body
    r"(?P<mclose>\$)?"          # optional math-close glued to the URL
    r"(?P<close>&gt;|>)?",      # optional closing angle wrapper
    re.IGNORECASE,
)
_EXISTING_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)
_STASH0_RE = re.compile("\x00(\\d+)\x00")


def linkify_urls(text: str, *, html: bool = True) -> str:
    r"""Linkify bare / angle-wrapped URLs; strip the wrapper + glued ``$`` (ITEM 1).

    ``html=True`` (rendered body): each URL → ``<a href="…" rel="noopener"
    class="mathjax_ignore">…</a>``. ``html=False`` (plain chunk / sidecar text):
    each URL → the bare normalized URL string. In BOTH modes the ``<…>`` /
    ``&lt;…&gt;`` angle wrapper and any ``$`` delimiter glued directly onto the
    URL are dropped, and an OCR-spaced scheme (``https : //``) is normalized to
    ``https://``. A fast guard keeps the common no-URL path allocation-free.
    """
    if not text or "http" not in text.lower():
        return text or ""
    stash: list[str] = []

    def _protect(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    s = text
    if html:
        # Idempotency: never re-linkify a URL already inside an <a> anchor.
        s = _EXISTING_ANCHOR_RE.sub(_protect, s)

    def _sub(m: "re.Match[str]") -> str:
        url = f"{m.group('scheme').lower()}://{m.group('body')}"
        if html:
            return (
                f'<a href="{url}" rel="noopener" '
                f'class="mathjax_ignore">{url}</a>'
            )
        return url

    s = _URL_RE.sub(_sub, s)
    if html and stash:
        s = _STASH0_RE.sub(lambda m: stash[int(m.group(1))], s)
    return s


def fold_math(text: str) -> str:
    """Fold LaTeX / unicode-math / digit-letter notation onto one vocabulary.

    Deterministic and idempotent-safe; a strict no-op on plain English (no
    ``\\``, no math unicode, no digit↔letter adjacency → unchanged). Applied to
    BOTH candidate and gold text before shingle / junk / garbage / repetition
    scoring so the two representations compare content, not notation.
    """
    if not text:
        return text or ""
    # 1+2. LaTeX commands: known symbol-commands → canonical word, else drop.
    s = _LATEX_CMD_RE.sub(_sub_latex_cmd, text)
    # 3. Unicode math symbols → the same canonical words / ASCII look-alikes.
    s = _UNICODE_RE.sub(lambda m: _UNICODE_MAP[m.group(0)], s)
    # 5. Exercise part markers: circled ``ⓐ`` and parenthesized ``(a)`` → the
    #    bare letter, so the two marker notations meet at the same token.
    s = _CIRCLED_LETTER_RE.sub(lambda m: _CIRCLED_LETTER_MAP[m.group(0)], s)
    s = _PAREN_LETTER_RE.sub(lambda m: f" {m.group(1)} ", s)
    # 4. Residual LaTeX delimiters/punctuation → space.
    s = _LATEX_PUNCT_RE.sub(" ", s)
    # 6. Split digit↔letter fusions in both directions.
    s = _DIGIT_LETTER_RE.sub(" ", s)
    return s


def count_math_folds(text: str) -> dict[str, int]:
    """Count the folding operations :func:`fold_math` would apply to ``text``.

    Surfaced in the report's ``normalization`` block so an audit can see how
    much of a chapter's junk / garbage was representation, not defect. All
    counts are zero on plain English.
    """
    if not text:
        return {
            "latex_commands": 0,
            "unicode_symbols": 0,
            "digit_letter_fusions": 0,
            "part_markers": 0,
        }
    return {
        "latex_commands": len(_LATEX_CMD_RE.findall(text)),
        "unicode_symbols": len(_UNICODE_RE.findall(text)),
        "digit_letter_fusions": len(_DIGIT_LETTER_RE.findall(text)),
        # Circled-letter + parenthesized single-letter exercise part markers.
        "part_markers": (
            len(_CIRCLED_LETTER_RE.findall(text))
            + len(_PAREN_LETTER_RE.findall(text))
        ),
    }
