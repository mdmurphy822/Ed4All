r"""Deterministic LaTeX → presentation-MathML render-time transform.

Why
---
``SEMANTIK_VLM_FUSION`` transcribes scanned math as inline LaTeX and — by its
documented contract — keeps it **VERBATIM as text** (``$``-delimiters
preserved). Nothing downstream ever turned that LaTeX into MathML, so a
scanned-textbook chapter shipped ~3.7k spans of literal ``$\frac{1}{2}$`` inside
``<p>`` bodies. A screen reader pronounces that as "dollar backslash frac one
two dollar" — a WCAG 1.3.1 failure and a catastrophic listening experience,
while the vendor's accessible HTML ships ~2.9k real ``<math>`` elements.

The math content is ALREADY correct — only its *representation* is wrong. This
module is the missing representation step: a pure, deterministic
LaTeX-string → presentation-MathML emitter over the arithmetic/algebra grammar
this corpus actually uses (measured, not guessed: ``\frac`` dominates, then
``\cdot`` / ``\div`` / ``\text`` / ``\left…\right`` / ``\sqrt`` / ``\quad`` /
``\neq`` / greek / ``^`` / ``_``).

Contract
--------
* **Reuses the real accept gate.** Every emitted ``<math>`` is validated by
  ``semantik_structure.gates.mathml_check.validate_mathml`` — the SAME function
  the cascade's Stage-7 MathML gate runs (loaded from its own source file so the
  heavy ``gates/__init__`` chain, which pulls the ``[semantik]``-extra-only
  playwright/axe stack, is never imported into a plain Ed4All venv). Not a
  re-implementation: identical code, identical verdicts.
* **Fail-soft, never mangled.** A span whose LaTeX falls outside the supported
  grammar — an unknown control sequence, an ``array``/``aligned`` environment, a
  ``\\`` row break, an alignment ``&``, a bare ``%`` — is DECLINED and left
  EXACTLY as it is today (verbatim LaTeX text). A span whose emitted MathML
  fails the gate is likewise declined. Partial coverage is expected and honest;
  silent corruption is not. A decline is never a half-conversion.
* **HTML-ONLY.** ``raw_text`` / ``repaired_text`` stay VERBATIM — they are the
  content-hash sourceId basis (mirrors the documented OCR-repair posture), so
  sourceIds / chunk refs / citation deep-links are byte-stable. The conversion
  happens at RENDER, exactly like ``SEMANTIK_BOX_TITLE_HEADINGS``.
* **Lossless.** The emitted element carries the original LaTeX twice — as the
  ``alttext`` attribute (WCAG SC 1.1.1, and what the gate requires) and as a
  ``<annotation encoding="application/x-tex">`` inside ``<semantics>`` (the
  standard LaTeX round-trip carrier; the gate explicitly exempts annotation
  CONTENTS from its presentation allowlist). The shape is a strict superset of
  the vendor gold's ``<math display=…><semantics><mrow>…</mrow></semantics></math>``.
* **NO LLM call site → NO DecisionCapture obligation.** This is a pure
  deterministic string → markup transform (house convention: the capture
  contract binds LLM call sites only).

Gated by ``SEMANTIK_LATEX_MATHML`` at the adapter seam; flag OFF is
byte-identical (this module is not even imported).
"""
from __future__ import annotations

import html as _html
import importlib.util
import re
import sys
import types as _pytypes
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

__all__ = [
    "MathMLGateUnavailableError",
    "convert_latex_spans",
    "latex_to_mathml",
    "load_validate_mathml",
]


# ---------------------------------------------------------------------------
# Accept gate — reuse the cascade's real ``validate_mathml``.
# ---------------------------------------------------------------------------
class MathMLGateUnavailableError(RuntimeError):
    """Raised when the MathML validity gate cannot be loaded.

    The accept gate is the whole safety property of this pass: without it we
    would be emitting unvalidated markup into a learner page. So we raise
    loudly (with operator guidance) rather than soft-passing — the same posture
    ``mathml_check.LxmlUnavailableError`` takes, and the same posture the
    SemantiK bridge takes when its runtime is unconfigured. The pass is opt-in,
    and the raise happens at pass entry BEFORE any mutation, so a misconfigured
    box fails fast and clean instead of shipping silently-degraded math.
    """


_GATES_DIR = (
    Path(__file__).resolve().parents[2] / "SemantiK" / "semantik_structure" / "gates"
)
_PKG = "_semantik_gates_lite"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_validate_mathml():
    """Return the cascade's ``validate_mathml``; raise if it cannot be loaded.

    Prefers a normal package import (the in-process SemantiK venv, where
    ``semantik_structure`` is already importable). Falls back to loading
    ``gates/mathml_check.py`` + its ``gates/types.py`` sibling DIRECTLY by path
    under a private package name — the identical source, but without executing
    ``gates/__init__``, whose eager ``hard_document`` import drags in the
    playwright/axe stack that only ships in the ``[semantik]`` extra. Either way
    the function object is the cascade's own; this module never re-implements
    the gate.
    """
    try:  # in-process SemantiK venv — the real package is importable.
        from semantik_structure.gates.mathml_check import (  # type: ignore
            validate_mathml,
        )

        return validate_mathml
    except Exception:  # noqa: BLE001 — fall through to the by-path loader.
        pass

    try:
        if _PKG not in sys.modules:
            pkg = _pytypes.ModuleType(_PKG)
            pkg.__path__ = [str(_GATES_DIR)]  # type: ignore[attr-defined]
            sys.modules[_PKG] = pkg
        if f"{_PKG}.types" not in sys.modules:
            _load_module(f"{_PKG}.types", _GATES_DIR / "types.py")
        module = sys.modules.get(f"{_PKG}.mathml_check") or _load_module(
            f"{_PKG}.mathml_check", _GATES_DIR / "mathml_check.py"
        )
        return module.validate_mathml
    except Exception as exc:  # noqa: BLE001
        raise MathMLGateUnavailableError(
            "SEMANTIK_LATEX_MATHML is enabled but the MathML validity gate "
            f"({_GATES_DIR / 'mathml_check.py'}) could not be loaded: {exc}. "
            "Every emitted <math> must pass that gate, so the pass refuses to "
            "run rather than ship unvalidated markup. Ensure the SemantiK tree "
            "is present and `lxml` is installed, or unset SEMANTIK_LATEX_MATHML."
        ) from exc


# ---------------------------------------------------------------------------
# Grammar tables — the arithmetic/algebra subset this corpus actually uses.
# ---------------------------------------------------------------------------

# Control sequences that render as an OPERATOR (<mo>).
_OPERATORS = {
    "cdot": "⋅", "times": "×", "div": "÷", "pm": "±",
    "mp": "∓", "ast": "∗", "star": "⋆",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "ne": "≠", "neq": "≠", "approx": "≈", "equiv": "≡",
    "sim": "∼", "simeq": "≃", "propto": "∝",
    "ll": "≪", "gg": "≫",
    "circ": "∘", "odot": "⊙", "oplus": "⊕", "otimes": "⊗",
    "bigcirc": "○", "bullet": "∙",
    "ldots": "…", "dots": "…", "cdots": "⋯", "vdots": "⋮",
    "to": "→", "rightarrow": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐",
    "leftrightarrow": "↔", "Leftrightarrow": "⇔", "mapsto": "↦",
    "lvert": "|", "rvert": "|", "vert": "|", "mid": "|",
    "lVert": "‖", "rVert": "‖",
    "cup": "∪", "cap": "∩", "in": "∈", "notin": "∉",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "supseteq": "⊇", "emptyset": "∅", "varnothing": "∅",
    "neg": "¬", "forall": "∀", "exists": "∃",
    "angle": "∠", "perp": "⊥", "parallel": "∥",
    "prime": "′", "degree": "°",
    "%": "%", "$": "$", "#": "#", "&": "&", "_": "_",
    "{": "{", "}": "}",
    "langle": "⟨", "rangle": "⟩",
    "lfloor": "⌊", "rfloor": "⌋",
    "lceil": "⌈", "rceil": "⌉",
    "colon": ":",
}

# Control sequences that render as an IDENTIFIER (<mi>) — greek + named consts.
_IDENTIFIERS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "infty": "∞",
}

# Named function/limit operators — rendered as an upright <mi> per MathML custom.
_FUNCTIONS = {
    "sin", "cos", "tan", "csc", "sec", "cot", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "log", "ln", "exp", "lim", "max", "min", "gcd",
    "det", "dim", "deg", "arg",
}

# Spacing control sequences → <mspace width=…> (mirrors the vendor gold, which
# emits <mspace width="1.5em"> for its own spacing runs).
_SPACING = {
    "quad": "1em", "qquad": "2em", ",": "0.167em", ";": "0.278em",
    ":": "0.222em", ">": "0.222em", " ": "0.25em", "enspace": "0.5em",
    "thinspace": "0.167em", "hspace": None,  # \hspace takes an arg → declined
}

# Fraction-family commands (2 args) and their MathML tag.
_FRACS = {"frac", "dfrac", "tfrac", "cfrac"}

# Single-arg commands that wrap their argument.
_OVER_UNDER = {
    "overline": ("mover", "¯"),
    "underline": ("munder", "_"),
    "bar": ("mover", "¯"),
    "hat": ("mover", "^"),
    "vec": ("mover", "→"),
    "tilde": ("mover", "~"),
    "overbrace": ("mover", "⏞"),
    "underbrace": ("munder", "⏟"),
}

# Single-arg commands whose argument is RAW TEXT (not math) → <mtext>.
_TEXT_ARG = {"text", "textbf", "textit", "textrm", "mbox", "textnormal"}

# Single-arg commands whose argument is math, rendered upright / transparently.
_MATH_STYLE_ARG = {"mathrm", "mathbf", "mathit", "mathsf", "mathtt", "operatorname"}

# Delimiters legal after \left / \right.
_DELIMS = {
    "(": "(", ")": ")", "[": "[", "]": "]", "|": "|", ".": "",
    "\\{": "{", "\\}": "}", "\\lvert": "|", "\\rvert": "|",
    "\\vert": "|", "\\|": "‖",
    "\\langle": "⟨", "\\rangle": "⟩",
    "\\lfloor": "⌊", "\\rfloor": "⌋",
    "\\lceil": "⌈", "\\rceil": "⌉",
}

# Bare operator characters.
_OP_CHARS = set("+-*/=<>(),.;:!?|[]'−–—×÷±"
                "≤≥≠≈∞°′")

# Characters we normalize on the way in (unicode look-alikes → canonical math).
_CHAR_FOLD = {"–": "−", "—": "−", "-": "−"}

_TOKEN_RE = re.compile(
    r"""
      (?P<cmd>\\[a-zA-Z]+)             # \frac, \cdot, \alpha …
    | (?P<esc>\\[,;:!>\ %$#&_{}|])     # \, \; \% \{ … (escaped / spacing)
    | (?P<num>\d+(?:\.\d+)?)           # 12, 3.5
    | (?P<letter>[A-Za-z])             # a, x, Y  (one <mi> each)
    | (?P<sup>\^)
    | (?P<sub>_)
    | (?P<lbrace>\{)
    | (?P<rbrace>\})
    | (?P<lbrack>\[)
    | (?P<rbrack>\])
    | (?P<ws>\s+)
    | (?P<op>[^\sA-Za-z0-9\\{}\[\]^_])  # any other single char → operator
    """,
    re.VERBOSE,
)

# Constructs we DECLINE outright (never a mangled half-conversion):
#   \begin/\end  — array / aligned / cases: 2-D layout the flat grammar can't hold
#   \\           — a row break, meaningless outside a table environment
#   bare &       — an alignment tab with no environment to align in
#   bare %       — a LaTeX comment marker; semantics are ambiguous in OCR output
_DECLINE_RE = re.compile(r"\\begin\b|\\end\b|\\\\|(?<!\\)&|(?<!\\)%")


class _Decline(Exception):
    """Internal: this span falls outside the supported grammar — leave verbatim."""


class _Tok(NamedTuple):
    """One token, carrying its SOURCE OFFSETS so a raw-text argument can be
    sliced back out of the original string verbatim (the tokenizer discards
    whitespace, so a token stream cannot be faithfully re-joined)."""

    kind: str
    text: str
    start: int
    end: int


# A ``\text{…}`` / ``\mbox{…}`` argument is DECLARED prose — legitimate inside
# math (a worked example's "Simplify." step label that OCR fused into the span).
# Bare prose in MATH position is the danger: a mis-wrapped ``$…$`` prose blob
# would otherwise become one ``<mi>`` per letter — a screen reader spelling the
# sentence out character by character, strictly WORSE than the LaTeX text it
# replaced. So the prose test runs on the residue AFTER the declared text args
# are removed, reusing ``math_fold``'s tested predicate rather than a new one.
_TEXT_ARG_RE = re.compile(
    r"\\(?:text|textbf|textit|textrm|mbox|textnormal)\s*\{[^{}]*\}"
)


def _reads_as_prose(source: str) -> bool:
    from lib.semantik.math_fold import _is_prose_math_span  # noqa: PLC0415

    return _is_prose_math_span(_TEXT_ARG_RE.sub(" ", source))


# ---------------------------------------------------------------------------
# Emit helpers.
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    """Escape XML character data (element content)."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _esc_attr(text: str) -> str:
    return _esc(text).replace('"', "&quot;")


def _row(nodes: List[str]) -> str:
    """Collapse a node list to exactly ONE element (MathML script/frac args)."""
    if len(nodes) == 1:
        return nodes[0]
    return "<mrow>" + "".join(nodes) + "</mrow>"


# ---------------------------------------------------------------------------
# Parser — recursive descent over the token stream.
# ---------------------------------------------------------------------------
class _Parser:
    def __init__(self, tokens: List["_Tok"], source: str):
        self.toks = tokens
        self.source = source
        self.i = 0

    def peek(self) -> Optional["_Tok"]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> "_Tok":
        tok = self.peek()
        if tok is None:
            raise _Decline("unexpected end of input")
        self.i += 1
        return tok

    # -- argument capture ---------------------------------------------------
    def parse_arg(self) -> str:
        """One MathML element: a braced group, or a single following atom.

        A script arg is parsed WITHOUT script attachment (``parse_base_only``),
        so ``x_1^2`` binds BOTH scripts to ``x`` (``<msubsup>``) rather than
        letting the ``^2`` bind to the subscript's ``1``.
        """
        tok = self.peek()
        if tok is None:
            raise _Decline("command missing its argument")
        if tok.kind == "lbrace":
            self.next()
            nodes = self.parse_until_rbrace()
            return _row(nodes) if nodes else "<mrow></mrow>"
        return _row(self.parse_base_only())

    def parse_base_only(self) -> List[str]:
        """One atom with NO ``^``/``_`` attachment (a script/command argument)."""
        base = self._parse_base()
        if not base:
            raise _Decline("empty argument")
        return base

    def parse_raw_text_arg(self) -> str:
        r"""The RAW source text of a braced group (``\text{…}``) — NOT math-parsed.

        Slices the ORIGINAL source between the braces, so interior whitespace is
        preserved verbatim ("Use definition of exponent." must not collapse to
        "Usedefinitionofexponent." — the tokenizer discards whitespace, so the
        token stream cannot be re-joined faithfully).
        """
        tok = self.peek()
        if tok is None or tok.kind != "lbrace":
            raise _Decline("text command without a braced argument")
        self.next()
        start = tok.end  # first character AFTER the opening brace
        depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise _Decline("unbalanced brace in text argument")
            self.next()
            if tok.kind == "lbrace":
                depth += 1
            elif tok.kind == "rbrace":
                depth -= 1
                if depth == 0:
                    return self.source[start : tok.start]

    def parse_optional_bracket_arg(self) -> Optional[str]:
        """``[n]`` after ``\\sqrt`` → the index element, else ``None``."""
        tok = self.peek()
        if tok is None or tok.kind != "lbrack":
            return None
        self.next()
        nodes: List[str] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise _Decline("unbalanced [ in optional argument")
            if tok.kind == "rbrack":
                self.next()
                break
            nodes.extend(self.parse_atom())
        return _row(nodes) if nodes else "<mrow></mrow>"

    def parse_until_rbrace(self) -> List[str]:
        nodes: List[str] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise _Decline("unbalanced { in group")
            if tok.kind == "rbrace":
                self.next()
                return nodes
            nodes.extend(self.parse_atom())

    # -- atoms --------------------------------------------------------------
    def parse_atom(self) -> List[str]:
        """Parse ONE atom plus any ``^``/``_`` scripts attached to it."""
        tok = self.peek()
        if tok is not None and tok.kind in ("sup", "sub"):
            # A script with NO base: OCR/VLM regularly emits a bare exponent or
            # degree marker whose base sits in the preceding text run
            # (``$^2$``, ``$^\circ\text{F}$``). An empty <mrow> base is the
            # canonical MathML spelling and reads correctly ("to the power 2"),
            # so this is faithful, not a fabrication.
            return [self._attach_scripts("<mrow></mrow>")]
        base = self._parse_base()
        if not base:
            return []
        base_el = _row(base) if len(base) > 1 else base[0]
        scripted = self._attach_scripts(base_el)
        return [scripted] if scripted != base_el else base

    def _attach_scripts(self, base: str) -> str:
        sup: Optional[str] = None
        sub: Optional[str] = None
        while True:
            tok = self.peek()
            if tok is None or tok.kind not in ("sup", "sub"):
                break
            self.next()
            arg = self.parse_arg()
            if tok.kind == "sup":
                if sup is not None:
                    raise _Decline("double superscript")
                sup = arg
            else:
                if sub is not None:
                    raise _Decline("double subscript")
                sub = arg
        if sup is not None and sub is not None:
            return f"<msubsup>{base}{sub}{sup}</msubsup>"
        if sup is not None:
            return f"<msup>{base}{sup}</msup>"
        if sub is not None:
            return f"<msub>{base}{sub}</msub>"
        return base

    def _parse_base(self) -> List[str]:
        tok = self.next()
        kind, text = tok.kind, tok.text

        if kind == "ws":
            return []
        if kind == "num":
            return [f"<mn>{text}</mn>"]
        if kind == "letter":
            return [f"<mi>{text}</mi>"]
        if kind == "lbrace":
            nodes = self.parse_until_rbrace()
            return [_row(nodes)] if nodes else []
        if kind == "rbrace":
            raise _Decline("unbalanced }")
        if kind in ("sup", "sub"):
            raise _Decline("script with no base")
        if kind in ("lbrack", "rbrack"):
            return [f"<mo>{text}</mo>"]
        if kind == "op":
            ch = _CHAR_FOLD.get(text, text)
            if ch not in _OP_CHARS and not _is_safe_op_char(ch):
                raise _Decline(f"unsupported character {text!r}")
            return [f"<mo>{_esc(ch)}</mo>"]
        if kind == "esc":
            return self._parse_escape(text[1:])
        if kind == "cmd":
            return self._parse_command(text[1:])
        raise _Decline(f"unsupported token {kind}")  # pragma: no cover

    def _parse_escape(self, ch: str) -> List[str]:
        if ch in _SPACING and _SPACING[ch] is not None:
            return [f'<mspace width="{_SPACING[ch]}"></mspace>']
        if ch == "!":
            return []  # negative thin space — no visual/semantic content
        if ch in _OPERATORS:
            return [f"<mo>{_esc(_OPERATORS[ch])}</mo>"]
        raise _Decline(f"unsupported escape \\{ch}")

    def _parse_command(self, name: str) -> List[str]:
        if name in _FRACS:
            num = self.parse_arg()
            den = self.parse_arg()
            return [f"<mfrac>{num}{den}</mfrac>"]

        if name == "sqrt":
            index = self.parse_optional_bracket_arg()
            radicand = self.parse_arg()
            if index is not None:
                return [f"<mroot>{radicand}{index}</mroot>"]
            return [f"<msqrt>{radicand}</msqrt>"]

        if name in _TEXT_ARG:
            raw = self.parse_raw_text_arg()
            return [f"<mtext>{_esc(raw)}</mtext>"] if raw else []

        if name in _MATH_STYLE_ARG:
            return [self.parse_arg()]

        if name in _OVER_UNDER:
            tag, accent = _OVER_UNDER[name]
            arg = self.parse_arg()
            return [f"<{tag}>{arg}<mo>{_esc(accent)}</mo></{tag}>"]

        if name == "boxed":
            arg = self.parse_arg()
            return [f'<menclose notation="box">{arg}</menclose>']

        if name in ("left", "right"):
            return self._parse_left_right()

        if name in _SPACING and _SPACING[name] is not None:
            return [f'<mspace width="{_SPACING[name]}"></mspace>']

        if name in _FUNCTIONS:
            return [f"<mi>{name}</mi>"]

        if name in _IDENTIFIERS:
            return [f"<mi>{_esc(_IDENTIFIERS[name])}</mi>"]

        if name in _OPERATORS:
            return [f"<mo>{_esc(_OPERATORS[name])}</mo>"]

        raise _Decline(f"unsupported command \\{name}")

    def _parse_left_right(self) -> List[str]:
        """``\\left(`` / ``\\right)`` — emit the delimiter, drop the sizing verb."""
        tok = self.peek()
        if tok is None:
            raise _Decline("\\left/\\right with no delimiter")
        self.next()
        delim = _DELIMS.get(tok.text)
        if delim is None:
            raise _Decline(f"unsupported \\left/\\right delimiter {tok.text!r}")
        if delim == "":  # ``\left.`` / ``\right.`` — an invisible fence.
            return []
        return [f'<mo stretchy="true">{_esc(delim)}</mo>']


def _is_safe_op_char(ch: str) -> bool:
    """A printable symbol we are willing to emit verbatim as an <mo>."""
    return ch in {"•", "…", "⋅", "⊕", "⊙", "○"}


# ---------------------------------------------------------------------------
# Public: one LaTeX body → one <math> fragment (or None to decline).
# ---------------------------------------------------------------------------
def latex_to_mathml(latex: str, *, display: bool = False) -> Optional[str]:
    r"""Convert ONE LaTeX body to a gate-valid ``<math>`` fragment.

    ``latex`` is the span CONTENT (delimiters already stripped), as it appears in
    the rendered HTML — so it may still carry ``&lt;`` / ``&amp;`` entities from
    the upstream ``_escape_math_angle_brackets`` pass; they are decoded here and
    re-escaped on emit.

    Returns ``None`` (a DECLINE) when the LaTeX falls outside the supported
    grammar or the emitted markup fails the MathML gate. A decline means the
    caller must leave the span EXACTLY as it found it.
    """
    source = _html.unescape(latex or "").strip()
    if not source:
        return None
    # A raw HTML tag inside the span means a real element straddles the math
    # delimiters — not our business, and unsafe to rewrite.
    if "<" in (latex or "") or ">" in (latex or ""):
        return None
    if _DECLINE_RE.search(source):
        return None
    if _reads_as_prose(source):
        return None

    try:
        tokens = _tokenize(source)
        parser = _Parser(tokens, source)
        nodes: List[str] = []
        while parser.peek() is not None:
            nodes.extend(parser.parse_atom())
    except _Decline:
        return None
    except RecursionError:  # pragma: no cover - pathological nesting
        return None

    if not nodes:
        return None

    body = "<mrow>" + "".join(nodes) + "</mrow>"
    disp = "block" if display else "inline"
    alt = _esc_attr(source)
    mathml = (
        f'<math display="{disp}" alttext="{alt}">'
        f"<semantics>{body}"
        f'<annotation encoding="application/x-tex">{_esc(source)}</annotation>'
        f"</semantics></math>"
    )

    validate = load_validate_mathml()
    outcome = validate(mathml)
    if not getattr(outcome, "passed", False):
        return None
    return mathml


def _tokenize(source: str) -> List[_Tok]:
    tokens: List[_Tok] = []
    pos = 0
    while pos < len(source):
        match = _TOKEN_RE.match(source, pos)
        if match is None:
            raise _Decline(f"untokenizable input at {pos}")
        start, pos = match.start(), match.end()
        kind = match.lastgroup or ""
        if kind == "ws":
            continue
        tokens.append(_Tok(kind, match.group(), start, pos))
    return tokens


# ---------------------------------------------------------------------------
# Public: walk a rendered HTML fragment and convert every convertible span.
# ---------------------------------------------------------------------------
# The SAME span iterator the round-8/9/11 math passes use, so this pass sees
# exactly the delimited spans they normalized (never re-deriving the boundaries).
from lib.semantik.math_fold import _MATH_SPAN_ANGLE_RE  # noqa: E402

_DISPLAY_OPENERS = ("$$", "\\[")


def _split_span(span: str) -> Tuple[str, bool]:
    """Return ``(body, is_display)`` for one delimited math span."""
    if span.startswith("$$") and span.endswith("$$"):
        return span[2:-2], True
    if span.startswith("\\[") and span.endswith("\\]"):
        return span[2:-2], True
    if span.startswith("\\(") and span.endswith("\\)"):
        return span[2:-2], False
    if span.startswith("$") and span.endswith("$"):
        return span[1:-1], False
    return span, False  # pragma: no cover - the regex cannot produce this


def convert_latex_spans(text: str) -> Tuple[str, int, int]:
    r"""Convert every convertible ``$…$`` / ``\(…\)`` span in ``text`` to MathML.

    Returns ``(html, converted, declined)``. A declined span is emitted BYTE-FOR-
    BYTE as it arrived — this function never produces a partially-converted span.
    A strict no-op (and a fixed point) on text with no delimited math, so
    re-rendering an already-converted page is idempotent.
    """
    if not text or ("$" not in text and "\\(" not in text and "\\[" not in text):
        return text or "", 0, 0

    stats = [0, 0]  # converted, declined

    def _sub(match: "re.Match[str]") -> str:
        span = match.group(0)
        body, display = _split_span(span)
        mathml = latex_to_mathml(body, display=display)
        if mathml is None:
            stats[1] += 1
            return span
        stats[0] += 1
        return mathml

    out = _MATH_SPAN_ANGLE_RE.sub(_sub, text)
    return out, stats[0], stats[1]
