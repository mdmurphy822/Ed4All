"""Deterministic presentation-MathML emitter for the reconstructed tree.

Walks the :mod:`.structure_infer` node tree and serialises well-formed
MathML 4.0 presentation markup wrapped in a single ``<math>`` carrying a
non-empty ``alttext`` (WCAG SC 1.1.1) — exactly the shape
:func:`semantik_structure.gates.mathml_check.validate_mathml` accepts. Every
element emitted is on that gate's 50-element presentation allowlist
(``mi`` / ``mn`` / ``mo`` / ``mtext`` / ``mrow`` / ``msup`` / ``msub`` /
``mfrac`` / ``math``).

Pure string building; no IO, no models.
"""
from __future__ import annotations

from html import escape

from .structure_infer import Atom, Frac, Row, Script

# Characters stripped from the alttext source (undecodable glyph
# placeholders carry no readable content).
import re as _re

_CID_RE = _re.compile(r"\(cid:\d+\)")


def _esc_attr(text: str) -> str:
    return escape(text, quote=True)


def _esc_text(text: str) -> str:
    # Escape XML special chars in element content; keep it well-formed.
    return escape(text, quote=False)


def _emit_node(node) -> str:
    if isinstance(node, Atom):
        kind = node.kind if node.kind in ("mi", "mn", "mo", "mtext") else "mtext"
        return f"<{kind}>{_esc_text(node.text)}</{kind}>"
    if isinstance(node, Script):
        tag = node.kind if node.kind in ("msup", "msub") else "msup"
        return (
            f"<{tag}>{_wrap_child(node.base)}{_wrap_child(node.script)}</{tag}>"
        )
    if isinstance(node, Frac):
        return f"<mfrac>{_wrap_child(node.num)}{_wrap_child(node.den)}</mfrac>"
    if isinstance(node, Row):
        inner = "".join(_emit_node(c) for c in node.children)
        return f"<mrow>{inner}</mrow>"
    # Unknown node — fail safe to a text node rather than malformed XML.
    return f"<mtext>{_esc_text(str(node))}</mtext>"


def _wrap_child(node) -> str:
    """Script/fraction arguments must be a SINGLE element. Wrap multi-child
    rows in <mrow>; a single Atom/Script/Frac is emitted bare."""
    if isinstance(node, Row):
        if len(node.children) == 1:
            return _emit_node(node.children[0])
        return _emit_node(node)  # emits <mrow>...</mrow>
    return _emit_node(node)


def alttext_from_src(src_text: str) -> str:
    """A readable, non-empty alttext linearisation of the source string."""
    cleaned = _CID_RE.sub(" ", src_text or "")
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or "math expression"


def emit_mathml(root: Row, *, alttext: str, display: bool = True) -> str:
    """Serialise the tree to a complete ``<math alttext=...>`` fragment."""
    alt = _esc_attr(alttext.strip() or "math expression")
    disp = "block" if display else "inline"
    body = "".join(_emit_node(c) for c in root.children)
    if not body:
        body = "<merror><mtext>{}</mtext></merror>".format(
            _esc_text(alttext.strip() or "math expression")
        )
    return f'<math display="{disp}" alttext="{alt}">{body}</math>'


def accessible_uniform_mathml(src_text: str, *, display: bool = True,
                              malformed: bool = False) -> str:
    """The LAST-RESORT uniform accessible representation — NEVER raw soup.

    Emits a valid ``<math>`` whose visible content is the source string as
    ``<mtext>`` (or ``<merror><mtext>`` when the source is unusable), with
    the source carried verbatim in ``alttext``. Always passes the MathML
    gate by construction, so a region that defeats both the deterministic
    reconstructor and the local generative specialist still ships valid,
    screen-reader-addressable MathML instead of a ``<span class=...>`` of
    undecoded glyphs.
    """
    alt = alttext_from_src(src_text)
    alt_attr = _esc_attr(alt)
    disp = "block" if display else "inline"
    inner_text = _esc_text(alt)
    if malformed or not (src_text or "").strip():
        body = f"<merror><mtext>{inner_text}</mtext></merror>"
    else:
        body = f"<mtext>{inner_text}</mtext>"
    return f'<math display="{disp}" alttext="{alt_attr}">{body}</math>'


__all__ = [
    "emit_mathml",
    "accessible_uniform_mathml",
    "alttext_from_src",
]
