"""MathML validity check for Stage 7 hard gate (math regions only).

Validates four properties of a candidate MathML fragment:

1. The fragment is well-formed XML.
2. The root element is ``<math>`` — either in the MathML namespace OR
   with no namespace at all. A bare ``<math>`` fragment carries no
   ``xmlns`` because in HTML5 the element is implicitly MathML-namespaced
   (this is how ar5iv ground-truth and the math adapter both emit it); we
   only reject a ``<math>`` that declares a *different* namespace.
3. The root carries a non-empty ``alttext`` attribute (WCAG SC 1.1.1).
4. Only MathML 4.0 presentation elements appear (50-element allowlist),
   **excluding** the contents of ``<annotation>`` / ``<annotation-xml>``
   subtrees — those hold non-presentation encodings (Content MathML,
   LaTeX, …) that are opaque to assistive tech and not subject to the
   presentation allowlist.

Returns a :class:`CheckOutcome` parameterised on
:attr:`GateCheck.MATHML_VALIDITY`.
"""
from __future__ import annotations

from .types import CheckOutcome, GateCheck

MATHML_NS = "http://www.w3.org/1998/Math/MathML"
MATHML_NS_PREFIX = f"{{{MATHML_NS}}}"


class LxmlUnavailableError(RuntimeError):
    """Raised when the MathML validity check cannot import ``lxml``.

    ``lxml`` is a mandatory runtime dependency (see ``pyproject.toml``:
    ``lxml>=5.1``), so this should never fire in a correctly installed
    environment. We raise rather than soft-pass: silently returning a
    benign-looking ``passed=True`` would let unvalidated math regions
    ship as if they were verified-clean — exactly the silent-fallback
    pattern the project forbids (feedback_no_silent_fallbacks.md).
    """

# MathML 4.0 element allowlist — covers presentation MathML (the part
# Qwen produces) plus the semantics annotation hooks. Content MathML is
# intentionally not allowlisted; if the math specialist starts emitting
# it we should explicitly approve each element.
MATHML_ALLOWED = frozenset({
    "math", "mi", "mo", "mn", "ms", "mtext", "mspace", "mrow",
    "mfrac", "msqrt", "mroot", "mstyle", "merror", "mpadded",
    "mphantom", "mfenced", "menclose", "msup", "msub", "msubsup",
    "munder", "mover", "munderover", "mmultiscripts", "mtable",
    "mtr", "mtd", "mlabeledtr", "maligngroup", "malignmark",
    "mstack", "mlongdiv", "msgroup", "msrow", "mscarries", "mscarry",
    "msline", "mglyph", "mprescripts", "none", "annotation",
    "annotation-xml", "semantics", "maction",
})


def _strip_ns(tag: str) -> str:
    """Drop the ``{http://www.w3.org/1998/Math/MathML}`` prefix from a
    Clark-notation tag string. Returns the local name unchanged when
    the namespace prefix is absent (the test suite uses unqualified
    tags to exercise the no-namespace failure path)."""
    if isinstance(tag, str) and tag.startswith(MATHML_NS_PREFIX):
        return tag[len(MATHML_NS_PREFIX):]
    return tag


def validate_mathml(xml: str) -> CheckOutcome:
    """Run all four MathML checks. Returns one :class:`CheckOutcome`.

    The first failure short-circuits — there's no point reporting an
    alttext miss on a fragment that didn't parse.
    """
    try:
        from lxml import etree
    except ImportError as exc:
        # lxml is a mandatory dependency (pyproject.toml: lxml>=5.1). If
        # the import fails the environment is broken; raise loudly rather
        # than soft-passing an unvalidated math region as verified-clean.
        raise LxmlUnavailableError(
            "lxml is required for MathML validity checks but could not be "
            "imported; it is a mandatory dependency (pyproject.toml: "
            "lxml>=5.1). Reinstall the project dependencies."
        ) from exc

    if not isinstance(xml, (str, bytes)):
        return CheckOutcome(
            check=GateCheck.MATHML_VALIDITY,
            passed=False,
            message=f"non-string input: type={type(xml).__name__}",
        )

    payload = xml.encode("utf-8") if isinstance(xml, str) else xml
    try:
        root = etree.fromstring(payload)
    except Exception as exc:
        return CheckOutcome(
            check=GateCheck.MATHML_VALIDITY,
            passed=False,
            message=f"malformed: {exc}",
        )

    # Check 1: root must be <math>, in the MathML namespace or none.
    # A namespaced root in a *different* namespace is rejected; a bare
    # (no-namespace) <math> is accepted as HTML5-implicit MathML.
    if not isinstance(root.tag, str):
        return CheckOutcome(
            check=GateCheck.MATHML_VALIDITY,
            passed=False,
            message=f"root not an element: tag={root.tag!r}",
        )
    if root.tag.startswith("{") and not root.tag.startswith(MATHML_NS_PREFIX):
        return CheckOutcome(
            check=GateCheck.MATHML_VALIDITY,
            passed=False,
            message=f"root <math> in non-mathml namespace: tag={root.tag!r}",
        )
    if _strip_ns(root.tag) != "math":
        return CheckOutcome(
            check=GateCheck.MATHML_VALIDITY,
            passed=False,
            message=f"root is not <math>: tag={root.tag!r}",
        )

    # Check 2: alttext (WCAG SC 1.1.1).
    alttext = (root.get("alttext") or "").strip()
    if not alttext:
        return CheckOutcome(
            check=GateCheck.MATHML_VALIDITY,
            passed=False,
            message="missing or empty alttext (SC 1.1.1)",
        )

    # Check 3: every presentation element must be a known MathML element.
    # The CONTENTS of <annotation>/<annotation-xml> are excluded — they
    # carry non-presentation encodings (Content MathML / LaTeX / …) that
    # are not rendered and not subject to the presentation allowlist. The
    # annotation element itself is allowlisted and still validated; only
    # its descendant subtree is skipped.
    annotated_subtree: set = set()
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if _strip_ns(el.tag) in ("annotation", "annotation-xml"):
            annotated_subtree.update(el.iterdescendants())
    for el in root.iter():
        # Skip lxml comment / PI nodes — they have callable .tag.
        if not isinstance(el.tag, str):
            continue
        if el in annotated_subtree:
            continue
        local = _strip_ns(el.tag)
        if local not in MATHML_ALLOWED:
            return CheckOutcome(
                check=GateCheck.MATHML_VALIDITY,
                passed=False,
                message=f"non-mathml element: {el.tag}",
            )

    return CheckOutcome(
        check=GateCheck.MATHML_VALIDITY,
        passed=True,
        message="ok",
    )


__all__ = ["validate_mathml", "MATHML_NS", "MATHML_ALLOWED", "LxmlUnavailableError"]
