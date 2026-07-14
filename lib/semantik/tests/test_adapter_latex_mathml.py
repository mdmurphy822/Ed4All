r"""Tests for the ``SEMANTIK_LATEX_MATHML`` LaTeX → presentation-MathML pass.

Covers the contract stated in ``lib/semantik/latex_mathml.py``:

* the REAL corpus samples (``$15 \div 3$``, ``$5 \cdot 3$``, ``$2 \cdot 2 \cdot 3$``)
* fractions / roots / superscripts / subscripts / grouping / operators
* an unparseable span stays EXACTLY verbatim (never a half-conversion)
* flag OFF is byte-identical
* ``alttext`` preserves the original LaTeX
* every emitted ``<math>`` passes the cascade's own ``validate_mathml``
"""
from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _latex_to_mathml,
    _resolve_latex_mathml,
)
from lib.semantik.latex_mathml import (
    convert_latex_spans,
    latex_to_mathml,
    load_validate_mathml,
)

validate_mathml = load_validate_mathml()


def _assert_gate_valid(mathml: str) -> None:
    outcome = validate_mathml(mathml)
    assert outcome.passed, f"MathML failed the gate: {outcome.message}\n{mathml}"


# ---------------------------------------------------------------------------
# 1. The real corpus samples from the ch01 scan conversion.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "latex",
    [
        r"15 \div 3",
        r"5 \cdot 3",
        r"2 \cdot 2 \cdot 3",
        r"\frac{1 \cdot 4}{2 \cdot 4} = \frac{4}{8}",
        r"\frac{4}{5} \div \frac{3}{4}",
        r"y = -\frac{1}{4}",
        r"-\frac{74}{100}",
        r"1 \cdot a = a",
        r"z = -12",
        r"4y - 4",
    ],
)
def test_real_corpus_samples_convert_and_validate(latex: str) -> None:
    mathml = latex_to_mathml(latex)
    assert mathml is not None, f"declined a real corpus span: {latex}"
    _assert_gate_valid(mathml)
    assert mathml.startswith("<math ")


def _presentation_body(mathml: str) -> str:
    """The rendered (screen-reader-visible) subtree — excludes alttext + annotation."""
    return re.search(r"<semantics>(.*?)<annotation", mathml, re.S).group(1)


def test_div_sample_emits_the_right_operator() -> None:
    mathml = latex_to_mathml(r"15 \div 3")
    assert "<mn>15</mn>" in mathml
    assert "<mo>÷</mo>" in mathml
    assert "<mn>3</mn>" in mathml
    # The literal LaTeX must NOT survive in what a screen reader announces (it
    # is retained ONLY in the alttext attribute + the x-tex annotation).
    assert r"\div" not in _presentation_body(mathml)


# ---------------------------------------------------------------------------
# 2. Grammar: fraction / root / superscript / subscript / grouping.
# ---------------------------------------------------------------------------
def test_fraction() -> None:
    mathml = latex_to_mathml(r"\frac{1}{2}")
    _assert_gate_valid(mathml)
    assert "<mfrac><mn>1</mn><mn>2</mn></mfrac>" in mathml


def test_nested_fraction() -> None:
    mathml = latex_to_mathml(r"\frac{\frac{1}{2}}{3}")
    _assert_gate_valid(mathml)
    assert mathml.count("<mfrac>") == 2


def test_square_root() -> None:
    mathml = latex_to_mathml(r"\sqrt{16}")
    _assert_gate_valid(mathml)
    assert "<msqrt><mn>16</mn></msqrt>" in mathml


def test_nth_root() -> None:
    mathml = latex_to_mathml(r"\sqrt[3]{27}")
    _assert_gate_valid(mathml)
    assert "<mroot><mn>27</mn><mn>3</mn></mroot>" in mathml


def test_superscript() -> None:
    mathml = latex_to_mathml(r"x^2")
    _assert_gate_valid(mathml)
    assert "<msup><mi>x</mi><mn>2</mn></msup>" in mathml


def test_subscript() -> None:
    mathml = latex_to_mathml(r"a_1")
    _assert_gate_valid(mathml)
    assert "<msub><mi>a</mi><mn>1</mn></msub>" in mathml


def test_sub_and_superscript() -> None:
    mathml = latex_to_mathml(r"x_1^2")
    _assert_gate_valid(mathml)
    assert "<msubsup>" in mathml


def test_multi_char_exponent() -> None:
    # A single-element group needs no <mrow> wrapper (the mathml_emit discipline).
    mathml = latex_to_mathml(r"x^{12}")
    _assert_gate_valid(mathml)
    assert "<msup><mi>x</mi><mn>12</mn></msup>" in mathml


def test_multi_element_exponent_is_mrow_wrapped() -> None:
    # A script argument MUST be exactly one element, so a multi-node group wraps.
    mathml = latex_to_mathml(r"x^{n+1}")
    _assert_gate_valid(mathml)
    assert "<msup><mi>x</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msup>" in mathml


def test_bare_superscript_gets_empty_base() -> None:
    # ``$^\circ\text{F}$`` — OCR left the base in the preceding text run.
    mathml = latex_to_mathml(r"^\circ\text{F}")
    _assert_gate_valid(mathml)
    assert "<msup><mrow></mrow><mo>∘</mo></msup>" in mathml
    assert "<mtext>F</mtext>" in mathml


def test_text_argument_becomes_mtext() -> None:
    mathml = latex_to_mathml(r"4^2 \text{Use definition of exponent.}")
    _assert_gate_valid(mathml)
    assert "<mtext>Use definition of exponent.</mtext>" in mathml


def test_left_right_delimiters() -> None:
    mathml = latex_to_mathml(r"\left(\frac{1}{2}\right)")
    _assert_gate_valid(mathml)
    assert '<mo stretchy="true">(</mo>' in mathml
    assert '<mo stretchy="true">)</mo>' in mathml


@pytest.mark.parametrize(
    "latex,glyph",
    [
        (r"a \times b", "×"),
        (r"a \div b", "÷"),
        (r"a \cdot b", "⋅"),
        (r"a \pm b", "±"),
        (r"a \le b", "≤"),
        (r"a \ge b", "≥"),
        (r"a \neq b", "≠"),
        (r"a \approx b", "≈"),
    ],
)
def test_operators(latex: str, glyph: str) -> None:
    mathml = latex_to_mathml(latex)
    _assert_gate_valid(mathml)
    assert f"<mo>{glyph}</mo>" in mathml


def test_greek_identifier() -> None:
    mathml = latex_to_mathml(r"\alpha + \beta")
    _assert_gate_valid(mathml)
    assert "<mi>α</mi>" in mathml and "<mi>β</mi>" in mathml


def test_inequality_entity_is_decoded_then_reescaped() -> None:
    # The upstream ``_escape_math_angle_brackets`` pass leaves ``&lt;`` in the span.
    mathml = latex_to_mathml("a &lt; b")
    _assert_gate_valid(mathml)
    assert "<mo>&lt;</mo>" in mathml


# ---------------------------------------------------------------------------
# 3. alttext preservation + the x-tex annotation round-trip.
# ---------------------------------------------------------------------------
def test_alttext_preserves_the_original_latex() -> None:
    latex = r"\frac{1}{2} + \sqrt{9}"
    mathml = latex_to_mathml(latex)
    _assert_gate_valid(mathml)
    alt = re.search(r'alttext="([^"]*)"', mathml).group(1)
    # XML-escaped in the attribute, but otherwise the source verbatim.
    assert alt.replace("&amp;", "&") == latex


def test_annotation_carries_the_verbatim_tex() -> None:
    latex = r"\frac{3}{4}"
    mathml = latex_to_mathml(latex)
    body = re.search(
        r'<annotation encoding="application/x-tex">(.*?)</annotation>', mathml
    ).group(1)
    assert body == latex


def test_display_mode() -> None:
    assert 'display="block"' in latex_to_mathml(r"\frac{1}{2}", display=True)
    assert 'display="inline"' in latex_to_mathml(r"\frac{1}{2}", display=False)


# ---------------------------------------------------------------------------
# 4. Fail-soft: an unparseable span stays EXACTLY verbatim.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "latex",
    [
        r"\begin{array}{ll} a & b \\ c & d \end{array}",  # 2-D layout
        r"\somethingUnknown{x}",                           # unknown control seq
        r"x \\ y",                                         # row break
        r"a & b",                                          # alignment tab
        r"50 % off",                                       # bare comment marker
        r"\frac{1}",                                       # missing argument
        r"{unbalanced",                                    # unbalanced brace
    ],
)
def test_unparseable_latex_is_declined(latex: str) -> None:
    assert latex_to_mathml(latex) is None


def test_declined_span_stays_byte_for_byte_verbatim() -> None:
    html = r"<p>before $\begin{array}{ll} a & b \end{array}$ after</p>"
    out, converted, declined = convert_latex_spans(html)
    assert out == html          # byte-for-byte
    assert converted == 0
    assert declined == 1


def test_prose_blob_wrapped_in_dollars_is_declined() -> None:
    # A mis-wrapped prose span must never become one <mi> per letter.
    latex = "5.97 as 6 minus 0.03 and then using the distributive property"
    assert latex_to_mathml(latex) is None


def test_mixed_block_converts_only_the_convertible_span() -> None:
    html = r"<p>$\frac{1}{2}$ and $\begin{array}{l} x \end{array}$</p>"
    out, converted, declined = convert_latex_spans(html)
    assert converted == 1 and declined == 1
    assert "<mfrac>" in out
    assert r"$\begin{array}{l} x \end{array}$" in out  # untouched


def test_no_math_is_a_strict_no_op() -> None:
    html = "<p>Plain prose with no math at all.</p>"
    assert convert_latex_spans(html) == (html, 0, 0)


def test_conversion_is_idempotent() -> None:
    html = r"<p>$\frac{1}{2}$</p>"
    once, _, _ = convert_latex_spans(html)
    twice, converted, _ = convert_latex_spans(once)
    assert twice == once
    assert converted == 0  # nothing left to convert


# ---------------------------------------------------------------------------
# 5. Every emitted <math> passes the gate (the accept-gate contract).
# ---------------------------------------------------------------------------
def test_every_emitted_math_passes_validate_mathml() -> None:
    corpus = [
        r"15 \div 3",
        r"\frac{1}{2}",
        r"\sqrt[3]{x^2 + 1}",
        r"\left(\frac{20x}{1}\right)",
        r"a \neq b",
        r"\alpha^2 + \beta_1",
        r"\overline{AB}",
        r"\boxed{a} = x = 5",
        r"4^2 \text{Use definition of exponent.}",
        r"\quad 5 \cdot 3",
    ]
    for latex in corpus:
        mathml = latex_to_mathml(latex)
        assert mathml is not None, latex
        _assert_gate_valid(mathml)


# ---------------------------------------------------------------------------
# 6. The adapter seam: flag OFF is byte-identical; flag ON converts.
# ---------------------------------------------------------------------------
def _chapters(html: str) -> list:
    block = _AdapterBlock(
        html=html,
        region_kind="paragraph",
        raw_block_index=0,
        raw_text=r"$\frac{1}{2}$",
    )
    return [_AdapterChapter(title="Ch 1", blocks=[block])]


def test_flag_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMANTIK_LATEX_MATHML", raising=False)
    html = r"<p>$\frac{1}{2}$</p>"
    chapters = _chapters(html)
    _latex_to_mathml(chapters)
    assert chapters[0].blocks[0].html == html


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "garbage"])
def test_flag_falsey_and_garbage_are_off(
    monkeypatch: pytest.MonkeyPatch, val: str
) -> None:
    monkeypatch.setenv("SEMANTIK_LATEX_MATHML", val)
    assert _resolve_latex_mathml() is False
    html = r"<p>$\frac{1}{2}$</p>"
    chapters = _chapters(html)
    _latex_to_mathml(chapters)
    assert chapters[0].blocks[0].html == html


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON", "True"])
def test_flag_truthy_enables(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("SEMANTIK_LATEX_MATHML", val)
    assert _resolve_latex_mathml() is True


def test_flag_on_converts_html_only_and_leaves_raw_text_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEMANTIK_LATEX_MATHML", "1")
    chapters = _chapters(r"<p>$\frac{1}{2}$</p>")
    _latex_to_mathml(chapters)
    block = chapters[0].blocks[0]
    assert "<math " in block.html
    assert "<mfrac>" in block.html
    assert "$" not in block.html
    # raw_text is the content-hash sourceId basis — it MUST stay verbatim.
    assert block.raw_text == r"$\frac{1}{2}$"
