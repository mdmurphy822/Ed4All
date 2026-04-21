"""Wave 16 tests for ``DART.converter.mathml``.

Coverage: LaTeX delimiter detection (``$...$``, ``\\(...\\)``,
``\\[...\\]``), plain-text equation detection, rendered MathML shape,
and guardrails against false positives on ordinary prose.
"""

from __future__ import annotations

import pytest

from DART.converter.mathml import (
    DetectedFormula,
    detect_formulas,
    render_block_mathml,
    render_mathml,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestDetectFormulas:
    def test_dollar_delimited_latex(self):
        found = detect_formulas("Consider $x^2 + y^2 = z^2$ today.")
        assert len(found) == 1
        assert found[0].body == "x^2 + y^2 = z^2"
        assert found[0].delimiter == "dollar"

    def test_paren_delimited_latex(self):
        found = detect_formulas(r"Let \(a + b = c\) hold.")
        assert len(found) == 1
        assert found[0].body == "a + b = c"
        assert found[0].delimiter == "paren"

    def test_bracket_delimited_display_latex(self):
        found = detect_formulas(r"\[\sum_{i=1}^n i = n(n+1)/2\]")
        assert len(found) == 1
        assert found[0].delimiter == "bracket"
        assert "sum" in found[0].body

    def test_plain_equation_on_a_line(self):
        found = detect_formulas("E = mc^2")
        assert len(found) == 1
        assert found[0].delimiter == "plain"
        assert found[0].body == "E = mc^2"

    def test_empty_text_returns_empty(self):
        assert detect_formulas("") == []
        assert detect_formulas("   \n\t  ") == []

    def test_prose_not_detected(self):
        # An English sentence with an '=' but no operators and long words
        # must not match the plain-equation pattern.
        found = detect_formulas("We agreed that this approach equals good")
        assert found == []

    def test_currency_not_detected(self):
        # A ``$5.00`` style string must not match the dollar LaTeX pattern.
        found = detect_formulas("It cost $5 and 00 cents.")
        assert found == []

    def test_multiple_inline_formulas_detected(self):
        found = detect_formulas("We have $a$ and $b$ and $c$.")
        assert len(found) == 3
        assert [f.body for f in found] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestRenderMathml:
    def test_mathml_shape(self):
        out = render_mathml("E = mc^2")
        assert out.startswith("<math")
        assert "xmlns=\"http://www.w3.org/1998/Math/MathML\"" in out
        assert "<semantics>" in out
        assert "<mtext>E = mc^2</mtext>" in out
        assert '<annotation encoding="text/plain">E = mc^2</annotation>' in out
        assert out.endswith("</math>")

    def test_mathml_escapes_html_special_chars(self):
        out = render_mathml("a < b & c > d")
        assert "&lt;" in out
        assert "&amp;" in out
        assert "&gt;" in out

    def test_mathml_display_attribute(self):
        out = render_mathml("x", display="inline")
        assert 'display="inline"' in out

    def test_mathml_fallback_independent_from_body(self):
        out = render_mathml("x^2", fallback="x squared")
        assert "<mtext>x^2</mtext>" in out
        assert ">x squared</annotation>" in out

    def test_render_block_mathml_detects_first_formula(self):
        out = render_block_mathml("Consider $a = b$ here.")
        assert out is not None
        assert "<mtext>a = b</mtext>" in out

    def test_render_block_mathml_returns_none_for_prose(self):
        assert render_block_mathml("This is just prose.") is None

    def test_render_block_mathml_returns_none_for_empty(self):
        assert render_block_mathml("") is None
        assert render_block_mathml("   ") is None
