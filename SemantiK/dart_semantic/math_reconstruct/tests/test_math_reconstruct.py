"""T4 — deterministic MathML reconstruction tests.

Covers: super/subscript (unicode + geometry), built-up fraction, bracket
grouping -> valid MathML through the Stage-7 gate; the fallback NEVER
emitting raw ``math-source`` glyph soup when the feature is on; an
end-to-end reconstruct on a sample math region; and the flag-off
byte-identical (soup) behaviour.
"""
from __future__ import annotations

import pytest

from dart_semantic.assembler.fallbacks import fallback_math
from dart_semantic.gates.mathml_check import validate_mathml
from dart_semantic.math_reconstruct.glyph_extract import Glyph, glyphs_from_src_text
from dart_semantic.math_reconstruct.mathml_emit import (
    accessible_uniform_mathml,
    emit_mathml,
)
from dart_semantic.math_reconstruct.policy import (
    reconstruct_math,
    resolve_math_reconstruct_mode,
    stamp_math_reconstruction,
)
from dart_semantic.math_reconstruct.structure_infer import (
    Frac,
    Script,
    infer_structure,
)
from dart_semantic.structure_graph import Region

_FLAG = "SEMANTIK_MATH_RECONSTRUCT"


def _assert_valid(mathml: str) -> None:
    outcome = validate_mathml(mathml)
    assert outcome.passed, f"MathML failed gate: {outcome.message}\n{mathml}"


# ---------------------------------------------------------------------------
# Flag resolver.
# ---------------------------------------------------------------------------


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert resolve_math_reconstruct_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "On"])
def test_flag_truthy(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert resolve_math_reconstruct_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "garbage", ""])
def test_flag_falsey_or_garbage(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert resolve_math_reconstruct_mode() is False


# ---------------------------------------------------------------------------
# Super/subscript.
# ---------------------------------------------------------------------------


def test_unicode_superscript():
    glyphs = glyphs_from_src_text("x²")
    res = infer_structure(glyphs, geometry_source="src_text")
    assert res.can_reconstruct
    assert any(isinstance(n, Script) and n.kind == "msup" for n in res.root.children)
    mathml = emit_mathml(res.root, alttext="x squared")
    _assert_valid(mathml)
    assert "<msup>" in mathml


def test_geometry_superscript():
    # base "x" full size on baseline 10; "2" shrunk + raised -> superscript.
    glyphs = [
        Glyph("x", 0.0, 6.0, 0.0, 10.0, 10.0),
        Glyph("2", 6.0, 9.0, 0.0, 6.0, 6.0),
    ]
    res = infer_structure(glyphs, geometry_source="pdfplumber")
    assert res.complexity == "scripts"
    node = res.root.children[0]
    assert isinstance(node, Script) and node.kind == "msup"
    _assert_valid(emit_mathml(res.root, alttext="x squared"))


def test_geometry_subscript():
    # base "H" on baseline 10; "2" shrunk + lowered -> subscript.
    glyphs = [
        Glyph("H", 0.0, 6.0, 0.0, 10.0, 10.0),
        Glyph("2", 6.0, 9.0, 7.0, 13.0, 6.0),
    ]
    res = infer_structure(glyphs, geometry_source="pdfplumber")
    node = res.root.children[0]
    assert isinstance(node, Script) and node.kind == "msub"
    mathml = emit_mathml(res.root, alttext="H 2")
    _assert_valid(mathml)
    assert "<msub>" in mathml


# ---------------------------------------------------------------------------
# Fraction.
# ---------------------------------------------------------------------------


def test_geometry_fraction():
    glyphs = [
        Glyph("a", 0.0, 6.0, 0.0, 8.0, 8.0),     # numerator (above bar)
        Glyph("—", 0.0, 12.0, 9.0, 10.0, 8.0),    # wide fraction bar
        Glyph("b", 0.0, 6.0, 11.0, 19.0, 8.0),    # denominator (below bar)
    ]
    res = infer_structure(glyphs, geometry_source="pdfplumber")
    assert res.complexity == "fraction"
    assert isinstance(res.root.children[0], Frac)
    mathml = emit_mathml(res.root, alttext="a over b")
    _assert_valid(mathml)
    assert "<mfrac>" in mathml


# ---------------------------------------------------------------------------
# Bracket grouping.
# ---------------------------------------------------------------------------


def test_bracket_grouping():
    glyphs = glyphs_from_src_text("(a+b)")
    res = infer_structure(glyphs, geometry_source="src_text")
    mathml = emit_mathml(res.root, alttext="open a plus b close")
    _assert_valid(mathml)
    # The balanced () run nests into an <mrow>.
    assert "<mrow>" in mathml


def test_unbalanced_brackets_still_valid():
    glyphs = glyphs_from_src_text("a+b)")
    res = infer_structure(glyphs, geometry_source="src_text")
    _assert_valid(emit_mathml(res.root, alttext="a plus b close"))


# ---------------------------------------------------------------------------
# Unsupported shapes -> routed (not reconstructed deterministically).
# ---------------------------------------------------------------------------


def test_integral_routed_not_reconstructed():
    glyphs = glyphs_from_src_text("∫f(x)dx")
    res = infer_structure(glyphs, geometry_source="src_text")
    assert res.can_reconstruct is False
    assert res.complexity == "unsupported"


def test_multiline_routed():
    # Two genuine baseline rows, no fraction bar -> multiline -> unsupported.
    glyphs = [
        Glyph("a", 0.0, 6.0, 0.0, 8.0, 8.0),
        Glyph("=", 6.0, 12.0, 0.0, 8.0, 8.0),
        Glyph("b", 0.0, 6.0, 20.0, 28.0, 8.0),
        Glyph("=", 6.0, 12.0, 20.0, 28.0, 8.0),
    ]
    res = infer_structure(glyphs, geometry_source="pdfplumber")
    assert res.can_reconstruct is False


# ---------------------------------------------------------------------------
# End-to-end reconstruct_math.
# ---------------------------------------------------------------------------


def test_reconstruct_math_deterministic_unicode():
    res = reconstruct_math(src_text="x²+1=0", display=True)
    assert res.method == "deterministic"
    assert res.ok
    _assert_valid(res.mathml)
    assert "math-source" not in res.mathml


def test_reconstruct_math_blind_structured_routes_to_uniform():
    # No geometry, no unicode scripts, but the council saw 2-D structure
    # (subsup_count>0) -> deterministic declines -> no generative_fn ->
    # accessible-uniform last resort (still valid, never soup).
    res = reconstruct_math(
        src_text="x 2 + 1",
        glyph_density_features={"subsup_count": 2.0},
    )
    assert res.method == "accessible_uniform"
    _assert_valid(res.mathml)
    assert "math-source" not in res.mathml


def test_reconstruct_math_generative_tier_local():
    # Provide a fake LOCAL generative_fn; deterministic declines (structured
    # but blind) -> generative tier accepts a gate-valid candidate.
    def fake_local(src):
        return '<math alttext="g"><mrow><mi>g</mi></mrow></math>'

    res = reconstruct_math(
        src_text="x 2",
        glyph_density_features={"subsup_count": 1.0},
        generative_fn=fake_local,
    )
    assert res.method == "generative_local"
    _assert_valid(res.mathml)


def test_generative_invalid_falls_to_uniform():
    def bad_local(src):
        return "<span>not mathml</span>"

    res = reconstruct_math(
        src_text="x 2",
        glyph_density_features={"subsup_count": 1.0},
        generative_fn=bad_local,
    )
    assert res.method == "accessible_uniform"
    _assert_valid(res.mathml)


# ---------------------------------------------------------------------------
# Accessible-uniform last resort: NEVER raw soup.
# ---------------------------------------------------------------------------


def test_accessible_uniform_is_valid_math():
    mathml = accessible_uniform_mathml("∑ x_i / y")
    _assert_valid(mathml)
    assert "math-source" not in mathml
    assert mathml.startswith("<math")


def test_accessible_uniform_empty_source_merror():
    mathml = accessible_uniform_mathml("")
    _assert_valid(mathml)
    assert "<merror>" in mathml


# ---------------------------------------------------------------------------
# fallback_math: flag-off soup byte-identical; flag-on NEVER soup.
# ---------------------------------------------------------------------------


def test_fallback_math_flag_off_is_legacy_soup(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    region = Region(kind="math", feature_block_indices=(),
                    payload={"src_text": "x^2 + 1"})
    out = fallback_math(region, [])
    assert out == '<span class="math-source">x^2 + 1</span>'


def test_fallback_math_flag_on_never_soup(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    region = Region(kind="math", feature_block_indices=(),
                    payload={"src_text": "x^2 + 1"})
    out = fallback_math(region, [])
    assert "math-source" not in out
    assert out.startswith("<math")
    _assert_valid(out)


def test_fallback_math_prefers_stamped_deterministic(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    stamped = '<math display="block" alttext="x squared"><msup><mi>x</mi><mn>2</mn></msup></math>'
    region = Region(
        kind="math",
        feature_block_indices=(),
        payload={"src_text": "x 2", "math_reconstruct": {
            "method": "deterministic", "mathml": stamped}},
    )
    out = fallback_math(region, [])
    assert out == stamped


# ---------------------------------------------------------------------------
# stamp_math_reconstruction (Stage-5 payload stamping).
# ---------------------------------------------------------------------------


def test_stamp_deterministic_payload():
    payload = {"src_text": "x²+1", "display": True, "glyph_density_features": {}}
    stamped = stamp_math_reconstruction(payload)
    assert stamped is not None
    assert payload["math_reconstruct"]["method"] == "deterministic"
    _assert_valid(payload["math_reconstruct"]["mathml"])
    assert "glyph_geometry" in payload["math_reconstruct"]


def test_stamp_skips_unreconstructable():
    # blind + structured -> deterministic declines -> NOT stamped (Stage-6
    # generative + assembler fallback own the rest).
    payload = {"src_text": "x 2", "display": True,
               "glyph_density_features": {"subsup_count": 2.0}}
    stamped = stamp_math_reconstruction(payload)
    assert stamped is None
    assert "math_reconstruct" not in payload
