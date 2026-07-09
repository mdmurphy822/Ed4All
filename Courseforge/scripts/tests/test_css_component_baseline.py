"""7B→Sonnet parity P1: COURSEFORGE_CSS component baseline + page shell.

Asserts the Sonnet IxD component classes (ported from
``plans/finegrain/sonnet-ixd-component-spec.md``) are codified in the single
``COURSEFORGE_CSS`` constant, and that a ``_wrap_page`` page carries the full
stylesheet AND a real document ``<head>``. No LLM, no pipeline run.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_course import (  # noqa: E402
    COURSEFORGE_CSS,
    COURSEFORGE_RICHER_CSS,
    _THEME_OVERRIDE_CSS,
    _wrap_page,
    patch_css_in_export,
    patch_css_in_html,
)


SONNET_COMPONENT_CLASSES = [
    ".example-box",
    ".definition-box",
    ".key-rule",
    ".worked-example",
    ".formula-box",
    ".self-check-item",
    "table.place-value",
    ".alert-info",
    ".alert-warning",
    ".alert-success",
]


def test_css_carries_sonnet_component_classes():
    for cls in SONNET_COMPONENT_CLASSES:
        assert cls in COURSEFORGE_CSS, f"{cls} missing from COURSEFORGE_CSS"


def test_worked_example_substructure_styled():
    # Multi-step worked-example sub-elements have declarations.
    for cls in (".step-label", ".step-row", ".solution-line"):
        assert cls in COURSEFORGE_CSS, f"{cls} missing from COURSEFORGE_CSS"


def test_box_color_accents_present():
    # Left-border color accents (the visual signature of the boxes).
    assert "border-left: 4px solid #007bff" in COURSEFORGE_CSS  # example-box
    assert "border-left: 4px solid #ffc107" in COURSEFORGE_CSS  # definition-box
    assert "border-left: 4px solid #17a2b8" in COURSEFORGE_CSS  # key-rule


# ---------------------------------------------------------------------------
# P3 — instruction-block variety: six block types that previously rendered as
# bare <section> prose now carry a distinct styled component class.
# ---------------------------------------------------------------------------

BLOCK_VARIETY_COMPONENT_CLASSES = [
    ".takeaway-card",
    ".misconception-card",
    ".prereq-card",
    ".recap-box",
    ".reflection-prompt",
    ".discussion-prompt",
]


def test_css_carries_block_variety_component_classes():
    for cls in BLOCK_VARIETY_COMPONENT_CLASSES:
        assert cls in COURSEFORGE_CSS, f"{cls} missing from COURSEFORGE_CSS"


def test_block_variety_components_have_explicit_dark_text_color():
    # Each new card is a light/colored background box that MUST declare an
    # explicit dark text color so it never inherits inverted near-white
    # (Defect-D parity).
    for cls in BLOCK_VARIETY_COMPONENT_CLASSES:
        body = _rule_body(COURSEFORGE_CSS, cls)
        m = _TEXT_COLOR_RE.search(body)
        assert m, f"{cls} has a background but no explicit text color (Defect D)"
        hexval = m.group(1).lstrip("#")
        if len(hexval) == 3:
            hexval = "".join(c * 2 for c in hexval)
        r, g, b = (int(hexval[i:i + 2], 16) for i in (0, 2, 4))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert lum < 64, (
            f"{cls} text color {m.group(1)} is too light (luminance {lum:.0f})"
        )


# ---------------------------------------------------------------------------
# Defect D: white-on-white / contrast regression.
# ---------------------------------------------------------------------------

# Light-background component classes that the deleted dark-mode block left
# inheriting near-white body text. Each MUST now declare an explicit text
# color so it never collides regardless of color scheme.
LIGHT_BG_COMPONENTS = [
    ".example-box",
    ".definition-box",
    ".key-rule",
    ".formula-box",
    ".worked-example",
    ".self-check-item",
    ".callout-warning",
    ".callout-success",
    ".sc-option.correct",
    ".sc-option.incorrect",
    # Wave-2 block-variety component classes. Each is a light-background
    # box that MUST declare an explicit dark text color (Defect-D parity).
    ".scenario-card",
    ".problem-card",
    ".vocab-card",
    ".formula-card",
    ".checklist",
]


def _rule_body(css: str, selector: str) -> str:
    """Return the declaration block ``{...}`` for an exact selector rule."""
    # Anchor the selector so ``.worked-example`` doesn't match
    # ``.worked-example .solution-line``; require the selector be followed
    # by optional whitespace then ``{``.
    pat = re.compile(
        r"(?:^|\n)\s*" + re.escape(selector) + r"\s*\{([^}]*)\}"
    )
    m = pat.search(css)
    assert m, f"rule for {selector} not found in COURSEFORGE_CSS"
    return m.group(1)


# A bare ``color:`` declaration (NOT ``border-color:`` / ``background-color:``).
_TEXT_COLOR_RE = re.compile(r"(?<![-\w])color:\s*(#[0-9a-fA-F]{3,6})")


def test_no_light_bg_component_lacks_explicit_text_color():
    for cls in LIGHT_BG_COMPONENTS:
        body = _rule_body(COURSEFORGE_CSS, cls)
        assert _TEXT_COLOR_RE.search(body), (
            f"{cls} has a light background but no explicit text color — "
            "it will inherit inverted/near-white body text (Defect D)."
        )


def test_solution_line_has_explicit_text_color():
    # Nested rule carries its own color too.
    body = _rule_body(COURSEFORGE_CSS, ".worked-example .solution-line")
    assert _TEXT_COLOR_RE.search(body)


def test_dark_mode_media_block_is_gone():
    # The OS-dark-scheme media block was the root cause — it must be deleted
    # entirely (matching Sonnet's baseline with no scheme override).
    assert "prefers-color-scheme: dark" not in COURSEFORGE_CSS
    # The body-text inversion it set (near-white #e2e8f0 ON body) is gone —
    # body no longer flips to a light text color under any scheme. (#e2e8f0
    # legitimately survives as the .callout border-color, so we assert the
    # specific inversion declaration, not the bare hex.)
    assert "color: #e2e8f0" not in COURSEFORGE_CSS
    assert "color:#e2e8f0" not in COURSEFORGE_CSS


def test_light_bg_text_color_is_dark_for_aa_contrast():
    # The explicit color is dark (#1a1a1a-class), giving >7:1 (AAA) on the
    # light component backgrounds — never the near-white that caused Defect D.
    for cls in LIGHT_BG_COMPONENTS:
        body = _rule_body(COURSEFORGE_CSS, cls)
        m = _TEXT_COLOR_RE.search(body)
        assert m, f"{cls} text color not a hex literal"
        hexval = m.group(1).lstrip("#")
        if len(hexval) == 3:
            hexval = "".join(c * 2 for c in hexval)
        r, g, b = (int(hexval[i:i + 2], 16) for i in (0, 2, 4))
        # Relative luminance proxy — dark text means low luminance.
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert lum < 64, (
            f"{cls} text color {m.group(1)} is too light "
            f"(luminance {lum:.0f}); would risk low contrast on a light box."
        )


def test_patch_css_in_html_swaps_style_block():
    stale = (
        "<!DOCTYPE html><html><head>"
        "<style>.example-box { background: #f8f9fa; }"
        "@media (prefers-color-scheme: dark){body{color:#e2e8f0}}</style>"
        "</head><body>x</body></html>"
    )
    patched, changed = patch_css_in_html(stale)
    assert changed is True
    assert "prefers-color-scheme: dark" not in patched
    # The fixed CSS is now embedded.
    assert "color: #1a1a1a" in patched
    # Body content preserved.
    assert ">x<" in patched


def test_patch_css_in_html_is_idempotent_noop_when_already_fixed():
    fixed = f"<html><head><style>{COURSEFORGE_CSS}</style></head><body/></html>"
    patched, changed = patch_css_in_html(fixed)
    assert changed is False
    assert patched == fixed


def test_patch_css_in_html_noop_when_no_style_block():
    plain = "<html><head></head><body>nothing</body></html>"
    patched, changed = patch_css_in_html(plain)
    assert changed is False
    assert patched == plain


def test_patch_css_in_export_round_trip(tmp_path):
    week = tmp_path / "week_01"
    week.mkdir()
    stale_css = (
        "<style>.example-box { background:#f8f9fa; }"
        "@media (prefers-color-scheme: dark){body{color:#e2e8f0}}</style>"
    )
    p1 = week / "week_01_content_01.html"
    p1.write_text(f"<html><head>{stale_css}</head><body>a</body></html>", encoding="utf-8")
    # A file with no <style> must be left untouched.
    p2 = tmp_path / "readme.html"
    p2.write_text("<html><body>no style</body></html>", encoding="utf-8")

    # Dry-run reports the change without writing.
    dry = patch_css_in_export(tmp_path, dry_run=True)
    assert p1 in dry and p2 not in dry
    assert "prefers-color-scheme: dark" in p1.read_text(encoding="utf-8")

    # Real run patches p1 only.
    changed = patch_css_in_export(tmp_path)
    assert changed == [p1]
    out = p1.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" not in out
    assert ">a<" in out
    assert p2.read_text(encoding="utf-8") == "<html><body>no style</body></html>"

    # Second run is a no-op (idempotent).
    assert patch_css_in_export(tmp_path) == []


# ---------------------------------------------------------------------------
# §5.2 — block-TYPE identity CSS (left rail + uppercase eyebrow), keyed on the
# already-emitted data-cf-content-type / data-cf-teaching-role attributes. Lives
# ONLY in COURSEFORGE_RICHER_CSS so flags-off output stays byte-identical.
# ---------------------------------------------------------------------------

# (content_type value, eyebrow label, role token) triples the identity CSS
# gives a rail + eyebrow. Exposition (explanation/overview/comparison) is
# DELIBERATELY excluded (stays calm prose).
_IDENTITY_CONTENT_TYPES = [
    ("definition", "Definition", "--cf-role-definition"),
    ("example", "Example", "--cf-role-example"),
    ("procedure", "Procedure", "--cf-role-practice"),
    ("exercise", "Practice", "--cf-role-practice"),
    ("summary", "Summary", "--cf-role-summary"),
]


def test_block_identity_rail_and_eyebrow_in_richer_css():
    for ct, label, token in _IDENTITY_CONTENT_TYPES:
        # A heading-scoped rail selector + a matching ::before eyebrow.
        assert f'h2[data-cf-content-type="{ct}"]' in COURSEFORGE_RICHER_CSS, (
            f"no rail selector for content-type {ct}"
        )
        assert f'h3[data-cf-content-type="{ct}"]::before' in COURSEFORGE_RICHER_CSS
        assert f'content: "{label}"' in COURSEFORGE_RICHER_CSS, (
            f"eyebrow label {label!r} missing for {ct}"
        )
        assert f"var({token})" in COURSEFORGE_RICHER_CSS


def test_block_identity_self_check_via_teaching_role():
    # self-check carries data-cf-teaching-role="assess" (map_role output).
    assert '[data-cf-teaching-role="assess"]::before' in COURSEFORGE_RICHER_CSS
    assert 'content: "Self-Check"' in COURSEFORGE_RICHER_CSS
    assert "var(--cf-role-assess)" in COURSEFORGE_RICHER_CSS


def test_block_identity_eyebrow_is_uppercase_block():
    # The eyebrow ::before is a block-level uppercase label.
    assert "text-transform: uppercase" in COURSEFORGE_RICHER_CSS


def test_block_identity_leaves_exposition_calm():
    # Anti-over-decoration: the default/most-common content-type "explanation"
    # (and the calm exposition siblings) get NO identity rail/eyebrow.
    for ct in ("explanation", "overview", "comparison"):
        assert f'data-cf-content-type="{ct}"]::before' not in COURSEFORGE_RICHER_CSS, (
            f"exposition content-type {ct} should stay calm (no eyebrow)"
        )


def test_block_identity_absent_from_legacy_css():
    # Byte-stability: the identity CSS never touches the always-emitted baseline.
    assert '[data-cf-content-type="example"]::before' not in COURSEFORGE_CSS
    assert '[data-cf-teaching-role="assess"]::before' not in COURSEFORGE_CSS


def test_wrapped_page_has_real_head_and_stylesheet():
    page = _wrap_page(
        title="week_01_content_01",
        course_code="PHYS_101",
        week_num=1,
        body_html="<section><p>Body.</p></section>",
    )
    # Real document head.
    assert "<!DOCTYPE html>" in page
    assert "<head>" in page and "</head>" in page
    assert '<meta charset="UTF-8">' in page
    # Full stylesheet inlined, incl. a component class.
    assert "<style>" in page
    assert ".example-box" in page
    # Body chrome from _wrap_page.
    assert 'id="main-content"' in page
