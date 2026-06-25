"""Richer Visual System — Phase 0 (richer-visual-system-plan-2026-06.md §4).

Phase 0 is the keystone: (a) the emitted course pages hardcode hex referencing
ZERO --cf-* tokens, AND (b) never inject the a11y themes — so high_contrast.css
/ dyslexia_friendly.css cannot reskin the output. Phase 0 fixes BOTH behind a
default-OFF flag (``ED4ALL_RICHER_VISUAL_SYSTEM``), byte-identical when off.

These tests prove:
- (a) flag-OFF emitted page + <style> are BYTE-IDENTICAL to the HEAD baseline
      (and the patch helpers stay byte-identical too);
- (b) the token prelude stays in sync with templates/_base/variables.css;
- (c) compute_content_hash is invariant across flag on/off (META-only);
- (d) THE KEYSTONE PROOF — with the flag ON + a theme appended, a representative
      box resolves THROUGH a token to the THEME's value, demonstrating the
      themes can now reskin (they cannot today).

No LLM, no pipeline run.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_course as gc  # noqa: E402
from generate_course import (  # noqa: E402
    COURSEFORGE_CSS,
    COURSEFORGE_RICHER_CSS,
    COURSEFORGE_TOKENS_CSS,
    _page_style_block,
    _wrap_page,
    patch_css_in_html,
)

_VARIABLES_CSS = (
    Path(__file__).resolve().parents[2]
    / "templates"
    / "_base"
    / "variables.css"
)


def _wrap_kwargs():
    return dict(
        title="week_01_content_01",
        course_code="PHYS_101",
        week_num=1,
        body_html="<section><p>Body.</p></section>",
    )


# ---------------------------------------------------------------------------
# (a) BYTE-STABILITY WHEN FLAG OFF
# ---------------------------------------------------------------------------


def test_style_block_off_is_exactly_courseforge_css(monkeypatch):
    # No richer flag, no typed-callout flag → the <style> inner CSS is exactly
    # COURSEFORGE_CSS (the historical baseline), byte-for-byte.
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    monkeypatch.delenv("ED4ALL_CALLOUT_TYPED", raising=False)
    assert _page_style_block() == COURSEFORGE_CSS
    assert _page_style_block(theme="high_contrast") == COURSEFORGE_CSS


def test_wrapped_page_off_byte_identical_to_legacy_shell(monkeypatch):
    # Re-derive the legacy <style> emission and confirm the live _wrap_page
    # output embeds exactly that — proving flag-off is byte-identical.
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    monkeypatch.delenv("ED4ALL_CALLOUT_TYPED", raising=False)
    page = _wrap_page(**_wrap_kwargs())
    assert f"<style>{COURSEFORGE_CSS}</style>" in page
    # The richer prelude must NOT leak into a flag-off page.
    assert ":root" not in page
    assert "--cf-ink" not in page


def test_wrapped_page_off_stable_regardless_of_theme_arg(monkeypatch):
    # A theme arg is inert while the flag is off (the page is byte-identical
    # whether or not a theme is requested).
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    monkeypatch.delenv("ED4ALL_CALLOUT_TYPED", raising=False)
    a = _wrap_page(**_wrap_kwargs())
    b = _wrap_page(theme="high_contrast", **_wrap_kwargs())
    assert a == b


def test_patch_helper_off_byte_identical(monkeypatch):
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    monkeypatch.delenv("ED4ALL_CALLOUT_TYPED", raising=False)
    # A page already carrying the canonical <style> is a no-op under patch.
    fixed = f"<html><head><style>{COURSEFORGE_CSS}</style></head><body/></html>"
    patched, changed = patch_css_in_html(fixed)
    assert changed is False
    assert patched == fixed


# ---------------------------------------------------------------------------
# (b) TOKEN PRELUDE <-> variables.css SYNC
# ---------------------------------------------------------------------------

_DECL_RE = re.compile(r"(--cf-[a-z0-9-]+)\s*:\s*([^;]+);")


def _tokens_from(css: str) -> dict:
    """Extract every ``--cf-NAME: VALUE;`` declaration as a name->value map."""
    out = {}
    for name, value in _DECL_RE.findall(css):
        out[name] = " ".join(value.split())  # normalize internal whitespace
    return out


def test_richer_token_prelude_in_sync():
    variables = _tokens_from(_VARIABLES_CSS.read_text(encoding="utf-8"))
    prelude = _tokens_from(COURSEFORGE_TOKENS_CSS)
    assert variables, "no --cf-* tokens found in variables.css"
    assert prelude, "no --cf-* tokens found in COURSEFORGE_TOKENS_CSS"
    # The checked-in mirror must declare exactly the variables.css token set,
    # name-for-name and value-for-value.
    assert prelude == variables, (
        "COURSEFORGE_TOKENS_CSS drifted from variables.css. "
        f"only-in-variables={set(variables) - set(prelude)} "
        f"only-in-prelude={set(prelude) - set(variables)} "
        f"value-mismatches="
        + str({k: (variables[k], prelude[k]) for k in set(variables) & set(prelude)
                if variables[k] != prelude[k]})
    )


def test_prelude_declares_new_exact_value_tokens():
    prelude = _tokens_from(COURSEFORGE_TOKENS_CSS)
    # Plan §2.1 — exact emitted hex, NOT near-neighbors.
    assert prelude["--cf-primary-tint"] == "#ebf8ff"
    assert prelude["--cf-ink"] == "#1a1a1a"
    assert prelude["--cf-heading"] == "#1a365d"
    assert prelude["--cf-heading-strong"] == "#2d3748"
    assert prelude["--cf-heading-muted"] == "#495057"
    # Darkened signal variants (>=3:1) for orange/teal.
    assert prelude["--cf-accent-orange-strong"] == "#b35900"
    assert prelude["--cf-accent-teal-strong"] == "#0f7c63"


# ---------------------------------------------------------------------------
# (c) compute_content_hash INVARIANT across flag on/off (META-only)
# ---------------------------------------------------------------------------


def _representative_block():
    from blocks import Block  # noqa: WPS433 (local import; sys.path set above)

    return Block(
        block_id="week_01_content_01#misconception_demo_0",
        block_type="misconception",
        page_id="week_01_content_01",
        sequence=0,
        content="A common mistake learners make here, and its correction.",
        key_terms=("misconception",),
        objective_ids=("CO-01",),
        bloom_level="understand",
    )


def test_content_hash_invariant_across_flag(monkeypatch):
    block = _representative_block()
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    off = block.compute_content_hash()
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    on = block.compute_content_hash()
    assert off == on, "contentHash must be META-only — CSS flag must not drift it"


# ---------------------------------------------------------------------------
# (d) THE KEYSTONE PROOF — themes can now reskin the emitted boxes
# ---------------------------------------------------------------------------


def _rule_body(css: str, selector: str) -> str:
    pat = re.compile(r"(?:^|\n)\s*" + re.escape(selector) + r"\s*\{([^}]*)\}")
    m = pat.search(css)
    assert m, f"rule for {selector} not found"
    return m.group(1)


def _resolve_var(token_css: str, var_name: str, _depth: int = 0) -> str:
    """Resolve a --cf-* token through the :root declarations to a literal.

    Follows ``var(--cf-...)`` indirection (the tokens re-point to each other).
    """
    assert _depth < 20, f"cycle resolving {var_name}"
    tokens = _tokens_from(token_css)
    value = tokens[var_name]
    m = re.fullmatch(r"var\((--cf-[a-z0-9-]+)\)", value.strip())
    if m:
        return _resolve_var(token_css, m.group(1), _depth + 1)
    return value.strip()


def test_keystone_default_box_resolves_to_baseline_hex():
    # With NO theme override, the richer .misconception-card resolves through
    # its tokens to the SAME hex the legacy rule hardcoded — proving the
    # token-consuming re-statement is value-identical (zero default visual diff).
    body = _rule_body(COURSEFORGE_RICHER_CSS, ".misconception-card")
    # Background re-points at --cf-danger-light; text at --cf-ink.
    assert "var(--cf-danger-light)" in body
    assert "var(--cf-ink)" in body
    assert _resolve_var(COURSEFORGE_TOKENS_CSS, "--cf-danger-light") == "#f8d7da"
    assert _resolve_var(COURSEFORGE_TOKENS_CSS, "--cf-ink") == "#1a1a1a"


def test_keystone_high_contrast_theme_reskins_a_box(monkeypatch):
    # THE PROOF the defect is fixed: render flag-ON + high_contrast theme; the
    # representative box's text + background tokens now resolve to the THEME's
    # high-contrast values (#000000 / #cce0ff), demonstrating the theme finally
    # reskins the emitted box — which it CANNOT do on HEAD (COURSEFORGE_CSS
    # references zero tokens, and the theme is never injected at all).
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block(theme="high_contrast")

    # The page now carries BOTH the token prelude AND the theme override.
    assert style.count(":root") >= 2, "expected token prelude + theme override"
    # The theme override is appended AFTER the prelude (cascade-by-source-order).
    assert style.rindex("--cf-ink: #000000") > style.index("--cf-ink: #1a1a1a")

    # Resolve --cf-ink / --cf-danger-light against the FINAL (post-theme)
    # cascade: the theme's :root overrides win because it is declared last.
    final = _last_root_token_values(style)
    assert final["--cf-ink"] == "#000000", "HC theme failed to reskin box text"
    assert final["--cf-danger-light"] == "#ffcccc", "HC theme failed to reskin box bg"
    assert final["--cf-primary-tint"] == "#cce0ff"


def test_keystone_seam_absent_on_head_baseline():
    # Sanity: the legacy COURSEFORGE_CSS references ZERO --cf-* tokens (the
    # root cause). This is the "they cannot today" half of the keystone claim.
    assert "var(--cf" not in COURSEFORGE_CSS


def test_dyslexia_theme_reskins_font_family(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block(theme="dyslexia_friendly")
    final = _last_root_token_values(style)
    assert "OpenDyslexic" in final["--cf-font-family"]
    # The richer body rule consumes --cf-ink (color); the dyslexia override
    # leaves color untouched but re-points the font/spacing tokens — proving the
    # SAME seam carries a typographic theme, not only a color theme.
    assert final["--cf-line-height-base"] == "1.8"


def test_unknown_theme_is_inert_but_richer_css_still_present(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block(theme="does_not_exist")
    # Tokens + components are present; no second :root override appended.
    assert "--cf-ink: #1a1a1a" in style
    assert style.count(":root") == 1


# ---------------------------------------------------------------------------
# Cascade helper — collapse all :root blocks in source order (last wins).
# ---------------------------------------------------------------------------


def _last_root_token_values(css: str) -> dict:
    """Apply every ``:root{...}`` block in source order; last declaration wins.

    Models the browser cascade for same-specificity ``:root`` token declarations
    across appended <style> segments.
    """
    out: dict = {}
    for m in re.finditer(r":root\s*\{([^}]*)\}", css, re.DOTALL):
        for name, value in _DECL_RE.findall(m.group(1)):
            out[name] = " ".join(value.split())
    return out


# ===========================================================================
# PHASE 1 — style the previously-UNSTYLED block types (0 CSS rules on HEAD)
# ===========================================================================

# The block types that carried ZERO CSS rules on HEAD (plan §1 / §4 Phase 1).
_PHASE1_NEW_CLASSES = [
    ".multimedia",
    ".media-pending",
    ".media-a11y-badge",
    ".diagram",
    ".diagram-longdesc",
    ".diagram-pending",
    ".hook",
    ".resources",
    ".guided-practice",
    ".key-terms",
    ".reflection-calibration",
    ".reveal",
    ".confidence-capture",
    ".confidence-option",
    ".scenario-debrief",
    ".assessment-item",
    ".stakes-label",
]


def test_phase1_new_type_classes_appear_in_richer_css():
    # Each previously-0-rule class now has at least one rule in the richer CSS.
    for cls in _PHASE1_NEW_CLASSES:
        assert re.search(r"(?:^|\n|\s)" + re.escape(cls) + r"[\s{,:.\[]", COURSEFORGE_RICHER_CSS), (
            f"{cls} has no rule in COURSEFORGE_RICHER_CSS"
        )


def test_phase1_new_type_classes_absent_from_legacy_css():
    # Proof these were unstyled on HEAD: the legacy COURSEFORGE_CSS carries none
    # of the IB5 / new-type classes.
    for cls in (".multimedia", ".diagram", ".hook", ".resources",
                ".guided-practice", ".key-terms", ".reflection-calibration",
                ".scenario-debrief", ".assessment-item"):
        assert cls not in COURSEFORGE_CSS, f"{cls} unexpectedly present in legacy CSS"


def test_phase1_rules_consume_tokens_not_raw_hex():
    # Every Phase-1 rule body must resolve through --cf-* tokens (no raw hex in
    # the new-type rule bodies — plan: "never raw hex"). Spot-check the media
    # card + fade rail + diagram table.
    for selector in (".multimedia", ".diagram", ".hook", ".resources",
                     ".key-terms", ".reflection-calibration",
                     "[data-cf-fade-state=\"completion\"]"):
        body = _rule_body(COURSEFORGE_RICHER_CSS, selector)
        assert "var(--cf-" in body, f"{selector} does not consume a token"
        assert not re.search(r"#[0-9a-fA-F]{3,6}\b", body), (
            f"{selector} carries a raw hex literal: {body!r}"
        )


def test_worked_example_targets_actual_emitted_classes():
    # REGRESSION on the dead-CSS bug (plan §1): the richer CSS must target the
    # ACTUAL emitted worked_example classes, not the dead .step-row/.solution-line.
    for cls in (".worked-example-problem", ".worked-example-steps",
                ".subgoal-label", ".worked-example .why"):
        assert cls in COURSEFORGE_RICHER_CSS, f"{cls} missing — dead-CSS bug not fixed"


def test_assessment_item_distinct_from_self_check():
    # assessment_item reads as GRADED (heavier frame, --cf-role-assess /
    # primary-dark) vs the lighter formative .self-check.
    body = _rule_body(COURSEFORGE_RICHER_CSS, ".assessment-item")
    assert "var(--cf-role-assess)" in body
    assert "2px solid" in body  # heavier frame than self-check's 1px


def test_fade_completion_uses_orange_strong_not_weak_orange():
    # The completion rail uses --cf-accent-orange-strong (>=3:1), never the weak
    # --cf-accent-orange as a SOLE signal (plan §2.2 / §3 1.4.11).
    assert _resolve_var(COURSEFORGE_TOKENS_CSS, "--cf-fade-completion") == "#b35900"
    assert _resolve_var(COURSEFORGE_TOKENS_CSS, "--cf-accent-orange-strong") == "#b35900"


def test_group_b_exposition_stays_calm_prose():
    # Anti-over-decoration: no NEW box/rail rule is added for the exposition
    # group selectors (concept/explanation prose sections). The richer CSS does
    # not introduce a .concept-section / .explanation-section box.
    assert ".concept-section" not in COURSEFORGE_RICHER_CSS
    assert ".explanation-section" not in COURSEFORGE_RICHER_CSS


# ===========================================================================
# PHASE 1 — additive renderer emits (flag-gated, byte-stable off)
# ===========================================================================


def _block(block_type, **kw):
    from blocks import Block  # noqa: WPS433
    base = dict(block_id=f"p#{block_type}_x_0", block_type=block_type,
                page_id="p", sequence=0, content={})
    base.update(kw)
    return Block(**base)


def test_multimedia_badge_row_gated(monkeypatch):
    blk = _block("multimedia", content={"caption": "c"})
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    off = gc._render_multimedia_section(blk)
    assert "media-a11y-badge" not in off
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    on = gc._render_multimedia_section(blk)
    assert "media-a11y-badge" in on
    assert "Captions" in on and "Audio description" in on
    assert 'aria-hidden="true"' in on  # decorative glyph


def test_worked_example_eyebrow_gated(monkeypatch):
    blk = _block("worked_example",
                 content={"problem": "P", "steps": [{"subgoal": "S", "body": "B"}]},
                 fade_state="completion")
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    off = gc._render_worked_example_section(blk)
    assert "fade-state-eyebrow" not in off
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    on = gc._render_worked_example_section(blk)
    assert "fade-state-eyebrow" in on
    assert "Completion practice" in on


def test_self_check_stakes_label_gated(monkeypatch):
    qs = [{"question": "Q?", "options": [{"text": "a", "correct": True},
                                          {"text": "b"}]}]
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    off = gc._render_self_check(qs, page_id="p")
    assert "stakes-label" not in off
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    on = gc._render_self_check(qs, page_id="p")
    assert "stakes-label" in on and "stakes-formative" in on
    assert "not graded" in on


# ===========================================================================
# PHASE 3 — theme parity + motion / print / focus
# ===========================================================================


def test_focus_visible_ring_present_when_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block()
    assert ":focus-visible" in style
    assert "var(--cf-focus-color)" in style
    # HEAD's emitted CSS defines NO :focus rule.
    assert ":focus-visible" not in COURSEFORGE_CSS


def test_print_media_query_present_when_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block()
    assert "@media print" in style
    # Force-open <details> for transcript / long-desc / reveal.
    assert "details" in style.split("@media print", 1)[1][:600]
    assert "@media print" not in COURSEFORGE_CSS


def test_reduced_motion_query_present_when_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block()
    assert "@media (prefers-reduced-motion: reduce)" in style


def test_target_size_floor_on_interactive_targets():
    # 2.5.8 — >=24px min on the interactive targets.
    body = _rule_body(COURSEFORGE_RICHER_CSS, ".sc-option, .confidence-option, .flip-card")
    assert "24px" in body


def test_hc_theme_reskins_a_new_accent_token(monkeypatch):
    # Phase-3 theme parity: a representative NEW accent/role token now has an HC
    # variant, so an HC user's box rail is high-contrast (it had ZERO HC
    # coverage before — un-overridable low-contrast accent).
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    style = _page_style_block(theme="high_contrast")
    final = _last_root_token_values(style)
    # --cf-role-scenario (purple #6f42c1 default) reskins to the HC purple.
    assert final["--cf-role-scenario"] == "#4b0082"
    # --cf-fade-completion (orange-strong default) reskins to the HC amber.
    assert final["--cf-fade-completion"] == "#664400"
    # --cf-accent-teal reskins to the HC teal.
    assert final["--cf-accent-teal"] == "#004466"


def test_hc_theme_file_mirrors_new_tokens():
    # The source-of-truth high_contrast.css file carries the same new-token
    # parity block (so a server-render path that links the file, not the
    # embedded mirror, also reskins).
    hc = (Path(__file__).resolve().parents[2] / "templates" / "theme"
          / "color_schemes" / "high_contrast.css").read_text(encoding="utf-8")
    for tok in ("--cf-role-scenario", "--cf-fade-completion", "--cf-accent-teal",
                "--cf-stage-apply", "--cf-ink"):
        assert tok in hc, f"{tok} missing from high_contrast.css parity block"


def test_dyslexia_reaches_new_boxes_via_font_token(monkeypatch):
    # The Phase-1 box rules consume var(--cf-font-family) / var(--cf-line-height-
    # base), so the dyslexia theme's font/spacing override reaches them.
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    for selector in (".multimedia", ".diagram", ".hook", ".reflection-calibration"):
        body = _rule_body(COURSEFORGE_RICHER_CSS, selector)
        assert "var(--cf-font-family)" in body, f"{selector} not dyslexia-reachable"
    style = _page_style_block(theme="dyslexia_friendly")
    final = _last_root_token_values(style)
    assert "OpenDyslexic" in final["--cf-font-family"]


# ===========================================================================
# FLAG-OFF BYTE-STABILITY + contentHash invariance (Phase 1 + 3 additions)
# ===========================================================================


def test_full_page_off_byte_identical_with_new_renderers(monkeypatch):
    # A page whose body exercises the new renderers stays byte-identical when
    # the richer flag is off (the renderer emits are suppressed).
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    monkeypatch.delenv("ED4ALL_CALLOUT_TYPED", raising=False)
    qs = [{"question": "Q?", "options": [{"text": "a", "correct": True}]}]
    body = gc._render_self_check(qs, page_id="p")
    kwargs = dict(_wrap_kwargs())
    kwargs["body_html"] = body
    page = _wrap_page(**kwargs)
    assert f"<style>{COURSEFORGE_CSS}</style>" in page
    assert "stakes-label" not in page
    assert "media-a11y-badge" not in page
    assert ":root" not in page


def test_content_hash_invariant_across_flag_phase1and3(monkeypatch):
    block = _representative_block()
    monkeypatch.delenv("ED4ALL_RICHER_VISUAL_SYSTEM", raising=False)
    off = block.compute_content_hash()
    monkeypatch.setenv("ED4ALL_RICHER_VISUAL_SYSTEM", "1")
    on = block.compute_content_hash()
    assert off == on, "contentHash must be META-only — Phase-1/3 must not drift it"
