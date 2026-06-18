"""Phase 0 design-token layer — additive, zero-visual-change foundation.

Asserts the shared token file exists with the required theming + motion
contract, that all three shells link it FIRST (before their surface CSS so
existing hardcoded values still win and rendered output is byte-identical),
and that the two Phase-0 a11y fixes landed (operator dark theme + polite
toast container, plus the styles.css reduced-motion guard).

Pure static-text assertions: no fastapi / torch / LibV2 dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "gui" / "static"
TOKENS = STATIC / "shared" / "tokens.css"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_tokens_file_exists_and_defines_required_blocks():
    assert TOKENS.is_file(), f"missing {TOKENS}"
    css = _read(TOKENS)
    # (a) light :root + dark theme override + spin keyframes + reduced-motion.
    assert ":root {" in css or ":root{" in css
    assert re.search(r':root\[data-theme="dark"\]\s*\{', css), \
        "tokens.css must define :root[data-theme=\"dark\"]"
    assert re.search(r'@keyframes\s+spin\b', css), \
        "tokens.css must define @keyframes spin"
    assert "@media (prefers-reduced-motion: reduce)" in css, \
        "tokens.css must include a prefers-reduced-motion block"


SHELLS = {
    "operator": STATIC / "index.html",
    "learner": STATIC / "learn" / "index.html",
    "studio": STATIC / "studio" / "index.html",
}
SURFACE_CSS = {
    # GUI-redesign Phase 2: the operator SPA moved under /advanced/, so its own
    # stylesheet ref is now /advanced/styles.css (the learner/studio shells are
    # unchanged — they keep their own-prefix surface CSS).
    "operator": "/advanced/styles.css",
    "learner": "/learn/learn.css",
    "studio": "/studio/studio.css",
}


def test_all_shells_link_tokens_before_surface_css():
    for name, shell in SHELLS.items():
        html = _read(shell)
        tok_idx = html.find('href="/shared/tokens.css"')
        surf_idx = html.find(f'href="{SURFACE_CSS[name]}"')
        assert tok_idx != -1, f"{name} shell does not link /shared/tokens.css"
        assert surf_idx != -1, f"{name} shell does not link {SURFACE_CSS[name]}"
        assert tok_idx < surf_idx, \
            f"{name} shell must link tokens.css BEFORE {SURFACE_CSS[name]}"


def test_operator_shell_is_dark_and_toast_polite():
    html = _read(SHELLS["operator"])
    assert 'data-theme="dark"' in html, "operator index.html must set data-theme=\"dark\""
    assert 'id="toasts" aria-live="polite"' in html, \
        "operator toast container must be aria-live=\"polite\""
    assert 'aria-live="assertive"' not in html, \
        "operator shell must not retain the assertive toast bug"


def test_author_and_learner_shells_stay_light():
    for name in ("learner", "studio"):
        html = _read(SHELLS[name])
        assert "data-theme" not in html, f"{name} shell must stay light (no data-theme)"


def test_styles_css_has_reduced_motion_guard():
    css = _read(STATIC / "styles.css")
    assert "@media (prefers-reduced-motion: reduce)" in css, \
        "styles.css must carry a prefers-reduced-motion guard for slidein"
