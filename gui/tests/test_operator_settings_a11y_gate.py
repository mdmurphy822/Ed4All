"""WCAG 2.2 AA gate for the operator Settings tab (GUI redesign Phase 6).

Phase 6 §3 ("plain-language by routing") re-renders the operator Settings tab
(``gui/static/app.js::renderSettings`` / ``fieldFor``) so a human sees a friendly
LABEL + help with the bare env-var name kept only as a small secondary
``<code>`` hint, GROUPED under a friendly category heading. This is a
behaviour-preserving relabel — same fields, same PATCH save path, same secret
masking — so the gate validates the NEW markup the JS emits is WCAG-clean and
keeps the label/control pairing the whole relabel turns on.

Mirrors the studio / dev a11y gates: zero CRITICAL (WCAG A) and zero HIGH
(WCAG AA) ``WCAGValidator`` findings on the reconstructed view, plus structural
assertions the validator does not fully cover (every control label-paired via
``<label for>``, the help wired via ``aria-describedby``, the bare env-var name
retained as a ``<code>`` hint, friendly group headings). ``WCAGValidator`` is
bs4-only (a base dependency), so this gate runs on a default install with NO
fastapi/torch dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from lib.validators.wcag import IssueSeverity, WCAGValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "gui" / "static"
APP_INDEX = STATIC_DIR / "index.html"
APP_JS = STATIC_DIR / "app.js"

_BLOCKING = {IssueSeverity.CRITICAL, IssueSeverity.HIGH}


def _bs4():
    from bs4 import BeautifulSoup  # noqa: PLC0415

    return BeautifulSoup


def _soup(html: str):
    return _bs4()(html, "html.parser")


def _gate(html: str) -> Tuple[List, List]:
    report = WCAGValidator().validate(html)
    blocking = [i for i in report.issues if i.severity in _BLOCKING]
    diagnostics = [i for i in report.issues if i.severity not in _BLOCKING]
    return blocking, diagnostics


def _assert_clean(variant: str, html: str) -> None:
    blocking, diagnostics = _gate(html)
    for d in diagnostics:
        print(f"[a11y-diag] {variant}: {d.severity.value} WCAG {d.criterion} — {d.message}")
    assert not blocking, (
        f"{variant}: {len(blocking)} CRITICAL/HIGH WCAG finding(s) — "
        + "; ".join(f"{i.severity.value} {i.criterion} {i.message}" for i in blocking)
    )


def _shell_with_view(view_inner: str) -> str:
    """Read the served operator shell and inject a rendered #view subtree."""
    soup = _soup(APP_INDEX.read_text(encoding="utf-8"))
    v = soup.find(id="view")
    assert v is not None, "operator index.html must carry #view"
    v.append(_bs4()(view_inner, "html.parser"))
    return str(soup)


# --------------------------------------------------------------------------- #
# Reconstruct the rendered Settings view (exactly as app.js fieldFor builds it).
# Two category cards: credentials (a secret + masked badge) and courseforge (an
# enum select + a friendly-labelled field). Each field is a fieldRow-style row:
# a <label for> bound control, then a .help block carrying the friendly help
# text AND the bare env-var name in a <code> (the secondary technical hint).
# --------------------------------------------------------------------------- #

_SETTINGS_INNER = """
<h1>Settings / API Keys</h1>
<p class="subtitle">Plain-language settings grouped by AI tier. Each field shows its underlying environment variable as a hint. Secrets are write-only — they send only on save.</p>
<div class="card">
  <h2>API keys &amp; credentials</h2>
  <p class="subtitle group-cat-hint"><span>Settings group: </span><code class="kv">credentials</code></p>
  <div class="grid2">
    <div class="field field-row">
      <label class="field-label" for="cfg-1">Anthropic API Key</label>
      <div class="inline">
        <input id="cfg-1" type="password" aria-describedby="cfg-h-1" autocomplete="off" placeholder="•••••••• (leave blank to keep)">
        <span class="badge set">set</span>
      </div>
      <div class="help" id="cfg-h-1">
        <span class="help-text">Required for --mode api with the Anthropic backend.</span>
        <span class="help-env"><code class="kv">ANTHROPIC_API_KEY</code><span class="help-applies"> · applies to: anthropic</span></span>
      </div>
    </div>
  </div>
</div>
<div class="card">
  <h2>Course authoring (Courseforge)</h2>
  <p class="subtitle group-cat-hint"><span>Settings group: </span><code class="kv">courseforge</code></p>
  <div class="grid2">
    <div class="field field-row">
      <label class="field-label" for="cfg-2">Outline Provider</label>
      <select id="cfg-2" aria-describedby="cfg-h-2"><option value="">— unset —</option><option value="local">local</option></select>
      <div class="help" id="cfg-h-2">
        <span class="help-text">LLM provider for the Courseforge outline tier.</span>
        <span class="help-env"><code class="kv">COURSEFORGE_OUTLINE_PROVIDER</code><span class="help-applies"> · applies to: courseforge_outline</span></span>
      </div>
    </div>
    <div class="field field-row">
      <label class="field-label" for="cfg-3">Courseforge Two-Pass</label>
      <label class="toggle"><input id="cfg-3" type="checkbox" aria-describedby="cfg-h-3"><span>disabled</span></label>
      <div class="help" id="cfg-h-3">
        <span class="help-text">Enable the Courseforge outline -> validate -> rewrite two-pass pipeline.</span>
        <span class="help-env"><code class="kv">COURSEFORGE_TWO_PASS</code><span class="help-applies"> · applies to: courseforge</span></span>
      </div>
    </div>
  </div>
</div>
<div class="btn-row"><button type="button">Save changes</button><button type="button" class="ghost">Apply to process env</button></div>
"""


def test_settings_view_zero_aa_findings():
    _assert_clean("operator-settings", _shell_with_view(_SETTINGS_INNER))


def test_settings_single_h1():
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    assert len(soup.find_all("h1")) == 1


def test_settings_fields_are_label_paired():
    """The whole point of the fieldRow relabel: every control stays bound to a
    real <label for> + its help is wired via aria-describedby."""
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    for cid in ("cfg-1", "cfg-2", "cfg-3"):
        ctl = soup.find(id=cid)
        assert ctl is not None, f"missing control {cid}"
        assert soup.find("label", attrs={"for": cid}) is not None, f"{cid} needs a <label for>"
        desc = ctl.get("aria-describedby")
        assert desc, f"{cid} must wire its help via aria-describedby"
        assert soup.find(id=desc) is not None, f"{cid} aria-describedby must resolve"


def test_settings_labels_are_friendly_not_bare_env():
    """The label the eye lands on is the friendly one, not the raw env-var."""
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    labels = {lbl.get("for"): lbl.get_text(strip=True) for lbl in soup.find_all("label", class_="field-label")}
    assert labels.get("cfg-2") == "Outline Provider", "field shows a friendly label"
    assert "COURSEFORGE_OUTLINE_PROVIDER" not in labels.get("cfg-2", ""), (
        "the friendly label must NOT be the bare env-var name"
    )


def test_settings_retains_bare_env_name_as_code_hint():
    """The operator surface keeps the raw env-var visible as a <code> hint."""
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    help_block = soup.find(id="cfg-h-2")
    assert help_block is not None
    code = help_block.find("code", class_="kv")
    assert code is not None and code.get_text(strip=True) == "COURSEFORGE_OUTLINE_PROVIDER", (
        "the bare env-var name must remain available as a <code> hint"
    )


def test_settings_groups_use_friendly_headings():
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    headings = {h.get_text(strip=True) for h in soup.find_all("h2")}
    assert "API keys & credentials" in headings
    assert "Course authoring (Courseforge)" in headings
    # The raw category slug is kept only as a secondary hint, not the heading.
    assert "credentials" not in headings and "courseforge" not in headings
    hint = soup.find("p", class_="group-cat-hint")
    assert hint is not None and hint.find("code", class_="kv") is not None, (
        "the raw category slug stays available as a secondary <code> hint"
    )


def test_app_js_settings_render_uses_friendly_labels_and_env_hint():
    js = APP_JS.read_text(encoding="utf-8")
    assert "GROUP_TITLES" in js, "settings render must map categories to friendly group headings"
    assert "help-env" in js, "settings render must keep the bare env-var as a secondary hint"
    assert "field-label" in js, "settings fields must carry a friendly field label"
    # Behaviour-preserving: the PATCH save path + secret masking are unchanged.
    assert "'/api/settings', 'PATCH'" in js, "settings still saves via PATCH /api/settings"
