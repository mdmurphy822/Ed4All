"""WCAG 2.2 AA gate for the developer console (Marketable-v1 C4).

Mirrors the studio / learner a11y gates. Validates (a) the static shell
``gui/static/dev/index.html`` as served, and (b) the natively-built panes
reconstructed in Python (bs4) exactly as ``dev.js`` builds them — the Run
Inspector (run list + log + gate table), the Env Catalog editor, the Events
feed, and the API Explorer.

The bar (same as the studio gate): zero CRITICAL (WCAG A) and zero HIGH
(WCAG AA) ``WCAGValidator`` findings on every variant a user can see, plus
structural assertions the validator does not fully cover (nav landmark, single
h1, live regions, drawer disclosure, CSS focus-visible / target-size). bs4-only
so the gate runs on a default install with NO fastapi/torch dependency.

The six ported tabs (upload/settings/routing/courses/retrieval) are EMBEDDED —
they reuse the legacy ``/app.js`` render functions verbatim, whose a11y is
already covered by the operator SPA's own surface; this gate covers the new
shell + the natively-built panes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest

from DART.pdf_converter.wcag_validator import IssueSeverity, WCAGValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "gui" / "static" / "dev"
DEV_INDEX = DEV_DIR / "index.html"
DEV_CSS = DEV_DIR / "dev.css"

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


def _shell_with_pane(pane_inner: str) -> str:
    """Read the served shell and inject a rendered #dev-main subtree (the swap)."""
    soup = _soup(DEV_INDEX.read_text(encoding="utf-8"))
    m = soup.find(id="dev-main")
    assert m is not None, "dev index.html must carry #dev-main"
    if pane_inner.strip():
        m.append(_bs4()(pane_inner, "html.parser"))
    return str(soup)


# --------------------------------------------------------------------------- #
# Reconstructed pane views (exactly as dev.js builds them)
# --------------------------------------------------------------------------- #

_RUN_INSPECTOR_INNER = """
<h1>Run Inspector</h1>
<p class="subtitle">Inspect runs.</p>
<div class="dev-inspector">
  <div class="dev-run-list card">
    <div class="flex wrap"><h2 style="flex:1;margin:0;border:none">Runs</h2><button type="button" class="ghost sm">Refresh</button></div>
    <div class="recent-runs">
      <table>
        <thead><tr><th>Run</th><th>Workflow</th><th>Status</th></tr></thead>
        <tbody>
          <tr class="clickable"><td><code class="kv">GUI-1</code></td><td>textbook_to_course</td>
            <td><span class="run-status"><span class="dot completed"></span><span>completed</span></span></td></tr>
        </tbody>
      </table>
    </div>
  </div>
  <div class="dev-run-detail card">
    <h2>Detail</h2>
    <div class="dev-detail-head"><code class="kv">GUI-1</code><span class="dev-chip-holder"><span class="run-status"><span class="dot completed"></span><span>completed</span></span></span></div>
    <div class="flex wrap" style="gap:8px;margin:8px 0"><button type="button" class="ghost sm">Download log</button></div>
    <pre class="console" tabindex="0" aria-label="Run log" role="log" aria-live="polite"></pre>
    <div class="dev-gate-wrap">
      <h3>Gate results</h3>
      <table>
        <thead><tr><th>Gate</th><th>Result</th><th>Severity</th><th>Detail</th></tr></thead>
        <tbody><tr><td>curie_anchoring</td><td class="gate-fail">fail</td><td>critical</td><td>pair anchoring 0.81</td></tr></tbody>
      </table>
    </div>
    <div class="dev-report-wrap">
      <div class="dev-failbox"><h3>Failure</h3><p><strong>failed_phase: </strong><code>inter_tier_validation</code></p></div>
    </div>
  </div>
</div>
"""

_ENV_CATALOG_INNER = """
<h1>Env Catalog</h1>
<p class="subtitle">The full environment catalog.</p>
<div class="card"><div class="field"><label for="filt-x">Filter</label><input id="filt-x" type="search" aria-label="Filter env catalog"></div></div>
<div class="card">
  <table>
    <thead><tr><th>Key</th><th>Category</th><th>Type</th><th>Value</th><th></th></tr></thead>
    <tbody>
      <tr><td><code>ANTHROPIC_API_KEY</code></td><td>credentials</td><td>secret</td>
        <td><input type="password" aria-label="Value for ANTHROPIC_API_KEY" autocomplete="off" placeholder="•••••• (set — leave blank to keep)"></td>
        <td><button type="button" class="primary sm">Save</button></td></tr>
    </tbody>
  </table>
</div>
"""

_EVENTS_INNER = """
<h1>Events</h1>
<p class="subtitle">The Claude / GUI activity bridge.</p>
<div class="flex wrap" style="gap:8px;margin:8px 0"><button type="button" class="ghost sm">Pause</button></div>
<p class="visually-hidden" role="status" aria-live="polite">1 event received.</p>
<div class="card dev-events" role="log" aria-label="Activity events" aria-live="polite">
  <div class="dev-event src-claude"><span class="dev-event-meta">#1 claude note</span><div class="dev-event-text">hello</div></div>
</div>
"""

_API_EXPLORER_INNER = """
<h1>API Explorer</h1>
<p class="subtitle">Send raw requests. The docs open in a
  <a href="/docs" target="_blank" rel="noopener" aria-label="Open OpenAPI docs, opens in new tab">new tab (Swagger UI)</a>.</p>
<form class="card dev-api-form">
  <div class="flex wrap" style="gap:8px">
    <div class="field"><label for="m-x">Method</label><select id="m-x" aria-label="HTTP method"><option>GET</option></select></div>
    <div class="field"><label for="p-x">Path</label><input id="p-x" type="text" value="/api/health" aria-label="Request path"></div>
  </div>
  <div class="field"><label for="h-x">Headers</label><textarea id="h-x" rows="3" aria-label="Request headers"></textarea></div>
  <div class="field"><label for="b-x">Body</label><textarea id="b-x" rows="5" aria-label="Request body"></textarea></div>
  <button type="button" class="primary">Send</button>
</form>
<div class="card dev-api-resp"><p class="muted">No request sent yet.</p></div>
<div class="card dev-api-hist"><h2>Request history</h2><p class="muted">No requests yet.</p></div>
"""


@pytest.mark.parametrize(
    "label,inner",
    [
        ("shell-idle", ""),
        ("run-inspector", _RUN_INSPECTOR_INNER),
        ("env-catalog", _ENV_CATALOG_INNER),
        ("events", _EVENTS_INNER),
        ("api-explorer", _API_EXPLORER_INNER),
    ],
)
def test_dev_panes_zero_aa_findings(label, inner):
    _assert_clean(f"dev-{label}", _shell_with_pane(inner))


# --------------------------------------------------------------------------- #
# Structural assertions
# --------------------------------------------------------------------------- #


def test_shell_has_nav_landmark_and_skip_link():
    soup = _soup(DEV_INDEX.read_text(encoding="utf-8"))
    # The pane rail is a navigation landmark with an accessible name.
    rail = soup.find("nav", id="dev-rail")
    assert rail is not None and rail.get("aria-label"), "rail must be a labelled <nav>"
    # The main region is a landmark + focus target for the skip link.
    main = soup.find("main", id="dev-main")
    assert main is not None and main.get("tabindex") == "-1"
    skip = soup.find("a", class_="skip-link")
    assert skip is not None and skip.get("href") == "#dev-main"
    # Skip link is the FIRST focusable element.
    focusable = soup.find(
        lambda t: (t.name == "a" and t.get("href")) or t.name in ("button", "input", "select", "textarea")
    )
    assert focusable is skip


def test_shell_nav_links_are_a_list():
    soup = _soup(DEV_INDEX.read_text(encoding="utf-8"))
    ul = soup.find("ul", id="dev-navlist")
    assert ul is not None, "pane links must be a semantic <ul>"
    links = ul.find_all("a")
    assert len(links) == 9, "nine panes in the rail"
    for a in links:
        assert a.get("data-pane"), "each pane link carries a data-pane id"


def test_shell_loads_module_and_shared_toolkit():
    html = DEV_INDEX.read_text(encoding="utf-8")
    assert 'type="module"' in html
    js = (DEV_DIR / "dev.js").read_text(encoding="utf-8")
    for mod in ("/shared/api.js", "/shared/dom.js", "/shared/toast.js"):
        assert mod in js, f"dev.js must import the shared module {mod}"


def test_error_drawer_is_a_disclosure():
    soup = _soup(DEV_INDEX.read_text(encoding="utf-8"))
    toggle = soup.find("button", id="dev-drawer-toggle")
    assert toggle is not None and toggle.get("aria-expanded") == "false"
    panel_id = toggle.get("aria-controls")
    assert panel_id, "drawer toggle must reference its panel by aria-controls"
    panel = soup.find(id=panel_id)
    assert panel is not None and panel.has_attr("hidden"), "collapsed drawer panel must be hidden"
    # The error count has a polite live region so new errors are announced.
    count = soup.find(id="dev-drawer-count")
    assert count is not None and count.get("aria-live") == "polite"


def test_each_pane_view_has_single_h1():
    for inner in (_RUN_INSPECTOR_INNER, _ENV_CATALOG_INNER, _EVENTS_INNER, _API_EXPLORER_INNER):
        soup = _soup(_shell_with_pane(inner))
        assert len(soup.find_all("h1")) == 1


def test_run_log_is_a_live_region():
    soup = _soup(_shell_with_pane(_RUN_INSPECTOR_INNER))
    console = soup.find("pre", class_="console")
    assert console is not None
    assert console.get("aria-live") == "polite"
    assert console.get("aria-label"), "run log needs an accessible name"


def test_events_feed_is_a_live_region():
    soup = _soup(_shell_with_pane(_EVENTS_INNER))
    feed = soup.find("div", class_="dev-events")
    assert feed is not None
    assert feed.get("role") == "log" and feed.get("aria-live") == "polite"


def test_api_explorer_docs_link_announces_new_tab():
    soup = _soup(_shell_with_pane(_API_EXPLORER_INNER))
    link = soup.find("a", href="/docs")
    assert link is not None and link.get("target") == "_blank"
    assert "noopener" in (link.get("rel") or [])
    assert "opens in new tab" in (link.get("aria-label") or "")


def test_focus_visible_styles_not_suppressed_in_css():
    css = DEV_CSS.read_text(encoding="utf-8")
    assert re.search(r":focus-visible[^{]*\{[^}]*outline\s*:\s*[1-9]", css), (
        "dev.css must define a :focus-visible outline (>=1px)"
    )
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).lower(), m.group(2)
        if re.search(r"outline\s*:\s*(none|0)\b", body):
            assert ":focus:not(:focus-visible)" in selector, (
                f"bare outline:none in selector {selector!r} suppresses focus"
            )


def test_interactive_target_sizes_meet_minimum_in_css():
    css = DEV_CSS.read_text(encoding="utf-8")
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).strip().lower(), m.group(2).lower()
        if not re.search(r"\b(button|input|select|textarea|\.btn)\b", selector):
            continue
        for dim in re.finditer(r"(?:min-)?(?:width|height)\s*:\s*(\d+)px", body):
            px = int(dim.group(1))
            assert not (0 < px < 24), f"selector {selector!r} sizes a control at {px}px (< 24px)"
    assert re.search(r"min-height\s*:\s*2[4-9]px", css) or re.search(
        r"min-height\s*:\s*([3-9]\d)px", css
    ) or re.search(r"min-height\s*:\s*2(\.\d+)?rem", css), (
        "dev.css must size controls at >= 24px"
    )
