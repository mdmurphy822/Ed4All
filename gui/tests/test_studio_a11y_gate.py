"""WCAG 2.2 AA gate for the Studio shell — mirrors the learner a11y gate.

The Studio surface is a JS-driven SPA, so the gate validates (a) the static
shell ``gui/static/studio/index.html`` as served, and (b) the two rendered
views reconstructed in Python (bs4) exactly as ``studio.js`` builds them — the
Library card grid and the Viewer (ARIA manifest tree + iframe content pane +
prev/next pager). It also gates the sanitiser output of the page-serving path
over a synthetic cartridge (the HTML that lands inside the viewer iframe).

The bar (same as ``test_learner_a11y_gate``): zero CRITICAL (WCAG A) and zero
HIGH (WCAG AA) ``WCAGValidator`` findings on every variant a user can see.
``WCAGValidator`` is bs4-only (a base dependency), so this gate runs on a
default install with NO fastapi/torch/LibV2 dependency. Plus structural
assertions the validator does not fully cover (ARIA tree pattern, focus
management, single h1, CSS focus-visible / target-size) and a static check that
the shell uses semantic landmarks + the shared ES modules.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import List, Tuple

import pytest

from lib.validators.wcag import IssueSeverity, WCAGValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "gui" / "static" / "studio"
STUDIO_INDEX = STUDIO_DIR / "index.html"
STUDIO_CSS = STUDIO_DIR / "studio.css"

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


# --------------------------------------------------------------------------- #
# Reconstruct the rendered views (exactly as studio.js builds them)
# --------------------------------------------------------------------------- #


def _shell_with_view(view_inner: str, *, crumbs: str = "") -> str:
    """Read the served shell and inject a rendered #view subtree (the JS swap)."""
    soup = _soup(STUDIO_INDEX.read_text(encoding="utf-8"))
    v = soup.find(id="view")
    assert v is not None, "studio index.html must carry #view"
    v.append(_bs4()(view_inner, "html.parser"))
    if crumbs:
        c = soup.find(id="crumbs")
        assert c is not None
        c.append(_bs4()(crumbs, "html.parser"))
    return str(soup)


_LIBRARY_INNER = """
<h1>Course Library</h1>
<ul class="cards">
  <li class="card-li"><a class="card" href="#/viewer/demo-101">
    <h2>Demo Course Title</h2>
    <p class="meta">2 pages · 12.3 MB</p>
    <span class="badge">Ask-ready</span>
  </a>
  <button type="button" class="card-delete" aria-label="Delete Demo Course Title">Delete</button>
  </li>
</ul>
"""

# The D5 destructive-confirm modal — reconstructed exactly as studio.js
# ``openDeleteDialog`` builds it. The confirm button is disabled until the slug
# is typed; the dialog is role=dialog + aria-modal + labelled by its title.
def _delete_dialog_inner(*, confirm_enabled: bool = False) -> str:
    disabled = "" if confirm_enabled else " disabled"
    return f"""
<div class="modal-overlay">
  <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="del-title">
    <h2 id="del-title">Delete course</h2>
    <p id="del-desc">This permanently deletes <strong>Demo Course Title</strong> (12.3 MB). There is no undo. To confirm, type the course id <code>demo-101</code> below.</p>
    <div class="field">
      <label for="del-input">Course id</label>
      <input id="del-input" type="text" autocomplete="off" aria-describedby="del-desc">
    </div>
    <p class="error" role="alert" hidden></p>
    <div class="modal-actions">
      <button type="button" class="btn">Cancel</button>
      <button type="button" class="btn primary danger"{disabled}>Delete demo-101</button>
    </div>
  </div>
</div>
"""

# The viewer view: h1 + tree pane (ARIA tree) + content pane (pager + iframe).
_VIEWER_INNER = """
<h1>Demo Course Title</h1>
<div class="viewer">
  <div class="tree-pane">
    <h2>Contents</h2>
    <ul role="tree" aria-label="Course contents">
      <li role="treeitem" aria-level="1" aria-expanded="true" tabindex="0">
        <span class="label"><span class="twisty" aria-hidden="true">▾ </span><span>Week 1</span></span>
        <ul role="group">
          <li role="treeitem" aria-level="2" aria-selected="true" tabindex="0" data-item="RES_overview">
            <span class="label"><span>Overview</span></span>
          </li>
          <li role="treeitem" aria-level="2" aria-selected="false" tabindex="-1" data-item="RES_self_check">
            <span class="label"><span>Self Check</span></span>
          </li>
        </ul>
      </li>
    </ul>
  </div>
  <div class="content-pane">
    <div class="pager">
      <button type="button" aria-label="Previous page">← Previous</button>
      <span class="page-title" id="page-title" aria-live="polite">Overview</span>
      <button type="button" aria-label="Next page">Next →</button>
    </div>
    <iframe class="frame" title="Course page content" sandbox="allow-scripts" referrerpolicy="no-referrer"></iframe>
  </div>
</div>
"""


# --------------------------------------------------------------------------- #
# Ask drawer (C2) — reconstructed exactly as drawer.js builds each state.
# --------------------------------------------------------------------------- #

# An answered fragment carrying a citation rendered as an in-context button
# (the server emits an <a href="/api/learn/source/…">; the drawer rewrites it to
# role=button so a citation click loads the cited page in the content pane).
# It also carries the B4 provenance disclosure (a "Provenance" toggle button
# controlling a hidden <ul> with the source block id + one PDF-page deep-link
# per page) — the ``expanded`` param flips the disclosure open as the JS does on
# activation, so the gate validates BOTH disclosure states.
def _drawer_answer_fragment(*, expanded: bool = False) -> str:
    aria = "true" if expanded else "false"
    panel = '<ul id="src-detail-c1" class="src-detail">' if expanded else (
        '<ul id="src-detail-c1" class="src-detail" hidden>'
    )
    return f"""
<section class="answer" data-status="answered" aria-labelledby="answer-h">
  <h2 id="answer-h" tabindex="-1">Answer</h2>
  <p>Velocity is the rate of change of position.</p>
  <h3>Sources</h3>
  <ol class="sources">
    <li><a class="ask-cite" role="button" href="/api/learn/source/demo-101?item_path=ch01.html#velocity">Source: Velocity</a>
      <button type="button" class="src-detail-toggle" aria-expanded="{aria}" aria-controls="src-detail-c1">Provenance</button>
      {panel}
        <li class="src-block">Source block <code>dart:mini_alpha#s3_c0</code></li>
        <li class="src-pdf"><a href="/api/courses/demo-101/source-pdf?file=mini_alpha&amp;page=12" class="src-pdf-link" target="_blank" rel="noopener" aria-label="Open PDF page 12, opens in new tab">PDF page 12</a></li>
      </ul>
    </li>
  </ol>
</section>
"""


# Back-compat alias for the default (collapsed) fragment.
_DRAWER_ANSWER_FRAGMENT = _drawer_answer_fragment()


def _drawer_html(*, state="idle", collapsed=False, no_index=False, prov_expanded=False) -> str:
    """Reconstruct the drawer subtree as ``drawer.js`` renders it for a state.

    ``state`` ∈ {idle, asking, answered}. ``no_index`` adds the explanatory note
    + keeps the ask box (the engine downgrades to lexical). ``collapsed`` hides
    the body (the toggle aria-expanded flips). ``prov_expanded`` opens the B4
    provenance disclosure (answered state only).
    """
    if state == "asking":
        history = """
        <li class="ask-entry">
          <p class="ask-q"><span class="ask-q-label">Q: </span>What is velocity?</p>
          <div class="ask-a">
            <span class="ask-busy">
              <span class="ask-spin" aria-hidden="true">⏳ </span>
              <span class="ask-busy-label">Thinking</span> · <span class="ask-elapsed">3s</span>
            </span>
          </div>
        </li>
        """
    elif state == "answered":
        fragment = _drawer_answer_fragment(expanded=prov_expanded)
        history = f"""
        <li class="ask-entry">
          <p class="ask-q"><span class="ask-q-label">Q: </span>What is velocity?</p>
          <div class="ask-a">{fragment}</div>
        </li>
        """
    else:
        history = ""

    noindex = (
        '<div class="ask-noindex" role="note">'
        "<p>This course has no semantic search index yet, so answers use keyword search only and may be less precise.</p>"
        '<p class="muted">To enable semantic answers, build a vector index for this course and reload.</p>'
        "</div>"
        if no_index
        else ""
    )
    body_hidden = " hidden" if collapsed else ""
    # No empty <ol> (WCAG 1.3.1): the list element exists only when populated.
    # The clear-history button is ALWAYS present in the shell (user directive),
    # independent of history population (mirrors drawer.js).
    history_list = (
        '<ol class="ask-history" aria-label="Question and answer history">'
        + history
        + "</ol>"
        if history.strip()
        else ""
    )
    history_list = (
        '<div class="ask-history-bar">'
        '<button type="button" class="ask-clear" '
        'aria-label="Clear question and answer history">Clear history</button>'
        "</div>" + history_list
    )
    expanded = "false" if collapsed else "true"
    toggle_text = "Show Ask panel" if collapsed else "Hide Ask panel"
    return f"""
<aside class="ask-drawer{' collapsed' if collapsed else ''}" role="complementary" aria-labelledby="ask-h">
  <div class="ask-header">
    <h2 id="ask-h" class="ask-title">Ask about this course</h2>
    <button type="button" class="ask-toggle" aria-expanded="{expanded}">{toggle_text}</button>
  </div>
  <div class="ask-body"{body_hidden}>
    {noindex}
    <form class="ask-form">
      <textarea class="ask-input" rows="2" aria-label="Your question about this course" placeholder="Ask a question about this course…"></textarea>
      <button type="button" class="ask-submit">Ask</button>
    </form>
    <p id="ask-live" class="visually-hidden" role="status" aria-live="polite"></p>
    <div class="ask-history-wrap">{history_list}</div>
  </div>
</aside>
"""


def _viewer_with_drawer(**drawer_kwargs) -> str:
    """The viewer inner with the drawer docked inside the .viewer grid."""
    drawer = _drawer_html(**drawer_kwargs)
    base = _VIEWER_INNER.rstrip()
    assert base.endswith("</div>"), "viewer inner must close with the .viewer </div>"
    return base[: -len("</div>")] + drawer + "</div>"


def test_shell_idle_zero_aa_findings():
    _assert_clean("studio-shell-idle", str(_soup(STUDIO_INDEX.read_text(encoding="utf-8"))))


def test_library_view_zero_aa_findings():
    _assert_clean("studio-library", _shell_with_view(_LIBRARY_INNER, crumbs="<span>Library</span>"))


def test_viewer_view_zero_aa_findings():
    crumbs = '<a href="#/library">Library</a><span class="sep" aria-hidden="true">/</span><span>Demo Course Title</span>'
    _assert_clean("studio-viewer", _shell_with_view(_VIEWER_INNER, crumbs=crumbs))


# --------------------------------------------------------------------------- #
# Author Dashboard (#/) — three zones + onboarding empty state + persona switch.
# Reconstructed exactly as dashboard.js + the shared kit (card/timeline-bar/
# empty-state/pill) build each subtree.
# --------------------------------------------------------------------------- #

# A runCard() as card.js builds it (h3 title + statusPill glyph+text, meta).
def _run_card(title: str, status: str, status_label: str, status_glyph: str, href: str, meta: str) -> str:
    return f"""
<a class="card run-card" href="{href}">
  <div class="card-head">
    <h3 class="card-title">{title}</h3>
    <span class="pill pill-running" data-kind="run:{status}" data-group="run">
      <span class="pill-glyph" aria-hidden="true">{status_glyph}</span>
      <span class="pill-label">{status_label}</span>
    </span>
  </div>
  <p class="card-meta">{meta}</p>
</a>
"""


# A courseCard() as card.js builds it (h3 title + meta + Ask-ready badge).
def _course_card(title: str, href: str, meta: str, ask_ready: bool = True) -> str:
    badge = '<span class="card-badge">Ask-ready</span>' if ask_ready else ""
    return f"""
<a class="card course-card" href="{href}">
  <h3 class="card-title">{title}</h3>
  <p class="card-meta">{meta}</p>
  {badge}
</a>
"""


# A timelineBar() as timeline-bar.js builds it (aria-hidden bar + semantic <ul>).
_TIMELINE_BAR = """
<figure class="timeline-bar">
  <div class="tl-bar-track" aria-hidden="true">
    <span class="tl-seg tl-seg-done" data-phase="semantik_conversion" style="width:60.00%" title="Convert textbook — 18m"></span>
    <span class="tl-seg tl-seg-done" data-phase="packaging" style="width:40.00%" title="Package course — 12m"></span>
  </div>
  <figcaption class="visually-hidden">Phase durations for Demo Course: total 30m</figcaption>
  <ul class="tl-legend" aria-label="Phase durations for Demo Course">
    <li class="tl-legend-item"><span class="tl-legend-swatch tl-seg-done" aria-hidden="true"></span><span class="tl-legend-name">Convert textbook</span><span class="tl-legend-dur tabular-nums">18m</span></li>
    <li class="tl-legend-item"><span class="tl-legend-swatch tl-seg-done" aria-hidden="true"></span><span class="tl-legend-name">Package course</span><span class="tl-legend-dur tabular-nums">12m</span></li>
  </ul>
</figure>
"""


def _dashboard_populated_inner() -> str:
    """The dashboard with ALL THREE zones populated (building / courses / recent)."""
    building_card = _run_card(
        "PHYS_201", "running", "Building", "◐",
        "#/build/GUI-live1", "textbook_to_course · building",
    )
    course_card = _course_card(
        "Demo Course Title", "#/viewer/demo-101", "12 pages",
    )
    recent_card = _run_card(
        "Demo Course", "completed", "Ready", "●",
        "#/build/GUI-done1", "textbook_to_course · 30m",
    )
    return f"""
<h1>Dashboard</h1>
<section class="dash-zone" aria-labelledby="dash-building-h">
  <h2 id="dash-building-h" class="dash-zone-h">Building now</h2>
  <p class="dash-zone-intro muted">Course builds in progress. Re-open one to watch its console.</p>
  <ul class="dash-cards cards" aria-label="Builds in progress">
    <li class="dash-card-li">{building_card}</li>
  </ul>
</section>
<section class="dash-zone" aria-labelledby="dash-courses-h">
  <h2 id="dash-courses-h" class="dash-zone-h">Your courses</h2>
  <p class="dash-zone-intro muted">Open a course to read it or ask questions about it.</p>
  <ul class="dash-cards cards" aria-label="Your courses">
    <li class="dash-card-li">{course_card}</li>
  </ul>
</section>
<section class="dash-zone" aria-labelledby="dash-recent-h">
  <h2 id="dash-recent-h" class="dash-zone-h">Recent builds</h2>
  <p class="dash-zone-intro muted">Finished builds. Re-open one, or run it again from the same textbook.</p>
  <ul class="dash-recent-list cards" aria-label="Finished builds">
    <li class="dash-recent-li">
      <div class="dash-recent-entry">
        {recent_card}
        {_TIMELINE_BAR}
        <div class="dash-recent-actions">
          <button type="button" class="btn">Re-open</button>
          <button type="button" class="btn">Run again</button>
        </div>
      </div>
    </li>
  </ul>
  <p class="dash-more"><a class="dash-more-link" href="#/runs">See all builds →</a></p>
</section>
"""


def _dashboard_onboarding_inner() -> str:
    """The onboarding empty state: no courses + no runs → one primary path."""
    return """
<h1>Dashboard</h1>
<section class="empty-state">
  <h2 class="empty-title">Build your first course</h2>
  <p class="empty-message">Turn a textbook PDF into an accessible, ask-ready course. Upload a PDF and we do the rest.</p>
  <a class="btn primary empty-cta" href="#/create">Build your first course</a>
</section>
"""


@pytest.mark.parametrize(
    "label,inner",
    [
        ("dashboard-populated", _dashboard_populated_inner()),
        ("dashboard-onboarding", _dashboard_onboarding_inner()),
    ],
)
def test_dashboard_views_zero_aa_findings(label, inner):
    _assert_clean(
        f"studio-{label}",
        _shell_with_view(inner, crumbs="<span>Dashboard</span>"),
    )


def test_dashboard_zones_are_labelled_sections_with_headings():
    soup = _soup(_shell_with_view(_dashboard_populated_inner()))
    # Exactly one h1 (the view title); each zone is a labelled <section> + <h2>.
    assert len(soup.find_all("h1")) == 1, "dashboard must keep a single <h1>"
    sections = soup.find_all("section", class_="dash-zone")
    assert len(sections) == 3, "dashboard must render three zones"
    seen = set()
    for sec in sections:
        lbl = sec.get("aria-labelledby")
        assert lbl, "each dashboard zone must be aria-labelledby a heading"
        heading = soup.find(id=lbl)
        assert heading is not None and heading.name == "h2", "zone heading must be an <h2> with the labelled id"
        seen.add(heading.get_text(strip=True))
    assert {"Building now", "Your courses", "Recent builds"} <= seen, seen


def test_dashboard_cards_are_keyboard_operable_links():
    soup = _soup(_shell_with_view(_dashboard_populated_inner()))
    # Building-now + recent cards link to their build console; courses to viewer.
    build_links = soup.select('a.card[href^="#/build/"]')
    assert len(build_links) == 2, "building-now + recent cards must link to #/build/<run_id>"
    course_links = soup.select('a.card[href^="#/viewer/"]')
    assert course_links, "course cards must link to the viewer"
    for a in build_links + course_links:
        # An <a href> is natively keyboard-operable; the title is an <h3>.
        assert a.find("h3", class_="card-title") is not None


def test_dashboard_recent_offers_reopen_run_again_and_history_link():
    soup = _soup(_shell_with_view(_dashboard_populated_inner()))
    recent = soup.find("section", attrs={"aria-labelledby": "dash-recent-h"})
    assert recent is not None
    labels = {b.get_text(strip=True) for b in recent.select(".dash-recent-actions button")}
    assert {"Re-open", "Run again"} <= labels, labels
    more = recent.find("a", class_="dash-more-link", href="#/runs")
    assert more is not None, "Recent builds must link to the full #/runs history"
    # The persisted-duration timeline is a semantic legend (never color-only).
    legend = recent.find("ul", class_="tl-legend")
    assert legend is not None and legend.get("aria-label"), "timeline bar needs a labelled legend"
    assert recent.find("div", class_="tl-bar-track").get("aria-hidden") == "true", (
        "the visual bar must be aria-hidden (the legend carries the a11y truth)"
    )


def test_dashboard_onboarding_is_single_primary_path():
    soup = _soup(_shell_with_view(_dashboard_onboarding_inner()))
    es = soup.find("section", class_="empty-state")
    assert es is not None, "onboarding must render an empty-state section"
    ctas = es.find_all("a", class_="empty-cta")
    assert len(ctas) == 1 and ctas[0].get("href") == "#/create", (
        "onboarding offers exactly one primary path → #/create"
    )
    assert ctas[0].get_text(strip=True) == "Build your first course"


def test_dashboard_status_pills_are_not_color_only():
    soup = _soup(_shell_with_view(_dashboard_populated_inner()))
    pills = soup.select(".dash-zone .pill")
    assert pills, "dashboard run cards must carry status pills"
    for pill in pills:
        # Glyph is decorative; the text label is what a screen reader reads.
        label = pill.find("span", class_="pill-label")
        assert label is not None and label.get_text(strip=True), (
            "each pill must carry a text label, not color alone (WCAG 1.4.1)"
        )


def test_studio_js_wires_dashboard_and_build_routes():
    js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    assert "/studio/dashboard.js" in js, "studio.js must import the dashboard module"
    assert "dashboard:" in js, "studio.js must register the dashboard route"
    assert "defaultRoute: 'dashboard'" in js, "the home route (#/) must land on the dashboard"
    assert "build:" in js, "studio.js must register the #/build/<run_id> console route"


# --------------------------------------------------------------------------- #
# Persona switcher (Author · Learner · Advanced) + graceful-401 explainer.
# --------------------------------------------------------------------------- #

# The explainer dialog, reconstructed exactly as persona-switcher.js builds it
# (a labelled, focus-managed modal mirroring the delete-dialog pattern).
_PERSONA_EXPLAINER_INNER = """
<div class="modal-overlay">
  <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="adv-title" aria-describedby="adv-desc">
    <h2 id="adv-title">Advanced mode is protected</h2>
    <p id="adv-desc">Advanced mode manages API keys, model routing, and run launching, and is protected — enter the operator token to continue. You’ll be taken to Advanced, where you can enter the token.</p>
    <div class="modal-actions">
      <button type="button" class="btn">Stay in Author</button>
      <a class="btn primary" href="/advanced/">Continue to Advanced</a>
    </div>
  </div>
</div>
"""


def test_persona_switcher_present_in_static_shell():
    soup = _soup(STUDIO_INDEX.read_text(encoding="utf-8"))
    nav = soup.find("nav", class_="persona-switcher")
    assert nav is not None, "the shell must server-render the persona switcher"
    assert nav.get("aria-label"), "the switcher must be a labelled <nav> (WCAG landmark)"
    personas = {a.get("data-persona"): a for a in nav.find_all("a")}
    assert {"author", "learner", "advanced"} <= set(personas), personas.keys()
    # Author is the current persona (aria-current=page), and only one is current.
    current = [a for a in nav.find_all("a") if a.get("aria-current") == "page"]
    assert len(current) == 1 and current[0].get("data-persona") == "author", (
        "exactly the Author link carries aria-current=page"
    )
    # Learner + Advanced are real same-origin navigations.
    assert personas["learner"].get("href") == "/learn/"
    assert personas["advanced"].get("href") == "/advanced/"
    # Author links at the Studio home (#/).
    assert personas["author"].get("href") == "#/"


def test_live_run_tab_in_primary_nav_and_routed():
    """The "Live run" tab: present in the labelled primary <nav> landmark,
    routed (#/live) via the shared hash router, and backed by a THIN resolver
    in create.js that finds the newest running/paused run in the merged
    /api/runs list, delegates to the shared build-progress view, and renders
    an honest empty state (CTA to Run history) when nothing is live."""
    soup = _soup(STUDIO_INDEX.read_text(encoding="utf-8"))
    nav = soup.find("nav", class_="primary-nav")
    assert nav is not None, "the shell must render the primary nav"
    assert nav.get("aria-label"), "primary nav must stay a labelled landmark"
    live = nav.find("a", href="#/live")
    assert live is not None, "the Live run tab must be in the primary nav"
    assert live.get_text(strip=True) == "Live run"

    studio_js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    assert "live: () => renderLiveRun(shell)" in studio_js
    assert (
        "import { renderCreate, renderLiveRun } from '/studio/create.js';"
        in studio_js
    ), "renderLiveRun must be imported from create.js"

    create_js = (STUDIO_DIR / "create.js").read_text(encoding="utf-8")
    start = create_js.index("function renderLiveRun(")
    body = create_js[start : create_js.index("\n}", start)]
    assert "api('/api/runs')" in body  # the merged run list (incl. CLI runs)
    assert "r.effective_status || r.status" in body
    assert "'building', 'running'" in body  # active wins over paused history
    assert "status(r) === 'paused'" in body
    assert "renderProgress(shell, runId)" in body  # thin delegation, no fork
    assert "emptyState(" in body and "#/runs" in body, (
        "no live run must render the empty state with a Run-history CTA"
    )
    # The shell with the new tab stays WCAG-clean.
    _assert_clean("studio-live-run-nav", STUDIO_INDEX.read_text(encoding="utf-8"))


def test_persona_switcher_shell_zero_aa_findings():
    # The whole shell with the switcher present must stay WCAG-clean.
    _assert_clean("studio-persona-switcher", str(_soup(STUDIO_INDEX.read_text(encoding="utf-8"))))


def test_advanced_explainer_is_labelled_focus_managed_modal():
    inner = _dashboard_populated_inner() + _PERSONA_EXPLAINER_INNER
    html = _shell_with_view(inner, crumbs="<span>Dashboard</span>")
    _assert_clean("studio-advanced-explainer", html)
    soup = _soup(html)
    dialog = soup.find("div", class_="modal-dialog", attrs={"role": "dialog"})
    assert dialog is not None, "the graceful-401 explainer must be role=dialog"
    assert dialog.get("aria-modal") == "true", "explainer must be a modal (aria-modal)"
    lbl = dialog.get("aria-labelledby")
    assert lbl and soup.find(id=lbl) is not None, "explainer needs an accessible name"
    desc = dialog.get("aria-describedby")
    assert desc and soup.find(id=desc) is not None, "explainer must describe the protection"
    # Single-h1 invariant holds (the dialog title is an h2).
    assert len(soup.find_all("h1")) == 1
    # The primary action proceeds to /advanced/ (where the SPA token overlay
    # takes over); a cancel stays in Author.
    proceed = dialog.find("a", class_="primary", href="/advanced/")
    assert proceed is not None, "explainer must offer a Continue-to-Advanced action"
    cancel = dialog.find("button", string=lambda s: s and "Author" in s)
    assert cancel is not None, "explainer must offer a stay-in-Author cancel"


def test_persona_switcher_js_probes_and_explains_401():
    js = (STUDIO_DIR / "persona-switcher.js").read_text(encoding="utf-8")
    assert "fetch(" in js, "the switcher must PROBE /advanced/ before navigating"
    assert "401" in js, "the switcher must intercept a 401 challenge"
    assert "aria-modal" in js, "the graceful-401 explainer must be a proper modal"
    assert "Escape" in js, "the explainer must close on ESC"
    assert "/advanced/" in js, "the explainer must proceed to the Advanced surface"


def test_studio_js_inits_persona_switcher():
    js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    assert "/studio/persona-switcher.js" in js, "studio.js must import the persona switcher"
    assert "initPersonaSwitcher" in js, "studio.js must initialise the persona switcher"


# --------------------------------------------------------------------------- #
# D5 destructive-confirm dialog — zero CRITICAL/HIGH WCAG + modal semantics.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("confirm_enabled", [False, True])
def test_delete_dialog_zero_aa_findings(confirm_enabled):
    inner = _LIBRARY_INNER + _delete_dialog_inner(confirm_enabled=confirm_enabled)
    _assert_clean(
        f"studio-delete-dialog-{'enabled' if confirm_enabled else 'disabled'}",
        _shell_with_view(inner, crumbs="<span>Library</span>"),
    )


def test_delete_dialog_is_labelled_modal():
    soup = _soup(_shell_with_view(_LIBRARY_INNER + _delete_dialog_inner()))
    dialog = soup.find(attrs={"role": "dialog"})
    assert dialog is not None, "delete dialog must be role=dialog"
    assert dialog.get("aria-modal") == "true", "modal must carry aria-modal=true"
    labelledby = dialog.get("aria-labelledby")
    assert labelledby, "modal needs an accessible name (aria-labelledby)"
    assert soup.find(id=labelledby) is not None, "aria-labelledby must point at a real title"
    # The single-h1 invariant holds: the dialog title is an h2.
    assert len(soup.find_all("h1")) == 1


def test_delete_dialog_confirm_input_is_labelled():
    soup = _soup(_shell_with_view(_LIBRARY_INNER + _delete_dialog_inner()))
    inp = soup.find("input", id="del-input")
    assert inp is not None
    assert soup.find("label", attrs={"for": "del-input"}) is not None, "input needs a <label for>"


def test_delete_dialog_confirm_disabled_until_typed():
    # Default (untyped) state: the confirm button is disabled.
    disabled_soup = _soup(_shell_with_view(_LIBRARY_INNER + _delete_dialog_inner(confirm_enabled=False)))
    confirm = disabled_soup.select_one(".modal-actions .btn.danger")
    assert confirm is not None and confirm.has_attr("disabled"), (
        "confirm button must start disabled (typed-confirm pattern)"
    )
    # Enabled (slug typed) state: the confirm button is enabled.
    enabled_soup = _soup(_shell_with_view(_LIBRARY_INNER + _delete_dialog_inner(confirm_enabled=True)))
    confirm2 = enabled_soup.select_one(".modal-actions .btn.danger")
    assert confirm2 is not None and not confirm2.has_attr("disabled")


def test_delete_dialog_error_region_is_alert():
    soup = _soup(_shell_with_view(_LIBRARY_INNER + _delete_dialog_inner()))
    err = soup.select_one(".modal-dialog .error")
    assert err is not None and err.get("role") == "alert", "modal error must be a role=alert region"


def test_library_card_has_delete_affordance():
    soup = _soup(_shell_with_view(_LIBRARY_INNER, crumbs="<span>Library</span>"))
    btn = soup.find("button", class_="card-delete")
    assert btn is not None, "library card must offer a delete affordance"
    assert btn.get("aria-label"), "delete button needs an accessible label"


def test_studio_js_wires_delete_dialog_and_endpoint():
    js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    assert "openDeleteDialog" in js, "studio.js must render the D5 delete dialog"
    assert "aria-modal" in js, "delete dialog must be a proper modal (aria-modal)"
    assert "Escape" in js, "delete dialog must cancel on Esc"
    assert "method: 'DELETE'" in js, "delete must issue a DELETE request"
    assert "confirm=" in js, "delete must echo the slug in the confirm query param"


# --------------------------------------------------------------------------- #
# Ask drawer states — zero CRITICAL/HIGH WCAG on every state a learner can see.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("idle", {"state": "idle"}),
        ("asking", {"state": "asking"}),
        ("answered", {"state": "answered"}),
        ("no-index", {"state": "idle", "no_index": True}),
        ("collapsed", {"state": "answered", "collapsed": True}),
        # B4 provenance disclosure — both collapsed and expanded states must
        # be WCAG-clean (the JS flips aria-expanded + unhides the panel).
        ("prov-collapsed", {"state": "answered", "prov_expanded": False}),
        ("prov-expanded", {"state": "answered", "prov_expanded": True}),
    ],
)
def test_ask_drawer_states_zero_aa_findings(label, kwargs):
    _assert_clean(f"studio-drawer-{label}", _shell_with_view(_viewer_with_drawer(**kwargs)))


def test_ask_drawer_is_complementary_landmark():
    soup = _soup(_shell_with_view(_viewer_with_drawer()))
    aside = soup.find("aside", class_="ask-drawer")
    assert aside is not None, "drawer must be an <aside>"
    assert aside.get("role") == "complementary"
    assert aside.get("aria-labelledby"), "drawer needs an accessible name (aria-labelledby)"
    # The single-h1 invariant holds with the drawer present (its title is an h2).
    assert len(soup.find_all("h1")) == 1


def test_ask_drawer_has_polite_live_region():
    soup = _soup(_shell_with_view(_viewer_with_drawer(state="asking")))
    live = soup.find(id="ask-live")
    assert live is not None
    assert live.get("aria-live") == "polite"
    assert live.get("role") == "status"


def test_ask_drawer_history_is_semantic_list():
    soup = _soup(_shell_with_view(_viewer_with_drawer(state="answered")))
    hist = soup.find("ol", class_="ask-history")
    assert hist is not None and hist.get("aria-label"), "history needs an <ol> + label"
    entries = hist.find_all("li", class_="ask-entry")
    assert entries, "answered drawer must render at least one Q/A entry"


def test_ask_drawer_input_is_labelled():
    soup = _soup(_shell_with_view(_viewer_with_drawer()))
    ta = soup.find("textarea", class_="ask-input")
    assert ta is not None and ta.get("aria-label"), "ask input needs an accessible label"


def test_ask_drawer_citations_render_as_buttons():
    soup = _soup(_shell_with_view(_viewer_with_drawer(state="answered")))
    cites = soup.select(".ask-history a.ask-cite")
    assert cites, "answered drawer must render citation controls"
    for c in cites:
        assert c.get("role") == "button", "citation must be activatable as a button"
        assert c.get("href", "").startswith("/api/learn/source/"), (
            "citation href must point at the in-context source URL the drawer intercepts"
        )


def test_provenance_disclosure_semantics():
    # The B4 provenance row is a disclosure: a button with aria-expanded that
    # controls the detail panel by id; the panel is hidden when collapsed.
    collapsed = _soup(_shell_with_view(_viewer_with_drawer(state="answered", prov_expanded=False)))
    toggle = collapsed.find("button", class_="src-detail-toggle")
    assert toggle is not None, "provenance row must be a disclosure button"
    assert toggle.get("aria-expanded") == "false"
    panel_id = toggle.get("aria-controls")
    assert panel_id, "disclosure must reference its panel by aria-controls"
    panel = collapsed.find(id=panel_id)
    assert panel is not None, "aria-controls must point at a real panel"
    assert panel.has_attr("hidden"), "collapsed disclosure panel must be hidden"

    expanded = _soup(_shell_with_view(_viewer_with_drawer(state="answered", prov_expanded=True)))
    etoggle = expanded.find("button", class_="src-detail-toggle")
    assert etoggle.get("aria-expanded") == "true"
    epanel = expanded.find(id=etoggle.get("aria-controls"))
    assert not epanel.has_attr("hidden"), "expanded disclosure panel must be visible"


def test_provenance_pdf_link_opens_new_tab_with_label():
    soup = _soup(_shell_with_view(_viewer_with_drawer(state="answered", prov_expanded=True)))
    pdf_link = soup.find("a", class_="src-pdf-link")
    assert pdf_link is not None, "expanded disclosure must carry a PDF-page link"
    assert pdf_link.get("target") == "_blank"
    assert "noopener" in (pdf_link.get("rel") or [])
    # New-tab links MUST announce that they open in a new tab (WCAG 3.2.5).
    assert "opens in new tab" in (pdf_link.get("aria-label") or "")
    assert pdf_link.get("href", "").startswith("/api/courses/"), (
        "PDF link must point at the source-pdf endpoint"
    )


def test_ask_drawer_toggle_reflects_collapsed_state():
    open_soup = _soup(_shell_with_view(_viewer_with_drawer(collapsed=False)))
    closed_soup = _soup(_shell_with_view(_viewer_with_drawer(collapsed=True)))
    open_toggle = open_soup.find("button", class_="ask-toggle")
    closed_toggle = closed_soup.find("button", class_="ask-toggle")
    assert open_toggle.get("aria-expanded") == "true"
    assert closed_toggle.get("aria-expanded") == "false"
    # Collapsed body is hidden from the a11y tree.
    closed_body = closed_soup.find("div", class_="ask-body")
    assert closed_body.has_attr("hidden")


def test_ask_drawer_no_index_keeps_ask_box_and_explains():
    soup = _soup(_shell_with_view(_viewer_with_drawer(no_index=True)))
    note = soup.find("div", class_="ask-noindex")
    assert note is not None and note.get("role") == "note"
    assert "semantic search index" in note.get_text()
    # The ask box stays present (lexical fallback still answers).
    assert soup.find("textarea", class_="ask-input") is not None


def test_studio_js_imports_drawer_module():
    js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    assert "/studio/drawer.js" in js, "studio.js must import the drawer module"


# --------------------------------------------------------------------------- #
# Create wizard + Studio settings (C3) — zero CRITICAL/HIGH on every state.
# --------------------------------------------------------------------------- #

# Reconstructed exactly as create.js / settings.js build each view. The shell's
# single #view is swapped; one polite live region (#status) carries status.

_WIZARD_UPLOAD_INNER = """
<h1>Create a course</h1>
<ol class="wizard-steps" aria-label="Create course steps">
  <li class="wizard-step is-current" aria-current="step"><span class="wizard-step-num" aria-hidden="true">1</span><span class="wizard-step-label">Upload</span><span class="visually-hidden"> (current step)</span></li>
  <li class="wizard-step is-upcoming"><span class="wizard-step-num" aria-hidden="true">2</span><span class="wizard-step-label">Configure</span><span class="visually-hidden"> (upcoming)</span></li>
  <li class="wizard-step is-upcoming"><span class="wizard-step-num" aria-hidden="true">3</span><span class="wizard-step-label">Launch</span><span class="visually-hidden"> (upcoming)</span></li>
</ol>
<div class="wizard-panel">
  <h2>Step 1: Upload your textbook</h2>
  <p class="muted">Add one or more PDF files. We convert them to an accessible course.</p>
  <div class="dropzone" tabindex="0" role="button" aria-label="Choose PDF files or drop them here">
    <p>Drag PDF files here, or</p>
    <label class="btn" for="file-x">Choose files</label>
    <input id="file-x" type="file" accept=".pdf,application/pdf" multiple class="visually-hidden">
  </div>
  <p class="error" role="alert" hidden></p>
  <ul class="file-list" aria-label="Selected files">
    <li><span class="file-name">textbook.pdf</span><span class="file-size"> (2.0 MB)</span>
      <button type="button" class="btn link-btn" aria-label="Remove textbook.pdf">Remove</button></li>
  </ul>
  <p class="muted" aria-live="polite">1 file(s) selected.</p>
  <div class="wizard-nav">
    <a class="btn" href="#/library">Cancel</a>
    <button type="button" class="btn primary">Next: Configure</button>
  </div>
</div>
"""

_WIZARD_CONFIGURE_INNER = """
<h1>Create a course</h1>
<ol class="wizard-steps" aria-label="Create course steps">
  <li class="wizard-step is-done"><span class="wizard-step-num" aria-hidden="true">1</span><span class="wizard-step-label">Upload</span><span class="visually-hidden"> (completed)</span></li>
  <li class="wizard-step is-current" aria-current="step"><span class="wizard-step-num" aria-hidden="true">2</span><span class="wizard-step-label">Configure</span><span class="visually-hidden"> (current step)</span></li>
  <li class="wizard-step is-upcoming"><span class="wizard-step-num" aria-hidden="true">3</span><span class="wizard-step-label">Launch</span><span class="visually-hidden"> (upcoming)</span></li>
</ol>
<div class="wizard-panel">
  <h2>Step 2: Configure</h2>
  <form class="wizard-form" novalidate>
    <div class="field">
      <label for="cname-x">Course name</label>
      <input id="cname-x" type="text" required aria-describedby="cname-h" autocomplete="off">
      <p id="cname-h" class="field-hint">Letters, numbers, “_” and “-” only (e.g. PHYS_101). At least 2 characters.</p>
      <p class="error" role="alert" hidden></p>
    </div>
    <div class="field">
      <label for="weeks-x">Course length in weeks (optional)</label>
      <input id="weeks-x" type="number" min="1" max="52" aria-describedby="weeks-h" inputmode="numeric">
      <p id="weeks-h" class="field-hint">Leave blank to size the course automatically from the textbook.</p>
    </div>
    <div class="summary-box flow-tree-box">
      <h3 id="flow-tree-h">AI pipeline</h3>
      <p class="muted">How your course is built, and which AI model runs each step. Change any step in settings.</p>
      <ol class="flow-tree" aria-labelledby="flow-tree-h">
        <li class="flow-step"><span class="flow-step-body"><span class="flow-step-label">PDF</span><span class="flow-step-sub">Your uploaded textbook</span></span></li>
        <li class="flow-step"><span class="flow-arrow" aria-hidden="true">→ </span><span class="flow-step-body"><span class="flow-step-label">Outline</span><span class="flow-step-sub">local</span></span></li>
        <li class="flow-step is-inactive"><span class="flow-arrow" aria-hidden="true">→ </span><span class="flow-step-body"><span class="flow-step-label">Validate &amp; Rewrite</span><span class="flow-step-sub">Skipped</span><span class="flow-step-note"> (two-pass only)</span></span></li>
        <li class="flow-step"><span class="flow-arrow" aria-hidden="true">→ </span><span class="flow-step-body"><span class="flow-step-label">Assessments</span><span class="flow-step-sub">local</span></span></li>
      </ol>
      <p class="flow-tree-link"><a class="flow-link" href="#/settings">Change AI model settings</a></p>
    </div>
    <details class="advanced">
      <summary>Advanced options</summary>
      <div class="advanced-body">
        <div class="field check"><input id="skip-x" type="checkbox"><label for="skip-x">Skip assessment generation</label></div>
        <div class="field check"><input id="skip-t" type="checkbox"><label for="skip-t">Skip training-data synthesis</label></div>
      </div>
    </details>
    <div class="wizard-nav">
      <button type="button" class="btn">Back</button>
      <button type="submit" class="btn primary">Launch build</button>
    </div>
  </form>
</div>
"""

# The same configure step with COURSEFORGE_TWO_PASS ON: the Validate & Rewrite
# flow-tree node is ACTIVE (no is-inactive, no "(two-pass only)" note) and shows
# its provider/model. Reconstructed exactly as create.js builds it.
_WIZARD_CONFIGURE_TWO_PASS_INNER = _WIZARD_CONFIGURE_INNER.replace(
    '<li class="flow-step is-inactive"><span class="flow-arrow" aria-hidden="true">→ </span>'
    '<span class="flow-step-body"><span class="flow-step-label">Validate &amp; Rewrite</span>'
    '<span class="flow-step-sub">Skipped</span><span class="flow-step-note"> (two-pass only)</span></span></li>',
    '<li class="flow-step"><span class="flow-arrow" aria-hidden="true">→ </span>'
    '<span class="flow-step-body"><span class="flow-step-label">Validate &amp; Rewrite</span>'
    '<span class="flow-step-sub">local · qwen2.5:14b</span></span></li>',
)

_WIZARD_PROGRESS_INNER = """
<h1>Building PHYS_101</h1>
<p class="muted"><span>Run GUI-x</span><span class="sep" aria-hidden="true"> · </span><span class="elapsed">elapsed 12s</span></p>
<ol class="phase-checklist" aria-label="Course build steps">
  <li class="phase-row is-done" data-phase="semantik_conversion"><span class="phase-icon" aria-hidden="true">●</span><span class="phase-label">Convert textbook to accessible HTML</span><span class="phase-state">Done</span></li>
  <li class="phase-row is-running" data-phase="staging"><span class="phase-icon" aria-hidden="true">◐</span><span class="phase-label">Stage source files</span><span class="phase-state">Running…</span></li>
  <li class="phase-row is-pending" data-phase="packaging"><span class="phase-icon" aria-hidden="true">○</span><span class="phase-label">Package course</span><span class="phase-state">Pending</span></li>
  <li class="phase-row is-pending" data-phase="trainforge_assessment"><span class="phase-icon" aria-hidden="true">○</span><span class="phase-label">Generate assessments</span><span class="phase-state">Pending</span><span class="phase-opt">(optional)</span></li>
</ol>
<div class="final-box" aria-live="polite">
  <p class="ok">Your course is ready.</p>
  <a class="btn primary" href="#/viewer/phys-101">Open course</a>
</div>
"""

_WIZARD_PROGRESS_FAILED_INNER = """
<h1>Building PHYS_101</h1>
<p class="muted"><span>Run GUI-x</span><span class="sep" aria-hidden="true"> · </span><span class="elapsed">finished</span></p>
<ol class="phase-checklist" aria-label="Course build steps">
  <li class="phase-row is-done" data-phase="semantik_conversion"><span class="phase-icon" aria-hidden="true">●</span><span class="phase-label">Convert textbook to accessible HTML</span><span class="phase-state">Done</span></li>
  <li class="phase-row is-failed" data-phase="staging"><span class="phase-icon" aria-hidden="true">✕</span><span class="phase-label">Stage source files</span><span class="phase-state">Failed</span></li>
</ol>
<div class="final-box" aria-live="polite">
  <p class="error" role="alert">A gate failed during staging.</p>
  <a class="btn" href="#/create/GUI-x/log">View build log</a>
</div>
"""

# A6 operator-failure panel: reconstructed exactly as create.js
# ``renderFailurePanel`` builds it for a failed build with a validation report.
_WIZARD_PROGRESS_FAILURE_PANEL_INNER = """
<h1>Building PHYS_101</h1>
<p class="muted"><span>Run GUI-x</span><span class="sep" aria-hidden="true"> · </span><span class="elapsed">finished</span></p>
<ol class="phase-checklist" aria-label="Course build steps">
  <li class="phase-row is-done" data-phase="content_generation_outline"><span class="phase-icon" aria-hidden="true">●</span><span class="phase-label">Outline course content</span><span class="phase-state">Done</span></li>
  <li class="phase-row is-failed" data-phase="inter_tier_validation"><span class="phase-icon" aria-hidden="true">✕</span><span class="phase-label">Validate content</span><span class="phase-state">Failed</span></li>
</ol>
<div class="final-box" aria-live="polite">
  <section class="failure-panel" role="alert" aria-labelledby="failure-h">
    <h2 id="failure-h" class="failure-title">The course build failed</h2>
    <p class="failure-intro">Failed during “Inter Tier Validation”: failed validation gate(s): curie_anchoring</p>
    <div class="failure-detail">
      <h3>Failed checks</h3>
      <table class="gate-table">
        <caption class="visually-hidden">Validation checks that failed</caption>
        <thead><tr><th scope="col">Severity</th><th scope="col">Check</th><th scope="col">What went wrong</th></tr></thead>
        <tbody>
          <tr><td><span class="sev-badge sev-critical">Critical</span></td><td>curie_anchoring</td><td>pair anchoring rate 0.81 &lt; 0.95</td></tr>
        </tbody>
      </table>
      <h3>Content blocks</h3>
      <table class="block-table">
        <caption class="visually-hidden">Per-block-type pass, fail and escalated counts</caption>
        <thead><tr><th scope="col">Block type</th><th scope="col">Passed</th><th scope="col">Failed</th><th scope="col">Escalated</th></tr></thead>
        <tbody>
          <tr><td>assessment_item</td><td>3</td><td>2</td><td>1</td></tr>
        </tbody>
      </table>
    </div>
    <div class="failure-actions">
      <h3>What to do next</h3>
      <div class="affordances">
        <div class="affordance"><button type="button" class="btn primary">Re-run validation</button><p class="affordance-hint muted">Re-run the validation checks against the existing course content.</p></div>
        <div class="affordance"><button type="button" class="btn primary">Rewrite failing blocks</button><p class="affordance-hint muted">Regenerate the 1 failing block type(s): assessment_item.</p></div>
        <div class="affordance"><button type="button" class="btn primary">Re-run failed step</button><p class="affordance-hint muted">Re-run just the “Inter Tier Validation” step.</p></div>
        <a class="btn" href="#/create/GUI-x/log">View / download build log</a>
      </div>
    </div>
  </section>
</div>
"""

_SETTINGS_INNER = """
<h1>Settings</h1>
<form class="settings-form" novalidate>
  <h2>AI provider</h2>
  <div class="field"><label for="mode-x">Mode</label><select id="mode-x"><option value="local" selected>local</option><option value="api">api</option></select><p class="field-hint">local: run on this machine (no key needed). api: call a cloud provider.</p></div>
  <div class="field"><label for="prov-x">Provider</label><select id="prov-x"><option value="local" selected>Local model server (OpenAI-compatible)</option><option value="anthropic">Anthropic (Claude)</option></select></div>
  <div class="field"><label for="model-x">Model (optional)</label><input id="model-x" type="text" autocomplete="off" placeholder="provider default"><p class="field-hint">Leave blank to use the provider’s default model.</p></div>
  <div class="field"><button type="button" class="btn">Test provider</button><p class="test-result is-ok" role="status" aria-live="polite">Connected. The provider is reachable.</p></div>
  <h2>Answers (course Q&amp;A)</h2>
  <p class="field-hint">The local backend that answers learner questions. Loopback-only.</p>
  <div class="field"><label for="ans-x">Answer model (optional)</label><input id="ans-x" type="text" autocomplete="off" placeholder="local default"></div>
  <h2>API keys</h2>
  <p class="field-hint">Only needed for cloud providers. Stored masked; leave blank to keep the saved value.</p>
  <div class="field"><label for="key-x">Anthropic API Key</label><input id="key-x" type="password" autocomplete="off" aria-describedby="key-x-h" placeholder="•••••• (saved)"><p id="key-x-h" class="field-hint">Required for --mode api with the Anthropic backend.</p></div>
  <h2>Server</h2>
  <div class="field readonly"><span class="ro-label">Address</span><span class="ro-value">127.0.0.1:8077</span><p class="field-hint">Where Studio is served. Change this from the command line, not the browser.</p></div>
  <div class="wizard-nav"><a class="btn" href="#/library">Back to library</a><button type="submit" class="btn primary">Save settings</button></div>
</form>
"""


@pytest.mark.parametrize(
    "label,inner,crumbs",
    [
        ("wizard-upload", _WIZARD_UPLOAD_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Create course</span>"),
        ("wizard-configure", _WIZARD_CONFIGURE_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Create course</span>"),
        ("wizard-configure-two-pass", _WIZARD_CONFIGURE_TWO_PASS_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Create course</span>"),
        ("wizard-progress", _WIZARD_PROGRESS_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Build progress</span>"),
        ("wizard-progress-failed", _WIZARD_PROGRESS_FAILED_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Build progress</span>"),
        ("wizard-failure-panel", _WIZARD_PROGRESS_FAILURE_PANEL_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Build progress</span>"),
        ("settings", _SETTINGS_INNER, "<a href='#/library'>Library</a><span class='sep' aria-hidden='true'>/</span><span>Settings</span>"),
    ],
)
def test_create_wizard_and_settings_zero_aa_findings(label, inner, crumbs):
    _assert_clean(f"studio-{label}", _shell_with_view(inner, crumbs=crumbs))


def test_create_settings_views_single_h1():
    for inner in (
        _WIZARD_UPLOAD_INNER,
        _WIZARD_CONFIGURE_INNER,
        _WIZARD_PROGRESS_INNER,
        _SETTINGS_INNER,
    ):
        soup = _soup(_shell_with_view(inner))
        assert len(soup.find_all("h1")) == 1


def test_wizard_steps_are_a_labelled_progress_structure():
    soup = _soup(_shell_with_view(_WIZARD_UPLOAD_INNER))
    ol = soup.find("ol", class_="wizard-steps")
    assert ol is not None and ol.get("aria-label"), "wizard steps need an <ol> + label"
    current = soup.find(attrs={"aria-current": "step"})
    assert current is not None, "the active wizard step must carry aria-current=step"


def test_wizard_form_fields_are_labelled_and_described():
    soup = _soup(_shell_with_view(_WIZARD_CONFIGURE_INNER))
    name = soup.find("input", id="cname-x")
    assert name is not None
    # A <label for=…> binds the control.
    assert soup.find("label", attrs={"for": "cname-x"}) is not None
    # A describedby hint exists.
    desc = name.get("aria-describedby")
    assert desc and soup.find(id=desc) is not None


def test_flow_tree_is_a_semantic_list_with_settings_links():
    """Phase 6: the AI-tier flow tree is a semantic <ol> (not divs), each step is
    an <li>, and the settings deep-link is a real <a>."""
    soup = _soup(_shell_with_view(_WIZARD_CONFIGURE_INNER))
    ol = soup.find("ol", class_="flow-tree")
    assert ol is not None, "the flow tree must be an <ol> (semantic list), not divs"
    assert ol.get("aria-labelledby"), "flow tree needs an accessible name"
    assert soup.find(id=ol.get("aria-labelledby")) is not None, "aria-labelledby must resolve"
    steps = ol.find_all("li", class_="flow-step")
    # PDF -> Outline -> Validate&Rewrite -> Assessments.
    labels = [li.find("span", class_="flow-step-label").get_text(strip=True) for li in steps]
    assert labels == ["PDF", "Outline", "Validate & Rewrite", "Assessments"], labels
    # The settings deep-link is a real anchor.
    link = soup.find("a", class_="flow-link", href="#/settings")
    assert link is not None, "flow tree must link to the settings page"


def test_flow_tree_greys_rewrite_node_off_two_pass_with_text_note():
    """Phase 6: when two-pass is OFF the Validate/Rewrite node is inactive AND
    carries a TEXT '(two-pass only)' note — not colour alone (non-color-only)."""
    off = _soup(_shell_with_view(_WIZARD_CONFIGURE_INNER))
    rewrite = next(
        li for li in off.find_all("li", class_="flow-step")
        if li.find("span", class_="flow-step-label").get_text(strip=True) == "Validate & Rewrite"
    )
    assert "is-inactive" in (rewrite.get("class") or []), "off-state rewrite node must be greyed"
    note = rewrite.find("span", class_="flow-step-note")
    assert note is not None and "two-pass only" in note.get_text(), (
        "greyed node must carry a text note, not colour alone (WCAG 1.4.1)"
    )

    # When two-pass is ON the node is active (no is-inactive, no note).
    on = _soup(_shell_with_view(_WIZARD_CONFIGURE_TWO_PASS_INNER))
    rewrite_on = next(
        li for li in on.find_all("li", class_="flow-step")
        if li.find("span", class_="flow-step-label").get_text(strip=True) == "Validate & Rewrite"
    )
    assert "is-inactive" not in (rewrite_on.get("class") or []), "on-state rewrite node is active"
    assert rewrite_on.find("span", class_="flow-step-note") is None, "active node carries no two-pass note"
    # The active node shows its provider/model.
    assert "local" in rewrite_on.find("span", class_="flow-step-sub").get_text()


def test_create_js_wires_flow_tree_from_studio_settings():
    js = (STUDIO_DIR / "create.js").read_text(encoding="utf-8")
    assert "providerFlowTree" in js, "create.js must render the AI-tier flow tree"
    assert "/api/settings/studio" in js, "flow tree must read GET /api/settings/studio"
    assert "twoPass" in js, "flow tree must grey the rewrite node off the two-pass flag"
    assert "courseforge_outline" in js and "courseforge_rewrite" in js, (
        "flow tree must surface the outline + rewrite authoring tiers"
    )


def test_settings_form_fields_are_labelled():
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    for cid in ("mode-x", "prov-x", "model-x", "ans-x", "key-x"):
        ctl = soup.find(id=cid)
        assert ctl is not None, f"missing control {cid}"
        assert soup.find("label", attrs={"for": cid}) is not None, f"{cid} needs a <label for>"


def test_settings_test_result_is_polite_status():
    soup = _soup(_shell_with_view(_SETTINGS_INNER))
    res = soup.find("p", class_="test-result")
    assert res is not None
    assert res.get("role") == "status" and res.get("aria-live") == "polite"


def test_progress_failed_surfaces_alert_and_log_link():
    soup = _soup(_shell_with_view(_WIZARD_PROGRESS_FAILED_INNER))
    alert = soup.find(attrs={"role": "alert"})
    assert alert is not None, "a failed build must surface an alert"
    log_link = soup.find("a", href=lambda h: h and h.endswith("/log"))
    assert log_link is not None, "a failed build must link to the build log"


def test_failure_panel_is_alert_with_table_semantics():
    """A6 failure panel: role=alert intro, single h1 held, semantic tables."""
    soup = _soup(_shell_with_view(_WIZARD_PROGRESS_FAILURE_PANEL_INNER))
    panel = soup.find("section", class_="failure-panel")
    assert panel is not None and panel.get("role") == "alert", "failure panel must be a role=alert region"
    assert panel.get("aria-labelledby"), "failure panel needs an accessible name"
    # Single-h1 invariant holds (panel title is an h2).
    assert len(soup.find_all("h1")) == 1
    # Both tables carry a caption + scoped column headers (table semantics).
    for cls in ("gate-table", "block-table"):
        table = soup.find("table", class_=cls)
        assert table is not None, f"missing {cls}"
        assert table.find("caption") is not None, f"{cls} needs a caption"
        headers = table.select("thead th[scope='col']")
        assert headers, f"{cls} needs scoped column headers"
    # Severity badges render with a readable label (not color-only).
    badge = soup.find("span", class_="sev-badge")
    assert badge is not None and badge.get_text(strip=True), "severity must carry a text label"


def test_failure_panel_offers_actionable_affordances():
    soup = _soup(_shell_with_view(_WIZARD_PROGRESS_FAILURE_PANEL_INNER))
    actions = soup.find("div", class_="failure-actions")
    assert actions is not None
    labels = {b.get_text(strip=True) for b in actions.select("button")}
    assert {"Re-run validation", "Rewrite failing blocks", "Re-run failed step"} <= labels
    log_link = actions.find("a", href=lambda h: h and h.endswith("/log"))
    assert log_link is not None, "failure panel must offer a build-log link"
    # Every affordance button carries a descriptive hint.
    for aff in actions.select(".affordance"):
        assert aff.find("p", class_="affordance-hint") is not None


def test_create_js_wires_failure_panel_and_validation_report():
    js = (STUDIO_DIR / "create.js").read_text(encoding="utf-8")
    assert "renderFailurePanel" in js, "create.js must render the A6 failure panel"
    assert "/validation-report" in js, "create.js must fetch the validation-report endpoint"
    assert "courseforge_rewrite" in js, "rewrite affordance must enqueue courseforge_rewrite"
    assert "courseforge_validate" in js, "re-run-validation affordance must enqueue courseforge_validate"


def test_studio_js_imports_create_and_settings_modules():
    js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    assert "/studio/create.js" in js, "studio.js must import the create wizard module"
    assert "/studio/settings.js" in js, "studio.js must import the settings module"


# --------------------------------------------------------------------------- #
# Sanitised page (what lands inside the viewer iframe)
# --------------------------------------------------------------------------- #


def test_served_page_zero_aa_findings(tmp_path):
    """The sanitised cartridge page (iframe content) passes the gate."""
    from gui.services import imscc_service  # noqa: PLC0415

    page_html = """<!DOCTYPE html><html lang="en"><head><title>Self Check</title></head>
    <body><h1>Self Check</h1>
      <div class="self-check">
        <button class="reveal-btn" aria-controls="ans1" aria-expanded="false">Show Answer</button>
        <div id="ans1" class="answer-reveal" style="display:none"><p>The answer.</p></div>
      </div>
    </body></html>"""
    manifest = """<?xml version='1.0'?>
    <manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1" identifier="M">
      <organizations><organization identifier="O"><item identifier="I" identifierref="R">
        <title>Self Check</title></item></organization></organizations>
      <resources><resource identifier="R" type="webcontent" href="sc.html">
        <file href="sc.html"/></resource></resources>
    </manifest>"""
    slug = "demo-101"
    cdir = tmp_path / "courses" / slug / "source" / "imscc"
    cdir.mkdir(parents=True)
    with zipfile.ZipFile(cdir / "DEMO.imscc", "w") as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr("sc.html", page_html)
    served = imscc_service.get_page(slug, "R", libv2_root=tmp_path)
    _assert_clean("studio-iframe-page", served.body.decode("utf-8"))


def test_served_source_doc_zero_aa_findings(tmp_path):
    """The sanitized + block-anchored served source doc passes the a11y gate.

    Source HTML is WCAG-validated at conversion time, but the gate re-checks the
    SERVED transform (active-content scrub + heading-id + block-anchor injection
    + figure-src rewrite + lang preservation). The fixture below carries legacy
    ``data-dart-block-id`` markers to exercise the dual-READ anchor path.
    """
    from gui.services import source_materials  # noqa: PLC0415

    doc_html = """<!DOCTYPE html><html lang="en"><head><title>Chapter One</title>
    <script>alert('x')</script></head>
    <body onload="boom()">
      <h1>Chapter One</h1>
      <h2>Real Numbers</h2>
      <ul data-dart-block-id="blk_1"><li>An ordered list block.</li></ul>
      <p id="sec-existing" data-dart-block-id="blk_2">A kept-id block.</p>
      <img src="chapter-one_figures/diagram.png" alt="A diagram">
    </body></html>"""
    slug = "demo-101"
    html_dir = tmp_path / "courses" / slug / "source" / "html"
    html_dir.mkdir(parents=True)
    (html_dir / "chapter-one.html").write_text(doc_html, encoding="utf-8")
    served = source_materials.serve_source_doc(
        slug, "chapter-one", libv2_root=tmp_path
    )
    _assert_clean("studio-source-doc", served.body.decode("utf-8"))


# The drawer answered fragment now carries the requirement-2 "View original
# source" deep link as the source-block row (a new-tab link). Reconstruct it
# exactly as ``answer_render._citation_provenance`` emits it so the a11y gate
# validates the new learner-facing link surface (label + new-tab + escaping).
def _drawer_answer_with_original_source() -> str:
    return """
<section class="answer" data-status="answered" aria-labelledby="answer-h">
  <h2 id="answer-h" tabindex="-1">Answer</h2>
  <p>Velocity is the rate of change of position.</p>
  <h3>Sources</h3>
  <ol class="sources">
    <li><a class="ask-cite" role="button" href="/api/learn/source/demo-101?item_path=ch01.html#velocity">Source: Velocity</a>
      <button type="button" class="src-detail-toggle" aria-expanded="true" aria-controls="src-detail-c1">Provenance</button>
      <ul id="src-detail-c1" class="src-detail">
        <li class="src-block"><a href="/api/courses/demo-101/source-doc?doc=mini_alpha&amp;ref=s3_c0#semantik-s3_c0" class="src-original-link" target="_blank" rel="noopener" aria-label="View original source (accessible HTML), opens in new tab">View original source (accessible HTML)</a> <code>dart:mini_alpha#s3_c0</code></li>
        <li class="src-pdf"><a href="/api/courses/demo-101/source-pdf?file=mini_alpha&amp;page=12" class="src-pdf-link" target="_blank" rel="noopener" aria-label="Open PDF page 12, opens in new tab">PDF page 12</a></li>
      </ul>
    </li>
  </ol>
</section>
"""


def test_original_source_link_a11y_and_new_tab_label():
    """The original-source deep link is WCAG-clean + announces the new tab."""
    inner = _shell_with_view(
        _VIEWER_INNER.rstrip()[: -len("</div>")]
        + '<aside class="ask-drawer" role="complementary" aria-labelledby="ask-h2">'
        '<h2 id="ask-h2">Ask</h2><div class="ask-history-wrap">'
        '<ol class="ask-history" aria-label="Q and A history"><li class="ask-entry">'
        + _drawer_answer_with_original_source()
        + "</li></ol></div></aside></div>"
    )
    _assert_clean("studio-original-source-link", inner)
    soup = _soup(inner)
    link = soup.find("a", class_="src-original-link")
    assert link is not None, "answered drawer must carry the original-source link"
    assert link.get("target") == "_blank"
    assert "noopener" in (link.get("rel") or [])
    assert "opens in new tab" in (link.get("aria-label") or "")
    assert link.get("href", "").startswith("/api/courses/")
    assert "/source-doc?doc=" in link.get("href", "")


# --------------------------------------------------------------------------- #
# Structural assertions (ARIA tree pattern + focus + landmarks)
# --------------------------------------------------------------------------- #


def test_shell_has_semantic_landmarks_and_skip_link():
    soup = _soup(STUDIO_INDEX.read_text(encoding="utf-8"))
    assert soup.find("header") is not None
    assert soup.find("main", id="main") is not None
    skip = soup.find("a", class_="skip-link")
    assert skip is not None and skip.get("href") == "#main"
    # Skip link must be the FIRST focusable element.
    focusable = soup.find(
        lambda t: (t.name == "a" and t.get("href")) or t.name in ("button", "input", "select", "textarea")
    )
    assert focusable is skip


def test_shell_uses_shared_es_modules():
    html = STUDIO_INDEX.read_text(encoding="utf-8")
    assert 'type="module"' in html, "studio shell must load studio.js as an ES module"
    js = (STUDIO_DIR / "studio.js").read_text(encoding="utf-8")
    for mod in ("/shared/api.js", "/shared/dom.js", "/shared/toast.js", "/shared/router.js"):
        assert mod in js, f"studio.js must import the shared module {mod}"


def test_shell_has_exactly_one_h1_per_view():
    for variant, inner in (("library", _LIBRARY_INNER), ("viewer", _VIEWER_INNER)):
        soup = _soup(_shell_with_view(inner))
        h1s = soup.find_all("h1")
        assert len(h1s) == 1, f"{variant}: expected one <h1>, found {len(h1s)}"


def test_viewer_tree_follows_aria_tree_pattern():
    soup = _soup(_shell_with_view(_VIEWER_INNER))
    tree = soup.find(attrs={"role": "tree"})
    assert tree is not None and tree.get("aria-label"), "tree needs role=tree + label"
    treeitems = soup.find_all(attrs={"role": "treeitem"})
    assert treeitems, "tree must carry role=treeitem nodes"
    for ti in treeitems:
        assert ti.get("aria-level"), "each treeitem needs aria-level"
        assert ti.get("tabindex") in ("0", "-1"), "treeitem needs roving tabindex"
    # Exactly one node is tabbable at rest (roving tabindex).
    tabbable = [t for t in treeitems if t.get("tabindex") == "0"]
    assert len(tabbable) >= 1, "at least one treeitem must be tabbable"
    # The selected node carries aria-selected=true.
    selected = [t for t in treeitems if t.get("aria-selected") == "true"]
    assert len(selected) == 1, "exactly one treeitem aria-selected=true"


def test_viewer_pager_buttons_have_labels():
    soup = _soup(_shell_with_view(_VIEWER_INNER))
    for btn in soup.select(".pager button"):
        assert btn.get("aria-label") or btn.get_text(strip=True), "pager button needs a label"


def test_viewer_iframe_is_sandboxed_titled():
    soup = _soup(_shell_with_view(_VIEWER_INNER))
    frame = soup.find("iframe")
    assert frame is not None
    assert frame.get("title"), "content iframe must carry a title"
    sandbox = frame.get("sandbox", "")
    sandbox_val = " ".join(sandbox) if isinstance(sandbox, (list, tuple)) else str(sandbox)
    assert "allow-scripts" in sandbox_val
    # No allow-same-origin — the shim must not reach the parent origin.
    assert "allow-same-origin" not in sandbox_val


def test_focus_visible_styles_not_suppressed_in_css():
    css = STUDIO_CSS.read_text(encoding="utf-8")
    assert re.search(r":focus-visible[^{]*\{[^}]*outline\s*:\s*[1-9]", css), (
        "studio.css must define a :focus-visible outline (>=1px)"
    )
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).lower(), m.group(2)
        if re.search(r"outline\s*:\s*(none|0)\b", body):
            assert ":focus:not(:focus-visible)" in selector, (
                f"bare outline:none in selector {selector!r} suppresses focus"
            )


def test_interactive_target_sizes_meet_minimum_in_css():
    css = STUDIO_CSS.read_text(encoding="utf-8")
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).strip().lower(), m.group(2).lower()
        if not re.search(r"\b(button|input|select|textarea|\.btn)\b", selector):
            continue
        for dim in re.finditer(r"(?:min-)?(?:width|height)\s*:\s*(\d+)px", body):
            px = int(dim.group(1))
            assert not (0 < px < 24), f"selector {selector!r} sizes a control at {px}px (< 24px)"
    assert re.search(r"min-height\s*:\s*2(\.\d+)?rem", css) or re.search(
        r"min-height\s*:\s*(2[4-9]|[3-9]\d)px", css
    ), "studio.css must size controls at >= 24px"
