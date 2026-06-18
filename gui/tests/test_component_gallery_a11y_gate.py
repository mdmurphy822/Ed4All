"""WCAG 2.2 AA gate for the shared component gallery (redesign Phase 1).

Mirrors ``test_studio_a11y_gate`` / ``test_dev_console_a11y_gate``. The gallery
is a JS-rendered page (``gui/static/dev/gallery.js`` builds it from the shared
component kit), so — exactly like the other gates — this validates (a) the
served static shell ``gui/static/dev/gallery.html`` and (b) a representative
static HTML snapshot of EACH component's rendered output, reconstructed in
Python (bs4) precisely as the component modules emit it.

The bar (same as the studio/dev gates): zero CRITICAL (WCAG A) and zero HIGH
(WCAG AA) ``WCAGValidator`` findings on every variant a user can see, plus
structural assertions the validator does not fully cover (single live region,
rings aria-hidden, non-color-only pills/chips, focus-visible / target-size in
the kit CSS). bs4-only, so the gate runs on a default install with NO
fastapi/torch dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest

from DART.pdf_converter.wcag_validator import IssueSeverity, WCAGValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "gui" / "static" / "dev"
GALLERY_INDEX = DEV_DIR / "gallery.html"
GALLERY_CSS = DEV_DIR / "gallery.css"
COMPONENTS_DIR = REPO_ROOT / "gui" / "static" / "shared" / "components"
COMPONENTS_CSS = COMPONENTS_DIR / "components.css"

_BLOCKING = {IssueSeverity.CRITICAL, IssueSeverity.HIGH}

_COMPONENT_MODULES = (
    "ring.js",
    "phase-row.js",
    "phase-timeline.js",
    "elapsed.js",
    "pill.js",
    "gate-chip.js",
    "card.js",
    "empty-state.js",
    "field-row.js",
    "timeline-bar.js",
)


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


def _shell_with_gallery(inner: str) -> str:
    """Read the served gallery shell and inject a rendered #gallery-main subtree."""
    soup = _soup(GALLERY_INDEX.read_text(encoding="utf-8"))
    m = soup.find(id="gallery-main")
    assert m is not None, "gallery.html must carry #gallery-main"
    if inner.strip():
        m.append(_bs4()(inner, "html.parser"))
    return str(soup)


# --------------------------------------------------------------------------- #
# Reconstructed component output (exactly as the component modules emit it)
# --------------------------------------------------------------------------- #

# A ring: aria-hidden SVG (decoration) + a visually-hidden label (the a11y truth).
def _ring(state: str, size: int = 28) -> str:
    return (
        f'<span class="ring ring-{state}" data-state="{state}">'
        f'<svg class="ring-svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" aria-hidden="true" focusable="false">'
        '<circle class="ring-track" fill="none" stroke="var(--border)"></circle>'
        '<circle class="ring-arc" fill="none" stroke="var(--phase-' + state + ')"></circle>'
        "</svg>"
        f'<span class="visually-hidden">{state}</span>'
        "</span>"
    )


_RING_INNER = (
    '<section class="gallery-section"><h2>Progress ring</h2>'
    '<p>Five states at three sizes.</p>'
    '<div class="gallery-row">'
    + "".join(
        f'<div class="gallery-cell">{_ring(s, sz)}<span class="cell-caption">{s} · {sz}px</span></div>'
        for sz in (20, 28, 48)
        for s in ("pending", "running", "done", "skipped", "failed")
    )
    + "</div></section>"
)


def _pill(cls: str, glyph: str, label: str) -> str:
    return (
        f'<span class="pill pill-{cls}">'
        f'<span class="pill-glyph" aria-hidden="true">{glyph}</span>'
        f'<span class="pill-label">{label}</span></span>'
    )


_PILL_INNER = (
    '<section class="gallery-section"><h2>Status pill</h2>'
    '<p>Run, answer, and severity statuses — glyph + text, never color-only.</p>'
    '<div class="gallery-row">'
    + _pill("pending", "◷", "Queued")
    + _pill("running", "◐", "Building")
    + _pill("done", "●", "Ready")
    + _pill("failed", "✕", "Failed")
    + _pill("skipped", "–", "Cancelled")
    + _pill("done", "●", "Answered")
    + _pill("warn", "△", "Couldn't verify")
    + _pill("skipped", "–", "No clear answer")
    + _pill("pass", "✓", "Passing")
    + _pill("warn", "△", "Warning")
    + _pill("fail", "✗", "Failed")
    + "</div></section>"
)


def _gate_chip(cls: str, glyph: str, word: str, name: str = "curie_anchoring") -> str:
    return (
        f'<span class="gate-chip gate-chip-{cls}" data-result="{cls}" title="{word}: {name}">'
        f'<span class="gate-chip-glyph" aria-hidden="true">{glyph}</span>'
        f'<span class="visually-hidden">{word}: {name}</span></span>'
    )


_GATE_INNER = (
    '<section class="gallery-section"><h2>Gate chip</h2>'
    '<p>Icon + visually-hidden text label (never color-only).</p>'
    '<div class="gallery-row">'
    + _gate_chip("pass", "✓", "Passed")
    + _gate_chip("warn", "△", "Warning")
    + _gate_chip("fail", "✗", "Failed")
    + "</div></section>"
)


def _gate_summary(result: str, glyph: str, text: str) -> str:
    """The per-phase "N of M checks passing" summary chip, reconstructed exactly
    as phase-row.js::setGateSummary emits it: glyph (aria-hidden) + VISIBLE text,
    never color-only. The tone (pass/warn/fail) is the WORST gate seen so a
    warning reads CALM (amber △ "warnings"), not alarming."""
    return (
        f'<span class="phase-gate-summary is-{result}" data-result="{result}">'
        f'<span class="phase-gate-summary-glyph" aria-hidden="true">{glyph}</span>'
        f'<span class="phase-gate-summary-text">{text}</span></span>'
    )

_ELAPSED_INNER = (
    '<section class="gallery-section"><h2>Elapsed timer</h2>'
    '<p>The single canonical timer (tabular-nums, aria-hidden).</p>'
    '<div class="gallery-row">'
    '<div class="gallery-cell"><span class="elapsed tabular-nums" aria-hidden="true">elapsed 1m 13s</span><span class="cell-caption">live</span></div>'
    '<div class="gallery-cell"><span class="elapsed tabular-nums" aria-hidden="true">elapsed 28m 20s</span><span class="cell-caption">frozen</span></div>'
    "</div></section>"
)


def _phase_row(state: str, *, tasks: str = "", gates: str = "") -> str:
    text = {
        "pending": "Pending", "running": "Running…", "done": "Done",
        "skipped": "Skipped", "failed": "Failed",
    }[state]
    task_span = (
        f'<span class="phase-tasks tabular-nums">{tasks}</span>'
        if tasks
        else '<span class="phase-tasks tabular-nums" hidden=""></span>'
    )
    return (
        f'<li class="phase-row is-{state}" data-phase="phase_{state}">'
        f'<span class="phase-ring">{_ring(state)}</span>'
        f'<span class="phase-label">Phase ({state})</span>'
        f'<span class="phase-state">{text}</span>'
        f"{task_span}"
        f'<span class="phase-gates" role="group" aria-label="Checks for Phase ({state})">{gates}</span>'
        '<div class="phase-detail" hidden=""></div>'
        "</li>"
    )


_PHASE_ROW_INNER = (
    '<section class="gallery-section"><h2>Phase row</h2>'
    '<p>Each of the five states.</p>'
    '<ol class="phase-checklist" aria-label="Phase row states">'
    + _phase_row("pending")
    + _phase_row("running", tasks="· 23/50 tasks")
    + _phase_row("done", gates=_gate_chip("pass", "✓", "Passed", "source_refs") + _gate_chip("warn", "△", "Warning", "co_terminal_alignment"))
    + _phase_row("skipped")
    + _phase_row("failed", gates=_gate_chip("fail", "✗", "Failed", "curie_anchoring"))
    + "</ol></section>"
)


# The Phase-5 gate strips: a phase with all gates passing ("37 of 37 checks
# passing"), a phase with a CALM amber warning, and a phase with a critical fail.
# Built from the phase-row markup with the summary chip + per-gate chips inside
# the .phase-gates strip (exactly as gallery.js::gateStripSection renders them).
_GATE_STRIP_INNER = (
    '<section class="gallery-section"><h2>Gate strips (live ticker)</h2>'
    '<p>Per-phase gate strips: all-passing reassurance, a CALM amber warning, and a critical fail — icon + text, never color-only.</p>'
    '<ol class="phase-checklist" aria-label="Gate strip states">'
    + _phase_row(
        "done",
        gates=_gate_summary("pass", "✓", "37 of 37 checks passing")
        + _gate_chip("pass", "✓", "Passed", "source_refs")
        + _gate_chip("pass", "✓", "Passed", "curie_anchoring")
        + _gate_chip("pass", "✓", "Passed", "manifest_completeness"),
    ).replace("phase_done", "validate_pass").replace("Phase (done)", "Validate content")
    + _phase_row(
        "done",
        gates=_gate_summary("warn", "△", "11 of 12 checks passing · 1 warning")
        + _gate_chip("pass", "✓", "Passed", "terminal_objective_coverage")
        + _gate_chip("warn", "△", "Warning", "co_terminal_alignment"),
    ).replace("phase_done", "plan_warn").replace("Phase (done)", "Plan course structure")
    + _phase_row(
        "failed",
        gates=_gate_summary("fail", "✗", "8 of 9 checks passing · 1 failed")
        + _gate_chip("pass", "✓", "Passed", "block_prose_entailment")
        + _gate_chip("fail", "✗", "Failed", "numeric_literal_grounding"),
    ).replace("phase_failed", "rewrite_fail").replace("Phase (failed)", "Rewrite content")
    + "</ol></section>"
)


def _course_card(title: str, *, href: str = "", meta: str = "", ask: bool = False) -> str:
    tag, attrs = ("a", f' href="{href}"') if href else ("article", "")
    badge = '<span class="card-badge">Ask-ready</span>' if ask else ""
    metap = f'<p class="card-meta">{meta}</p>' if meta else ""
    return f'<{tag} class="card course-card"{attrs}><h3 class="card-title">{title}</h3>{metap}{badge}</{tag}>'


def _run_card(title: str, status: str, glyph: str, label: str, *, href: str = "", meta: str = "") -> str:
    tag, attrs = ("a", f' href="{href}"') if href else ("article", "")
    metap = f'<p class="card-meta">{meta}</p>' if meta else ""
    return (
        f'<{tag} class="card run-card"{attrs}>'
        f'<div class="card-head"><h3 class="card-title">{title}</h3>{_pill(status, glyph, label)}</div>'
        f"{metap}</{tag}>"
    )


_CARD_INNER = (
    '<section class="gallery-section"><h2>Cards</h2>'
    '<p>courseCard() + runCard().</p>'
    '<div class="gallery-grid">'
    + _course_card("Algebra (OpenStax)", href="#/viewer/algebra", meta="24 pages · 18.2 MB", ask=True)
    + _course_card("Physics 101", meta="12 pages · 9.0 MB")
    + _run_card("PHYS_101", "running", "◐", "Building", href="#/build/GUI-1", meta="textbook_to_course · building")
    + _run_card("BIO_201", "done", "●", "Ready", href="#/build/GUI-2", meta="textbook_to_course · 28m")
    + _run_card("CHEM_101", "failed", "✕", "Failed", href="#/build/GUI-3", meta="textbook_to_course · 6m")
    + "</div></section>"
)

_EMPTY_INNER = (
    '<section class="gallery-section"><h2>Empty state</h2>'
    '<p>Message + a required CTA slot.</p>'
    '<section class="empty-state">'
    '<h2 class="empty-title">No courses yet</h2>'
    '<p class="empty-message">Upload a textbook to build your first course.</p>'
    '<a class="btn primary empty-cta" href="#/create">Build your first course</a>'
    "</section></section>"
)

# A field row: <label for>, control, describedby hint, role=alert error (hidden).
_FIELD_INNER = (
    '<section class="gallery-section"><h2>Field row</h2>'
    '<p>Label-paired input rows.</p>'
    '<form class="gallery-form" novalidate="">'
    '<div class="field field-row">'
    '<label class="field-label" for="fr-1">Course name</label>'
    '<input id="fr-1" name="course_name" type="text" aria-describedby="fr-1-h" required="">'
    '<p id="fr-1-h" class="field-hint">Letters, numbers, “_” and “-” only.</p>'
    '<p class="error field-error" role="alert" hidden=""></p>'
    "</div>"
    '<div class="field field-row">'
    '<label class="field-label" for="fr-2">Outline course model</label>'
    '<select id="fr-2" name="outline_provider" aria-describedby="fr-2-h">'
    '<option value="local" selected="">Ollama (local)</option>'
    '<option value="anthropic">Anthropic (Claude)</option></select>'
    '<p id="fr-2-h" class="field-hint">The AI provider that drafts the course outline.</p>'
    '<p class="error field-error" role="alert" hidden=""></p>'
    "</div>"
    "</form></section>"
)


def _tl_seg(state: str, pct: str) -> str:
    return f'<span class="tl-seg tl-seg-{state}" style="width:{pct}%"></span>'


def _tl_li(state: str, name: str, dur: str) -> str:
    return (
        f'<li class="tl-legend-item">'
        f'<span class="tl-legend-swatch tl-seg-{state}" aria-hidden="true"></span>'
        f'<span class="tl-legend-name">{name}</span>'
        f'<span class="tl-legend-dur tabular-nums">{dur}</span></li>'
    )


_TIMELINE_INNER = (
    '<section class="gallery-section"><h2>Timeline bar</h2>'
    '<p>A stacked horizontal bar of phase durations.</p>'
    '<figure class="timeline-bar">'
    '<div class="tl-bar-track" aria-hidden="true">'
    + _tl_seg("done", "20.00") + _tl_seg("done", "67.50") + _tl_seg("done", "3.75") + _tl_seg("skipped", "8.75")
    + "</div>"
    '<figcaption class="visually-hidden">Phase durations: total 26m 40s</figcaption>'
    '<ul class="tl-legend" aria-label="Phase durations">'
    + _tl_li("done", "Convert", "5m 20s") + _tl_li("done", "Generate content", "18m")
    + _tl_li("done", "Package", "1m") + _tl_li("skipped", "Assessments", "2m 20s")
    + "</ul></figure></section>"
)

_LIVE_DEMO_INNER = (
    '<section class="gallery-section"><h2>Live region demo</h2>'
    '<p>One role="status" aria-live region announces phase transitions, debounced.</p>'
    '<div><div class="gallery-row"><button type="button" class="btn primary">Replay phase sweep</button></div>'
    '<ol class="phase-checklist" aria-label="Course build steps">'
    + _phase_row("running").replace("phase_running", "a").replace("Phase (running)", "Convert textbook")
    + "</ol></div></section>"
)

# The assembled Build Console (run-progress.js). In the gallery it is given the
# shell's single #gallery-live region, so it does NOT mint its own — the markup
# is a .run-progress root carrying the meta line (aria-hidden elapsed) + the
# phase-checklist of ring rows. Reconstructed exactly as run-progress.js emits.
def _overall_header(eta_text: str, *, rough: bool = False, cancelling: bool = False) -> str:
    """The run-progress header: aria-hidden overall ring + ETA + sticky cancel,
    reconstructed exactly as run-progress.js emits it."""
    eta_cls = "run-eta tabular-nums muted" + (" is-rough" if rough else "")
    cancel_text = "Cancelling… (finishing current step)" if cancelling else "Cancel build"
    cancel_attr = ' disabled=""' if cancelling else ""
    return (
        '<div class="run-progress-header">'
        '<span class="run-overall-ring" aria-hidden="true">' + _ring("running") + "</span>"
        '<span class="run-overall-text">'
        '<span class="run-overall-count tabular-nums muted">Step 3 of 5 · </span>'
        f'<span class="{eta_cls}">{eta_text}</span></span>'
        f'<button type="button" class="btn run-cancel-btn"{cancel_attr}>{cancel_text}</button>'
        "</div>"
    )


# The assembled Build Console (run-progress.js): meta line + the overall header
# (ring + honest-ETA range + sticky cancel) + the phase checklist. The ETA is a
# RANGE ("about 19m left"), never a zeroing countdown.
_CONSOLE_INNER = (
    '<section class="gallery-section"><h2>Build Console</h2>'
    '<p>The assembled run-progress.js console — overall ring + honest ETA + sticky cancel + the single narrative line.</p>'
    '<div><div class="gallery-row">'
    '<button type="button" class="btn primary">Replay build</button></div>'
    '<div class="run-progress kit">'
    '<p class="run-progress-meta muted">'
    '<span class="elapsed tabular-nums" aria-hidden="true">elapsed 0s</span></p>'
    + _overall_header("about 19m left")
    + '<ol class="phase-checklist" aria-label="Course build steps">'
    + _phase_row("done").replace("phase_done", "dart_conversion").replace("Phase (done)", "Convert textbook to accessible HTML")
    + _phase_row("done").replace("phase_done", "staging").replace("Phase (done)", "Stage source files")
    + _phase_row("running", tasks="· 50/50 tasks").replace("phase_running", "content_generation").replace("Phase (running)", "Generate course content")
    + _phase_row("pending").replace("phase_pending", "packaging").replace("Phase (pending)", "Package course")
    + _phase_row("pending").replace("phase_pending", "trainforge_assessment").replace("Phase (pending)", "Generate assessments")
    + "</ol></div></div></section>"
)

# The ETA-discipline + cancel states: estimating (<2 runs), rough estimate
# (static prior), and the two-stage "Cancelling…" cancel button.
_CONSOLE_ETA_INNER = (
    '<section class="gallery-section"><h2>Build Console — ETA &amp; cancel states</h2>'
    '<p>The honest ETA is a coarse range, never a zeroing countdown.</p>'
    '<div class="gallery-row">'
    '<div class="gallery-cell"><div class="run-progress kit">'
    '<p class="run-progress-meta muted"><span class="elapsed tabular-nums" aria-hidden="true">elapsed 0s</span></p>'
    + _overall_header("estimating…")
    + '<ol class="phase-checklist" aria-label="Course build steps">'
    + _phase_row("running").replace("phase_running", "a").replace("Phase (running)", "Convert")
    + "</ol></div><span class=\"cell-caption\">estimating (&lt;2 runs)</span></div>"
    '<div class="gallery-cell"><div class="run-progress kit">'
    '<p class="run-progress-meta muted"><span class="elapsed tabular-nums" aria-hidden="true">elapsed 0s</span></p>'
    + _overall_header("about 6m left (rough estimate)", rough=True)
    + '<ol class="phase-checklist" aria-label="Course build steps">'
    + _phase_row("running").replace("phase_running", "a").replace("Phase (running)", "Convert")
    + "</ol></div><span class=\"cell-caption\">rough estimate (prior)</span></div>"
    '<div class="gallery-cell"><div class="run-progress kit">'
    '<p class="run-progress-meta muted"><span class="elapsed tabular-nums" aria-hidden="true">elapsed 0s</span></p>'
    + _overall_header("about 19m left", cancelling=True)
    + '<ol class="phase-checklist" aria-label="Course build steps">'
    + _phase_row("running").replace("phase_running", "a").replace("Phase (running)", "Convert")
    + "</ol></div><span class=\"cell-caption\">Cancelling…</span></div>"
    "</div></section>"
)


def _run_history_entry(title: str, status: str, glyph: str, label: str, *, href: str,
                       meta: str, durs: str = "", actions: str = "") -> str:
    """One Run History entry: a runCard + (optional) timelineBar + actions."""
    bar = f'<figure class="timeline-bar">{durs}</figure>' if durs else ""
    act = f'<div class="run-history-actions">{actions}</div>' if actions else ""
    return (
        '<li class="run-history-li"><div class="run-history-entry">'
        + _run_card(title, status, glyph, label, href=href, meta=meta)
        + bar + act + "</div></li>"
    )


_RUN_HISTORY_INNER = (
    '<section class="gallery-section"><h2>Run History</h2>'
    '<p>The Studio #/runs view — runCards with persisted-duration timelineBars.</p>'
    '<ul class="run-history-list cards" aria-label="Course builds">'
    + _run_history_entry(
        "PHYS_101", "done", "●", "Ready", href="#/create/GUI-1", meta="textbook_to_course · 28m",
        durs=(
            '<div class="tl-bar-track" aria-hidden="true">' + _tl_seg("done", "50.00") + _tl_seg("done", "50.00") + "</div>"
            '<figcaption class="visually-hidden">Phase durations for PHYS_101: total 23m</figcaption>'
            '<ul class="tl-legend" aria-label="Phase durations for PHYS_101">'
            + _tl_li("done", "Convert", "5m 20s") + _tl_li("done", "Generate content", "18m") + "</ul>"
        ),
        actions='<button type="button" class="btn">Run again</button>',
    )
    + _run_history_entry(
        "BIO_201", "running", "◐", "Building", href="#/create/GUI-2", meta="textbook_to_course · building",
        actions='<button type="button" class="btn">Re-open build</button>',
    )
    + _run_history_entry(
        "CHEM_101", "failed", "✕", "Failed", href="#/create/GUI-3", meta="textbook_to_course · 6m",
        durs=(
            '<div class="tl-bar-track" aria-hidden="true">' + _tl_seg("done", "80.00") + _tl_seg("failed", "20.00") + "</div>"
            '<figcaption class="visually-hidden">Phase durations for CHEM_101: total 6m</figcaption>'
            '<ul class="tl-legend" aria-label="Phase durations for CHEM_101">'
            + _tl_li("done", "Convert", "5m") + _tl_li("failed", "Generate content", "1m") + "</ul>"
        ),
        actions='<button type="button" class="btn">Run again</button>',
    )
    + "</ul></section>"
)

_ALL_SECTIONS = (
    _RING_INNER + _PILL_INNER + _GATE_INNER + _GATE_STRIP_INNER + _ELAPSED_INNER
    + _PHASE_ROW_INNER + _CARD_INNER + _EMPTY_INNER + _FIELD_INNER + _TIMELINE_INNER
    + _LIVE_DEMO_INNER + _CONSOLE_INNER + _CONSOLE_ETA_INNER + _RUN_HISTORY_INNER
)


# --------------------------------------------------------------------------- #
# Zero-AA gate over the shell + each component section + the full gallery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,inner",
    [
        ("shell-idle", ""),
        ("ring", _RING_INNER),
        ("pill", _PILL_INNER),
        ("gate-chip", _GATE_INNER),
        ("gate-strip", _GATE_STRIP_INNER),
        ("elapsed", _ELAPSED_INNER),
        ("phase-row", _PHASE_ROW_INNER),
        ("card", _CARD_INNER),
        ("empty-state", _EMPTY_INNER),
        ("field-row", _FIELD_INNER),
        ("timeline-bar", _TIMELINE_INNER),
        ("live-demo", _LIVE_DEMO_INNER),
        ("build-console", _CONSOLE_INNER),
        ("build-console-eta", _CONSOLE_ETA_INNER),
        ("run-history", _RUN_HISTORY_INNER),
        ("full-gallery", _ALL_SECTIONS),
    ],
)
def test_gallery_components_zero_aa_findings(label, inner):
    _assert_clean(f"gallery-{label}", _shell_with_gallery(inner))


# --------------------------------------------------------------------------- #
# Build Console (run-progress.js) — the Phase-3 assembled console fixture
# --------------------------------------------------------------------------- #


def test_build_console_fixture_passes_wcag_with_single_live_region():
    """The assembled Build Console fixture (rings + elapsed + checklist) is
    zero-AA, and the WHOLE page exposes EXACTLY ONE role=status region — the
    shell's #gallery-live. The console reuses it (it does NOT mint a second),
    so the §9 single-live-truth contract holds with the console mounted."""
    html = _shell_with_gallery(_CONSOLE_INNER)
    _assert_clean("gallery-build-console", html)
    soup = _soup(html)
    # Exactly one role=status aria-live=polite region on the page.
    lives = soup.find_all(attrs={"aria-live": True})
    assert len(lives) == 1, f"expected one aria-live region, found {len(lives)}"
    assert lives[0].get("role") == "status" and lives[0].get("aria-live") == "polite"
    # The console mounted: its root + the phase checklist + ring decorations.
    rp = soup.find("div", class_="run-progress")
    assert rp is not None, "the Build Console root must render"
    assert rp.find("ol", class_="phase-checklist") is not None
    # The elapsed timer is aria-hidden decoration (no SR chatter).
    timer = rp.find("span", class_="elapsed")
    assert timer is not None and timer.get("aria-hidden") == "true"
    # Every ring SVG inside the console is aria-hidden decoration.
    for svg in rp.find_all("svg", class_="ring-svg"):
        assert svg.get("aria-hidden") == "true"


def test_gallery_js_imports_and_drives_run_progress_console():
    """gallery.js must import run-progress.js and feed it ISO-stamped lines
    (the [phase] / [progress] replay that exercises Tier-1 timing)."""
    js = (DEV_DIR / "gallery.js").read_text(encoding="utf-8")
    assert "/shared/components/run-progress.js" in js, "gallery.js must import run-progress.js"
    assert "runProgressConsole(" in js, "gallery.js must build the console"
    assert "[progress]" in js and "[phase]" in js, "the replay must drive [phase]/[progress] lines"
    assert "toISOString" in js, "the replay must ISO-stamp lines (Tier-1 timing)"


# --------------------------------------------------------------------------- #
# Phase 4: overall ring + honest ETA + cancel + Run History
# --------------------------------------------------------------------------- #


def test_build_console_eta_and_cancel_fixture_single_live_region():
    """The ETA/cancel-state Build Console fixture is zero-AA AND keeps the page
    at EXACTLY ONE role=status region (the shell's #gallery-live). The overall
    ring is aria-hidden decoration; the ETA + cancel surface NO new live region."""
    html = _shell_with_gallery(_CONSOLE_ETA_INNER)
    _assert_clean("gallery-build-console-eta", html)
    soup = _soup(html)
    lives = soup.find_all(attrs={"aria-live": True})
    assert len(lives) == 1, f"expected one aria-live region, found {len(lives)}"
    assert lives[0].get("role") == "status" and lives[0].get("aria-live") == "polite"
    # The overall ring is aria-hidden decoration (truth is the single status line).
    for slot in soup.find_all("span", class_="run-overall-ring"):
        assert slot.get("aria-hidden") == "true", "overall ring slot must be aria-hidden"
        for svg in slot.find_all("svg", class_="ring-svg"):
            assert svg.get("aria-hidden") == "true"
    # The ETA line is NOT a live region (no aria-live on .run-eta) — announcements
    # route through the single #gallery-live region, not a per-ETA chatter region.
    for eta in soup.find_all("span", class_="run-eta"):
        assert eta.get("aria-live") is None, ".run-eta must NOT mint its own live region"


def test_eta_is_a_range_never_a_zeroing_countdown():
    """ETA discipline (hard): the rendered ETA is a coarse RANGE ('about Nm
    left') / 'estimating…' / 'rough estimate', NEVER a precise sub-minute
    countdown. The console source proves the bucketing + first-run discipline."""
    soup = _soup(_shell_with_gallery(_CONSOLE_ETA_INNER))
    etas = [e.get_text(strip=True) for e in soup.find_all("span", class_="run-eta")]
    assert etas, "the ETA fixture must render .run-eta lines"
    # Every ETA is one of the three honest forms (range / estimating / rough).
    for t in etas:
        assert (
            t == "estimating…"
            or t.startswith("about ")
            or "rough estimate" in t
        ), f"ETA {t!r} is not an honest range/estimating/rough form"
    # The static prior path carries the 'rough estimate' qualifier.
    assert any("rough estimate" in t for t in etas), "the prior-source ETA must say 'rough estimate'"
    assert any(t == "estimating…" for t in etas), "the <2-run ETA must say 'estimating…'"

    # The run-progress.js source proves the discipline: a coarse bucket, the
    # 'estimating…' first-run guard, the 'rough estimate' prior label, and NO
    # second-by-second zeroing countdown (the phrase is bucketed minutes).
    js = (COMPONENTS_DIR / "run-progress.js").read_text(encoding="utf-8")
    assert "estimating…" in js, "console must render 'estimating…' on <2 historical runs"
    assert "rough estimate" in js, "console must label the static-prior ETA 'rough estimate'"
    assert "tooFewSamples" in js, "console must guard the first-run (too-few-samples) case"
    assert "_etaPhrase" in js and "about " in js, "ETA must be a coarse 'about Nm left' range"


def test_cancel_button_is_real_button_with_two_stage_text():
    """The sticky cancel is a real <button> (focus-visible, >=24px via kit CSS);
    its two-stage text is 'Cancel build' → 'Cancelling… (finishing current
    step)', never a claim of instant stop."""
    soup = _soup(_shell_with_gallery(_CONSOLE_INNER + _CONSOLE_ETA_INNER))
    cancels = soup.find_all("button", class_="run-cancel-btn")
    assert cancels, "the console must render a run-cancel-btn"
    texts = {c.get_text(strip=True) for c in cancels}
    assert "Cancel build" in texts, "the idle cancel button reads 'Cancel build'"
    assert any("Cancelling" in t and "finishing current step" in t for t in texts), (
        "the requested cancel must read 'Cancelling… (finishing current step)'"
    )
    # The two-stage text + 202 handling live in the run-progress + host source.
    js = (COMPONENTS_DIR / "run-progress.js").read_text(encoding="utf-8")
    assert "Cancelling… (finishing current step)" in js
    assert "setCancelling" in js, "console must expose the two-stage setCancelling()"
    create = (REPO_ROOT / "gui" / "static" / "studio" / "create.js").read_text(encoding="utf-8")
    assert "cancel_requested" in create or "setCancelling" in create
    assert "/cancel" in create, "create.js host must own the cancel POST"


def test_run_history_fixture_cards_and_timeline_bars():
    """The Run History fixture renders runCards (statusPill, non-color-only) +
    aria-hidden timeline bars with a semantic legend, and is zero-AA."""
    html = _shell_with_gallery(_RUN_HISTORY_INNER)
    _assert_clean("gallery-run-history", html)
    soup = _soup(html)
    cards = soup.find_all(class_="run-card")
    assert len(cards) == 3, "run history must render one runCard per build"
    for c in cards:
        # statusPill present (glyph aria-hidden + a text label, never color-only).
        pill = c.find("span", class_="pill")
        assert pill is not None
        assert pill.find("span", class_="pill-glyph").get("aria-hidden") == "true"
        assert pill.find("span", class_="pill-label").get_text(strip=True)
    # Timeline bars: aria-hidden track + a semantic legend (the a11y truth).
    bars = soup.find_all("figure", class_="timeline-bar")
    assert bars, "run history must render timeline bars"
    for b in bars:
        assert b.find("div", class_="tl-bar-track").get("aria-hidden") == "true"
        legend = b.find("ul", class_="tl-legend")
        assert legend is not None and legend.get("aria-label")
    # No new live region introduced by Run History.
    lives = soup.find_all(attrs={"aria-live": True})
    assert len(lives) == 1, f"Run History must not add a live region; found {len(lives)}"


def test_run_history_lives_in_studio_router_no_auth_remount():
    """Run History is reachable via the EXISTING studio hash router at #/runs —
    NO Phase-2 auth remount, NO route/auth change. Confirm the route is wired in
    studio.js (the shared createRouter) and a nav link exists."""
    studio = (REPO_ROOT / "gui" / "static" / "studio" / "studio.js").read_text(encoding="utf-8")
    assert "renderRunHistory" in studio, "studio.js must import the run-history view"
    assert "runs:" in studio, "studio.js must register the #/runs route in its router"
    # No auth/remount surface touched: the run-history module + studio router use
    # the shared api()/router, not a new auth guard.
    rh = (REPO_ROOT / "gui" / "static" / "studio" / "run-history.js").read_text(encoding="utf-8")
    assert "createRouter" not in rh, "run-history must reuse the shell router, not mint a new one"
    assert "/api/runs" in rh, "run-history reads GET /api/runs"
    index = (REPO_ROOT / "gui" / "static" / "studio" / "index.html").read_text(encoding="utf-8")
    assert 'href="#/runs"' in index, "studio shell must expose a Run history nav link"


def test_gallery_js_drives_overall_ring_eta_and_run_history():
    """gallery.js must exercise the overall ring + ETA states + cancel + Run
    History fixtures (the Phase-4 additions)."""
    js = (DEV_DIR / "gallery.js").read_text(encoding="utf-8")
    assert "durationsSource" in js, "gallery must feed the console a durations source (ETA)"
    assert "estimating" in js or "durations: {}" in js, "gallery must demo the estimating ETA state"
    assert "prior" in js, "gallery must demo the rough-estimate (prior) ETA state"
    assert "onCancel" in js and "setCancelling" in js, "gallery must demo the two-stage cancel"
    assert "runHistorySection" in js, "gallery must render the Run History fixture"
    assert "timelineBar(" in js and "runCard(" in js, "Run History uses timelineBar + runCard"


# --------------------------------------------------------------------------- #
# Structural assertions the validator does not fully cover
# --------------------------------------------------------------------------- #


def test_shell_links_tokens_and_kit_css_and_loads_module():
    html = GALLERY_INDEX.read_text(encoding="utf-8")
    tok = html.find('href="/shared/tokens.css"')
    kit = html.find('href="/shared/components/components.css"')
    assert tok != -1, "gallery must link the shared tokens"
    assert kit != -1, "gallery must link the component-kit CSS"
    assert tok < kit, "tokens must load BEFORE the kit CSS"
    assert 'type="module"' in html, "gallery must load gallery.js as an ES module"


def test_shell_has_single_h1_skip_link_and_one_live_region():
    soup = _soup(GALLERY_INDEX.read_text(encoding="utf-8"))
    assert len(soup.find_all("h1")) == 1, "the gallery shell carries exactly one h1"
    skip = soup.find("a", class_="skip-link")
    assert skip is not None and skip.get("href") == "#gallery-main"
    main = soup.find("main", id="gallery-main")
    assert main is not None and main.get("tabindex") == "-1"
    # Exactly ONE role=status aria-live=polite region (the single live truth).
    lives = soup.find_all(attrs={"aria-live": True})
    assert len(lives) == 1, f"expected one aria-live region, found {len(lives)}"
    live = lives[0]
    assert live.get("role") == "status" and live.get("aria-live") == "polite"


def test_rings_are_aria_hidden_decoration():
    soup = _soup(_shell_with_gallery(_RING_INNER))
    svgs = soup.find_all("svg", class_="ring-svg")
    assert svgs, "ring section must render ring SVGs"
    for svg in svgs:
        assert svg.get("aria-hidden") == "true", "every ring SVG must be aria-hidden"
    # Each ring carries a visually-hidden label (the a11y truth).
    rings = soup.find_all("span", class_="ring")
    for r in rings:
        assert r.find("span", class_="visually-hidden") is not None


def test_pills_are_not_color_only():
    soup = _soup(_shell_with_gallery(_PILL_INNER))
    pills = soup.find_all("span", class_="pill")
    assert pills, "pill section must render pills"
    for p in pills:
        glyph = p.find("span", class_="pill-glyph")
        label = p.find("span", class_="pill-label")
        assert glyph is not None and glyph.get("aria-hidden") == "true"
        assert label is not None and label.get_text(strip=True), "pill must carry a text label"


def test_gate_chips_carry_visually_hidden_label():
    soup = _soup(_shell_with_gallery(_GATE_INNER))
    chips = soup.find_all("span", class_="gate-chip")
    assert chips, "gate-chip section must render chips"
    for c in chips:
        assert c.find("span", class_="visually-hidden") is not None, "chip needs a hidden text label"
        assert c.find("span", class_="gate-chip-glyph").get("aria-hidden") == "true"


def test_gate_strip_fixture_passes_wcag_and_adds_no_live_region():
    """The Phase-5 gate-strip fixture (per-phase gate strips + 'N of M checks
    passing' summary chips) is zero-AA AND introduces NO new role=status region
    — any gate announcement routes through the shell's single #gallery-live."""
    html = _shell_with_gallery(_GATE_STRIP_INNER)
    _assert_clean("gallery-gate-strip", html)
    soup = _soup(html)
    # Exactly one role=status aria-live=polite region on the whole page.
    lives = soup.find_all(attrs={"aria-live": True})
    assert len(lives) == 1, f"gate strip must not add a live region; found {len(lives)}"
    assert lives[0].get("role") == "status" and lives[0].get("aria-live") == "polite"
    # The gate-strip summary chips are NOT live regions (no aria-live on them).
    summaries = soup.find_all("span", class_="phase-gate-summary")
    assert len(summaries) == 3, "fixture renders three per-phase summary chips"
    for s in summaries:
        assert s.get("aria-live") is None, "summary chip must NOT mint its own live region"


def test_gate_summary_chips_are_not_color_only_and_warning_is_calm():
    """Each summary chip pairs a glyph (aria-hidden) with VISIBLE text (never
    color-only). The warning reads CALM ('warning', amber △), not alarming."""
    soup = _soup(_shell_with_gallery(_GATE_STRIP_INNER))
    summaries = soup.find_all("span", class_="phase-gate-summary")
    assert summaries, "gate-strip section must render summary chips"
    texts = []
    for s in summaries:
        glyph = s.find("span", class_="phase-gate-summary-glyph")
        label = s.find("span", class_="phase-gate-summary-text")
        assert glyph is not None and glyph.get("aria-hidden") == "true"
        assert label is not None and label.get_text(strip=True), "summary needs visible text"
        texts.append(label.get_text(strip=True))
    # All-passing reassurance: "37 of 37 checks passing".
    assert any(t == "37 of 37 checks passing" for t in texts), "all-pass chip must read 'N of N checks passing'"
    # The warning is CALM: it says "warning", not "fail"/"error"/"critical".
    warn = next(s for s in summaries if "is-warn" in (s.get("class") or []))
    wtext = warn.find("span", class_="phase-gate-summary-text").get_text(strip=True).lower()
    assert "warning" in wtext and "fail" not in wtext and "error" not in wtext, (
        f"the warning summary must read calmly, got {wtext!r}"
    )
    assert warn.find("span", class_="phase-gate-summary-glyph").get_text(strip=True) == "△"
    # The critical fail uses the ✗ glyph + names the failure.
    fail = next(s for s in summaries if "is-fail" in (s.get("class") or []))
    assert fail.find("span", class_="phase-gate-summary-glyph").get_text(strip=True) == "✗"


def test_run_progress_parses_gate_lines_and_summary():
    """run-progress.js must defensively parse the [gate] per-gate + __summary__
    lines, route per-gate to addGate and the summary to the 'N of M' chip, and
    keep the calm-warning treatment (a warning-severity fail → 'warn', not
    'fail')."""
    js = (COMPONENTS_DIR / "run-progress.js").read_text(encoding="utf-8")
    assert "[gate]" in js, "console must parse [gate] lines"
    assert "__summary__" in js, "console must parse the __summary__ aggregate line"
    assert "setGateSummary" in js, "console must drive the per-phase summary chip"
    assert "_gateResult" in js and "warn" in js, "warning-severity fail must read as a calm warn"
    # phase-row.js exposes the summary setter the console calls.
    pr = (COMPONENTS_DIR / "phase-row.js").read_text(encoding="utf-8")
    assert "setGateSummary" in pr, "phase-row must expose setGateSummary"
    assert "checks passing" in pr, "the summary chip text reads 'N of M checks passing'"


def test_gallery_js_renders_gate_strip_section():
    """gallery.js must render the Phase-5 gate-strip fixture (all-pass / calm
    warning / critical fail) via setGateSummary on phase rows."""
    js = (DEV_DIR / "gallery.js").read_text(encoding="utf-8")
    assert "gateStripSection" in js, "gallery must render the gate-strip fixture"
    assert "setGateSummary(" in js, "gate-strip fixture drives the summary chip"
    assert "37, 37" in js, "the all-passing strip reads '37 of 37 checks passing'"


def test_elapsed_is_aria_hidden_tabular():
    soup = _soup(_shell_with_gallery(_ELAPSED_INNER))
    for t in soup.find_all("span", class_="elapsed"):
        assert t.get("aria-hidden") == "true", "elapsed timer must be aria-hidden (no SR chatter)"
        assert "tabular-nums" in (t.get("class") or [])


def test_empty_state_has_required_cta():
    soup = _soup(_shell_with_gallery(_EMPTY_INNER))
    es = soup.find("section", class_="empty-state")
    assert es is not None
    cta = es.find(class_="empty-cta")
    assert cta is not None and cta.get_text(strip=True), "empty state must carry a CTA"


def test_field_rows_are_labelled_and_described():
    soup = _soup(_shell_with_gallery(_FIELD_INNER))
    for ctl_id in ("fr-1", "fr-2"):
        ctl = soup.find(id=ctl_id)
        assert ctl is not None
        assert soup.find("label", attrs={"for": ctl_id}) is not None, f"{ctl_id} needs a <label for>"
        desc = ctl.get("aria-describedby")
        assert desc and soup.find(id=desc) is not None, f"{ctl_id} needs a real describedby hint"


def test_timeline_bar_is_aria_hidden_with_semantic_legend():
    soup = _soup(_shell_with_gallery(_TIMELINE_INNER))
    track = soup.find("div", class_="tl-bar-track")
    assert track is not None and track.get("aria-hidden") == "true"
    legend = soup.find("ul", class_="tl-legend")
    assert legend is not None and legend.get("aria-label")
    assert legend.find_all("li"), "legend must carry per-phase rows (the a11y truth)"


def test_gallery_js_imports_every_component_module():
    js = (DEV_DIR / "gallery.js").read_text(encoding="utf-8")
    for mod in _COMPONENT_MODULES:
        assert f"/shared/components/{mod}" in js, f"gallery.js must import {mod}"


def test_dev_index_links_gallery_without_breaking_navlist():
    """The added gallery link lives OUTSIDE #dev-navlist (keeps the 9-link count)."""
    soup = _soup((DEV_DIR / "index.html").read_text(encoding="utf-8"))
    link = soup.find("a", id="dev-gallery-link")
    assert link is not None and link.get("href") == "/dev/gallery.html"
    navlist = soup.find("ul", id="dev-navlist")
    assert link not in navlist.find_all("a"), "gallery link must not be inside the pane navlist"


def test_kit_css_focus_visible_not_suppressed_and_targets_meet_minimum():
    css = COMPONENTS_CSS.read_text(encoding="utf-8")
    assert re.search(r":focus-visible[^{]*\{[^}]*outline\s*:\s*[1-9]", css), (
        "components.css must define a :focus-visible outline (>=1px)"
    )
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).lower(), m.group(2)
        if re.search(r"outline\s*:\s*(none|0)\b", body):
            assert ":focus:not(:focus-visible)" in selector, (
                f"bare outline:none in selector {selector!r} suppresses focus"
            )
    # No interactive control sized < 24px.
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).strip().lower(), m.group(2).lower()
        if not re.search(r"\b(button|input|select|textarea|\.btn|\.pill|\.gate-chip)\b", selector):
            continue
        for dim in re.finditer(r"(?:min-)?(?:width|height)\s*:\s*(\d+)px", body):
            px = int(dim.group(1))
            assert not (0 < px < 24), f"selector {selector!r} sizes a control at {px}px (< 24px)"


def test_kit_css_uses_tokens_not_raw_hex():
    """The kit consumes var(--token); a raw hex would break the frozen contract."""
    css = COMPONENTS_CSS.read_text(encoding="utf-8")
    # Strip comments, then assert no #rrggbb / #rgb literal survives.
    no_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", no_comments)
    assert not hexes, f"components.css must use tokens, found raw hex: {hexes}"


def test_no_unguarded_animation_in_kit_css():
    """Any animation must be the tokens.css `spin` (already reduced-motion guarded)."""
    css = COMPONENTS_CSS.read_text(encoding="utf-8")
    no_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    # No @keyframes defined locally (motion lives in tokens.css).
    assert "@keyframes" not in no_comments, "kit CSS must not define its own keyframes (motion lives in tokens.css)"
    for m in re.finditer(r"animation\s*:\s*([^;]+);", no_comments):
        assert "spin" in m.group(1), (
            f"kit animation must reference the guarded `spin` keyframe, got: {m.group(1)!r}"
        )
