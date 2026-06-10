"""WS4 milestone gate — automated WCAG 2.2 AA validation of the learner UI.

THE milestone bar (master plan M5–M7): **zero CRITICAL (Level A) and zero HIGH
(Level AA) ``WCAGValidator`` findings** on every server-rendered page variant a
learner can actually see — the idle learner page, the busy state, the page with
each answer-fragment status swapped in (exactly as ``learn.js`` does the swap,
performed here in Python via bs4), each typed-error fragment, and the
source-viewer output over the WS1 mini-course fixture (dart + imscc arms).

The validator (``DART/pdf_converter/wcag_validator.py``) is bs4-only (bs4 is a
base dependency), so this gate runs on a default install with NO fastapi /
torch / LibV2 / model dependency — it is a pure ``string -> ValidationReport``
transform over the *real* rendered output of the WS4-owned renderers. The
source-viewer arm needs the WS1 mini-course fixture on disk under an isolated
``libv2_root`` (the ``conftest.libv2_root`` env seam) but still loads no model.

CRITICAL = WCAG Level A failure, HIGH = WCAG Level AA failure (the validator's
own ``IssueSeverity`` mapping). MEDIUM / LOW findings are collected and printed
as non-blocking diagnostics (they do not move the milestone bar) — the gate
NEVER weakens the assertion by severity blanket; where a finding is a genuine
validator false-positive for a fragment, the gate assembles the FULL document so
the validator sees real, complete output (plan § 6.1 / R5).

The opt-in real-server smoke (``@pytest.mark.real_models``) spins the learner
app via fastapi ``TestClient`` and asserts the served ``/learn/`` page passes
the same gate; the model-dependent ask round-trip is gated behind an Ollama
opt-in env (it needs a live local answer backend).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Tuple

import pytest

from DART.pdf_converter.wcag_validator import IssueSeverity, WCAGValidator
from gui.services.answer_render import (
    ERROR_COPY,
    render_answer_fragment,
    render_error_fragment,
)

# --------------------------------------------------------------------------- #
# Paths / fixtures
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARN_INDEX = REPO_ROOT / "gui" / "static" / "learn" / "index.html"
MINI_COURSE = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"

# The six WS3 pipeline status constants (refused/blocked render no answer text
# and no citations; answered/answered_with_warnings carry both).
_REFUSED_BLOCKED = (
    "refused_low_confidence",
    "refused_not_in_course",
    "blocked_invalid_citation",
    "blocked_citation_gate",
)

# Severities that move the milestone bar.
_BLOCKING = {IssueSeverity.CRITICAL, IssueSeverity.HIGH}


# --------------------------------------------------------------------------- #
# Canned payloads (GroundedAnswer.to_dict() shapes — no model, no LibV2)
# --------------------------------------------------------------------------- #


def _answered_payload(status: str) -> dict:
    """A representative answered / answered_with_warnings payload.

    Carries multi-paragraph answer text and a citation that exercises every
    optional sub-element of the renderer (module tag, approximate-location
    marker, text quote) so the gate validates the richest fragment shape.
    """
    return {
        "status": status,
        "course_slug": "mini-dart",
        "answer_text": (
            "A vector store indexes embedding vectors for nearest-neighbour "
            "search.\n\n"
            "Retrieval quality is commonly measured with recall at k."
        ),
        "citations": [
            {
                "page_label": "Alpha Page",
                "item_path": "alpha.html",
                "module_id": "M1",
                "anchor_status": "resolved_exact",
                "text_quote": "A vector store is a database that indexes vectors.",
                "link_target": {
                    "item_path": "alpha.html",
                    "fragment": {"kind": "heading", "value": "retrieval-quality"},
                },
            },
            {
                # Second citation: approximate anchor, no module, no quote —
                # exercises the "(approximate location)" marker + top-of-page
                # link (no heading fragment).
                "page_label": "Beta Page",
                "item_path": "beta.html",
                "anchor_status": "approx_section",
                "link_target": {"item_path": "beta.html", "fragment": {"kind": None}},
            },
        ],
    }


def _refused_payload(status: str) -> dict:
    """A refused/blocked payload — contract guarantees no text, no citations."""
    return {"status": status, "course_slug": "mini-dart", "answer_text": None, "citations": []}


# --------------------------------------------------------------------------- #
# Page assembly — reproduce the JS swap in Python (bs4), exactly as learn.js
# inserts the server fragment into #answer-region.
# --------------------------------------------------------------------------- #


def _bs4():
    from bs4 import BeautifulSoup  # noqa: PLC0415

    return BeautifulSoup


def _assemble_page(
    fragment_html: str = "", *, busy: bool = False, status_text: str = ""
) -> str:
    """Read the served learner index.html and swap a fragment into the answer
    region — exactly the JS swap path (``region.innerHTML = html``), done here in
    Python so the validator sees the full document a learner actually gets.

    ``busy`` flips ``aria-busy="true"`` on the answer region; ``status_text``
    writes the polite live-region message (the busy-state snapshot).
    """
    BeautifulSoup = _bs4()
    soup = BeautifulSoup(LEARN_INDEX.read_text(encoding="utf-8"), "html.parser")
    region = soup.find(id="answer-region")
    assert region is not None, "learner index.html must carry #answer-region"
    if busy:
        region["aria-busy"] = "true"
    if status_text:
        status_el = soup.find(id="status")
        assert status_el is not None, "learner index.html must carry #status"
        status_el.string = status_text
    if fragment_html:
        region.append(BeautifulSoup(fragment_html, "html.parser"))
    return str(soup)


def _gate(html: str) -> Tuple[List, List]:
    """Validate ``html``; return ``(blocking_issues, diagnostic_issues)``."""
    report = WCAGValidator().validate(html)
    blocking = [i for i in report.issues if i.severity in _BLOCKING]
    diagnostics = [i for i in report.issues if i.severity not in _BLOCKING]
    return blocking, diagnostics


def _assert_clean(variant: str, html: str) -> None:
    """Assert zero CRITICAL/HIGH; surface diagnostics on the captured stream."""
    blocking, diagnostics = _gate(html)
    if diagnostics:
        for d in diagnostics:
            print(
                f"[a11y-diag] {variant}: {d.severity.value} WCAG {d.criterion} "
                f"— {d.message}"
            )
    assert not blocking, (
        f"{variant}: {len(blocking)} CRITICAL/HIGH WCAG finding(s) — "
        + "; ".join(f"{i.severity.value} {i.criterion} {i.message}" for i in blocking)
    )


# --------------------------------------------------------------------------- #
# Source-viewer fixture (mini-course copied into an isolated libv2_root)
# --------------------------------------------------------------------------- #


def _make_course(libv2_root: Path, slug: str, *, kind: str) -> Path:
    """Copy the WS1 mini-course source tree into ``libv2_root/courses/<slug>``.

    ``kind`` (``dart`` / ``imscc``) selects the chunks dir so
    ``_infer_chunkset_kind`` returns the kind under test.
    """
    course_dir = libv2_root / "courses" / slug
    course_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MINI_COURSE / "source", course_dir / "source")
    chunks_dir = {"dart": "dart_chunks", "imscc": "imscc_chunks"}[kind]
    cdir = course_dir / chunks_dir
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    return course_dir


# --------------------------------------------------------------------------- #
# 1) The WCAG 2.2 AA gate — every learner page variant
# --------------------------------------------------------------------------- #


def test_idle_page_zero_aa_findings():
    """The learner page as served (no fragment) has zero CRITICAL/HIGH."""
    _assert_clean("idle", _assemble_page())


def test_busy_state_zero_aa_findings():
    """The busy snapshot (aria-busy=true + polite status text) is clean."""
    _assert_clean(
        "busy",
        _assemble_page(
            busy=True,
            status_text="Searching the course materials. This can take up to a minute.",
        ),
    )


@pytest.mark.parametrize("status", ["answered", "answered_with_warnings"])
def test_answered_variants_zero_aa_findings(status):
    """Answered + answered_with_warnings (with rich citations) are clean."""
    frag = render_answer_fragment(_answered_payload(status))
    _assert_clean(status, _assemble_page(frag))


@pytest.mark.parametrize("status", _REFUSED_BLOCKED)
def test_refused_blocked_variants_zero_aa_findings(status):
    """Each refused/blocked status fragment is clean (no text, no citations)."""
    frag = render_answer_fragment(_refused_payload(status))
    _assert_clean(status, _assemble_page(frag))


@pytest.mark.parametrize("error_key", sorted(ERROR_COPY.keys()))
def test_error_fragment_variants_zero_aa_findings(error_key):
    """Every typed-error copy key renders a clean error fragment."""
    frag = render_error_fragment(error_key)
    _assert_clean(error_key, _assemble_page(frag))


def test_unknown_status_falls_back_clean():
    """An unknown status renders the generic-error fragment (fail-safe), clean."""
    frag = render_answer_fragment({"status": "totally_unknown", "course_slug": "x"})
    _assert_clean("unknown_status_fallback", _assemble_page(frag))


@pytest.mark.parametrize("slug,kind", [("mini-dart", "dart"), ("mini-imscc", "imscc")])
def test_source_viewer_zero_aa_findings(slug, kind, libv2_root):
    """The viewer-wrapped archived source page (banner + injected ids) is clean.

    Covers both the dart on-disk arm and the imscc in-memory-zip-member arm of
    ``resolve_source_page``. The banner nav + heading-id injection must not break
    heading hierarchy or the document lang.
    """
    from gui.services.source_page import render_source_page  # noqa: PLC0415

    _make_course(libv2_root, slug, kind=kind)
    # Heading-fragment arm: alpha.html with a #fragment pointing at a heading.
    res = render_source_page(slug, "alpha.html", libv2_root, fragment="retrieval-quality")
    _assert_clean(f"viewer-{kind}-alpha-heading", res.html)
    # No-fragment arm: beta.html (banner emits the "cited section is on this
    # page" note instead of a #fragment).
    res2 = render_source_page(slug, "beta.html", libv2_root, fragment=None)
    _assert_clean(f"viewer-{kind}-beta-nofrag", res2.html)


# --------------------------------------------------------------------------- #
# 2) Structural assertions the validator doesn't fully cover (plan § 5 D2)
# --------------------------------------------------------------------------- #


def _soup(html: str):
    return _bs4()(html, "html.parser")


def test_assembled_page_has_exactly_one_h1():
    """Every assembled learner page has exactly one <h1> (the page heading)."""
    for status in ("answered", *_REFUSED_BLOCKED):
        payload = (
            _answered_payload(status)
            if status == "answered"
            else _refused_payload(status)
        )
        html = _assemble_page(render_answer_fragment(payload))
        h1s = _soup(html).find_all("h1")
        assert len(h1s) == 1, f"{status}: expected one <h1>, found {len(h1s)}"


def test_answered_fragment_heading_order_no_skip():
    """The answered fragment goes h1 (page) → h2 (answer) → h3 (sources): no skip."""
    html = _assemble_page(render_answer_fragment(_answered_payload("answered")))
    levels = [int(h.name[1]) for h in _soup(html).find_all(re.compile(r"^h[1-6]$"))]
    assert levels[0] == 1, "first heading must be the page h1"
    prev = 0
    for lvl in levels:
        assert lvl <= prev + 1, f"heading level skip in {levels}"
        prev = lvl
    # The sources subheading must be present as an h3 under the answer h2.
    assert 3 in levels, "answered fragment must carry an <h3>Sources</h3>"


def test_answer_heading_carries_tabindex_for_focus_move():
    """The answer <h2> carries id=answer-h + tabindex=-1 so JS can move focus."""
    html = _assemble_page(render_answer_fragment(_refused_payload("refused_low_confidence")))
    h2 = _soup(html).find("h2", id="answer-h")
    assert h2 is not None, "answer fragment must carry <h2 id='answer-h'>"
    assert h2.get("tabindex") == "-1", "answer h2 must carry tabindex='-1'"


def test_single_polite_status_live_region():
    """Exactly one role=status live region, polite, present on the page."""
    soup = _soup(_assemble_page())
    status_regions = soup.find_all(attrs={"role": "status"})
    assert len(status_regions) == 1, "exactly one role=status live region required"
    region = status_regions[0]
    assert region.get("aria-live") == "polite", "status region must be aria-live=polite"
    assert region.get("id") == "status"


def test_every_citation_link_is_focusable_contained_source_url():
    """Each citation <li> link resolves to /api/learn/source/... with a contained
    item_path (no traversal, no off-course href), and carries full link text."""
    html = _assemble_page(render_answer_fragment(_answered_payload("answered")))
    soup = _soup(html)
    links = soup.select("ol.sources li a")
    assert links, "answered fragment must render at least one citation link"
    for a in links:
        href = a.get("href", "")
        assert href.startswith("/api/learn/source/"), f"bad citation href: {href}"
        # The item_path query must not carry an encoded traversal sequence.
        assert "%2e%2e" not in href.lower(), f"citation href encodes '..': {href}"
        assert ".." not in href, f"citation href carries '..': {href}"
        text = a.get_text(strip=True)
        assert text.startswith("Source:"), f"citation link text must lead 'Source:': {text!r}"
        assert len(text) > len("Source:"), "citation link needs a page label after 'Source:'"


def test_form_controls_are_label_paired():
    """Every form control (select, textarea) is associated with a <label for=…>."""
    soup = _soup(_assemble_page())
    for control in soup.select("form select, form textarea, form input"):
        ctype = control.get("type", "")
        if ctype in ("hidden", "submit", "button", "reset"):
            continue
        cid = control.get("id")
        assert cid, f"control {control.name} needs an id for label pairing"
        if control.get("aria-label") or control.get("aria-labelledby"):
            continue
        assert soup.find("label", {"for": cid}) is not None, (
            f"control #{cid} has no associated <label for>"
        )


def test_advisory_note_first_child_of_warnings_fragment():
    """answered_with_warnings prepends the advisory <p role=note> as first child."""
    frag = render_answer_fragment(_answered_payload("answered_with_warnings"))
    section = _soup(frag).find("section", class_="answer")
    # First non-heading child after the h2 must be the advisory note.
    children = [c for c in section.find_all(recursive=False)]
    # h2 is first; the advisory paragraph is the next element.
    advisory = section.find("p", class_="advisory")
    assert advisory is not None, "warnings fragment must carry an advisory note"
    assert advisory.get("role") == "note", "advisory must carry role=note"
    # Confirm it precedes the answer body / sources (first <p> in the section).
    first_p = section.find("p")
    assert first_p is advisory, "advisory must be the FIRST paragraph in the fragment"


# --------------------------------------------------------------------------- #
# 3) Manual-pass prep — automatable preconditions from the checklist doc
#    (docs/accessibility/learner-ui-manual-pass.md). What a screen-reader pass
#    cannot be done in CI; these assert the preconditions that ARE expressible.
# --------------------------------------------------------------------------- #


def test_skip_link_is_first_focusable_and_targets_main():
    """The skip link is the FIRST focusable element and points at #main."""
    soup = _soup(_assemble_page())
    # First focusable in DOM order: <a href>, button, input, select, textarea,
    # or anything with a non-negative tabindex.
    focusable = soup.find(
        lambda t: (
            (t.name == "a" and t.get("href"))
            or t.name in ("button", "input", "select", "textarea")
            or (t.get("tabindex") not in (None, "-1") if hasattr(t, "get") else False)
        )
    )
    assert focusable is not None, "page must have a focusable element"
    assert focusable.name == "a" and "skip-link" in (focusable.get("class") or []), (
        f"first focusable must be the skip link, got <{focusable.name}>"
    )
    assert focusable.get("href") == "#main", "skip link must target #main"
    assert soup.find(id="main") is not None, "page must carry an element with id=main"


def test_focus_visible_styles_not_suppressed_in_css():
    """learn.css keeps a real :focus-visible outline (never outline:none alone)."""
    css = (REPO_ROOT / "gui" / "static" / "learn" / "learn.css").read_text(encoding="utf-8")
    # A :focus-visible rule that sets a non-zero outline must exist.
    assert re.search(r":focus-visible[^{]*\{[^}]*outline\s*:\s*[1-9]", css), (
        "learn.css must define a :focus-visible outline (>=1px) — keyboard focus"
    )
    # No bare `outline: none` / `outline: 0` that isn't paired with the
    # :focus:not(:focus-visible) opt-out (the validator's accepted pattern).
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).lower(), m.group(2)
        if re.search(r"outline\s*:\s*(none|0)\b", body):
            assert ":focus:not(:focus-visible)" in selector, (
                f"bare outline:none in selector {selector!r} suppresses focus"
            )


def test_interactive_target_sizes_meet_minimum_in_css():
    """learn.css sizes interactive controls at >= 24px (WCAG 2.5.8)."""
    css = (REPO_ROOT / "gui" / "static" / "learn" / "learn.css").read_text(encoding="utf-8")
    # The button + form controls carry an explicit min-height; assert no rule
    # pins a button/input/select/textarea below 24px (the validator's check).
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).strip().lower(), m.group(2).lower()
        if not re.search(r"\b(button|input|select|textarea|\.btn)\b", selector):
            continue
        for dim in re.finditer(r"(?:min-)?(?:width|height)\s*:\s*(\d+)px", body):
            px = int(dim.group(1))
            # A 0px or a >=24px is fine; anything in (0,24) is a target-size fail.
            assert not (0 < px < 24), (
                f"selector {selector!r} sizes a control at {px}px (< 24px target)"
            )
    # And the canonical interactive controls carry a >= 24px min-height somewhere.
    assert re.search(r"min-height\s*:\s*2(\.\d+)?rem", css) or re.search(
        r"min-height\s*:\s*(2[4-9]|[3-9]\d)px", css
    ), "learn.css must size controls at >= 24px (rem >= 2 or px >= 24)"


def test_live_region_is_outside_the_visual_counter():
    """The visual-only elapsed counter is aria-hidden (no SR per-second chatter)."""
    soup = _soup(_assemble_page())
    elapsed = soup.find(id="elapsed")
    assert elapsed is not None, "page must carry the #elapsed visual counter"
    assert elapsed.get("aria-hidden") == "true", "elapsed counter must be aria-hidden"
    # It must NOT be a live region.
    assert not elapsed.get("aria-live"), "elapsed counter must not be a live region"


# --------------------------------------------------------------------------- #
# 4) Opt-in real-server smoke (fastapi TestClient) — the served page passes the
#    same gate end-to-end. The model-dependent ask round-trip is gated behind a
#    live-Ollama opt-in env (no model in CI).
# --------------------------------------------------------------------------- #


@pytest.mark.real_models
def test_served_learner_page_passes_gate(libv2_root, monkeypatch):
    """Serve the learner app and assert the GET ``/`` (learner-only) page passes
    the same WCAG gate the rendered fragments do — end-to-end through fastapi."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    # Learner-only app: `/` IS the learner page (StaticFiles html=True).
    client = TestClient(create_app(learner_only=True))
    resp = client.get("/")
    assert resp.status_code == 200, resp.text
    _assert_clean("served-learner-index", resp.text)
    # Operator surface must NOT be mounted in learner-only mode.
    assert client.get("/api/settings").status_code == 404


@pytest.mark.real_models
def test_real_ask_roundtrip_renders_resolvable_source(libv2_root, monkeypatch):
    """Opt-in (needs a live local answer backend): an ``answered`` response
    renders >= 1 source link that the viewer then serves 200.

    Skipped unless ``ED4ALL_A11Y_SMOKE_OLLAMA=1`` AND fastapi/the model backend
    are available — this needs a real Ollama-served answer model.
    """
    import os  # noqa: PLC0415

    if os.environ.get("ED4ALL_A11Y_SMOKE_OLLAMA") != "1":
        pytest.skip("requires a live local answer backend (set ED4ALL_A11Y_SMOKE_OLLAMA=1)")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    _make_course(libv2_root, "mini-dart", kind="dart")
    client = TestClient(create_app())
    resp = client.post(
        "/api/learn/ask",
        json={"slug": "mini-dart", "query": "What is a vector store?", "engine": "lexical"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_clean("served-ask-fragment", _assemble_page(body["html"]))
    if body["answer"]["status"] in ("answered", "answered_with_warnings"):
        soup = _soup(body["html"])
        links = soup.select("ol.sources li a")
        assert links, "an answered response must render >= 1 source link"
        href = links[0]["href"]
        viewer = client.get(href)
        assert viewer.status_code == 200, viewer.text
