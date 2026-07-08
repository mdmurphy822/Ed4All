"""Lane learner-api — WCAG + catalog checks for the new Ask-drawer surfaces.

The Studio drawer (``gui/static/studio/drawer.js``) gains three surfaces in this
lane: the L4 passages-first preview pane, the I6 answer-feedback bar, and the L3
"search all courses" toggle. There is no JS runner in CI, so — mirroring
``test_studio_a11y_gate`` — the markup each surface builds is reconstructed here
in Python (byte-for-byte with ``drawer.js``) and validated with the bs4-only
``WCAGValidator``: zero CRITICAL (Level A) / HIGH (Level AA) findings.

Also asserts the ``ED4ALL_ANSWER_LIBRARY_WIDE`` env-catalog row exists (L3).
"""
from __future__ import annotations

from typing import List

from lib.validators.wcag import IssueSeverity, WCAGValidator

_BLOCKING = {IssueSeverity.CRITICAL, IssueSeverity.HIGH}


def _doc(fragment: str) -> str:
    """Wrap a drawer fragment in a minimal valid page so the validator sees a
    complete document (lang, title, one h1, main landmark) — no page-shell
    false positives, only the fragment's own findings."""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Ask drawer surface</title></head><body>"
        "<main id='main'><h1>Course</h1>"
        f"{fragment}"
        "</main></body></html>"
    )


def _blocking(html: str) -> List:
    report = WCAGValidator().validate(html)
    return [i for i in report.issues if i.severity in _BLOCKING]


# The three surfaces exactly as drawer.js constructs them.
_PASSAGES_PANE = (
    "<div class='ask-passages' role='region' aria-live='polite' "
    "aria-label='Passages found while composing an answer'>"
    "<p class='ask-passages-head muted'>Found these passages — composing an answer…</p>"
    "<ol class='ask-passages-list'>"
    "<li class='ask-passage'><p class='ask-passage-src'>Velocity</p>"
    "<p class='ask-passage-text'>Velocity is a vector.</p></li>"
    "</ol></div>"
)

_FEEDBACK_BAR = (
    "<div class='ask-feedback'>"
    "<p id='ask-fb-l-1' class='ask-feedback-label'>Was this answer helpful?</p>"
    "<div class='ask-feedback-btns' role='group' aria-labelledby='ask-fb-l-1'>"
    "<button type='button' class='ask-feedback-up' "
    "aria-label='Yes, this answer was helpful'>👍 Helpful</button>"
    "<button type='button' class='ask-feedback-down' "
    "aria-label='No, this answer was not helpful'>👎 Not helpful</button>"
    "</div>"
    "<label class='ask-feedback-comment-label' for='ask-fb-c-1'>Add a comment (optional)</label>"
    "<textarea id='ask-fb-c-1' class='ask-feedback-comment' rows='2'></textarea>"
    "</div>"
)

_FEEDBACK_THANKS = (
    "<div class='ask-feedback'>"
    "<p class='ask-feedback-thanks' role='status'>Thanks for your feedback.</p>"
    "</div>"
)

_LIBWIDE_TOGGLE = (
    "<div class='ask-libwide'>"
    "<input type='checkbox' id='ask-libwide-1' class='ask-libwide-input'>"
    "<label class='ask-libwide-label' for='ask-libwide-1'>Search all courses</label>"
    "</div>"
)


def test_passages_pane_zero_aa_findings():
    blocking = _blocking(_doc(_PASSAGES_PANE))
    assert not blocking, [f"{i.criterion} {i.message}" for i in blocking]


def test_feedback_bar_zero_aa_findings():
    blocking = _blocking(_doc(_FEEDBACK_BAR))
    assert not blocking, [f"{i.criterion} {i.message}" for i in blocking]


def test_feedback_thanks_zero_aa_findings():
    blocking = _blocking(_doc(_FEEDBACK_THANKS))
    assert not blocking, [f"{i.criterion} {i.message}" for i in blocking]


def test_libwide_toggle_zero_aa_findings():
    blocking = _blocking(_doc(_LIBWIDE_TOGGLE))
    assert not blocking, [f"{i.criterion} {i.message}" for i in blocking]


def test_feedback_comment_textarea_is_label_paired():
    """The optional comment control carries an associated <label for=…>."""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(_FEEDBACK_BAR, "html.parser")
    ta = soup.find("textarea")
    assert ta is not None and ta.get("id")
    assert soup.find("label", {"for": ta.get("id")}) is not None


def test_libwide_checkbox_is_label_paired():
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(_LIBWIDE_TOGGLE, "html.parser")
    cb = soup.find("input", {"type": "checkbox"})
    assert cb is not None and cb.get("id")
    assert soup.find("label", {"for": cb.get("id")}) is not None


def test_env_catalog_has_library_wide_row():
    from gui import env_catalog  # noqa: PLC0415

    rows = [r for r in env_catalog.CATALOG if r["key"] == "ED4ALL_ANSWER_LIBRARY_WIDE"]
    assert len(rows) == 1, "ED4ALL_ANSWER_LIBRARY_WIDE must have exactly one catalog row"
    row = rows[0]
    assert row["type"] == "boolean"
    assert row["default"] is False
    # Required keys mirror the other answer rows.
    assert {"key", "label", "category", "type", "default", "help"} <= set(row)
