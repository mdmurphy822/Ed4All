"""FR-A11Y-03 / FR-INT-05 — typed-callout render + self-check aria-live.

Covers:

* ``_render_self_check`` emits ``aria-live="polite"`` on the feedback region
  (FR-INT-05) so revealed feedback is announced.
* The callout renderer is BYTE-STABLE when ``ED4ALL_CALLOUT_TYPED`` is unset
  (default off) even when a section carries a ``kind``.
* With the flag on AND a ``kind`` present, the callout gains a redundant
  non-color contract — a visible LABEL + icon + a per-kind border class
  (never color-only).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import _render_content_sections, _render_self_check  # noqa: E402


def test_self_check_feedback_has_aria_live():
    html = _render_self_check(
        [
            {
                "question": "2+2?",
                "options": [
                    {"text": "4", "correct": True, "feedback": "Right."},
                    {"text": "5", "correct": False, "feedback": "Off by one."},
                ],
                "bloom_level": "remember",
            }
        ]
    )
    assert 'class="sc-feedback" aria-live="polite"' in html


_TYPED_SECTION = [
    {
        "heading": "A Section",
        "paragraphs": ["Body."],
        "callout": {
            "type": "callout-warning",
            "heading": "Careful",
            "kind": "warning",
            "items": ["Watch out."],
        },
    }
]


def test_callout_byte_stable_when_flag_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_CALLOUT_TYPED", raising=False)
    html = _render_content_sections(_TYPED_SECTION)
    assert "callout-kind-warning" not in html
    assert "callout-label" not in html
    assert "callout-kind-label" not in html


def test_callout_typed_emits_redundant_coding(monkeypatch):
    monkeypatch.setenv("ED4ALL_CALLOUT_TYPED", "1")
    html = _render_content_sections(_TYPED_SECTION)
    # Border class (color) AND a visible label/icon row (non-color) — redundant.
    assert "callout-kind-warning" in html
    assert "callout-label" in html
    assert "callout-kind-label" in html
    assert "Warning" in html


def test_callout_typed_off_no_kind_unchanged(monkeypatch):
    """Flag on but no kind → legacy markup (no label row, no kind class)."""
    monkeypatch.setenv("ED4ALL_CALLOUT_TYPED", "1")
    sections = [
        {
            "heading": "A Section",
            "paragraphs": ["Body."],
            "callout": {
                "type": "callout-info",
                "heading": "Note",
                "items": ["Be aware."],
            },
        }
    ]
    html = _render_content_sections(sections)
    assert "callout-kind-" not in html
    assert "callout-label" not in html
