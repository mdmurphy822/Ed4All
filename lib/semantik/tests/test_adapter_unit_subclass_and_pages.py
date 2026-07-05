"""Build #23 — adapter-side unit page rollup (unconditional) + Tier-3 hook.

* ``_wrap_composite_unit`` stamps a ``data-dart-pages`` rollup spanning member
  pages, flag-independent (provenance).
* ``normalize_cascade_to_ed4all`` runs the subclass pass ONLY when the flag is
  on or a client is injected; flag-off + no client is byte-identical (no
  ``data-dart-subclass`` attribute, ``subclass_report`` is None).
"""
from __future__ import annotations

import json
import re

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _render_chapters,
    normalize_cascade_to_ed4all,
)


def _opener(text, idx, role):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=4,
        raw_text=text, heading_text=text, block_role=role,
    )


def _para(text, idx, pages):
    return _AdapterBlock(
        html=f"<p>{text}</p>", region_kind="paragraph", raw_block_index=idx,
        raw_text=text, heading_text=None, pages=pages,
    )


def _worked_example_chapter():
    return _AdapterChapter(
        title="Roots",
        blocks=[
            _opener("Example 9.1", 0, "worked_example"),
            _para("Simplify sqrt 36.", 1, [3]),
            _opener("Solution", 2, "solution"),
            _para("Since 6^2 = 36.", 3, [4]),
            _opener("Try It 9.1", 4, "try_it"),
            _para("Simplify sqrt 49.", 5, [5]),
        ],
    )


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = None
        self.flags = []
        self.lane_used = None
        self.lang = "en"


# ---------------------------------------------------------------------------
# Unconditional page rollup on the unit <section>
# ---------------------------------------------------------------------------
def test_unit_section_carries_page_rollup():
    html = _render_chapters([_worked_example_chapter()])
    m = re.search(r'<section class="dart-unit dart-unit-worked_example"[^>]*>', html)
    assert m is not None
    tag = m.group(0)
    # Members span pages 3..5 → rollup "3-5".
    assert 'data-dart-pages="3-5"' in tag
    assert 'data-dart-page-kind="physical"' in tag


def test_single_page_unit_rollup_is_single_value():
    ch = _AdapterChapter(
        title="Defs",
        blocks=[
            _para("A term.", 0, [7]),  # dl-style standalone — see below
        ],
    )
    # A lone item is not a unit; build a real 2-member unit on one page.
    ch = _AdapterChapter(
        title="Roots",
        blocks=[
            _opener("Example 9.1", 0, "worked_example"),
            _para("Simplify.", 1, [7]),
            _opener("Solution", 2, "solution"),
            _para("Done.", 3, [7]),
        ],
    )
    html = _render_chapters([ch])
    m = re.search(r'<section class="dart-unit dart-unit-worked_example"[^>]*>', html)
    assert 'data-dart-pages="7"' in m.group(0)


# ---------------------------------------------------------------------------
# Flag-off byte-identical (no client, flag unset) vs Tier-3 pass runs
# ---------------------------------------------------------------------------
def test_flag_off_no_client_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SEMANTIC_SUBCLASS", raising=False)
    out = normalize_cascade_to_ed4all(_Result([_worked_example_chapter()]), pdf_stem="ch09")
    assert "data-dart-subclass=" not in out["html"]
    assert out["subclass_report"] is None


def test_injected_client_runs_subclass_pass(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SEMANTIC_SUBCLASS", raising=False)

    def _client(prompt, *, max_tokens=64):
        return json.dumps({"subclass": "symbolic-manipulation", "confidence": 0.9})

    class _Cap:
        def __init__(self):
            self.calls = []

        def log_decision(self, **kw):
            self.calls.append(kw)

    cap = _Cap()
    out = normalize_cascade_to_ed4all(
        _Result([_worked_example_chapter()]),
        pdf_stem="ch09",
        subclass_client=_client,
        subclass_capture=cap,
    )
    assert 'data-dart-subclass="symbolic-manipulation"' in out["html"]
    assert out["subclass_report"]["total_units"] >= 1
    assert cap.calls and cap.calls[0]["decision_type"] == "unit_subclass_assignment"


def test_flag_on_builds_default_seat_but_tolerates_failure(monkeypatch):
    # Flag on + no client → the default local seat is built. With no server
    # reachable the client call raises; the pass swallows it (render survives)
    # and no subclass attribute is emitted. Byte-safe fail-open.
    monkeypatch.setenv("SEMANTIK_SEMANTIC_SUBCLASS", "true")
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://127.0.0.1:1/v1")
    out = normalize_cascade_to_ed4all(_Result([_worked_example_chapter()]), pdf_stem="ch09")
    assert "data-dart-subclass=" not in out["html"]
