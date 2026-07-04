"""Wave #22 Tier-2 — composite-unit wrapping at the adapter render seam.

Synthetic block IR only (no corpus files). Exercises the ``dart-unit`` grouping
in ``_render_chapters``: adjacent sibling openers / sections that form one
pedagogical whole are wrapped in a ``<section role="group">`` with an accessible
name, while lone boxes and cross-boundary runs stay ungrouped.
"""
from __future__ import annotations

import re

from lib.semantik.adapter import _AdapterBlock, _AdapterChapter, _render_chapters


def _opener(text, idx, role):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=4,
        raw_text=text, heading_text=text, block_role=role,
    )


def _para(text, idx, html=None):
    return _AdapterBlock(
        html=html if html is not None else f"<p>{text}</p>",
        region_kind="paragraph", raw_block_index=idx, raw_text=text,
        heading_text=None,
    )


def _section(text, idx):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=3,
        raw_text=text, heading_text=text,
    )


def _unit_section(html, unit_type):
    m = re.search(
        rf'<section class="dart-unit dart-unit-{unit_type}"[^>]*>(.*?)</section>\s*$',
        html, re.DOTALL,
    )
    return m


def test_worked_example_unit_wraps_example_solution_practice():
    ch = _AdapterChapter(
        title="Roots",
        blocks=[
            _opener("Example 9.1", 0, "worked_example"),
            _para("Simplify sqrt 36.", 1),
            _opener("Solution", 2, "solution"),
            _para("Since 6^2 = 36.", 3),
            _opener("Try It 9.1", 4, "try_it"),
            _para("Simplify sqrt 49.", 5),
        ],
    )
    html = _render_chapters([ch])
    assert 'data-dart-unit="worked_example"' in html
    assert html.count('data-dart-unit="worked_example"') == 1
    # role=group with an accessible name pointing at the example heading id.
    assert 'role="group"' in html
    assert 'aria-labelledby="example-9-1"' in html
    # The three callout-group boxes are all inside the one unit.
    unit = re.search(
        r'<section class="dart-unit dart-unit-worked_example"[^>]*>(.*)',
        html, re.DOTALL,
    ).group(1)
    assert unit.count("data-dart-opener-group=") == 3


def test_section_opener_unit_objectives_plus_readiness():
    ch = _AdapterChapter(
        title="Sec",
        blocks=[
            _opener("Learning Objectives", 0, "objectives"),
            _para("By the end you will simplify.", 1),
            _opener("Be Prepared 9.1", 2, "readiness_check"),
            _para("Before you begin.", 3),
        ],
    )
    html = _render_chapters([ch])
    assert 'data-dart-unit="section_opener"' in html
    assert 'aria-labelledby="learning-objectives"' in html


def test_lone_example_not_wrapped_in_unit():
    ch = _AdapterChapter(
        title="Lone",
        blocks=[_opener("Example 1.1", 0, "worked_example"), _para("body", 1)],
    )
    html = _render_chapters([ch])
    assert "data-dart-unit=" not in html
    # The callout-group box still renders.
    assert 'data-dart-opener-group="worked_example"' in html


def test_unit_never_crosses_section_heading():
    ch = _AdapterChapter(
        title="Cross",
        blocks=[
            _opener("Example 1.1", 0, "worked_example"),
            _para("ex body", 1),
            _section("9.2 New Section", 2),
            _opener("Solution", 3, "solution"),
            _para("sol body", 4),
        ],
    )
    html = _render_chapters([ch])
    # No worked_example unit forms across the <h3> boundary.
    assert "data-dart-unit=" not in html
    assert "<h3" in html and "New Section" in html


def test_exercise_set_unit_two_exercise_lists():
    xl = '<ol class="dart-exercise-list"><li><ol type="a"><li>$x$</li></ol></li></ol>'
    ch = _AdapterChapter(
        title="Ex",
        blocks=[
            _para("list one", 0, html=xl),
            _para("list two", 1, html=xl),
        ],
    )
    html = _render_chapters([ch])
    assert 'data-dart-unit="exercise_set"' in html
    # No heading on the lead -> aria-label fallback carries the accessible name.
    assert 'aria-label="Exercise set"' in html


def test_group_role_always_has_accessible_name():
    ch = _AdapterChapter(
        title="Named",
        blocks=[
            _opener("Example 2.1", 0, "worked_example"),
            _para("b", 1),
            _opener("Solution", 2, "solution"),
            _para("c", 3),
        ],
    )
    html = _render_chapters([ch])
    for tag in re.findall(r'<section class="dart-unit[^>]*>', html):
        assert "aria-labelledby=" in tag or "aria-label=" in tag
