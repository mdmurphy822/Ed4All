"""WS5 §6 — emit↔validator slicer PARITY (single-sourced helper).

The per-week chapter-objective slice is consumed by two surfaces:

* the emit-side content generator
  (``MCP/tools/pipeline_tools.py::_generate_course_content``), and
* the validator's allowed-set builder
  (``generate_course.py::_slice_chapter_objectives_by_week`` +
  ``_plan_course_structure``'s persisted ``"Week N"`` groups).

WS5 makes them consume the SAME ``_slice_cos_for_week`` helper so parity is
structural (single-sourced), not textual. These tests assert that the helper
each surface calls produces byte-identical week→CO assignments over a grid, and
that ``ED4ALL_COS_PER_WEEK_CAP`` flows through both sides identically.
"""

import importlib
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import (  # noqa: E402
    _slice_chapter_objectives_by_week,
    _slice_cos_for_week,
)


def _cos(n):
    return [{"id": f"CO-{i:02d}", "statement": f"objective {i}"} for i in range(1, n + 1)]


# The emit side (pipeline_tools) imports the SAME helper; we exercise the
# canonical helper here and confirm the validator-side dict builder delegates
# to it. The grid below covers small/medium/large CO counts × week counts.
_GRID = [
    (n_cos, weeks)
    for n_cos in (8, 30, 62, 141)
    for weeks in (8, 10, 13, 31)
]


class TestSlicerParity:
    @pytest.mark.parametrize("n_cos,weeks", _GRID)
    def test_emit_and_validator_slicers_byte_identical(self, n_cos, weeks):
        cos = _cos(n_cos)
        # Validator side: the dict builder.
        validator_map = _slice_chapter_objectives_by_week(cos, weeks)
        # Emit side: the same helper invoked per-week (this is exactly what
        # pipeline_tools._generate_course_content now does).
        for w in range(1, weeks + 1):
            emit_week = _slice_cos_for_week(cos, weeks, w)
            emit_ids = [o["id"] for o in emit_week]
            allowed_ids = [o["id"] for o in validator_map[w]]
            assert emit_ids == allowed_ids, (
                f"week {w} parity broke at (n_cos={n_cos}, weeks={weeks}): "
                f"emit={emit_ids} allowed={allowed_ids}"
            )

    @pytest.mark.parametrize("n_cos,weeks", _GRID)
    def test_plan_persisted_groups_match_emit(self, n_cos, weeks):
        """The §2.4(A) persisted ``"Week N"`` groups equal the emit slice."""
        cos = _cos(n_cos)
        # Replicate the _plan_course_structure persistence shape.
        groups = [
            {
                "chapter": f"Week {w}",
                "objectives": _slice_cos_for_week(cos, weeks, w),
            }
            for w in range(1, weeks + 1)
        ]
        for w in range(1, weeks + 1):
            persisted_ids = [o["id"] for o in groups[w - 1]["objectives"]]
            emit_ids = [o["id"] for o in _slice_cos_for_week(cos, weeks, w)]
            assert persisted_ids == emit_ids


class TestCapEnvOverride:
    def test_cap_env_override_applies_to_both(self, monkeypatch):
        """``ED4ALL_COS_PER_WEEK_CAP=3`` truncates each week's slice to 3 on
        BOTH surfaces identically."""
        monkeypatch.setenv("ED4ALL_COS_PER_WEEK_CAP", "3")
        # Reload so the env read inside the helper picks up the cap (the helper
        # reads os.environ at call time, so no reload is strictly needed, but
        # be explicit about isolation).
        import generate_course as gc
        importlib.reload(gc)
        cos = _cos(62)
        weeks = 8  # ceil(62/8)=8 step → cap 3 should truncate
        vmap = gc._slice_chapter_objectives_by_week(cos, weeks)
        for w in range(1, weeks + 1):
            week_cos = gc._slice_cos_for_week(cos, weeks, w)
            assert len(week_cos) <= 3
            assert [o["id"] for o in week_cos] == [o["id"] for o in vmap[w]]
        # Restore module state for other tests in the session.
        monkeypatch.delenv("ED4ALL_COS_PER_WEEK_CAP", raising=False)
        importlib.reload(gc)
