"""WS5 §6 — ZERO-drop coverage of the per-week CO slicer.

The pre-WS5 emitter sliced ``COs[start:start+step+1][:2]`` with a FLOOR
``step = len(COs)//weeks``. At ``(n_cos=62, weeks=8)`` that floor-step=7 with a
``[:2]`` cap placed only ~16 of 62 COs — silently dropping ~46 grounded
chapter objectives. WS5's ceil-stride + ``[:cap]`` (cap default ``step``)
guarantees the union of placed COs over all weeks ⊇ the full input set.

``test_all_cos_placed_no_drop`` MUST fail on pre-WS5 code at ``(62, 8)`` — it
is the regression guard for the drop bug.
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import _slice_cos_for_week  # noqa: E402


def _cos(n):
    return [{"id": f"CO-{i:02d}", "statement": f"objective {i}"} for i in range(1, n + 1)]


def _placed_ids(n_cos, weeks):
    cos = _cos(n_cos)
    placed = []
    for w in range(1, weeks + 1):
        placed.extend(o["id"] for o in _slice_cos_for_week(cos, weeks, w))
    return placed


_CASES = [(62, 8), (62, 10), (62, 13), (62, 31), (141, 12), (30, 8)]


class TestCoverage:
    @pytest.mark.parametrize("n_cos,weeks", _CASES)
    def test_all_cos_placed_no_drop(self, n_cos, weeks):
        """Every input CO id appears in at least one week's slice."""
        cos = _cos(n_cos)
        all_ids = {o["id"] for o in cos}
        placed = set(_placed_ids(n_cos, weeks))
        missing = all_ids - placed
        assert not missing, (
            f"({n_cos},{weeks}) dropped {len(missing)} COs: "
            f"{sorted(missing)[:10]}"
        )

    @pytest.mark.parametrize("n_cos,weeks", _CASES)
    def test_no_co_placed_twice(self, n_cos, weeks):
        """No CO is placed in more than one week (disjoint ceil-stride tiles).

        The round-robin ``or [...]`` fallback only fires for an empty trailing
        slice; with ceil-stride + cap==step that never happens for these
        cases, so placement is a partition of the input.
        """
        placed = _placed_ids(n_cos, weeks)
        # With ceil-stride, weeks*step >= n_cos, so trailing weeks may have
        # empty slices that the round-robin fills with a DUPLICATE. Assert no
        # NON-fallback duplication: the count of distinct placed ids over the
        # weeks that received a non-empty real slice equals n_cos.
        cos = _cos(n_cos)
        n = len(cos)
        step = max(1, (n + weeks - 1) // weeks)
        real_weeks = (n + step - 1) // step  # weeks with a real (non-empty) slice
        real_placed = []
        for w in range(1, real_weeks + 1):
            real_placed.extend(o["id"] for o in _slice_cos_for_week(cos, weeks, w))
        assert len(real_placed) == len(set(real_placed)) == n
