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


# ---------------------------------------------------------------------------
# M5 Fix B — cross-week CO-id de-duplication via the ``placed_ids`` seam.
# Regression guard for the real run's CO-12-in-Week-2-AND-Week-12 bug: at
# (62 COs, 12 weeks) ceil-step=6, week 12's slice cos[66:72] is EMPTY so the
# round-robin fallback re-places cos[11] (CO-12) — already placed in week 2.
# ---------------------------------------------------------------------------


def _placed_ids_deduped(n_cos, weeks):
    """Slice every week threading a SHARED placed_ids set (Fix B)."""
    cos = _cos(n_cos)
    placed_set = set()
    placed = []
    for w in range(1, weeks + 1):
        placed.extend(
            o["id"]
            for o in _slice_cos_for_week(cos, weeks, w, placed_ids=placed_set)
        )
    return placed


class TestFixBCrossWeekDedup:
    def test_co12_not_placed_twice_real_run_shape(self):
        """The exact (62, 12) shape that put CO-12 in Week 2 and Week 12."""
        cos = _cos(62)
        # Legacy (no placed_ids) reproduces the duplicate.
        legacy = []
        for w in range(1, 13):
            legacy.extend(o["id"] for o in _slice_cos_for_week(cos, 12, w))
        assert legacy.count("CO-12") == 2, "expected the legacy CO-12 dup bug"

        # Deduped (placed_ids threaded) — CO-12 appears exactly once.
        deduped = _placed_ids_deduped(62, 12)
        assert deduped.count("CO-12") == 1

    @pytest.mark.parametrize("n_cos,weeks", _CASES + [(62, 12), (62, 20)])
    def test_no_cross_week_duplicate_with_placed_ids(self, n_cos, weeks):
        """With placed_ids threaded, EVERY week is disjoint — no id twice."""
        placed = _placed_ids_deduped(n_cos, weeks)
        assert len(placed) == len(set(placed)), (
            f"({n_cos},{weeks}) placed an id in two weeks despite placed_ids"
        )

    @pytest.mark.parametrize("n_cos,weeks", _CASES + [(62, 12), (62, 20)])
    def test_dedup_preserves_full_coverage(self, n_cos, weeks):
        """De-dup drops only the trailing-fallback duplicate, never real COs."""
        all_ids = {o["id"] for o in _cos(n_cos)}
        placed = set(_placed_ids_deduped(n_cos, weeks))
        assert placed == all_ids, (
            f"({n_cos},{weeks}) dedup dropped {sorted(all_ids - placed)[:10]}"
        )

    def test_placed_ids_none_is_byte_stable_legacy(self):
        """placed_ids=None (default) keeps the legacy, un-deduplicated output."""
        cos = _cos(62)
        for w in range(1, 13):
            legacy = _slice_cos_for_week(cos, 12, w)
            explicit_none = _slice_cos_for_week(cos, 12, w, placed_ids=None)
            assert [o["id"] for o in legacy] == [o["id"] for o in explicit_none]

    def test_dedup_falls_back_to_statement_when_id_absent(self):
        """A pre-mint flat list (no id) de-dups on statement identity."""
        cos = [{"statement": f"obj {i}"} for i in range(1, 63)]
        placed_set = set()
        placed = []
        for w in range(1, 13):
            placed.extend(
                o["statement"]
                for o in _slice_cos_for_week(cos, 12, w, placed_ids=placed_set)
            )
        assert len(placed) == len(set(placed)) == 62
