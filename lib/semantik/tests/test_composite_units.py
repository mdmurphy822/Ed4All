"""Wave #22 Tier-2 — pure composite-unit planner (no HTML, no corpus files)."""
from __future__ import annotations

from lib.semantik.composite_units import (
    MAX_UNIT_BLOCKS,
    ROLE_DEFINITION,
    ROLE_DIRECTIVE,
    ROLE_EXAMPLE,
    ROLE_EXERCISE_LIST,
    ROLE_FIGURE,
    ROLE_OBJECTIVES,
    ROLE_PRACTICE,
    ROLE_PROCEDURE,
    ROLE_READINESS,
    ROLE_SOLUTION,
    ROLE_STEP,
    UnitItem,
    plan_units,
)


def _it(role, boundary=False, members=1, has_heading=True):
    return UnitItem(role=role, boundary=boundary, members=members, has_heading=has_heading)


def _types(spans):
    return [s.unit_type for s in spans]


# ---------------------------------------------------------------------------
# worked_example
# ---------------------------------------------------------------------------


def test_worked_example_example_plus_solution():
    spans = plan_units([_it(ROLE_EXAMPLE), _it(ROLE_SOLUTION)])
    assert len(spans) == 1
    s = spans[0]
    assert s.unit_type == "worked_example" and s.start == 0 and s.end == 2
    assert s.lead_index == 0


def test_worked_example_example_solution_practice_chain():
    spans = plan_units(
        [_it(ROLE_EXAMPLE), _it(ROLE_SOLUTION), _it(ROLE_PRACTICE), _it(ROLE_PRACTICE)]
    )
    assert _types(spans) == ["worked_example"]
    assert spans[0].end == 4


def test_lone_example_is_not_grouped():
    # A lone example box stays ungrouped (>= 2 members guard).
    spans = plan_units([_it(ROLE_EXAMPLE)])
    assert _types(spans) == [None]


def test_example_then_unrelated_prose_not_grouped():
    spans = plan_units([_it(ROLE_EXAMPLE), _it(None)])
    assert _types(spans) == [None, None]


# ---------------------------------------------------------------------------
# section_opener
# ---------------------------------------------------------------------------


def test_section_opener_objectives_plus_readiness():
    spans = plan_units([_it(ROLE_OBJECTIVES), _it(ROLE_READINESS), _it(ROLE_READINESS)])
    assert _types(spans) == ["section_opener"]
    assert spans[0].end == 3


def test_lone_objectives_not_grouped():
    assert _types(plan_units([_it(ROLE_OBJECTIVES)])) == [None]


# ---------------------------------------------------------------------------
# procedure / exercise_set / definition_group / figure_group
# ---------------------------------------------------------------------------


def test_procedure_with_steps():
    spans = plan_units([_it(ROLE_PROCEDURE), _it(ROLE_STEP), _it(ROLE_EXERCISE_LIST)])
    assert _types(spans) == ["procedure"]
    assert spans[0].end == 3


def test_exercise_set_two_lists():
    spans = plan_units([_it(ROLE_EXERCISE_LIST, has_heading=False),
                        _it(ROLE_EXERCISE_LIST, has_heading=False)])
    assert _types(spans) == ["exercise_set"]
    assert spans[0].lead_index is None  # no heading -> aria-label fallback


def test_exercise_set_directive_plus_list():
    spans = plan_units([_it(ROLE_DIRECTIVE, has_heading=False),
                        _it(ROLE_EXERCISE_LIST, has_heading=False)])
    assert _types(spans) == ["exercise_set"]


def test_directive_without_list_not_grouped():
    spans = plan_units([_it(ROLE_DIRECTIVE, has_heading=False), _it(None)])
    assert _types(spans) == [None, None]


def test_definition_group_dl_plus_example():
    spans = plan_units([_it(ROLE_DEFINITION, has_heading=False), _it(ROLE_EXAMPLE)])
    assert _types(spans) == ["definition_group"]
    assert spans[0].lead_index == 1  # the example has the heading


def test_definition_group_does_not_split_worked_example():
    # dl + example + solution: the example leads a worked example, so the dl is
    # left standalone and the worked example forms instead (conservative).
    spans = plan_units(
        [_it(ROLE_DEFINITION, has_heading=False), _it(ROLE_EXAMPLE), _it(ROLE_SOLUTION)]
    )
    assert _types(spans) == [None, "worked_example"]


def test_figure_group_two_figures():
    spans = plan_units([_it(ROLE_FIGURE, has_heading=False),
                        _it(ROLE_FIGURE, has_heading=False)])
    assert _types(spans) == ["figure_group"]


# ---------------------------------------------------------------------------
# Guards: boundary, adjacency, cap.
# ---------------------------------------------------------------------------


def test_boundary_closes_unit():
    # A genuine heading between example and solution splits them.
    spans = plan_units([_it(ROLE_EXAMPLE), _it(None, boundary=True), _it(ROLE_SOLUTION)])
    assert _types(spans) == [None, None, None]


def test_no_gap_jumping():
    # example, prose, solution: the prose is not a member, so no unit forms.
    spans = plan_units([_it(ROLE_EXAMPLE), _it(None), _it(ROLE_SOLUTION)])
    assert _types(spans) == [None, None, None]


def test_block_cap_stops_extension():
    # example(6) + solution(6) = 12 ok; a 3rd practice(1) would exceed 12.
    spans = plan_units(
        [_it(ROLE_EXAMPLE, members=6), _it(ROLE_SOLUTION, members=6), _it(ROLE_PRACTICE)]
    )
    assert spans[0].unit_type == "worked_example"
    assert spans[0].end == 2  # third item excluded by the cap
    assert _types(spans) == ["worked_example", None]


def test_cap_constant():
    assert MAX_UNIT_BLOCKS == 12


def test_empty_input():
    assert plan_units([]) == []
