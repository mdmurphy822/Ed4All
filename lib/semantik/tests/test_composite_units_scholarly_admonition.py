"""Task #24 — composite-unit planner grammar for the scholarly / admonition profiles.

Pure :func:`plan_units` tests (no HTML, no corpus files), mirroring the
``test_composite_units.py`` idiom. Exercises the two new abstract-role grammars:
``theorem_block`` (a statement + its proof / trailing remarks) and the SINGLETON
``admonition`` (one label + its body). The lexicon PROFILES map concrete labels
(Theorem / Proof / Note / Warning …) onto these abstract roles; the planner only
ever sees the roles, so it stays corpus-agnostic.
"""
from __future__ import annotations

from lib.semantik.composite_units import (
    MIN_UNIT_ITEMS,
    ROLE_ADMONITION,
    ROLE_PROOF,
    ROLE_REMARK,
    ROLE_THEOREM,
    SINGLETON_UNIT_TYPES,
    UNIT_TYPES,
    UnitItem,
    plan_units,
)


def _it(role, boundary=False, members=1, has_heading=True):
    return UnitItem(role=role, boundary=boundary, members=members, has_heading=has_heading)


def _types(spans):
    return [s.unit_type for s in spans]


# ---------------------------------------------------------------------------
# theorem_block: theorem (+ proof|remark)+
# ---------------------------------------------------------------------------


def test_theorem_block_theorem_plus_proof():
    spans = plan_units([_it(ROLE_THEOREM), _it(ROLE_PROOF)])
    assert len(spans) == 1
    s = spans[0]
    assert s.unit_type == "theorem_block" and s.start == 0 and s.end == 2
    assert s.lead_index == 0  # aria-labelledby -> the theorem heading


def test_theorem_block_theorem_proof_remark_chain():
    # theorem + proof + remark (a trailing corollary maps to ``remark``).
    spans = plan_units([_it(ROLE_THEOREM), _it(ROLE_PROOF), _it(ROLE_REMARK)])
    assert _types(spans) == ["theorem_block"]
    assert spans[0].end == 3


def test_theorem_block_theorem_plus_remark_no_proof():
    spans = plan_units([_it(ROLE_THEOREM), _it(ROLE_REMARK)])
    assert _types(spans) == ["theorem_block"]


def test_lone_theorem_not_grouped():
    assert _types(plan_units([_it(ROLE_THEOREM)])) == [None]


def test_theorem_boundary_closes_block():
    # A genuine section heading between theorem and proof splits them.
    spans = plan_units([_it(ROLE_THEOREM), _it(None, boundary=True), _it(ROLE_PROOF)])
    assert _types(spans) == [None, None, None]


def test_theorem_no_gap_jumping():
    # theorem, inert prose, proof: prose is not a member, so no block forms.
    spans = plan_units([_it(ROLE_THEOREM), _it(None), _it(ROLE_PROOF)])
    assert _types(spans) == [None, None, None]


def test_proof_alone_is_not_a_head():
    # A proof with no preceding theorem never heads a block.
    assert _types(plan_units([_it(ROLE_PROOF), _it(ROLE_REMARK)])) == [None, None]


# ---------------------------------------------------------------------------
# admonition: SINGLETON (one label + its body)
# ---------------------------------------------------------------------------


def test_admonition_singleton_with_body():
    # members >= 2 -> the label folded its body block; a complete single-item unit.
    spans = plan_units([_it(ROLE_ADMONITION, members=2)])
    assert len(spans) == 1
    s = spans[0]
    assert s.unit_type == "admonition" and s.start == 0 and s.end == 1
    assert s.lead_index == 0


def test_admonition_bare_label_not_grouped():
    # A bare label (members == 1, no body) is never wrapped.
    assert _types(plan_units([_it(ROLE_ADMONITION, members=1)])) == [None]


def test_two_adjacent_admonitions_stay_separate():
    spans = plan_units(
        [_it(ROLE_ADMONITION, members=2), _it(ROLE_ADMONITION, members=3)]
    )
    assert _types(spans) == ["admonition", "admonition"]
    assert spans[0].end == 1 and spans[1].start == 1 and spans[1].end == 2


def test_admonition_no_heading_uses_label_fallback():
    spans = plan_units([_it(ROLE_ADMONITION, members=2, has_heading=False)])
    assert _types(spans) == ["admonition"]
    assert spans[0].lead_index is None  # aria-label fallback


# ---------------------------------------------------------------------------
# Registration + backward-compat.
# ---------------------------------------------------------------------------


def test_new_unit_types_registered():
    assert "theorem_block" in UNIT_TYPES
    assert "admonition" in UNIT_TYPES


def test_admonition_is_the_sole_singleton_type():
    assert SINGLETON_UNIT_TYPES == frozenset({"admonition"})
    assert MIN_UNIT_ITEMS == 2


def test_existing_grammars_untouched_by_new_roles():
    # An inert-prose stream with none of the new roles plans identically to before.
    assert _types(plan_units([_it(None), _it(None)])) == [None, None]
