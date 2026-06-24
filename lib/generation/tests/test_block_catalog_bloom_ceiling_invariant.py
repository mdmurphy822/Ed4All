"""FR-CAT-01 — catalog ``bloom_ceiling`` invariant guard.

The catalog header (``Courseforge/config/block_catalog.yaml``) documents that an
entry's optional ``bloom_ceiling`` "Must be a valid Bloom level >= the entry's
bloom_fit" — but nothing enforced it, so a future edit could silently set a
ceiling BELOW the type's lowest fit level. That would make the IB7.6b
``_apply_bloom_ceilings`` re-route (``lib/generation/block_planner.py``) and the
``bloom_type_range`` gate (``lib/validators/bloom_type_range.py``) misbehave:
the ceiling would be lower than the levels the type is explicitly fit for.

This guard asserts, for EVERY catalog entry carrying both ``bloom_fit`` and
``bloom_ceiling``, that the ceiling's index on the canonical Bloom order
(``lib/ontology/bloom.py::BLOOM_LEVELS`` — remember < understand < apply <
analyze < evaluate < create) is >= the index of the FIRST element of
``bloom_fit``. Catches future drift the same way
``test_framework_block_map_consistency`` catches re-parenting drift.
"""

from __future__ import annotations

from lib.generation.block_catalog import load_block_catalog
from lib.ontology.bloom import BLOOM_LEVELS

_BLOOM_INDEX = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}


def test_bloom_ceiling_at_least_first_bloom_fit_for_every_entry() -> None:
    """No catalog entry may declare a ceiling below its lowest fit level."""
    violations = []
    for entry in load_block_catalog():
        bloom_fit = entry.get("bloom_fit")
        bloom_ceiling = entry.get("bloom_ceiling")
        # Only entries carrying BOTH fields are constrained by the invariant.
        if not bloom_fit or bloom_ceiling is None:
            continue

        block_type = entry.get("block_type")
        first_fit = bloom_fit[0]

        assert first_fit in _BLOOM_INDEX, (
            f"{block_type}: bloom_fit[0]={first_fit!r} is not a valid "
            f"BLOOM_LEVELS member {BLOOM_LEVELS}"
        )
        assert bloom_ceiling in _BLOOM_INDEX, (
            f"{block_type}: bloom_ceiling={bloom_ceiling!r} is not a valid "
            f"BLOOM_LEVELS member {BLOOM_LEVELS}"
        )

        if _BLOOM_INDEX[bloom_ceiling] < _BLOOM_INDEX[first_fit]:
            violations.append(
                f"{block_type}: bloom_ceiling={bloom_ceiling!r} "
                f"(index {_BLOOM_INDEX[bloom_ceiling]}) is BELOW the first "
                f"bloom_fit level {first_fit!r} (index {_BLOOM_INDEX[first_fit]})"
            )

    assert not violations, (
        "block_catalog.yaml entries violate the bloom_ceiling >= bloom_fit[0] "
        "invariant:\n  " + "\n  ".join(violations)
    )


def test_misconception_and_formula_resolve_a_bloom_ceiling() -> None:
    """FR-CAT-01 regression: the two B03 types now carry a ceiling.

    ``misconception`` and ``formula`` lacked ``bloom_ceiling`` (unlike their B03
    siblings ``concept``/``explanation``), so the IB7.6b re-route + the
    ``bloom_type_range`` gate never fired for them. Pin that they resolve a
    valid ceiling so a future deletion is caught.
    """
    by_type = {e.get("block_type"): e for e in load_block_catalog()}
    for block_type in ("misconception", "formula"):
        entry = by_type.get(block_type)
        assert entry is not None, f"{block_type} missing from catalog"
        ceiling = entry.get("bloom_ceiling")
        assert ceiling in _BLOOM_INDEX, (
            f"{block_type}: expected a valid bloom_ceiling, got {ceiling!r}"
        )
