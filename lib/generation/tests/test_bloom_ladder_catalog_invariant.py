"""Bloom-ladder initiative, WI-10 — block-catalog ``ladder_rungs`` invariant guard.

Mirrors ``test_block_catalog_bloom_ceiling_invariant.py``'s posture: the
catalog header (``Courseforge/config/block_catalog.yaml``) documents that an
entry's optional ``ladder_rungs`` is "the taxonomy's blocks_permitted_at
inverse for this block_type, capped so no listed rung exceeds the entry's own
bloom_ceiling" — but nothing enforced it, so a future hand-edit could smuggle
in a fabricated rung the taxonomy never granted, or a rung above the type's
own ceiling. Either would make the ladder-capping content-planner logic
(ED4ALL_BLOOM_LADDER) offer a block type at a rung it isn't authoritatively
cleared for.

This guard asserts, for EVERY catalog entry carrying ``ladder_rungs``:

1. Every listed rung is a member of the Bloom-ladder taxonomy's
   ``blocks_permitted_at`` inverse for that ``block_type`` (no fabricated
   rungs) — ``schemas/taxonomies/bloom_ladder_blocks.json`` via the
   ``lib/ontology/bloom_ladder.py::blocks_permitted_at`` loader.
2. No listed rung's ordinal (canonical Bloom order,
   ``lib/ontology/bloom.py::BLOOM_LEVELS``) exceeds the entry's own
   ``bloom_ceiling`` ordinal, when a ceiling is declared.

Plus two regression pins: the two entries whose taxonomy-named rung set has
an EMPTY intersection with their own ``bloom_ceiling`` (``misconception``,
``resources``) correctly omit ``ladder_rungs`` rather than smuggling in an
over-ceiling rung; and ``chrome`` (non-pedagogical scaffolding, no
``framework_block``) is correctly never named by the taxonomy at all.
"""

from __future__ import annotations

from typing import Dict, Set

from lib.generation.block_catalog import load_block_catalog
from lib.ontology.bloom import BLOOM_LEVELS
from lib.ontology.bloom_ladder import blocks_permitted_at

_BLOOM_INDEX = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}


def _taxonomy_inverse() -> Dict[str, Set[str]]:
    """block_type -> the set of Bloom levels the ladder taxonomy names it at.

    Built by inverting ``blocks_permitted_at(level)`` over every canonical
    level -- the ladder taxonomy's own source of truth, never re-derived from
    the catalog itself.
    """
    inverse: Dict[str, Set[str]] = {}
    for level in BLOOM_LEVELS:
        for block_type in blocks_permitted_at(level):
            inverse.setdefault(block_type, set()).add(level)
    return inverse


def test_ladder_rungs_consistent_with_taxonomy_inverse() -> None:
    """No catalog entry may list a rung the ladder taxonomy doesn't grant it."""
    inverse = _taxonomy_inverse()
    violations = []
    for entry in load_block_catalog():
        ladder_rungs = entry.get("ladder_rungs")
        if not ladder_rungs:
            continue

        block_type = entry.get("block_type")
        permitted = inverse.get(block_type, set())
        fabricated = set(ladder_rungs) - permitted
        if fabricated:
            violations.append(
                f"{block_type}: ladder_rungs {ladder_rungs!r} names rung(s) "
                f"{sorted(fabricated)} the taxonomy does not permit for it "
                f"(taxonomy permits {sorted(permitted)})"
            )

    assert not violations, (
        "block_catalog.yaml ladder_rungs entries diverge from the "
        "bloom_ladder_blocks.json blocks_permitted_at inverse:\n  "
        + "\n  ".join(violations)
    )


def test_ladder_rungs_never_exceed_bloom_ceiling() -> None:
    """No listed ladder_rung may exceed the entry's own bloom_ceiling."""
    violations = []
    for entry in load_block_catalog():
        ladder_rungs = entry.get("ladder_rungs")
        bloom_ceiling = entry.get("bloom_ceiling")
        if not ladder_rungs or bloom_ceiling is None:
            continue

        block_type = entry.get("block_type")
        assert bloom_ceiling in _BLOOM_INDEX, (
            f"{block_type}: bloom_ceiling={bloom_ceiling!r} is not a valid "
            f"BLOOM_LEVELS member {BLOOM_LEVELS}"
        )
        for rung in ladder_rungs:
            assert rung in _BLOOM_INDEX, (
                f"{block_type}: ladder_rungs entry {rung!r} is not a valid "
                f"BLOOM_LEVELS member {BLOOM_LEVELS}"
            )
            if _BLOOM_INDEX[rung] > _BLOOM_INDEX[bloom_ceiling]:
                violations.append(
                    f"{block_type}: ladder_rung {rung!r} (index "
                    f"{_BLOOM_INDEX[rung]}) exceeds bloom_ceiling "
                    f"{bloom_ceiling!r} (index {_BLOOM_INDEX[bloom_ceiling]})"
                )

    assert not violations, (
        "block_catalog.yaml ladder_rungs entries exceed their own "
        "bloom_ceiling:\n  " + "\n  ".join(violations)
    )


def test_every_taxonomy_named_block_type_exists_in_catalog() -> None:
    """Every block_type the ladder taxonomy names must exist in the catalog
    (with or without ladder_rungs -- see the empty-intersection regression
    below for the two entries that legitimately omit it)."""
    inverse = _taxonomy_inverse()
    catalog_types = {e.get("block_type") for e in load_block_catalog()}
    missing = set(inverse) - catalog_types
    assert not missing, (
        "Bloom-ladder taxonomy names block_type(s) absent from "
        f"block_catalog.yaml: {sorted(missing)}"
    )


def test_misconception_and_resources_omit_ladder_rungs() -> None:
    """Regression: misconception (bloom_ceiling=apply, taxonomy-named at
    analyze) and resources (bloom_ceiling=understand, taxonomy-named at
    create) both have an EMPTY intersection between their taxonomy-named
    rung(s) and their own bloom_ceiling, so they correctly omit
    ladder_rungs rather than smuggling in a rung above their ceiling."""
    by_type = {e.get("block_type"): e for e in load_block_catalog()}
    for block_type in ("misconception", "resources"):
        entry = by_type.get(block_type)
        assert entry is not None, f"{block_type} missing from catalog"
        assert not entry.get("ladder_rungs"), (
            f"{block_type}: expected no ladder_rungs (empty bloom_ceiling "
            f"intersection with the taxonomy-named rung set), got "
            f"{entry.get('ladder_rungs')!r}"
        )


def test_chrome_not_named_by_ladder_taxonomy() -> None:
    """chrome is non-pedagogical scaffolding (framework_block=null); the
    ladder taxonomy correctly never names it, so it carries no ladder_rungs."""
    inverse = _taxonomy_inverse()
    assert "chrome" not in inverse

    by_type = {e.get("block_type"): e for e in load_block_catalog()}
    assert not by_type["chrome"].get("ladder_rungs")
