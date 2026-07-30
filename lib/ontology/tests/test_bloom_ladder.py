"""Bloom-ladder initiative (WI-01) — bloom_ladder_blocks loader tests.

Confirms ``lib/ontology/bloom_ladder.py`` loads
``schemas/taxonomies/bloom_ladder_blocks.json`` correctly: six rungs,
ordinal-ordered ladder capping, per-rung ``block_types`` drawn only from
the canonical Courseforge ``BLOCK_TYPES`` palette, and per-rung
misconception-card probe guidance pinned verbatim to the existing
``misconception-card`` / ``misconception-claim`` / ``misconception-correction``
class-token contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Courseforge.scripts.blocks import BLOCK_TYPES
from lib.ontology.bloom import BLOOM_LEVELS
from lib.ontology.bloom_ladder import (
    PROBE_STRATEGIES,
    blocks_permitted_at,
    load_bloom_ladder,
    misconception_probe_for,
    rungs_up_to,
)

_FIXTURES_DIR = (
    Path(__file__).resolve().parents[3]
    / "schemas"
    / "tests"
    / "fixtures"
    / "bloom_ladder"
)


# ---------------------------------------------------------------------------
# load_bloom_ladder
# ---------------------------------------------------------------------------


def test_load_bloom_ladder_has_all_six_levels():
    raw = load_bloom_ladder()
    assert set(raw) == set(BLOOM_LEVELS)


def test_load_bloom_ladder_is_cached_singleton():
    assert load_bloom_ladder() is load_bloom_ladder()


def test_every_rung_block_type_is_canonical():
    """No rung may declare a BLOCK_TYPES token outside the 30-member palette."""
    raw = load_bloom_ladder()
    for level, rung in raw.items():
        for bt in rung["block_types"]:
            assert bt in BLOCK_TYPES, f"{level} declares unknown block_type {bt!r}"


def test_ordinals_match_bloom_level_order():
    raw = load_bloom_ladder()
    for idx, level in enumerate(BLOOM_LEVELS):
        assert raw[level]["ordinal"] == idx + 1


# ---------------------------------------------------------------------------
# rungs_up_to
# ---------------------------------------------------------------------------


def test_rungs_up_to_remember_returns_only_remember():
    rungs = rungs_up_to("remember")
    assert [r.level for r in rungs] == ["remember"]


def test_rungs_up_to_create_returns_all_six_in_order():
    rungs = rungs_up_to("create")
    assert [r.level for r in rungs] == list(BLOOM_LEVELS)
    assert [r.ordinal for r in rungs] == [1, 2, 3, 4, 5, 6]


def test_rungs_up_to_is_the_ladder_cumulative_prefix():
    """rungs_up_to(ceiling) is exactly the ordinal-ascending prefix ending at ceiling."""
    for idx, ceiling in enumerate(BLOOM_LEVELS):
        rungs = rungs_up_to(ceiling)
        assert [r.level for r in rungs] == list(BLOOM_LEVELS[: idx + 1])


def test_rungs_up_to_never_admits_a_rung_above_ceiling():
    rungs = rungs_up_to("apply")
    levels = {r.level for r in rungs}
    assert "analyze" not in levels
    assert "evaluate" not in levels
    assert "create" not in levels


def test_rungs_up_to_unknown_level_raises():
    with pytest.raises(ValueError, match="Unrecognized Bloom level"):
        rungs_up_to("bogus_level")


# ---------------------------------------------------------------------------
# blocks_permitted_at
# ---------------------------------------------------------------------------


def test_blocks_permitted_at_every_level_nonempty_subset_of_canonical():
    for level in BLOOM_LEVELS:
        blocks = blocks_permitted_at(level)
        assert isinstance(blocks, frozenset)
        assert blocks
        assert blocks <= BLOCK_TYPES


def test_blocks_permitted_at_unknown_level_raises():
    with pytest.raises(ValueError, match="Unrecognized Bloom level"):
        blocks_permitted_at("not_a_level")


def test_blocks_permitted_at_matches_rungs_up_to_own_rung():
    for level in BLOOM_LEVELS:
        own_rung = rungs_up_to(level)[-1]
        assert own_rung.level == level
        assert own_rung.block_types == blocks_permitted_at(level)


# ---------------------------------------------------------------------------
# misconception_probe_for
# ---------------------------------------------------------------------------


def test_misconception_probe_for_every_level_pins_zero_touch_class_tokens():
    for level in BLOOM_LEVELS:
        probe = misconception_probe_for(level)
        assert probe["card_class"] == "misconception-card"
        assert probe["claim_class"] == "misconception-claim"
        assert probe["correction_class"] == "misconception-correction"
        assert probe["probe_strategy"] in PROBE_STRATEGIES


def test_misconception_probe_strategies_are_distinct_per_rung():
    """Six rungs, six canonical strategies -- each rung gets its own."""
    strategies = [misconception_probe_for(level)["probe_strategy"] for level in BLOOM_LEVELS]
    assert len(set(strategies)) == len(BLOOM_LEVELS)
    assert set(strategies) == set(PROBE_STRATEGIES)


def test_misconception_probe_for_returns_defensive_copy():
    probe = misconception_probe_for("remember")
    probe["probe_strategy"] = "mutated"
    assert misconception_probe_for("remember")["probe_strategy"] != "mutated"


def test_misconception_probe_for_unknown_level_raises():
    with pytest.raises(ValueError, match="Unrecognized Bloom level"):
        misconception_probe_for("nope")


# ---------------------------------------------------------------------------
# Wave-namespaced fixtures — the "ladder capped at the objective's own
# bloom_level" design decision.
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict:
    path = _FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_accept_rung_at_own_ceiling():
    fixture = _load_fixture("bloom_ladder_accept_remember_ceiling.json")
    assert fixture["expect"] == "accept"
    admissible_levels = {r.level for r in rungs_up_to(fixture["objective_bloom_level"])}
    assert fixture["requested_rung"] in admissible_levels


def test_fixture_reject_rung_above_ceiling():
    fixture = _load_fixture("bloom_ladder_reject_rung_above_ceiling.json")
    assert fixture["expect"] == "reject"
    admissible_levels = {r.level for r in rungs_up_to(fixture["objective_bloom_level"])}
    assert fixture["requested_rung"] not in admissible_levels
