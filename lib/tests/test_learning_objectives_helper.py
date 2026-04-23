"""Tests for lib.ontology.learning_objectives (Wave 24).

Covers the canonical LO-ID mint / validate / hierarchy helpers:
  * mint_lo_id round-trips with validate_lo_id for both hierarchies.
  * Invalid IDs fail validation.
  * Counter edge cases (1, 99, 100, 0, negative).
  * hierarchy_from_id returns correct labels + raises on junk.
  * split_terminal_chapter honors ratio + min/max bounds.
  * assign_lo_ids produces correct ordering and count.
"""

from __future__ import annotations

import pytest

from lib.ontology.learning_objectives import (
    LO_ID_PATTERN,
    assign_lo_ids,
    hierarchy_from_id,
    mint_lo_id,
    split_terminal_chapter,
    validate_lo_id,
)

# ---------------------------------------------------------------------------
# mint_lo_id
# ---------------------------------------------------------------------------


def test_mint_terminal_basic():
    assert mint_lo_id("terminal", 1) == "TO-01"
    assert mint_lo_id("terminal", 2) == "TO-02"
    assert mint_lo_id("terminal", 42) == "TO-42"


def test_mint_chapter_basic():
    assert mint_lo_id("chapter", 1) == "CO-01"
    assert mint_lo_id("chapter", 99) == "CO-99"


def test_mint_handles_three_digit_counters():
    # Should expand to 3 digits naturally.
    assert mint_lo_id("terminal", 100) == "TO-100"
    assert mint_lo_id("chapter", 123) == "CO-123"


def test_mint_rejects_bad_hierarchy():
    with pytest.raises(ValueError, match="hierarchy"):
        mint_lo_id("module", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hierarchy"):
        mint_lo_id("", 1)  # type: ignore[arg-type]


def test_mint_rejects_non_positive_counter():
    with pytest.raises(ValueError, match="counter"):
        mint_lo_id("terminal", 0)
    with pytest.raises(ValueError, match="counter"):
        mint_lo_id("chapter", -5)


# ---------------------------------------------------------------------------
# validate_lo_id + LO_ID_PATTERN
# ---------------------------------------------------------------------------


def test_validate_accepts_canonical_ids():
    assert validate_lo_id("TO-01")
    assert validate_lo_id("CO-42")
    # Longer prefixes are allowed (e.g., BIO-01 for domain-scoped variants).
    assert validate_lo_id("BIO-01")
    # Three-digit counters valid.
    assert validate_lo_id("TO-100")


def test_validate_rejects_drift_shapes():
    # Pre-Wave-24 phantom shape.
    assert not validate_lo_id("PHYS_101_OBJ_1")
    # Too few digits.
    assert not validate_lo_id("TO-1")
    # Single-letter prefix.
    assert not validate_lo_id("T-01")
    # Lowercase prefix.
    assert not validate_lo_id("to-01")
    # Whitespace padded.
    assert not validate_lo_id(" TO-01")
    # Empty.
    assert not validate_lo_id("")
    # Non-string.
    assert not validate_lo_id(None)  # type: ignore[arg-type]
    assert not validate_lo_id(123)  # type: ignore[arg-type]


def test_lo_id_pattern_equivalence_with_schema():
    # Guard: keep the pattern byte-identical with the courseforge schema.
    # (schema pattern: ``^[A-Z]{2,}-\d{2,}$``)
    assert LO_ID_PATTERN.pattern == r"^[A-Z]{2,}-\d{2,}$"


# ---------------------------------------------------------------------------
# hierarchy_from_id
# ---------------------------------------------------------------------------


def test_hierarchy_from_id_round_trip():
    for counter in (1, 2, 5, 42, 99):
        to_id = mint_lo_id("terminal", counter)
        assert hierarchy_from_id(to_id) == "terminal"
        co_id = mint_lo_id("chapter", counter)
        assert hierarchy_from_id(co_id) == "chapter"


def test_hierarchy_from_id_rejects_unknown_prefix():
    # Valid pattern but not a known hierarchy prefix.
    with pytest.raises(ValueError, match="prefix"):
        hierarchy_from_id("BIO-01")
    with pytest.raises(ValueError, match="prefix"):
        hierarchy_from_id("XX-99")


def test_hierarchy_from_id_rejects_invalid():
    with pytest.raises(ValueError, match="canonical"):
        hierarchy_from_id("PHYS_101_OBJ_1")
    with pytest.raises(ValueError, match="canonical"):
        hierarchy_from_id("to-01")


# ---------------------------------------------------------------------------
# split_terminal_chapter
# ---------------------------------------------------------------------------


def test_split_basic_ratio():
    # 20 objectives, 0.25 ratio → 5, but max_terminal=6 clamps to 5.
    terminal, chapter = split_terminal_chapter(20)
    assert terminal == 5
    assert chapter == 15
    assert terminal + chapter == 20


def test_split_respects_min_terminal():
    # 4 objectives, ratio 0.25 → 1 raw, clamped up to min_terminal=2.
    terminal, chapter = split_terminal_chapter(4)
    assert terminal == 2
    assert chapter == 2


def test_split_respects_max_terminal():
    # 100 objectives, ratio 0.25 → 25, clamped down to max_terminal=6.
    terminal, chapter = split_terminal_chapter(100)
    assert terminal == 6
    assert chapter == 94


def test_split_handles_small_totals():
    # Total < min_terminal → terminal clamped to total, chapter = 0.
    terminal, chapter = split_terminal_chapter(1)
    assert terminal == 1
    assert chapter == 0


def test_split_zero_or_negative():
    assert split_terminal_chapter(0) == (0, 0)
    assert split_terminal_chapter(-3) == (0, 0)


def test_split_configurable_bounds():
    # Override bounds to emulate the legacy max_terminal=2 policy.
    terminal, chapter = split_terminal_chapter(10, max_terminal=2)
    assert terminal == 2
    assert chapter == 8


# ---------------------------------------------------------------------------
# assign_lo_ids
# ---------------------------------------------------------------------------


def test_assign_lo_ids_empty():
    assert assign_lo_ids(0) == []
    assert assign_lo_ids(-1) == []


def test_assign_lo_ids_default_split():
    ids = assign_lo_ids(10)
    assert len(ids) == 10
    # Terminal outcomes emit first.
    terminals = [x for x in ids if x[1] == "terminal"]
    chapters = [x for x in ids if x[1] == "chapter"]
    assert len(terminals) + len(chapters) == 10
    # All IDs are canonical.
    for lo_id, _ in ids:
        assert validate_lo_id(lo_id)


def test_assign_lo_ids_ordering_contract():
    ids = assign_lo_ids(8)
    # Terminal block comes first, chapter block after.
    seen_chapter = False
    for lo_id, level in ids:
        if level == "chapter":
            seen_chapter = True
        elif level == "terminal":
            # Once we've seen a CO, we should not see another TO.
            assert not seen_chapter, (
                f"Terminal {lo_id} emitted after chapter entry"
            )
