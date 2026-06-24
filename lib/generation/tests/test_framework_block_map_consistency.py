"""IB2.7 — catalog ⇄ §map drift guard.

A single regression net that the IB2 reconciliation map (the authoritative
§map table in
``plans/finegrain/ib2-catalog-reconciliation-rubric-scaffold-2026-06-22.md``)
is faithfully encoded in ``block_catalog.yaml``. Catches a future catalog edit
silently re-parenting a type — same posture as
``test_catalog_coverage_equals_block_types_exactly``.
"""

from __future__ import annotations

from lib.generation.block_catalog import load_block_catalog

# Hardcoded expected primary B-code per type, copied verbatim from the IB2
# §map table. A catalog edit that diverges from this dict fails the guard.
_EXPECTED_PRIMARY = {
    "objective": "B01",
    "recap": "B02",
    "prereq_set": "B02",
    "concept": "B03",
    "explanation": "B03",
    "example": "B05",
    "formula": "B03",
    "misconception": "B03",
    "table": "B06",
    "acronym": "B06",
    "self_check_question": "B07",
    "assessment_item": "B14",
    "activity": "B08",
    "problem": "B08",
    "checklist": "B08",
    "scenario": "B09",
    "discussion_prompt": "B10",
    "reflection_prompt": "B11",
    "callout": "B12",
    "key_idea": "B12",
    "vocab_card": "B12",
    "flip_card_grid": "B12",
    "summary_takeaway": "B13",
    "chrome": None,
    # IB5 framework-aligned pedagogical block types.
    "hook": "B02",
    "multimedia": "B04",
    "worked_example": "B05",
    "diagram": "B06",
    # IB8/cycle-1: B15 Resources — closes the last canonical-catalog gap.
    "resources": "B15",
}

# Expected secondary B-code (only for the types that straddle two parents).
_EXPECTED_SECONDARY = {
    "example": "B03",
    "formula": "B06",
    "misconception": "B12",
    "table": "B03",
    "acronym": "B12",
    "checklist": "B13",
    "vocab_card": "B03",
    "flip_card_grid": "B03",
    # IB5 framework-aligned pedagogical block types that straddle two parents.
    "worked_example": "B08",
    "diagram": "B03",
}


def test_catalog_primary_matches_expected_map():
    actual = {e["block_type"]: e.get("framework_block") for e in load_block_catalog()}
    assert actual == _EXPECTED_PRIMARY


def test_catalog_secondary_matches_expected_map():
    actual = {
        e["block_type"]: e["framework_block_secondary"]
        for e in load_block_catalog()
        if "framework_block_secondary" in e
    }
    assert actual == _EXPECTED_SECONDARY
