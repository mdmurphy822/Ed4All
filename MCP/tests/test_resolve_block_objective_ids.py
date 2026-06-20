"""Unit tests for the CO-id fan-out fix helper.

``MCP.tools.pipeline_tools._resolve_block_objective_ids`` decides WHICH
objective ids a planned content block is stamped with: the planner's specific
``target_co_ids`` (filtered to the course's valid CO set) when present, else the
week-level TERMINAL objective ids (byte-identical legacy fallback). Slug-agnostic
synthetic inline fixtures only — no course names / device paths.

CPU-pin guidance: ``ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu`` (the
helper itself loads no model, but importing the module is kept device-safe).
"""

from __future__ import annotations

from MCP.tools.pipeline_tools import _resolve_block_objective_ids


_WEEK_TO = ("TO-01",)
_VALID_COS = {"CO-01", "CO-02", "CO-03"}


def test_planner_co_target_stamps_specific_co():
    # A planned block with a valid CO target → stamped with that CO, NOT the TO.
    assert _resolve_block_objective_ids(
        ["CO-02"], _WEEK_TO, _VALID_COS
    ) == ("CO-02",)


def test_empty_target_falls_back_to_week_to():
    # A cross-cutting block (empty target_co_ids) keeps the week-TO fallback.
    assert _resolve_block_objective_ids(
        [], _WEEK_TO, _VALID_COS
    ) == ("TO-01",)


def test_invalid_co_id_falls_back_to_week_to():
    # A hallucinated CO id (not in the valid set) is dropped → week-TO fallback
    # (anti-fabrication: never stamp an id the course doesn't have).
    assert _resolve_block_objective_ids(
        ["CO-99"], _WEEK_TO, _VALID_COS
    ) == ("TO-01",)


def test_multiple_valid_cos_preserved_in_order():
    assert _resolve_block_objective_ids(
        ["CO-03", "CO-01"], _WEEK_TO, _VALID_COS
    ) == ("CO-03", "CO-01")


def test_mixed_valid_and_invalid_keeps_only_valid():
    assert _resolve_block_objective_ids(
        ["CO-99", "CO-02", "CO-77"], _WEEK_TO, _VALID_COS
    ) == ("CO-02",)


def test_duplicate_valid_cos_deduped_order_preserving():
    assert _resolve_block_objective_ids(
        ["CO-02", "CO-02", "CO-01"], _WEEK_TO, _VALID_COS
    ) == ("CO-02", "CO-01")


def test_multi_to_week_fallback_preserved():
    # A week with two TOs and a cross-cutting block keeps BOTH TOs.
    assert _resolve_block_objective_ids(
        [], ("TO-01", "TO-02"), _VALID_COS
    ) == ("TO-01", "TO-02")


def test_non_string_target_ignored():
    # Defensive: non-str entries are ignored without crashing.
    assert _resolve_block_objective_ids(
        [None, 7, "CO-01"], _WEEK_TO, _VALID_COS  # type: ignore[list-item]
    ) == ("CO-01",)
