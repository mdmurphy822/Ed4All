"""Param-transport regression for the rewrite-eviction token params.

``WorkflowRunner._route_params`` comma-joins list workflow params into a single
STRING when it builds phase-task params (so tools like the SemantiK converter
keep receiving ``pdf_paths`` as a comma-joined string). The three
rewrite-eviction token params ride the same wire, so they reach the
``_run_content_generation_rewrite`` consumer as a comma-joined STRING, not the
original list:

- ``target_block_ids`` (stage-1, block TYPES)
- ``target_block_instance_ids`` (I4 stage-2, exact block-instance ids)
- ``target_page_ids`` (I4 stage-2, page/module ids)

The pre-fix consumer built its target set with ``{str(_t) for _t in value}``,
which iterates a STRING into CHARACTERS — exploding one block id into
``['#', '-', '0', '1', ...]`` and then failing the loud unknown-token guard.
The fix is a shared consumer-side normalizer,
:func:`MCP.tools.pipeline_tools._normalize_token_list_param`, that SPLITS on
',' instead of iterating. These tests pin (a) the normalizer, and (b) the
consumer seam (normalizer → real eviction helper) with the string wire form for
each of the three params.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _evict_rewrite_cache_by_block_id,
    _evict_rewrite_cache_by_block_type,
    _evict_rewrite_cache_by_page,
    _normalize_token_list_param,
)


def _blk(block_id: str, block_type: str, page_id: str) -> SimpleNamespace:
    """Minimal duck-typed stand-in for a Courseforge Block."""
    return SimpleNamespace(
        block_id=block_id, block_type=block_type, page_id=page_id
    )


def _fixture():
    """Two weeks, mixed block types on distinct pages (real-looking ids)."""
    blocks = [
        _blk("week_08_content_01#example_derivative_03", "example",
             "week_08_content_01"),
        _blk("week_08_content_01#objective_intro_0", "objective",
             "week_08_content_01"),
        _blk("week_08_content_02#concept_chain_rule_0", "concept",
             "week_08_content_02"),
        _blk("week_09_content_01#objective_x_0", "objective",
             "week_09_content_01"),
        _blk("week_09_content_01#concept_y_0", "concept",
             "week_09_content_01"),
    ]
    cache = {b.block_id: {"block_id": b.block_id, "content": "<p>x</p>"}
             for b in blocks}
    return blocks, cache


# --------------------------------------------------------------------------- #
# _normalize_token_list_param — the shared consumer-side re-splitter
# --------------------------------------------------------------------------- #


class TestNormalizeTokenListParam:
    def test_none_is_empty(self):
        assert _normalize_token_list_param(None) == []

    def test_empty_string_is_empty(self):
        assert _normalize_token_list_param("") == []
        assert _normalize_token_list_param("   ") == []

    def test_single_token_string(self):
        # The exact wire shape that used to explode into characters.
        assert _normalize_token_list_param(
            "week_08_content_01#example_derivative_03"
        ) == ["week_08_content_01#example_derivative_03"]

    def test_comma_joined_string_splits_on_comma(self):
        assert _normalize_token_list_param("objective,concept") == [
            "objective", "concept",
        ]

    def test_tokens_with_spaces_are_stripped(self):
        assert _normalize_token_list_param(" objective , concept ") == [
            "objective", "concept",
        ]

    def test_trailing_and_empty_tokens_dropped(self):
        assert _normalize_token_list_param("objective,,concept,") == [
            "objective", "concept",
        ]

    def test_list_passthrough_stringifies_and_strips(self):
        assert _normalize_token_list_param([" a ", "b", ""]) == ["a", "b"]

    def test_tuple_and_set_forms(self):
        assert _normalize_token_list_param(("a", "b")) == ["a", "b"]
        assert sorted(_normalize_token_list_param({"a", "b"})) == ["a", "b"]

    def test_list_drops_none_members(self):
        assert _normalize_token_list_param(["a", None, "b"]) == ["a", "b"]

    def test_scalar_non_string_coerced_to_single_token(self):
        assert _normalize_token_list_param(7) == ["7"]

    def test_string_is_never_iterated_into_characters(self):
        # The bug signature: a bare id must NOT become ['#', '-', '0', ...].
        out = _normalize_token_list_param("a#b_0")
        assert out == ["a#b_0"]
        assert "#" not in out


# --------------------------------------------------------------------------- #
# Consumer seam — normalizer → real eviction helper, STRING wire form.
# Mirrors exactly what _run_content_generation_rewrite does after the fix.
# --------------------------------------------------------------------------- #


class TestBlockInstanceIdStringTransport:
    def test_single_id_string_resolves_to_id_not_chars(self):
        blocks, cache = _fixture()
        wire = "week_08_content_01#example_derivative_03"  # comma-joined-of-one
        targets = set(_normalize_token_list_param(wire))
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, targets,
        )
        assert unknown == []  # NOT ['#', '-', '0', ...]
        assert removed == 1
        assert "week_08_content_01#example_derivative_03" not in cache
        # Sibling of the same type survives.
        assert "week_09_content_01#objective_x_0" in cache

    def test_two_id_string_resolves_both(self):
        blocks, cache = _fixture()
        wire = (
            "week_08_content_01#example_derivative_03,"
            "week_09_content_01#concept_y_0"
        )
        targets = set(_normalize_token_list_param(wire))
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, targets,
        )
        assert unknown == []
        assert removed == 2
        assert "week_08_content_01#example_derivative_03" not in cache
        assert "week_09_content_01#concept_y_0" not in cache

    def test_unknown_id_string_still_fails_loud(self):
        blocks, cache = _fixture()
        targets = set(_normalize_token_list_param("week_99_ghost#objective_0"))
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, targets,
        )
        assert removed == 0
        assert unknown == ["week_99_ghost#objective_0"]
        assert len(cache) == 5  # loud path never mutates on unknown


class TestPageIdStringTransport:
    def test_single_page_string_resolves(self):
        blocks, cache = _fixture()
        targets = set(_normalize_token_list_param("week_08_content_01"))
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, targets,
        )
        assert unknown == []
        assert removed == 2  # example + objective on that page
        assert "week_08_content_01#example_derivative_03" not in cache
        assert "week_08_content_01#objective_intro_0" not in cache

    def test_module_prefix_string_resolves_whole_week(self):
        blocks, cache = _fixture()
        targets = set(_normalize_token_list_param("week_08"))
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, targets,
        )
        assert unknown == []
        assert removed == 3  # all week_08 blocks
        assert not any(k.startswith("week_08") for k in cache)

    def test_two_page_string_resolves_both(self):
        blocks, cache = _fixture()
        targets = set(
            _normalize_token_list_param("week_08_content_02, week_09_content_01")
        )
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, targets,
        )
        assert unknown == []
        assert removed == 3  # 1 on content_02 + 2 on week_09_content_01

    def test_unknown_page_string_still_fails_loud(self):
        blocks, cache = _fixture()
        targets = set(_normalize_token_list_param("week_77"))
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, targets,
        )
        assert removed == 0
        assert unknown == ["week_77"]


class TestBlockTypeStringTransport:
    def test_single_type_string_resolves_type(self):
        blocks, cache = _fixture()
        targets = set(_normalize_token_list_param("objective"))
        removed = _evict_rewrite_cache_by_block_type(cache, blocks, targets)
        assert removed == 2  # both objective instances
        assert "week_08_content_01#objective_intro_0" not in cache
        assert "week_09_content_01#objective_x_0" not in cache
        # Non-objective blocks survive.
        assert "week_08_content_02#concept_chain_rule_0" in cache

    def test_two_type_string_resolves_both_types(self):
        blocks, cache = _fixture()
        targets = set(_normalize_token_list_param("objective,concept"))
        removed = _evict_rewrite_cache_by_block_type(cache, blocks, targets)
        assert removed == 4  # 2 objective + 2 concept
        # The lone example survives.
        assert "week_08_content_01#example_derivative_03" in cache

    def test_unset_string_is_byte_identical_noop(self):
        blocks, cache = _fixture()
        before = dict(cache)
        targets = set(_normalize_token_list_param(None))
        removed = _evict_rewrite_cache_by_block_type(cache, blocks, targets)
        assert removed == 0
        assert cache == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
