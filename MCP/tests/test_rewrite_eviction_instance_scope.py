"""I4 stage 2 — block-instance / page-scoped rewrite-cache eviction.

Stage 1 (``--blocks``, type-level) evicts EVERY cached block of a named type.
Stage 2 lets an instructor re-roll ONE specific block instance
(``--block-ids``) or one page/module (``--pages``) without evicting every
block of that type. Both new seams live beside
``_evict_rewrite_cache_by_block_type`` in ``MCP/tools/pipeline_tools.py``:

* ``_evict_rewrite_cache_by_block_id`` — ADDITIVE eviction by EXACT block id.
* ``_evict_rewrite_cache_by_page`` — ADDITIVE eviction by page/module
  membership (exact ``page_id`` or module prefix via
  ``_page_membership_match``).

Both return ``(removed_count, unknown_tokens)``: an id/page that resolves
against NO outline block is UNKNOWN and returned so the rewrite handler fails
LOUD (never a silent no-op). All three filters unset → byte-identical
failure-driven reuse (asserted here).
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
    _page_membership_match,
)


def _blk(block_id: str, block_type: str, page_id: str) -> SimpleNamespace:
    """Minimal duck-typed stand-in for a Courseforge Block."""
    return SimpleNamespace(
        block_id=block_id, block_type=block_type, page_id=page_id
    )


def _fixture():
    """Two weeks, mixed block types on distinct pages."""
    blocks = [
        _blk("week_01_overview#objective_intro_0", "objective",
             "week_01_overview"),
        _blk("week_01_content_01#concept_a_0", "concept",
             "week_01_content_01"),
        _blk("week_01_content_01#example_a_1", "example",
             "week_01_content_01"),
        _blk("week_02_content_01#objective_x_0", "objective",
             "week_02_content_01"),
        _blk("week_02_content_01#concept_y_0", "concept",
             "week_02_content_01"),
    ]
    cache = {b.block_id: {"block_id": b.block_id, "content": "<p>x</p>"}
             for b in blocks}
    return blocks, cache


# --------------------------------------------------------------------------- #
# _page_membership_match
# --------------------------------------------------------------------------- #


class TestPageMembershipMatch:
    def test_exact_page_matches(self):
        assert _page_membership_match("week_01_content_01", "week_01_content_01")

    def test_module_prefix_matches(self):
        # ``week_01`` is a module prefix of every week-01 page.
        assert _page_membership_match("week_01_overview", "week_01")
        assert _page_membership_match("week_01_content_02", "week_01")

    def test_prefix_requires_separator_boundary(self):
        # ``week_01`` must NOT match ``week_010`` (no false-positive substring).
        assert not _page_membership_match("week_010_overview", "week_01")

    def test_non_matching_page(self):
        assert not _page_membership_match("week_02_overview", "week_01")

    def test_empty_operands(self):
        assert not _page_membership_match("", "week_01")
        assert not _page_membership_match("week_01_overview", "")


# --------------------------------------------------------------------------- #
# _evict_rewrite_cache_by_block_id (--block-ids)
# --------------------------------------------------------------------------- #


class TestEvictByBlockId:
    def test_unset_is_byte_identical_noop(self):
        blocks, cache = _fixture()
        before = dict(cache)
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, set()
        )
        assert removed == 0
        assert unknown == []
        assert cache == before

    def test_evicts_single_instance_only(self):
        """One id evicted; sibling of the SAME type is untouched."""
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, {"week_01_overview#objective_intro_0"},
        )
        assert removed == 1
        assert unknown == []
        assert "week_01_overview#objective_intro_0" not in cache
        # The OTHER objective instance keeps byte-identical reuse.
        assert "week_02_content_01#objective_x_0" in cache

    def test_multiple_ids(self):
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache,
            blocks,
            {"week_01_content_01#concept_a_0", "week_02_content_01#concept_y_0"},
        )
        assert removed == 2
        assert unknown == []
        assert "week_01_content_01#concept_a_0" not in cache
        assert "week_02_content_01#concept_y_0" not in cache

    def test_unknown_id_is_loud(self):
        """An id not carried by any outline block is reported UNKNOWN."""
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, {"week_99_ghost#objective_x_0"},
        )
        assert removed == 0
        assert unknown == ["week_99_ghost#objective_x_0"]
        # Cache untouched — the loud path never mutates on an unknown target.
        assert len(cache) == 5

    def test_unknown_detected_even_with_empty_cache(self):
        """Validation is against the outline set, not the cache — so an
        unknown id is caught on a first pass with no cache on disk."""
        blocks, _ = _fixture()
        removed, unknown = _evict_rewrite_cache_by_block_id(
            {}, blocks, {"ghost#x_0"},
        )
        assert removed == 0
        assert unknown == ["ghost#x_0"]

    def test_known_id_not_in_cache_is_clean(self):
        """A KNOWN id missing from the cache (never authored) is not an
        error and evicts nothing."""
        blocks, cache = _fixture()
        del cache["week_01_content_01#example_a_1"]
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, {"week_01_content_01#example_a_1"},
        )
        assert removed == 0
        assert unknown == []


# --------------------------------------------------------------------------- #
# _evict_rewrite_cache_by_page (--pages)
# --------------------------------------------------------------------------- #


class TestEvictByPage:
    def test_unset_is_byte_identical_noop(self):
        blocks, cache = _fixture()
        before = dict(cache)
        removed, unknown = _evict_rewrite_cache_by_page(cache, blocks, set())
        assert removed == 0
        assert unknown == []
        assert cache == before

    def test_exact_page_evicts_all_its_blocks(self):
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, {"week_01_content_01"},
        )
        assert removed == 2  # concept_a + example_a on that page
        assert unknown == []
        assert "week_01_content_01#concept_a_0" not in cache
        assert "week_01_content_01#example_a_1" not in cache
        # Other pages keep byte-identical reuse.
        assert "week_01_overview#objective_intro_0" in cache
        assert "week_02_content_01#objective_x_0" in cache

    def test_module_prefix_evicts_whole_week(self):
        """``--pages week_01`` re-rolls every week-01 page."""
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, {"week_01"},
        )
        assert removed == 3  # overview + 2 content-01 blocks
        assert unknown == []
        assert not any(k.startswith("week_01") for k in cache)
        # Week 02 untouched.
        assert "week_02_content_01#objective_x_0" in cache
        assert "week_02_content_01#concept_y_0" in cache

    def test_unknown_page_is_loud(self):
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, {"week_99"},
        )
        assert removed == 0
        assert unknown == ["week_99"]
        assert len(cache) == 5

    def test_unknown_detected_with_empty_cache(self):
        blocks, _ = _fixture()
        removed, unknown = _evict_rewrite_cache_by_page(
            {}, blocks, {"week_99_content_01"},
        )
        assert removed == 0
        assert unknown == ["week_99_content_01"]

    def test_mixed_known_and_unknown_reports_only_unknown(self):
        blocks, cache = _fixture()
        removed, unknown = _evict_rewrite_cache_by_page(
            cache, blocks, {"week_01", "week_77"},
        )
        # Loud path: any unknown reports; caller fails before re-running.
        assert unknown == ["week_77"]
        # The known week-01 blocks are still evicted (caller decides to abort).
        assert removed == 3


# --------------------------------------------------------------------------- #
# Combined default: stage-1 + stage-2 all unset → byte-identical reuse.
# --------------------------------------------------------------------------- #


class TestAllFiltersUnsetByteIdentical:
    def test_all_three_unset_is_noop(self):
        blocks, cache = _fixture()
        before = dict(cache)
        n_type = _evict_rewrite_cache_by_block_type(cache, blocks, set())
        n_id, unk_id = _evict_rewrite_cache_by_block_id(cache, blocks, set())
        n_pg, unk_pg = _evict_rewrite_cache_by_page(cache, blocks, set())
        assert (n_type, n_id, n_pg) == (0, 0, 0)
        assert unk_id == [] and unk_pg == []
        assert cache == before

    def test_stage2_is_additive_over_stage1(self):
        """Type eviction + an instance eviction of a DIFFERENT type compose."""
        blocks, cache = _fixture()
        _evict_rewrite_cache_by_block_type(cache, blocks, {"objective"})
        removed, unknown = _evict_rewrite_cache_by_block_id(
            cache, blocks, {"week_01_content_01#concept_a_0"},
        )
        assert unknown == []
        # Both objective instances gone (type) + one concept (instance).
        assert "week_01_overview#objective_intro_0" not in cache
        assert "week_02_content_01#objective_x_0" not in cache
        assert "week_01_content_01#concept_a_0" not in cache
        # The other concept (not named, not that type) survives.
        assert "week_02_content_01#concept_y_0" in cache


# --------------------------------------------------------------------------- #
# GUI seam — gui/services/run_service._normalize_blocks_param (block_ids/pages)
#
# Lives here (not gui/tests) because this lane owns MCP/tests; the helper is a
# backend run-service seam that turns the studio failure panel's ``block_ids``
# / ``pages`` CSV options into the ``target_block_instance_ids`` /
# ``target_page_ids`` params the pipeline routing consumes.
# --------------------------------------------------------------------------- #


class TestGuiInstancePageNormalization:
    def test_block_ids_csv_becomes_instance_param(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"course_name": "X", "block_ids": "p1#a_0, p2#b_1"}
        _normalize_blocks_param(wp)
        assert wp["target_block_instance_ids"] == ["p1#a_0", "p2#b_1"]
        assert "block_ids" not in wp

    def test_pages_csv_becomes_page_param(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"pages": "week_01, week_02_content_01"}
        _normalize_blocks_param(wp)
        assert wp["target_page_ids"] == ["week_01", "week_02_content_01"]
        assert "pages" not in wp

    def test_all_three_scopes_compose(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {
            "blocks": "objective",
            "block_ids": "p1#a_0",
            "pages": "week_03",
        }
        _normalize_blocks_param(wp)
        assert wp["target_block_ids"] == ["objective"]
        assert wp["target_block_instance_ids"] == ["p1#a_0"]
        assert wp["target_page_ids"] == ["week_03"]

    def test_list_form_and_dedup(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"block_ids": ["p1#a_0", "p1#a_0", "p2#b_0"]}
        _normalize_blocks_param(wp)
        assert wp["target_block_instance_ids"] == ["p1#a_0", "p2#b_0"]

    def test_absent_is_noop(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"course_name": "X"}
        _normalize_blocks_param(wp)
        assert "target_block_instance_ids" not in wp
        assert "target_page_ids" not in wp

    def test_explicit_param_wins(self):
        from gui.services.run_service import _normalize_blocks_param

        wp = {"pages": "week_01", "target_page_ids": ["week_09"]}
        _normalize_blocks_param(wp)
        assert wp["target_page_ids"] == ["week_09"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
