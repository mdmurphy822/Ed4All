"""Regression: the post_rewrite ``_entry_to_block`` loader must source its
``content_type_label`` enum from the canonical taxonomy
(``lib/validators/content_type.py::get_valid_chunk_types`` →
``schemas/taxonomies/content_type.json#/$defs/ChunkType``), NOT a stale
hardcoded set.

Bug history: the loader nested in
``MCP/tools/pipeline_tools.py::_run_post_rewrite_validation`` carried a stale
8-value hardcoded ``_CONTENT_TYPE_ENUM`` (``definition/example/procedure/
principle/fact/concept/rule/narrative``). Any rewrite-tier block whose
``content_type_label`` was a MODERN taxonomy value (``explanation``,
``key_idea``, ``exercise``, ``acronym``, ``table`` — all legitimately emitted)
fell through the ``out_of_enum`` branch and had its ``content_type`` field
silently DROPPED, malforming the block and cascading across the post_rewrite
gates. This test asserts the loader now PRESERVES canonical ChunkType values.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from lib.validators.content_type import get_valid_chunk_types
from MCP.tools.pipeline_tools import _run_post_rewrite_validation


def _run(**kwargs) -> dict:
    return json.loads(asyncio.run(_run_post_rewrite_validation(**kwargs)))


@pytest.mark.parametrize("ctype", ["explanation", "key_idea", "exercise"])
def test_loader_preserves_canonical_content_type_label(tmp_path, ctype):
    """A block whose ``content_type_label`` is a canonical ChunkType member
    is NOT dropped by the loader (no ``content_type`` metadata_drop)."""
    # Pre-condition: the value is a member of the canonical enum.
    assert ctype in get_valid_chunk_types()

    blocks_path = tmp_path / "blocks_final.jsonl"
    entry = {
        "block_id": "week_01_overview#concept_sample_0",
        "block_type": "concept",
        "page_id": "week_01_overview",
        "sequence": 0,
        "content": (
            f'<section data-cf-content-type="{ctype}" '
            'data-cf-source-ids="dart:src#blk_1">'
            "<h2>Sample heading</h2><p>Grounded prose.</p></section>"
        ),
        "content_type_label": ctype,
        "source_ids": ["dart:src#blk_1"],
    }
    blocks_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    out = _run(blocks_final_path=str(blocks_path), project_id="")
    assert out.get("success") is True, out

    # The canonical value must survive — no content_type drop record.
    content_type_drops = [
        d for d in out.get("metadata_drops", [])
        if d.get("field_name") == "content_type"
    ]
    assert content_type_drops == [], (
        f"content_type_label={ctype!r} (a canonical ChunkType member) was "
        f"dropped by the loader; metadata_drops={out.get('metadata_drops')}"
    )


def test_loader_drops_genuinely_off_taxonomy_content_type_label(tmp_path):
    """Anti-regression for the other direction: a value NOT in the canonical
    enum is still dropped (the loader stays fail-closed on garbage)."""
    bogus = "definitely_not_a_chunk_type"
    assert bogus not in get_valid_chunk_types()

    blocks_path = tmp_path / "blocks_final.jsonl"
    entry = {
        "block_id": "week_01_overview#concept_sample_1",
        "block_type": "concept",
        "page_id": "week_01_overview",
        "sequence": 0,
        "content": (
            '<section data-cf-content-type="explanation" '
            'data-cf-source-ids="dart:src#blk_1">'
            "<h2>Sample heading</h2><p>Grounded prose.</p></section>"
        ),
        "content_type_label": bogus,
        "source_ids": ["dart:src#blk_1"],
    }
    blocks_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    out = _run(blocks_final_path=str(blocks_path), project_id="")
    assert out.get("success") is True, out

    content_type_drops = [
        d for d in out.get("metadata_drops", [])
        if d.get("field_name") == "content_type"
        and d.get("reason") == "out_of_enum"
    ]
    assert len(content_type_drops) == 1, (
        f"off-taxonomy content_type_label={bogus!r} should be dropped once; "
        f"metadata_drops={out.get('metadata_drops')}"
    )
