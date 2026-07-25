"""Regression net for the deterministic required-`data-cf-*` attribute stamp.

The `rewrite_html_shape` gate requires per-block-type wrapper attributes. When
the rewrite model omits one, a re-roll is the wrong instrument: the value is
not something the model must invent — the Block already declares it. A
measured 15-block targeted re-roll fixed 11 and left exactly this class behind
(4 `assessment_item` blocks missing `data-cf-objective-ref` while each carried
its `objective_ids`).

Anti-fabrication is the load-bearing contract: nothing declared -> nothing
stamped, so the block fails the gate honestly instead of gaining a made-up ref.
"""

import dataclasses

import pytest

from Courseforge.scripts.blocks import Block
from MCP.tools.pipeline_tools import _stamp_missing_required_attrs


def _blk(**over):
    base = dict(
        block_id="p#assessment_item_x_0",
        block_type="assessment_item",
        page_id="p",
        sequence=0,
        content='<section data-cf-block-id="p#assessment_item_x_0">Q?</section>',
        objective_ids=["CO-01"],
    )
    base.update(over)
    fields = {f.name for f in dataclasses.fields(Block)}
    return Block(**{k: v for k, v in base.items() if k in fields})


def test_stamps_declared_objective_ref():
    out, n = _stamp_missing_required_attrs([_blk()])
    assert n == 1
    assert 'data-cf-objective-ref="CO-01"' in out[0].content


def test_does_not_fabricate_when_nothing_declared():
    out, n = _stamp_missing_required_attrs([_blk(objective_ids=[])])
    assert n == 0
    assert "data-cf-objective-ref" not in out[0].content


def test_never_overwrites_an_existing_attribute():
    existing = '<section data-cf-objective-ref="CO-99">Q?</section>'
    out, n = _stamp_missing_required_attrs([_blk(content=existing)])
    assert n == 0
    assert out[0].content == existing


def test_joins_multiple_declared_objectives():
    out, _ = _stamp_missing_required_attrs([_blk(objective_ids=["CO-01", "CO-02"])])
    assert 'data-cf-objective-ref="CO-01,CO-02"' in out[0].content


def test_only_the_first_wrapper_tag_is_stamped():
    out, _ = _stamp_missing_required_attrs([
        _blk(content="<section><div>a</div><div>b</div></section>")
    ])
    assert out[0].content.count("data-cf-objective-ref") == 1
    assert out[0].content.startswith('<section data-cf-objective-ref="CO-01">')


@pytest.mark.parametrize("block_type", ["concept", "example", "callout", "chrome"])
def test_untargeted_block_types_are_untouched(block_type):
    original = "<section>body</section>"
    out, n = _stamp_missing_required_attrs([
        _blk(block_type=block_type, content=original)
    ])
    assert n == 0 and out[0].content == original


@pytest.mark.parametrize("content", [None, "", "   ", 123, {"div": 1}])
def test_non_string_or_empty_content_is_untouched(content):
    out, n = _stamp_missing_required_attrs([_blk(content=content)])
    assert n == 0 and out[0].content == content


def test_content_with_no_tag_is_untouched():
    out, n = _stamp_missing_required_attrs([_blk(content="bare text, no tags")])
    assert n == 0


def test_empty_input_is_a_noop():
    assert _stamp_missing_required_attrs([]) == ([], 0)
