"""Track-A planner -> outline-Block seam tests.

Covers the end-to-end threading of the four default-off ``recall_self_check`` /
``misconception_rich`` stamp fields (``recall_format`` / ``mc_named_concept`` /
``mc_predict_prompt`` / ``mc_reconcile``) from the planner ``selected`` list,
through ``block_planner._to_page_plan``'s trailing-extras-dict tuple element,
into the ``MCP/tools/pipeline_tools.py`` outline-Block consumer unpack, and onto
the ``Block(...)`` stub.

The byte-stability contract: when NO block carries any of the four fields,
``_to_page_plan`` emits the legacy 3/4/5-tuple shape with NO 6-element tuple
(this is what guarantees a flag-off run is byte-identical, since the fields are
only ever set when the flags are on).
"""

from __future__ import annotations

from lib.generation.block_planner import CANONICAL_PAGE_TYPES, _to_page_plan


_TRACK_A_FIELDS = (
    "recall_format",
    "mc_named_concept",
    "mc_predict_prompt",
    "mc_reconcile",
)


def _consumer_unpack(spec):
    """Replicate the pipeline_tools outline consumer's arity-tolerant unpack.

    Mirrors ``MCP/tools/pipeline_tools.py::_run_content_generation_outline``'s
    per-spec read so the test pins the exact extraction contract the Block stub
    relies on.
    """
    spec_type = spec[0]
    spec_bloom = spec[1]
    spec_co_ids = tuple(spec[2]) if len(spec) > 2 and spec[2] else ()
    spec_itype = str(spec[3]) if len(spec) > 3 and spec[3] else None
    spec_fade = str(spec[4]) if len(spec) > 4 and spec[4] else None
    spec_extras = spec[5] if len(spec) > 5 and isinstance(spec[5], dict) else {}
    return spec_type, spec_bloom, spec_co_ids, spec_itype, spec_fade, spec_extras


def test_to_page_plan_emits_extras_dict_at_index_5():
    """(a) A selected block carrying track-A fields -> 6-tuple, dict at idx 5."""
    selected = [
        {
            "page_type": "content",
            "block_type": "self_check_question",
            "target_bloom": "remember",
            "target_co_ids": ["CO-01"],
            "recall_format": "cloze",
        },
        {
            "page_type": "content",
            "block_type": "misconception",
            "target_bloom": "understand",
            "target_co_ids": ["CO-02"],
            "mc_named_concept": "order-of-operations",
            "mc_predict_prompt": "What do you expect?",
            "mc_reconcile": "Why the faulty model fails.",
        },
    ]
    plan = _to_page_plan(selected)
    content = plan["content"]
    assert len(content) == 2

    recall_tuple = content[0]
    assert len(recall_tuple) == 6
    assert isinstance(recall_tuple[5], dict)
    assert recall_tuple[5] == {"recall_format": "cloze"}
    # itype/fade forced present (None) so positional consistency holds.
    assert recall_tuple[3] is None
    assert recall_tuple[4] is None

    mc_tuple = content[1]
    assert len(mc_tuple) == 6
    assert mc_tuple[5] == {
        "mc_named_concept": "order-of-operations",
        "mc_predict_prompt": "What do you expect?",
        "mc_reconcile": "Why the faulty model fails.",
    }


def test_to_page_plan_byte_identical_legacy_shape_without_track_a():
    """(b) No track-A field -> legacy 3/4/5-tuple shape, ZERO 6-element tuples."""
    selected = [
        # Bare 3-tuple block.
        {
            "page_type": "content",
            "block_type": "concept",
            "target_bloom": "understand",
            "target_co_ids": ["CO-01"],
        },
        # 4-tuple block (interaction_type set).
        {
            "page_type": "content",
            "block_type": "self_check_question",
            "target_bloom": "remember",
            "target_co_ids": ["CO-01"],
            "interaction_type": "select",
        },
        # 5-tuple block (fade_state set).
        {
            "page_type": "application",
            "block_type": "worked_example",
            "target_bloom": "apply",
            "target_co_ids": ["CO-02"],
            "fade_state": "worked",
        },
    ]
    plan = _to_page_plan(selected)
    all_tuples = [t for tuples in plan.values() for t in tuples]
    # No 6-element tuples appear when no block carries a track-A field.
    assert all(len(t) < 6 for t in all_tuples)
    # Exact legacy shapes preserved.
    assert plan["content"][0] == ("concept", "understand", ["CO-01"])
    assert plan["content"][1] == (
        "self_check_question", "remember", ["CO-01"], "select",
    )
    assert plan["application"][0] == (
        "worked_example", "apply", ["CO-02"], None, "worked",
    )


def test_every_canonical_page_type_present():
    """The plan keys every canonical page type (consumer loop never KeyErrors)."""
    plan = _to_page_plan([])
    for ptype in CANONICAL_PAGE_TYPES:
        assert ptype in plan
        assert plan[ptype] == []


def test_consumer_unpack_extracts_extras():
    """(c) The consumer unpack pulls the four fields off the 6-tuple."""
    selected = [
        {
            "page_type": "content",
            "block_type": "misconception",
            "target_bloom": "understand",
            "target_co_ids": ["CO-02"],
            "recall_format": "free_recall",
            "mc_named_concept": "slug-x",
        },
    ]
    spec = _to_page_plan(selected)["content"][0]
    (_t, _b, _co, _it, _fd, extras) = _consumer_unpack(spec)
    assert extras.get("recall_format") == "free_recall"
    assert extras.get("mc_named_concept") == "slug-x"
    assert extras.get("mc_predict_prompt") is None
    assert extras.get("mc_reconcile") is None


def test_consumer_unpack_legacy_tuple_yields_empty_extras():
    """A legacy 3/4/5-tuple yields an empty extras dict (defaults all None)."""
    for spec in (
        ("concept", "understand", ["CO-01"]),
        ("self_check_question", "remember", ["CO-01"], "select"),
        ("worked_example", "apply", ["CO-02"], None, "worked"),
    ):
        (*_rest, extras) = _consumer_unpack(spec)
        assert extras == {}
        assert all(extras.get(f) is None for f in _TRACK_A_FIELDS)


def test_block_accepts_track_a_kwargs_from_extras():
    """Block(...) accepts the four extras-sourced kwargs and carries them."""
    from Courseforge.scripts.blocks import Block

    selected = [
        {
            "page_type": "content",
            "block_type": "misconception",
            "target_bloom": "understand",
            "target_co_ids": ["CO-02"],
            "recall_format": "cloze",
            "mc_named_concept": "slug-x",
            "mc_predict_prompt": "predict?",
            "mc_reconcile": "reconcile.",
        },
    ]
    spec = _to_page_plan(selected)["content"][0]
    (_t, _b, _co, _it, _fd, extras) = _consumer_unpack(spec)
    block = Block(
        block_id="blk-1",
        block_type="misconception",
        page_id="page-1",
        sequence=0,
        content="",
        objective_ids=("CO-02",),
        recall_format=extras.get("recall_format"),
        mc_named_concept=extras.get("mc_named_concept"),
        mc_predict_prompt=extras.get("mc_predict_prompt"),
        mc_reconcile=extras.get("mc_reconcile"),
    )
    assert block.recall_format == "cloze"
    assert block.mc_named_concept == "slug-x"
    assert block.mc_predict_prompt == "predict?"
    assert block.mc_reconcile == "reconcile."


def test_block_track_a_fields_excluded_from_content_hash():
    """The four fields stay out of compute_content_hash's allowlist (no drift)."""
    from Courseforge.scripts.blocks import Block

    base = dict(
        block_id="blk-1",
        block_type="misconception",
        page_id="page-1",
        sequence=0,
        content="body",
        objective_ids=("CO-02",),
    )
    plain = Block(**base)
    stamped = Block(
        **base,
        recall_format="cloze",
        mc_named_concept="slug-x",
        mc_predict_prompt="predict?",
        mc_reconcile="reconcile.",
    )
    assert plain.compute_content_hash() == stamped.compute_content_hash()
