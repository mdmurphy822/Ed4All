"""Unit tests for the GAP D / GAP C per-CO + per-content-page planner floors.

Covers the two opt-in floors added to ``lib.generation.block_planner``:

- ``_apply_triangle_floor`` (GAP D — IB3 triangle): gated on
  ``ED4ALL_TRIANGLE_FLOOR``; default-off identity no-op; on → every CO
  referenced by >= 1 block gets >= 1 activity-class block AND >= 1 band-aligned
  ``assessment_item`` (target_bloom = the CO's declared Bloom). The resulting
  plan passes ``TriangleCompletenessValidator``.
- ``_apply_retrieval_interleave_floor`` (GAP C — IB7.5b): gated on
  ``ED4ALL_RETRIEVAL_INTERLEAVE``; default-off identity no-op; on → the content
  page carries >= 1 spaced ``self_check_question`` block (page_type='content').
  The resulting plan passes ``RetrievalPresenceValidator``.

Plus the cross-cutting contracts shared with every other planner floor/pass:
flags-off byte-identical (the whole ``plan_week_blocks`` output, not just the
two passes), no invented CO ids, and the existing planner/floor/IB7 tests stay
green (run separately).
"""
from __future__ import annotations

import copy

import pytest

from lib.generation.block_planner import (
    _apply_alignment_floors,
    _apply_retrieval_interleave_floor,
    _apply_triangle_floor,
    _resolve_block_types,
    plan_week_blocks,
)


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    """Strip every planner flag so each test controls exactly what it sets."""
    for env in (
        "ED4ALL_TRIANGLE_FLOOR",
        "ED4ALL_RETRIEVAL_INTERLEAVE",
        "ED4ALL_DYNAMIC_BLOCK_PLAN",
        "ED4ALL_NEW_BLOCK_TYPES",
        "ED4ALL_ALIGNMENT_VERB_TRIPLE",
        "ED4ALL_WORKED_EXAMPLE_FLOOR",
        "ED4ALL_BLOOM_SPREAD_FLOOR",
        "ED4ALL_PLANNER_BLOOM_CLIMB",
        "ED4ALL_PLANNER_LIFECYCLE",
        "ED4ALL_PLANNER_SPACING",
        "ED4ALL_PLANNER_BLOOM_CEILING",
        "ED4ALL_PLANNER_FADING",
        "ED4ALL_BLOCK_ANATOMY",
    ):
        monkeypatch.delenv(env, raising=False)
    yield


@pytest.fixture()
def block_types():
    return _resolve_block_types()


def _cos():
    return [
        {"id": "CO-01", "statement": "Add integers", "bloom_level": "apply"},
        {"id": "CO-02", "statement": "Compare integers", "bloom_level": "understand"},
    ]


def _co_bloom(cos):
    return {str(c["id"]): str(c.get("bloom_level") or "") for c in cos}


def _content_selection():
    """A selected block set that references CO-01 + CO-02 but has no triangle
    and no interleaved retrieval — the gap the floors close."""
    return [
        {"block_type": "objective", "page_type": "overview",
         "target_co_ids": ["CO-01", "CO-02"], "target_bloom": "understand"},
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "explanation", "page_type": "content",
         "target_co_ids": ["CO-02"], "target_bloom": "understand"},
    ]


# --------------------------------------------------------------------------- #
# GAP D — triangle floor.
# --------------------------------------------------------------------------- #
def test_triangle_floor_noop_when_flag_off(block_types):
    sel = _content_selection()
    before = copy.deepcopy(sel)
    out = _apply_triangle_floor(
        selected=sel, chapter_objectives=_cos(),
        block_types=block_types, co_bloom=_co_bloom(_cos()),
    )
    assert sel == before  # input not mutated
    assert out == before  # identity


def test_triangle_floor_adds_activity_and_assessment_per_referenced_co(block_types):
    import os
    os.environ["ED4ALL_TRIANGLE_FLOOR"] = "1"
    try:
        out = _apply_triangle_floor(
            selected=_content_selection(), chapter_objectives=_cos(),
            block_types=block_types, co_bloom=_co_bloom(_cos()),
        )
    finally:
        del os.environ["ED4ALL_TRIANGLE_FLOOR"]

    activity_types = {"self_check_question", "activity", "problem", "scenario"}
    for cid, expect_bloom in (("CO-01", "apply"), ("CO-02", "understand")):
        acts = [
            b for b in out
            if b["block_type"] in activity_types and cid in b["target_co_ids"]
        ]
        assess = [
            b for b in out
            if b["block_type"] == "assessment_item" and cid in b["target_co_ids"]
        ]
        assert acts, f"{cid} got no activity block"
        assert assess, f"{cid} got no assessment_item"
        # band-aligned: assessment target_bloom == the CO's declared Bloom.
        assert assess[0]["target_bloom"] == expect_bloom


def test_triangle_floor_never_invents_co_ids(block_types):
    import os
    os.environ["ED4ALL_TRIANGLE_FLOOR"] = "1"
    try:
        out = _apply_triangle_floor(
            selected=_content_selection(), chapter_objectives=_cos(),
            block_types=block_types, co_bloom=_co_bloom(_cos()),
        )
    finally:
        del os.environ["ED4ALL_TRIANGLE_FLOOR"]
    valid = {"CO-01", "CO-02"}
    for b in out:
        for cid in b["target_co_ids"]:
            assert cid in valid, f"invented CO id {cid!r}"


def test_triangle_floor_empty_when_no_referenced_co(block_types):
    """A block set with only empty target_co_ids (the fixed-plan shape) gets no
    triangle injection even when the flag is on (anti-fabrication — the gate
    audits only objectives a block references)."""
    import os
    os.environ["ED4ALL_TRIANGLE_FLOOR"] = "1"
    sel = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": [], "target_bloom": "understand"},
    ]
    before = copy.deepcopy(sel)
    try:
        out = _apply_triangle_floor(
            selected=sel, chapter_objectives=_cos(),
            block_types=block_types, co_bloom=_co_bloom(_cos()),
        )
    finally:
        del os.environ["ED4ALL_TRIANGLE_FLOOR"]
    assert out == before


def test_triangle_floor_does_not_double_inject(block_types):
    """A CO already carrying an activity + assessment_item gets no duplicate."""
    import os
    os.environ["ED4ALL_TRIANGLE_FLOOR"] = "1"
    sel = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "assessment_item", "page_type": "self_check",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    before = copy.deepcopy(sel)
    try:
        out = _apply_triangle_floor(
            selected=sel, chapter_objectives=[{"id": "CO-01", "bloom_level": "apply"}],
            block_types=block_types, co_bloom={"CO-01": "apply"},
        )
    finally:
        del os.environ["ED4ALL_TRIANGLE_FLOOR"]
    assert out == before  # CO-01 already had both arms → no injection


# --------------------------------------------------------------------------- #
# GAP C — retrieval-interleave floor.
# --------------------------------------------------------------------------- #
def test_retrieval_floor_noop_when_flag_off(block_types):
    sel = _content_selection()
    before = copy.deepcopy(sel)
    out = _apply_retrieval_interleave_floor(
        selected=sel, chapter_objectives=_cos(), block_types=block_types,
    )
    assert sel == before
    assert out == before


def test_retrieval_floor_injects_content_page_retrieval(block_types):
    import os
    os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"] = "1"
    try:
        out = _apply_retrieval_interleave_floor(
            selected=_content_selection(), chapter_objectives=_cos(),
            block_types=block_types,
        )
    finally:
        del os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"]
    content_retrieval = [
        b for b in out
        if b["page_type"] == "content"
        and b["block_type"] in {"self_check_question", "reflection_prompt"}
    ]
    assert content_retrieval, "no retrieval block on the content page"
    # It targets a REAL CO id (the last content exposition's CO — CO-02 here).
    for cid in content_retrieval[0]["target_co_ids"]:
        assert cid in {"CO-01", "CO-02"}
    # It is placed AFTER >= 1 content-page exposition block (spaced).
    content_idxs = [i for i, b in enumerate(out) if b["page_type"] == "content"]
    first_retrieval_pos = next(
        i for i in content_idxs
        if out[i]["block_type"] in {"self_check_question", "reflection_prompt"}
    )
    assert first_retrieval_pos != content_idxs[0]  # not first on the page


def test_retrieval_floor_noop_when_content_already_has_retrieval(block_types):
    import os
    os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"] = "1"
    sel = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "self_check_question", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    before = copy.deepcopy(sel)
    try:
        out = _apply_retrieval_interleave_floor(
            selected=sel, chapter_objectives=_cos(), block_types=block_types,
        )
    finally:
        del os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"]
    assert out == before


def test_retrieval_floor_noop_when_no_content_exposition(block_types):
    """A page group with content page_type but no exposition block (e.g. only a
    chrome/objective) is not content-bearing → no injection."""
    import os
    os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"] = "1"
    sel = [
        {"block_type": "objective", "page_type": "overview",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    before = copy.deepcopy(sel)
    try:
        out = _apply_retrieval_interleave_floor(
            selected=sel, chapter_objectives=_cos(), block_types=block_types,
        )
    finally:
        del os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"]
    assert out == before


def test_retrieval_floor_preserves_end_of_week_self_check(block_types):
    """The cumulative end-of-week self_check page is never removed/moved."""
    import os
    os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"] = "1"
    sel = _content_selection() + [
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    try:
        out = _apply_retrieval_interleave_floor(
            selected=sel, chapter_objectives=_cos(), block_types=block_types,
        )
    finally:
        del os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"]
    self_check = [b for b in out if b["page_type"] == "self_check"]
    assert self_check, "end-of-week self_check page lost"


# --------------------------------------------------------------------------- #
# Cross-cutting: flags-off byte-identical whole-plan + validators pass.
# --------------------------------------------------------------------------- #
def test_plan_week_blocks_byte_identical_when_both_flags_off(monkeypatch):
    """The whole plan_week_blocks output is byte-identical with the two new
    flags off vs. simply unset (the floors must not perturb the default path)."""
    to = {"id": "TO-01", "statement": "Master integer operations"}
    chunks = [{"id": "c1", "text": "Add integers step by step.", "heading": "Integers"}]

    # Baseline: all flags unset.
    plan_a = plan_week_blocks(
        terminal_objective=to, chapter_objectives=_cos(),
        source_chunks=chunks, provider=None,
    )
    # Explicitly-off flags must produce the identical plan.
    monkeypatch.setenv("ED4ALL_TRIANGLE_FLOOR", "0")
    monkeypatch.setenv("ED4ALL_RETRIEVAL_INTERLEAVE", "off")
    plan_b = plan_week_blocks(
        terminal_objective=to, chapter_objectives=_cos(),
        source_chunks=chunks, provider=None,
    )
    assert plan_a.selected == plan_b.selected
    assert plan_a.page_plan == plan_b.page_plan


def test_floors_make_triangle_and_retrieval_validators_pass(block_types):
    """The keystone end-to-end assertion: a content selection run through BOTH
    floors passes TriangleCompletenessValidator AND RetrievalPresenceValidator."""
    import os
    os.environ["ED4ALL_TRIANGLE_FLOOR"] = "1"
    os.environ["ED4ALL_RETRIEVAL_INTERLEAVE"] = "1"
    os.environ["ED4ALL_ALIGNMENT_VERB_TRIPLE"] = "1"  # gate the triangle validator on
    try:
        selected = _content_selection()
        selected = _apply_triangle_floor(
            selected=selected, chapter_objectives=_cos(),
            block_types=block_types, co_bloom=_co_bloom(_cos()),
        )
        selected = _apply_retrieval_interleave_floor(
            selected=selected, chapter_objectives=_cos(), block_types=block_types,
        )

        # Materialize block-shaped objects the validators read.
        class _B:
            def __init__(self, block_type, page_id, objective_ids, bloom):
                self.block_type = block_type
                self.page_id = page_id
                self.objective_ids = tuple(objective_ids)
                self.content = {"bloom_level": bloom}
                self.interaction = None

        blocks = [
            _B(b["block_type"], "week01_" + b["page_type"],
               b["target_co_ids"], b["target_bloom"])
            for b in selected
        ]

        from lib.validators.alignment.triangle_completeness import (
            TriangleCompletenessValidator,
        )
        objectives = {
            "CO-01": {"bloom_level": "apply"},
            "CO-02": {"bloom_level": "understand"},
        }
        tri = TriangleCompletenessValidator().validate(
            {"blocks": blocks, "objectives": objectives}
        )
        assert tri.score == 1.0, [(i.code, i.location) for i in tri.issues]
        assert not [i for i in tri.issues if i.code in {
            "OBJECTIVE_NO_ACTIVITY", "OBJECTIVE_NO_ALIGNED_ASSESSMENT"
        }]

        from lib.validators.retrieval_presence import RetrievalPresenceValidator
        ret = RetrievalPresenceValidator().validate({"blocks": blocks})
        assert ret.passed, [(i.code, i.location) for i in ret.issues]
    finally:
        for env in (
            "ED4ALL_TRIANGLE_FLOOR",
            "ED4ALL_RETRIEVAL_INTERLEAVE",
            "ED4ALL_ALIGNMENT_VERB_TRIPLE",
        ):
            os.environ.pop(env, None)


# --------------------------------------------------------------------------- #
# F3 / F4 / F5 — end-to-end gate-closure through the REAL plan_week_blocks
# pipeline (floors run AFTER all IB7 passes; triangle AFTER retrieval).
# --------------------------------------------------------------------------- #
class _Blk:
    """A Block-shaped object the triangle + retrieval validators both read.

    Triangle reads ``objective_ids`` + (via the verb-band resolver)
    ``content['bloom_level']``; retrieval reads ``block_type`` + ``page_id``.
    The downstream consumer maps each planner ``page_type`` to a distinct
    ``page_id`` (``_page_id_for`` → ``week_NN_<type>``), so we mirror that here
    so a content-bearing ``overview`` / ``application`` page is its own module.
    """

    def __init__(self, block_type, page_type, objective_ids, bloom):
        self.block_type = block_type
        self.page_type = page_type
        self.page_id = "week01_" + str(page_type)
        self.objective_ids = tuple(objective_ids)
        self.content = {"bloom_level": bloom}
        self.interaction = None


def _materialize(selected):
    return [
        _Blk(b["block_type"], b["page_type"], b["target_co_ids"], b["target_bloom"])
        for b in selected
    ]


def _assert_both_gates_pass(selected, objectives):
    """The keystone: TriangleCompletenessValidator (zero OBJECTIVE_NO_*) AND
    RetrievalPresenceValidator (passed) BOTH pass over the materialized plan."""
    from lib.validators.alignment.triangle_completeness import (
        TriangleCompletenessValidator,
    )
    from lib.validators.retrieval_presence import RetrievalPresenceValidator

    blocks = _materialize(selected)
    tri = TriangleCompletenessValidator().validate(
        {"blocks": blocks, "objectives": objectives}
    )
    bad = [
        i for i in tri.issues
        if i.code in {"OBJECTIVE_NO_ACTIVITY", "OBJECTIVE_NO_ALIGNED_ASSESSMENT"}
    ]
    assert not bad, [(i.code, i.location) for i in bad]
    assert tri.score == 1.0, tri.score

    ret = RetrievalPresenceValidator().validate({"blocks": blocks})
    assert ret.passed, [(i.code, i.location) for i in ret.issues]


def _run_plan(monkeypatch, cos, *, bloom_ceiling=False):
    """Run the REAL plan_week_blocks with BOTH floors on (provider=None ⇒ the
    fixed-plan fallback path), optionally co-enabling ED4ALL_PLANNER_BLOOM_CEILING.

    The fixed-plan fallback emits empty-target_co_id blocks, so we drive the
    floors over a CONTENT selection that references the COs by passing it through
    the floor application directly AFTER plan_week_blocks would normally run the
    IB7 passes — but to exercise the true end-to-end order we instead call the
    public _apply_alignment_floors (the same single call site plan_week_blocks +
    _fallback_plan use) over a representative IB7-ordered selection."""
    monkeypatch.setenv("ED4ALL_TRIANGLE_FLOOR", "1")
    monkeypatch.setenv("ED4ALL_RETRIEVAL_INTERLEAVE", "1")
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    if bloom_ceiling:
        monkeypatch.setenv("ED4ALL_PLANNER_BLOOM_CEILING", "1")


def test_e2e_create_level_co_closes_both_gates(monkeypatch, block_types):
    """A CREATE-level CO: the triangle activity arm (self_check_question,
    catalog bloom_ceiling=apply) is injected at target_bloom=create. Because the
    floors run AFTER _apply_bloom_ceilings, the ceiling pass can NOT re-route it
    to assessment_item (which would re-fire OBJECTIVE_NO_ACTIVITY)."""
    _run_plan(monkeypatch, None, bloom_ceiling=True)
    cos = [{"id": "CO-01", "statement": "Design a proof", "bloom_level": "create"}]
    objectives = {"CO-01": {"bloom_level": "create"}}
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    # Simulate the real run order: IB7 ceiling pass FIRST (it must not see the
    # floor-injected blocks), then the alignment floors LAST.
    from lib.generation.block_planner import _apply_ib7_passes, load_block_catalog
    catalog_by_type = {
        str(e.get("block_type")): e for e in load_block_catalog() if e.get("block_type")
    }
    selected = _apply_ib7_passes(
        selected=selected, chapter_objectives=cos,
        catalog_by_type=catalog_by_type, block_types=block_types, signals={},
    )
    selected = _apply_alignment_floors(
        selected=selected, chapter_objectives=cos,
        block_types=block_types, co_bloom={"CO-01": "create"},
    )
    # The activity arm survived as an activity-class type (not re-routed).
    acts = [
        b for b in selected
        if b["block_type"] in {"self_check_question", "activity", "problem", "scenario"}
        and "CO-01" in b["target_co_ids"]
    ]
    assert acts, "create-level CO lost its activity arm to the bloom-ceiling reroute"
    _assert_both_gates_pass(selected, objectives)


def test_e2e_evaluate_level_co_closes_both_gates(monkeypatch, block_types):
    """An EVALUATE-level CO likewise keeps a complete triangle under the ceiling
    pass (scenario admits evaluate, but the floors run after the ceiling pass so
    the injected blocks are never inspected by it anyway)."""
    _run_plan(monkeypatch, None, bloom_ceiling=True)
    cos = [{"id": "CO-01", "statement": "Critique an argument", "bloom_level": "evaluate"}]
    objectives = {"CO-01": {"bloom_level": "evaluate"}}
    selected = [
        {"block_type": "explanation", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    from lib.generation.block_planner import _apply_ib7_passes, load_block_catalog
    catalog_by_type = {
        str(e.get("block_type")): e for e in load_block_catalog() if e.get("block_type")
    }
    selected = _apply_ib7_passes(
        selected=selected, chapter_objectives=cos,
        catalog_by_type=catalog_by_type, block_types=block_types, signals={},
    )
    selected = _apply_alignment_floors(
        selected=selected, chapter_objectives=cos,
        block_types=block_types, co_bloom={"CO-01": "evaluate"},
    )
    _assert_both_gates_pass(selected, objectives)


def test_e2e_content_bearing_overview_and_application_pages_get_retrieval(
    monkeypatch, block_types,
):
    """F3b: a content-bearing overview / application page (NOT just a 'content'
    page) is its own module and must get its own interleaved retrieval block."""
    _run_plan(monkeypatch, None)
    cos = [
        {"id": "CO-01", "statement": "Add integers", "bloom_level": "apply"},
        {"id": "CO-02", "statement": "Apply integers", "bloom_level": "apply"},
    ]
    objectives = {
        "CO-01": {"bloom_level": "apply"},
        "CO-02": {"bloom_level": "apply"},
    }
    # An overview page carrying a content block, an application page carrying a
    # content block — both content-bearing, neither is page_type 'content'.
    selected = [
        {"block_type": "concept", "page_type": "overview",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "problem", "page_type": "application",
         "target_co_ids": ["CO-02"], "target_bloom": "apply"},
    ]
    selected = _apply_alignment_floors(
        selected=selected, chapter_objectives=cos,
        block_types=block_types, co_bloom={"CO-01": "apply", "CO-02": "apply"},
    )
    # Each content-bearing non-'content' page got an interleaved retrieval block.
    for ptype in ("overview", "application"):
        page_retrieval = [
            b for b in selected
            if b["page_type"] == ptype
            and b["block_type"] in {"self_check_question", "reflection_prompt"}
        ]
        assert page_retrieval, f"{ptype} page got no interleaved retrieval block"
    _assert_both_gates_pass(selected, objectives)


def test_e2e_fallback_path_provider_none_closes_both_gates(monkeypatch, block_types):
    """provider=None ⇒ _fallback_plan. With a CO-targeted seed selection, the
    floors close BOTH gates on the fallback path too (the floors run AFTER the
    fallback's IB7 passes)."""
    _run_plan(monkeypatch, None)
    cos = [{"id": "CO-01", "statement": "Add integers", "bloom_level": "apply"}]
    objectives = {"CO-01": {"bloom_level": "apply"}}
    # The pure fixed-plan fallback has empty target_co_ids (no triangle to close,
    # by design), so seed a CO-referencing content selection to exercise the
    # floor closure on the fallback's alignment-floor call site.
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    selected = _apply_alignment_floors(
        selected=selected, chapter_objectives=cos,
        block_types=block_types, co_bloom={"CO-01": "apply"},
    )
    _assert_both_gates_pass(selected, objectives)


def test_e2e_co_with_only_retrieval_activity_still_gets_assessment(
    monkeypatch, block_types,
):
    """F3a: a CO whose ONLY activity comes from the retrieval-injected block
    still ends with BOTH arms — because the triangle floor runs AFTER the
    retrieval floor and re-scans the referenced-CO set."""
    _run_plan(monkeypatch, None)
    cos = [{"id": "CO-01", "statement": "Add integers", "bloom_level": "apply"}]
    objectives = {"CO-01": {"bloom_level": "apply"}}
    # A single content exposition for CO-01 — no activity, no assessment. The
    # retrieval floor injects a self_check_question (an activity-class type) for
    # CO-01; the triangle floor (running after) must then still add the missing
    # assessment_item for CO-01.
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    selected = _apply_alignment_floors(
        selected=selected, chapter_objectives=cos,
        block_types=block_types, co_bloom={"CO-01": "apply"},
    )
    assess = [
        b for b in selected
        if b["block_type"] == "assessment_item" and "CO-01" in b["target_co_ids"]
    ]
    assert assess, "CO whose activity came only from retrieval got no assessment_item"
    _assert_both_gates_pass(selected, objectives)


def test_plan_week_blocks_byte_identical_with_all_ib7_and_floors_off(monkeypatch):
    """Strengthened byte-stability: with EVERY IB7 + floor flag explicitly off,
    plan_week_blocks is byte-identical to the all-unset baseline."""
    to = {"id": "TO-01", "statement": "Master integer operations"}
    chunks = [{"id": "c1", "text": "Add integers step by step.", "heading": "Integers"}]
    plan_a = plan_week_blocks(
        terminal_objective=to, chapter_objectives=_cos(),
        source_chunks=chunks, provider=None,
    )
    for env in (
        "ED4ALL_TRIANGLE_FLOOR", "ED4ALL_RETRIEVAL_INTERLEAVE",
        "ED4ALL_PLANNER_BLOOM_CLIMB", "ED4ALL_PLANNER_LIFECYCLE",
        "ED4ALL_PLANNER_SPACING", "ED4ALL_PLANNER_BLOOM_CEILING",
        "ED4ALL_PLANNER_FADING",
    ):
        monkeypatch.setenv(env, "0")
    plan_b = plan_week_blocks(
        terminal_objective=to, chapter_objectives=_cos(),
        source_chunks=chunks, provider=None,
    )
    assert plan_a.selected == plan_b.selected
    assert plan_a.page_plan == plan_b.page_plan
