"""IB3 recalibration — asymmetric over-shoot tolerance for the verb-triple axis.

The 2-corpus calibration gap-map measured the strict band-EQUALITY verb-triple
fire at ~32%, mostly FALSE POSITIVES: blocks whose RESOLVED Bloom landed ABOVE
the objective band (math-idiom prose inflation + genuinely higher-order blocks
that still serve the objective). The recalibration makes the mismatch fire
ASYMMETRIC: ONLY a block strictly BELOW the objective band (genuine
UNDER-delivery) fires; at-or-above (equal OR over-shoot) is ALIGNED.

These tests are the keystone-safety contract:
  * a block teaching ABOVE its objective band MUST NOT fire (over-shoot OK).
  * a block teaching AT the band MUST NOT fire.
  * a block teaching strictly BELOW the band MUST still fire (under-delivery
    is the real defect the keystone exists to catch — coverage preserved).
  * the declared-bloom anchor (idiom-proof) takes precedence over the
    inflating prose scan.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.alignment.verb_triple import (  # noqa: E402
    resolve_block_verb_level,
    verb_triple_misaligned,
    verbs_share_band,
)
from lib.validators.block_objective_delivery import (  # noqa: E402
    BlockObjectiveDeliveryValidator,
    _CODE_VERB_TRIPLE_MISMATCH,
)


# --------------------------------------------------------------------- #
# Unit: the asymmetric helper itself.
# --------------------------------------------------------------------- #

def test_misaligned_only_fires_on_under_delivery():
    # Strictly below the objective band → genuine under-delivery → fire.
    assert verb_triple_misaligned("evaluate", "apply") is True
    assert verb_triple_misaligned("evaluate", "remember") is True
    assert verb_triple_misaligned("apply", "understand") is True


def test_misaligned_tolerates_equal_band():
    # At the band → aligned → no fire.
    assert verb_triple_misaligned("apply", "apply") is False
    assert verb_triple_misaligned("evaluate", "evaluate") is False


def test_misaligned_tolerates_overshoot():
    # ABOVE the objective band → over-shoot → aligned → no fire.
    assert verb_triple_misaligned("apply", "evaluate") is False
    assert verb_triple_misaligned("remember", "create") is False
    assert verb_triple_misaligned("understand", "apply") is False


def test_misaligned_none_safe():
    assert verb_triple_misaligned(None, "apply") is False
    assert verb_triple_misaligned("apply", None) is False
    assert verb_triple_misaligned("bogus", "apply") is False


def test_verbs_share_band_unchanged_strict():
    # The positive-polarity band check (consumed by triangle_completeness for
    # "is this assessment band-aligned") MUST stay strict equality — over-shoot
    # is not the same band.
    assert verbs_share_band("apply", "apply") is True
    assert verbs_share_band("apply", "evaluate") is False
    assert verbs_share_band("evaluate", "apply") is False


# --------------------------------------------------------------------- #
# Unit: declared-bloom anchor beats the inflating prose scan.
# --------------------------------------------------------------------- #

def test_declared_bloom_in_content_beats_math_idiom():
    # "Evaluate the expression to find the sum" is arithmetic (Apply), not
    # Bloom Evaluate. With a declared bloom_level the resolver uses it.
    block = {
        "block_type": "self_check_question",
        "content": {
            "bloom_level": "understand",
            "text": "Evaluate the expression to find the sum.",
        },
    }
    _, level = resolve_block_verb_level(block)
    assert level == "understand"


def test_declared_bloom_top_level_attr_beats_idiom():
    block = Block(
        block_id="p#activity_d_0", block_type="activity", page_id="p",
        sequence=0, content="Create a number line and plot the integers.",
        bloom_level="apply",
    )
    _, level = resolve_block_verb_level(block)
    assert level == "apply"  # NOT "create" from the idiom prose scan


def test_prose_fallback_when_no_declared_bloom():
    # No declared bloom anywhere → prose scan still runs (last resort). The
    # idiom inflation it produces is tolerated by verb_triple_misaligned, not
    # by demoting it here.
    block = Block(
        block_id="p#activity_p_0", block_type="activity", page_id="p",
        sequence=0, content="Solve the equation for x.",
    )
    _, level = resolve_block_verb_level(block)
    assert level == "apply"


# --------------------------------------------------------------------- #
# Integration: the gate fires on under-delivery, not over-shoot.
# --------------------------------------------------------------------- #

def _validate(blocks, objectives, monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    res = BlockObjectiveDeliveryValidator(nli=None).validate({
        "blocks": blocks,
        "objectives": objectives,
        "objective_statements": {
            oid: o.get("statement", "") for oid, o in objectives.items()
        },
        "return_touched_blocks": True,
    })
    return res


def _codes(res):
    return [i.code for i in res.issues]


def test_gate_does_not_fire_on_overshoot(monkeypatch):
    """A block teaching ABOVE its objective band must NOT fire (keystone-safe).

    Objective = Apply; block resolves to Evaluate (the math-idiom inflation
    case). Pre-recalibration this fired as a strict-band-equality FP. Now it
    is aligned (over-shoot).
    """
    objectives = {
        "CO-01": {
            "id": "CO-01", "bloom_level": "apply", "bloom_verb": "solve",
            "statement": "Apply the method",
            "abcd": {"behavior": {"verb": "solve"}},
        },
    }
    # No declared bloom on the block → prose scan inflates to "evaluate".
    block = Block(
        block_id="p#activity_o_0", block_type="activity", page_id="p",
        sequence=0, content="Evaluate the expression to judge the result.",
        objective_ids=("CO-01",),
    )
    res = _validate([block], objectives, monkeypatch)
    # Sanity: the block really did resolve above the objective band.
    blk_level = resolve_block_verb_level(block)[1]
    assert blk_level == "evaluate"
    # Keystone-safety: over-shoot must NOT fire.
    assert _CODE_VERB_TRIPLE_MISMATCH not in _codes(res)


def test_gate_fires_on_under_delivery(monkeypatch):
    """A block teaching strictly BELOW its objective band MUST still fire.

    Coverage of the genuine constructive-alignment defect must NOT regress:
    an Evaluate objective "checked" by a Remember (list) block fails.
    """
    objectives = {
        "CO-01": {
            "id": "CO-01", "bloom_level": "evaluate", "bloom_verb": "critique",
            "statement": "Critique the argument",
            "abcd": {"behavior": {"verb": "critique"}},
        },
    }
    block = Block(
        block_id="p#activity_u_0", block_type="activity", page_id="p",
        sequence=0, content="List the steps of the process.",
        objective_ids=("CO-01",),
    )
    res = _validate([block], objectives, monkeypatch)
    assert resolve_block_verb_level(block)[1] == "remember"
    assert _CODE_VERB_TRIPLE_MISMATCH in _codes(res)


def test_gate_does_not_fire_on_equal_band(monkeypatch):
    objectives = {
        "CO-01": {
            "id": "CO-01", "bloom_level": "apply", "bloom_verb": "solve",
            "statement": "Apply the method",
            "abcd": {"behavior": {"verb": "solve"}},
        },
    }
    block = Block(
        block_id="p#activity_e_0", block_type="activity", page_id="p",
        sequence=0, content="Solve the practice equation by applying the rule.",
        objective_ids=("CO-01",),
    )
    res = _validate([block], objectives, monkeypatch)
    assert _CODE_VERB_TRIPLE_MISMATCH not in _codes(res)


def test_overshoot_status_aligned_in_alignment_entry(monkeypatch):
    """The persisted verb_triple_status for an over-shoot pair is 'aligned'."""
    objectives = {
        "CO-01": {
            "id": "CO-01", "bloom_level": "understand", "bloom_verb": "explain",
            "statement": "Explain the concept",
            "abcd": {"behavior": {"verb": "explain"}},
        },
    }
    block = Block(
        block_id="p#self_check_question_o_0", block_type="self_check_question",
        page_id="p", sequence=0,
        content="Evaluate the two arguments and judge which is stronger.",
        objective_ids=("CO-01",),
    )
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    inputs = {
        "blocks": [block], "objectives": objectives,
        "objective_statements": {"CO-01": "Explain the concept"},
        "return_touched_blocks": True,
    }
    res = BlockObjectiveDeliveryValidator(nli=None).validate(inputs)
    assert _CODE_VERB_TRIPLE_MISMATCH not in [i.code for i in res.issues]
    ali = inputs["_touched_blocks"][0].objective_alignment[0]
    assert ali["verb_triple_status"] == "aligned"
    assert ali["block_bloom"] == "evaluate"
    assert ali["objective_bloom"] == "understand"


def test_prose_only_cohort_overshoot_tolerated_under_delivery_fires(monkeypatch):
    """cal2 finding — prose-only blocks (no declared bloom) are the real corpus
    shape, and the over-shoot tolerance carries the FP reduction there.

    cal2 (66-block sample-algebra rewrite) blocks carry NO declared bloom
    anywhere — ``content`` is an HTML string, so ``resolve_block_verb_level``
    always falls through to the math-idiom-vulnerable prose scan. On that
    cohort 29 audited verb-triple pairs resolved ABOVE the objective band
    (over-shoot, now ALIGNED) and only the genuine under-delivery pairs fire.
    This pins that mixed shape: an over-shoot prose block does NOT fire while a
    co-located strict under-delivery block DOES — so the recalibration's FP
    reduction holds on the production cohort, not just on declared-bloom
    fixtures.
    """
    objectives = {
        # Objective band Apply — an over-shoot prose block resolving to Evaluate
        # must be tolerated.
        "CO-OS": {
            "id": "CO-OS", "bloom_level": "apply", "bloom_verb": "solve",
            "statement": "Apply the rule",
            "abcd": {"behavior": {"verb": "solve"}},
        },
        # Objective band Evaluate — an under-delivering Remember prose block
        # must still fire.
        "CO-UD": {
            "id": "CO-UD", "bloom_level": "evaluate", "bloom_verb": "critique",
            "statement": "Critique the result",
            "abcd": {"behavior": {"verb": "critique"}},
        },
    }
    overshoot = Block(
        block_id="week_01_self_check#self_check_question_os",
        block_type="self_check_question", page_id="week_01_self_check",
        sequence=0,
        content="Evaluate the two strategies and judge which is stronger.",
        objective_ids=("CO-OS",),
    )
    underdeliver = Block(
        block_id="week_02_self_check#self_check_question_ud",
        block_type="self_check_question", page_id="week_02_self_check",
        sequence=0,
        content="Identify the term and list its parts.",
        objective_ids=("CO-UD",),
    )
    # Neither block declares a bloom_level → both use the prose-scan path.
    assert resolve_block_verb_level(overshoot)[1] == "evaluate"
    assert resolve_block_verb_level(underdeliver)[1] == "remember"
    res = _validate([overshoot, underdeliver], objectives, monkeypatch)
    codes = _codes(res)
    # Exactly one verb-triple mismatch: the under-deliverer, not the over-shoot.
    assert codes.count(_CODE_VERB_TRIPLE_MISMATCH) == 1
    msgs = [i for i in res.issues if i.code == _CODE_VERB_TRIPLE_MISMATCH]
    assert msgs[0].location == "week_02_self_check#self_check_question_ud"


def test_assessment_under_delivery_still_sets_cap(monkeypatch):
    """Under-delivery on an assessment block still fires AND sets the
    alignment-cap-at-1 signal — the cap is the assessment-Bloom-below-objective
    contract and must NOT be weakened by the over-shoot tolerance."""
    objectives = {
        "CO-01": {
            "id": "CO-01", "bloom_level": "apply", "bloom_verb": "solve",
            "statement": "Apply the method",
            "abcd": {"behavior": {"verb": "solve"}},
        },
    }
    block = Block(
        block_id="p#assessment_item_u_0", block_type="assessment_item",
        page_id="p", sequence=0,
        content="Define the term and list its parts.",
        objective_ids=("CO-01",),
    )
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    inputs = {
        "blocks": [block], "objectives": objectives,
        "objective_statements": {"CO-01": "Apply the method"},
        "return_touched_blocks": True,
    }
    res = BlockObjectiveDeliveryValidator(nli=None).validate(inputs)
    assert _CODE_VERB_TRIPLE_MISMATCH in [i.code for i in res.issues]
    ali = inputs["_touched_blocks"][0].objective_alignment[0]
    assert ali["verb_triple_status"] == "mismatch"
    assert ali.get("alignment_cap_at_1") is True
