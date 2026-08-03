"""Unit tests for the SIBLING-PAGE fallback tier of the outline CURIE minter.

The minter (``MCP/tools/pipeline_tools.py::_build_outline_curie_minter``)
mints per-course domain CURIEs onto curie-less outline blocks by matching
vocabulary surface forms against (1) key_claims, then (2) source-chunk text,
then (3) objective-refs statements. A handful of enriched-page-plan support
blocks (callout / flip_card_grid / extra example slots) miss ALL THREE
surfaces. The third fallback tier — ``sibling_fallback_sweep`` — lets such a
block inherit the MODAL minted domain CURIE(s) (capped at 2) from its
same-page siblings that DID end up with a minted domain CURIE. Grounded
inheritance (shared page chunk universe), not fabrication; still fail-closed
when no sibling carries a domain CURIE.

Covered:
- (a) inheritance fires for a claims-miss + chunks-miss block when a sibling
  minted;
- (b) modal selection with the cap of 2 (+ deterministic lexicographic
  tie-break);
- (c) no sibling with a domain CURIE -> no mint (fail closed);
- (d) order-independence: a candidate EARLIER in the list than its minted
  sibling still inherits (second sweep observes the final state);
- (e) the ``curie_minting`` decision capture fires with the
  ``sibling_page_fallback`` marker;
- (f) blocks already carrying a domain CURIE are left untouched;
- the str-content (rewrite-tier) path is NOT touched by the sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_outline_curie_minter,
    _emit_curie_minting_capture,
    _mint_outline_curies,
)
from blocks import Block  # noqa: E402


_VOCAB = {
    "course_id": "introbio101",
    "concepts": [
        {"canonical": "slope", "aliases": ["gradient", "steepness"]},
        {"canonical": "intercept", "aliases": ["y-intercept"]},
        {"canonical": "function", "aliases": ["mapping"]},
    ],
}


def _minter(tmp_path):
    vocab_path = tmp_path / "domain_concept_vocabulary.json"
    vocab_path.write_text(json.dumps(_VOCAB), encoding="utf-8")
    kwargs = {
        "phase_outputs": {
            "concept_extraction": {
                "domain_concept_vocabulary_path": str(vocab_path),
            }
        }
    }
    minter = _build_outline_curie_minter(
        course_code="introbio101", kwargs=kwargs,
    )
    assert minter is not None
    return minter


def _blk(block_id, *, curies, key_claims=None, page_id=None):
    return Block(
        block_id=block_id,
        block_type="callout",
        page_id=page_id or block_id.split("#", 1)[0],
        sequence=0,
        content={
            "curies": list(curies),
            "key_claims": list(key_claims or []),
            "content_type": "definition",
        },
        objective_ids=("CO-01",),
    )


def _curie_for(minter, text):
    """Mint a probe block whose key_claims mentions a concept and return the
    single minted domain CURIE — a robust way to learn the concept's real
    per-course CURIE string without hardcoding the slug scheme."""
    probe = _blk("probe_page#concept_probe_0", curies=[], key_claims=[text])
    nb, meta = minter.mint_block(probe, None)
    assert nb is not None, f"probe failed to mint for {text!r}"
    return nb.content["curies"][0]


# ---------------------------------------------------------------------------
# (a) inheritance fires for a claims-miss + chunks-miss block.
# ---------------------------------------------------------------------------


def test_sibling_fallback_fires_when_sibling_minted(tmp_path):
    minter = _minter(tmp_path)
    slope_curie = _curie_for(minter, "The slope of a line measures steepness.")

    # Sibling on the same page carries a minted domain CURIE; the support
    # block matches no vocab surface form anywhere.
    sibling = _blk("week_01_content_01#concept_slope_0", curies=[slope_curie])
    support = _blk(
        "week_01_content_01#callout_eval_1",
        curies=[],
        key_claims=["Evaluate the expression: 3x + 4 when x = -2"],
    )
    # The support block indeed misses all vocab surfaces on its own.
    assert minter.mint_block(support, None) == (None, None)

    blocks = [sibling, support]
    results = minter.sibling_fallback_sweep(blocks)
    assert len(results) == 1
    idx, new_block, meta = results[0]
    assert idx == 1
    assert new_block.content["curies"] == [slope_curie]
    assert meta["matched_surface"] == "sibling_page_fallback"
    assert meta["minted_curies"] == [slope_curie]
    assert meta["sibling_count"] == 1


# ---------------------------------------------------------------------------
# (b) modal selection with the cap of 2 + lexicographic tie-break.
# ---------------------------------------------------------------------------


def test_modal_selection_cap_two_and_lexical_tiebreak(tmp_path):
    minter = _minter(tmp_path)
    slope_curie = _curie_for(minter, "The slope of a line measures steepness.")
    intercept_curie = _curie_for(minter, "The y-intercept is where it crosses.")
    function_curie = _curie_for(minter, "A function maps inputs to outputs.")
    assert len({slope_curie, intercept_curie, function_curie}) == 3

    # slope appears on TWO siblings (modal); intercept + function once each
    # (tied at count 1). Cap of 2 -> [slope, lexicographically-first of the
    # tied pair].
    siblings = [
        _blk("week_02_content_01#concept_slope_0", curies=[slope_curie]),
        _blk("week_02_content_01#concept_slope_1", curies=[slope_curie]),
        _blk("week_02_content_01#concept_int_2", curies=[intercept_curie]),
        _blk("week_02_content_01#concept_fn_3", curies=[function_curie]),
    ]
    support = _blk(
        "week_02_content_01#callout_x_4",
        curies=[],
        key_claims=["Try this practice problem on your own."],
    )
    blocks = siblings + [support]
    results = minter.sibling_fallback_sweep(blocks)
    assert len(results) == 1
    _, new_block, meta = results[0]
    inherited = new_block.content["curies"]
    assert len(inherited) == 2  # cap of 2
    assert inherited[0] == slope_curie  # highest frequency
    # Second slot is the lexicographically-first of the count-1 tie.
    tied_low = sorted([intercept_curie, function_curie])[0]
    assert inherited[1] == tied_low
    assert meta["minted_curies"] == inherited
    assert meta["sibling_count"] == 4


# ---------------------------------------------------------------------------
# (c) no sibling with a domain CURIE -> no mint (fail closed).
# ---------------------------------------------------------------------------


def test_no_sibling_domain_curie_no_mint(tmp_path):
    minter = _minter(tmp_path)
    # A page whose only other block also carries no domain CURIE (just a
    # generic token). Nothing to inherit -> fail closed.
    other = _blk(
        "week_03_content_01#text_intro_0",
        curies=["outline:sample"],
        key_claims=["Nothing in the vocabulary here."],
    )
    support = _blk(
        "week_03_content_01#callout_y_1",
        curies=[],
        key_claims=["Evaluate the expression: 3x + 4 when x = -2"],
    )
    results = minter.sibling_fallback_sweep([other, support])
    assert results == []


# ---------------------------------------------------------------------------
# (d) order-independence: candidate BEFORE its minted sibling still inherits.
# ---------------------------------------------------------------------------


def test_order_independence_candidate_before_sibling(tmp_path):
    minter = _minter(tmp_path)
    slope_curie = _curie_for(minter, "The slope of a line measures steepness.")

    support = _blk(
        "week_04_content_01#callout_eval_0",
        curies=[],
        key_claims=["Evaluate the expression: 3x + 4 when x = -2"],
    )
    sibling = _blk("week_04_content_01#concept_slope_1", curies=[slope_curie])
    # Candidate EARLIER in the list than the minted sibling.
    blocks = [support, sibling]
    results = minter.sibling_fallback_sweep(blocks)
    assert len(results) == 1
    idx, new_block, _ = results[0]
    assert idx == 0
    assert new_block.content["curies"] == [slope_curie]


# ---------------------------------------------------------------------------
# (e) capture fires with the sibling_page_fallback marker.
# ---------------------------------------------------------------------------


class _RecordingCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def _libv2_vocab_tree(tmp_path):
    """Write the vocab under a fake LibV2 course tree so _mint_outline_curies
    can auto-locate it via libv2_root."""
    course_dir = tmp_path / "courses" / "fxdemo-101" / "concept_graph"
    course_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "schema_version": "v1",
        "course_id": "FXDEMO_101",
        "course_slug": "fxdemo-101",
        "concept_count": 2,
        "concepts": [
            {"canonical": "slope", "aliases": ["gradient"]},
            {"canonical": "intercept", "aliases": ["y-intercept"]},
        ],
    }
    (course_dir / "domain_concept_vocabulary.json").write_text(
        json.dumps(vocab), encoding="utf-8",
    )


def test_capture_fires_with_sibling_fallback_marker(tmp_path):
    _libv2_vocab_tree(tmp_path)
    kwargs = {"libv2_root": str(tmp_path)}
    # Sibling whose key_claims mint slope; support block that matches nothing.
    sibling = _blk(
        "week_01_content_01#concept_slope_0",
        curies=[],
        key_claims=["The slope of a line measures its steepness."],
    )
    support = _blk(
        "week_01_content_01#callout_eval_1",
        curies=[],
        key_claims=["Evaluate the expression: 3x + 4 when x = -2"],
    )
    blocks = [sibling, support]
    capture = _RecordingCapture()
    _mint_outline_curies(
        outline_blocks=blocks,
        course_code="FXDEMO_101",
        kwargs=kwargs,
        capture=capture,
    )
    # The support block inherited a domain CURIE via the sibling fallback.
    assert blocks[1].content["curies"], "support block should have inherited"
    fallback_calls = [
        c for c in capture.calls
        if c.get("ml_features", {}).get("matched_surface")
        == "sibling_page_fallback"
    ]
    assert len(fallback_calls) == 1
    call = fallback_calls[0]
    assert call["decision_type"] == "curie_minting"
    assert len(call["rationale"]) >= 20
    # Dynamic rationale interpolates the sibling count + inherited CURIEs.
    inherited = blocks[1].content["curies"]
    assert str(inherited) in call["rationale"] or all(
        c in call["rationale"] for c in inherited
    )
    assert "sibling" in call["rationale"].lower()
    assert call["ml_features"]["sibling_donor_count"] == 1
    # Capture-quality contract (proficient floor): the fallback event
    # references the real inputs it consumed (the block + the donor
    # siblings) and the genuine do-nothing alternative (leave the block
    # un-minted and fail the anchoring gate).
    inputs_ref = call["inputs_ref"]
    assert {
        "source_type": "block",
        "path_or_id": "week_01_content_01#callout_eval_1",
    } in inputs_ref
    assert any(
        r.get("source_type") == "sibling_page_blocks" for r in inputs_ref
    )
    alternatives = call["alternatives_considered"]
    assert alternatives
    assert any("BlockCurieAnchoringValidator" in a for a in alternatives)

    # The sibling's own (non-fallback) mint event references the actual
    # matched vocabulary concept(s) as inputs.
    direct_calls = [
        c for c in capture.calls
        if c.get("ml_features", {}).get("matched_surface")
        != "sibling_page_fallback"
    ]
    assert direct_calls, "sibling's own mint should also have fired"
    direct = direct_calls[0]
    assert any(
        r.get("source_type") == "domain_concept"
        for r in direct["inputs_ref"]
    )
    assert direct["alternatives_considered"]


# ---------------------------------------------------------------------------
# (f) blocks already carrying a domain CURIE are left untouched.
# ---------------------------------------------------------------------------


def test_block_with_domain_curie_untouched(tmp_path):
    minter = _minter(tmp_path)
    slope_curie = _curie_for(minter, "The slope of a line measures steepness.")
    intercept_curie = _curie_for(minter, "The y-intercept is where it crosses.")

    already = _blk("week_05_content_01#concept_a_0", curies=[intercept_curie])
    donor = _blk("week_05_content_01#concept_b_1", curies=[slope_curie])
    results = minter.sibling_fallback_sweep([already, donor])
    # Neither the already-anchored block nor the donor is re-processed.
    assert results == []


# ---------------------------------------------------------------------------
# str-content (rewrite-tier) path is NOT touched by the sweep.
# ---------------------------------------------------------------------------


def test_str_content_block_not_processed(tmp_path):
    minter = _minter(tmp_path)
    slope_curie = _curie_for(minter, "The slope of a line measures steepness.")
    donor = _blk("week_06_content_01#concept_a_0", curies=[slope_curie])
    # A rewrite-tier block whose content is a raw HTML string, not a dict.
    str_block = Block(
        block_id="week_06_content_01#callout_html_1",
        block_type="callout",
        page_id="week_06_content_01",
        sequence=1,
        content="<div>Evaluate the expression: 3x + 4</div>",
        objective_ids=("CO-01",),
    )
    results = minter.sibling_fallback_sweep([donor, str_block])
    # The str-content block is skipped (dict-path fallback only).
    assert results == []
