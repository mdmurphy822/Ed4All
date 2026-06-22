"""IB7.6c — tests for BloomTypeRangeValidator.

Confirms the per-type Bloom-range ceiling gate (with a passed-in
``bloom_ceilings`` override so the test is independent of the catalog gaining
``bloom_ceiling`` keys):

  * an over-ceiling block (concept @ analyze, ceiling apply) → FLAGS
    ``BLOCK_BLOOM_OVER_CEILING`` (warning, passed=False, action="regenerate");
  * an in-range block (concept @ understand, ceiling apply) → PASS;
  * a block whose type has no ceiling → skip-PASS;
  * a block whose Bloom is unresolvable → skip-PASS;
  * empty input → graceful PASS;
  * the ``block_validation_action`` decision-capture fires.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from Courseforge.scripts.blocks import Block
from lib.validators.bloom_type_range import BloomTypeRangeValidator


class _SpyCapture:
    """DecisionCapture-shaped spy collecting every emitted event."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        **kw: Any,
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kw,
            }
        )


# A test-local ceiling map (independent of the catalog).
_CEILINGS = {"concept": "apply", "explanation": "apply", "scenario": "evaluate"}


def _block(
    block_type: str,
    *,
    target_bloom: Optional[str] = None,
    bloom_level: Optional[str] = None,
    block_id: str = "page#b1",
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page",
        sequence=0,
        content="<p>body</p>",
        target_bloom=target_bloom,
        bloom_level=bloom_level,
    )


def _validate(
    blocks: List[Block],
    capture: Optional[_SpyCapture] = None,
    ceilings: Optional[Dict[str, str]] = _CEILINGS,
):
    v = BloomTypeRangeValidator()
    inputs: Dict[str, Any] = {"blocks": blocks}
    if ceilings is not None:
        inputs["bloom_ceilings"] = ceilings
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Over-ceiling → FLAGS
# ---------------------------------------------------------------------------


def test_over_ceiling_block_flags():
    capture = _SpyCapture()
    blocks = [_block("concept", target_bloom="analyze")]
    result = _validate(blocks, capture)

    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "BLOCK_BLOOM_OVER_CEILING" in codes
    issue = result.issues[0]
    assert issue.severity == "warning"
    assert "analyze" in issue.message
    assert "apply" in issue.message


def test_over_ceiling_via_bloom_level_fallback():
    # No target_bloom → resolve via bloom_level; still over ceiling.
    blocks = [_block("concept", bloom_level="evaluate")]
    result = _validate(blocks)
    assert result.passed is False
    assert any(i.code == "BLOCK_BLOOM_OVER_CEILING" for i in result.issues)


# ---------------------------------------------------------------------------
# In-range → PASS
# ---------------------------------------------------------------------------


def test_in_range_block_passes():
    capture = _SpyCapture()
    blocks = [_block("concept", target_bloom="understand")]
    result = _validate(blocks, capture)
    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_at_ceiling_passes():
    # At-ceiling (== not >) is in range.
    blocks = [_block("concept", target_bloom="apply")]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# No ceiling for type → skip-PASS
# ---------------------------------------------------------------------------


def test_type_without_ceiling_skip_passes():
    # assessment_item is not in _CEILINGS → no ceiling → skip-pass even at
    # create.
    blocks = [_block("assessment_item", target_bloom="create")]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Unresolvable Bloom → skip-PASS
# ---------------------------------------------------------------------------


def test_unresolvable_bloom_skip_passes():
    blocks = [_block("concept")]  # no target_bloom, no bloom_level
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


def test_invalid_target_bloom_falls_back_then_unresolvable():
    blocks = [_block("concept", target_bloom="not-a-level")]
    result = _validate(blocks)
    # target_bloom invalid, bloom_level absent → unresolvable → skip-pass.
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Empty input → graceful PASS
# ---------------------------------------------------------------------------


def test_empty_input_graceful_pass():
    result = _validate([])
    assert result.passed is True
    assert result.action is None
    assert result.issues == []


# ---------------------------------------------------------------------------
# Decision capture fires (only for ceiling-carrying blocks)
# ---------------------------------------------------------------------------


def test_decision_capture_fires_on_over_ceiling():
    capture = _SpyCapture()
    _validate([_block("concept", target_bloom="analyze")], capture)

    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == "block_validation_action"
    assert ev["decision"] == "over_ceiling"
    assert len(ev["rationale"]) >= 20
    feats = ev["ml_features"]
    assert feats["block_id"] == "page#b1"
    assert feats["block_type"] == "concept"
    assert feats["observed_bloom"] == "analyze"
    assert feats["bloom_ceiling"] == "apply"
    assert feats["over_ceiling"] is True
    assert feats["passed"] is False


def test_decision_capture_not_emitted_for_no_ceiling_type():
    capture = _SpyCapture()
    _validate([_block("assessment_item", target_bloom="create")], capture)
    assert capture.events == []


def test_constructor_ceiling_override_used():
    # The constructor kwarg is used when inputs carry no override.
    v = BloomTypeRangeValidator(bloom_ceilings={"concept": "remember"})
    result = v.validate({"blocks": [_block("concept", target_bloom="apply")]})
    assert result.passed is False
    assert any(i.code == "BLOCK_BLOOM_OVER_CEILING" for i in result.issues)
