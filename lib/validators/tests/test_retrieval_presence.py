"""IB7.5b — tests for RetrievalPresenceValidator.

Confirms the spacing-aware retrieval-presence floor:

  * a content module with ZERO retrieval blocks → FAIL (warning
    ``MODULE_NO_RETRIEVAL``, passed=False, action="regenerate");
  * a content module with a SPACED self_check_question (concept first, then
    self_check_question) → PASS;
  * a retrieval block that is the FIRST block on the page → FLAGS
    ``RETRIEVAL_UNSPACED``;
  * empty input → graceful PASS;
  * a page with no content-bearing blocks → graceful PASS (not a module);
  * the ``content_selection`` decision-capture fires.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from Courseforge.scripts.blocks import Block
from lib.validators.retrieval_presence import RetrievalPresenceValidator


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


def _block(
    block_type: str,
    sequence: int,
    *,
    page_id: str = "page1",
    content: str = "<p>body</p>",
) -> Block:
    return Block(
        block_id=f"{page_id}#{block_type}_{sequence}",
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content=content,
    )


def _validate(blocks: List[Block], capture: Optional[_SpyCapture] = None):
    v = RetrievalPresenceValidator()
    inputs: Dict[str, Any] = {"blocks": blocks}
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Zero retrieval blocks → FAIL
# ---------------------------------------------------------------------------


def test_content_module_zero_retrieval_fails():
    capture = _SpyCapture()
    blocks = [
        _block("concept", 0),
        _block("example", 1),
        _block("explanation", 2),
    ]
    result = _validate(blocks, capture)

    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "MODULE_NO_RETRIEVAL" in codes
    issue = next(i for i in result.issues if i.code == "MODULE_NO_RETRIEVAL")
    assert issue.severity == "warning"


# ---------------------------------------------------------------------------
# Spaced retrieval → PASS
# ---------------------------------------------------------------------------


def test_spaced_self_check_passes():
    capture = _SpyCapture()
    blocks = [
        _block("concept", 0),
        _block("explanation", 1),
        _block("self_check_question", 2),
    ]
    result = _validate(blocks, capture)

    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_spaced_reflection_prompt_passes():
    blocks = [
        _block("concept", 0),
        _block("reflection_prompt", 1),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Retrieval block first on page → UNSPACED
# ---------------------------------------------------------------------------


def test_retrieval_first_block_flags_unspaced():
    blocks = [
        _block("self_check_question", 0),
        _block("concept", 1),
        _block("explanation", 2),
    ]
    result = _validate(blocks)
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "RETRIEVAL_UNSPACED" in codes
    issue = next(i for i in result.issues if i.code == "RETRIEVAL_UNSPACED")
    assert issue.severity == "warning"


def test_retrieval_first_but_second_retrieval_spaced_passes():
    # First block is retrieval (unspaced), but a SECOND retrieval block sits
    # spaced after the exposition → the floor is satisfied.
    blocks = [
        _block("self_check_question", 0),
        _block("concept", 1),
        _block("self_check_question", 2),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Empty / no-content input → graceful PASS
# ---------------------------------------------------------------------------


def test_empty_input_graceful_pass():
    result = _validate([])
    assert result.passed is True
    assert result.action is None
    assert result.issues == []


def test_no_content_bearing_blocks_graceful_pass():
    # A page carrying only navigation / retrieval blocks (no content set
    # member) is not a content-bearing module → not audited.
    blocks = [
        _block("self_check_question", 0),
        _block("reflection_prompt", 1),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Multi-page grouping
# ---------------------------------------------------------------------------


def test_per_page_grouping_isolates_modules():
    # page1 is good (spaced check); page2 is missing retrieval → only page2
    # flags.
    blocks = [
        _block("concept", 0, page_id="page1"),
        _block("self_check_question", 1, page_id="page1"),
        _block("concept", 0, page_id="page2"),
        _block("example", 1, page_id="page2"),
    ]
    result = _validate(blocks)
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert codes == ["MODULE_NO_RETRIEVAL"]
    issue = result.issues[0]
    assert "page2" in issue.message


# ---------------------------------------------------------------------------
# Decision capture fires
# ---------------------------------------------------------------------------


def test_decision_capture_fires_on_audit():
    capture = _SpyCapture()
    blocks = [
        _block("concept", 0),
        _block("explanation", 1),
        _block("self_check_question", 2),
    ]
    _validate(blocks, capture)

    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == "content_selection"
    assert ev["decision"] == "spaced_retrieval_present"
    assert len(ev["rationale"]) >= 20
    feats = ev["ml_features"]
    assert feats["page_id"] == "page1"
    assert feats["n_content_blocks"] == 2
    assert feats["n_retrieval_blocks"] == 1
    assert feats["spaced"] is True
    assert feats["passed"] is True


def test_decision_capture_fires_on_miss():
    capture = _SpyCapture()
    blocks = [_block("concept", 0), _block("example", 1)]
    _validate(blocks, capture)

    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision"] == "retrieval_floor_miss"
    feats = ev["ml_features"]
    assert feats["n_retrieval_blocks"] == 0
    assert feats["spaced"] is False
    assert feats["passed"] is False
