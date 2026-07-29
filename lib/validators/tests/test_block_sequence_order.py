"""FR-COURSE-02 — tests for BlockSequenceOrderValidator.

Confirms the pedagogical block-sequence-order floor (cloned from
RetrievalPresenceValidator — no flag, runs warning-day-1):

  * guided practice before worked_example → ``WORKED_BEFORE_GUIDED_OUT_OF_ORDER``;
  * worked_example before guided practice → PASS;
  * a check massed against same-objective exposition → ``CHECK_NOT_SPACED``;
  * a spaced check → PASS;
  * a scenario opening a module → ``SCENARIO_OPENS_TO``;
  * a scenario after exposition → PASS;
  * empty input / no content-bearing blocks → graceful PASS;
  * the ``content_selection`` decision-capture fires.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from Courseforge.scripts.blocks import Block
from lib.validators.block_sequence_order import BlockSequenceOrderValidator


class _SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **kw: Any
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
    objective_ids=(),
    content: str = "<p>body</p>",
) -> Block:
    return Block(
        block_id=f"{page_id}#{block_type}_{sequence}",
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content=content,
        objective_ids=tuple(objective_ids),
    )


def _validate(blocks: List[Block], capture: Optional[_SpyCapture] = None):
    v = BlockSequenceOrderValidator()
    inputs: Dict[str, Any] = {"blocks": blocks}
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Worked-example / guided-practice order
# ---------------------------------------------------------------------------


def test_guided_before_worked_flags():
    blocks = [
        _block("concept", 0),
        _block("problem", 1),          # guided practice FIRST
        _block("worked_example", 2),   # modelling AFTER → out of order
    ]
    result = _validate(blocks)
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "WORKED_BEFORE_GUIDED_OUT_OF_ORDER" in codes
    issue = next(
        i for i in result.issues if i.code == "WORKED_BEFORE_GUIDED_OUT_OF_ORDER"
    )
    assert issue.severity == "warning"


def test_worked_before_guided_passes():
    blocks = [
        _block("concept", 0),
        _block("worked_example", 1),
        _block("problem", 2),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


def test_guided_before_another_objectives_worked_example_not_flagged():
    """A page holds several independent objective sequences.

    CO-01 is taught then practised; CO-02 is taught, MODELLED, then practised.
    Both fade gradients are correct, but CO-01's practice sits earlier on the
    page than CO-02's worked_example. A positional reading calls that
    out-of-order; the fade gradient is per skill, so it is not.
    """
    blocks = [
        _block("concept", 0, objective_ids=("CO-01",)),
        _block("example", 1, objective_ids=("CO-01",)),
        _block("activity", 2, objective_ids=("CO-01",)),
        _block("concept", 3, objective_ids=("CO-02",)),
        _block("worked_example", 4, objective_ids=("CO-02",)),
        _block("activity", 5, objective_ids=("CO-02",)),
    ]
    result = _validate(blocks)
    codes = [i.code for i in result.issues]
    assert "WORKED_BEFORE_GUIDED_OUT_OF_ORDER" not in codes


def test_guided_before_its_own_objectives_worked_example_flags():
    """The same objective IS practised before it is modelled — a real miss."""
    blocks = [
        _block("concept", 0, objective_ids=("CO-01",)),
        _block("activity", 1, objective_ids=("CO-01",)),
        _block("worked_example", 2, objective_ids=("CO-01",)),
    ]
    result = _validate(blocks)
    assert result.passed is False
    assert "WORKED_BEFORE_GUIDED_OUT_OF_ORDER" in [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# Check spacing
# ---------------------------------------------------------------------------


def test_check_massed_against_same_objective_exposition_flags():
    blocks = [
        _block("concept", 0, objective_ids=("CO-01",)),
        _block("self_check_question", 1, objective_ids=("CO-01",)),  # massed
    ]
    result = _validate(blocks)
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "CHECK_NOT_SPACED" in codes


def test_check_spaced_passes():
    # An intervening block teaching a DIFFERENT objective sits between the
    # CO-01 exposition and the CO-01 check → spaced (not massed).
    blocks = [
        _block("concept", 0, objective_ids=("CO-01",)),
        _block("explanation", 1, objective_ids=("CO-02",)),  # intervening
        _block("self_check_question", 2, objective_ids=("CO-01",)),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


def test_check_different_objective_not_flagged():
    # A check immediately after exposition for a DIFFERENT objective is not a
    # massed-retrieval violation.
    blocks = [
        _block("concept", 0, objective_ids=("CO-01",)),
        _block("self_check_question", 1, objective_ids=("CO-02",)),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Scenario opens TO
# ---------------------------------------------------------------------------


def test_scenario_opens_module_flags():
    blocks = [
        _block("scenario", 0),
        _block("concept", 1),
    ]
    result = _validate(blocks)
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "SCENARIO_OPENS_TO" in codes


def test_scenario_after_exposition_passes():
    blocks = [
        _block("concept", 0),
        _block("explanation", 1),
        _block("scenario", 2),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Empty / no-content → graceful PASS
# ---------------------------------------------------------------------------


def test_empty_input_graceful_pass():
    result = _validate([])
    assert result.passed is True
    assert result.issues == []
    assert result.action is None


def test_no_content_bearing_blocks_graceful_pass():
    blocks = [
        _block("self_check_question", 0),
        _block("reflection_prompt", 1),
    ]
    result = _validate(blocks)
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# Decision capture fires
# ---------------------------------------------------------------------------


def test_decision_capture_fires():
    capture = _SpyCapture()
    blocks = [
        _block("concept", 0),
        _block("worked_example", 1),
        _block("problem", 2),
    ]
    _validate(blocks, capture)
    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == "content_selection"
    assert ev["decision"] == "block_sequence_ok"
    assert len(ev["rationale"]) >= 20
    feats = ev["ml_features"]
    assert feats["passed"] is True
