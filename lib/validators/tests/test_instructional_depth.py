"""Worker W7 — tests for InstructionalDepthValidator.

Per the W7 spec: per-page pedagogical-density floors —
``min_concepts_per_page=2``, ``min_examples_per_concept=1.0``,
``min_explanation_tokens_per_concept=80``. Each below-threshold metric
fires an ``INSTRUCTIONAL_DEPTH_<METRIC>_BELOW_THRESHOLD`` critical
GateIssue, and exactly one ``instructional_depth_check`` decision
event lands per ``validate()`` call carrying the three metrics in
``ml_features``.

Fixtures construct ``Courseforge.scripts.blocks.Block`` instances at
the requested density so the validator's per-page metric computation
fires deterministically. No LLM in the loop — pure structural test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Repo root + Courseforge scripts dir on sys.path so ``from blocks``
# resolves against the canonical ``Courseforge/scripts/blocks.py``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.instructional_depth import (  # noqa: E402
    DEFAULT_MIN_CONCEPTS_PER_PAGE,
    DEFAULT_MIN_EXAMPLES_PER_CONCEPT,
    DEFAULT_MIN_EXPLANATION_TOKENS_PER_CONCEPT,
    InstructionalDepthValidator,
)


# ---------------------------------------------------------------------------
# Decision-capture stub
# ---------------------------------------------------------------------------


class _StubCapture:
    """Records every ``log_decision`` call as a tuple in ``calls``."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, str, Dict[str, Any]]] = []

    def log_decision(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        **kwargs: Any,
    ) -> None:
        self.calls.append((decision_type, decision, rationale, kwargs))


# ---------------------------------------------------------------------------
# Block fixtures
# ---------------------------------------------------------------------------


def _make_concept(
    *,
    page_id: str = "page_01",
    sequence: int = 0,
    body: str = "",
    block_id: Optional[str] = None,
) -> Block:
    """A ``concept`` Block whose body is an arbitrary token-rich
    sentence string. Defaults to empty body so callers can target the
    explanation-tokens floor without polluting their assertions."""
    bid = block_id or f"{page_id}#concept_demo_{sequence}"
    return Block(
        block_id=bid,
        block_type="concept",
        page_id=page_id,
        sequence=sequence,
        content={"body": body, "key_claims": []},
    )


def _make_example(
    *,
    page_id: str = "page_01",
    sequence: int = 0,
    block_id: Optional[str] = None,
) -> Block:
    bid = block_id or f"{page_id}#example_demo_{sequence}"
    return Block(
        block_id=bid,
        block_type="example",
        page_id=page_id,
        sequence=sequence,
        content={"body": "EXAMPLE: an illustrative scenario."},
    )


def _make_explanation(
    *,
    page_id: str = "page_01",
    sequence: int = 0,
    body: str = "",
    block_id: Optional[str] = None,
) -> Block:
    bid = block_id or f"{page_id}#explanation_demo_{sequence}"
    return Block(
        block_id=bid,
        block_type="explanation",
        page_id=page_id,
        sequence=sequence,
        content={"body": body},
    )


def _word_padding(n: int) -> str:
    """Generate a string with exactly ``n`` whitespace-separated word
    tokens. Used to drive the explanation-tokens-per-concept floor."""
    return " ".join(f"word{i}" for i in range(n))


# ---------------------------------------------------------------------------
# 1. Happy path — all three metrics above threshold.
# ---------------------------------------------------------------------------


def test_happy_path_three_concepts_four_examples_passes() -> None:
    """Page with 3 concepts, 4 examples, 100+ tokens of explanation
    per concept → passes; no critical issues; ``score == 1.0``."""
    blocks = [
        _make_concept(sequence=i, body=_word_padding(100))
        for i in range(3)
    ]
    blocks.extend(_make_example(sequence=10 + i) for i in range(4))

    result = InstructionalDepthValidator().validate({"blocks": blocks})

    assert result.passed is True
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert critical_codes == []
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# 2. Below-min-concepts — page with 1 concept fires the concepts code.
# ---------------------------------------------------------------------------


def test_below_min_concepts_per_page_fires_critical() -> None:
    """1 concept on a page (floor=2) →
    ``INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD``."""
    blocks = [
        _make_concept(sequence=0, body=_word_padding(120)),
        # One example so we don't double-fire the examples ratio code
        # (1 ex / 1 concept = 1.0 ratio = floor met).
        _make_example(sequence=10),
    ]

    result = InstructionalDepthValidator().validate({"blocks": blocks})

    assert result.passed is False
    codes = {i.code for i in result.issues if i.severity == "critical"}
    assert (
        "INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD" in codes
    )


# ---------------------------------------------------------------------------
# 3. Below examples-per-concept — 3 concepts with 1 example total.
# ---------------------------------------------------------------------------


def test_below_examples_per_concept_fires_critical() -> None:
    """3 concepts + 1 example total → ratio 0.33 < 1.0 floor →
    ``INSTRUCTIONAL_DEPTH_EXAMPLES_PER_CONCEPT_BELOW_THRESHOLD``."""
    blocks = [
        _make_concept(sequence=i, body=_word_padding(100))
        for i in range(3)
    ]
    blocks.append(_make_example(sequence=10))

    result = InstructionalDepthValidator().validate({"blocks": blocks})

    assert result.passed is False
    codes = {i.code for i in result.issues if i.severity == "critical"}
    assert (
        "INSTRUCTIONAL_DEPTH_EXAMPLES_PER_CONCEPT_BELOW_THRESHOLD"
        in codes
    )
    # Concepts + tokens floors are satisfied — this should be the only
    # critical code on the page.
    assert (
        "INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD"
        not in codes
    )
    assert (
        "INSTRUCTIONAL_DEPTH_EXPLANATION_TOKENS_PER_CONCEPT"
        "_BELOW_THRESHOLD"
    ) not in codes


# ---------------------------------------------------------------------------
# 4. Below explanation tokens — 3 concepts × 30 tokens each.
# ---------------------------------------------------------------------------


def test_below_explanation_tokens_per_concept_fires_critical() -> None:
    """3 concepts × 30 tokens each → 30 < 80 floor →
    ``INSTRUCTIONAL_DEPTH_EXPLANATION_TOKENS_PER_CONCEPT_BELOW_THRESHOLD``."""
    blocks = [
        _make_concept(sequence=i, body=_word_padding(30))
        for i in range(3)
    ]
    # 4 examples so the examples-per-concept floor is satisfied.
    blocks.extend(_make_example(sequence=10 + i) for i in range(4))

    result = InstructionalDepthValidator().validate({"blocks": blocks})

    assert result.passed is False
    codes = {i.code for i in result.issues if i.severity == "critical"}
    assert (
        "INSTRUCTIONAL_DEPTH_EXPLANATION_TOKENS_PER_CONCEPT"
        "_BELOW_THRESHOLD"
    ) in codes
    assert (
        "INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD"
        not in codes
    )
    assert (
        "INSTRUCTIONAL_DEPTH_EXAMPLES_PER_CONCEPT_BELOW_THRESHOLD"
        not in codes
    )


# ---------------------------------------------------------------------------
# 5. Capture-emit test — exactly one ``instructional_depth_check``
#    event with all three metrics on ``ml_features``.
# ---------------------------------------------------------------------------


def test_capture_emit_records_all_three_metrics() -> None:
    blocks = [
        _make_concept(sequence=0, body=_word_padding(120)),
        _make_concept(sequence=1, body=_word_padding(120)),
        _make_concept(sequence=2, body=_word_padding(120)),
        _make_example(sequence=10),
        _make_example(sequence=11),
        _make_example(sequence=12),
        _make_example(sequence=13),
    ]
    capture = _StubCapture()

    InstructionalDepthValidator().validate({
        "blocks": blocks,
        "decision_capture": capture,
    })

    assert len(capture.calls) == 1
    decision_type, decision, rationale, kwargs = capture.calls[0]
    assert decision_type == "instructional_depth_check"
    assert decision == "passed"
    # Rationale carries the three dynamic metric signals.
    assert "avg_concepts_per_page=" in rationale
    assert "avg_examples_per_concept=" in rationale
    assert "avg_explanation_tokens_per_concept=" in rationale
    # ml_features carries the three metrics + the three thresholds.
    ml_features = kwargs.get("ml_features")
    assert isinstance(ml_features, dict)
    assert ml_features["avg_concepts_per_page"] == pytest.approx(3.0)
    assert ml_features["avg_examples_per_concept"] == pytest.approx(
        4.0 / 3.0
    )
    assert ml_features["avg_explanation_tokens_per_concept"] == pytest.approx(
        120.0
    )
    assert ml_features["min_concepts_per_page"] == pytest.approx(
        float(DEFAULT_MIN_CONCEPTS_PER_PAGE)
    )
    assert ml_features["min_examples_per_concept"] == pytest.approx(
        DEFAULT_MIN_EXAMPLES_PER_CONCEPT
    )
    assert ml_features["min_explanation_tokens_per_concept"] == pytest.approx(
        float(DEFAULT_MIN_EXPLANATION_TOKENS_PER_CONCEPT)
    )
    assert ml_features["pages_audited"] == 1
    assert ml_features["passed"] is True
    assert ml_features["failure_codes"] == []


# ---------------------------------------------------------------------------
# 6. Threshold override — relaxed floors flip a previously-failing case
#    to passing.
# ---------------------------------------------------------------------------


def test_threshold_override_relaxes_gate() -> None:
    """A page that fails default floors passes with the floors lowered
    via ``inputs['thresholds']``."""
    # 1 concept, 0 examples, 30 tokens — fails all three default floors.
    blocks = [_make_concept(sequence=0, body=_word_padding(30))]

    result = InstructionalDepthValidator().validate({
        "blocks": blocks,
        "thresholds": {
            "min_concepts_per_page": 1,
            "min_examples_per_concept": 0.0,
            "min_explanation_tokens_per_concept": 10,
        },
    })

    assert result.passed is True
    assert [i for i in result.issues if i.severity == "critical"] == []


# ---------------------------------------------------------------------------
# 7. Multi-page input — failures attribute to the right page_id.
# ---------------------------------------------------------------------------


def test_multi_page_failure_locates_to_correct_page() -> None:
    """Two pages; only page_02 trips the concepts floor. The emitted
    GateIssue's ``location`` must point at page_02."""
    page_01_blocks: List[Block] = [
        _make_concept(
            page_id="page_01",
            sequence=i,
            body=_word_padding(120),
        )
        for i in range(3)
    ]
    page_01_blocks.extend(
        _make_example(page_id="page_01", sequence=10 + i)
        for i in range(4)
    )
    page_02_blocks = [
        _make_concept(
            page_id="page_02",
            sequence=0,
            body=_word_padding(120),
        ),
        _make_example(page_id="page_02", sequence=10),
    ]
    blocks = page_01_blocks + page_02_blocks

    result = InstructionalDepthValidator().validate({"blocks": blocks})

    assert result.passed is False
    concept_floor_issues = [
        i for i in result.issues
        if i.code
        == "INSTRUCTIONAL_DEPTH_CONCEPTS_PER_PAGE_BELOW_THRESHOLD"
    ]
    assert len(concept_floor_issues) == 1
    assert concept_floor_issues[0].location == "page_02"


# ---------------------------------------------------------------------------
# 8. Missing input handling — the validator surfaces a critical fail
#    rather than crashing.
# ---------------------------------------------------------------------------


def test_missing_blocks_input_fails_critical() -> None:
    result = InstructionalDepthValidator().validate({})
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "MISSING_BLOCKS_INPUT" in codes


def test_invalid_blocks_input_fails_critical() -> None:
    result = InstructionalDepthValidator().validate({"blocks": "not-a-list"})
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "INVALID_BLOCKS_INPUT" in codes


def test_empty_blocks_list_passes_without_issues() -> None:
    """An empty block list (page with zero authored content) returns a
    pass — the upstream phase is responsible for catching that surface
    failure; this gate is a depth gate, not a population gate."""
    result = InstructionalDepthValidator().validate({"blocks": []})
    assert result.passed is True
    assert result.issues == []


# ---------------------------------------------------------------------------
# 9. Score signal — degraded depth pulls the aggregate score down.
# ---------------------------------------------------------------------------


def test_score_drops_when_metrics_below_thresholds() -> None:
    """Score is the harmonic-style normalised average of the three
    signals. A page that violates all three floors should score
    well below 1.0."""
    blocks = [_make_concept(sequence=0, body=_word_padding(20))]
    result = InstructionalDepthValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.score is not None
    assert result.score < 0.5


# ---------------------------------------------------------------------------
# 10. Adjacent ``explanation`` blocks contribute to per-concept token
#     count — a thin concept body still passes if surrounded by
#     explanation prose.
# ---------------------------------------------------------------------------


def test_explanation_blocks_contribute_to_token_count() -> None:
    """A page with 2 thin concepts (10 tokens each) but rich
    surrounding explanation prose (200 tokens) clears the
    tokens-per-concept floor: total 220 tokens / 2 concepts = 110."""
    blocks = [
        _make_concept(sequence=0, body=_word_padding(10)),
        _make_concept(sequence=1, body=_word_padding(10)),
        _make_explanation(sequence=2, body=_word_padding(200)),
        _make_example(sequence=10),
        _make_example(sequence=11),
    ]
    result = InstructionalDepthValidator().validate({"blocks": blocks})
    assert result.passed is True
