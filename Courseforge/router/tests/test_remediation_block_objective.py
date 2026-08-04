"""Tests for the block-objective delivery remediation suffix.

Covers the rewrite-tier remediation-loop integration of the three
``BlockObjectiveDeliveryValidator`` issue codes
(``BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED`` /
``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` / ``BLOCK_OBJECTIVE_VERB_ABSENT``)
plus the new ``block_objective_undelivered`` escalation marker that
fires when the rewrite-tier regeneration budget exhausts purely on
block-objective delivery codes.

The canonical cases are:

1. ``test_remediation_suffix_for_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED``
   — assert the rendered suffix contains the objective_id, the
   truncated objective_statement (≤80 chars), the entailment score
   formatted to 2 decimals, and the floor.
2. ``test_remediation_suffix_for_BLOCK_OBJECTIVE_BLOOM_UNDERMET`` —
   assert the rendered suffix names the bloom_gap and both the
   declared / observed Bloom levels.
3. ``test_remediation_suffix_for_BLOCK_OBJECTIVE_VERB_ABSENT`` —
   assert the rendered suffix lists the bloom_verb plus the
   synonym preview list.
4. ``test_rewrite_regen_loop_consumes_block_objective_action_regenerate``
   — full integration: outline emits a Bloom-undermet block, inter-tier
   validator fires regenerate, rewrite tier sees the suffix, the
   re-rolled block passes the post-rewrite gate.
5. ``test_block_objective_undelivered_marker_when_budget_exhausted``
   — exhaust the rewrite regen budget with persistent
   ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` failures; assert the surviving
   best-effort block carries
   ``escalation_marker="block_objective_undelivered"`` AND
   ``objective_alignment[*].status == "unverifiable"``.

Mirrors the fixture style in ``test_remediation_injection.py`` for
parallel structure with the existing rewrite-tier remediation
regression suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blocks import Block  # noqa: E402

from Courseforge.router.remediation import (  # noqa: E402
    _CODE_BLOCK_OBJECTIVE_BLOOM_UNDERMET,
    _CODE_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED,
    _CODE_BLOCK_OBJECTIVE_VERB_ABSENT,
    _MAX_OBJECTIVE_STATEMENT_CHARS,
    _append_remediation_for_gates,
)
from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from MCP.hardening.validation_gates import GateIssue, GateResult  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "concept",
    block_id: str = "page1#concept_intro_0",
    content: Any = "hello",
    objective_ids: tuple = ("TO-01",),
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
        objective_ids=objective_ids,
    )


def _statement_undersupported_issue(
    *,
    block_id: str = "page1#concept_intro_0",
    obj_id: str = "TO-01",
    block_type: str = "concept",
    entailment_score: float = 0.12,
    entailment_floor: float = 0.30,
    contradiction_score: float = 0.78,
    contradiction_floor: float = 0.50,
) -> GateIssue:
    """Build a GateIssue whose .message matches the validator's
    ``BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED`` f-string format."""
    return GateIssue(
        severity="warning",
        code=_CODE_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED,
        message=(
            f"Block {block_id!r} (block_type={block_type!r}) prose did "
            f"not entail objective {obj_id!r}. NLI entailment="
            f"{entailment_score:.4f} (floor {entailment_floor:.4f}); "
            f"contradiction={contradiction_score:.4f} (floor "
            f"{contradiction_floor:.4f})."
        ),
        location=block_id,
    )


def _bloom_undermet_issue(
    *,
    block_id: str = "page1#concept_intro_0",
    obj_id: str = "TO-01",
    block_type: str = "concept",
    observed_bloom: str = "remember",
    declared_bloom: str = "apply",
    bloom_gap: int = 2,
) -> GateIssue:
    """Build a GateIssue whose .message matches the validator's
    ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` f-string format."""
    return GateIssue(
        severity="warning",
        code=_CODE_BLOCK_OBJECTIVE_BLOOM_UNDERMET,
        message=(
            f"Block {block_id!r} (block_type={block_type!r}) "
            f"bloom_level={observed_bloom!r} is {bloom_gap} levels "
            f"below the declared objective {obj_id!r}'s "
            f"bloom_level={declared_bloom!r}. Re-emit at or above the "
            f"declared level — scaffold up to the declared cognitive "
            f"demand."
        ),
        location=block_id,
    )


def _verb_absent_issue(
    *,
    block_id: str = "page1#concept_intro_0",
    obj_id: str = "TO-01",
    block_type: str = "concept",
    bloom_verb: str = "apply",
    declared_bloom: str = "apply",
    synonyms_preview: str = "apply, demonstrate, employ, execute, illustrate, implement, solve, use",
) -> GateIssue:
    """Build a GateIssue whose .message matches the validator's
    ``BLOCK_OBJECTIVE_VERB_ABSENT`` f-string format."""
    return GateIssue(
        severity="warning",
        code=_CODE_BLOCK_OBJECTIVE_VERB_ABSENT,
        message=(
            f"Block {block_id!r} (block_type={block_type!r}) prose "
            f"contains no synonym of the objective {obj_id!r}'s "
            f"bloom_verb={bloom_verb!r}. Re-emit prose using "
            f"{bloom_verb!r} or one of the {declared_bloom!r}-level "
            f"synonyms (e.g. {synonyms_preview})."
        ),
        location=block_id,
    )


def _gate_result_from_issue(
    issue: GateIssue,
    *,
    gate_id: str = "outline_block_objective_delivery",
    action: str = "regenerate",
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        validator_name="block_objective_delivery",
        validator_version="1.0.0",
        passed=True,  # warning-severity → passed=True per Day-1 contract
        issues=[issue],
        action=action,
    )


# ---------------------------------------------------------------------------
# Test 1 — STATEMENT_UNDERSUPPORTED suffix
# ---------------------------------------------------------------------------


def test_remediation_suffix_for_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED():
    """The rendered suffix interpolates the objective_id, the
    objective_statement (truncated at 80 chars), the entailment score
    formatted to 2 decimals, and the entailment floor."""
    issue = _statement_undersupported_issue(
        obj_id="TO-03",
        entailment_score=0.18,
        entailment_floor=0.45,
    )
    statement_long = (
        "Apply linear-regression diagnostics to identify residual "
        "outliers, leverage points, and influential observations across "
        "a multi-predictor model."
    )
    assert len(statement_long) > _MAX_OBJECTIVE_STATEMENT_CHARS

    suffix = _append_remediation_for_gates(
        "",
        [_gate_result_from_issue(issue)],
        objective_statements={"TO-03": statement_long},
    )

    # Failing objective_id is named.
    assert "'TO-03'" in suffix
    # Score and floor render at two decimals per the prompt contract.
    assert "0.18" in suffix
    assert "0.45" in suffix
    # Statement is truncated — full string should NOT appear, but the
    # leading prefix MUST appear.
    assert statement_long not in suffix
    assert statement_long[:40] in suffix
    # Truncation suffix marker (ellipsis) must be present.
    assert "…" in suffix
    # The canonical statement-support directive must appear.
    assert "did not" in suffix
    assert "semantically support" in suffix
    assert "behavioral outcome" in suffix


# ---------------------------------------------------------------------------
# Test 2 — BLOOM_UNDERMET suffix
# ---------------------------------------------------------------------------


def test_remediation_suffix_for_BLOCK_OBJECTIVE_BLOOM_UNDERMET():
    """The rendered suffix names the bloom_gap and both the declared and
    observed Bloom levels, plus the failing objective_id."""
    issue = _bloom_undermet_issue(
        obj_id="CO-04",
        observed_bloom="understand",
        declared_bloom="evaluate",
        bloom_gap=3,
    )
    suffix = _append_remediation_for_gates(
        "",
        [_gate_result_from_issue(issue)],
    )

    # Failing objective_id is named.
    assert "'CO-04'" in suffix
    # Both Bloom levels are named.
    assert "'understand'" in suffix
    assert "'evaluate'" in suffix
    # Bloom-gap integer is named.
    assert "3 levels below" in suffix
    # The canonical Bloom-demand directive must appear.
    assert "scaffold up to the declared" in suffix
    assert "cognitive demand" in suffix


# ---------------------------------------------------------------------------
# Test 3 — VERB_ABSENT suffix
# ---------------------------------------------------------------------------


def test_remediation_suffix_for_BLOCK_OBJECTIVE_VERB_ABSENT():
    """The rendered suffix lists the bloom_verb and the synonym
    preview the validator emitted in its ``e.g. <list>`` clause."""
    issue = _verb_absent_issue(
        bloom_verb="evaluate",
        declared_bloom="evaluate",
        synonyms_preview="appraise, assess, critique, judge, justify, rate",
    )
    suffix = _append_remediation_for_gates(
        "",
        [_gate_result_from_issue(issue)],
    )

    # Failing bloom_verb is named.
    assert "'evaluate'" in suffix
    # Synonym preview list lands in the directive verbatim.
    assert "appraise" in suffix
    assert "critique" in suffix
    assert "justify" in suffix
    # The canonical verb-presence directive must appear.
    assert "no synonym" in suffix
    assert "Bloom-level synonyms" in suffix


# ---------------------------------------------------------------------------
# Test 4 — full rewrite-tier integration
# ---------------------------------------------------------------------------


class _RecordingRewriteProvider:
    """Records every ``generate_rewrite`` call including the
    ``remediation_suffix`` kwarg so the integration test can verify the
    suffix flow."""

    def __init__(self, outputs: List[Block]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        remediation_suffix: Optional[str] = None,
        **kwargs: Any,
    ) -> Block:
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append(
            {
                "block_id": block.block_id,
                "remediation_suffix": remediation_suffix,
            }
        )
        return self._outputs[idx]


class _SequenceValidator:
    """Returns canned ``GateResult`` instances in sequence."""

    name = "sequence_validator"
    version = "1.0.0"

    def __init__(self, results: List[GateResult]) -> None:
        if not results:
            raise ValueError(
                "_SequenceValidator needs at least one result"
            )
        self._results = list(results)
        self.calls: int = 0

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def test_rewrite_regen_loop_consumes_block_objective_action_regenerate(
    monkeypatch,
):
    """Integration: outline emits a Bloom-undermet block, the rewrite-tier
    validator chain returns ``action="regenerate"`` on the first
    candidate, the next candidate's prompt carries the objective-delivery
    suffix interpolating the failing objective_id + Bloom gap, and the
    re-rolled block passes the post-rewrite gate.

    Mocks both the rewrite provider and the validator chain so the test
    runs without LLM dispatch.
    """
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)

    blk = _block(objective_ids=("TO-07",))
    rewrite = _RecordingRewriteProvider([blk, blk])
    validator = _SequenceValidator(
        [
            _gate_result_from_issue(
                _bloom_undermet_issue(
                    obj_id="TO-07",
                    observed_bloom="remember",
                    declared_bloom="analyze",
                    bloom_gap=3,
                ),
                gate_id="rewrite_block_objective_delivery",
            ),
            _gate_result_from_issue(
                _bloom_undermet_issue(),
                gate_id="rewrite_block_objective_delivery",
                action="pass",
            ).__class__(
                gate_id="rewrite_block_objective_delivery",
                validator_name="block_objective_delivery",
                validator_version="1.0.0",
                passed=True,
                issues=[],
                action="pass",
            ),
        ]
    )
    objectives = [
        {
            "id": "TO-07",
            "statement": "Analyze the residual structure.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
        },
    ]

    r = CourseforgeRouter(rewrite_provider=rewrite)
    out = r.route_rewrite_with_remediation(
        blk,
        n_candidates=2,
        regen_budget=5,
        validators=[validator],
        objectives=objectives,
    )

    # First candidate dispatched without a suffix; second carries the
    # Bloom-undermet directive.
    assert len(rewrite.calls) == 2
    assert rewrite.calls[0]["remediation_suffix"] is None
    second = rewrite.calls[1]["remediation_suffix"]
    assert second is not None
    # Failing objective_id + both Bloom levels surface in the suffix.
    assert "'TO-07'" in second
    assert "'remember'" in second
    assert "'analyze'" in second
    assert "3 levels below" in second
    # Canonical Bloom-demand directive copy.
    assert "scaffold up to the declared" in second
    # The re-rolled block is the winner — no escalation marker.
    assert out.escalation_marker is None


# ---------------------------------------------------------------------------
# Test 5 — escalation marker when budget exhausted
# ---------------------------------------------------------------------------


def test_block_objective_undelivered_marker_when_budget_exhausted(monkeypatch):
    """Exhaust the rewrite regen budget with persistent
    ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` failures; assert the surviving
    best-effort block carries
    ``escalation_marker="block_objective_undelivered"`` AND
    ``objective_alignment[*].status == "unverifiable"`` for every
    declared objective_id."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)

    blk = _block(objective_ids=("TO-09", "CO-02"))
    rewrite = _RecordingRewriteProvider([blk, blk, blk])
    # Every candidate fails with BLOOM_UNDERMET.
    validator = _SequenceValidator(
        [
            _gate_result_from_issue(
                _bloom_undermet_issue(obj_id="TO-09"),
                gate_id="rewrite_block_objective_delivery",
            ),
        ]
    )
    objectives = [
        {
            "id": "TO-09",
            "statement": "Apply the diagnostic.",
            "bloom_level": "apply",
            "bloom_verb": "apply",
        },
        {
            "id": "CO-02",
            "statement": "Identify the residual.",
            "bloom_level": "remember",
            "bloom_verb": "identify",
        },
    ]

    r = CourseforgeRouter(rewrite_provider=rewrite)
    out = r.route_rewrite_with_remediation(
        blk,
        n_candidates=3,
        regen_budget=3,
        validators=[validator],
        objectives=objectives,
    )

    # A budget-exhausted block carries the objective-delivery marker
    # (NOT the generic validator_consensus_fail marker).
    assert out.escalation_marker == "block_objective_undelivered"
    # objective_alignment carries one entry per declared objective_id,
    # each with status="unverifiable".
    assert len(out.objective_alignment) == 2
    obj_ids_in_alignment = {
        entry["objective_id"] for entry in out.objective_alignment
    }
    assert obj_ids_in_alignment == {"TO-09", "CO-02"}
    statuses = {entry["status"] for entry in out.objective_alignment}
    assert statuses == {"unverifiable"}
