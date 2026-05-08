"""GPT Feedback v2 Wave 1 (W1.C) — course-status helper tests.

Pins the public surface of :mod:`lib.governance.course_status`:

  * :func:`load_course_status_schema` returns a dict carrying all five
    canonical enum values.
  * :func:`compose_course_status` returns ``"failed"`` when any arrow has
    ``promotion_decision == "fail"``.
  * :func:`compose_course_status` returns ``"non_certified_archive"`` when
    arrows 1-5 pass and arrow 6 is ``"missing"``.
  * :func:`compose_course_status` raises :class:`NotImplementedError` for
    the three ``certified_*`` branches with a message pointing at Wave 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Repo-root sys.path bootstrap (mirror lib/tests/test_decision_capture.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.governance.course_status import (  # noqa: E402
    ACCESSIBILITY_GATE_IDS,
    INSTRUCTIONAL_GATE_IDS,
    TRAINABLE_GATE_IDS,
    compose_course_status,
    derive_course_status,
    load_course_status_schema,
)

EXPECTED_ENUM = [
    "certified_accessible",
    "certified_instructional",
    "certified_trainable",
    "non_certified_archive",
    "failed",
]


# ---------------------------------------------------------------------------
# load_course_status_schema
# ---------------------------------------------------------------------------


def test_load_course_status_schema_returns_dict_with_full_enum():
    schema = load_course_status_schema()
    assert isinstance(schema, dict)
    assert "enum" in schema
    assert schema["enum"] == EXPECTED_ENUM
    assert schema["type"] == "string"
    # Defensive-copy contract: mutating the returned dict must not pollute
    # subsequent loads (mirrors lib.ontology.taxonomy semantics).
    schema["enum"].append("totally_made_up")
    again = load_course_status_schema()
    assert again["enum"] == EXPECTED_ENUM


# ---------------------------------------------------------------------------
# compose_course_status — failed branch
# ---------------------------------------------------------------------------


def test_compose_course_status_returns_failed_on_any_fail():
    assert compose_course_status({"arrow1": "fail"}) == "failed"


def test_compose_course_status_returns_failed_when_one_of_many_fails():
    decisions = {
        "arrow1": "pass",
        "arrow2": "pass",
        "arrow3": "fail",
        "arrow4": "pass",
        "arrow5": "pass",
    }
    assert compose_course_status(decisions) == "failed"


# ---------------------------------------------------------------------------
# compose_course_status — non_certified_archive branch
# ---------------------------------------------------------------------------


def test_compose_course_status_returns_non_certified_archive_when_arrows_1_5_pass_and_6_missing():
    decisions = {
        "arrow1": "pass",
        "arrow2": "pass",
        "arrow3": "pass",
        "arrow4": "pass",
        "arrow5": "pass",
        "arrow6": "missing",
    }
    assert compose_course_status(decisions) == "non_certified_archive"


def test_compose_course_status_non_certified_archive_when_only_arrows_1_5_present():
    # When the trainable slice is entirely absent (no arrow6+ keys at all),
    # the helper still routes to non_certified_archive — arrows 1-5 pass
    # and there is no passing trainable arrow.
    decisions = {
        "arrow1": "pass",
        "arrow2": "warn",
        "arrow3": "pass",
        "arrow4": "pass",
        "arrow5": "pass",
    }
    assert compose_course_status(decisions) == "non_certified_archive"


# ---------------------------------------------------------------------------
# compose_course_status — certified_* (Wave 3 NotImplementedError)
# ---------------------------------------------------------------------------


def test_compose_course_status_raises_not_implemented_for_full_chain_pass():
    decisions = {f"arrow{i}": "pass" for i in range(1, 10)}
    with pytest.raises(NotImplementedError) as excinfo:
        compose_course_status(decisions)
    message = str(excinfo.value)
    assert "Wave 3" in message
    assert "certified" in message.lower()


def test_compose_course_status_not_implemented_message_points_at_plan():
    decisions = {f"arrow{i}": "pass" for i in range(1, 10)}
    with pytest.raises(NotImplementedError) as excinfo:
        compose_course_status(decisions)
    # Pin the deferral citation so a future drift on the message is loud.
    assert "gpt-feedback-2-wave1-schemas-2026-05" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Wave W-D11.B — claim_support / pair_claim_support cohort credit
# ---------------------------------------------------------------------------
#
# Wave W-D11 added two NLI-entailment gates (block-side ``claim_support``
# fired at ``post_rewrite_validation`` and pair-side ``pair_claim_support``
# fired at ``training_synthesis``). They're warning-severity day-1 per the
# calibration-deferred contract; once operator calibration flips them to
# critical, ``derive_course_status`` MUST credit them to the right cohort
# rather than short-circuit to ``"failed"`` via the critical-cohort branch.
# These tests pin the cohort assignment ahead of the calibration flip:
# the cohort entries are dormant until severity flips, but the membership
# table must already be correct so the flip lands without a silent
# misclassification.


def test_claim_support_in_instructional_cohort():
    """``claim_support`` is sibling to ``content_grounding`` / ``source_refs``."""
    assert "claim_support" in INSTRUCTIONAL_GATE_IDS
    # Critical cohort containment guard: ``claim_support`` must NOT be in
    # the accessibility cohort, otherwise a critical-fail would
    # short-circuit to ``"failed"`` instead of gating instructional
    # certification.
    assert "claim_support" not in ACCESSIBILITY_GATE_IDS
    assert "claim_support" not in TRAINABLE_GATE_IDS


def test_pair_claim_support_in_trainable_cohort():
    """``pair_claim_support`` is sibling to ``synthesis_diversity``."""
    assert "pair_claim_support" in TRAINABLE_GATE_IDS
    assert "pair_claim_support" not in ACCESSIBILITY_GATE_IDS
    assert "pair_claim_support" not in INSTRUCTIONAL_GATE_IDS


def _arrow(
    arrow_id: int,
    *,
    validator_set,
    promotion_decision: str,
):
    """Minimal arrow-row builder mirroring the promotion-chain shape."""
    return {
        "arrow_id": arrow_id,
        "name": f"arrow_{arrow_id}",
        "input_hash": None,
        "output_hash": None,
        "validator_set": list(validator_set),
        "passed": promotion_decision in ("pass", "warn"),
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": promotion_decision,
    }


def _accessibility_passing_arrows():
    """Arrows 1-5 clean, with WCAG attached on arrow 5 so the cohort certifies."""
    arrows = [
        _arrow(i, validator_set=[], promotion_decision="pass")
        for i in range(1, 5)
    ]
    arrows.append(
        _arrow(5, validator_set=["wcag_compliance"], promotion_decision="pass")
    )
    return arrows


def test_critical_fail_on_claim_support_does_not_short_circuit_to_failed():
    """A critical-fail attached only to ``claim_support`` (and the rest of
    the instructional cohort otherwise complete) MUST gate the
    instructional cohort rather than short-circuit to ``"failed"``.

    The chain has WCAG passing (so accessibility certifies). Arrow 3
    fails with ``claim_support`` in its validator_set — the only
    instructional-cohort gate fired — so the instructional cohort is
    disqualified by ``_cohort_passes``'s any-fail rule. Expected:
    ``certified_accessible`` (the highest tier the WCAG-only cohort
    earns), NOT ``"failed"`` (which would be the silent-misclassification
    outcome if ``claim_support`` were in the critical cohort).
    """
    arrows = _accessibility_passing_arrows()
    arrows.append(
        _arrow(
            3,
            validator_set=["claim_support"],
            promotion_decision="fail",
        )
    )
    decision = derive_course_status(arrows)
    assert decision == "certified_accessible", (
        f"claim_support fail must gate instructional cohort, not "
        f"short-circuit to failed; got {decision!r}"
    )


def test_critical_fail_on_pair_claim_support_does_not_short_circuit_to_failed():
    """A critical-fail attached only to ``pair_claim_support`` MUST gate
    the trainable cohort rather than short-circuit to ``"failed"``.

    The chain has WCAG passing AND every instructional-cohort gate
    passing on arrow 6 (so the chain ceilings at
    ``certified_instructional`` if the trainable cohort fails). Arrow
    7 fails with ``pair_claim_support`` in its validator_set — the only
    trainable-cohort gate fired — so the trainable cohort is
    disqualified. Expected: ``certified_instructional``, NOT
    ``"failed"``.
    """
    arrows = _accessibility_passing_arrows()
    # Arrow 6 — full instructional cohort passing.
    arrows.append(
        _arrow(
            6,
            validator_set=sorted(INSTRUCTIONAL_GATE_IDS),
            promotion_decision="pass",
        )
    )
    # Arrow 7 — pair_claim_support fails (calibration-flipped to
    # critical). No other trainable-cohort gates attached so the
    # cohort fails purely on this gate.
    arrows.append(
        _arrow(
            7,
            validator_set=["pair_claim_support"],
            promotion_decision="fail",
        )
    )
    decision = derive_course_status(arrows)
    assert decision == "certified_instructional", (
        f"pair_claim_support fail must gate trainable cohort, not "
        f"short-circuit to failed; got {decision!r}"
    )


def test_warning_only_claim_support_does_not_block_promotion():
    """The day-1 warning-severity contract: ``claim_support`` firing as
    a non-blocking ``"warn"`` (the current default) MUST NOT prevent
    promotion through the instructional + trainable cohorts.

    Mirrors the calibration-deferred posture: until operator
    calibration flips severity to critical, ``claim_support`` /
    ``pair_claim_support`` warnings are advisory and the chain
    promotes on the strength of the other cohort gates.
    """
    arrows = _accessibility_passing_arrows()
    # Arrow 3 — claim_support warns (non-blocking).
    arrows.append(
        _arrow(
            3,
            validator_set=["claim_support"],
            promotion_decision="warn",
        )
    )
    # Arrow 6 — full instructional cohort passing.
    arrows.append(
        _arrow(
            6,
            validator_set=sorted(INSTRUCTIONAL_GATE_IDS),
            promotion_decision="pass",
        )
    )
    # Arrow 7 — full trainable cohort passing.
    arrows.append(
        _arrow(
            7,
            validator_set=sorted(TRAINABLE_GATE_IDS),
            promotion_decision="pass",
        )
    )
    decision = derive_course_status(arrows)
    assert decision == "certified_trainable", (
        f"warning-only claim_support / pair_claim_support must not "
        f"block trainable promotion; got {decision!r}"
    )


def test_warning_only_pair_claim_support_does_not_block_promotion():
    """Symmetric warning-only check for ``pair_claim_support``."""
    arrows = _accessibility_passing_arrows()
    # Arrow 6 — full instructional cohort passing.
    arrows.append(
        _arrow(
            6,
            validator_set=sorted(INSTRUCTIONAL_GATE_IDS),
            promotion_decision="pass",
        )
    )
    # Arrow 7 — pair_claim_support warns; the rest of the trainable
    # cohort passes.
    arrows.append(
        _arrow(
            7,
            validator_set=sorted(TRAINABLE_GATE_IDS),
            promotion_decision="pass",
        )
    )
    arrows.append(
        _arrow(
            7,
            validator_set=["pair_claim_support"],
            promotion_decision="warn",
        )
    )
    decision = derive_course_status(arrows)
    assert decision == "certified_trainable", (
        f"warning-only pair_claim_support must not block trainable "
        f"promotion; got {decision!r}"
    )
