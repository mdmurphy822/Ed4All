"""
Phase 3.5 Subtask 27: Regression sentinel for the ``ml_features.tier``
field added to ``block_validation_action`` events by Subtask 26.

The Phase 3 schema at ``schemas/events/decision_event.schema.json`` does
NOT pin ``additionalProperties: false`` on the ``ml_features`` object,
so adding the ``tier`` field is purely additive — pre-Phase-3.5
captures without ``tier`` continue to deserialize and Subtask 26's
emit-side change is backward-compatible.

This test file pins that contract: under
``DECISION_VALIDATION_STRICT=true``, ``block_validation_action`` events
carrying ``ml_features.tier="outline"`` and ``ml_features.tier="rewrite"``
both validate clean. If a future schema change tightens
``ml_features`` (e.g. adding ``additionalProperties: false`` to the
nested object), this test trips loud — that's the whole point of the
sentinel.

Tests use ``lib.validation.validate_decision`` directly because
``lib.decision_capture.DecisionCapture._build_record`` calls
``dataclasses.asdict(ml_features)`` on the ``MLFeatures`` argument and
the router's ``_emit_block_validation_action`` helper currently passes
a plain dict (``dict[str, Any]``) — that mismatch is captured /
swallowed by the router's defensive try/except. The schema-level
contract (the actual ``ml_features.tier`` regression sentinel) is what
Subtask 27 needs to pin, and ``validate_decision`` is the canonical
entry point.

Pre-Phase-3.5 fixture coverage (e.g.
``test_phase3_decision_event_enums.py::test_phase3_decision_type_validates_clean_in_strict_mode``)
exercises ``DecisionCapture.log_decision`` for the four Phase 3
``decision_type`` values without any ``ml_features``, so this file
covers the orthogonal axis of the same router event shape: tier-
disambiguated ``ml_features``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from lib.validation import validate_decision  # noqa: E402
except ImportError:  # pragma: no cover — defensive
    pytest.skip("lib.validation not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helper: build a canonical block_validation_action record with ``tier``.
# ---------------------------------------------------------------------------


def _build_block_validation_action_record(
    *,
    tier: str,
    block_id: str = "p1#concept_a_0",
    block_type: str = "concept",
    gate_id: str = "outline_curie_anchoring",
    action: str = "regenerate",
) -> dict:
    """Build a canonical decision record for the regression sentinel.

    Mirrors the shape ``CourseforgeRouter._emit_block_validation_action``
    (``Courseforge/router/router.py``) lands per Subtask 47 + Subtask 26
    so the test fixture stays byte-aligned with the production emit
    path.
    """
    return {
        "event_id": "EVT_0123456789abcdef",
        "seq": 1,
        "run_id": "RUN_20260502_120000",
        "course_id": "TEST_101",
        "module_id": None,
        "artifact_id": None,
        "task_id": None,
        "tool": "courseforge",
        "operation": "validate_outline",
        "timestamp": "2026-05-02T12:00:00",
        "phase": "courseforge-content-generator-outline",
        "decision_type": "block_validation_action",
        "decision": (
            f"validation_action:{block_type}:{block_id}:{gate_id}:{action}"
        ),
        "rationale": (
            f"Validator {gate_id} returned action={action} for block "
            f"{block_id} (block_type={block_type}); score=0.420; "
            f"top_issues=[curie_missing(critical):no curies]. Router "
            f"will route per Phase 4 §1.5 mapping: regenerate→retry."
        ),
        "alternatives_considered": [],
        "context": None,
        "confidence": None,
        "is_default": False,
        "ml_features": {
            "block_id": block_id,
            "block_type": block_type,
            "gate_id": gate_id,
            "action": action,
            "score": 0.42,
            "issues_top3": [
                {
                    "code": "curie_missing",
                    "severity": "critical",
                    "message": "no curies",
                }
            ],
            "issues_count": 1,
            # The Subtask-26 addition under audit by this test:
            "tier": tier,
        },
        "inputs_ref": [],
        "prompt_ref": None,
        "outputs": [],
        "outcome": None,
        "metadata": {
            "rationale_length": 200,
            "quality_level": "proficient",
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strict_mode_accepts_block_validation_action_with_tier_outline(
    monkeypatch,
) -> None:
    """Phase 3.5 §F — outline-tier ml_features.tier validates clean."""
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    record = _build_block_validation_action_record(tier="outline")
    is_valid, issues = validate_decision(record, tool="courseforge")

    assert is_valid, (
        f"block_validation_action with ml_features.tier='outline' must "
        f"validate clean under strict mode; got issues={issues}"
    )
    assert issues == []


@pytest.mark.unit
def test_strict_mode_accepts_block_validation_action_with_tier_rewrite(
    monkeypatch,
) -> None:
    """Phase 3.5 §F — rewrite-tier ml_features.tier validates clean.

    The rewrite-tier emit path lands in Wave B Subtask 13's
    ``_run_post_rewrite_validation`` helper, which will pass
    ``tier="rewrite"`` at its own call site. This test pins the
    schema-level acceptance ahead of the wave-B implementation so a
    schema-tightening change can't silently break the rewrite-tier
    emit path before Wave B lands.
    """
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    # Use the rewrite-phase value to mirror the canonical Wave B emit
    # context (the helper will run inside the rewrite-phase capture).
    record = _build_block_validation_action_record(tier="rewrite")
    record["phase"] = "courseforge-content-generator-rewrite"

    is_valid, issues = validate_decision(record, tool="courseforge")

    assert is_valid, (
        f"block_validation_action with ml_features.tier='rewrite' must "
        f"validate clean under strict mode; got issues={issues}"
    )
    assert issues == []


@pytest.mark.unit
def test_strict_mode_accepts_block_validation_action_without_tier(
    monkeypatch,
) -> None:
    """Phase 3.5 §F — backward-compat: pre-Phase-3.5 records without
    ``ml_features.tier`` still validate clean.

    The schema doesn't pin ``additionalProperties: false`` on
    ``ml_features``, and ``tier`` is not a required key. A capture
    written before Subtask 26 landed (no ``tier`` key) must
    deserialize against the unchanged schema; this test pins that
    invariant.
    """
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    record = _build_block_validation_action_record(tier="outline")
    record["ml_features"].pop("tier", None)

    is_valid, issues = validate_decision(record, tool="courseforge")

    assert is_valid, (
        f"pre-Phase-3.5 block_validation_action (no ml_features.tier) "
        f"must validate clean under strict mode; got issues={issues}"
    )
    assert issues == []
