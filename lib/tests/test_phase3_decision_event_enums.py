"""
Phase 3 Subtask 8: Regression tests for the strict-mode validator's frozen
enum cache after adding 4 new ``decision_type`` values + 2 new ``phase``
values for the two-pass router.

These tests guard against drift in two specific contracts:

1. The schema enum at ``schemas/events/decision_event.schema.json`` is the
   single source of truth for ``ALLOWED_DECISION_TYPES`` (loaded at module
   import via ``lib.decision_capture._load_schema``).
2. Under ``DECISION_VALIDATION_STRICT=true``, every Phase 3 enum value must
   validate clean (no ValueError raised) — and known fail-loud paths
   (rationale too short) must still raise.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.decision_capture import ALLOWED_DECISION_TYPES, DecisionCapture
except ImportError:
    pytest.skip("decision_capture not available", allow_module_level=True)


# Phase 3 two-pass router additions to the decision_type enum.
PHASE3_DECISION_TYPES = (
    "block_escalation",
    "block_outline_call",
    "block_rewrite_call",
    "block_validation_action",
)

# Phase 3 two-pass router additions to the phase enum.
PHASE3_PHASES = (
    "courseforge-content-generator-outline",
    "courseforge-content-generator-rewrite",
)


@pytest.fixture
def mock_libv2_storage(tmp_path):
    """Mock LibV2Storage to use a temp directory."""
    with patch("lib.decision_capture.LibV2Storage") as mock_cls:
        storage = Mock()
        capture_path = tmp_path / "libv2" / "training"
        capture_path.mkdir(parents=True)
        storage.get_training_capture_path.return_value = capture_path
        mock_cls.return_value = storage
        yield mock_cls


@pytest.fixture
def mock_legacy_dir(tmp_path):
    """Mock legacy training directory."""
    legacy_dir = tmp_path / "legacy-training"
    legacy_dir.mkdir(parents=True)
    with patch("lib.decision_capture.LEGACY_TRAINING_DIR", legacy_dir):
        yield legacy_dir


@pytest.fixture
def capture_outline(mock_libv2_storage, mock_legacy_dir):
    """Capture instance pinned to the outline phase."""
    return DecisionCapture(
        course_code="TEST_101",
        phase="courseforge-content-generator-outline",
        streaming=False,
    )


@pytest.fixture
def capture_rewrite(mock_libv2_storage, mock_legacy_dir):
    """Capture instance pinned to the rewrite phase."""
    return DecisionCapture(
        course_code="TEST_101",
        phase="courseforge-content-generator-rewrite",
        streaming=False,
    )


# =============================================================================
# Frozen-enum cache regression: every Phase 3 type is in ALLOWED_DECISION_TYPES
# =============================================================================

@pytest.mark.unit
def test_phase3_decision_types_loaded_into_allowlist():
    """Every Phase 3 decision_type must be in ALLOWED_DECISION_TYPES."""
    for dtype in PHASE3_DECISION_TYPES:
        assert dtype in ALLOWED_DECISION_TYPES, (
            f"Phase 3 decision_type {dtype!r} missing from ALLOWED_DECISION_TYPES "
            f"(schema enum drift — re-run after schema edit)"
        )


# =============================================================================
# Strict-mode pass cases: every Phase 3 type validates clean
# =============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("dtype", PHASE3_DECISION_TYPES)
def test_phase3_decision_type_validates_clean_in_strict_mode(
    capture_outline, monkeypatch, dtype
):
    """Each Phase 3 decision_type must NOT raise under strict mode."""
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")

    capture_outline.log_decision(
        decision_type=dtype,
        decision=f"Phase 3 router emitted {dtype} event",
        rationale=(
            "Phase 3 two-pass router strict-mode regression — this rationale "
            "must pass minLength=20 to exercise the enum cache."
        ),
    )

    assert len(capture_outline.decisions) == 1
    assert capture_outline.decisions[0]["decision_type"] == dtype


# =============================================================================
# Strict-mode fail-loud cases: short rationale still raises for Phase 3 types
# =============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("dtype", PHASE3_DECISION_TYPES)
def test_phase3_decision_type_short_rationale_raises_in_strict_mode(
    capture_outline, monkeypatch, dtype
):
    """Phase 3 types must still trip the rationale<20 fail-loud path."""
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")

    with pytest.raises(ValueError, match="validation failed"):
        capture_outline.log_decision(
            decision_type=dtype,
            decision="short",
            rationale="too short",  # 9 chars < 20-char minimum
        )

    # Record was NOT written (fail-closed contract).
    assert len(capture_outline.decisions) == 0


# =============================================================================
# Phase enum: the two new courseforge-content-generator-* phases survive
# round-trip through the capture record.
# =============================================================================

@pytest.mark.unit
def test_phase3_outline_phase_round_trips_through_record(capture_outline):
    """The outline phase value must round-trip through the emitted record."""
    capture_outline.log_decision(
        decision_type="block_outline_call",
        decision="Outline call dispatched to local provider",
        rationale=(
            "Phase 3 two-pass router emitted an outline call event under the "
            "courseforge-content-generator-outline phase context."
        ),
    )
    assert capture_outline.decisions[0]["phase"] == (
        "courseforge-content-generator-outline"
    )


@pytest.mark.unit
def test_phase3_rewrite_phase_round_trips_through_record(capture_rewrite):
    """The rewrite phase value must round-trip through the emitted record."""
    capture_rewrite.log_decision(
        decision_type="block_rewrite_call",
        decision="Rewrite call dispatched to anthropic provider",
        rationale=(
            "Phase 3 two-pass router emitted a rewrite call event under the "
            "courseforge-content-generator-rewrite phase context."
        ),
    )
    assert capture_rewrite.decisions[0]["phase"] == (
        "courseforge-content-generator-rewrite"
    )


# =============================================================================
# Dict-shaped decision payload (router emits structured decisions)
# =============================================================================

@pytest.mark.unit
def test_phase3_block_escalation_with_dict_decision(capture_outline, monkeypatch):
    """block_escalation events frequently carry dict-shaped decision payloads."""
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
    monkeypatch.setenv("VALIDATE_DECISIONS", "true")

    capture_outline.log_decision(
        decision_type="block_escalation",
        decision=(
            "{'block_id': 'p1#concept_x_0', 'marker': 'outline_budget_exhausted', "
            "'attempts': 3}"
        ),
        rationale=(
            "Phase 3 escalation event emitted after outline regen budget "
            "exhausted; router falling back to rewrite tier with enriched prompt."
        ),
    )

    assert len(capture_outline.decisions) == 1
    assert capture_outline.decisions[0]["decision_type"] == "block_escalation"
