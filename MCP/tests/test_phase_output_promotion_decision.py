"""GPT Feedback v2 Wave 1 (W1.D) — PhaseOutput.promotion_decision round-trip.

Pins ``MCP/orchestrator/worker_contracts.py::PhaseOutput.promotion_decision``:

  * PhaseOutput constructed without the field has ``promotion_decision is
    None`` and round-trips clean through ``to_dict()`` / ``from_dict()``.
  * PhaseOutput constructed with ``promotion_decision="pass"`` round-trips
    through ``to_dict()`` (key present in dict) -> ``from_dict()`` (value
    restored).
  * Legacy serialized PhaseOutput JSON (no key) deserializes via
    ``from_dict()`` to a PhaseOutput with ``promotion_decision is None``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.orchestrator.worker_contracts import GateResult, PhaseOutput


VALID_PROMOTION_DECISIONS = ("pass", "warn", "fail", "escalate")


def _sample_gate_results() -> dict:
    return {
        "content_structure": GateResult(
            gate_id="content_structure",
            severity="critical",
            passed=True,
            issues=[],
            details={},
        )
    }


# ---------------------------------------------------------------------------
# Default behavior: field defaults to None and round-trips clean.
# ---------------------------------------------------------------------------


def test_phase_output_default_promotion_decision_is_none():
    """Constructing PhaseOutput without the field leaves it None."""
    output = PhaseOutput(
        run_id="WF-20260506-test0001",
        phase_name="content_generation_outline",
    )
    assert output.promotion_decision is None


def test_phase_output_default_round_trip_clean():
    """A default-constructed PhaseOutput round-trips through
    to_dict() / from_dict() without losing any field value."""
    original = PhaseOutput(
        run_id="WF-20260506-test0002",
        phase_name="content_generation_rewrite",
        outputs={"blocks_rewritten": 7},
        artifacts=[Path("/tmp/run/04_rewrite/page_001.json")],
        gate_results=_sample_gate_results(),
        status="ok",
    )
    serialized = original.to_dict()
    # Default-None field must still appear in the dict so the schema's
    # optional declaration is exercised on every emit.
    assert "promotion_decision" in serialized
    assert serialized["promotion_decision"] is None

    restored = PhaseOutput.from_dict(serialized)
    assert restored.promotion_decision is None
    assert restored.run_id == original.run_id
    assert restored.phase_name == original.phase_name
    assert restored.status == original.status
    assert list(restored.gate_results.keys()) == list(
        original.gate_results.keys()
    )


# ---------------------------------------------------------------------------
# Explicit-value round-trip across each enum member.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALID_PROMOTION_DECISIONS)
def test_phase_output_explicit_promotion_decision_round_trip(value):
    """Each of the four enum values round-trips through to_dict() ->
    from_dict() preserving the value exactly."""
    original = PhaseOutput(
        run_id="WF-20260506-test0003",
        phase_name="post_rewrite_validation",
        promotion_decision=value,
    )
    serialized = original.to_dict()
    assert serialized["promotion_decision"] == value

    restored = PhaseOutput.from_dict(serialized)
    assert restored.promotion_decision == value


def test_phase_output_promotion_decision_round_trip_via_json_string():
    """Round-trip via to_json() / json.loads() / from_dict() keeps the
    value stable — confirms the field survives the full JSON encode/decode
    path used by checkpoint writers."""
    original = PhaseOutput(
        run_id="WF-20260506-test0004",
        phase_name="post_rewrite_validation",
        promotion_decision="pass",
        gate_results=_sample_gate_results(),
        status="ok",
    )
    encoded = original.to_json()
    decoded = json.loads(encoded)
    assert decoded["promotion_decision"] == "pass"

    restored = PhaseOutput.from_dict(decoded)
    assert restored.promotion_decision == "pass"


# ---------------------------------------------------------------------------
# Legacy compatibility: dicts without the key deserialize to None.
# ---------------------------------------------------------------------------


def test_legacy_serialized_phase_output_deserializes_to_none():
    """A serialized PhaseOutput dict from BEFORE Wave 1 (no
    promotion_decision key) must deserialize cleanly via from_dict() to a
    PhaseOutput with promotion_decision=None — preserves backward
    compatibility for already-checkpointed phase outputs."""
    legacy_dict = {
        "run_id": "WF-20260420-legacy01",
        "phase_name": "content_generation",
        "outputs": {"weeks_generated": 8},
        "artifacts": ["/old/run/page_01.html"],
        "gate_results": {
            "page_objectives": {
                "gate_id": "page_objectives",
                "severity": "critical",
                "passed": True,
                "issues": [],
                "details": {},
            }
        },
        "decision_captures_path": None,
        "status": "ok",
        "error": None,
        "metrics": {"elapsed_seconds": 12.3},
    }
    assert "promotion_decision" not in legacy_dict

    restored = PhaseOutput.from_dict(legacy_dict)
    assert restored.promotion_decision is None
    assert restored.run_id == "WF-20260420-legacy01"
    assert restored.phase_name == "content_generation"
    assert restored.status == "ok"
    # Round-trip back through to_dict() now stamps the new key with None,
    # which is the post-Wave-1 canonical shape.
    re_serialized = restored.to_dict()
    assert "promotion_decision" in re_serialized
    assert re_serialized["promotion_decision"] is None
