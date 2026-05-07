"""Wave W-D1 T1.4 — `PairObjectiveDeliveryValidator.validate()` walk tests.

Audit-walk only: asserts every pair on disk carries
`pair_objective_alignment` + `pair_objective_alignment_pass_rate` keys.
Per the validator docstring at `pair_objective_delivery.py:1038-1068`,
this is NOT a re-run of the per-pair tri-axis NLI / Bloom-gap / verb
fan-out; that path requires the synthesized-objectives map and is the
call-site's responsibility.

Mirror of `lib/validators/tests/test_assessment_objective_alignment.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.pair_objective_delivery import (  # noqa: E402
    PairObjectiveDeliveryValidator,
)


# T1.4 stub — used in lib/validators/tests/test_pair_*.py (see plan §2
# "Shared test-stub helpers"; inline by design).
class _StubCapture:
    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "kwargs": kw,
        })


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _pair_with_audit_fields() -> dict:
    return {
        "prompt": "x" * 50,
        "completion": "x" * 60,
        "chunk_id": "c1",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "evt_test",
        "pair_objective_alignment": None,
        "pair_objective_alignment_pass_rate": None,
    }


def test_validator_instantiates() -> None:
    """Smoke: validator can be instantiated with no args."""
    validator = PairObjectiveDeliveryValidator()
    assert validator is not None
    assert callable(getattr(validator, "validate", None))


def test_empty_inputs_returns_gate_result_no_crash() -> None:
    """Empty-inputs case returns a GateResult (does not raise)."""
    result = PairObjectiveDeliveryValidator().validate({})
    assert result is not None
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "MISSING_INPUTS" in codes


def test_pairs_with_audit_fields_pass(tmp_path: Path) -> None:
    capture = _StubCapture()
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst, [_pair_with_audit_fields() for _ in range(3)])

    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is True
    assert len(capture.calls) == 1
    event = capture.calls[0]
    assert event["decision_type"] == "pair_objective_delivery_check"
    assert event["decision"].startswith("audit:passed")
    rationale = event["rationale"]
    assert len(rationale) >= 30
    assert "audited" in rationale


def test_missing_pair_objective_alignment_fires_critical(tmp_path: Path) -> None:
    capture = _StubCapture()
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_audit_fields()
    pair.pop("pair_objective_alignment")
    _write_jsonl(inst, [pair])

    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "MISSING_PAIR_OBJECTIVE_ALIGNMENT" in codes
    assert any(i.severity == "critical" for i in result.issues)
    assert capture.calls[0]["decision"].startswith("audit:failed")


def test_missing_pass_rate_fires_critical(tmp_path: Path) -> None:
    capture = _StubCapture()
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_audit_fields()
    pair.pop("pair_objective_alignment_pass_rate")
    _write_jsonl(inst, [pair])

    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "MISSING_PAIR_OBJECTIVE_ALIGNMENT_PASS_RATE" in codes


def test_no_capture_no_crash(tmp_path: Path) -> None:
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst, [_pair_with_audit_fields()])
    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
    })
    assert result.passed is True
