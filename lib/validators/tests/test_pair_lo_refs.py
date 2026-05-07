"""Wave W-D1 T1.4 — `PairLearningOutcomeRefsValidator.validate()` walk tests.

Pins the gate-runner walk surface (NOT the per-pair `validate_pair`):
- happy-path subset check passes
- phantom LO ref triggers `PHANTOM_PAIR_LO_REFS` + capture event
- W8 deterministic-template skip discriminator short-circuits the
  chunk-id resolution

Mirror of `lib/validators/tests/test_assessment_objective_alignment.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.pair_lo_refs import (  # noqa: E402
    PairLearningOutcomeRefsValidator,
)


# T1.4 stub — used in lib/validators/tests/test_pair_*.py (see plan §2
# "Shared test-stub helpers"; inline by design — silent-degradation
# precedent in the codebase keeps each test self-contained).
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


def _make_pair(
    chunk_id: str,
    lo_refs: list,
    kind: str = "instruction",
    extra: dict | None = None,
) -> dict:
    base = {
        "prompt": "x" * 50,
        "completion": "x" * 60,
        "chunk_id": chunk_id,
        "lo_refs": lo_refs,
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "evt_test",
    }
    if kind == "preference":
        base.pop("completion")
        base["chosen"] = "x" * 60
        base["rejected"] = "y" * 60
    if extra:
        base.update(extra)
    return base


def test_validator_instantiates() -> None:
    """Smoke: validator can be instantiated with no args."""
    validator = PairLearningOutcomeRefsValidator()
    assert validator is not None
    # Exposes the gate-runner contract.
    assert callable(getattr(validator, "validate", None))


def test_empty_inputs_returns_gate_result_no_crash() -> None:
    """Empty-inputs case returns a GateResult (does not raise)."""
    result = PairLearningOutcomeRefsValidator().validate({})
    # Missing chunks_path / course_dir → critical fail with structured
    # GateResult, not a Python exception.
    assert result is not None
    assert result.passed is False
    assert hasattr(result, "issues")
    assert any(
        i.code in {"MISSING_CHUNKS_PATH", "CHUNKS_NOT_FOUND"}
        for i in result.issues
    )


def test_happy_path_subset_check_passes(tmp_path: Path) -> None:
    capture = _StubCapture()
    chunks = tmp_path / "chunks.jsonl"
    inst = tmp_path / "instruction_pairs.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [
        {"id": "c1", "learning_outcome_refs": ["TO-01", "CO-01"]},
    ])
    _write_jsonl(inst, [_make_pair("c1", ["to-01"])])
    _write_jsonl(pref, [_make_pair("c1", ["co-01"], kind="preference")])

    result = PairLearningOutcomeRefsValidator().validate({
        "instruction_pairs_path": str(inst),
        "preference_pairs_path": str(pref),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert result.passed is True
    assert result.score == 1.0
    assert result.action is None
    assert len(capture.calls) == 1
    event = capture.calls[0]
    assert event["decision_type"] == "pair_lo_refs_check"
    assert event["decision"] == "passed"
    rationale = event["rationale"]
    assert len(rationale) >= 60
    assert "audited" in rationale
    assert "phantom" in rationale


def test_phantom_lo_ref_fires_critical(tmp_path: Path) -> None:
    capture = _StubCapture()
    chunks = tmp_path / "chunks.jsonl"
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(chunks, [
        {"id": "c1", "learning_outcome_refs": ["TO-01"]},
    ])
    _write_jsonl(inst, [_make_pair("c1", ["TO-99"])])  # phantom

    result = PairLearningOutcomeRefsValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    assert "PHANTOM_PAIR_LO_REFS" in codes
    assert any(i.severity == "critical" for i in result.issues)
    # Decision capture surfaces the failure with a structured shape.
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"].startswith("failed:")
    assert "phantom=1" in capture.calls[0]["decision"]


def test_deterministic_template_skip_short_circuits(tmp_path: Path) -> None:
    """W8 deterministic-pair audit-stamp: pairs carrying
    ``pair_lo_resolution.skipped == "deterministic_template"`` skip the
    chunk-id resolution path (their synthetic chunk_ids don't appear in
    chunks.jsonl by construction). Pinned at pair_lo_refs.py:464-486."""
    capture = _StubCapture()
    chunks = tmp_path / "chunks.jsonl"
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(chunks, [
        {"id": "c1", "learning_outcome_refs": ["TO-01"]},
    ])
    deterministic_pair = _make_pair(
        "violation_fixture:datatype_int_age",
        ["TO-01"],
        extra={
            "pair_lo_resolution": {
                "declared_los": ["TO-01"],
                "chunk_los": [],
                "phantom_los": [],
                "skipped": "deterministic_template",
            },
        },
    )
    _write_jsonl(inst, [deterministic_pair])

    result = PairLearningOutcomeRefsValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    # Deterministic pair short-circuits — no PAIR_CHUNK_NOT_FOUND fires
    # against the synthetic ``violation_fixture:...`` chunk_id.
    assert result.passed is True
    assert all(
        i.code != "PAIR_CHUNK_NOT_FOUND" for i in result.issues
    )


def test_no_capture_no_emit_no_crash(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(chunks, [{"id": "c1", "learning_outcome_refs": ["TO-01"]}])
    _write_jsonl(inst, [_make_pair("c1", ["to-01"])])
    base = PairLearningOutcomeRefsValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunks_path": str(chunks),
    })
    captured = PairLearningOutcomeRefsValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunks_path": str(chunks),
        "decision_capture": _StubCapture(),
    })
    assert base.passed == captured.passed
    assert base.score == captured.score
