"""Regression: ``ValidationGateManager.run_gate`` forwards a gate's
declared ``threshold:`` dial into the validator inputs.

Pre-fix contract bug: the gate manager merged only ``gate.config`` into
the validator inputs, never ``gate.threshold``. ``_apply_thresholds``
recognises only the result-level keys (``max_critical_issues`` /
``max_issues`` / ``min_score`` / ``required_score``). Validators that read
per-dimension floors from their inputs — notably
``KGQualityValidator`` (``min_completeness`` / ``min_consistency`` /
``min_accuracy`` / ``min_coverage``) — therefore never saw the
YAML-configured floor and silently fell back to a 0.0 default, so every
metric breach passed with a warning. This test pins the
threshold -> inputs forward (with ``setdefault`` override semantics) and
the end-to-end fail-closed verdict for a below-floor kg_quality report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import (
    GateBehavior,
    GateConfig,
    GateResult,
    GateSeverity,
    ValidationGateManager,
)
from lib.validators.kg_quality import KGQualityValidator


class _SnapshotValidator:
    """Validator stub that records the inputs dict it is handed."""

    name = "snapshot_threshold"
    version = "1.0.0"

    def __init__(self) -> None:
        self.captured_inputs: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        self.captured_inputs.append(dict(inputs))
        return GateResult(
            gate_id="snapshot",
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
        )


def _gate_with_threshold(
    stub_path: str,
    threshold: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> GateConfig:
    return GateConfig(
        gate_id="threshold_gate",
        validator_path=stub_path,
        severity=GateSeverity.CRITICAL,
        threshold=threshold,
        config=config or {},
        behavior_on_fail=GateBehavior.BLOCK,
        behavior_on_error=GateBehavior.FAIL_CLOSED,
    )


def test_run_gate_forwards_threshold_keys_into_inputs() -> None:
    """A gate's ``threshold:`` keys reach the validator's inputs dict."""
    manager = ValidationGateManager()
    stub = _SnapshotValidator()
    sentinel = "lib.validators.kg_quality.SnapshotThresholdOne"
    manager._validators[sentinel] = stub

    gate = _gate_with_threshold(
        sentinel,
        {
            "min_completeness": 0.95,
            "min_consistency": 0.95,
            "min_accuracy": 0.95,
            "min_coverage": 0.5,
        },
    )
    manager.run_gate(gate, inputs={"course_slug": "syn"})

    seen = stub.captured_inputs[0]
    assert seen["min_completeness"] == 0.95
    assert seen["min_consistency"] == 0.95
    assert seen["min_accuracy"] == 0.95
    assert seen["min_coverage"] == 0.5
    # Pre-existing input keys survive the merge.
    assert seen["course_slug"] == "syn"


def test_run_gate_threshold_setdefault_preserves_explicit_input() -> None:
    """An explicit per-call input overrides the gate's threshold value."""
    manager = ValidationGateManager()
    stub = _SnapshotValidator()
    sentinel = "lib.validators.kg_quality.SnapshotThresholdTwo"
    manager._validators[sentinel] = stub

    gate = _gate_with_threshold(sentinel, {"min_completeness": 0.95})
    manager.run_gate(
        gate, inputs={"min_completeness": 0.10, "course_slug": "syn"}
    )

    seen = stub.captured_inputs[0]
    # Explicit per-call value wins over the YAML threshold (setdefault).
    assert seen["min_completeness"] == 0.10


# ---------------------------------------------------------------------
# End-to-end: the forwarded floor actually blocks a degraded kg_quality
# report, and clears when every dimension is at/above its floor.
# ---------------------------------------------------------------------


class _CannedReporter:
    """Stub KGQualityReporter returning pre-canned dimension scores."""

    def __init__(self, *, course_slug: str, run_id: str,
                 output_dir: Path, scores: Dict[str, float]) -> None:
        self.course_slug = course_slug
        self.run_id = run_id
        self.output_dir = output_dir
        self._scores = scores

    def compute(self, **_: Any) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "course_slug": self.course_slug,
            "dimensions": {
                k: {"score": v, "metric": "stub"}
                for k, v in self._scores.items()
            },
            "per_shape": [],
            "rule_outputs": [],
        }

    def write(self, _report: Dict[str, Any]) -> Path:
        return self.output_dir / "kg_quality_report.json"


def _kg_inputs(tmp_path: Path) -> Dict[str, Any]:
    concept = tmp_path / "concept_graph.json"
    semantic = tmp_path / "concept_graph_semantic.json"
    concept.write_text("{}", encoding="utf-8")
    semantic.write_text("{}", encoding="utf-8")
    return {
        "course_slug": "syn-course",
        "run_id": "syn-run-001",
        "output_dir": str(tmp_path / "out"),
        "concept_graph_path": str(concept),
        "semantic_graph_path": str(semantic),
    }


def _kg_validator(scores: Dict[str, float]) -> KGQualityValidator:
    def factory(*, course_slug: str, run_id: str, output_dir: Path):
        return _CannedReporter(
            course_slug=course_slug, run_id=run_id,
            output_dir=output_dir, scores=scores,
        )

    return KGQualityValidator(reporter_factory=factory)


def test_kg_quality_gate_blocks_below_floor_via_threshold_forward(
    tmp_path: Path,
) -> None:
    """A below-floor report fails the critical/block kg_quality gate
    when the floor is declared only under the gate's ``threshold:``."""
    manager = ValidationGateManager()
    validator = _kg_validator({
        "completeness": 0.50, "consistency": 0.96,
        "accuracy": 0.97, "coverage": 0.80,
    })
    sentinel = "lib.validators.kg_quality.KGQualityFloorBlock"
    manager._validators[sentinel] = validator

    gate = _gate_with_threshold(
        sentinel,
        {
            "min_completeness": 0.95,
            "min_consistency": 0.95,
            "min_accuracy": 0.95,
            "min_coverage": 0.5,
        },
    )
    result = manager.run_gate(gate, _kg_inputs(tmp_path))

    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    assert "KG_QUALITY_COMPLETENESS_BELOW_THRESHOLD" in codes
    assert all(i.severity == "critical" for i in result.issues)


def test_kg_quality_gate_passes_at_or_above_floor(tmp_path: Path) -> None:
    """A report at/above every floor passes the gate cleanly."""
    manager = ValidationGateManager()
    validator = _kg_validator({
        "completeness": 0.95, "consistency": 0.96,
        "accuracy": 0.97, "coverage": 0.50,
    })
    sentinel = "lib.validators.kg_quality.KGQualityFloorPass"
    manager._validators[sentinel] = validator

    gate = _gate_with_threshold(
        sentinel,
        {
            "min_completeness": 0.95,
            "min_consistency": 0.95,
            "min_accuracy": 0.95,
            "min_coverage": 0.5,
        },
    )
    result = manager.run_gate(gate, _kg_inputs(tmp_path))

    assert result.passed is True
    assert result.issues == []
