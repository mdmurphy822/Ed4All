"""Tests for ``lib.validators.kg_quality.KGQualityValidator``.

Mocks the ``KGQualityReporter`` so the wrapper's threshold-comparison
logic is exercised in isolation from the underlying aggregator. Full
end-to-end coverage of the aggregator lives in
``Trainforge/tests/test_kg_quality_report.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.kg_quality import KGQualityValidator  # noqa: E402


class _MockReporter:
    """Stub reporter — returns scores from a pre-canned dict."""

    def __init__(self, *, course_slug: str, run_id: str,
                 output_dir: Path, scores: Optional[Dict[str, float]] = None,
                 written: Optional[list] = None) -> None:
        self.course_slug = course_slug
        self.run_id = run_id
        self.output_dir = output_dir
        self._scores = scores or {
            "completeness": 1.0, "consistency": 1.0,
            "accuracy": 1.0, "coverage": 1.0,
        }
        self._written = written if written is not None else []

    def compute(self, **kwargs: Any) -> Dict[str, Any]:
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

    def write(self, report: Dict[str, Any]) -> Path:
        path = self.output_dir / "kg_quality_report.json"
        self._written.append(path)
        return path


def _factory(scores: Dict[str, float], written: Optional[list] = None):
    def make(*, course_slug: str, run_id: str, output_dir: Path):
        return _MockReporter(
            course_slug=course_slug, run_id=run_id, output_dir=output_dir,
            scores=scores, written=written,
        )
    return make


def _base_inputs(tmp_path: Path) -> Dict[str, Any]:
    concept = tmp_path / "concept_graph.json"
    semantic = tmp_path / "concept_graph_semantic.json"
    concept.write_text("{}", encoding="utf-8")
    semantic.write_text("{}", encoding="utf-8")
    return {
        "course_slug": "test-course",
        "run_id": "test-run-001",
        "output_dir": str(tmp_path / "out"),
        "concept_graph_path": str(concept),
        "semantic_graph_path": str(semantic),
    }


def test_passes_with_perfect_scores(tmp_path: Path):
    """All dims at 1.0 + default thresholds 0 -> no issues, score 1.0."""
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 1.0, "consistency": 1.0,
        "accuracy": 1.0, "coverage": 1.0,
    }))
    result = validator.validate(_base_inputs(tmp_path))
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []


def test_emits_warning_when_below_threshold(tmp_path: Path):
    """Score below configured min -> one warning issue per breach."""
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 0.5, "consistency": 0.95,
        "accuracy": 0.99, "coverage": 1.0,
    }))
    inputs = _base_inputs(tmp_path)
    inputs["min_completeness"] = 0.8
    inputs["min_consistency"] = 0.9
    result = validator.validate(inputs)
    # Always advisory — passed True even with breaches.
    assert result.passed is True
    codes = sorted(i.code for i in result.issues)
    # Only completeness breach (consistency 0.95 > 0.9 cleared).
    assert codes == ["KG_QUALITY_COMPLETENESS_BELOW_THRESHOLD"]
    assert all(i.severity == "warning" for i in result.issues)


def test_zero_threshold_never_breaches(tmp_path: Path):
    """Default 0.0 threshold should not trigger warnings even when
    scores are 0.0 (advisory at roll-out)."""
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 0.0, "consistency": 0.0,
        "accuracy": 0.0, "coverage": 0.0,
    }))
    result = validator.validate(_base_inputs(tmp_path))
    assert result.passed is True
    assert result.issues == []
    assert result.score == 0.0


def test_composite_score_is_unweighted_mean(tmp_path: Path):
    """The GateResult score == mean of the 4 dimensions."""
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 0.8, "consistency": 0.9,
        "accuracy": 1.0, "coverage": 0.5,
    }))
    result = validator.validate(_base_inputs(tmp_path))
    expected = round((0.8 + 0.9 + 1.0 + 0.5) / 4, 4)
    assert result.score == expected


def test_validator_fails_closed_on_missing_pedagogy_graph(tmp_path: Path):
    """Audit C3: missing graph inputs are now a critical fail-closed.

    Previously this returned ``passed=True`` with a warning issue —
    silently shipping an empty knowledge graph to LibV2 when upstream
    concept_extraction failed. The contract per
    ``config/workflows.yaml::textbook_to_course::libv2_archival`` is
    critical-severity, so missing graphs now block.
    """
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 1.0, "consistency": 1.0,
        "accuracy": 1.0, "coverage": 1.0,
    }))
    result = validator.validate({"course_slug": "test"})
    assert result.passed is False
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == "KG_QUALITY_PEDAGOGY_GRAPH_MISSING"
    assert result.issues[0].severity == "critical"


def test_validator_fails_closed_on_reporter_exception(tmp_path: Path):
    """Audit C3: a reporter raise was previously swallowed as
    ``passed=True``; now it critical-fails with the exception class +
    message threaded through the GateIssue."""

    class _RaisingReporter:
        def __init__(self, **_: Any) -> None:
            pass

        def compute(self, **_: Any) -> Dict[str, Any]:
            raise RuntimeError("simulated reporter failure")

        def write(self, _: Dict[str, Any]) -> Path:
            raise AssertionError("write should not be reached")

    def raising_factory(**kw: Any) -> Any:
        return _RaisingReporter(**kw)

    validator = KGQualityValidator(reporter_factory=raising_factory)
    result = validator.validate(_base_inputs(tmp_path))
    assert result.passed is False
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == "KG_QUALITY_REPORTER_ERROR"
    assert result.issues[0].severity == "critical"
    # Exception class + message are surfaced for triage.
    assert "RuntimeError" in result.issues[0].message
    assert "simulated reporter failure" in result.issues[0].message


class _MockCapture:
    """Minimal DecisionCapture stub — records every log_decision call."""

    def __init__(self) -> None:
        self.calls: list = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_validator_emits_decision_capture_on_pass(tmp_path: Path):
    """Audit H3 (partial): one ``kg_quality_report_check`` event per
    ``validate()`` call, with computed metrics + verdict."""
    capture = _MockCapture()
    validator = KGQualityValidator(
        reporter_factory=_factory({
            "completeness": 0.9, "consistency": 0.8,
            "accuracy": 0.7, "coverage": 0.6,
        }),
        decision_capture=capture,
    )
    validator.validate(_base_inputs(tmp_path))
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "kg_quality_report_check"
    assert call["decision"] == "passed"
    metrics = call["metrics"]
    assert metrics["completeness"] == 0.9
    assert metrics["consistency"] == 0.8
    assert metrics["accuracy"] == 0.7
    assert metrics["coverage"] == 0.6
    assert metrics["passed"] is True
    # Composite is the unweighted mean.
    assert abs(metrics["composite"] - (0.9 + 0.8 + 0.7 + 0.6) / 4) < 1e-9


def test_validator_emits_decision_capture_on_missing_graph(tmp_path: Path):
    """Decision capture must fire on the fail-closed missing-graph
    path so post-hoc replay can distinguish it from below-threshold."""
    capture = _MockCapture()
    validator = KGQualityValidator(
        reporter_factory=_factory({
            "completeness": 1.0, "consistency": 1.0,
            "accuracy": 1.0, "coverage": 1.0,
        }),
        decision_capture=capture,
    )
    validator.validate({"course_slug": "test"})
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "kg_quality_report_check"
    assert call["decision"] == "failed:KG_QUALITY_PEDAGOGY_GRAPH_MISSING"
    assert call["metrics"]["failure_code"] == (
        "KG_QUALITY_PEDAGOGY_GRAPH_MISSING"
    )
    assert call["metrics"]["passed"] is False


def test_validator_emits_decision_capture_on_reporter_exception(tmp_path: Path):
    """Decision capture fires on the reporter-exception fail-closed."""

    def raising_factory(**_: Any) -> Any:
        class _R:
            def compute(self, **__: Any) -> Dict[str, Any]:
                raise ValueError("boom")

            def write(self, _r: Dict[str, Any]) -> Path:
                raise AssertionError("unreached")

        return _R()

    capture = _MockCapture()
    validator = KGQualityValidator(
        reporter_factory=raising_factory,
        decision_capture=capture,
    )
    validator.validate(_base_inputs(tmp_path))
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "kg_quality_report_check"
    assert call["metrics"]["failure_code"] == "KG_QUALITY_REPORTER_ERROR"
    assert call["metrics"]["passed"] is False


def test_validator_threads_capture_via_inputs(tmp_path: Path):
    """Per-call ``inputs['decision_capture']`` overrides the
    constructor-injected one (workflow-runner dispatch path)."""
    constructor_capture = _MockCapture()
    per_call_capture = _MockCapture()
    validator = KGQualityValidator(
        reporter_factory=_factory({
            "completeness": 1.0, "consistency": 1.0,
            "accuracy": 1.0, "coverage": 1.0,
        }),
        decision_capture=constructor_capture,
    )
    inputs = _base_inputs(tmp_path)
    inputs["decision_capture"] = per_call_capture
    validator.validate(inputs)
    # Per-call wins; constructor capture is untouched.
    assert len(per_call_capture.calls) == 1
    assert len(constructor_capture.calls) == 0


def test_writes_report_to_disk(tmp_path: Path):
    """Validator must call reporter.write() on a successful compute."""
    written: list = []
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 1.0, "consistency": 1.0,
        "accuracy": 1.0, "coverage": 1.0,
    }, written=written))
    validator.validate(_base_inputs(tmp_path))
    assert len(written) == 1
    assert written[0].name == "kg_quality_report.json"


def test_all_four_dimensions_can_breach(tmp_path: Path):
    """Every dim below its threshold -> one issue per dim."""
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 0.1, "consistency": 0.2,
        "accuracy": 0.3, "coverage": 0.4,
    }))
    inputs = _base_inputs(tmp_path)
    inputs["min_completeness"] = 0.5
    inputs["min_consistency"] = 0.5
    inputs["min_accuracy"] = 0.5
    inputs["min_coverage"] = 0.5
    result = validator.validate(inputs)
    codes = sorted(i.code for i in result.issues)
    assert codes == [
        "KG_QUALITY_ACCURACY_BELOW_THRESHOLD",
        "KG_QUALITY_COMPLETENESS_BELOW_THRESHOLD",
        "KG_QUALITY_CONSISTENCY_BELOW_THRESHOLD",
        "KG_QUALITY_COVERAGE_BELOW_THRESHOLD",
    ]
