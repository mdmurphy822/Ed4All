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


def test_missing_input_yields_warning_not_block(tmp_path: Path):
    """Missing required inputs -> advisory warning, passed=True."""
    validator = KGQualityValidator(reporter_factory=_factory({
        "completeness": 1.0, "consistency": 1.0,
        "accuracy": 1.0, "coverage": 1.0,
    }))
    result = validator.validate({"course_slug": "test"})
    assert result.passed is True  # advisory only
    assert len(result.issues) == 1
    assert result.issues[0].code == "KG_QUALITY_INPUT_MISSING"
    assert result.issues[0].severity == "warning"


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
