"""Wave 108 / Phase B: EvalGatingValidator must fail closed when
post-training eval scores fall below thresholds (regression / yes-bias /
no-bias / source-match drop)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.validators.eval_gating import EvalGatingValidator


def _write_report(model_dir: Path, **fields: Any) -> Path:
    """Build a baseline-passing eval_report.json under model_dir, with
    overrides applied via fields."""
    report: Dict[str, Any] = {
        "faithfulness": 0.80,
        "coverage": 0.90,
        "profile": "rdf_shacl",
        "per_tier": {},
        "per_invariant": {},
        "baseline_delta": 0.10,
        "source_match": 0.65,
        "negative_grounding_accuracy": 0.70,
        "yes_rate": 0.55,
        "metrics": {"hallucination_rate": 0.20},
    }
    report.update(fields)
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    p = eval_dir / "eval_report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


def test_passing_report_has_no_critical_issues(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-v1"
    model_dir.mkdir(parents=True)
    _write_report(model_dir)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is True
    assert not [i for i in result.issues if i.severity == "critical"]


def test_faithfulness_below_threshold_fails_critical(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-v2"
    model_dir.mkdir(parents=True)
    _write_report(model_dir, faithfulness=0.40)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_FAITHFULNESS_BELOW_THRESHOLD" in codes


def test_yes_rate_above_threshold_fails_critical(tmp_path: Path) -> None:
    """The exact regression class Phase B catches: a 'yes always'
    template-recognizer adapter would have yes_rate ~= 1.0."""
    model_dir = tmp_path / "models" / "test-v3"
    model_dir.mkdir(parents=True)
    _write_report(model_dir, yes_rate=0.95)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_YES_BIAS_DETECTED" in codes


def test_negative_grounding_accuracy_below_threshold_fails_critical(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "test-v4"
    model_dir.mkdir(parents=True)
    _write_report(model_dir, negative_grounding_accuracy=0.20)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_NEGATIVE_GROUNDING_BELOW_THRESHOLD" in codes


def test_baseline_delta_negative_fails_critical(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-v5"
    model_dir.mkdir(parents=True)
    _write_report(model_dir, baseline_delta=-0.05)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_BASELINE_REGRESSION" in codes


def test_source_match_below_threshold_fails_critical(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-v6"
    model_dir.mkdir(parents=True)
    _write_report(model_dir, source_match=0.10)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_SOURCE_MATCH_BELOW_THRESHOLD" in codes


def test_missing_eval_report_fails_critical(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-v7"
    model_dir.mkdir(parents=True)
    result = EvalGatingValidator().validate({"model_dir": str(model_dir)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_REPORT_NOT_FOUND" in codes


def test_capture_emits_eval_gating_decision(tmp_path: Path) -> None:
    """CLAUDE.md mandate: every load-bearing decision logs to capture."""
    class _Capture:
        def __init__(self) -> None:
            self.events = []

        def log_decision(self, **kwargs: Any) -> None:
            self.events.append(dict(kwargs))

    model_dir = tmp_path / "models" / "test-cap"
    model_dir.mkdir(parents=True)
    _write_report(model_dir)
    capture = _Capture()
    EvalGatingValidator().validate({
        "model_dir": str(model_dir),
        "capture": capture,
    })
    assert any(e["decision_type"] == "eval_gating_decision" for e in capture.events)
    rationale = capture.events[0]["rationale"]
    assert len(rationale) >= 20
    assert any(s in rationale for s in ("faithfulness", "yes_rate", "baseline_delta"))


def test_per_property_accuracy_below_floor_fails_critical(tmp_path: Path) -> None:
    """A property scoring below its min_accuracy fails the gate."""
    model_dir = tmp_path / "models" / "test-prop"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        per_property_accuracy={
            "sh_datatype": 0.80,
            "sh_class": 0.10,        # below 0.40 floor
            "owl_sameas": None,      # unscored — skipped
        },
    )
    result = EvalGatingValidator().validate({
        "model_dir": str(model_dir),
        "thresholds": {"min_per_property_accuracy": 0.40},
    })
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EVAL_PER_PROPERTY_BELOW_THRESHOLD" in codes
    msg = " ".join(i.message for i in result.issues if i.severity == "critical")
    assert "sh_class" in msg


def test_per_property_passes_when_all_scored_above_floor(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "test-prop-pass"
    model_dir.mkdir(parents=True)
    _write_report(
        model_dir,
        per_property_accuracy={
            "sh_datatype": 0.50, "sh_class": 0.55, "owl_sameas": None,
        },
    )
    result = EvalGatingValidator().validate({
        "model_dir": str(model_dir),
        "thresholds": {"min_per_property_accuracy": 0.40},
    })
    assert result.passed is True
