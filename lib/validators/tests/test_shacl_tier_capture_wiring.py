"""H3 wave W3 — capture-wiring regression for the SHACL-tier validators.

Pins one ``decision_capture`` event per ``validate()`` call across:

* ``PageObjectivesShaclValidator``  → ``page_objectives_shacl_check``
* ``CourseforgeOutlineShaclValidator`` → ``courseforge_outline_shacl_check``
* ``SemanticGraphRuleOutputValidator`` → ``semantic_graph_rule_output_check``

Pattern A cardinality (one event per ``validate()`` call). The test
matrix asserts:

1. Capture fires when ``inputs['decision_capture']`` is wired in.
2. Each emit carries the canonical ``decision_type`` for its validator.
3. Each emit carries the SHACL violation counts (per-shape conform /
   non-conform), the rule / shape names that fired, and the input graph
   size in its ``metrics`` blob.
4. Rationale length ≥ 60 chars (regression-pins against static
   boilerplate rationales — H3 rationale-quality contract).
5. Absent ``decision_capture`` → no emit, no crash, identical
   ``GateResult``.

Each validator exposes a deps-skip / disabled path the test exercises
without paying the pyld / pyshacl import tax: that path emits the
canonical decision shape so ops can spot the gate skipping in
production.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.courseforge_outline_shacl import (  # noqa: E402
    CourseforgeOutlineShaclValidator,
)
from lib.validators.semantic_graph_rule_output import (  # noqa: E402
    SemanticGraphRuleOutputValidator,
)
from lib.validators.shacl_runner import (  # noqa: E402
    PageObjectivesShaclValidator,
)


class _MockCapture:
    """Minimal DecisionCapture stub — records every log_decision call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# --------------------------------------------------------------------- #
# PageObjectivesShaclValidator
# --------------------------------------------------------------------- #


def test_page_objectives_shacl_emits_capture_on_missing_content_dir():
    """Missing-content-dir path emits one ``page_objectives_shacl_check``."""
    capture = _MockCapture()
    v = PageObjectivesShaclValidator()

    result = v.validate({"decision_capture": capture})

    assert result.passed is False
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "page_objectives_shacl_check"
    assert call["decision"].startswith("failed:")
    assert call["metrics"]["failure_code"] == "MISSING_CONTENT_DIR"
    assert call["metrics"]["passed"] is False
    assert call["metrics"]["payloads_audited"] == 0
    # Per-shape conform/non-conform counts present.
    assert "violations_count" in call["metrics"]
    assert "critical_count" in call["metrics"]
    assert "warning_count" in call["metrics"]
    assert "shape_iri_counts" in call["metrics"]
    # Rationale length pin — guards against boilerplate static strings.
    assert len(call["rationale"]) >= 60


def test_page_objectives_shacl_emits_capture_on_empty_corpus(tmp_path: Path):
    """Empty content_dir (no week_* pages) → passed=True + one emit."""
    capture = _MockCapture()
    v = PageObjectivesShaclValidator()

    result = v.validate({
        "content_dir": str(tmp_path),
        "decision_capture": capture,
    })

    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "page_objectives_shacl_check"
    assert call["decision"] == "passed"
    assert call["metrics"]["passed"] is True
    assert call["metrics"]["payloads_audited"] == 0
    assert call["metrics"]["pages_scanned"] == 0
    assert call["metrics"]["target_class"] == "schema:WebPage"
    assert len(call["rationale"]) >= 60


def test_page_objectives_shacl_emits_capture_on_deps_missing(
    monkeypatch, tmp_path: Path,
):
    """SHACL deps missing → capture fires with SHACL_DEPS_MISSING.

    Builds a synthetic ``week_01/page.html`` with one JSON-LD block
    so the validator gets past the empty-corpus short-circuit and
    hits the deps-missing path. ``_ensure_deps`` is patched to raise
    so the test runs even when pyld / pyshacl extras ARE installed.
    """
    week_dir = tmp_path / "week_01"
    week_dir.mkdir()
    (week_dir / "page.html").write_text(
        '<html><head><script type="application/ld+json">'
        '{"@type": "WebPage"}</script></head><body></body></html>',
        encoding="utf-8",
    )

    from lib.validators import shacl_runner as mod
    from lib.validators.shacl_runner import ShaclDepsMissing

    def _raise():
        raise ShaclDepsMissing("pyshacl not importable in this env")

    monkeypatch.setattr(mod, "_ensure_deps", _raise)

    capture = _MockCapture()
    v = PageObjectivesShaclValidator()
    result = v.validate({
        "content_dir": str(tmp_path),
        "decision_capture": capture,
    })

    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "page_objectives_shacl_check"
    assert call["metrics"]["failure_code"] == "SHACL_DEPS_MISSING"
    assert call["metrics"]["passed"] is True
    assert call["metrics"]["warning_count"] == 1
    assert len(call["rationale"]) >= 60


def test_page_objectives_shacl_no_capture_no_emit_no_crash(tmp_path: Path):
    """Absent ``decision_capture`` → no emit, no crash, identical pass."""
    v = PageObjectivesShaclValidator()
    result = v.validate({"content_dir": str(tmp_path)})
    assert result.passed is True
    assert result.issues == []


# --------------------------------------------------------------------- #
# CourseforgeOutlineShaclValidator
# --------------------------------------------------------------------- #


def test_courseforge_outline_shacl_emits_capture_on_missing_input():
    """Missing blocks/blocks_path path → one ``courseforge_outline_shacl_check``."""
    capture = _MockCapture()
    v = CourseforgeOutlineShaclValidator()

    result = v.validate({"decision_capture": capture})

    assert result.passed is False
    assert result.action == "block"
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "courseforge_outline_shacl_check"
    assert call["decision"].startswith("failed:")
    assert call["metrics"]["failure_code"] == "MISSING_BLOCKS_INPUT"
    assert call["metrics"]["passed"] is False
    assert call["metrics"]["target_class"] == "ed4all:Block"
    assert "violations_count" in call["metrics"]
    assert "critical_count" in call["metrics"]
    assert "warning_count" in call["metrics"]
    assert "shape_iri_counts" in call["metrics"]
    assert "block_type_counts" in call["metrics"]
    assert len(call["rationale"]) >= 60


def test_courseforge_outline_shacl_emits_capture_on_empty_blocks():
    """Empty blocks list → passed=True + one emit + canonical metrics."""
    capture = _MockCapture()
    v = CourseforgeOutlineShaclValidator()

    result = v.validate({"blocks": [], "decision_capture": capture})

    assert result.passed is True
    assert result.action is None
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "courseforge_outline_shacl_check"
    assert call["decision"] == "passed"
    assert call["metrics"]["passed"] is True
    assert call["metrics"]["payloads_audited"] == 0
    assert call["metrics"]["block_type_counts"] == {}
    assert len(call["rationale"]) >= 60


def test_courseforge_outline_shacl_emits_capture_on_deps_missing(monkeypatch):
    """SHACL deps missing → capture fires with SHACL_DEPS_MISSING +
    block_type_counts populated from the input payloads.

    Exercises the deps-missing path WITHOUT requiring pyld / pyshacl
    extras — patches ``_ensure_deps`` to raise.
    """
    from lib.validators import courseforge_outline_shacl as mod
    from lib.validators.shacl_runner import ShaclDepsMissing

    def _raise():
        raise ShaclDepsMissing("pyshacl not importable in this env")

    monkeypatch.setattr(mod, "_ensure_deps", _raise)

    capture = _MockCapture()
    v = CourseforgeOutlineShaclValidator()
    blocks = [
        {"blockId": "w01#concept_0", "blockType": "concept", "sequence": 0},
        {"blockId": "w01#concept_1", "blockType": "concept", "sequence": 1},
        {"blockId": "w01#example_0", "blockType": "example", "sequence": 2},
    ]

    result = v.validate({"blocks": blocks, "decision_capture": capture})

    assert result.passed is True
    assert result.action is None
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "courseforge_outline_shacl_check"
    assert call["metrics"]["failure_code"] == "SHACL_DEPS_MISSING"
    assert call["metrics"]["passed"] is True
    assert call["metrics"]["payloads_audited"] == 3
    # Block-type distribution is the input-graph signal — verify
    # the gate captured it on the deps-missing skip path.
    assert call["metrics"]["block_type_counts"] == {"concept": 2, "example": 1}
    assert len(call["rationale"]) >= 60


def test_courseforge_outline_shacl_no_capture_no_emit_no_crash():
    """Absent ``decision_capture`` → no emit, no crash."""
    v = CourseforgeOutlineShaclValidator()
    result = v.validate({"blocks": []})
    assert result.passed is True
    assert result.issues == []


# --------------------------------------------------------------------- #
# SemanticGraphRuleOutputValidator
# --------------------------------------------------------------------- #


def _write_graph(path: Path, edges: List[Dict[str, Any]],
                 rule_versions: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps({"edges": edges, "rule_versions": rule_versions}),
        encoding="utf-8",
    )


def test_semantic_graph_rule_output_emits_capture_when_disabled():
    """Behaviour-flag-off short-circuit → still emits one event tagged
    DISABLED so ops can spot the gate skipping."""
    capture = _MockCapture()
    v = SemanticGraphRuleOutputValidator()

    # enabled=False explicit (avoids env-var dependency in the test).
    result = v.validate({"enabled": False, "decision_capture": capture})

    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "semantic_graph_rule_output_check"
    assert call["metrics"]["enabled"] is False
    assert call["metrics"]["failure_code"] == "DISABLED"
    assert call["metrics"]["rule_zero_drop_count"] == 0
    assert call["metrics"]["total_rules_evaluated"] == 0
    assert len(call["rationale"]) >= 60


def test_semantic_graph_rule_output_emits_capture_on_silent_zero(
    tmp_path: Path,
):
    """Silent-zero regression path → capture fires with the rule names
    that dropped to zero AND the per-rule + total edge counts."""
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    # Baseline: rule_a has 50 edges; rule_b has 30 edges.
    _write_graph(
        baseline_path,
        edges=(
            [{"provenance": {"rule": "rule_a"}}] * 50
            + [{"provenance": {"rule": "rule_b"}}] * 30
        ),
        rule_versions={"rule_a": "1.0.0", "rule_b": "1.0.0"},
    )
    # Current: rule_a still has 50 edges; rule_b silently drops to zero
    # with the same rule_version.
    _write_graph(
        current_path,
        edges=[{"provenance": {"rule": "rule_a"}}] * 50,
        rule_versions={"rule_a": "1.0.0", "rule_b": "1.0.0"},
    )

    capture = _MockCapture()
    v = SemanticGraphRuleOutputValidator()
    result = v.validate({
        "enabled": True,
        "current_path": str(current_path),
        "baseline_path": str(baseline_path),
        "decision_capture": capture,
    })

    assert result.passed is False
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "semantic_graph_rule_output_check"
    assert call["decision"].startswith("failed:")
    assert call["metrics"]["failure_code"] == "SILENT_ZERO_REGRESSION"
    assert call["metrics"]["rule_zero_drop_count"] == 1
    assert call["metrics"]["zero_drop_rule_names"] == ["rule_b"]
    assert call["metrics"]["total_rules_evaluated"] == 2
    # Total edge counts = input graph size signal.
    assert call["metrics"]["current_total_edges"] == 50
    assert call["metrics"]["baseline_total_edges"] == 80
    assert "rule_b" in call["rationale"]
    assert len(call["rationale"]) >= 60


def test_semantic_graph_rule_output_no_capture_no_emit_no_crash():
    """Absent ``decision_capture`` → no emit, no crash."""
    v = SemanticGraphRuleOutputValidator()
    result = v.validate({"enabled": False})
    assert result.passed is True
    assert result.issues == []


# --------------------------------------------------------------------- #
# Cross-validator parametrized rationale-quality regression
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "validator_factory,inputs_factory,expected_decision_type",
    [
        (
            PageObjectivesShaclValidator,
            lambda tmp_path: {"content_dir": str(tmp_path)},
            "page_objectives_shacl_check",
        ),
        (
            CourseforgeOutlineShaclValidator,
            lambda tmp_path: {"blocks": []},
            "courseforge_outline_shacl_check",
        ),
        (
            SemanticGraphRuleOutputValidator,
            lambda tmp_path: {"enabled": False},
            "semantic_graph_rule_output_check",
        ),
    ],
)
def test_shacl_tier_validators_emit_canonical_decision_type(
    tmp_path: Path,
    validator_factory: Any,
    inputs_factory: Any,
    expected_decision_type: str,
):
    """Cardinality + decision_type contract per W1 acceptance criteria.

    For each SHACL-tier validator, one ``validate()`` call → one capture
    event with the canonical ``decision_type`` enum value.
    """
    capture = _MockCapture()
    v = validator_factory()
    inputs = inputs_factory(tmp_path)
    inputs["decision_capture"] = capture

    v.validate(inputs)

    assert len(capture.calls) == 1, (
        f"{validator_factory.__name__} must emit exactly one capture event "
        f"per validate() call (Pattern A cardinality)."
    )
    assert capture.calls[0]["decision_type"] == expected_decision_type
    assert len(capture.calls[0]["rationale"]) >= 60
