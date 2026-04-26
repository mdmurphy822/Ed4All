"""Wave 82 tests for SemanticGraphRuleOutputValidator.

Pins the silent-zero detection contract that closes the rdf-shacl-551
audit's load-bearing finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.validators.semantic_graph_rule_output import (
    SemanticGraphRuleOutputValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    edges_per_rule: dict,
    rule_versions: dict | None = None,
) -> dict:
    """Construct a concept_graph_semantic.json shape with N edges per rule."""
    edges = []
    for rule, count in edges_per_rule.items():
        for i in range(count):
            edges.append({
                "source": f"src-{rule}-{i}",
                "target": f"tgt-{rule}-{i}",
                "type": rule.replace("_from_", "-from-").replace("_", "-"),
                "confidence": 0.7,
                "provenance": {
                    "rule": rule,
                    "rule_version": (rule_versions or {}).get(rule, 1),
                    "evidence": {},
                },
            })
    return {
        "kind": "typed_concept_graph",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "rule_versions": rule_versions or {r: 1 for r in edges_per_rule},
        "nodes": [],
        "edges": edges,
    }


def _write_graph(path: Path, graph: dict) -> Path:
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Behaviour-flag gating
# ---------------------------------------------------------------------------


class TestEnabledGate:
    def test_disabled_returns_pass_short_circuit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRAINFORGE_VALIDATE_RULE_OUTPUTS", raising=False)
        v = SemanticGraphRuleOutputValidator()
        result = v.validate({})  # no inputs at all — disabled by default
        assert result.passed is True
        assert result.issues == []

    def test_explicit_disabled_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRAINFORGE_VALIDATE_RULE_OUTPUTS", "true")
        v = SemanticGraphRuleOutputValidator()
        result = v.validate({"enabled": False})
        assert result.passed is True

    def test_env_true_enables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRAINFORGE_VALIDATE_RULE_OUTPUTS", "true")
        v = SemanticGraphRuleOutputValidator()
        # Missing current_path with enabled=true fails fast.
        result = v.validate({})
        assert result.passed is False
        assert any(i.code == "MISSING_CURRENT_PATH" for i in result.issues)


# ---------------------------------------------------------------------------
# Audit-case reproduction: silent-zero detection
# ---------------------------------------------------------------------------


class TestSilentZeroDetection:
    def test_audit_failure_mode_reproduces(self, tmp_path):
        # Baseline: the wave76.bak shape (2,004 edges across 7 rules).
        baseline = _make_graph({
            "prerequisite_from_lo_order": 210,
            "related_from_cooccurrence": 120,
            "defined_by_from_first_mention": 424,
            "derived_from_lo_ref": 969,
            "exemplifies_from_example_chunks": 205,
            "assesses_from_question_lo": 53,
            "misconception_of_from_misconception_ref": 23,
        })
        # Current: the rdf-shacl-551 broken state — 5 rules dropped to 0,
        # rule_versions unchanged.
        current = _make_graph(
            {
                "prerequisite_from_lo_order": 210,
                "related_from_cooccurrence": 120,
                "defined_by_from_first_mention": 0,
                "derived_from_lo_ref": 0,
                "exemplifies_from_example_chunks": 0,
                "assesses_from_question_lo": 0,
                "misconception_of_from_misconception_ref": 0,
            },
            # Same rule_versions as baseline (the audit's smoking gun).
            rule_versions={
                "prerequisite_from_lo_order": 1,
                "related_from_cooccurrence": 1,
                "defined_by_from_first_mention": 2,
                "derived_from_lo_ref": 2,
                "exemplifies_from_example_chunks": 1,
                "assesses_from_question_lo": 1,
                "misconception_of_from_misconception_ref": 1,
            },
        )
        # Set baseline rule_versions to match.
        baseline["rule_versions"] = current["rule_versions"]

        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "enabled": True,
        })
        assert result.passed is False
        # All 5 zero-edge rules produce SILENT_ZERO_REGRESSION issues.
        zero_codes = [i for i in result.issues if i.code == "SILENT_ZERO_REGRESSION"]
        assert len(zero_codes) == 5
        flagged = {issue.message.split("'")[1] for issue in zero_codes}
        assert flagged == {
            "defined_by_from_first_mention",
            "derived_from_lo_ref",
            "exemplifies_from_example_chunks",
            "assesses_from_question_lo",
            "misconception_of_from_misconception_ref",
        }

    def test_version_bump_exempts_rule(self, tmp_path):
        # Same input shape as audit case, but rule_version bumped on one
        # zero-output rule → that rule is exempt (intentional change).
        baseline = _make_graph(
            {"defined_by_from_first_mention": 100, "other_rule": 50},
            rule_versions={"defined_by_from_first_mention": 1, "other_rule": 1},
        )
        current = _make_graph(
            {"defined_by_from_first_mention": 0, "other_rule": 50},
            rule_versions={"defined_by_from_first_mention": 2, "other_rule": 1},
        )
        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "enabled": True,
        })
        assert result.passed is True

    def test_baseline_rule_below_floor_is_exempt(self, tmp_path):
        # Rule produced 5 edges in baseline (below default floor of 10).
        # Current produces 0 → exempt.
        baseline = _make_graph({"rare_rule": 5, "common_rule": 100})
        current = _make_graph({"rare_rule": 0, "common_rule": 100})
        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "enabled": True,
        })
        assert result.passed is True

    def test_floor_overridable_via_input(self, tmp_path):
        # min_baseline_edges=3 → "rare_rule" with 5 in baseline is now load-bearing.
        baseline = _make_graph({"rare_rule": 5})
        current = _make_graph({"rare_rule": 0})
        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "min_baseline_edges": 3,
            "enabled": True,
        })
        assert result.passed is False

    def test_partial_regression_passes(self, tmp_path):
        # Wave 82 keeps the gate strict-zero only — a rule dropping from
        # 100 to 5 edges currently passes. Documented limitation.
        baseline = _make_graph({"r": 100})
        current = _make_graph({"r": 5})
        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "enabled": True,
        })
        assert result.passed is True

    def test_clean_run_passes(self, tmp_path):
        baseline = _make_graph({"r1": 100, "r2": 50})
        current = _make_graph({"r1": 99, "r2": 50})
        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "enabled": True,
        })
        assert result.passed is True
        assert result.issues == []


# ---------------------------------------------------------------------------
# Robustness — malformed inputs, missing files, no-baseline
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_no_baseline_passes_with_info(self, tmp_path):
        current = _make_graph({"r": 100})
        cur_path = _write_graph(tmp_path / "current.json", current)

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({"current_path": cur_path, "enabled": True})
        assert result.passed is True
        assert any(i.code == "NO_BASELINE" for i in result.issues)

    def test_missing_current_fails_critical(self, tmp_path):
        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": tmp_path / "does-not-exist.json",
            "enabled": True,
        })
        assert result.passed is False
        assert any(i.code == "CURRENT_UNREADABLE" for i in result.issues)

    def test_unreadable_baseline_warns_but_passes(self, tmp_path):
        current = _make_graph({"r": 100})
        cur_path = _write_graph(tmp_path / "current.json", current)
        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": tmp_path / "no-baseline.json",
            "enabled": True,
        })
        assert result.passed is True
        assert any(i.code == "BASELINE_UNREADABLE" for i in result.issues)

    def test_malformed_json_baseline_warns(self, tmp_path):
        current = _make_graph({"r": 100})
        cur_path = _write_graph(tmp_path / "current.json", current)
        bad_baseline = tmp_path / "bad.json"
        bad_baseline.write_text("not valid json {{{", encoding="utf-8")

        v = SemanticGraphRuleOutputValidator()
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bad_baseline,
            "enabled": True,
        })
        assert result.passed is True  # warns, doesn't fail
        assert any(i.code == "BASELINE_UNREADABLE" for i in result.issues)

    def test_handles_edges_without_provenance(self, tmp_path):
        # Defensive: edges missing provenance shouldn't crash the counter.
        current = {
            "rule_versions": {"r": 1},
            "edges": [
                {"source": "a", "target": "b", "type": "x"},
                {"source": "c", "target": "d", "type": "y", "provenance": "not-a-dict"},
                {"source": "e", "target": "f", "type": "z", "provenance": {"rule": "r"}},
            ],
        }
        baseline = _make_graph({"r": 50})
        cur_path = _write_graph(tmp_path / "current.json", current)
        bak_path = _write_graph(tmp_path / "baseline.json", baseline)

        v = SemanticGraphRuleOutputValidator()
        # 1 edge has provenance.rule=r in current; baseline has 50 — still
        # passes because current > 0.
        result = v.validate({
            "current_path": cur_path,
            "baseline_path": bak_path,
            "enabled": True,
        })
        assert result.passed is True
