"""SemanticGraphRuleOutputValidator — Wave 82 silent-zero regression gate.

The rdf-shacl-551-2 audit found that 5 of 7 inference rules in a shipped
``concept_graph_semantic.json`` silently emitted zero edges, while the
``rule_versions`` block was byte-identical to a baseline that produced
2,004 edges across all 7 rules. The pipeline reported success while
shipping a degraded provenance graph because no per-rule output
monitoring existed — only aggregate edge counts were validated.

This validator closes that gap. Given a current semantic graph and a
baseline, it flags rules that:

* Had ≥ ``min_baseline_edges`` (default 10) in the baseline
* Have ZERO edges in the current run
* Carry the same ``rule_version`` between baseline and current

A version bump exempts the rule (intentional removal / behavior change).
A baseline rule below the floor is also exempt (legitimately rare rules
shouldn't gate the pipeline).

Behavior-flagged: only executes when
``TRAINFORGE_VALIDATE_RULE_OUTPUTS=true``. Off by default to avoid
breaking corpora that lack a baseline. The ``inputs['enabled']`` toggle
also exists for explicit per-call control.

Wired into ``config/workflows.yaml`` as a warning-severity gate on the
``textbook_to_course::libv2_archival`` phase (semantic graph is
finalized by archival time). Phase A3 of plans/wave-82-consolidation/.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_semantic_graph_rule_output_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    rule_zero_drop_count: int,
    total_rules_evaluated: int,
    rules_below_floor: int,
    rules_version_bumped: int,
    current_total_edges: int,
    baseline_total_edges: int,
    current_path: Optional[str],
    baseline_path: Optional[str],
    min_baseline_edges: int,
    enabled: bool,
    zero_drop_rule_names: List[str],
) -> None:
    """Emit one ``semantic_graph_rule_output_check`` decision per validate() call.

    H3 wave W3 closure for the silent-zero rule-drop gate. Pattern A
    cardinality (one event per ``validate()``); rationale interpolates
    the per-rule drop counts plus the rule names that fired plus the
    input graph size so post-hoc replay can trace which rules silently
    regressed without re-running the whole concept_extraction phase.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rule_names_str = ", ".join(sorted(zero_drop_rule_names)[:8]) or "none"
    rationale = (
        "semantic_graph_rule_output gate verdict: "
        f"enabled={enabled}, rule_zero_drop_count={rule_zero_drop_count}, "
        f"total_rules_evaluated={total_rules_evaluated}, "
        f"rules_below_floor={rules_below_floor}, "
        f"rules_version_bumped={rules_version_bumped}, "
        f"current_total_edges={current_total_edges}, "
        f"baseline_total_edges={baseline_total_edges}, "
        f"min_baseline_edges={min_baseline_edges}, "
        f"silent_zero_rules=({rule_names_str}), "
        f"current_path={current_path or 'n/a'}, "
        f"baseline_path={baseline_path or 'n/a'}, "
        f"failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "rule_zero_drop_count": int(rule_zero_drop_count),
        "total_rules_evaluated": int(total_rules_evaluated),
        "rules_below_floor": int(rules_below_floor),
        "rules_version_bumped": int(rules_version_bumped),
        "current_total_edges": int(current_total_edges),
        "baseline_total_edges": int(baseline_total_edges),
        "min_baseline_edges": int(min_baseline_edges),
        "enabled": bool(enabled),
        "current_path": current_path,
        "baseline_path": baseline_path,
        "zero_drop_rule_names": list(zero_drop_rule_names),
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="semantic_graph_rule_output_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "semantic_graph_rule_output_check: %s",
            exc,
        )


def _count_edges_by_rule(graph: Mapping[str, Any]) -> Counter:
    """Tally edges per rule name (provenance.rule). Empty Counter on empty graph."""
    counts: Counter = Counter()
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        rule = (
            edge.get("provenance", {}).get("rule")
            if isinstance(edge.get("provenance"), dict)
            else None
        )
        if isinstance(rule, str) and rule:
            counts[rule] += 1
    return counts


def _rule_versions(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the rule_versions dict (or empty when missing/malformed)."""
    rv = graph.get("rule_versions") if isinstance(graph, Mapping) else None
    return dict(rv) if isinstance(rv, dict) else {}


def _load_graph(path: Path) -> Optional[Dict[str, Any]]:
    """Read a concept_graph_semantic.json. None on missing/malformed file."""
    if not path or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


class SemanticGraphRuleOutputValidator:
    """Detect silent-zero regressions in concept_graph_semantic.json.

    Inputs (dict):
        current_path: Path to the just-emitted concept_graph_semantic.json.
        baseline_path: Path to the baseline concept_graph_semantic.json
            (e.g., a prior known-good run or a fixture under
            ``Trainforge/tests/fixtures/``).
        min_baseline_edges: Minimum edge count in baseline to consider
            the rule "load-bearing" (default 10). Below this, the rule
            is exempt (legitimately rare).
        enabled: Optional explicit toggle. When False the validator
            short-circuits to pass. When None (default), reads the
            TRAINFORGE_VALIDATE_RULE_OUTPUTS env var.

    A failed rule produces a critical-severity GateIssue. A version bump
    exempts the rule. A degraded-but-nonzero rule (e.g. dropped from 100
    to 5 edges) currently passes — Wave 82 keeps the gate strict-zero
    only; partial regressions are out-of-scope for this round.
    """

    name: str = "SemanticGraphRuleOutputValidator"
    version: str = "1.0.0"
    gate_id: str = "semantic_graph_rule_output"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        issues: List[GateIssue] = []
        capture = inputs.get("decision_capture")

        # Behaviour-flag short-circuit. Returns "passed" so the gate
        # doesn't accidentally hard-fail in environments without the flag.
        enabled = inputs.get("enabled")
        if enabled is None:
            enabled = (
                os.getenv("TRAINFORGE_VALIDATE_RULE_OUTPUTS", "").lower() == "true"
            )
        current_path_raw = inputs.get("current_path")
        baseline_path_raw = inputs.get("baseline_path")
        min_baseline_edges = int(inputs.get("min_baseline_edges", 10))

        if not enabled:
            _emit_semantic_graph_rule_output_decision(
                capture,
                passed=True,
                code="DISABLED",
                rule_zero_drop_count=0,
                total_rules_evaluated=0,
                rules_below_floor=0,
                rules_version_bumped=0,
                current_total_edges=0,
                baseline_total_edges=0,
                current_path=str(current_path_raw) if current_path_raw else None,
                baseline_path=str(baseline_path_raw) if baseline_path_raw else None,
                min_baseline_edges=min_baseline_edges,
                enabled=False,
                zero_drop_rule_names=[],
            )
            return GateResult(
                gate_id=self.gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[],
            )

        current_path = current_path_raw
        baseline_path = baseline_path_raw

        if not current_path:
            issues.append(GateIssue(
                severity="critical",
                code="MISSING_CURRENT_PATH",
                message="No current_path supplied to the validator.",
                suggestion=(
                    "Pass inputs={'current_path': <path to "
                    "concept_graph_semantic.json>, ...}."
                ),
            ))
            _emit_semantic_graph_rule_output_decision(
                capture,
                passed=False,
                code="MISSING_CURRENT_PATH",
                rule_zero_drop_count=0,
                total_rules_evaluated=0,
                rules_below_floor=0,
                rules_version_bumped=0,
                current_total_edges=0,
                baseline_total_edges=0,
                current_path=None,
                baseline_path=str(baseline_path) if baseline_path else None,
                min_baseline_edges=min_baseline_edges,
                enabled=True,
                zero_drop_rule_names=[],
            )
            return GateResult(
                gate_id=self.gate_id, validator_name=self.name,
                validator_version=self.version, passed=False, issues=issues,
            )

        current = _load_graph(Path(current_path))
        if current is None:
            issues.append(GateIssue(
                severity="critical",
                code="CURRENT_UNREADABLE",
                message=f"Could not load current semantic graph at {current_path}",
                location=str(current_path),
                suggestion=(
                    "Verify the file exists and contains valid JSON; check the "
                    "concept-graph emit step in Trainforge.process_course."
                ),
            ))
            _emit_semantic_graph_rule_output_decision(
                capture,
                passed=False,
                code="CURRENT_UNREADABLE",
                rule_zero_drop_count=0,
                total_rules_evaluated=0,
                rules_below_floor=0,
                rules_version_bumped=0,
                current_total_edges=0,
                baseline_total_edges=0,
                current_path=str(current_path),
                baseline_path=str(baseline_path) if baseline_path else None,
                min_baseline_edges=min_baseline_edges,
                enabled=True,
                zero_drop_rule_names=[],
            )
            return GateResult(
                gate_id=self.gate_id, validator_name=self.name,
                validator_version=self.version, passed=False, issues=issues,
            )

        current_counts = _count_edges_by_rule(current)
        current_total_edges = sum(current_counts.values())

        # No baseline → soft pass with an info issue. Useful on first runs
        # of a corpus where no prior known-good output exists yet.
        if not baseline_path:
            issues.append(GateIssue(
                severity="info",
                code="NO_BASELINE",
                message=(
                    "No baseline_path supplied; skipping silent-zero detection. "
                    "Provide a known-good concept_graph_semantic.json baseline "
                    "to enable regression detection."
                ),
            ))
            _emit_semantic_graph_rule_output_decision(
                capture,
                passed=True,
                code="NO_BASELINE",
                rule_zero_drop_count=0,
                total_rules_evaluated=len(current_counts),
                rules_below_floor=0,
                rules_version_bumped=0,
                current_total_edges=current_total_edges,
                baseline_total_edges=0,
                current_path=str(current_path),
                baseline_path=None,
                min_baseline_edges=min_baseline_edges,
                enabled=True,
                zero_drop_rule_names=[],
            )
            return GateResult(
                gate_id=self.gate_id, validator_name=self.name,
                validator_version=self.version, passed=True, issues=issues,
            )
        baseline = _load_graph(Path(baseline_path))
        if baseline is None:
            issues.append(GateIssue(
                severity="warning",
                code="BASELINE_UNREADABLE",
                message=f"Baseline graph at {baseline_path} could not be loaded.",
                location=str(baseline_path),
                suggestion=(
                    "Skipping rule-output comparison. Repair the baseline file "
                    "or remove the baseline_path input to suppress this warning."
                ),
            ))
            _emit_semantic_graph_rule_output_decision(
                capture,
                passed=True,
                code="BASELINE_UNREADABLE",
                rule_zero_drop_count=0,
                total_rules_evaluated=len(current_counts),
                rules_below_floor=0,
                rules_version_bumped=0,
                current_total_edges=current_total_edges,
                baseline_total_edges=0,
                current_path=str(current_path),
                baseline_path=str(baseline_path),
                min_baseline_edges=min_baseline_edges,
                enabled=True,
                zero_drop_rule_names=[],
            )
            return GateResult(
                gate_id=self.gate_id, validator_name=self.name,
                validator_version=self.version, passed=True, issues=issues,
            )

        baseline_counts = _count_edges_by_rule(baseline)
        baseline_total_edges = sum(baseline_counts.values())
        current_versions = _rule_versions(current)
        baseline_versions = _rule_versions(baseline)

        zero_drop_rules: List[str] = []
        rules_below_floor = 0
        rules_version_bumped = 0

        # For each rule that produced ≥ floor in baseline, demand non-zero
        # in current OR a rule_version bump.
        for rule_name, baseline_n in sorted(baseline_counts.items()):
            if baseline_n < min_baseline_edges:
                rules_below_floor += 1
                continue
            current_n = current_counts.get(rule_name, 0)
            if current_n > 0:
                continue
            cur_v = current_versions.get(rule_name)
            base_v = baseline_versions.get(rule_name)
            if cur_v is not None and base_v is not None and cur_v != base_v:
                # Rule_version changed → intentional behavior change → exempt.
                rules_version_bumped += 1
                continue
            zero_drop_rules.append(rule_name)
            issues.append(GateIssue(
                severity="critical",
                code="SILENT_ZERO_REGRESSION",
                message=(
                    f"Rule '{rule_name}' produced 0 edges in current run but "
                    f"{baseline_n} edges in baseline (rule_version unchanged: "
                    f"{cur_v!r}). This is the failure mode the rdf-shacl-551 "
                    f"audit flagged — pipeline shipped a degraded semantic "
                    f"graph while reporting success."
                ),
                location=f"{current_path}#provenance.rule={rule_name}",
                suggestion=(
                    f"Inspect the rule's input shape: "
                    f"Trainforge/rag/inference_rules/{rule_name}.py. "
                    f"Verify the orchestrator passes correct chunks/course/"
                    f"concept_graph kwargs. Re-run the pipeline against fresh "
                    f"chunks if upstream data was modified post-emit."
                ),
            ))

        passed = not any(i.severity == "critical" for i in issues)
        total_rules_evaluated = len(
            set(baseline_counts.keys()) | set(current_counts.keys())
        )
        _emit_semantic_graph_rule_output_decision(
            capture,
            passed=passed,
            code=None if passed else "SILENT_ZERO_REGRESSION",
            rule_zero_drop_count=len(zero_drop_rules),
            total_rules_evaluated=total_rules_evaluated,
            rules_below_floor=rules_below_floor,
            rules_version_bumped=rules_version_bumped,
            current_total_edges=current_total_edges,
            baseline_total_edges=baseline_total_edges,
            current_path=str(current_path),
            baseline_path=str(baseline_path),
            min_baseline_edges=min_baseline_edges,
            enabled=True,
            zero_drop_rule_names=zero_drop_rules,
        )
        return GateResult(
            gate_id=self.gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            issues=issues,
        )


__all__ = ["SemanticGraphRuleOutputValidator"]
